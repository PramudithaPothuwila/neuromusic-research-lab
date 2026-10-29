"""Experiment orchestration with full provenance tracking."""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd
from pydantic import BaseModel
from scipy import stats
from statsmodels.stats.multitest import multipletests

from neuromusic import services
from neuromusic.analysis import descriptive
from neuromusic.analysis.comparisons import compare_songs, compare_songs_between
from neuromusic.analysis.correlations import rating_associations, song_attribute_associations
from neuromusic.analysis.mixed_effects import fit_song_models
from neuromusic.analysis.similarity import (
    attribute_consistency,
    distance_matrix,
    mantel,
    nearest_neighbours,
    pair_table,
    zscore,
)
from neuromusic.analysis.synchronization import check_preconditions, compute_isc
from neuromusic.audio.stimulus_features import (
    StimulusAudio,
    recording_feature_table,
    song_feature_table,
)
from neuromusic.clustering.evaluation import run_clustering
from neuromusic.config import (
    ClassificationConfig,
    ClusteringConfig,
    RegressionConfig,
    SongSimilarityConfig,
    StatisticsConfig,
    SynchronizationConfig,
)
from neuromusic.domain.enums import ExecutionStatus, ExperimentType
from neuromusic.experiments.persistence import ExperimentStore
from neuromusic.experiments.registry import ExperimentRecord, make_experiment_id
from neuromusic.features.extractor import feature_columns
from neuromusic.ingestion.ratings import load_ratings
from neuromusic.ingestion.song_attributes import attribute_columns, load_song_attributes
from neuromusic.ingestion.stimuli import load_catalogue
from neuromusic.modeling.classifiers import CLASSIFIERS, available_classifiers
from neuromusic.modeling.datasets import build_dataset
from neuromusic.modeling.metrics import classification_metrics, regression_metrics
from neuromusic.modeling.regressors import REGRESSORS, SONG_ONLY_MODELS
from neuromusic.modeling.validation import permutation_test, run_loso
from neuromusic.workspace import Workspace

Progress = Callable[[int, int, str], None] | None


class ExperimentError(RuntimeError):
    pass


class ExperimentRunner:
    def __init__(self, ws: Workspace) -> None:
        self.ws = ws
        self.store = ExperimentStore(ws.experiments_dir)

    # ------------------------------------------------------------------ helpers

    def _features(self, level: str = "trial") -> pd.DataFrame:
        frame = services.load_features(self.ws, level)
        if frame is None:
            raise ExperimentError("features have not been built for the current configuration; "
                                  "run feature extraction first")
        return frame

    def _begin(self, etype: ExperimentType, cfg: BaseModel | dict[str, Any],
               trials: pd.DataFrame, *, models: list[str] | None = None,
               cv: str | None = None, seed: int) -> ExperimentRecord:
        cfg_dict = cfg.model_dump(mode="json") if isinstance(cfg, BaseModel) else cfg
        fingerprint = self.ws.registry.fingerprint()
        now = datetime.now()
        exp_id = make_experiment_id(etype, cfg_dict, fingerprint, now)
        manifest = services.feature_store(self.ws).manifest(self.ws.preprocessing_hash,
                                                            self.ws.feature_hash)
        record = ExperimentRecord(
            experiment_id=exp_id, experiment_type=etype, created_at=now,
            dataset_fingerprint=fingerprint,
            is_synthetic_data=bool(manifest.get("is_synthetic_data", False)),
            n_participants=int(trials["participant_id"].nunique()),
            song_ids=sorted(trials["song_id"].dropna().unique().tolist()),
            recording_ids=sorted(trials["recording_id"].unique().tolist()),
            preprocessing_hash=self.ws.preprocessing_hash, feature_hash=self.ws.feature_hash,
            preprocessing_config=self.ws.preprocessing.model_dump(mode="json"),
            feature_config=self.ws.features.model_dump(mode="json"),
            experiment_config=cfg_dict, models=models or [], cv_strategy=cv, random_seed=seed,
            results_path=str(self.store.path(exp_id)),
        )
        self.store.create(record)
        return record

    def _finish(self, record: ExperimentRecord, error: Exception | None = None) -> None:
        record.status = ExecutionStatus.FAILED if error else ExecutionStatus.COMPLETED
        record.error = f"{type(error).__name__}: {error}" if error else None
        self.store.write_record(record)

    # --------------------------------------------------------------- statistics

    def run_statistics(self, cfg: StatisticsConfig | None = None,
                       features: list[str] | None = None) -> ExperimentRecord:
        cfg = cfg or self.ws.statistics
        trials = self._features("trial")
        windows = self._features("window")
        valid = trials[trials["trial_valid"]]
        record = self._begin(ExperimentType.STATISTICS, cfg, valid,
                             cv=None, seed=cfg.random_seed)
        try:
            feats = features or feature_columns(valid)
            primary = [f for f in feats if any(f.startswith(p)
                                               for p in cfg.primary_feature_prefixes)]
            recs = self.ws.registry.recordings_frame()
            eid = record.experiment_id
            self.store.save_table(eid, "participants_per_song",
                                  descriptive.participants_per_song(recs))
            self.store.save_table(eid, "recordings_per_participant",
                                  descriptive.recordings_per_participant(recs))
            self.store.save_table(eid, "window_summary", descriptive.window_summary(windows))
            self.store.save_table(eid, "feature_summary_by_song",
                                  descriptive.feature_summary_by_song(valid, primary or feats))
            self.store.save_table(eid, "variability", descriptive.variability_summary(
                valid, primary or feats))
            if cfg.design == "between":
                comp = compare_songs_between(valid, feats, cfg, cfg.min_participants_per_song)
            else:
                comp = compare_songs(valid, feats, cfg)
                if comp.summary.empty and cfg.design == "auto":
                    within_notes = comp.notes
                    comp = compare_songs_between(valid, feats, cfg,
                                                 cfg.min_participants_per_song)
                    comp.notes = [*within_notes,
                                  "Falling back to the between-recording design.", *comp.notes]
            self.store.save_table(eid, "comparisons", comp.summary)
            self.store.save_table(eid, "posthoc", comp.posthoc)
            notes = list(comp.notes)
            me = pd.DataFrame()
            if cfg.mixed_effects_enabled and primary:
                me = fit_song_models(valid, primary[: cfg.mixed_effects_max_features],
                                     cfg.correction)
                self.store.save_table(eid, "mixed_effects", me)
                if len(primary) > cfg.mixed_effects_max_features:
                    notes.append(f"Mixed models fitted for the first "
                                 f"{cfg.mixed_effects_max_features} primary features only.")
            attributes = load_song_attributes(self.ws.metadata_dir)
            if not attributes.empty:
                assoc = song_attribute_associations(valid, attributes, primary or feats,
                                                    attribute_columns(attributes),
                                                    cfg.correction)
                if not assoc.empty:
                    self.store.save_table(eid, "song_attribute_associations", assoc)
                    notes.append("Song-attribute associations are exploratory (n = number of "
                                 "songs); attributes are song-level, not participant ratings.")
            ratings = load_ratings(self.ws.metadata_dir)
            if not ratings.empty:
                self.store.save_table(eid, "rating_associations",
                                      rating_associations(valid, ratings, primary or feats,
                                                          cfg.correction))
            summary = comp.summary
            record.metrics = {
                "n_features_tested": len(summary),
                "n_primary_features": int(summary["primary"].sum()) if not summary.empty else 0,
                "n_significant_primary": int((summary["primary"] & summary["significant"]).sum())
                if not summary.empty else 0,
                "n_significant_exploratory": int(
                    (~summary["primary"] & summary["significant"]).sum())
                if not summary.empty else 0,
                "n_participants_complete": comp.n_participants_complete,
                "songs": comp.songs,
                "test": summary["test"].iloc[0] if not summary.empty else None,
                "songs_tested": comp.songs,
                "n_mixed_models_ok": int((me["status"] == "ok").sum()) if not me.empty else 0,
            }
            record.notes = notes
            self._finish(record)
        except Exception as exc:
            self._finish(record, exc)
            raise
        return record

    # ----------------------------------------------------------- classification

    def run_classification(self, cfg: ClassificationConfig | None = None,
                           progress: Progress = None) -> ExperimentRecord:
        cfg = cfg or self.ws.experiments.classification
        features = self._features(cfg.observation_level)
        ds = build_dataset(features, cfg.feature_prefixes, cfg.target, cfg.observation_level,
                           cfg.min_participants)
        if len(ds.classes) < 2:
            raise ExperimentError("classification requires at least two labelled songs")
        unknown = [m for m in cfg.models if m not in available_classifiers()]
        if unknown:
            raise ExperimentError(f"unavailable models: {unknown}")
        models = cfg.models if "dummy" in cfg.models else ["dummy", *cfg.models]
        trials_meta = ds.meta.drop_duplicates("recording_id")
        record = self._begin(ExperimentType.CLASSIFICATION, cfg, trials_meta, models=models,
                             cv="leave-one-subject-out; inner GroupKFold "
                                f"({cfg.inner_cv_splits}) for tuning", seed=cfg.random_seed)
        eid = record.experiment_id
        try:
            all_pred, all_folds, all_imp, metrics = [], [], [], {}
            for i, name in enumerate(models):
                if progress:
                    progress(i, len(models), name)
                res = run_loso(ds, CLASSIFIERS[name], tune=cfg.tune_hyperparameters,
                               inner_splits=cfg.inner_cv_splits, scoring=cfg.tuning_scoring,
                               seed=cfg.random_seed,
                               importance_repeats=cfg.permutation_importance_repeats,
                               n_jobs=cfg.n_jobs)
                all_pred.append(res.predictions)
                all_folds.append(res.folds.assign(model=name))
                if not res.importance.empty:
                    all_imp.append(res.importance.assign(model=name))
                m = classification_metrics(res.predictions, ds.classes, seed=cfg.random_seed)
                m["mean_train_accuracy"] = float(res.folds["train_score"].mean())
                if res.trial_predictions is not None:
                    m["trial_majority_vote"] = classification_metrics(
                        res.trial_predictions, ds.classes, seed=cfg.random_seed)
                metrics[name] = m
            learned = [m for m in models if m != "dummy"]
            if cfg.n_permutations > 0 and learned:
                best = max(learned, key=lambda m: metrics[m]["balanced_accuracy"])
                if progress:
                    progress(len(models), len(models) + 1, f"permutation test ({best})")
                perm = permutation_test(
                    ds, CLASSIFIERS[best], metrics[best]["balanced_accuracy"],
                    cfg.n_permutations, tune=cfg.tune_hyperparameters,
                    inner_splits=cfg.inner_cv_splits, scoring=cfg.tuning_scoring,
                    seed=cfg.random_seed, n_jobs=cfg.n_jobs)
                perm["model"] = best
                metrics[best]["permutation_test"] = perm
                self.store.save_json(eid, "permutation_test", perm)
            self.store.save_table(eid, "predictions", pd.concat(all_pred, ignore_index=True))
            self.store.save_table(eid, "folds", pd.concat(all_folds, ignore_index=True))
            if all_imp:
                self.store.save_table(eid, "importance", pd.concat(all_imp, ignore_index=True))
            self.store.save_json(eid, "metrics", metrics)
            self.store.save_json(eid, "dataset", {
                "n_observations": len(ds.y), "level": ds.level,
                "n_participants": ds.n_participants, "classes": ds.classes,
                "class_support": {str(k): int(v) for k, v in
                                  zip(*np.unique(ds.y, return_counts=True), strict=True)},
                "features": ds.feature_names})
            record.metrics = {
                name: {k: m[k] for k in ("accuracy", "balanced_accuracy", "macro_f1",
                                         "balanced_accuracy_ci", "mean_train_accuracy")}
                | ({"permutation_p": m["permutation_test"]["p_value"]}
                   if "permutation_test" in m else {})
                for name, m in metrics.items()}
            record.hyperparameters = {
                name: {s.name: s.param_grid for s in [CLASSIFIERS[name]]}[name]
                for name in models}
            record.notes = [
                "Held-out metrics come from participants never seen in training or tuning.",
                f"{len(ds.classes)} songs: results do not establish generalisation to new songs.",
                "Song order, session and other confounds may contribute to classification.",
            ]
            if progress:
                progress(1, 1, "done")
            self._finish(record)
        except Exception as exc:
            self._finish(record, exc)
            raise
        return record

    # --------------------------------------------------------------- regression

    def run_regression(self, cfg: RegressionConfig | None = None) -> ExperimentRecord:
        cfg = cfg or self.ws.experiments.regression
        ratings = load_ratings(self.ws.metadata_dir)
        if ratings.empty:
            raise ExperimentError("no emotion ratings imported")
        rating_type = cfg.rating_type or sorted(ratings["rating_type"].unique())[0]
        sel = ratings[ratings["rating_type"] == rating_type]
        trials = self._features("trial")
        keys = ["participant_id", "song_id", "trial_id"]
        merged = trials.merge(sel[[*keys, "rating_value"]], on=keys, how="inner")
        if merged.empty:
            raise ExperimentError(f"no trials matched ratings of type {rating_type!r}")
        ds = build_dataset(merged, cfg.feature_prefixes, "rating_value", "trial")
        ds.y = ds.y.astype(float)
        song_codes, _ = pd.factorize(ds.meta["song_id"])
        record = self._begin(ExperimentType.REGRESSION, cfg, ds.meta,
                             models=["mean_baseline", "song_baseline", *cfg.models],
                             cv="leave-one-subject-out", seed=cfg.random_seed)
        eid = record.experiment_id
        try:
            preds, metrics = [], {}
            for name in ["mean_baseline", "song_baseline", *cfg.models]:
                spec = REGRESSORS[name]
                x_override = song_codes.reshape(-1, 1).astype(float) \
                    if name in SONG_ONLY_MODELS else None
                res = run_loso(ds, spec, task="regression", tune=cfg.tune_hyperparameters,
                               inner_splits=cfg.inner_cv_splits,
                               scoring="neg_mean_absolute_error", seed=cfg.random_seed,
                               x_override=x_override)
                preds.append(res.predictions)
                metrics[name] = regression_metrics(res.predictions, seed=cfg.random_seed)
            self.store.save_table(eid, "predictions", pd.concat(preds, ignore_index=True))
            self.store.save_json(eid, "metrics", metrics)
            record.metrics = {"rating_type": rating_type, **{
                n: {"mae": m["mae"], "rmse": m["rmse"]} for n, m in metrics.items()}}
            record.notes = [f"Rating type '{rating_type}' used exactly as defined in the "
                            "questionnaire; no mapping to other constructs."]
            self._finish(record)
        except Exception as exc:
            self._finish(record, exc)
            raise
        return record

    # --------------------------------------------------------------- clustering

    def run_clustering(self, cfg: ClusteringConfig | None = None) -> ExperimentRecord:
        cfg = cfg or self.ws.experiments.clustering
        trials = self._features("trial")
        valid = trials[trials["trial_valid"]].reset_index(drop=True)
        feats = feature_columns(valid, cfg.feature_prefixes)
        if len(valid) < 4 or not feats:
            raise ExperimentError("clustering needs at least 4 valid trials and matching features")
        record = self._begin(ExperimentType.CLUSTERING, cfg, valid, models=["pca", "kmeans"],
                             cv=None, seed=cfg.random_seed)
        eid = record.experiment_id
        try:
            res = run_clustering(valid, feats, cfg)
            self.store.save_table(eid, "pca_scores", res.pca.scores)
            self.store.save_table(eid, "pca_loadings", res.pca.loadings.reset_index(
                names="feature"))
            self.store.save_table(eid, "k_diagnostics", res.k_diagnostics)
            self.store.save_table(eid, "assignments", res.assignments)
            self.store.save_json(eid, "explained_variance",
                                 res.pca.explained_variance_ratio.tolist())
            record.metrics = {
                "explained_variance_ratio": res.pca.explained_variance_ratio.tolist(),
                "k_diagnostics": res.k_diagnostics.to_dict(orient="records"),
            }
            record.notes = res.notes
            self._finish(record)
        except Exception as exc:
            self._finish(record, exc)
            raise
        return record

    # ---------------------------------------------------------- synchronization

    def run_synchronization(self, song_id: str, cfg: SynchronizationConfig | None = None
                            ) -> ExperimentRecord:
        cfg = cfg or self.ws.experiments.synchronization
        recordings = self.ws.registry.analysable_recordings()
        avail = check_preconditions(recordings, song_id)
        if not avail.available:
            raise ExperimentError("synchronization analysis unavailable: "
                                  + " ".join(avail.reasons))
        results = [services.load_preprocessed(self.ws, rid) for rid in
                   avail.eligible_recording_ids]
        meta = pd.DataFrame([{"recording_id": r.recording_id, "participant_id": r.participant_id,
                              "song_id": r.song_id} for r in recordings
                             if r.recording_id in avail.eligible_recording_ids])
        record = self._begin(ExperimentType.SYNCHRONIZATION, cfg.model_dump(mode="json")
                             | {"song_id": song_id}, meta, models=["isc"],
                             cv="circular-shift surrogates", seed=cfg.random_seed)
        eid = record.experiment_id
        try:
            res = compute_isc(results, recordings, song_id, cfg)
            self.store.save_table(eid, "isc_per_channel", res.per_channel)
            self.store.save_table(eid, "isc_pairs", res.pairs)
            self.store.save_table(eid, "isc_sliding", res.sliding)
            record.metrics = {"song_id": song_id, "duration_s": res.duration_s,
                              "mean_isc": float(res.per_channel["isc"].mean()),
                              "n_channels_significant": int(
                                  (res.per_channel["p_adjusted"] < 0.05).sum())}
            record.notes = [*avail.reasons, *res.warnings,
                            "Correlated EEG time courses do not imply identical subjective "
                            "experience."]
            self._finish(record)
        except Exception as exc:
            self._finish(record, exc)
            raise
        return record

    # ---------------------------------------------------------- song similarity

    def run_song_similarity(self, cfg: SongSimilarityConfig | None = None) -> ExperimentRecord:
        """Compare songs with each other (audio, attribute table) and with EEG (RSA)."""
        cfg = cfg or self.ws.experiments.song_similarity
        trials = self._features("trial")
        valid = trials[trials["trial_valid"]].reset_index(drop=True)
        eeg_cols = feature_columns(valid, cfg.eeg_feature_prefixes)
        if not eeg_cols:
            raise ExperimentError("no EEG features match the configured prefixes")
        audio = StimulusAudio(load_catalogue(self.ws.metadata_dir),
                              self.ws.root / "data" / "stimuli" / "audio", cfg.sample_rate)
        songs_with_audio = audio.available(sorted(valid["song_id"].unique()))
        valid = valid[valid["song_id"].isin(songs_with_audio)].reset_index(drop=True)
        if valid["song_id"].nunique() < 3:
            raise ExperimentError("song similarity needs at least 3 songs with EEG and audio")
        record = self._begin(ExperimentType.SONG_SIMILARITY, cfg, valid,
                             models=["rsa_mantel"], cv=None, seed=cfg.random_seed)
        eid = record.experiment_id
        rng = np.random.default_rng(cfg.random_seed)
        try:
            notes: list[str] = []
            # ---- part 1: songs compared with each other
            catalogue_songs = audio.available(
                sorted({*songs_with_audio, *load_song_attributes(self.ws.metadata_dir).index}))
            song_audio = song_feature_table(audio, catalogue_songs, cfg.song_segment_seconds)
            acoustic = [c for c in song_audio.columns
                        if c.split(".")[0] in cfg.acoustic_groups]
            timbre = [c for c in song_audio.columns if c.startswith("timbre.")]
            song_rdms: dict[str, pd.DataFrame] = {
                "audio_acoustic": distance_matrix(song_audio[acoustic]),
                "audio_timbre": distance_matrix(song_audio[timbre]),
            }
            attrs = load_song_attributes(self.ws.metadata_dir)
            if not attrs.empty:
                in_both = [s for s in song_audio.index if s in attrs.index]
                proxy = [c for c in attribute_columns(attrs) if not c.endswith("_survey")]
                survey = [c for c in attribute_columns(attrs) if c.endswith("_survey")]
                if proxy:
                    song_rdms["table_proxies"] = distance_matrix(attrs.loc[in_both, proxy])
                if survey:
                    song_rdms["table_survey"] = distance_matrix(attrs.loc[in_both, survey])
                # The table describes entire songs, so check it against whole-song features.
                song_full = song_feature_table(audio, catalogue_songs, None)
                self.store.save_table(eid, "song_audio_features_full", song_full.reset_index())
                consistency = attribute_consistency(song_full, attrs)
                self.store.save_table(eid, "attribute_consistency", consistency)
                n_bad = int((consistency["status"] != "consistent").sum())
                if n_bad:
                    notes.append(
                        f"{n_bad} of {len(consistency)} song-attribute rows do not match the "
                        "supplied audio of the same song (several match another song's audio "
                        "almost exactly). Table-based comparisons are reported but flagged; "
                        "audio features computed from the exact stimulus files are primary.")
            self.store.save_table(eid, "song_audio_features", song_audio.reset_index())
            for name, m in song_rdms.items():
                self.store.save_table(eid, f"song_rdm_{name}", m.reset_index())
            self.store.save_table(eid, "song_neighbours_audio",
                                  nearest_neighbours(song_rdms["audio_acoustic"]))
            between_desc = []
            names = list(song_rdms)
            for i, a in enumerate(names):
                for b in names[i + 1:]:
                    common = [s for s in song_rdms[a].index if s in song_rdms[b].index]
                    res = mantel(song_rdms[a].loc[common, common],
                                 song_rdms[b].loc[common, common], cfg.n_permutations, rng)
                    between_desc.append({"rdm_a": a, "rdm_b": b, **res})
            self.store.save_table(eid, "song_description_agreement", pd.DataFrame(between_desc))

            # ---- part 2a: song-level RSA (song-mean EEG profiles)
            eeg_z = zscore(valid[eeg_cols]).assign(participant_id=valid["participant_id"],
                                                   song_id=valid["song_id"])
            per_participant = eeg_z.groupby(["participant_id", "song_id"]).mean(
                numeric_only=True)
            song_eeg = per_participant.groupby("song_id").mean()
            eeg_song_rdm = distance_matrix(song_eeg)
            self.store.save_table(eid, "song_rdm_eeg", eeg_song_rdm.reset_index())
            song_rsa = []
            for name, m in song_rdms.items():
                common = [s for s in eeg_song_rdm.index if s in m.index]
                res = mantel(m.loc[common, common], eeg_song_rdm.loc[common, common],
                             cfg.n_permutations, rng)
                song_rsa.append({"level": "song", "model_rdm": name, **res})

            # ---- part 2b: recording-level RSA (audio each participant actually heard)
            recs = {r.recording_id: r for r in self.ws.registry.analysable_recordings()}
            rec_audio = recording_feature_table(
                audio, [recs[rid] for rid in valid["recording_id"] if rid in recs])
            rec_ids = list(rec_audio.index)
            eeg_rec = zscore(valid.set_index("recording_id").loc[rec_ids, eeg_cols])
            eeg_rec_rdm = distance_matrix(eeg_rec)
            participants = rec_audio["participant_id"].to_numpy()
            songs = rec_audio["song_id"].to_numpy()
            mask = None
            if cfg.exclude_same_participant_pairs:
                mask = participants[:, None] != participants[None, :]
            rec_rdms = {
                "audio_acoustic": distance_matrix(rec_audio[acoustic].astype(float)),
                "audio_timbre": distance_matrix(rec_audio[timbre].astype(float)),
                "different_song": pd.DataFrame((songs[:, None] != songs[None, :]).astype(float),
                                               index=rec_ids, columns=rec_ids),
                "duration_difference": pd.DataFrame(
                    np.abs(rec_audio["heard_s"].to_numpy(float)[:, None]
                           - rec_audio["heard_s"].to_numpy(float)[None, :]),
                    index=rec_ids, columns=rec_ids),
            }
            rec_rsa = [{"level": "recording", "model_rdm": name,
                        **mantel(m, eeg_rec_rdm, cfg.n_permutations, rng, mask)}
                       for name, m in rec_rdms.items()]
            rsa = pd.DataFrame(song_rsa + rec_rsa)
            if not rsa.empty:
                ok = rsa["p_value"].notna()
                rsa["p_adjusted"] = np.nan
                rsa.loc[ok, "p_adjusted"] = multipletests(rsa.loc[ok, "p_value"],
                                                          method="holm")[1]
            self.store.save_table(eid, "rsa", rsa)
            pairs = pair_table({"eeg": eeg_rec_rdm, **rec_rdms}, mask)
            pairs["song_a"] = pairs["item_a"].map(rec_audio["song_id"])
            pairs["song_b"] = pairs["item_b"].map(rec_audio["song_id"])
            self.store.save_table(eid, "recording_pairs", pairs)
            self.store.save_table(eid, "recording_audio_features", rec_audio.reset_index())

            # ---- part 2c: which audio features track which EEG features (exploratory)
            assoc = []
            for af in acoustic:
                for ef in eeg_cols:
                    res = stats.spearmanr(rec_audio[af].astype(float),
                                          valid.set_index("recording_id").loc[rec_ids, ef])
                    assoc.append({"audio_feature": af, "eeg_feature": ef, "n": len(rec_ids),
                                  "rho": float(res.statistic), "p_value": float(res.pvalue)})
            assoc_df = pd.DataFrame(assoc)
            ok = assoc_df["p_value"].notna()
            assoc_df["p_adjusted"] = np.nan
            assoc_df.loc[ok, "p_adjusted"] = multipletests(assoc_df.loc[ok, "p_value"],
                                                           method="fdr_bh")[1]
            self.store.save_table(eid, "audio_eeg_associations",
                                  assoc_df.sort_values("p_value").reset_index(drop=True))

            notes += [
                "RSA: Spearman correlation between dissimilarity matrices; Mantel permutation "
                f"test ({cfg.n_permutations} permutations); Holm correction across RSA tests.",
                "Song-level EEG profiles average different participants per song, so song and "
                "participant differences are confounded at that level.",
                "Recording-level RSA uses the audio segment each participant heard"
                + ("; pairs from the same participant are excluded."
                   if cfg.exclude_same_participant_pairs else "."),
                "Audio-EEG feature associations are exploratory (FDR-corrected).",
            ]
            record.metrics = {
                "n_songs_compared": len(song_audio),
                "n_songs_with_eeg": len(eeg_song_rdm),
                "n_recordings": len(rec_ids),
                "n_recording_pairs": int(rsa.loc[rsa["level"] == "recording", "n_pairs"].max()),
                "rsa": rsa[["level", "model_rdm", "rho", "p_value", "p_adjusted"]]
                .to_dict(orient="records"),
                "n_audio_eeg_associations_significant": int((assoc_df["p_adjusted"] < 0.05)
                                                            .sum()),
            }
            record.notes = notes
            self._finish(record)
        except Exception as exc:
            self._finish(record, exc)
            raise
        return record
