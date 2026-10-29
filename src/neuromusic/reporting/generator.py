"""Research report generation (Markdown, HTML, CSV, JSON, PNG).

The report is assembled from persisted artifacts only (registry, feature manifest, experiment
records). Nothing is recomputed or invented; missing analyses are reported as missing.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
from jinja2 import Environment, FileSystemLoader, select_autoescape

from neuromusic import __version__, services
from neuromusic.analysis import descriptive
from neuromusic.experiments.persistence import ExperimentStore
from neuromusic.experiments.registry import ExperimentRecord, environment_info
from neuromusic.features.extractor import feature_unit
from neuromusic.reporting import exports, figures
from neuromusic.workspace import Workspace

TEMPLATE_DIR = Path(__file__).parent / "templates"


@dataclass
class Table:
    caption: str
    frame: pd.DataFrame
    name: str


@dataclass
class Figure:
    caption: str
    path: str  # relative to the report directory


@dataclass
class Section:
    title: str
    paragraphs: list[str] = field(default_factory=list)
    bullets: list[str] = field(default_factory=list)
    tables: list[Table] = field(default_factory=list)
    figures: list[Figure] = field(default_factory=list)


def _latest(records: list[ExperimentRecord], etype: str) -> ExperimentRecord | None:
    return next((r for r in records if r.experiment_type.value == etype
                 and r.status.value == "completed"), None)


def _select_experiments(store: ExperimentStore, ids: list[str] | None
                        ) -> list[ExperimentRecord]:
    if ids:
        return [store.load_record(i) for i in ids]
    records = store.list_records()
    chosen = [_latest(records, t) for t in
              ("statistics", "classification", "clustering", "regression", "synchronization",
               "song_similarity")]
    return [r for r in chosen if r is not None]


class ReportBuilder:
    def __init__(self, ws: Workspace, out_dir: Path) -> None:
        self.ws = ws
        self.out = out_dir
        self.store = ExperimentStore(ws.experiments_dir)
        self.dpi = ws.application.figure_dpi
        self.titles = services.song_titles(ws)
        self.synthetic = False
        self.payload: dict[str, Any] = {}

    def _table(self, caption: str, frame: pd.DataFrame, name: str) -> Table:
        exports.save_csv(frame, self.out / "tables" / f"{name}.csv")
        return Table(caption, frame, name)

    def _figure(self, caption: str, path: Path) -> Figure:
        return Figure(caption, path.relative_to(self.out).as_posix())

    # ------------------------------------------------------------------ sections

    def dataset_section(self) -> Section:
        reg = self.ws.registry
        recs = reg.recordings_frame()
        overview = descriptive.dataset_overview(recs)
        analysable = [r for r in reg.analysable_recordings()]
        self.synthetic = any(r.is_synthetic for r in analysable)
        manifest = reg.manifest_frame(self.titles)
        prov = manifest["provenance"].value_counts().rename_axis("provenance").reset_index(
            name="n_files") if not manifest.empty else pd.DataFrame()
        self.payload["dataset"] = {**overview, "provenance": prov.to_dict(orient="records"),
                                   "fingerprint": reg.fingerprint()}
        sec = Section("Dataset summary")
        sec.bullets = [
            f"Files registered: {len(manifest)} (incl. byte-identical copies)",
            f"Analysable independent recordings: {overview.get('n_analysable', 0)}",
            f"Participants: {overview.get('n_participants', 0)}; songs: "
            f"{overview.get('n_songs', 0)}",
            f"Total analysable duration: {overview.get('total_duration_s', 0.0) / 60:.1f} min",
            f"Sampling rate(s): {overview.get('sampling_rates', [])} Hz",
            f"Dataset fingerprint: `{reg.fingerprint()}`",
        ]
        if self.synthetic:
            sec.paragraphs.append("**WARNING: synthetic recordings are included in the "
                                  "analysable set. Results are not study findings.**")
        sec.tables.append(self._table("File provenance after the near-duplicate audit", prov,
                                      "provenance"))
        rec_ok = recs[recs["recording_id"].isin([r.recording_id for r in analysable])]
        per_song = descriptive.participants_per_song(rec_ok)
        if not per_song.empty:
            per_song.insert(1, "song", per_song["song_id"].map(lambda s: self.titles.get(s, s)))
            sec.tables.append(self._table("Analysable recordings per song", per_song,
                                          "participants_per_song"))
            path = figures.recordings_per_song(per_song.rename(columns={}), self.out / "figures"
                                               / "recordings_per_song.png", self.dpi,
                                               self.synthetic)
            sec.figures.append(self._figure("Analysable independent recordings per song.",
                                            path))
        per_part = descriptive.recordings_per_participant(rec_ok)
        if not per_part.empty:
            dist = per_part["n_songs"].value_counts().sort_index().rename_axis(
                "songs_per_participant").reset_index(name="n_participants")
            sec.tables.append(self._table("Songs heard per participant (analysable)", dist,
                                          "songs_per_participant"))
        exports.save_csv(manifest, self.out / "tables" / "recording_manifest.csv")
        return sec

    def methodology_section(self) -> Section:
        p = self.ws.preprocessing
        f = self.ws.features
        sec = Section("Methodology")
        sec.bullets = [
            "Raw CSVs are immutable; validation checks channels, numeric content, sample-index "
            "monotonicity, duplicates, timestamp gaps, detected sampling rate, flatlines and "
            "extreme excursions.",
            "Near-duplicate recordings were detected by correlating first-difference signal "
            "signatures; only one member of each signal family is retained, and families with "
            "conflicting song labels are excluded entirely.",
            f"Preprocessing: median DC removal, notch {'on' if p.notch.enabled else 'off'}, "
            f"zero-phase FIR band-pass {p.bandpass.low_hz}-{p.bandpass.high_hz} Hz "
            f"({'on' if p.bandpass.enabled else 'off'}), reference "
            f"`{p.reference.method.value}`, ICA {'on' if p.ica.enabled else 'off'}.",
            f"Windows: {p.windows.duration_seconds} s with {p.windows.overlap_seconds} s overlap;"
            " artifact flags are data-driven (robust z-scores) because units are unconfirmed; "
            "automatic rejection "
            f"{'on' if p.artifact_handling.automatic_rejection_enabled else 'off'}.",
            "Spectral features: Welch PSD; band power integrated over half-open bands "
            + ", ".join(f"{k} [{a}, {b}) Hz" for k, (a, b) in f.frequency_bands.items())
            + f"; relative power to {f.total_power_range[0]}-{f.total_power_range[1]} Hz.",
            f"Trial features: {f.trial_aggregation} over valid windows; trials with fewer than "
            f"{f.min_valid_windows_per_trial} valid windows are excluded.",
            "Statistics use participants as the unit of analysis (within-participant tests on "
            "complete cases; mixed models with participant random intercepts).",
            "Classification uses leave-one-subject-out validation with all transforms and "
            "tuning fitted on training participants only.",
        ]
        cfg = pd.DataFrame([
            {"setting": k, "value": str(v)} for k, v in
            pd.json_normalize(p.model_dump(mode="json"), sep=".").iloc[0].items()])
        sec.tables.append(self._table("Preprocessing configuration", cfg,
                                      "preprocessing_config"))
        self.payload["preprocessing_config"] = p.model_dump(mode="json")
        self.payload["feature_config"] = f.model_dump(mode="json")
        return sec

    def statistics_section(self, rec: ExperimentRecord) -> Section:
        eid = rec.experiment_id
        sec = Section(f"Descriptive and inferential statistics (`{eid}`)")
        sec.paragraphs.extend(rec.notes)
        comp = self.store.load_table(eid, "comparisons") if self.store.has(eid, "comparisons") \
            else pd.DataFrame()
        if comp.empty:
            sec.paragraphs.append("No cross-song comparison could be computed (insufficient "
                                  "participants with multiple songs).")
        else:
            cols = [c for c in ("feature", "test", "n_participants", "statistic", "p_value",
                                "p_adjusted", "effect_size", "effect_size_name", "estimate",
                                "ci_low", "ci_high") if c in comp.columns]
            prim = comp[comp["primary"]][cols]
            sec.tables.append(self._table("Primary features: within-participant comparison "
                                          "across songs", prim, "comparisons_primary"))
            exports.save_csv(comp, self.out / "tables" / "comparisons_all.csv")
            if not prim.empty:
                trials = services.load_features(self.ws)
                feat = str(prim.iloc[0]["feature"])
                if trials is not None and feat in trials.columns:
                    path = figures.feature_by_song(
                        trials[trials["trial_valid"]], feat,
                        feature_unit(feat, self.ws.signal_unit_label),
                        self.out / "figures" / "top_primary_feature.png", self.dpi,
                        self.synthetic, self.titles)
                    sec.figures.append(self._figure(
                        f"Distribution of `{feat}` by song (smallest primary p-value). "
                        "Exploratory display; see adjusted p-values.", path))
        if self.store.has(eid, "mixed_effects"):
            me = self.store.load_table(eid, "mixed_effects")
            keep = [c for c in ("feature", "status", "n_observations", "n_participants",
                                "lr_statistic", "df", "p_value", "p_adjusted", "converged",
                                "singular_fit") if c in me.columns]
            sec.tables.append(self._table("Mixed-effects models (song fixed effect, "
                                          "participant random intercept; LR test)",
                                          me[keep], "mixed_effects"))
        if self.store.has(eid, "song_attribute_associations"):
            assoc = self.store.load_table(eid, "song_attribute_associations")
            sec.tables.append(self._table(
                "Exploratory song-level associations between song-mean EEG features and song "
                "attributes (n = songs)", assoc.head(20), "song_attribute_associations"))
        self.payload["statistics"] = rec.metrics
        return sec

    def classification_section(self, rec: ExperimentRecord) -> Section:
        eid = rec.experiment_id
        sec = Section(f"Song classification, leave-one-subject-out (`{eid}`)")
        sec.paragraphs.extend(rec.notes)
        metrics = self.store.load_json(eid, "metrics")
        dataset = self.store.load_json(eid, "dataset") if self.store.has(eid, "dataset") else {}
        rows = []
        for name, m in metrics.items():
            ci = m.get("balanced_accuracy_ci", [float("nan")] * 2)
            rows.append({"model": name, "accuracy": m["accuracy"],
                         "balanced_accuracy": m["balanced_accuracy"], "ci_low": ci[0],
                         "ci_high": ci[1], "macro_f1": m["macro_f1"],
                         "mean_train_accuracy": m.get("mean_train_accuracy"),
                         "permutation_p": m.get("permutation_test", {}).get("p_value"),
                         "chance": m["chance_level_uniform"]})
        table = pd.DataFrame(rows)
        sec.bullets = [f"Observations: {dataset.get('n_observations')} "
                       f"({dataset.get('level')}-level); participants: "
                       f"{dataset.get('n_participants')}; classes: "
                       f"{len(dataset.get('classes', []))}",
                       f"Class support: {dataset.get('class_support')}"]
        sec.tables.append(self._table("Held-out performance (training accuracy shown "
                                      "separately for reference only)", table,
                                      "classification_metrics"))
        if not table.empty:
            path = figures.model_comparison(table, self.out / "figures" /
                                            "model_comparison.png", self.dpi, self.synthetic)
            sec.figures.append(self._figure("Balanced accuracy on held-out participants.", path))
            learned = table[table["model"] != "dummy"]
            if not learned.empty:
                best = str(learned.sort_values("balanced_accuracy").iloc[-1]["model"])
                m = metrics[best]
                path = figures.confusion_matrix(
                    m["confusion_matrix"], [self.titles.get(c, c) for c in m["classes"]],
                    f"Confusion matrix: {best} (held-out)",
                    self.out / "figures" / "confusion_matrix.png", self.dpi, self.synthetic)
                sec.figures.append(self._figure(f"Held-out confusion matrix for {best}.", path))
        self.payload["classification"] = metrics
        return sec

    def clustering_section(self, rec: ExperimentRecord) -> Section:
        eid = rec.experiment_id
        sec = Section(f"Exploratory unsupervised analysis (`{eid}`)")
        sec.paragraphs.extend(rec.notes)
        diag = self.store.load_table(eid, "k_diagnostics")
        sec.tables.append(self._table("K-Means diagnostics", diag, "k_diagnostics"))
        explained = self.store.load_json(eid, "explained_variance")
        scores = self.store.load_table(eid, "pca_scores")
        path = figures.pca_scatter(scores, explained, self.out / "figures" / "pca.png",
                                   self.dpi, self.synthetic)
        sec.figures.append(self._figure("PCA of trial-level features coloured by song "
                                        "(exploratory).", path))
        self.payload["clustering"] = rec.metrics
        return sec

    def regression_section(self, rec: ExperimentRecord) -> Section:
        sec = Section(f"Rating prediction (`{rec.experiment_id}`)")
        sec.paragraphs.extend(rec.notes)
        metrics = self.store.load_json(rec.experiment_id, "metrics")
        sec.tables.append(self._table("LOSO regression error", pd.DataFrame(
            [{"model": k, "mae": v["mae"], "rmse": v["rmse"]} for k, v in metrics.items()]),
            "regression_metrics"))
        self.payload["regression"] = metrics
        return sec

    def synchronization_section(self, rec: ExperimentRecord) -> Section:
        sec = Section(f"Inter-subject correlation (`{rec.experiment_id}`)")
        sec.paragraphs.extend(rec.notes)
        sec.tables.append(self._table("Per-channel ISC with circular-shift surrogate tests",
                                      self.store.load_table(rec.experiment_id,
                                                            "isc_per_channel"), "isc"))
        self.payload["synchronization"] = rec.metrics
        return sec

    def song_similarity_section(self, rec: ExperimentRecord) -> Section:
        eid = rec.experiment_id
        sec = Section(f"Song similarity and EEG (`{eid}`)")
        sec.paragraphs.extend(rec.notes)
        for name, caption in (
                ("rsa", "Representational similarity: model dissimilarity vs EEG dissimilarity"),
                ("song_description_agreement", "Agreement between song descriptions"),
                ("song_neighbours_audio", "Acoustically nearest songs"),
                ("attribute_consistency", "Attribute table rows checked against the audio"),
                ("audio_eeg_associations", "Audio feature vs EEG feature (exploratory, top 20)")):
            if self.store.has(eid, name):
                frame = self.store.load_table(eid, name)
                sec.tables.append(self._table(caption, frame.head(20) if name.startswith(
                    "audio_eeg") else frame, f"song_similarity_{name}"))
        self.payload["song_similarity"] = rec.metrics
        return sec

    def limitations_section(self) -> Section:
        ws = self.ws
        acq = ws.dataset.acquisition
        sec = Section("Limitations")
        items = []
        if not ws.units_validated:
            items.append("Signal units and scaling are unconfirmed; amplitudes and powers are "
                         "in arbitrary units and no microvolt thresholds were applied.")
        if acq.acquisition_reference is None:
            items.append("The acquisition reference is unknown; asymmetry and regional "
                         "measures depend on it.")
        if acq.recording_device is None:
            items.append("The recording device is unverified (the montage resembles a "
                         "14-channel consumer headset).")
        if not acq.montage_positions_verified:
            items.append("Electrode positions are not verified; no topographic maps shown.")
        manifest = ws.registry.manifest_frame(self.titles)
        n_bad = int(manifest["provenance"].isin(["derived_copy", "identical_copy",
                                                 "label_conflict"]).sum()) \
            if not manifest.empty else 0
        if n_bad:
            items.append(f"{n_bad} file(s) were copies of other recordings (some relabelled as "
                         "different participants or songs) and were excluded. Data-collection "
                         "provenance should be confirmed with the data owner.")
        if not any(r.alignment_verified for r in ws.registry.analysable_recordings()):
            items.append("Playback-relative timing is unverified, so inter-subject "
                         "synchronization was not analysed.")
        items += [
            "Presentation order, familiarity, listening volume, fatigue and baseline state "
            "were not recorded and may confound song differences.",
            "Hundreds of windows per trial are not independent observations; inference uses "
            "participants (or trials) as the unit.",
            "Song classification across the studied stimuli does not demonstrate "
            "generalisation to unseen songs.",
            "Spectral differences are not direct measures of emotion; song attributes are "
            "song-level aggregates, not participants' own ratings.",
        ]
        sec.bullets = items
        return sec

    def reproducibility_section(self, records: list[ExperimentRecord]) -> Section:
        env = environment_info()
        sec = Section("Reproducibility")
        sec.bullets = [
            f"Software: neuromusic {__version__}; Python {env['python']}; {env['platform']}",
            "Packages: " + ", ".join(f"{k} {v}" for k, v in env["packages"].items()),
            f"Preprocessing hash `{self.ws.preprocessing_hash}`; feature hash "
            f"`{self.ws.feature_hash}`; dataset fingerprint "
            f"`{self.ws.registry.fingerprint()}`",
            *[f"Experiment `{r.experiment_id}` ({r.experiment_type.value}), seed "
              f"{r.random_seed}, CV: {r.cv_strategy or 'n/a'}" for r in records],
        ]
        self.payload["environment"] = env
        self.payload["experiments"] = [r.model_dump(mode="json") for r in records]
        return sec


def generate_report(ws: Workspace, experiment_ids: list[str] | None = None,
                    formats: tuple[str, ...] = ("md", "html")) -> Path:
    """Write a report directory and return its path. Existing reports are never overwritten."""
    now = datetime.now()
    out = ws.reports_dir / f"report-{now:%Y%m%d-%H%M%S}"
    suffix = 1
    while out.exists():
        out = ws.reports_dir / f"report-{now:%Y%m%d-%H%M%S}-{suffix}"
        suffix += 1
    (out / "tables").mkdir(parents=True)
    (out / "figures").mkdir()
    builder = ReportBuilder(ws, out)
    records = _select_experiments(builder.store, experiment_ids)
    sections = [builder.dataset_section(), builder.methodology_section()]
    handlers = {"statistics": builder.statistics_section,
                "classification": builder.classification_section,
                "clustering": builder.clustering_section,
                "regression": builder.regression_section,
                "synchronization": builder.synchronization_section,
                "song_similarity": builder.song_similarity_section}
    included = []
    for rec in records:
        if rec.status.value != "completed":
            continue
        sections.append(handlers[rec.experiment_type.value](rec))
        included.append(rec)
        builder.synthetic = builder.synthetic or rec.is_synthetic_data
    if not included:
        sections.append(Section("Results", paragraphs=[
            "No completed experiments were found; only dataset and methodology sections are "
            "included."]))
    sections += [builder.limitations_section(), builder.reproducibility_section(included)]

    context = {"title": ws.application.project_name + " - research report",
               "generated": now.isoformat(timespec="seconds"), "sections": sections,
               "synthetic": builder.synthetic, "md_table": exports.markdown_table,
               "html_table": exports.html_table}
    env = Environment(loader=FileSystemLoader(TEMPLATE_DIR),
                      autoescape=select_autoescape(enabled_extensions=("html.j2",),
                                                   default_for_string=False))
    if "md" in formats:
        (out / "report.md").write_text(env.get_template("report.md.j2").render(**context),
                                       encoding="utf-8")
    if "html" in formats:
        (out / "report.html").write_text(env.get_template("report.html.j2").render(**context),
                                         encoding="utf-8")
    exports.save_json({"generated": context["generated"], "synthetic": builder.synthetic,
                       **builder.payload}, out / "results.json")
    return out
