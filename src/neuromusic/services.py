"""High-level data-pipeline services shared by the CLI and the dashboard."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import pandas as pd

from neuromusic.features.aggregation import aggregate_trials
from neuromusic.features.extractor import extract_window_features
from neuromusic.features.store import FeatureStore
from neuromusic.ingestion.provenance import DEFAULT_THRESHOLD
from neuromusic.ingestion.song_attributes import load_song_attributes
from neuromusic.preprocessing.pipeline import (
    PreprocessedStore,
    PreprocessingError,
    PreprocessingResult,
    run_preprocessing,
)
from neuromusic.workspace import Workspace

Progress = Callable[[int, int, str], None] | None


@dataclass(frozen=True)
class StepResult:
    recording_id: str
    status: str  # "ok" | "skipped" | "failed"
    message: str


MANIFEST_FILE = "recording_manifest.csv"


def song_titles(ws: Workspace) -> dict[str, str]:
    attrs = load_song_attributes(ws.metadata_dir)
    return {} if attrs.empty else {str(k): str(v) for k, v in attrs["song_name"].items()}


def sync_song_metadata(ws: Workspace) -> None:
    """Use titles and attributes from song_attributes.csv for registered songs."""
    attrs = load_song_attributes(ws.metadata_dir)
    if attrs.empty:
        return
    for song_id, song in ws.registry.state.songs.items():
        if song_id in attrs.index:
            row = attrs.loc[song_id]
            song.song_name = str(row["song_name"])
            song.source_metadata = {str(k): (None if pd.isna(v) else v)
                                    for k, v in row.drop("song_name").items()}
    ws.registry.save()


def import_and_audit(ws: Workspace, threshold: float = DEFAULT_THRESHOLD,
                     exclude_synthetic: bool = True) -> pd.DataFrame:
    """Scan the raw directory, run the provenance audit, sync song metadata, write manifest."""
    ws.registry.scan_directory()
    ws.registry.audit_provenance(threshold, exclude_synthetic=exclude_synthetic)
    sync_song_metadata(ws)
    return write_manifest(ws)


def write_manifest(ws: Workspace) -> pd.DataFrame:
    manifest = ws.registry.manifest_frame(song_titles(ws))
    manifest.to_csv(ws.metadata_dir / MANIFEST_FILE, index=False)
    return manifest


def preprocessed_store(ws: Workspace) -> PreprocessedStore:
    return PreprocessedStore(ws.interim_dir, ws.preprocessing_hash)


def feature_store(ws: Workspace) -> FeatureStore:
    return FeatureStore(ws.processed_dir)


def preprocess_one(ws: Workspace, recording_id: str) -> PreprocessingResult:
    """Run preprocessing for one recording in memory (nothing is persisted)."""
    rec = ws.registry.recordings[recording_id]
    return run_preprocessing(rec, ws.registry.resolve_path(rec), ws.preprocessing, ws.dataset)


def preprocess_recordings(ws: Workspace, recording_ids: list[str] | None = None,
                          force: bool = False, progress: Progress = None) -> list[StepResult]:
    store = preprocessed_store(ws)
    recs = ws.registry.analysable_recordings()
    if recording_ids is not None:
        recs = [r for r in recs if r.recording_id in set(recording_ids)]
    results = []
    for i, rec in enumerate(recs):
        if progress:
            progress(i, len(recs), rec.recording_id)
        if store.exists(rec.recording_id) and not force:
            results.append(StepResult(rec.recording_id, "skipped", "already processed"))
            continue
        try:
            res = run_preprocessing(rec, ws.registry.resolve_path(rec), ws.preprocessing,
                                    ws.dataset)
            store.save(res, ws.preprocessing, save_signals=ws.preprocessing.save_cleaned_data)
            q = res.run.quality_results
            results.append(StepResult(rec.recording_id, "ok",
                                      f"{q['n_windows']} windows, {q['n_flags']} flags"))
        except (PreprocessingError, ValueError, FileNotFoundError) as exc:
            results.append(StepResult(rec.recording_id, "failed", str(exc)))
    if progress:
        progress(len(recs), len(recs), "done")
    return results


def load_preprocessed(ws: Workspace, recording_id: str) -> PreprocessingResult:
    store = preprocessed_store(ws)
    try:
        return store.load_result(recording_id)
    except FileNotFoundError:
        return preprocess_one(ws, recording_id)


def build_features(ws: Workspace, progress: Progress = None) -> FeatureStore:
    """Preprocess any missing recordings, extract window features, aggregate to trials, save."""
    preprocess_recordings(ws, progress=progress)
    store = preprocessed_store(ws)
    recs = [r for r in ws.registry.analysable_recordings() if store.exists(r.recording_id)]
    frames = []
    for i, rec in enumerate(recs):
        if progress:
            progress(i, len(recs), f"features {rec.recording_id}")
        frames.append(extract_window_features(load_preprocessed(ws, rec.recording_id),
                                              ws.features))
    if not frames:
        raise ValueError("no preprocessed recordings available for feature extraction")
    window = pd.concat(frames, ignore_index=True)
    trial = aggregate_trials(window, ws.features)
    fstore = feature_store(ws)
    fstore.save(ws.preprocessing_hash, ws.feature_hash, window, trial, {
        "dataset_fingerprint": ws.registry.fingerprint(),
        "recording_ids": [r.recording_id for r in recs],
        "is_synthetic_data": any(r.is_synthetic for r in recs),
        "signal_units": ws.signal_unit_label,
        "preprocessing_config": ws.preprocessing.model_dump(mode="json"),
        "feature_config": ws.features.model_dump(mode="json"),
        "n_windows": len(window),
        "n_trials": len(trial),
        "n_valid_trials": int(trial["trial_valid"].sum()),
    })
    return fstore


def features_available(ws: Workspace) -> bool:
    return feature_store(ws).exists(ws.preprocessing_hash, ws.feature_hash)


def features_current(ws: Workspace) -> bool:
    """True when stored features match the current configs and dataset fingerprint."""
    if not features_available(ws):
        return False
    manifest = feature_store(ws).manifest(ws.preprocessing_hash, ws.feature_hash)
    return manifest.get("dataset_fingerprint") == ws.registry.fingerprint()


def load_features(ws: Workspace, level: str = "trial") -> pd.DataFrame | None:
    if not features_available(ws):
        return None
    return feature_store(ws).load(ws.preprocessing_hash, ws.feature_hash, level)


def quality_tables(ws: Workspace) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(channel quality, artifact flags) across all processed recordings."""
    store = preprocessed_store(ws)
    quality, flags = [], []
    for rid in store.processed_ids():
        if rid not in ws.registry.recordings:
            continue
        quality.append(store.load_table(rid, "quality").assign(recording_id=rid))
        f = store.load_table(rid, "flags")
        if not f.empty:
            flags.append(f.assign(recording_id=rid))
    return (pd.concat(quality, ignore_index=True) if quality else pd.DataFrame(),
            pd.concat(flags, ignore_index=True) if flags else pd.DataFrame())
