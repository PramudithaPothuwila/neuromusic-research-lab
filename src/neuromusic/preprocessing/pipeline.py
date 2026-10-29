"""Configurable preprocessing pipeline with persisted provenance.

Order: channel selection -> sampling-rate confirmation -> analysis span (trim / playback onset)
-> DC removal -> notch -> band-pass -> re-reference -> optional ICA -> artifact detection ->
segmentation -> quality metrics. The raw recording is read-only; outputs are written to
``<interim>/cleaned/<config_hash>/``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import mne
import numpy as np
import pandas as pd

from neuromusic import PIPELINE_VERSION
from neuromusic.config import DatasetConfig, PreprocessingConfig, config_hash
from neuromusic.domain.enums import ExecutionStatus, QualityStatus
from neuromusic.domain.models import ProcessingRun, Recording
from neuromusic.ingestion.csv_reader import RawEEG, read_eeg_csv, select_channels
from neuromusic.ingestion.validator import detect_sampling_rate
from neuromusic.preprocessing.artifacts import detect_artifacts, flags_to_frame
from neuromusic.preprocessing.filtering import bandpass, notch, remove_dc_offset
from neuromusic.preprocessing.quality import channel_quality, recording_quality_summary
from neuromusic.preprocessing.referencing import rereference
from neuromusic.preprocessing.segmentation import assign_window_quality, window_bounds


class PreprocessingError(RuntimeError):
    pass


@dataclass
class PreprocessingResult:
    recording_id: str
    sfreq: float
    channel_names: list[str]
    times: np.ndarray  # seconds, relative to the start of the raw recording
    raw_centered: np.ndarray
    cleaned: np.ndarray
    windows: pd.DataFrame
    flags: pd.DataFrame
    channel_quality: pd.DataFrame
    run: ProcessingRun


def preprocessing_hash(prep: PreprocessingConfig, dataset: DatasetConfig) -> str:
    """Hash of everything that changes preprocessing output."""
    return config_hash(prep, dataset.acquisition, length=12)


def resolve_sampling_rate(raw: RawEEG, prep: PreprocessingConfig) -> float:
    detected = detect_sampling_rate(raw.times)
    if prep.sampling_rate == "auto":
        if detected is None:
            raise PreprocessingError("sampling rate cannot be detected from timestamps")
        # Round to the nearest 1/1000 Hz to avoid floating noise from timestamp formatting.
        return round(detected, 3)
    configured = float(prep.sampling_rate)
    if detected is not None and abs(detected - configured) > 0.5:
        raise PreprocessingError(
            f"configured sampling rate {configured} Hz conflicts with timestamps "
            f"({detected:.3f} Hz)")
    return configured


def _apply_ica(data: np.ndarray, sfreq: float, channel_names: list[str],
               prep: PreprocessingConfig) -> np.ndarray:
    info = mne.create_info(channel_names, sfreq, ch_types="eeg")
    raw = mne.io.RawArray(data, info, verbose=False)
    ica = mne.preprocessing.ICA(n_components=prep.ica.n_components, method=prep.ica.method,
                                random_state=prep.ica.random_state, verbose=False)
    ica.fit(raw, verbose=False)
    ica.exclude = list(prep.ica.exclude_components)
    return ica.apply(raw.copy(), verbose=False).get_data()


def run_preprocessing(
    recording: Recording,
    raw_path: Path,
    prep: PreprocessingConfig,
    dataset: DatasetConfig,
) -> PreprocessingResult:
    if recording.participant_id is None or recording.song_id is None:
        raise PreprocessingError(f"{recording.recording_id}: participant/song metadata missing")
    raw = read_eeg_csv(raw_path, dataset.channel_prefix)
    if not np.isfinite(raw.data).all():
        raise PreprocessingError(f"{recording.recording_id}: non-finite samples; resolve "
                                 "validation errors before preprocessing")
    channels = [c for c in dataset.channels if c in raw.channel_names]
    if not channels:
        raise PreprocessingError(f"{recording.recording_id}: none of the configured channels")
    raw = select_channels(raw, channels)
    sfreq = resolve_sampling_rate(raw, prep)

    # Analysis span. Times are expressed relative to the first sample of the recording.
    times = raw.times - raw.times[0]
    onset = recording.playback_onset_s or 0.0
    start_s = max(prep.trim_start_seconds, onset)
    stop_s = times[-1] + 1.0 / sfreq - prep.trim_end_seconds
    start = int(np.searchsorted(times, start_s - 0.5 / sfreq))
    stop = int(np.searchsorted(times, stop_s - 0.5 / sfreq))
    if stop - start < int(prep.windows.duration_seconds * sfreq):
        raise PreprocessingError(f"{recording.recording_id}: analysis span shorter than one window")
    data = raw.data[:, start:stop]
    seg_times = times[start:stop]
    t0 = float(seg_times[0])

    raw_centered = remove_dc_offset(data) if prep.remove_dc_offset else data.copy()
    cleaned = notch(raw_centered, sfreq, prep.notch)
    cleaned = bandpass(cleaned, sfreq, prep.bandpass)
    cleaned = rereference(cleaned, channels, prep.reference)
    if prep.ica.enabled:
        cleaned = _apply_ica(cleaned, sfreq, channels, prep)

    flags = flags_to_frame(detect_artifacts(
        raw_centered, cleaned, sfreq, channels, prep.artifact_handling, time_offset=t0))

    bounds = window_bounds(cleaned.shape[1], sfreq, prep.windows)
    statuses, n_flagged = assign_window_quality(
        bounds, sfreq, flags,
        reject_if_channels_ge=prep.artifact_handling.reject_window_if_flagged_channels_ge,
        automatic_rejection=prep.artifact_handling.automatic_rejection_enabled,
        time_offset=t0,
    )
    windows = pd.DataFrame({
        "recording_id": recording.recording_id,
        "participant_id": recording.participant_id,
        "song_id": recording.song_id,
        "trial_id": recording.trial_id,
        "window_id": np.arange(len(bounds), dtype=np.int64),
        "start_time": t0 + bounds[:, 0] / sfreq if len(bounds) else np.empty(0),
        "end_time": t0 + bounds[:, 1] / sfreq if len(bounds) else np.empty(0),
        "start_sample": bounds[:, 0],
        "stop_sample": bounds[:, 1],
        "quality_status": [s.value for s in statuses],
        "n_flagged_channels": n_flagged,
    })
    if prep.baseline.enabled:
        in_baseline = (windows["start_time"] >= prep.baseline.start_s) & (
            windows["end_time"] <= prep.baseline.end_s)
        windows["segment"] = np.where(in_baseline, "baseline", "stimulus")
    else:
        windows["segment"] = "stimulus"

    duration = cleaned.shape[1] / sfreq
    quality = channel_quality(raw_centered, cleaned, sfreq, channels, flags, duration,
                              prep.artifact_handling.line_noise_hz)
    phash = preprocessing_hash(prep, dataset)
    run = ProcessingRun(
        processing_id=f"PROC-{recording.recording_id}-{phash}",
        recording_id=recording.recording_id,
        configuration_hash=phash,
        pipeline_version=PIPELINE_VERSION,
        quality_results={
            **recording_quality_summary(windows, quality, flags),
            "sampling_rate": sfreq,
            "analysis_start_s": t0,
            "analysis_duration_s": duration,
            "n_rejected_windows": int((windows["quality_status"]
                                       == QualityStatus.REJECTED.value).sum()),
        },
        execution_status=ExecutionStatus.COMPLETED,
    )
    return PreprocessingResult(
        recording_id=recording.recording_id, sfreq=sfreq, channel_names=channels,
        times=seg_times, raw_centered=raw_centered, cleaned=cleaned, windows=windows,
        flags=flags, channel_quality=quality, run=run,
    )


# --------------------------------------------------------------------- persistence


class PreprocessedStore:
    """Read/write preprocessed outputs under ``<interim>/cleaned/<hash>/``."""

    def __init__(self, interim_dir: Path, prep_hash: str) -> None:
        self.root = Path(interim_dir) / "cleaned" / prep_hash
        self.prep_hash = prep_hash

    def _path(self, recording_id: str, suffix: str) -> Path:
        return self.root / f"{recording_id}{suffix}"

    def exists(self, recording_id: str) -> bool:
        return self._path(recording_id, "_run.json").exists()

    def save(self, result: PreprocessingResult, prep: PreprocessingConfig,
             save_signals: bool = True) -> Path:
        self.root.mkdir(parents=True, exist_ok=True)
        rid = result.recording_id
        if save_signals:
            np.savez_compressed(self._path(rid, "_signals.npz"), cleaned=result.cleaned,
                                raw_centered=result.raw_centered, times=result.times,
                                sfreq=np.array(result.sfreq),
                                channel_names=np.array(result.channel_names))
        result.windows.to_parquet(self._path(rid, "_windows.parquet"), index=False)
        result.flags.to_parquet(self._path(rid, "_flags.parquet"), index=False)
        result.channel_quality.to_parquet(self._path(rid, "_quality.parquet"), index=False)
        result.run.artifact_path = str(self.root)
        self._path(rid, "_run.json").write_text(result.run.model_dump_json(indent=2),
                                                encoding="utf-8")
        config_file = self.root / "preprocessing_config.json"
        if not config_file.exists():
            config_file.write_text(json.dumps(prep.model_dump(mode="json"), indent=2),
                                   encoding="utf-8")
        return self.root

    def load_run(self, recording_id: str) -> ProcessingRun:
        return ProcessingRun.model_validate_json(
            self._path(recording_id, "_run.json").read_text(encoding="utf-8"))

    def load_table(self, recording_id: str, name: str) -> pd.DataFrame:
        return pd.read_parquet(self._path(recording_id, f"_{name}.parquet"))

    def load_signals(self, recording_id: str) -> dict[str, np.ndarray]:
        path = self._path(recording_id, "_signals.npz")
        if not path.exists():
            raise FileNotFoundError(f"cleaned signals not saved for {recording_id}")
        with np.load(path, allow_pickle=False) as npz:
            return {k: npz[k] for k in npz.files}

    def load_result(self, recording_id: str) -> PreprocessingResult:
        sig = self.load_signals(recording_id)
        return PreprocessingResult(
            recording_id=recording_id, sfreq=float(sig["sfreq"]),
            channel_names=[str(c) for c in sig["channel_names"]], times=sig["times"],
            raw_centered=sig["raw_centered"], cleaned=sig["cleaned"],
            windows=self.load_table(recording_id, "windows"),
            flags=self.load_table(recording_id, "flags"),
            channel_quality=self.load_table(recording_id, "quality"),
            run=self.load_run(recording_id),
        )

    def processed_ids(self) -> list[str]:
        if not self.root.exists():
            return []
        return sorted(p.name.removesuffix("_run.json") for p in self.root.glob("*_run.json"))
