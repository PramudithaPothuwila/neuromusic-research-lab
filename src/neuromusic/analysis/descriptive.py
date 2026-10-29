"""Descriptive summaries of the dataset, windows and features."""

from __future__ import annotations

from typing import Any

import pandas as pd

from neuromusic.domain.enums import QualityStatus


def dataset_overview(recordings: pd.DataFrame) -> dict[str, Any]:
    """Counts over registered recordings (``DatasetRegistry.recordings_frame``)."""
    if recordings.empty:
        return {"n_recordings": 0, "n_participants": 0, "n_songs": 0, "total_duration_s": 0.0,
                "n_analysable": 0, "n_synthetic": 0}
    analysable = recordings[
        (~recordings["excluded"])
        & recordings["participant_id"].notna()
        & recordings["song_id"].notna()
        & recordings["validation_status"].isin(["passed", "passed_with_warnings"])
    ]
    return {
        "n_recordings": len(recordings),
        "n_analysable": len(analysable),
        "n_participants": int(analysable["participant_id"].nunique()),
        "n_songs": int(analysable["song_id"].nunique()),
        "total_duration_s": float(analysable["duration_seconds"].sum()),
        "n_failed_validation": int((recordings["validation_status"] == "failed").sum()),
        "n_excluded": int(recordings["excluded"].sum()),
        "n_synthetic": int(recordings["is_synthetic"].sum()),
        "sampling_rates": sorted({round(float(x), 3)
                                  for x in analysable["sampling_rate"].dropna()}),
    }


def participants_per_song(recordings: pd.DataFrame) -> pd.DataFrame:
    rec = recordings.dropna(subset=["participant_id", "song_id"])
    return (rec.groupby("song_id")
            .agg(n_participants=("participant_id", "nunique"),
                 n_recordings=("recording_id", "size"),
                 mean_duration_s=("duration_seconds", "mean"))
            .reset_index())


def recordings_per_participant(recordings: pd.DataFrame) -> pd.DataFrame:
    rec = recordings.dropna(subset=["participant_id"])
    return (rec.groupby("participant_id")
            .agg(n_recordings=("recording_id", "size"), n_songs=("song_id", "nunique"),
                 total_duration_s=("duration_seconds", "sum"))
            .reset_index())


def window_summary(window_features: pd.DataFrame) -> pd.DataFrame:
    """Valid / flagged / rejected window counts per recording."""
    if window_features.empty:
        return pd.DataFrame()
    counts = (window_features
              .groupby(["recording_id", "participant_id", "song_id", "trial_id"])["quality_status"]
              .value_counts().unstack(fill_value=0).reset_index())
    for status in QualityStatus:
        if status.value not in counts.columns:
            counts[status.value] = 0
    counts["n_windows"] = counts[[s.value for s in QualityStatus]].sum(axis=1)
    counts["n_valid"] = counts["n_windows"] - counts[QualityStatus.REJECTED.value]
    return counts


def feature_summary_by_song(trials: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    """Mean, median, SD and participant count per song for each feature (long format)."""
    long = trials.melt(id_vars=["participant_id", "song_id"], value_vars=features,
                       var_name="feature", value_name="value")
    return (long.groupby(["feature", "song_id"])["value"]
            .agg(n="count", mean="mean", median="median", sd="std")
            .reset_index())


def variability_summary(trials: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    """Between-participant SD (per song) and between-song SD (per participant), averaged."""
    rows = []
    for feat in features:
        pivot = trials.pivot_table(index="participant_id", columns="song_id", values=feat,
                                   aggfunc="mean")
        rows.append({
            "feature": feat,
            "between_participant_sd": float(pivot.std(axis=0, ddof=1).mean()),
            "between_song_sd": float(pivot.std(axis=1, ddof=1).mean()),
        })
    return pd.DataFrame(rows)


def channel_quality_summary(quality: pd.DataFrame) -> pd.DataFrame:
    """Aggregate per-recording channel quality tables (with a ``recording_id`` column)."""
    if quality.empty:
        return quality
    return (quality.groupby("channel")
            .agg(n_recordings=("recording_id", "nunique"),
                 median_robust_sd=("robust_sd", "median"),
                 median_line_noise=("line_noise_fraction", "median"),
                 median_high_freq=("high_freq_fraction", "median"),
                 mean_flagged_fraction=("flagged_fraction", "mean"),
                 n_review=("quality", lambda s: int((s == "review").sum())))
            .reset_index())
