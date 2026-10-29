"""Aggregate window-level features to one row per participant-song trial.

Windows from one trial are not independent observations; trial-level rows are the default unit
of analysis for statistics and machine learning.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from neuromusic.config import FeatureConfig
from neuromusic.domain.enums import QualityStatus
from neuromusic.features.extractor import feature_columns

TRIAL_KEYS = ["recording_id", "participant_id", "song_id", "trial_id"]


def valid_window_mask(windows: pd.DataFrame) -> pd.Series:
    mask = windows["quality_status"] != QualityStatus.REJECTED.value
    if "segment" in windows.columns:
        mask &= windows["segment"] == "stimulus"
    return mask


def aggregate_trials(window_features: pd.DataFrame, cfg: FeatureConfig) -> pd.DataFrame:
    if window_features.empty:
        return pd.DataFrame(columns=[*TRIAL_KEYS, "n_windows_total", "n_windows_valid",
                                     "trial_valid"])
    feats = feature_columns(window_features)
    valid = window_features[valid_window_mask(window_features)]
    totals = window_features.groupby("recording_id").size().rename("n_windows_total")
    grouped = valid.groupby(TRIAL_KEYS, sort=True)
    central = (grouped[feats].median() if cfg.trial_aggregation == "median"
               else grouped[feats].mean())
    parts = [central]
    var_feats = [c for c in feats
                 if any(c.startswith(f"{fam}.") for fam in cfg.variability_families)]
    if var_feats:
        for stat in cfg.trial_variability:
            if stat == "std":
                spread = grouped[var_feats].std(ddof=1)
            else:
                spread = grouped[var_feats].quantile(0.75) - grouped[var_feats].quantile(0.25)
            spread.columns = [f"var_{stat}.{c}" for c in var_feats]
            parts.append(spread)
    trial = pd.concat(parts, axis=1).reset_index()
    trial["n_windows_valid"] = grouped.size().to_numpy()
    trial = trial.merge(totals, left_on="recording_id", right_index=True, how="left")
    trial["trial_valid"] = trial["n_windows_valid"] >= cfg.min_valid_windows_per_trial
    ordered = [*TRIAL_KEYS, "n_windows_total", "n_windows_valid", "trial_valid"]
    rest = [c for c in trial.columns if c not in ordered]
    out = trial[ordered + rest]
    # Recordings with zero valid windows still appear, marked invalid.
    missing = set(window_features["recording_id"]) - set(out["recording_id"])
    if missing:
        stub = (window_features[window_features["recording_id"].isin(missing)]
                .groupby(TRIAL_KEYS).size().rename("n_windows_total").reset_index())
        stub["n_windows_valid"] = 0
        stub["trial_valid"] = False
        out = pd.concat([out, stub], ignore_index=True)
        for c in rest:
            out[c] = out[c].astype(float)
    out[rest] = out[rest].replace([np.inf, -np.inf], np.nan)
    return out.sort_values(TRIAL_KEYS).reset_index(drop=True)
