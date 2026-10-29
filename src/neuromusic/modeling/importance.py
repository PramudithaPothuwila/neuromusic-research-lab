"""Permutation importance evaluated on held-out participants only."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance


def fold_importance(estimator, x_test: np.ndarray, y_test: np.ndarray, feature_names: list[str],
                    n_repeats: int, seed: int, scoring: str) -> pd.DataFrame:
    """Importance of each feature for a model fitted on training participants, measured as the
    score drop on the held-out participant's observations."""
    res = permutation_importance(estimator, x_test, y_test, n_repeats=n_repeats,
                                 random_state=seed, scoring=scoring)
    return pd.DataFrame({"feature": feature_names, "importance_mean": res.importances_mean,
                         "importance_sd": res.importances_std})


def aggregate_importance(fold_frames: list[pd.DataFrame]) -> pd.DataFrame:
    if not fold_frames:
        return pd.DataFrame(columns=["feature", "importance_mean", "importance_sd_across_folds",
                                     "n_folds"])
    stacked = pd.concat(fold_frames, ignore_index=True)
    out = (stacked.groupby("feature")["importance_mean"]
           .agg(importance_mean="mean", importance_sd_across_folds="std", n_folds="count")
           .reset_index())
    return out.sort_values("importance_mean", ascending=False).reset_index(drop=True)
