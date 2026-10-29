"""Evaluation metrics computed from held-out predictions."""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    precision_recall_fscore_support,
    root_mean_squared_error,
)


def _participant_bootstrap(pred: pd.DataFrame, func, n_boot: int, seed: int,
                           alpha: float = 0.05) -> tuple[float, float]:
    """Percentile CI resampling participants (the independent unit) with replacement."""
    rng = np.random.default_rng(seed)
    groups = {p: g for p, g in pred.groupby("participant_id")}
    ids = list(groups)
    stats = []
    with warnings.catch_warnings():
        # Resamples often lack some classes; sklearn warns on every such resample.
        warnings.simplefilter("ignore", UserWarning)
        for _ in range(n_boot):
            sample = pd.concat([groups[ids[i]] for i in rng.integers(0, len(ids), len(ids))])
            try:
                stats.append(func(sample))
            except ValueError:
                continue
    if not stats:
        return (float("nan"), float("nan"))
    return (float(np.quantile(stats, alpha / 2)), float(np.quantile(stats, 1 - alpha / 2)))


def classification_metrics(pred: pd.DataFrame, classes: list[str], n_boot: int = 1000,
                           seed: int = 42) -> dict[str, Any]:
    """``pred`` needs columns participant_id, y_true, y_pred (held-out only)."""
    y_true, y_pred = pred["y_true"].astype(str), pred["y_pred"].astype(str)
    prec, rec, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=classes, zero_division=0)
    per_participant = (pred.assign(correct=y_true == y_pred)
                       .groupby("participant_id")["correct"].mean())

    def bal(df: pd.DataFrame) -> float:
        return balanced_accuracy_score(df["y_true"].astype(str), df["y_pred"].astype(str))

    return {
        "n_predictions": len(pred),
        "n_participants": int(pred["participant_id"].nunique()),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=classes, average="macro",
                                   zero_division=0)),
        "balanced_accuracy_ci": list(_participant_bootstrap(pred, bal, n_boot, seed)),
        "chance_level_uniform": 1.0 / len(classes),
        "classes": classes,
        "per_class": [
            {"class": c, "precision": float(p), "recall": float(r), "f1": float(f),
             "support": int(s)}
            for c, p, r, f, s in zip(classes, prec, rec, f1, support, strict=True)
        ],
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=classes).tolist(),
        "participant_accuracy": {str(k): float(v) for k, v in per_participant.items()},
        "participant_accuracy_mean": float(per_participant.mean()),
        "participant_accuracy_sd": float(per_participant.std(ddof=1))
        if len(per_participant) > 1 else float("nan"),
    }


def regression_metrics(pred: pd.DataFrame, n_boot: int = 1000, seed: int = 42) -> dict[str, Any]:
    y_true, y_pred = pred["y_true"].astype(float), pred["y_pred"].astype(float)

    def mae(df: pd.DataFrame) -> float:
        return mean_absolute_error(df["y_true"], df["y_pred"])

    per_participant = pd.Series({
        pid: mean_absolute_error(g["y_true"], g["y_pred"])
        for pid, g in pred.groupby("participant_id")})
    return {
        "n_predictions": len(pred),
        "n_participants": int(pred["participant_id"].nunique()),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(root_mean_squared_error(y_true, y_pred)),
        "mae_ci": list(_participant_bootstrap(pred, mae, n_boot, seed)),
        "participant_mae": {str(k): float(v) for k, v in per_participant.items()},
    }
