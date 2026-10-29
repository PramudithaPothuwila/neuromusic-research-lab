"""Leave-One-Subject-Out (LOSO) evaluation with nested, group-aware tuning.

For every fold one participant is held out. Imputation, scaling and hyperparameter search are
fitted on the remaining participants only (inner GroupKFold), then the refitted model predicts
the held-out participant. Window-level experiments keep every window of the held-out participant
out of training and additionally report trial-level majority-vote predictions.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import GridSearchCV, GroupKFold, LeaveOneGroupOut

from neuromusic.modeling.classifiers import ModelSpec
from neuromusic.modeling.datasets import ModelingDataset
from neuromusic.modeling.importance import aggregate_importance, fold_importance


@dataclass
class LOSOResult:
    model: str
    predictions: pd.DataFrame
    folds: pd.DataFrame
    importance: pd.DataFrame
    trial_predictions: pd.DataFrame | None = None
    permutation: dict[str, Any] = field(default_factory=dict)


def _fit_fold(spec: ModelSpec, x_tr: np.ndarray, y_tr: np.ndarray, g_tr: np.ndarray,
              tune: bool, inner_splits: int, scoring: str, seed: int, n_jobs: int
              ) -> tuple[Any, dict[str, Any], float]:
    estimator = spec.factory(seed)
    n_groups = len(np.unique(g_tr))
    if tune and spec.param_grid and n_groups >= 2 and len(np.unique(y_tr)) >= 2:
        search = GridSearchCV(estimator, spec.param_grid,
                              cv=GroupKFold(n_splits=min(inner_splits, n_groups)),
                              scoring=scoring, n_jobs=n_jobs, refit=True, error_score=np.nan)
        search.fit(x_tr, y_tr, groups=g_tr)
        return search.best_estimator_, dict(search.best_params_), float(search.best_score_)
    estimator.fit(x_tr, y_tr)
    return estimator, {}, float("nan")


def run_loso(
    ds: ModelingDataset,
    spec: ModelSpec,
    *,
    task: str = "classification",
    tune: bool = True,
    inner_splits: int = 3,
    scoring: str = "balanced_accuracy",
    seed: int = 42,
    importance_repeats: int = 0,
    n_jobs: int = 1,
    x_override: np.ndarray | None = None,
) -> LOSOResult:
    """Run LOSO for one model. ``x_override`` replaces features (song-only baselines)."""
    x = ds.x if x_override is None else x_override
    pred_rows: list[dict[str, Any]] = []
    fold_rows: list[dict[str, Any]] = []
    imp_frames: list[pd.DataFrame] = []
    imp_scoring = "accuracy" if task == "classification" else "neg_mean_absolute_error"
    for fold, (tr, te) in enumerate(LeaveOneGroupOut().split(x, ds.y, ds.groups)):
        held_out = str(ds.groups[te][0])
        if held_out in set(ds.groups[tr]):
            raise RuntimeError("participant leakage across LOSO split")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            est, params, inner = _fit_fold(spec, x[tr], ds.y[tr], ds.groups[tr], tune,
                                           inner_splits, scoring, seed, n_jobs)
        y_hat = est.predict(x[te])
        train_hat = est.predict(x[tr])
        if task == "classification":
            train_score = float(np.mean(train_hat == ds.y[tr]))
            test_score = float(np.mean(y_hat == ds.y[te]))
        else:
            train_score = float(np.mean(np.abs(train_hat - ds.y[tr].astype(float))))
            test_score = float(np.mean(np.abs(y_hat - ds.y[te].astype(float))))
        fold_rows.append({
            "fold": fold, "held_out_participant": held_out, "n_train": len(tr),
            "n_test": len(te), "n_train_participants": len(np.unique(ds.groups[tr])),
            "train_score": train_score, "test_score": test_score,
            "inner_cv_score": inner, "best_params": params,
        })
        meta = ds.meta.iloc[te]
        for i, (yt, yp) in enumerate(zip(ds.y[te], y_hat, strict=True)):
            row = meta.iloc[i].to_dict()
            row.update({"fold": fold, "y_true": yt, "y_pred": yp, "model": spec.name})
            pred_rows.append(row)
        if importance_repeats > 0 and x_override is None and spec.name not in (
                "dummy", "mean_baseline"):
            frame = fold_importance(est, x[te], ds.y[te], ds.feature_names,
                                    importance_repeats, seed, imp_scoring)
            frame["fold"] = fold
            imp_frames.append(frame)

    predictions = pd.DataFrame(pred_rows)
    trial_predictions = None
    if ds.level == "window" and task == "classification":
        trial_predictions = (
            predictions.groupby(["recording_id", "participant_id"])
            .agg(y_true=("y_true", "first"),
                 y_pred=("y_pred", lambda s: s.value_counts().idxmax()),
                 n_windows=("y_pred", "size"))
            .reset_index())
    return LOSOResult(spec.name, predictions, pd.DataFrame(fold_rows),
                      aggregate_importance(imp_frames), trial_predictions)


def permute_within_participants(y: np.ndarray, groups: np.ndarray, trial_ids: np.ndarray,
                                rng: np.random.Generator) -> np.ndarray:
    """Shuffle labels among each participant's trials, keeping all windows of a trial together.

    Preserves each participant's label set and the trial grouping of windows."""
    y_perm = y.copy()
    for g in np.unique(groups):
        idx = np.flatnonzero(groups == g)
        trials = pd.unique(trial_ids[idx])
        labels = {t: y[idx][trial_ids[idx] == t][0] for t in trials}
        shuffled = rng.permutation([labels[t] for t in trials])
        mapping = dict(zip(trials, shuffled, strict=True))
        y_perm[idx] = [mapping[t] for t in trial_ids[idx]]
    return y_perm


def permutation_test(ds: ModelingDataset, spec: ModelSpec, observed: float, n_permutations: int,
                     *, tune: bool, inner_splits: int, scoring: str, seed: int, n_jobs: int = 1
                     ) -> dict[str, Any]:
    """Null distribution of LOSO balanced accuracy under within-participant label permutation.
    The full pipeline (including tuning) is rerun for every permutation."""
    rng = np.random.default_rng(seed)
    perms = [permute_within_participants(ds.y, ds.groups, ds.trial_ids, rng)
             for _ in range(n_permutations)]

    def one(y_perm: np.ndarray) -> float:
        perm_ds = ModelingDataset(ds.x, y_perm, ds.groups, ds.trial_ids, ds.feature_names,
                                  ds.meta, ds.level)
        res = run_loso(perm_ds, spec, tune=tune, inner_splits=inner_splits, scoring=scoring,
                       seed=seed)
        return float(balanced_accuracy_score(res.predictions["y_true"].astype(str),
                                             res.predictions["y_pred"].astype(str)))

    null = Parallel(n_jobs=n_jobs)(delayed(one)(p) for p in perms)
    null_arr = np.asarray(null)
    return {
        "n_permutations": n_permutations,
        "observed_balanced_accuracy": observed,
        "null_mean": float(null_arr.mean()),
        "null_95th": float(np.quantile(null_arr, 0.95)),
        "p_value": float((1 + (null_arr >= observed - 1e-12).sum()) / (1 + n_permutations)),
        "null_distribution": null_arr.tolist(),
        "scheme": "labels permuted among each participant's trials; windows of a trial "
                  "move together",
    }

