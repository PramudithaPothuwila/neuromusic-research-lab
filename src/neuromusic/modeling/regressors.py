"""Regressor registry for optional rating prediction."""

from __future__ import annotations

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.svm import SVR

from neuromusic.modeling.baselines import GroupMeanRegressor, mean_regressor
from neuromusic.modeling.classifiers import ModelSpec, _pipeline

REGRESSORS: dict[str, ModelSpec] = {
    "mean_baseline": ModelSpec("mean_baseline", lambda s: _pipeline(mean_regressor(), False), {},
                               "Training-mean baseline"),
    "song_baseline": ModelSpec("song_baseline", lambda s: GroupMeanRegressor(), {},
                               "Per-song training mean (song identity only)"),
    "ridge": ModelSpec("ridge", lambda s: _pipeline(Ridge(alpha=1.0)),
                       {"model__alpha": [0.1, 1.0, 10.0, 100.0]}, "Ridge regression"),
    "svr": ModelSpec("svr", lambda s: _pipeline(SVR(C=1.0)),
                     {"model__C": [0.1, 1.0, 10.0], "model__epsilon": [0.1, 0.5]},
                     "Support vector regression"),
    "random_forest": ModelSpec(
        "random_forest",
        lambda s: _pipeline(RandomForestRegressor(n_estimators=300, random_state=s, n_jobs=1),
                            False),
        {"model__max_depth": [None, 3, 6]}, "Random forest regression"),
}

# Models that receive the song label (as the final column) instead of EEG features.
SONG_ONLY_MODELS = {"song_baseline"}
