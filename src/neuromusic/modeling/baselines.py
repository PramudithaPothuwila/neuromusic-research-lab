"""Reference models every learned model must be compared against."""

from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.dummy import DummyClassifier, DummyRegressor


def dummy_classifier(random_state: int = 42) -> DummyClassifier:
    """Predicts the most frequent training class (chance-level reference)."""
    return DummyClassifier(strategy="most_frequent", random_state=random_state)


def mean_regressor() -> DummyRegressor:
    """Predicts the training-set mean rating."""
    return DummyRegressor(strategy="mean")


class GroupMeanRegressor(RegressorMixin, BaseEstimator):
    """Predicts the training mean of the target for each song (song-identity-only baseline).

    The song label is passed as the *last* column of X (see ``modeling.regressors``) and is
    ignored by EEG models; unseen songs fall back to the overall training mean.
    """

    def fit(self, x: np.ndarray, y: np.ndarray) -> GroupMeanRegressor:
        keys = x[:, -1]
        self.means_ = {k: float(np.mean(y[keys == k])) for k in np.unique(keys)}
        self.overall_ = float(np.mean(y))
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        return np.array([self.means_.get(k, self.overall_) for k in x[:, -1]])
