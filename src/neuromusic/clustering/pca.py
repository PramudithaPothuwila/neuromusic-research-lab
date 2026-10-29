"""Exploratory PCA on standardised trial-level features."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler


@dataclass
class PCAResult:
    scores: pd.DataFrame  # one row per trial, PC columns + identifiers
    loadings: pd.DataFrame  # features x PCs
    explained_variance_ratio: np.ndarray
    pipeline: Pipeline


def preprocess_pipeline() -> Pipeline:
    """Median imputation + robust scaling (median/IQR), resistant to outlying trials."""
    return Pipeline([("impute", SimpleImputer(strategy="median")),
                     ("scale", RobustScaler())])


def fit_pca(trials: pd.DataFrame, features: list[str], n_components: int,
            random_state: int = 42) -> PCAResult:
    x = trials[features].to_numpy(dtype=float)
    n_components = max(1, min(n_components, x.shape[0] - 1, x.shape[1]))
    pipe = Pipeline([*preprocess_pipeline().steps,
                     ("pca", PCA(n_components=n_components, random_state=random_state))])
    scores = pipe.fit_transform(x)
    pcs = [f"PC{i + 1}" for i in range(n_components)]
    score_df = pd.DataFrame(scores, columns=pcs)
    for col in ("recording_id", "participant_id", "song_id", "trial_id"):
        score_df[col] = trials[col].to_numpy()
    pca: PCA = pipe.named_steps["pca"]
    loadings = pd.DataFrame(pca.components_.T, index=features, columns=pcs)
    return PCAResult(score_df, loadings, pca.explained_variance_ratio_, pipe)
