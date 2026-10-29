"""K-Means and optional hierarchical clustering on standardised features (exploratory).

The number of clusters is never chosen from the number of songs; a range of k is evaluated and
reported with silhouette and stability diagnostics.
"""

from __future__ import annotations

import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage
from sklearn.cluster import KMeans


def fit_kmeans(x: np.ndarray, k: int, random_state: int) -> np.ndarray:
    return KMeans(n_clusters=k, n_init=20, random_state=random_state).fit_predict(x)


def fit_hierarchical(x: np.ndarray, k: int, method: str = "ward") -> tuple[np.ndarray, np.ndarray]:
    """Returns (labels 0..k-1, linkage matrix)."""
    z = linkage(x, method=method)
    return fcluster(z, t=k, criterion="maxclust") - 1, z
