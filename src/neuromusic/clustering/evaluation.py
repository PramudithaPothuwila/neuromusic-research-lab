"""Cluster diagnostics and the exploratory clustering workflow."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score, silhouette_score

from neuromusic.clustering.kmeans import fit_hierarchical, fit_kmeans
from neuromusic.clustering.pca import PCAResult, fit_pca, preprocess_pipeline
from neuromusic.config import ClusteringConfig


@dataclass
class ClusteringResult:
    pca: PCAResult
    k_diagnostics: pd.DataFrame
    assignments: pd.DataFrame  # identifiers + cluster label per k
    song_contingency: dict[int, pd.DataFrame]
    participant_contingency: dict[int, pd.DataFrame]
    linkage: np.ndarray | None
    notes: list[str]


def bootstrap_stability(x: np.ndarray, k: int, n_runs: int, seed: int) -> float:
    """Mean ARI between the full-data solution and solutions on bootstrap resamples,
    compared on the resampled points."""
    if n_runs <= 0:
        return float("nan")
    rng = np.random.default_rng(seed)
    reference = fit_kmeans(x, k, seed)
    aris = []
    for r in range(n_runs):
        idx = rng.choice(len(x), size=len(x), replace=True)
        if len(np.unique(idx)) <= k:
            continue
        labels = fit_kmeans(x[idx], k, seed + r + 1)
        aris.append(adjusted_rand_score(reference[idx], labels))
    return float(np.mean(aris)) if aris else float("nan")


def run_clustering(trials: pd.DataFrame, features: list[str], cfg: ClusteringConfig
                   ) -> ClusteringResult:
    notes = ["Exploratory analysis: clusters are not emotional states and are not evidence of "
             "out-of-sample predictive performance."]
    data = trials.reset_index(drop=True)
    pca = fit_pca(data, features, cfg.n_components, cfg.random_seed)
    x = preprocess_pipeline().fit_transform(data[features].to_numpy(dtype=float))
    rows = []
    assignments = data[["recording_id", "participant_id", "song_id", "trial_id"]].copy()
    song_ct: dict[int, pd.DataFrame] = {}
    part_ct: dict[int, pd.DataFrame] = {}
    valid_k = [k for k in cfg.k_values if 2 <= k < len(data)]
    if len(valid_k) < len(cfg.k_values):
        notes.append(f"k values {sorted(set(cfg.k_values) - set(valid_k))} skipped: need "
                     "2 <= k < number of trials.")
    for k in valid_k:
        labels = fit_kmeans(x, k, cfg.random_seed)
        sizes = np.bincount(labels, minlength=k)
        sil = silhouette_score(x, labels) if len(np.unique(labels)) > 1 else np.nan
        rows.append({
            "k": k,
            "silhouette": float(sil),
            "inertia": float(
                sum(((x[labels == c] - x[labels == c].mean(axis=0)) ** 2).sum()
                    for c in range(k) if (labels == c).any())),
            "min_cluster_size": int(sizes.min()),
            "max_cluster_size": int(sizes.max()),
            "stability_ari": bootstrap_stability(x, k, cfg.n_stability_runs, cfg.random_seed),
            "ari_vs_song": float(adjusted_rand_score(data["song_id"], labels)),
            "ari_vs_participant": float(adjusted_rand_score(data["participant_id"], labels)),
        })
        assignments[f"cluster_k{k}"] = labels
        song_ct[k] = pd.crosstab(pd.Series(labels, name="cluster"), data["song_id"])
        part_ct[k] = pd.crosstab(pd.Series(labels, name="cluster"), data["participant_id"])
    link = None
    if cfg.hierarchical_enabled and valid_k:
        for k in valid_k:
            labels, link = fit_hierarchical(x, k)
            assignments[f"hier_k{k}"] = labels
    notes.append("ari_vs_participant > ari_vs_song indicates that structure reflects individual "
                 "differences more than stimuli.")
    return ClusteringResult(pca, pd.DataFrame(rows), assignments, song_ct, part_ct, link, notes)
