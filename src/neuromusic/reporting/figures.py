"""Static (Matplotlib) figures for exported reports."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SYNTHETIC_LABEL = "SYNTHETIC DATA - NOT STUDY RESULTS"


def _finish(fig: plt.Figure, path: Path, dpi: int, synthetic: bool) -> Path:
    if synthetic:
        fig.text(0.5, 0.5, SYNTHETIC_LABEL, ha="center", va="center", fontsize=16,
                 color="red", alpha=0.25, rotation=25, transform=fig.transFigure)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi)
    plt.close(fig)
    return path


def recordings_per_song(counts: pd.DataFrame, path: Path, dpi: int, synthetic: bool) -> Path:
    """``counts`` columns: song (title), n_recordings, n_participants."""
    fig, ax = plt.subplots(figsize=(8, max(3, 0.3 * len(counts) + 1)))
    data = counts.sort_values("n_recordings")
    ax.barh(data["song"], data["n_recordings"], color="#4C72B0")
    ax.set_xlabel("Analysable independent recordings")
    ax.set_title("Recordings per song (after provenance audit)")
    return _finish(fig, path, dpi, synthetic)


def confusion_matrix(matrix: list[list[int]], classes: list[str], title: str, path: Path,
                     dpi: int, synthetic: bool) -> Path:
    cm = np.asarray(matrix, dtype=float)
    size = max(4, 0.45 * len(classes) + 2)
    fig, ax = plt.subplots(figsize=(size, size))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(classes)), classes, rotation=60, ha="right")
    ax.set_yticks(range(len(classes)), classes)
    ax.set_xlabel("Predicted song")
    ax.set_ylabel("True song")
    ax.set_title(title)
    if len(classes) <= 12:
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, str(int(cm[i, j])), ha="center", va="center",
                        color="white" if cm[i, j] > cm.max() / 2 else "black")
    fig.colorbar(im, ax=ax, label="Held-out predictions")
    return _finish(fig, path, dpi, synthetic)


def model_comparison(metrics: pd.DataFrame, path: Path, dpi: int, synthetic: bool) -> Path:
    """``metrics`` columns: model, balanced_accuracy, ci_low, ci_high, chance."""
    fig, ax = plt.subplots(figsize=(7, 4))
    x = np.arange(len(metrics))
    err = np.vstack([metrics["balanced_accuracy"] - metrics["ci_low"],
                     metrics["ci_high"] - metrics["balanced_accuracy"]]).clip(min=0)
    ax.bar(x, metrics["balanced_accuracy"], yerr=err, capsize=4, color="#55A868")
    ax.axhline(float(metrics["chance"].iloc[0]), color="grey", ls="--", label="Uniform chance")
    ax.set_xticks(x, metrics["model"], rotation=20)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Balanced accuracy (held-out participants)")
    ax.set_title("LOSO model comparison (95% participant-bootstrap CI)")
    ax.legend()
    return _finish(fig, path, dpi, synthetic)


def feature_by_song(trials: pd.DataFrame, feature: str, unit: str, path: Path, dpi: int,
                    synthetic: bool, titles: dict[str, str] | None = None) -> Path:
    titles = titles or {}
    songs = sorted(trials["song_id"].unique())
    fig, ax = plt.subplots(figsize=(max(6, 0.5 * len(songs) + 2), 4))
    data = [trials.loc[trials["song_id"] == s, feature].dropna().to_numpy() for s in songs]
    ax.boxplot(data, showfliers=False)
    for i, values in enumerate(data, start=1):
        ax.scatter(np.full(values.size, i) + np.random.default_rng(i).uniform(-0.1, 0.1,
                                                                               values.size),
                   values, s=12, color="#C44E52", zorder=3)
    ax.set_xticks(range(1, len(songs) + 1), [titles.get(s, s) for s in songs], rotation=60,
                  ha="right")
    ax.set_ylabel(f"{feature} ({unit})" if unit else feature)
    ax.set_title(f"{feature} by song (one point per trial)")
    return _finish(fig, path, dpi, synthetic)


def pca_scatter(scores: pd.DataFrame, explained: list[float], path: Path, dpi: int,
                synthetic: bool) -> Path:
    fig, ax = plt.subplots(figsize=(6.5, 5))
    for song, grp in scores.groupby("song_id"):
        ax.scatter(grp["PC1"], grp["PC2"] if "PC2" in grp else np.zeros(len(grp)), label=song,
                   s=25)
    ax.set_xlabel(f"PC1 ({explained[0]:.1%} variance)")
    if len(explained) > 1:
        ax.set_ylabel(f"PC2 ({explained[1]:.1%} variance)")
    ax.set_title("Exploratory PCA of trial-level features")
    if scores["song_id"].nunique() <= 20:
        ax.legend(fontsize=7, ncol=2, loc="best")
    return _finish(fig, path, dpi, synthetic)
