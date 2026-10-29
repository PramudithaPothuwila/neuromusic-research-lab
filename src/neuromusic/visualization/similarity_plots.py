"""Figures for song-to-song similarity and RSA."""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go

from neuromusic.visualization import SEQUENTIAL, finalize


def plot_rdm(rdm: pd.DataFrame, title: str, labels: dict[str, str] | None = None, *,
             synthetic: bool = False) -> go.Figure:
    """Heatmap of a square dissimilarity matrix (rows/columns in the given order)."""
    labels = labels or {}
    names = [labels.get(str(i), str(i)) for i in rdm.index]
    fig = go.Figure(go.Heatmap(
        z=rdm.to_numpy(), x=names, y=names, colorscale=SEQUENTIAL, reversescale=True,
        colorbar={"title": "distance"},
        hovertemplate="%{y} vs %{x}<br>distance %{z:.2f}<extra></extra>"))
    fig.update_yaxes(autorange="reversed")
    return finalize(fig, title, synthetic=synthetic, height=520,
                    caption="Euclidean distance between z-scored descriptors; darker = more "
                            "similar.")


def plot_rsa_bars(rsa: pd.DataFrame, *, synthetic: bool = False) -> go.Figure:
    """Spearman rho of each model dissimilarity matrix with the EEG matrix."""
    data = rsa.assign(label=rsa["level"] + ": " + rsa["model_rdm"])
    colors = ["#2a6fdb" if p < 0.05 else "#9aa5b1" for p in data["p_value"].fillna(1.0)]
    fig = go.Figure(go.Bar(
        x=data["rho"], y=data["label"], orientation="h", marker_color=colors,
        customdata=data[["p_value", "p_adjusted", "n_pairs"]].to_numpy(),
        hovertemplate="rho %{x:.3f}<br>p %{customdata[0]:.4f} (Holm %{customdata[1]:.4f})"
                      "<br>pairs %{customdata[2]}<extra></extra>"))
    fig.add_vline(x=0, line_color="#5f6b7a")
    fig.update_xaxes(title="Spearman rho with EEG dissimilarity")
    return finalize(fig, "Do songs that differ more evoke more different EEG?",
                    synthetic=synthetic, height=380,
                    caption="Blue: uncorrected Mantel p < 0.05. See the table for Holm-adjusted "
                            "p-values.")


def plot_pair_scatter(pairs: pd.DataFrame, model: str, *, synthetic: bool = False) -> go.Figure:
    """EEG distance against a model distance for every recording pair."""
    same = pairs["song_a"] == pairs["song_b"]
    fig = go.Figure()
    for flag, name, color in ((False, "different songs", "#9aa5b1"),
                              (True, "same song", "#e07a1f")):
        sel = pairs[same == flag]
        fig.add_trace(go.Scatter(
            x=sel[model], y=sel["eeg"], mode="markers", name=name,
            marker={"color": color, "size": 7, "opacity": 0.75},
            text=sel["song_a"] + " / " + sel["song_b"],
            hovertemplate="%{text}<br>model %{x:.2f}<br>EEG %{y:.2f}<extra></extra>"))
    fig.update_xaxes(title=f"{model} distance")
    fig.update_yaxes(title="EEG feature distance")
    return finalize(fig, f"Recording pairs: EEG vs {model}", synthetic=synthetic, height=420)
