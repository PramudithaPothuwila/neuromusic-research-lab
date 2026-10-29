"""Frequency-domain figures."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from neuromusic.visualization import CATEGORICAL, SEQUENTIAL, empty_figure, finalize


def plot_psd(freqs: np.ndarray, psd: np.ndarray, channel_names: list[str], unit: str, *,
             bands: dict[str, tuple[float, float]] | None = None, fmax: float = 45.0,
             log_scale: bool = True, synthetic: bool = False,
             title: str = "Power spectral density (Welch)") -> go.Figure:
    mask = freqs <= fmax
    fig = go.Figure()
    for i, ch in enumerate(channel_names):
        fig.add_trace(go.Scatter(x=freqs[mask], y=psd[i, mask], mode="lines", name=ch,
                                 line={"width": 1.2, "color": CATEGORICAL[i % len(CATEGORICAL)]}))
    for name, (lo, hi) in (bands or {}).items():
        fig.add_vrect(x0=lo, x1=hi, fillcolor="#9aa5b1", opacity=0.08, line_width=0,
                      annotation_text=name, annotation_position="top left")
    fig.update_xaxes(title="Frequency (Hz)")
    fig.update_yaxes(title=f"PSD ({unit}²/Hz)", type="log" if log_scale else "linear")
    return finalize(fig, title, synthetic=synthetic, height=420)


def plot_band_power_by_channel(long: pd.DataFrame, unit_label: str, *,
                               synthetic: bool = False,
                               title: str = "Band power by channel") -> go.Figure:
    """``long`` columns: channel, band, value."""
    if long.empty:
        return empty_figure("No band-power values", title)
    fig = px.bar(long, x="channel", y="value", color="band", barmode="group",
                 color_discrete_sequence=CATEGORICAL)
    fig.update_xaxes(title="Channel")
    fig.update_yaxes(title=unit_label)
    return finalize(fig, title, synthetic=synthetic, height=400)


def plot_band_power_distribution(frame: pd.DataFrame, value_col: str, group_col: str,
                                 unit_label: str, *, synthetic: bool = False,
                                 title: str = "Band-power distribution") -> go.Figure:
    if frame.empty:
        return empty_figure("No values", title)
    fig = px.box(frame, x=group_col, y=value_col, points="all", color=group_col,
                 color_discrete_sequence=CATEGORICAL)
    fig.update_xaxes(title=group_col.replace("_", " ").capitalize())
    fig.update_yaxes(title=unit_label)
    fig.update_layout(showlegend=False)
    return finalize(fig, title, synthetic=synthetic, height=400)


def plot_time_varying(window_features: pd.DataFrame, columns: list[str], unit_label: str, *,
                      synthetic: bool = False,
                      title: str = "Band power over time (per analysis window)") -> go.Figure:
    if window_features.empty or not columns:
        return empty_figure("No windows", title)
    centre = (window_features["start_time"] + window_features["end_time"]) / 2
    fig = go.Figure()
    for i, col in enumerate(columns):
        fig.add_trace(go.Scatter(x=centre, y=window_features[col], mode="lines+markers",
                                 name=col, marker={"size": 4},
                                 line={"color": CATEGORICAL[i % len(CATEGORICAL)]}))
    if "quality_status" in window_features.columns:
        bad = window_features[window_features["quality_status"] != "good"]
        for row in bad.itertuples(index=False):
            fig.add_vrect(x0=row.start_time, x1=row.end_time, fillcolor="#e07a1f", opacity=0.06,
                          line_width=0)
    fig.update_xaxes(title="Window centre (s)")
    fig.update_yaxes(title=unit_label)
    return finalize(fig, title, synthetic=synthetic, height=400,
                    caption="Shaded: windows flagged or rejected by artifact detection.")


def plot_spectrogram(times: np.ndarray, freqs: np.ndarray, power: np.ndarray, channel: str,
                     unit: str, *, fmax: float = 45.0, synthetic: bool = False) -> go.Figure:
    """Time-frequency representation; ``power`` shape (n_freqs, n_times)."""
    mask = freqs <= fmax
    fig = go.Figure(go.Heatmap(x=times, y=freqs[mask], z=10 * np.log10(power[mask] + 1e-30),
                               colorscale=SEQUENTIAL,
                               colorbar={"title": f"dB ({unit}²/Hz)"}))
    fig.update_xaxes(title="Time (s)")
    fig.update_yaxes(title="Frequency (Hz)")
    return finalize(fig, f"Time-frequency representation — {channel}", synthetic=synthetic,
                    height=400)
