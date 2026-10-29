"""Time-domain EEG figures."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from neuromusic.visualization import CATEGORICAL, empty_figure, finalize

MAX_POINTS_PER_TRACE = 20_000

FLAG_COLORS = {
    "flatline": "#7d4cc2", "step": "#c0392b", "excessive_amplitude": "#e07a1f",
    "high_frequency": "#d6619b", "possible_movement": "#8c6d31", "drift": "#5f6b7a",
    "channel_variance": "#1fa5b4", "poor_contact_possible": "#a5b41f",
}


def _decimate(times: np.ndarray, data: np.ndarray) -> tuple[np.ndarray, np.ndarray, int]:
    """Display-only stride decimation (does not alter stored data)."""
    step = max(1, int(np.ceil(times.size / MAX_POINTS_PER_TRACE)))
    return times[::step], data[..., ::step], step


def plot_traces(times: np.ndarray, data: np.ndarray, channel_names: list[str], unit: str,
                title: str = "EEG traces", *, flags: pd.DataFrame | None = None,
                spacing: float | None = None, synthetic: bool = False) -> go.Figure:
    """Vertically offset traces (one per channel) with optional shaded artifact intervals."""
    if data.size == 0:
        return empty_figure("No samples to display", title)
    t, d, step = _decimate(times, data)
    centred = d - np.median(d, axis=1, keepdims=True)
    if spacing is None:
        spread = np.nanpercentile(np.abs(centred), 99, axis=1)
        spacing = float(2.0 * np.nanmedian(spread)) or 1.0
    fig = go.Figure()
    offsets = spacing * np.arange(len(channel_names))[::-1]
    for i, ch in enumerate(channel_names):
        fig.add_trace(go.Scattergl(
            x=t, y=centred[i] + offsets[i], mode="lines", name=ch,
            line={"width": 1, "color": CATEGORICAL[i % len(CATEGORICAL)]},
            hovertemplate=f"{ch}<br>t=%{{x:.3f}} s<extra></extra>"))
    if flags is not None and not flags.empty:
        local = flags[flags["scope"] == "segment"] if "scope" in flags.columns else flags
        seen: set[str] = set()
        for row in local.itertuples(index=False):
            kind = str(row.kind)
            color = FLAG_COLORS.get(kind, "#999999")
            fig.add_vrect(x0=row.start_s, x1=row.end_s, fillcolor=color, opacity=0.12,
                          line_width=0, layer="below")
            if kind not in seen:
                seen.add(kind)
                fig.add_trace(go.Scatter(x=[None], y=[None], mode="markers",
                                         marker={"color": color, "size": 10, "symbol": "square"},
                                         name=f"flag: {kind}"))
    fig.update_yaxes(tickmode="array", tickvals=offsets, ticktext=channel_names,
                     title=f"Channel (offset; scale {spacing:.3g} {unit} per step)")
    fig.update_xaxes(title="Time (s)", rangeslider={"visible": False})
    caption = f"Display decimated by {step}x." if step > 1 else None
    return finalize(fig, title, synthetic=synthetic, height=max(400, 45 * len(channel_names)),
                    caption=caption)


def plot_raw_vs_cleaned(times: np.ndarray, raw: np.ndarray, cleaned: np.ndarray, channel: str,
                        unit: str, *, synthetic: bool = False) -> go.Figure:
    t, both, step = _decimate(times, np.vstack([raw, cleaned]))
    fig = go.Figure()
    fig.add_trace(go.Scattergl(x=t, y=both[0], mode="lines", name="Raw (DC removed)",
                               line={"width": 1, "color": "#9aa5b1"}))
    fig.add_trace(go.Scattergl(x=t, y=both[1], mode="lines", name="Cleaned",
                               line={"width": 1, "color": CATEGORICAL[0]}))
    fig.update_xaxes(title="Time (s)")
    fig.update_yaxes(title=f"Amplitude ({unit})")
    caption = f"Display decimated by {step}x." if step > 1 else None
    return finalize(fig, f"Raw vs cleaned signal — {channel}", synthetic=synthetic, height=380,
                    caption=caption)


def plot_flag_timeline(flags: pd.DataFrame, duration_s: float, *,
                       synthetic: bool = False) -> go.Figure:
    title = "Artifact flags by channel"
    if flags.empty:
        return empty_figure("No artifact flags", title)
    fig = go.Figure()
    frame = flags.assign(channel=flags["channel"].fillna("all channels"))
    for kind, group in frame.groupby("kind"):
        color = FLAG_COLORS.get(str(kind), "#999999")
        fig.add_trace(go.Bar(
            base=group["start_s"], x=group["end_s"] - group["start_s"], y=group["channel"],
            orientation="h", name=str(kind), marker_color=color,
            customdata=group[["reason"]].to_numpy(),
            hovertemplate="%{y}: %{base:.2f}–%{x:.2f} s<br>%{customdata[0]}<extra></extra>"))
    fig.update_layout(barmode="overlay")
    fig.update_xaxes(title="Time (s)", range=[0, duration_s])
    fig.update_yaxes(title="Channel")
    return finalize(fig, title, synthetic=synthetic, height=420)
