"""Plotly figure builders.

Every builder is a pure function returning ``plotly.graph_objects.Figure`` with a title, axis
labels carrying the supplied unit label ("a.u." unless units are validated) and, when
``synthetic=True``, a prominent "SYNTHETIC DATA" annotation.
"""

from __future__ import annotations

import plotly.graph_objects as go

FONT_FAMILY = "Inter, Segoe UI, Helvetica, Arial, sans-serif"
CATEGORICAL = ["#2a6fdb", "#e07a1f", "#2a9d6f", "#c0392b", "#7d4cc2", "#8c6d31",
               "#d6619b", "#5f6b7a", "#a5b41f", "#1fa5b4"]
SEQUENTIAL = "Blues"
DIVERGING = "RdBu_r"


def finalize(fig: go.Figure, title: str, *, synthetic: bool = False, height: int | None = None,
             caption: str | None = None) -> go.Figure:
    """Apply the shared layout, the title and optional synthetic marker / caption."""
    fig.update_layout(
        title={"text": title, "x": 0.0, "xanchor": "left", "font": {"size": 16}},
        font={"family": FONT_FAMILY, "size": 12},
        template="plotly_white",
        colorway=CATEGORICAL,
        margin={"l": 60, "r": 20, "t": 60, "b": 60 if caption is None else 90},
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.0, "xanchor": "right", "x": 1.0},
    )
    if height is not None:
        fig.update_layout(height=height)
    if synthetic:
        fig.add_annotation(text="SYNTHETIC DATA", xref="paper", yref="paper", x=0.5, y=0.5,
                           showarrow=False, opacity=0.18, textangle=-20,
                           font={"size": 48, "color": "#c0392b"})
    if caption:
        fig.add_annotation(text=caption, xref="paper", yref="paper", x=0.0, y=-0.18,
                           xanchor="left", yanchor="top", showarrow=False, align="left",
                           font={"size": 11, "color": "#5f6b7a"})
    return fig


def empty_figure(message: str, title: str = "") -> go.Figure:
    fig = go.Figure()
    fig.add_annotation(text=message, xref="paper", yref="paper", x=0.5, y=0.5,
                       showarrow=False, font={"size": 14, "color": "#5f6b7a"})
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    return finalize(fig, title, height=300)


__all__ = ["CATEGORICAL", "DIVERGING", "SEQUENTIAL", "empty_figure", "finalize"]
