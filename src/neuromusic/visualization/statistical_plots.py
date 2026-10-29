"""Dataset, quality, comparison and clustering figures."""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from neuromusic.visualization import CATEGORICAL, SEQUENTIAL, empty_figure, finalize


def plot_coverage_matrix(matrix: pd.DataFrame, *, synthetic: bool = False) -> go.Figure:
    title = "Dataset coverage (analysable recordings per participant × song)"
    if matrix.empty:
        return empty_figure("No analysable recordings", title)
    fig = go.Figure(go.Heatmap(
        z=matrix.to_numpy(), x=list(matrix.columns), y=list(matrix.index),
        colorscale=SEQUENTIAL, zmin=0, colorbar={"title": "Recordings"},
        hovertemplate="%{y} × %{x}: %{z}<extra></extra>"))
    fig.update_xaxes(title="Song")
    fig.update_yaxes(title="Participant", autorange="reversed")
    return finalize(fig, title, synthetic=synthetic,
                    height=max(320, 18 * len(matrix.index) + 140))


def plot_provenance_summary(manifest: pd.DataFrame) -> go.Figure:
    title = "File provenance"
    if manifest.empty:
        return empty_figure("No files registered", title)
    counts = manifest["provenance"].value_counts().reset_index()
    counts.columns = ["provenance", "files"]
    fig = px.bar(counts, x="provenance", y="files", color="provenance",
                 color_discrete_sequence=CATEGORICAL, text="files")
    fig.update_xaxes(title="Provenance category")
    fig.update_yaxes(title="Number of files")
    fig.update_layout(showlegend=False)
    return finalize(fig, title, height=320)


def plot_channel_metric(quality: pd.DataFrame, metric: str, label: str, *,
                        synthetic: bool = False) -> go.Figure:
    title = f"{label} by channel"
    if quality.empty:
        return empty_figure("No quality data", title)
    fig = px.box(quality, x="channel", y=metric, points="all",
                 color_discrete_sequence=CATEGORICAL)
    fig.update_xaxes(title="Channel")
    fig.update_yaxes(title=label)
    return finalize(fig, title, synthetic=synthetic, height=380,
                    caption="Each point is one recording.")


def plot_song_boxplot(trials: pd.DataFrame, feature: str, unit_label: str,
                      titles: dict[str, str] | None = None, *,
                      synthetic: bool = False) -> go.Figure:
    title = f"{feature} by song"
    if trials.empty:
        return empty_figure("No trials", title)
    frame = trials.assign(song=trials["song_id"].map(lambda s: (titles or {}).get(s, s)))
    fig = px.box(frame, x="song", y=feature, points="all", color="song",
                 hover_data=["participant_id", "recording_id"],
                 color_discrete_sequence=CATEGORICAL)
    fig.update_xaxes(title="Song")
    fig.update_yaxes(title=f"{feature} ({unit_label})")
    fig.update_layout(showlegend=False)
    return finalize(fig, title, synthetic=synthetic, height=420,
                    caption="Each point is one trial (participant × song).")


def plot_paired(trials: pd.DataFrame, feature: str, unit_label: str,
                titles: dict[str, str] | None = None, *, synthetic: bool = False) -> go.Figure:
    """Participant-level lines across songs (only participants with >= 2 songs)."""
    title = f"{feature}: within-participant comparison"
    means = trials.groupby(["participant_id", "song_id"])[feature].mean().reset_index()
    multi = means.groupby("participant_id")["song_id"].transform("nunique") >= 2
    means = means[multi]
    if means.empty:
        return empty_figure("No participant listened to two or more songs", title)
    means["song"] = means["song_id"].map(lambda s: (titles or {}).get(s, s))
    fig = px.line(means, x="song", y=feature, color="participant_id", markers=True,
                  color_discrete_sequence=CATEGORICAL)
    fig.update_xaxes(title="Song")
    fig.update_yaxes(title=f"{feature} ({unit_label})")
    return finalize(fig, title, synthetic=synthetic, height=420)


def plot_effect_sizes(summary: pd.DataFrame, top_n: int = 25, *,
                      synthetic: bool = False) -> go.Figure:
    title = "Effect sizes with bootstrap confidence intervals"
    if summary.empty:
        return empty_figure("No comparison results", title)
    top = summary.nsmallest(top_n, "p_value").iloc[::-1]
    color = ["#c0392b" if s else "#5f6b7a" for s in top["significant"]]
    fig = go.Figure(go.Scatter(
        x=top["estimate"], y=top["feature"], mode="markers", marker={"color": color, "size": 8},
        error_x={"type": "data", "symmetric": False,
                 "array": (top["ci_high"] - top["estimate"]).to_numpy(),
                 "arrayminus": (top["estimate"] - top["ci_low"]).to_numpy()},
        customdata=top[["p_value", "p_adjusted"]].to_numpy(),
        hovertemplate="%{y}<br>estimate %{x:.3g}<br>p=%{customdata[0]:.3g}, "
                      "adjusted p=%{customdata[1]:.3g}<extra></extra>",
        name="estimate"))
    name = str(top["estimate_name"].iloc[0]).replace("_", " ")
    fig.update_xaxes(title=name)
    fig.update_yaxes(title="")
    return finalize(fig, title, synthetic=synthetic, height=max(380, 22 * len(top) + 120),
                    caption="Red: significant after multiple-comparison adjustment.")


def plot_pca_scatter(scores: pd.DataFrame, color_by: str, explained: list[float],
                     x: str = "PC1", y: str = "PC2", *, synthetic: bool = False) -> go.Figure:
    title = f"PCA of trial-level features (coloured by {color_by}) — exploratory"
    ix, iy = int(x[2:]) - 1, int(y[2:]) - 1
    fig = px.scatter(scores, x=x, y=y, color=scores[color_by].astype(str),
                     hover_data=["participant_id", "song_id", "recording_id"],
                     color_discrete_sequence=CATEGORICAL)
    fig.update_traces(marker={"size": 10, "line": {"width": 0.5, "color": "white"}})
    fig.update_xaxes(title=f"{x} ({explained[ix]:.1%} variance)")
    fig.update_yaxes(title=f"{y} ({explained[iy]:.1%} variance)")
    fig.update_layout(legend_title_text=color_by)
    return finalize(fig, title, synthetic=synthetic, height=480)


def plot_loadings(loadings: pd.DataFrame, component: str, top_n: int = 20) -> go.Figure:
    title = f"Top feature loadings on {component}"
    top = loadings[component].abs().nlargest(top_n).index
    values = loadings.loc[top, component].iloc[::-1]
    fig = go.Figure(go.Bar(x=values.to_numpy(), y=list(values.index), orientation="h",
                           marker_color=["#2a6fdb" if v >= 0 else "#c0392b" for v in values]))
    fig.update_xaxes(title="Loading (standardised features)")
    return finalize(fig, title, height=max(360, 22 * len(values) + 120))


def plot_explained_variance(explained: list[float]) -> go.Figure:
    pcs = [f"PC{i + 1}" for i in range(len(explained))]
    cumulative = pd.Series(explained).cumsum()
    fig = go.Figure()
    fig.add_trace(go.Bar(x=pcs, y=explained, name="Per component", marker_color=CATEGORICAL[0]))
    fig.add_trace(go.Scatter(x=pcs, y=cumulative, name="Cumulative", mode="lines+markers",
                             line={"color": CATEGORICAL[1]}))
    fig.update_yaxes(title="Explained variance ratio", tickformat=".0%")
    fig.update_xaxes(title="Principal component")
    return finalize(fig, "Explained variance", height=340)


def plot_cluster_diagnostics(k_diag: pd.DataFrame) -> go.Figure:
    title = "Cluster diagnostics by k"
    if k_diag.empty:
        return empty_figure("No clustering solutions", title)
    fig = go.Figure()
    for i, (col, label) in enumerate([("silhouette", "Silhouette"),
                                      ("stability_ari", "Bootstrap stability (ARI)"),
                                      ("ari_vs_song", "ARI vs song"),
                                      ("ari_vs_participant", "ARI vs participant")]):
        fig.add_trace(go.Scatter(x=k_diag["k"], y=k_diag[col], mode="lines+markers", name=label,
                                 line={"color": CATEGORICAL[i]}))
    fig.update_xaxes(title="Number of clusters (k)", dtick=1)
    fig.update_yaxes(title="Score")
    return finalize(fig, title, height=380)


def plot_contingency(table: pd.DataFrame, title: str) -> go.Figure:
    if table.empty:
        return empty_figure("No data", title)
    fig = go.Figure(go.Heatmap(z=table.to_numpy(), x=[str(c) for c in table.columns],
                               y=[f"cluster {i}" for i in table.index], colorscale=SEQUENTIAL,
                               text=table.to_numpy(), texttemplate="%{text}",
                               colorbar={"title": "Trials"}))
    return finalize(fig, title, height=max(300, 40 * len(table.index) + 140))
