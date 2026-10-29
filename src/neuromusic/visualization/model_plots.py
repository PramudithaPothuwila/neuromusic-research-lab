"""Model-evaluation and synchronization figures."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from neuromusic.visualization import CATEGORICAL, DIVERGING, SEQUENTIAL, empty_figure, finalize


def plot_confusion_matrix(matrix: list[list[int]], classes: list[str], *,
                          normalize: bool = True, title: str = "Confusion matrix (held-out)",
                          synthetic: bool = False) -> go.Figure:
    cm = np.asarray(matrix, dtype=float)
    if cm.size == 0:
        return empty_figure("No predictions", title)
    if normalize:
        rows = cm.sum(axis=1, keepdims=True)
        shown = np.divide(cm, rows, out=np.zeros_like(cm), where=rows > 0)
        text = [[f"{shown[i, j]:.0%}<br>({int(cm[i, j])})" for j in range(cm.shape[1])]
                for i in range(cm.shape[0])]
    else:
        shown, text = cm, cm.astype(int).astype(str).tolist()
    fig = go.Figure(go.Heatmap(z=shown, x=classes, y=classes, colorscale=SEQUENTIAL,
                               text=text, texttemplate="%{text}", zmin=0,
                               zmax=1 if normalize else None,
                               colorbar={"title": "Row %" if normalize else "Count"}))
    fig.update_xaxes(title="Predicted song")
    fig.update_yaxes(title="True song", autorange="reversed")
    return finalize(fig, title, synthetic=synthetic, height=max(380, 40 * len(classes) + 160))


def plot_model_comparison(metrics: dict[str, dict[str, Any]], metric: str = "balanced_accuracy",
                          *, synthetic: bool = False) -> go.Figure:
    title = f"Held-out {metric.replace('_', ' ')} by model (LOSO)"
    if not metrics:
        return empty_figure("No models", title)
    names = list(metrics)
    values = [metrics[n][metric] for n in names]
    fig = go.Figure(go.Bar(x=names, y=values, name="Held-out",
                           marker_color=[("#9aa5b1" if n == "dummy" else CATEGORICAL[0])
                                         for n in names]))
    if metric == "balanced_accuracy":
        lo = [metrics[n].get("balanced_accuracy_ci", [np.nan, np.nan])[0] for n in names]
        hi = [metrics[n].get("balanced_accuracy_ci", [np.nan, np.nan])[1] for n in names]
        fig.update_traces(error_y={"type": "data", "symmetric": False,
                                   "array": np.array(hi) - np.array(values),
                                   "arrayminus": np.array(values) - np.array(lo)})
    if "dummy" in metrics:
        fig.add_hline(y=metrics["dummy"][metric], line_dash="dash", line_color="#5f6b7a",
                      annotation_text="dummy baseline")
    fig.update_yaxes(title=metric.replace("_", " ").capitalize(), range=[0, 1])
    fig.update_xaxes(title="Model")
    return finalize(fig, title, synthetic=synthetic, height=400,
                    caption="Error bars: 95% bootstrap CI resampling participants.")


def plot_train_vs_test(folds: pd.DataFrame) -> go.Figure:
    title = "Training vs held-out accuracy per fold"
    if folds.empty:
        return empty_figure("No folds", title)
    long = folds.melt(id_vars=["model", "held_out_participant"],
                      value_vars=["train_score", "test_score"], var_name="split",
                      value_name="accuracy")
    long["split"] = long["split"].map({"train_score": "training (in-sample)",
                                       "test_score": "held-out participant"})
    fig = px.box(long, x="model", y="accuracy", color="split", points="all",
                 color_discrete_sequence=["#9aa5b1", CATEGORICAL[0]])
    fig.update_yaxes(title="Accuracy", range=[0, 1.05])
    fig.update_xaxes(title="Model")
    return finalize(fig, title, height=400,
                    caption="Training accuracy is optimistic; only held-out values estimate "
                            "generalisation to new participants.")


def plot_participant_accuracy(per_participant: dict[str, float], model: str) -> go.Figure:
    title = f"Per-participant held-out accuracy — {model}"
    if not per_participant:
        return empty_figure("No participants", title)
    s = pd.Series(per_participant).sort_values()
    fig = go.Figure(go.Bar(x=s.values, y=list(s.index), orientation="h",
                           marker_color=CATEGORICAL[0]))
    fig.update_xaxes(title="Accuracy", range=[0, 1])
    fig.update_yaxes(title="Held-out participant")
    return finalize(fig, title, height=max(320, 18 * len(s) + 120))


def plot_permutation_null(null: list[float], observed: float, p_value: float) -> go.Figure:
    title = "Permutation test (labels permuted within participants)"
    fig = go.Figure(go.Histogram(x=null, nbinsx=30, marker_color="#9aa5b1", name="Null"))
    fig.add_vline(x=observed, line_color="#c0392b", line_width=2,
                  annotation_text=f"observed {observed:.3f} (p={p_value:.3f})")
    fig.update_xaxes(title="Balanced accuracy")
    fig.update_yaxes(title="Permutations")
    return finalize(fig, title, height=360)


def plot_importance(importance: pd.DataFrame, top_n: int = 20, model: str = "") -> go.Figure:
    title = f"Permutation importance on held-out participants {model}".strip()
    if importance.empty:
        return empty_figure("No importance values", title)
    top = importance.nlargest(top_n, "importance_mean").iloc[::-1]
    fig = go.Figure(go.Bar(
        x=top["importance_mean"], y=top["feature"], orientation="h", marker_color=CATEGORICAL[0],
        error_x={"type": "data", "array": top["importance_sd_across_folds"].fillna(0).to_numpy()}))
    fig.update_xaxes(title="Mean accuracy drop when permuted")
    return finalize(fig, title, height=max(360, 22 * len(top) + 120),
                    caption="Computed per LOSO fold on the held-out participant only; noisy "
                            "with few trials per participant.")


def plot_isc_channels(per_channel: pd.DataFrame, *, synthetic: bool = False) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Bar(x=per_channel["channel"], y=per_channel["isc"], name="Observed ISC",
                         marker_color=["#c0392b" if p < 0.05 else CATEGORICAL[0]
                                       for p in per_channel["p_adjusted"].fillna(1)]))
    if "null_95" in per_channel.columns:
        fig.add_trace(go.Scatter(x=per_channel["channel"], y=per_channel["null_95"],
                                 mode="markers", name="Surrogate 95th percentile",
                                 marker={"symbol": "line-ew-open", "size": 18,
                                         "color": "#5f6b7a"}))
    fig.update_yaxes(title="Mean pairwise correlation (r)")
    fig.update_xaxes(title="Channel")
    return finalize(fig, "Inter-subject correlation per channel", synthetic=synthetic,
                    height=380, caption="Red: FDR-adjusted surrogate p < 0.05.")


def plot_isc_sliding(sliding: pd.DataFrame, *, synthetic: bool = False) -> go.Figure:
    title = "Time-varying inter-subject correlation"
    if sliding.empty:
        return empty_figure("Recording too short for sliding windows", title)
    fig = go.Figure(go.Scatter(x=sliding["time_s"], y=sliding["isc_mean"], mode="lines+markers",
                               line={"color": CATEGORICAL[0]}))
    fig.update_xaxes(title="Time from playback onset (s)")
    fig.update_yaxes(title="Mean ISC across channels (r)")
    return finalize(fig, title, synthetic=synthetic, height=340)


def plot_isc_pairs(pairs: pd.DataFrame, *, synthetic: bool = False) -> go.Figure:
    title = "Participant-pair similarity"
    if pairs.empty:
        return empty_figure("No pairs", title)
    ids = sorted(set(pairs["participant_a"]) | set(pairs["participant_b"]))
    mat = pd.DataFrame(np.nan, index=ids, columns=ids)
    for row in pairs.itertuples(index=False):
        mat.loc[row.participant_a, row.participant_b] = row.mean_r
        mat.loc[row.participant_b, row.participant_a] = row.mean_r
    lim = float(np.nanmax(np.abs(mat.to_numpy()))) if mat.notna().any().any() else 1.0
    fig = go.Figure(go.Heatmap(z=mat.to_numpy(), x=ids, y=ids, colorscale=DIVERGING,
                               zmin=-lim, zmax=lim, colorbar={"title": "r"}))
    return finalize(fig, title, synthetic=synthetic, height=max(360, 22 * len(ids) + 140))
