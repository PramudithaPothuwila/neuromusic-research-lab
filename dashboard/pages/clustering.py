"""Page 8 — Clustering (exploratory)."""

import sys
from pathlib import Path

_DASHBOARD = str(Path(__file__).resolve().parents[1])
if _DASHBOARD not in sys.path:
    sys.path.insert(0, _DASHBOARD)

import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from components import state, widgets  # noqa: E402
from neuromusic.experiments.runner import ExperimentError, ExperimentRunner  # noqa: E402
from neuromusic.features.extractor import feature_columns, feature_families  # noqa: E402
from neuromusic.visualization.statistical_plots import (  # noqa: E402
    plot_cluster_diagnostics,
    plot_contingency,
    plot_explained_variance,
    plot_loadings,
    plot_pca_scatter,
)

ws = state.page_header(
    "Clustering",
    "Exploratory PCA and K-Means on standardised trial-level features. Clusters are not "
    "emotional states and say nothing about out-of-sample prediction.")
trials = state.require_features("trial")
if trials is None:
    st.stop()

cfg = ws.experiments.clustering
with st.expander("Run a clustering experiment", expanded=False):
    families = feature_families(feature_columns(trials))
    prefixes = st.multiselect("Feature families", families,
                              default=[p for p in cfg.feature_prefixes if p in families])
    c1, c2, c3 = st.columns(3)
    n_comp = c1.number_input("PCA components", 2, 20, cfg.n_components)
    k_values = c2.multiselect("k values", list(range(2, 9)), default=cfg.k_values)
    runs = c3.number_input("Stability bootstrap runs", 0, 500, cfg.n_stability_runs)
    if st.button("Run clustering", type="primary"):
        try:
            record = ExperimentRunner(ws).run_clustering(cfg.model_copy(update={
                "feature_prefixes": prefixes, "n_components": int(n_comp),
                "k_values": sorted(k_values), "n_stability_runs": int(runs)}))
            st.success(f"Saved {record.experiment_id}.")
        except ExperimentError as exc:
            st.error(str(exc))

record = widgets.experiment_selector("clustering", key="clust_exp")
if record is None:
    st.stop()
widgets.record_summary(record)
eid = record.experiment_id
scores = widgets.load_table(eid, "pca_scores")
loadings = widgets.load_table(eid, "pca_loadings")
diag = widgets.load_table(eid, "k_diagnostics")
assign = widgets.load_table(eid, "assignments")
explained = widgets.load_json(eid, "explained_variance") or []
if scores.empty:
    st.info("This experiment has no saved results.")
    st.stop()

scores = scores.merge(assign, on=["recording_id", "participant_id", "song_id", "trial_id"],
                      how="left")
scores["song"] = scores["song_id"].map(state.song_label)
cluster_cols = [c for c in assign.columns if c.startswith(("cluster_k", "hier_k"))]
pcs = [c for c in scores.columns if c.startswith("PC")]
c1, c2, c3 = st.columns(3)
color_by = c1.selectbox("Colour by", ["song", "participant_id", *cluster_cols])
x = c2.selectbox("x", pcs, index=0)
y = c3.selectbox("y", pcs, index=min(1, len(pcs) - 1))
widgets.plot(plot_pca_scatter(scores, color_by, explained, x, y,
                              synthetic=record.is_synthetic_data), key="pca")
c1, c2 = st.columns(2)
with c1:
    widgets.plot(plot_explained_variance(explained), key="ev")
with c2:
    widgets.plot(plot_cluster_diagnostics(diag), key="diag")
if not loadings.empty:
    comp = st.selectbox("Loadings for", pcs)
    widgets.plot(plot_loadings(loadings.set_index("feature"), comp), key="load")
st.dataframe(diag, hide_index=True, width="stretch")
if cluster_cols:
    col = st.selectbox("Cluster solution", cluster_cols)
    c1, c2 = st.columns(2)
    with c1:
        widgets.plot(plot_contingency(pd.crosstab(scores[col], scores["song"]),
                                      "Songs across clusters"), key="ct_song")
    with c2:
        widgets.plot(plot_contingency(pd.crosstab(scores[col], scores["participant_id"]),
                                      "Participants across clusters"), key="ct_part")
