"""Page 7 — Song comparison."""

import sys
from pathlib import Path

_DASHBOARD = str(Path(__file__).resolve().parents[1])
if _DASHBOARD not in sys.path:
    sys.path.insert(0, _DASHBOARD)

import streamlit as st  # noqa: E402

from components import state, widgets  # noqa: E402
from neuromusic.experiments.runner import ExperimentError, ExperimentRunner  # noqa: E402
from neuromusic.features.extractor import feature_columns, feature_unit  # noqa: E402
from neuromusic.visualization.statistical_plots import (  # noqa: E402
    plot_effect_sizes,
    plot_paired,
    plot_song_boxplot,
)

ws = state.page_header(
    "Song comparison",
    "Within-participant comparisons of trial-level EEG features across songs, with effect "
    "sizes, confidence intervals and multiple-comparison correction.")
state.units_notice()
trials = state.require_features("trial")
if trials is None:
    st.stop()
valid = trials[trials["trial_valid"]]
synthetic = state.uses_synthetic_data()
titles = state.song_titles()

st.subheader("Descriptive view")
features = feature_columns(valid)
default = next((f for f in features if f.startswith("region.log_abs_power.alpha.")),
               features[0])
feature = st.selectbox("Feature", features, index=features.index(default))
c1, c2 = st.columns(2)
with c1:
    widgets.plot(plot_song_boxplot(valid, feature, feature_unit(feature, ws.signal_unit_label),
                                   titles, synthetic=synthetic), key="box")
with c2:
    widgets.plot(plot_paired(valid, feature, feature_unit(feature, ws.signal_unit_label), titles,
                             synthetic=synthetic), key="paired")

st.subheader("Inferential statistics")
st.caption("Never interpret a band-power difference as evidence of a specific emotional state.")
if st.button("Run statistics experiment", type="primary", icon=":material/calculate:"):
    with st.spinner("Running comparisons, permutation tests and mixed models…"):
        try:
            record = ExperimentRunner(ws).run_statistics()
            st.success(f"Saved {record.experiment_id}.")
        except ExperimentError as exc:
            st.error(str(exc))

record = widgets.experiment_selector("statistics", key="stats_exp")
if record is None:
    st.stop()
widgets.record_summary(record)
st.json(record.metrics, expanded=False)
eid = record.experiment_id
summary = widgets.load_table(eid, "comparisons")
if summary.empty:
    st.info("No inferential tests were possible (see notes above). With few participants per "
            "song, within-participant comparisons require participants who heard every "
            "compared song.")
else:
    scope = st.radio("Features", ["Primary", "All", "Region-level", "Channel-level"],
                     horizontal=True)
    shown = summary
    if scope == "Primary":
        shown = summary[summary["primary"]]
    elif scope == "Region-level":
        shown = summary[summary["feature"].str.startswith(("region.", "asymmetry."))]
    elif scope == "Channel-level":
        shown = summary[summary["feature"].str.startswith("spectral.")]
    widgets.plot(plot_effect_sizes(shown, synthetic=record.is_synthetic_data), key="effects")
    st.dataframe(shown, hide_index=True, width="stretch")
    widgets.download_csv(summary, f"{eid}_comparisons.csv", "Download comparisons",
                         key="dl_cmp")

for name, title in [("posthoc", "Post-hoc pairwise tests (Holm within feature)"),
                    ("mixed_effects", "Mixed-effects models: feature ~ song + (1 | participant)"),
                    ("feature_summary_by_song", "Feature summary by song"),
                    ("variability", "Participant- and song-level variability"),
                    ("song_attribute_associations",
                     "Song-attribute associations (exploratory, n = songs)"),
                    ("rating_associations", "EEG-rating associations")]:
    table = widgets.load_table(eid, name)
    if not table.empty:
        with st.expander(title):
            st.dataframe(table, hide_index=True, width="stretch")
