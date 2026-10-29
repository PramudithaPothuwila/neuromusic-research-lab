"""Page 4 — Signal quality."""

import sys
from pathlib import Path

_DASHBOARD = str(Path(__file__).resolve().parents[1])
if _DASHBOARD not in sys.path:
    sys.path.insert(0, _DASHBOARD)

import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from components import state, widgets  # noqa: E402
from neuromusic import services  # noqa: E402
from neuromusic.visualization.eeg_plots import plot_flag_timeline  # noqa: E402
from neuromusic.visualization.statistical_plots import plot_channel_metric  # noqa: E402

ws = state.page_header(
    "Signal quality",
    "Relative, unit-agnostic quality indicators: robust variability, line-noise and "
    "high-frequency fractions, drift, flatlines and flagged segments.")
state.units_notice()
store = services.preprocessed_store(ws)

quality, flags = state.quality_tables()
if quality.empty:
    st.info("No recordings have been preprocessed with the current configuration.")
    if st.button("Preprocess all analysable recordings", type="primary"):
        bar = st.progress(0.0)
        services.preprocess_recordings(
            ws, progress=lambda i, n, msg: bar.progress(i / max(n, 1), text=msg))
        state.refresh()
        st.rerun()
    st.stop()

st.subheader("Across recordings")
metric = st.selectbox("Metric", {
    "robust_sd": f"Robust SD ({ws.signal_unit_label})",
    "line_noise_fraction": "Line-noise power fraction",
    "high_freq_fraction": "High-frequency (45 Hz+) power fraction",
    "low_freq_fraction": "Sub-1 Hz power fraction (raw)",
    "flagged_fraction": "Fraction of time flagged",
    "kurtosis": "Kurtosis",
}.items(), format_func=lambda kv: kv[1])
widgets.plot(plot_channel_metric(quality, metric[0], metric[1],
                                 synthetic=state.uses_synthetic_data()), key="metric")
if not flags.empty:
    counts = flags.groupby(["kind", "scope"]).size().rename("n_flags").reset_index()
    st.dataframe(counts, hide_index=True, width="stretch")
widgets.download_csv(quality, "channel_quality_all.csv", "Download quality report (all)",
                     key="dl_q_all")
if not flags.empty:
    widgets.download_csv(flags, "artifact_flags_all.csv", "Download artifact flags (all)",
                         key="dl_f_all")

st.subheader("Single recording")
rec = widgets.recording_selector("quality")
if rec is None:
    st.stop()
if not store.exists(rec.recording_id):
    st.info("This recording has not been preprocessed with the current configuration.")
    if st.button("Preprocess this recording"):
        result = services.preprocess_recordings(ws, [rec.recording_id])
        if result and result[0].status == "failed":
            st.error(result[0].message)
        state.refresh()
        st.rerun()
    st.stop()

rq = store.load_table(rec.recording_id, "quality")
rf = store.load_table(rec.recording_id, "flags")
windows = store.load_table(rec.recording_id, "windows")
run = store.load_run(rec.recording_id)
q = run.quality_results
widgets.metric_row([
    ("Windows", q.get("n_windows", 0)),
    ("Flagged windows", q.get("window_status_counts", {}).get("flagged", 0)),
    ("Rejected windows", q.get("n_rejected_windows", 0)),
    ("Channels for review", len(q.get("channels_for_review", []))),
])
st.dataframe(rq.style.apply(
    lambda row: ["background-color: #fdecea" if row["quality"] == "review" else ""] * len(row),
    axis=1), hide_index=True, width="stretch")
widgets.plot(plot_flag_timeline(rf, float(q.get("analysis_start_s", 0))
                                + float(q.get("analysis_duration_s", 0)),
                                synthetic=rec.is_synthetic), key="timeline")
with st.expander(f"Flag details ({len(rf)})"):
    st.dataframe(rf, hide_index=True, width="stretch")
with st.expander("Window quality"):
    st.dataframe(windows, hide_index=True, width="stretch")
report = pd.concat([rq.assign(section="channel_quality"),
                    rf.assign(section="artifact_flags")], ignore_index=True)
widgets.download_csv(report, f"quality_{rec.recording_id}.csv",
                     "Download quality report (this recording)", key="dl_q_one")
