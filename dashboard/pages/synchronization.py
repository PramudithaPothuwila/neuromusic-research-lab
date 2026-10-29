"""Page 11 — Neural synchronization (inter-subject correlation)."""

import sys
from pathlib import Path

_DASHBOARD = str(Path(__file__).resolve().parents[1])
if _DASHBOARD not in sys.path:
    sys.path.insert(0, _DASHBOARD)

import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from components import state, widgets  # noqa: E402
from neuromusic.analysis.synchronization import check_preconditions  # noqa: E402
from neuromusic.experiments.runner import ExperimentError, ExperimentRunner  # noqa: E402
from neuromusic.visualization.model_plots import (  # noqa: E402
    plot_isc_channels,
    plot_isc_pairs,
    plot_isc_sliding,
)

ws = state.page_header(
    "Neural synchronization",
    "Inter-subject correlation of EEG time courses during the same song. Enabled only when "
    "recordings have verified playback-relative alignment.")
st.caption("Correlated EEG activity does not indicate identical subjective experience.")

recs = ws.registry.analysable_recordings()
songs = sorted({r.song_id for r in recs if r.song_id})
rows, available = [], []
for song in songs:
    avail = check_preconditions(recs, song)
    rows.append({"song": state.song_label(song), "song_id": song, "available": avail.available,
                 "aligned recordings": len(avail.eligible_recording_ids),
                 "reasons": " ".join(avail.reasons)})
    if avail.available:
        available.append(song)
st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")

if not available:
    st.info("Synchronization analysis is unavailable: no song has at least two participants "
            "with a verified playback onset. Set `playback_onset_s` and `alignment_verified` "
            "in the **Dataset manager** once playback alignment is documented.")
else:
    song = st.selectbox("Song", available, format_func=state.song_label)
    if st.button("Run synchronization analysis", type="primary"):
        with st.spinner("Computing ISC and surrogates…"):
            try:
                record = ExperimentRunner(ws).run_synchronization(song)
                st.success(f"Saved {record.experiment_id}.")
            except ExperimentError as exc:
                st.error(str(exc))

record = widgets.experiment_selector("synchronization", key="sync_exp")
if record is not None:
    widgets.record_summary(record)
    eid = record.experiment_id
    per_channel = widgets.load_table(eid, "isc_per_channel")
    if per_channel.empty:
        st.info("No results saved for this experiment.")
    else:
        st.json(record.metrics, expanded=False)
        widgets.plot(plot_isc_channels(per_channel, synthetic=record.is_synthetic_data),
                     key="isc_ch")
        c1, c2 = st.columns(2)
        with c1:
            widgets.plot(plot_isc_sliding(widgets.load_table(eid, "isc_sliding"),
                                          synthetic=record.is_synthetic_data), key="isc_t")
        with c2:
            widgets.plot(plot_isc_pairs(widgets.load_table(eid, "isc_pairs"),
                                        synthetic=record.is_synthetic_data), key="isc_p")
        st.dataframe(per_channel, hide_index=True, width="stretch")
