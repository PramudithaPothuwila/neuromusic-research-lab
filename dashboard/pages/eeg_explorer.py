"""Page 3 — EEG explorer."""

import sys
from pathlib import Path

_DASHBOARD = str(Path(__file__).resolve().parents[1])
if _DASHBOARD not in sys.path:
    sys.path.insert(0, _DASHBOARD)

import numpy as np  # noqa: E402
import streamlit as st  # noqa: E402

from components import state, widgets  # noqa: E402
from neuromusic.preprocessing.pipeline import PreprocessingError  # noqa: E402
from neuromusic.visualization.eeg_plots import plot_raw_vs_cleaned, plot_traces  # noqa: E402

ws = state.page_header("EEG explorer", "Inspect raw and cleaned time series with artifact flags.")
state.units_notice()
unit = ws.signal_unit_label

show_all = st.toggle("Include excluded / unanalysable recordings", value=False)
rec = widgets.recording_selector("explorer", analysable_only=not show_all)
if rec is None:
    st.stop()
if rec.excluded:
    st.warning(f"Excluded recording: {rec.exclusion_reason}")

raw = state.load_raw(rec.recording_id)
times = raw.times - raw.times[0]
widgets.metric_row([
    ("Samples", f"{raw.n_samples:,}"),
    ("Duration", f"{rec.duration_seconds:.2f} s"),
    ("Detected rate", f"{rec.sampling_rate:.3f} Hz" if rec.sampling_rate else "—"),
    ("Validation", rec.validation_status.value),
])
st.caption(f"Timestamps: first {raw.times[0]:.6f} s, last {raw.times[-1]:.6f} s "
           f"(file `{rec.source_filename}`). Playback onset: "
           + (f"{rec.playback_onset_s:.3f} s" if rec.playback_onset_s is not None else "unknown")
           + (" (verified)" if rec.alignment_verified else " (not verified)"))

channels = st.multiselect("Channels", raw.channel_names, default=raw.channel_names[:6])
duration = float(times[-1]) if times.size else 0.0
t0, t1 = st.slider("Time range (s)", 0.0, max(duration, 0.1),
                   (0.0, min(duration, 20.0)), step=0.5)
mode = st.radio("Signal", ["Raw (DC removed)", "Cleaned", "Raw vs cleaned"], horizontal=True)

if not channels:
    st.info("Select at least one channel.")
    st.stop()

if mode == "Raw (DC removed)":
    mask = (times >= t0) & (times <= t1)
    idx = [raw.channel_names.index(c) for c in channels]
    data = raw.data[idx][:, mask]
    data = data - np.nanmedian(data, axis=1, keepdims=True)
    widgets.plot(plot_traces(times[mask], data, channels, unit,
                             f"Raw EEG — {rec.participant_id} · {state.song_label(rec.song_id)}",
                             synthetic=rec.is_synthetic), key="raw")
else:
    try:
        res = state.load_preprocessed(rec.recording_id)
    except (PreprocessingError, ValueError) as exc:
        st.error(f"Preprocessing failed: {exc}")
        st.stop()
    mask = (res.times >= t0) & (res.times <= t1)
    if not mask.any():
        st.info("Selected range lies outside the analysed span.")
        st.stop()
    idx = [res.channel_names.index(c) for c in channels if c in res.channel_names]
    if mode == "Cleaned":
        widgets.plot(plot_traces(res.times[mask], res.cleaned[idx][:, mask],
                                 [res.channel_names[i] for i in idx], unit,
                                 f"Cleaned EEG — {rec.participant_id} · "
                                 f"{state.song_label(rec.song_id)}",
                                 flags=res.flags, synthetic=rec.is_synthetic), key="clean")
    else:
        channel = st.selectbox("Channel for comparison", [res.channel_names[i] for i in idx])
        c = res.channel_names.index(channel)
        widgets.plot(plot_raw_vs_cleaned(res.times[mask], res.raw_centered[c, mask],
                                         res.cleaned[c, mask], channel, unit,
                                         synthetic=rec.is_synthetic), key="cmp")
    st.markdown("#### Flagged intervals")
    flags = res.flags
    in_range = flags[(flags["end_s"] >= t0) & (flags["start_s"] <= t1)] if not flags.empty \
        else flags
    if in_range.empty:
        st.caption("No flags in the selected range.")
    else:
        st.dataframe(in_range, hide_index=True, width="stretch")
