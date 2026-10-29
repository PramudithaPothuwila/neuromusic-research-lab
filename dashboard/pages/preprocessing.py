"""Page 5 — EEG preprocessing."""

import sys
from pathlib import Path

_DASHBOARD = str(Path(__file__).resolve().parents[1])
if _DASHBOARD not in sys.path:
    sys.path.insert(0, _DASHBOARD)

import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402
from pydantic import ValidationError  # noqa: E402

from components import state, widgets  # noqa: E402
from neuromusic import services  # noqa: E402
from neuromusic.config import PreprocessingConfig  # noqa: E402
from neuromusic.domain.enums import ReferenceMethod, SignalUnits  # noqa: E402
from neuromusic.preprocessing.pipeline import PreprocessingError  # noqa: E402
from neuromusic.visualization.eeg_plots import plot_raw_vs_cleaned  # noqa: E402

ws = state.page_header(
    "Preprocessing",
    "Configure, run and inspect the preprocessing pipeline. Each configuration produces a "
    "separate, hash-versioned output set; raw files are untouched.")
cfg = ws.preprocessing.model_dump(mode="json")
channels = ws.dataset.channels

with st.form("prep_config"):
    st.markdown(f"Current configuration hash: `{ws.preprocessing_hash}`")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("**Filtering**")
        cfg["remove_dc_offset"] = st.checkbox("Remove DC offset (median)",
                                              cfg["remove_dc_offset"])
        cfg["bandpass"]["enabled"] = st.checkbox("Band-pass", cfg["bandpass"]["enabled"])
        cfg["bandpass"]["low_hz"] = st.number_input("Low cut-off (Hz)", 0.1, 20.0,
                                                    float(cfg["bandpass"]["low_hz"]), 0.1)
        cfg["bandpass"]["high_hz"] = st.number_input("High cut-off (Hz)", 5.0, 120.0,
                                                     float(cfg["bandpass"]["high_hz"]), 1.0)
        cfg["notch"]["enabled"] = st.checkbox("Notch filter", cfg["notch"]["enabled"])
        cfg["notch"]["frequency_hz"] = st.selectbox(
            "Line frequency (Hz)", [50.0, 60.0],
            index=0 if cfg["notch"]["frequency_hz"] == 50.0 else 1)
    with c2:
        st.markdown("**Reference and windows**")
        methods = [m.value for m in ReferenceMethod]
        cfg["reference"]["method"] = st.selectbox(
            "Re-reference", methods, index=methods.index(cfg["reference"]["method"]),
            help="The acquisition reference is unknown; 'preserve_original' is the default.")
        cfg["reference"]["channels"] = st.multiselect("Reference channels (method 'channels')",
                                                      channels, cfg["reference"]["channels"])
        cfg["windows"]["duration_seconds"] = st.number_input(
            "Window length (s)", 0.5, 30.0, float(cfg["windows"]["duration_seconds"]), 0.5)
        cfg["windows"]["overlap_seconds"] = st.number_input(
            "Window overlap (s)", 0.0, 29.5, float(cfg["windows"]["overlap_seconds"]), 0.5)
        cfg["trim_start_seconds"] = st.number_input("Trim start (s)", 0.0, 60.0,
                                                    float(cfg["trim_start_seconds"]), 0.5)
        cfg["trim_end_seconds"] = st.number_input("Trim end (s)", 0.0, 60.0,
                                                  float(cfg["trim_end_seconds"]), 0.5)
    with c3:
        st.markdown("**Artifacts and units**")
        art = cfg["artifact_handling"]
        art["detection_enabled"] = st.checkbox("Artifact detection", art["detection_enabled"])
        art["automatic_rejection_enabled"] = st.checkbox(
            "Automatic window rejection", art["automatic_rejection_enabled"],
            help="Off by default: windows are flagged, not removed.")
        art["reject_window_if_flagged_channels_ge"] = st.number_input(
            "Reject window if flagged channels ≥", 1, len(channels),
            int(art["reject_window_if_flagged_channels_ge"]))
        art["amplitude_robust_z"] = st.number_input("Amplitude threshold (robust z)", 2.0, 50.0,
                                                    float(art["amplitude_robust_z"]), 0.5)
        art["step_robust_z"] = st.number_input("Step threshold (robust z)", 2.0, 100.0,
                                               float(art["step_robust_z"]), 1.0)
        units = [u.value for u in SignalUnits]
        cfg["signal_units"] = st.selectbox(
            "Signal units", units, index=units.index(cfg["signal_units"]),
            help="Set a physical unit only after the device scaling has been confirmed.")
        cfg["ica"]["enabled"] = st.checkbox("ICA (requires justified component exclusion)",
                                            cfg["ica"]["enabled"])
        cfg["save_cleaned_data"] = st.checkbox("Save cleaned signals", cfg["save_cleaned_data"])
    submitted = st.form_submit_button("Validate and save configuration", type="primary")

if submitted:
    try:
        new_cfg = PreprocessingConfig.model_validate(cfg)
    except ValidationError as exc:
        st.error(f"Invalid configuration: {exc}")
    else:
        path = ws.save_config("preprocessing", new_cfg)
        state.refresh()
        st.success(f"Saved {path.name}; new configuration hash `{ws.preprocessing_hash}`.")

st.subheader("Run pipeline")
store = services.preprocessed_store(ws)
n_ok = len(ws.registry.analysable_recordings())
st.caption(f"{len(store.processed_ids())} of {n_ok} analysable recordings processed with the "
           "current configuration.")
c1, c2 = st.columns(2)
force = c1.checkbox("Re-run already processed recordings")
if c1.button("Run preprocessing", icon=":material/play_arrow:"):
    bar = st.progress(0.0)
    results = services.preprocess_recordings(
        ws, force=force, progress=lambda i, n, msg: bar.progress(i / max(n, 1), text=msg))
    state.refresh()
    frame = pd.DataFrame([r.__dict__ for r in results])
    failed = frame[frame["status"] == "failed"] if not frame.empty else frame
    st.success(f"{(frame['status'] == 'ok').sum() if not frame.empty else 0} processed, "
               f"{len(failed)} failed.")
    if not failed.empty:
        st.dataframe(failed, hide_index=True, width="stretch")
if c2.button("Build features (preprocess missing first)", icon=":material/build:"):
    bar = st.progress(0.0)
    try:
        services.build_features(
            ws, progress=lambda i, n, msg: bar.progress(min(i / max(n, 1), 1.0), text=msg))
    except ValueError as exc:
        st.error(str(exc))
    else:
        state.refresh()
        st.success("Window- and trial-level features saved.")

st.subheader("Before / after")
state.units_notice()
rec = widgets.recording_selector("prep")
if rec is not None:
    try:
        res = state.load_preprocessed(rec.recording_id)
    except (PreprocessingError, ValueError) as exc:
        st.error(f"Preprocessing failed: {exc}")
    else:
        channel = st.selectbox("Channel", res.channel_names, key="prep_channel")
        c = res.channel_names.index(channel)
        widgets.plot(plot_raw_vs_cleaned(res.times, res.raw_centered[c], res.cleaned[c], channel,
                                         ws.signal_unit_label, synthetic=rec.is_synthetic),
                     key="prep_cmp")
        q = res.run.quality_results
        st.caption(f"{q['n_windows']} windows · {q['n_flags']} flags · status counts "
                   f"{q['window_status_counts']}")
