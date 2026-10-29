"""Page 6 — Spectral analysis."""

import sys
from pathlib import Path

_DASHBOARD = str(Path(__file__).resolve().parents[1])
if _DASHBOARD not in sys.path:
    sys.path.insert(0, _DASHBOARD)

import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from components import compute, state, widgets  # noqa: E402
from neuromusic.features.extractor import feature_unit  # noqa: E402
from neuromusic.preprocessing.pipeline import PreprocessingError  # noqa: E402
from neuromusic.visualization.spectral_plots import (  # noqa: E402
    plot_band_power_by_channel,
    plot_band_power_distribution,
    plot_psd,
    plot_spectrogram,
    plot_time_varying,
)

ws = state.page_header("Spectral analysis",
                       "Welch power spectra, band power, and time-varying spectral content.")
state.units_notice()
unit = ws.signal_unit_label
bands = ws.features.frequency_bands
synthetic = state.uses_synthetic_data()

tab_rec, tab_trials = st.tabs(["Single recording", "Across trials"])

with tab_rec:
    rec = widgets.recording_selector("spectral")
    if rec is not None:
        try:
            freqs, psd, channels = compute.recording_psd(rec.recording_id, ws.preprocessing_hash)
        except (PreprocessingError, ValueError) as exc:
            st.error(f"Preprocessing failed: {exc}")
            st.stop()
        chosen = st.multiselect("Channels", channels, default=channels[:4], key="psd_ch")
        idx = [channels.index(c) for c in chosen]
        log_scale = st.toggle("Logarithmic power axis", value=True)
        widgets.plot(plot_psd(freqs, psd[idx], chosen, unit, bands=bands, log_scale=log_scale,
                              synthetic=rec.is_synthetic,
                              title=f"PSD — {rec.participant_id} · "
                                    f"{state.song_label(rec.song_id)}"), key="psd")

        windows = state.load_features("window")
        if windows is None:
            state.require_features("window")
        else:
            wf = windows[windows["recording_id"] == rec.recording_id]
            measure = st.radio("Band-power measure",
                               ["log_abs_power", "rel_power", "abs_power"], horizontal=True)
            valid = wf[wf["quality_status"] != "rejected"]
            long = pd.DataFrame([
                {"channel": ch, "band": band,
                 "value": float(valid[f"spectral.{measure}.{band}.{ch}"].median())}
                for band in bands for ch in channels
                if f"spectral.{measure}.{band}.{ch}" in valid.columns])
            label = feature_unit(f"spectral.{measure}.alpha.X", unit)
            widgets.plot(plot_band_power_by_channel(
                long, label, synthetic=rec.is_synthetic,
                title="Median band power across valid windows"), key="bpc")
            channel = st.selectbox("Channel for time course", channels, key="tv_ch")
            cols = [f"spectral.{measure}.{b}.{channel}" for b in bands]
            widgets.plot(plot_time_varying(wf, [c for c in cols if c in wf.columns], label,
                                           synthetic=rec.is_synthetic), key="tv")
        channel = st.selectbox("Channel for time-frequency map", channels, key="tf_ch")
        f, t, p = compute.recording_spectrogram(rec.recording_id, channel, ws.preprocessing_hash)
        widgets.plot(plot_spectrogram(t, f, p, channel, unit, synthetic=rec.is_synthetic),
                     key="tf")

with tab_trials:
    trials = state.require_features("trial")
    if trials is not None:
        valid = trials[trials["trial_valid"]]
        c1, c2, c3 = st.columns(3)
        measure = c1.selectbox("Measure", ["log_abs_power", "rel_power", "abs_power"])
        band = c2.selectbox("Band", list(bands))
        regions = sorted({c.split(".")[-1] for c in valid.columns
                          if c.startswith(f"region.{measure}.{band}.")})
        region = c3.selectbox("Region", regions)
        col = f"region.{measure}.{band}.{region}"
        if col in valid.columns:
            frame = valid.assign(song=valid["song_id"].map(state.song_label))
            widgets.plot(plot_band_power_distribution(
                frame, col, "song", feature_unit(col, unit), synthetic=synthetic,
                title=f"{band} {measure.replace('_', ' ')} — {region}, per trial"), key="dist")
            st.caption("Each point is one trial. Songs differ in which participants heard them; "
                       "between-song differences here are not within-participant contrasts.")
        if not ws.dataset.acquisition.montage_positions_verified:
            st.caption("Scalp topographic maps are disabled until electrode positions are "
                       "verified (`dataset.acquisition.montage_positions_verified`).")
