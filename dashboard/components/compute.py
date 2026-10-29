"""Light, cached computations needed only for display (PSD, spectrogram)."""

from __future__ import annotations

import numpy as np
import streamlit as st
from scipy import signal as sps

from components import state
from neuromusic.features.spectral import welch_psd


@st.cache_data(show_spinner=False)
def recording_psd(recording_id: str, prep_hash: str) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Welch PSD of the cleaned recording using the configured PSD parameters."""
    del prep_hash
    ws = state.get_workspace()
    res = state.load_preprocessed(recording_id)
    freqs, psd = welch_psd(res.cleaned, res.sfreq, ws.features.psd)
    return freqs, psd, res.channel_names


@st.cache_data(show_spinner=False)
def recording_spectrogram(recording_id: str, channel: str, prep_hash: str,
                          segment_seconds: float = 2.0
                          ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    del prep_hash
    res = state.load_preprocessed(recording_id)
    idx = res.channel_names.index(channel)
    nperseg = int(segment_seconds * res.sfreq)
    freqs, times, power = sps.spectrogram(res.cleaned[idx], fs=res.sfreq, nperseg=nperseg,
                                          noverlap=nperseg // 2, window="hann")
    return freqs, times + float(res.times[0]), power
