"""Zero-phase FIR filtering via MNE."""

from __future__ import annotations

import mne
import numpy as np

from neuromusic.config import BandpassConfig, NotchConfig


def remove_dc_offset(data: np.ndarray) -> np.ndarray:
    """Subtract each channel's median. Preserves shape; returns a new array."""
    return data - np.median(data, axis=1, keepdims=True)


def bandpass(data: np.ndarray, sfreq: float, cfg: BandpassConfig) -> np.ndarray:
    if not cfg.enabled:
        return data.copy()
    nyquist = sfreq / 2.0
    if cfg.high_hz >= nyquist:
        raise ValueError(f"bandpass high edge {cfg.high_hz} Hz must be below Nyquist {nyquist} Hz")
    return mne.filter.filter_data(
        data.astype(np.float64), sfreq, l_freq=cfg.low_hz, h_freq=cfg.high_hz,
        method="fir", phase="zero", fir_design="firwin", pad="reflect_limited", verbose=False,
    )


def notch(data: np.ndarray, sfreq: float, cfg: NotchConfig) -> np.ndarray:
    if not cfg.enabled:
        return data.copy()
    nyquist = sfreq / 2.0
    freqs = (
        np.arange(cfg.frequency_hz, nyquist, cfg.frequency_hz) if cfg.harmonics
        else np.array([cfg.frequency_hz])
    )
    freqs = freqs[freqs < nyquist - 1.0]
    if freqs.size == 0:
        return data.copy()
    return mne.filter.notch_filter(
        data.astype(np.float64), sfreq, freqs=freqs, method="fir", phase="zero",
        fir_design="firwin", verbose=False,
    )
