"""Spectral features from Welch power spectral density estimates.

Band power is the rectangle-rule integral of the PSD over the half-open interval [low, high),
so adjacent bands sharing an edge never count the same frequency bin twice.
"""

from __future__ import annotations

import itertools

import numpy as np
from scipy import signal as sps

from neuromusic.config import PSDConfig

EPS = np.finfo(float).tiny


def welch_psd(data: np.ndarray, sfreq: float, cfg: PSDConfig) -> tuple[np.ndarray, np.ndarray]:
    """PSD along the last axis. Returns (freqs, psd) with psd shape ``data.shape[:-1] + (F,)``."""
    n = data.shape[-1]
    nperseg = min(n, round(cfg.segment_seconds * sfreq))
    noverlap = round(nperseg * cfg.overlap_fraction)
    freqs, psd = sps.welch(data, fs=sfreq, window=cfg.window, nperseg=nperseg,
                           noverlap=noverlap, average=cfg.average, axis=-1, detrend="constant")
    return freqs, psd


def band_mask(freqs: np.ndarray, band: tuple[float, float]) -> np.ndarray:
    return (freqs >= band[0]) & (freqs < band[1])


def band_power(freqs: np.ndarray, psd: np.ndarray, band: tuple[float, float]) -> np.ndarray:
    """Absolute band power (signal units squared)."""
    mask = band_mask(freqs, band)
    if not mask.any():
        raise ValueError(f"band {band} contains no frequency bins at resolution "
                         f"{freqs[1] - freqs[0]:.3f} Hz")
    df = float(freqs[1] - freqs[0])
    return psd[..., mask].sum(axis=-1) * df


def spectral_entropy(freqs: np.ndarray, psd: np.ndarray, fmin: float, fmax: float) -> np.ndarray:
    """Normalised Shannon entropy of the PSD within [fmin, fmax) in [0, 1]."""
    mask = band_mask(freqs, (fmin, fmax))
    p = psd[..., mask]
    total = p.sum(axis=-1, keepdims=True)
    p = np.divide(p, total, out=np.zeros_like(p), where=total > 0)
    with np.errstate(divide="ignore", invalid="ignore"):
        h = -np.where(p > 0, p * np.log(p), 0.0).sum(axis=-1)
    return h / np.log(mask.sum())


def check_bands(bands: dict[str, tuple[float, float]], sfreq: float,
                total_range: tuple[float, float]) -> None:
    nyquist = sfreq / 2.0
    for name, (lo, hi) in bands.items():
        if hi > nyquist:
            raise ValueError(f"band {name} upper edge {hi} Hz exceeds Nyquist {nyquist} Hz")
        if lo < total_range[0] or hi > total_range[1]:
            raise ValueError(f"band {name} lies outside the total-power range {total_range}")
    ordered = sorted(bands.items(), key=lambda kv: kv[1][0])
    for (n1, (_, hi1)), (n2, (lo2, _)) in itertools.pairwise(ordered):
        if lo2 < hi1:
            raise ValueError(f"bands {n1} and {n2} overlap; overlapping definitions double-count")
