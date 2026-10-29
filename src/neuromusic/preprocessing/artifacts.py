"""Data-driven artifact detection.

Signal units are unconfirmed, so every criterion is relative: robust z-scores against the
channel's own distribution (time-localised flags) or against the other channels
(recording-level flags). Each flag records its location, reason, observed value and threshold.
Detection never alters data.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import signal as sps

from neuromusic.config import ArtifactConfig
from neuromusic.domain.models import ArtifactFlag

MAD_SCALE = 1.4826
FLAG_COLUMNS = ["kind", "reason", "channel", "scope", "start_s", "end_s", "value", "threshold"]


def robust_z(values: np.ndarray, axis: int | None = None) -> np.ndarray:
    median = np.median(values, axis=axis, keepdims=True)
    mad = np.median(np.abs(values - median), axis=axis, keepdims=True) * MAD_SCALE
    mad = np.where(mad <= np.finfo(float).eps, np.finfo(float).eps, mad)
    return (values - median) / mad


def _segments(n_samples: int, seg_len: int) -> np.ndarray:
    n_seg = n_samples // seg_len
    starts = np.arange(n_seg) * seg_len
    return np.column_stack([starts, starts + seg_len])


def _merge(mask: np.ndarray, segs: np.ndarray) -> list[tuple[int, int, int, int]]:
    """Merge consecutive flagged segments. Returns (first_seg, last_seg, start, stop)."""
    out: list[tuple[int, int, int, int]] = []
    idx = np.flatnonzero(mask)
    if idx.size == 0:
        return out
    run_start = prev = int(idx[0])
    for i in idx[1:]:
        i = int(i)
        if i != prev + 1:
            out.append((run_start, prev, int(segs[run_start, 0]), int(segs[prev, 1])))
            run_start = i
        prev = i
    out.append((run_start, prev, int(segs[run_start, 0]), int(segs[prev, 1])))
    return out


def _band_ratio(freqs: np.ndarray, psd: np.ndarray, band: tuple[float, float],
                total: tuple[float, float]) -> np.ndarray:
    in_band = (freqs >= band[0]) & (freqs < band[1])
    in_total = (freqs >= total[0]) & (freqs < total[1])
    num = psd[..., in_band].sum(axis=-1)
    den = psd[..., in_total].sum(axis=-1)
    return np.divide(num, den, out=np.zeros_like(num), where=den > 0)


def detect_artifacts(
    raw_centered: np.ndarray,
    filtered: np.ndarray,
    sfreq: float,
    channel_names: list[str],
    cfg: ArtifactConfig,
    time_offset: float = 0.0,
) -> list[ArtifactFlag]:
    """Detect artifacts.

    ``raw_centered`` is the unfiltered signal with DC removed (used for flatlines, steps, drift,
    line noise and high-frequency content); ``filtered`` is the analysis signal (amplitude).
    """
    if not cfg.detection_enabled:
        return []
    flags: list[ArtifactFlag] = []
    n_ch, n_samples = raw_centered.shape
    seg_len = max(round(cfg.detection_window_seconds * sfreq), 8)
    segs = _segments(n_samples, seg_len)
    if segs.shape[0] < 3:
        return flags
    nyquist = sfreq / 2.0

    def t(sample: int) -> float:
        return time_offset + sample / sfreq

    # Per-segment matrices (channels x segments)
    raw_segs = np.stack([raw_centered[:, s:e] for s, e in segs], axis=1)
    filt_segs = np.stack([filtered[:, s:e] for s, e in segs], axis=1)
    seg_std = raw_segs.std(axis=-1)
    seg_ptp = np.ptp(filt_segs, axis=-1)

    hf_lo = cfg.hf_band_hz[0]
    hf_hi = min(cfg.hf_band_hz[1], nyquist - 1.0)
    hf_ratio = None
    if hf_hi > hf_lo:
        freqs, psd = sps.periodogram(raw_segs, fs=sfreq, window="hann", axis=-1)
        hf_ratio = _band_ratio(freqs, psd, (hf_lo, hf_hi), (1.0, nyquist))

    diffs = np.abs(np.diff(raw_centered, axis=1))
    amplitude_mask = np.zeros_like(seg_ptp, dtype=bool)

    for c, ch in enumerate(channel_names):
        # Flatline: segment variability far below the channel's typical variability.
        typical = np.median(seg_std[c])
        if typical > 0:
            flat = seg_std[c] < cfg.flatline_relative_std * typical
            for _, _, s, e in _merge(flat, segs):
                flags.append(ArtifactFlag(
                    kind="flatline", channel=ch, start_s=t(s), end_s=t(e),
                    value=float(seg_std[c, s // seg_len]), threshold=cfg.flatline_relative_std,
                    reason=f"segment SD below {cfg.flatline_relative_std:g} x channel median SD"))

        # Sudden steps: sample-to-sample jumps far outside the channel's increment distribution.
        dz = robust_z(diffs[c])
        step_idx = np.flatnonzero(dz > cfg.step_robust_z)
        if step_idx.size:
            step_mask = np.zeros(segs.shape[0], dtype=bool)
            step_mask[np.minimum(step_idx // seg_len, segs.shape[0] - 1)] = True
            for a, b, s, e in _merge(step_mask, segs):
                in_run = step_idx[(step_idx >= s) & (step_idx < e)]
                peak = float(dz[in_run].max()) if in_run.size else float("nan")
                flags.append(ArtifactFlag(
                    kind="step", channel=ch, start_s=t(s), end_s=t(e), value=peak,
                    threshold=cfg.step_robust_z,
                    reason=f"sample-to-sample jump of {peak:.1f} robust SD "
                    f"({b - a + 1} segment(s))"))

        # Excessive amplitude change in the analysis signal.
        pz = robust_z(seg_ptp[c])
        amplitude_mask[c] = pz > cfg.amplitude_robust_z
        for _, _, s, e in _merge(amplitude_mask[c], segs):
            k = s // seg_len
            flags.append(ArtifactFlag(
                kind="excessive_amplitude", channel=ch, start_s=t(s), end_s=t(e),
                value=float(pz[k]), threshold=cfg.amplitude_robust_z,
                reason="peak-to-peak amplitude far above the channel's typical segment"))

        # High-frequency (likely muscle) contamination.
        if hf_ratio is not None:
            hz = robust_z(hf_ratio[c])
            for _, _, s, e in _merge(hz > cfg.hf_ratio_robust_z, segs):
                k = s // seg_len
                flags.append(ArtifactFlag(
                    kind="high_frequency", channel=ch, start_s=t(s), end_s=t(e),
                    value=float(hf_ratio[c, k]), threshold=cfg.hf_ratio_robust_z,
                    reason=f"{hf_lo:g}-{hf_hi:g} Hz power fraction elevated "
                    f"(robust z {hz[k]:.1f})"))

    # Possible movement: amplitude excursions on at least half the channels simultaneously.
    simultaneous = amplitude_mask.sum(axis=0) >= max(2, n_ch // 2)
    for _, _, s, e in _merge(simultaneous, segs):
        flags.append(ArtifactFlag(
            kind="possible_movement", channel=None, start_s=t(s), end_s=t(e),
            value=float(amplitude_mask[:, s // seg_len].sum()), threshold=float(max(2, n_ch // 2)),
            reason="concurrent amplitude excursions across many channels"))

    # Recording-level indicators from the whole-channel spectrum.
    nperseg = int(min(n_samples, max(seg_len * 8, int(sfreq * 8))))
    freqs, psd = sps.welch(raw_centered, fs=sfreq, nperseg=nperseg, axis=-1)
    total_hi = nyquist
    drift_frac = _band_ratio(freqs, psd, (0.0, cfg.drift_cutoff_hz), (0.0, total_hi))
    line_frac = None
    if cfg.line_noise_hz < nyquist - 1:
        line_frac = _band_ratio(freqs, psd, (cfg.line_noise_hz - 1.0, cfg.line_noise_hz + 1.0),
                                (1.0, total_hi))
    log_var = np.log(filtered.var(axis=1) + np.finfo(float).eps)
    var_z = robust_z(log_var) if n_ch >= 4 else np.zeros(n_ch)
    end = t(n_samples)
    for c, ch in enumerate(channel_names):
        if drift_frac[c] > cfg.drift_power_fraction:
            flags.append(ArtifactFlag(
                kind="drift", channel=ch, scope="recording", start_s=t(0), end_s=end,
                value=float(drift_frac[c]), threshold=cfg.drift_power_fraction,
                reason=f"{drift_frac[c]:.0%} of raw power below {cfg.drift_cutoff_hz:g} Hz"))
        if abs(var_z[c]) > cfg.channel_variance_robust_z:
            direction = "high" if var_z[c] > 0 else "low"
            flags.append(ArtifactFlag(
                kind="channel_variance", channel=ch, scope="recording", start_s=t(0), end_s=end,
                value=float(var_z[c]), threshold=cfg.channel_variance_robust_z,
                reason=f"filtered variance unusually {direction} relative to other channels"))
        if line_frac is not None and line_frac[c] > cfg.line_noise_ratio_warning:
            flags.append(ArtifactFlag(
                kind="poor_contact_possible", channel=ch, scope="recording", start_s=t(0),
                end_s=end, value=float(line_frac[c]), threshold=cfg.line_noise_ratio_warning,
                reason=f"{line_frac[c]:.0%} of power near {cfg.line_noise_hz:g} Hz line "
                "frequency; may indicate poor electrode contact"))
    return flags


def flags_to_frame(flags: list[ArtifactFlag]) -> pd.DataFrame:
    if not flags:
        return pd.DataFrame(columns=FLAG_COLUMNS).astype(
            {"start_s": float, "end_s": float, "value": float, "threshold": float})
    return pd.DataFrame([f.model_dump() for f in flags], columns=FLAG_COLUMNS)
