"""Per-channel signal-quality metrics (relative, unit-agnostic)."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import signal as sps
from scipy import stats

from neuromusic.preprocessing.artifacts import MAD_SCALE


def channel_quality(
    raw_centered: np.ndarray,
    filtered: np.ndarray,
    sfreq: float,
    channel_names: list[str],
    flags: pd.DataFrame,
    duration_s: float,
    line_noise_hz: float = 50.0,
) -> pd.DataFrame:
    nyquist = sfreq / 2.0
    nperseg = int(min(raw_centered.shape[1], sfreq * 4))
    freqs, psd = sps.welch(raw_centered, fs=sfreq, nperseg=nperseg, axis=-1)

    def frac(lo: float, hi: float, tot: tuple[float, float] = (1.0, nyquist)) -> np.ndarray:
        num = psd[:, (freqs >= lo) & (freqs < hi)].sum(axis=1)
        den = psd[:, (freqs >= tot[0]) & (freqs < tot[1])].sum(axis=1)
        return np.divide(num, den, out=np.full_like(num, np.nan), where=den > 0)

    hf_hi = min(100.0, nyquist - 1.0)
    rows = []
    for c, ch in enumerate(channel_names):
        sig = filtered[c]
        ch_flags = flags[flags["channel"] == ch] if not flags.empty else flags
        local = ch_flags[ch_flags["scope"] == "segment"] if not ch_flags.empty else ch_flags
        flagged_s = float((local["end_s"] - local["start_s"]).sum()) if not local.empty else 0.0
        rows.append({
            "channel": ch,
            "variance": float(sig.var()),
            "robust_sd": float(np.median(np.abs(sig - np.median(sig))) * MAD_SCALE),
            "kurtosis": float(stats.kurtosis(sig)),
            "line_noise_fraction": float(frac(line_noise_hz - 1, line_noise_hz + 1)[c])
            if line_noise_hz < nyquist - 1 else np.nan,
            "high_freq_fraction": float(frac(45.0, hf_hi)[c]) if hf_hi > 45 else np.nan,
            "low_freq_fraction": float(frac(0.0, 1.0, (0.0, nyquist))[c]),
            "n_flags": len(ch_flags),
            "flag_kinds": ", ".join(sorted(set(ch_flags["kind"]))) if not ch_flags.empty else "",
            "flagged_fraction": min(1.0, flagged_s / duration_s) if duration_s > 0 else 0.0,
            "recording_level_flag": bool(
                not ch_flags.empty and (ch_flags["scope"] == "recording").any()),
        })
    frame = pd.DataFrame(rows)
    frame["quality"] = np.where(
        frame["recording_level_flag"] | (frame["flagged_fraction"] > 0.2), "review", "acceptable")
    return frame


def recording_quality_summary(windows: pd.DataFrame, chan_quality: pd.DataFrame,
                              flags: pd.DataFrame) -> dict[str, object]:
    status_counts = windows["quality_status"].value_counts().to_dict() if not windows.empty else {}
    return {
        "n_windows": len(windows),
        "window_status_counts": {str(k): int(v) for k, v in status_counts.items()},
        "n_flags": len(flags),
        "flag_kind_counts": {str(k): int(v) for k, v in flags["kind"].value_counts().items()}
        if not flags.empty else {},
        "channels_for_review": chan_quality.loc[chan_quality["quality"] == "review",
                                                "channel"].tolist(),
    }
