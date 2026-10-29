"""Sliding-window segmentation."""

from __future__ import annotations

import numpy as np
import pandas as pd

from neuromusic.config import WindowConfig
from neuromusic.domain.enums import QualityStatus


def window_bounds(n_samples: int, sfreq: float, cfg: WindowConfig,
                  start_sample: int = 0, stop_sample: int | None = None) -> np.ndarray:
    """Array of shape (n_windows, 2) with half-open sample bounds [start, stop)."""
    stop_sample = n_samples if stop_sample is None else min(stop_sample, n_samples)
    length = round(cfg.duration_seconds * sfreq)
    step = round((cfg.duration_seconds - cfg.overlap_seconds) * sfreq)
    if length <= 0 or step <= 0:
        raise ValueError("window length and step must be at least one sample")
    available = stop_sample - start_sample
    if available < length:
        return np.empty((0, 2), dtype=np.int64)
    starts = start_sample + np.arange(0, available - length + 1, step, dtype=np.int64)
    return np.column_stack([starts, starts + length])


def assign_window_quality(
    bounds: np.ndarray,
    sfreq: float,
    flags: pd.DataFrame,
    reject_if_channels_ge: int,
    automatic_rejection: bool,
    time_offset: float = 0.0,
) -> tuple[list[QualityStatus], list[int]]:
    """Quality status and number of flagged channels per window.

    Only time-localised (``scope == "segment"``) flags are considered. Windows reaching the
    channel threshold are REJECTED when automatic rejection is enabled, otherwise FLAGGED.
    """
    statuses: list[QualityStatus] = []
    counts: list[int] = []
    local = flags[flags["scope"] == "segment"] if not flags.empty else flags
    for start, stop in bounds:
        t0 = time_offset + start / sfreq
        t1 = time_offset + stop / sfreq
        hits = local if local.empty else local[(local["start_s"] < t1) & (local["end_s"] > t0)]
        global_hit = (not hits.empty) and hits["channel"].isna().any()
        n_channels = int(hits["channel"].dropna().nunique()) if not hits.empty else 0
        counts.append(n_channels)
        if (n_channels >= reject_if_channels_ge or global_hit) and automatic_rejection:
            statuses.append(QualityStatus.REJECTED)
        elif n_channels > 0 or global_hit:
            statuses.append(QualityStatus.FLAGGED)
        else:
            statuses.append(QualityStatus.GOOD)
    return statuses, counts
