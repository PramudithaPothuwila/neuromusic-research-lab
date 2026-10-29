"""Time-domain amplitude statistics computed along the last axis."""

from __future__ import annotations

import numpy as np


def temporal_stats(x: np.ndarray) -> dict[str, np.ndarray]:
    median = np.median(x, axis=-1, keepdims=True)
    return {
        "mean": x.mean(axis=-1),
        "std": x.std(axis=-1, ddof=1),
        "var": x.var(axis=-1, ddof=1),
        "rms": np.sqrt(np.mean(x**2, axis=-1)),
        "ptp": np.ptp(x, axis=-1),
        "mad": np.median(np.abs(x - median), axis=-1),
    }
