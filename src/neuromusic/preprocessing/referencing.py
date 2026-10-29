"""EEG re-referencing.

The acquisition reference is unknown, so the default preserves the original reference.
Average referencing with only 14 channels is a coarse approximation and should be justified.
"""

from __future__ import annotations

import numpy as np

from neuromusic.config import ReferenceConfig
from neuromusic.domain.enums import ReferenceMethod


def rereference(data: np.ndarray, channel_names: list[str], cfg: ReferenceConfig) -> np.ndarray:
    if cfg.method is ReferenceMethod.PRESERVE_ORIGINAL:
        return data.copy()
    if cfg.method is ReferenceMethod.AVERAGE:
        return data - data.mean(axis=0, keepdims=True)
    if cfg.method is ReferenceMethod.CHANNELS:
        if not cfg.channels:
            raise ValueError("reference method 'channels' requires reference.channels")
        missing = [c for c in cfg.channels if c not in channel_names]
        if missing:
            raise ValueError(f"reference channels not recorded: {missing}")
        idx = [channel_names.index(c) for c in cfg.channels]
        return data - data[idx].mean(axis=0, keepdims=True)
    raise ValueError(f"unsupported reference method {cfg.method}")
