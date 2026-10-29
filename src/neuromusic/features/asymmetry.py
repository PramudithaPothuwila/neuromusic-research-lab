"""Region-level power and hemispheric asymmetry.

Regions and pairs are configurable and restricted to recorded channels. Asymmetry is
ln(P_right) - ln(P_left); its interpretation depends on the (currently unverified) reference.
"""

from __future__ import annotations

import numpy as np


def region_power(abs_power: dict[str, np.ndarray], channel_names: list[str],
                 regions: dict[str, list[str]]) -> dict[str, np.ndarray]:
    """Mean absolute power per region. ``abs_power`` values have shape (..., n_channels)."""
    out: dict[str, np.ndarray] = {}
    for region, members in regions.items():
        idx = [channel_names.index(c) for c in members if c in channel_names]
        if not idx:
            continue
        for band, power in abs_power.items():
            out[f"{band}.{region}"] = power[..., idx].mean(axis=-1)
    return out


def asymmetry(abs_power: dict[str, np.ndarray], channel_names: list[str],
              pairs: list[tuple[str, str]], bands: list[str]) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    tiny = np.finfo(float).tiny
    for right, left in pairs:
        if right not in channel_names or left not in channel_names:
            continue
        r, lft = channel_names.index(right), channel_names.index(left)
        for band in bands:
            if band not in abs_power:
                continue
            p = abs_power[band]
            out[f"{band}.{right}-{left}"] = np.log(p[..., r] + tiny) - np.log(p[..., lft] + tiny)
    return out
