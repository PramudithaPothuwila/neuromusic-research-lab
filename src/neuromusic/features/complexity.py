"""Hjorth parameters (Hjorth, 1970) along the last axis."""

from __future__ import annotations

import numpy as np


def hjorth(x: np.ndarray) -> dict[str, np.ndarray]:
    """Activity = var(x); mobility = sqrt(var(x') / var(x));
    complexity = mobility(x') / mobility(x). Derivatives are first differences (per sample)."""
    dx = np.diff(x, axis=-1)
    ddx = np.diff(dx, axis=-1)
    var_x = x.var(axis=-1)
    var_dx = dx.var(axis=-1)
    var_ddx = ddx.var(axis=-1)
    with np.errstate(divide="ignore", invalid="ignore"):
        mobility = np.sqrt(np.divide(var_dx, var_x, out=np.zeros_like(var_x), where=var_x > 0))
        mobility_dx = np.sqrt(
            np.divide(var_ddx, var_dx, out=np.zeros_like(var_dx), where=var_dx > 0))
        complexity = np.divide(mobility_dx, mobility, out=np.zeros_like(mobility),
                               where=mobility > 0)
    return {"activity": var_x, "mobility": mobility, "complexity": complexity}
