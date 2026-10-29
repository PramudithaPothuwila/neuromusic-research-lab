"""Window-level feature extraction.

Column naming: ``<family>.<measure>[.<band>].<scope>``. Families: ``spectral``, ``temporal``,
``hjorth``, ``region``, ``asymmetry``. Power is in (signal units)^2; with unconfirmed units these
are arbitrary units (a.u.).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from neuromusic.config import FeatureConfig
from neuromusic.domain.schemas import WINDOW_KEY_COLUMNS
from neuromusic.features.asymmetry import asymmetry, region_power
from neuromusic.features.complexity import hjorth
from neuromusic.features.spectral import (
    band_power,
    check_bands,
    spectral_entropy,
    welch_psd,
)
from neuromusic.features.temporal import temporal_stats
from neuromusic.preprocessing.pipeline import PreprocessingResult

EPS = np.finfo(float).tiny


def _epochs(cleaned: np.ndarray, windows: pd.DataFrame) -> np.ndarray:
    """Stack windows into (n_windows, n_channels, n_samples)."""
    return np.stack([cleaned[:, int(s):int(e)]
                     for s, e in zip(windows["start_sample"], windows["stop_sample"],
                                     strict=True)])


def compute_window_features(epochs: np.ndarray, sfreq: float, channel_names: list[str],
                            cfg: FeatureConfig) -> pd.DataFrame:
    """Feature matrix with one row per epoch."""
    check_bands(cfg.frequency_bands, sfreq, cfg.total_power_range)
    cols: dict[str, np.ndarray] = {}
    freqs, psd = welch_psd(epochs, sfreq, cfg.psd)
    total = band_power(freqs, psd, cfg.total_power_range)
    abs_power: dict[str, np.ndarray] = {}
    for band, edges in cfg.frequency_bands.items():
        abs_power[band] = band_power(freqs, psd, edges)
    for c, ch in enumerate(channel_names):
        for band, p in abs_power.items():
            cols[f"spectral.abs_power.{band}.{ch}"] = p[:, c]
            cols[f"spectral.log_abs_power.{band}.{ch}"] = np.log10(p[:, c] + EPS)
            cols[f"spectral.rel_power.{band}.{ch}"] = p[:, c] / (total[:, c] + EPS)
        for num, den in cfg.ratios:
            if num in abs_power and den in abs_power:
                cols[f"spectral.ratio.{num}_{den}.{ch}"] = (
                    abs_power[num][:, c] / (abs_power[den][:, c] + EPS))
    if cfg.include_spectral_entropy:
        ent = spectral_entropy(freqs, psd, cfg.psd.fmin, cfg.psd.fmax)
        for c, ch in enumerate(channel_names):
            cols[f"spectral.entropy.{ch}"] = ent[:, c]
    if cfg.include_temporal:
        for name, values in temporal_stats(epochs).items():
            for c, ch in enumerate(channel_names):
                cols[f"temporal.{name}.{ch}"] = values[:, c]
    if cfg.include_hjorth:
        for name, values in hjorth(epochs).items():
            for c, ch in enumerate(channel_names):
                cols[f"hjorth.{name}.{ch}"] = values[:, c]
    if cfg.regional_features_enabled:
        reg_abs = region_power(abs_power, channel_names, cfg.regions)
        reg_total = region_power({"total": total}, channel_names, cfg.regions)
        for key, p in reg_abs.items():
            region = key.split(".", 1)[1]
            cols[f"region.abs_power.{key}"] = p
            cols[f"region.log_abs_power.{key}"] = np.log10(p + EPS)
            cols[f"region.rel_power.{key}"] = p / (reg_total[f"total.{region}"] + EPS)
        for key, value in asymmetry(abs_power, channel_names, cfg.asymmetry_pairs,
                                    cfg.asymmetry_bands).items():
            cols[f"asymmetry.{key}"] = value
    return pd.DataFrame(cols)


def extract_window_features(result: PreprocessingResult, cfg: FeatureConfig) -> pd.DataFrame:
    windows = result.windows.reset_index(drop=True)
    if windows.empty:
        return pd.DataFrame(columns=[*WINDOW_KEY_COLUMNS, "segment"])
    epochs = _epochs(result.cleaned, windows)
    features = compute_window_features(epochs, result.sfreq, result.channel_names, cfg)
    keys = windows[[*WINDOW_KEY_COLUMNS, "segment", "n_flagged_channels"]]
    frame = pd.concat([keys, features], axis=1)
    baseline = frame["segment"] == "baseline"
    if baseline.any():
        log_cols = [c for c in features.columns if c.startswith("spectral.log_abs_power.")]
        ref = frame.loc[baseline, log_cols].median()
        corrected = frame[log_cols] - ref
        corrected.columns = [c.replace("spectral.log_abs_power.", "spectral.bc_log_abs_power.")
                             for c in log_cols]
        frame = pd.concat([frame, corrected], axis=1)
    return frame


def feature_columns(frame: pd.DataFrame, prefixes: list[str] | None = None) -> list[str]:
    """Numeric feature columns, optionally restricted to those starting with any prefix."""
    excluded = {*WINDOW_KEY_COLUMNS, "segment", "n_flagged_channels", "n_windows_total",
                "n_windows_valid", "trial_valid", "source_id"}
    cols = [c for c in frame.columns
            if c not in excluded and "." in c and pd.api.types.is_numeric_dtype(frame[c])]
    if prefixes:
        cols = [c for c in cols if any(c.startswith(p) for p in prefixes)]
    return cols


def feature_families(columns: list[str]) -> list[str]:
    """Unique ``family.measure.`` prefixes, e.g. ``spectral.rel_power.``."""
    seen: dict[str, None] = {}
    for c in columns:
        parts = c.split(".")
        seen[".".join(parts[:2]) + "."] = None
    return list(seen)


def feature_unit(column: str, signal_unit: str) -> str:
    """Unit string for a feature column. ``signal_unit`` is 'a.u.' when not validated."""
    measure = column.split(".")[1] if "." in column else ""
    family = column.split(".")[0]
    if family == "var_iqr":
        return feature_unit(column.split(".", 1)[1], signal_unit)
    if family in ("spectral", "region"):
        return {
            "abs_power": f"{signal_unit}²",
            "log_abs_power": f"log10({signal_unit}²)",
            "bc_log_abs_power": "log10 ratio to baseline",
            "rel_power": "fraction of 1-40 Hz power",
            "ratio": "ratio",
            "entropy": "normalised (0-1)",
        }.get(measure, "")
    if family == "asymmetry":
        return "ln(right) - ln(left)"
    if family == "temporal":
        return f"{signal_unit}²" if measure == "var" else signal_unit
    if family == "hjorth":
        return {"activity": f"{signal_unit}²", "mobility": "per sample",
                "complexity": "dimensionless"}.get(measure, "")
    return ""
