"""Read raw EEG CSV files into an in-memory representation without modifying them."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from neuromusic.domain.schemas import SAMPLE_INDEX_COLUMN, TIME_COLUMN


class EEGReadError(ValueError):
    """Raised when a file cannot be parsed as an EEG CSV."""


@dataclass(frozen=True)
class RawEEG:
    """Raw time series. ``data`` has shape (n_channels, n_samples), dtype float64."""

    data: np.ndarray
    times: np.ndarray
    sample_index: np.ndarray
    channel_names: list[str]
    source_path: Path
    non_numeric_counts: dict[str, int] = field(default_factory=dict)

    @property
    def n_samples(self) -> int:
        return int(self.data.shape[1])

    @property
    def n_channels(self) -> int:
        return int(self.data.shape[0])


def file_sha256(path: Path, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        while chunk := fh.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def read_header(path: Path) -> list[str]:
    try:
        return list(pd.read_csv(path, nrows=0).columns)
    except (pd.errors.ParserError, pd.errors.EmptyDataError, UnicodeDecodeError) as exc:
        raise EEGReadError(f"{path.name}: unreadable CSV header ({exc})") from exc


def read_eeg_csv(path: Path, channel_prefix: str = "EEG.") -> RawEEG:
    """Parse an EEG CSV.

    Values that cannot be parsed as numbers are converted to NaN and counted per column so the
    validator can report them; nothing is silently dropped or resampled.
    """
    path = Path(path)
    try:
        frame = pd.read_csv(path, dtype=str, keep_default_na=False)
    except (pd.errors.ParserError, pd.errors.EmptyDataError, UnicodeDecodeError) as exc:
        raise EEGReadError(f"{path.name}: unreadable CSV ({exc})") from exc

    columns = list(frame.columns)
    for required in (SAMPLE_INDEX_COLUMN, TIME_COLUMN):
        if required not in columns:
            raise EEGReadError(f"{path.name}: required column {required!r} missing")
    eeg_columns = [c for c in columns if c.startswith(channel_prefix)]
    if not eeg_columns:
        raise EEGReadError(f"{path.name}: no columns with prefix {channel_prefix!r}")

    numeric: dict[str, np.ndarray] = {}
    non_numeric: dict[str, int] = {}
    for col in [SAMPLE_INDEX_COLUMN, TIME_COLUMN, *eeg_columns]:
        raw = frame[col].astype(str).str.strip()
        values = pd.to_numeric(raw, errors="coerce").to_numpy(dtype=np.float64)
        # Empty strings are missing values; anything else that failed to parse is non-numeric.
        bad = np.isnan(values) & (raw.to_numpy() != "") & (raw.str.lower().to_numpy() != "nan")
        if bad.any():
            non_numeric[col] = int(bad.sum())
        numeric[col] = values

    data = np.vstack([numeric[c] for c in eeg_columns])
    return RawEEG(
        data=data,
        times=numeric[TIME_COLUMN],
        sample_index=numeric[SAMPLE_INDEX_COLUMN],
        channel_names=[c[len(channel_prefix):] for c in eeg_columns],
        source_path=path,
        non_numeric_counts=non_numeric,
    )


def select_channels(raw: RawEEG, channels: list[str]) -> RawEEG:
    """Return a copy restricted to ``channels`` in the given order."""
    missing = [c for c in channels if c not in raw.channel_names]
    if missing:
        raise EEGReadError(f"channels not present in recording: {missing}")
    idx = [raw.channel_names.index(c) for c in channels]
    return RawEEG(
        data=raw.data[idx].copy(),
        times=raw.times.copy(),
        sample_index=raw.sample_index.copy(),
        channel_names=list(channels),
        source_path=raw.source_path,
        non_numeric_counts=dict(raw.non_numeric_counts),
    )
