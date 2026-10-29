from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from neuromusic.config import DatasetConfig
from neuromusic.domain.schemas import DEFAULT_CHANNELS

SFREQ = 256.0


def make_frame(duration_s: float = 20.0, sfreq: float = SFREQ,
               channels: tuple[str, ...] = DEFAULT_CHANNELS, seed: int = 0,
               freq_hz: float = 10.0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n = int(duration_s * sfreq)
    t = np.arange(n) / sfreq
    data = {f"EEG.{c}": 4300 + 10 * np.sin(2 * np.pi * freq_hz * t + i)
            + rng.normal(0, 2, n) for i, c in enumerate(channels)}
    frame = pd.DataFrame(data)
    frame.insert(0, "time_s", np.round(t, 6))
    frame.insert(0, "sample_index", np.arange(n))
    return frame


@pytest.fixture
def dataset_config() -> DatasetConfig:
    return DatasetConfig()


@pytest.fixture
def write_csv(tmp_path: Path) -> Callable[..., Path]:
    def _write(name: str = "SRC001_SOURCE_P001_Amude.csv", frame: pd.DataFrame | None = None,
               directory: Path | None = None, **kwargs) -> Path:
        directory = directory or tmp_path
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / name
        (frame if frame is not None else make_frame(**kwargs)).to_csv(path, index=False)
        return path

    return _write
