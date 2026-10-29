"""Build modelling datasets (X, y, groups) from feature tables."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from neuromusic.features.aggregation import valid_window_mask
from neuromusic.features.extractor import feature_columns


class DatasetError(ValueError):
    pass


@dataclass
class ModelingDataset:
    x: np.ndarray
    y: np.ndarray
    groups: np.ndarray  # participant ids
    trial_ids: np.ndarray  # recording ids, used to aggregate window-level predictions
    feature_names: list[str]
    meta: pd.DataFrame
    level: str

    @property
    def n_participants(self) -> int:
        return len(np.unique(self.groups))

    @property
    def classes(self) -> list[str]:
        return sorted(np.unique(self.y).tolist())


def build_dataset(features: pd.DataFrame, prefixes: list[str] | None, target: str = "song_id",
                  level: str = "trial", min_participants: int = 3,
                  target_values: pd.Series | None = None) -> ModelingDataset:
    """One row per trial (default) or per valid window.

    ``target_values`` (indexed like ``features``) overrides ``features[target]``; used for ratings.
    """
    frame = features
    if level == "trial" and "trial_valid" in frame.columns:
        frame = frame[frame["trial_valid"]]
    elif level == "window":
        frame = frame[valid_window_mask(frame)]
    if target_values is not None:
        frame = frame.assign(_target=target_values.reindex(frame.index))
        target = "_target"
    frame = frame.dropna(subset=[target, "participant_id"])
    cols = feature_columns(frame, prefixes)
    if not cols:
        raise DatasetError(f"no feature columns match prefixes {prefixes}")
    if frame["participant_id"].nunique() < min_participants:
        raise DatasetError(f"{frame['participant_id'].nunique()} participant(s); at least "
                           f"{min_participants} required for subject-independent evaluation")
    meta_cols = [c for c in ("recording_id", "participant_id", "song_id", "trial_id", "window_id")
                 if c in frame.columns]
    return ModelingDataset(
        x=frame[cols].to_numpy(dtype=float),
        y=frame[target].to_numpy(),
        groups=frame["participant_id"].astype(str).to_numpy(),
        trial_ids=frame["recording_id"].astype(str).to_numpy(),
        feature_names=cols,
        meta=frame[meta_cols].reset_index(drop=True),
        level=level,
    )
