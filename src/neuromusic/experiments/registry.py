"""Experiment records: everything needed to reproduce and audit a run."""

from __future__ import annotations

import hashlib
import json
import platform
import sys
from datetime import datetime
from importlib import metadata
from typing import Any

from pydantic import BaseModel, Field

from neuromusic import PIPELINE_VERSION, __version__
from neuromusic.domain.enums import ExecutionStatus, ExperimentType

TRACKED_PACKAGES = ("numpy", "pandas", "scipy", "mne", "statsmodels", "scikit-learn",
                    "xgboost", "plotly", "matplotlib", "streamlit", "pyarrow", "pydantic")


def environment_info() -> dict[str, Any]:
    packages = {}
    for name in TRACKED_PACKAGES:
        try:
            packages[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            continue
    return {"python": sys.version.split()[0], "platform": platform.platform(),
            "packages": packages}


class ExperimentRecord(BaseModel):
    experiment_id: str
    experiment_type: ExperimentType
    created_at: datetime
    status: ExecutionStatus = ExecutionStatus.PENDING
    dataset_fingerprint: str
    is_synthetic_data: bool = False
    n_participants: int
    song_ids: list[str]
    recording_ids: list[str]
    preprocessing_hash: str
    feature_hash: str
    preprocessing_config: dict[str, Any]
    feature_config: dict[str, Any]
    experiment_config: dict[str, Any]
    models: list[str] = Field(default_factory=list)
    hyperparameters: dict[str, Any] = Field(default_factory=dict)
    cv_strategy: str | None = None
    random_seed: int
    metrics: dict[str, Any] = Field(default_factory=dict)
    results_path: str
    software_version: str = __version__
    pipeline_version: str = PIPELINE_VERSION
    environment: dict[str, Any] = Field(default_factory=environment_info)
    notes: list[str] = Field(default_factory=list)
    error: str | None = None


def make_experiment_id(experiment_type: ExperimentType, config: dict[str, Any],
                       fingerprint: str, when: datetime) -> str:
    """``<type>-<YYYYmmdd-HHMMSS>-<hash(config, data)>``; the hash identifies the setup, the
    timestamp distinguishes repeated runs."""
    digest = hashlib.sha256(json.dumps([config, fingerprint], sort_keys=True,
                                       default=str).encode()).hexdigest()[:8]
    return f"{experiment_type.value}-{when:%Y%m%d-%H%M%S}-{digest}"
