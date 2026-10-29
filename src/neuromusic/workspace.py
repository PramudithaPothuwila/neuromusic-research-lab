"""Workspace: project root, configuration, directory layout and registry access."""

from __future__ import annotations

import os
from functools import cached_property
from pathlib import Path

from pydantic import BaseModel

from neuromusic.config import (
    CONFIG_FILES,
    ApplicationConfig,
    DatasetConfig,
    ExperimentsConfig,
    FeatureConfig,
    PreprocessingConfig,
    StatisticsConfig,
    config_hash,
    dump_yaml_model,
    load_yaml_model,
)
from neuromusic.domain.enums import SignalUnits
from neuromusic.ingestion.dataset_registry import DatasetRegistry

ROOT_ENV = "NEUROMUSIC_ROOT"


def find_root(start: Path | None = None) -> Path:
    """Project root: $NEUROMUSIC_ROOT, else the nearest ancestor containing ``configs/``."""
    if env := os.environ.get(ROOT_ENV):
        return Path(env).resolve()
    current = (start or Path.cwd()).resolve()
    for candidate in (current, *current.parents):
        if (candidate / "configs").is_dir():
            return candidate
    return current


class Workspace:
    def __init__(self, root: Path | None = None) -> None:
        self.root = Path(root).resolve() if root is not None else find_root()
        self.config_dir = self.root / "configs"
        self.reload_configs()

    # ------------------------------------------------------------------ configs

    def reload_configs(self) -> None:
        def load(name: str) -> BaseModel:
            filename, model, key = CONFIG_FILES[name]
            return load_yaml_model(self.config_dir / filename, model, key)

        self.application: ApplicationConfig = load("application")  # type: ignore[assignment]
        self.dataset: DatasetConfig = load("dataset")  # type: ignore[assignment]
        self.preprocessing: PreprocessingConfig = load("preprocessing")  # type: ignore[assignment]
        self.features: FeatureConfig = load("features")  # type: ignore[assignment]
        self.statistics: StatisticsConfig = load("statistics")  # type: ignore[assignment]
        self.experiments: ExperimentsConfig = load("experiments")  # type: ignore[assignment]
        self.__dict__.pop("registry", None)

    def save_config(self, name: str, cfg: BaseModel) -> Path:
        filename, model, key = CONFIG_FILES[name]
        validated = model.model_validate(cfg.model_dump())
        path = self.config_dir / filename
        dump_yaml_model(validated, path, key)
        self.reload_configs()
        return path

    # -------------------------------------------------------------------- paths

    def _path(self, rel: str) -> Path:
        path = Path(rel)
        return path if path.is_absolute() else self.root / path

    @property
    def raw_dir(self) -> Path:
        return self._path(self.dataset.raw_dir)

    @property
    def metadata_dir(self) -> Path:
        return self._path(self.dataset.metadata_dir)

    @property
    def interim_dir(self) -> Path:
        return self._path(self.application.interim_dir)

    @property
    def processed_dir(self) -> Path:
        return self._path(self.application.processed_dir)

    @property
    def experiments_dir(self) -> Path:
        return self._path(self.application.experiments_dir)

    @property
    def reports_dir(self) -> Path:
        return self._path(self.application.reports_dir)

    # -------------------------------------------------------------- registry

    @cached_property
    def registry(self) -> DatasetRegistry:
        return DatasetRegistry(self.raw_dir, self.metadata_dir, self.dataset)

    # ---------------------------------------------------------------- hashes

    @property
    def preprocessing_hash(self) -> str:
        from neuromusic.preprocessing.pipeline import preprocessing_hash

        return preprocessing_hash(self.preprocessing, self.dataset)

    @property
    def feature_hash(self) -> str:
        return config_hash(self.features)

    # ---------------------------------------------------------------- units

    @property
    def units_validated(self) -> bool:
        return self.preprocessing.signal_units is not SignalUnits.UNKNOWN

    @property
    def signal_unit_label(self) -> str:
        """Unit used in labels: the configured physical unit, or 'a.u.' if unvalidated."""
        if not self.units_validated:
            return "a.u."
        return "µV" if self.preprocessing.signal_units is SignalUnits.MICROVOLTS else "V"
