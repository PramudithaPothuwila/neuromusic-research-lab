"""Experiment artifact storage. Existing experiments are never overwritten implicitly."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from neuromusic.experiments.registry import ExperimentRecord


class ExperimentExistsError(FileExistsError):
    pass


class ExperimentStore:
    def __init__(self, root: Path) -> None:
        self.root = Path(root)

    def path(self, experiment_id: str) -> Path:
        if "/" in experiment_id or "\\" in experiment_id or ".." in experiment_id:
            raise ValueError(f"invalid experiment id {experiment_id!r}")
        return self.root / experiment_id

    def create(self, record: ExperimentRecord) -> Path:
        directory = self.path(record.experiment_id)
        if directory.exists():
            raise ExperimentExistsError(f"experiment {record.experiment_id} already exists")
        directory.mkdir(parents=True)
        self.write_record(record)
        return directory

    def write_record(self, record: ExperimentRecord) -> None:
        (self.path(record.experiment_id) / "record.json").write_text(
            record.model_dump_json(indent=2), encoding="utf-8")

    def save_table(self, experiment_id: str, name: str, frame: pd.DataFrame) -> Path:
        path = self.path(experiment_id) / f"{name}.parquet"
        out = frame.copy()
        for col in out.columns:
            if out[col].dtype == object:
                out[col] = out[col].map(
                    lambda v: json.dumps(v, default=str) if isinstance(v, dict | list) else v)
        out.to_parquet(path, index=False)
        out.to_csv(path.with_suffix(".csv"), index=False)
        return path

    def save_json(self, experiment_id: str, name: str, payload: Any) -> Path:
        path = self.path(experiment_id) / f"{name}.json"
        path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
        return path

    def save_model(self, experiment_id: str, name: str, model: Any) -> Path:
        path = self.path(experiment_id) / "models" / f"{name}.joblib"
        path.parent.mkdir(exist_ok=True)
        joblib.dump(model, path)
        return path

    def load_record(self, experiment_id: str) -> ExperimentRecord:
        return ExperimentRecord.model_validate_json(
            (self.path(experiment_id) / "record.json").read_text(encoding="utf-8"))

    def load_table(self, experiment_id: str, name: str) -> pd.DataFrame:
        return pd.read_parquet(self.path(experiment_id) / f"{name}.parquet")

    def load_json(self, experiment_id: str, name: str) -> Any:
        return json.loads((self.path(experiment_id) / f"{name}.json").read_text(encoding="utf-8"))

    def has(self, experiment_id: str, name: str) -> bool:
        directory = self.path(experiment_id)
        return any((directory / f"{name}{ext}").exists() for ext in (".parquet", ".json"))

    def list_records(self, experiment_type: str | None = None) -> list[ExperimentRecord]:
        if not self.root.exists():
            return []
        records = []
        for path in self.root.glob("*/record.json"):
            try:
                rec = ExperimentRecord.model_validate_json(path.read_text(encoding="utf-8"))
            except ValueError:
                continue
            if experiment_type is None or rec.experiment_type.value == experiment_type:
                records.append(rec)
        return sorted(records, key=lambda r: r.created_at, reverse=True)

    def delete(self, experiment_id: str, *, confirm: bool) -> None:
        """Remove an experiment directory. Requires ``confirm=True`` (explicit user action)."""
        if not confirm:
            raise PermissionError("deleting an experiment requires explicit confirmation")
        shutil.rmtree(self.path(experiment_id))
