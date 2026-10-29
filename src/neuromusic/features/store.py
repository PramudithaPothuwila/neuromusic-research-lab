"""Parquet persistence for feature tables with version and configuration provenance."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from neuromusic import PIPELINE_VERSION, __version__


class FeatureStore:
    def __init__(self, processed_dir: Path) -> None:
        self.root = Path(processed_dir) / "features"

    def set_dir(self, prep_hash: str, feat_hash: str) -> Path:
        return self.root / f"{prep_hash}_{feat_hash}"

    def save(self, prep_hash: str, feat_hash: str, window_features: pd.DataFrame,
             trial_features: pd.DataFrame, manifest: dict[str, Any]) -> Path:
        directory = self.set_dir(prep_hash, feat_hash)
        directory.mkdir(parents=True, exist_ok=True)
        window_features.to_parquet(directory / "window_features.parquet", index=False)
        trial_features.to_parquet(directory / "trial_features.parquet", index=False)
        full = {
            "preprocessing_hash": prep_hash,
            "feature_hash": feat_hash,
            "software_version": __version__,
            "pipeline_version": PIPELINE_VERSION,
            "created_at": datetime.now().isoformat(timespec="seconds"),
            **manifest,
        }
        (directory / "manifest.json").write_text(json.dumps(full, indent=2, default=str),
                                                 encoding="utf-8")
        return directory

    def exists(self, prep_hash: str, feat_hash: str) -> bool:
        return (self.set_dir(prep_hash, feat_hash) / "manifest.json").exists()

    def load(self, prep_hash: str, feat_hash: str, level: str = "trial") -> pd.DataFrame:
        return pd.read_parquet(self.set_dir(prep_hash, feat_hash) / f"{level}_features.parquet")

    def manifest(self, prep_hash: str, feat_hash: str) -> dict[str, Any]:
        path = self.set_dir(prep_hash, feat_hash) / "manifest.json"
        return json.loads(path.read_text(encoding="utf-8"))

    def list_sets(self) -> list[dict[str, Any]]:
        if not self.root.exists():
            return []
        out = []
        for path in sorted(self.root.glob("*/manifest.json")):
            info = json.loads(path.read_text(encoding="utf-8"))
            info["path"] = str(path.parent)
            out.append(info)
        return sorted(out, key=lambda m: m.get("created_at", ""), reverse=True)
