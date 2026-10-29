"""Optional emotion/questionnaire rating import.

Ratings are kept exactly as supplied: rating types are not mapped onto one another (e.g.
"happiness" is not treated as valence) and no labels are derived from song names or EEG.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from neuromusic.domain.models import EmotionRating

REQUIRED_COLUMNS = ("participant_id", "song_id", "rating_type", "rating_value",
                    "scale_min", "scale_max")
RATINGS_FILE = "emotion_ratings.csv"


class RatingsError(ValueError):
    pass


def parse_ratings(frame: pd.DataFrame) -> list[EmotionRating]:
    missing = [c for c in REQUIRED_COLUMNS if c not in frame.columns]
    if missing:
        raise RatingsError(f"ratings file missing columns: {missing}")
    frame = frame.copy()
    if "trial_id" not in frame.columns:
        frame["trial_id"] = "T01"
    for col in ("rating_value", "scale_min", "scale_max"):
        frame[col] = pd.to_numeric(frame[col], errors="coerce")
    bad = frame[["rating_value", "scale_min", "scale_max"]].isna().any(axis=1)
    if bad.any():
        raise RatingsError(f"{int(bad.sum())} row(s) with non-numeric rating or scale values")
    out_of_range = (frame["rating_value"] < frame["scale_min"]) | (
        frame["rating_value"] > frame["scale_max"])
    if out_of_range.any():
        raise RatingsError(f"{int(out_of_range.sum())} rating(s) outside their declared scale")
    ratings = []
    for i, row in enumerate(frame.to_dict(orient="records")):
        definition = row.get("definition")
        ratings.append(EmotionRating(
            rating_id=f"R{i + 1:05d}",
            participant_id=str(row["participant_id"]),
            song_id=str(row["song_id"]),
            trial_id=str(row["trial_id"]),
            rating_type=str(row["rating_type"]),
            rating_value=float(row["rating_value"]),
            scale_min=float(row["scale_min"]),
            scale_max=float(row["scale_max"]),
            definition=None if definition is None or pd.isna(definition) else str(definition),
        ))
    return ratings


def import_ratings(source: Path, metadata_dir: Path) -> list[EmotionRating]:
    ratings = parse_ratings(pd.read_csv(source))
    frame = pd.DataFrame([r.model_dump() for r in ratings])
    metadata_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(metadata_dir / RATINGS_FILE, index=False)
    return ratings


def load_ratings(metadata_dir: Path) -> pd.DataFrame:
    path = metadata_dir / RATINGS_FILE
    if not path.exists():
        return pd.DataFrame(columns=["rating_id", *REQUIRED_COLUMNS, "trial_id", "definition"])
    return pd.read_csv(path, dtype={"participant_id": str, "song_id": str, "trial_id": str})
