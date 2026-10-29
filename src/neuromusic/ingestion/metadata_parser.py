"""Configurable filename parsing.

The default pattern interprets ``SRC001_SOURCE_P001_Amude.csv`` as source ``SRC001``,
participant ``P001`` and song ``Amude``. This interpretation is provisional; the meaning of the
second token (``SOURCE``) is not assumed. Parsed metadata can be corrected via the registry.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from neuromusic.config import FilenamePatternConfig


@dataclass(frozen=True)
class ParsedFilename:
    source_id: str | None
    participant_id: str | None
    song_name: str | None
    song_id: str | None
    trial_id: str | None
    matched: bool


def slugify(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", value.strip().lower()).strip("_")
    return slug or "unknown"


def song_key(value: str) -> str:
    """Canonical song id: lowercase alphanumerics only, so "Hadisida Manda",
    "Hadisida_Manda" and "HadisidaManda" all map to ``hadisidamanda``."""
    key = re.sub(r"[^a-z0-9]+", "", value.strip().lower())
    return key or "unknown"


class FilenameParser:
    def __init__(self, config: FilenamePatternConfig | None = None) -> None:
        self.config = config or FilenamePatternConfig()
        self._pattern = re.compile(self.config.regex)
        missing = {"participant_id", "song_id"} - set(self._pattern.groupindex)
        if missing:
            raise ValueError(f"filename regex must define named groups: {sorted(missing)}")

    def parse(self, filename: str | Path) -> ParsedFilename:
        stem = Path(filename).stem
        match = self._pattern.match(stem)
        if match is None:
            return ParsedFilename(None, None, None, None, None, matched=False)
        groups = match.groupdict()
        song_name = groups.get("song_id")
        if song_name is not None:
            song_name = self.config.song_aliases.get(song_name, song_name)
        return ParsedFilename(
            source_id=groups.get("source_id"),
            participant_id=groups.get("participant_id"),
            song_name=song_name,
            song_id=song_key(song_name) if song_name else None,
            trial_id=groups.get("trial_id"),
            matched=True,
        )
