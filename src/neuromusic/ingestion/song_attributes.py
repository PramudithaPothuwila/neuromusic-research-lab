"""Song-level stimulus attributes (``data/metadata/song_attributes.csv``).

The source file is kept verbatim (including original header spellings). Columns are mapped to
normalised names on load. Provenance assumptions, all unverified:

* ``*_proxy`` and acoustic columns (tempo, energy, brightness, flatness) appear to be computed
  from audio; the extraction tool and scales are undocumented.
* ``*_survey`` columns appear to be aggregate (song-level) survey ratings; the respondents,
  scale anchors and whether respondents overlap with EEG participants are undocumented.

These are properties of songs, not of participants' trials. They must not be used as
per-participant emotion labels; any association with EEG has n = number of songs.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from neuromusic.ingestion.metadata_parser import song_key

SONG_ATTRIBUTES_FILE = "song_attributes.csv"

COLUMN_MAP = {
    "Title": "song_name",
    "Valence_Proxy": "valence_proxy",
    "Valence_Survay": "valence_survey",
    "Arousal_Proxy": "arousal_proxy",
    "Arousal_Survay": "arousal_survey",
    "Danceability": "danceability_proxy",
    "Danceabitity(Survay)": "danceability_survey",
    "Tempo_BPM": "tempo_bpm",
    "Average of Perceived Speed (Tempo / BPM Equivalent)(Survay)": "perceived_speed_survey",
    "Energy": "energy_proxy",
    "Brightness_Hz": "brightness_hz",
    "Flatness": "spectral_flatness",
}


def load_song_attributes(metadata_dir: Path) -> pd.DataFrame:
    """Attributes indexed by ``song_id`` (``song_key`` of the title); empty if absent."""
    path = Path(metadata_dir) / SONG_ATTRIBUTES_FILE
    if not path.exists():
        return pd.DataFrame()
    frame = pd.read_csv(path).rename(columns=COLUMN_MAP)
    frame["song_id"] = frame["song_name"].map(song_key)
    if frame["song_id"].duplicated().any():
        raise ValueError("song attribute titles map to duplicate song ids")
    return frame.set_index("song_id")


def attribute_columns(frame: pd.DataFrame) -> list[str]:
    return [c for c in frame.columns if c != "song_name"]


def match_report(attributes: pd.DataFrame, registered_song_ids: list[str]) -> dict[str, list[str]]:
    """Which registered songs have attributes and which attribute rows have no recordings."""
    known = set(attributes.index) if not attributes.empty else set()
    registered = set(registered_song_ids)
    return {
        "matched": sorted(registered & known),
        "recordings_without_attributes": sorted(registered - known),
        "attributes_without_recordings": sorted(known - registered),
    }
