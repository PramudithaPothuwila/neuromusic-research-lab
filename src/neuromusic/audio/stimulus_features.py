"""Acoustic features of the stimulus audio, per song and per recording (heard segment)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from neuromusic.audio.decoding import decode_mono
from neuromusic.audio.features import audio_features
from neuromusic.domain.models import Recording
from neuromusic.ingestion.stimuli import StimulusSource, audio_path, resolve


class MissingAudioError(FileNotFoundError):
    pass


@dataclass
class StimulusAudio:
    """Decodes each song once and serves features for arbitrary opening segments."""

    catalogue: list[StimulusSource]
    audio_dir: Path
    sample_rate: int = 22050

    def __post_init__(self) -> None:
        self._signals: dict[str, np.ndarray] = {}

    def signal(self, song_id: str) -> np.ndarray:
        if song_id not in self._signals:
            source = resolve(self.catalogue, song_id)
            path = audio_path(self.audio_dir, source) if source else None
            if path is None:
                raise MissingAudioError(f"no stimulus audio for song {song_id!r}")
            self._signals[song_id] = decode_mono(path, self.sample_rate).astype(np.float64)
        return self._signals[song_id]

    def duration(self, song_id: str) -> float:
        return self.signal(song_id).size / self.sample_rate

    def features(self, song_id: str, seconds: float | None = None) -> dict[str, float]:
        """Features of the first ``seconds`` of the song (entire song when None)."""
        x = self.signal(song_id)
        if seconds is not None:
            x = x[: round(seconds * self.sample_rate)]
        return audio_features(x, self.sample_rate)

    def available(self, song_ids: list[str]) -> list[str]:
        out = []
        for sid in song_ids:
            source = resolve(self.catalogue, sid)
            if source is not None and audio_path(self.audio_dir, source) is not None:
                out.append(sid)
        return out


def song_feature_table(audio: StimulusAudio, song_ids: list[str],
                       segment_seconds: float | None) -> pd.DataFrame:
    rows = {sid: {"segment_s": min(segment_seconds or np.inf, audio.duration(sid)),
                  "song_duration_s": audio.duration(sid),
                  **audio.features(sid, segment_seconds)} for sid in song_ids}
    return pd.DataFrame(rows).T.rename_axis("song_id")


def recording_feature_table(audio: StimulusAudio, recordings: list[Recording]) -> pd.DataFrame:
    """Features of the audio segment each recording covers (song onset to recording end)."""
    rows = {}
    for rec in recordings:
        if rec.song_id is None:
            continue
        onset = rec.playback_onset_s or 0.0
        heard = max(rec.duration_seconds - onset, 0.0)
        rows[rec.recording_id] = {"song_id": rec.song_id, "participant_id": rec.participant_id,
                                  "heard_s": min(heard, audio.duration(rec.song_id)),
                                  **audio.features(rec.song_id, heard)}
    return pd.DataFrame(rows).T.rename_axis("recording_id")
