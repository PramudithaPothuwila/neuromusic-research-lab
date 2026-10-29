"""Synthetic EEG generator for tests and demonstrations.

Every generated file carries ``SYNTHETIC`` in its filename; the registry marks such recordings
and the dashboard and reports label them as synthetic. Synthetic data must never be presented
as study results.
"""

from __future__ import annotations

import zlib
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from neuromusic.domain.schemas import DEFAULT_CHANNELS, SAMPLE_INDEX_COLUMN, TIME_COLUMN

POSTERIOR = {"O1", "O2", "P7", "P8"}
FRONTAL = {"AF3", "AF4", "F3", "F4", "F7", "F8", "FC5", "FC6"}


@dataclass
class SyntheticSpec:
    n_participants: int = 8
    songs: tuple[str, ...] = ("SongA", "SongB", "SongC")
    duration_s: float = 60.0
    sfreq: float = 256.0
    channels: tuple[str, ...] = DEFAULT_CHANNELS
    # Multiplicative alpha (posterior) and beta (frontal) amplitude per song.
    alpha_gain: dict[str, float] = field(default_factory=lambda: {
        "SongA": 1.0, "SongB": 1.6, "SongC": 0.8})
    beta_gain: dict[str, float] = field(default_factory=lambda: {
        "SongA": 1.0, "SongB": 0.9, "SongC": 1.5})
    shared_response_strength: float = 0.6
    add_artifacts: bool = True
    dc_offset: float = 4300.0
    seed: int = 7


def _pink_noise(rng: np.random.Generator, n_ch: int, n: int) -> np.ndarray:
    white = rng.standard_normal((n_ch, n))
    spectrum = np.fft.rfft(white, axis=1)
    f = np.fft.rfftfreq(n)
    f[0] = f[1]
    spectrum /= np.sqrt(f)
    pink = np.fft.irfft(spectrum, n=n, axis=1)
    return pink / pink.std(axis=1, keepdims=True)


def generate_recording(spec: SyntheticSpec, participant: int, song: str,
                       rng: np.random.Generator) -> pd.DataFrame:
    n = int(spec.duration_s * spec.sfreq)
    t = np.arange(n) / spec.sfreq
    channels = list(spec.channels)
    n_ch = len(channels)
    subj = np.random.default_rng(spec.seed * 1000 + participant)
    subj_scale = subj.uniform(0.7, 1.4)
    alpha_freq = subj.uniform(9.0, 11.5)
    data = 6.0 * _pink_noise(rng, n_ch, n) * subj_scale

    song_rng = np.random.default_rng(zlib.crc32(song.encode()))
    # Song-locked slow envelope shared across participants (enables synchronization demos).
    envelope = np.convolve(song_rng.standard_normal(n), np.ones(256) / 256, mode="same")
    envelope = 1.0 + spec.shared_response_strength * envelope / (envelope.std() + 1e-12)
    for c, ch in enumerate(channels):
        phase = rng.uniform(0, 2 * np.pi)
        alpha_amp = 4.0 * spec.alpha_gain.get(song, 1.0) * (1.6 if ch in POSTERIOR else 0.6)
        beta_amp = 1.5 * spec.beta_gain.get(song, 1.0) * (1.4 if ch in FRONTAL else 0.7)
        theta_amp = 2.0
        data[c] += alpha_amp * subj_scale * envelope * np.sin(2 * np.pi * alpha_freq * t + phase)
        data[c] += beta_amp * np.sin(2 * np.pi * 20.0 * t + rng.uniform(0, 2 * np.pi))
        data[c] += theta_amp * np.sin(2 * np.pi * 6.0 * t + rng.uniform(0, 2 * np.pi))
        data[c] += spec.dc_offset + 300 * subj.standard_normal() + 2.0 * t / spec.duration_s

    if spec.add_artifacts:
        # One movement burst across all channels and a brief flatline on one channel.
        burst = int(rng.uniform(0.3, 0.7) * n)
        width = int(0.5 * spec.sfreq)
        data[:, burst:burst + width] += 120.0 * np.hanning(width)
        flat_ch = int(rng.integers(n_ch))
        f0 = int(0.15 * n)
        data[flat_ch, f0:f0 + int(1.5 * spec.sfreq)] = data[flat_ch, f0]

    frame = pd.DataFrame(data.T, columns=[f"EEG.{c}" for c in channels])
    frame.insert(0, TIME_COLUMN, np.round(t, 6))
    frame.insert(0, SAMPLE_INDEX_COLUMN, np.arange(n))
    return frame


def write_dataset(directory: Path, spec: SyntheticSpec | None = None) -> list[Path]:
    """Write one CSV per participant x song. Returns written paths."""
    spec = spec or SyntheticSpec()
    directory.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(spec.seed)
    paths = []
    for p in range(1, spec.n_participants + 1):
        for song in spec.songs:
            frame = generate_recording(spec, p, song, rng)
            path = directory / f"SYN001_SYNTHETIC_P{p:03d}_{song}.csv"
            frame.to_csv(path, index=False, float_format="%.6f")
            paths.append(path)
    return paths
