"""Audio decoding through the ``ffmpeg`` executable (handles Opus, MP3, AAC, WAV, ...)."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import numpy as np


class AudioDecodeError(RuntimeError):
    pass


def ffmpeg_available() -> bool:
    return shutil.which("ffmpeg") is not None


def decode_mono(path: Path, sample_rate: int = 22050, duration_s: float | None = None,
                offset_s: float = 0.0) -> np.ndarray:
    """Decode ``path`` to a mono float32 signal at ``sample_rate``.

    Arguments are passed as a list (no shell). ``duration_s`` limits decoding to the segment
    [offset_s, offset_s + duration_s).
    """
    exe = shutil.which("ffmpeg")
    if exe is None:
        raise AudioDecodeError("ffmpeg executable not found on PATH")
    path = Path(path)
    if not path.is_file():
        raise AudioDecodeError(f"audio file not found: {path}")
    cmd = [exe, "-v", "error", "-nostdin"]
    if offset_s > 0:
        cmd += ["-ss", f"{offset_s:.3f}"]
    cmd += ["-i", str(path)]
    if duration_s is not None:
        cmd += ["-t", f"{duration_s:.3f}"]
    cmd += ["-ac", "1", "-ar", str(sample_rate), "-f", "f32le", "-"]
    proc = subprocess.run(cmd, capture_output=True, check=False)
    if proc.returncode != 0:
        raise AudioDecodeError(f"ffmpeg failed for {path.name}: "
                               f"{proc.stderr.decode(errors='replace').strip()[:300]}")
    signal = np.frombuffer(proc.stdout, dtype=np.float32)
    if signal.size == 0:
        raise AudioDecodeError(f"no audio decoded from {path.name}")
    return signal


def probe_duration(path: Path, sample_rate: int = 22050) -> float:
    """Duration in seconds (decodes the file; exact for any container)."""
    return decode_mono(path, sample_rate).size / sample_rate
