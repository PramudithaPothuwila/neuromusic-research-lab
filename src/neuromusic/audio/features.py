"""Acoustic descriptors of a mono audio signal.

Frame-based STFT (Hann window). Feature groups:

* ``loudness``: RMS level mean / SD / dynamic range in dB relative to full scale. Absolute
  level depends on mastering; playback volume during acquisition is unknown.
* ``spectral``: centroid ("brightness", Hz), bandwidth (Hz), 85 % roll-off (Hz), flatness,
  zero-crossing rate, low/mid/high energy fractions (< 250 Hz, 250-4000 Hz, > 4000 Hz).
* ``rhythm``: onset (spectral-flux) rate and strength, and a tempo estimate from the
  prior-weighted autocorrelation of the onset envelope (60-200 BPM; octave errors remain
  possible).
* ``timbre``: mean MFCC 1-12 (40-band mel filterbank, log power, orthonormal DCT-II).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import signal as sps
from scipy.fft import dct

EPS = 1e-12
FEATURE_GROUPS = ("loudness", "spectral", "rhythm", "timbre")


@dataclass(frozen=True)
class STFTConfig:
    n_fft: int = 2048
    hop: int = 512
    n_mels: int = 40
    n_mfcc: int = 12
    tempo_range_bpm: tuple[float, float] = (60.0, 200.0)


def _frames(x: np.ndarray, n_fft: int, hop: int) -> np.ndarray:
    if x.size < n_fft:
        x = np.pad(x, (0, n_fft - x.size))
    return np.lib.stride_tricks.sliding_window_view(x, n_fft)[::hop]


def mel_filterbank(sr: int, n_fft: int, n_mels: int, fmin: float = 0.0,
                   fmax: float | None = None) -> np.ndarray:
    """Triangular mel filters (HTK mel scale), shape (n_mels, n_fft // 2 + 1)."""
    fmax = fmax or sr / 2.0

    def hz_to_mel(f: np.ndarray | float) -> np.ndarray:
        return 2595.0 * np.log10(1.0 + np.asarray(f) / 700.0)

    def mel_to_hz(m: np.ndarray) -> np.ndarray:
        return 700.0 * (10.0 ** (m / 2595.0) - 1.0)

    edges = mel_to_hz(np.linspace(hz_to_mel(fmin), hz_to_mel(fmax), n_mels + 2))
    freqs = np.fft.rfftfreq(n_fft, 1.0 / sr)
    bank = np.zeros((n_mels, freqs.size))
    for m in range(n_mels):
        lo, mid, hi = edges[m], edges[m + 1], edges[m + 2]
        rise = (freqs - lo) / max(mid - lo, EPS)
        fall = (hi - freqs) / max(hi - mid, EPS)
        bank[m] = np.clip(np.minimum(rise, fall), 0.0, None)
    return bank


def estimate_tempo(onset_env: np.ndarray, frame_rate: float,
                   bpm_range: tuple[float, float] = (60.0, 200.0),
                   prior_bpm: float = 120.0, prior_octaves: float = 1.0) -> float:
    """Tempo (BPM) from the autocorrelation of the smoothed onset envelope.

    The unbiased autocorrelation is weighted by a log-normal tempo prior centred on
    ``prior_bpm`` (SD ``prior_octaves`` octaves), the usual guard against octave errors.
    """
    env = np.convolve(onset_env, np.hanning(5) / np.hanning(5).sum(), mode="same")
    env = env - env.mean()
    if env.size < 4 or not np.any(env):
        return float("nan")
    ac = sps.correlate(env, env, mode="full", method="fft")[env.size - 1:]
    ac = ac / (env.size - np.arange(ac.size))  # unbiased: remove the bias towards short lags
    lags = np.arange(ac.size)
    lo = int(np.floor(60.0 * frame_rate / bpm_range[1]))
    hi = int(np.ceil(60.0 * frame_rate / bpm_range[0]))
    sel = (lags >= max(lo, 1)) & (lags <= min(hi, ac.size // 2))
    if not sel.any():
        return float("nan")
    bpm = 60.0 * frame_rate / lags[sel]
    weight = np.exp(-0.5 * (np.log2(bpm / prior_bpm) / prior_octaves) ** 2)
    best = lags[sel][np.argmax(ac[sel] * weight)]
    return float(60.0 * frame_rate / best)


def audio_features(x: np.ndarray, sr: int, cfg: STFTConfig | None = None) -> dict[str, float]:
    cfg = cfg or STFTConfig()
    x = np.asarray(x, dtype=np.float64)
    frames = _frames(x, cfg.n_fft, cfg.hop)
    window = np.hanning(cfg.n_fft)
    spec = np.abs(np.fft.rfft(frames * window, axis=1))
    power = spec**2
    freqs = np.fft.rfftfreq(cfg.n_fft, 1.0 / sr)
    total = power.sum(axis=1) + EPS
    frame_rate = sr / cfg.hop

    rms_db = 10.0 * np.log10(np.mean(frames**2, axis=1) + EPS)
    audible = rms_db > rms_db.max() - 60.0  # ignore digital silence for spectral averages
    p = power[audible]
    t = total[audible]
    # Centroid, bandwidth and roll-off are magnitude-weighted (conventional definition).
    mag = spec[audible]
    m_total = mag.sum(axis=1) + EPS
    centroid = (mag * freqs).sum(axis=1) / m_total
    bandwidth = np.sqrt((mag * (freqs[None, :] - centroid[:, None]) ** 2).sum(axis=1) / m_total)
    cumulative = np.cumsum(mag, axis=1) / m_total[:, None]
    rolloff = freqs[np.argmax(cumulative >= 0.85, axis=1)]
    flatness = np.exp(np.mean(np.log(p + EPS), axis=1)) / (np.mean(p, axis=1) + EPS)
    zcr = np.mean(np.abs(np.diff(np.signbit(frames[audible]), axis=1)), axis=1)

    def band(lo: float, hi: float) -> float:
        mask = (freqs >= lo) & (freqs < hi)
        return float(np.mean(p[:, mask].sum(axis=1) / t))

    log_spec = np.log1p(1000.0 * spec)
    flux = np.maximum(np.diff(log_spec, axis=0), 0.0).sum(axis=1)
    flux = flux / (flux.max() + EPS)
    peaks, _ = sps.find_peaks(flux, height=flux.mean() + flux.std(),
                              distance=max(1, int(0.1 * frame_rate)))
    duration = x.size / sr

    mel = mel_filterbank(sr, cfg.n_fft, cfg.n_mels) @ p.T
    mfcc = dct(np.log(mel + EPS), type=2, norm="ortho", axis=0)[1:cfg.n_mfcc + 1]

    feats = {
        "loudness.rms_db_mean": float(rms_db[audible].mean()),
        "loudness.rms_db_sd": float(rms_db[audible].std()),
        "loudness.dynamic_range_db": float(np.percentile(rms_db[audible], 95)
                                           - np.percentile(rms_db[audible], 5)),
        "spectral.centroid_hz": float(centroid.mean()),
        "spectral.bandwidth_hz": float(bandwidth.mean()),
        "spectral.rolloff_hz": float(rolloff.mean()),
        "spectral.flatness": float(flatness.mean()),
        "spectral.zero_crossing_rate": float(zcr.mean()),
        "spectral.low_energy_fraction": band(0.0, 250.0),
        "spectral.mid_energy_fraction": band(250.0, 4000.0),
        "spectral.high_energy_fraction": band(4000.0, sr / 2.0 + 1.0),
        "rhythm.onset_rate_hz": float(len(peaks) / duration) if duration > 0 else float("nan"),
        "rhythm.onset_strength": float(flux.mean()),
        "rhythm.tempo_bpm": estimate_tempo(flux, frame_rate, cfg.tempo_range_bpm),
    }
    for i, value in enumerate(mfcc.mean(axis=1), start=1):
        feats[f"timbre.mfcc{i:02d}"] = float(value)
    return feats
