"""Inter-subject correlation (ISC) of EEG time courses during the same song.

This analysis is only meaningful when every recording has a verified playback onset. Without
trustworthy alignment it is reported as unavailable rather than computed. Significance uses
circular-shift surrogates, which preserve each signal's autocorrelation while destroying
stimulus-locked alignment. Shared EEG time courses do not imply shared subjective experience.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from statsmodels.stats.multitest import multipletests

from neuromusic.config import SynchronizationConfig
from neuromusic.domain.models import Recording
from neuromusic.preprocessing.pipeline import PreprocessingResult


@dataclass
class SyncAvailability:
    available: bool
    reasons: list[str]
    eligible_recording_ids: list[str] = field(default_factory=list)


@dataclass
class SyncResult:
    song_id: str
    channels: list[str]
    recording_ids: list[str]
    participant_ids: list[str]
    duration_s: float
    per_channel: pd.DataFrame
    pairs: pd.DataFrame
    sliding: pd.DataFrame
    warnings: list[str]


def check_preconditions(recordings: list[Recording], song_id: str) -> SyncAvailability:
    reasons: list[str] = []
    song_recs = [r for r in recordings if r.song_id == song_id and r.is_analysable]
    eligible = [r for r in song_recs if r.alignment_verified and r.playback_onset_s is not None]
    unaligned = len(song_recs) - len(eligible)
    if unaligned:
        reasons.append(f"{unaligned} recording(s) of '{song_id}' lack a verified playback onset.")
    if len({r.participant_id for r in eligible}) < 2:
        reasons.append("At least two participants with verified playback alignment are required.")
    available = len({r.participant_id for r in eligible}) >= 2
    rates = {round(r.sampling_rate or 0.0, 3) for r in eligible}
    if len(rates) > 1:
        reasons.append(f"Sampling rates differ across recordings: {sorted(rates)}.")
        available = False
    if eligible and not set(eligible[0].channel_names).intersection(
            *[r.channel_names for r in eligible]):
        reasons.append("No channels are shared by all aligned recordings.")
        available = False
    return SyncAvailability(available=available, reasons=reasons,
                            eligible_recording_ids=[r.recording_id for r in eligible])


def _zscore(x: np.ndarray, axis: int = -1) -> np.ndarray:
    sd = x.std(axis=axis, keepdims=True)
    return (x - x.mean(axis=axis, keepdims=True)) / np.where(sd > 0, sd, 1.0)


def mean_pairwise_r(z: np.ndarray) -> np.ndarray:
    """Mean pairwise Pearson r across subjects for z-scored data of shape (n_subj, n_ch, T).

    Uses sum_{i<j} z_i . z_j = (|sum_i z_i|^2 - sum_i |z_i|^2) / 2 for O(n) cost.
    """
    n, _, t = z.shape
    total = z.sum(axis=0)
    cross = ((total**2).sum(axis=-1) - (z**2).sum(axis=(0, 2))) / 2.0
    return cross / (t * n * (n - 1) / 2.0)


def compute_isc(
    results: list[PreprocessingResult],
    recordings: list[Recording],
    song_id: str,
    cfg: SynchronizationConfig,
) -> SyncResult:
    rec_by_id = {r.recording_id: r for r in recordings}
    warnings_: list[str] = []
    channels = cfg.channels or sorted(
        set(results[0].channel_names).intersection(*[r.channel_names for r in results]),
        key=results[0].channel_names.index)
    sfreq = results[0].sfreq
    segments = []
    for res in results:
        rec = rec_by_id[res.recording_id]
        onset = rec.playback_onset_s or 0.0
        start = int(np.searchsorted(res.times, onset - 0.5 / sfreq))
        idx = [res.channel_names.index(c) for c in channels]
        segments.append(res.cleaned[idx, start:])
    length = min(s.shape[1] for s in segments)
    if length < sfreq * cfg.sliding_window_seconds:
        warnings_.append("Common aligned duration is shorter than one sliding window.")
    data = np.stack([s[:, :length] for s in segments])
    z = _zscore(data)
    n_subj = z.shape[0]

    observed = mean_pairwise_r(z)
    rng = np.random.default_rng(cfg.random_seed)
    min_shift = int(cfg.min_shift_seconds * sfreq)
    null = np.empty((cfg.n_surrogates, len(channels)))
    if length > 2 * min_shift:
        for k in range(cfg.n_surrogates):
            shifts = rng.integers(min_shift, length - min_shift, size=n_subj)
            shifted = np.stack([np.roll(z[i], shifts[i], axis=-1) for i in range(n_subj)])
            null[k] = mean_pairwise_r(shifted)
        p = (1 + (null >= observed).sum(axis=0)) / (1 + cfg.n_surrogates)
    else:
        warnings_.append("Recording too short for circular-shift surrogates.")
        p = np.full(len(channels), np.nan)
    per_channel = pd.DataFrame({
        "channel": channels, "isc": observed, "p_surrogate": p,
        "null_mean": null.mean(axis=0) if length > 2 * min_shift else np.nan,
        "null_95": np.quantile(null, 0.95, axis=0) if length > 2 * min_shift else np.nan,
    })
    ok = per_channel["p_surrogate"].notna()
    per_channel["p_adjusted"] = np.nan
    if ok.any():
        per_channel.loc[ok, "p_adjusted"] = multipletests(per_channel.loc[ok, "p_surrogate"],
                                                          method="fdr_bh")[1]

    pair_r = np.einsum("ict,jct->ijc", z, z) / length
    ids = [r.recording_id for r in results]
    parts = [rec_by_id[i].participant_id or i for i in ids]
    pairs = pd.DataFrame([
        {"recording_a": ids[i], "recording_b": ids[j], "participant_a": parts[i],
         "participant_b": parts[j], "mean_r": float(pair_r[i, j].mean())}
        for i in range(n_subj) for j in range(i + 1, n_subj)
    ])

    win = int(cfg.sliding_window_seconds * sfreq)
    step = int(cfg.sliding_step_seconds * sfreq)
    sliding_rows = []
    for start in range(0, max(0, length - win + 1), max(step, 1)):
        zw = _zscore(data[:, :, start:start + win])
        sliding_rows.append({"time_s": (start + win / 2) / sfreq,
                             "isc_mean": float(mean_pairwise_r(zw).mean())})
    return SyncResult(
        song_id=song_id, channels=channels, recording_ids=ids, participant_ids=parts,
        duration_s=length / sfreq, per_channel=per_channel, pairs=pairs,
        sliding=pd.DataFrame(sliding_rows), warnings=warnings_,
    )
