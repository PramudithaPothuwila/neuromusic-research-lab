"""Detection of recordings derived from other recordings.

Identical-content files are caught by SHA-256. This module catches *near*-duplicates: files
whose signal is a copy of another file with small perturbations, offsets or relabelling. Each
recording is summarised by the z-scored first differences of the first ``n_samples`` samples of
a few channels (differencing removes DC offsets); pairs whose signatures correlate above a
threshold are linked, and connected components form "signal families".

Within a family only one member can be an independent measurement. Members carrying different
song labels make every label in the family untrustworthy.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.sparse.csgraph import connected_components

from neuromusic.ingestion.csv_reader import RawEEG

SIGNATURE_SAMPLES = 4096
SIGNATURE_CHANNELS = ("AF3", "O1", "T8")
DEFAULT_THRESHOLD = 0.9


def signal_signature(raw: RawEEG, n_samples: int = SIGNATURE_SAMPLES,
                     channels: tuple[str, ...] = SIGNATURE_CHANNELS) -> np.ndarray | None:
    """Normalised signature vector, or None if the recording is too short or non-finite."""
    idx = [raw.channel_names.index(c) for c in channels if c in raw.channel_names]
    if not idx or raw.n_samples <= n_samples:
        return None
    seg = np.diff(raw.data[idx, : n_samples + 1], axis=1).ravel()
    if not np.isfinite(seg).all() or seg.std() == 0:
        return None
    return ((seg - seg.mean()) / seg.std()).astype(np.float32)


@dataclass
class SignalFamily:
    members: list[str]  # recording ids, primary first
    max_r: float
    song_ids: list[str | None]
    label_conflict: bool
    pair_r: dict[tuple[str, str], float] = field(default_factory=dict)


def find_families(signatures: dict[str, np.ndarray], order: list[str],
                  threshold: float = DEFAULT_THRESHOLD,
                  songs: dict[str, str | None] | None = None) -> list[SignalFamily]:
    """Group recordings whose signatures correlate >= ``threshold``.

    ``order`` ranks recordings (e.g. by source id); the earliest member of a family is reported
    first and treated as the presumed original. Only families with >= 2 members are returned.
    """
    ids = [i for i in order if i in signatures]
    if len(ids) < 2:
        return []
    length = min(signatures[i].size for i in ids)
    mat = np.stack([signatures[i][:length] for i in ids])
    corr = mat @ mat.T / length
    adjacency = corr >= threshold
    np.fill_diagonal(adjacency, False)
    n_comp, labels = connected_components(adjacency, directed=False)
    songs = songs or {}
    families = []
    for comp in range(n_comp):
        members_idx = np.flatnonzero(labels == comp)
        if members_idx.size < 2:
            continue
        members = [ids[k] for k in members_idx]  # already in `order`
        sub = corr[np.ix_(members_idx, members_idx)]
        pairs = {(ids[a], ids[b]): float(corr[a, b])
                 for x, a in enumerate(members_idx) for b in members_idx[x + 1:]}
        member_songs = [songs.get(m) for m in members]
        families.append(SignalFamily(
            members=members,
            max_r=float(sub[~np.eye(len(members_idx), dtype=bool)].max()),
            song_ids=member_songs,
            label_conflict=len({s for s in member_songs if s is not None}) > 1,
            pair_r=pairs,
        ))
    return families
