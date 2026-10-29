"""Song-to-song similarity and representational similarity analysis (RSA) against EEG.

Songs (or recordings) are described by feature vectors; each description yields a
dissimilarity matrix (Euclidean distance between z-scored features). Two matrices are compared
with the Spearman correlation of their off-diagonal entries; significance comes from a Mantel
permutation test that permutes the rows and columns of one matrix jointly.

At the recording level, pairs from the same participant can be masked out so that stable
individual differences cannot create a spurious stimulus-EEG relationship.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
from scipy import stats


def zscore(frame: pd.DataFrame) -> pd.DataFrame:
    """Column-wise z-scores; constant or all-missing columns are dropped."""
    sd = frame.std(ddof=0)
    keep = sd[(sd > 0) & sd.notna()].index
    return (frame[keep] - frame[keep].mean()) / sd[keep]


def distance_matrix(features: pd.DataFrame) -> pd.DataFrame:
    """Euclidean distance between rows of z-scored features, divided by sqrt(n_features)."""
    z = zscore(features.astype(float)).fillna(0.0)
    x = z.to_numpy()
    diff = x[:, None, :] - x[None, :, :]
    d = np.sqrt((diff**2).sum(axis=-1) / max(x.shape[1], 1))
    return pd.DataFrame(d, index=features.index, columns=features.index)


def upper_triangle(matrix: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
    iu = np.triu_indices(matrix.shape[0], k=1)
    values = matrix[iu]
    if mask is not None:
        values = values[mask[iu]]
    return values


def mantel(a: pd.DataFrame, b: pd.DataFrame, n_permutations: int, rng: np.random.Generator,
           mask: np.ndarray | None = None) -> dict[str, float]:
    """Spearman correlation between two dissimilarity matrices with a Mantel permutation test.

    ``mask`` (boolean, same shape) selects the pairs entering the correlation; it stays fixed
    while ``a`` is permuted, so excluded pairs remain excluded under the null.
    """
    if list(a.index) != list(b.index):
        raise ValueError("dissimilarity matrices must share the same item order")
    am, bm = a.to_numpy(), b.to_numpy()
    va, vb = upper_triangle(am, mask), upper_triangle(bm, mask)
    n_pairs = int(va.size)
    if n_pairs < 3 or np.ptp(va) == 0 or np.ptp(vb) == 0:
        return {"rho": math.nan, "p_value": math.nan, "n_items": len(a), "n_pairs": n_pairs}
    observed = float(stats.spearmanr(va, vb).statistic)
    rb = stats.rankdata(vb)
    null = np.empty(n_permutations)
    n = am.shape[0]
    for k in range(n_permutations):
        perm = rng.permutation(n)
        null[k] = np.corrcoef(stats.rankdata(upper_triangle(am[np.ix_(perm, perm)], mask)),
                              rb)[0, 1]
    p = float((1 + np.sum(np.abs(null) >= abs(observed) - 1e-12)) / (1 + n_permutations))
    return {"rho": observed, "p_value": p, "n_items": n, "n_pairs": n_pairs,
            "null_95": float(np.quantile(np.abs(null), 0.95))}


def pair_table(matrices: dict[str, pd.DataFrame], mask: np.ndarray | None = None
               ) -> pd.DataFrame:
    """Long table of item pairs with one column per dissimilarity matrix."""
    first = next(iter(matrices.values()))
    items = list(first.index)
    iu = np.triu_indices(len(items), k=1)
    keep = mask[iu] if mask is not None else np.ones(iu[0].size, dtype=bool)
    out = pd.DataFrame({"item_a": np.asarray(items)[iu[0]][keep],
                        "item_b": np.asarray(items)[iu[1]][keep]})
    for name, m in matrices.items():
        out[name] = m.to_numpy()[iu][keep]
    return out


def nearest_neighbours(dist: pd.DataFrame, k: int = 3) -> pd.DataFrame:
    rows = []
    for item in dist.index:
        others = dist.loc[item].drop(item).sort_values()
        rows.append({"item": item, **{f"nearest_{i + 1}": f"{o} ({d:.2f})"
                                      for i, (o, d) in enumerate(others.head(k).items())}})
    return pd.DataFrame(rows)


def attribute_consistency(audio: pd.DataFrame, attributes: pd.DataFrame,
                          centroid_col: str = "spectral.centroid_hz",
                          tempo_col: str = "rhythm.tempo_bpm") -> pd.DataFrame:
    """Check each attribute-table row against the audio file of the same song.

    Compares the table's brightness/tempo with the computed spectral centroid/tempo, and
    reports which other song's audio matches the row best. Rows whose values match another
    song's audio much better than their own indicate mislabelled or mismatched entries.
    """
    common = [s for s in attributes.index if s in audio.index]
    rows = []
    for song in common:
        b = float(attributes.loc[song, "brightness_hz"])
        t = float(attributes.loc[song, "tempo_bpm"])
        rel = (audio[centroid_col] - b).abs() / b
        tempo_eq = (audio[tempo_col] - t).abs() < 0.01 * t
        best = rel.idxmin()
        exact_others = [s for s in audio.index
                        if s != song and rel[s] < 0.01 and bool(tempo_eq[s])]
        own_ok = rel[song] < 0.05
        rows.append({
            "song_id": song,
            "table_brightness_hz": b,
            "audio_centroid_hz": float(audio.loc[song, centroid_col]),
            "centroid_diff_pct": float(100 * rel[song]),
            "table_tempo_bpm": t,
            "audio_tempo_bpm": float(audio.loc[song, tempo_col]),
            "tempo_equal": bool(tempo_eq[song]),
            "closest_audio_by_centroid": best,
            "other_audio_matching_row": ", ".join(exact_others),
            "status": "consistent" if own_ok and not exact_others
            else "matches other song" if exact_others else "inconsistent",
        })
    return pd.DataFrame(rows)
