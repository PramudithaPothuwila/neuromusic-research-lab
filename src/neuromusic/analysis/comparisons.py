"""Within-participant comparisons of trial-level features across songs.

The unit of analysis is the participant: multiple trials of the same participant-song are
averaged first, and only participants with every compared song are included (complete cases).

* Two songs: paired t-test and Wilcoxon signed-rank; Cohen's d_z, rank-biserial r, bootstrap
  CI of the mean paired difference, and a sign-flip permutation p-value.
* Three or more songs: Friedman test with Kendall's W, a within-participant permutation
  p-value, bootstrap CI for W, and Holm-corrected pairwise Wilcoxon post-hoc tests.

p-values are adjusted across features separately for the pre-specified primary family and the
exploratory remainder.
"""

from __future__ import annotations

import itertools
import math
import warnings
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests

from neuromusic.config import StatisticsConfig

EXACT_SIGNFLIP_MAX_N = 14


@dataclass
class ComparisonResults:
    summary: pd.DataFrame
    posthoc: pd.DataFrame
    songs: list[str]
    n_participants_complete: int
    notes: list[str]


def participant_song_matrix(trials: pd.DataFrame, feature: str,
                            songs: list[str] | None = None) -> pd.DataFrame:
    pivot = trials.pivot_table(index="participant_id", columns="song_id", values=feature,
                               aggfunc="mean")
    if songs is not None:
        pivot = pivot.reindex(columns=songs)
    return pivot.dropna(axis=0, how="any")


# ------------------------------------------------------------------- two conditions


def _signflip_pvalue(d: np.ndarray, n_perm: int, rng: np.random.Generator) -> float:
    n = d.size
    observed = abs(d.mean())
    if n <= EXACT_SIGNFLIP_MAX_N:
        signs = np.array(list(itertools.product([1.0, -1.0], repeat=n)))
        null = np.abs((signs * d).mean(axis=1))
        return float((null >= observed - 1e-12).mean())
    signs = rng.choice([1.0, -1.0], size=(n_perm, n))
    null = np.abs((signs * d).mean(axis=1))
    return float((1 + (null >= observed - 1e-12).sum()) / (1 + n_perm))


def _bootstrap_ci(values: np.ndarray, func, n_boot: int, rng: np.random.Generator,
                  alpha: float) -> tuple[float, float]:
    """Percentile bootstrap over participants. ``func`` maps (n_boot, n, ...) -> (n_boot,)."""
    if n_boot <= 0:
        return (math.nan, math.nan)
    n = values.shape[0]
    boots = np.asarray(func(values[rng.integers(0, n, size=(n_boot, n))]), dtype=float)
    boots = boots[np.isfinite(boots)]
    if boots.size == 0:
        return (math.nan, math.nan)
    return (float(np.quantile(boots, alpha / 2)), float(np.quantile(boots, 1 - alpha / 2)))


def compare_two(a: np.ndarray, b: np.ndarray, cfg: StatisticsConfig,
                rng: np.random.Generator) -> dict[str, float | str]:
    d = b - a
    n = d.size
    sd = d.std(ddof=1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        t_res = stats.ttest_rel(b, a)
        if np.allclose(d, 0):
            w_stat, w_p, rbc = math.nan, 1.0, 0.0
        else:
            w = stats.wilcoxon(d, zero_method="wilcox")
            w_stat, w_p = float(w.statistic), float(w.pvalue)
            nz = d[d != 0]
            ranks = stats.rankdata(np.abs(nz))
            rbc = float((ranks[nz > 0].sum() - ranks[nz < 0].sum()) / ranks.sum())
    ci = _bootstrap_ci(d, lambda b: b.mean(axis=-1), cfg.n_bootstrap, rng, cfg.alpha)
    return {
        "test": "wilcoxon_signed_rank",
        "statistic": w_stat,
        "p_value": w_p,
        "p_permutation": (_signflip_pvalue(d, cfg.n_permutations, rng)
                          if cfg.n_permutations > 0 else math.nan),
        "paired_t": float(t_res.statistic),
        "p_paired_t": float(t_res.pvalue),
        "effect_size": rbc,
        "effect_size_name": "rank_biserial_r",
        "cohens_dz": float(d.mean() / sd) if sd > 0 else math.nan,
        "estimate": float(d.mean()),
        "estimate_name": "mean_difference",
        "ci_low": ci[0],
        "ci_high": ci[1],
        "n_participants": n,
    }


# -------------------------------------------------------------- k >= 3 conditions


def _friedman_stat(x: np.ndarray) -> np.ndarray:
    """Friedman chi-square without tie correction. ``x`` shape (..., n, k)."""
    n, k = x.shape[-2], x.shape[-1]
    ranks = stats.rankdata(x, axis=-1)
    rank_sums = ranks.sum(axis=-2)
    return 12.0 / (n * k * (k + 1)) * (rank_sums**2).sum(axis=-1) - 3.0 * n * (k + 1)


def _kendalls_w(x: np.ndarray) -> np.ndarray:
    n, k = x.shape[-2], x.shape[-1]
    return _friedman_stat(x) / (n * (k - 1))


def compare_many(x: np.ndarray, songs: list[str], cfg: StatisticsConfig,
                 rng: np.random.Generator) -> tuple[dict[str, float | str], list[dict]]:
    n, k = x.shape
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fr = stats.friedmanchisquare(*[x[:, j] for j in range(k)])
    observed = _friedman_stat(x)
    perm_idx = np.argsort(rng.random((cfg.n_permutations, n, k)), axis=-1)
    permuted = np.take_along_axis(np.broadcast_to(x, perm_idx.shape), perm_idx, axis=-1)
    null = _friedman_stat(permuted)
    p_perm = float((1 + (null >= observed - 1e-12).sum()) / (1 + cfg.n_permutations))
    w = float(_kendalls_w(x))
    ci = _bootstrap_ci(x, _kendalls_w, min(cfg.n_bootstrap, 1000), rng, cfg.alpha)
    summary: dict[str, float | str] = {
        "test": "friedman",
        "statistic": float(fr.statistic),
        "p_value": float(fr.pvalue),
        "p_permutation": p_perm,
        "effect_size": w,
        "effect_size_name": "kendalls_w",
        "estimate": w,
        "estimate_name": "kendalls_w",
        "ci_low": ci[0],
        "ci_high": ci[1],
        "n_participants": n,
    }
    posthoc = []
    for (i, si), (j, sj) in itertools.combinations(enumerate(songs), 2):
        res = compare_two(x[:, i], x[:, j], cfg.model_copy(update={"n_permutations": 0,
                                                                     "n_bootstrap": 500}), rng)
        posthoc.append({"song_a": si, "song_b": sj, "statistic": res["statistic"],
                        "p_value": res["p_value"], "effect_size": res["effect_size"],
                        "cohens_dz": res["cohens_dz"], "mean_difference": res["estimate"],
                        "ci_low": res["ci_low"], "ci_high": res["ci_high"]})
    if posthoc:
        p_adj = multipletests([r["p_value"] for r in posthoc], method="holm")[1]
        for r, p in zip(posthoc, p_adj, strict=True):
            r["p_holm"] = float(p)
    return summary, posthoc


# ------------------------------------------------------------------------- driver


def adjust_pvalues(p: pd.Series, method: str) -> pd.Series:
    out = pd.Series(np.nan, index=p.index)
    mask = p.notna()
    if mask.any():
        out[mask] = multipletests(p[mask].to_numpy(), method=method)[1]
    return out


def compare_songs(trials: pd.DataFrame, features: list[str], cfg: StatisticsConfig,
                  songs: list[str] | None = None) -> ComparisonResults:
    """Run within-participant comparisons for each feature."""
    rng = np.random.default_rng(cfg.random_seed)
    notes: list[str] = []
    data = trials[trials["trial_valid"]] if "trial_valid" in trials.columns else trials
    songs = sorted(data["song_id"].dropna().unique()) if songs is None else songs
    if len(songs) < 2:
        return ComparisonResults(pd.DataFrame(), pd.DataFrame(), list(songs), 0,
                                 ["Fewer than two songs: cross-song comparison not possible."])
    if data.groupby(["participant_id", "song_id"]).size().gt(1).any():
        notes.append("Some participants have repeated trials per song; trials were averaged "
                     "within participant-song before testing.")
    complete = data.groupby("participant_id")["song_id"].apply(lambda s: set(songs) <= set(s))
    n_complete = int(complete.sum())
    if n_complete < len(complete):
        notes.append(f"{len(complete) - n_complete} participant(s) lack at least one song and "
                     "are excluded from within-participant tests (complete-case analysis).")
    if n_complete < cfg.min_participants_for_inference:
        notes.append(f"Only {n_complete} participant(s) with all songs; at least "
                     f"{cfg.min_participants_for_inference} required for inference.")
        return ComparisonResults(pd.DataFrame(), pd.DataFrame(), list(songs), n_complete, notes)

    rows: list[dict[str, Any]] = []
    post_rows: list[dict[str, Any]] = []
    for feat in features:
        matrix = participant_song_matrix(data, feat, songs)
        if len(matrix) < cfg.min_participants_for_inference:
            continue
        x = matrix.to_numpy(dtype=float)
        if len(songs) == 2:
            res = compare_two(x[:, 0], x[:, 1], cfg, rng)
            res["contrast"] = f"{songs[1]} - {songs[0]}"
        else:
            res, post = compare_many(x, songs, cfg, rng)
            post_rows.extend({"feature": feat, **r} for r in post)
        res["feature"] = feat
        res["primary"] = any(feat.startswith(p) for p in cfg.primary_feature_prefixes)
        for song, value in zip(songs, x.mean(axis=0), strict=True):
            res[f"mean.{song}"] = float(value)
        rows.append(res)

    summary = pd.DataFrame(rows)
    if not summary.empty:
        summary["p_adjusted"] = np.nan
        summary["p_permutation_adjusted"] = np.nan
        for flag in (True, False):
            sel = summary["primary"] == flag
            summary.loc[sel, "p_adjusted"] = adjust_pvalues(summary.loc[sel, "p_value"],
                                                            cfg.correction)
            summary.loc[sel, "p_permutation_adjusted"] = adjust_pvalues(
                summary.loc[sel, "p_permutation"], cfg.correction)
        summary["significant"] = summary["p_adjusted"] < cfg.alpha
        lead = ["feature", "primary", "test", "n_participants", "statistic", "p_value",
                "p_permutation", "p_adjusted", "p_permutation_adjusted", "significant",
                "effect_size", "effect_size_name", "estimate", "estimate_name", "ci_low",
                "ci_high"]
        summary = summary[[c for c in lead if c in summary.columns]
                          + [c for c in summary.columns if c not in lead]]
        summary = summary.sort_values(["primary", "p_value"], ascending=[False, True])
    notes.append(f"Multiple-comparison correction: {cfg.correction}, applied separately to "
                 "primary and exploratory feature families.")
    return ComparisonResults(summary.reset_index(drop=True), pd.DataFrame(post_rows),
                             list(songs), n_complete, notes)


# ------------------------------------------------------------- between-recording design


def _kruskal_h(values: np.ndarray, codes: np.ndarray, k: int) -> np.ndarray:
    """Kruskal-Wallis H (no tie correction) for each row of ``values`` (n_perm, n)."""
    n = values.shape[-1]
    ranks = stats.rankdata(values, axis=-1)
    sums = np.stack([ranks[..., codes == j].sum(axis=-1) for j in range(k)], axis=-1)
    counts = np.bincount(codes, minlength=k)
    return 12.0 / (n * (n + 1)) * (sums**2 / counts).sum(axis=-1) - 3.0 * (n + 1)


def compare_songs_between(trials: pd.DataFrame, features: list[str], cfg: StatisticsConfig,
                          min_per_song: int = 2) -> ComparisonResults:
    """Kruskal-Wallis across songs, one observation per participant-song (trials averaged).

    Used when too few participants heard multiple songs for within-participant tests. The
    permutation p-value shuffles song labels across observations. Participants contributing
    several songs violate strict independence; the mixed-effects models account for this.
    """
    rng = np.random.default_rng(cfg.random_seed)
    data = trials[trials["trial_valid"]] if "trial_valid" in trials.columns else trials
    obs = data.groupby(["participant_id", "song_id"], as_index=False)[features].mean()
    per_song = obs.groupby("song_id")["participant_id"].nunique()
    songs = sorted(per_song[per_song >= min_per_song].index)
    dropped = sorted(per_song[per_song < min_per_song].index)
    notes = ["Between-recording design: songs compared across participants (Kruskal-Wallis, "
             "epsilon-squared effect size). Participant and song effects are confounded."]
    if dropped:
        notes.append(f"{len(dropped)} song(s) with fewer than {min_per_song} participants "
                     f"excluded from tests: {', '.join(dropped)}.")
    obs = obs[obs["song_id"].isin(songs)]
    n_multi = int((obs.groupby("participant_id").size() > 1).sum())
    if n_multi:
        notes.append(f"{n_multi} participant(s) contribute more than one song; observations "
                     "are not fully independent.")
    if len(songs) < 2 or len(obs) < cfg.min_participants_for_inference:
        notes.append("Fewer than two songs with enough participants: no test performed.")
        return ComparisonResults(pd.DataFrame(), pd.DataFrame(), songs,
                                 int(obs["participant_id"].nunique()), notes)
    codes = pd.Categorical(obs["song_id"], categories=songs).codes.astype(int)
    k, n = len(songs), len(obs)
    perm_codes = np.stack([rng.permutation(codes) for _ in range(cfg.n_permutations)]) \
        if cfg.n_permutations > 0 else None
    rows = []
    for feat in features:
        y = obs[feat].to_numpy(dtype=float)
        if not np.isfinite(y).all() or np.ptp(y) == 0:
            continue
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = stats.kruskal(*[y[codes == j] for j in range(k)])
        p_perm = math.nan
        if perm_codes is not None:
            observed = _kruskal_h(y, codes, k)
            ranks = stats.rankdata(y)
            sums = np.stack([(ranks * (perm_codes == j)).sum(axis=1) for j in range(k)], axis=1)
            counts = np.bincount(codes, minlength=k)
            null = 12.0 / (n * (n + 1)) * (sums**2 / counts).sum(axis=1) - 3.0 * (n + 1)
            p_perm = float((1 + (null >= observed - 1e-9).sum()) / (1 + len(null)))
        eps2 = float((res.statistic - k + 1) / (n - k)) if n > k else math.nan
        row = {"feature": feat, "primary": any(feat.startswith(p)
                                               for p in cfg.primary_feature_prefixes),
               "test": "kruskal_wallis", "n_participants": int(obs["participant_id"].nunique()),
               "n_observations": n, "statistic": float(res.statistic),
               "p_value": float(res.pvalue), "p_permutation": p_perm, "effect_size": eps2,
               "effect_size_name": "epsilon_squared", "estimate": eps2,
               "estimate_name": "epsilon_squared", "ci_low": math.nan, "ci_high": math.nan}
        for song in songs:
            row[f"mean.{song}"] = float(y[codes == songs.index(song)].mean())
        rows.append(row)
    summary = pd.DataFrame(rows)
    if not summary.empty:
        for col in ("p_adjusted", "p_permutation_adjusted"):
            summary[col] = np.nan
        for flag in (True, False):
            sel = summary["primary"] == flag
            summary.loc[sel, "p_adjusted"] = adjust_pvalues(summary.loc[sel, "p_value"],
                                                            cfg.correction)
            summary.loc[sel, "p_permutation_adjusted"] = adjust_pvalues(
                summary.loc[sel, "p_permutation"], cfg.correction)
        summary["significant"] = summary["p_adjusted"] < cfg.alpha
        summary = summary.sort_values(["primary", "p_value"], ascending=[False, True])
    notes.append(f"Multiple-comparison correction: {cfg.correction}, applied separately to "
                 "primary and exploratory feature families.")
    return ComparisonResults(summary.reset_index(drop=True), pd.DataFrame(), songs,
                             int(obs["participant_id"].nunique()), notes)
