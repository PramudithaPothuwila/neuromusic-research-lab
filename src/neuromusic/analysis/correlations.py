"""Feature inter-correlations and EEG-rating associations."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests


def feature_correlation_matrix(trials: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    return trials[features].corr(method="spearman")


def song_attribute_associations(trials: pd.DataFrame, attributes: pd.DataFrame,
                                features: list[str], attribute_cols: list[str],
                                correction: str = "fdr_bh") -> pd.DataFrame:
    """Spearman correlation between song-mean EEG features and song attributes.

    Features are averaged within participant first, then across participants per song, so the
    sample size is the number of songs. Exploratory: with few songs these estimates are
    imprecise, and song attributes are confounded with one another.
    """
    if attributes.empty:
        return pd.DataFrame()
    per_participant = trials.groupby(["participant_id", "song_id"])[features].mean()
    song_means = per_participant.groupby("song_id").mean()
    joined = song_means.join(attributes[attribute_cols], how="inner")
    if len(joined) < 4:
        return pd.DataFrame()
    rows = []
    for attr in attribute_cols:
        for feat in features:
            res = stats.spearmanr(joined[feat], joined[attr], nan_policy="omit")
            rows.append({"attribute": attr, "feature": feat, "n_songs": len(joined),
                         "rho": float(res.statistic), "p_value": float(res.pvalue)})
    frame = pd.DataFrame(rows)
    ok = frame["p_value"].notna()
    frame["p_adjusted"] = np.nan
    if ok.any():
        frame.loc[ok, "p_adjusted"] = multipletests(frame.loc[ok, "p_value"], method=correction)[1]
    return frame.sort_values("p_value").reset_index(drop=True)


def rating_associations(trials: pd.DataFrame, ratings: pd.DataFrame, features: list[str],
                        correction: str = "fdr_bh") -> pd.DataFrame:
    """Spearman associations between features and each rating type.

    Reports a pooled correlation and a within-participant correlation (both variables centred
    on each participant's mean), which removes stable between-participant differences.
    """
    if ratings.empty:
        return pd.DataFrame()
    rows = []
    keys = ["participant_id", "song_id", "trial_id"]
    for rating_type, group in ratings.groupby("rating_type"):
        merged = trials.merge(group[[*keys, "rating_value"]], on=keys, how="inner")
        if merged["participant_id"].nunique() < 3 or len(merged) < 5:
            continue
        centred = merged.copy()
        cols = [*features, "rating_value"]
        centred[cols] = merged[cols] - merged.groupby("participant_id")[cols].transform("mean")
        for feat in features:
            pooled = stats.spearmanr(merged[feat], merged["rating_value"], nan_policy="omit")
            within = stats.spearmanr(centred[feat], centred["rating_value"], nan_policy="omit")
            rows.append({"rating_type": rating_type, "feature": feat, "n": len(merged),
                         "rho_pooled": float(pooled.statistic), "p_pooled": float(pooled.pvalue),
                         "rho_within": float(within.statistic),
                         "p_within": float(within.pvalue)})
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    for col in ("p_pooled", "p_within"):
        frame[f"{col}_adjusted"] = np.nan
        for _, idx in frame.groupby("rating_type").groups.items():
            p = frame.loc[idx, col]
            ok = p.notna()
            if ok.any():
                frame.loc[p[ok].index, f"{col}_adjusted"] = multipletests(
                    p[ok], method=correction)[1]
    return frame.sort_values("p_within").reset_index(drop=True)
