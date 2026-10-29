"""Linear mixed-effects models: feature ~ song + (1 | participant).

Fitted by maximum likelihood so that the song effect can be tested with a likelihood-ratio test
against the intercept-only model. Convergence problems and singular fits are reported, not hidden.
"""

from __future__ import annotations

import math
import warnings

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats
from statsmodels.stats.multitest import multipletests


def fit_song_model(trials: pd.DataFrame, feature: str, reference_song: str | None = None
                   ) -> dict[str, object]:
    data = trials[["participant_id", "song_id", feature]].dropna().rename(columns={feature: "y"})
    songs = sorted(data["song_id"].unique())
    result: dict[str, object] = {"feature": feature, "n_observations": len(data),
                                 "n_participants": int(data["participant_id"].nunique())}
    if len(songs) < 2 or data["participant_id"].nunique() < 3:
        return {**result, "status": "insufficient_data"}
    ref = reference_song if reference_song in songs else songs[0]
    formula = f"y ~ C(song_id, Treatment(reference='{ref}'))"
    caught: list[str] = []
    with warnings.catch_warnings(record=True) as wlist:
        warnings.simplefilter("always")
        try:
            full = smf.mixedlm(formula, data, groups=data["participant_id"]).fit(
                reml=False, method=["lbfgs", "powell"])
            null = smf.mixedlm("y ~ 1", data, groups=data["participant_id"]).fit(
                reml=False, method=["lbfgs", "powell"])
        except (np.linalg.LinAlgError, ValueError) as exc:
            return {**result, "status": "failed", "message": str(exc)}
        caught = sorted({str(w.message).split("\n")[0] for w in wlist})
    lr = max(0.0, 2.0 * (full.llf - null.llf))
    df = len(songs) - 1
    group_var = float(np.asarray(full.cov_re)[0, 0]) if full.cov_re.size else math.nan
    out: dict[str, object] = {
        **result,
        "status": "ok",
        "reference_song": ref,
        "lr_statistic": lr,
        "df": df,
        "p_value": float(stats.chi2.sf(lr, df)),
        "converged": bool(full.converged and null.converged),
        "participant_variance": group_var,
        "residual_variance": float(full.scale),
        "singular_fit": bool(group_var < 1e-10),
        "warnings": "; ".join(caught),
    }
    ci = full.conf_int()
    for name in full.fe_params.index:
        if "song_id" not in name:
            continue
        song = name.split("[T.")[-1].rstrip("]")
        out[f"coef.{song}"] = float(full.fe_params[name])
        out[f"ci_low.{song}"] = float(ci.loc[name, 0])
        out[f"ci_high.{song}"] = float(ci.loc[name, 1])
    return out


def fit_song_models(trials: pd.DataFrame, features: list[str], correction: str = "fdr_bh"
                    ) -> pd.DataFrame:
    rows = [fit_song_model(trials, f) for f in features]
    frame = pd.DataFrame(rows)
    if "p_value" in frame.columns:
        mask = frame["p_value"].notna()
        frame["p_adjusted"] = np.nan
        if mask.any():
            frame.loc[mask, "p_adjusted"] = multipletests(frame.loc[mask, "p_value"],
                                                          method=correction)[1]
    return frame
