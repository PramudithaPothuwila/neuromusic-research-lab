from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from neuromusic.analysis.comparisons import compare_songs, compare_songs_between
from neuromusic.analysis.mixed_effects import fit_song_model
from neuromusic.config import StatisticsConfig

CFG = StatisticsConfig(n_permutations=500, n_bootstrap=500)


def trials(n_participants: int, effects: dict[str, float], noise: float = 0.3,
           seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for p in range(n_participants):
        offset = rng.normal(0, 1)
        for song, effect in effects.items():
            rows.append({"recording_id": f"R{p}{song}", "participant_id": f"P{p:02d}",
                         "song_id": song, "trial_id": "T01", "trial_valid": True,
                         "region.log_abs_power.alpha.occipital": offset + effect
                         + rng.normal(0, noise),
                         "temporal.rms.O1": rng.normal(0, 1)})
    return pd.DataFrame(rows)


FEATS = ["region.log_abs_power.alpha.occipital", "temporal.rms.O1"]


def test_paired_comparison_detects_effect():
    res = compare_songs(trials(12, {"a": 0.0, "b": 1.0}), FEATS, CFG)
    row = res.summary.set_index("feature").loc[FEATS[0]]
    assert row["test"] == "wilcoxon_signed_rank" and row["p_value"] < 0.01
    assert row["estimate"] == pytest.approx(1.0, abs=0.3)
    assert row["ci_low"] < row["estimate"] < row["ci_high"]
    assert bool(row["primary"])


def test_friedman_with_posthoc():
    res = compare_songs(trials(12, {"a": 0.0, "b": 1.0, "c": 2.0}), FEATS, CFG)
    row = res.summary.set_index("feature").loc[FEATS[0]]
    assert row["test"] == "friedman" and row["p_permutation"] < 0.01
    assert row["effect_size"] > 0.5  # Kendall's W
    assert len(res.posthoc[res.posthoc["feature"] == FEATS[0]]) == 3
    assert "p_holm" in res.posthoc.columns


def test_null_feature_not_significant():
    res = compare_songs(trials(12, {"a": 0.0, "b": 0.0, "c": 0.0}, seed=3), FEATS, CFG)
    assert not res.summary["significant"].any()


def test_insufficient_participants_reported():
    res = compare_songs(trials(2, {"a": 0.0, "b": 1.0}), FEATS, CFG)
    assert res.summary.empty and any("required" in n for n in res.notes)


def test_between_design_with_unbalanced_data():
    data = trials(30, {"a": 0.0, "b": 1.5, "c": 0.0})
    # Keep one song per participant (between-subjects structure).
    data = data.groupby("participant_id").sample(n=1, random_state=0)
    res = compare_songs_between(data, FEATS, CFG)
    row = res.summary.set_index("feature").loc[FEATS[0]]
    assert row["test"] == "kruskal_wallis" and row["p_value"] < 0.05


def test_mixed_model_song_effect():
    out = fit_song_model(trials(10, {"a": 0.0, "b": 1.0, "c": 0.5}), FEATS[0])
    assert out["status"] == "ok" and out["p_value"] < 0.01
    assert out["coef.b"] == pytest.approx(1.0, abs=0.3)
