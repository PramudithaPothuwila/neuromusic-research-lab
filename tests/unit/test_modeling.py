from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from neuromusic.modeling.classifiers import CLASSIFIERS, ModelSpec
from neuromusic.modeling.datasets import ModelingDataset, build_dataset
from neuromusic.modeling.metrics import classification_metrics
from neuromusic.modeling.validation import permute_within_participants, run_loso


def feature_table(n_participants: int = 8, songs=("a", "b", "c"), effect: float = 3.0,
                  seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for p in range(n_participants):
        for i, song in enumerate(songs):
            rows.append({"recording_id": f"R{p}{song}", "participant_id": f"P{p}",
                         "song_id": song, "trial_id": "T01", "trial_valid": True,
                         "spectral.rel_power.alpha.O1": i * effect + rng.normal(0, 1),
                         "spectral.rel_power.beta.O1": rng.normal(0, 1)})
    return pd.DataFrame(rows)


class FitRecorder(TransformerMixin, BaseEstimator):
    """Records the participants seen at fit time to prove fold isolation."""

    seen: list[set[str]] = []  # noqa: RUF012

    def __init__(self, groups_lookup=None):
        self.groups_lookup = groups_lookup

    def fit(self, x, y=None):
        FitRecorder.seen.append({self.groups_lookup[tuple(row)] for row in x})
        return self

    def transform(self, x):
        return x


def test_loso_never_trains_on_held_out_participant():
    ds = build_dataset(feature_table(), ["spectral."], level="trial")
    lookup = {tuple(row): g for row, g in zip(ds.x, ds.groups, strict=True)}
    FitRecorder.seen = []
    spec = ModelSpec("recorder", lambda s: Pipeline([
        ("rec", FitRecorder(lookup)), ("model", LogisticRegression())]), {}, "test")
    res = run_loso(ds, spec, tune=False)
    assert len(FitRecorder.seen) == ds.n_participants
    for fold_row, seen in zip(res.folds.itertuples(), FitRecorder.seen, strict=True):
        assert fold_row.held_out_participant not in seen
        assert len(seen) == ds.n_participants - 1


def test_loso_separable_data_beats_dummy():
    ds = build_dataset(feature_table(), ["spectral."], level="trial")
    learned = run_loso(ds, CLASSIFIERS["logistic_regression"], tune=True, inner_splits=3)
    dummy = run_loso(ds, CLASSIFIERS["dummy"], tune=False)
    m = classification_metrics(learned.predictions, ds.classes, n_boot=200)
    d = classification_metrics(dummy.predictions, ds.classes, n_boot=200)
    assert m["balanced_accuracy"] > 0.8 > d["balanced_accuracy"]
    assert len(m["confusion_matrix"]) == 3
    assert set(m["participant_accuracy"]) == set(ds.groups)


def test_permutation_within_participants_preserves_structure():
    rng = np.random.default_rng(0)
    groups = np.array(["P1"] * 6 + ["P2"] * 6)
    trial_ids = np.array(["t1", "t1", "t2", "t2", "t3", "t3"] * 2)
    y = np.array(["a", "a", "b", "b", "c", "c"] * 2)
    for _ in range(20):
        perm = permute_within_participants(y, groups, trial_ids, rng)
        for g in ("P1", "P2"):
            sel = groups == g
            assert sorted(perm[sel]) == sorted(y[sel])
            for t in ("t1", "t2", "t3"):
                assert len(set(perm[sel & (trial_ids == t)])) == 1


def test_build_dataset_requires_participants():
    data = feature_table(n_participants=2)
    try:
        build_dataset(data, ["spectral."], min_participants=3)
    except ValueError as exc:
        assert "participant" in str(exc)
    else:
        raise AssertionError("expected DatasetError")


def test_window_level_majority_vote():
    base = feature_table(n_participants=5)
    windows = base.loc[base.index.repeat(4)].reset_index(drop=True)
    windows["window_id"] = windows.groupby("recording_id").cumcount()
    windows["quality_status"] = "good"
    windows["segment"] = "stimulus"
    ds = build_dataset(windows, ["spectral."], level="window")
    assert isinstance(ds, ModelingDataset) and len(ds.y) == 60
    res = run_loso(ds, CLASSIFIERS["logistic_regression"], tune=False)
    assert res.trial_predictions is not None and len(res.trial_predictions) == 15
