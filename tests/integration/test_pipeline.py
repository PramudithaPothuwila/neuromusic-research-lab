"""End-to-end run on a small synthetic workspace."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from neuromusic import services
from neuromusic.cli import cmd_init
from neuromusic.experiments.persistence import ExperimentExistsError, ExperimentStore
from neuromusic.experiments.runner import ExperimentRunner
from neuromusic.reporting.generator import generate_report
from neuromusic.synthetic import SyntheticSpec, write_dataset
from neuromusic.workspace import Workspace


@pytest.fixture(scope="module")
def workspace(tmp_path_factory) -> Workspace:
    root = tmp_path_factory.mktemp("ws")
    import argparse

    cmd_init(argparse.Namespace(root=str(root)))
    write_dataset(root / "data/raw/eeg", SyntheticSpec(n_participants=5, duration_s=30.0))
    ws = Workspace(root)
    manifest = services.import_and_audit(ws, exclude_synthetic=False)
    assert manifest["analysable"].sum() == 15
    services.build_features(ws)
    return ws


def test_synthetic_files_excluded_by_default(tmp_path):
    write_dataset(tmp_path / "data/raw/eeg", SyntheticSpec(n_participants=1, duration_s=15.0))
    (tmp_path / "configs").mkdir()
    ws = Workspace(tmp_path)
    manifest = services.import_and_audit(ws)
    assert not manifest["analysable"].any()
    assert set(manifest["provenance"]) == {"synthetic"}


def test_features_built(workspace):
    trials = services.load_features(workspace)
    assert trials is not None and len(trials) == 15 and trials["trial_valid"].all()
    manifest = services.feature_store(workspace).manifest(workspace.preprocessing_hash,
                                                          workspace.feature_hash)
    assert manifest["is_synthetic_data"] is True
    assert services.features_current(workspace)


def test_experiments_and_report(workspace):
    runner = ExperimentRunner(workspace)
    stats = runner.run_statistics()
    assert stats.status.value == "completed"
    assert stats.metrics["test"] == "friedman"
    assert stats.metrics["n_significant_primary"] > 0  # synthetic alpha/beta effects

    cfg = workspace.experiments.classification.model_copy(update={
        "models": ["logistic_regression"], "tune_hyperparameters": False,
        "permutation_importance_repeats": 2})
    clf = runner.run_classification(cfg)
    assert clf.metrics["logistic_regression"]["balanced_accuracy"] > \
        clf.metrics["dummy"]["balanced_accuracy"]
    assert clf.is_synthetic_data and clf.cv_strategy.startswith("leave-one-subject-out")

    clus = runner.run_clustering()
    assert clus.metrics["explained_variance_ratio"]

    store = ExperimentStore(workspace.experiments_dir)
    with pytest.raises(ExperimentExistsError):
        store.create(clf)

    out = generate_report(workspace)
    md = (out / "report.md").read_text(encoding="utf-8")
    assert "SYNTHETIC DATA WARNING" in md and "Limitations" in md
    assert (out / "report.html").exists()
    assert (out / "figures" / "confusion_matrix.png").exists()
    payload = json.loads((out / "results.json").read_text(encoding="utf-8"))
    assert payload["synthetic"] is True and "classification" in payload


def test_synchronization_unavailable_without_alignment(workspace):
    from neuromusic.experiments.runner import ExperimentError

    with pytest.raises(ExperimentError, match="unavailable"):
        ExperimentRunner(workspace).run_synchronization("songa")


def test_synchronization_with_verified_alignment(workspace):
    reg = workspace.registry
    song_recs = [r for r in reg.analysable_recordings() if r.song_id == "songa"]
    for rec in song_recs:
        reg.update_recording(rec.recording_id, playback_onset_s=0.0, alignment_verified=True,
                             save=False)
    reg.save()
    cfg = workspace.experiments.synchronization.model_copy(update={"n_surrogates": 20})
    rec = ExperimentRunner(workspace).run_synchronization("songa", cfg)
    assert rec.status.value == "completed" and rec.metrics["duration_s"] > 20
    table = ExperimentStore(workspace.experiments_dir).load_table(rec.experiment_id,
                                                                  "isc_per_channel")
    assert len(table) == 14 and Path(rec.results_path).exists()
