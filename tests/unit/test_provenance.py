from __future__ import annotations

import numpy as np

from neuromusic.ingestion.dataset_registry import DatasetRegistry
from tests.conftest import make_frame


def test_near_duplicates_and_label_conflicts(tmp_path, write_csv, dataset_config):
    raw = tmp_path / "raw"
    base = make_frame(seed=1)
    write_csv("SRC001_SOURCE_P001_Amude.csv", frame=base, directory=raw)
    # Derived copy: same signal plus small noise, relabelled as another participant.
    copy = base.copy()
    eeg = [c for c in copy.columns if c.startswith("EEG.")]
    copy[eeg] += np.random.default_rng(9).normal(0, 0.05, copy[eeg].shape)
    write_csv("SRC002_SOURCE_P002_Amude.csv", frame=copy, directory=raw)
    # Independent recording.
    write_csv("SRC003_SOURCE_P003_Dhole.csv", frame=make_frame(seed=2), directory=raw)
    # Byte-identical copy of SRC003 carrying a different song label.
    (raw / "SRC004_SOURCE_P004_Pandama.csv").write_bytes(
        (raw / "SRC003_SOURCE_P003_Dhole.csv").read_bytes())

    reg = DatasetRegistry(raw, tmp_path / "meta", dataset_config)
    reg.scan_directory()
    families = reg.audit_provenance()
    by_src = {r.source_id: r for r in reg.recordings.values()}

    assert len(families) == 1 and not families[0].label_conflict
    assert not by_src["SRC001"].excluded
    assert by_src["SRC002"].excluded and by_src["SRC002"].duplicate_of == \
        by_src["SRC001"].recording_id
    assert by_src["SRC003"].excluded and "conflicting" in by_src["SRC003"].exclusion_reason

    manifest = reg.manifest_frame()
    prov = dict(zip(manifest["source_filename"], manifest["provenance"], strict=True))
    assert prov["SRC001_SOURCE_P001_Amude.csv"] == "original"
    assert prov["SRC002_SOURCE_P002_Amude.csv"] == "derived_copy"
    assert prov["SRC004_SOURCE_P004_Pandama.csv"] == "label_conflict"
    assert [r.source_id for r in reg.analysable_recordings()] == ["SRC001"]


def test_manual_inclusion_survives_reaudit(tmp_path, write_csv, dataset_config):
    raw = tmp_path / "raw"
    base = make_frame(seed=1)
    write_csv("SRC001_SOURCE_P001_Amude.csv", frame=base, directory=raw)
    shifted = base.copy()
    shifted[[c for c in base.columns if c.startswith("EEG.")]] += 100.0
    write_csv("SRC002_SOURCE_P002_Amude.csv", frame=shifted, directory=raw)
    reg = DatasetRegistry(raw, tmp_path / "meta", dataset_config)
    reg.scan_directory()
    reg.audit_provenance()
    dup = next(r for r in reg.recordings.values() if r.source_id == "SRC002")
    assert dup.excluded
    reg.update_recording(dup.recording_id, excluded=False)
    reg.audit_provenance()
    assert not reg.recordings[dup.recording_id].excluded
