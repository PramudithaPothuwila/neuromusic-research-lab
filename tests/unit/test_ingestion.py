from __future__ import annotations

import zipfile
from pathlib import Path

import numpy as np
import pytest

from neuromusic.domain.enums import ImportStatus, ValidationStatus
from neuromusic.ingestion.csv_reader import EEGReadError, read_eeg_csv
from neuromusic.ingestion.dataset_registry import DatasetRegistry, RegistryImportError
from neuromusic.ingestion.metadata_parser import FilenameParser
from neuromusic.ingestion.validator import detect_sampling_rate, validate_recording
from tests.conftest import make_frame


def _codes(report) -> set[str]:
    return {i.code for i in report.issues}


def test_valid_csv_passes(write_csv, dataset_config):
    raw = read_eeg_csv(write_csv())
    report = validate_recording(raw, dataset_config)
    assert report.status is ValidationStatus.PASSED
    assert report.detected_sampling_rate == pytest.approx(256.0, abs=1e-3)
    assert raw.channel_names[0] == "AF3" and raw.n_channels == 14


def test_rounded_timestamps_give_exact_rate():
    # Real files store 6-decimal timestamps; the median increment alone gives 256.016 Hz.
    times = np.round(np.arange(31412) / 256.0, 6)
    assert detect_sampling_rate(times) == pytest.approx(256.0, abs=1e-3)


def test_missing_channel_is_error(write_csv, dataset_config):
    frame = make_frame().drop(columns=["EEG.O1"])
    report = validate_recording(read_eeg_csv(write_csv(frame=frame)), dataset_config)
    assert report.status is ValidationStatus.FAILED
    assert "missing_channels" in _codes(report)


def test_duplicate_sample_indices(write_csv, dataset_config):
    frame = make_frame()
    frame.loc[100, "sample_index"] = 99
    report = validate_recording(read_eeg_csv(write_csv(frame=frame)), dataset_config)
    assert "duplicate_samples" in _codes(report)
    assert report.status is ValidationStatus.FAILED


def test_timestamp_gap_flagged(write_csv, dataset_config):
    frame = make_frame()
    frame.loc[2000:, "time_s"] += 0.5
    report = validate_recording(read_eeg_csv(write_csv(frame=frame)), dataset_config)
    gaps = [i for i in report.issues if i.code == "timestamp_gap"]
    assert gaps and gaps[0].start_s == pytest.approx(2000 / 256 - 1 / 256, abs=0.01)


def test_incorrect_sampling_frequency(write_csv, dataset_config):
    frame = make_frame(sfreq=128.0)
    report = validate_recording(read_eeg_csv(write_csv(frame=frame)), dataset_config)
    assert "sampling_rate_mismatch" in _codes(report)
    assert report.detected_sampling_rate == pytest.approx(128.0, abs=1e-3)


def test_non_numeric_values(write_csv, dataset_config):
    frame = make_frame().astype({"EEG.F3": object})
    frame.loc[10, "EEG.F3"] = "bad"
    raw = read_eeg_csv(write_csv(frame=frame))
    assert raw.non_numeric_counts == {"EEG.F3": 1}
    report = validate_recording(raw, dataset_config)
    assert {"non_numeric", "missing_values"} <= _codes(report)


def test_missing_required_column(write_csv):
    with pytest.raises(EEGReadError):
        read_eeg_csv(write_csv(frame=make_frame().drop(columns=["time_s"])))


def test_flatline_detected(write_csv, dataset_config):
    frame = make_frame()
    frame.loc[1000:1600, "EEG.T7"] = 4300.0
    report = validate_recording(read_eeg_csv(write_csv(frame=frame)), dataset_config)
    flat = [i for i in report.issues if i.code == "flatline"]
    assert flat and flat[0].channel == "T7"


def test_filename_parser_variants():
    parser = FilenameParser()
    p = parser.parse("SRC001_SOURCE_P001_Amude.csv")
    assert (p.source_id, p.participant_id, p.song_id) == ("SRC001", "P001", "amude")
    p = parser.parse("SRC056_SOURCE_P042_Wahi_Bindu_Watenwanam_from_SRC032.csv")
    assert p.song_id == "wahibinduwatenwanam"
    p = parser.parse("SRC010_SOURCE_P007_Parasathu_Yaya_T02.csv")
    assert (p.song_id, p.trial_id) == ("parasathuyaya", "T02")
    assert not parser.parse("recording.csv").matched


def _registry(tmp_path: Path, dataset_config) -> DatasetRegistry:
    return DatasetRegistry(tmp_path / "raw", tmp_path / "meta", dataset_config)


def test_missing_participant_metadata_not_analysable(tmp_path, write_csv, dataset_config):
    reg = _registry(tmp_path, dataset_config)
    res = reg.import_file(write_csv("unlabelled_recording.csv"))
    rec = reg.recordings[res.recording_id]
    assert rec.participant_id is None and rec.metadata_source == "unparsed"
    assert not rec.is_analysable
    reg.update_recording(rec.recording_id, participant_id="P009", song_id="Amude")
    rec = reg.recordings[res.recording_id]
    assert rec.is_analysable and rec.metadata_source == "manual" and rec.song_id == "amude"


def test_reimport_detected(tmp_path, write_csv, dataset_config):
    reg = _registry(tmp_path, dataset_config)
    first = reg.import_file(write_csv())
    again = reg.import_file(write_csv(directory=tmp_path / "other"))
    assert first.status is ImportStatus.IMPORTED
    assert again.status is ImportStatus.DUPLICATE and again.recording_id == first.recording_id
    assert len(reg.recordings) == 1


def test_unsupported_type_rejected(tmp_path, dataset_config):
    reg = _registry(tmp_path, dataset_config)
    bad = tmp_path / "notes.txt"
    bad.write_text("x")
    assert reg.import_file(bad).status is ImportStatus.REJECTED


def test_invalid_zip_rejected(tmp_path, dataset_config):
    reg = _registry(tmp_path, dataset_config)
    fake = tmp_path / "data.zip"
    fake.write_bytes(b"not a zip")
    with pytest.raises(RegistryImportError):
        reg.import_zip(fake)


def test_zip_path_traversal_rejected(tmp_path, dataset_config):
    reg = _registry(tmp_path, dataset_config)
    archive = tmp_path / "evil.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("../../escape.csv", "a,b\n1,2\n")
    with pytest.raises(RegistryImportError):
        reg.import_zip(archive)
    assert not (tmp_path / "escape.csv").exists()


def test_zip_import(tmp_path, write_csv, dataset_config):
    reg = _registry(tmp_path, dataset_config)
    src = write_csv("SRC002_SOURCE_P002_Dhole.csv", seed=3)
    archive = tmp_path / "batch.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.write(src, "nested/SRC002_SOURCE_P002_Dhole.csv")
        zf.writestr("readme.txt", "ignored")
    results = reg.import_zip(archive)
    statuses = sorted(r.status.value for r in results)
    assert statuses == ["imported", "rejected"]
    assert reg.analysable_recordings()[0].song_id == "dhole"
