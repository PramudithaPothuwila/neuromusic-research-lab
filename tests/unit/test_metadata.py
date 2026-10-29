from __future__ import annotations

import json

import pandas as pd
import pytest

from neuromusic.ingestion import stimuli
from neuromusic.ingestion.ratings import RatingsError, parse_ratings
from neuromusic.ingestion.song_attributes import load_song_attributes


def test_song_attributes_keyed_by_song_key(tmp_path):
    pd.DataFrame({"Title": ["Hadisida Manda", "Amude"], "Valence_Survay": [2.3, 3.2]}).to_csv(
        tmp_path / "song_attributes.csv", index=False)
    attrs = load_song_attributes(tmp_path)
    assert list(attrs.index) == ["hadisidamanda", "amude"]
    assert "valence_survey" in attrs.columns


def test_stimulus_aliases_resolve(tmp_path):
    (tmp_path / stimuli.SOURCES_FILE).write_text(json.dumps([
        {"title": "Ravana Teledrama theme song", "video_id": "abc",
         "aliases": ["Ravana Telegrama theme song"]}]), encoding="utf-8")
    catalogue = stimuli.load_catalogue(tmp_path)
    source = stimuli.resolve(catalogue, "ravanatelegramathemesong")
    assert source is not None and source.url == "https://www.youtube.com/watch?v=abc"
    assert not source.identity_verified


def test_fetch_requires_rights_confirmation(tmp_path):
    source = stimuli.StimulusSource("amude", "Amude", "x", (), "", "", False)
    with pytest.raises(stimuli.StimulusError, match="confirm_rights"):
        stimuli.fetch_youtube_audio([source], tmp_path, "licence", confirm_rights=False)


def test_import_audio_records_provenance(tmp_path):
    source = stimuli.StimulusSource("amude", "Amude", "x", (), "", "", False)
    clip = tmp_path / "supplied.wav"
    clip.write_bytes(b"RIFF0000WAVE")
    target = stimuli.import_audio(clip, source, tmp_path / "audio", "licensed for research")
    sidecar = json.loads(target.with_suffix(".json").read_text(encoding="utf-8"))
    assert target.name == "amude.wav" and sidecar["rights_statement"] == "licensed for research"
    with pytest.raises(stimuli.StimulusError):
        stimuli.import_audio(clip, source, tmp_path / "audio", "again")


def test_ratings_validation():
    good = pd.DataFrame({"participant_id": ["P1"], "song_id": ["amude"],
                         "rating_type": ["valence"], "rating_value": [4], "scale_min": [1],
                         "scale_max": [5]})
    ratings = parse_ratings(good)
    assert ratings[0].trial_id == "T01" and ratings[0].rating_type == "valence"
    with pytest.raises(RatingsError, match="outside"):
        parse_ratings(good.assign(rating_value=[9]))
    with pytest.raises(RatingsError, match="missing"):
        parse_ratings(good.drop(columns=["scale_max"]))


def test_scan_incoming_matches_titles(tmp_path):
    catalogue = [stimuli.StimulusSource("hadisidamanda", "Hadisida Manda", "x", (), "", "",
                                        False)]
    incoming = tmp_path / "incoming"
    incoming.mkdir()
    catalogue.append(stimuli.StimulusSource("udurawi", "Udurawi", "y", (), "", "", False))
    title = "Kanchana Anuradhi - UDURAWEE (\u0d8b) Official Lyric Video.opus"
    (incoming / title).write_bytes(b"OggS")
    (incoming / "Hadisida_Manda.mp3").write_bytes(b"ID3")
    (incoming / "Unknown Song.mp3").write_bytes(b"ID3")
    (incoming / "notes.txt").write_text("x")
    results = stimuli.scan_incoming(incoming, catalogue, tmp_path / "audio", "licensed")
    assert results["Hadisida_Manda.mp3"].startswith("-> hadisidamanda.mp3")
    assert results[title].startswith("-> udurawi.opus")
    assert results["Unknown Song.mp3"].startswith("unmatched")
    assert results["notes.txt"].startswith("skipped")
    assert not (incoming / "Hadisida_Manda.mp3").exists()  # moved, not copied
    sidecar = json.loads((tmp_path / "audio" / "udurawi.json").read_text(encoding="utf-8"))
    assert sidecar["original_filename"] == title and sidecar["title_match_score"] > 0.8


def test_ambiguous_titles_not_imported(tmp_path):
    catalogue = [stimuli.StimulusSource("sandanari", "Sandanari", "a", (), "", "", False),
                 stimuli.StimulusSource("sandanaro", "Sandanaro", "b", (), "", "", False)]
    source, _, reason = stimuli.match_source("Sandanar official", catalogue)
    assert source is None and reason.startswith("ambiguous")
