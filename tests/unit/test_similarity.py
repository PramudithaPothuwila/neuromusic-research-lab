from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from neuromusic.analysis.similarity import (
    attribute_consistency,
    distance_matrix,
    mantel,
    nearest_neighbours,
)
from neuromusic.audio.features import audio_features, estimate_tempo

SR = 22050


def test_mantel_detects_shared_structure():
    rng = np.random.default_rng(0)
    items = [f"s{i}" for i in range(15)]
    base = pd.DataFrame(rng.normal(size=(15, 4)), index=items)
    related = base + rng.normal(scale=0.2, size=base.shape)
    unrelated = pd.DataFrame(rng.normal(size=(15, 4)), index=items)
    a, b, c = (distance_matrix(f) for f in (base, related, unrelated))
    strong = mantel(a, b, 999, np.random.default_rng(1))
    assert strong["rho"] > 0.7 and strong["p_value"] < 0.01
    assert mantel(a, c, 999, np.random.default_rng(1))["p_value"] > 0.05


def test_mantel_mask_excludes_pairs():
    items = list("abcd")
    m = distance_matrix(pd.DataFrame({"x": [0.0, 1.0, 3.0, 6.0]}, index=items))
    mask = np.ones((4, 4), dtype=bool)
    mask[0, 1] = mask[1, 0] = False
    assert mantel(m, m, 10, np.random.default_rng(0), mask)["n_pairs"] == 5


def test_distance_matrix_symmetric_zero_diagonal():
    d = distance_matrix(pd.DataFrame({"x": [1.0, 2.0, 4.0], "y": [0.0, 1.0, 0.0]},
                                     index=["a", "b", "c"]))
    assert np.allclose(d.to_numpy(), d.to_numpy().T) and np.allclose(np.diag(d), 0)
    assert nearest_neighbours(d, k=1).loc[0, "nearest_1"].startswith("b")


def test_audio_centroid_of_sine():
    t = np.arange(SR * 5) / SR
    feats = audio_features(0.5 * np.sin(2 * np.pi * 1000 * t), SR)
    assert feats["spectral.centroid_hz"] == pytest.approx(1000, rel=0.02)
    assert feats["spectral.mid_energy_fraction"] > 0.95


@pytest.mark.parametrize("bpm", [80, 100, 120, 150])
def test_tempo_of_click_track(bpm):
    clicks = np.zeros(SR * 30)
    clicks[(np.arange(0, 30, 60 / bpm) * SR).astype(int)] = 1.0
    clicks = np.convolve(clicks, np.hanning(200), mode="same")
    clicks += 1e-4 * np.random.default_rng(0).normal(size=clicks.size)
    assert audio_features(clicks, SR)["rhythm.tempo_bpm"] == pytest.approx(bpm, rel=0.03)


def test_tempo_nan_for_silence():
    assert np.isnan(estimate_tempo(np.zeros(100), 43.0))


def test_attribute_consistency_flags_swapped_rows():
    audio = pd.DataFrame({"spectral.centroid_hz": [1000.0, 3000.0],
                          "rhythm.tempo_bpm": [100.0, 150.0]}, index=["a", "b"])
    table = pd.DataFrame({"brightness_hz": [3000.0, 1000.0], "tempo_bpm": [150.0, 100.0]},
                         index=["a", "b"])
    out = attribute_consistency(audio, table).set_index("song_id")
    assert (out["status"] == "matches other song").all()
    assert out.loc["a", "other_audio_matching_row"] == "b"
