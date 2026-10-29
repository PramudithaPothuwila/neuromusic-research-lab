from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from neuromusic.config import (
    ArtifactConfig,
    BandpassConfig,
    FeatureConfig,
    PSDConfig,
    WindowConfig,
)
from neuromusic.domain.models import Recording
from neuromusic.features.complexity import hjorth
from neuromusic.features.extractor import compute_window_features
from neuromusic.features.spectral import band_power, check_bands, welch_psd
from neuromusic.preprocessing.artifacts import detect_artifacts, flags_to_frame
from neuromusic.preprocessing.filtering import bandpass
from neuromusic.preprocessing.segmentation import assign_window_quality, window_bounds

FS = 256.0


def sine(freq: float, seconds: float = 8.0, amp: float = 1.0) -> np.ndarray:
    t = np.arange(int(seconds * FS)) / FS
    return amp * np.sin(2 * np.pi * freq * t)


@pytest.mark.parametrize(("freq", "band"), [(6.0, "theta"), (10.0, "alpha"), (20.0, "beta")])
def test_band_power_detects_known_sinusoid(freq, band):
    cfg = FeatureConfig()
    # Mean averaging: the default median average is bias-corrected for noise, not pure tones.
    freqs, psd = welch_psd(sine(freq)[None, :], FS, PSDConfig(average="mean"))
    powers = {b: band_power(freqs, psd, edges)[0] for b, edges in cfg.frequency_bands.items()}
    assert max(powers, key=powers.get) == band
    # Power of a unit sine is 0.5; nearly all of it falls in the containing band.
    assert powers[band] == pytest.approx(0.5, rel=0.1)


def test_band_edges_are_not_double_counted():
    freqs = np.arange(0, 40.5, 0.5)
    psd = np.ones_like(freqs)
    lo = band_power(freqs, psd, (4.0, 8.0))
    hi = band_power(freqs, psd, (8.0, 13.0))
    both = band_power(freqs, psd, (4.0, 13.0))
    assert lo + hi == pytest.approx(both)


def test_overlapping_bands_rejected():
    with pytest.raises(ValueError, match="overlap"):
        check_bands({"a": (4.0, 9.0), "b": (8.0, 13.0)}, FS, (1.0, 40.0))


def test_bandpass_attenuates_out_of_band():
    x = (sine(10.0) + sine(60.0) + sine(0.2))[None, :]
    y = bandpass(x, FS, BandpassConfig(low_hz=1.0, high_hz=40.0))
    freqs, p_in = welch_psd(x, FS, PSDConfig(segment_seconds=4.0))
    _, p_out = welch_psd(y, FS, PSDConfig(segment_seconds=4.0))

    def at(f: float, p: np.ndarray) -> float:
        return float(p[0, np.argmin(np.abs(freqs - f))])

    assert at(10.0, p_out) == pytest.approx(at(10.0, p_in), rel=0.1)
    assert at(60.0, p_out) < 1e-3 * at(60.0, p_in)


def test_window_lengths_and_overlap():
    bounds = window_bounds(int(20 * FS), FS, WindowConfig(duration_seconds=4.0,
                                                          overlap_seconds=2.0))
    assert len(bounds) == 9
    assert set(bounds[:, 1] - bounds[:, 0]) == {int(4 * FS)}
    assert np.all(np.diff(bounds[:, 0]) == int(2 * FS))
    no_overlap = window_bounds(int(20 * FS), FS, WindowConfig(duration_seconds=4.0,
                                                              overlap_seconds=0.0))
    assert len(no_overlap) == 5


def test_invalid_window_config():
    with pytest.raises(ValueError):
        WindowConfig(duration_seconds=2.0, overlap_seconds=2.0)


def _noisy(n_ch: int = 6, seconds: float = 30.0, seed: int = 1) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.normal(0, 1, (n_ch, int(seconds * FS)))


def test_artifact_flags_identify_synthetic_anomalies():
    data = _noisy()
    channels = [f"C{i}" for i in range(data.shape[0])]
    data[1, int(5 * FS):int(8 * FS)] = 0.0  # flatline
    data[2, int(15 * FS):] += 80.0  # step
    data[:, int(20 * FS):int(20.5 * FS)] += 40.0 * np.hanning(int(0.5 * FS))  # movement
    flags = flags_to_frame(detect_artifacts(data, data, FS, channels, ArtifactConfig()))
    flat = flags[flags["kind"] == "flatline"]
    assert (flat["channel"] == "C1").all() and not flat.empty
    assert flat["start_s"].min() == pytest.approx(5.0, abs=1.0)
    step = flags[(flags["kind"] == "step") & (flags["channel"] == "C2")]
    assert not step.empty and step["start_s"].min() == pytest.approx(15.0, abs=1.0)
    move = flags[flags["kind"] == "possible_movement"]
    assert not move.empty and move["start_s"].min() == pytest.approx(20.0, abs=1.0)
    assert flags["reason"].str.len().gt(0).all()


def test_clean_noise_has_no_segment_flags():
    data = _noisy(seed=5)
    flags = flags_to_frame(detect_artifacts(data, data, FS, [f"C{i}" for i in range(6)],
                                            ArtifactConfig()))
    assert flags[flags["scope"] == "segment"].empty


def test_window_quality_from_flags():
    bounds = window_bounds(int(20 * FS), FS, WindowConfig())
    flags = pd.DataFrame([{"kind": "step", "reason": "x", "channel": c, "scope": "segment",
                           "start_s": 5.0, "end_s": 6.0, "value": 1.0, "threshold": 1.0}
                          for c in ("A", "B", "C")])
    statuses, counts = assign_window_quality(bounds, FS, flags, 3, automatic_rejection=True)
    assert counts[1] == 3 and statuses[1].value == "rejected"
    assert statuses[-1].value == "good"
    statuses, _ = assign_window_quality(bounds, FS, flags, 3, automatic_rejection=False)
    assert statuses[1].value == "flagged"


def test_features_finite_and_named():
    rng = np.random.default_rng(0)
    epochs = rng.normal(0, 1, (5, 14, int(4 * FS)))
    channels = ["AF3", "F7", "F3", "FC5", "T7", "P7", "O1", "O2", "P8", "T8", "FC6", "F4",
                "F8", "AF4"]
    feats = compute_window_features(epochs, FS, channels, FeatureConfig())
    assert len(feats) == 5
    assert np.isfinite(feats.to_numpy()).all()
    for col in ("spectral.rel_power.alpha.O1", "hjorth.mobility.F3",
                "region.log_abs_power.alpha.occipital", "asymmetry.alpha.F4-F3",
                "spectral.ratio.alpha_beta.AF3", "spectral.entropy.T7"):
        assert col in feats.columns


def test_hjorth_of_sine():
    x = sine(10.0)
    params = hjorth(x)
    # Mobility of a sinusoid ~ 2*pi*f/fs (per sample); complexity ~ 1.
    assert params["mobility"] == pytest.approx(2 * np.pi * 10 / FS, rel=0.02)
    assert params["complexity"] == pytest.approx(1.0, abs=0.05)


def test_recording_model_analysable_flag():
    rec = Recording(recording_id="R", participant_id="P1", song_id="s", source_filename="f",
                    raw_data_path="f", content_sha256="0", validation_status="passed")
    assert rec.is_analysable
    rec.excluded = True
    assert not rec.is_analysable
