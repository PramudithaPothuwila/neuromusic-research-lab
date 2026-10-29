"""Typed configuration models loaded from ``configs/*.yaml``.

Every processing step is parameterised by one of these models. ``config_hash`` produces a
stable identifier used to version derived artifacts.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from neuromusic.domain.enums import ReferenceMethod, SignalUnits
from neuromusic.domain.schemas import DEFAULT_CHANNELS


class _Config(BaseModel):
    model_config = ConfigDict(extra="forbid")


# --------------------------------------------------------------------------- dataset


class FilenamePatternConfig(_Config):
    """Regex with named groups ``participant_id`` and ``song_id`` (``source_id``, ``trial_id``
    optional). The default interpretation of ``SRC001_SOURCE_P001_Amude.csv`` is provisional."""

    regex: str = (
        r"^(?P<source_id>[A-Za-z0-9]+)_(?P<source_label>[A-Za-z0-9]+)_(?P<participant_id>P\d+)"
        r"_(?P<song_id>.+?)(?:_from_(?P<derived_from>SRC\d+))?(?:_(?P<trial_id>T\d+))?$"
    )
    song_aliases: dict[str, str] = Field(default_factory=dict)


class AcquisitionConfig(_Config):
    expected_sampling_rate: float | None = 256.0
    sampling_rate_tolerance_hz: float = 0.5
    timestamp_jitter_tolerance: float = 0.1  # fraction of one sample interval
    unit_scale_to_microvolts: float | None = None
    recording_device: str | None = None
    acquisition_reference: str | None = None
    montage: str | None = "standard_1020"
    montage_positions_verified: bool = False


class ValidationConfig(_Config):
    min_duration_seconds: float = 10.0
    flatline_min_seconds: float = 1.0
    flatline_epsilon: float = 1e-9
    extreme_robust_z: float = 20.0
    extreme_fraction_warning: float = 0.01


class DatasetConfig(_Config):
    raw_dir: str = "data/raw/eeg"
    metadata_dir: str = "data/metadata"
    channels: list[str] = Field(default_factory=lambda: list(DEFAULT_CHANNELS))
    channel_prefix: str = "EEG."
    allow_extra_channels: bool = True
    filename: FilenamePatternConfig = Field(default_factory=FilenamePatternConfig)
    acquisition: AcquisitionConfig = Field(default_factory=AcquisitionConfig)
    validation: ValidationConfig = Field(default_factory=ValidationConfig)
    max_zip_members: int = 2000
    max_file_size_mb: float = 500.0


# --------------------------------------------------------------------- preprocessing


class BandpassConfig(_Config):
    enabled: bool = True
    low_hz: float = 1.0
    high_hz: float = 40.0

    @model_validator(mode="after")
    def _check(self) -> BandpassConfig:
        if self.low_hz <= 0 or self.high_hz <= self.low_hz:
            raise ValueError("bandpass requires 0 < low_hz < high_hz")
        return self


class NotchConfig(_Config):
    enabled: bool = False
    frequency_hz: float = 50.0
    harmonics: bool = True


class ReferenceConfig(_Config):
    method: ReferenceMethod = ReferenceMethod.PRESERVE_ORIGINAL
    channels: list[str] = Field(default_factory=list)


class WindowConfig(_Config):
    duration_seconds: float = 4.0
    overlap_seconds: float = 2.0

    @model_validator(mode="after")
    def _check(self) -> WindowConfig:
        if self.duration_seconds <= 0:
            raise ValueError("window duration must be positive")
        if not 0 <= self.overlap_seconds < self.duration_seconds:
            raise ValueError("overlap must satisfy 0 <= overlap < duration")
        return self


class ArtifactConfig(_Config):
    """Data-driven artifact thresholds. All amplitude criteria are relative (robust z-scores),
    because signal units are not confirmed."""

    detection_enabled: bool = True
    automatic_rejection_enabled: bool = False
    detection_window_seconds: float = 1.0
    flatline_relative_std: float = 0.01
    step_robust_z: float = 15.0
    amplitude_robust_z: float = 8.0
    hf_band_hz: tuple[float, float] = (45.0, 100.0)
    hf_ratio_robust_z: float = 5.0
    drift_cutoff_hz: float = 0.5
    drift_power_fraction: float = 0.9
    channel_variance_robust_z: float = 5.0
    line_noise_hz: float = 50.0
    line_noise_ratio_warning: float = 0.3
    reject_window_if_flagged_channels_ge: int = 3


class ICAConfig(_Config):
    enabled: bool = False
    n_components: int | None = None
    method: Literal["fastica", "infomax", "picard"] = "fastica"
    exclude_components: list[int] = Field(default_factory=list)
    random_state: int = 42


class BaselineConfig(_Config):
    enabled: bool = False
    start_s: float = 0.0
    end_s: float = 0.0


class PreprocessingConfig(_Config):
    sampling_rate: float | Literal["auto"] = "auto"
    # Physical units are shown only when set to a validated value; "unknown" -> arbitrary units.
    signal_units: SignalUnits = SignalUnits.UNKNOWN
    remove_dc_offset: bool = True
    bandpass: BandpassConfig = Field(default_factory=BandpassConfig)
    notch: NotchConfig = Field(default_factory=NotchConfig)
    reference: ReferenceConfig = Field(default_factory=ReferenceConfig)
    windows: WindowConfig = Field(default_factory=WindowConfig)
    artifact_handling: ArtifactConfig = Field(default_factory=ArtifactConfig)
    ica: ICAConfig = Field(default_factory=ICAConfig)
    baseline: BaselineConfig = Field(default_factory=BaselineConfig)
    trim_start_seconds: float = 0.0
    trim_end_seconds: float = 0.0
    save_cleaned_data: bool = True


# -------------------------------------------------------------------------- features


class PSDConfig(_Config):
    method: Literal["welch"] = "welch"
    segment_seconds: float = 2.0
    overlap_fraction: float = 0.5
    window: str = "hann"
    average: Literal["mean", "median"] = "median"
    fmin: float = 1.0
    fmax: float = 40.0


def _default_variability() -> list[Literal["std", "iqr"]]:
    return ["iqr"]


class FeatureConfig(_Config):
    frequency_bands: dict[str, tuple[float, float]] = Field(
        default_factory=lambda: {"theta": (4.0, 8.0), "alpha": (8.0, 13.0), "beta": (13.0, 30.0)}
    )
    total_power_range: tuple[float, float] = (1.0, 40.0)
    psd: PSDConfig = Field(default_factory=PSDConfig)
    ratios: list[tuple[str, str]] = Field(
        default_factory=lambda: [("alpha", "beta"), ("theta", "beta")]
    )
    regions: dict[str, list[str]] = Field(
        default_factory=lambda: {
            "frontal_left": ["AF3", "F7", "F3", "FC5"],
            "frontal_right": ["AF4", "F8", "F4", "FC6"],
            "temporal_left": ["T7"],
            "temporal_right": ["T8"],
            "parietal": ["P7", "P8"],
            "occipital": ["O1", "O2"],
        }
    )
    asymmetry_pairs: list[tuple[str, str]] = Field(
        default_factory=lambda: [("F4", "F3"), ("AF4", "AF3"), ("F8", "F7"), ("FC6", "FC5")]
    )
    asymmetry_bands: list[str] = Field(default_factory=lambda: ["alpha"])
    regional_features_enabled: bool = True
    include_temporal: bool = True
    include_hjorth: bool = True
    include_spectral_entropy: bool = True
    trial_aggregation: Literal["median", "mean"] = "median"
    trial_variability: list[Literal["std", "iqr"]] = Field(default_factory=_default_variability)
    variability_families: list[str] = Field(default_factory=lambda: ["region"])
    min_valid_windows_per_trial: int = 5

    @field_validator("frequency_bands")
    @classmethod
    def _bands_valid(cls, bands: dict[str, tuple[float, float]]) -> dict[str, tuple[float, float]]:
        for name, (lo, hi) in bands.items():
            if not 0 <= lo < hi:
                raise ValueError(f"band {name!r} must satisfy 0 <= low < high")
        return bands


# ------------------------------------------------------------------------ statistics


class StatisticsConfig(_Config):
    alpha: float = 0.05
    correction: Literal["fdr_bh", "holm", "bonferroni"] = "fdr_bh"
    primary_feature_prefixes: list[str] = Field(
        default_factory=lambda: ["region.log_abs_power.", "region.rel_power.", "asymmetry."]
    )
    n_bootstrap: int = 2000
    n_permutations: int = 2000
    # "within": repeated-measures tests; "between": Kruskal-Wallis across participants;
    # "auto": within when enough participants heard every song, otherwise between.
    design: Literal["auto", "within", "between"] = "auto"
    min_participants_per_song: int = 2
    mixed_effects_enabled: bool = True
    mixed_effects_max_features: int = 60
    min_participants_for_inference: int = 3
    random_seed: int = 42


# ------------------------------------------------------------------------ experiments


class ClassificationConfig(_Config):
    models: list[str] = Field(
        default_factory=lambda: ["dummy", "logistic_regression", "svm", "random_forest"]
    )
    feature_prefixes: list[str] = Field(
        default_factory=lambda: ["spectral.log_abs_power.", "spectral.rel_power."]
    )
    observation_level: Literal["trial", "window"] = "trial"
    target: str = "song_id"
    tune_hyperparameters: bool = True
    inner_cv_splits: int = 3
    tuning_scoring: str = "balanced_accuracy"
    n_permutations: int = 0
    permutation_importance_repeats: int = 10
    min_participants: int = 3
    random_seed: int = 42
    n_jobs: int = 1


class RegressionConfig(_Config):
    models: list[str] = Field(default_factory=lambda: ["ridge", "svr", "random_forest"])
    feature_prefixes: list[str] = Field(default_factory=lambda: ["region.log_abs_power."])
    rating_type: str | None = None
    tune_hyperparameters: bool = True
    inner_cv_splits: int = 3
    random_seed: int = 42


class ClusteringConfig(_Config):
    feature_prefixes: list[str] = Field(
        default_factory=lambda: ["region.log_abs_power.", "region.rel_power."]
    )
    n_components: int = 5
    k_values: list[int] = Field(default_factory=lambda: [2, 3, 4, 5])
    n_stability_runs: int = 50
    hierarchical_enabled: bool = True
    random_seed: int = 42


class SynchronizationConfig(_Config):
    channels: list[str] | None = None
    sliding_window_seconds: float = 10.0
    sliding_step_seconds: float = 5.0
    n_surrogates: int = 200
    min_shift_seconds: float = 5.0
    random_seed: int = 42


class SongSimilarityConfig(_Config):
    """Song-to-song comparison and representational similarity analysis against EEG."""

    eeg_feature_prefixes: list[str] = Field(
        default_factory=lambda: ["region.log_abs_power.", "region.rel_power.", "asymmetry."])
    # Song-level comparisons use the same opening segment for every song (recordings start
    # at song onset and last ~120 s); None compares entire songs.
    song_segment_seconds: float | None = 120.0
    acoustic_groups: list[str] = Field(
        default_factory=lambda: ["loudness", "spectral", "rhythm"])
    sample_rate: int = 22050
    exclude_same_participant_pairs: bool = True
    n_permutations: int = 5000
    random_seed: int = 42


class ExperimentsConfig(_Config):
    classification: ClassificationConfig = Field(default_factory=ClassificationConfig)
    regression: RegressionConfig = Field(default_factory=RegressionConfig)
    clustering: ClusteringConfig = Field(default_factory=ClusteringConfig)
    synchronization: SynchronizationConfig = Field(default_factory=SynchronizationConfig)
    song_similarity: SongSimilarityConfig = Field(default_factory=SongSimilarityConfig)


class ApplicationConfig(_Config):
    project_name: str = "NeuroMusic Research Lab"
    interim_dir: str = "data/interim"
    processed_dir: str = "data/processed"
    experiments_dir: str = "experiments"
    reports_dir: str = "reports"
    artifacts_dir: str = "artifacts"
    figure_dpi: int = 300
    random_seed: int = 42


CONFIG_FILES: dict[str, tuple[str, type[BaseModel], str | None]] = {
    "application": ("application.yaml", ApplicationConfig, None),
    "dataset": ("dataset.yaml", DatasetConfig, "dataset"),
    "preprocessing": ("preprocessing.yaml", PreprocessingConfig, "preprocessing"),
    "features": ("features.yaml", FeatureConfig, "features"),
    "statistics": ("statistics.yaml", StatisticsConfig, "statistics"),
    "experiments": ("experiments.yaml", ExperimentsConfig, "experiments"),
}


def load_yaml_model[M: BaseModel](path: Path, model: type[M], root_key: str | None = None) -> M:
    """Load ``path`` into ``model``; a missing file yields the defaults."""
    if not path.exists():
        return model()
    with path.open(encoding="utf-8") as fh:
        raw: Any = yaml.safe_load(fh) or {}
    if root_key is not None and isinstance(raw, dict) and root_key in raw:
        raw = raw[root_key]
    return model.model_validate(raw)


def dump_yaml_model(cfg: BaseModel, path: Path, root_key: str | None = None) -> None:
    data = cfg.model_dump(mode="json")
    if root_key is not None:
        data = {root_key: data}
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        yaml.safe_dump(data, fh, sort_keys=False)


def config_hash(*configs: BaseModel, length: int = 12) -> str:
    """Stable short hash of one or more configuration objects."""
    payload = json.dumps(
        [c.model_dump(mode="json") for c in configs], sort_keys=True, separators=(",", ":")
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:length]
