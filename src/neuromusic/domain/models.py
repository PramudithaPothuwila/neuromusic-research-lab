"""Typed domain entities persisted in the dataset registry."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from neuromusic.domain.enums import (
    ExecutionStatus,
    ImportStatus,
    QualityStatus,
    Severity,
    ValidationStatus,
)


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)


class Participant(_Model):
    """Pseudonymous participant. Demographics are never inferred from filenames."""

    participant_id: str
    group: str | None = None
    demographics: dict[str, Any] = Field(default_factory=dict)


class Song(_Model):
    song_id: str
    song_name: str
    artist: str | None = None
    duration_seconds: float | None = None
    source_metadata: dict[str, Any] = Field(default_factory=dict)


class ValidationIssue(_Model):
    code: str
    severity: Severity
    message: str
    channel: str | None = None
    start_s: float | None = None
    end_s: float | None = None
    value: float | None = None


class ValidationReport(_Model):
    status: ValidationStatus
    detected_sampling_rate: float | None = None
    expected_sampling_rate: float | None = None
    sample_count: int = 0
    duration_seconds: float = 0.0
    channel_names: list[str] = Field(default_factory=list)
    issues: list[ValidationIssue] = Field(default_factory=list)

    @property
    def errors(self) -> list[ValidationIssue]:
        return [i for i in self.issues if i.severity is Severity.ERROR]

    @property
    def warnings(self) -> list[ValidationIssue]:
        return [i for i in self.issues if i.severity is Severity.WARNING]


class Recording(_Model):
    """One CSV file: a participant listening to a song (one trial).

    ``metadata_source`` records whether participant/song were inferred from the
    filename or corrected manually; manual values are never overwritten by re-parsing.
    """

    recording_id: str
    participant_id: str | None
    song_id: str | None
    trial_id: str = "T01"
    source_id: str | None = None
    source_filename: str
    raw_data_path: str
    content_sha256: str
    sampling_rate: float | None = None
    channel_names: list[str] = Field(default_factory=list)
    sample_count: int = 0
    duration_seconds: float = 0.0
    import_status: ImportStatus = ImportStatus.IMPORTED
    validation_status: ValidationStatus = ValidationStatus.NOT_VALIDATED
    validation: ValidationReport | None = None
    metadata_source: str = "filename"
    is_synthetic: bool = False
    excluded: bool = False
    exclusion_reason: str | None = None
    exclusion_source: str | None = None  # "manual" or "provenance_audit"
    duplicate_of: str | None = None  # recording_id of the presumed original signal
    signal_family: str | None = None
    # Playback alignment: seconds into the recording at which the stimulus started.
    playback_onset_s: float | None = None
    alignment_verified: bool = False
    alignment_evidence: str | None = None
    imported_at: datetime = Field(default_factory=datetime.now)

    @property
    def is_analysable(self) -> bool:
        return (
            not self.excluded
            and self.participant_id is not None
            and self.song_id is not None
            and self.validation_status
            in (ValidationStatus.PASSED, ValidationStatus.PASSED_WITH_WARNINGS)
        )


class ExperimentSession(_Model):
    session_id: str
    participant_id: str
    recording_date: str | None = None
    song_presentation_order: list[str] | None = None
    recording_device: str | None = None
    electrode_reference: str | None = None
    playback_alignment_metadata: dict[str, Any] = Field(default_factory=dict)


class EmotionRating(_Model):
    rating_id: str
    participant_id: str
    song_id: str
    trial_id: str = "T01"
    rating_type: str
    rating_value: float
    scale_min: float
    scale_max: float
    definition: str | None = None


class ArtifactFlag(_Model):
    """A detected signal problem. ``channel`` is None for whole-recording events."""

    kind: str
    reason: str
    channel: str | None
    scope: str = "segment"  # "segment" (localised in time) or "recording" (whole channel)
    start_s: float
    end_s: float
    value: float | None = None
    threshold: float | None = None


class WindowInfo(_Model):
    recording_id: str
    participant_id: str
    song_id: str
    trial_id: str
    window_id: int
    start_time: float
    end_time: float
    quality_status: QualityStatus


class ProcessingRun(_Model):
    processing_id: str
    recording_id: str
    configuration_hash: str
    pipeline_version: str
    quality_results: dict[str, Any] = Field(default_factory=dict)
    artifact_path: str | None = None
    execution_status: ExecutionStatus = ExecutionStatus.PENDING
    error: str | None = None
    created_at: datetime = Field(default_factory=datetime.now)
