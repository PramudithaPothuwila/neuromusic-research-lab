"""Enumerations shared across the domain."""

from enum import StrEnum


class ImportStatus(StrEnum):
    IMPORTED = "imported"
    DUPLICATE = "duplicate"
    REJECTED = "rejected"


class ValidationStatus(StrEnum):
    PASSED = "passed"
    PASSED_WITH_WARNINGS = "passed_with_warnings"
    FAILED = "failed"
    NOT_VALIDATED = "not_validated"


class Severity(StrEnum):
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"


class QualityStatus(StrEnum):
    GOOD = "good"
    FLAGGED = "flagged"
    REJECTED = "rejected"


class ExecutionStatus(StrEnum):
    PENDING = "pending"
    COMPLETED = "completed"
    FAILED = "failed"


class SignalUnits(StrEnum):
    """Amplitude units of the raw signal. ``UNKNOWN`` disables physical-unit labelling."""

    UNKNOWN = "unknown"
    MICROVOLTS = "uV"
    VOLTS = "V"


class ReferenceMethod(StrEnum):
    PRESERVE_ORIGINAL = "preserve_original"
    AVERAGE = "average"
    CHANNELS = "channels"


class ExperimentType(StrEnum):
    CLASSIFICATION = "classification"
    REGRESSION = "regression"
    CLUSTERING = "clustering"
    STATISTICS = "statistics"
    SYNCHRONIZATION = "synchronization"
    SONG_SIMILARITY = "song_similarity"


class ObservationLevel(StrEnum):
    TRIAL = "trial"
    WINDOW = "window"
