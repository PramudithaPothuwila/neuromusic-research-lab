"""Column-level schema constants for raw CSVs and derived tables."""

SAMPLE_INDEX_COLUMN = "sample_index"
TIME_COLUMN = "time_s"
EEG_COLUMN_PREFIX = "EEG."

# Verified from SRC001_SOURCE_P001_Amude.csv. Configurable in configs/dataset.yaml.
DEFAULT_CHANNELS: tuple[str, ...] = (
    "AF3", "F7", "F3", "FC5", "T7", "P7", "O1",
    "O2", "P8", "T8", "FC6", "F4", "F8", "AF4",
)

# Identifier columns carried by every window- and trial-level feature table.
WINDOW_KEY_COLUMNS: tuple[str, ...] = (
    "recording_id", "participant_id", "song_id", "trial_id",
    "window_id", "start_time", "end_time", "quality_status",
)
TRIAL_KEY_COLUMNS: tuple[str, ...] = (
    "recording_id", "participant_id", "song_id", "trial_id",
    "n_windows_total", "n_windows_valid",
)

# Feature column naming: "<family>.<measure>[.<band>].<scope>", e.g.
#   spectral.log_abs_power.alpha.AF3, temporal.rms.O1, region.rel_power.alpha.frontal_left,
#   asymmetry.alpha.F4-F3
FEATURE_SEPARATOR = "."


def is_feature_column(column: str) -> bool:
    return column not in WINDOW_KEY_COLUMNS and column not in TRIAL_KEY_COLUMNS and (
        FEATURE_SEPARATOR in column
    )
