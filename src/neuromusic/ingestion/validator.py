"""Structural and signal-level validation of raw recordings.

Validation never modifies data. Errors make a recording unanalysable until resolved; warnings
are recorded and surfaced to the user.
"""

from __future__ import annotations

import numpy as np

from neuromusic.config import DatasetConfig
from neuromusic.domain.enums import Severity, ValidationStatus
from neuromusic.domain.models import ValidationIssue, ValidationReport
from neuromusic.ingestion.csv_reader import RawEEG

MAX_REPORTED_LOCATIONS = 20


def detect_sampling_rate(times: np.ndarray) -> float | None:
    """Sampling rate from timestamps.

    The median increment is robust to isolated gaps but sensitive to timestamp rounding
    (e.g. 6-decimal timestamps at 256 Hz give 256.016 Hz). When the full-span estimate
    (n - 1) / (t_last - t_first) agrees with it to within 1 %, the span estimate is returned.
    """
    finite = times[np.isfinite(times)]
    if finite.size < 2:
        return None
    diffs = np.diff(finite)
    diffs = diffs[diffs > 0]
    if diffs.size == 0:
        return None
    median_rate = float(1.0 / np.median(diffs))
    span = float(finite[-1] - finite[0])
    if span > 0:
        span_rate = (finite.size - 1) / span
        if abs(span_rate - median_rate) / median_rate < 0.01:
            return span_rate
    return median_rate


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """Half-open [start, stop) index runs where ``mask`` is True."""
    if not mask.any():
        return []
    padded = np.concatenate([[False], mask, [False]])
    edges = np.flatnonzero(np.diff(padded.astype(np.int8)))
    return list(zip(edges[::2].tolist(), edges[1::2].tolist(), strict=True))


def validate_recording(raw: RawEEG, config: DatasetConfig) -> ValidationReport:
    issues: list[ValidationIssue] = []
    add = issues.append
    acq = config.acquisition
    vcfg = config.validation

    # ---- channel set
    expected = list(config.channels)
    missing = [c for c in expected if c not in raw.channel_names]
    extra = [c for c in raw.channel_names if c not in expected]
    if missing:
        add(ValidationIssue(code="missing_channels", severity=Severity.ERROR,
                            message=f"Expected channels missing: {', '.join(missing)}"))
    if extra:
        add(ValidationIssue(
            code="unexpected_channels",
            severity=Severity.WARNING if config.allow_extra_channels else Severity.ERROR,
            message=f"Unexpected channels present: {', '.join(extra)}"))
    duplicates = sorted({c for c in raw.channel_names if raw.channel_names.count(c) > 1})
    if duplicates:
        add(ValidationIssue(code="duplicate_channels", severity=Severity.ERROR,
                            message=f"Duplicate channel columns: {', '.join(duplicates)}"))

    # ---- numeric content
    for col, count in raw.non_numeric_counts.items():
        add(ValidationIssue(code="non_numeric", severity=Severity.ERROR, channel=col,
                            value=float(count),
                            message=f"{count} non-numeric value(s) in column {col!r}"))
    n_missing = int(np.isnan(raw.data).sum())
    if n_missing:
        add(ValidationIssue(code="missing_values", severity=Severity.ERROR, value=float(n_missing),
                            message=f"{n_missing} missing or non-numeric EEG value(s)"))
    n_inf = int(np.isinf(raw.data).sum())
    if n_inf:
        add(ValidationIssue(code="non_finite", severity=Severity.ERROR, value=float(n_inf),
                            message=f"{n_inf} infinite EEG value(s)"))

    # ---- sample index
    idx = raw.sample_index
    if not np.all(np.isfinite(idx)):
        add(ValidationIssue(code="sample_index_invalid", severity=Severity.ERROR,
                            message="sample_index contains missing or non-numeric values"))
    else:
        d_idx = np.diff(idx)
        n_dup = int(idx.size - np.unique(idx).size)
        if n_dup:
            add(ValidationIssue(code="duplicate_samples", severity=Severity.ERROR,
                                value=float(n_dup),
                                message=f"{n_dup} duplicated sample index value(s)"))
        if np.any(d_idx < 0):
            add(ValidationIssue(code="sample_index_not_monotonic", severity=Severity.ERROR,
                                message="sample_index decreases at "
                                f"{int((d_idx < 0).sum())} position(s)"))
        elif np.any(d_idx > 1):
            for start, _ in _runs(d_idx > 1)[:MAX_REPORTED_LOCATIONS]:
                add(ValidationIssue(code="sample_index_gap", severity=Severity.WARNING,
                                    value=float(d_idx[start]),
                                    message=f"sample_index jumps by {int(d_idx[start])} after "
                                    f"index {int(idx[start])}"))

    # ---- timestamps and sampling rate
    times = raw.times
    sfreq = detect_sampling_rate(times)
    duration = 0.0
    if sfreq is None or not np.all(np.isfinite(times)):
        add(ValidationIssue(code="timestamps_invalid", severity=Severity.ERROR,
                            message="Timestamps are missing, non-numeric or non-increasing"))
    else:
        duration = float(times[-1] - times[0] + 1.0 / sfreq)
        dt = np.diff(times)
        if np.any(dt <= 0):
            add(ValidationIssue(code="timestamps_not_increasing", severity=Severity.ERROR,
                                message=f"Timestamps fail to increase at {int((dt <= 0).sum())} "
                                "position(s)"))
        nominal = 1.0 / sfreq
        deviation = np.abs(dt - nominal) > acq.timestamp_jitter_tolerance * nominal
        gaps = deviation & (dt > nominal)
        for start, stop in _runs(gaps)[:MAX_REPORTED_LOCATIONS]:
            add(ValidationIssue(code="timestamp_gap", severity=Severity.WARNING,
                                start_s=float(times[start]), end_s=float(times[stop]),
                                value=float(dt[start]),
                                message=f"Timestamp discontinuity of {dt[start]:.6f} s at "
                                f"t={times[start]:.4f} s (nominal {nominal:.6f} s)"))
        n_irregular = int(deviation.sum())
        if n_irregular and not gaps.any():
            add(ValidationIssue(code="timestamp_jitter", severity=Severity.WARNING,
                                value=float(n_irregular),
                                message=f"{n_irregular} irregular timestamp increment(s)"))
        if acq.expected_sampling_rate is not None and (
            abs(sfreq - acq.expected_sampling_rate) > acq.sampling_rate_tolerance_hz
        ):
            add(ValidationIssue(code="sampling_rate_mismatch", severity=Severity.ERROR,
                                value=sfreq,
                                message=f"Detected sampling rate {sfreq:.3f} Hz differs from "
                                f"configured {acq.expected_sampling_rate:.3f} Hz"))
        if duration < vcfg.min_duration_seconds:
            add(ValidationIssue(code="too_short", severity=Severity.ERROR, value=duration,
                                message=f"Duration {duration:.2f} s is below the minimum "
                                f"{vcfg.min_duration_seconds:.2f} s"))

    # ---- signal-level checks (only on finite data)
    if sfreq is not None and np.isfinite(raw.data).all():
        min_run = max(2, round(vcfg.flatline_min_seconds * sfreq))
        for ch, signal in zip(raw.channel_names, raw.data, strict=True):
            flat = np.abs(np.diff(signal)) <= vcfg.flatline_epsilon
            for start, stop in _runs(flat):
                if stop - start + 1 >= min_run:
                    add(ValidationIssue(code="flatline", severity=Severity.WARNING, channel=ch,
                                        start_s=float(times[start]), end_s=float(times[stop]),
                                        message=f"{ch}: constant signal for "
                                        f"{(stop - start + 1) / sfreq:.2f} s"))
            median = np.median(signal)
            mad = np.median(np.abs(signal - median)) * 1.4826
            if mad <= vcfg.flatline_epsilon:
                add(ValidationIssue(code="channel_flat", severity=Severity.ERROR, channel=ch,
                                    message=f"{ch}: channel has (near) zero variability"))
                continue
            robust_z = np.abs(signal - median) / mad
            frac = float((robust_z > vcfg.extreme_robust_z).mean())
            if frac > 0:
                add(ValidationIssue(
                    code="extreme_excursions",
                    severity=Severity.WARNING if frac >= vcfg.extreme_fraction_warning
                    else Severity.INFO,
                    channel=ch, value=frac,
                    message=f"{ch}: {frac:.2%} of samples exceed {vcfg.extreme_robust_z:g} "
                    "robust SD from the channel median"))

    if any(i.severity is Severity.ERROR for i in issues):
        status = ValidationStatus.FAILED
    elif any(i.severity is Severity.WARNING for i in issues):
        status = ValidationStatus.PASSED_WITH_WARNINGS
    else:
        status = ValidationStatus.PASSED

    return ValidationReport(
        status=status,
        detected_sampling_rate=sfreq,
        expected_sampling_rate=acq.expected_sampling_rate,
        sample_count=raw.n_samples,
        duration_seconds=duration,
        channel_names=list(raw.channel_names),
        issues=issues,
    )
