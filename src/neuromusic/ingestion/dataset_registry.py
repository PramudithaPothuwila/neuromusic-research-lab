"""Persistent registry of participants, songs, recordings and sessions.

Raw files live under ``raw_dir`` and are never modified. Files imported from elsewhere are copied
into ``raw_dir/imports``; ZIP archives are extracted (after safety checks) into
``raw_dir/extracted/<archive>``. Re-imports are detected by SHA-256 content hash.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import stat
import tempfile
import zipfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path, PurePosixPath

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

from neuromusic.config import DatasetConfig
from neuromusic.domain.enums import ImportStatus, ValidationStatus
from neuromusic.domain.models import ExperimentSession, Participant, Recording, Song
from neuromusic.ingestion.csv_reader import EEGReadError, file_sha256, read_eeg_csv
from neuromusic.ingestion.metadata_parser import FilenameParser, slugify, song_key
from neuromusic.ingestion.provenance import (
    DEFAULT_THRESHOLD,
    SignalFamily,
    find_families,
    signal_signature,
)
from neuromusic.ingestion.validator import validate_recording

SUPPORTED_SUFFIXES = {".csv"}
SYNTHETIC_MARKER = "SYNTHETIC"


class RegistryImportError(ValueError):
    """Raised for unsafe or unsupported imports."""


class RejectedFile(BaseModel):
    path: str
    reason: str
    rejected_at: datetime = Field(default_factory=datetime.now)


class IdenticalCopy(BaseModel):
    """A file whose bytes equal an already-registered recording (not registered separately)."""

    raw_data_path: str
    source_filename: str
    duplicate_of: str
    participant_id: str | None = None
    song_id: str | None = None
    source_id: str | None = None


class RegistryState(BaseModel):
    schema_version: int = 1
    participants: dict[str, Participant] = Field(default_factory=dict)
    songs: dict[str, Song] = Field(default_factory=dict)
    recordings: dict[str, Recording] = Field(default_factory=dict)
    sessions: dict[str, ExperimentSession] = Field(default_factory=dict)
    rejected: list[RejectedFile] = Field(default_factory=list)
    identical_copies: list[IdenticalCopy] = Field(default_factory=list)


@dataclass(frozen=True)
class ImportResult:
    path: str
    status: ImportStatus
    recording_id: str | None
    message: str


class DatasetRegistry:
    REGISTRY_FILE = "registry.json"

    def __init__(self, raw_dir: Path, metadata_dir: Path, config: DatasetConfig) -> None:
        self.raw_dir = Path(raw_dir).resolve()
        self.metadata_dir = Path(metadata_dir).resolve()
        self.config = config
        self.parser = FilenameParser(config.filename)
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_dir.mkdir(parents=True, exist_ok=True)
        self.state = self._load()

    # ------------------------------------------------------------------ persistence

    @property
    def registry_path(self) -> Path:
        return self.metadata_dir / self.REGISTRY_FILE

    def _load(self) -> RegistryState:
        if not self.registry_path.exists():
            return RegistryState()
        return RegistryState.model_validate_json(self.registry_path.read_text(encoding="utf-8"))

    def reload(self) -> None:
        self.state = self._load()

    def save(self) -> None:
        payload = self.state.model_dump_json(indent=2)
        fd, tmp = tempfile.mkstemp(dir=self.metadata_dir, suffix=".tmp")
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(payload)
        os.replace(tmp, self.registry_path)

    # ---------------------------------------------------------------------- queries

    @property
    def recordings(self) -> dict[str, Recording]:
        return self.state.recordings

    def resolve_path(self, recording: Recording) -> Path:
        return self.raw_dir / PurePosixPath(recording.raw_data_path)

    def analysable_recordings(self) -> list[Recording]:
        return sorted(
            (r for r in self.state.recordings.values() if r.is_analysable),
            key=lambda r: (r.participant_id or "", r.song_id or "", r.trial_id),
        )

    def has_synthetic_data(self) -> bool:
        return any(r.is_synthetic for r in self.state.recordings.values())

    def recordings_frame(self) -> pd.DataFrame:
        rows = [
            {
                "recording_id": r.recording_id,
                "participant_id": r.participant_id,
                "song_id": r.song_id,
                "trial_id": r.trial_id,
                "source_id": r.source_id,
                "source_filename": r.source_filename,
                "sampling_rate": r.sampling_rate,
                "n_channels": len(r.channel_names),
                "sample_count": r.sample_count,
                "duration_seconds": r.duration_seconds,
                "validation_status": r.validation_status.value,
                "n_errors": len(r.validation.errors) if r.validation else 0,
                "n_warnings": len(r.validation.warnings) if r.validation else 0,
                "metadata_source": r.metadata_source,
                "excluded": r.excluded,
                "is_synthetic": r.is_synthetic,
                "alignment_verified": r.alignment_verified,
                "playback_onset_s": r.playback_onset_s,
                "alignment_evidence": r.alignment_evidence,
            }
            for r in self.state.recordings.values()
        ]
        return pd.DataFrame(rows)

    def coverage_matrix(self, analysable_only: bool = True) -> pd.DataFrame:
        """Participants x songs table of recording counts (0 = missing)."""
        recs = (
            self.analysable_recordings()
            if analysable_only
            else [r for r in self.state.recordings.values() if r.participant_id and r.song_id]
        )
        if not recs:
            return pd.DataFrame()
        frame = pd.DataFrame([{"participant_id": r.participant_id, "song_id": r.song_id}
                              for r in recs])
        matrix = pd.crosstab(frame["participant_id"], frame["song_id"])
        songs = sorted(self.state.songs)
        participants = sorted(self.state.participants)
        return matrix.reindex(index=participants, columns=songs, fill_value=0)

    def missing_recordings(self) -> list[tuple[str, str]]:
        matrix = self.coverage_matrix()
        if matrix.empty:
            return []
        return [(p, s) for p in matrix.index for s in matrix.columns if matrix.loc[p, s] == 0]

    def fingerprint(self) -> str:
        """Hash identifying the analysable dataset content and its metadata."""
        items = sorted(
            (r.recording_id, r.content_sha256, r.participant_id, r.song_id, r.trial_id)
            for r in self.analysable_recordings()
        )
        return hashlib.sha256(json.dumps(items).encode()).hexdigest()[:16]

    # ---------------------------------------------------------------------- imports

    def _relative(self, path: Path) -> str:
        return path.resolve().relative_to(self.raw_dir).as_posix()

    def _is_inside_raw(self, path: Path) -> bool:
        try:
            path.resolve().relative_to(self.raw_dir)
        except ValueError:
            return False
        return True

    def _find_by_sha(self, sha: str) -> Recording | None:
        return next((r for r in self.state.recordings.values() if r.content_sha256 == sha), None)

    def _next_trial_id(self, participant_id: str | None, song_id: str | None) -> str:
        existing = [
            r for r in self.state.recordings.values()
            if r.participant_id == participant_id and r.song_id == song_id
        ]
        return f"T{len(existing) + 1:02d}"

    def register_file(self, path: Path) -> ImportResult:
        """Register a CSV that already resides inside ``raw_dir``."""
        path = Path(path)
        if not self._is_inside_raw(path):
            raise RegistryImportError(f"{path} is outside the raw data directory; use import_file")
        if path.suffix.lower() not in SUPPORTED_SUFFIXES:
            return self._reject(path, f"unsupported file type {path.suffix!r}")
        size_mb = path.stat().st_size / 1e6
        if size_mb > self.config.max_file_size_mb:
            return self._reject(path, f"file exceeds {self.config.max_file_size_mb} MB limit")

        sha = file_sha256(path)
        existing = self._find_by_sha(sha)
        if existing is not None:
            self._record_identical_copy(path, existing)
            return ImportResult(str(path), ImportStatus.DUPLICATE, existing.recording_id,
                                f"identical content already registered as "
                                f"{existing.recording_id} ({existing.source_filename})")
        try:
            raw = read_eeg_csv(path, self.config.channel_prefix)
        except EEGReadError as exc:
            return self._reject(path, str(exc))

        report = validate_recording(raw, self.config)
        parsed = self.parser.parse(path.name)
        participant_id = parsed.participant_id
        song_id = parsed.song_id
        trial_id = parsed.trial_id or self._next_trial_id(participant_id, song_id)
        recording = Recording(
            recording_id=f"REC-{sha[:12]}",
            participant_id=participant_id,
            song_id=song_id,
            trial_id=trial_id,
            source_id=parsed.source_id,
            source_filename=path.name,
            raw_data_path=self._relative(path),
            content_sha256=sha,
            sampling_rate=report.detected_sampling_rate,
            channel_names=report.channel_names,
            sample_count=report.sample_count,
            duration_seconds=report.duration_seconds,
            validation_status=report.status,
            validation=report,
            metadata_source="filename" if parsed.matched else "unparsed",
            is_synthetic=SYNTHETIC_MARKER in path.name.upper(),
        )
        if participant_id is not None:
            self.state.participants.setdefault(participant_id,
                                               Participant(participant_id=participant_id))
        if song_id is not None and parsed.song_name is not None:
            self.state.songs.setdefault(song_id, Song(song_id=song_id, song_name=parsed.song_name))
        self.state.recordings[recording.recording_id] = recording
        message = f"validation {report.status.value}"
        if not parsed.matched:
            message += "; filename did not match pattern, participant/song require manual entry"
        return ImportResult(str(path), ImportStatus.IMPORTED, recording.recording_id, message)

    def _record_identical_copy(self, path: Path, existing: Recording) -> None:
        rel = self._relative(path)
        if rel == existing.raw_data_path or any(
                c.raw_data_path == rel for c in self.state.identical_copies):
            return
        parsed = self.parser.parse(path.name)
        self.state.identical_copies.append(IdenticalCopy(
            raw_data_path=rel, source_filename=path.name, duplicate_of=existing.recording_id,
            participant_id=parsed.participant_id, song_id=parsed.song_id,
            source_id=parsed.source_id))

    def _reject(self, path: Path, reason: str) -> ImportResult:
        self.state.rejected.append(RejectedFile(path=str(path), reason=reason))
        return ImportResult(str(path), ImportStatus.REJECTED, None, reason)

    def import_file(self, source: Path, save: bool = True) -> ImportResult:
        """Import one CSV (copied into raw_dir/imports unless already inside raw_dir)."""
        source = Path(source)
        if source.suffix.lower() == ".zip":
            raise RegistryImportError("use import_zip for archives")
        if source.suffix.lower() not in SUPPORTED_SUFFIXES:
            result = self._reject(source, f"unsupported file type {source.suffix!r}")
        elif self._is_inside_raw(source):
            result = self.register_file(source)
        else:
            sha = file_sha256(source)
            existing = self._find_by_sha(sha)
            if existing is not None:
                result = ImportResult(str(source), ImportStatus.DUPLICATE, existing.recording_id,
                                      f"identical content already registered as "
                                      f"{existing.recording_id}")
            else:
                target = self._unique_target(self.raw_dir / "imports", source.name)
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, target)
                result = self.register_file(target)
        if save:
            self.save()
        return result

    @staticmethod
    def _unique_target(directory: Path, name: str) -> Path:
        target = directory / name
        stem, suffix, n = Path(name).stem, Path(name).suffix, 1
        while target.exists():
            target = directory / f"{stem}__{n}{suffix}"
            n += 1
        return target

    def import_zip(self, archive: Path, save: bool = True) -> list[ImportResult]:
        """Extract CSV members of ``archive`` safely and register them.

        Rejects absolute paths, ``..`` components, symlinks, non-CSV members, and archives
        exceeding member-count or uncompressed-size limits. No member is executed.
        """
        archive = Path(archive)
        if not zipfile.is_zipfile(archive):
            raise RegistryImportError(f"{archive.name} is not a valid ZIP archive")
        dest_root = self.raw_dir / "extracted" / slugify(archive.stem)
        results: list[ImportResult] = []
        with zipfile.ZipFile(archive) as zf:
            members = [m for m in zf.infolist() if not m.is_dir()]
            if len(members) > self.config.max_zip_members:
                raise RegistryImportError(f"archive has {len(members)} members "
                                   f"(limit {self.config.max_zip_members})")
            limit = self.config.max_file_size_mb * 1e6
            for member in members:
                name = member.filename
                pure = PurePosixPath(name.replace("\\", "/"))
                if pure.is_absolute() or ".." in pure.parts or ":" in pure.parts[0]:
                    raise RegistryImportError(f"unsafe path in archive: {name!r}")
                if stat.S_ISLNK(member.external_attr >> 16):
                    raise RegistryImportError(f"symbolic link in archive: {name!r}")
                if member.file_size > limit:
                    raise RegistryImportError(f"member {name!r} exceeds size limit")
            for member in members:
                pure = PurePosixPath(member.filename.replace("\\", "/"))
                if pure.name.startswith(".") or "__MACOSX" in pure.parts:
                    continue
                if pure.suffix.lower() not in SUPPORTED_SUFFIXES:
                    results.append(self._reject(Path(member.filename),
                                                f"unsupported file type {pure.suffix!r}"))
                    continue
                target = (dest_root / Path(*pure.parts)).resolve()
                if not target.is_relative_to(dest_root.resolve()):
                    raise RegistryImportError(f"unsafe path in archive: {member.filename!r}")
                if target.exists():
                    results.append(self.register_file(target))
                    continue
                target.parent.mkdir(parents=True, exist_ok=True)
                with zf.open(member) as src, target.open("wb") as dst:
                    shutil.copyfileobj(src, dst)
                results.append(self.register_file(target))
        if save:
            self.save()
        return results

    def scan_directory(self, directory: Path | None = None, recursive: bool = True,
                       save: bool = True) -> list[ImportResult]:
        """Register every CSV (and extract every ZIP) found under ``directory``."""
        directory = Path(directory) if directory is not None else self.raw_dir
        if not directory.is_dir():
            raise RegistryImportError(f"{directory} is not a directory")
        pattern = "**/*" if recursive else "*"
        results: list[ImportResult] = []
        for path in sorted(directory.glob(pattern)):
            if not path.is_file() or path.name.startswith("."):
                continue
            suffix = path.suffix.lower()
            if suffix == ".zip":
                results.extend(self.import_zip(path, save=False))
            elif suffix in SUPPORTED_SUFFIXES:
                results.append(self.import_file(path, save=False))
        if save:
            self.save()
        return results

    # ------------------------------------------------------------- metadata edits

    def update_recording(
        self,
        recording_id: str,
        *,
        participant_id: str | None = None,
        song_id: str | None = None,
        trial_id: str | None = None,
        excluded: bool | None = None,
        exclusion_reason: str | None = None,
        playback_onset_s: float | None = None,
        alignment_verified: bool | None = None,
        alignment_evidence: str | None = None,
        save: bool = True,
    ) -> Recording:
        """Apply manual metadata corrections; the raw file is untouched."""
        rec = self.state.recordings[recording_id]
        updates: dict[str, object] = {}
        if participant_id is not None:
            participant_id = participant_id.strip()
            updates["participant_id"] = participant_id
            self.state.participants.setdefault(participant_id,
                                               Participant(participant_id=participant_id))
        if song_id is not None:
            song_id = song_key(song_id)
            updates["song_id"] = song_id
            self.state.songs.setdefault(song_id, Song(song_id=song_id, song_name=song_id))
        if trial_id is not None:
            updates["trial_id"] = trial_id
        if participant_id is not None or song_id is not None or trial_id is not None:
            updates["metadata_source"] = "manual"
        if excluded is not None:
            updates["excluded"] = excluded
            updates["exclusion_reason"] = exclusion_reason if excluded else None
            updates["exclusion_source"] = "manual"
        if playback_onset_s is not None:
            updates["playback_onset_s"] = playback_onset_s
        if alignment_verified is not None:
            updates["alignment_verified"] = alignment_verified
        if alignment_evidence is not None:
            updates["alignment_evidence"] = alignment_evidence
        updated = rec.model_copy(update=updates)
        self.state.recordings[recording_id] = Recording.model_validate(updated.model_dump())
        self._prune_orphans()
        if save:
            self.save()
        return self.state.recordings[recording_id]

    def update_song(self, song_id: str, *, song_name: str | None = None,
                    artist: str | None = None, duration_seconds: float | None = None,
                    save: bool = True) -> Song:
        song = self.state.songs[song_id]
        if song_name is not None:
            song.song_name = song_name
        if artist is not None:
            song.artist = artist
        if duration_seconds is not None:
            song.duration_seconds = duration_seconds
        if save:
            self.save()
        return song

    def upsert_session(self, session: ExperimentSession, save: bool = True) -> None:
        self.state.sessions[session.session_id] = session
        if save:
            self.save()

    def revalidate(self, recording_id: str, save: bool = True) -> Recording:
        rec = self.state.recordings[recording_id]
        raw = read_eeg_csv(self.resolve_path(rec), self.config.channel_prefix)
        report = validate_recording(raw, self.config)
        rec.validation = report
        rec.validation_status = report.status
        rec.sampling_rate = report.detected_sampling_rate
        if save:
            self.save()
        return rec

    def remove_recording(self, recording_id: str, save: bool = True) -> None:
        """Unregister a recording. The raw file is left on disk."""
        self.state.recordings.pop(recording_id)
        self._prune_orphans()
        if save:
            self.save()

    def _prune_orphans(self) -> None:
        used_p = {r.participant_id for r in self.state.recordings.values()}
        used_s = {r.song_id for r in self.state.recordings.values()}
        for pid in [p for p in self.state.participants if p not in used_p]:
            del self.state.participants[pid]
        for sid in [s for s in self.state.songs if s not in used_s]:
            del self.state.songs[sid]

    # ------------------------------------------------------------- provenance

    SIGNATURE_CACHE = "signal_signatures.npz"

    def _signatures(self) -> dict[str, np.ndarray]:
        """Signal signatures keyed by recording id, cached by content hash."""
        cache_path = self.metadata_dir / self.SIGNATURE_CACHE
        cache: dict[str, np.ndarray] = {}
        if cache_path.exists():
            with np.load(cache_path, allow_pickle=False) as npz:
                cache = {k: npz[k] for k in npz.files}
        out: dict[str, np.ndarray] = {}
        changed = False
        for rec in self.state.recordings.values():
            if rec.content_sha256 not in cache:
                try:
                    sig = signal_signature(read_eeg_csv(self.resolve_path(rec),
                                                        self.config.channel_prefix))
                except (EEGReadError, FileNotFoundError):
                    sig = None
                cache[rec.content_sha256] = sig if sig is not None else np.empty(0, np.float32)
                changed = True
            if cache[rec.content_sha256].size:
                out[rec.recording_id] = cache[rec.content_sha256]
        if changed:
            np.savez(cache_path, **cache)  # type: ignore[arg-type]
        return out

    @staticmethod
    def _source_order(rec: Recording) -> tuple[int, str, str]:
        digits = "".join(ch for ch in (rec.source_id or "") if ch.isdigit())
        return (int(rec.is_synthetic), f"{int(digits):09d}" if digits else "~",
                rec.source_filename)

    def audit_provenance(self, threshold: float = DEFAULT_THRESHOLD, save: bool = True,
                         exclude_synthetic: bool = True) -> list[SignalFamily]:
        """Find near-duplicate signal families and exclude non-independent recordings.

        Policy (recorded on each recording; manual exclusions are never overridden):
        * synthetic files are excluded (unless ``exclude_synthetic`` is False, for demos);
        * in a family whose members share one song label, the earliest source is kept and the
          others are excluded as derived copies;
        * in a family with conflicting song labels, every member is excluded because the true
          stimulus cannot be determined.
        """
        recs = sorted(self.state.recordings.values(), key=self._source_order)
        order = [r.recording_id for r in recs]
        families = find_families(self._signatures(), order, threshold,
                                 {r.recording_id: r.song_id for r in recs})
        for rec in recs:
            if rec.exclusion_source == "provenance_audit":
                rec.excluded, rec.exclusion_reason, rec.exclusion_source = False, None, None
            rec.duplicate_of = None
            rec.signal_family = None
        for n, fam in enumerate(families, start=1):
            fam_id = f"FAM{n:03d}"
            primary = self.state.recordings[fam.members[0]]
            for rid in fam.members:
                rec = self.state.recordings[rid]
                rec.signal_family = fam_id
                if rid != primary.recording_id:
                    rec.duplicate_of = primary.recording_id
                if rec.exclusion_source == "manual":
                    continue
                if fam.label_conflict:
                    labels = sorted({s for s in fam.song_ids if s})
                    self._auto_exclude(rec, f"signal family {fam_id} shares one signal across "
                                       f"conflicting song labels {labels}")
                elif rid != primary.recording_id:
                    r = fam.pair_r.get((primary.recording_id, rid), fam.max_r)
                    self._auto_exclude(rec, f"near-duplicate of {primary.source_filename} "
                                       f"(signature r={r:.3f})")
        for rec in recs:
            copy_songs = {c.song_id for c in self.state.identical_copies
                          if c.duplicate_of == rec.recording_id and c.song_id}
            if rec.exclusion_source != "manual" and copy_songs - {rec.song_id}:
                labels = sorted(copy_songs | {rec.song_id or "?"})
                self._auto_exclude(rec, "byte-identical copies carry conflicting song labels "
                                   f"{labels}")
            if exclude_synthetic and rec.is_synthetic and rec.exclusion_source != "manual":
                self._auto_exclude(rec, "synthetic file")
        if save:
            self.save()
        return families

    @staticmethod
    def _auto_exclude(rec: Recording, reason: str) -> None:
        rec.excluded = True
        rec.exclusion_reason = reason
        rec.exclusion_source = "provenance_audit"

    def manifest_frame(self, song_titles: dict[str, str] | None = None) -> pd.DataFrame:
        """One row per registered file with its identified song and provenance status."""
        song_titles = song_titles or {}
        rows = []
        for rec in sorted(self.state.recordings.values(), key=self._source_order):
            if rec.is_synthetic:
                provenance = "synthetic"
            elif rec.exclusion_source == "provenance_audit" and "conflicting" in (
                    rec.exclusion_reason or ""):  # near-duplicate or byte-identical conflict
                provenance = "label_conflict"
            elif rec.duplicate_of:
                provenance = "derived_copy"
            else:
                provenance = "original"
            title = song_titles.get(rec.song_id or "") or (
                self.state.songs[rec.song_id].song_name if rec.song_id in self.state.songs
                else None)
            dup = self.state.recordings.get(rec.duplicate_of or "")
            clean = None
            if rec.participant_id:
                label = (title or rec.song_id or "unknown").replace(" ", "")
                clean = f"{rec.participant_id}_{label}_{rec.trial_id}"
            for copy in self.state.identical_copies:
                if copy.duplicate_of != rec.recording_id:
                    continue
                conflict = copy.song_id != rec.song_id
                copy_title = song_titles.get(copy.song_id or "") or copy.song_id
                rows.append({
                    "source_filename": copy.source_filename,
                    "recording_id": None,
                    "source_id": copy.source_id,
                    "participant_id": copy.participant_id,
                    "song_id": copy.song_id,
                    "song_title": copy_title,
                    "song_in_attribute_table": copy.song_id in song_titles,
                    "trial_id": None,
                    "clean_name": None,
                    "duration_seconds": round(rec.duration_seconds, 3),
                    "validation_status": rec.validation_status.value,
                    "provenance": "label_conflict" if conflict else "identical_copy",
                    "signal_family": rec.signal_family,
                    "duplicate_of_file": rec.source_filename,
                    "excluded": True,
                    "exclusion_reason": "byte-identical to " + rec.source_filename
                    + (" with a different song label" if conflict else ""),
                    "analysable": False,
                })
            rows.append({
                "source_filename": rec.source_filename,
                "recording_id": rec.recording_id,
                "source_id": rec.source_id,
                "participant_id": rec.participant_id,
                "song_id": rec.song_id,
                "song_title": title,
                "song_in_attribute_table": rec.song_id in song_titles,
                "trial_id": rec.trial_id,
                "clean_name": clean,
                "duration_seconds": round(rec.duration_seconds, 3),
                "validation_status": rec.validation_status.value,
                "provenance": provenance,
                "signal_family": rec.signal_family,
                "duplicate_of_file": dup.source_filename if dup else None,
                "excluded": rec.excluded,
                "exclusion_reason": rec.exclusion_reason,
                "analysable": rec.is_analysable,
            })
        frame = pd.DataFrame(rows)
        return frame.sort_values("source_filename", key=lambda c: c.str.upper()).reset_index(
            drop=True) if not frame.empty else frame

    def validation_summary(self) -> dict[str, int]:
        counts = {s.value: 0 for s in ValidationStatus}
        for rec in self.state.recordings.values():
            counts[rec.validation_status.value] += 1
        return counts
