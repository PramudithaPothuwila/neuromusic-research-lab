"""Stimulus (song audio) catalogue and licensed audio acquisition.

``data/metadata/song_sources.json`` lists reference videos per song. Matches between a video
and the audio actually played during EEG acquisition are unverified
(``recording_identity_verified: false``) until confirmed by the data owner.

Audio is stored under ``data/stimuli/audio/<song_key>.<ext>`` with a provenance sidecar
``<song_key>.json`` (source, licence statement, SHA-256, timestamp). Audio files are never
committed. Two acquisition routes exist:

* ``import_audio``: copy a file supplied by the rights holder (preferred).
* ``fetch_youtube_audio``: download via the optional ``yt-dlp`` package. It requires an explicit
  rights confirmation; the operator is responsible for compliance with the platform's terms of
  service, which are separate from the copyright licence.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from dataclasses import dataclass
from datetime import datetime
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

from neuromusic.ingestion.metadata_parser import song_key

SOURCES_FILE = "song_sources.json"
AUDIO_SUFFIXES = {".wav", ".flac", ".mp3", ".m4a", ".ogg", ".opus", ".webm"}


class StimulusError(RuntimeError):
    pass


@dataclass(frozen=True)
class StimulusSource:
    key: str  # canonical song key used across the platform
    title: str
    video_id: str | None
    aliases: tuple[str, ...]
    status: str
    evidence: str
    identity_verified: bool

    @property
    def url(self) -> str | None:
        return f"https://www.youtube.com/watch?v={self.video_id}" if self.video_id else None

    @property
    def keys(self) -> set[str]:
        return {self.key, *(song_key(a) for a in self.aliases)}


def load_catalogue(metadata_dir: Path) -> list[StimulusSource]:
    path = Path(metadata_dir) / SOURCES_FILE
    if not path.exists():
        return []
    entries: list[dict[str, Any]] = json.loads(path.read_text(encoding="utf-8"))
    return [StimulusSource(
        key=song_key(e["title"]), title=e["title"], video_id=e.get("video_id"),
        aliases=tuple(e.get("aliases", [])), status=e.get("status", ""),
        evidence=e.get("evidence", ""),
        identity_verified=bool(e.get("recording_identity_verified", False)))
        for e in entries]


def resolve(catalogue: list[StimulusSource], song_id: str) -> StimulusSource | None:
    """Catalogue entry whose title or alias matches a registry song id."""
    return next((s for s in catalogue if song_id in s.keys), None)


def audio_path(audio_dir: Path, source: StimulusSource) -> Path | None:
    matches = [p for p in Path(audio_dir).glob(f"{source.key}.*")
               if p.suffix.lower() in AUDIO_SUFFIXES]
    return matches[0] if matches else None


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        while chunk := fh.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def _write_sidecar(audio: Path, source: StimulusSource, origin: str, rights: str,
                   extra: dict[str, Any] | None = None) -> None:
    audio.with_suffix(".json").write_text(json.dumps({
        "song_key": source.key, "title": source.title, "video_id": source.video_id,
        "origin": origin, "rights_statement": rights, "file": audio.name,
        "sha256": _sha256(audio), "acquired_at": datetime.now().isoformat(timespec="seconds"),
        "recording_identity_verified": source.identity_verified, **(extra or {}),
    }, indent=2, ensure_ascii=False), encoding="utf-8")


def import_audio(file: Path, source: StimulusSource, audio_dir: Path, rights: str,
                 move: bool = False, extra: dict[str, Any] | None = None) -> Path:
    """Store a rights-holder-supplied audio file as ``<song_key><ext>`` (copy or move)."""
    file = Path(file)
    if file.suffix.lower() not in AUDIO_SUFFIXES:
        raise StimulusError(f"unsupported audio type {file.suffix!r}")
    audio_dir.mkdir(parents=True, exist_ok=True)
    target = audio_dir / f"{source.key}{file.suffix.lower()}"
    if target.exists():
        raise StimulusError(f"{target.name} already exists; remove it explicitly to replace")
    if move:
        shutil.move(str(file), target)
    else:
        shutil.copy2(file, target)
    _write_sidecar(target, source, f"file:{file.name}", rights,
                   {"original_filename": file.name, **(extra or {})})
    return target


def _partial_ratio(key: str, text: str) -> float:
    """Best similarity of ``key`` to any similar-length substring of ``text`` (0-1)."""
    if key in text:
        return 1.0
    best = 0.0
    for width in range(max(1, len(key) - 3), len(key) + 4):
        for i in range(max(1, len(text) - width + 1)):
            best = max(best, SequenceMatcher(None, key, text[i:i + width]).ratio())
    return best


def match_source(name: str, catalogue: list[StimulusSource], min_score: float = 0.8,
                 min_margin: float = 0.1) -> tuple[StimulusSource | None, float, str]:
    """Match a free-form file name (e.g. a video title) to a catalogue entry.

    Exact title/alias containment scores 1.0; otherwise fuzzy substring similarity on the
    ASCII-normalised name (non-Latin script is ignored). The best entry is accepted only if it
    scores >= ``min_score`` and beats the runner-up by >= ``min_margin``.
    """
    text = song_key(name)
    scored = sorted(((max(_partial_ratio(k, text) for k in s.keys), i)
                     for i, s in enumerate(catalogue)), reverse=True)
    if not scored:
        return None, 0.0, "empty catalogue"
    best, idx = scored[0]
    runner_up = scored[1][0] if len(scored) > 1 else 0.0
    if best < min_score:
        return None, best, f"no title scores >= {min_score} (best {best:.2f})"
    if best - runner_up < min_margin:
        return None, best, f"ambiguous: best {best:.2f}, runner-up {runner_up:.2f}"
    return catalogue[idx], best, "ok"


def scan_incoming(incoming_dir: Path, catalogue: list[StimulusSource], audio_dir: Path,
                  rights: str, move: bool = True) -> dict[str, str]:
    """Rename audio files dropped into ``incoming_dir`` to ``audio_dir/<song_key><ext>``.

    Files may carry arbitrary titles (e.g. "Kanchana Anuradhi - UDURAWEE (...) Official
    Lyric Video.opus"); see ``match_source``. Matching is one-to-one: if several files match
    the same song, none of them is imported. The original name and match score are recorded in
    the provenance sidecar. Returns {filename: status}.
    """
    incoming = Path(incoming_dir)
    files = sorted(f for f in incoming.iterdir()
                   if f.is_file() and not f.name.startswith(".")) if incoming.is_dir() else []
    results: dict[str, str] = {}
    matches: dict[Path, tuple[StimulusSource, float]] = {}
    for file in files:
        if file.suffix.lower() not in AUDIO_SUFFIXES:
            results[file.name] = f"skipped: unsupported type {file.suffix!r}"
            continue
        source, score, reason = match_source(file.stem, catalogue)
        if source is None:
            results[file.name] = f"unmatched: {reason}"
        else:
            matches[file] = (source, score)
    claimed: dict[str, list[Path]] = {}
    for file, (source, _) in matches.items():
        claimed.setdefault(source.key, []).append(file)
    for file, (source, score) in matches.items():
        if len(claimed[source.key]) > 1:
            results[file.name] = f"conflict: {len(claimed[source.key])} files match {source.key}"
            continue
        existing = audio_path(audio_dir, source)
        if existing is not None:
            results[file.name] = f"exists: {existing.name} (left in place)"
            continue
        target = import_audio(file, source, audio_dir, rights, move=move,
                              extra={"title_match_score": round(score, 3)})
        results[file.name] = f"-> {target.name} (match {score:.2f})"
    return results


def fetch_youtube_audio(sources: list[StimulusSource], audio_dir: Path, rights: str,
                        confirm_rights: bool) -> dict[str, str]:
    """Download best-quality audio for each source with ``yt-dlp`` (optional dependency).

    Returns {song_key: status}. Existing files are skipped, never overwritten.
    """
    if not confirm_rights:
        raise StimulusError("downloading requires confirm_rights=True: confirm you hold rights "
                            "to use these recordings and are permitted to download them")
    try:
        from yt_dlp import YoutubeDL
    except ImportError as exc:
        raise StimulusError("yt-dlp is not installed; run `uv sync --extra stimuli`") from exc
    audio_dir.mkdir(parents=True, exist_ok=True)
    results: dict[str, str] = {}
    for source in sources:
        if source.video_id is None:
            results[source.key] = "no video reference"
            continue
        if audio_path(audio_dir, source) is not None:
            results[source.key] = "exists"
            continue
        options = {"format": "bestaudio/best", "quiet": True, "noprogress": True,
                   "outtmpl": str(audio_dir / f"{source.key}.%(ext)s"), "noplaylist": True}
        try:
            with YoutubeDL(options) as ydl:
                ydl.download([source.url])
        except Exception as exc:  # yt-dlp raises many error types; record and continue
            results[source.key] = f"failed: {exc}"
            continue
        downloaded = audio_path(audio_dir, source)
        if downloaded is None:
            results[source.key] = "failed: no audio file produced"
            continue
        _write_sidecar(downloaded, source, f"youtube:{source.video_id}", rights)
        results[source.key] = f"downloaded {downloaded.name}"
    return results


def confirm_identity(metadata_dir: Path, audio_dir: Path, evidence: str) -> list[str]:
    """Record an owner confirmation that stored audio files are the exact stimuli played.

    Updates the catalogue entry and the provenance sidecar of every song that has a stored
    audio file. Prior evidence is preserved. Returns the confirmed song keys.
    """
    path = Path(metadata_dir) / SOURCES_FILE
    entries: list[dict[str, Any]] = json.loads(path.read_text(encoding="utf-8"))
    stamp = datetime.now().isoformat(timespec="seconds")
    confirmed = []
    for entry in entries:
        source = next(s for s in load_catalogue(metadata_dir) if s.title == entry["title"])
        audio = audio_path(audio_dir, source)
        if audio is None:
            continue
        entry["recording_identity_verified"] = True
        entry["identity_evidence"] = evidence
        entry["identity_confirmed_at"] = stamp
        entry["stimulus_file"] = audio.name
        sidecar = audio.with_suffix(".json")
        meta = json.loads(sidecar.read_text(encoding="utf-8"))
        meta.update({"recording_identity_verified": True, "identity_evidence": evidence,
                     "identity_confirmed_at": stamp})
        sidecar.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
        confirmed.append(source.key)
    path.write_text(json.dumps(entries, indent=2, ensure_ascii=False), encoding="utf-8")
    return confirmed


def catalogue_status(catalogue: list[StimulusSource], audio_dir: Path,
                     registered_song_ids: list[str]) -> list[dict[str, Any]]:
    rows = []
    for s in catalogue:
        path = audio_path(audio_dir, s)
        rows.append({"song_key": s.key, "title": s.title, "video_id": s.video_id,
                     "has_eeg": any(sid in s.keys for sid in registered_song_ids),
                     "audio_file": path.name if path else None,
                     "identity_verified": s.identity_verified, "status": s.status})
    return rows
