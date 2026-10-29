"""Command-line interface: ``neuromusic <command>``."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from neuromusic import __version__, services
from neuromusic.config import CONFIG_FILES, dump_yaml_model
from neuromusic.workspace import ROOT_ENV, Workspace


def _ws(args: argparse.Namespace) -> Workspace:
    return Workspace(Path(args.root) if args.root else None)


def _progress(i: int, n: int, label: str) -> None:
    print(f"  [{i}/{n}] {label}", file=sys.stderr)


def _print_records(results) -> None:
    for r in results:
        print(f"{r.status.value:9s} {Path(r.path).name}: {r.message}")


# ------------------------------------------------------------------------- commands


def cmd_init(args: argparse.Namespace) -> int:
    root = Path(args.root or ".").resolve()
    for _, (filename, model, key) in CONFIG_FILES.items():
        path = root / "configs" / filename
        if path.exists():
            print(f"exists   {path}")
            continue
        dump_yaml_model(model(), path, key)
        print(f"created  {path}")
    for sub in ("data/raw/eeg", "data/metadata", "data/interim", "data/processed",
                "experiments", "reports"):
        (root / sub).mkdir(parents=True, exist_ok=True)
    return 0


def cmd_import(args: argparse.Namespace) -> int:
    ws = _ws(args)
    reg = ws.registry
    for raw in args.paths:
        path = Path(raw)
        if path.is_dir():
            _print_records(reg.scan_directory(path))
        elif path.suffix.lower() == ".zip":
            _print_records(reg.import_zip(path))
        else:
            _print_records([reg.import_file(path)])
    if not args.no_audit:
        manifest = services.import_and_audit(ws)
        print(manifest["provenance"].value_counts().to_string())
    return 0


def cmd_scan(args: argparse.Namespace) -> int:
    ws = _ws(args)
    manifest = services.import_and_audit(ws, args.threshold)
    counts = manifest["provenance"].value_counts().to_dict() if not manifest.empty else {}
    print(f"files: {len(manifest)}  provenance: {counts}")
    print(f"analysable: {int(manifest['analysable'].sum()) if not manifest.empty else 0}")
    print(f"manifest:   {ws.metadata_dir / services.MANIFEST_FILE}")
    return 0


def cmd_validate(args: argparse.Namespace) -> int:
    ws = _ws(args)
    for rid in list(ws.registry.recordings):
        rec = ws.registry.revalidate(rid, save=False)
        if rec.validation and rec.validation.issues:
            codes = sorted({i.code for i in rec.validation.issues})
            print(f"{rec.validation_status.value:22s} {rec.source_filename}: {', '.join(codes)}")
    ws.registry.save()
    print(ws.registry.validation_summary())
    return 0


def cmd_align(args: argparse.Namespace) -> int:
    """Record playback onsets (seconds into each recording at which the song started)."""
    ws = _ws(args)
    reg = ws.registry
    targets = args.recordings or [r.recording_id for r in reg.recordings.values()
                                  if not r.is_synthetic]
    for rid in targets:
        reg.update_recording(rid, playback_onset_s=args.onset, alignment_verified=True,
                             alignment_evidence=args.evidence, save=False)
    reg.save()
    print(f"playback onset {args.onset} s recorded for {len(targets)} recording(s)")
    return 0


def cmd_preprocess(args: argparse.Namespace) -> int:
    ws = _ws(args)
    results = services.preprocess_recordings(ws, force=args.force, progress=_progress)
    for r in results:
        if r.status != "skipped":
            print(f"{r.status:7s} {r.recording_id}: {r.message}")
    failed = sum(r.status == "failed" for r in results)
    print(f"done: {len(results) - failed} ok/skipped, {failed} failed "
          f"(hash {ws.preprocessing_hash})")
    return 1 if failed and failed == len(results) else 0


def cmd_features(args: argparse.Namespace) -> int:
    ws = _ws(args)
    store = services.build_features(ws, progress=_progress)
    manifest = store.manifest(ws.preprocessing_hash, ws.feature_hash)
    print(json.dumps({k: manifest[k] for k in ("n_windows", "n_trials", "n_valid_trials",
                                                "dataset_fingerprint")}, indent=2))
    print(store.set_dir(ws.preprocessing_hash, ws.feature_hash))
    return 0


def _runner(args: argparse.Namespace):
    from neuromusic.experiments.runner import ExperimentRunner

    return ExperimentRunner(_ws(args))


def _show(record) -> None:
    print(f"experiment: {record.experiment_id} ({record.status.value})")
    print(json.dumps(record.metrics, indent=2, default=str))
    for note in record.notes:
        print(f"note: {note}")


def cmd_stats(args: argparse.Namespace) -> int:
    _show(_runner(args).run_statistics())
    return 0


def cmd_classify(args: argparse.Namespace) -> int:
    runner = _runner(args)
    cfg = runner.ws.experiments.classification
    updates = {k: v for k, v in {
        "models": args.models, "feature_prefixes": args.prefixes,
        "observation_level": args.level, "n_permutations": args.permutations,
        "n_jobs": args.jobs}.items() if v is not None}
    if args.no_tune:
        updates["tune_hyperparameters"] = False
    _show(runner.run_classification(cfg.model_copy(update=updates), progress=_progress))
    return 0


def cmd_cluster(args: argparse.Namespace) -> int:
    _show(_runner(args).run_clustering())
    return 0


def cmd_regress(args: argparse.Namespace) -> int:
    runner = _runner(args)
    cfg = runner.ws.experiments.regression
    if args.rating_type:
        cfg = cfg.model_copy(update={"rating_type": args.rating_type})
    _show(runner.run_regression(cfg))
    return 0


def cmd_sync(args: argparse.Namespace) -> int:
    _show(_runner(args).run_synchronization(args.song))
    return 0


def cmd_songs(args: argparse.Namespace) -> int:
    runner = _runner(args)
    cfg = runner.ws.experiments.song_similarity
    if args.full_songs:
        cfg = cfg.model_copy(update={"song_segment_seconds": None})
    if args.permutations is not None:
        cfg = cfg.model_copy(update={"n_permutations": args.permutations})
    _show(runner.run_song_similarity(cfg))
    return 0


def cmd_ratings(args: argparse.Namespace) -> int:
    from neuromusic.ingestion.ratings import import_ratings

    ws = _ws(args)
    ratings = import_ratings(Path(args.csv), ws.metadata_dir)
    print(f"imported {len(ratings)} rating(s); types: "
          f"{sorted({r.rating_type for r in ratings})}")
    return 0


def cmd_stimuli(args: argparse.Namespace) -> int:
    from neuromusic.ingestion import stimuli

    ws = _ws(args)
    catalogue = stimuli.load_catalogue(ws.metadata_dir)
    audio_dir = ws.root / "data" / "stimuli" / "audio"
    registered = sorted(ws.registry.state.songs)
    if args.action == "list":
        for row in stimuli.catalogue_status(catalogue, audio_dir, registered):
            print(f"{row['song_key']:28s} eeg={'yes' if row['has_eeg'] else 'no ':3s} "
                  f"audio={row['audio_file'] or '-':32s} video={row['video_id']}")
        return 0
    if args.action == "import":
        if not args.file or not args.song:
            raise ValueError("stimuli import requires --song and --file")
        source = stimuli.resolve(catalogue, args.song)
        if source is None:
            raise ValueError(f"song {args.song!r} not in {stimuli.SOURCES_FILE}")
        print(stimuli.import_audio(Path(args.file), source, audio_dir, args.rights))
        return 0
    if args.action == "confirm":
        if not args.evidence:
            raise ValueError("stimuli confirm requires --evidence describing the confirmation")
        keys = stimuli.confirm_identity(ws.metadata_dir, audio_dir, args.evidence)
        print(f"confirmed {len(keys)} stimulus file(s): {', '.join(keys)}")
        return 0
    if args.action == "scan":
        results = stimuli.scan_incoming(ws.root / "data" / "stimuli" / "incoming", catalogue,
                                        audio_dir, args.rights)
        for name, status in results.items():
            print(f"{status:40s} <- {name}")
        if not results:
            print("no files in data/stimuli/incoming")
        return 0
    selected = [s for s in catalogue if not args.song or args.song in s.keys]
    results = stimuli.fetch_youtube_audio(selected, audio_dir, args.rights,
                                          confirm_rights=args.confirm_rights)
    for key, status in results.items():
        print(f"{key:28s} {status}")
    return 0


def cmd_report(args: argparse.Namespace) -> int:
    from neuromusic.reporting.generator import generate_report

    ws = _ws(args)
    out = generate_report(ws, args.experiments, tuple(args.formats.split(",")))
    print(out)
    return 0


def cmd_experiments(args: argparse.Namespace) -> int:
    from neuromusic.experiments.persistence import ExperimentStore

    ws = _ws(args)
    for rec in ExperimentStore(ws.experiments_dir).list_records(args.type):
        flag = " [synthetic]" if rec.is_synthetic_data else ""
        print(f"{rec.experiment_id}  {rec.status.value:9s} participants={rec.n_participants} "
              f"songs={len(rec.song_ids)}{flag}")
    return 0


def cmd_demo(args: argparse.Namespace) -> int:
    """Create a synthetic workspace and run the full pipeline (clearly labelled synthetic)."""
    from neuromusic.experiments.runner import ExperimentRunner
    from neuromusic.reporting.generator import generate_report
    from neuromusic.synthetic import SyntheticSpec, write_dataset

    root = Path(args.root or "demo_workspace").resolve()
    cmd_init(argparse.Namespace(root=str(root)))
    write_dataset(root / "data/raw/eeg", SyntheticSpec(n_participants=args.participants,
                                                       duration_s=args.duration))
    ws = Workspace(root)
    services.import_and_audit(ws, exclude_synthetic=False)
    services.build_features(ws, progress=_progress)
    runner = ExperimentRunner(ws)
    runner.run_statistics()
    runner.run_clustering()
    cfg = ws.experiments.classification.model_copy(update={"n_permutations": 0})
    runner.run_classification(cfg, progress=_progress)
    print(generate_report(ws))
    return 0


def cmd_dashboard(args: argparse.Namespace) -> int:
    app = Path(__file__).resolve().parents[2] / "dashboard" / "app.py"
    env = dict(os.environ)
    if args.root:
        env[ROOT_ENV] = str(Path(args.root).resolve())
    return subprocess.call([sys.executable, "-m", "streamlit", "run", str(app),
                            "--server.port", str(args.port)], env=env)


# --------------------------------------------------------------------------- parser


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="neuromusic", description="NeuroMusic Research Lab")
    parser.add_argument("--version", action="version", version=__version__)
    parser.add_argument("--root", help="project root (default: nearest dir with configs/)")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("init", help="write default configs and directories").set_defaults(
        func=cmd_init)
    p = sub.add_parser("import", help="import CSV files, ZIP archives or directories")
    p.add_argument("paths", nargs="+")
    p.add_argument("--no-audit", action="store_true", help="skip the provenance audit")
    p.set_defaults(func=cmd_import)
    p = sub.add_parser("scan", help="register raw_dir, audit provenance, write manifest")
    p.add_argument("--threshold", type=float, default=0.9,
                   help="signature correlation linking near-duplicates")
    p.set_defaults(func=cmd_scan)
    sub.add_parser("validate", help="re-validate all registered recordings").set_defaults(
        func=cmd_validate)
    p = sub.add_parser("align", help="record verified playback onsets")
    p.add_argument("--onset", type=float, required=True,
                   help="seconds into the recording at which the song started")
    p.add_argument("--evidence", required=True, help="source of the alignment information")
    p.add_argument("--recordings", nargs="+", help="recording ids (default: all non-synthetic)")
    p.set_defaults(func=cmd_align)
    p = sub.add_parser("preprocess", help="run preprocessing for analysable recordings")
    p.add_argument("--force", action="store_true")
    p.set_defaults(func=cmd_preprocess)
    sub.add_parser("features", help="extract window and trial features").set_defaults(
        func=cmd_features)
    sub.add_parser("stats", help="descriptive and inferential statistics").set_defaults(
        func=cmd_stats)
    p = sub.add_parser("classify", help="LOSO song classification")
    p.add_argument("--models", nargs="+")
    p.add_argument("--prefixes", nargs="+", help="feature prefixes, e.g. region.rel_power.")
    p.add_argument("--level", choices=["trial", "window"])
    p.add_argument("--permutations", type=int)
    p.add_argument("--jobs", type=int)
    p.add_argument("--no-tune", action="store_true")
    p.set_defaults(func=cmd_classify)
    sub.add_parser("cluster", help="exploratory PCA and clustering").set_defaults(
        func=cmd_cluster)
    p = sub.add_parser("regress", help="LOSO rating regression (requires ratings)")
    p.add_argument("--rating-type")
    p.set_defaults(func=cmd_regress)
    p = sub.add_parser("sync", help="inter-subject correlation for one song")
    p.add_argument("--song", required=True)
    p.set_defaults(func=cmd_sync)
    p = sub.add_parser("songs", help="compare songs with each other and with EEG (RSA)")
    p.add_argument("--full-songs", action="store_true",
                   help="compare entire songs instead of the heard opening segment")
    p.add_argument("--permutations", type=int)
    p.set_defaults(func=cmd_songs)
    p = sub.add_parser("ratings", help="import an emotion-rating CSV")
    p.add_argument("csv")
    p.set_defaults(func=cmd_ratings)
    p = sub.add_parser("stimuli",
                       help="song audio catalogue: list, scan, import, fetch, confirm")
    p.add_argument("action", choices=["list", "scan", "import", "fetch", "confirm"])
    p.add_argument("--evidence", help="who confirmed the stored audio is what was played (confirm)")
    p.add_argument("--song", help="song key or alias (default for fetch: all)")
    p.add_argument("--file", help="audio file supplied by the rights holder (import)")
    p.add_argument("--rights", default="Licensed by the rights holder for research use in "
                   "this project", help="licence statement recorded with each file")
    p.add_argument("--confirm-rights", action="store_true",
                   help="confirm you are permitted to download the referenced audio")
    p.set_defaults(func=cmd_stimuli)
    p = sub.add_parser("report", help="generate a research report")
    p.add_argument("--experiments", nargs="+")
    p.add_argument("--formats", default="md,html")
    p.set_defaults(func=cmd_report)
    p = sub.add_parser("experiments", help="list experiments")
    p.add_argument("--type")
    p.set_defaults(func=cmd_experiments)
    p = sub.add_parser("demo", help="build a synthetic demo workspace and run the pipeline")
    p.add_argument("--participants", type=int, default=8)
    p.add_argument("--duration", type=float, default=60.0)
    p.set_defaults(func=cmd_demo)
    p = sub.add_parser("dashboard", help="launch the Streamlit dashboard")
    p.add_argument("--port", type=int, default=8501)
    p.set_defaults(func=cmd_dashboard)
    return parser


def main(argv: list[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):  # non-Latin titles on Windows consoles
            stream.reconfigure(encoding="utf-8", errors="replace")
    args = build_parser().parse_args(argv)
    try:
        return int(args.func(args) or 0)
    except (ValueError, RuntimeError, FileNotFoundError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
