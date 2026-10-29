"""Page 2 — Dataset manager."""

import sys
import tempfile
from pathlib import Path

_DASHBOARD = str(Path(__file__).resolve().parents[1])
if _DASHBOARD not in sys.path:
    sys.path.insert(0, _DASHBOARD)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from components import state, widgets  # noqa: E402
from neuromusic import services  # noqa: E402
from neuromusic.ingestion.dataset_registry import RegistryImportError  # noqa: E402
from neuromusic.ingestion.provenance import DEFAULT_THRESHOLD  # noqa: E402
from neuromusic.ingestion.song_attributes import match_report  # noqa: E402
from neuromusic.visualization.statistical_plots import (  # noqa: E402
    plot_coverage_matrix,
    plot_provenance_summary,
)

ws = state.page_header("Dataset manager",
                       "Import, validate and annotate recordings. Raw files are never modified.")
registry = ws.registry


def show_results(results: list) -> None:
    if results:
        st.dataframe(pd.DataFrame([{"file": Path(r.path).name, "status": r.status.value,
                                    "recording": r.recording_id, "message": r.message}
                                   for r in results]), hide_index=True, width="stretch")


tab_import, tab_meta, tab_prov, tab_cov, tab_songs = st.tabs(
    ["Import", "Recordings & metadata", "Provenance", "Coverage", "Songs"])

with tab_import:
    st.markdown(f"Raw data directory: `{registry.raw_dir}`")
    uploads = st.file_uploader("Upload EEG CSV files or ZIP archives", type=["csv", "zip"],
                               accept_multiple_files=True)
    if uploads and st.button("Import uploaded files", type="primary"):
        results = []
        with tempfile.TemporaryDirectory() as tmp:
            for up in uploads:
                target = Path(tmp) / Path(up.name).name
                target.write_bytes(up.getbuffer())
                try:
                    if target.suffix.lower() == ".zip":
                        results.extend(registry.import_zip(target))
                    else:
                        results.append(registry.import_file(target))
                except RegistryImportError as exc:
                    st.error(f"{up.name}: {exc}")
        state.refresh()
        show_results(results)
    c1, c2 = st.columns(2)
    if c1.button("Scan raw data directory", icon=":material/search:"):
        with st.spinner("Scanning…"):
            try:
                results = registry.scan_directory()
            except RegistryImportError as exc:
                st.error(str(exc))
                results = []
        state.refresh()
        new = [r for r in results if r.status.value == "imported"]
        st.success(f"{len(new)} new recording(s) registered.")
        show_results(new)
    if registry.state.rejected:
        with st.expander(f"Rejected files ({len(registry.state.rejected)})"):
            st.dataframe(pd.DataFrame([r.model_dump() for r in registry.state.rejected]),
                         hide_index=True, width="stretch")

with tab_meta:
    frame = registry.recordings_frame()
    if frame.empty:
        st.info("No recordings registered.")
    else:
        st.caption("Edit participant, song, trial, exclusion and playback alignment. Changes are "
                   "recorded as manual metadata; filename parsing never overwrites them.")
        editable = ["participant_id", "song_id", "trial_id", "excluded", "exclusion_reason",
                    "playback_onset_s", "alignment_verified"]
        reasons = {r.recording_id: r.exclusion_reason for r in registry.recordings.values()}
        frame["exclusion_reason"] = frame["recording_id"].map(reasons)
        view = frame[["recording_id", "source_filename", *editable, "validation_status",
                      "n_errors", "n_warnings", "duration_seconds", "metadata_source"]]
        edited = st.data_editor(
            view, hide_index=True, width="stretch", key="meta_editor",
            disabled=["recording_id", "source_filename", "validation_status", "n_errors",
                      "n_warnings", "duration_seconds", "metadata_source"],
            column_config={
                "playback_onset_s": st.column_config.NumberColumn("playback onset (s)",
                                                                  min_value=0.0, format="%.3f"),
                "duration_seconds": st.column_config.NumberColumn("duration (s)", format="%.1f"),
            })
        if st.button("Save metadata changes", type="primary"):
            changed = 0
            for (_, old), (_, new) in zip(view.iterrows(), edited.iterrows(), strict=True):
                diff = {}
                for col in editable:
                    a, b = old[col], new[col]
                    if (pd.isna(a) and pd.isna(b)) or a == b:
                        continue
                    diff[col] = b
                if not diff:
                    continue
                kwargs = {}
                for col in ("participant_id", "song_id", "trial_id"):
                    if col in diff and isinstance(diff[col], str) and diff[col].strip():
                        kwargs[col] = diff[col]
                if "excluded" in diff or "exclusion_reason" in diff:
                    reason = new["exclusion_reason"]
                    kwargs["excluded"] = bool(new["excluded"])
                    kwargs["exclusion_reason"] = (
                        reason if isinstance(reason, str) and reason.strip()
                        else "manual exclusion") if new["excluded"] else None
                if "playback_onset_s" in diff and not pd.isna(diff["playback_onset_s"]):
                    kwargs["playback_onset_s"] = float(diff["playback_onset_s"])
                if "alignment_verified" in diff:
                    kwargs["alignment_verified"] = bool(np.bool_(diff["alignment_verified"]))
                if kwargs:
                    registry.update_recording(new["recording_id"], save=False, **kwargs)
                    changed += 1
            registry.save()
            services.write_manifest(ws)
            state.refresh()
            st.success(f"Updated {changed} recording(s).")

        st.markdown("#### Validation issues")
        rec = st.selectbox("Recording", list(registry.recordings.values()),
                           format_func=lambda r: f"{r.source_filename} ({r.validation_status})",
                           key="issues_rec")
        if rec is not None and rec.validation is not None:
            issues = pd.DataFrame([i.model_dump() for i in rec.validation.issues])
            if issues.empty:
                st.success("No validation issues.")
            else:
                st.dataframe(issues, hide_index=True, width="stretch")
            if st.button("Re-validate recording"):
                registry.revalidate(rec.recording_id)
                state.refresh()
                st.rerun()

with tab_prov:
    st.markdown(
        "Recordings whose signals are copies of other recordings are **not independent "
        "participants**. The audit links files whose signal signatures correlate above the "
        "threshold, keeps the earliest source of each signal, excludes derived copies, and "
        "excludes every member of a signal family whose copies carry conflicting song labels. "
        "Manual inclusion/exclusion decisions are never overridden.")
    threshold = st.slider("Signature correlation threshold", 0.5, 0.999, DEFAULT_THRESHOLD,
                          0.01)
    if st.button("Run provenance audit", type="primary", icon=":material/fact_check:"):
        with st.spinner("Comparing signal signatures…"):
            families = registry.audit_provenance(threshold)
            services.sync_song_metadata(ws)
            services.write_manifest(ws)
        state.refresh()
        st.success(f"Audit complete: {len(families)} signal family(ies) with shared signals.")
    manifest = ws.registry.manifest_frame(services.song_titles(ws))
    if not manifest.empty:
        widgets.plot(plot_provenance_summary(manifest), key="prov_ds")
        kinds = st.multiselect("Show provenance", sorted(manifest["provenance"].unique()),
                               default=sorted(manifest["provenance"].unique()))
        st.dataframe(manifest[manifest["provenance"].isin(kinds)], hide_index=True,
                     width="stretch")
        widgets.download_csv(manifest, "recording_manifest.csv", "Download manifest CSV",
                             key="dl_manifest")

with tab_cov:
    matrix = registry.coverage_matrix().rename(columns=state.song_label)
    widgets.plot(plot_coverage_matrix(matrix, synthetic=state.uses_synthetic_data()),
                 key="cov_ds")
    missing = registry.missing_recordings()
    st.caption(f"{len(missing)} participant × song combinations have no analysable recording.")
    if missing:
        with st.expander("Missing combinations"):
            st.dataframe(pd.DataFrame(missing, columns=["participant_id", "song_id"])
                         .assign(song=lambda d: d["song_id"].map(state.song_label)),
                         hide_index=True, width="stretch")

with tab_songs:
    attrs = state.song_attributes()
    songs = pd.DataFrame([s.model_dump(exclude={"source_metadata"})
                          for s in registry.state.songs.values()])
    st.markdown("#### Registered songs")
    st.dataframe(songs, hide_index=True, width="stretch")
    if attrs.empty:
        st.info("No `data/metadata/song_attributes.csv` found.")
    else:
        report = match_report(attrs, list(registry.state.songs))
        widgets.metric_row([("Matched to attributes", len(report["matched"])),
                            ("Recordings without attributes",
                             len(report["recordings_without_attributes"])),
                            ("Attribute rows without recordings",
                             len(report["attributes_without_recordings"]))])
        st.markdown("#### Song attributes (song-level, provenance unverified)")
        st.caption("Proxy and acoustic values appear audio-derived; survey values appear to be "
                   "song-level aggregates. They are not per-participant ratings.")
        st.dataframe(attrs.reset_index(), hide_index=True, width="stretch")
