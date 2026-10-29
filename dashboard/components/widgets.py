"""Reusable selectors and display helpers."""

from __future__ import annotations

import json
from typing import Any

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from components import state
from neuromusic.domain.models import Recording
from neuromusic.experiments.registry import ExperimentRecord


def recording_selector(key: str, *, analysable_only: bool = True,
                       recordings: list[Recording] | None = None) -> Recording | None:
    """Participant -> song -> recording cascade."""
    ws = state.get_workspace()
    recs = recordings if recordings is not None else (
        ws.registry.analysable_recordings() if analysable_only
        else list(ws.registry.recordings.values()))
    if not recs:
        st.info("No recordings available.")
        return None
    participants = sorted({r.participant_id or "unassigned" for r in recs})
    c1, c2, c3 = st.columns(3)
    pid = c1.selectbox("Participant", participants, key=f"{key}_participant")
    by_p = [r for r in recs if (r.participant_id or "unassigned") == pid]
    songs = sorted({r.song_id or "unassigned" for r in by_p})
    sid = c2.selectbox("Song", songs, key=f"{key}_song", format_func=state.song_label)
    options = [r for r in by_p if (r.song_id or "unassigned") == sid]
    rec = c3.selectbox("Recording", options, key=f"{key}_recording",
                       format_func=lambda r: f"{r.trial_id} · {r.source_filename}")
    return rec


def plot(fig: go.Figure, key: str | None = None) -> None:
    st.plotly_chart(fig, width="stretch", key=key,
                    config={"displaylogo": False,
                            "toImageButtonOptions": {"format": "png", "scale": 3}})


def metric_row(items: list[tuple[str, Any]]) -> None:
    cols = st.columns(len(items))
    for col, (label, value) in zip(cols, items, strict=True):
        col.metric(label, value)


def experiment_selector(experiment_type: str, key: str) -> ExperimentRecord | None:
    records = state.experiment_store().list_records(experiment_type)
    if not records:
        st.info(f"No {experiment_type} experiments have been run yet.")
        return None

    def label(r: ExperimentRecord) -> str:
        flag = " · SYNTHETIC" if r.is_synthetic_data else ""
        return f"{r.experiment_id} [{r.status.value}]{flag}"

    return st.selectbox("Experiment", records, format_func=label, key=key)


def record_summary(record: ExperimentRecord) -> None:
    if record.is_synthetic_data:
        st.warning("This experiment used synthetic data.", icon=":material/science:")
    if record.error:
        st.error(f"Experiment failed: {record.error}")
    with st.expander("Provenance and reproducibility"):
        st.json({
            "experiment_id": record.experiment_id,
            "created_at": record.created_at.isoformat(timespec="seconds"),
            "dataset_fingerprint": record.dataset_fingerprint,
            "n_participants": record.n_participants,
            "songs": record.song_ids,
            "n_recordings": len(record.recording_ids),
            "preprocessing_hash": record.preprocessing_hash,
            "feature_hash": record.feature_hash,
            "cv_strategy": record.cv_strategy,
            "random_seed": record.random_seed,
            "software_version": record.software_version,
            "environment": record.environment,
            "experiment_config": record.experiment_config,
        }, expanded=False)
    for note in record.notes:
        st.caption(f"• {note}")


def load_table(experiment_id: str, name: str) -> pd.DataFrame:
    store = state.experiment_store()
    if not store.has(experiment_id, name):
        return pd.DataFrame()
    return store.load_table(experiment_id, name)


def load_json(experiment_id: str, name: str) -> Any:
    store = state.experiment_store()
    if not store.has(experiment_id, name):
        return None
    return store.load_json(experiment_id, name)


def parse_json_cell(value: Any) -> Any:
    if isinstance(value, str) and value[:1] in "[{":
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return value
    return value


def download_csv(frame: pd.DataFrame, filename: str, label: str, key: str) -> None:
    st.download_button(label, frame.to_csv(index=False).encode("utf-8"), file_name=filename,
                       mime="text/csv", key=key, icon=":material/download:")
