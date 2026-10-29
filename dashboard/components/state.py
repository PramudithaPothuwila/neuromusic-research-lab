"""Shared dashboard state: workspace, cached loaders and page chrome."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from neuromusic import services
from neuromusic.experiments.persistence import ExperimentStore
from neuromusic.features.store import FeatureStore
from neuromusic.ingestion.csv_reader import RawEEG, read_eeg_csv
from neuromusic.ingestion.song_attributes import load_song_attributes
from neuromusic.preprocessing.pipeline import PreprocessingResult
from neuromusic.workspace import Workspace, find_root

DASHBOARD_DIR = Path(__file__).resolve().parents[1]


@st.cache_resource(show_spinner=False)
def get_workspace() -> Workspace:
    return Workspace(find_root(DASHBOARD_DIR.parent))


def refresh() -> None:
    """Reload configs/registry from disk and drop cached data (after any mutation)."""
    get_workspace().reload_configs()
    st.cache_data.clear()


def experiment_store() -> ExperimentStore:
    return ExperimentStore(get_workspace().experiments_dir)


# ---------------------------------------------------------------- cached loaders


def _features_path(ws: Workspace, level: str) -> Path:
    return FeatureStore(ws.processed_dir).set_dir(ws.preprocessing_hash, ws.feature_hash) / (
        f"{level}_features.parquet")


@st.cache_data(show_spinner=False)
def _read_parquet(path: str, mtime: float) -> pd.DataFrame:
    del mtime  # part of the cache key only
    return pd.read_parquet(path)


def load_features(level: str = "trial") -> pd.DataFrame | None:
    """Feature table for the current configuration, or None if not built yet."""
    ws = get_workspace()
    path = _features_path(ws, level)
    if not path.exists():
        return None
    return _read_parquet(str(path), path.stat().st_mtime)


@st.cache_data(show_spinner="Reading recording…")
def _read_raw(path: str, prefix: str, mtime: float) -> RawEEG:
    del mtime
    return read_eeg_csv(Path(path), prefix)


def load_raw(recording_id: str) -> RawEEG:
    ws = get_workspace()
    path = ws.registry.resolve_path(ws.registry.recordings[recording_id])
    return _read_raw(str(path), ws.dataset.channel_prefix, path.stat().st_mtime)


@st.cache_data(show_spinner="Loading preprocessed signals…")
def _load_preprocessed(recording_id: str, prep_hash: str, fingerprint: str) -> PreprocessingResult:
    del prep_hash, fingerprint
    return services.load_preprocessed(get_workspace(), recording_id)


def load_preprocessed(recording_id: str) -> PreprocessingResult:
    ws = get_workspace()
    return _load_preprocessed(recording_id, ws.preprocessing_hash, ws.registry.fingerprint())


@st.cache_data(show_spinner=False)
def _quality_tables(prep_hash: str, n_processed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    del prep_hash, n_processed
    return services.quality_tables(get_workspace())


def quality_tables() -> tuple[pd.DataFrame, pd.DataFrame]:
    ws = get_workspace()
    n = len(services.preprocessed_store(ws).processed_ids())
    return _quality_tables(ws.preprocessing_hash, n)


def song_titles() -> dict[str, str]:
    ws = get_workspace()
    titles = {sid: s.song_name for sid, s in ws.registry.state.songs.items()}
    titles.update(services.song_titles(ws))
    return titles


def song_label(song_id: str | None) -> str:
    if song_id is None:
        return "—"
    return song_titles().get(song_id, song_id)


def song_attributes() -> pd.DataFrame:
    return load_song_attributes(get_workspace().metadata_dir)


# ------------------------------------------------------------------- chrome


def uses_synthetic_data() -> bool:
    return any(r.is_synthetic for r in get_workspace().registry.analysable_recordings())


def page_header(title: str, description: str | None = None) -> Workspace:
    """Title, description, synthetic-data banner and unit notice; returns the workspace."""
    ws = get_workspace()
    st.title(title)
    if description:
        st.caption(description)
    if uses_synthetic_data():
        st.warning("**Synthetic data is included in the analysable dataset.** Results shown "
                   "here are not study findings.", icon=":material/science:")
    return ws


def units_notice() -> None:
    ws = get_workspace()
    if not ws.units_validated:
        st.caption("Signal units, scaling and acquisition reference are not validated. "
                   "Amplitudes and powers are shown in arbitrary units (a.u.).")


def require_features(level: str = "trial") -> pd.DataFrame | None:
    """Return features or render guidance when they have not been built."""
    frame = load_features(level)
    if frame is None:
        st.info("Features have not been built for the current preprocessing and feature "
                "configuration. Run the pipeline on the **Preprocessing** page "
                "(or `neuromusic features`).", icon=":material/info:")
    return frame


def render_sidebar() -> None:
    ws = get_workspace()
    with st.sidebar:
        st.markdown(f"**{ws.application.project_name}**")
        st.caption(f"Workspace: `{ws.root.name}`")
        st.caption(f"Preprocessing `{ws.preprocessing_hash}` · features `{ws.feature_hash}`")
        if st.button("Reload from disk", icon=":material/refresh:", width="stretch"):
            refresh()
            st.rerun()
