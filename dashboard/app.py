"""NeuroMusic Research Lab dashboard entry point: ``streamlit run dashboard/app.py``."""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

DASHBOARD_DIR = Path(__file__).resolve().parent
if str(DASHBOARD_DIR) not in sys.path:
    sys.path.insert(0, str(DASHBOARD_DIR))

from components import state  # noqa: E402

st.set_page_config(page_title="NeuroMusic Research Lab", page_icon=":material/neurology:",
                   layout="wide")


def page(file: str, title: str, icon: str, default: bool = False) -> st.Page:
    return st.Page(str(DASHBOARD_DIR / "pages" / file), title=title, icon=icon, default=default)


navigation = st.navigation({
    "Study": [
        page("overview.py", "Research overview", ":material/dashboard:", default=True),
        page("datasets.py", "Dataset manager", ":material/folder_open:"),
    ],
    "Signals": [
        page("eeg_explorer.py", "EEG explorer", ":material/monitor_heart:"),
        page("signal_quality.py", "Signal quality", ":material/verified:"),
        page("preprocessing.py", "Preprocessing", ":material/tune:"),
        page("spectral_analysis.py", "Spectral analysis", ":material/graphic_eq:"),
    ],
    "Analysis": [
        page("song_comparison.py", "Song comparison", ":material/compare_arrows:"),
        page("song_similarity.py", "Song similarity", ":material/library_music:"),
        page("clustering.py", "Clustering", ":material/bubble_chart:"),
        page("synchronization.py", "Neural synchronization", ":material/sync:"),
    ],
    "Machine learning": [
        page("model_training.py", "Model training", ":material/model_training:"),
        page("model_evaluation.py", "Model evaluation", ":material/insights:"),
    ],
    "Output": [
        page("reports.py", "Research reports", ":material/description:"),
    ],
})
state.render_sidebar()
navigation.run()
