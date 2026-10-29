"""Page 1 — Research overview."""

import sys
from pathlib import Path

_DASHBOARD = str(Path(__file__).resolve().parents[1])
if _DASHBOARD not in sys.path:
    sys.path.insert(0, _DASHBOARD)

import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from components import state, widgets  # noqa: E402
from neuromusic import services  # noqa: E402
from neuromusic.analysis.descriptive import channel_quality_summary, dataset_overview  # noqa: E402
from neuromusic.visualization.statistical_plots import (  # noqa: E402
    plot_coverage_matrix,
    plot_provenance_summary,
)

PROVENANCE_HELP = {
    "original": "earliest copy of an independent signal; kept",
    "derived_copy": "near-duplicate of another file's signal; excluded (not an independent "
                    "participant)",
    "identical_copy": "byte-identical to another file; not registered separately",
    "label_conflict": "one signal carries different song labels; all copies excluded because "
                      "the true stimulus is unknown",
    "synthetic": "generated data; excluded from analysis",
}

ws = state.page_header(
    "Research overview",
    "Music-evoked EEG: spectral analysis, unsupervised structure and cross-participant "
    "generalisation.")

recs = ws.registry.recordings_frame()
overview = dataset_overview(recs)
widgets.metric_row([
    ("Participants", overview["n_participants"]),
    ("Songs", overview["n_songs"]),
    ("Analysable recordings", f"{overview['n_analysable']} / {overview['n_recordings']}"),
    ("Total analysable EEG", f"{overview['total_duration_s'] / 60:.1f} min"),
])
st.caption("Counts refer to analysable recordings: validated, with participant and song "
           "metadata, and not excluded.")

if recs.empty:
    st.info("No recordings registered. Copy CSV files into `data/raw/eeg/` and use the "
            "**Dataset manager** to scan them.")
    st.stop()

left, right = st.columns(2)
with left:
    st.subheader("File provenance")
    manifest = ws.registry.manifest_frame(services.song_titles(ws))
    widgets.plot(plot_provenance_summary(manifest), key="prov")
    counts = manifest["provenance"].value_counts()
    for name, help_text in PROVENANCE_HELP.items():
        if name in counts:
            st.caption(f"**{name}** ({counts[name]}): {help_text}")
with right:
    st.subheader("Validation")
    st.dataframe(pd.Series(ws.registry.validation_summary(), name="recordings")
                 .rename_axis("status").reset_index(), hide_index=True, width="stretch")
    analysable = ws.registry.analysable_recordings()
    channels = sorted({c for r in analysable for c in r.channel_names})
    st.markdown(f"**Channel coverage:** {len(channels)} channels — {', '.join(channels)}")
    st.markdown(f"**Sampling rates (Hz):** {overview.get('sampling_rates', [])}")
    state.units_notice()

st.subheader("Coverage")
matrix = ws.registry.coverage_matrix().rename(columns=state.song_label)
widgets.plot(plot_coverage_matrix(matrix, synthetic=state.uses_synthetic_data()), key="cov")
multi = int((matrix.gt(0).sum(axis=1) >= 2).sum()) if not matrix.empty else 0
st.caption(f"{multi} participant(s) have analysable recordings for two or more songs; "
           "within-participant song comparisons are only possible for them.")

st.subheader("Data quality")
quality, flags = state.quality_tables()
if quality.empty:
    st.info("No recordings preprocessed yet.")
else:
    st.dataframe(channel_quality_summary(quality), hide_index=True, width="stretch")

st.subheader("Pipeline status")
features = state.load_features("trial")
store = state.experiment_store()
records = store.list_records()
status = pd.DataFrame([
    {"step": "Recordings validated", "status": f"{len(recs)} registered"},
    {"step": "Preprocessing (current config)",
     "status": f"{len(services.preprocessed_store(ws).processed_ids())} recordings processed"},
    {"step": "Features (current config)",
     "status": "built" + ("" if services.features_current(ws) else " — dataset changed since")
     if features is not None else "not built"},
    *[{"step": f"{t} experiments",
       "status": str(sum(r.experiment_type.value == t for r in records))}
      for t in ("statistics", "clustering", "classification", "regression", "synchronization")],
])
st.dataframe(status, hide_index=True, width="stretch")

st.subheader("Recent results")
if not records:
    st.info("No experiments yet.")
else:
    st.dataframe(pd.DataFrame([{
        "experiment": r.experiment_id, "type": r.experiment_type.value, "status": r.status.value,
        "participants": r.n_participants, "synthetic": r.is_synthetic_data,
        "created": r.created_at.strftime("%Y-%m-%d %H:%M"),
    } for r in records[:8]]), hide_index=True, width="stretch")
