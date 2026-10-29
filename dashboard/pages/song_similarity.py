"""Song similarity: songs compared with each other, then with EEG (RSA)."""

import sys
from pathlib import Path

_DASHBOARD = str(Path(__file__).resolve().parents[1])
if _DASHBOARD not in sys.path:
    sys.path.insert(0, _DASHBOARD)

import streamlit as st  # noqa: E402

from components import state, widgets  # noqa: E402
from neuromusic.audio.decoding import ffmpeg_available  # noqa: E402
from neuromusic.experiments.runner import ExperimentError, ExperimentRunner  # noqa: E402
from neuromusic.visualization.similarity_plots import (  # noqa: E402
    plot_pair_scatter,
    plot_rdm,
    plot_rsa_bars,
)

ws = state.page_header(
    "Song similarity",
    "Songs compared with each other from their audio (and the song attribute table), then "
    "related to EEG with representational similarity analysis.")
st.caption("Audio features are computed from the exact stimulus files over the segment "
           "participants heard (recordings start at song onset). Exploratory analysis.")

if not ffmpeg_available():
    st.error("ffmpeg is not on PATH; stimulus audio cannot be decoded.")
elif state.require_features() is not None and st.button("Run song similarity analysis",
                                                        type="primary"):
    with st.spinner("Decoding audio, computing features and permutation tests…"):
        try:
            rec = ExperimentRunner(ws).run_song_similarity()
            st.success(f"Saved {rec.experiment_id}.")
        except (ExperimentError, FileNotFoundError) as exc:
            st.error(str(exc))

record = widgets.experiment_selector("song_similarity", key="songsim_exp")
if record is not None:
    widgets.record_summary(record)
    eid = record.experiment_id
    titles = state.song_titles()
    syn = record.is_synthetic_data

    def rdm(name: str):
        table = widgets.load_table(eid, name)
        return table.set_index(table.columns[0]) if not table.empty else table

    tab1, tab2, tab3 = st.tabs(["Songs vs songs", "Songs vs EEG", "Attribute table check"])
    with tab1:
        options = {"Acoustic (loudness, spectrum, rhythm)": "song_rdm_audio_acoustic",
                   "Timbre (MFCC)": "song_rdm_audio_timbre",
                   "Attribute table: proxies": "song_rdm_table_proxies",
                   "Attribute table: survey": "song_rdm_table_survey",
                   "EEG (song-mean features)": "song_rdm_eeg"}
        choice = st.selectbox("Description", list(options))
        matrix = rdm(options[choice])
        if matrix.empty:
            st.info("Not available for this experiment.")
        else:
            widgets.plot(plot_rdm(matrix, f"Song dissimilarity: {choice}", titles,
                                  synthetic=syn), key="rdm")
        st.subheader("Nearest songs (acoustic)")
        st.dataframe(widgets.load_table(eid, "song_neighbours_audio"), hide_index=True,
                     width="stretch")
        st.subheader("Agreement between song descriptions")
        st.dataframe(widgets.load_table(eid, "song_description_agreement"), hide_index=True,
                     width="stretch")
        with st.expander("Audio features (heard segment)"):
            st.dataframe(widgets.load_table(eid, "song_audio_features"), hide_index=True,
                         width="stretch")
    with tab2:
        rsa = widgets.load_table(eid, "rsa")
        if not rsa.empty:
            widgets.plot(plot_rsa_bars(rsa, synthetic=syn), key="rsa")
            st.dataframe(rsa, hide_index=True, width="stretch")
        pairs = widgets.load_table(eid, "recording_pairs")
        if not pairs.empty:
            model = st.selectbox("Model distance", ["audio_timbre", "audio_acoustic",
                                                    "duration_difference"])
            widgets.plot(plot_pair_scatter(pairs, model, synthetic=syn), key="pairs")
        st.subheader("Audio feature vs EEG feature (exploratory, FDR)")
        st.dataframe(widgets.load_table(eid, "audio_eeg_associations").head(30),
                     hide_index=True, width="stretch")
    with tab3:
        st.caption("Each attribute-table row compared with the audio file of the same song "
                   "(whole song). 'matches other song' means the row's brightness and tempo "
                   "match a different song's audio almost exactly.")
        st.dataframe(widgets.load_table(eid, "attribute_consistency"), hide_index=True,
                     width="stretch")
