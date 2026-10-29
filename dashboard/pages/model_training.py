"""Page 9 — Machine learning: training with subject-independent validation."""

import sys
from pathlib import Path

_DASHBOARD = str(Path(__file__).resolve().parents[1])
if _DASHBOARD not in sys.path:
    sys.path.insert(0, _DASHBOARD)

import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from components import state, widgets  # noqa: E402
from neuromusic.experiments.runner import ExperimentError, ExperimentRunner  # noqa: E402
from neuromusic.features.extractor import feature_columns, feature_families  # noqa: E402
from neuromusic.modeling.classifiers import CLASSIFIERS, available_classifiers  # noqa: E402
from neuromusic.modeling.datasets import DatasetError  # noqa: E402

ws = state.page_header(
    "Model training",
    "Predict the song from EEG features with leave-one-subject-out cross-validation. All "
    "scaling, imputation and tuning are fitted on training participants only.")
trials = state.require_features("trial")
if trials is None:
    st.stop()

valid = trials[trials["trial_valid"]]
per_song = valid.groupby("song_id")["participant_id"].nunique()
widgets.metric_row([("Valid trials", len(valid)),
                    ("Participants", valid["participant_id"].nunique()),
                    ("Songs", valid["song_id"].nunique()),
                    ("Min participants per song", int(per_song.min()) if len(per_song) else 0)])
if len(per_song) and per_song.min() < 3:
    st.warning("Some songs have fewer than three participants. Held-out estimates for those "
               "classes will be extremely unstable; consider restricting the analysis to songs "
               "with adequate coverage before interpreting results.")

cfg = ws.experiments.classification
families = feature_families(feature_columns(trials))
with st.form("clf"):
    models = st.multiselect(
        "Models", available_classifiers(),
        default=[m for m in cfg.models if m in available_classifiers()],
        format_func=lambda m: f"{m} — {CLASSIFIERS[m].description}")
    prefixes = st.multiselect("Feature families", families,
                              default=[p for p in cfg.feature_prefixes if p in families])
    c1, c2, c3, c4 = st.columns(4)
    level = c1.selectbox("Observation level", ["trial", "window"],
                         index=["trial", "window"].index(cfg.observation_level),
                         help="Window level keeps every window of the held-out participant out "
                              "of training; windows are not independent participants.")
    tune = c2.checkbox("Tune hyperparameters (inner GroupKFold)", cfg.tune_hyperparameters)
    n_perm = c3.number_input("Permutations (0 = off)", 0, 1000, cfg.n_permutations,
                             help="Labels are permuted within participants; the full pipeline "
                                  "is rerun each time.")
    repeats = c4.number_input("Importance repeats", 0, 100, cfg.permutation_importance_repeats)
    seed = st.number_input("Random seed", 0, 2**31 - 1, cfg.random_seed)
    run = st.form_submit_button("Train and evaluate", type="primary")

if run:
    new_cfg = cfg.model_copy(update={
        "models": models, "feature_prefixes": prefixes, "observation_level": level,
        "tune_hyperparameters": tune, "n_permutations": int(n_perm),
        "permutation_importance_repeats": int(repeats), "random_seed": int(seed)})
    bar = st.progress(0.0, text="Starting…")
    try:
        record = ExperimentRunner(ws).run_classification(
            new_cfg, progress=lambda i, n, msg: bar.progress(min(i / max(n, 1), 1.0), text=msg))
    except (ExperimentError, DatasetError, ValueError) as exc:
        st.error(f"Experiment could not run: {exc}")
    else:
        bar.progress(1.0, text="Done")
        st.success(f"Saved {record.experiment_id}. Inspect it on **Model evaluation**.")
        st.dataframe(pd.DataFrame(record.metrics).T, width="stretch")

st.subheader("Saved classification experiments")
records = state.experiment_store().list_records("classification")
if not records:
    st.info("No classification experiments yet.")
else:
    st.dataframe(pd.DataFrame([{
        "experiment": r.experiment_id, "status": r.status.value,
        "level": r.experiment_config.get("observation_level"),
        "features": ", ".join(r.experiment_config.get("feature_prefixes", [])),
        "tuned": r.experiment_config.get("tune_hyperparameters"),
        **{f"bal_acc[{m}]": round(v.get("balanced_accuracy", float("nan")), 3)
           for m, v in r.metrics.items() if isinstance(v, dict)},
        "synthetic": r.is_synthetic_data,
    } for r in records]), hide_index=True, width="stretch")
