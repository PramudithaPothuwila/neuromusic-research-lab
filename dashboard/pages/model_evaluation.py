"""Page 10 — Model evaluation."""

import sys
from pathlib import Path

_DASHBOARD = str(Path(__file__).resolve().parents[1])
if _DASHBOARD not in sys.path:
    sys.path.insert(0, _DASHBOARD)

import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from components import state, widgets  # noqa: E402
from neuromusic.visualization.model_plots import (  # noqa: E402
    plot_confusion_matrix,
    plot_importance,
    plot_model_comparison,
    plot_participant_accuracy,
    plot_permutation_null,
    plot_train_vs_test,
)

ws = state.page_header("Model evaluation",
                       "Held-out (leave-one-subject-out) performance compared with the dummy "
                       "baseline. Training scores are shown separately and are optimistic.")
record = widgets.experiment_selector("classification", key="eval_exp")
if record is None:
    st.stop()
widgets.record_summary(record)
eid = record.experiment_id
metrics = widgets.load_json(eid, "metrics")
if not metrics:
    st.info("This experiment has no saved metrics.")
    st.stop()
dataset = widgets.load_json(eid, "dataset") or {}
titles = state.song_titles()

widgets.metric_row([("Observations", dataset.get("n_observations", "—")),
                    ("Level", dataset.get("level", "—")),
                    ("Participants", dataset.get("n_participants", "—")),
                    ("Classes", len(dataset.get("classes", [])))])
if dataset.get("class_support"):
    st.caption("Class support: " + ", ".join(
        f"{titles.get(k, k)}: {v}" for k, v in dataset["class_support"].items()))

table = pd.DataFrame([{
    "model": name,
    "held-out accuracy": m["accuracy"],
    "held-out balanced accuracy": m["balanced_accuracy"],
    "balanced accuracy 95% CI": f"[{m['balanced_accuracy_ci'][0]:.3f}, "
                                f"{m['balanced_accuracy_ci'][1]:.3f}]",
    "held-out macro-F1": m["macro_f1"],
    "training accuracy (in-sample)": m.get("mean_train_accuracy"),
    "permutation p": m.get("permutation_test", {}).get("p_value"),
} for name, m in metrics.items()])
st.subheader("Held-out performance")
st.dataframe(table, hide_index=True, width="stretch")
widgets.plot(plot_model_comparison(metrics, synthetic=record.is_synthetic_data), key="cmp")

folds = widgets.load_table(eid, "folds")
if not folds.empty:
    widgets.plot(plot_train_vs_test(folds), key="tt")

model = st.selectbox("Model", list(metrics), index=min(1, len(metrics) - 1))
m = metrics[model]
classes = [titles.get(c, c) for c in m["classes"]]
c1, c2 = st.columns(2)
with c1:
    normalize = st.toggle("Normalise rows", value=True)
    widgets.plot(plot_confusion_matrix(m["confusion_matrix"], classes, normalize=normalize,
                                       synthetic=record.is_synthetic_data), key="cm")
with c2:
    widgets.plot(plot_participant_accuracy(m["participant_accuracy"], model), key="pa")
per_class = pd.DataFrame(m["per_class"])
per_class["class"] = per_class["class"].map(lambda c: titles.get(c, c))
st.dataframe(per_class, hide_index=True, width="stretch")
if "trial_majority_vote" in m:
    st.caption(f"Trial-level majority vote over windows: balanced accuracy "
               f"{m['trial_majority_vote']['balanced_accuracy']:.3f}.")

perm = m.get("permutation_test")
if perm:
    widgets.plot(plot_permutation_null(perm["null_distribution"],
                                       perm["observed_balanced_accuracy"], perm["p_value"]),
                 key="perm")
importance = widgets.load_table(eid, "importance")
if not importance.empty and "model" in importance.columns:
    widgets.plot(plot_importance(importance[importance["model"] == model], model=f"— {model}"),
                 key="imp")
predictions = widgets.load_table(eid, "predictions")
if not predictions.empty:
    with st.expander("Held-out predictions"):
        st.dataframe(predictions[predictions["model"] == model], hide_index=True,
                     width="stretch")
st.caption(f"{len(m['classes'])} songs were studied: performance does not demonstrate "
           "generalisation to new songs, and song order or session effects may contribute.")
