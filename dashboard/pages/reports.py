"""Page 12 — Research reports."""

import sys
from pathlib import Path

_DASHBOARD = str(Path(__file__).resolve().parents[1])
if _DASHBOARD not in sys.path:
    sys.path.insert(0, _DASHBOARD)

import streamlit as st  # noqa: E402

from components import state  # noqa: E402

MIME = {".md": "text/markdown", ".html": "text/html", ".csv": "text/csv",
        ".json": "application/json", ".png": "image/png"}

ws = state.page_header(
    "Research reports",
    "Generate a report with dataset summary, methodology, preprocessing configuration, "
    "statistics, model results, limitations and reproducibility information.")

try:
    from neuromusic.reporting.generator import generate_report
except ImportError as exc:
    st.error(f"Report generation is unavailable: {exc}")
    st.stop()

records = state.experiment_store().list_records()
completed = [r for r in records if r.status.value == "completed"]
chosen = st.multiselect(
    "Experiments to include (empty = latest completed of each type)", completed,
    format_func=lambda r: f"{r.experiment_id}" + (" · SYNTHETIC" if r.is_synthetic_data else ""))
formats = st.multiselect("Formats", ["md", "html", "csv", "json", "png"],
                         default=["md", "html", "csv", "json", "png"])

if st.button("Generate report", type="primary", icon=":material/description:"):
    with st.spinner("Generating report…"):
        try:
            out = generate_report(ws, [r.experiment_id for r in chosen] or None,
                                  tuple(formats))
        except (ValueError, FileNotFoundError, RuntimeError) as exc:
            st.error(f"Report generation failed: {exc}")
        else:
            st.session_state["report_dir"] = str(out)
            st.success(f"Report written to `{out}`.")

report_dir = st.session_state.get("report_dir")
if report_dir and Path(report_dir).is_dir():
    root = Path(report_dir)
    markdown = root / "report.md"
    if markdown.exists():
        # Rendered as Markdown (no script execution) rather than embedding the HTML report.
        with st.expander("Preview", expanded=True):
            st.markdown(markdown.read_text(encoding="utf-8"))
    st.markdown("#### Files")
    for i, path in enumerate(sorted(p for p in root.rglob("*") if p.is_file())):
        st.download_button(str(path.relative_to(root)), path.read_bytes(), file_name=path.name,
                           mime=MIME.get(path.suffix, "application/octet-stream"),
                           key=f"dl_{i}", icon=":material/download:")
elif not records:
    st.info("No experiments yet; a report will contain the dataset summary only.")
