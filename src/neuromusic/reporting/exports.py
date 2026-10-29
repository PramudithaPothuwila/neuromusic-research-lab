"""Table and payload export helpers for reports."""

from __future__ import annotations

import html
import json
import math
from pathlib import Path
from typing import Any

import pandas as pd


def _fmt(value: Any, digits: int = 3) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        if math.isnan(value):
            return ""
        if value != 0 and (abs(value) < 10 ** -digits or abs(value) >= 1e5):
            return f"{value:.{digits}g}"
        return f"{value:.{digits}f}"
    if isinstance(value, list | tuple):
        return ", ".join(_fmt(v, digits) for v in value)
    return str(value)


def markdown_table(frame: pd.DataFrame, digits: int = 3, max_rows: int = 30) -> str:
    if frame is None or frame.empty:
        return "_No data._"
    shown = frame.head(max_rows)
    header = "| " + " | ".join(str(c) for c in shown.columns) + " |"
    sep = "| " + " | ".join("---" for _ in shown.columns) + " |"
    rows = ["| " + " | ".join(_fmt(v, digits).replace("|", "\\|") for v in row) + " |"
            for row in shown.itertuples(index=False)]
    more = [f"\n_{len(frame) - max_rows} more row(s) in the CSV export._"] \
        if len(frame) > max_rows else []
    return "\n".join([header, sep, *rows, *more])


def html_table(frame: pd.DataFrame, digits: int = 3, max_rows: int = 30) -> str:
    if frame is None or frame.empty:
        return "<p><em>No data.</em></p>"
    shown = frame.head(max_rows)
    head = "".join(f"<th>{html.escape(str(c))}</th>" for c in shown.columns)
    body = "".join(
        "<tr>" + "".join(f"<td>{html.escape(_fmt(v, digits))}</td>" for v in row) + "</tr>"
        for row in shown.itertuples(index=False))
    more = (f"<p class='more'>{len(frame) - max_rows} more row(s) in the CSV export.</p>"
            if len(frame) > max_rows else "")
    return f"<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>{more}"


def save_csv(frame: pd.DataFrame, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    return path


def save_json(payload: Any, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    return path
