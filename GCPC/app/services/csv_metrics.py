# -*- coding: utf-8 -*-
from __future__ import annotations

import csv
from typing import Dict, Iterable

from app.utils.config import APP_DIR

METRIC_FIELDS = [
    "session_id",
    "backend",
    "providers",
    "FPS_avg",
    "e2e_p50",
    "e2e_p95",
    "interaction",
    "scenario",
    "task_time_ms",
    "task_success",
    "false_activations",
]

EVAL_FIELDS = [
    "session_id",
    "datetime",
    "condition",
    "hand",
    "gesture_target",
    "reps",
    "hits",
    "wrongs",
    "misses",
    "accuracy",
    "false_rate",
    "avg_time_to_hit_ms",
]


def append_csv_row(file_name: str, field_names: Iterable[str], row: Dict[str, object]) -> None:
    """Persist a CSV row ensuring a stable header order."""
    fields = list(field_names)
    path = APP_DIR.parent / file_name
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        if not exists:
            writer.writeheader()
        writer.writerow({key: row.get(key, "") for key in fields})


def append_metrics_row(row: Dict[str, object]) -> None:
    """Persist a single metrics row to ``metrics.csv``."""
    append_csv_row("metrics.csv", METRIC_FIELDS, row)


def append_eval_row(row: Dict[str, object]) -> None:
    """Persist a single eval row to ``gesture_eval.csv``."""
    append_csv_row("gesture_eval.csv", EVAL_FIELDS, row)


def write_eval_report(session_id: str, report_text: str) -> str:
    """Write evaluation report to a text file and return its path."""
    path = APP_DIR.parent / f"gesture_eval_report_{session_id}.txt"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        file.write(report_text)
    return str(path)

