# -*- coding: utf-8 -*-
from __future__ import annotations

import csv
import logging
from typing import Dict, Iterable

from app.utils.config import APP_DIR

LOGGER = logging.getLogger(__name__)

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
    "test_name",
    "condition",
    "expected_gesture",
    "detected_gesture",
    "attempt_index",
    "result",
    "time_to_recognize_ms",
]

GESTURE_SUMMARY_FIELDS = [
    "session_id",
    "datetime",
    "test_name",
    "condition",
    "gesture",
    "attempts",
    "hits",
    "wrongs",
    "misses",
    "accuracy",
    "avg_time_to_recognize_ms",
]

SCENARIO_EVAL_FIELDS = [
    "session_id",
    "datetime",
    "scenario_name",
    "duration_ms",
    "success",
    "error_count",
    "critical_error_count",
    "comment",
]


def append_csv_row(file_name: str, field_names: Iterable[str], row: Dict[str, object]) -> None:
    """Persist a CSV row ensuring a stable header order."""
    fields = list(field_names)
    path = APP_DIR.parent / file_name
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    try:
        with path.open("a", encoding="utf-8", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=fields)
            if not exists:
                writer.writeheader()
            writer.writerow({key: row.get(key, "") for key in fields})
    except (OSError, csv.Error):
        LOGGER.exception("Failed to append CSV row to %s", path)
        raise


def append_metrics_row(row: Dict[str, object]) -> None:
    """Persist a single metrics row to ``metrics.csv``."""
    append_csv_row("metrics.csv", METRIC_FIELDS, row)


def append_eval_row(row: Dict[str, object]) -> None:
    """Persist a single gesture-attempt eval row to ``gesture_eval.csv``."""
    append_csv_row("gesture_eval.csv", EVAL_FIELDS, row)


def append_gesture_summary_row(row: Dict[str, object]) -> None:
    """Persist a per-gesture eval summary row."""
    append_csv_row("gesture_eval_summary.csv", GESTURE_SUMMARY_FIELDS, row)


def append_scenario_eval_row(row: Dict[str, object]) -> None:
    """Persist a single applied-scenario eval row."""
    append_csv_row("scenario_eval.csv", SCENARIO_EVAL_FIELDS, row)


def write_eval_report(session_id: str, report_text: str) -> str:
    """Write evaluation report to a text file and return its path."""
    path = APP_DIR.parent / f"gesture_eval_report_{session_id}.txt"
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("w", encoding="utf-8") as file:
            file.write(report_text)
    except OSError:
        LOGGER.exception("Failed to write evaluation report to %s", path)
        raise
    return str(path)

