# -*- coding: utf-8 -*-
from __future__ import annotations

import statistics
import time
import uuid
from datetime import datetime
from typing import Any

from PySide6 import QtCore, QtWidgets

from app.services.csv_metrics import (
    append_eval_row,
    append_gesture_summary_row,
    append_scenario_eval_row,
    write_eval_report,
)
from app.utils.config import resolve_side


DEFAULT_GESTURES = [
    "PINCH",
    "PINCH_MIDDLE",
    "FIST",
    "THUMBS_UP",
    "OPEN_PALM",
    "SWIPE_RIGHT",
    "SWIPE_LEFT",
]


def _coerce_int(value: Any, fallback: int, minimum: int | None = None) -> int:
    try:
        result = int(float(value))
    except (TypeError, ValueError):
        result = fallback
    if minimum is not None:
        result = max(minimum, result)
    return result


def _coerce_float(value: Any, fallback: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return fallback


def _evaluation_cfg(cfg: dict[str, Any]) -> dict[str, Any]:
    return cfg.get("evaluation", {}) or {}


def _gesture_test_cfg(cfg: dict[str, Any]) -> dict[str, Any]:
    evaluation = _evaluation_cfg(cfg)
    current = evaluation.get("gesture_test", {}) or {}
    legacy = cfg.get("eval_single", {}) or {}
    merged = dict(legacy)
    merged.update(current)
    return merged


def _scenario_test_cfg(cfg: dict[str, Any]) -> dict[str, Any]:
    evaluation = _evaluation_cfg(cfg)
    return evaluation.get("scenario_test", {}) or {}


class GestureEvalSession:
    """Runs hidden developer gesture-recognition tests and writes analysis CSVs."""

    def __init__(self, cfg: dict[str, Any], hands: dict[str, str]):
        self.active = False
        self.session_id = ""
        self.session_datetime = ""
        self.gesture_idx = 0
        self.attempt_index = 1
        self.trial_start_ms = 0
        self.hits = 0
        self.wrongs = 0
        self.misses = 0
        self.hit_times: list[float] = []
        self.summary_rows: list[dict[str, str | float | int]] = []
        self.aborted = False
        self.notice_until_ms = 0
        self.notice_top = ""
        self.notice_sub = ""

        self.test_name = "gesture_test"
        self.condition = ""
        self.gestures = DEFAULT_GESTURES[:]
        self.reps_target = 10
        self.timeout_ms = 2500
        self.pass_accuracy = 0.9
        self.pass_wrong_max = 3
        self.hand_setting = "RIGHT"
        self.hand_label = "dominant"
        self.reload_config(cfg, hands)

    def reload_config(self, cfg: dict[str, Any], hands: dict[str, str]) -> None:
        eval_cfg = _gesture_test_cfg(cfg)
        gestures = eval_cfg.get("gestures", DEFAULT_GESTURES)
        self.gestures = [str(item).strip().upper() for item in gestures if str(item).strip()]
        if not self.gestures:
            self.gestures = DEFAULT_GESTURES[:]
        self.test_name = str(eval_cfg.get("test_name", "gesture_test") or "gesture_test")
        self.condition = str(eval_cfg.get("condition", "") or "")
        self.reps_target = _coerce_int(eval_cfg.get("reps"), 10, minimum=1)
        self.timeout_ms = _coerce_int(eval_cfg.get("timeout_ms"), 2500, minimum=100)
        self.pass_accuracy = _coerce_float(eval_cfg.get("pass_accuracy"), 0.9)
        self.pass_wrong_max = _coerce_int(eval_cfg.get("pass_wrong_max"), 3, minimum=0)
        self.reconfigure_hands(cfg, hands)

    def reconfigure_hands(self, cfg: dict[str, Any], hands: dict[str, str]) -> None:
        eval_cfg = _gesture_test_cfg(cfg)
        hand_setting = resolve_side(eval_cfg.get("hand", "dominant"), hands)
        if hand_setting not in ("RIGHT", "LEFT"):
            hand_setting = resolve_side("dominant", hands)
        self.hand_setting = hand_setting
        self.hand_label = (
            "dominant"
            if self.hand_setting == hands.get("dominant", "RIGHT")
            else "non_dominant"
        )

    def current_target(self) -> str | None:
        if 0 <= self.gesture_idx < len(self.gestures):
            return self.gestures[self.gesture_idx]
        return None

    def start(self, now_ms: int) -> bool:
        if self.active or not self.gestures:
            return False
        self.active = True
        self.aborted = False
        self.session_id = uuid.uuid4().hex
        self.session_datetime = datetime.now().isoformat(timespec="seconds")
        self.gesture_idx = 0
        self.summary_rows = []
        self._reset_gesture(now_ms)
        self.notice_until_ms = 0
        print(
            f"[EVAL] START gesture_test session={self.session_id} "
            f"test={self.test_name!r} condition={self.condition!r}"
        )
        return True

    def stop(self) -> bool:
        if not self.active:
            return False
        self.aborted = True
        if self._attempts_done() > 0:
            self._finalize_gesture()
        self._finalize_session()
        print(f"[EVAL] STOP gesture_test session={self.session_id}")
        return True

    def process(self, now_ms: int, detected_gesture: str) -> None:
        if not self.active:
            return
        target = self.current_target()
        if not target:
            return

        detected = str(detected_gesture or "").strip().upper()
        if detected:
            result = "hit" if detected == target else "wrong"
            self._record_attempt(now_ms, detected, result, now_ms - self.trial_start_ms)
        elif (now_ms - self.trial_start_ms) >= self.timeout_ms:
            self._record_attempt(now_ms, "", "miss", now_ms - self.trial_start_ms)

    def status_text(self) -> tuple[str, str]:
        target = self.current_target() or "-"
        attempt = min(self.attempt_index, self.reps_target)
        top = f"GESTURE TEST: {target}"
        sub = (
            f"{self.test_name} | {self.condition or 'no condition'} | "
            f"{attempt}/{self.reps_target} | H/W/M: {self.hits}/{self.wrongs}/{self.misses}"
        )
        return top, sub

    def notice_text(self, now_ms: int) -> tuple[str, str] | None:
        if self.notice_until_ms and now_ms <= self.notice_until_ms:
            return self.notice_top, self.notice_sub
        return None

    def _reset_gesture(self, now_ms: int) -> None:
        self.attempt_index = 1
        self.trial_start_ms = now_ms
        self.hits = 0
        self.wrongs = 0
        self.misses = 0
        self.hit_times = []

    def _attempts_done(self) -> int:
        return self.hits + self.wrongs + self.misses

    def _record_attempt(
        self,
        now_ms: int,
        detected: str,
        result: str,
        elapsed_ms: float,
    ) -> None:
        target = self.current_target() or ""
        elapsed_ms = max(0.0, float(elapsed_ms))
        append_eval_row(
            {
                "session_id": self.session_id,
                "datetime": self.session_datetime,
                "test_name": self.test_name,
                "condition": self.condition,
                "expected_gesture": target,
                "detected_gesture": detected,
                "attempt_index": self.attempt_index,
                "result": result,
                "time_to_recognize_ms": round(elapsed_ms, 3),
            }
        )
        if result == "hit":
            self.hits += 1
            self.hit_times.append(elapsed_ms)
        elif result == "wrong":
            self.wrongs += 1
        else:
            self.misses += 1

        self.attempt_index += 1
        self.trial_start_ms = now_ms
        if self.attempt_index > self.reps_target:
            self._finalize_gesture()
            self.gesture_idx += 1
            if self.gesture_idx >= len(self.gestures):
                self._finalize_session()
            else:
                self._reset_gesture(now_ms)

    def _finalize_gesture(self) -> None:
        target = self.current_target() or ""
        attempts = self._attempts_done()
        accuracy = self.hits / attempts if attempts else 0.0
        avg_hit_ms = statistics.mean(self.hit_times) if self.hit_times else None
        row: dict[str, str | float | int] = {
            "session_id": self.session_id,
            "datetime": self.session_datetime,
            "test_name": self.test_name,
            "condition": self.condition,
            "gesture": target,
            "attempts": attempts,
            "hits": self.hits,
            "wrongs": self.wrongs,
            "misses": self.misses,
            "accuracy": round(accuracy, 3),
            "avg_time_to_recognize_ms": round(avg_hit_ms, 3) if avg_hit_ms is not None else "",
        }
        append_gesture_summary_row(row)
        self.summary_rows.append(row)

    def _finalize_session(self) -> None:
        total_attempts = sum(int(item["attempts"]) for item in self.summary_rows)
        total_hits = sum(int(item["hits"]) for item in self.summary_rows)
        total_wrongs = sum(int(item["wrongs"]) for item in self.summary_rows)
        total_misses = sum(int(item["misses"]) for item in self.summary_rows)
        accuracy = total_hits / total_attempts if total_attempts else 0.0
        passed = accuracy >= self.pass_accuracy and total_wrongs <= self.pass_wrong_max

        lines = [
            "GESTURE TEST REPORT",
            f"session_id: {self.session_id}",
            f"datetime: {self.session_datetime}",
            f"test_name: {self.test_name}",
            f"condition: {self.condition}",
            f"hand: {self.hand_label}",
            f"attempts: {total_attempts}",
            f"hits: {total_hits}",
            f"wrongs: {total_wrongs}",
            f"misses: {total_misses}",
            f"accuracy: {accuracy:.3f}",
            f"pass_criteria: accuracy >= {self.pass_accuracy} and wrongs <= {self.pass_wrong_max}",
            f"result: {'PASS' if passed else 'FAIL'}",
        ]
        if self.aborted:
            lines.append("status: ABORTED")
        lines.append("")
        lines.append("Per-gesture summary:")
        for item in self.summary_rows:
            lines.append(
                f"- {item['gesture']}: attempts={item['attempts']}, hits={item['hits']}, "
                f"wrongs={item['wrongs']}, misses={item['misses']}, "
                f"accuracy={item['accuracy']}, avg_ms={item['avg_time_to_recognize_ms']}"
            )

        if self.summary_rows:
            report_path = write_eval_report(self.session_id, "\n".join(lines))
            print(f"[EVAL] Report saved to {report_path}")

        self.active = False
        self.notice_top = "GESTURE TEST DONE"
        self.notice_sub = (
            f"accuracy={accuracy:.3f} | hits={total_hits} wrongs={total_wrongs} misses={total_misses}"
        )
        self.notice_until_ms = int(time.time() * 1000) + 6000


class ScenarioEvalDialog(QtWidgets.QDialog):
    """Non-modal applied-scenario timer and CSV writer for developer mode."""

    def __init__(self, cfg: dict[str, Any], parent: QtWidgets.QWidget | None = None):
        super().__init__(parent)
        self.cfg = cfg
        self.session_id = uuid.uuid4().hex
        self.started_ns: int | None = None
        self.duration_ms: float | None = None
        self.saved = False

        self.setWindowTitle("Scenario test")
        self.setWindowModality(QtCore.Qt.NonModal)
        self.resize(520, 360)

        root = QtWidgets.QVBoxLayout(self)
        form = QtWidgets.QFormLayout()
        root.addLayout(form)

        scenario_cfg = _scenario_test_cfg(cfg)
        scenarios = scenario_cfg.get("scenarios", []) or []
        self.scenario_combo = QtWidgets.QComboBox(self)
        self.scenario_combo.setEditable(True)
        for item in scenarios:
            text = str(item).strip()
            if text:
                self.scenario_combo.addItem(text)
        default_name = str(scenario_cfg.get("default_name", "") or "")
        if default_name:
            index = self.scenario_combo.findText(default_name)
            if index < 0:
                self.scenario_combo.addItem(default_name)
                index = self.scenario_combo.findText(default_name)
            self.scenario_combo.setCurrentIndex(index)
        form.addRow("Scenario name", self.scenario_combo)

        self.elapsed_label = QtWidgets.QLabel("0 ms", self)
        form.addRow("Duration", self.elapsed_label)

        self.success_check = QtWidgets.QCheckBox("Success", self)
        self.success_check.setChecked(True)
        form.addRow("Result", self.success_check)

        self.error_spin = QtWidgets.QSpinBox(self)
        self.error_spin.setRange(0, 999)
        form.addRow("Errors", self.error_spin)

        self.critical_error_spin = QtWidgets.QSpinBox(self)
        self.critical_error_spin.setRange(0, 999)
        form.addRow("Critical errors", self.critical_error_spin)

        self.comment_edit = QtWidgets.QPlainTextEdit(self)
        self.comment_edit.setPlaceholderText("Optional short comment")
        self.comment_edit.setFixedHeight(72)
        form.addRow("Comment", self.comment_edit)

        self.status_label = QtWidgets.QLabel("Enter or choose a scenario, then press Start.", self)
        self.status_label.setWordWrap(True)
        root.addWidget(self.status_label)

        button_row = QtWidgets.QHBoxLayout()
        self.start_btn = QtWidgets.QPushButton("Start / Почати", self)
        self.finish_btn = QtWidgets.QPushButton("Finish / Завершити", self)
        self.save_btn = QtWidgets.QPushButton("Save result", self)
        self.close_btn = QtWidgets.QPushButton("Close", self)
        button_row.addWidget(self.start_btn)
        button_row.addWidget(self.finish_btn)
        button_row.addWidget(self.save_btn)
        button_row.addStretch(1)
        button_row.addWidget(self.close_btn)
        root.addLayout(button_row)

        self.timer = QtCore.QTimer(self)
        self.timer.setInterval(100)
        self.timer.timeout.connect(self._refresh_elapsed)

        self.start_btn.clicked.connect(self._start)
        self.finish_btn.clicked.connect(self._finish)
        self.save_btn.clicked.connect(self._save)
        self.close_btn.clicked.connect(self.close)

        self._set_result_fields_enabled(False)
        self.finish_btn.setEnabled(False)
        self.save_btn.setEnabled(False)

    def _set_result_fields_enabled(self, enabled: bool) -> None:
        self.success_check.setEnabled(enabled)
        self.error_spin.setEnabled(enabled)
        self.critical_error_spin.setEnabled(enabled)
        self.comment_edit.setEnabled(enabled)

    def _scenario_name(self) -> str:
        return self.scenario_combo.currentText().strip()

    def _start(self) -> None:
        if not self._scenario_name():
            self.status_label.setText("Scenario name is required.")
            return
        self.session_id = uuid.uuid4().hex
        self.started_ns = time.perf_counter_ns()
        self.duration_ms = None
        self.saved = False
        self.scenario_combo.setEnabled(False)
        self.start_btn.setEnabled(False)
        self.finish_btn.setEnabled(True)
        self.save_btn.setEnabled(False)
        self._set_result_fields_enabled(False)
        self.status_label.setText("Timer is running. Complete the scenario, then press Finish.")
        self.timer.start()

    def _finish(self) -> None:
        if self.started_ns is None:
            return
        now_ns = time.perf_counter_ns()
        self.duration_ms = (now_ns - self.started_ns) / 1e6
        self.timer.stop()
        self._refresh_elapsed()
        self.finish_btn.setEnabled(False)
        self.save_btn.setEnabled(True)
        self._set_result_fields_enabled(True)
        self.status_label.setText("Scenario finished. Fill result fields and press Save result.")

    def _save(self) -> None:
        if self.duration_ms is None:
            self.status_label.setText("Finish the scenario before saving.")
            return
        append_scenario_eval_row(
            {
                "session_id": self.session_id,
                "datetime": datetime.now().isoformat(timespec="seconds"),
                "scenario_name": self._scenario_name(),
                "duration_ms": round(self.duration_ms, 3),
                "success": 1 if self.success_check.isChecked() else 0,
                "error_count": int(self.error_spin.value()),
                "critical_error_count": int(self.critical_error_spin.value()),
                "comment": self.comment_edit.toPlainText().strip(),
            }
        )
        self.saved = True
        self.save_btn.setEnabled(False)
        self.start_btn.setEnabled(True)
        self.scenario_combo.setEnabled(True)
        self.status_label.setText(
            f"Saved to scenario_eval.csv. Duration: {self.duration_ms:.0f} ms."
        )

    def _refresh_elapsed(self) -> None:
        if self.duration_ms is not None:
            elapsed = self.duration_ms
        elif self.started_ns is not None:
            elapsed = (time.perf_counter_ns() - self.started_ns) / 1e6
        else:
            elapsed = 0.0
        self.elapsed_label.setText(f"{elapsed:.0f} ms")

    def closeEvent(self, event) -> None:
        self.timer.stop()
        super().closeEvent(event)
