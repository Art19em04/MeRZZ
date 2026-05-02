# -*- coding: utf-8 -*-
from __future__ import annotations

import statistics
from datetime import datetime

from app.services.csv_metrics import append_eval_row, write_eval_report
from app.utils.config import resolve_side


class EvalSingleSession:
    """Stateful EVAL_SINGLE engine extracted from the main runtime loop."""

    def __init__(self, cfg: dict, hands: dict, panel=None):
        eval_cfg = cfg.get("eval_single", {})
        gestures = eval_cfg.get(
            "gestures",
            [
                "PINCH",
                "PINCH_MIDDLE",
                "FIST",
                "THUMBS_UP",
                "OPEN_PALM",
                "SWIPE_RIGHT",
                "SWIPE_LEFT",
            ],
        )
        self.gestures = [str(item).upper() for item in gestures]
        self.reps_target = int(eval_cfg.get("reps", 30))
        self.timeout_ms = int(eval_cfg.get("timeout_ms", 2500))
        self.condition = str(eval_cfg.get("condition", "ideal"))
        self.pass_accuracy = float(eval_cfg.get("pass_accuracy", 0.9))
        self.pass_wrong_max = int(eval_cfg.get("pass_wrong_max", 3))

        self.hand_setting = "RIGHT"
        self.hand_label = "dominant"
        self.reconfigure_hands(cfg, hands)

        self.panel = panel
        self.active = False
        self.session_id = ""
        self.gesture_idx = 0
        self.rep = 1
        self.hits = 0
        self.wrongs = 0
        self.misses = 0
        self.trial_start_ms = 0
        self.hit_times: list[int] = []
        self.summary_rows: list[dict[str, str | float | int]] = []
        self.aborted = False

    def reconfigure_hands(self, cfg: dict, hands: dict) -> None:
        eval_cfg = cfg.get("eval_single", {})
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

    def _reset_stats(self, now_ms: int) -> None:
        self.rep = 1
        self.hits = 0
        self.wrongs = 0
        self.misses = 0
        self.trial_start_ms = now_ms
        self.hit_times = []

    def _finalize_gesture(self) -> None:
        reps_done = max(0, self.rep - 1)
        if reps_done <= 0:
            accuracy = 0.0
            false_rate = 0.0
        else:
            accuracy = self.hits / reps_done
            false_rate = self.wrongs / reps_done

        avg_hit_ms = statistics.mean(self.hit_times) if self.hit_times else None
        row: dict[str, str | float | int] = {
            "session_id": self.session_id,
            "datetime": datetime.now().isoformat(timespec="seconds"),
            "condition": self.condition,
            "hand": self.hand_label,
            "gesture_target": self.current_target() or "",
            "reps": reps_done,
            "hits": self.hits,
            "wrongs": self.wrongs,
            "misses": self.misses,
            "accuracy": round(accuracy, 3),
            "false_rate": round(false_rate, 3),
            "avg_time_to_hit_ms": round(avg_hit_ms, 3) if avg_hit_ms is not None else "",
        }
        append_eval_row(row)
        self.summary_rows.append(row)

    def _finalize_session(self) -> None:
        if not self.summary_rows:
            self.active = False
            if self.panel:
                self.panel.set_eval_single(False)
            return

        total_reps = sum(int(item["reps"]) for item in self.summary_rows)
        total_hits = sum(int(item["hits"]) for item in self.summary_rows)
        total_wrongs = sum(int(item["wrongs"]) for item in self.summary_rows)
        total_misses = sum(int(item["misses"]) for item in self.summary_rows)
        accuracy = total_hits / total_reps if total_reps else 0.0
        false_rate = total_wrongs / total_reps if total_reps else 0.0
        passed = accuracy >= self.pass_accuracy and total_wrongs <= self.pass_wrong_max

        lines = [
            "EVAL_SINGLE REPORT",
            f"session_id: {self.session_id}",
            f"datetime: {datetime.now().isoformat(timespec='seconds')}",
            f"condition: {self.condition}",
            f"hand: {self.hand_label}",
            f"total_reps: {total_reps}",
            f"hits: {total_hits}",
            f"wrongs: {total_wrongs}",
            f"misses: {total_misses}",
            f"accuracy: {accuracy:.3f}",
            f"false_rate: {false_rate:.3f}",
            f"pass_criteria: accuracy >= {self.pass_accuracy} and wrongs <= {self.pass_wrong_max}",
            f"result: {'PASS' if passed else 'FAIL'}",
        ]
        if self.aborted:
            lines.append("status: ABORTED")
        lines.append("")
        lines.append("Per-gesture summary:")
        for item in self.summary_rows:
            lines.append(
                f"- {item['gesture_target']}: reps={item['reps']}, hits={item['hits']}, "
                f"wrongs={item['wrongs']}, misses={item['misses']}, "
                f"accuracy={item['accuracy']}, false_rate={item['false_rate']}"
            )

        report_path = write_eval_report(self.session_id, "\n".join(lines))
        print(f"[EVAL] Report saved to {report_path}")
        self.active = False
        if self.panel:
            self.panel.set_eval_single(False)

    def start(self, now_ms: int, session_id: str) -> bool:
        if self.active or not self.gestures:
            return False
        self.active = True
        self.aborted = False
        self.gesture_idx = 0
        self.summary_rows = []
        self.session_id = session_id
        self._reset_stats(now_ms)
        print("[EVAL] START EVAL_SINGLE")
        return True

    def stop(self) -> bool:
        if not self.active:
            return False
        self.aborted = True
        self._finalize_gesture()
        self._finalize_session()
        print("[EVAL] STOP EVAL_SINGLE")
        return True

    def process(self, now_ms: int, event: str) -> None:
        if not self.active:
            return
        target = self.current_target()
        if not target:
            return

        if event:
            if event == target:
                self.hits += 1
                self.hit_times.append(now_ms - self.trial_start_ms)
            else:
                self.wrongs += 1
            self.rep += 1
            self.trial_start_ms = now_ms
        elif (now_ms - self.trial_start_ms) >= self.timeout_ms:
            self.misses += 1
            self.rep += 1
            self.trial_start_ms = now_ms

        if self.rep > self.reps_target:
            self._finalize_gesture()
            self.gesture_idx += 1
            if self.gesture_idx >= len(self.gestures):
                self._finalize_session()
            else:
                self._reset_stats(now_ms)

    def status_text(self) -> tuple[str, str]:
        target = self.current_target() or "-"
        top = f"EVAL {target} {min(self.rep, self.reps_target)}/{self.reps_target}"
        sub = (
            f"COND: {self.condition} | HAND: {self.hand_label} | "
            f"H/W/M: {self.hits}/{self.wrongs}/{self.misses}"
        )
        return top, sub

