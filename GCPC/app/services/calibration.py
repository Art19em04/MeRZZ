# -*- coding: utf-8 -*-
from __future__ import annotations

import time
from typing import Dict, List

from app.gestures import (
    INDEX_MCP,
    INDEX_TIP,
    MIDDLE_MCP,
    MIDDLE_TIP,
    PINKY_MCP,
    THUMB_TIP,
    WRIST,
    finger_flexion,
)
from app.utils.config import save_config


def _dist(a, b) -> float:
    dx, dy = a[0] - b[0], a[1] - b[1]
    return (dx * dx + dy * dy) ** 0.5


def _percentile(values: List[float], fraction: float) -> float | None:
    if not values:
        return None
    sorted_values = sorted(values)
    index = int((len(sorted_values) - 1) * max(0.0, min(1.0, fraction)))
    return sorted_values[index]


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


class CalibrationSession:
    """Stateful gesture-threshold calibration extracted from the main loop."""

    def __init__(self, cfg: dict):
        self.cfg = cfg
        calib_cfg = cfg.get("calibration", {})
        self.enabled = bool(calib_cfg.get("enabled", True))
        self.duration_ms = int(calib_cfg.get("duration_ms", 12000))
        self.trigger_key = str(calib_cfg.get("trigger_key", "c")).lower()

        self.stage_defs = [
            {"name": "PINCH", "hint": "Pinch thumb + index"},
            {"name": "PINCH_MIDDLE", "hint": "Pinch thumb + middle"},
            {"name": "FIST", "hint": "Make a fist"},
            {"name": "THUMBS_UP", "hint": "Thumb up, other fingers bent"},
            {"name": "OPEN_PALM", "hint": "Open palm with straight fingers"},
            {"name": "SWIPE_RIGHT", "hint": "Swipe hand to the right"},
            {"name": "SWIPE_LEFT", "hint": "Swipe hand to the left"},
        ]
        stage_durations = self._split_duration(self.duration_ms, len(self.stage_defs))
        self.stages = [
            self.stage_defs[index] | {"dur_ms": stage_durations[index]}
            for index in range(len(self.stage_defs))
        ]
        self.total_ms = sum(stage["dur_ms"] for stage in self.stages)

        self.active = False
        self.start_ms = 0
        self.stage_ms = 0
        self.stage_idx = -1
        self.data: Dict[str, List[float]] = self._new_data()
        self.motion_prev: Dict[str, tuple[int, tuple[float, float]] | None] = {
            "RIGHT": None,
            "LEFT": None,
        }

    @staticmethod
    def _split_duration(total_ms: int, parts: int) -> List[int]:
        base = total_ms // parts
        result = [base for _ in range(parts)]
        for index in range(total_ms - base * parts):
            result[index % parts] += 1
        return result

    @staticmethod
    def _new_data() -> Dict[str, List[float]]:
        return {
            "pinch": [],
            "pinch_middle": [],
            "fist": [],
            "thumbs_thumb": [],
            "thumbs_others": [],
            "open": [],
            "swipe_right_dx": [],
            "swipe_left_dx": [],
            "swipe_speed": [],
            "swipe_ratio": [],
        }

    def current_stage(self):
        if 0 <= self.stage_idx < len(self.stages):
            return self.stages[self.stage_idx]
        return None

    def _begin_stage(self, index: int, now_ms: int) -> None:
        self.stage_idx = index
        self.stage_ms = now_ms
        self.motion_prev["RIGHT"] = None
        self.motion_prev["LEFT"] = None
        stage = self.current_stage()
        if not stage:
            return
        print(
            f"[CALIBRATION] Stage {index + 1}/{len(self.stages)}: "
            f"{stage['name']} ({stage['dur_ms']} ms) - {stage['hint']}"
        )

    def start(self, now_ms: int) -> bool:
        if not self.enabled:
            return False
        self.active = True
        self.start_ms = now_ms
        self.data = self._new_data()
        self._begin_stage(0, now_ms)
        return True

    def stop(self) -> None:
        self.active = False

    def _palm_span(self, lm) -> float:
        span = _dist(lm[INDEX_MCP], lm[PINKY_MCP])
        anchor = _dist(lm[WRIST], lm[MIDDLE_MCP])
        return max(span, anchor, 1e-4)

    def record(self, lm, side: str | None) -> None:
        if not self.active:
            return
        stage = self.current_stage()
        if not stage:
            return
        stage_name = stage["name"]
        now_ms = int(time.time() * 1000)
        flex = finger_flexion(lm)
        pinch_d = _dist(lm[THUMB_TIP], lm[INDEX_TIP])
        middle_pinch_d = _dist(lm[THUMB_TIP], lm[MIDDLE_TIP])
        span = self._palm_span(lm)
        avg_other = (flex["middle"] + flex["ring"] + flex["pinky"]) / 3.0

        prev = None
        if side:
            prev = self.motion_prev.get(side)
            self.motion_prev[side] = (now_ms, lm[WRIST])

        if stage_name == "PINCH":
            if pinch_d / span < 0.8 and flex["index"] > 0.12 and flex["thumb"] > 0.12:
                self.data["pinch"].append(pinch_d)
        elif stage_name == "PINCH_MIDDLE":
            if middle_pinch_d / span < 0.8 and flex["middle"] > 0.12 and flex["thumb"] > 0.12:
                self.data["pinch_middle"].append(middle_pinch_d)
        elif stage_name == "FIST":
            avg_fist = (flex["index"] + flex["middle"] + flex["ring"] + flex["pinky"]) / 4.0
            if avg_fist > 0.35:
                self.data["fist"].append(avg_fist)
        elif stage_name == "THUMBS_UP":
            if flex["thumb"] < 0.5 and avg_other > 0.35:
                self.data["thumbs_thumb"].append(flex["thumb"])
                self.data["thumbs_others"].append(avg_other)
        elif stage_name == "OPEN_PALM":
            max_open = max(flex["index"], flex["middle"], flex["ring"], flex["pinky"])
            if max_open < 0.55 and pinch_d / span > 0.9:
                self.data["open"].append(max_open)
        elif stage_name in {"SWIPE_RIGHT", "SWIPE_LEFT"} and side and prev:
            prev_ts, prev_pt = prev
            dt = now_ms - prev_ts
            if dt <= 0:
                return
            dx = lm[WRIST][0] - prev_pt[0]
            dy = lm[WRIST][1] - prev_pt[1]
            ratio = abs(dy) / max(abs(dx), 1e-4)
            speed = dx / (dt / 1000.0)
            if stage_name == "SWIPE_RIGHT" and dx > 0:
                self.data["swipe_right_dx"].append(dx)
                self.data["swipe_speed"].append(speed)
                self.data["swipe_ratio"].append(ratio)
            elif stage_name == "SWIPE_LEFT" and dx < 0:
                self.data["swipe_left_dx"].append(-dx)
                self.data["swipe_speed"].append(-speed)
                self.data["swipe_ratio"].append(ratio)

    def _finalize(self, now_ms: int) -> None:
        self.active = False
        ge_cfg = self.cfg.get("gesture_engine", {})
        self.stage_idx = -1

        def or_default(values: List[float], fraction: float, transform, fallback_key: str):
            value = _percentile(values, fraction)
            if value is None:
                return ge_cfg.get(fallback_key)
            return transform(value)

        updates = {
            "pinch_threshold": or_default(
                self.data["pinch"],
                0.9,
                lambda v: _clamp(v * 1.1, 0.01, 0.2),
                "pinch_threshold",
            ),
            "middle_pinch_threshold": or_default(
                self.data["pinch_middle"],
                0.9,
                lambda v: _clamp(v * 1.1, 0.01, 0.2),
                "middle_pinch_threshold",
            ),
            "fist_threshold": or_default(
                self.data["fist"],
                0.25,
                lambda v: _clamp(v * 0.9, 0.15, 0.95),
                "fist_threshold",
            ),
            "thumbs_up_thumb_max_flex": or_default(
                self.data["thumbs_thumb"],
                0.8,
                lambda v: _clamp(v * 1.1, 0.05, 0.8),
                "thumbs_up_thumb_max_flex",
            ),
            "thumbs_up_others_min_flex": or_default(
                self.data["thumbs_others"],
                0.2,
                lambda v: _clamp(v * 0.9, 0.3, 0.95),
                "thumbs_up_others_min_flex",
            ),
            "open_palm_max_flex": or_default(
                self.data["open"],
                0.9,
                lambda v: _clamp(v * 1.05, 0.1, 0.7),
                "open_palm_max_flex",
            ),
            "swipe_min_dx": or_default(
                self.data["swipe_right_dx"] + self.data["swipe_left_dx"],
                0.6,
                lambda v: _clamp(v * 0.8, 0.02, 0.5),
                "swipe_min_dx",
            ),
            "swipe_min_speed": or_default(
                self.data["swipe_speed"],
                0.4,
                lambda v: _clamp(v * 0.8, 0.15, 3.0),
                "swipe_min_speed",
            ),
            "swipe_max_dy_ratio": or_default(
                self.data["swipe_ratio"],
                0.8,
                lambda v: _clamp(v * 1.2, 0.05, 1.5),
                "swipe_max_dy_ratio",
            ),
        }

        ge_cfg.update(updates)
        self.cfg["gesture_engine"] = ge_cfg
        save_config(self.cfg)
        print(f"[CALIBRATION] Updated gesture thresholds at {now_ms} ms: {updates}")

    def advance(self, now_ms: int) -> bool:
        if not self.active:
            return False
        while True:
            stage = self.current_stage()
            if not stage:
                return False
            elapsed = now_ms - self.stage_ms
            if elapsed < stage["dur_ms"]:
                return False
            next_index = self.stage_idx + 1
            if next_index >= len(self.stages):
                self._finalize(now_ms)
                return True
            self._begin_stage(next_index, now_ms)

    def status_text(self, now_ms: int) -> tuple[str, str]:
        stage = self.current_stage()
        stage_elapsed = now_ms - self.stage_ms
        stage_remaining = max(0, (stage["dur_ms"] if stage else 0) - stage_elapsed)
        total_elapsed = now_ms - self.start_ms
        total_remaining = max(0, self.total_ms - total_elapsed)
        stage_name = stage["name"] if stage else "DONE"
        hint = stage["hint"] if stage else "Calibration completed"
        top = f"CAL {self.stage_idx + 1}/{len(self.stages)}: {stage_name}"
        sub = (
            f"{hint} | Stage: {stage_remaining / 1000:.1f}s | "
            f"Total: {total_remaining / 1000:.1f}s"
        )
        return top, sub

