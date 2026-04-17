# -*- coding: utf-8 -*-
from __future__ import annotations

import importlib
import inspect
import sys
import time
from pathlib import Path

from app.utils.config import ROOT


def _clamp01(value: float) -> float:
    """Clamp numeric value to normalized [0..1] range."""
    return max(0.0, min(1.0, value))


def _resolve_legacy_hands_api():
    """Try resolving legacy ``mp.solutions.hands`` API."""
    errors: list[str] = []
    try:
        from mediapipe import solutions as mp_solutions

        return mp_solutions.hands, errors
    except Exception as exc:
        errors.append(f"from mediapipe import solutions failed: {exc!r}")

    try:
        solutions_mod = importlib.import_module("mediapipe.solutions")
        return solutions_mod.hands, errors
    except Exception as exc:
        errors.append(f"import mediapipe.solutions failed: {exc!r}")

    return None, errors


def _find_task_model(mp_module) -> Path | None:
    """Find hand landmarker task model within installed mediapipe package."""
    base = Path(getattr(mp_module, "__file__", "")).resolve().parent
    runtime_candidates = [
        ROOT / "models" / "hand_landmarker.task",
        ROOT / "hand_landmarker.task",
    ]
    if getattr(sys, "frozen", False):
        mei_dir = Path(getattr(sys, "_MEIPASS", ""))
        if str(mei_dir):
            runtime_candidates.extend(
                [
                    mei_dir / "models" / "hand_landmarker.task",
                    mei_dir / "hand_landmarker.task",
                ]
            )

    for candidate in runtime_candidates:
        if candidate.exists():
            return candidate

    candidates = [
        base / "modules" / "hand_landmarker" / "hand_landmarker.task",
        base / "modules" / "hand_landmarker" / "hand_landmarker_lite.task",
        base / "modules" / "hand_landmarker" / "hand_landmarker_full.task",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate

    hand_landmarker_dir = base / "modules" / "hand_landmarker"
    if hand_landmarker_dir.exists():
        for candidate in hand_landmarker_dir.glob("*.task"):
            if candidate.exists():
                return candidate

    return None


class MediaPipeHandTracker:
    """Wrapper around MediaPipe Hands with legacy and tasks fallbacks."""

    def __init__(self, min_det=0.6, min_trk=0.5, max_hands=2, model_complexity=1):
        import mediapipe as mp

        self.mp = mp
        self.providers = ["mediapipe"]
        self._backend = "legacy"
        self._task_landmarker = None
        self._last_ts_ms = 0

        hands_api, legacy_errors = _resolve_legacy_hands_api()
        if hands_api is not None:
            self.hands = hands_api.Hands(
                static_image_mode=False,
                max_num_hands=max_hands,
                model_complexity=model_complexity,
                min_detection_confidence=min_det,
                min_tracking_confidence=min_trk,
            )
            self.providers.append("mediapipe.solutions")
            return

        task_errors = []
        try:
            self._init_tasks_backend(min_det=min_det, min_trk=min_trk, max_hands=max_hands)
            self._backend = "tasks"
            self.providers.append("mediapipe.tasks")
            self.hands = None
            return
        except Exception as exc:
            task_errors.append(f"mediapipe tasks init failed: {exc!r}")

        version = getattr(mp, "__version__", "unknown")
        details = "\n".join([*legacy_errors, *task_errors]) or "No extra details."
        raise RuntimeError(
            f"Unsupported mediapipe package layout (version={version}). "
            f"Cannot initialize hand tracking.\n{details}"
        )

    def _init_tasks_backend(self, min_det=0.6, min_trk=0.5, max_hands=2):
        """Initialize MediaPipe Tasks hand landmarker backend."""
        mp = self.mp
        model_path = _find_task_model(mp)
        if model_path is None:
            raise RuntimeError(
                "No hand landmarker task model found inside mediapipe package."
            )

        vision_mod = importlib.import_module("mediapipe.tasks.python.vision")
        base_options_mod = importlib.import_module("mediapipe.tasks.python.core.base_options")
        BaseOptions = getattr(base_options_mod, "BaseOptions")
        HandLandmarker = getattr(vision_mod, "HandLandmarker")
        HandLandmarkerOptions = getattr(vision_mod, "HandLandmarkerOptions")
        running_mode = getattr(vision_mod, "RunningMode", None)
        if running_mode is None:
            running_mode_mod = importlib.import_module(
                "mediapipe.tasks.python.vision.core.vision_task_running_mode"
            )
            running_mode = (
                getattr(running_mode_mod, "VisionTaskRunningMode", None)
                or getattr(running_mode_mod, "VisionRunningMode", None)
            )
        if running_mode is None:
            raise RuntimeError("Cannot resolve MediaPipe Tasks running mode enum.")

        options_kwargs = {
            "base_options": BaseOptions(model_asset_path=str(model_path.resolve())),
            "running_mode": running_mode.VIDEO,
            "num_hands": int(max_hands),
            "min_hand_detection_confidence": float(min_det),
            "min_tracking_confidence": float(min_trk),
            "min_hand_presence_confidence": float(min_det),
        }
        valid_keys = set(inspect.signature(HandLandmarkerOptions).parameters.keys())
        options = HandLandmarkerOptions(**{k: v for k, v in options_kwargs.items() if k in valid_keys})
        self._task_landmarker = HandLandmarker.create_from_options(options)

    def _process_legacy(self, rgb):
        """Run legacy mp.solutions backend and return normalized landmarks."""
        results = self.hands.process(rgb)
        if not results.multi_hand_landmarks:
            return []

        out = []
        for lm, handed in zip(results.multi_hand_landmarks, results.multi_handedness):
            pts = [(_clamp01(float(p.x)), _clamp01(float(p.y))) for p in lm.landmark]
            if handed and handed.classification:
                score = float(handed.classification[0].score)
                label = handed.classification[0].label
            else:
                score = 0.0
                label = "Unknown"
            out.append({"lm": pts, "label": label, "score": score})
        return out

    def _process_tasks(self, rgb):
        """Run Tasks backend and return normalized landmarks in legacy shape."""
        now_ms = int(time.time() * 1000)
        if now_ms <= self._last_ts_ms:
            now_ms = self._last_ts_ms + 1
        self._last_ts_ms = now_ms

        image = self.mp.Image(image_format=self.mp.ImageFormat.SRGB, data=rgb)
        result = self._task_landmarker.detect_for_video(image, now_ms)

        hand_landmarks = list(getattr(result, "hand_landmarks", []) or [])
        handedness = list(getattr(result, "handedness", []) or [])
        out = []
        for idx, lm_list in enumerate(hand_landmarks):
            pts = [(_clamp01(float(p.x)), _clamp01(float(p.y))) for p in lm_list]
            label = "Unknown"
            score = 0.0
            if idx < len(handedness) and handedness[idx]:
                category = handedness[idx][0]
                label = (
                    getattr(category, "category_name", None)
                    or getattr(category, "display_name", None)
                    or "Unknown"
                )
                score = float(getattr(category, "score", 0.0) or 0.0)
            out.append({"lm": pts, "label": label, "score": score})
        return out

    def process(self, rgb):
        """Run active backend on an RGB frame and return normalized landmarks."""
        if self._backend == "legacy":
            return self._process_legacy(rgb)
        return self._process_tasks(rgb)
