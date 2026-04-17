# -*- coding: utf-8 -*-
from __future__ import annotations

import cv2
import numpy as np

from app.gestures import GestureState


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def active_pose_name(state: GestureState, fallback: str) -> str:
    for name in ("PINCH", "PINCH_MIDDLE", "FIST", "THUMBS_UP", "OPEN_PALM"):
        if state.pose_flags.get(name):
            return name
    return fallback or "-"


def draw_pose_label(frame, lm, text, color=(0, 200, 255)) -> None:
    if not text or not lm:
        return
    frame_h, frame_w = frame.shape[:2]
    xs = [point[0] * frame_w for point in lm]
    ys = [point[1] * frame_h for point in lm]
    x = int(min(xs))
    y = int(min(ys)) - 10
    y = max(20, y)
    cv2.putText(
        frame,
        text,
        (x, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        color,
        2,
        cv2.LINE_AA,
    )


def hand_crop(frame, lm, padding: float) -> np.ndarray | None:
    if not lm:
        return None
    frame_h, frame_w = frame.shape[:2]
    xs = [point[0] for point in lm]
    ys = [point[1] for point in lm]

    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    pad_x = (max_x - min_x) * padding
    pad_y = (max_y - min_y) * padding

    min_x -= pad_x
    max_x += pad_x
    min_y -= pad_y
    max_y += pad_y

    center_x = (min_x + max_x) / 2.0
    center_y = (min_y + max_y) / 2.0
    size = max(max_x - min_x, max_y - min_y)

    min_x = center_x - size / 2.0
    max_x = center_x + size / 2.0
    min_y = center_y - size / 2.0
    max_y = center_y + size / 2.0

    min_x = _clamp(min_x, 0.0, 1.0)
    max_x = _clamp(max_x, 0.0, 1.0)
    min_y = _clamp(min_y, 0.0, 1.0)
    max_y = _clamp(max_y, 0.0, 1.0)

    x1 = int(min_x * frame_w)
    x2 = int(max_x * frame_w)
    y1 = int(min_y * frame_h)
    y2 = int(max_y * frame_h)
    if x2 <= x1 or y2 <= y1:
        return None
    return frame[y1:y2, x1:x2].copy()


def render_hand_window(title: str, crop, label: str, size: int, color) -> None:
    if crop is None:
        tile = np.zeros((size, size, 3), dtype=np.uint8)
    else:
        tile = cv2.resize(crop, (size, size), interpolation=cv2.INTER_AREA)
    cv2.rectangle(tile, (0, 0), (size, 28), (0, 0, 0), -1)
    cv2.putText(
        tile,
        label,
        (8, 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        color,
        2,
        cv2.LINE_AA,
    )
    cv2.imshow(title, tile)

