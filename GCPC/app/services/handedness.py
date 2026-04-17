# -*- coding: utf-8 -*-
from __future__ import annotations

from dataclasses import dataclass

from app.gestures import PINKY_MCP, THUMB_TIP


def normalize_handedness_label(raw_label: object) -> str:
    token = str(raw_label or "").strip().upper()
    if token in {"RIGHT", "R", "RH"}:
        return "Right"
    if token in {"LEFT", "L", "LH"}:
        return "Left"
    return ""


def swap_handedness_label(label: str) -> str:
    if label == "Right":
        return "Left"
    if label == "Left":
        return "Right"
    return label


@dataclass
class HandednessResolver:
    """Resolve stable Right/Left labels from model output and geometry fallback."""

    strategy: str
    mirror: bool
    using_tasks_backend: bool
    swap_labels: bool
    prefer_geometry_on_conflict: bool = False

    def __post_init__(self) -> None:
        token = str(self.strategy or "auto").strip().lower()
        if token not in {"auto", "label", "geometry"}:
            token = "auto"
        self.strategy = token

    def infer_side_from_geometry(self, lm) -> str:
        if not lm or len(lm) <= max(THUMB_TIP, PINKY_MCP):
            return "Right"
        thumb_x = float(lm[THUMB_TIP][0])
        pinky_x = float(lm[PINKY_MCP][0])
        if self.mirror:
            return "Right" if thumb_x > pinky_x else "Left"
        return "Right" if thumb_x < pinky_x else "Left"

    def resolve_label(self, hand_obj: dict) -> str:
        geometry_label = self.infer_side_from_geometry(hand_obj.get("lm"))
        reported_label = normalize_handedness_label(hand_obj.get("label"))
        if self.swap_labels and reported_label:
            reported_label = swap_handedness_label(reported_label)

        if self.strategy == "geometry":
            return geometry_label
        if self.strategy == "label":
            return reported_label or geometry_label

        if self.using_tasks_backend:
            return reported_label or geometry_label

        if reported_label and geometry_label and reported_label != geometry_label:
            if self.prefer_geometry_on_conflict:
                return geometry_label
        return reported_label or geometry_label

