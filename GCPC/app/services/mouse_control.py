# -*- coding: utf-8 -*-
"""Mouse mode configuration helpers."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Mapping

from app.gestures import WRIST
from app.utils.bindings import binding_notation, parse_single_binding, trigger_label
from app.utils.config import resolve_side

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class MouseControlSettings:
    """Parsed mouse mode settings used by the main loop."""

    enabled: bool
    status_label: str
    smoothing_alpha: float
    pointer_side: str
    pointer_landmark: int
    rect: dict[str, float]
    left_binding: dict[str, Any]
    left_label: str
    right_binding: dict[str, Any]
    right_label: str
    scroll_enabled: bool
    scroll_side: str
    scroll_gesture: str
    scroll_landmark: int
    scroll_speed: float
    scroll_deadzone: float
    scroll_interval_ms: int
    scroll_label: str
    active_hint: str


def clamp_rect(x: float, y: float, width: float, height: float) -> dict[str, float]:
    """Clamp a normalized pointer-control rectangle to the frame bounds."""
    x = max(0.0, min(1.0, x))
    y = max(0.0, min(1.0, y))
    width = max(1e-4, min(1.0 - x, width))
    height = max(1e-4, min(1.0 - y, height))
    return {"x": x, "y": y, "w": width, "h": height}


def _resolve_mouse_hand(
    raw_value: Any,
    hands: Mapping[str, str],
    fallback_side: str,
    setting_name: str,
) -> str:
    side = resolve_side(raw_value, dict(hands))
    if side in ("RIGHT", "LEFT"):
        return side
    LOGGER.warning(
        "Invalid mouse %s=%r resolved to %r; falling back to %s",
        setting_name,
        raw_value,
        side,
        fallback_side,
    )
    return fallback_side


def _read_mouse_binding(
    mouse_cfg: Mapping[str, Any],
    key: str,
    hands: Mapping[str, str],
    dominant_side: str,
    support_side: str,
) -> tuple[dict[str, Any], str]:
    binding = parse_single_binding(mouse_cfg.get(key), dict(hands))
    gesture = binding.get("gesture") or ""
    binding["gesture"] = str(gesture).upper()
    label = binding_notation(binding, dominant_side, support_side)
    return binding, label


def build_mouse_control_settings(
    mouse_cfg: Mapping[str, Any],
    hands: Mapping[str, str],
    dominant_side: str,
    support_side: str,
    mouse_trigger: Mapping[str, Any] | None,
) -> MouseControlSettings:
    """Parse mouse mode configuration into a compact runtime object."""
    enabled = bool(mouse_cfg.get("enabled", False))
    status_label = str(mouse_cfg.get("status_label", "MOUSE"))
    smoothing_alpha = max(
        0.0,
        min(1.0, float(mouse_cfg.get("smoothing_alpha", 0.25))),
    )
    pointer_side = _resolve_mouse_hand(
        mouse_cfg.get("pointer_hand"),
        hands,
        dominant_side,
        "pointer_hand",
    )
    pointer_landmark = int(mouse_cfg.get("pointer_landmark", 8))

    rect_cfg = mouse_cfg.get("control_rect", {}) or {}
    rect = clamp_rect(
        float(rect_cfg.get("x", 0.0) or 0.0),
        float(rect_cfg.get("y", 0.0) or 0.0),
        float(rect_cfg.get("width", 1.0) or 1.0),
        float(rect_cfg.get("height", 1.0) or 1.0),
    )

    left_binding, left_label = _read_mouse_binding(
        mouse_cfg,
        "left_click_binding",
        hands,
        dominant_side,
        support_side,
    )
    right_binding, right_label = _read_mouse_binding(
        mouse_cfg,
        "right_click_binding",
        hands,
        dominant_side,
        support_side,
    )

    scroll_cfg = mouse_cfg.get("scroll", {}) or {}
    scroll_enabled = bool(scroll_cfg.get("enabled", True))
    scroll_side = _resolve_mouse_hand(
        scroll_cfg.get("hand", "RIGHT"),
        hands,
        "RIGHT",
        "scroll.hand",
    )
    scroll_gesture = str(scroll_cfg.get("gesture", "FIST")).upper()
    scroll_landmark = int(scroll_cfg.get("landmark", WRIST))
    scroll_speed = float(scroll_cfg.get("speed", 1200.0))
    scroll_deadzone = float(scroll_cfg.get("deadzone", 0.01))
    scroll_interval_ms = int(scroll_cfg.get("interval_ms", 30))
    scroll_label = binding_notation(
        {"hand": scroll_side, "gesture": scroll_gesture},
        dominant_side,
        support_side,
    )

    active_hint = mouse_cfg.get("active_hint")
    if not active_hint:
        trigger_hint = trigger_label(mouse_trigger, dominant_side, support_side)
        pointer_hint = binding_notation(
            {"hand": pointer_side, "gesture": "PINCH"},
            dominant_side,
            support_side,
        )
        base_hint = f"CURSOR: {pointer_hint} | LMB: {left_label} | RMB: {right_label}"
        if scroll_enabled:
            base_hint = f"{base_hint} | SCROLL: {scroll_label}"
        active_hint = f"{trigger_hint} | {base_hint}" if trigger_hint else base_hint

    return MouseControlSettings(
        enabled=enabled,
        status_label=status_label,
        smoothing_alpha=smoothing_alpha,
        pointer_side=pointer_side,
        pointer_landmark=pointer_landmark,
        rect=rect,
        left_binding=left_binding,
        left_label=left_label,
        right_binding=right_binding,
        right_label=right_label,
        scroll_enabled=scroll_enabled,
        scroll_side=scroll_side,
        scroll_gesture=scroll_gesture,
        scroll_landmark=scroll_landmark,
        scroll_speed=scroll_speed,
        scroll_deadzone=scroll_deadzone,
        scroll_interval_ms=scroll_interval_ms,
        scroll_label=scroll_label,
        active_hint=str(active_hint),
    )
