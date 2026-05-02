# -*- coding: utf-8 -*-
"""Helpers for high-level GCPC interaction modes."""
from __future__ import annotations

from typing import Any, Mapping, MutableMapping

from app.gestures import GestureState
from app.utils.bindings import parse_mapping_key


ModeBinding = dict[str, str]
ModeTriggers = dict[str, ModeBinding | None]

_MODE_COMMANDS = {
    "MODE_ONE_HAND": "one_hand",
    "MODE_RECORD": "record",
    "MODE_MOUSE": "mouse",
    "MODE_EXIT": "exit",
    "MODE_CALIBRATE": "calibrate",
}


def build_mode_triggers(
    functional_raw: Mapping[str, Any],
    hands: Mapping[str, str],
) -> tuple[int, int, ModeTriggers]:
    """Build mode trigger bindings from functional gesture mappings."""
    mode_refractory_ms = int(functional_raw.get("refractory_ms", 800))
    exit_hold_ms = int(functional_raw.get("exit_hold_ms", 500))
    mode_triggers: ModeTriggers = {}

    for raw_key, combo in functional_raw.items():
        if not isinstance(combo, str) or not combo.startswith("MODE_"):
            continue
        parsed = parse_mapping_key(raw_key, dict(hands))
        if not parsed:
            raise ValueError(f"[GESTURE] bad functional mapping key: {raw_key!r}")
        side, gestures = parsed
        assert gestures, "parse_mapping_key returned an empty gesture list"
        mode_key = _MODE_COMMANDS.get(combo)
        if mode_key:
            mode_triggers[mode_key] = {"hand": side, "gesture": gestures[0]}

    mode_triggers.setdefault("calibrate", None)
    return mode_refractory_ms, exit_hold_ms, mode_triggers


def binding_active(
    binding: Mapping[str, Any] | None,
    dominant_side: str,
    support_side: str,
    g_right: GestureState,
    g_left: GestureState,
    right_present: bool,
    left_present: bool,
    event_right: str,
    event_left: str,
) -> bool:
    """Check whether a gesture binding is active for the current hand state."""
    if not binding:
        return False
    gesture = str(binding.get("gesture") or "").upper()
    if not gesture:
        return False
    hand = str(binding.get("hand") or "").upper()

    def _side_active(side_name: str) -> bool:
        if side_name == "RIGHT":
            if not right_present:
                return False
            state = g_right
            event = event_right
        elif side_name == "LEFT":
            if not left_present:
                return False
            state = g_left
            event = event_left
        else:
            return False
        return bool(state.pose_flags.get(gesture)) or event == gesture

    if hand == "BOTH":
        return _side_active("RIGHT") and _side_active("LEFT")
    if hand in ("ANY", "EITHER"):
        return _side_active("RIGHT") or _side_active("LEFT")
    if hand == "RIGHT":
        return _side_active("RIGHT")
    if hand == "LEFT":
        return _side_active("LEFT")
    if hand == dominant_side:
        return _side_active(dominant_side)
    if hand == support_side:
        return _side_active(support_side)
    return False


def trigger_fired(
    trigger: Mapping[str, Any] | None,
    *,
    dominant_side: str,
    support_side: str,
    g_right: GestureState,
    g_left: GestureState,
    right_present: bool,
    left_present: bool,
    event_right: str,
    event_left: str,
    dominant_event: str,
    support_event: str,
    both_pose_latched: MutableMapping[str, bool],
) -> bool:
    """Return whether a mode trigger fired on this frame."""
    if not trigger:
        return False
    gesture = trigger.get("gesture")
    hand = trigger.get("hand")
    if not gesture:
        return False
    hand = str(hand or "").upper()
    if not hand:
        return False
    if hand == "BOTH":
        active = bool(
            right_present
            and left_present
            and g_right.pose_flags.get(gesture, False)
            and g_left.pose_flags.get(gesture, False)
        )
        if not active:
            both_pose_latched[gesture] = False
            return False
        if both_pose_latched.get(gesture):
            return False
        both_pose_latched[gesture] = True
        return True
    if hand in ("EITHER", "ANY"):
        return (event_right == gesture) or (event_left == gesture)
    if hand == "RIGHT":
        return event_right == gesture
    if hand == "LEFT":
        return event_left == gesture
    if hand == dominant_side:
        return dominant_event == gesture
    if hand == support_side:
        return support_event == gesture
    return False
