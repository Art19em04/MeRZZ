# -*- coding: utf-8 -*-
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from app.utils.bindings import parse_mapping_key


POSE_GESTURES = ("PINCH", "PINCH_MIDDLE", "FIST", "THUMBS_UP", "OPEN_PALM")


@dataclass(frozen=True)
class OneHandAction:
    """A configured one-hand gesture action."""

    side: str
    gesture: str
    combo: str
    label: str


@dataclass(frozen=True)
class OneHandDispatch:
    """Command selected by the one-hand dispatcher."""

    side: str
    gesture: str
    combo: str
    label: str


class OneHandCommandDispatcher:
    """Dispatch ``single_gestures`` directly while one-hand mode is active."""

    def __init__(
        self,
        raw_map: Mapping[str, str],
        hands: Mapping[str, str],
        refractory_ms: int,
    ):
        self.actions = self._build_actions(raw_map, hands)
        self.refractory_ms = max(0, int(refractory_ms))
        self._latched: set[tuple[int, str]] = set()
        self._last_sent_ms = 0

    @staticmethod
    def _build_actions(
        raw_map: Mapping[str, str],
        hands: Mapping[str, str],
    ) -> list[OneHandAction]:
        actions: list[OneHandAction] = []
        for raw_key, raw_combo in (raw_map or {}).items():
            combo = str(raw_combo or "").strip()
            if not combo:
                continue
            parsed = parse_mapping_key(str(raw_key), dict(hands))
            if not parsed:
                raise ValueError(f"[ONE-HAND] invalid single mapping: {raw_key!r}")
            side, gestures = parsed
            if not gestures:
                raise ValueError(f"[ONE-HAND] empty single mapping: {raw_key!r}")
            gesture = str(gestures[0]).upper()
            actions.append(
                OneHandAction(
                    side=str(side).upper(),
                    gesture=gesture,
                    combo=combo,
                    label=str(raw_key),
                )
            )
        return actions

    @staticmethod
    def _gesture_candidates(event: str, pose_flags: Mapping[str, Any]) -> set[str]:
        candidates: set[str] = set()
        event_token = str(event or "").upper()
        if event_token:
            candidates.add(event_token)
        for gesture in POSE_GESTURES:
            if pose_flags.get(gesture):
                candidates.add(gesture)
        return candidates

    @staticmethod
    def _is_active_for_side(
        action: OneHandAction,
        side: str,
        candidates_by_side: Mapping[str, set[str]],
    ) -> bool:
        return action.gesture in candidates_by_side.get(side, set())

    def _active_source(
        self,
        action: OneHandAction,
        candidates_by_side: Mapping[str, set[str]],
    ) -> str | None:
        if action.side in {"RIGHT", "LEFT"}:
            if self._is_active_for_side(action, action.side, candidates_by_side):
                return action.side
            return None
        if action.side in {"EITHER", "ANY"}:
            for side in ("RIGHT", "LEFT"):
                if self._is_active_for_side(action, side, candidates_by_side):
                    return side
            return None
        if action.side == "BOTH":
            if all(
                self._is_active_for_side(action, side, candidates_by_side)
                for side in ("RIGHT", "LEFT")
            ):
                return "BOTH"
            return None
        return None

    def reset(self, now_ms: int = 0) -> None:
        self._latched.clear()
        self._last_sent_ms = int(now_ms)

    def update(
        self,
        now_ms: int,
        *,
        right_present: bool,
        left_present: bool,
        right_event: str,
        left_event: str,
        right_pose_flags: Mapping[str, Any],
        left_pose_flags: Mapping[str, Any],
    ) -> OneHandDispatch | None:
        candidates_by_side = {
            "RIGHT": (
                self._gesture_candidates(right_event, right_pose_flags)
                if right_present
                else set()
            ),
            "LEFT": (
                self._gesture_candidates(left_event, left_pose_flags)
                if left_present
                else set()
            ),
        }
        active_latches: set[tuple[int, str]] = set()
        selected: OneHandDispatch | None = None

        for index, action in enumerate(self.actions):
            source = self._active_source(action, candidates_by_side)
            if not source:
                continue

            latch_key = (index, source)
            active_latches.add(latch_key)
            if latch_key in self._latched:
                continue
            if selected is not None:
                continue
            if now_ms - self._last_sent_ms < self.refractory_ms:
                continue

            selected = OneHandDispatch(
                side=source,
                gesture=action.gesture,
                combo=action.combo,
                label=action.label,
            )
            self._last_sent_ms = now_ms

        self._latched.intersection_update(active_latches)
        self._latched.update(active_latches)
        return selected

    def hint(self, limit: int = 4) -> str:
        if not self.actions:
            return "No one-hand gesture actions configured"
        items = [f"{action.label} -> {action.combo}" for action in self.actions[:limit]]
        if len(self.actions) > limit:
            items.append(f"+{len(self.actions) - limit} more")
        return " | ".join(items)
