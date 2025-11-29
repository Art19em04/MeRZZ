"""Gesture binding helpers."""
from __future__ import annotations

from typing import Any, Dict, List, Mapping, MutableMapping, Tuple

from app.utils.config import resolve_side


GestureTuple = Tuple[str, ...]


def parse_mapping_key(raw_key: str, hands: Dict[str, str]) -> Tuple[str, List[str]] | None:
    """Parse a gesture mapping string into (side, [gestures]).

    The notation uses hyphen-delimited tokens where the first token optionally
    denotes which hand initiates the mapping (e.g. ``DOMINANT`` or ``LEFT``)
    and the last token of each segment represents the actual gesture name.
    Segments are chained with ``>`` to describe ordered sequences.  For
    example, ``DOMINANT-SEQUENCE-SWIPE_LEFT > DOMINANT-SEQUENCE-SWIPE_RIGHT``
    reads as "start with the dominant hand performing ``SWIPE_LEFT`` inside
    the sequence track, then expect the same hand to perform ``SWIPE_RIGHT``
    as the next sequence gesture".
    """
    steps = raw_key.replace(" ", "").split(">")
    gestures: List[str] = []
    side = None

    for idx, step in enumerate(steps):
        if not step:
            continue
        tokens = step.replace(" ", "").upper().split("-")
        if not tokens:
            continue
        first = tokens[0]
        resolved = resolve_side(first, hands)
        if resolved in {"RIGHT", "LEFT", "BOTH", "EITHER", "ANY"}:
            if side is None and idx == 0:
                side = resolved
            if len(tokens) > 1:
                tokens = tokens[1:]
            else:
                tokens = []
        if not tokens:
            continue
        gesture = tokens[-1]
        gestures.append(gesture)
    if not gestures:
        return None
    if side is None:
        side = resolve_side("DOMINANT", hands)
    return side, gestures


def lookup_mapping(table: Mapping[str, Mapping[Any, Any]] | None, side: str | None, key: Any) -> Any:
    """Look up mapping entry prioritizing specific hand side and fallbacks."""
    if not table or not side:
        return None
    bucket = table.get(side)
    if bucket is None:
        return None
    return bucket.get(key)


def binding_from_string(raw_binding: str, hands: Dict[str, str]) -> Tuple[str, List[str]]:
    """Parse user-facing binding string into hand side and gesture sequence."""
    if not isinstance(raw_binding, str):
        raise ValueError("[GESTURE] SMTH INCORRECT")
    parsed = parse_mapping_key(raw_binding, hands)
    if not parsed:
        raise ValueError("[GESTURE] failed to parse binding")
    side, gestures = parsed
    if not gestures:
        raise ValueError("[GESTURE] no gestures parsed from binding")
    return side, gestures


def parse_single_binding(raw_value: str, hands: Dict[str, str]) -> Dict[str, Any]:
    """Normalize single-gesture binding configuration to a dict."""
    side, gestures = binding_from_string(raw_value, hands)
    return {"hand": side, "gesture": gestures[-1]}


def parse_sequence_binding(raw_value: str, hands: Dict[str, str]) -> Dict[str, Any]:
    """Normalize sequence binding configuration to a dict with gestures list."""
    side, gestures = binding_from_string(raw_value, hands)
    return {"hand": side, "gestures": gestures}


def hand_token_label(side: str | None, dominant_side: str, support_side: str) -> str:
    """Convert hand token to consistent label honoring dominant/support roles."""
    if not side:
        return "DOMINANT"
    side = side.upper()
    if side in ("BOTH", "EITHER", "ANY", "RIGHT", "LEFT"):
        return side
    if side == dominant_side:
        return "DOMINANT"
    if side == support_side:
        return "NON_DOMINANT"
    return side


def binding_notation(binding: Mapping[str, Any], dominant_side: str, support_side: str) -> str:
    """Compose notation string like DOMINANT-SINGLE-GESTURE for display."""
    if not binding:
        raise ValueError("[BINDING] SMTH")
    gesture = binding.get("gesture")
    hand = binding.get("hand")
    token = hand_token_label(hand, dominant_side, support_side)
    if not gesture:
        return token
    return f"{token}-{gesture}"


def trigger_label(trig: Mapping[str, Any], dominant_side: str, support_side: str) -> str:
    """Format human readable label for trigger binding."""
    gesture = trig.get("gesture", "")
    if not gesture:
        return ""
    gesture_txt = str(gesture).replace("_", " ")
    hand = str(trig.get("hand", support_side)).upper()
    if hand == "BOTH":
        prefix = "BOTH"
    elif hand in ("EITHER", "ANY"):
        prefix = "EITHER"
    elif hand == dominant_side:
        prefix = dominant_side
    elif hand == support_side:
        prefix = support_side
    else:
        prefix = hand
    return f"{prefix} {gesture_txt}".strip()


def build_single_map(raw_map: Mapping[str, str], hands: Dict[str, str]) -> Dict[str, Dict[str, str]]:
    """Build mapping for single gesture bindings grouped by hand side."""
    single_map: Dict[str, Dict[str, str]] = {}
    for raw_key, combo in (raw_map or {}).items():
        parsed = parse_mapping_key(raw_key, hands)
        if not parsed:
            raise ValueError("[GESTURE] invalid single mapping")
        side, gestures = parsed
        if not gestures:
            raise ValueError("[GESTURE] empty single mapping")
        bucket = single_map.setdefault(side, {})
        bucket[gestures[0]] = combo
    return single_map


def build_sequence_map(raw_map: Mapping[str, str], hands: Dict[str, str]) -> Dict[str, Dict[GestureTuple, str]]:
    """Build mapping for complex (sequence) gesture bindings."""
    seq_map: Dict[str, Dict[GestureTuple, str]] = {}
    for raw_key, combo in (raw_map or {}).items():
        parsed = parse_mapping_key(raw_key, hands)
        if not parsed:
            raise ValueError("[GESTURE] invalid complex mapping")
        side, gestures = parsed
        if not gestures:
            raise ValueError("[GESTURE] empty complex mapping")
        bucket = seq_map.setdefault(side, {})
        bucket[tuple(gestures)] = combo
    return seq_map


def merge_single_into_sequences(seq_map: MutableMapping[str, Dict[GestureTuple, str]], single_map: Mapping[str, Mapping[str, str]]) -> None:
    """Ensure single-gesture combos are available inside sequence table."""
    for side, bucket in single_map.items():
        seq_bucket = seq_map.setdefault(side, {})
        for gesture, combo in bucket.items():
            seq_bucket.setdefault((gesture,), combo)
