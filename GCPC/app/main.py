# -*- coding: utf-8 -*-
import json
import os
import time

import cv2
from PySide6 import QtWidgets

from app.gestures import GestureState
from app.os_events_win import press_combo, mouse_move_normalized, mouse_press, mouse_release
from app.osd import OSD
from app.tracker_mediapipe import MediaPipeHandTracker

APP_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(APP_DIR)


def load_config():
    """Load application configuration from the root config.json file."""
    with open(os.path.join(ROOT, "config.json"), "r", encoding="utf-8") as f:
        return json.load(f)


def build_hands(cfg):
    """Construct a mapping describing dominant and support hands."""
    hands_cfg = cfg.get("hands")
    dominant = hands_cfg.get("dominant")
    if dominant != "RIGHT" and dominant != "LEFT":
        raise ValueError("[CFG] INCORRECT DOMINANT HAND")
    support = hands_cfg.get("support")
    if dominant != "RIGHT" and dominant != "LEFT":
        raise ValueError("[CFG] INCORRECT SUPPORT HAND")
    return {"dominant": dominant, "support": support}


def resolve_side(tag, hands):
    """Resolve textual hand tag into a concrete side using provided defaults."""
    dominant_side = hands.get("dominant")
    support_side = hands.get("support")
    if tag is None:
        return dominant_side
    tag = str(tag).strip().upper()
    if not tag:
        return dominant_side
    if tag in ("RIGHT", "LEFT", "BOTH", "EITHER", "ANY"):
        return tag
    if tag == "DOMINANT":
        return dominant_side
    if tag == "NON_DOMINANT":
        return support_side
    return tag or dominant_side


def parse_mapping_key(raw_key, hands):
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
    gestures = []
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


def lookup_mapping(table, side, key):
    """Look up mapping entry prioritizing specific hand side and fallbacks."""
    if not table or not side:
        return None
    bucket = table.get(side)
    if bucket is None:
        return None
    return bucket.get(key)


class DebouncedTrigger:
    """Utility that fires after a dwell time while respecting a refractory window."""

    def __init__(self, dwell_ms=260, refractory_ms=900):
        self.dwell_ms = dwell_ms
        self.refractory_ms = refractory_ms
        self.candidate_since = None
        self.last_fire = 0

    def update(self, now_ms, active):
        """Return True when active has stayed asserted long enough to trigger."""
        if not active:
            self.candidate_since = None
            return False
        if self.candidate_since is None:
            self.candidate_since = now_ms
            return False
        if (now_ms - self.candidate_since) >= self.dwell_ms and (now_ms - self.last_fire) >= self.refractory_ms:
            self.last_fire = now_ms
            self.candidate_since = None
            return True
        return False


def open_camera(idx, w, h):
    """Try to open camera by preferred index with fallbacks across APIs."""

    def try_open(i, api):
        """Attempt to open a specific camera index using selected backend."""
        cap = cv2.VideoCapture(i, api)
        if cap.isOpened():
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
            ok, _ = cap.read()
            if ok:
                return cap
            cap.release()
        raise ValueError("[DEVICE] WRONG DEVICE SETUP")

    def probe():
        for probe in range(0, 6):
            for api in (cv2.CAP_MSMF, cv2.CAP_DSHOW, cv2.CAP_ANY):
                cap = try_open(probe, api)
                if cap:
                    print(f"[videoio] open idx={probe} api={api}")
                    return cap

    for api in (cv2.CAP_MSMF, cv2.CAP_DSHOW, cv2.CAP_ANY):
        cap = try_open(idx, api)
        if cap:
            print(f"[videoio] open idx={idx} api={api}")
            return cap

    if idx == -1:
        return probe()

    return None


def draw_landmarks(frame, lm):
    """Draw simple circles for each landmark on a frame in-place."""
    h, w = frame.shape[:2]
    for (x, y) in lm:
        cv2.circle(frame, (int(x * w), int(y * h)), 4, (0, 255, 0), -1)


def main():
    cfg = load_config()
    app = QtWidgets.QApplication([])
    osd = OSD()
    osd.show()

    vcfg = cfg["video"]
    idx = int(vcfg.get("camera_index", 0))
    w = int(vcfg.get("width", 1280))
    h = int(vcfg.get("height", 720))
    mirror = bool(vcfg.get("mirror", True))
    show_fps = bool(vcfg.get("show_fps", False))

    cap = open_camera(idx, w, h)

    dcfg = cfg.get("detector", {})
    tracker = MediaPipeHandTracker(
        min_det=dcfg.get("min_detection_confidence"),
        min_trk=dcfg.get("min_tracking_confidence"),
        max_hands=dcfg.get("max_num_hands"),
        model_complexity=dcfg.get("model_complexity"),
    )

    gR = GestureState(cfg["gesture_engine"])
    gL = GestureState(cfg["gesture_engine"])

    seq_cfg = cfg.get("sequence", {})
    arm_delay_ms = int(seq_cfg.get("arm_delay_ms"))
    refractory_ms = int(seq_cfg.get("refractory_ms"))
    cancel_exit_ms = int(seq_cfg.get("cancel_on_hand_exit_ms"))
    auto_exit = bool(seq_cfg.get("auto_exit_on_hand_exit"))
    max_len = int(seq_cfg.get("max_len"))

    hands = build_hands(cfg)
    dominant_side = hands["dominant"]
    support_side = hands["support"]
    dominant_is_right = dominant_side == "RIGHT"

    controls_cfg = cfg.get("controls", {})
    seq_ctrl = controls_cfg.get("sequence", {})
    seq_input_side = resolve_side(seq_ctrl.get("input_hand", "dominant"), hands)
    candidate_ignore = {str(g).upper() for g in seq_ctrl.get("candidate_ignore")}

    def _binding_from_string(raw_binding):
        """Parse user-facing binding string into hand side and gesture tuple."""
        if not isinstance(raw_binding, str):
            raise ValueError("[GESTURE] SMTH INCORRECT")
        parsed = parse_mapping_key(raw_binding, hands)
        if not parsed:
            return None
        side, gestures = parsed
        if not gestures:
            raise ValueError("[GESTURE] GESTURE MAPPING GOES WRONG")
        return side, gestures

    def _parse_single_binding(raw_value: str):
        """Normalize single-gesture binding configuration to a dict."""
        candidate = raw_value
        parsed = _binding_from_string(candidate)
        if not parsed:
            raise ValueError("[GESTURE] failed to parse binding")
        side, gestures = parsed
        if not gestures:
            raise ValueError("[GESTURE] no gestures parsed from binding")
        return {
            "hand": side,
            "gesture": gestures[-1],
        }

    def _parse_sequence_binding(raw_value: str):
        """Normalize sequence binding configuration to a dict with gestures list."""
        candidate = raw_value
        parsed = _binding_from_string(candidate)
        if not parsed:
            raise ValueError(f"[GESTURE] failed to parse sequence binding: {candidate!r}")
        side, gestures = parsed
        if not gestures:
            raise ValueError(f"[GESTURE] no gestures parsed from sequence binding: {candidate!r}")
        return {
            "hand": side,
            "gestures": gestures,
        }

    confirm_cfg = seq_ctrl.get("confirm")
    confirm_binding_value = (
        confirm_cfg.get("binding")
        if isinstance(confirm_cfg, dict)
        else confirm_cfg
    )
    confirm_binding = _parse_single_binding(confirm_binding_value)
    confirm_hand = confirm_binding["hand"]
    confirm_gesture = confirm_binding["gesture"]
    confirm_deb = DebouncedTrigger(
        int(confirm_cfg.get("dwell_ms", 220)),
        int(confirm_cfg.get("refractory_ms", 700)),
    )

    undo_cfg = seq_ctrl.get("undo")
    undo_binding_value = (
        undo_cfg.get("binding")
        if isinstance(undo_cfg, dict)
        else undo_cfg
    )
    undo_binding = _parse_sequence_binding(undo_binding_value)
    undo_hand = undo_binding["hand"]
    undo_steps = undo_binding["gestures"]
    undo_start = undo_steps[0]
    undo_end = undo_steps[-1] if len(undo_steps) > 1 else undo_steps[0]
    undo_window_ms = int(undo_cfg.get("window_ms"))

    commit_cfg = seq_ctrl.get("commit")
    commit_binding_value = (
        commit_cfg.get("binding")
        if isinstance(commit_cfg, dict)
        else commit_cfg
    )
    commit_binding = _parse_single_binding(
        commit_binding_value
    )
    commit_hand = commit_binding["hand"]
    commit_gesture = commit_binding["gesture"]
    commit_deb = DebouncedTrigger(
        int(commit_cfg.get("dwell_ms")),
        int(commit_cfg.get("refractory_ms")),
    )
    exit_on_commit = bool(commit_cfg.get("exit_on_commit", seq_cfg.get("exit_on_commit", True)))

    cmd_map = cfg.get("command_mappings") or {}

    single_map_raw = cmd_map.get("single_gestures") or {}
    complex_map_raw = cmd_map.get("complex_gestures") or {}
    functional_raw = cmd_map.get("functional") or {}
    mode_refractory_ms = int(functional_raw.get("refractory_ms", 800))
    mode_triggers = {}

    for raw_key, combo in functional_raw.items():
        if not isinstance(combo, str) or not combo.startswith("MODE_"):
            continue

        parsed = parse_mapping_key(raw_key, hands)
        if not parsed:
            raise ValueError(f"[GESTURE] bad functional mapping key: {raw_key!r}")
        side, gestures = parsed
        if not gestures:
            raise ValueError(f"[GESTURE] empty gestures for key: {raw_key!r}")

        binding = {"hand": side, "gesture": gestures[0]}

        if combo == "MODE_ONE_HAND":
            mode_triggers["one_hand"] = binding
        elif combo == "MODE_RECORD":
            mode_triggers["record"] = binding
        elif combo == "MODE_MOUSE":
            mode_triggers["mouse"] = binding
        elif combo == "MODE_EXIT":
            mode_triggers["exit"] = binding

    single_map = {}
    for raw_key, combo in single_map_raw.items():
        parsed = parse_mapping_key(raw_key, hands)
        if not parsed:
            raise ValueError("[GESTURE] invalid single mapping")
        side, gestures = parsed
        if not gestures:
            raise ValueError("[GESTURE] empty single mapping")
        bucket = single_map.setdefault(side, {})
        bucket[gestures[0]] = combo

    seq_map = {}
    for raw_key, combo in complex_map_raw.items():
        parsed = parse_mapping_key(raw_key, hands)
        if not parsed:
            raise ValueError("[GESTURE] invalid complex mapping")
        side, gestures = parsed
        if not gestures:
            raise ValueError("[GESTURE] empty complex mapping")
        bucket = seq_map.setdefault(side, {})
        bucket[tuple(gestures)] = combo

    for side, bucket in single_map.items():
        seq_bucket = seq_map.setdefault(side, {})
        for gesture, combo in bucket.items():
            seq_bucket.setdefault((gesture,), combo)


    def _trigger_label(trig):
        """Format human readable label for trigger binding."""
        gesture = trig.get("gesture", "")
        if not gesture:
            return ""
        gesture_txt = gesture.replace("_", " ")
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

    def _hand_token_label(side):
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

    def _binding_notation(binding):
        """Compose notation string like DOMINANT-SINGLE-GESTURE for display."""
        if not binding:
            raise ValueError("[BINDING] SMTH")
        gesture = binding.get("gesture")
        hand = binding.get("hand")
        token = _hand_token_label(hand)
        if not gesture:
            return token
        return f"{token}-{gesture}"

    one_cfg = cfg.get("one_hand_mode", {})
    one_enabled = bool(one_cfg.get("enabled", True))
    one_status_label = one_cfg.get("status_label")

    func = cfg.get("command_mappings", {}).get("functional", {})
    exit_gesture = next((g for g, cmd in func.items() if cmd == "MODE_EXIT"), None)

    base_hint = one_cfg.get("active_hint") or ""

    if exit_gesture:
        exit_hint = f"To exit mode: {exit_gesture}"
    else:
        exit_hint = ""

    if base_hint and exit_hint:
        one_active_hint = f"{base_hint} | {exit_hint}"
    else:
        one_active_hint = base_hint or exit_hint or ""

    dispatch_side = resolve_side(one_cfg.get("dispatch_hand", "dominant"), hands)

    if not one_active_hint:
        raise ValueError("[GESTURE] SMTH")

    def _binding_active(binding, right_present, left_present, event_right, event_left):
        """Check whether gesture binding is satisfied given hand presence and events."""
        if not binding:
            return False
        gesture = binding.get("gesture")
        if not gesture:
            return False
        hand = (binding.get("hand") or "").upper()
        gesture = str(gesture).upper()

        def _side_active(side_name):
            """Return True if specific side has active pose or event."""
            if side_name == "RIGHT":
                if not right_present:
                    return False
                state = gR
                event = event_right
            elif side_name == "LEFT":
                if not left_present:
                    return False
                state = gL
                event = event_left
            else:
                return False
            flag = state.pose_flags.get(gesture)
            if flag:
                return True
            return event == gesture

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

    mouse_cfg = cfg.get("mouse_control", {})
    mouse_enabled = bool(mouse_cfg.get("enabled"))
    mouse_status_label = mouse_cfg.get("status_label")
    mouse_smooth = max(0.0, min(1.0, float(mouse_cfg.get("smoothing_alpha", 0.25))))
    pointer_hand_token = mouse_cfg.get("pointer_hand")
    pointer_side = resolve_side(pointer_hand_token, hands)
    if pointer_side not in ("RIGHT", "LEFT"):
        pointer_side = dominant_side
    pointer_landmark = int(mouse_cfg.get("pointer_landmark", 8))

    def _read_mouse_binding(key):
        raw_value = mouse_cfg.get(key)
        binding = _parse_single_binding(raw_value)
        gesture = binding.get("gesture") or ""
        binding["gesture"] = gesture.upper()
        label = _binding_notation(binding)
        return binding, label

    mouse_left_binding, mouse_left_label = _read_mouse_binding(
        "left_click_binding"
    )
    mouse_right_binding, mouse_right_label = _read_mouse_binding(
        "right_click_binding"
    )
    mouse_active_hint = mouse_cfg.get("active_hint")
    if not mouse_active_hint:
        trig_label = _trigger_label(mode_triggers["mouse"])
        pointer_hint = _binding_notation({"hand": pointer_side, "gesture": "POINT"})
        base_hint = f"CURSOR: {pointer_hint} | LMB: {mouse_left_label} | RMB: {mouse_right_label}"
        mouse_active_hint = f"{trig_label} | {base_hint}" if trig_label else base_hint

    mouse_prev = None
    mouse_left_down = False
    mouse_right_down = False

    seq_active = False
    seq_buffer = []
    seq_pending = None
    last_seq_event_ms = 0
    last_evt_ms = 0
    last_sent_ms = 0

    last_seen_R = int(time.time() * 1000)
    last_seen_L = int(time.time() * 1000)
    undo_open_ts = {"RIGHT": None, "LEFT": None}

    last_R_label = ""
    last_L_label = ""

    one_hand_active = False
    mouse_active = False
    last_single_action = ""

    current_mode = "idle"
    mode_last_change_ms = 0

    both_pose_latched = {}

    def switch_mode(new_mode, now_ms, force_reset=False):
        """Transition between operation modes and reset related state when needed."""
        nonlocal current_mode, seq_active, one_hand_active, mouse_active
        nonlocal mouse_prev, mouse_left_down, mouse_right_down
        nonlocal last_single_action, seq_buffer, seq_pending, last_evt_ms, mode_last_change_ms
        prev = current_mode
        if new_mode == prev and not force_reset:
            mode_last_change_ms = now_ms
            return
        if new_mode == "record":
            seq_buffer.clear()
            seq_pending = None
            seq_active = True
            last_evt_ms = now_ms
        else:
            seq_active = False
            if prev == "record" or force_reset:
                seq_buffer.clear()
                seq_pending = None
        if new_mode != "mouse" or prev == "mouse" or force_reset:
            if mouse_left_down:
                mouse_release("left")
                mouse_left_down = False
            if mouse_right_down:
                mouse_release("right")
                mouse_right_down = False
            mouse_prev = None
        elif new_mode == "mouse":
            mouse_prev = None
        if new_mode != "one_hand":
            last_single_action = ""
        one_hand_active = (new_mode == "one_hand") and one_enabled
        mouse_active = (new_mode == "mouse") and mouse_enabled
        current_mode = new_mode
        mode_last_change_ms = now_ms
        labels = {"idle": "IDLE", "record": "RECORD", "mouse": mouse_status_label, "one_hand": one_status_label}
        print(f"[MODE] -> {labels.get(new_mode, new_mode.upper())}")

    fps = None
    last_frame_time = time.time()

    while True:
        QtWidgets.QApplication.processEvents()
        ret, frame = cap.read()
        if not ret:
            break
        if mirror:
            frame = cv2.flip(frame, 1)

        now = time.time()
        dt = now - last_frame_time
        last_frame_time = now
        now_ms = int(now * 1000)
        if show_fps and dt > 0:
            inst = 1.0 / dt
            fps = inst if fps is None else (0.9 * fps + 0.1 * inst)

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        hands = tracker.process(rgb)

        right = left = None
        rights = []
        lefts = []
        for hnd in hands:
            if hnd.get("label", "") == "Right":
                rights.append(hnd)
            elif hnd.get("label", "") == "Left":
                lefts.append(hnd)
        if not rights and not lefts and hands:
            srt = sorted(hands, key=lambda h: h["lm"][0][0])
            if len(srt) == 2:
                lefts = [srt[0]]
                rights = [srt[1]]
            elif len(srt) == 1:
                lm = srt[0]["lm"]
                lbl = "Right" if lm[4][0] > lm[20][0] else "Left"
                if lbl == "Right":
                    rights = [srt[0]]
                else:
                    lefts = [srt[0]]

        right = max(rights, key=lambda h: h.get("score", 0)) if rights else None
        left = max(lefts, key=lambda h: h.get("score", 0)) if lefts else None

        evR = None
        evL = None

        if right:
            last_seen_R = now_ms
            evR = gR.update_and_classify(right["lm"]) or ""
            if evR:
                last_R_label = evR
        else:
            gR.pose_flags.clear()
            if seq_active and seq_input_side == "RIGHT" and cancel_exit_ms > 0 and (
                    now_ms - last_seen_R) >= cancel_exit_ms:
                seq_buffer.clear()
                seq_pending = None
                if auto_exit:
                    switch_mode("idle", now_ms, force_reset=True)
                else:
                    seq_active = False
                print("[SEQ] Авто-отмена: правая рука вне кадра")

        if left:
            last_seen_L = now_ms
            evL = gL.update_and_classify(left["lm"]) or ""
            if evL:
                last_L_label = evL
        else:
            gL.pose_flags.clear()
            if seq_active and seq_input_side == "LEFT" and cancel_exit_ms > 0 and (
                    now_ms - last_seen_L) >= cancel_exit_ms:
                seq_buffer.clear()
                seq_pending = None
                if auto_exit:
                    switch_mode("idle", now_ms, force_reset=True)
                else:
                    seq_active = False
                print("[SEQ] Авто-отмена: левая рука вне кадра")

        dom_event = evR if dominant_side == "RIGHT" else evL
        support_event = evR if support_side == "RIGHT" else evL

        def _trigger_fired(trig):
            """Return True when a mode trigger gesture has just been activated."""
            nonlocal both_pose_latched
            gesture = trig.get("gesture")
            hand = trig.get("hand")
            if not gesture:
                return False
            hand = (hand or "").upper()
            if not hand:
                return False
            if hand == "BOTH":
                active = bool(
                    right and left and gR.pose_flags.get(gesture, False) and gL.pose_flags.get(gesture, False))
                if not active:
                    both_pose_latched[gesture] = False
                    return False
                if both_pose_latched.get(gesture):
                    return False
                both_pose_latched[gesture] = True
                return True
            if hand in ("EITHER", "ANY"):
                return (evR == gesture) or (evL == gesture)
            if hand == "RIGHT":
                return evR == gesture
            if hand == "LEFT":
                return evL == gesture
            if hand == dominant_side:
                return dom_event == gesture
            if hand == support_side:
                return support_event == gesture
            return False

        time_since_change = now_ms - mode_last_change_ms

        if current_mode != "idle" and _trigger_fired(mode_triggers["exit"]):
            switch_mode("idle", now_ms, force_reset=True)
        elif _trigger_fired(mode_triggers["record"]) and time_since_change >= mode_refractory_ms:
            if current_mode == "record":
                switch_mode("record", now_ms, force_reset=True)
            elif current_mode == "idle":
                switch_mode("record", now_ms)
        elif mouse_enabled and _trigger_fired(mode_triggers["mouse"]) and time_since_change >= mode_refractory_ms:
            if current_mode == "mouse":
                switch_mode("mouse", now_ms, force_reset=True)
            elif current_mode == "idle":
                switch_mode("mouse", now_ms)
        elif one_enabled and _trigger_fired(mode_triggers["one_hand"]) and time_since_change >= mode_refractory_ms:
            if current_mode == "one_hand":
                switch_mode("one_hand", now_ms, force_reset=True)
            elif current_mode == "idle":
                switch_mode("one_hand", now_ms)

        if current_mode != "mouse":
            mouse_prev = None
            if mouse_left_down:
                mouse_release("left")
                mouse_left_down = False
            if mouse_right_down:
                mouse_release("right")
                mouse_right_down = False

        if one_hand_active and not seq_active and not mouse_active:
            sides = []
            if dispatch_side in ("EITHER", "ANY"):
                sides = ["RIGHT", "LEFT"]
            elif dispatch_side == "BOTH":
                sides = ["RIGHT", "LEFT"]
            else:
                sides = [dispatch_side]
            triggered = False
            for side in sides:
                event = evR if side == "RIGHT" else evL
                present = bool(right) if side == "RIGHT" else bool(left)
                if not event or not present:
                    continue
                combo = lookup_mapping(single_map, side, event)
                if combo and (now_ms - last_sent_ms) >= refractory_ms:
                    print(f"[ONE-HAND] {side} {event} -> {combo}")
                    press_combo(combo)
                    last_sent_ms = now_ms
                    last_single_action = f"LAST: {side} {event} → {combo}"
                    triggered = True
                    break
            if not triggered and dispatch_side == "BOTH":
                for side in ("RIGHT", "LEFT"):
                    event = evR if side == "RIGHT" else evL
                    present = bool(right) if side == "RIGHT" else bool(left)
                    if not event or not present:
                        continue
                    combo = lookup_mapping(single_map, "EITHER", event)
                    if combo and (now_ms - last_sent_ms) >= refractory_ms:
                        print(f"[ONE-HAND] {side} {event} -> {combo}")
                        press_combo(combo)
                        last_sent_ms = now_ms
                        last_single_action = f"LAST: {side} {event} → {combo}"
                        break

        if mouse_active:
            pointer_source = right if pointer_side == "RIGHT" else left
            right_present = bool(right)
            left_present = bool(left)
            if pointer_source:
                tip = pointer_source["lm"][pointer_landmark]
                target = (tip[0], tip[1])
                if mouse_prev is None or mouse_smooth <= 0.0:
                    mouse_prev = target
                else:
                    prev_x, prev_y = mouse_prev
                    mouse_prev = (
                        prev_x + (target[0] - prev_x) * (1.0 - mouse_smooth),
                        prev_y + (target[1] - prev_y) * (1.0 - mouse_smooth),
                    )
                mx, my = mouse_prev
                mouse_move_normalized(mx, my)
                left_active = _binding_active(
                    mouse_left_binding,
                    right_present,
                    left_present,
                    evR,
                    evL,
                )
                right_active = _binding_active(
                    mouse_right_binding,
                    right_present,
                    left_present,
                    evR,
                    evL,
                )
                if left_active and not mouse_left_down:
                    mouse_press("left")
                    mouse_left_down = True
                if not left_active and mouse_left_down:
                    mouse_release("left")
                    mouse_left_down = False
                if right_active and not mouse_right_down:
                    mouse_press("right")
                    mouse_right_down = True
                if not right_active and mouse_right_down:
                    mouse_release("right")
                    mouse_right_down = False
            else:
                if mouse_left_down:
                    mouse_release("left")
                    mouse_left_down = False
                if mouse_right_down:
                    mouse_release("right")
                    mouse_right_down = False

        seq_hand = seq_input_side if seq_input_side in ("RIGHT", "LEFT") else ("RIGHT" if dominant_is_right else "LEFT")
        seq_last_label = last_R_label if seq_hand == "RIGHT" else last_L_label
        seq_present = bool(right) if seq_hand == "RIGHT" else bool(left)

        if seq_active and seq_present:
            candidate = seq_last_label
            if candidate and candidate not in candidate_ignore:
                prev_candidate = seq_pending
                if (prev_candidate is None) or (
                        prev_candidate != candidate and (now_ms - last_seq_event_ms) >= arm_delay_ms):
                    seq_pending = candidate
                    last_seq_event_ms = now_ms

        confirm_active = False
        if confirm_hand == "RIGHT":
            confirm_active = last_R_label == confirm_gesture
        elif confirm_hand == "LEFT":
            confirm_active = last_L_label == confirm_gesture
        elif confirm_hand == "BOTH":
            confirm_active = (last_R_label == confirm_gesture) and (last_L_label == confirm_gesture)
        else:
            confirm_active = (last_R_label == confirm_gesture) or (last_L_label == confirm_gesture)

        if seq_active and seq_pending and confirm_deb.update(now_ms, confirm_active):
            if len(seq_buffer) < max_len and (now_ms - last_evt_ms) >= arm_delay_ms:
                seq_buffer.append(seq_pending)
                print(f"[SEQ] +{seq_pending}  buffer={seq_buffer}")
                seq_pending = None
                last_evt_ms = now_ms

        if seq_active:
            def _update_undo_for_side(side_name):
                """Handle undo gesture sequence for a specific hand side."""
                state = gR if side_name == "RIGHT" else gL
                if state.pose_flags.get(undo_start, False):
                    undo_open_ts[side_name] = now_ms
                if (
                        state.pose_flags.get(undo_end, False)
                        and undo_open_ts.get(side_name)
                        and (now_ms - undo_open_ts[side_name]) <= undo_window_ms
                ):
                    if seq_buffer:
                        popped = seq_buffer.pop()
                        print(f"[SEQ] UNDO -{popped}  buffer={seq_buffer}")
                    undo_open_ts[side_name] = None

            if undo_hand in ("RIGHT", "LEFT"):
                _update_undo_for_side(undo_hand)
            elif undo_hand in ("EITHER", "ANY"):
                _update_undo_for_side("RIGHT")
                _update_undo_for_side("LEFT")

        commit_active = False
        if commit_hand == "BOTH":
            commit_active = bool(gR.pose_flags.get(commit_gesture, False) and gL.pose_flags.get(commit_gesture, False))
        elif commit_hand == "RIGHT":
            commit_active = bool(gR.pose_flags.get(commit_gesture, False))
        elif commit_hand == "LEFT":
            commit_active = bool(gL.pose_flags.get(commit_gesture, False))
        else:
            commit_active = bool(gR.pose_flags.get(commit_gesture, False) or gL.pose_flags.get(commit_gesture, False))

        if seq_active and commit_deb.update(now_ms, commit_active):
            key_tuple = tuple(seq_buffer)
            combo = lookup_mapping(seq_map, seq_hand, key_tuple) if key_tuple else None
            if combo and (now_ms - last_sent_ms) >= refractory_ms:
                joined = " > ".join(seq_buffer)
                print(f"[SEQ-COMMIT] {seq_hand} {joined} -> {combo}")
                press_combo(combo)
                last_sent_ms = now_ms
            else:
                joined = " > ".join(seq_buffer) if seq_buffer else "—"
                print(f"[SEQ-COMMIT] Нет маппинга для: {seq_hand} {joined}")
            seq_buffer.clear()
            seq_pending = None
            last_evt_ms = now_ms
            if exit_on_commit:
                switch_mode("idle", now_ms, force_reset=True)

        if seq_active:
            top = "REC"
            sub = ("BUF: " + " > ".join(seq_buffer[-6:])) if seq_buffer else "BUF: —"
            if seq_pending:
                sub += f"   |   CAND: {seq_pending}"
            if mouse_active and mouse_enabled:
                sub += f"   |   {mouse_status_label}"
            if one_hand_active and one_enabled:
                sub += f"   |   {one_status_label}"
        else:
            if mouse_active and mouse_enabled:
                top = mouse_status_label
                sub = mouse_active_hint
            elif one_hand_active and one_enabled:
                top = one_status_label
                sub = one_active_hint
                if last_single_action:
                    sub += f"   |   {last_single_action}"
            else:
                top = "IDLE"
                sub = "BUF: —"
        osd.set_text(top, sub)

        if right:
            draw_landmarks(frame, right["lm"])
        if left:
            draw_landmarks(frame, left["lm"])
        if show_fps and fps is not None:
            cv2.putText(
                frame,
                f"FPS: {fps:5.1f}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )
        cv2.imshow("GCPC - Camera", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
