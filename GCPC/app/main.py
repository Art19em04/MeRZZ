import time
from typing import Dict, List

import cv2
from PySide6 import QtWidgets

from app.gestures import GestureState, THUMB_TIP, INDEX_TIP, finger_flexion
from app.os_events_win import mouse_move_normalized, mouse_press, mouse_release, press_combo
from app.osd import OSD
from app.tracker_mediapipe import MediaPipeHandTracker
from app.utils.bindings import (
    binding_notation,
    build_sequence_map,
    build_single_map,
    lookup_mapping,
    merge_single_into_sequences,
    parse_mapping_key,
    parse_sequence_binding,
    parse_single_binding,
    trigger_label,
)
from app.utils.camera import draw_landmarks, open_camera
from app.utils.config import build_hands, load_config, resolve_side, save_config
from app.utils.triggers import DebouncedTrigger


def main():
    cfg = load_config()
    app = QtWidgets.QApplication([])
    osd = OSD()
    osd.show()

    def _dist(a, b):
        dx, dy = a[0] - b[0], a[1] - b[1]
        return (dx * dx + dy * dy) ** 0.5

    def _percentile(values: List[float], fraction: float) -> float | None:
        if not values:
            return None
        srt = sorted(values)
        idx = int((len(srt) - 1) * max(0.0, min(1.0, fraction)))
        return srt[idx]

    def _clamp(val: float, lo: float, hi: float) -> float:
        return max(lo, min(hi, val))

    def _active_pose_name(state: GestureState, fallback: str) -> str:
        for name in ("PINCH", "FIST", "THUMBS_UP", "POINT", "OPEN_PALM", "SWIPE_LEFT", "SWIPE_RIGHT"):
            if state.pose_flags.get(name):
                return name
        return fallback or "—"

    def _draw_pose_label(frame, lm, text, color=(0, 200, 255)):
        if not text or not lm:
            return
        h, w = frame.shape[:2]
        xs = [p[0] * w for p in lm]
        ys = [p[1] * h for p in lm]
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

    vcfg = cfg["video"]
    idx = int(vcfg.get("camera_index", 0))
    w = int(vcfg.get("width", 1280))
    h = int(vcfg.get("height", 720))
    mirror = bool(vcfg.get("mirror", True))
    show_fps = bool(vcfg.get("show_fps", False))

    calib_cfg = cfg.get("calibration", {})
    calibration_enabled = bool(calib_cfg.get("enabled", True))
    calibration_duration_ms = int(calib_cfg.get("duration_ms", 12000))
    calibration_trigger_key = str(calib_cfg.get("trigger_key", "c")).lower()

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

    confirm_cfg = seq_ctrl.get("confirm")
    confirm_binding_value = (
        confirm_cfg.get("binding")
        if isinstance(confirm_cfg, dict)
        else confirm_cfg
    )
    confirm_binding = parse_single_binding(confirm_binding_value, hands)
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
    undo_binding = parse_sequence_binding(undo_binding_value, hands)
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
    commit_binding = parse_single_binding(
        commit_binding_value,
        hands,
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
        elif combo == "MODE_CALIBRATE":
            mode_triggers["calibrate"] = binding

    mode_triggers.setdefault("calibrate", None)
    single_map = build_single_map(single_map_raw, hands)

    seq_map = build_sequence_map(complex_map_raw, hands)

    merge_single_into_sequences(seq_map, single_map)


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

    rect_cfg = mouse_cfg.get("control_rect", {}) or {}
    rect_x = float(rect_cfg.get("x", 0.0) or 0.0)
    rect_y = float(rect_cfg.get("y", 0.0) or 0.0)
    rect_w = float(rect_cfg.get("width", 1.0) or 1.0)
    rect_h = float(rect_cfg.get("height", 1.0) or 1.0)

    def _clamp_rect(x, y, w, h):
        """Ensure control rectangle fits into normalized [0..1] frame."""
        x = max(0.0, min(1.0, x))
        y = max(0.0, min(1.0, y))
        w = max(1e-4, min(1.0 - x, w))
        h = max(1e-4, min(1.0 - y, h))
        return {"x": x, "y": y, "w": w, "h": h}

    mouse_rect = _clamp_rect(rect_x, rect_y, rect_w, rect_h)

    def _read_mouse_binding(key):
        raw_value = mouse_cfg.get(key)
        binding = parse_single_binding(raw_value, hands)
        gesture = binding.get("gesture") or ""
        binding["gesture"] = gesture.upper()
        label = binding_notation(binding, dominant_side, support_side)
        return binding, label

    mouse_left_binding, mouse_left_label = _read_mouse_binding(
        "left_click_binding"
    )
    mouse_right_binding, mouse_right_label = _read_mouse_binding(
        "right_click_binding"
    )
    mouse_active_hint = mouse_cfg.get("active_hint")
    if not mouse_active_hint:
        trig_label = trigger_label(mode_triggers["mouse"], dominant_side, support_side)
        pointer_hint = binding_notation({"hand": pointer_side, "gesture": "POINT"}, dominant_side, support_side)
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

    calibration_active = False
    calibration_start_ms = 0
    calibration_data: Dict[str, List[float]] = {}

    one_hand_active = False
    mouse_active = False
    last_single_action = ""

    current_mode = "idle"
    mode_last_change_ms = 0

    both_pose_latched = {}

    def _new_calibration_data() -> Dict[str, List[float]]:
        return {
            "pinch": [],
            "fist": [],
            "thumbs_thumb": [],
            "thumbs_others": [],
            "point_index": [],
            "point_others": [],
            "swipe_speed": [],
        }

    def _record_calibration(state: GestureState, lm):
        nonlocal calibration_data
        if not calibration_active:
            return
        flex = finger_flexion(lm)
        pinch_d = _dist(lm[THUMB_TIP], lm[INDEX_TIP])
        avg_other = (flex["middle"] + flex["ring"] + flex["pinky"]) / 3.0
        if state.pose_flags.get("PINCH"):
            calibration_data["pinch"].append(pinch_d)
        if state.pose_flags.get("FIST"):
            avg_fist = (flex["index"] + flex["middle"] + flex["ring"] + flex["pinky"]) / 4.0
            calibration_data["fist"].append(avg_fist)
        if state.pose_flags.get("THUMBS_UP"):
            calibration_data["thumbs_thumb"].append(flex["thumb"])
            calibration_data["thumbs_others"].append(avg_other)
        if state.pose_flags.get("POINT"):
            calibration_data["point_index"].append(flex["index"])
            calibration_data["point_others"].append(avg_other)
        if state.pose_flags.get("SWIPE_LEFT") or state.pose_flags.get("SWIPE_RIGHT"):
            if len(state.wrist_hist) >= 2:
                t0, p0 = state.wrist_hist[0]
                t1, p1 = state.wrist_hist[-1]
                dt = max(1e-3, (t1 - t0) / 1000.0)
                vx = abs((p1[0] - p0[0]) / dt)
                calibration_data["swipe_speed"].append(vx)

    def _finalize_calibration(now_ms: int):
        nonlocal calibration_active, cfg
        if not calibration_active:
            return
        calibration_active = False
        ge_cfg = cfg.get("gesture_engine", {})

        def or_default(values: List[float], fraction: float, transform, fallback_key: str):
            val = _percentile(values, fraction)
            if val is None:
                return ge_cfg.get(fallback_key)
            return transform(val)

        pinch_thr = or_default(
            calibration_data["pinch"],
            0.9,
            lambda v: _clamp(v * 1.1, 0.01, 0.2),
            "pinch_threshold",
        )
        fist_thr = or_default(
            calibration_data["fist"],
            0.25,
            lambda v: _clamp(v * 0.9, 0.15, 0.95),
            "fist_threshold",
        )
        thumbs_thumb = or_default(
            calibration_data["thumbs_thumb"],
            0.8,
            lambda v: _clamp(v * 1.1, 0.05, 0.8),
            "thumbs_up_thumb_max_flex",
        )
        thumbs_other = or_default(
            calibration_data["thumbs_others"],
            0.2,
            lambda v: _clamp(v * 0.9, 0.3, 0.95),
            "thumbs_up_others_min_flex",
        )
        point_idx = or_default(
            calibration_data["point_index"],
            0.8,
            lambda v: _clamp(v * 1.1, 0.05, 0.7),
            "point_index_max_flex",
        )
        point_others = or_default(
            calibration_data["point_others"],
            0.2,
            lambda v: _clamp(v * 0.9, 0.3, 0.95),
            "point_others_min_flex",
        )
        swipe_thr = or_default(
            calibration_data["swipe_speed"],
            0.5,
            lambda v: int(_clamp(v * 800 * 0.9, 200, 2000)),
            "swipe_speed_px",
        )

        updates = {
            "pinch_threshold": pinch_thr,
            "fist_threshold": fist_thr,
            "thumbs_up_thumb_max_flex": thumbs_thumb,
            "thumbs_up_others_min_flex": thumbs_other,
            "point_index_max_flex": point_idx,
            "point_others_min_flex": point_others,
            "swipe_speed_px": swipe_thr,
        }
        ge_cfg.update(updates)
        cfg["gesture_engine"] = ge_cfg
        save_config(cfg)
        print(f"[CALIBRATION] Updated gesture thresholds at {now_ms} ms: {updates}")

    def switch_mode(new_mode, now_ms, force_reset=False):
        nonlocal current_mode, seq_active, one_hand_active, mouse_active
        nonlocal mouse_prev, mouse_left_down, mouse_right_down
        nonlocal last_single_action, seq_buffer, seq_pending, last_evt_ms, mode_last_change_ms
        nonlocal calibration_active, calibration_start_ms, calibration_data

        prev = current_mode

        if new_mode == prev and not force_reset:
            return

        # --- sequence ---
        if prev == "record" or force_reset:
            seq_active = False
            seq_buffer.clear()
            seq_pending = None

        if new_mode == "record":
            seq_active = True
            seq_buffer.clear()
            seq_pending = None
            last_evt_ms = now_ms
        else:
            seq_active = False

        # --- mouse ---
        if prev == "mouse" or force_reset:
            if mouse_left_down:
                mouse_release("left")
                mouse_left_down = False
            if mouse_right_down:
                mouse_release("right")
                mouse_right_down = False

        if new_mode == "mouse" and mouse_enabled:
            mouse_prev = None
            mouse_active = True
        else:
            mouse_prev = None
            mouse_active = False

        # --- calibration ---
        if prev == "calibrate" or force_reset:
            calibration_active = False

        if new_mode == "calibrate" and calibration_enabled:
            calibration_active = True
            calibration_start_ms = now_ms
            calibration_data = _new_calibration_data()

        # --- one-hand ---
        if new_mode != "one_hand":
            last_single_action = ""

        one_hand_active = (new_mode == "one_hand") and one_enabled

        current_mode = new_mode
        mode_last_change_ms = now_ms

        labels = {
            "idle": "IDLE",
            "record": "RECORD",
            "mouse": mouse_status_label,
            "one_hand": one_status_label,
            "calibrate": "CALIBRATION",
        }
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
            _record_calibration(gR, right["lm"])
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
            _record_calibration(gL, left["lm"])
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
        elif calibration_enabled and mode_triggers.get("calibrate") and _trigger_fired(mode_triggers["calibrate"]) and time_since_change >= mode_refractory_ms:
            switch_mode("calibrate", now_ms, force_reset=True)
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
                target = (
                    (tip[0] - mouse_rect["x"]) / mouse_rect["w"],
                    (tip[1] - mouse_rect["y"]) / mouse_rect["h"],
                )
                if mouse_prev is None or mouse_smooth <= 0.0:
                    smoothed = target
                else:
                    prev_x, prev_y = mouse_prev
                    smoothed = (
                        prev_x + (target[0] - prev_x) * (1.0 - mouse_smooth),
                        prev_y + (target[1] - prev_y) * (1.0 - mouse_smooth),
                    )
                mx = max(0.0, min(1.0, smoothed[0]))
                my = max(0.0, min(1.0, smoothed[1]))
                mouse_prev = (mx, my)
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

        if calibration_active and (now_ms - calibration_start_ms) >= calibration_duration_ms:
            _finalize_calibration(now_ms)
            switch_mode("idle", now_ms, force_reset=True)

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

        if calibration_active:
            elapsed = now_ms - calibration_start_ms
            remaining = max(0, calibration_duration_ms - elapsed)
            top = "CALIBRATION"
            sub = f"Показывайте разные жесты | Осталось: {remaining / 1000:.1f}с"
        elif seq_active:
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
            label = _active_pose_name(gR, last_R_label)
            _draw_pose_label(frame, right["lm"], f"R: {label}", (0, 200, 255))
        if left:
            draw_landmarks(frame, left["lm"])
            label = _active_pose_name(gL, last_L_label)
            _draw_pose_label(frame, left["lm"], f"L: {label}", (0, 255, 180))
        if mouse_active:
            fh, fw = frame.shape[:2]
            tl = (int(mouse_rect["x"] * fw), int(mouse_rect["y"] * fh))
            br = (
                int((mouse_rect["x"] + mouse_rect["w"]) * fw),
                int((mouse_rect["y"] + mouse_rect["h"]) * fh),
            )
            cv2.rectangle(frame, tl, br, (0, 255, 255), 2)
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
        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            break
        if calibration_enabled and chr(key).lower() == calibration_trigger_key:
            switch_mode("calibrate", now_ms, force_reset=True)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
