# -*- coding: utf-8 -*-
import os
import json
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
    with open(os.path.join(ROOT, "config.json"), "r", encoding="utf-8") as f:
        return json.load(f)


def _normalize_hand_choice(value, fallback):
    if value is None:
        return fallback
    token = str(value).strip().lower()
    if token in ("right", "r"):
        return "RIGHT"
    if token in ("left", "l"):
        return "LEFT"
    return fallback


def build_hands(cfg):
    hands_cfg = cfg.get("hands", {}) if isinstance(cfg.get("hands"), dict) else {}
    dominant_raw = hands_cfg.get("dominant", cfg.get("dominant_hand", "right"))
    dominant = _normalize_hand_choice(dominant_raw, "RIGHT")
    support = _normalize_hand_choice(hands_cfg.get("support"), None)
    if support not in ("LEFT", "RIGHT"):
        support = "LEFT" if dominant == "RIGHT" else "RIGHT"
    return {"dominant": dominant, "support": support}


def resolve_side(tag, hands):
    dominant_side = hands.get("dominant", "RIGHT")
    support_side = hands.get("support") or ("LEFT" if dominant_side == "RIGHT" else "RIGHT")
    if tag is None:
        return dominant_side
    tag = str(tag).strip().upper()
    if not tag:
        return dominant_side
    if tag in ("RIGHT", "LEFT", "BOTH", "EITHER", "ANY"):
        return tag
    if tag in ("DOMINANT", "MAIN", "PRIMARY"):
        return dominant_side
    if tag in ("NON_DOMINANT", "NON-DOMINANT", "OFFHAND", "SUPPORT", "SECONDARY", "AUXILIARY", "FUNCTIONAL"):
        return support_side
    if tag in ("OPPOSITE", "OTHER"):
        return support_side if support_side != dominant_side else dominant_side
    return tag or dominant_side


def parse_mapping_key(raw_key, hands):
    if not isinstance(raw_key, str):
        return None
    steps = [step.strip() for step in raw_key.split(">")]
    gestures = []
    side = None
    for idx, step in enumerate(steps):
        if not step:
            continue
        tokens = [tok.strip().upper() for tok in step.split("-") if tok.strip()]
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
    if not table:
        return None
    order = []
    if side:
        order.append(side)
    if side != "EITHER":
        order.append("EITHER")
    order.append("ANY")
    seen = set()
    for cand in order:
        if cand in seen:
            continue
        seen.add(cand)
        bucket = table.get(cand)
        if bucket and key in bucket:
            return bucket[key]
    return None


class DebouncedTrigger:
    def __init__(self, dwell_ms=260, refractory_ms=900):
        self.dwell_ms = dwell_ms
        self.refractory_ms = refractory_ms
        self.candidate_since = None
        self.last_fire = 0

    def update(self, now_ms, active):
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
    def try_open(i, api):
        cap = cv2.VideoCapture(i, api)
        if cap.isOpened():
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
            ok, _ = cap.read()
            if ok:
                return cap
            cap.release()
        return None

    for api in (cv2.CAP_MSMF, cv2.CAP_DSHOW, cv2.CAP_ANY):
        cap = try_open(idx, api)
        if cap:
            print(f"[videoio] open idx={idx} api={api}")
            return cap
    for probe in range(0, 6):
        for api in (cv2.CAP_MSMF, cv2.CAP_DSHOW, cv2.CAP_ANY):
            cap = try_open(probe, api)
            if cap:
                print(f"[videoio] open idx={probe} api={api}")
                return cap
    return None


def draw_landmarks(frame, lm):
    h, w = frame.shape[:2]
    for (x, y) in lm:
        cv2.circle(frame, (int(x * w), int(y * h)), 3, (0, 255, 0), -1)


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
    if cap is None:
        raise RuntimeError("Не удалось открыть камеру")

    dcfg = cfg.get("detector", {})
    tracker = MediaPipeHandTracker(
        min_det=dcfg.get("min_detection_confidence", 0.6),
        min_trk=dcfg.get("min_tracking_confidence", 0.5),
        max_hands=dcfg.get("max_num_hands", 2),
        model_complexity=dcfg.get("model_complexity", 1),
    )

    gR = GestureState(cfg["gesture_engine"])
    gL = GestureState(cfg["gesture_engine"])

    seq_cfg = cfg.get("sequence", {})
    arm_delay_ms = int(seq_cfg.get("arm_delay_ms", 420))
    refractory_ms = int(seq_cfg.get("refractory_ms", 1100))
    cancel_exit_ms = int(seq_cfg.get("cancel_on_hand_exit_ms", 900))
    auto_exit = bool(seq_cfg.get("auto_exit_on_hand_exit", True))
    max_len = int(seq_cfg.get("max_len", 6))

    hands = build_hands(cfg)
    dominant_side = hands["dominant"]
    support_side = hands["support"]
    dominant_is_right = dominant_side == "RIGHT"

    controls_cfg = cfg.get("controls", {})
    seq_ctrl = controls_cfg.get("sequence", {})
    seq_input_side = resolve_side(seq_ctrl.get("input_hand", "dominant"), hands)
    candidate_ignore = {str(g).upper() for g in seq_ctrl.get("candidate_ignore", ["OPEN_PALM", "FIST"])}

    confirm_cfg = seq_ctrl.get("confirm", {})
    confirm_hand = resolve_side(confirm_cfg.get("hand", "non_dominant"), hands)
    confirm_gesture = str(confirm_cfg.get("gesture", "PINCH_TAP")).upper()
    confirm_deb = DebouncedTrigger(
        int(confirm_cfg.get("dwell_ms", 220)),
        int(confirm_cfg.get("refractory_ms", 700)),
    )

    undo_cfg = seq_ctrl.get("undo", {})
    undo_hand = resolve_side(undo_cfg.get("hand", "non_dominant"), hands)
    undo_start = str(undo_cfg.get("start_gesture", "OPEN_PALM")).upper()
    undo_end = str(undo_cfg.get("end_gesture", "FIST")).upper()
    undo_window_ms = int(undo_cfg.get("window_ms", 900))

    commit_cfg = seq_ctrl.get("commit", {})
    commit_hand = resolve_side(commit_cfg.get("hand", "both"), hands)
    commit_gesture = str(commit_cfg.get("gesture", "FIST")).upper()
    commit_deb = DebouncedTrigger(
        int(commit_cfg.get("dwell_ms", 260)),
        int(commit_cfg.get("refractory_ms", 1200)),
    )
    exit_on_commit = bool(commit_cfg.get("exit_on_commit", seq_cfg.get("exit_on_commit", True)))

    cmd_map = cfg.get("command_mappings", {})
    single_map_raw = cmd_map.get("single_gestures", {})
    complex_map_raw = cmd_map.get("complex_gestures", {})

    single_map = {}
    for raw_key, combo in single_map_raw.items():
        parsed = parse_mapping_key(raw_key, hands)
        if not parsed:
            continue
        side, gestures = parsed
        if not gestures:
            continue
        bucket = single_map.setdefault(side, {})
        bucket[gestures[0]] = combo

    seq_map = {}
    for raw_key, combo in complex_map_raw.items():
        parsed = parse_mapping_key(raw_key, hands)
        if not parsed:
            continue
        side, gestures = parsed
        if not gestures:
            continue
        bucket = seq_map.setdefault(side, {})
        bucket[tuple(gestures)] = combo
    # allow single gesture sequences as well
    for side, bucket in single_map.items():
        seq_bucket = seq_map.setdefault(side, {})
        for gesture, combo in bucket.items():
            seq_bucket.setdefault((gesture,), combo)

    mode_cfg = cfg.get("mode_switches", {})
    mode_refractory_ms = int(mode_cfg.get("refractory_ms", 800))

    def _parse_trigger(name, default_hand, default_gesture):
        entry = mode_cfg.get(name, {})
        if isinstance(entry, str):
            entry = {"gesture": entry}
        result = {"hand": default_hand, "gesture": default_gesture}
        if isinstance(entry, dict):
            if "hand" in entry and entry["hand"] is not None:
                result["hand"] = entry["hand"]
            if "gesture" in entry and entry["gesture"] is not None:
                result["gesture"] = str(entry["gesture"]).upper()
        result["hand"] = resolve_side(result["hand"], hands)
        result["gesture"] = result["gesture"].upper()
        return result

    mode_triggers = {
        "one_hand": _parse_trigger("one_hand", "non_dominant", "FIST"),
        "record": _parse_trigger("record", "both", "OPEN_PALM"),
        "mouse": _parse_trigger("mouse", "non_dominant", "THUMBS_UP"),
        "exit": _parse_trigger("exit", "non_dominant", "SWIPE_LEFT"),
    }

    def _trigger_label(trig):
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

    one_cfg = cfg.get("one_hand_mode", {})
    one_enabled = bool(one_cfg.get("enabled", True))
    one_status_label = one_cfg.get("status_label") or "ONE-HAND"
    one_active_hint = one_cfg.get("active_hint")
    dispatch_side = resolve_side(one_cfg.get("dispatch_hand", "dominant"), hands)
    if not one_active_hint:
        trig_label = _trigger_label(mode_triggers["one_hand"])
        one_active_hint = f"{trig_label} → SINGLE" if trig_label else "ONE-HAND"

    mouse_cfg = cfg.get("mouse_control", {})
    mouse_enabled = bool(mouse_cfg.get("enabled", True))
    mouse_status_label = mouse_cfg.get("status_label") or "MOUSE"
    mouse_smooth = max(0.0, min(1.0, float(mouse_cfg.get("smoothing_alpha", 0.25))))
    pointer_side = resolve_side(mouse_cfg.get("pointer_hand", "dominant"), hands)
    if pointer_side not in ("RIGHT", "LEFT"):
        pointer_side = dominant_side
    pointer_landmark = int(mouse_cfg.get("pointer_landmark", 8))
    if pointer_landmark < 0 or pointer_landmark > 20:
        pointer_landmark = 8
    mouse_left_pose = str(mouse_cfg.get("left_click_pose", "FIST")).upper()
    mouse_right_pose = str(mouse_cfg.get("right_click_pose", "OPEN_PALM")).upper()
    mouse_active_hint = mouse_cfg.get("active_hint")
    if not mouse_active_hint:
        trig_label = _trigger_label(mode_triggers["mouse"])
        pointer_txt = pointer_side
        mouse_active_hint = (
            f"{trig_label} | CURSOR: {pointer_txt} POINT | LMB: {mouse_left_pose} | RMB: {mouse_right_pose}"
            if trig_label
            else "CURSOR CONTROL"
        )

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
            if seq_active and seq_input_side == "RIGHT" and cancel_exit_ms > 0 and (now_ms - last_seen_R) >= cancel_exit_ms:
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
            if seq_active and seq_input_side == "LEFT" and cancel_exit_ms > 0 and (now_ms - last_seen_L) >= cancel_exit_ms:
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
            nonlocal both_pose_latched
            gesture = trig.get("gesture")
            hand = trig.get("hand")
            if not gesture:
                return False
            hand = (hand or "").upper()
            if not hand:
                return False
            if hand == "BOTH":
                active = bool(right and left and gR.pose_flags.get(gesture, False) and gL.pose_flags.get(gesture, False))
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
            else:
                switch_mode("record", now_ms)
        elif mouse_enabled and _trigger_fired(mode_triggers["mouse"]) and time_since_change >= mode_refractory_ms and current_mode != "mouse":
            switch_mode("mouse", now_ms)
        elif one_enabled and _trigger_fired(mode_triggers["one_hand"]) and time_since_change >= mode_refractory_ms and current_mode != "one_hand":
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
            pointer_state = gR if pointer_side == "RIGHT" else gL
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
                left_active = bool(pointer_state.pose_flags.get(mouse_left_pose, False))
                right_active = bool(pointer_state.pose_flags.get(mouse_right_pose, False))
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
                if (prev_candidate is None) or (prev_candidate != candidate and (now_ms - last_seq_event_ms) >= arm_delay_ms):
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

