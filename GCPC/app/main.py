import csv
import statistics
import time
import uuid
from typing import Dict, List

import cv2
from PySide6 import QtCore, QtGui, QtWidgets

from app.gestures import (
    GestureState,
    INDEX_MCP,
    INDEX_TIP,
    MIDDLE_MCP,
    PINKY_MCP,
    THUMB_TIP,
    WRIST,
    finger_flexion,
)
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
from app.utils.config import APP_DIR, build_hands, load_config, resolve_side, save_config
from app.utils.triggers import DebouncedTrigger


METRIC_FIELDS = [
    "session_id",
    "backend",
    "providers",
    "FPS_avg",
    "e2e_p50",
    "e2e_p95",
    "interaction",
    "scenario",
    "task_time_ms",
    "task_success",
    "false_activations",
]


def _append_metrics_row(row):
    """Persist a single metrics row to ``metrics.csv`` with a shared header."""
    path = APP_DIR.parent / "metrics.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=METRIC_FIELDS)
        if not exists:
            writer.writeheader()
        writer.writerow({key: row.get(key, "") for key in METRIC_FIELDS})


class ControlPanel(QtWidgets.QWidget):
    """Tiny always-on-top widget for interaction mode and armed toggles."""

    def __init__(self):
        super().__init__(None, QtCore.Qt.Tool | QtCore.Qt.WindowStaysOnTopHint)
        self.setWindowTitle("GCPC Controls")
        self.interaction_mode = "gestures"
        self.armed = True

        layout = QtWidgets.QVBoxLayout(self)
        self.interaction_btn = QtWidgets.QPushButton(self._interaction_label())
        self.interaction_btn.clicked.connect(self._toggle_interaction)
        self.armed_btn = QtWidgets.QPushButton(self._armed_label())
        self.armed_btn.setCheckable(True)
        self.armed_btn.setChecked(True)
        self.armed_btn.clicked.connect(self._toggle_armed)

        layout.addWidget(self.interaction_btn)
        layout.addWidget(self.armed_btn)

    def _interaction_label(self):
        mode = "GESTURES" if self.interaction_mode == "gestures" else "MK"
        return f"Interaction: {mode}"

    def _toggle_interaction(self):
        self.interaction_mode = "mk" if self.interaction_mode == "gestures" else "gestures"
        self.interaction_btn.setText(self._interaction_label())

    def _armed_label(self):
        return "Armed" if self.armed else "Disarmed"

    def _toggle_armed(self):
        self.armed = self.armed_btn.isChecked()
        self.armed_btn.setText(self._armed_label())

    def current_interaction(self) -> str:
        return self.interaction_mode

    def is_armed(self) -> bool:
        return self.armed


def main():
    cfg = load_config()
    app = QtWidgets.QApplication([])
    osd = OSD()
    osd.show()

    ui_cfg = cfg.get("ui", {})
    panel_cfg = ui_cfg.get("controls_panel", {})
    panel_enabled = bool(panel_cfg.get("enabled", True))
    start_scenario_key = str(panel_cfg.get("start_scenario_shortcut", "Alt+S"))
    end_scenario_key = str(panel_cfg.get("end_scenario_shortcut", "Alt+E"))

    panel = ControlPanel() if panel_enabled else None
    if panel:
        panel.show()

    session_id = uuid.uuid4().hex
    latencies: List[float] = []
    frames_processed = 0
    last_frame_capture_ns: int | None = None
    session_start_ns = time.perf_counter_ns()
    false_activations_total = 0
    scenario_active = False
    scenario_name = ""
    scenario_start_ns: int | None = None
    scenario_false_baseline = 0
    def _interaction_mode():
        return panel.current_interaction() if panel else "gestures"

    def _armed_state():
        return panel.is_armed() if panel else True

    scenario_interaction = _interaction_mode()

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
        for name in ("PINCH", "FIST", "THUMBS_UP", "OPEN_PALM"):
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

    def _split_duration(total_ms: int, parts: int) -> List[int]:
        base = total_ms // parts
        res = [base for _ in range(parts)]
        for i in range(total_ms - base * parts):
            res[i % parts] += 1
        return res

    stage_defs = [
        {"name": "PINCH", "hint": "Сомкните большой+указательный"},
        {"name": "FIST", "hint": "Сожмите кулак"},
        {"name": "THUMBS_UP", "hint": "Большой вверх, остальные согнуты"},
        {"name": "OPEN_PALM", "hint": "Разожмите ладонь и выпрямите пальцы"},
        {"name": "SWIPE_RIGHT", "hint": "Проведите ладонью вправо"},
        {"name": "SWIPE_LEFT", "hint": "Проведите ладонью влево"},
    ]
    stage_durations = _split_duration(calibration_duration_ms, len(stage_defs))
    calibration_stages = [
        stage_defs[i] | {"dur_ms": stage_durations[i]}
        for i in range(len(stage_defs))
    ]
    calibration_total_ms = sum(stage["dur_ms"] for stage in calibration_stages)

    cap = open_camera(idx, w, h)

    dcfg = cfg.get("detector", {})
    tracker = MediaPipeHandTracker(
        min_det=dcfg.get("min_detection_confidence"),
        min_trk=dcfg.get("min_tracking_confidence"),
        max_hands=dcfg.get("max_num_hands"),
        model_complexity=dcfg.get("model_complexity"),
    )

    providers = getattr(tracker, "providers", []) or []
    backend_kind = "gpu" if any("CUDA" in p.upper() or "GPU" in p.upper() for p in providers) else "cpu"
    providers_str = str(providers)
    print(f"[SESSION] id={session_id} backend={backend_kind} providers={providers}")
    _append_metrics_row(
        {
            "session_id": session_id,
            "backend": backend_kind,
            "providers": providers_str,
        }
    )

    def _session_summary_row(final_ns: int | None = None) -> Dict[str, str | float]:
        """Prepare a metrics row describing session-level performance."""
        end_ns = final_ns or time.perf_counter_ns()
        duration_s = max(1e-9, (end_ns - session_start_ns) / 1e9)
        fps_avg = frames_processed / duration_s if frames_processed else 0.0
        e2e_p50 = statistics.median(latencies) if latencies else None
        e2e_p95 = _percentile(latencies, 0.95)
        return {
            "session_id": session_id,
            "backend": backend_kind,
            "providers": providers_str,
            "FPS_avg": round(fps_avg, 3),
            "e2e_p50": round(e2e_p50, 3) if e2e_p50 is not None else "",
            "e2e_p95": round(e2e_p95, 3) if e2e_p95 is not None else "",
        }

    def _start_scenario():
        nonlocal scenario_active, scenario_name, scenario_start_ns, scenario_false_baseline, scenario_interaction
        if scenario_active:
            print("[SCENARIO] Already active; end current scenario first.")
            return
        text, ok = QtWidgets.QInputDialog.getText(
            osd, "Start scenario", "Scenario name:", text=scenario_name or "",
        )
        if not ok or not text.strip():
            return
        scenario_name = text.strip()
        scenario_interaction = _interaction_mode()
        scenario_start_ns = time.perf_counter_ns()
        scenario_false_baseline = false_activations_total
        scenario_active = True
        print(f"[SCENARIO] START name={scenario_name!r} interaction={scenario_interaction}")

    def _end_scenario():
        nonlocal scenario_active, scenario_start_ns
        if not scenario_active:
            print("[SCENARIO] No active scenario to end.")
            return
        choice, ok = QtWidgets.QInputDialog.getItem(
            osd,
            "End scenario",
            "Success? (1=yes, 0=no)",
            ["1", "0"],
            0,
            False,
        )
        if not ok:
            return
        now_ns = time.perf_counter_ns()
        elapsed_ms = (now_ns - (scenario_start_ns or now_ns)) / 1e6
        task_success = int(choice)
        false_in_scenario = false_activations_total - scenario_false_baseline
        _append_metrics_row(
            {
                "session_id": session_id,
                "backend": backend_kind,
                "providers": providers_str,
                "interaction": scenario_interaction,
                "scenario": scenario_name,
                "task_time_ms": round(elapsed_ms, 3),
                "task_success": task_success,
                "false_activations": false_in_scenario,
            }
        )
        scenario_active = False
        scenario_start_ns = None
        print(
            f"[SCENARIO] END name={scenario_name!r} success={task_success} "
            f"time_ms={elapsed_ms:.3f} false={false_in_scenario}"
        )

    def _send_hotkey(combo: str):
        nonlocal false_activations_total
        now_ns = time.perf_counter_ns()
        if last_frame_capture_ns is not None:
            latencies.append((now_ns - last_frame_capture_ns) / 1e6)
        armed = _armed_state()
        active = scenario_active
        if not (active and armed):
            false_activations_total += 1
        press_combo(combo)

    shortcut_parent = panel if panel else osd
    start_shortcut = QtGui.QShortcut(QtGui.QKeySequence(start_scenario_key), shortcut_parent)
    start_shortcut.setContext(QtCore.Qt.ApplicationShortcut)
    start_shortcut.activated.connect(_start_scenario)
    end_shortcut = QtGui.QShortcut(QtGui.QKeySequence(end_scenario_key), shortcut_parent)
    end_shortcut.setContext(QtCore.Qt.ApplicationShortcut)
    end_shortcut.activated.connect(_end_scenario)

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
        pointer_hint = binding_notation({"hand": pointer_side, "gesture": "PINCH"}, dominant_side, support_side)
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
    calibration_stage_ms = 0
    calibration_stage_idx = -1
    calibration_data: Dict[str, List[float]] = {}
    calibration_motion_prev: Dict[str, tuple[int, tuple[float, float]] | None] = {
        "RIGHT": None,
        "LEFT": None,
    }

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
            "open": [],
            "swipe_right_dx": [],
            "swipe_left_dx": [],
            "swipe_speed": [],
            "swipe_ratio": [],
        }

    def _current_calibration_stage():
        if 0 <= calibration_stage_idx < len(calibration_stages):
            return calibration_stages[calibration_stage_idx]
        return None

    def _begin_calibration_stage(idx: int, now_ms: int):
        nonlocal calibration_stage_idx, calibration_stage_ms
        calibration_stage_idx = idx
        calibration_stage_ms = now_ms
        calibration_motion_prev["RIGHT"] = None
        calibration_motion_prev["LEFT"] = None
        stage = _current_calibration_stage()
        _announce_stage(stage)

    def _announce_stage(stage):
        if not stage:
            return
        idx = calibration_stage_idx if calibration_stage_idx >= 0 else calibration_stages.index(stage)
        print(
            f"[CALIBRATION] Stage {idx + 1}/{len(calibration_stages)}: "
            f"{stage['name']} ({stage['dur_ms']} ms) — {stage['hint']}"
        )

    def _palm_span(lm) -> float:
        span = _dist(lm[INDEX_MCP], lm[PINKY_MCP])
        anchor = _dist(lm[WRIST], lm[MIDDLE_MCP])
        return max(span, anchor, 1e-4)

    def _record_calibration(stage_name: str | None, lm, side: str | None):
        nonlocal calibration_data
        if not calibration_active or not stage_name:
            return
        now_ms = int(time.time() * 1000)
        flex = finger_flexion(lm)
        pinch_d = _dist(lm[THUMB_TIP], lm[INDEX_TIP])
        span = _palm_span(lm)
        avg_other = (flex["middle"] + flex["ring"] + flex["pinky"]) / 3.0
        prev = None
        if side:
            prev = calibration_motion_prev.get(side)
            calibration_motion_prev[side] = (now_ms, lm[WRIST])

        if stage_name == "PINCH":
            if pinch_d / span < 0.8 and flex["index"] > 0.12 and flex["thumb"] > 0.12:
                calibration_data["pinch"].append(pinch_d)
        elif stage_name == "FIST":
            avg_fist = (flex["index"] + flex["middle"] + flex["ring"] + flex["pinky"]) / 4.0
            if avg_fist > 0.35:
                calibration_data["fist"].append(avg_fist)
        elif stage_name == "THUMBS_UP":
            if flex["thumb"] < 0.5 and avg_other > 0.35:
                calibration_data["thumbs_thumb"].append(flex["thumb"])
                calibration_data["thumbs_others"].append(avg_other)
        elif stage_name == "OPEN_PALM":
            max_open = max(flex["index"], flex["middle"], flex["ring"], flex["pinky"])
            if max_open < 0.55 and pinch_d / span > 0.9:
                calibration_data["open"].append(max_open)
        elif stage_name in {"SWIPE_RIGHT", "SWIPE_LEFT"} and side and prev:
            prev_ts, prev_pt = prev
            dt = now_ms - prev_ts
            if dt > 0:
                dx = lm[WRIST][0] - prev_pt[0]
                dy = lm[WRIST][1] - prev_pt[1]
                ratio = abs(dy) / max(abs(dx), 1e-4)
                speed = dx / (dt / 1000.0)
                if stage_name == "SWIPE_RIGHT" and dx > 0:
                    calibration_data["swipe_right_dx"].append(dx)
                    calibration_data["swipe_speed"].append(speed)
                    calibration_data["swipe_ratio"].append(ratio)
                elif stage_name == "SWIPE_LEFT" and dx < 0:
                    calibration_data["swipe_left_dx"].append(-dx)
                    calibration_data["swipe_speed"].append(-speed)
                    calibration_data["swipe_ratio"].append(ratio)

    def _advance_calibration_stage(now_ms: int) -> bool:
        nonlocal calibration_stage_idx, calibration_stage_ms
        if not calibration_active:
            return False
        while True:
            stage = _current_calibration_stage()
            if not stage:
                return False
            elapsed = now_ms - calibration_stage_ms
            if elapsed < stage["dur_ms"]:
                return False
            next_idx = calibration_stage_idx + 1
            if next_idx >= len(calibration_stages):
                _finalize_calibration(now_ms)
                return True
            _begin_calibration_stage(next_idx, now_ms)

    def _finalize_calibration(now_ms: int):
        nonlocal calibration_active, cfg, calibration_stage_idx
        if not calibration_active:
            return
        calibration_active = False
        ge_cfg = cfg.get("gesture_engine", {})

        # Reset stage index so a new calibration starts from the first stage label.
        calibration_stage_idx = -1

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
        open_max = or_default(
            calibration_data["open"],
            0.9,
            lambda v: _clamp(v * 1.05, 0.1, 0.7),
            "open_palm_max_flex",
        )
        swipe_min_dx = or_default(
            calibration_data["swipe_right_dx"] + calibration_data["swipe_left_dx"],
            0.6,
            lambda v: _clamp(v * 0.8, 0.02, 0.5),
            "swipe_min_dx",
        )
        swipe_min_speed = or_default(
            calibration_data["swipe_speed"],
            0.4,
            lambda v: _clamp(v * 0.8, 0.15, 3.0),
            "swipe_min_speed",
        )
        swipe_ratio_max = or_default(
            calibration_data["swipe_ratio"],
            0.8,
            lambda v: _clamp(v * 1.2, 0.05, 1.5),
            "swipe_max_dy_ratio",
        )
        updates = {
            "pinch_threshold": pinch_thr,
            "fist_threshold": fist_thr,
            "thumbs_up_thumb_max_flex": thumbs_thumb,
            "thumbs_up_others_min_flex": thumbs_other,
            "open_palm_max_flex": open_max,
            "swipe_min_dx": swipe_min_dx,
            "swipe_min_speed": swipe_min_speed,
            "swipe_max_dy_ratio": swipe_ratio_max,
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
            _begin_calibration_stage(0, now_ms)

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
        frame_capture_ns = time.perf_counter_ns()
        last_frame_capture_ns = frame_capture_ns
        frames_processed += 1

        now = time.time()
        dt = now - last_frame_time
        last_frame_time = now
        now_ms = int(now * 1000)
        finished_calibration = False
        if calibration_active:
            finished_calibration = _advance_calibration_stage(now_ms)
        if finished_calibration:
            switch_mode("idle", now_ms, force_reset=True)
        current_stage = _current_calibration_stage() if calibration_active else None
        current_stage_name = current_stage["name"] if current_stage else None
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
            _record_calibration(current_stage_name, right["lm"], "RIGHT")
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
            _record_calibration(current_stage_name, left["lm"], "LEFT")
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

        if not calibration_active:
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
                    _send_hotkey(combo)
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
                        _send_hotkey(combo)
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
                _send_hotkey(combo)
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
            stage = current_stage
            stage_elapsed = now_ms - calibration_stage_ms
            stage_remaining = max(0, (stage["dur_ms"] if stage else 0) - stage_elapsed)
            total_elapsed = now_ms - calibration_start_ms
            total_remaining = max(0, calibration_total_ms - total_elapsed)
            stage_name = stage["name"] if stage else "DONE"
            hint = stage["hint"] if stage else "Калибровка завершена"
            top = f"CAL {calibration_stage_idx + 1}/{len(calibration_stages)}: {stage_name}"
            sub = (
                f"{hint} | Этап: {stage_remaining / 1000:.1f}с | "
                f"Всего: {total_remaining / 1000:.1f}с"
            )
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
        raw_key = cv2.waitKey(1)
        key = raw_key & 0xFF if raw_key != -1 else -1
        if raw_key == 27 or key == 27:
            break
        if calibration_enabled and key != -1:
            try:
                key_chr = chr(key).lower()
            except ValueError:
                key_chr = ""
            if key_chr == calibration_trigger_key:
                switch_mode("calibrate", now_ms, force_reset=True)

    _append_metrics_row(_session_summary_row())
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
