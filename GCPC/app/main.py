# -*- coding: utf-8 -*-
import copy
import logging
import statistics
import time
import uuid
from typing import Any, Dict, List, Mapping

import cv2
from PySide6 import QtCore, QtGui, QtWidgets

from app.gestures import GestureState, WRIST
from app.os_events_win import (
    mouse_move_normalized,
    mouse_press,
    mouse_release,
    mouse_scroll,
    press_combo,
)
from app.osd import OSD
from app.services.calibration import CalibrationSession
from app.services.csv_metrics import append_metrics_row
from app.services.eval_single import EvalSingleSession
from app.services.handedness import HandednessResolver
from app.services.one_hand import OneHandCommandDispatcher
from app.services.rendering import (
    active_pose_name,
    draw_pose_label,
    hand_crop,
    render_hand_window,
)
from app.settings_dialog import GestureSettingsDialog
from app.tracker_mediapipe import MediaPipeHandTracker
from app.ui.control_panel import ControlPanel
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
from app.utils.runtime import install_exception_hooks, report_fatal_exception, setup_logging
from app.utils.triggers import DebouncedTrigger

LOGGER = logging.getLogger(__name__)


def _percentile(values: List[float], fraction: float) -> float | None:
    if not values:
        return None
    sorted_values = sorted(values)
    index = int((len(sorted_values) - 1) * max(0.0, min(1.0, fraction)))
    return sorted_values[index]


def _optional_int(value) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _binding_active(
    binding: dict | None,
    dominant_side: str,
    support_side: str,
    g_right: GestureState,
    g_left: GestureState,
    right_present: bool,
    left_present: bool,
    event_right: str,
    event_left: str,
) -> bool:
    """Check whether gesture binding is satisfied given hand presence and events."""
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


def _clamp_rect(x: float, y: float, w_rect: float, h_rect: float):
    x = max(0.0, min(1.0, x))
    y = max(0.0, min(1.0, y))
    w_rect = max(1e-4, min(1.0 - x, w_rect))
    h_rect = max(1e-4, min(1.0 - y, h_rect))
    return {"x": x, "y": y, "w": w_rect, "h": h_rect}


def _build_mode_triggers(
    functional_raw: Mapping[str, Any],
    hands: Mapping[str, str],
) -> tuple[int, int, Dict[str, dict | None]]:
    mode_refractory_ms = int(functional_raw.get("refractory_ms", 800))
    exit_hold_ms = int(functional_raw.get("exit_hold_ms", 500))
    mode_triggers: Dict[str, dict | None] = {}

    for raw_key, combo in functional_raw.items():
        if not isinstance(combo, str) or not combo.startswith("MODE_"):
            continue
        parsed = parse_mapping_key(raw_key, dict(hands))
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
    return mode_refractory_ms, exit_hold_ms, mode_triggers


def main():
    cfg = load_config()
    LOGGER.info("Starting GCPC application")
    measurements_enabled = bool((cfg.get("measurements") or {}).get("enabled", False))

    app = QtWidgets.QApplication([])
    osd = OSD()
    osd.show()

    ui_cfg = cfg.get("ui", {})
    panel_cfg = ui_cfg.get("controls_panel", {})
    panel_enabled = True

    start_scenario_key = str(panel_cfg.get("start_scenario_shortcut", "Alt+S"))
    end_scenario_key = str(panel_cfg.get("end_scenario_shortcut", "Alt+E"))
    settings_shortcut_key = str(panel_cfg.get("settings_shortcut", "Ctrl+,"))

    vcfg = cfg.get("video", {})
    idx = int(vcfg.get("camera_index", 0))
    last_working_camera_idx = _optional_int(vcfg.get("last_working_camera_index"))
    w = int(vcfg.get("width", 1280))
    h = int(vcfg.get("height", 720))
    mirror = bool(vcfg.get("mirror", True))
    show_fps = bool(vcfg.get("show_fps", False))

    hand_windows_cfg = ui_cfg.get("hand_windows", {})
    hand_windows_enabled = bool(hand_windows_cfg.get("enabled", False))
    hand_window_size = int(hand_windows_cfg.get("size", 220))
    hand_window_padding = float(hand_windows_cfg.get("padding", 0.2))
    hand_window_margin = int(hand_windows_cfg.get("margin_px", 16))
    show_full_camera = bool(hand_windows_cfg.get("show_full_camera", True))

    panel = (
        ControlPanel(
            default_camera_enabled=bool(panel_cfg.get("camera_enabled", False)),
            default_hand_enabled=bool(panel_cfg.get("hand_control_enabled", False)),
            default_resolution=(w, h),
        )
        if panel_enabled
        else None
    )
    if panel:
        panel.show()

    calibration_requested_from_ui = False

    def _persist_camera_resolution(width: int, height: int) -> None:
        width = int(width)
        height = int(height)
        video_cfg = cfg.setdefault("video", {})
        try:
            current = (int(video_cfg.get("width")), int(video_cfg.get("height")))
        except (TypeError, ValueError):
            current = (None, None)
        if current == (width, height):
            return
        video_cfg["width"] = width
        video_cfg["height"] = height
        save_config(cfg)
        LOGGER.info("Camera resolution saved: %sx%s", width, height)

    def _persist_last_camera_index(camera_idx: int | None) -> None:
        nonlocal last_working_camera_idx
        if camera_idx is None or camera_idx < 0:
            return
        if last_working_camera_idx == camera_idx:
            return
        video_cfg = cfg.setdefault("video", {})
        video_cfg["last_working_camera_index"] = camera_idx
        last_working_camera_idx = camera_idx
        save_config(cfg)
        LOGGER.info("Last working camera index saved: %s", camera_idx)

    def _open_settings():
        nonlocal calibration_requested_from_ui
        dialog_parent = panel if panel else osd
        previous_cfg = copy.deepcopy(cfg)

        def _request_calibration():
            nonlocal calibration_requested_from_ui
            calibration_requested_from_ui = True

        try:
            dialog = GestureSettingsDialog(
                cfg,
                dialog_parent,
                on_request_calibration=_request_calibration,
            )
            if dialog.exec() != QtWidgets.QDialog.Accepted:
                return
            restart_notes = _apply_runtime_settings(int(time.time() * 1000))
            save_config(cfg)
            LOGGER.info("Gesture settings saved to config.json and applied live")
            message = "Gesture settings were saved and applied immediately."
            if restart_notes:
                message += "\nRestart GCPC to apply: " + ", ".join(restart_notes)
            QtWidgets.QMessageBox.information(
                dialog_parent,
                "Settings saved",
                message,
            )
        except Exception:
            cfg.clear()
            cfg.update(previous_cfg)
            try:
                _apply_runtime_settings(int(time.time() * 1000))
            except Exception:
                LOGGER.exception("Failed to restore runtime settings after settings error")
            LOGGER.exception("Failed to save gesture settings")
            QtWidgets.QMessageBox.critical(
                dialog_parent,
                "Settings error",
                "Could not apply or save gesture settings. Check logs/gcpc.log for details.",
            )

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

    dcfg = cfg.get("detector", {})
    handedness_cfg = dcfg.get("handedness") or {}
    handedness_strategy = str(handedness_cfg.get("strategy", "auto")).strip().lower()
    if handedness_strategy not in {"auto", "label", "geometry"}:
        handedness_strategy = "auto"
    swap_handedness_labels_cfg = handedness_cfg.get("swap_labels")
    prefer_geometry_on_conflict = bool(
        handedness_cfg.get("prefer_geometry_on_conflict", False)
    )

    tracker = MediaPipeHandTracker(
        min_det=dcfg.get("min_detection_confidence"),
        min_trk=dcfg.get("min_tracking_confidence"),
        max_hands=dcfg.get("max_num_hands"),
        model_complexity=dcfg.get("model_complexity"),
    )

    providers = getattr(tracker, "providers", []) or []
    using_tasks_backend = any("tasks" in str(p).lower() for p in providers)
    if swap_handedness_labels_cfg is None:
        # Tasks backend on mirrored frames commonly needs label swap.
        swap_handedness_labels = bool(using_tasks_backend and mirror)
    else:
        swap_handedness_labels = bool(swap_handedness_labels_cfg)

    resolver = HandednessResolver(
        strategy=handedness_strategy,
        mirror=mirror,
        using_tasks_backend=using_tasks_backend,
        swap_labels=swap_handedness_labels,
        prefer_geometry_on_conflict=prefer_geometry_on_conflict,
    )

    backend_kind = (
        "gpu" if any("CUDA" in str(p).upper() or "GPU" in str(p).upper() for p in providers) else "cpu"
    )
    providers_str = str(providers)
    print(f"[SESSION] id={session_id} backend={backend_kind} providers={providers}")
    print(
        "[HANDEDNESS] "
        f"backend={'tasks' if using_tasks_backend else 'legacy'} "
        f"strategy={resolver.strategy} "
        f"swap_labels={resolver.swap_labels} "
        f"mirror={mirror}"
    )

    if measurements_enabled:
        append_metrics_row(
            {
                "session_id": session_id,
                "backend": backend_kind,
                "providers": providers_str,
            }
        )

    def _session_summary_row(final_ns: int | None = None) -> Dict[str, str | float]:
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

    hands = build_hands(cfg)
    dominant_side = hands["dominant"]
    support_side = hands["support"]
    dominant_is_right = dominant_side == "RIGHT"

    eval_session = EvalSingleSession(cfg, hands, panel=panel)
    calibration = CalibrationSession(cfg)

    def _start_scenario():
        nonlocal scenario_active, scenario_name, scenario_start_ns, scenario_false_baseline, scenario_interaction
        if scenario_active:
            print("[SCENARIO] Already active; end current scenario first.")
            return
        text, ok = QtWidgets.QInputDialog.getText(
            osd, "Start scenario", "Scenario name:", text=scenario_name or ""
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
        if measurements_enabled:
            append_metrics_row(
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
        if eval_session.active:
            return
        now_ns = time.perf_counter_ns()
        if measurements_enabled and last_frame_capture_ns is not None:
            latencies.append((now_ns - last_frame_capture_ns) / 1e6)
        armed = _armed_state()
        active = scenario_active
        if measurements_enabled and not (active and armed):
            false_activations_total += 1
        press_combo(combo)

    shortcut_parent = panel if panel else osd
    if measurements_enabled:
        start_shortcut = QtGui.QShortcut(QtGui.QKeySequence(start_scenario_key), shortcut_parent)
        start_shortcut.setContext(QtCore.Qt.ApplicationShortcut)
        start_shortcut.activated.connect(_start_scenario)
        end_shortcut = QtGui.QShortcut(QtGui.QKeySequence(end_scenario_key), shortcut_parent)
        end_shortcut.setContext(QtCore.Qt.ApplicationShortcut)
        end_shortcut.activated.connect(_end_scenario)

    settings_shortcut = QtGui.QShortcut(QtGui.QKeySequence(settings_shortcut_key), shortcut_parent)
    settings_shortcut.setContext(QtCore.Qt.ApplicationShortcut)
    settings_shortcut.activated.connect(_open_settings)
    if panel:
        panel.settings_requested.connect(_open_settings)
        panel.camera_resolution_changed.connect(_persist_camera_resolution)

    def _safe_destroy_window(title: str) -> None:
        try:
            cv2.destroyWindow(title)
        except Exception:
            pass

    cap = None
    active_resolution = (w, h)
    camera_error = ""

    def _close_camera() -> None:
        nonlocal cap
        if cap is not None:
            cap.release()
            cap = None
        _safe_destroy_window("GCPC - Camera")
        _safe_destroy_window("GCPC - Left Hand")
        _safe_destroy_window("GCPC - Right Hand")

    gR = GestureState(cfg["gesture_engine"])
    gL = GestureState(cfg["gesture_engine"])

    seq_cfg = cfg.get("sequence", {})
    arm_delay_ms = int(seq_cfg.get("arm_delay_ms", 420))
    refractory_ms = int(seq_cfg.get("refractory_ms", 1100))
    cancel_exit_ms = int(seq_cfg.get("cancel_on_hand_exit_ms", 900))
    max_len = int(seq_cfg.get("max_len", 6))

    controls_cfg = cfg.get("controls", {})
    seq_ctrl = controls_cfg.get("sequence", {})
    seq_input_side = resolve_side(seq_ctrl.get("input_hand", "dominant"), hands)
    candidate_ignore = {str(gesture).upper() for gesture in seq_ctrl.get("candidate_ignore", [])}

    confirm_cfg = seq_ctrl.get("confirm") or {}
    confirm_binding_value = (
        confirm_cfg.get("binding") if isinstance(confirm_cfg, dict) else confirm_cfg
    )
    confirm_binding = parse_single_binding(confirm_binding_value, hands)
    confirm_hand = confirm_binding["hand"]
    confirm_gesture = confirm_binding["gesture"]
    confirm_deb = DebouncedTrigger(
        int(confirm_cfg.get("dwell_ms", 220)),
        int(confirm_cfg.get("refractory_ms", 700)),
    )

    undo_cfg = seq_ctrl.get("undo") or {}
    undo_binding_value = (
        undo_cfg.get("binding") if isinstance(undo_cfg, dict) else undo_cfg
    )
    undo_binding = parse_sequence_binding(undo_binding_value, hands)
    undo_hand = undo_binding["hand"]
    undo_steps = undo_binding["gestures"]
    undo_start = undo_steps[0]
    undo_end = undo_steps[-1] if len(undo_steps) > 1 else undo_steps[0]
    undo_window_ms = int(undo_cfg.get("window_ms", 900))

    commit_cfg = seq_ctrl.get("commit") or {}
    commit_binding_value = (
        commit_cfg.get("binding") if isinstance(commit_cfg, dict) else commit_cfg
    )
    commit_binding = parse_single_binding(commit_binding_value, hands)
    commit_hand = commit_binding["hand"]
    commit_gesture = commit_binding["gesture"]
    commit_deb = DebouncedTrigger(
        int(commit_cfg.get("dwell_ms", 260)),
        int(commit_cfg.get("refractory_ms", 1200)),
    )

    cmd_map = cfg.get("command_mappings") or {}
    single_map_raw = cmd_map.get("single_gestures") or {}
    complex_map_raw = cmd_map.get("complex_gestures") or {}
    functional_raw = cmd_map.get("functional") or {}
    mode_refractory_ms, exit_hold_ms, mode_triggers = _build_mode_triggers(
        functional_raw,
        hands,
    )
    single_map = build_single_map(single_map_raw, hands)
    one_hand_dispatcher = OneHandCommandDispatcher(
        single_map_raw,
        hands,
        refractory_ms=refractory_ms,
    )
    seq_map = build_sequence_map(complex_map_raw, hands)
    merge_single_into_sequences(seq_map, single_map)

    one_cfg = cfg.get("one_hand_mode", {})
    one_enabled = bool(one_cfg.get("enabled", True))
    one_status_label = str(one_cfg.get("status_label", "ONE-HAND"))
    func = cfg.get("command_mappings", {}).get("functional", {})
    exit_gesture = next((gesture for gesture, cmd in func.items() if cmd == "MODE_EXIT"), None)
    base_hint = one_cfg.get("active_hint") or ""
    exit_hint = f"To exit mode: {exit_gesture}" if exit_gesture else ""
    if base_hint and exit_hint:
        one_active_hint = f"{base_hint} | {exit_hint}"
    elif base_hint:
        one_active_hint = base_hint
    else:
        action_hint = one_hand_dispatcher.hint()
        one_active_hint = f"{action_hint} | {exit_hint}" if exit_hint else action_hint

    mouse_cfg = cfg.get("mouse_control", {})
    mouse_enabled = bool(mouse_cfg.get("enabled", False))
    mouse_status_label = str(mouse_cfg.get("status_label", "MOUSE"))
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
    mouse_rect = _clamp_rect(rect_x, rect_y, rect_w, rect_h)

    def _read_mouse_binding(
        key: str,
        source_cfg: Mapping[str, Any] | None = None,
        source_hands: Mapping[str, str] | None = None,
        source_dominant_side: str | None = None,
        source_support_side: str | None = None,
    ):
        binding_cfg = source_cfg if source_cfg is not None else mouse_cfg
        binding_hands = dict(source_hands if source_hands is not None else hands)
        binding_dominant = source_dominant_side or dominant_side
        binding_support = source_support_side or support_side
        raw_value = binding_cfg.get(key)
        binding = parse_single_binding(raw_value, binding_hands)
        gesture = binding.get("gesture") or ""
        binding["gesture"] = str(gesture).upper()
        label = binding_notation(binding, binding_dominant, binding_support)
        return binding, label

    mouse_left_binding, mouse_left_label = _read_mouse_binding("left_click_binding")
    mouse_right_binding, mouse_right_label = _read_mouse_binding("right_click_binding")

    scroll_cfg = mouse_cfg.get("scroll", {}) or {}
    scroll_enabled = bool(scroll_cfg.get("enabled", True))
    scroll_hand_token = scroll_cfg.get("hand", "RIGHT")
    scroll_side = resolve_side(scroll_hand_token, hands)
    if scroll_side not in ("RIGHT", "LEFT"):
        scroll_side = "RIGHT"
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
    mouse_active_hint = mouse_cfg.get("active_hint")
    if not mouse_active_hint:
        trig_label = trigger_label(mode_triggers.get("mouse"), dominant_side, support_side)
        pointer_hint = binding_notation(
            {"hand": pointer_side, "gesture": "PINCH"},
            dominant_side,
            support_side,
        )
        base_hint = f"CURSOR: {pointer_hint} | LMB: {mouse_left_label} | RMB: {mouse_right_label}"
        if scroll_enabled:
            base_hint = f"{base_hint} | SCROLL: {scroll_label}"
        mouse_active_hint = f"{trig_label} | {base_hint}" if trig_label else base_hint

    mouse_prev = None
    mouse_left_down = False
    mouse_right_down = False
    scroll_prev_y = None
    scroll_last_ms = 0

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
    exit_hold_since = None

    def switch_mode(new_mode: str, now_ms: int, force_reset: bool = False):
        nonlocal current_mode, seq_active, one_hand_active, mouse_active
        nonlocal mouse_prev, mouse_left_down, mouse_right_down
        nonlocal last_single_action, seq_buffer, seq_pending, last_evt_ms, mode_last_change_ms

        prev = current_mode
        if new_mode == prev and not force_reset:
            return
        if new_mode == prev and force_reset and new_mode == "idle":
            if not seq_active and not one_hand_active and not mouse_active and not calibration.active:
                return

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

        if prev == "calibrate" or force_reset:
            calibration.stop()
        if new_mode == "calibrate" and calibration.enabled:
            calibration.start(now_ms)

        if new_mode != "one_hand":
            last_single_action = ""
        one_hand_active = (new_mode == "one_hand") and one_enabled
        if prev == "one_hand" or one_hand_active or force_reset:
            one_hand_dispatcher.reset(now_ms)

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

    def _apply_runtime_settings(now_ms: int) -> list[str]:
        nonlocal hands, dominant_side, support_side, dominant_is_right, eval_session
        nonlocal arm_delay_ms, refractory_ms, cancel_exit_ms, max_len
        nonlocal seq_input_side, candidate_ignore
        nonlocal confirm_hand, confirm_gesture, confirm_deb
        nonlocal undo_hand, undo_steps, undo_start, undo_end, undo_window_ms
        nonlocal commit_hand, commit_gesture, commit_deb
        nonlocal single_map_raw, complex_map_raw, functional_raw
        nonlocal mode_refractory_ms, exit_hold_ms, mode_triggers
        nonlocal single_map, one_hand_dispatcher, seq_map
        nonlocal one_enabled, one_status_label, one_active_hint
        nonlocal mouse_cfg, mouse_enabled, mouse_status_label, mouse_smooth
        nonlocal pointer_side, pointer_landmark, mouse_rect
        nonlocal mouse_left_binding, mouse_left_label
        nonlocal mouse_right_binding, mouse_right_label
        nonlocal scroll_enabled, scroll_side, scroll_gesture, scroll_landmark
        nonlocal scroll_speed, scroll_deadzone, scroll_interval_ms, scroll_label
        nonlocal mouse_active_hint, mouse_prev, mouse_left_down, mouse_right_down
        nonlocal scroll_prev_y, scroll_last_ms
        nonlocal seq_buffer, seq_pending, last_seq_event_ms, last_evt_ms, last_sent_ms
        nonlocal undo_open_ts, last_R_label, last_L_label, last_single_action
        nonlocal both_pose_latched, exit_hold_since
        nonlocal hand_windows_enabled, hand_window_size, hand_window_padding
        nonlocal hand_window_margin, show_full_camera, hand_windows_placed

        previous_hands = (dominant_side, support_side)
        previous_mode_triggers = mode_triggers
        previous_single_map_raw = single_map_raw
        previous_complex_map_raw = complex_map_raw
        previous_mouse_enabled = mouse_enabled
        previous_pointer_side = pointer_side
        previous_pointer_landmark = pointer_landmark
        previous_mouse_rect = dict(mouse_rect)
        previous_mouse_left_binding = dict(mouse_left_binding)
        previous_mouse_right_binding = dict(mouse_right_binding)
        previous_scroll_state = (
            scroll_enabled,
            scroll_side,
            scroll_gesture,
            scroll_landmark,
            scroll_speed,
            scroll_deadzone,
            scroll_interval_ms,
        )
        previous_hand_window_state = (
            hand_windows_enabled,
            hand_window_size,
            hand_window_padding,
            hand_window_margin,
            show_full_camera,
        )

        new_hands = build_hands(cfg)
        new_dominant_side = new_hands["dominant"]
        new_support_side = new_hands["support"]
        new_dominant_is_right = new_dominant_side == "RIGHT"

        new_seq_cfg = cfg.get("sequence", {}) or {}
        new_arm_delay_ms = int(new_seq_cfg.get("arm_delay_ms", 420))
        new_refractory_ms = int(new_seq_cfg.get("refractory_ms", 1100))
        new_cancel_exit_ms = int(new_seq_cfg.get("cancel_on_hand_exit_ms", 900))
        new_max_len = int(new_seq_cfg.get("max_len", 6))

        new_controls_cfg = cfg.get("controls", {}) or {}
        new_seq_ctrl = new_controls_cfg.get("sequence", {}) or {}
        new_seq_input_side = resolve_side(
            new_seq_ctrl.get("input_hand", "dominant"),
            new_hands,
        )
        new_candidate_ignore = {
            str(gesture).upper()
            for gesture in new_seq_ctrl.get("candidate_ignore", [])
        }

        new_confirm_cfg = new_seq_ctrl.get("confirm") or {}
        new_confirm_binding_value = (
            new_confirm_cfg.get("binding")
            if isinstance(new_confirm_cfg, dict)
            else new_confirm_cfg
        )
        new_confirm_options = (
            new_confirm_cfg if isinstance(new_confirm_cfg, dict) else {}
        )
        new_confirm_binding = parse_single_binding(
            new_confirm_binding_value,
            new_hands,
        )
        new_confirm_deb = DebouncedTrigger(
            int(new_confirm_options.get("dwell_ms", 220)),
            int(new_confirm_options.get("refractory_ms", 700)),
        )

        new_undo_cfg = new_seq_ctrl.get("undo") or {}
        new_undo_binding_value = (
            new_undo_cfg.get("binding") if isinstance(new_undo_cfg, dict) else new_undo_cfg
        )
        new_undo_options = new_undo_cfg if isinstance(new_undo_cfg, dict) else {}
        new_undo_binding = parse_sequence_binding(new_undo_binding_value, new_hands)
        new_undo_steps = new_undo_binding["gestures"]
        new_undo_start = new_undo_steps[0]
        new_undo_end = (
            new_undo_steps[-1] if len(new_undo_steps) > 1 else new_undo_steps[0]
        )

        new_commit_cfg = new_seq_ctrl.get("commit") or {}
        new_commit_binding_value = (
            new_commit_cfg.get("binding")
            if isinstance(new_commit_cfg, dict)
            else new_commit_cfg
        )
        new_commit_options = new_commit_cfg if isinstance(new_commit_cfg, dict) else {}
        new_commit_binding = parse_single_binding(new_commit_binding_value, new_hands)
        new_commit_deb = DebouncedTrigger(
            int(new_commit_options.get("dwell_ms", 260)),
            int(new_commit_options.get("refractory_ms", 1200)),
        )

        new_cmd_map = cfg.get("command_mappings") or {}
        new_single_map_raw = new_cmd_map.get("single_gestures") or {}
        new_complex_map_raw = new_cmd_map.get("complex_gestures") or {}
        new_functional_raw = new_cmd_map.get("functional") or {}
        (
            new_mode_refractory_ms,
            new_exit_hold_ms,
            new_mode_triggers,
        ) = _build_mode_triggers(new_functional_raw, new_hands)
        new_single_map = build_single_map(new_single_map_raw, new_hands)
        new_one_hand_dispatcher = OneHandCommandDispatcher(
            new_single_map_raw,
            new_hands,
            refractory_ms=new_refractory_ms,
        )
        new_seq_map = build_sequence_map(new_complex_map_raw, new_hands)
        merge_single_into_sequences(new_seq_map, new_single_map)

        new_one_cfg = cfg.get("one_hand_mode", {}) or {}
        new_one_enabled = bool(new_one_cfg.get("enabled", True))
        new_one_status_label = str(new_one_cfg.get("status_label", "ONE-HAND"))
        new_func = (cfg.get("command_mappings", {}) or {}).get("functional", {}) or {}
        new_exit_gesture = next(
            (gesture for gesture, cmd in new_func.items() if cmd == "MODE_EXIT"),
            None,
        )
        new_base_hint = new_one_cfg.get("active_hint") or ""
        new_exit_hint = f"To exit mode: {new_exit_gesture}" if new_exit_gesture else ""
        if new_base_hint and new_exit_hint:
            new_one_active_hint = f"{new_base_hint} | {new_exit_hint}"
        elif new_base_hint:
            new_one_active_hint = new_base_hint
        else:
            new_action_hint = new_one_hand_dispatcher.hint()
            new_one_active_hint = (
                f"{new_action_hint} | {new_exit_hint}"
                if new_exit_hint
                else new_action_hint
            )

        new_mouse_cfg = cfg.get("mouse_control", {}) or {}
        new_mouse_enabled = bool(new_mouse_cfg.get("enabled", False))
        new_mouse_status_label = str(new_mouse_cfg.get("status_label", "MOUSE"))
        new_mouse_smooth = max(
            0.0,
            min(1.0, float(new_mouse_cfg.get("smoothing_alpha", 0.25))),
        )
        new_pointer_side = resolve_side(new_mouse_cfg.get("pointer_hand"), new_hands)
        if new_pointer_side not in ("RIGHT", "LEFT"):
            new_pointer_side = new_dominant_side
        new_pointer_landmark = int(new_mouse_cfg.get("pointer_landmark", 8))

        new_rect_cfg = new_mouse_cfg.get("control_rect", {}) or {}
        new_mouse_rect = _clamp_rect(
            float(new_rect_cfg.get("x", 0.0) or 0.0),
            float(new_rect_cfg.get("y", 0.0) or 0.0),
            float(new_rect_cfg.get("width", 1.0) or 1.0),
            float(new_rect_cfg.get("height", 1.0) or 1.0),
        )
        new_mouse_left_binding, new_mouse_left_label = _read_mouse_binding(
            "left_click_binding",
            new_mouse_cfg,
            new_hands,
            new_dominant_side,
            new_support_side,
        )
        new_mouse_right_binding, new_mouse_right_label = _read_mouse_binding(
            "right_click_binding",
            new_mouse_cfg,
            new_hands,
            new_dominant_side,
            new_support_side,
        )

        new_scroll_cfg = new_mouse_cfg.get("scroll", {}) or {}
        new_scroll_enabled = bool(new_scroll_cfg.get("enabled", True))
        new_scroll_side = resolve_side(new_scroll_cfg.get("hand", "RIGHT"), new_hands)
        if new_scroll_side not in ("RIGHT", "LEFT"):
            new_scroll_side = "RIGHT"
        new_scroll_gesture = str(new_scroll_cfg.get("gesture", "FIST")).upper()
        new_scroll_landmark = int(new_scroll_cfg.get("landmark", WRIST))
        new_scroll_speed = float(new_scroll_cfg.get("speed", 1200.0))
        new_scroll_deadzone = float(new_scroll_cfg.get("deadzone", 0.01))
        new_scroll_interval_ms = int(new_scroll_cfg.get("interval_ms", 30))
        new_scroll_label = binding_notation(
            {"hand": new_scroll_side, "gesture": new_scroll_gesture},
            new_dominant_side,
            new_support_side,
        )
        new_mouse_active_hint = new_mouse_cfg.get("active_hint")
        if not new_mouse_active_hint:
            new_trig_label = trigger_label(
                new_mode_triggers.get("mouse"),
                new_dominant_side,
                new_support_side,
            )
            new_pointer_hint = binding_notation(
                {"hand": new_pointer_side, "gesture": "PINCH"},
                new_dominant_side,
                new_support_side,
            )
            new_base_mouse_hint = (
                f"CURSOR: {new_pointer_hint} | "
                f"LMB: {new_mouse_left_label} | RMB: {new_mouse_right_label}"
            )
            if new_scroll_enabled:
                new_base_mouse_hint = (
                    f"{new_base_mouse_hint} | SCROLL: {new_scroll_label}"
                )
            new_mouse_active_hint = (
                f"{new_trig_label} | {new_base_mouse_hint}"
                if new_trig_label
                else new_base_mouse_hint
            )

        new_hand_windows_cfg = (cfg.get("ui", {}) or {}).get("hand_windows", {}) or {}
        new_hand_windows_enabled = bool(new_hand_windows_cfg.get("enabled", False))
        new_hand_window_size = int(new_hand_windows_cfg.get("size", 220))
        new_hand_window_padding = float(new_hand_windows_cfg.get("padding", 0.2))
        new_hand_window_margin = int(new_hand_windows_cfg.get("margin_px", 16))
        new_show_full_camera = bool(new_hand_windows_cfg.get("show_full_camera", True))

        hand_roles_changed = previous_hands != (new_dominant_side, new_support_side)
        binding_settings_changed = (
            previous_mode_triggers != new_mode_triggers
            or previous_single_map_raw != new_single_map_raw
            or previous_complex_map_raw != new_complex_map_raw
            or seq_input_side != new_seq_input_side
            or confirm_hand != new_confirm_binding["hand"]
            or confirm_gesture != new_confirm_binding["gesture"]
            or undo_hand != new_undo_binding["hand"]
            or undo_steps != new_undo_steps
            or commit_hand != new_commit_binding["hand"]
            or commit_gesture != new_commit_binding["gesture"]
        )
        mouse_settings_changed = (
            previous_mouse_enabled != new_mouse_enabled
            or previous_pointer_side != new_pointer_side
            or previous_pointer_landmark != new_pointer_landmark
            or previous_mouse_rect != new_mouse_rect
            or previous_mouse_left_binding != new_mouse_left_binding
            or previous_mouse_right_binding != new_mouse_right_binding
            or previous_scroll_state
            != (
                new_scroll_enabled,
                new_scroll_side,
                new_scroll_gesture,
                new_scroll_landmark,
                new_scroll_speed,
                new_scroll_deadzone,
                new_scroll_interval_ms,
            )
        )
        hand_window_state_changed = previous_hand_window_state != (
            new_hand_windows_enabled,
            new_hand_window_size,
            new_hand_window_padding,
            new_hand_window_margin,
            new_show_full_camera,
        )

        hands = new_hands
        dominant_side = new_dominant_side
        support_side = new_support_side
        dominant_is_right = new_dominant_is_right
        if eval_session.active:
            eval_session.reconfigure_hands(cfg, hands)
        else:
            eval_session = EvalSingleSession(cfg, hands, panel=panel)

        gR.cfg = cfg.get("gesture_engine", {}) or {}
        gL.cfg = cfg.get("gesture_engine", {}) or {}

        arm_delay_ms = new_arm_delay_ms
        refractory_ms = new_refractory_ms
        cancel_exit_ms = new_cancel_exit_ms
        max_len = new_max_len
        seq_input_side = new_seq_input_side
        candidate_ignore = new_candidate_ignore
        confirm_hand = new_confirm_binding["hand"]
        confirm_gesture = new_confirm_binding["gesture"]
        confirm_deb = new_confirm_deb
        undo_hand = new_undo_binding["hand"]
        undo_steps = new_undo_steps
        undo_start = new_undo_start
        undo_end = new_undo_end
        undo_window_ms = int(new_undo_options.get("window_ms", 900))
        commit_hand = new_commit_binding["hand"]
        commit_gesture = new_commit_binding["gesture"]
        commit_deb = new_commit_deb
        single_map_raw = new_single_map_raw
        complex_map_raw = new_complex_map_raw
        functional_raw = new_functional_raw
        mode_refractory_ms = new_mode_refractory_ms
        exit_hold_ms = new_exit_hold_ms
        mode_triggers = new_mode_triggers
        single_map = new_single_map
        one_hand_dispatcher = new_one_hand_dispatcher
        seq_map = new_seq_map
        one_enabled = new_one_enabled
        one_status_label = new_one_status_label
        one_active_hint = new_one_active_hint

        mouse_cfg = new_mouse_cfg
        mouse_enabled = new_mouse_enabled
        mouse_status_label = new_mouse_status_label
        mouse_smooth = new_mouse_smooth
        pointer_side = new_pointer_side
        pointer_landmark = new_pointer_landmark
        mouse_rect = new_mouse_rect
        mouse_left_binding = new_mouse_left_binding
        mouse_left_label = new_mouse_left_label
        mouse_right_binding = new_mouse_right_binding
        mouse_right_label = new_mouse_right_label
        scroll_enabled = new_scroll_enabled
        scroll_side = new_scroll_side
        scroll_gesture = new_scroll_gesture
        scroll_landmark = new_scroll_landmark
        scroll_speed = new_scroll_speed
        scroll_deadzone = new_scroll_deadzone
        scroll_interval_ms = new_scroll_interval_ms
        scroll_label = new_scroll_label
        mouse_active_hint = new_mouse_active_hint

        hand_windows_enabled = new_hand_windows_enabled
        hand_window_size = new_hand_window_size
        hand_window_padding = new_hand_window_padding
        hand_window_margin = new_hand_window_margin
        show_full_camera = new_show_full_camera

        if hand_roles_changed or binding_settings_changed:
            seq_buffer.clear()
            seq_pending = None
            last_seq_event_ms = now_ms
            last_evt_ms = now_ms
            last_sent_ms = now_ms
            undo_open_ts = {"RIGHT": None, "LEFT": None}
            both_pose_latched.clear()
            exit_hold_since = None
            last_R_label = ""
            last_L_label = ""
            last_single_action = ""
            gR.pose_flags.clear()
            gL.pose_flags.clear()
            one_hand_dispatcher.reset(now_ms)

        if mouse_settings_changed:
            mouse_prev = None
            scroll_prev_y = None
            scroll_last_ms = now_ms
            if mouse_left_down:
                mouse_release("left")
                mouse_left_down = False
            if mouse_right_down:
                mouse_release("right")
                mouse_right_down = False

        if current_mode == "mouse" and not mouse_enabled:
            switch_mode("idle", now_ms, force_reset=True)
        elif current_mode == "one_hand" and not one_enabled:
            switch_mode("idle", now_ms, force_reset=True)
        elif current_mode == "one_hand":
            one_hand_dispatcher.reset(now_ms)

        if hand_window_state_changed:
            hand_windows_placed = False
            _safe_destroy_window("GCPC - Left Hand")
            _safe_destroy_window("GCPC - Right Hand")
            if hand_windows_enabled and not show_full_camera:
                _safe_destroy_window("GCPC - Camera")

        return []

    fps = None
    last_frame_time = time.time()
    hand_windows_placed = False
    hand_control_prev = panel.is_hand_control_enabled() if panel else True
    camera_requested_prev = panel.is_camera_enabled() if panel else True

    while True:
        QtWidgets.QApplication.processEvents()
        if panel and not panel.isVisible():
            break

        camera_requested = panel.is_camera_enabled() if panel else True
        requested_resolution = panel.selected_camera_resolution() if panel else (w, h)
        requested_resolution = (int(requested_resolution[0]), int(requested_resolution[1]))

        if camera_requested and requested_resolution != active_resolution:
            _persist_camera_resolution(*requested_resolution)
            _close_camera()
            active_resolution = requested_resolution
            hand_windows_placed = False

        if camera_requested and cap is None:
            cap, opened_camera_idx = open_camera(
                idx,
                requested_resolution[0],
                requested_resolution[1],
                preferred_idx=last_working_camera_idx,
            )
            if cap is None:
                camera_error = (
                    f"Unable to open camera index={idx} "
                    f"at {requested_resolution[0]}x{requested_resolution[1]}"
                )
                osd.set_text("CAMERA ERROR", camera_error)
                QtCore.QThread.msleep(120)
                continue
            _persist_last_camera_index(opened_camera_idx)
            active_resolution = requested_resolution
            w, h = active_resolution
            camera_error = ""
            hand_windows_placed = False

        if not camera_requested:
            if camera_requested_prev:
                switch_mode("idle", int(time.time() * 1000), force_reset=True)
                _close_camera()
            camera_requested_prev = False
            osd.set_text("CAMERA OFF", "Enable camera in GCPC Controls")
            QtCore.QThread.msleep(80)
            continue
        camera_requested_prev = True

        if cap is None:
            osd.set_text("CAMERA OFF", camera_error or "Camera is not initialized")
            QtCore.QThread.msleep(80)
            continue

        ret, frame = cap.read()
        if not ret:
            camera_error = "Camera read failed. Trying to reopen..."
            switch_mode("idle", int(time.time() * 1000), force_reset=True)
            _close_camera()
            osd.set_text("CAMERA ERROR", camera_error)
            QtCore.QThread.msleep(120)
            continue

        if mirror:
            frame = cv2.flip(frame, 1)
        frame_capture_ns = time.perf_counter_ns()
        last_frame_capture_ns = frame_capture_ns
        frames_processed += 1

        now = time.time()
        dt = now - last_frame_time
        last_frame_time = now
        now_ms = int(now * 1000)

        hand_control_enabled = panel.is_hand_control_enabled() if panel else True
        if hand_control_prev and not hand_control_enabled:
            switch_mode("idle", now_ms, force_reset=True)
        hand_control_prev = hand_control_enabled

        if calibration_requested_from_ui:
            if hand_control_enabled:
                switch_mode("calibrate", now_ms, force_reset=True)
                calibration_requested_from_ui = False
            else:
                osd.set_text("CALIBRATION WAIT", "Enable hand control first.")

        if not hand_control_enabled:
            if show_fps and dt > 0:
                inst = 1.0 / dt
                fps = inst if fps is None else (0.9 * fps + 0.1 * inst)
            cv2.putText(
                frame,
                "HAND CONTROL: OFF",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 200, 255),
                2,
                cv2.LINE_AA,
            )
            if show_fps and fps is not None:
                cv2.putText(
                    frame,
                    f"FPS: {fps:5.1f}",
                    (10, 58),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA,
                )
            osd.set_text("IDLE", "Enable hand control in GCPC Controls")
            if not hand_windows_enabled or show_full_camera:
                cv2.imshow("GCPC - Camera", frame)
            else:
                _safe_destroy_window("GCPC - Camera")
            _safe_destroy_window("GCPC - Left Hand")
            _safe_destroy_window("GCPC - Right Hand")
            raw_key = cv2.waitKey(1)
            key = raw_key & 0xFF if raw_key != -1 else -1
            if raw_key == 27 or key == 27:
                break
            continue

        finished_calibration = False
        if calibration.active:
            finished_calibration = calibration.advance(now_ms)
        if finished_calibration:
            switch_mode("idle", now_ms, force_reset=True)

        if show_fps and dt > 0:
            inst = 1.0 / dt
            fps = inst if fps is None else (0.9 * fps + 0.1 * inst)

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        hands_data = tracker.process(rgb)
        for hand in hands_data:
            hand["label"] = resolver.resolve_label(hand)

        rights = []
        lefts = []
        for hand in hands_data:
            if hand.get("label", "") == "Right":
                rights.append(hand)
            elif hand.get("label", "") == "Left":
                lefts.append(hand)
        if not rights and not lefts and hands_data:
            sorted_hands = sorted(hands_data, key=lambda item: item["lm"][0][0])
            if len(sorted_hands) == 2:
                lefts = [sorted_hands[0]]
                rights = [sorted_hands[1]]
            elif len(sorted_hands) == 1:
                label = resolver.infer_side_from_geometry(sorted_hands[0]["lm"])
                if label == "Right":
                    rights = [sorted_hands[0]]
                else:
                    lefts = [sorted_hands[0]]

        right = max(rights, key=lambda item: item.get("score", 0)) if rights else None
        left = max(lefts, key=lambda item: item.get("score", 0)) if lefts else None

        evR = ""
        evL = ""
        if right:
            last_seen_R = now_ms
            evR = gR.update_and_classify(right["lm"]) or ""
            if evR:
                last_R_label = evR
            if calibration.active:
                calibration.record(right["lm"], "RIGHT")
        else:
            gR.pose_flags.clear()
            if seq_active and seq_input_side == "RIGHT" and cancel_exit_ms > 0:
                if (now_ms - last_seen_R) >= cancel_exit_ms:
                    if seq_buffer or seq_pending:
                        seq_buffer.clear()
                        seq_pending = None
                        print("[SEQ] Cleared buffer: right hand out of frame")

        if left:
            last_seen_L = now_ms
            evL = gL.update_and_classify(left["lm"]) or ""
            if evL:
                last_L_label = evL
            if calibration.active:
                calibration.record(left["lm"], "LEFT")
        else:
            gL.pose_flags.clear()
            if seq_active and seq_input_side == "LEFT" and cancel_exit_ms > 0:
                if (now_ms - last_seen_L) >= cancel_exit_ms:
                    if seq_buffer or seq_pending:
                        seq_buffer.clear()
                        seq_pending = None
                        print("[SEQ] Cleared buffer: left hand out of frame")

        dom_event = evR if dominant_side == "RIGHT" else evL
        support_event = evR if support_side == "RIGHT" else evL

        eval_requested = measurements_enabled and (panel.is_eval_single() if panel else False)
        if eval_requested and not eval_session.active:
            if eval_session.start(now_ms, session_id):
                switch_mode("idle", now_ms, force_reset=True)
        elif not eval_requested and eval_session.active:
            eval_session.stop()

        def _trigger_fired(trigger: dict | None) -> bool:
            nonlocal both_pose_latched
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
                    right
                    and left
                    and gR.pose_flags.get(gesture, False)
                    and gL.pose_flags.get(gesture, False)
                )
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

        exit_binding = mode_triggers.get("exit")
        exit_active = False
        if exit_binding:
            exit_active = _binding_active(
                exit_binding,
                dominant_side,
                support_side,
                gR,
                gL,
                bool(right),
                bool(left),
                evR,
                evL,
            )
        if exit_active:
            exit_hold_since = exit_hold_since or now_ms
        else:
            exit_hold_since = None
        exit_ready = bool(exit_hold_since and (now_ms - exit_hold_since) >= exit_hold_ms)

        if not calibration.active and not eval_session.active:
            if current_mode != "idle" and exit_ready:
                switch_mode("idle", now_ms, force_reset=True)
            elif _trigger_fired(mode_triggers.get("record")) and time_since_change >= mode_refractory_ms:
                if current_mode == "idle":
                    switch_mode("record", now_ms)
            elif mouse_enabled and _trigger_fired(mode_triggers.get("mouse")) and time_since_change >= mode_refractory_ms:
                if current_mode == "idle":
                    switch_mode("mouse", now_ms)
            elif calibration.enabled and mode_triggers.get("calibrate") and _trigger_fired(mode_triggers.get("calibrate")) and time_since_change >= mode_refractory_ms:
                switch_mode("calibrate", now_ms, force_reset=True)
            elif one_enabled and _trigger_fired(mode_triggers.get("one_hand")) and time_since_change >= mode_refractory_ms:
                if current_mode == "idle":
                    switch_mode("one_hand", now_ms)

        if current_mode != "mouse" or eval_session.active:
            mouse_prev = None
            if mouse_left_down:
                mouse_release("left")
                mouse_left_down = False
            if mouse_right_down:
                mouse_release("right")
                mouse_right_down = False

        if one_hand_active and not seq_active and not mouse_active and not eval_session.active:
            dispatch = one_hand_dispatcher.update(
                now_ms,
                right_present=bool(right),
                left_present=bool(left),
                right_event=evR,
                left_event=evL,
                right_pose_flags=gR.pose_flags,
                left_pose_flags=gL.pose_flags,
            )
            if dispatch:
                print(
                    f"[ONE-HAND] {dispatch.label} "
                    f"({dispatch.side} {dispatch.gesture}) -> {dispatch.combo}"
                )
                _send_hotkey(dispatch.combo)
                last_single_action = (
                    f"LAST: {dispatch.label} -> {dispatch.combo}"
                )

        if mouse_active and not eval_session.active:
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
                    dominant_side,
                    support_side,
                    gR,
                    gL,
                    right_present,
                    left_present,
                    evR,
                    evL,
                )
                right_active = _binding_active(
                    mouse_right_binding,
                    dominant_side,
                    support_side,
                    gR,
                    gL,
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

            scroll_source = right if scroll_side == "RIGHT" else left
            scroll_state = gR if scroll_side == "RIGHT" else gL
            scroll_active = bool(
                scroll_enabled
                and scroll_source
                and scroll_state.pose_flags.get(scroll_gesture, False)
            )
            if scroll_active:
                tip = scroll_source["lm"][scroll_landmark]
                if scroll_prev_y is None:
                    scroll_prev_y = tip[1]
                else:
                    dy = tip[1] - scroll_prev_y
                    if abs(dy) >= scroll_deadzone and (now_ms - scroll_last_ms) >= scroll_interval_ms:
                        delta = int(-dy * scroll_speed)
                        if delta != 0:
                            mouse_scroll(delta)
                            scroll_last_ms = now_ms
                        scroll_prev_y = tip[1]
            else:
                scroll_prev_y = None

        seq_hand = seq_input_side if seq_input_side in ("RIGHT", "LEFT") else ("RIGHT" if dominant_is_right else "LEFT")
        seq_last_label = last_R_label if seq_hand == "RIGHT" else last_L_label
        seq_present = bool(right) if seq_hand == "RIGHT" else bool(left)

        if seq_active and seq_present and not eval_session.active:
            candidate = seq_last_label
            if candidate and candidate not in candidate_ignore:
                prev_candidate = seq_pending
                if (prev_candidate is None) or (
                    prev_candidate != candidate and (now_ms - last_seq_event_ms) >= arm_delay_ms
                ):
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

        if seq_active and seq_pending and confirm_deb.update(now_ms, confirm_active) and not eval_session.active:
            if len(seq_buffer) < max_len and (now_ms - last_evt_ms) >= arm_delay_ms:
                seq_buffer.append(seq_pending)
                print(f"[SEQ] +{seq_pending}  buffer={seq_buffer}")
                seq_pending = None
                last_evt_ms = now_ms

        if seq_active and not eval_session.active:
            def _update_undo_for_side(side_name: str):
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
            commit_active = bool(
                gR.pose_flags.get(commit_gesture, False)
                and gL.pose_flags.get(commit_gesture, False)
            )
        elif commit_hand == "RIGHT":
            commit_active = bool(gR.pose_flags.get(commit_gesture, False))
        elif commit_hand == "LEFT":
            commit_active = bool(gL.pose_flags.get(commit_gesture, False))
        else:
            commit_active = bool(
                gR.pose_flags.get(commit_gesture, False)
                or gL.pose_flags.get(commit_gesture, False)
            )

        if seq_active and commit_deb.update(now_ms, commit_active) and not eval_session.active:
            if not seq_buffer:
                print("[SEQ-COMMIT] Empty sequence, commit skipped")
            else:
                key_tuple = tuple(seq_buffer)
                combo = lookup_mapping(seq_map, seq_hand, key_tuple) if key_tuple else None
                if combo and (now_ms - last_sent_ms) >= refractory_ms:
                    joined = " > ".join(seq_buffer)
                    print(f"[SEQ-COMMIT] {seq_hand} {joined} -> {combo}")
                    _send_hotkey(combo)
                    last_sent_ms = now_ms
                else:
                    joined = " > ".join(seq_buffer) if seq_buffer else "-"
                    print(f"[SEQ-COMMIT] No mapping for: {seq_hand} {joined}")
            seq_buffer.clear()
            seq_pending = None
            last_evt_ms = now_ms

        if eval_session.active:
            eval_event = evR if eval_session.hand_setting == "RIGHT" else evL
            eval_session.process(now_ms, eval_event)

        if eval_session.active:
            top, sub = eval_session.status_text()
        elif calibration.active:
            top, sub = calibration.status_text(now_ms)
        elif seq_active:
            top = "REC"
            sub = ("BUF: " + " > ".join(seq_buffer[-6:])) if seq_buffer else "BUF: -"
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
                sub = "BUF: -"
        osd.set_text(top, sub)

        hand_frame = frame.copy() if hand_windows_enabled else frame
        right_label = "R: -"
        if right:
            draw_landmarks(frame, right["lm"])
            label = active_pose_name(gR, last_R_label)
            right_label = f"R: {label}"
            draw_pose_label(frame, right["lm"], right_label, (0, 200, 255))
        left_label = "L: -"
        if left:
            draw_landmarks(frame, left["lm"])
            label = active_pose_name(gL, last_L_label)
            left_label = f"L: {label}"
            draw_pose_label(frame, left["lm"], left_label, (0, 255, 180))

        if mouse_active and not eval_session.active:
            frame_h, frame_w = frame.shape[:2]
            top_left = (int(mouse_rect["x"] * frame_w), int(mouse_rect["y"] * frame_h))
            bottom_right = (
                int((mouse_rect["x"] + mouse_rect["w"]) * frame_w),
                int((mouse_rect["y"] + mouse_rect["h"]) * frame_h),
            )
            cv2.rectangle(frame, top_left, bottom_right, (0, 255, 255), 2)

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

        if hand_windows_enabled:
            right_crop = hand_crop(hand_frame, right["lm"], hand_window_padding) if right else None
            left_crop = hand_crop(hand_frame, left["lm"], hand_window_padding) if left else None
            render_hand_window("GCPC - Right Hand", right_crop, right_label, hand_window_size, (0, 200, 255))
            render_hand_window("GCPC - Left Hand", left_crop, left_label, hand_window_size, (0, 255, 180))
            if not hand_windows_placed:
                cv2.moveWindow("GCPC - Left Hand", hand_window_margin, hand_window_margin)
                cv2.moveWindow(
                    "GCPC - Right Hand",
                    hand_window_margin * 2 + hand_window_size,
                    hand_window_margin,
                )
                hand_windows_placed = True

        if not hand_windows_enabled or show_full_camera:
            cv2.imshow("GCPC - Camera", frame)

        raw_key = cv2.waitKey(1)
        key = raw_key & 0xFF if raw_key != -1 else -1
        if raw_key == 27 or key == 27:
            break
        if calibration.enabled and key != -1:
            try:
                key_chr = chr(key).lower()
            except ValueError:
                key_chr = ""
            if key_chr == calibration.trigger_key:
                switch_mode("calibrate", now_ms, force_reset=True)

    if measurements_enabled:
        append_metrics_row(_session_summary_row())
    if cap is not None:
        cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    setup_logging()
    install_exception_hooks()
    try:
        main()
    except Exception:
        report_fatal_exception(context="Fatal error in GCPC main loop")
        raise SystemExit(1)
