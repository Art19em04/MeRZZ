"""Dialog for simplified gesture configuration."""
from __future__ import annotations

from typing import Any, Dict, Iterable, Tuple

from PySide6 import QtWidgets


HAND_OPTIONS = (
    ("DOMINANT", "Dominant"),
    ("NON_DOMINANT", "Non-dominant"),
    ("RIGHT", "Right"),
    ("LEFT", "Left"),
    ("BOTH", "Both"),
    ("EITHER", "Either"),
)

SINGLE_HAND_OPTIONS = (
    ("DOMINANT", "Dominant"),
    ("NON_DOMINANT", "Non-dominant"),
    ("RIGHT", "Right"),
    ("LEFT", "Left"),
)

SIDE_OPTIONS = (
    ("RIGHT", "Right"),
    ("LEFT", "Left"),
)

GESTURE_OPTIONS = (
    ("PINCH", "PINCH"),
    ("PINCH_MIDDLE", "PINCH_MIDDLE"),
    ("FIST", "FIST"),
    ("THUMBS_UP", "THUMBS_UP"),
    ("OPEN_PALM", "OPEN_PALM"),
    ("SWIPE_RIGHT", "SWIPE_RIGHT"),
    ("SWIPE_LEFT", "SWIPE_LEFT"),
)

POSE_GESTURE_OPTIONS = (
    ("PINCH", "PINCH"),
    ("PINCH_MIDDLE", "PINCH_MIDDLE"),
    ("FIST", "FIST"),
    ("THUMBS_UP", "THUMBS_UP"),
    ("OPEN_PALM", "OPEN_PALM"),
)

DISPATCH_OPTIONS = (
    ("DOMINANT", "Dominant"),
    ("NON_DOMINANT", "Non-dominant"),
    ("RIGHT", "Right"),
    ("LEFT", "Left"),
    ("BOTH", "Both"),
    ("EITHER", "Either"),
)


def _normalized_token(value: Any, fallback: str) -> str:
    if value is None:
        return fallback
    token = str(value).strip().upper()
    return token or fallback


def _parse_single_binding(raw_value: Any, fallback_hand: str, fallback_gesture: str) -> Tuple[str, str]:
    raw = _normalized_token(raw_value, "")
    if not raw:
        return fallback_hand, fallback_gesture

    if "-" in raw:
        hand, gesture = raw.rsplit("-", 1)
        hand = _normalized_token(hand, fallback_hand)
        gesture = _normalized_token(gesture, fallback_gesture)
        return hand, gesture

    return fallback_hand, _normalized_token(raw, fallback_gesture)


def _parse_sequence_binding(
    raw_value: Any,
    fallback_hand: str,
    fallback_start: str,
    fallback_end: str,
) -> Tuple[str, str, str]:
    raw = str(raw_value or "")
    parts = [part.strip() for part in raw.split(">") if part.strip()]
    if not parts:
        return fallback_hand, fallback_start, fallback_end

    hand, start = _parse_single_binding(parts[0], fallback_hand, fallback_start)
    end_hand, end = _parse_single_binding(parts[-1], hand, fallback_end)
    return _normalized_token(end_hand, hand), start, end


def _compose_single_binding(hand: str, gesture: str) -> str:
    return f"{hand}-{gesture}"


def _coerce_int(value: Any, fallback: int) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return fallback


def _get_mode_binding(functional_cfg: Dict[str, Any], mode_name: str, fallback: Tuple[str, str]) -> Tuple[str, str]:
    for raw_key, target in functional_cfg.items():
        if target != mode_name or not isinstance(raw_key, str):
            continue
        return _parse_single_binding(raw_key, fallback[0], fallback[1])
    return fallback


class GestureSettingsDialog(QtWidgets.QDialog):
    """Simple UI for the most frequently changed gesture bindings."""

    def __init__(
        self,
        cfg: Dict[str, Any],
        parent: QtWidgets.QWidget | None = None,
        on_request_calibration=None,
    ):
        super().__init__(parent)
        self.cfg = cfg
        self._on_request_calibration = on_request_calibration
        self.setWindowTitle("GCPC Gesture Settings")
        self.resize(720, 700)

        root_layout = QtWidgets.QVBoxLayout(self)

        scroll = QtWidgets.QScrollArea(self)
        scroll.setWidgetResizable(True)
        root_layout.addWidget(scroll)

        content = QtWidgets.QWidget()
        scroll.setWidget(content)
        content_layout = QtWidgets.QVBoxLayout(content)
        content_layout.setSpacing(12)

        self._build_hands_group(content_layout)
        self._build_camera_group(content_layout)
        self._build_mode_group(content_layout)
        self._build_sequence_group(content_layout)
        self._build_mouse_group(content_layout)
        self._build_keymap_group(content_layout)
        self._build_calibration_group(content_layout)

        note = QtWidgets.QLabel(
            "Saved to config.json. Restart GCPC to apply all runtime changes."
        )
        note.setWordWrap(True)
        content_layout.addWidget(note)
        content_layout.addStretch(1)

        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Save | QtWidgets.QDialogButtonBox.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        root_layout.addWidget(buttons)

    def _make_combo(self, options: Iterable[Tuple[str, str]], selected: str) -> QtWidgets.QComboBox:
        combo = QtWidgets.QComboBox(self)
        selected = _normalized_token(selected, "")
        found_index = 0
        for index, (value, label) in enumerate(options):
            combo.addItem(label, value)
            if value == selected:
                found_index = index
        combo.setCurrentIndex(found_index)
        return combo

    def _make_spin(self, value: int, minimum: int, maximum: int, step: int = 10) -> QtWidgets.QSpinBox:
        spin = QtWidgets.QSpinBox(self)
        spin.setRange(minimum, maximum)
        spin.setSingleStep(step)
        spin.setValue(max(minimum, min(maximum, int(value))))
        return spin

    def _add_binding_row(
        self,
        form: QtWidgets.QFormLayout,
        label: str,
        hand: str,
        gesture: str,
        hand_options: Iterable[Tuple[str, str]] = HAND_OPTIONS,
        gesture_options: Iterable[Tuple[str, str]] = GESTURE_OPTIONS,
    ) -> Tuple[QtWidgets.QComboBox, QtWidgets.QComboBox]:
        row = QtWidgets.QHBoxLayout()
        hand_combo = self._make_combo(hand_options, hand)
        gesture_combo = self._make_combo(gesture_options, gesture)
        row.addWidget(hand_combo, 1)
        row.addWidget(gesture_combo, 1)

        wrapper = QtWidgets.QWidget(self)
        wrapper.setLayout(row)
        form.addRow(label, wrapper)
        return hand_combo, gesture_combo

    def _build_hands_group(self, parent_layout: QtWidgets.QVBoxLayout) -> None:
        group = QtWidgets.QGroupBox("Hands", self)
        form = QtWidgets.QFormLayout(group)

        hands_cfg = self.cfg.get("hands") or {}
        dominant = _normalized_token(hands_cfg.get("dominant"), "RIGHT")
        support = _normalized_token(hands_cfg.get("support"), "LEFT")

        self.dominant_combo = self._make_combo(SIDE_OPTIONS, dominant)
        self.support_combo = self._make_combo(SIDE_OPTIONS, support)
        form.addRow("Dominant hand", self.dominant_combo)
        form.addRow("Support hand", self.support_combo)
        parent_layout.addWidget(group)

    def _build_camera_group(self, parent_layout: QtWidgets.QVBoxLayout) -> None:
        group = QtWidgets.QGroupBox("Camera", self)
        layout = QtWidgets.QVBoxLayout(group)
        ui_cfg = self.cfg.get("ui") or {}
        hand_windows_cfg = ui_cfg.get("hand_windows") or {}
        hands_only = bool(hand_windows_cfg.get("enabled", False)) and not bool(
            hand_windows_cfg.get("show_full_camera", True)
        )
        self.hands_only_camera_checkbox = QtWidgets.QCheckBox("Display only hands windows", self)
        self.hands_only_camera_checkbox.setChecked(hands_only)
        layout.addWidget(self.hands_only_camera_checkbox)
        parent_layout.addWidget(group)

    def _build_mode_group(self, parent_layout: QtWidgets.QVBoxLayout) -> None:
        group = QtWidgets.QGroupBox("Mode switching gestures", self)
        form = QtWidgets.QFormLayout(group)

        functional = (self.cfg.get("command_mappings") or {}).get("functional") or {}
        mode_record = _get_mode_binding(functional, "MODE_RECORD", ("BOTH", "FIST"))
        mode_mouse = _get_mode_binding(functional, "MODE_MOUSE", ("NON_DOMINANT", "THUMBS_UP"))
        mode_one_hand = _get_mode_binding(functional, "MODE_ONE_HAND", ("BOTH", "THUMBS_UP"))
        mode_exit = _get_mode_binding(functional, "MODE_EXIT", ("BOTH", "OPEN_PALM"))

        self.record_hand_combo, self.record_gesture_combo = self._add_binding_row(
            form, "Record mode", mode_record[0], mode_record[1]
        )
        self.mouse_hand_combo, self.mouse_gesture_combo = self._add_binding_row(
            form, "Mouse mode", mode_mouse[0], mode_mouse[1]
        )
        self.one_hand_mode_hand_combo, self.one_hand_mode_gesture_combo = self._add_binding_row(
            form, "One-hand mode", mode_one_hand[0], mode_one_hand[1]
        )
        self.exit_mode_hand_combo, self.exit_mode_gesture_combo = self._add_binding_row(
            form, "Exit mode", mode_exit[0], mode_exit[1]
        )
        parent_layout.addWidget(group)

    def _build_sequence_group(self, parent_layout: QtWidgets.QVBoxLayout) -> None:
        group = QtWidgets.QGroupBox("Sequence controls", self)
        form = QtWidgets.QFormLayout(group)

        controls = (self.cfg.get("controls") or {}).get("sequence") or {}
        input_hand = _normalized_token(controls.get("input_hand"), "DOMINANT")
        confirm_cfg = controls.get("confirm") or {}
        undo_cfg = controls.get("undo") or {}
        commit_cfg = controls.get("commit") or {}

        confirm_hand, confirm_gesture = _parse_single_binding(
            confirm_cfg.get("binding"),
            "NON_DOMINANT",
            "PINCH",
        )
        undo_hand, undo_start, undo_end = _parse_sequence_binding(
            undo_cfg.get("binding"),
            "NON_DOMINANT",
            "OPEN_PALM",
            "FIST",
        )
        commit_hand, commit_gesture = _parse_single_binding(
            commit_cfg.get("binding"),
            "BOTH",
            "FIST",
        )

        self.seq_input_hand_combo = self._make_combo(DISPATCH_OPTIONS, input_hand)
        form.addRow("Input hand", self.seq_input_hand_combo)

        self.confirm_hand_combo, self.confirm_gesture_combo = self._add_binding_row(
            form, "Confirm", confirm_hand, confirm_gesture
        )
        self.confirm_dwell_spin = self._make_spin(confirm_cfg.get("dwell_ms", 220), 50, 2000, 10)
        self.confirm_refractory_spin = self._make_spin(
            confirm_cfg.get("refractory_ms", 700), 50, 5000, 10
        )
        form.addRow("Confirm dwell (ms)", self.confirm_dwell_spin)
        form.addRow("Confirm refractory (ms)", self.confirm_refractory_spin)

        self.undo_hand_combo = self._make_combo(HAND_OPTIONS, undo_hand)
        self.undo_start_combo = self._make_combo(GESTURE_OPTIONS, undo_start)
        self.undo_end_combo = self._make_combo(GESTURE_OPTIONS, undo_end)
        undo_row = QtWidgets.QHBoxLayout()
        undo_row.addWidget(self.undo_hand_combo, 1)
        undo_row.addWidget(self.undo_start_combo, 1)
        undo_row.addWidget(self.undo_end_combo, 1)
        undo_wrap = QtWidgets.QWidget(self)
        undo_wrap.setLayout(undo_row)
        form.addRow("Undo sequence", undo_wrap)
        self.undo_window_spin = self._make_spin(undo_cfg.get("window_ms", 900), 100, 5000, 10)
        form.addRow("Undo window (ms)", self.undo_window_spin)

        self.commit_hand_combo, self.commit_gesture_combo = self._add_binding_row(
            form, "Commit", commit_hand, commit_gesture
        )
        self.commit_dwell_spin = self._make_spin(commit_cfg.get("dwell_ms", 260), 50, 3000, 10)
        self.commit_refractory_spin = self._make_spin(
            commit_cfg.get("refractory_ms", 1200), 50, 5000, 10
        )
        form.addRow("Commit dwell (ms)", self.commit_dwell_spin)
        form.addRow("Commit refractory (ms)", self.commit_refractory_spin)
        parent_layout.addWidget(group)

    def _build_mouse_group(self, parent_layout: QtWidgets.QVBoxLayout) -> None:
        group = QtWidgets.QGroupBox("Mouse actions", self)
        form = QtWidgets.QFormLayout(group)

        mouse_cfg = self.cfg.get("mouse_control") or {}
        pointer_hand = _normalized_token(mouse_cfg.get("pointer_hand"), "DOMINANT")
        rect_cfg = mouse_cfg.get("control_rect") or {}
        scroll_cfg = mouse_cfg.get("scroll") or {}
        scroll_hand = _normalized_token(scroll_cfg.get("hand"), "RIGHT")
        scroll_gesture = _normalized_token(scroll_cfg.get("gesture"), "FIST")
        left_hand, left_gesture = _parse_single_binding(
            mouse_cfg.get("left_click_binding"),
            "NON_DOMINANT",
            "PINCH",
        )
        right_hand, right_gesture = _parse_single_binding(
            mouse_cfg.get("right_click_binding"),
            "NON_DOMINANT",
            "PINCH_MIDDLE",
        )

        self.pointer_hand_combo = self._make_combo(HAND_OPTIONS, pointer_hand)
        form.addRow("Pointer hand", self.pointer_hand_combo)
        self.mouse_left_hand_combo, self.mouse_left_gesture_combo = self._add_binding_row(
            form, "Left click", left_hand, left_gesture
        )
        self.mouse_right_hand_combo, self.mouse_right_gesture_combo = self._add_binding_row(
            form, "Right click", right_hand, right_gesture
        )
        self.scroll_hand_combo, self.scroll_gesture_combo = self._add_binding_row(
            form,
            "Scroll gesture",
            scroll_hand,
            scroll_gesture,
            hand_options=SINGLE_HAND_OPTIONS,
            gesture_options=POSE_GESTURE_OPTIONS,
        )
        self.scroll_speed_spin = self._make_spin(
            _coerce_int(scroll_cfg.get("speed", 1200), 1200),
            0,
            10000,
            50,
        )
        form.addRow("Scroll speed", self.scroll_speed_spin)
        self.mouse_rect_width_spin = QtWidgets.QDoubleSpinBox(self)
        self.mouse_rect_width_spin.setRange(0.1, 1.0)
        self.mouse_rect_width_spin.setSingleStep(0.05)
        self.mouse_rect_width_spin.setDecimals(2)
        self.mouse_rect_width_spin.setValue(float(rect_cfg.get("width", 0.5)))
        self.mouse_rect_height_spin = QtWidgets.QDoubleSpinBox(self)
        self.mouse_rect_height_spin.setRange(0.1, 1.0)
        self.mouse_rect_height_spin.setSingleStep(0.05)
        self.mouse_rect_height_spin.setDecimals(2)
        self.mouse_rect_height_spin.setValue(float(rect_cfg.get("height", 0.5)))
        form.addRow("Mouse area width", self.mouse_rect_width_spin)
        form.addRow("Mouse area height", self.mouse_rect_height_spin)
        parent_layout.addWidget(group)

    def _new_mapping_table(self, mapping: Dict[str, Any]) -> QtWidgets.QTableWidget:
        table = QtWidgets.QTableWidget(self)
        table.setColumnCount(2)
        table.setHorizontalHeaderLabels(["Gesture binding", "Hotkey"])
        table.horizontalHeader().setStretchLastSection(True)
        table.verticalHeader().setVisible(False)
        table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        table.setSelectionMode(QtWidgets.QAbstractItemView.SingleSelection)
        for key, value in (mapping or {}).items():
            self._add_mapping_row(table, str(key), str(value))
        return table

    def _add_mapping_row(self, table: QtWidgets.QTableWidget, key: str = "", value: str = "") -> None:
        row = table.rowCount()
        table.insertRow(row)
        table.setItem(row, 0, QtWidgets.QTableWidgetItem(key))
        table.setItem(row, 1, QtWidgets.QTableWidgetItem(value))

    def _remove_mapping_row(self, table: QtWidgets.QTableWidget) -> None:
        row = table.currentRow()
        if row >= 0:
            table.removeRow(row)

    def _read_mapping_table(self, table: QtWidgets.QTableWidget) -> Dict[str, str]:
        result: Dict[str, str] = {}
        for row in range(table.rowCount()):
            key_item = table.item(row, 0)
            value_item = table.item(row, 1)
            key = (key_item.text() if key_item else "").strip()
            value = (value_item.text() if value_item else "").strip()
            if key and value:
                result[key] = value
        return result

    def _build_keymap_group(self, parent_layout: QtWidgets.QVBoxLayout) -> None:
        group = QtWidgets.QGroupBox("Hotkey mappings", self)
        layout = QtWidgets.QVBoxLayout(group)

        mappings = self.cfg.get("command_mappings") or {}
        single_map = mappings.get("single_gestures") or {}
        complex_map = mappings.get("complex_gestures") or {}

        single_label = QtWidgets.QLabel("Single gestures")
        self.single_map_table = self._new_mapping_table(single_map)
        single_controls = QtWidgets.QHBoxLayout()
        single_add_btn = QtWidgets.QPushButton("Add")
        single_add_btn.clicked.connect(lambda: self._add_mapping_row(self.single_map_table))
        single_remove_btn = QtWidgets.QPushButton("Remove selected")
        single_remove_btn.clicked.connect(lambda: self._remove_mapping_row(self.single_map_table))
        single_controls.addWidget(single_add_btn)
        single_controls.addWidget(single_remove_btn)
        single_controls.addStretch(1)

        complex_label = QtWidgets.QLabel("Complex gestures (sequence)")
        self.complex_map_table = self._new_mapping_table(complex_map)
        complex_controls = QtWidgets.QHBoxLayout()
        complex_add_btn = QtWidgets.QPushButton("Add")
        complex_add_btn.clicked.connect(lambda: self._add_mapping_row(self.complex_map_table))
        complex_remove_btn = QtWidgets.QPushButton("Remove selected")
        complex_remove_btn.clicked.connect(lambda: self._remove_mapping_row(self.complex_map_table))
        complex_controls.addWidget(complex_add_btn)
        complex_controls.addWidget(complex_remove_btn)
        complex_controls.addStretch(1)

        layout.addWidget(single_label)
        layout.addWidget(self.single_map_table)
        layout.addLayout(single_controls)
        layout.addWidget(complex_label)
        layout.addWidget(self.complex_map_table)
        layout.addLayout(complex_controls)
        parent_layout.addWidget(group)

    def _build_calibration_group(self, parent_layout: QtWidgets.QVBoxLayout) -> None:
        group = QtWidgets.QGroupBox("Calibration", self)
        row = QtWidgets.QHBoxLayout(group)
        self.calibrate_btn = QtWidgets.QPushButton("Run calibration now")
        self.calibrate_btn.clicked.connect(self._request_calibration)
        row.addWidget(self.calibrate_btn)
        row.addStretch(1)
        parent_layout.addWidget(group)

    def _request_calibration(self) -> None:
        if callable(self._on_request_calibration):
            self._on_request_calibration()
            QtWidgets.QMessageBox.information(
                self,
                "Calibration",
                "Calibration was requested. Ensure camera and hand control are enabled.",
            )
            return
        QtWidgets.QMessageBox.warning(
            self,
            "Calibration",
            "Calibration is unavailable in current runtime context.",
        )

    def _combo_value(self, combo: QtWidgets.QComboBox, fallback: str) -> str:
        value = combo.currentData()
        return _normalized_token(value, fallback)

    def _save_to_config(self) -> None:
        dominant = self._combo_value(self.dominant_combo, "RIGHT")
        support = self._combo_value(self.support_combo, "LEFT")
        if dominant == support:
            raise ValueError("Dominant and support hand must be different.")

        hands_cfg = self.cfg.setdefault("hands", {})
        hands_cfg["dominant"] = dominant
        hands_cfg["support"] = support

        command_mappings = self.cfg.setdefault("command_mappings", {})
        functional_cfg = command_mappings.setdefault("functional", {})
        keep_values = {
            key: value
            for key, value in functional_cfg.items()
            if not (isinstance(value, str) and value.startswith("MODE_"))
        }
        mode_bindings = {
            "MODE_RECORD": _compose_single_binding(
                self._combo_value(self.record_hand_combo, "BOTH"),
                self._combo_value(self.record_gesture_combo, "FIST"),
            ),
            "MODE_MOUSE": _compose_single_binding(
                self._combo_value(self.mouse_hand_combo, "NON_DOMINANT"),
                self._combo_value(self.mouse_gesture_combo, "THUMBS_UP"),
            ),
            "MODE_ONE_HAND": _compose_single_binding(
                self._combo_value(self.one_hand_mode_hand_combo, "BOTH"),
                self._combo_value(self.one_hand_mode_gesture_combo, "THUMBS_UP"),
            ),
            "MODE_EXIT": _compose_single_binding(
                self._combo_value(self.exit_mode_hand_combo, "BOTH"),
                self._combo_value(self.exit_mode_gesture_combo, "OPEN_PALM"),
            ),
        }
        if len(set(mode_bindings.values())) != len(mode_bindings):
            raise ValueError("Mode switch gestures must be unique.")

        functional_cfg.clear()
        functional_cfg.update(keep_values)
        for mode_name, binding in mode_bindings.items():
            functional_cfg[binding] = mode_name
        command_mappings["single_gestures"] = self._read_mapping_table(self.single_map_table)
        command_mappings["complex_gestures"] = self._read_mapping_table(self.complex_map_table)

        controls_cfg = self.cfg.setdefault("controls", {}).setdefault("sequence", {})
        controls_cfg["input_hand"] = self._combo_value(self.seq_input_hand_combo, "DOMINANT")

        confirm_cfg = controls_cfg.setdefault("confirm", {})
        confirm_cfg["binding"] = _compose_single_binding(
            self._combo_value(self.confirm_hand_combo, "NON_DOMINANT"),
            self._combo_value(self.confirm_gesture_combo, "PINCH"),
        )
        confirm_cfg["dwell_ms"] = int(self.confirm_dwell_spin.value())
        confirm_cfg["refractory_ms"] = int(self.confirm_refractory_spin.value())

        undo_cfg = controls_cfg.setdefault("undo", {})
        undo_hand = self._combo_value(self.undo_hand_combo, "NON_DOMINANT")
        undo_start = self._combo_value(self.undo_start_combo, "OPEN_PALM")
        undo_end = self._combo_value(self.undo_end_combo, "FIST")
        undo_cfg["binding"] = (
            f"{_compose_single_binding(undo_hand, undo_start)} > "
            f"{_compose_single_binding(undo_hand, undo_end)}"
        )
        undo_cfg["window_ms"] = int(self.undo_window_spin.value())

        commit_cfg = controls_cfg.setdefault("commit", {})
        commit_cfg["binding"] = _compose_single_binding(
            self._combo_value(self.commit_hand_combo, "BOTH"),
            self._combo_value(self.commit_gesture_combo, "FIST"),
        )
        commit_cfg["dwell_ms"] = int(self.commit_dwell_spin.value())
        commit_cfg["refractory_ms"] = int(self.commit_refractory_spin.value())

        mouse_cfg = self.cfg.setdefault("mouse_control", {})
        mouse_cfg["pointer_hand"] = self._combo_value(self.pointer_hand_combo, "DOMINANT")
        mouse_cfg["left_click_binding"] = _compose_single_binding(
            self._combo_value(self.mouse_left_hand_combo, "NON_DOMINANT"),
            self._combo_value(self.mouse_left_gesture_combo, "PINCH"),
        )
        mouse_cfg["right_click_binding"] = _compose_single_binding(
            self._combo_value(self.mouse_right_hand_combo, "NON_DOMINANT"),
            self._combo_value(self.mouse_right_gesture_combo, "PINCH_MIDDLE"),
        )
        rect_cfg = mouse_cfg.setdefault("control_rect", {})
        rect_cfg["width"] = float(self.mouse_rect_width_spin.value())
        rect_cfg["height"] = float(self.mouse_rect_height_spin.value())
        scroll_cfg = mouse_cfg.setdefault("scroll", {})
        scroll_cfg["hand"] = self._combo_value(self.scroll_hand_combo, "RIGHT")
        scroll_cfg["gesture"] = self._combo_value(self.scroll_gesture_combo, "FIST")
        scroll_cfg["speed"] = int(self.scroll_speed_spin.value())

        one_hand_cfg = self.cfg.setdefault("one_hand_mode", {})
        one_hand_cfg.pop("dispatch_hand", None)

        ui_cfg = self.cfg.setdefault("ui", {})
        hand_windows_cfg = ui_cfg.setdefault("hand_windows", {})
        hands_only = bool(self.hands_only_camera_checkbox.isChecked())
        hand_windows_cfg["enabled"] = hands_only
        hand_windows_cfg["show_full_camera"] = not hands_only

    def accept(self) -> None:
        try:
            self._save_to_config()
        except ValueError as exc:
            QtWidgets.QMessageBox.warning(self, "Validation error", str(exc))
            return
        super().accept()
