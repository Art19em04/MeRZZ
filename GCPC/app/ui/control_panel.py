# -*- coding: utf-8 -*-
from __future__ import annotations

from PySide6 import QtCore, QtWidgets


class ControlPanel(QtWidgets.QWidget):
    """Always-on-top launcher panel for camera, hand control and settings."""

    settings_requested = QtCore.Signal()
    camera_resolution_changed = QtCore.Signal(int, int)

    def __init__(
        self,
        default_camera_enabled: bool = False,
        default_hand_enabled: bool = False,
        default_resolution: tuple[int, int] = (640, 360),
    ):
        super().__init__(None, QtCore.Qt.Tool | QtCore.Qt.WindowStaysOnTopHint)
        self.setWindowTitle("GCPC Controls")
        self.hand_control_enabled = bool(default_hand_enabled)
        self.camera_enabled = bool(default_camera_enabled)

        layout = QtWidgets.QVBoxLayout(self)
        self.hand_btn = QtWidgets.QPushButton(self._hand_label())
        self.hand_btn.setCheckable(True)
        self.hand_btn.setChecked(self.hand_control_enabled)
        self.hand_btn.clicked.connect(self._toggle_hand_control)

        self.camera_btn = QtWidgets.QPushButton(self._camera_label())
        self.camera_btn.setCheckable(True)
        self.camera_btn.setChecked(self.camera_enabled)
        self.camera_btn.clicked.connect(self._toggle_camera)

        self.camera_resolution_combo = QtWidgets.QComboBox(self)
        selected_index = 0
        default_width = int(default_resolution[0])
        default_height = int(default_resolution[1])
        default_found = False
        for index, (text, width, height) in enumerate(self._resolution_options()):
            self.camera_resolution_combo.addItem(text, (width, height))
            if (int(width), int(height)) == (default_width, default_height):
                selected_index = index
                default_found = True
        if not default_found:
            selected_index = self.camera_resolution_combo.count()
            self.camera_resolution_combo.addItem(
                f"{default_width}x{default_height}",
                (default_width, default_height),
            )
        self.camera_resolution_combo.setCurrentIndex(selected_index)
        self.camera_resolution_combo.currentIndexChanged.connect(
            self._emit_camera_resolution_changed
        )

        self.settings_btn = QtWidgets.QPushButton("Gesture settings")
        self.settings_btn.clicked.connect(self.settings_requested.emit)

        camera_row = QtWidgets.QHBoxLayout()
        camera_row.addWidget(self.camera_btn, 1)
        camera_row.addWidget(self.camera_resolution_combo, 1)

        layout.addWidget(self.hand_btn)
        layout.addLayout(camera_row)
        layout.addWidget(self.settings_btn)

    @staticmethod
    def _resolution_options():
        return [
            ("640x360", 640, 360),
            ("640x480", 640, 480),
            ("960x540", 960, 540),
            ("1280x720", 1280, 720),
            ("1920x1080", 1920, 1080),
        ]

    def _hand_label(self) -> str:
        return "Hand control: ON" if self.hand_control_enabled else "Hand control: OFF"

    def _camera_label(self) -> str:
        return "Camera: ON" if self.camera_enabled else "Camera: OFF"

    def _toggle_hand_control(self) -> None:
        self.hand_control_enabled = self.hand_btn.isChecked()
        self.hand_btn.setText(self._hand_label())

    def _toggle_camera(self) -> None:
        self.camera_enabled = self.camera_btn.isChecked()
        self.camera_btn.setText(self._camera_label())

    def current_interaction(self) -> str:
        return "gestures"

    def is_armed(self) -> bool:
        return True

    def is_hand_control_enabled(self) -> bool:
        return self.hand_control_enabled

    def is_camera_enabled(self) -> bool:
        return self.camera_enabled

    def selected_camera_resolution(self) -> tuple[int, int]:
        data = self.camera_resolution_combo.currentData()
        if isinstance(data, tuple) and len(data) == 2:
            return int(data[0]), int(data[1])
        return 640, 360

    def _emit_camera_resolution_changed(self, _index: int) -> None:
        width, height = self.selected_camera_resolution()
        self.camera_resolution_changed.emit(width, height)
