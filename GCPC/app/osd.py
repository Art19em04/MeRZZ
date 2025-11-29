# -*- coding: utf-8 -*-
import ctypes

from PySide6 import QtWidgets, QtCore, QtGui

WS_EX_LAYERED = 0x00080000
WS_EX_TRANSPARENT = 0x00000020


class OSD(QtWidgets.QWidget):
    """On-screen display widget showing current mode and hints."""
    def __init__(self):
        super().__init__(None, QtCore.Qt.FramelessWindowHint | QtCore.Qt.Tool | QtCore.Qt.WindowStaysOnTopHint)
        self.setAttribute(QtCore.Qt.WA_TranslucentBackground, True)
        self.setWindowFlag(QtCore.Qt.WindowDoesNotAcceptFocus, True)
        self.text_main = "";
        self.text_sub = ""
        self.font_main = QtGui.QFont("Segoe UI", 12, QtGui.QFont.Bold)
        self.font_sub = QtGui.QFont("Segoe UI", 10)
        self._install_click_through();
        self._resize_top()
        self.timer = QtCore.QTimer(self);
        self.timer.setInterval(33);
        self.timer.timeout.connect(self.update);
        self.timer.start()

    def _install_click_through(self):
        """Allow mouse clicks to pass through the widget."""
        hwnd = int(self.winId())
        GWL_EXSTYLE = -20
        GetWindowLong = ctypes.windll.user32.GetWindowLongW
        SetWindowLong = ctypes.windll.user32.SetWindowLongW
        ex = GetWindowLong(hwnd, GWL_EXSTYLE)
        SetWindowLong(hwnd, GWL_EXSTYLE, ex | WS_EX_LAYERED | WS_EX_TRANSPARENT)

    def _resize_top(self):
        """Stretch the overlay across the top of the primary screen."""
        scr = QtWidgets.QApplication.primaryScreen().availableGeometry()
        self.setGeometry(scr.left(), scr.top(), scr.width(), 52)

    def set_text(self, main, sub=""):
        """Update displayed main and secondary text."""
        self.text_main = main or ""
        self.text_sub = sub or ""
        self.update()

    def paintEvent(self, e):
        """Render overlay background and text."""
        p = QtGui.QPainter(self)
        p.setRenderHint(QtGui.QPainter.Antialiasing)
        r = self.rect()
        p.fillRect(r, QtGui.QColor(0, 0, 0, 120))
        pen = QtGui.QPen(QtGui.QColor(80, 200, 255, 180))
        pen.setWidth(2)
        p.setPen(pen)
        p.drawRect(r.adjusted(1, 1, -2, -2))
        p.setPen(QtGui.QColor(230, 255, 255))
        p.setFont(self.font_main)
        p.drawText(r.adjusted(12, 8, -12, -8), QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter, self.text_main)
        p.setFont(self.font_sub)
        p.setPen(QtGui.QColor(180, 235, 235))
        p.drawText(r.adjusted(12, 26, -12, -8), QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter, self.text_sub)

    def showEvent(self, e):
        """Ensure overlay is resized when shown."""
        super().showEvent(e)
        self._resize_top()
