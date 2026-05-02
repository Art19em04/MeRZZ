"""Runtime logging and user-facing fatal error handling."""
from __future__ import annotations

import logging
import sys
import threading
import traceback
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Tuple

from PySide6 import QtWidgets

from app.utils.config import ROOT


LOG_DIR = ROOT / "logs"
LOG_FILE = LOG_DIR / "gcpc.log"
LOGGER_NAME = "gcpc.runtime"


def setup_logging() -> Path:
    """Configure rotating file logging and console mirror."""
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    root = logging.getLogger()
    root.setLevel(logging.INFO)

    file_handler_exists = False
    for handler in root.handlers:
        base_name = getattr(handler, "baseFilename", "")
        if base_name and Path(base_name).resolve() == LOG_FILE.resolve():
            file_handler_exists = True
            break

    if not file_handler_exists:
        file_handler = RotatingFileHandler(
            LOG_FILE,
            maxBytes=2_000_000,
            backupCount=5,
            encoding="utf-8",
        )
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(
            logging.Formatter(
                "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
        )
        root.addHandler(file_handler)

    console_handler_exists = any(
        isinstance(handler, logging.StreamHandler) and not getattr(handler, "baseFilename", None)
        for handler in root.handlers
    )
    if not console_handler_exists:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(logging.Formatter("%(levelname)s %(name)s: %(message)s"))
        root.addHandler(console_handler)

    logging.getLogger(LOGGER_NAME).info("Logging initialized at %s", LOG_FILE)
    return LOG_FILE


def _show_error_dialog(title: str, message: str, details: str) -> None:
    """Show fatal error details to the user if Qt is available."""
    try:
        app = QtWidgets.QApplication.instance()
        owns_app = False
        if app is None:
            app = QtWidgets.QApplication([])
            owns_app = True

        dialog = QtWidgets.QMessageBox()
        dialog.setIcon(QtWidgets.QMessageBox.Critical)
        dialog.setWindowTitle(title)
        dialog.setText(message)
        dialog.setDetailedText(details)
        dialog.setStandardButtons(QtWidgets.QMessageBox.Ok)
        dialog.exec()

        if owns_app:
            app.quit()
    except Exception:
        logging.getLogger(LOGGER_NAME).exception(
            "Failed to show Qt error dialog; falling back to console output"
        )
        print(title)
        print(message)
        if details:
            print(details)


def report_fatal_exception(
    exc_info: Tuple[type[BaseException], BaseException, object] | None = None,
    *,
    context: str = "Fatal application error",
) -> None:
    """Write full traceback to log and show a user-readable error dialog."""
    if exc_info is None:
        exc_info = sys.exc_info()
    exc_type, exc_value, exc_tb = exc_info
    if not exc_type or not exc_value:
        return

    if issubclass(exc_type, KeyboardInterrupt):
        return

    logger = logging.getLogger(LOGGER_NAME)
    logger.exception(context, exc_info=(exc_type, exc_value, exc_tb))

    details = "".join(traceback.format_exception(exc_type, exc_value, exc_tb))
    message = (
        "GCPC encountered an unexpected error and needs to stop.\n"
        f"Details were saved to:\n{LOG_FILE}"
    )
    _show_error_dialog("GCPC Error", message, details)


def install_exception_hooks() -> None:
    """Install global exception hooks for main and worker threads."""

    def _sys_hook(exc_type, exc_value, exc_tb):
        report_fatal_exception(
            (exc_type, exc_value, exc_tb),
            context="Unhandled exception in main thread",
        )

    def _thread_hook(args):
        report_fatal_exception(
            (args.exc_type, args.exc_value, args.exc_traceback),
            context=f"Unhandled exception in thread {args.thread.name}",
        )

    sys.excepthook = _sys_hook
    if hasattr(threading, "excepthook"):
        threading.excepthook = _thread_hook
