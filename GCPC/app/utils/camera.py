"""Camera and drawing helpers."""
from __future__ import annotations

import cv2


def open_camera(idx: int, w: int, h: int):
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
