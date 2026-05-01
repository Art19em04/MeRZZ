"""Camera and drawing helpers."""

from typing import Iterable

import cv2


_BACKENDS = tuple(
    dict.fromkeys(
        api
        for api in (
            getattr(cv2, "CAP_MSMF", None),
            getattr(cv2, "CAP_DSHOW", None),
            getattr(cv2, "CAP_ANY", None),
        )
        if api is not None
    )
)
_PROBE_RANGE = range(0, 6)


def _unique_indices(indices: Iterable[int]) -> tuple[int, ...]:
    result = []
    for index in indices:
        if index not in result:
            result.append(index)
    return tuple(result)


def _candidate_indices(idx: int, preferred_idx: int | None = None) -> tuple[int, ...]:
    candidates = []
    if preferred_idx is not None:
        candidates.append(preferred_idx)
    if idx == -1:
        candidates.extend((-1, *_PROBE_RANGE))
    else:
        candidates.append(idx)
    return _unique_indices(candidates)


def _open_with_backend(index: int, api: int, width: int, height: int):
    """Open camera index with a specific backend and verify first frame read."""
    cap = cv2.VideoCapture(index, api)
    if not cap or not cap.isOpened():
        if cap:
            cap.release()
        return None
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    ok, _ = cap.read()
    if not ok:
        cap.release()
        return None
    return cap


def open_camera(idx: int, w: int, h: int, preferred_idx: int | None = None):
    """Try to open camera by preferred index with fallbacks across APIs."""
    for cam_idx in _candidate_indices(idx, preferred_idx):
        for api in _BACKENDS:
            cap = _open_with_backend(cam_idx, api, w, h)
            if cap is not None:
                print(f"[videoio] open idx={cam_idx} api={api}")
                return cap, cam_idx
    return None, None


def draw_landmarks(frame, lm):
    """Draw simple circles for each landmark on a frame in-place."""
    h, w = frame.shape[:2]
    for (x, y) in lm:
        cv2.circle(frame, (int(x * w), int(y * h)), 4, (0, 255, 0), -1)
