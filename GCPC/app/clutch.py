import numpy as np
from enum import Enum

THUMB_TIP = 4
INDEX_TIP = 8
WRIST = 0
INDEX_MCP = 5


class ClutchState(Enum):
    IDLE = 0
    ARMED = 1
    WINDOW = 2


class PinchClutch:
    def __init__(self, cfg):
        ccfg = cfg["clutch"]
        self.open_thr = ccfg.get("pinch_open", 0.30)
        self.close_thr = ccfg.get("pinch_close", 0.22)
        self.hold_ms = ccfg.get("hold_ms", 80)
        self.window_ms = ccfg.get("window_ms", 2500)
        self.state = ClutchState.IDLE
        self.state_ts = 0

    def _norm_dist(self, pts):
        wrist = pts[WRIST][:2]
        imcp = pts[INDEX_MCP][:2]
        hand_size = np.linalg.norm(imcp - wrist) + 1e-6
        d = np.linalg.norm(pts[THUMB_TIP][:2] - pts[INDEX_TIP][:2])
        return d / hand_size

    def update(self, pts, t_ms):
        d = self._norm_dist(pts)

        if self.state == ClutchState.IDLE:
            if d < self.close_thr:
                self.state = ClutchState.ARMED
                self.state_ts = t_ms
        elif self.state == ClutchState.ARMED:
            if d >= self.open_thr:
                self.state = ClutchState.IDLE
            elif t_ms - self.state_ts >= self.hold_ms:
                self.state = ClutchState.WINDOW
                self.state_ts = t_ms
        elif self.state == ClutchState.WINDOW:
            if t_ms - self.state_ts > self.window_ms or d >= self.open_thr:
                self.state = ClutchState.IDLE

        return self.state
