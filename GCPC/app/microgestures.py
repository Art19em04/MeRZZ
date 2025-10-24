import numpy as np
from collections import deque
from dataclasses import dataclass
from enum import Enum

THUMB_TIP = 4
INDEX_MCP = 5
MIDDLE_MCP = 9
WRIST = 0

class G(Enum):
    NONE=0
    SWIPE_UP=1
    SWIPE_DOWN=2
    TAP_FORWARD=3
    TAP_BACK=4

@dataclass
class GestureConfig:
    conf_min: float
    min_duration_ms: int
    max_duration_ms: int
    cooldown_ms: int

@dataclass
class MGDecision:
    g: G
    confidence: float
    duration_ms: int

class MicroGestureDetector:
    def __init__(self, cfg):
        gcfg = cfg["gesture"]
        self.cfg = GestureConfig(
            conf_min=gcfg["conf_min"],
            min_duration_ms=gcfg["min_duration_ms"],
            max_duration_ms=gcfg["max_duration_ms"],
            cooldown_ms=gcfg["cooldown_ms"]
        )
        self.buf = deque(maxlen=24)  # ~ at 120fps equals 200ms window; scale with actual fps
        # thresholds from config (with sane defaults)
        gparams = cfg.get("gesture", {})
        self.thr_swipe = float(gparams.get("thr_swipe", 0.28))
        self.thr_tapz = float(gparams.get("thr_tapz", 0.06))

        self.last_ts = 0
        self.last_fire = 0

    def update(self, pts, t_ms, clutch_window: bool) -> MGDecision:
        # Нет руки или окно клатча закрыто — ничего не распознаём
        if (pts is None) or (not clutch_window):
            self.buf.clear()
            return MGDecision(G.NONE, 0.0, 0)

        # Копим только валидные точки внутри окна
        self.buf.append((t_ms, pts.copy()))

        # cooldown
        if t_ms - self.last_fire < self.cfg.cooldown_ms:
            return MGDecision(G.NONE, 0.0, 0)

        # Нужна хоть какая-то длительность
        if len(self.buf) < 3:
            return MGDecision(G.NONE, 0.0, 0)

        # Окно времени по первым/последним сэмплам
        t0, p0 = self.buf[0]
        t1, p1 = self.buf[-1]
        dur = t1 - t0
        if dur < self.cfg.min_duration_ms or dur > self.cfg.max_duration_ms:
            return MGDecision(G.NONE, 0.0, dur)

        # ----- дальше твоя логика осей/порогов без изменений -----
        wrist = p0[0][:2]  # WRIST=0
        mid = p0[9][:2]  # MIDDLE_MCP=9
        y_axis = mid - wrist
        y_axis = y_axis / (np.linalg.norm(y_axis) + 1e-6)
        x_axis = np.array([y_axis[1], -y_axis[0]])

        THUMB_TIP = 4;
        INDEX_MCP = 5
        thumb0 = p0[THUMB_TIP];
        thumb1 = p1[THUMB_TIP]
        delta = thumb1 - thumb0
        hs = np.linalg.norm(p0[INDEX_MCP][:2] - wrist) + 1e-6
        d2 = delta[:2] / hs
        dz = (thumb1[2] - thumb0[2])

        up_comp = float(np.dot(d2, y_axis))
        right_comp = float(np.dot(d2, x_axis))

        score_up = max(0.0, up_comp)
        score_down = max(0.0, -up_comp)
        score_fwd = max(0.0, -dz)  # к камере
        score_back = max(0.0, dz)  # от камеры

        THR_SWIPE = self.thr_swipe
        THR_TAPZ = self.thr_tapz

        cand = G.NONE;
        conf = 0.0
        if score_up > THR_SWIPE and score_up > score_down and score_up > score_fwd and score_up > score_back:
            cand, conf = G.SWIPE_UP, max(0.0, min(1.0, (score_up - THR_SWIPE) / max(1e-6, THR_SWIPE*1.6)))
        elif score_down > THR_SWIPE and score_down > score_up and score_down > score_fwd and score_down > score_back:
            cand, conf = G.SWIPE_DOWN, max(0.0, min(1.0, (score_down - THR_SWIPE) / max(1e-6, THR_SWIPE*1.6)))
        elif score_fwd > THR_TAPZ and score_fwd > score_back and score_fwd > score_up and score_fwd > score_down:
            cand, conf = G.TAP_FORWARD, max(0.0, min(1.0, (score_fwd - THR_TAPZ) / max(1e-6, THR_TAPZ*2.0)))
        elif score_back > THR_TAPZ and score_back > score_fwd and score_back > score_up and score_back > score_down:
            cand, conf = G.TAP_BACK, max(0.0, min(1.0, (score_back - THR_TAPZ) / max(1e-6, THR_TAPZ*2.0)))

        if cand != G.NONE and conf >= self.cfg.conf_min:
            self.last_fire = t_ms
            return MGDecision(cand, conf, dur)

        return MGDecision(G.NONE, conf, dur)