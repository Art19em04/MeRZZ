import os
from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np
import onnxruntime as ort


@dataclass
class HandResult:
    handedness: str
    landmarks: np.ndarray
    world_landmarks: Optional[np.ndarray] = None
    score: float = 0.0


class HandTracker:
    def __init__(self, cfg):
        self.cfg = cfg
        self.roi_norm = list(cfg["video"]["roi"])
        models = cfg["models"]
        self.lmk_path = models.get("hand_landmark")
        self.det_path = models.get("palm_detector")
        if not (self.lmk_path and os.path.exists(self.lmk_path)):
            raise FileNotFoundError("models.hand_landmark ONNX not found")

        opts = cfg.get("tracker_opts", {})
        self.det_every = int(opts.get("det_every", 5))
        self.min_det_conf = float(opts.get("min_det_conf", 0.5))
        self.pad_scale = float(opts.get("pad_scale", 1.3))
        self.lost_grace = int(opts.get("lost_grace_frames", 8))
        self.z_div = float(opts.get("z_div", 80.0))
        self.z_dir = float(opts.get("z_dir", -1.0))
        self.pres_high = float(opts.get("presence_high", 0.80))
        self.pres_low = float(opts.get("presence_low", 0.60))
        self.k_enter = int(opts.get("k_enter", 2))
        self.k_exit = int(opts.get("k_exit", 2))
        self.min_area = float(opts.get("min_box_area", 0.015))

        self._present = False
        self._cnt_hi = 0
        self._cnt_lo = 0

        providers = ort.get_available_providers()
        if "DmlExecutionProvider" not in providers:
            raise RuntimeError("DirectML execution provider is required")
        self.providers = ["DmlExecutionProvider"]
        print("[ORT] available providers:", providers, flush=True)

        session_opts = ort.SessionOptions()
        session_opts.log_severity_level = 3

        self.lmk_sess = ort.InferenceSession(self.lmk_path, sess_options=session_opts, providers=self.providers)
        print("[ORT] using (landmark):", self.lmk_sess.get_providers(), flush=True)
        lmk_input = self.lmk_sess.get_inputs()[0]
        self.lmk_in = lmk_input.name
        shape = lmk_input.shape
        if len(shape) == 4 and shape[1] in (1, 3):
            self.lmk_nchw, self.lmk_c, self.lmk_h, self.lmk_w = True, int(shape[1]), int(shape[2]), int(shape[3])
        elif len(shape) == 4 and shape[-1] in (1, 3):
            self.lmk_nchw, self.lmk_h, self.lmk_w, self.lmk_c = False, int(shape[1]), int(shape[2]), int(shape[3])
        else:
            self.lmk_nchw, self.lmk_c, self.lmk_h, self.lmk_w = True, 3, 224, 224

        dummy = (
            np.zeros((1, self.lmk_c, self.lmk_h, self.lmk_w), np.float32)
            if self.lmk_nchw
            else np.zeros((1, self.lmk_h, self.lmk_w, self.lmk_c), np.float32)
        )
        self.lmk_sess.run(None, {self.lmk_in: dummy})
        print("[ORT] warmup landmark done", flush=True)

        self.det_sess = None
        self.det_in = None
        self.det_nchw = True
        self.det_h = self.det_w = 192
        if self.det_path and os.path.exists(self.det_path):
            self.det_sess = ort.InferenceSession(self.det_path, sess_options=session_opts, providers=self.providers)
            print("[ORT] using (detector):", self.det_sess.get_providers(), flush=True)
            det_input = self.det_sess.get_inputs()[0]
            self.det_in = det_input.name
            d_shape = det_input.shape
            if len(d_shape) == 4 and d_shape[1] in (1, 3):
                self.det_nchw, self.det_c, self.det_h, self.det_w = True, int(d_shape[1]), int(d_shape[2]), int(d_shape[3])
            elif len(d_shape) == 4 and d_shape[-1] in (1, 3):
                self.det_nchw, self.det_h, self.det_w, self.det_c = False, int(d_shape[1]), int(d_shape[2]), int(d_shape[3])
            else:
                self.det_nchw, self.det_c, self.det_h, self.det_w = True, 3, 192, 192
            det_dummy = (
                np.zeros((1, self.det_c, self.det_h, self.det_w), np.float32)
                if self.det_nchw
                else np.zeros((1, self.det_h, self.det_w, self.det_c), np.float32)
            )
            self.det_sess.run(None, {self.det_in: det_dummy})
            print("[ORT] warmup detector done", flush=True)

        self.frame_idx = 0
        self.last_box_xyxy: Optional[Tuple[int, int, int, int]] = None
        self.last_conf = 0.0
        self.lost_frames = 0

    @staticmethod
    def _clip(v, lo, hi):
        return max(lo, min(hi, v))

    def _norm2abs_roi(self, h, w, roi):
        x, y, rw, rh = roi
        x0, y0 = int(x * w), int(y * h)
        x1, y1 = int((x + rw) * w), int((y + rh) * h)
        x0 = self._clip(x0, 0, w - 1)
        x1 = self._clip(x1, 1, w)
        y0 = self._clip(y0, 0, h - 1)
        y1 = self._clip(y1, 1, h)
        return x0, y0, x1, y1

    def _pad_box(self, x0, y0, x1, y1, h, w, scale):
        cx = (x0 + x1) / 2
        cy = (y0 + y1) / 2
        bw = (x1 - x0) * scale
        bh = (y1 - y0) * scale
        nx0 = int(self._clip(cx - bw / 2, 0, w - 1))
        ny0 = int(self._clip(cy - bh / 2, 0, h - 1))
        nx1 = int(self._clip(cx + bw / 2, 1, w))
        ny1 = int(self._clip(cy + bh / 2, 1, h))
        return nx0, ny0, nx1, ny1

    def _prep(self, bgr, h, w, nchw, dh, dw):
        img = cv2.resize(bgr, (dw, dh), interpolation=cv2.INTER_AREA)
        rgb = img[:, :, ::-1].astype(np.float32) / 255.0
        return np.transpose(rgb, (2, 0, 1))[None, ...] if nchw else rgb[None, ...]

    def _post_landmarks(self, out):
        arr = out[0]
        arr = arr[0] if hasattr(arr, "shape") and arr.shape[0] == 1 else arr
        pts = np.asarray(arr, dtype=np.float32).reshape(21, 3)
        pts[:, 0] = np.clip(pts[:, 0], 0, 1)
        pts[:, 1] = np.clip(pts[:, 1], 0, 1)
        if self.z_div > 0:
            pts[:, 2] = (pts[:, 2] / self.z_div) * self.z_dir
        return pts

    def _run_detector(self, frame_bgr):
        if self.det_sess is None:
            return None, 0.0
        h, w = frame_bgr.shape[:2]
        inp = self._prep(frame_bgr, h, w, self.det_nchw, self.det_h, self.det_w)
        outs = self.det_sess.run(None, {self.det_in: inp})
        candidates = [np.asarray(o) for o in outs if o is not None]
        best = None
        best_conf = 0.0
        for arr in candidates:
            flat = arr.reshape(-1, arr.shape[-1])
            for row in flat:
                if row.shape[-1] >= 5:
                    xc, yc, bw, bh, conf = map(float, row[:5])
                    if conf > best_conf and 0 <= xc <= 1 and 0 <= yc <= 1 and bw > 0 and bh > 0:
                        x0 = int((xc - bw / 2) * w)
                        y0 = int((yc - bh / 2) * h)
                        x1 = int((xc + bw / 2) * w)
                        y1 = int((yc + bh / 2) * h)
                        x0 = self._clip(x0, 0, w - 1)
                        y0 = self._clip(y0, 0, h - 1)
                        x1 = self._clip(x1, 1, w)
                        y1 = self._clip(y1, 1, h)
                        best = (x0, y0, x1, y1)
                        best_conf = conf
        return best, best_conf

    def process(self, frame_bgr, timestamp_ms: int) -> Optional[HandResult]:
        h, w = frame_bgr.shape[:2]
        self.frame_idx += 1

        need_det = (self.last_box_xyxy is None) or (self.frame_idx % self.det_every == 0) or (self.lost_frames > 0)
        if need_det:
            box, conf = self._run_detector(frame_bgr)
            if box is not None:
                area = ((box[2] - box[0]) * (box[3] - box[1])) / float(h * w)
                if area < self.min_area:
                    conf = 0.0
                else:
                    self.last_box_xyxy = self._pad_box(*box, h, w, self.pad_scale)
                    self.last_conf = conf
                    self.lost_frames = 0
            else:
                self.lost_frames += 1
                self.last_conf = 0.0

        conf = float(self.last_conf)
        if conf >= self.pres_high:
            self._cnt_hi += 1
            self._cnt_lo = max(0, self._cnt_lo - 1)
        elif conf <= self.pres_low:
            self._cnt_lo += 1
            self._cnt_hi = max(0, self._cnt_hi - 1)

        if not self._present and self._cnt_hi >= self.k_enter:
            self._present = True
            self._cnt_lo = 0
        elif self._present and self._cnt_lo >= self.k_exit:
            self._present = False
            self._cnt_hi = 0

        if self.det_sess is not None:
            if (not self._present and self.lost_frames > self.lost_grace) or (self.last_box_xyxy is None):
                return None
            x0, y0, x1, y1 = self.last_box_xyxy
            crop = frame_bgr[y0:y1, x0:x1]
            det_conf = self.last_conf
        else:
            x0, y0, x1, y1 = self._norm2abs_roi(h, w, self.roi_norm)
            crop = frame_bgr[y0:y1, x0:x1]
            det_conf = 1e-3

        inp = self._prep(crop, crop.shape[0], crop.shape[1], self.lmk_nchw, self.lmk_h, self.lmk_w)
        out = self.lmk_sess.run(None, {self.lmk_in: inp})
        pts = self._post_landmarks(out)
        return HandResult(handedness="Right", landmarks=pts, score=det_conf)
