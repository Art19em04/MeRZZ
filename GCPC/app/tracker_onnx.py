# -*- coding: utf-8 -*-
# Optional ONNX landmarks-only tracker (needs separate detector). Kept for CUDA experiments.
import os

import cv2
import numpy as np
import onnxruntime as ort


def _is_nchw(shape):
    """Heuristically decide if ONNX model uses NCHW layout."""
    if len(shape) != 4: return True
    if shape[1] == 3: return True
    if shape[-1] == 3: return False
    return True


def _input_hw(shape, is_nchw):
    """Extract expected input height/width from ONNX input shape."""
    if is_nchw:
        H = shape[2] if len(shape) > 2 and isinstance(shape[2], int) and shape[2] > 0 else 256
        W = shape[3] if len(shape) > 3 and isinstance(shape[3], int) and shape[3] > 0 else 256
    else:
        H = shape[1] if len(shape) > 1 and isinstance(shape[1], int) and shape[1] > 0 else 256
        W = shape[2] if len(shape) > 2 and isinstance(shape[2], int) and shape[2] > 0 else 256
    return int(H), int(W)


def _letterbox(rgb, out_h, out_w):
    """Resize with padding to preserve aspect ratio and return metadata."""
    ih, iw = rgb.shape[:2]
    s = min(out_w / iw, out_h / ih)
    nw, nh = int(round(iw * s)), int(round(ih * s))
    resized = cv2.resize(rgb, (nw, nh), interpolation=cv2.INTER_LINEAR)
    canvas = np.zeros((out_h, out_w, 3), dtype=np.uint8)
    padx = (out_w - nw) // 2
    pady = (out_h - nh) // 2
    canvas[pady:pady + nh, padx:padx + nw] = resized
    return canvas, {"in_w": iw, "in_h": ih, "out_w": out_w, "out_h": out_h, "scale": s, "pad_x": padx, "pad_y": pady}


def _map_to_original(xy_list, meta, assume_norm01=True):
    """Map normalized points from letterboxed space back to original image."""
    out = []
    iw, ih = meta["in_w"], meta["in_h"]
    ow, oh = meta["out_w"], meta["out_h"]
    px, py = meta["pad_x"], meta["pad_y"]
    s = meta["scale"]
    for (x, y) in xy_list:
        if assume_norm01:
            X = x * ow
            Y = y * oh
        else:
            X, Y = x, y
        ox = (X - px) / max(1e-8, s)
        oy = (Y - py) / max(1e-8, s)
        nx = float(np.clip(ox / max(1, iw), 0, 1))
        ny = float(np.clip(oy / max(1, ih), 0, 1))
        out.append((nx, ny))
    return out


def _argmax2d(hm):
    """Find maximum value in heatmap and return normalized coordinates."""
    H, W = hm.shape
    idx = int(np.argmax(hm))
    y, x = divmod(idx, W)
    v = float(hm[y, x])
    return (x / max(1, W - 1), y / max(1, H - 1), v)


class ONNXHandTracker:
    """ONNX-based hand landmark tracker expecting only landmark model."""
    def __init__(self, models_dir="models"):
        mp = os.environ.get("HAND_ONNX_PATH", "").strip() or os.path.join(models_dir, "hand_landmarks.onnx")
        if not os.path.isfile(mp):
            raise FileNotFoundError(
                f"Нет модели {mp}. Укажи HAND_ONNX_PATH или положи hand_landmarks.onnx в {models_dir}.")
        try:
            ort.preload_dlls()
        except Exception:
            pass
        self.sess = ort.InferenceSession(mp, providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
        self.providers = list(self.sess.get_providers())
        print("[ONNX] Providers:", self.sess.get_providers())
        i0 = self.sess.get_inputs()[0]
        self.input_name = i0.name
        self.is_nchw = _is_nchw(i0.shape)
        self.in_h, self.in_w = _input_hw(i0.shape, self.is_nchw)
        self.input_scale = 1 / 255.0
        self.input_mean = (0.0, 0.0, 0.0)
        self.input_std = (1.0, 1.0, 1.0)

    def _pre(self, rgb):
        """Preprocess RGB frame for ONNX inference and return meta data."""
        lb, meta = _letterbox(rgb, self.in_h, self.in_w)
        x = lb.astype(np.float32) * self.input_scale
        if any(abs(m) > 1e-6 for m in self.input_mean) or any(abs(s - 1) > 1e-6 for s in self.input_std):
            x = (x - np.array(self.input_mean, dtype=np.float32)) / np.array(self.input_std, dtype=np.float32)
        if self.is_nchw:
            x = np.transpose(x, (2, 0, 1))[None, ...]
        else:
            x = x[None, ...]
        return x, meta

    @staticmethod
    def _handed(lm):
        """Infer hand label from landmarks order (rough heuristic)."""
        try:
            return "Right" if lm[4][0] > lm[20][0] else "Left"
        except Exception:
            return "Unknown"

    def _parse(self, y, meta):
        """Convert raw ONNX outputs to normalized landmark structures."""
        y = y[0]
        lm = None
        score = 1.0
        if y.ndim == 2 and y.shape[0] == 1 and y.shape[1] in (63, 42):
            has_z = (y.shape[1] == 63)
            flat = y.reshape(-1)
            pts = []
            for i in range(21):
                xi = flat[i * (3 if has_z else 2) + 0]
                yi = flat[i * (3 if has_z else 2) + 1]
                pts.append((float(xi), float(yi)))
            assume_norm = bool(np.max(np.abs(np.array(pts))) <= 1.3)
            lm = _map_to_original(pts, meta, assume_norm01=assume_norm)
        elif y.ndim == 3 and y.shape[0] == 1 and y.shape[1] == 21 and y.shape[2] in (2, 3):
            pts = [(float(y[0, i, 0]), float(y[0, i, 1])) for i in range(21)]
            assume_norm = bool(np.max(np.abs(y[0, :, :2])) <= 1.3)
            lm = _map_to_original(pts, meta, assume_norm01=assume_norm)
            if y.shape[2] == 3:
                sc = float(np.mean(np.clip(y[0, :, 2], 0.0, 1.0)))
                import math
                if not math.isnan(sc): score = sc
        elif y.ndim == 4 and y.shape[0] == 1 and y.shape[1] == 21:
            pts = []
            vals = []
            for i in range(21):
                x01, y01, v = _argmax2d(y[0, i])
                pts.append((x01, y01))
                vals.append(v)
            lm = _map_to_original(pts, meta, assume_norm01=True)
            if vals: score = float(np.clip(np.mean(vals), 0.0, 1.0))
        if lm is None: return []
        return [{"lm": lm, "label": self._handed(lm), "score": float(score)}]

    def process(self, rgb):
        """Run ONNX landmark model on RGB frame and return detections."""
        if rgb is None or rgb.ndim != 3 or rgb.shape[2] != 3: return []
        x, meta = self._pre(rgb)
        out = self.sess.run(None, {self.input_name: x})
        return self._parse(out, meta)
