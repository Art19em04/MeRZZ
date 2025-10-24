# app/hands_cuda_onnx.py
import os, math, numpy as np, cv2
from dataclasses import dataclass
from typing import Optional, Tuple

try:
    import onnxruntime as ort
except Exception as e:
    raise RuntimeError("pip install onnxruntime-directml (или onnxruntime-gpu)") from e

@dataclass
class HandResult:
    handedness: str
    landmarks: np.ndarray    # (21,3) в ROI [0..1], z нормализована
    world_landmarks: Optional[np.ndarray] = None
    score: float = 0.0       # уверенность детектора (presence proxy)

class CudaHandTracker:
    def __init__(self, cfg):
        self.cfg = cfg
        self.roi_norm = list(cfg["video"]["roi"])  # [x,y,w,h] 0..1 — стартовый ROI
        m = cfg["models"]
        self.lmk_path = m.get("hand_landmark")
        self.det_path = m.get("palm_detector")     # рекомендовано; без него возможно FP «без руки»
        if not (self.lmk_path and os.path.exists(self.lmk_path)):
            raise FileNotFoundError("models.hand_landmark ONNX not found")

        topts = cfg.get("tracker_opts", {})
        self.det_every   = int(topts.get("det_every", 5))
        self.min_det_conf= float(topts.get("min_det_conf", 0.5))
        self.pad_scale   = float(topts.get("pad_scale", 1.3))
        self.lost_grace  = int(topts.get("lost_grace_frames", 8))
        self.z_div       = float(topts.get("z_div", 80.0))
        self.z_dir       = float(topts.get("z_dir", -1.0))

        # --- presence gate (гистерезис + серия кадров + min area) ---
        self.pres_high   = float(topts.get("presence_high", 0.80))  # вход
        self.pres_low    = float(topts.get("presence_low",  0.60))  # выход
        self.k_enter     = int(topts.get("k_enter", 2))             # N кадров подряд для входа
        self.k_exit      = int(topts.get("k_exit",  2))             # N кадров подряд для выхода
        self.min_area    = float(topts.get("min_box_area", 0.015))  # доля площади кадра (напр. 1.5%)

        self._present = False
        self._cnt_hi  = 0
        self._cnt_lo  = 0

        # EP: TensorRT -> CUDA -> DirectML -> CPU
        avail = ort.get_available_providers()
        prov = []
        if "TensorrtExecutionProvider" in avail: prov.append("TensorrtExecutionProvider")
        if "CUDAExecutionProvider"     in avail: prov.append("CUDAExecutionProvider")
        if "DmlExecutionProvider"      in avail: prov.append("DmlExecutionProvider")
        prov.append("CPUExecutionProvider")
        print("[ORT] available providers:", avail, flush=True)

        so = ort.SessionOptions(); so.log_severity_level = 3

        # landmark session
        self.lmk_sess = ort.InferenceSession(self.lmk_path, sess_options=so, providers=prov)
        print("[ORT] using (landmark):", self.lmk_sess.get_providers(), flush=True)
        l_in = self.lmk_sess.get_inputs()[0]; self.lmk_in = l_in.name
        l_shp = l_in.shape
        if len(l_shp)==4 and l_shp[1] in (1,3):
            self.lmk_nchw, self.lmk_c, self.lmk_h, self.lmk_w = True, int(l_shp[1]), int(l_shp[2]), int(l_shp[3])
        elif len(l_shp)==4 and l_shp[-1] in (1,3):
            self.lmk_nchw, self.lmk_h, self.lmk_w, self.lmk_c = False, int(l_shp[1]), int(l_shp[2]), int(l_shp[3])
        else:
            self.lmk_nchw, self.lmk_c, self.lmk_h, self.lmk_w = True, 3, 224, 224

        # warmup landmark
        dummy = (np.zeros((1, self.lmk_c, self.lmk_h, self.lmk_w), np.float32)
                 if self.lmk_nchw else
                 np.zeros((1, self.lmk_h, self.lmk_w, self.lmk_c), np.float32))
        _ = self.lmk_sess.run(None, {self.lmk_in: dummy})
        print("[ORT] warmup landmark done", flush=True)

        # detector session (опционально, но настоятельно рекомендуется)
        self.det_sess = None; self.det_in = None; self.det_nchw = True; self.det_h = self.det_w = 192
        if self.det_path and os.path.exists(self.det_path):
            self.det_sess = ort.InferenceSession(self.det_path, sess_options=so, providers=prov)
            print("[ORT] using (detector):", self.det_sess.get_providers(), flush=True)
            d_in = self.det_sess.get_inputs()[0]; self.det_in = d_in.name
            d_shp = d_in.shape
            if len(d_shp)==4 and d_shp[1] in (1,3):
                self.det_nchw, self.det_c, self.det_h, self.det_w = True, int(d_shp[1]), int(d_shp[2]), int(d_shp[3])
            elif len(d_shp)==4 and d_shp[-1] in (1,3):
                self.det_nchw, self.det_h, self.det_w, self.det_c = False, int(d_shp[1]), int(d_shp[2]), int(d_shp[3])
            else:
                self.det_nchw, self.det_c, self.det_h, self.det_w = True, 3, 192, 192
            d_dummy = (np.zeros((1, self.det_c, self.det_h, self.det_w), np.float32)
                       if self.det_nchw else
                       np.zeros((1, self.det_h, self.det_w, self.det_c), np.float32))
            _ = self.det_sess.run(None, {self.det_in: d_dummy})
            print("[ORT] warmup detector done", flush=True)

        # state
        self.frame_idx = 0
        self.last_box_xyxy: Optional[Tuple[int,int,int,int]] = None
        self.last_conf = 0.0
        self.lost_frames = 0

    # ---------- utils ----------
    @staticmethod
    def _clip(a, lo, hi): return max(lo, min(hi, a))

    def _norm2abs_roi(self, H, W, roi):
        x,y,w,h = roi
        x0,y0 = int(x*W), int(y*H)
        x1,y1 = int((x+w)*W), int((y+h)*H)
        x0 = self._clip(x0,0,W-1); x1 = self._clip(x1,1,W)
        y0 = self._clip(y0,0,H-1); y1 = self._clip(y1,1,H)
        return x0,y0,x1,y1

    def _pad_box(self, x0,y0,x1,y1, H,W, scale):
        cx = (x0+x1)/2; cy = (y0+y1)/2
        bw = (x1-x0)*scale; bh = (y1-y0)*scale
        nx0 = int(self._clip(cx-bw/2, 0, W-1)); ny0 = int(self._clip(cy-bh/2, 0, H-1))
        nx1 = int(self._clip(cx+bw/2, 1, W));   ny1 = int(self._clip(cy+bh/2, 1, H))
        return nx0,ny0,nx1,ny1

    def _prep(self, bgr, H, W, nchw, h, w):
        img = cv2.resize(bgr, (w,h), interpolation=cv2.INTER_AREA)
        rgb = img[:,:,::-1].astype(np.float32)/255.0
        return (np.transpose(rgb,(2,0,1))[None,...] if nchw else rgb[None,...])

    def _post_landmarks(self, out):
        arr = out[0];  arr = arr[0] if (hasattr(arr,'shape') and arr.shape[0]==1) else arr
        pts = np.asarray(arr, dtype=np.float32).reshape(21,3)
        pts[:,0] = np.clip(pts[:,0], 0, 1)
        pts[:,1] = np.clip(pts[:,1], 0, 1)
        if self.z_div>0: pts[:,2] = (pts[:,2]/self.z_div)*self.z_dir
        return pts

    def _run_detector(self, frame_bgr) -> Tuple[Optional[Tuple[int,int,int,int]], float]:
        if self.det_sess is None:
            return None, 0.0
        H,W = frame_bgr.shape[:2]
        inp = self._prep(frame_bgr, H, W, self.det_nchw, self.det_h, self.det_w)
        outs = self.det_sess.run(None, {self.det_in: inp})
        # Популярный формат *_post.onnx: [N,6] → xc,yc,w,h,score,class в нормал.коорд.
        flat = [np.asarray(o) for o in outs if o is not None]
        best = None; best_conf = 0.0
        for a in flat:
            a = a.reshape(-1, a.shape[-1])
            for row in a:
                if row.shape[-1] >= 5:
                    xc, yc, bw, bh, conf = float(row[0]), float(row[1]), float(row[2]), float(row[3]), float(row[4])
                    if conf > best_conf and 0<=xc<=1 and 0<=yc<=1 and bw>0 and bh>0:
                        x0 = int((xc - bw/2)*W); y0 = int((yc - bh/2)*H)
                        x1 = int((xc + bw/2)*W); y1 = int((yc + bh/2)*H)
                        x0 = self._clip(x0,0,W-1); y0 = self._clip(y0,0,H-1)
                        x1 = self._clip(x1,1,W);   y1 = self._clip(y1,1,H)
                        best = (x0,y0,x1,y1); best_conf = conf
        return best, best_conf

    # ---------- public ----------
    def process(self, frame_bgr, timestamp_ms: int) -> Optional[HandResult]:
        H,W = frame_bgr.shape[:2]
        self.frame_idx += 1

        # каждые det_every кадров — обновляем бокс
        need_det = (self.last_box_xyxy is None) or (self.frame_idx % self.det_every == 0) or (self.lost_frames > 0)
        if need_det:
            box, conf = self._run_detector(frame_bgr)
            if box is not None:
                # проверка минимальной площади бокса
                area = ( (box[2]-box[0]) * (box[3]-box[1]) ) / float(H*W)
                if area < self.min_area:
                    conf = 0.0
                else:
                    self.last_box_xyxy = self._pad_box(*box, H,W, self.pad_scale)
                    self.last_conf = conf
                    self.lost_frames = 0
            else:
                self.lost_frames += 1
                self.last_conf = 0.0

        # --- гистерезис presence по self.last_conf ---
        conf = float(self.last_conf)
        if conf >= self.pres_high:
            self._cnt_hi += 1; self._cnt_lo = max(0, self._cnt_lo - 1)
        elif conf <= self.pres_low:
            self._cnt_lo += 1; self._cnt_hi = max(0, self._cnt_hi - 1)

        if not self._present and self._cnt_hi >= self.k_enter:
            self._present = True;  self._cnt_lo = 0
        elif self._present and self._cnt_lo >= self.k_exit:
            self._present = False; self._cnt_hi = 0

        # --- без устойчивого presence / без бокса — руки «нет» ---
        if self.det_sess is not None:
            if (not self._present and self.lost_frames > self.lost_grace) or (self.last_box_xyxy is None):
                return None
            x0,y0,x1,y1 = self.last_box_xyxy
            crop = frame_bgr[y0:y1, x0:x1]
            det_conf = self.last_conf
        else:
            # без детектора — берём ROI из конфига (менее надёжно)
            x0,y0,x1,y1 = self._norm2abs_roi(H,W,self.roi_norm)
            crop = frame_bgr[y0:y1, x0:x1]
            det_conf = 1e-3  # почти ноль, для логов

        # landmarks в кропе
        inp = self._prep(crop, crop.shape[0], crop.shape[1], self.lmk_nchw, self.lmk_h, self.lmk_w)
        out = self.lmk_sess.run(None, {self.lmk_in: inp})
        pts = self._post_landmarks(out)
        return HandResult(handedness="Right", landmarks=pts, score=det_conf)
