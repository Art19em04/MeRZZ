import cv2
import numpy as np
import onnxruntime as ort

from .one_euro import OneEuro


def _letterbox(img, newh, neww):
    ih, iw = img.shape[:2]
    s = min(neww / iw, newh / ih)
    nw, nh = int(round(iw * s)), int(round(ih * s))
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    canvas = np.zeros((newh, neww, 3), dtype=np.uint8)
    pad_x = (neww - nw) // 2
    pad_y = (newh - nh) // 2
    canvas[pad_y:pad_y + nh, pad_x:pad_x + nw] = resized
    meta = {"in_w": iw, "in_h": ih, "out_w": neww, "out_h": newh, "scale": s, "pad_x": pad_x, "pad_y": pad_y}
    return canvas, meta


def _unletterbox_xyxy(xyxy, meta):
    # map [x1,y1,x2,y2] in letterboxed space back to original image
    iw, ih = meta["in_w"], meta["in_h"]
    s = meta["scale"]
    px, py = meta["pad_x"], meta["pad_y"]
    x1, y1, x2, y2 = xyxy
    x1 = (x1 - px) / max(1e-8, s)
    x2 = (x2 - px) / max(1e-8, s)
    y1 = (y1 - py) / max(1e-8, s)
    y2 = (y2 - py) / max(1e-8, s)
    return [float(np.clip(x1, 0, iw)), float(np.clip(y1, 0, ih)), float(np.clip(x2, 0, iw)), float(np.clip(y2, 0, ih))]


def _nms(boxes, scores, iou_th=0.45, topk=100):
    if len(boxes) == 0: return []
    boxes = np.array(boxes, dtype=np.float32)
    scores = np.array(scores, dtype=np.float32)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0 and len(keep) < topk:
        i = order[0];
        keep.append(i)
        if order.size == 1: break
        xx1 = np.maximum(boxes[i, 0], boxes[order[1:], 0])
        yy1 = np.maximum(boxes[i, 1], boxes[order[1:], 1])
        xx2 = np.minimum(boxes[i, 2], boxes[order[1:], 2])
        yy2 = np.minimum(boxes[i, 3], boxes[order[1:], 3])
        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h
        iou = inter / ((boxes[i, 2] - boxes[i, 0]) * (boxes[i, 3] - boxes[i, 1]) + (
                boxes[order[1:], 2] - boxes[order[1:], 0]) * (
                               boxes[order[1:], 3] - boxes[order[1:], 1]) - inter + 1e-6)
        inds = np.where(iou <= iou_th)[0]
        order = order[inds + 1]
    return keep


class DetectorONNX:
    def __init__(self, path, input_size):
        self.path = path
        try:
            ort.preload_dlls()
        except Exception:
            pass
        self.sess = ort.InferenceSession(path, providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
        self.inp = self.sess.get_inputs()[0]
        self.in_h, self.in_w = int(input_size[0]), int(input_size[1])
        self.out_names = [o.name for o in self.sess.get_outputs()]
        print("[DET] Providers:", self.sess.get_providers())

    def infer(self, rgb):
        lb, meta = _letterbox(rgb, self.in_h, self.in_w)
        x = lb.astype(np.float32) / 255.0
        x = np.transpose(x, (2, 0, 1))[None, ...]
        out = self.sess.run(self.out_names, {self.inp.name: x})
        y = out[0]
        boxes, scores = [], []
        if y.ndim == 3 and y.shape[-1] >= 6:
            # assume (1, N, 6+) -> [x1,y1,x2,y2, score, class ...]
            for row in y[0]:
                x1, y1, x2, y2, score, cls = float(row[0]), float(row[1]), float(row[2]), float(row[3]), float(
                    row[4]), float(row[5])
                # accept all classes as "hand"
                if score > 0: boxes.append([x1, y1, x2, y2]); scores.append(score)
        elif y.ndim == 2 and y.shape[-1] >= 6:
            for row in y:
                x1, y1, x2, y2, score, cls = float(row[0]), float(row[1]), float(row[2]), float(row[3]), float(
                    row[4]), float(row[5])
                if score > 0: boxes.append([x1, y1, x2, y2]); scores.append(score)
        else:
            # try alternate format: (1, 4, N) and (1, 1, N) etc. — not implemented fully
            return [], meta
        # NMS in letterboxed space
        if boxes:
            keep = _nms(np.array(boxes), np.array(scores), iou_th=0.45, topk=50)
            boxes = [boxes[i] for i in keep]
            scores = [scores[i] for i in keep]
        # map back to original
        xyxy = [_unletterbox_xyxy(b, meta) for b in boxes]
        return list(zip(xyxy, scores)), meta


class LandmarkONNX:
    def __init__(self, path, input_size, smooth=True, one_euro_cfg=None):
        try:
            ort.preload_dlls()
        except Exception:
            pass
        self.sess = ort.InferenceSession(path, providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
        self.inp = self.sess.get_inputs()[0]
        self.in_h, self.in_w = int(input_size[0]), int(input_size[1])
        self.out_names = [o.name for o in self.sess.get_outputs()]
        self.smooth = smooth
        self.filters = [OneEuro(**(one_euro_cfg or {"min_cutoff": 1.2, "beta": 0.025})) for _ in range(21 * 2)]
        print("[LMK] Providers:", self.sess.get_providers())

    def _pre(self, crop):
        x = cv2.resize(crop, (self.in_w, self.in_h), interpolation=cv2.INTER_LINEAR).astype(np.float32) / 255.0
        x = np.transpose(x, (2, 0, 1))[None, ...]
        return x

    @staticmethod
    def _argmax2d(hm):
        H, W = hm.shape
        idx = int(np.argmax(hm))
        y, x = divmod(idx, W)
        v = float(hm[y, x])
        return x / (W - 1 if W > 1 else 1), y / (H - 1 if H > 1 else 1), v

    def _parse(self, y):
        y = y[0]
        pts = None;
        conf = 1.0
        if y.ndim == 3 and y.shape[0] == 1 and y.shape[1] == 21 and y.shape[2] in (2, 3):
            pts = [(float(y[0, i, 0]), float(y[0, i, 1])) for i in range(21)]
            if y.shape[2] == 3: conf = float(np.mean(np.clip(y[0, :, 2], 0, 1)))
        elif y.ndim == 2 and y.shape[0] == 1 and y.shape[1] in (42, 63):
            has_z = (y.shape[1] == 63)
            flat = y.reshape(-1)
            pts = []
            for i in range(21):
                xi = flat[i * (3 if has_z else 2) + 0]
                yi = flat[i * (3 if has_z else 2) + 1]
                pts.append((float(xi), float(yi)))
        elif y.ndim == 4 and y.shape[0] == 1 and y.shape[1] == 21:
            pts = [];
            vs = []
            for i in range(21):
                x01, y01, v = self._argmax2d(y[0, i])
                pts.append((x01, y01))
                vs.append(v)
            conf = float(np.mean(vs))
        return pts, conf

    def infer(self, rgb_crop, bbox_xyxy):
        x = self._pre(rgb_crop)
        out = self.sess.run(self.out_names, {self.inp.name: x})
        pts01, conf = self._parse(out)
        if pts01 is None: return None, 0.0
        # map crop norm -> image coords
        x1, y1, x2, y2 = bbox_xyxy
        bw, bh = max(1.0, x2 - x1), max(1.0, y2 - y1)
        lm = []
        for i, (px, py) in enumerate(pts01):
            X = x1 + px * bw
            Y = y1 + py * bh
            if self.smooth:
                X = self.filters[i * 2 + 0].apply(X)
                Y = self.filters[i * 2 + 1].apply(Y)
            lm.append((float(X), float(Y)))
        # normalize to [0..1] later in main
        return lm, float(conf)


class TwoStageHandTracker:
    def __init__(self, det_path, det_input_size, lmk_path, lmk_input_size, max_hands=2, score_th=0.3, nms_th=0.45,
                 presence_th=0.35, smooth=True, one_euro=None):
        self.det = DetectorONNX(det_path, det_input_size)
        self.lmk = LandmarkONNX(lmk_path, lmk_input_size, smooth=smooth, one_euro_cfg=one_euro)
        self.max_hands = max_hands
        self.score_th = score_th
        self.nms_th = nms_th
        self.presence_th = presence_th

    def process(self, rgb):
        (detections, meta) = self.det.infer(rgb)
        H, W = rgb.shape[:2]
        hands_out = []
        dets = [(xy, sc) for (xy, sc) in detections if sc >= self.score_th]
        dets = sorted(dets, key=lambda t: t[1], reverse=True)[:self.max_hands]
        for (x1, y1, x2, y2), sc in dets:
            x1i, y1i, x2i, y2i = map(lambda v: int(round(v)), (x1, y1, x2, y2))
            x1i, y1i = max(0, x1i), max(0, y1i);
            x2i, y2i = min(W - 1, x2i), min(H - 1, y2i)
            if x2i <= x1i or y2i <= y1i: continue
            crop = rgb[y1i:y2i, x1i:x2i].copy()
            lm, conf = self.lmk.infer(crop, (x1i, y1i, x2i, y2i))
            if lm is None or conf < self.presence_th:
                continue
            lm01 = [(lx / W, ly / H) for (lx, ly) in lm]
            label = "Right" if np.mean([p[0] for p in lm01]) > 0.5 else "Left"
            hands_out.append({"lm": lm01, "label": label, "score": float(conf)})
        return hands_out
