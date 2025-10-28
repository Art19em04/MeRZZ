# -*- coding: utf-8 -*-
# Two-stage CUDA tracker: detector (bboxes) + landmark head (21 pts)
from numbers import Integral

import cv2
import math
import numpy as np

from .onnx_utils import create_onnx_session
from .one_euro import OneEuro


def _resolve_input_hw(requested_size, node_arg, log_prefix):
    """Определяет размер входа модели, учитывая фактические статические размеры."""

    fallback = None
    if requested_size:
        fallback = (int(requested_size[0]), int(requested_size[1]))

    shape = getattr(node_arg, "shape", None)
    guessed = None
    if shape:
        dims = []
        for dim in shape:
            if isinstance(dim, Integral):
                val = int(dim)
                if val > 4:
                    dims.append(val)
        if len(dims) >= 2:
            guessed = (dims[-2], dims[-1])

    if guessed:
        if fallback and fallback != guessed:
            print(f"{log_prefix} Игнорируем config размер {fallback} — модель ожидает {guessed}")
        return guessed

    if fallback:
        return fallback

    raise RuntimeError(
        f"{log_prefix} Не удалось определить входной размер для {getattr(node_arg, 'name', '<input>')} (shape={shape})"
    )

def _letterbox(img, newh, neww):
    ih, iw = img.shape[:2]
    s = min(neww/iw, newh/ih)
    nw, nh = int(round(iw*s)), int(round(ih*s))
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    canvas = np.zeros((newh, neww, 3), dtype=np.uint8)
    pad_x = (neww-nw)//2; pad_y=(newh-nh)//2
    canvas[pad_y:pad_y+nh, pad_x:pad_x+nw] = resized
    meta = {"in_w": iw, "in_h": ih, "out_w": neww, "out_h": newh, "scale": s, "pad_x": pad_x, "pad_y": pad_y}
    return canvas, meta

def _unletterbox_xyxy(xyxy, meta):
    # map [x1,y1,x2,y2] in letterboxed space back to original image
    iw, ih = meta["in_w"], meta["in_h"]; s=meta["scale"]; px,py=meta["pad_x"],meta["pad_y"]
    x1,y1,x2,y2 = xyxy
    x1 = (x1 - px) / max(1e-8, s); x2 = (x2 - px) / max(1e-8, s)
    y1 = (y1 - py) / max(1e-8, s); y2 = (y2 - py) / max(1e-8, s)
    return [float(np.clip(x1,0,iw)), float(np.clip(y1,0,ih)), float(np.clip(x2,0,iw)), float(np.clip(y2,0,ih))]


def _unletterbox_point(pt, meta):
    x, y = pt
    iw, ih = meta["in_w"], meta["in_h"]
    s = meta["scale"]; px, py = meta["pad_x"], meta["pad_y"]
    ox = (x - px) / max(1e-8, s)
    oy = (y - py) / max(1e-8, s)
    return float(np.clip(ox, 0.0, iw)), float(np.clip(oy, 0.0, ih))

def _nms(boxes, scores, iou_th=0.45, topk=100):
    if len(boxes)==0: return []
    boxes = np.array(boxes, dtype=np.float32)
    scores = np.array(scores, dtype=np.float32)
    order = scores.argsort()[::-1]
    keep=[]
    while order.size>0 and len(keep)<topk:
        i = order[0]; keep.append(i)
        if order.size==1: break
        xx1 = np.maximum(boxes[i,0], boxes[order[1:],0])
        yy1 = np.maximum(boxes[i,1], boxes[order[1:],1])
        xx2 = np.minimum(boxes[i,2], boxes[order[1:],2])
        yy2 = np.minimum(boxes[i,3], boxes[order[1:],3])
        w = np.maximum(0.0, xx2-xx1); h = np.maximum(0.0, yy2-yy1)
        inter = w*h
        iou = inter / ((boxes[i,2]-boxes[i,0])*(boxes[i,3]-boxes[i,1]) + (boxes[order[1:],2]-boxes[order[1:],0])*(boxes[order[1:],3]-boxes[order[1:],1]) - inter + 1e-6)
        inds = np.where(iou <= iou_th)[0]
        order = order[inds+1]
    return keep

def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def _softmax(x):
    x = x - np.max(x, axis=-1, keepdims=True)
    ex = np.exp(x)
    return ex / np.sum(ex, axis=-1, keepdims=True)


def _node_static_shape(node_arg):
    dims = []
    shape = getattr(node_arg, "shape", None)
    if not shape:
        return tuple(dims)
    for dim in shape:
        if isinstance(dim, Integral):
            dims.append(int(dim))
        else:
            dim_value = getattr(dim, "dim_value", None)
            if isinstance(dim_value, Integral) and int(dim_value) > 0:
                dims.append(int(dim_value))
            else:
                dims.append(None)
    return tuple(dims)


class _PalmDetectorDecoder:
    """Decodes BlazePalm-style detector outputs into xyxy boxes."""

    def __init__(self, input_hw, box_index, score_index):
        self.in_h, self.in_w = input_hw
        self.box_index = box_index
        self.score_index = score_index
        self.anchors = self._generate_anchors()
        self.x_scale = float(self.in_w)
        self.y_scale = float(self.in_h)
        self.w_scale = float(self.in_w)
        self.h_scale = float(self.in_h)
        self.score_clip = 100.0
        self._warned_anchor_mismatch = False
        self._max_keypoints = 7

    def _generate_anchors(self):
        # MediaPipe palm detector anchor spec
        strides = [8, 16, 32, 32]
        anchor_offset_x = 0.5
        anchor_offset_y = 0.5
        aspect_ratios = [1.0]
        interpolated_scale_aspect_ratio = 1.0

        min_scale = 0.1171875
        max_scale = 0.75

        num_layers = len(strides)
        scales = []
        for layer_id in range(num_layers):
            if num_layers == 1:
                scales.append(min_scale)
            else:
                scale = min_scale + (max_scale - min_scale) * layer_id / (num_layers - 1)
                scales.append(scale)

        anchors = []

        for layer_id, stride in enumerate(strides):
            feature_h = int(math.ceil(self.in_h / float(stride)))
            feature_w = int(math.ceil(self.in_w / float(stride)))

            scale = scales[layer_id]
            if layer_id == num_layers - 1:
                scale_next = 1.0
            else:
                scale_next = scales[layer_id + 1]

            for y in range(feature_h):
                for x in range(feature_w):
                    x_center = (x + anchor_offset_x) / feature_w
                    y_center = (y + anchor_offset_y) / feature_h

                    for aspect_ratio in aspect_ratios:
                        ratio_sqrt = math.sqrt(aspect_ratio)
                        anchor_h = scale / ratio_sqrt
                        anchor_w = scale * ratio_sqrt
                        anchors.append((x_center, y_center, anchor_w, anchor_h))

                    if interpolated_scale_aspect_ratio > 0.0:
                        scale_interp = math.sqrt(scale * scale_next)
                        ratio = interpolated_scale_aspect_ratio
                        ratio_sqrt = math.sqrt(ratio)
                        anchor_h = scale_interp / ratio_sqrt
                        anchor_w = scale_interp * ratio_sqrt
                        anchors.append((x_center, y_center, anchor_w, anchor_h))

        return np.array(anchors, dtype=np.float32)

    @classmethod
    def try_create(cls, session, input_hw):
        outputs = session.get_outputs()
        box_idx = None
        score_idx = None
        for idx, out in enumerate(outputs):
            shape = _node_static_shape(out)
            if len(shape) >= 3:
                last_dim = shape[-1]
                if last_dim == 18 or last_dim == 7 or last_dim == 16:
                    if box_idx is None:
                        box_idx = idx
                elif last_dim == 1 or last_dim == 2:
                    if score_idx is None:
                        score_idx = idx
        if box_idx is None or score_idx is None:
            return None
        return cls(input_hw, box_idx, score_idx)

    def decode(self, outputs):
        raw_boxes = np.array(outputs[self.box_index])
        raw_scores = np.array(outputs[self.score_index])

        raw_boxes = raw_boxes.reshape(-1, raw_boxes.shape[-1])
        raw_scores = raw_scores.reshape(-1, raw_scores.shape[-1])

        if raw_boxes.shape[0] != self.anchors.shape[0]:
            count = min(raw_boxes.shape[0], self.anchors.shape[0])
            raw_boxes = raw_boxes[:count]
            raw_scores = raw_scores[:count]
            anchors = self.anchors[:count]
        else:
            anchors = self.anchors

        if raw_scores.shape[-1] == 1:
            scores = _sigmoid(np.clip(raw_scores[:, 0], -self.score_clip, self.score_clip))
        else:
            probs = _softmax(np.clip(raw_scores, -self.score_clip, self.score_clip))
            scores = probs[:, -1]

        detections = self._decode_with_anchors(raw_boxes, scores, anchors)
        if detections:
            return detections

        # fallback heuristics when anchor metadata mismatches the export
        if not self._warned_anchor_mismatch:
            print("[DET] Не удалось сопоставить anchors с выходами модели — включаем эвристики декодирования")
            self._warned_anchor_mismatch = True
        detections = self._decode_direct_xywh(raw_boxes, scores)
        if detections:
            return detections

        return self._decode_direct_xyxy(raw_boxes, scores)

    def _decode_with_anchors(self, raw_boxes, scores, anchors):
        detections = []
        num_vals = raw_boxes.shape[1]
        num_keypoints = max(0, (num_vals - 4) // 2)
        kp_to_use = min(self._max_keypoints, num_keypoints)
        for i, rb in enumerate(raw_boxes):
            sc = float(scores[i])
            if sc <= 0:
                continue
            anchor = anchors[i]
            x_center = rb[0] / self.x_scale * anchor[2] + anchor[0]
            y_center = rb[1] / self.y_scale * anchor[3] + anchor[1]
            w = math.exp(rb[2] / self.w_scale) * anchor[2]
            h = math.exp(rb[3] / self.h_scale) * anchor[3]

            x1 = (x_center - 0.5 * w) * self.in_w
            y1 = (y_center - 0.5 * h) * self.in_h
            x2 = (x_center + 0.5 * w) * self.in_w
            y2 = (y_center + 0.5 * h) * self.in_h
            if not np.isfinite([x1, y1, x2, y2]).all():
                continue
            x1 = float(np.clip(x1, 0.0, self.in_w))
            y1 = float(np.clip(y1, 0.0, self.in_h))
            x2 = float(np.clip(x2, 0.0, self.in_w))
            y2 = float(np.clip(y2, 0.0, self.in_h))
            if x2 <= x1 or y2 <= y1:
                continue

            keypoints = []
            for kp_idx in range(kp_to_use):
                base = 4 + kp_idx * 2
                if base + 1 >= num_vals:
                    break
                kx = rb[base] / self.x_scale * anchor[2] + anchor[0]
                ky = rb[base + 1] / self.y_scale * anchor[3] + anchor[1]
                keypoints.append((float(kx * self.in_w), float(ky * self.in_h)))

            xc = float(x_center * self.in_w)
            yc = float(y_center * self.in_h)
            w_px = float(w * self.in_w)
            h_px = float(h * self.in_h)

            rotation = 0.0
            if len(keypoints) >= 2:
                ref = keypoints[2] if len(keypoints) > 2 else keypoints[1]
                vec_x = ref[0] - keypoints[0][0]
                vec_y = ref[1] - keypoints[0][1]
                rotation = 0.5 * math.pi - math.atan2(vec_y, vec_x)

            shift_x = 0.0
            shift_y = -0.5
            roi_center_x = xc + shift_x * w_px
            roi_center_y = yc + shift_y * h_px
            roi_size = max(w_px, h_px) * 2.6
            roi_size = max(roi_size, 1.0)

            detections.append(
                {
                    "xyxy": (x1, y1, x2, y2),
                    "score": sc,
                    "roi": {
                        "center": (roi_center_x, roi_center_y),
                        "size": roi_size,
                        "rotation": float(rotation),
                    },
                    "keypoints": keypoints,
                }
            )
        return detections

    def _decode_direct_xywh(self, raw_boxes, scores):
        detections = []
        for rb, sc in zip(raw_boxes, scores):
            sc = float(sc)
            if sc <= 0:
                continue
            vals = rb[:4]
            if not np.isfinite(vals).all():
                continue
            x_c, y_c, w, h = vals
            if max(abs(x_c), abs(y_c)) > 2.5:
                continue
            # treat as normalized center-width-height (0..1)
            if max(abs(w), abs(h)) <= 1.5:
                x1 = (x_c - 0.5 * w) * self.in_w
                y1 = (y_c - 0.5 * h) * self.in_h
                x2 = (x_c + 0.5 * w) * self.in_w
                y2 = (y_c + 0.5 * h) * self.in_h
            else:
                x1 = x_c - 0.5 * w
                y1 = y_c - 0.5 * h
                x2 = x_c + 0.5 * w
                y2 = y_c + 0.5 * h
            x1 = float(np.clip(x1, 0.0, self.in_w))
            y1 = float(np.clip(y1, 0.0, self.in_h))
            x2 = float(np.clip(x2, 0.0, self.in_w))
            y2 = float(np.clip(y2, 0.0, self.in_h))
            if x2 <= x1 or y2 <= y1:
                continue
            detections.append({"xyxy": (x1, y1, x2, y2), "score": sc})
        return detections

    def _decode_direct_xyxy(self, raw_boxes, scores):
        detections = []
        for rb, sc in zip(raw_boxes, scores):
            sc = float(sc)
            if sc <= 0:
                continue
            vals = rb[:4]
            if not np.isfinite(vals).all():
                continue
            x1, y1, x2, y2 = vals
            if max(abs(x1), abs(y1), abs(x2), abs(y2)) <= 1.5:
                x1 *= self.in_w
                y1 *= self.in_h
                x2 *= self.in_w
                y2 *= self.in_h
            x1 = float(np.clip(x1, 0.0, self.in_w))
            y1 = float(np.clip(y1, 0.0, self.in_h))
            x2 = float(np.clip(x2, 0.0, self.in_w))
            y2 = float(np.clip(y2, 0.0, self.in_h))
            if x2 <= x1 or y2 <= y1:
                continue
            detections.append({"xyxy": (x1, y1, x2, y2), "score": sc})
        return detections


class DetectorONNX:
    """Generic ONNX detector wrapper that auto-detects BlazePalm outputs."""

    def __init__(self, path, input_size):
        self.path = path
        self.sess = create_onnx_session(path, prefer_cuda=True, allow_fallback=True, log_prefix="[DET]")
        self.inp = self.sess.get_inputs()[0]
        self.in_h, self.in_w = _resolve_input_hw(input_size, self.inp, "[DET]")
        self.out_names = [o.name for o in self.sess.get_outputs()]
        self.decoder = _PalmDetectorDecoder.try_create(self.sess, (self.in_h, self.in_w))
    def infer(self, rgb):
        lb, meta = _letterbox(rgb, self.in_h, self.in_w)
        x = lb.astype(np.float32) / 127.5 - 1.0
        x = np.transpose(x,(2,0,1))[None,...]
        out = self.sess.run(self.out_names, {self.inp.name: x})

        detections = []
        if self.decoder is not None:
            detections = self.decoder.decode(out)
        else:
            y = out[0]
            if y.ndim == 3 and y.shape[-1] >= 6:
                for row in y[0]:
                    score = float(row[4])
                    if score <= 0:
                        continue
                    detections.append({"xyxy": (float(row[0]), float(row[1]), float(row[2]), float(row[3])), "score": score})
            elif y.ndim == 2 and y.shape[-1] >= 6:
                for row in y:
                    score = float(row[4])
                    if score <= 0:
                        continue
                    detections.append({"xyxy": (float(row[0]), float(row[1]), float(row[2]), float(row[3])), "score": score})
            else:
                return [], meta

        if not detections:
            return [], meta

        boxes = np.array([det["xyxy"] for det in detections], dtype=np.float32)
        scores = np.array([float(det.get("score", 0.0)) for det in detections], dtype=np.float32)
        keep = _nms(boxes, scores, iou_th=0.45, topk=50)
        detections = [detections[i] for i in keep]

        mapped = []
        scale = meta.get("scale", 1.0)
        for det in detections:
            det_xyxy = _unletterbox_xyxy(det["xyxy"], meta)
            mapped_det = {
                "xyxy": det_xyxy,
                "score": float(det.get("score", 0.0)),
            }
            roi = det.get("roi")
            if roi:
                center = _unletterbox_point(roi.get("center", (0.0, 0.0)), meta)
                size = float(roi.get("size", 0.0)) / max(1e-8, scale)
                rotation = float(roi.get("rotation", 0.0))
                mapped_det["roi"] = {"center": center, "size": max(size, 1.0), "rotation": rotation}
            keypoints = det.get("keypoints")
            if keypoints:
                mapped_det["keypoints"] = [_unletterbox_point(kp, meta) for kp in keypoints]
            mapped.append(mapped_det)

        return mapped, meta

class LandmarkONNX:
    """Generic 21-keypoint head (RTMPose-Hand style) returning either (1,21,2/3) or (1,42/63) or heatmaps (1,21,H,W)."""
    def __init__(self, path, input_size, smooth=True, one_euro_cfg=None):
        self.sess = create_onnx_session(path, prefer_cuda=True, allow_fallback=True, log_prefix="[LMK]")
        self.inp = self.sess.get_inputs()[0]
        self.in_h, self.in_w = _resolve_input_hw(input_size, self.inp, "[LMK]")
        self.out_names = [o.name for o in self.sess.get_outputs()]
        self.smooth = smooth
        self.filters = [OneEuro(**(one_euro_cfg or {"min_cutoff":1.2,"beta":0.025})) for _ in range(21*2)]
        self._warned_range = False

    def _pre(self, crop):
        if crop.shape[0] != self.in_h or crop.shape[1] != self.in_w:
            crop = cv2.resize(crop, (self.in_w, self.in_h), interpolation=cv2.INTER_LINEAR)
        x = crop.astype(np.float32)
        x = x / 127.5 - 1.0
        x = np.transpose(x,(2,0,1))[None,...]
        return x

    def _normalize_pts(self, pts):
        arr = np.array(pts, dtype=np.float32)
        if arr.ndim != 2 or arr.shape[0] == 0:
            return None
        raw_min = float(np.min(arr))
        raw_max = float(np.max(arr))
        if raw_min < -0.1 or raw_max > 1.1:
            arr = (arr + 1.0) * 0.5
            if not self._warned_range:
                print("[LMK] Выход landmark-модели в диапазоне [-1,1] — конвертируем к [0,1]")
                self._warned_range = True
        arr = np.clip(arr, 0.0, 1.0)
        return arr

    def _warp_roi(self, rgb, roi):
        cx, cy = roi.get("center", (0.0, 0.0))
        size = float(roi.get("size", 0.0))
        rotation = float(roi.get("rotation", 0.0))
        size = max(size, 1.0)
        half = size / 2.0
        cos_r = math.cos(rotation)
        sin_r = math.sin(rotation)
        offsets = [(-half, -half), (half, -half), (half, half)]
        src = []
        for dx, dy in offsets:
            rx = cx + dx * cos_r - dy * sin_r
            ry = cy + dx * sin_r + dy * cos_r
            src.append([rx, ry])
        src = np.float32(src)
        dst = np.float32([[0.0, 0.0], [self.in_w - 1.0, 0.0], [self.in_w - 1.0, self.in_h - 1.0]])
        M = cv2.getAffineTransform(src, dst)
        crop = cv2.warpAffine(
            rgb,
            M,
            (self.in_w, self.in_h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REPLICATE,
        )
        Minv = cv2.invertAffineTransform(M)
        return crop, Minv

    @staticmethod
    def _argmax2d(hm):
        H,W = hm.shape; idx = int(np.argmax(hm)); y,x = divmod(idx, W); v = float(hm[y,x])
        return (x/(W-1 if W>1 else 1), y/(H-1 if H>1 else 1), v)

    def _parse(self, outputs):
        y = outputs[0]
        conf = 1.0

        squeezed = np.squeeze(y)

        if squeezed.ndim == 3 and squeezed.shape[0] == 21 and squeezed.shape[1] > 4:
            pts = []
            vs = []
            for i in range(21):
                x01, y01, v = self._argmax2d(squeezed[i])
                pts.append((x01, y01))
                vs.append(v)
            conf = float(np.mean(vs))
            return np.array(pts, dtype=np.float32), conf

        if squeezed.ndim == 3 and squeezed.shape[0] == 21 and squeezed.shape[1] in (2, 3):
            pts = squeezed[:, :2].astype(np.float32)
            if squeezed.shape[1] >= 3:
                conf = float(np.mean(np.clip(squeezed[:, 2], 0.0, 1.0)))
            return pts, conf

        if squeezed.ndim == 2 and squeezed.shape[0] == 21:
            pts = squeezed[:, :2].astype(np.float32)
            if squeezed.shape[1] >= 3:
                conf = float(np.mean(np.clip(squeezed[:, 2], 0.0, 1.0)))
            return pts, conf

        if squeezed.ndim == 1 and squeezed.shape[0] in (42, 63):
            stride = 3 if squeezed.shape[0] == 63 else 2
            pts = squeezed.reshape(21, stride)[:, :2].astype(np.float32)
            return pts, conf

        if squeezed.ndim == 2 and squeezed.shape[0] in (42, 63):
            stride = 3 if squeezed.shape[0] == 63 else 2
            pts = squeezed.reshape(21, stride)[:, :2].astype(np.float32)
            return pts, conf

        return None, 0.0

    def infer(self, rgb, detection):
        roi = detection.get("roi") if isinstance(detection, dict) else None
        Minv = None
        if roi:
            crop, Minv = self._warp_roi(rgb, roi)
        else:
            x1, y1, x2, y2 = detection["xyxy"] if isinstance(detection, dict) else detection
            H, W = rgb.shape[:2]
            x1i, y1i, x2i, y2i = map(lambda v: int(round(v)), (x1, y1, x2, y2))
            x1i = int(np.clip(x1i, 0, max(0, W - 1)))
            y1i = int(np.clip(y1i, 0, max(0, H - 1)))
            x2i = int(np.clip(x2i, x1i + 1, max(1, W)))
            y2i = int(np.clip(y2i, y1i + 1, max(1, H)))
            crop = rgb[y1i:y2i, x1i:x2i].copy()
        if crop.size == 0:
            return None, 0.0

        x = self._pre(crop)
        out = self.sess.run(self.out_names, {self.inp.name: x})
        pts01, conf = self._parse(out)
        if pts01 is None:
            return None, 0.0

        pts01 = self._normalize_pts(pts01)
        if pts01 is None:
            return None, 0.0

        lm = []
        if roi and Minv is not None:
            for i, (px, py) in enumerate(pts01):
                pxn = float(px) * (self.in_w - 1.0)
                pyn = float(py) * (self.in_h - 1.0)
                X = float(Minv[0, 0] * pxn + Minv[0, 1] * pyn + Minv[0, 2])
                Y = float(Minv[1, 0] * pxn + Minv[1, 1] * pyn + Minv[1, 2])
                if self.smooth:
                    X = self.filters[i * 2 + 0].apply(X)
                    Y = self.filters[i * 2 + 1].apply(Y)
                lm.append((X, Y))
        else:
            x1, y1, x2, y2 = detection["xyxy"] if isinstance(detection, dict) else detection
            bw, bh = max(1.0, x2 - x1), max(1.0, y2 - y1)
            for i, (px, py) in enumerate(pts01):
                X = x1 + float(px) * bw
                Y = y1 + float(py) * bh
                if self.smooth:
                    X = self.filters[i * 2 + 0].apply(X)
                    Y = self.filters[i * 2 + 1].apply(Y)
                lm.append((float(X), float(Y)))

        return lm, float(conf)

class TwoStageHandTracker:
    def __init__(self, det_path, det_input_size, lmk_path, lmk_input_size, max_hands=2, score_th=0.3, nms_th=0.45, presence_th=0.35, smooth=True, one_euro=None):
        self.det = DetectorONNX(det_path, det_input_size)
        self.lmk = LandmarkONNX(lmk_path, lmk_input_size, smooth=smooth, one_euro_cfg=one_euro)
        self.max_hands = max_hands
        self.score_th = score_th
        self.nms_th = nms_th
        self.presence_th = presence_th

    def process(self, rgb):
        detections, _ = self.det.infer(rgb)
        H,W = rgb.shape[:2]
        hands_out = []
        # filter by threshold, keep top-k
        dets = [det for det in detections if float(det.get("score", 0.0)) >= self.score_th]
        dets = sorted(dets, key=lambda d: d.get("score", 0.0), reverse=True)[:self.max_hands]
        for det in dets:
            lm, conf = self.lmk.infer(rgb, det)
            if lm is None or conf < self.presence_th:
                continue  # suppress "flying points"
            # normalize to [0..1] for app
            lm01 = [(float(np.clip(lx / max(1.0, W), 0.0, 1.0)), float(np.clip(ly / max(1.0, H), 0.0, 1.0))) for (lx, ly) in lm]
            label = "Right" if np.mean([p[0] for p in lm01]) > 0.5 else "Left"
            hands_out.append({"lm": lm01, "label": label, "score": float(conf)})
        return hands_out
