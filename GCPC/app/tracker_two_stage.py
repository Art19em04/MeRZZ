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

    def _generate_anchors(self):
        # MediaPipe palm detector anchor spec
        strides = [8, 16, 32, 32]
        anchor_offset_x = 0.5
        anchor_offset_y = 0.5
        aspect_ratios = [1.0]
        interpolated_scale_aspect_ratio = 1.0

        anchors = []

        for stride in strides:
            feature_h = int(math.ceil(self.in_h / float(stride)))
            feature_w = int(math.ceil(self.in_w / float(stride)))

            for y in range(feature_h):
                for x in range(feature_w):
                    x_center = (x + anchor_offset_x) / feature_w
                    y_center = (y + anchor_offset_y) / feature_h

                    for _ in aspect_ratios:
                        anchors.append((x_center, y_center, 1.0, 1.0))

                    if interpolated_scale_aspect_ratio > 0.0:
                        anchors.append((x_center, y_center, 1.0, 1.0))

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

        boxes = self._decode_with_anchors(raw_boxes, scores, anchors)
        if boxes:
            return boxes

        # fallback heuristics when anchor metadata mismatches the export
        if not self._warned_anchor_mismatch:
            print("[DET] Не удалось сопоставить anchors с выходами модели — включаем эвристики декодирования")
            self._warned_anchor_mismatch = True
        boxes = self._decode_direct_xywh(raw_boxes, scores)
        if boxes:
            return boxes

        return self._decode_direct_xyxy(raw_boxes, scores)

    def _decode_with_anchors(self, raw_boxes, scores, anchors):
        boxes = []
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
            boxes.append((x1, y1, x2, y2, sc))
        return boxes

    def _decode_direct_xywh(self, raw_boxes, scores):
        boxes = []
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
            boxes.append((x1, y1, x2, y2, sc))
        return boxes

    def _decode_direct_xyxy(self, raw_boxes, scores):
        boxes = []
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
            boxes.append((x1, y1, x2, y2, sc))
        return boxes


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
        x = lb.astype(np.float32)/255.0
        x = np.transpose(x,(2,0,1))[None,...]
        out = self.sess.run(self.out_names, {self.inp.name: x})

        boxes, scores = [], []
        if self.decoder is not None:
            decoded = self.decoder.decode(out)
            for (x1, y1, x2, y2, sc) in decoded:
                boxes.append([float(x1), float(y1), float(x2), float(y2)])
                scores.append(float(sc))
        else:
            y = out[0]
            if y.ndim==3 and y.shape[-1] >= 6:
                # assume (1, N, 6+) -> [x1,y1,x2,y2, score, class ...]
                for row in y[0]:
                    x1,y1,x2,y2,score,cls = float(row[0]),float(row[1]),float(row[2]),float(row[3]),float(row[4]),float(row[5])
                    if score>0: boxes.append([x1,y1,x2,y2]); scores.append(score)
            elif y.ndim==2 and y.shape[-1] >= 6:
                for row in y:
                    x1,y1,x2,y2,score,cls = float(row[0]),float(row[1]),float(row[2]),float(row[3]),float(row[4]),float(row[5])
                    if score>0: boxes.append([x1,y1,x2,y2]); scores.append(score)
            else:
                return [], meta
        # NMS in letterboxed space
        if boxes:
            keep = _nms(np.array(boxes), np.array(scores), iou_th=0.45, topk=50)
            boxes = [boxes[i] for i in keep]; scores = [scores[i] for i in keep]
        # map back to original
        xyxy = [_unletterbox_xyxy(b, meta) for b in boxes]
        return list(zip(xyxy, scores)), meta

class LandmarkONNX:
    """Generic 21-keypoint head (RTMPose-Hand style) returning either (1,21,2/3) or (1,42/63) or heatmaps (1,21,H,W)."""
    def __init__(self, path, input_size, smooth=True, one_euro_cfg=None):
        self.sess = create_onnx_session(path, prefer_cuda=True, allow_fallback=True, log_prefix="[LMK]")
        self.inp = self.sess.get_inputs()[0]
        self.in_h, self.in_w = _resolve_input_hw(input_size, self.inp, "[LMK]")
        self.out_names = [o.name for o in self.sess.get_outputs()]
        self.smooth = smooth
        self.filters = [OneEuro(**(one_euro_cfg or {"min_cutoff":1.2,"beta":0.025})) for _ in range(21*2)]

    def _pre(self, crop):
        x = cv2.resize(crop, (self.in_w, self.in_h), interpolation=cv2.INTER_LINEAR).astype(np.float32)/255.0
        x = np.transpose(x,(2,0,1))[None,...]
        return x

    @staticmethod
    def _argmax2d(hm):
        H,W = hm.shape; idx = int(np.argmax(hm)); y,x = divmod(idx, W); v = float(hm[y,x])
        return (x/(W-1 if W>1 else 1), y/(H-1 if H>1 else 1), v)

    def _parse(self, y):
        y=y[0]
        pts=None; conf=1.0
        if y.ndim==3 and y.shape[0]==1 and y.shape[1]==21 and y.shape[2] in (2,3):
            pts = [(float(y[0,i,0]), float(y[0,i,1])) for i in range(21)]
            if y.shape[2]==3: conf=float(np.mean(np.clip(y[0,:,2],0,1)))
        elif y.ndim==2 and y.shape[0]==1 and y.shape[1] in (42,63):
            has_z=(y.shape[1]==63); flat=y.reshape(-1); pts=[]
            for i in range(21):
                xi=flat[i*(3 if has_z else 2)+0]; yi=flat[i*(3 if has_z else 2)+1]
                pts.append((float(xi), float(yi)))
        elif y.ndim==4 and y.shape[0]==1 and y.shape[1]==21:
            pts=[]; vs=[]
            for i in range(21):
                x01,y01,v=self._argmax2d(y[0,i]); pts.append((x01,y01)); vs.append(v)
            conf=float(np.mean(vs))
        return pts, conf

    def infer(self, rgb_crop, bbox_xyxy):
        x = self._pre(rgb_crop)
        out = self.sess.run(self.out_names, {self.inp.name: x})
        pts01, conf = self._parse(out)
        if pts01 is None: return None, 0.0
        # map crop norm -> image coords
        x1,y1,x2,y2 = bbox_xyxy
        bw, bh = max(1.0, x2-x1), max(1.0, y2-y1)
        lm=[]
        for i,(px,py) in enumerate(pts01):
            X = x1 + px * bw; Y = y1 + py * bh
            if self.smooth:
                X = self.filters[i*2+0].apply(X)
                Y = self.filters[i*2+1].apply(Y)
            lm.append((float(X), float(Y)))
        # normalize to [0..1] later in main
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
        (detections, meta) = self.det.infer(rgb)
        H,W = rgb.shape[:2]
        hands_out = []
        # filter by threshold, keep top-k
        dets = [(xy,sc) for (xy,sc) in detections if sc >= self.score_th]
        dets = sorted(dets, key=lambda t: t[1], reverse=True)[:self.max_hands]
        for (x1,y1,x2,y2), sc in dets:
            x1i,y1i,x2i,y2i = map(lambda v:int(round(v)), (x1,y1,x2,y2))
            x1i,y1i = max(0,x1i), max(0,y1i); x2i,y2i = min(W-1,x2i), min(H-1,y2i)
            if x2i<=x1i or y2i<=y1i: continue
            crop = rgb[y1i:y2i, x1i:x2i].copy()
            lm, conf = self.lmk.infer(crop, (x1i,y1i,x2i,y2i))
            if lm is None or conf < self.presence_th: 
                continue  # suppress "flying points"
            # normalize to [0..1] for app
            lm01 = [(lx/W, ly/H) for (lx,ly) in lm]
            label = "Right" if np.mean([p[0] for p in lm01]) > 0.5 else "Left"
            hands_out.append({"lm": lm01, "label": label, "score": float(conf)})
        return hands_out
