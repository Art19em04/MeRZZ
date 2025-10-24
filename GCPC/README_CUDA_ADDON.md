# CUDA Addon for Microgestures MVP

Drop-in CUDA (GPU) hand tracker using ONNX Runtime GPU as a replacement for MediaPipe Python Tasks.

Backends:
- tracker: "cuda_onnx" -> app/hands_cuda_onnx.py
- tracker: "mediapipe" -> app/hands.py

How to integrate
1) pip install onnxruntime-gpu
2) Put models into models/: hand_landmark.onnx and (optional) palm_detector.onnx
3) config.json additions:
{
  "tracker": "cuda_onnx",
  "models": {
    "palm_detector": "models/palm_detector.onnx",
    "hand_landmark": "models/hand_landmark.onnx"
  },
  "tracker_opts": {"det_every": 5, "pad_scale": 1.3, "min_det_conf": 0.4, "flip_lr": false}
}
4) from app.factory import get_hand_tracker; tracker = get_hand_tracker(cfg)
5) Run as module: app.main, working dir = project root, PYTHONUNBUFFERED=1
