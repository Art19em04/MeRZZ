def get_hand_tracker(cfg):
    backend = cfg.get("tracker", "mediapipe").lower()
    if backend == "cuda_onnx":
        from hands_cuda_onnx import CudaHandTracker
        return CudaHandTracker(cfg)
    elif backend == "mediapipe":
        from hands import HandTracker
        return HandTracker(cfg)
    else:
        raise ValueError(f"Unknown tracker backend: {backend}")
