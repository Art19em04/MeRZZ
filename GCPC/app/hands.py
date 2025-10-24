# app/hands.py
import os
from dataclasses import dataclass
from typing import Optional
import cv2
import numpy as np

import mediapipe as mp

try:
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision as mp_vision
    from mediapipe.tasks.python.vision import HandLandmarker, HandLandmarkerOptions, HandLandmarkerResult, RunningMode
    MP_OK = True
except Exception:
    MP_OK = False
    HandLandmarker = None
    HandLandmarkerOptions = None
    HandLandmarkerResult = None
    RunningMode = None

@dataclass
class HandResult:
    handedness: str
    landmarks: np.ndarray  # (21,3) normalized
    world_landmarks: Optional[np.ndarray] = None
    score: float = 0.0

class HandTracker:
    def __init__(self, cfg):
        if not MP_OK:
            raise RuntimeError("MediaPipe not available. Install mediapipe>=0.10.*")

        model_path = cfg["models"]["hand_landmarker"]
        model_path = os.environ.get("HAND_LANDMARKER_TASK", model_path)
        if not os.path.isabs(model_path):
            model_path = os.path.abspath(model_path)
        if not os.path.exists(model_path):
            raise RuntimeError(
                "Не найден файл модели Hand Landmarker:\n"
                f"  {model_path}\n"
                "Положи 'hand_landmarker.task' по этому пути, "
                "или пропиши абсолютный путь в config.json → models.hand_landmarker, "
                "или установи переменную окружения HAND_LANDMARKER_TASK."
            )

        base_opts = mp_python.BaseOptions(model_asset_path=model_path)
        opts = HandLandmarkerOptions(
            base_options=base_opts,
            num_hands=1,
            running_mode=RunningMode.VIDEO,
            min_hand_detection_confidence=0.5,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self.task = HandLandmarker.create_from_options(opts)
        self._vision = mp_vision  # храним для типов/RunningMode

    def process(self, frame_bgr, timestamp_ms: int):
        if frame_bgr is None:
            return None

        # 1) Нормализуем формát кадра из камеры
        if frame_bgr.ndim == 2:
            frame_bgr = cv2.cvtColor(frame_bgr, cv2.COLOR_GRAY2BGR)
        if frame_bgr.shape[2] == 4:  # некоторые backends дают BGRA
            frame_bgr = cv2.cvtColor(frame_bgr, cv2.COLOR_BGRA2BGR)

        # 2) Правильный RGB БЕЗ «::-1» (он даёт отрицательные страйды)
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        frame_rgb = np.ascontiguousarray(frame_rgb, dtype=np.uint8)  # на всякий случай

        # 3) Создаём mp.Image и взываем Tasks API
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        res = self.task.detect_for_video(mp_image, timestamp_ms)
        if not res or len(res.handedness) == 0 or len(res.hand_landmarks) == 0:
            return None

        idx = 0
        handed = res.handedness[idx][0].category_name
        score = res.handedness[idx][0].score
        pts = np.array([[lm.x, lm.y, lm.z] for lm in res.hand_landmarks[idx]], dtype=np.float32)
        return HandResult(handedness=handed, landmarks=pts, world_landmarks=None, score=score)
