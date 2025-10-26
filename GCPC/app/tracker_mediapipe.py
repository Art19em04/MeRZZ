# -*- coding: utf-8 -*-
import numpy as np

class MediaPipeHandTracker:
    """
    Обёртка над mediapipe.solutions.hands (детектор + лэндмарки).
    Возвращает список dict по рукам: {"lm":[(x,y)*21],"label":"Right/Left","score":float}
    Координаты нормализованы [0..1] в пространстве исходного кадра.
    """
    def __init__(self, min_det=0.6, min_trk=0.5, max_hands=2, model_complexity=1):
        import mediapipe as mp
        self.mp = mp
        self.hands = mp.solutions.hands.Hands(
            static_image_mode=False,
            max_num_hands=max_hands,
            model_complexity=model_complexity,
            min_detection_confidence=min_det,
            min_tracking_confidence=min_trk
        )

    def process(self, rgb):
        # Mediapipe ожидает RGB без копии, но важно не использовать cv2.cvtColor тут
        results = self.hands.process(rgb)
        out=[]
        if not results.multi_hand_landmarks: return out
        iw, ih = rgb.shape[1], rgb.shape[0]
        for lm, handed in zip(results.multi_hand_landmarks, results.multi_handedness):
            pts=[(float(p.x), float(p.y)) for p in lm.landmark]
            score = float(handed.classification[0].score) if handed and handed.classification else 0.0
            label = handed.classification[0].label if handed and handed.classification else "Unknown"
            # safety clamp
            pts=[(max(0.0,min(1.0,x)), max(0.0,min(1.0,y))) for (x,y) in pts]
            out.append({"lm": pts, "label": label, "score": score})
        return out
