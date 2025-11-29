# -*- coding: utf-8 -*-

class MediaPipeHandTracker:
    """Wrapper around MediaPipe Hands providing a common tracker interface."""
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
        """Run MediaPipe on an RGB frame and return normalized landmarks."""
        results = self.hands.process(rgb)
        out = []
        if not results.multi_hand_landmarks: return out
        iw, ih = rgb.shape[1], rgb.shape[0]
        for lm, handed in zip(results.multi_hand_landmarks, results.multi_handedness):
            pts = [(float(p.x), float(p.y)) for p in lm.landmark]
            score = float(handed.classification[0].score) if handed and handed.classification else 0.0
            label = handed.classification[0].label if handed and handed.classification else "Unknown"
            pts = [(max(0.0, min(1.0, x)), max(0.0, min(1.0, y))) for (x, y) in pts]
            out.append({"lm": pts, "label": label, "score": score})
        return out
