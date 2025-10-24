import platform

import cv2, time

class Video:
    def __init__(self, cfg):
        v = cfg["video"]
        idx = v.get("camera_index", 0)

        # 👉 на Windows стабильнее DSHOW, плюс MJPG
        backend = cv2.CAP_DSHOW if platform.system().lower().startswith("win") else 0
        self.cap = cv2.VideoCapture(idx, backend)

        # MJPG сильно разгружает CPU/USB
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)   # было 960×540 — снижаем
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)
        self.cap.set(cv2.CAP_PROP_FPS, v.get("fps_min", 60))

        # минимальный буфер, чтобы не копились старые кадры
        try:
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception:
            pass

    def read(self):
        ok, frame = self.cap.read()
        return frame if ok else None

    def release(self):
        if self.cap: self.cap.release()