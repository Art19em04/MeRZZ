import platform

import cv2


class Video:
    def __init__(self, cfg):
        video_cfg = cfg["video"]
        idx = video_cfg.get("camera_index", 0)
        backend = cv2.CAP_DSHOW if platform.system().lower().startswith("win") else 0
        self.cap = cv2.VideoCapture(idx, backend)
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, video_cfg.get("width", 640))
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, video_cfg.get("height", 360))
        self.cap.set(cv2.CAP_PROP_FPS, video_cfg.get("fps_min", 60))
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    def read(self):
        ok, frame = self.cap.read()
        return frame if ok else None

    def release(self):
        if self.cap:
            self.cap.release()
