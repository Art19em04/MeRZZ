import os, json, time
import numpy as np

def _json_default(o):
    # numpy → чистые Python-типы
    if isinstance(o, (np.floating,)):   # float32/64/...
        return float(o)
    if isinstance(o, (np.integer,)):    # int32/64/...
        return int(o)
    if isinstance(o, (np.ndarray,)):    # массивы → list из Python-скаляров
        return o.tolist()
    # на всякий случай — строка
    return str(o)

class Telemetry:
    def __init__(self, cfg):
        d = cfg["telemetry"]["dir"]
        pref = cfg["telemetry"].get("file_prefix", "session")
        os.makedirs(d, exist_ok=True)
        ts = time.strftime("%Y%m%d_%H%M%S")
        self.path = os.path.join(d, f"{pref}_{ts}.jsonl")

    def write(self, obj: dict):
        try:
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(json.dumps(obj, ensure_ascii=False, default=_json_default) + "\n")
        except Exception as e:
            # не валим пайплайн, если что-то пошло не так
            with open(self.path + ".err", "a", encoding="utf-8") as f:
                f.write(f"[ERR] {time.time()} {e}\n")
