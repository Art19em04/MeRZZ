import json
import os
import time
from typing import Any

import numpy as np


def _json_default(value: Any):
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


class Telemetry:
    def __init__(self, cfg):
        telemetry_cfg = cfg["telemetry"]
        directory = telemetry_cfg["dir"]
        prefix = telemetry_cfg.get("file_prefix", "session")
        os.makedirs(directory, exist_ok=True)
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        self.path = os.path.join(directory, f"{prefix}_{timestamp}.jsonl")

    def write(self, payload: dict):
        with open(self.path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False, default=_json_default) + "\n")
