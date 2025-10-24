import json
from pathlib import Path


def load_config(path: str = "config.json") -> dict:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)
