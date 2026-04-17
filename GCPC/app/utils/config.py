"""Configuration helpers for GCPC."""
import json
import sys
from pathlib import Path
from typing import Any, Dict

_SOURCE_APP_DIR = Path(__file__).resolve().parents[1]
if getattr(sys, "frozen", False):
    ROOT = Path(sys.executable).resolve().parent
    APP_DIR = ROOT / "app"
else:
    APP_DIR = _SOURCE_APP_DIR
    ROOT = APP_DIR.parent


def load_config(config_path: Path | str | None = None) -> Dict[str, Any]:
    """Load application configuration from the root ``config.json`` file."""
    path = Path(config_path) if config_path else ROOT / "config.json"
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_config(cfg: Dict[str, Any], config_path: Path | str | None = None) -> None:
    """Persist configuration back to disk with UTF-8 encoding."""
    path = Path(config_path) if config_path else ROOT / "config.json"
    with path.open("w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2)


def build_hands(cfg: Dict[str, Any]) -> Dict[str, str]:
    """Construct a mapping describing dominant and support hands."""
    hands_cfg = cfg.get("hands") or {}
    dominant = hands_cfg.get("dominant")
    if dominant not in {"RIGHT", "LEFT"}:
        raise ValueError("[CFG] INCORRECT DOMINANT HAND")

    support = hands_cfg.get("support")
    if support not in {"RIGHT", "LEFT"}:
        raise ValueError("[CFG] INCORRECT SUPPORT HAND")
    if support == dominant:
        raise ValueError("[CFG] dominant and support hands must differ")

    return {"dominant": dominant, "support": support}


def resolve_side(tag: Any, hands: Dict[str, str]) -> str:
    """Resolve textual hand tag into a concrete side using provided defaults."""
    dominant_side = hands.get("dominant")
    support_side = hands.get("support")
    if tag is None:
        return dominant_side
    tag = str(tag).strip().upper()
    if not tag:
        return dominant_side
    if tag in {"RIGHT", "LEFT", "BOTH", "EITHER", "ANY"}:
        return tag
    if tag == "DOMINANT":
        return dominant_side
    if tag == "NON_DOMINANT":
        return support_side
    return tag or dominant_side
