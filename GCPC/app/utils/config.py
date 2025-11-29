"""Configuration helpers for GCPC."""
import json
from pathlib import Path
from typing import Any, Dict

APP_DIR = Path(__file__).resolve().parents[1]
ROOT = APP_DIR.parent


def load_config(config_path: Path | str | None = None) -> Dict[str, Any]:
    """Load application configuration from the root ``config.json`` file."""
    path = Path(config_path) if config_path else ROOT / "config.json"
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def build_hands(cfg: Dict[str, Any]) -> Dict[str, str]:
    """Construct a mapping describing dominant and support hands."""
    hands_cfg = cfg.get("hands") or {}
    dominant = hands_cfg.get("dominant")
    if dominant not in {"RIGHT", "LEFT"}:
        raise ValueError("[CFG] INCORRECT DOMINANT HAND")
    support = hands_cfg.get("support")
    if support not in {"RIGHT", "LEFT"}:
        raise ValueError("[CFG] INCORRECT SUPPORT HAND")
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
