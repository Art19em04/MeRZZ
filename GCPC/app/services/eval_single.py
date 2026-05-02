# -*- coding: utf-8 -*-
from __future__ import annotations

from app.services.eval_manager import GestureEvalSession


class EvalSingleSession(GestureEvalSession):
    """Deprecated compatibility wrapper for the old single-gesture eval engine."""

    def __init__(self, cfg, hands, panel=None):
        _ = panel
        super().__init__(cfg, hands)

    def start(self, now_ms: int, session_id: str | None = None) -> bool:
        _ = session_id
        return super().start(now_ms)
