"""Trigger helpers."""

class DebouncedTrigger:
    """Utility that fires after a dwell time while respecting a refractory window."""

    def __init__(self, dwell_ms: int = 260, refractory_ms: int = 900):
        self.dwell_ms = dwell_ms
        self.refractory_ms = refractory_ms
        self.candidate_since = None
        self.last_fire = 0

    def update(self, now_ms: int, active: bool) -> bool:
        """Return True when active has stayed asserted long enough to trigger."""
        if not active:
            self.candidate_since = None
            return False
        if self.candidate_since is None:
            self.candidate_since = now_ms
            return False
        if (now_ms - self.candidate_since) >= self.dwell_ms and (now_ms - self.last_fire) >= self.refractory_ms:
            self.last_fire = now_ms
            self.candidate_since = None
            return True
        return False
