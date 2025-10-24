import time
from dataclasses import dataclass

def now_ns():
    return time.perf_counter_ns()

def ns_to_ms(ns):
    return ns / 1_000_000.0

@dataclass
class Tick:
    t_frame_in: int = 0
    t_landmarks: int = 0
    t_decision: int = 0
    t_os_event: int = 0

    def e2e_ms(self):
        if self.t_frame_in and self.t_os_event:
            return ns_to_ms(self.t_os_event - self.t_frame_in)
        return 0.0
