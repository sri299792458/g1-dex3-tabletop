"""Injectable monotonic clocks for deterministic control tests."""

from __future__ import annotations

import time
from typing import Protocol


class MonotonicClock(Protocol):
    def monotonic(self) -> float: ...


class SystemClock:
    def monotonic(self) -> float:
        return time.monotonic()


class ManualClock:
    def __init__(self, initial_s: float = 0.0) -> None:
        if initial_s < 0:
            raise ValueError("initial clock time must be non-negative")
        self._time_s = float(initial_s)

    def monotonic(self) -> float:
        return self._time_s

    def advance(self, duration_s: float) -> float:
        if duration_s < 0:
            raise ValueError("clock cannot move backwards")
        self._time_s += duration_s
        return self._time_s
