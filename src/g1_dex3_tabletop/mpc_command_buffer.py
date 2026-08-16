"""Validated rolling arm-command windows for an isolated MPC worker.

The CUDA process may optimize asynchronously, but it never publishes robot
commands.  This module is the small, ROS-free boundary that decides whether a
returned window is safe to expose to the existing 250 Hz command loop.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any

import numpy as np


def _sha256(document: dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(document, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def _q7(value: Any, *, name: str) -> tuple[float, ...]:
    array = np.asarray(value, dtype=np.float64).reshape(-1)
    if array.shape != (7,) or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain seven finite values")
    return tuple(float(item) for item in array)


@dataclass(frozen=True, slots=True)
class MPCCommandWindow:
    """One hash-bound, finite-horizon command returned by CuRobo MPC.

    ``sample_time_s[0]`` is always zero and ``command_q_rad[0]`` is the exact
    command active when the optimization request was made.  This makes every
    replacement continuous at the controller boundary instead of relying on
    the measured arm state and commanded arm state being identical.
    """

    generation: int
    plan_sha256: str
    state_monotonic_s: float
    sample_time_s: tuple[float, ...]
    command_q_rad: tuple[tuple[float, ...], ...]
    feasible: bool
    terminal: bool
    solve_time_s: float
    diagnostics: dict[str, Any]
    schema_version: int = 1

    def __post_init__(self) -> None:
        if self.generation < 0:
            raise ValueError("MPC generation must be non-negative")
        if len(self.plan_sha256) != 64:
            raise ValueError("MPC plan SHA-256 must contain 64 characters")
        state_time = float(self.state_monotonic_s)
        solve_time = float(self.solve_time_s)
        if not np.isfinite(state_time) or state_time < 0.0:
            raise ValueError("MPC state time must be finite and non-negative")
        if not np.isfinite(solve_time) or solve_time < 0.0:
            raise ValueError("MPC solve time must be finite and non-negative")
        times = np.asarray(self.sample_time_s, dtype=np.float64).reshape(-1)
        commands = np.asarray(self.command_q_rad, dtype=np.float64)
        if (
            len(times) < 2
            or times[0] != 0.0
            or not np.all(np.isfinite(times))
            or not np.all(np.diff(times) > 0.0)
        ):
            raise ValueError("MPC window times must start at zero and increase")
        if commands.shape != (len(times), 7) or not np.all(np.isfinite(commands)):
            raise ValueError("MPC window commands must be a finite N x 7 array")
        if not isinstance(self.diagnostics, dict):
            raise TypeError("MPC diagnostics must be a dictionary")
        object.__setattr__(self, "state_monotonic_s", state_time)
        object.__setattr__(self, "solve_time_s", solve_time)
        object.__setattr__(self, "sample_time_s", tuple(float(item) for item in times))
        object.__setattr__(
            self,
            "command_q_rad",
            tuple(tuple(float(item) for item in row) for row in commands),
        )
        # Prove diagnostics can cross the line-delimited JSON process boundary.
        json.dumps(self.diagnostics, sort_keys=True, separators=(",", ":"), allow_nan=False)

    @property
    def content_sha256(self) -> str:
        return _sha256(self.to_dict(include_hash=False))

    @property
    def duration_s(self) -> float:
        return self.sample_time_s[-1]

    def peak_velocity_rad_s(self) -> float:
        times = np.asarray(self.sample_time_s, dtype=np.float64)
        commands = np.asarray(self.command_q_rad, dtype=np.float64)
        return float(np.max(np.abs(np.diff(commands, axis=0)) / np.diff(times)[:, None]))

    def rebase_start(self, active_command_q_rad: Any) -> MPCCommandWindow:
        """Replace only the zero-time sample after an asynchronous solve.

        The old trajectory may continue moving while CUDA solves.  The
        returned future samples remain unchanged; the controller supplies its
        exact command at receipt time and then re-runs the complete velocity
        validation before accepting this rebased window.
        """

        active = _q7(active_command_q_rad, name="active command")
        original = np.asarray(self.command_q_rad[0], dtype=np.float64)
        diagnostics = dict(self.diagnostics)
        diagnostics["start_rebase_rad"] = float(
            np.max(np.abs(np.asarray(active, dtype=np.float64) - original))
        )
        diagnostics["worker_window_sha256"] = self.content_sha256
        return MPCCommandWindow(
            generation=self.generation,
            plan_sha256=self.plan_sha256,
            state_monotonic_s=self.state_monotonic_s,
            sample_time_s=self.sample_time_s,
            command_q_rad=(active, *self.command_q_rad[1:]),
            feasible=self.feasible,
            terminal=self.terminal,
            solve_time_s=self.solve_time_s,
            diagnostics=diagnostics,
        )

    def to_dict(self, *, include_hash: bool = True) -> dict[str, Any]:
        result = {
            "schema_version": self.schema_version,
            "generation": self.generation,
            "plan_sha256": self.plan_sha256,
            "state_monotonic_s": self.state_monotonic_s,
            "sample_time_s": list(self.sample_time_s),
            "command_q_rad": [list(row) for row in self.command_q_rad],
            "feasible": self.feasible,
            "terminal": self.terminal,
            "solve_time_s": self.solve_time_s,
            "diagnostics": self.diagnostics,
        }
        if include_hash:
            result["content_sha256"] = self.content_sha256
        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> MPCCommandWindow:
        values = dict(data)
        expected_hash = values.pop("content_sha256", None)
        result = cls(**values)
        if expected_hash is not None and expected_hash != result.content_sha256:
            raise ValueError("MPC command-window SHA-256 mismatch")
        return result


class RollingMPCCommandBuffer:
    """Continuously interpolate accepted MPC windows at the control rate."""

    def __init__(
        self,
        *,
        plan_sha256: str,
        maximum_velocity_rad_s: float,
        maximum_state_age_s: float,
        maximum_window_gap_s: float,
    ) -> None:
        if len(plan_sha256) != 64:
            raise ValueError("MPC plan SHA-256 must contain 64 characters")
        for name, value in (
            ("maximum_velocity_rad_s", maximum_velocity_rad_s),
            ("maximum_state_age_s", maximum_state_age_s),
            ("maximum_window_gap_s", maximum_window_gap_s),
        ):
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be positive and finite")
        self.plan_sha256 = plan_sha256
        self.maximum_velocity_rad_s = float(maximum_velocity_rad_s)
        self.maximum_state_age_s = float(maximum_state_age_s)
        self.maximum_window_gap_s = float(maximum_window_gap_s)
        self._window: MPCCommandWindow | None = None
        self._accepted_at_s: float | None = None
        self._last_generation = -1

    @property
    def last_generation(self) -> int:
        return self._last_generation

    @property
    def terminal(self) -> bool:
        return self._window is not None and self._window.terminal

    @property
    def duration_s(self) -> float:
        if self._window is None:
            raise RuntimeError("no MPC command window has been accepted")
        return self._window.duration_s

    def elapsed_s(self, *, now_s: float) -> float:
        if self._accepted_at_s is None:
            raise RuntimeError("no MPC command window has been accepted")
        elapsed = float(now_s) - self._accepted_at_s
        if not np.isfinite(elapsed) or elapsed < 0.0:
            raise ValueError("MPC command time moved backwards")
        return elapsed

    def remaining_s(self, *, now_s: float) -> float:
        return max(self.duration_s - self.elapsed_s(now_s=now_s), 0.0)

    def accept(
        self,
        window: MPCCommandWindow,
        *,
        now_s: float,
        active_command_q_rad: Any,
    ) -> None:
        now = float(now_s)
        if not np.isfinite(now) or now < 0.0:
            raise ValueError("current monotonic time must be finite and non-negative")
        active = np.asarray(_q7(active_command_q_rad, name="active command"))
        if window.plan_sha256 != self.plan_sha256:
            raise ValueError("MPC window belongs to a different frozen plan")
        if window.generation <= self._last_generation:
            raise ValueError("MPC window generation is stale or repeated")
        if not window.feasible:
            raise ValueError("MPC worker marked the command window infeasible")
        state_age = now - window.state_monotonic_s
        if state_age < 0.0 or state_age > self.maximum_state_age_s:
            raise ValueError(
                f"MPC source state age {state_age:.4f}s exceeds "
                f"{self.maximum_state_age_s:.4f}s"
            )
        commands = np.asarray(window.command_q_rad, dtype=np.float64)
        continuity_error = float(np.max(np.abs(commands[0] - active)))
        if continuity_error > 1.0e-8:
            raise ValueError(
                "MPC window does not start at the active command: "
                f"error={continuity_error:.9f}rad"
            )
        peak_velocity = window.peak_velocity_rad_s()
        if peak_velocity > self.maximum_velocity_rad_s + 1.0e-6:
            raise ValueError(
                f"MPC window velocity {peak_velocity:.4f}rad/s exceeds "
                f"{self.maximum_velocity_rad_s:.4f}rad/s"
            )
        self._window = window
        self._accepted_at_s = now
        self._last_generation = window.generation

    def command(self, *, now_s: float) -> np.ndarray:
        if self._window is None or self._accepted_at_s is None:
            raise RuntimeError("no MPC command window has been accepted")
        elapsed = float(now_s) - self._accepted_at_s
        if not np.isfinite(elapsed) or elapsed < 0.0:
            raise ValueError("MPC command time moved backwards")
        if elapsed > self._window.duration_s + self.maximum_window_gap_s:
            raise RuntimeError(
                f"MPC command window expired {elapsed - self._window.duration_s:.4f}s ago; "
                f"limit is {self.maximum_window_gap_s:.4f}s"
            )
        times = np.asarray(self._window.sample_time_s, dtype=np.float64)
        commands = np.asarray(self._window.command_q_rad, dtype=np.float64)
        sample_time = min(elapsed, self._window.duration_s)
        return np.asarray(
            [np.interp(sample_time, times, commands[:, index]) for index in range(7)],
            dtype=np.float64,
        )
