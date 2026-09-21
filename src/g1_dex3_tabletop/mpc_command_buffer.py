"""Immutable, absolute-time MPC trajectories for the 250 Hz arm controller.

The CUDA worker never publishes a robot command. It returns one trajectory
whose command and predicted-state paths have already been sampled and safety
checked on the controller grid. The executor may schedule that trajectory,
but it must never rewrite it after certification.
"""

from __future__ import annotations

import hashlib
import json
import math
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


def _sample_rows(times: np.ndarray, values: np.ndarray, sample_time_s: float) -> np.ndarray:
    """Sample one piecewise-linear N x 7 trajectory."""

    sample = float(sample_time_s)
    if not np.isfinite(sample) or sample < times[0] or sample > times[-1] + 1.0e-9:
        raise ValueError("trajectory sample time is outside the certified interval")
    sample = min(sample, float(times[-1]))
    return np.asarray(
        [np.interp(sample, times, values[:, index]) for index in range(7)],
        dtype=np.float64,
    )


def command_sequence_from_measured_plan(
    planned_measured_q_rad: Any,
    *,
    measured_q_rad: Any,
    active_command_q_rad: Any,
    future_sample_time_s: Any,
) -> tuple[np.ndarray, np.ndarray]:
    """Carry the boundary position-loop offset through one MPC horizon.

    CuRobo predicts measured joint motion. The G1 position controller can be
    holding a different desired position under gravity/load, so the complete
    short horizon retains the exact boundary offset. This conversion happens
    before strict validation and is never repeated by the executor.
    """

    planned = np.asarray(planned_measured_q_rad, dtype=np.float64)
    if planned.ndim != 2 or planned.shape[0] < 1 or planned.shape[1] != 7:
        raise ValueError("measured-state MPC plan must be a non-empty N x 7 array")
    if not np.all(np.isfinite(planned)):
        raise ValueError("measured-state MPC plan must contain only finite values")
    times = np.asarray(future_sample_time_s, dtype=np.float64).reshape(-1)
    if (
        times.shape != (len(planned),)
        or not np.all(np.isfinite(times))
        or times[0] <= 0.0
        or not np.all(np.diff(times) > 0.0)
    ):
        raise ValueError("future MPC sample times must be finite, positive, and increasing")
    measured = np.asarray(_q7(measured_q_rad, name="measured position"))
    active = np.asarray(_q7(active_command_q_rad, name="active command"))
    tracking_offset = active - measured
    return planned + tracking_offset[None, :], tracking_offset


@dataclass(frozen=True, slots=True)
class MPCHandoffBoundary:
    """One future splice point sampled from the trajectory being executed."""

    valid_from_monotonic_s: float
    command_q_rad: tuple[float, ...]
    predicted_q_rad: tuple[float, ...]
    predicted_dq_rad_s: tuple[float, ...]
    predicted_ddq_rad_s2: tuple[float, ...]
    predecessor_sha256: str | None
    committed_route_progress_index: int

    def __post_init__(self) -> None:
        valid_from = float(self.valid_from_monotonic_s)
        if not np.isfinite(valid_from) or valid_from < 0.0:
            raise ValueError("MPC handoff time must be finite and non-negative")
        if self.predecessor_sha256 is not None and len(self.predecessor_sha256) != 64:
            raise ValueError("MPC predecessor SHA-256 must contain 64 characters")
        if self.committed_route_progress_index < 0:
            raise ValueError("committed route progress must be non-negative")
        object.__setattr__(self, "valid_from_monotonic_s", valid_from)
        object.__setattr__(self, "command_q_rad", _q7(self.command_q_rad, name="handoff command"))
        object.__setattr__(
            self,
            "predicted_q_rad",
            _q7(self.predicted_q_rad, name="handoff predicted position"),
        )
        object.__setattr__(
            self,
            "predicted_dq_rad_s",
            _q7(self.predicted_dq_rad_s, name="handoff predicted velocity"),
        )
        object.__setattr__(
            self,
            "predicted_ddq_rad_s2",
            _q7(self.predicted_ddq_rad_s2, name="handoff predicted acceleration"),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "valid_from_monotonic_s": self.valid_from_monotonic_s,
            "command_q_rad": list(self.command_q_rad),
            "predicted_q_rad": list(self.predicted_q_rad),
            "predicted_dq_rad_s": list(self.predicted_dq_rad_s),
            "predicted_ddq_rad_s2": list(self.predicted_ddq_rad_s2),
            "predecessor_sha256": self.predecessor_sha256,
            "committed_route_progress_index": self.committed_route_progress_index,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> MPCHandoffBoundary:
        return cls(**data)


@dataclass(frozen=True, slots=True)
class MPCCommandWindow:
    """One immutable, hash-bound trajectory certified by the CUDA worker."""

    generation: int
    plan_sha256: str
    source_state_monotonic_s: float
    valid_from_monotonic_s: float
    sample_time_s: tuple[float, ...]
    command_q_rad: tuple[tuple[float, ...], ...]
    predicted_q_rad: tuple[tuple[float, ...], ...]
    predicted_dq_rad_s: tuple[tuple[float, ...], ...]
    predicted_ddq_rad_s2: tuple[tuple[float, ...], ...]
    predecessor_sha256: str | None
    feasible: bool
    terminal: bool
    solve_time_s: float
    diagnostics: dict[str, Any]
    schema_version: int = 3

    def __post_init__(self) -> None:
        if self.generation < 0:
            raise ValueError("MPC generation must be non-negative")
        if len(self.plan_sha256) != 64:
            raise ValueError("MPC plan SHA-256 must contain 64 characters")
        if self.predecessor_sha256 is not None and len(self.predecessor_sha256) != 64:
            raise ValueError("MPC predecessor SHA-256 must contain 64 characters")
        source_time = float(self.source_state_monotonic_s)
        valid_from = float(self.valid_from_monotonic_s)
        solve_time = float(self.solve_time_s)
        if not np.isfinite(source_time) or source_time < 0.0:
            raise ValueError("MPC source-state time must be finite and non-negative")
        if not np.isfinite(valid_from) or valid_from <= source_time:
            raise ValueError("MPC valid-from time must follow its source-state time")
        if not np.isfinite(solve_time) or solve_time < 0.0:
            raise ValueError("MPC solve time must be finite and non-negative")
        times = np.asarray(self.sample_time_s, dtype=np.float64).reshape(-1)
        commands = np.asarray(self.command_q_rad, dtype=np.float64)
        predicted = np.asarray(self.predicted_q_rad, dtype=np.float64)
        predicted_dq = np.asarray(self.predicted_dq_rad_s, dtype=np.float64)
        predicted_ddq = np.asarray(self.predicted_ddq_rad_s2, dtype=np.float64)
        if (
            len(times) < 2
            or times[0] != 0.0
            or not np.all(np.isfinite(times))
            or not np.all(np.diff(times) > 0.0)
        ):
            raise ValueError("MPC window times must start at zero and increase")
        expected_shape = (len(times), 7)
        for name, values in (
            ("commands", commands),
            ("predicted positions", predicted),
            ("predicted velocities", predicted_dq),
            ("predicted accelerations", predicted_ddq),
        ):
            if values.shape != expected_shape or not np.all(np.isfinite(values)):
                raise ValueError(f"MPC window {name} must be a finite N x 7 array")
        if not isinstance(self.diagnostics, dict):
            raise TypeError("MPC diagnostics must be a dictionary")
        object.__setattr__(self, "source_state_monotonic_s", source_time)
        object.__setattr__(self, "valid_from_monotonic_s", valid_from)
        object.__setattr__(self, "solve_time_s", solve_time)
        object.__setattr__(self, "sample_time_s", tuple(float(item) for item in times))
        for field_name, values in (
            ("command_q_rad", commands),
            ("predicted_q_rad", predicted),
            ("predicted_dq_rad_s", predicted_dq),
            ("predicted_ddq_rad_s2", predicted_ddq),
        ):
            object.__setattr__(
                self,
                field_name,
                tuple(tuple(float(item) for item in row) for row in values),
            )
        json.dumps(self.diagnostics, sort_keys=True, separators=(",", ":"), allow_nan=False)

    @property
    def content_sha256(self) -> str:
        return _sha256(self.to_dict(include_hash=False))

    @property
    def duration_s(self) -> float:
        return self.sample_time_s[-1]

    @property
    def expiration_monotonic_s(self) -> float:
        return self.valid_from_monotonic_s + self.duration_s

    def peak_velocity_rad_s(self) -> float:
        times = np.asarray(self.sample_time_s, dtype=np.float64)
        commands = np.asarray(self.command_q_rad, dtype=np.float64)
        return float(np.max(np.abs(np.diff(commands, axis=0)) / np.diff(times)[:, None]))

    def sample_command(self, *, monotonic_s: float) -> np.ndarray:
        return self._sample(self.command_q_rad, monotonic_s=monotonic_s)

    def sample_predicted_q(self, *, monotonic_s: float) -> np.ndarray:
        return self._sample(self.predicted_q_rad, monotonic_s=monotonic_s)

    def sample_predicted_dq(self, *, monotonic_s: float) -> np.ndarray:
        return self._sample(self.predicted_dq_rad_s, monotonic_s=monotonic_s)

    def sample_predicted_ddq(self, *, monotonic_s: float) -> np.ndarray:
        return self._sample(self.predicted_ddq_rad_s2, monotonic_s=monotonic_s)

    def _sample(self, values: Any, *, monotonic_s: float) -> np.ndarray:
        offset = float(monotonic_s) - self.valid_from_monotonic_s
        return _sample_rows(
            np.asarray(self.sample_time_s, dtype=np.float64),
            np.asarray(values, dtype=np.float64),
            offset,
        )

    def to_dict(self, *, include_hash: bool = True) -> dict[str, Any]:
        result = {
            "schema_version": self.schema_version,
            "generation": self.generation,
            "plan_sha256": self.plan_sha256,
            "source_state_monotonic_s": self.source_state_monotonic_s,
            "valid_from_monotonic_s": self.valid_from_monotonic_s,
            "sample_time_s": list(self.sample_time_s),
            "command_q_rad": [list(row) for row in self.command_q_rad],
            "predicted_q_rad": [list(row) for row in self.predicted_q_rad],
            "predicted_dq_rad_s": [list(row) for row in self.predicted_dq_rad_s],
            "predicted_ddq_rad_s2": [list(row) for row in self.predicted_ddq_rad_s2],
            "predecessor_sha256": self.predecessor_sha256,
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
    """Schedule immutable MPC trajectories on one absolute monotonic timeline."""

    def __init__(
        self,
        *,
        plan_sha256: str,
        maximum_velocity_rad_s: float,
        maximum_handoff_position_error_rad: float,
        maximum_handoff_velocity_error_rad_s: float,
        activation_lateness_s: float,
    ) -> None:
        if len(plan_sha256) != 64:
            raise ValueError("MPC plan SHA-256 must contain 64 characters")
        for name, value in (
            ("maximum_velocity_rad_s", maximum_velocity_rad_s),
            ("maximum_handoff_position_error_rad", maximum_handoff_position_error_rad),
            ("maximum_handoff_velocity_error_rad_s", maximum_handoff_velocity_error_rad_s),
            ("activation_lateness_s", activation_lateness_s),
        ):
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be positive and finite")
        self.plan_sha256 = plan_sha256
        self.maximum_velocity_rad_s = float(maximum_velocity_rad_s)
        self.maximum_handoff_position_error_rad = float(maximum_handoff_position_error_rad)
        self.maximum_handoff_velocity_error_rad_s = float(maximum_handoff_velocity_error_rad_s)
        self.activation_lateness_s = float(activation_lateness_s)
        self._active: MPCCommandWindow | None = None
        self._queued: MPCCommandWindow | None = None
        self._prestart_command_q: np.ndarray | None = None
        self._last_generation = -1
        self._stop_after_active = False

    @property
    def last_generation(self) -> int:
        return self._last_generation

    @property
    def active(self) -> bool:
        return self._active is not None

    @property
    def has_queued(self) -> bool:
        return self._queued is not None

    @property
    def terminal(self) -> bool:
        return self._active is not None and (self._active.terminal or self._stop_after_active)

    @property
    def terminal_pending(self) -> bool:
        return self.terminal or (self._queued is not None and self._queued.terminal)

    @property
    def stop_requested(self) -> bool:
        return self._stop_after_active

    @property
    def active_content_sha256(self) -> str | None:
        return None if self._active is None else self._active.content_sha256

    @property
    def duration_s(self) -> float:
        window = self._active if self._active is not None else self._queued
        if window is None:
            raise RuntimeError("no MPC command window has been installed")
        return window.duration_s

    @property
    def committed_route_progress_index(self) -> int:
        if self._active is None:
            return 0
        return int(self._active.diagnostics.get("proposed_route_progress_index", 0))

    def remaining_s(self, *, now_s: float) -> float:
        window = self._active if self._active is not None else self._queued
        if window is None:
            raise RuntimeError("no MPC command window has been installed")
        return max(window.expiration_monotonic_s - float(now_s), 0.0)

    def install(
        self,
        window: MPCCommandWindow,
        *,
        now_s: float,
        active_command_q_rad: Any,
        handoff_boundary: MPCHandoffBoundary | None = None,
    ) -> None:
        """Queue a worker-certified trajectory without changing one byte of it."""

        now = float(now_s)
        if not np.isfinite(now) or now < 0.0:
            raise ValueError("current monotonic time must be finite and non-negative")
        if window.valid_from_monotonic_s <= now:
            raise ValueError(
                "MPC trajectory missed its absolute handoff time by "
                f"{now - window.valid_from_monotonic_s:.6f}s"
            )
        if window.plan_sha256 != self.plan_sha256:
            raise ValueError("MPC window belongs to a different frozen plan")
        if window.generation <= self._last_generation:
            raise ValueError("MPC window generation is stale or repeated")
        if not window.feasible:
            raise ValueError("MPC worker marked the command window infeasible")
        if self._stop_after_active:
            raise RuntimeError("MPC stream is already finishing its active safe horizon")
        if self._queued is not None:
            raise RuntimeError("an MPC trajectory is already queued")
        peak_velocity = window.peak_velocity_rad_s()
        if peak_velocity > self.maximum_velocity_rad_s + 1.0e-6:
            raise ValueError(
                f"MPC window velocity {peak_velocity:.4f}rad/s exceeds "
                f"{self.maximum_velocity_rad_s:.4f}rad/s"
            )

        first_command = np.asarray(window.command_q_rad[0], dtype=np.float64)
        if self._active is None:
            if handoff_boundary is not None:
                raise ValueError("first MPC trajectory must not name a streaming handoff")
            if window.predecessor_sha256 is not None:
                raise ValueError("first MPC trajectory unexpectedly names a predecessor")
            active = np.asarray(_q7(active_command_q_rad, name="active command"))
            continuity_error = float(np.max(np.abs(first_command - active)))
            self._prestart_command_q = active.copy()
        else:
            if handoff_boundary is None:
                raise ValueError("replacement MPC trajectory requires its frozen handoff boundary")
            if handoff_boundary.predecessor_sha256 != self._active.content_sha256:
                raise ValueError("frozen MPC handoff belongs to a different active trajectory")
            if window.valid_from_monotonic_s != handoff_boundary.valid_from_monotonic_s:
                raise ValueError("MPC trajectory does not use its frozen handoff time")
            if window.predecessor_sha256 != self._active.content_sha256:
                raise ValueError("MPC trajectory predecessor is not the active trajectory")
            # A certified window ends in a stationary position target.  If
            # perception or optimization needs longer than that horizon, the
            # fixed-rate controller keeps publishing the endpoint.  A later
            # window therefore splices from that held endpoint rather than
            # turning planner availability into a robot-control fault.
            predecessor_sample_s = min(
                window.valid_from_monotonic_s,
                self._active.expiration_monotonic_s,
            )
            old_command = self._active.sample_command(monotonic_s=predecessor_sample_s)
            boundary_command = np.asarray(handoff_boundary.command_q_rad, dtype=np.float64)
            frozen_command_error = float(np.max(np.abs(boundary_command - old_command)))
            if frozen_command_error > 1.0e-8:
                raise ValueError(
                    "frozen MPC handoff command does not match the active trajectory: "
                    f"error={frozen_command_error:.9f}rad"
                )
            continuity_error = float(np.max(np.abs(first_command - old_command)))
            predicted_error = float(
                np.max(
                    np.abs(
                        np.asarray(window.predicted_q_rad[0], dtype=np.float64)
                        - np.asarray(handoff_boundary.predicted_q_rad, dtype=np.float64)
                    )
                )
            )
            if predicted_error > 1.0e-8:
                raise ValueError(
                    "MPC predicted handoff does not match the frozen live-reanchored boundary: "
                    f"error={predicted_error:.9f}rad"
                )
            for label, new_value, old_value, unit, tolerance in (
                (
                    "velocity",
                    window.predicted_dq_rad_s[0],
                    handoff_boundary.predicted_dq_rad_s,
                    "rad/s",
                    1.0e-5,
                ),
                (
                    "acceleration",
                    window.predicted_ddq_rad_s2[0],
                    handoff_boundary.predicted_ddq_rad_s2,
                    "rad/s^2",
                    1.0e-4,
                ),
            ):
                derivative_error = float(
                    np.max(np.abs(np.asarray(new_value, dtype=np.float64) - old_value))
                )
                if derivative_error > tolerance:
                    raise ValueError(
                        f"MPC predicted {label} handoff does not match the active "
                        f"trajectory: error={derivative_error:.9f}{unit}"
                    )
        if continuity_error > 1.0e-8:
            raise ValueError(
                f"MPC command handoff is discontinuous: error={continuity_error:.9f}rad"
            )
        self._queued = window
        self._last_generation = window.generation

    def finish_active_horizon(self) -> None:
        """Stop replenishing and settle at the active certified endpoint."""

        if self._active is None:
            raise RuntimeError("no active MPC trajectory can be finished")
        if self._queued is not None:
            raise RuntimeError("cannot stop while another MPC trajectory is queued")
        self._stop_after_active = True

    def handoff_boundary(
        self,
        *,
        now_s: float,
        minimum_lead_s: float,
        handoff_quantum_s: float,
        live_measured_q_rad: Any,
        live_active_command_q_rad: Any,
    ) -> MPCHandoffBoundary:
        """Freeze a command splice and reanchor its predicted state to live tracking."""

        if self._active is None:
            raise RuntimeError("no active MPC trajectory is available for a future handoff")
        if self._stop_after_active:
            raise RuntimeError("MPC stream is already finishing its active safe horizon")
        if self._queued is not None:
            raise RuntimeError("cannot plan another MPC handoff while one is queued")
        now = float(now_s)
        lead = float(minimum_lead_s)
        quantum = float(handoff_quantum_s)
        if not np.isfinite(lead) or lead <= 0.0:
            raise ValueError("MPC handoff lead must be positive and finite")
        if not np.isfinite(quantum) or quantum <= 0.0:
            raise ValueError("MPC handoff quantum must be positive and finite")
        live_measured = np.asarray(_q7(live_measured_q_rad, name="live measured position"))
        live_command = np.asarray(
            _q7(live_active_command_q_rad, name="live active command")
        )
        live_tracking_offset = live_command - live_measured
        if np.max(np.abs(live_tracking_offset)) > self.maximum_handoff_position_error_rad:
            raise RuntimeError(
                "MPC live tracking offset "
                f"{np.max(np.abs(live_tracking_offset)):.4f}rad exceeds "
                f"{self.maximum_handoff_position_error_rad:.4f}rad"
            )
        relative = max(now + lead - self._active.valid_from_monotonic_s, 0.0)
        handoff_offset = math.ceil((relative - 1.0e-12) / quantum) * quantum
        valid_from = self._active.valid_from_monotonic_s + handoff_offset
        predecessor_sample_s = min(valid_from, self._active.expiration_monotonic_s)
        future_command = self._active.sample_command(monotonic_s=predecessor_sample_s)
        return MPCHandoffBoundary(
            valid_from_monotonic_s=valid_from,
            command_q_rad=tuple(future_command),
            # The future command is immutable. Re-estimate the physical state
            # beneath it with the command-minus-measured offset observed now,
            # instead of carrying the previous window's stale prediction.
            predicted_q_rad=tuple(future_command - live_tracking_offset),
            predicted_dq_rad_s=tuple(
                self._active.sample_predicted_dq(monotonic_s=predecessor_sample_s)
            ),
            predicted_ddq_rad_s2=tuple(
                self._active.sample_predicted_ddq(monotonic_s=predecessor_sample_s)
            ),
            predecessor_sha256=self._active.content_sha256,
            committed_route_progress_index=self.committed_route_progress_index,
        )

    def command(
        self,
        *,
        now_s: float,
        measured_q_rad: Any,
        measured_dq_rad_s: Any,
    ) -> np.ndarray:
        now = float(now_s)
        measured_q = np.asarray(_q7(measured_q_rad, name="measured position"))
        measured_dq = np.asarray(_q7(measured_dq_rad_s, name="measured velocity"))
        if self._queued is not None and now >= self._queued.valid_from_monotonic_s:
            lateness = now - self._queued.valid_from_monotonic_s
            if lateness > self.activation_lateness_s:
                raise RuntimeError(
                    f"MPC handoff activation is {lateness:.6f}s late; "
                    f"limit is {self.activation_lateness_s:.6f}s"
                )
            expected_q = self._queued.sample_predicted_q(monotonic_s=now)
            expected_dq = self._queued.sample_predicted_dq(monotonic_s=now)
            q_error = float(np.max(np.abs(measured_q - expected_q)))
            dq_error = float(np.max(np.abs(measured_dq - expected_dq)))
            if q_error > self.maximum_handoff_position_error_rad:
                raise RuntimeError(
                    f"MPC live handoff position error {q_error:.4f}rad exceeds "
                    f"{self.maximum_handoff_position_error_rad:.4f}rad"
                )
            if dq_error > self.maximum_handoff_velocity_error_rad_s:
                raise RuntimeError(
                    f"MPC live handoff velocity error {dq_error:.4f}rad/s exceeds "
                    f"{self.maximum_handoff_velocity_error_rad_s:.4f}rad/s"
                )
            self._active = self._queued
            self._queued = None
            self._prestart_command_q = None

        if self._active is None:
            if self._prestart_command_q is None:
                raise RuntimeError("no MPC command trajectory has been installed")
            return self._prestart_command_q.copy()
        sample_time = min(
            max(now, self._active.valid_from_monotonic_s),
            self._active.expiration_monotonic_s,
        )
        return self._active.sample_command(monotonic_s=sample_time)
