"""Commissioned standing shoulder lifecycle from the earlier calibration collector.

DualArmClearanceExecutor is copied from robot-calibration-aprilcube-prototype's
2026-08-12 collector, without changing its acquisition, motion, settling, or
release methods. ShoulderClearancePlan adapts the CuRobo-certified straight
shoulder paths to that existing executor. No runtime planner or DDS owner is
created here.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from g1_aprilcube_calibration.clock import MonotonicClock
from g1_aprilcube_calibration.executor_state_machine import (
    ExecutorConfig,
    ExecutorEvent,
    ExecutorState,
)
from g1_aprilcube_calibration.gravity_compensation import ArmGravityFeedforward
from g1_aprilcube_calibration.joint_map import DUAL_ARM_DOF, G1_29_JOINT_NAMES, dual_arm_vector
from g1_aprilcube_calibration.motion_profile import velocity_limited_step
from g1_aprilcube_calibration.pose_schema import HANDOFF_POSE_ID
from g1_aprilcube_calibration.transports.base import ArmCommand, ArmTransport
from g1_dex3_tabletop.planning.contracts import (
    MAXIMUM_SHOULDER_CLEARANCE_OFFSET_RAD,
    Dex3PreparationPlan,
)

RIGHT_CLEARANCE_POSE_ID = "right_shoulder_clearance"
DUAL_CLEARANCE_POSE_ID = "dual_shoulder_clearance"
_COMMAND_COMPLETION_EPSILON_RAD = 1e-9


@dataclass(frozen=True, slots=True)
class ShoulderClearancePlan:
    preparation: Dex3PreparationPlan

    def __post_init__(self):
        # The commissioned executor traverses straight joint-space paths.
        # Never execute a CuRobo detour using that controller.
        for edge in (self.preparation.right_outbound, self.preparation.left_outbound):
            q = np.asarray(edge.command_q_rad)
            start, end = q[0], q[-1]
            if abs(end[1] - start[1]) > MAXIMUM_SHOULDER_CLEARANCE_OFFSET_RAD + 1e-9:
                raise ValueError(
                    "shoulder-clearance plan exceeds 0.14 rad; "
                    "adjust the hand position and rerun preflight"
                )
            fixed = np.arange(7) != 1
            if not np.allclose(q[:, fixed], start[fixed], atol=1e-9, rtol=0):
                raise ValueError("commissioned preparation must move only shoulder roll")
            delta = np.diff(q[:, 1]) * np.sign(end[1] - start[1])
            if np.any(delta < -1e-9):
                raise ValueError("commissioned shoulder path must be monotonic")
        if not np.allclose(
            self.target(DUAL_CLEARANCE_POSE_ID),
            self.preparation.dual_clearance_q14_rad,
            atol=1e-9,
            rtol=0,
        ):
            raise ValueError("shoulder preparation endpoint binding changed")
        for outbound, returned, direction in (
            (self.preparation.right_outbound, self.preparation.right_return, -1),
            (self.preparation.left_outbound, self.preparation.left_return, 1),
        ):
            q = np.asarray(outbound.command_q_rad)
            if direction * (q[-1, 1] - q[0, 1]) <= 0:
                raise ValueError("commissioned shoulders must move outward")
            if not np.array_equal(returned.command_q_rad, q[::-1]):
                raise ValueError("commissioned shoulder return must reverse the certified path")

    @property
    def source_q14(self):
        return (
            *self.preparation.left_outbound.command_q_rad[0],
            *self.preparation.right_outbound.command_q_rad[0],
        )

    @property
    def content_sha256(self):
        return self.preparation.content_sha256

    def target(self, pose_id):
        values = {
            HANDOFF_POSE_ID: self.source_q14,
            RIGHT_CLEARANCE_POSE_ID: (
                *self.preparation.left_outbound.command_q_rad[0],
                *self.preparation.right_outbound.command_q_rad[-1],
            ),
            DUAL_CLEARANCE_POSE_ID: (
                *self.preparation.left_outbound.command_q_rad[-1],
                *self.preparation.right_outbound.command_q_rad[-1],
            ),
        }
        if pose_id not in values:
            raise ValueError(f"unknown shoulder-clearance pose: {pose_id}")
        return np.asarray(values[pose_id], dtype=np.float64)

    def transition_is_validated(self, source, target):
        return (source, target) in {
            (HANDOFF_POSE_ID, RIGHT_CLEARANCE_POSE_ID),
            (RIGHT_CLEARANCE_POSE_ID, DUAL_CLEARANCE_POSE_ID),
            (DUAL_CLEARANCE_POSE_ID, RIGHT_CLEARANCE_POSE_ID),
            (RIGHT_CLEARANCE_POSE_ID, HANDOFF_POSE_ID),
        }


class DualArmClearanceExecutor:
    """Acquire, traverse the frozen shoulder route, and return before release."""

    def __init__(
        self,
        *,
        transport: ArmTransport,
        clock: MonotonicClock,
        plan: ShoulderClearancePlan,
        config: ExecutorConfig,
        gravity_feedforward: ArmGravityFeedforward | None = None,
    ) -> None:
        self.transport = transport
        self.clock = clock
        self.pose_set = plan
        self.plan = plan
        self.config = config
        self.gravity_feedforward = gravity_feedforward
        self.approved_validation_report_sha256 = plan.content_sha256
        self.state = ExecutorState.OBSERVING
        self.current_pose_id: str | None = None
        self.fault_reason: str | None = None
        self.events: list[ExecutorEvent] = []
        self._command_q14: np.ndarray | None = None
        self._goal_q14: np.ndarray | None = None
        self._pending_pose_id: str | None = None
        self._weight = 0.0
        self._phase_started_s: float | None = None
        self._motion_started_s: float | None = None
        self._last_tick_s: float | None = None
        self._settle_started_s: float | None = None
        self._settle_min_q: np.ndarray | None = None
        self._settle_max_q: np.ndarray | None = None
        self._fault_initial_weight = 0.0
        self._last_motion_phase: ExecutorState | None = None
        self._last_motion_elapsed_s: float | None = None
        self._last_motion_measured_q: np.ndarray | None = None
        self._last_motion_position_errors: np.ndarray | None = None
        self._last_command_remaining_rad: float | None = None
        self._last_settle_elapsed_s: float | None = None
        self._last_settle_spread_rad: float | None = None
        self._maximum_acquisition_position_change_rad = 0.0
        self._ready_reference_q14: np.ndarray | None = None

    @property
    def maximum_acquisition_position_change_rad(self) -> float:
        return self._maximum_acquisition_position_change_rad

    def acquire(self, *, operator_confirmed: bool) -> None:
        if self.state is not ExecutorState.OBSERVING:
            raise RuntimeError("clearance control can only be acquired once")
        if not operator_confirmed:
            raise ValueError("operator confirmation is required before acquisition")
        now = self.clock.monotonic()
        sample = self.transport.observe()
        self._validate_fresh_state(sample, now)
        measured = dual_arm_vector(sample.left_q, sample.right_q)
        source = np.asarray(self.plan.source_q14, dtype=np.float64)
        error = float(np.max(np.abs(measured - source)))
        if error > self.config.activation_position_tolerance_rad:
            raise ValueError(
                "live dual-arm state differs from the validated clearance source by "
                f"{error:.4f}rad; limit is "
                f"{self.config.activation_position_tolerance_rad:.4f}rad"
            )
        self._command_q14 = np.asarray(measured, dtype=np.float64).copy()
        self._goal_q14 = self._command_q14.copy()
        if self.gravity_feedforward is not None:
            self.gravity_feedforward.seed_reference(sample.position)
        self.current_pose_id = HANDOFF_POSE_ID
        self._phase_started_s = now
        self._last_tick_s = now
        self._weight = 0.0
        self._send(now)
        acquired_at = self.clock.monotonic()
        self._phase_started_s = acquired_at
        self._last_tick_s = acquired_at
        self._transition(
            ExecutorState.ACQUIRING,
            "operator confirmed measured-state clearance acquisition",
            acquired_at,
        )

    def start_pose(self, pose_id: str, *, operator_confirmed: bool) -> None:
        if self.state not in {ExecutorState.READY, ExecutorState.HOLDING}:
            raise RuntimeError("a clearance move can only start while ready")
        if not operator_confirmed:
            raise ValueError("operator confirmation is required for every move")
        if self.current_pose_id is None or not self.plan.transition_is_validated(
            self.current_pose_id, pose_id
        ):
            raise ValueError(
                f"clearance transition is not validated: {self.current_pose_id}->{pose_id}"
            )
        self._goal_q14 = self.plan.target(pose_id)
        self._pending_pose_id = pose_id
        self._ready_reference_q14 = None
        now = self.clock.monotonic()
        self._phase_started_s = now
        self._motion_started_s = now
        self._reset_settle_window()
        self._last_motion_phase = ExecutorState.MOVING
        self._last_motion_elapsed_s = 0.0
        self._last_motion_measured_q = None
        self._last_motion_position_errors = None
        self._last_command_remaining_rad = None
        self._last_settle_elapsed_s = 0.0
        self._last_settle_spread_rad = None
        self._transition(ExecutorState.MOVING, f"validated move to {pose_id}", now)

    def resume_owned_control(self) -> None:
        """Resume the unchanged full-weight clearance command after a handoff."""

        if self.state not in {ExecutorState.READY, ExecutorState.HOLDING}:
            raise RuntimeError("clearance control can only resume while held")
        if self.current_pose_id != DUAL_CLEARANCE_POSE_ID:
            raise RuntimeError("clearance control can only resume at dual clearance")
        if self._command_q14 is None or self._weight != 1.0:
            raise RuntimeError("clearance control is not held at full command weight")
        now = self.clock.monotonic()
        sample = self.transport.observe()
        self._validate_fresh_state(sample, now)
        self._ready_reference_q14 = dual_arm_vector(sample.left_q, sample.right_q)
        if self.gravity_feedforward is not None:
            self.gravity_feedforward.seed_reference(sample.position)
        self._last_tick_s = now
        self._send(now)
        sent_at = self.clock.monotonic()
        self._last_tick_s = sent_at
        self._transition(
            self.state,
            "resumed identical full-weight command after calibration route",
            sent_at,
        )

    def tick(self) -> ExecutorState:
        now = self.clock.monotonic()
        if self.state in {ExecutorState.STOPPED, ExecutorState.OBSERVING}:
            return self.state
        if self.state is ExecutorState.FAULT:
            self._tick_release(now, emergency=True)
            return self.state
        if self._last_tick_s is None:
            self._enter_fault("control loop has no previous tick", now)
            return self.state
        duration = now - self._last_tick_s
        if duration < 0.0:
            self._enter_fault("monotonic clock moved backwards", now)
            return self.state
        if duration > self.config.control_gap_fault_s:
            self._enter_fault(
                f"control loop gap {duration:.3f}s exceeds "
                f"hard limit {self.config.control_gap_fault_s:.3f}s",
                now,
            )
            return self.state
        self._last_tick_s = now
        try:
            sample = self.transport.observe()
            self._validate_fresh_state(sample, now)
        except (TypeError, ValueError, RuntimeError) as error:
            self._enter_fault(str(error), now)
            return self.state
        measured = dual_arm_vector(sample.left_q, sample.right_q)

        if self.state in {ExecutorState.READY, ExecutorState.HOLDING}:
            if self._ready_reference_q14 is None:
                self._enter_fault("clearance hold has no measured reference", now)
                return self.state
            drift = float(np.max(np.abs(measured - self._ready_reference_q14)))
            if drift > self.config.held_arm_position_tolerance_rad:
                self._enter_fault(
                    "clearance-held arms drifted by "
                    f"{drift:.4f}rad; limit is "
                    f"{self.config.held_arm_position_tolerance_rad:.4f}rad",
                    now,
                )
                return self.state

        if self.state is ExecutorState.ACQUIRING:
            assert self._command_q14 is not None
            acquisition_error = float(np.max(np.abs(measured - self._command_q14)))
            self._maximum_acquisition_position_change_rad = max(
                self._maximum_acquisition_position_change_rad, acquisition_error
            )
            if acquisition_error > self.config.ownership_transition_position_tolerance_rad:
                self._enter_fault(
                    "arm position changed by "
                    f"{acquisition_error:.4f}rad during ownership acquisition; "
                    "limit is "
                    f"{self.config.ownership_transition_position_tolerance_rad:.4f}rad",
                    now,
                )
                return self.state
            assert self._phase_started_s is not None
            elapsed = now - self._phase_started_s
            self._weight = min(elapsed / self.config.acquisition_ramp_s, 1.0)
            self._send(now)
            if self._weight >= 1.0:
                self._ready_reference_q14 = np.asarray(measured).copy()
                self._transition(
                    ExecutorState.READY,
                    "clearance acquisition ramp complete",
                    now,
                )
            return self.state

        if self.state is ExecutorState.RELEASING:
            self._tick_release(now, emergency=False)
            return self.state
        if self.state in {ExecutorState.HOLDING, ExecutorState.READY}:
            self._send(now)
            return self.state

        assert self._command_q14 is not None and self._goal_q14 is not None
        self._command_q14 = velocity_limited_step(
            self._command_q14,
            self._goal_q14,
            maximum_velocity_rad_s=self.config.maximum_joint_velocity_rad_s,
            duration_s=self.config.nominal_tick_period_s,
        )
        self._send(now)
        assert self._motion_started_s is not None
        position_errors = np.abs(np.asarray(measured) - self._goal_q14)
        position_error = float(np.max(position_errors))
        self._last_motion_elapsed_s = now - self._motion_started_s
        self._last_motion_measured_q = np.asarray(measured).copy()
        self._last_motion_position_errors = position_errors.copy()
        self._last_command_remaining_rad = float(
            np.max(np.abs(self._command_q14 - self._goal_q14))
        )
        if self.state is ExecutorState.MOVING:
            if self._last_command_remaining_rad <= _COMMAND_COMPLETION_EPSILON_RAD:
                self._reset_settle_window()
                self._transition(
                    ExecutorState.SETTLING,
                    "clearance command complete; waiting for measured settling",
                    now,
                )
        elif self._last_command_remaining_rad > _COMMAND_COMPLETION_EPSILON_RAD:
            self._reset_settle_window()
            self._transition(ExecutorState.MOVING, "command became incomplete", now)
        elif self._settle_started_s is None:
            self._settle_started_s = now
            self._settle_min_q = np.asarray(measured).copy()
            self._settle_max_q = np.asarray(measured).copy()
            self._last_settle_elapsed_s = 0.0
            self._last_settle_spread_rad = 0.0
        else:
            assert self._settle_min_q is not None and self._settle_max_q is not None
            self._settle_min_q = np.minimum(self._settle_min_q, measured)
            self._settle_max_q = np.maximum(self._settle_max_q, measured)
            maximum_spread = float(np.max(self._settle_max_q - self._settle_min_q))
            self._last_settle_elapsed_s = now - self._settle_started_s
            self._last_settle_spread_rad = maximum_spread
            if maximum_spread > self.config.settled_position_spread_rad:
                self._settle_started_s = now
                self._settle_min_q = np.asarray(measured).copy()
                self._settle_max_q = np.asarray(measured).copy()
                self._last_settle_elapsed_s = 0.0
            elif now - self._settle_started_s >= self.config.settle_dwell_s:
                if (
                    self.config.require_motion_endpoint_tolerance
                    and position_error > self.config.motion_position_tolerance_rad
                ):
                    self._enter_fault(
                        self.motion_diagnostic(
                            prefix="clearance motion settled outside endpoint tolerance"
                        ),
                        now,
                    )
                else:
                    self.current_pose_id = self._pending_pose_id
                    self._pending_pose_id = None
                    self._ready_reference_q14 = np.asarray(measured).copy()
                    self._transition(
                        ExecutorState.READY,
                        f"clearance settle passed; endpoint error {position_error:.4f}rad",
                        now,
                    )
        if self.state in {ExecutorState.MOVING, ExecutorState.SETTLING}:
            self._last_motion_phase = self.state
            if self._last_motion_elapsed_s > self.config.motion_timeout_s:
                self._enter_fault(self.motion_diagnostic(prefix="clearance motion timed out"), now)
        return self.state

    def motion_diagnostic(self, *, prefix: str = "motion status") -> str:
        if (
            self._pending_pose_id is None
            or self._last_motion_phase is None
            or self._last_motion_elapsed_s is None
            or self._last_motion_measured_q is None
            or self._last_motion_position_errors is None
            or self._last_command_remaining_rad is None
        ):
            return f"{prefix}: no active measured-motion diagnostic"
        worst = int(np.argmax(self._last_motion_position_errors))
        target = self.plan.target(self._pending_pose_id)
        spread = (
            "n/a"
            if self._last_settle_spread_rad is None
            else f"{self._last_settle_spread_rad:.4f}rad"
        )
        return (
            f"{prefix} after {self._last_motion_elapsed_s:.2f}s while "
            f"{self._last_motion_phase.value} for {self._pending_pose_id}: "
            f"{G1_29_JOINT_NAMES[15 + worst]} has maximum position error "
            f"{self._last_motion_position_errors[worst]:.4f}rad "
            f"(measured={self._last_motion_measured_q[worst]:.4f}, "
            f"target={target[worst]:.4f}, "
            f"limit={self.config.motion_position_tolerance_rad:.4f}rad); "
            f"command remaining={self._last_command_remaining_rad:.4f}rad; "
            f"settle window={self._last_settle_elapsed_s or 0.0:.2f}/"
            f"{self.config.settle_dwell_s:.2f}s, position spread={spread}"
        )

    def begin_clean_release(self, *, operator_confirmed: bool) -> None:
        if self.state not in {ExecutorState.READY, ExecutorState.HOLDING}:
            raise RuntimeError("clean release requires a settled clearance executor")
        if not operator_confirmed:
            raise ValueError("operator confirmation is required for clean release")
        if self.current_pose_id != HANDOFF_POSE_ID:
            raise ValueError("arms must return to the validated source before release")
        now = self.clock.monotonic()
        self._phase_started_s = now
        self._transition(ExecutorState.RELEASING, "clean release approved", now)

    def observe_state(self):
        """Return one transport observation under the synchronized wrapper lock."""

        return self.transport.observe()

    def emergency_stop(self, reason: str) -> None:
        if self.state in {ExecutorState.STOPPED, ExecutorState.OBSERVING}:
            if self.state is ExecutorState.OBSERVING:
                self.transport.close()
                self._transition(ExecutorState.STOPPED, reason, self.clock.monotonic())
            return
        self._enter_fault(reason, self.clock.monotonic())

    def confirm_external_damping(self, reason: str) -> None:
        if self.state is ExecutorState.STOPPED:
            return
        if not reason.strip():
            raise ValueError("external damping reason must be non-empty")
        self.fault_reason = reason.strip()
        self.transport.close_after_external_takeover()
        self._transition(
            ExecutorState.STOPPED,
            f"external damping confirmed: {reason.strip()}",
            self.clock.monotonic(),
        )

    def _validate_fresh_state(self, sample, now: float) -> None:
        if not sample.is_mode5:
            raise ValueError("robot state is not mode_machine=5")
        age = sample.age_s(now)
        if age > self.config.state_freshness_timeout_s:
            raise ValueError(
                f"robot state age {age:.3f}s exceeds {self.config.state_freshness_timeout_s:.3f}s"
            )

    def _reset_settle_window(self) -> None:
        self._settle_started_s = None
        self._settle_min_q = None
        self._settle_max_q = None
        self._last_settle_elapsed_s = 0.0
        self._last_settle_spread_rad = None

    def _send(self, now: float, *, emergency: bool = False) -> None:
        if self._command_q14 is None:
            raise RuntimeError("cannot command before measured-state seeding")
        torque = (
            np.zeros(DUAL_ARM_DOF, dtype=np.float64)
            if self.gravity_feedforward is None
            else self.gravity_feedforward.torque_for(self._command_q14)
        )
        self.transport.send_command(
            ArmCommand.create(
                self._command_q14,
                weight=self._weight,
                issued_monotonic_s=now,
                emergency_release=emergency,
                tau_ff14=torque,
            )
        )

    def _tick_release(self, now: float, *, emergency: bool) -> None:
        assert self._phase_started_s is not None
        elapsed = max(now - self._phase_started_s, 0.0)
        initial = self._fault_initial_weight if emergency else 1.0
        self._weight = initial * max(1.0 - elapsed / self.config.release_ramp_s, 0.0)
        self._send(now, emergency=emergency)
        if self._weight <= 0.0:
            self.transport.close()
            self._transition(
                ExecutorState.STOPPED,
                "emergency weight reached zero" if emergency else "clean release complete",
                now,
            )

    def _enter_fault(self, reason: str, now: float) -> None:
        if self.state in {ExecutorState.FAULT, ExecutorState.STOPPED}:
            return
        self.fault_reason = reason
        self._fault_initial_weight = self._weight
        self._phase_started_s = now
        self._transition(ExecutorState.FAULT, reason, now)
        self._send(now, emergency=True)

    def _transition(self, state: ExecutorState, reason: str, now: float) -> None:
        previous = self.state
        self.state = state
        self.events.append(
            ExecutorEvent(
                sequence=len(self.events),
                occurred_monotonic_s=now,
                previous_state=previous,
                state=state,
                reason=reason,
            )
        )
