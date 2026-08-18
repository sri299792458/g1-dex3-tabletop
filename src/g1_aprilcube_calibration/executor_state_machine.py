"""Safety-first deterministic G1 pose execution state machine."""

from __future__ import annotations

import hashlib
import re
from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum

import numpy as np

from g1_aprilcube_calibration.clock import MonotonicClock
from g1_aprilcube_calibration.gravity_compensation import ArmGravityFeedforward
from g1_aprilcube_calibration.joint_map import (
    arm_joint_names,
    dual_arm_vector,
    opposite_arm,
    validate_arm_vector,
)
from g1_aprilcube_calibration.models import RobotStateSample
from g1_aprilcube_calibration.motion_profile import velocity_limited_step
from g1_aprilcube_calibration.opposite_arm_hold import OppositeArmHold
from g1_aprilcube_calibration.pose_schema import HANDOFF_POSE_ID, PoseSet
from g1_aprilcube_calibration.transports.base import ArmCommand, ArmTransport
from g1_dex3_tabletop.mpc_command_buffer import (
    MPCCommandWindow,
    RollingMPCCommandBuffer,
)

_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_COMMAND_COMPLETION_EPSILON_RAD = 1e-9


class ExecutorState(str, Enum):
    OBSERVING = "observing"
    ACQUIRING = "acquiring"
    HOLDING = "holding"
    MOVING = "moving"
    SETTLING = "settling"
    READY = "ready"
    CAPTURING = "capturing"
    RELEASING = "releasing"
    FAULT = "fault"
    STOPPED = "stopped"


@dataclass(frozen=True, slots=True)
class ExecutorConfig:
    maximum_joint_velocity_rad_s: float = 0.2
    motion_position_tolerance_rad: float = 0.08
    require_motion_endpoint_tolerance: bool = True
    ownership_transition_position_tolerance_rad: float = 0.05
    activation_position_tolerance_rad: float = 0.02
    held_arm_position_tolerance_rad: float = 0.02
    settled_position_spread_rad: float = 0.01
    settle_dwell_s: float = 0.5
    state_freshness_timeout_s: float = 0.1
    nominal_tick_period_s: float = 0.004
    control_gap_fault_s: float = 0.25
    acquisition_ramp_s: float = 1.0
    release_ramp_s: float = 1.0
    motion_timeout_s: float = 30.0

    def __post_init__(self) -> None:
        if not isinstance(self.require_motion_endpoint_tolerance, bool):
            raise TypeError("require_motion_endpoint_tolerance must be boolean")
        for name in self.__dataclass_fields__:
            if name == "require_motion_endpoint_tolerance":
                continue
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.nominal_tick_period_s >= self.control_gap_fault_s:
            raise ValueError("nominal tick period must be below the fault limit")


@dataclass(frozen=True, slots=True)
class TransitionApproval:
    from_pose_id: str
    to_pose_id: str
    pose_set_sha256: str
    validation_report_sha256: str
    passed: bool

    def __post_init__(self) -> None:
        if not self.from_pose_id or not self.to_pose_id:
            raise ValueError("transition pose IDs must be non-empty")
        if not _SHA256_PATTERN.fullmatch(self.pose_set_sha256):
            raise ValueError("pose_set_sha256 must be lowercase SHA-256")
        if not _SHA256_PATTERN.fullmatch(self.validation_report_sha256):
            raise ValueError("validation_report_sha256 must be lowercase SHA-256")


@dataclass(frozen=True, slots=True)
class ExecutorEvent:
    sequence: int
    occurred_monotonic_s: float
    previous_state: ExecutorState
    state: ExecutorState
    reason: str


class PoseExecutor:
    def __init__(
        self,
        *,
        transport: ArmTransport,
        clock: MonotonicClock,
        pose_set: PoseSet,
        handoff_q: Sequence[float] | np.ndarray,
        hold_q: Sequence[float] | np.ndarray,
        approved_validation_report_sha256: str,
        config: ExecutorConfig | None = None,
        gravity_feedforward: ArmGravityFeedforward | None = None,
    ) -> None:
        if not _SHA256_PATTERN.fullmatch(approved_validation_report_sha256):
            raise ValueError("approved validation report hash must be lowercase SHA-256")
        self.transport = transport
        self.clock = clock
        self.pose_set = pose_set
        self.handoff_q = validate_arm_vector(handoff_q, side=pose_set.calibration_arm)
        self.hold_q = validate_arm_vector(hold_q, side=opposite_arm(pose_set.calibration_arm))
        self.approved_validation_report_sha256 = approved_validation_report_sha256
        self.config = config or ExecutorConfig()
        self.gravity_feedforward = gravity_feedforward
        self.state = ExecutorState.OBSERVING
        self.events: list[ExecutorEvent] = []
        self.fault_reason: str | None = None
        self.current_pose_id: str | None = None
        self._pending_pose_id: str | None = None
        self._goal_q14: np.ndarray | None = None
        self._command_q14: np.ndarray | None = None
        self._opposite_hold = OppositeArmHold(
            calibration_arm=pose_set.calibration_arm,
            command_q=self.hold_q,
        )
        self._calibration_goal_q: np.ndarray | None = None
        self._weight = 0.0
        self._phase_started_s: float | None = None
        self._motion_started_s: float | None = None
        self._settle_started_s: float | None = None
        self._settle_min_q: np.ndarray | None = None
        self._settle_max_q: np.ndarray | None = None
        self._last_tick_s: float | None = None
        self._fault_initial_weight = 0.0
        self._last_motion_phase: ExecutorState | None = None
        self._last_motion_elapsed_s: float | None = None
        self._last_motion_measured_q: np.ndarray | None = None
        self._last_motion_position_errors: np.ndarray | None = None
        self._last_command_remaining_rad: float | None = None
        self._last_settle_elapsed_s: float | None = None
        self._last_settle_spread_rad: float | None = None
        self._maximum_acquisition_position_change_rad = 0.0
        self._trajectory_time_s: np.ndarray | None = None
        self._trajectory_command_q: np.ndarray | None = None
        self._mpc_command_buffer: RollingMPCCommandBuffer | None = None
        self._active_motion_timeout_s = self.config.motion_timeout_s

    @property
    def maximum_acquisition_position_change_rad(self) -> float:
        return self._maximum_acquisition_position_change_rad

    @property
    def calibration_command_q(self) -> np.ndarray:
        """Return the exact calibration-arm command active in the control loop."""

        if self._command_q14 is None:
            raise RuntimeError("executor has no acquired arm command")
        if self.pose_set.calibration_arm == "left":
            return self._command_q14[:7].copy()
        return self._command_q14[7:].copy()

    def observe_control_input(self) -> tuple[RobotStateSample, np.ndarray]:
        """Return one fresh measurement paired with the exact active command."""

        return self.observe_state(), self.calibration_command_q

    def streaming_trajectory_status(self) -> dict[str, float | int | bool]:
        if self._mpc_command_buffer is None:
            raise RuntimeError("no streaming MPC trajectory is active")
        now = self.clock.monotonic()
        return {
            "generation": self._mpc_command_buffer.last_generation,
            "terminal": self._mpc_command_buffer.terminal,
            "remaining_s": self._mpc_command_buffer.remaining_s(now_s=now),
            "duration_s": self._mpc_command_buffer.duration_s,
        }

    def acquire(
        self,
        *,
        operator_confirmed: bool,
    ) -> None:
        if self.state is not ExecutorState.OBSERVING:
            raise RuntimeError("control can only be acquired from observing")
        if not operator_confirmed:
            raise ValueError("operator confirmation is required before acquisition")
        now = self.clock.monotonic()
        sample = self.transport.observe()
        self._validate_fresh_state(sample, now)
        hold_arm = opposite_arm(self.pose_set.calibration_arm)
        hold_error = float(np.max(np.abs(sample.arm_q(hold_arm) - self.hold_q)))
        if hold_error > self.config.activation_position_tolerance_rad:
            raise ValueError(
                f"measured {hold_arm} arm differs from pose-set hold by {hold_error:.4f}rad"
            )
        calibration_error = float(
            np.max(np.abs(sample.arm_q(self.pose_set.calibration_arm) - self.handoff_q))
        )
        if calibration_error > self.config.activation_position_tolerance_rad:
            raise ValueError(
                f"measured {self.pose_set.calibration_arm} arm differs from "
                f"handoff pose by {calibration_error:.4f}rad"
            )
        # Command the zero-displacement ownership sample, not the older
        # stationary-window median. Finite gains can create a small loaded
        # offset while ownership and optional gravity feedforward ramp up.
        # Keep the command fixed, but monitor steady drift from the measured
        # full-ownership equilibrium established below.
        self._opposite_hold.seed_from_sample(sample)
        self._calibration_goal_q = self.handoff_q.copy()
        self._command_q14 = dual_arm_vector(sample.left_q, sample.right_q)
        self._goal_q14 = self._command_q14
        if self.gravity_feedforward is not None:
            self.gravity_feedforward.seed_reference(sample.position)
        self.current_pose_id = HANDOFF_POSE_ID
        self._phase_started_s = now
        self._last_tick_s = now
        self._weight = 0.0
        self._send(now)
        # A direct-control transport may perform a bounded MotionSwitcher RPC
        # before its first packet. Start fixed-rate gap accounting only after
        # that ownership transition has completed.
        acquired_at = self.clock.monotonic()
        self._phase_started_s = acquired_at
        self._last_tick_s = acquired_at
        self._transition(
            ExecutorState.ACQUIRING,
            "operator confirmed acquisition",
            acquired_at,
        )

    def adopt_owned_control(
        self,
        *,
        previous_command_q14: Sequence[float] | np.ndarray,
    ) -> None:
        """Continue an identical full-weight command from another executor."""

        if self.state is not ExecutorState.OBSERVING:
            raise RuntimeError("owned control can only be adopted from observing")
        previous = np.asarray(previous_command_q14, dtype=np.float64).reshape(-1)
        if previous.shape != (14,) or not np.all(np.isfinite(previous)):
            raise ValueError("previous owned command must contain 14 finite values")
        expected = self._opposite_hold.compose(self.handoff_q)
        command_change = float(np.max(np.abs(previous - expected)))
        if command_change > _COMMAND_COMPLETION_EPSILON_RAD:
            raise ValueError(
                "owned-control handoff command differs from the authored handoff "
                f"by {command_change:.6f}rad"
            )
        now = self.clock.monotonic()
        sample = self.transport.observe()
        self._validate_fresh_state(sample, now)
        self._opposite_hold.rebase_monitor(sample)
        self._calibration_goal_q = self.handoff_q.copy()
        self._command_q14 = previous.copy()
        self._goal_q14 = previous.copy()
        if self.gravity_feedforward is not None:
            self.gravity_feedforward.seed_reference(sample.position)
        self.current_pose_id = HANDOFF_POSE_ID
        self._weight = 1.0
        self._phase_started_s = now
        self._last_tick_s = now
        self._send(now)
        sent_at = self.clock.monotonic()
        self._phase_started_s = sent_at
        self._last_tick_s = sent_at
        self._transition(
            ExecutorState.READY,
            "continued identical full-weight command from clearance executor",
            sent_at,
        )

    def start_pose(
        self,
        pose_id: str,
        *,
        approval: TransitionApproval,
        operator_confirmed: bool,
    ) -> None:
        if self.state not in {ExecutorState.HOLDING, ExecutorState.READY}:
            raise RuntimeError("a move can only start while holding or ready")
        if not operator_confirmed:
            raise ValueError("operator confirmation is required for every move")
        if self.current_pose_id is None or approval.from_pose_id != self.current_pose_id:
            raise ValueError("transition approval does not match the current pose")
        if approval.to_pose_id != pose_id:
            raise ValueError("transition approval does not match the requested pose")
        if approval.pose_set_sha256 != self.pose_set.content_sha256:
            raise ValueError("transition approval pose-set hash is stale")
        if approval.validation_report_sha256 != self.approved_validation_report_sha256:
            raise ValueError("transition approval validation-report hash is stale")
        if not approval.passed:
            raise ValueError("transition validation did not pass")
        if pose_id == HANDOFF_POSE_ID:
            calibration_target = self.handoff_q
        else:
            matching = [pose for pose in self.pose_set.poses if pose.id == pose_id]
            if len(matching) != 1:
                raise ValueError(f"pose ID is not present exactly once: {pose_id}")
            calibration_target = matching[0].command_calibration_q
        if self._command_q14 is None:
            raise RuntimeError("executor has no acquired arm state")
        calibration_goal = validate_arm_vector(
            calibration_target,
            side=self.pose_set.calibration_arm,
        )
        self._calibration_goal_q = calibration_goal
        self._goal_q14 = self._compose_command(calibration_goal)
        self._pending_pose_id = pose_id
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
        self._transition(ExecutorState.MOVING, f"approved move to {pose_id}", now)

    def start_trajectory(
        self,
        *,
        from_pose_id: str,
        to_pose_id: str,
        sample_time_s: Sequence[float] | np.ndarray,
        command_q_rad: Sequence[Sequence[float]] | np.ndarray,
        plan_sha256: str,
        operator_confirmed: bool,
    ) -> None:
        """Follow one immutable, externally collision-validated arm trajectory.

        CuRobo owns geometric planning and time parameterization. This method
        only interpolates its frozen samples inside the commissioned fixed-rate
        Unitree ownership, gravity, freshness, hold, and settling machinery.
        """

        if self.state not in {ExecutorState.HOLDING, ExecutorState.READY}:
            raise RuntimeError("a trajectory can only start while holding or ready")
        if not operator_confirmed:
            raise ValueError("operator confirmation is required for every trajectory")
        if self.current_pose_id != from_pose_id:
            raise ValueError(
                f"trajectory source {from_pose_id!r} does not match the current pose "
                f"{self.current_pose_id!r}"
            )
        if plan_sha256 != self.approved_validation_report_sha256:
            raise ValueError("trajectory belongs to a different approved plan")
        if to_pose_id != HANDOFF_POSE_ID and not any(
            pose.id == to_pose_id for pose in self.pose_set.poses
        ):
            raise ValueError(f"trajectory target is not in the installed plan: {to_pose_id}")
        times = np.asarray(sample_time_s, dtype=np.float64).reshape(-1)
        command = np.asarray(command_q_rad, dtype=np.float64)
        if (
            len(times) < 2
            or times[0] != 0.0
            or not np.all(np.isfinite(times))
            or not np.all(np.diff(times) > 0.0)
        ):
            raise ValueError("trajectory timestamps must start at zero and increase")
        if command.shape != (len(times), 7) or not np.all(np.isfinite(command)):
            raise ValueError("trajectory commands must be a finite N x 7 array")
        velocity = np.max(np.abs(np.diff(command, axis=0)) / np.diff(times)[:, None])
        if velocity > self.config.maximum_joint_velocity_rad_s + 1e-6:
            raise ValueError(
                f"trajectory velocity {velocity:.4f}rad/s exceeds controller limit "
                f"{self.config.maximum_joint_velocity_rad_s:.4f}rad/s"
            )
        if self._command_q14 is None:
            raise RuntimeError("executor has no acquired arm state")
        current_command = (
            self._command_q14[:7]
            if self.pose_set.calibration_arm == "left"
            else self._command_q14[7:]
        )
        start_error = float(np.max(np.abs(command[0] - current_command)))
        if start_error > _COMMAND_COMPLETION_EPSILON_RAD:
            raise ValueError(
                f"trajectory start differs from the current command by {start_error:.9f}rad"
            )

        self._trajectory_time_s = times.copy()
        self._trajectory_command_q = command.copy()
        self._calibration_goal_q = command[-1].copy()
        self._goal_q14 = self._compose_command(self._calibration_goal_q)
        self._pending_pose_id = to_pose_id
        now = self.clock.monotonic()
        self._phase_started_s = now
        self._motion_started_s = now
        self._active_motion_timeout_s = max(
            self.config.motion_timeout_s,
            float(times[-1]) + self.config.settle_dwell_s + 1.0,
        )
        self._reset_settle_window()
        self._last_motion_phase = ExecutorState.MOVING
        self._last_motion_elapsed_s = 0.0
        self._last_motion_measured_q = None
        self._last_motion_position_errors = None
        self._last_command_remaining_rad = None
        self._last_settle_elapsed_s = 0.0
        self._last_settle_spread_rad = None
        self._transition(
            ExecutorState.MOVING,
            f"approved frozen trajectory {from_pose_id}->{to_pose_id}",
            now,
        )

    def start_streaming_trajectory(
        self,
        *,
        from_pose_id: str,
        to_pose_id: str,
        window: MPCCommandWindow,
        plan_sha256: str,
        operator_confirmed: bool,
    ) -> MPCCommandWindow:
        """Start one continuously replenished, validated MPC arm trajectory."""

        if self.state not in {ExecutorState.HOLDING, ExecutorState.READY}:
            raise RuntimeError("a streaming trajectory can only start while holding or ready")
        if not operator_confirmed:
            raise ValueError("operator confirmation is required for every trajectory")
        if self.current_pose_id != from_pose_id:
            raise ValueError(
                f"streaming trajectory source {from_pose_id!r} does not match the "
                f"current pose {self.current_pose_id!r}"
            )
        if plan_sha256 != self.approved_validation_report_sha256:
            raise ValueError("streaming trajectory belongs to a different approved plan")
        if to_pose_id != HANDOFF_POSE_ID and not any(
            pose.id == to_pose_id for pose in self.pose_set.poses
        ):
            raise ValueError(
                f"streaming trajectory target is not in the installed plan: {to_pose_id}"
            )
        if self._command_q14 is None:
            raise RuntimeError("executor has no acquired arm state")

        now = self.clock.monotonic()
        buffer = RollingMPCCommandBuffer(
            plan_sha256=plan_sha256,
            maximum_velocity_rad_s=self.config.maximum_joint_velocity_rad_s,
            maximum_state_age_s=self.config.state_freshness_timeout_s,
            maximum_window_gap_s=self.config.control_gap_fault_s,
        )
        accepted = window.rebase_start(self.calibration_command_q)
        buffer.accept(
            accepted,
            now_s=now,
            active_command_q_rad=self.calibration_command_q,
        )
        self._mpc_command_buffer = buffer
        self._calibration_goal_q = np.asarray(accepted.command_q_rad[-1], dtype=np.float64)
        self._goal_q14 = self._compose_command(self._calibration_goal_q)
        self._pending_pose_id = to_pose_id
        self._phase_started_s = now
        self._motion_started_s = now
        self._active_motion_timeout_s = self.config.motion_timeout_s
        self._reset_settle_window()
        self._last_motion_phase = ExecutorState.MOVING
        self._last_motion_elapsed_s = 0.0
        self._last_motion_measured_q = None
        self._last_motion_position_errors = None
        self._last_command_remaining_rad = None
        self._last_settle_elapsed_s = 0.0
        self._last_settle_spread_rad = None
        self._transition(
            ExecutorState.MOVING,
            f"approved streaming MPC trajectory {from_pose_id}->{to_pose_id}",
            now,
        )
        return accepted

    def update_streaming_trajectory(
        self,
        *,
        window: MPCCommandWindow,
    ) -> MPCCommandWindow:
        """Atomically install the next MPC window at the current command."""

        if self.state is not ExecutorState.MOVING or self._mpc_command_buffer is None:
            raise RuntimeError("no streaming MPC trajectory is active")
        now = self.clock.monotonic()
        accepted = window.rebase_start(self.calibration_command_q)
        self._mpc_command_buffer.accept(
            accepted,
            now_s=now,
            active_command_q_rad=self.calibration_command_q,
        )
        self._calibration_goal_q = np.asarray(accepted.command_q_rad[-1], dtype=np.float64)
        self._goal_q14 = self._compose_command(self._calibration_goal_q)
        return accepted

    def install_validated_plan(
        self,
        *,
        pose_set: PoseSet,
        approved_validation_report_sha256: str,
        validated_reference_state: RobotStateSample,
    ) -> None:
        """Atomically install a plan validated at the loaded handoff state.

        This is intentionally narrower than a general runtime replan.  It is
        only legal before the first move, while holding the handoff pose at
        full command weight.  The newly validated handoff becomes the command
        origin, so the first changing target follows the exact pose set and
        edge report that were produced after ownership acquisition.
        """

        if self.state not in {ExecutorState.READY, ExecutorState.HOLDING}:
            raise RuntimeError("a validated plan can only be installed while ready or holding")
        if self.current_pose_id != HANDOFF_POSE_ID or self._pending_pose_id is not None:
            raise RuntimeError(
                "a validated plan can only be installed before the first handoff move"
            )
        if self._command_q14 is None or self._weight != 1.0:
            raise RuntimeError("a validated plan requires acquired control at full command weight")
        if not _SHA256_PATTERN.fullmatch(approved_validation_report_sha256):
            raise ValueError("approved validation report hash must be lowercase SHA-256")
        if pose_set.robot_model != self.pose_set.robot_model:
            raise ValueError("replacement pose set belongs to a different robot model")
        if pose_set.mode_machine != self.pose_set.mode_machine:
            raise ValueError("replacement pose set uses a different mode machine")
        if pose_set.urdf_sha256 != self.pose_set.urdf_sha256:
            raise ValueError("replacement pose set belongs to a different URDF")
        if pose_set.calibration_arm != self.pose_set.calibration_arm:
            raise ValueError("replacement pose set uses a different calibration arm")
        if not validated_reference_state.is_mode5:
            raise ValueError("validated plan reference is not mode_machine=5")

        now = self.clock.monotonic()
        live_state = self.transport.observe()
        self._validate_fresh_state(live_state, now)
        reference_drift = float(
            np.max(np.abs(live_state.position - validated_reference_state.position))
        )
        if reference_drift > self.config.settled_position_spread_rad:
            raise ValueError(
                "loaded state changed after path validation by "
                f"{reference_drift:.4f}rad; limit is "
                f"{self.config.settled_position_spread_rad:.4f}rad"
            )

        command_q14 = dual_arm_vector(
            validated_reference_state.left_q,
            validated_reference_state.right_q,
        )
        command_change = float(np.max(np.abs(command_q14 - self._command_q14)))
        if command_change > self.config.ownership_transition_position_tolerance_rad:
            raise ValueError(
                "loaded handoff command rebase is "
                f"{command_change:.4f}rad; limit is "
                f"{self.config.ownership_transition_position_tolerance_rad:.4f}rad"
            )

        self.pose_set = pose_set
        self.approved_validation_report_sha256 = approved_validation_report_sha256
        self.handoff_q = validated_reference_state.arm_q(pose_set.calibration_arm).copy()
        hold_arm = opposite_arm(pose_set.calibration_arm)
        self.hold_q = validated_reference_state.arm_q(hold_arm).copy()
        self._opposite_hold = OppositeArmHold(
            calibration_arm=pose_set.calibration_arm,
            command_q=self.hold_q,
        )
        self._opposite_hold.rebase_monitor(live_state)
        self._calibration_goal_q = self.handoff_q.copy()
        self._command_q14 = command_q14
        self._goal_q14 = command_q14.copy()
        self._reset_settle_window()
        if self.gravity_feedforward is not None:
            self.gravity_feedforward.seed_reference(validated_reference_state.position)
        self._send(now)
        self._transition(
            self.state,
            "post-acquisition validated plan installed at loaded handoff; "
            f"live reference drift {reference_drift:.4f}rad; "
            f"command rebase {command_change:.4f}rad",
            now,
        )

    def replace_validated_remaining_plan(
        self,
        *,
        pose_set: PoseSet,
        approved_validation_report_sha256: str,
        validated_reference_state: RobotStateSample,
    ) -> None:
        """Atomically replace future routes at an already-reached plan boundary.

        Unlike loaded-handoff installation, this does not rebase ownership,
        body commands, the opposite-arm hold, or gravity feedforward. The new
        plan must contain the current named boundary at the exact active arm
        command, and the live complete state must still match the state used by
        the planner.
        """

        if self.state not in {ExecutorState.READY, ExecutorState.HOLDING}:
            raise RuntimeError("a remaining plan can only be replaced while ready or holding")
        if self.current_pose_id in {None, HANDOFF_POSE_ID} or self._pending_pose_id is not None:
            raise RuntimeError("remaining-plan replacement requires a reached non-handoff pose")
        if self._command_q14 is None or self._weight != 1.0:
            raise RuntimeError("remaining-plan replacement requires full command ownership")
        if not _SHA256_PATTERN.fullmatch(approved_validation_report_sha256):
            raise ValueError("approved validation report hash must be lowercase SHA-256")
        if pose_set.robot_model != self.pose_set.robot_model:
            raise ValueError("replacement pose set belongs to a different robot model")
        if pose_set.mode_machine != self.pose_set.mode_machine:
            raise ValueError("replacement pose set uses a different mode machine")
        if pose_set.urdf_sha256 != self.pose_set.urdf_sha256:
            raise ValueError("replacement pose set belongs to a different URDF")
        if pose_set.calibration_arm != self.pose_set.calibration_arm:
            raise ValueError("replacement pose set uses a different calibration arm")
        if not validated_reference_state.is_mode5:
            raise ValueError("validated plan reference is not mode_machine=5")

        matching = [pose for pose in pose_set.poses if pose.id == self.current_pose_id]
        if len(matching) != 1:
            raise ValueError("replacement plan does not contain the current boundary exactly once")
        current_command = self.calibration_command_q
        boundary_error = float(
            np.max(np.abs(np.asarray(matching[0].command_calibration_q) - current_command))
        )
        if boundary_error > _COMMAND_COMPLETION_EPSILON_RAD:
            raise ValueError(
                f"replacement boundary differs from the current command by {boundary_error:.9f}rad"
            )

        now = self.clock.monotonic()
        live_state = self.transport.observe()
        self._validate_fresh_state(live_state, now)
        reference_drift = float(
            np.max(np.abs(live_state.position - validated_reference_state.position))
        )
        if reference_drift > self.config.settled_position_spread_rad:
            raise ValueError(
                "boundary state changed after path validation by "
                f"{reference_drift:.4f}rad; limit is "
                f"{self.config.settled_position_spread_rad:.4f}rad"
            )

        self.pose_set = pose_set
        self.approved_validation_report_sha256 = approved_validation_report_sha256
        self._calibration_goal_q = current_command.copy()
        self._goal_q14 = self._command_q14.copy()
        self._reset_settle_window()
        self._send(now)
        self._transition(
            self.state,
            "boundary-corrected remaining plan installed at "
            f"{self.current_pose_id}; live reference drift {reference_drift:.4f}rad; "
            f"command continuity {boundary_error:.9f}rad",
            now,
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
        if duration < 0:
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
            ownership_transition = self.state in {
                ExecutorState.ACQUIRING,
                ExecutorState.RELEASING,
            }
            self._opposite_hold.validate(
                sample,
                tolerance_rad=(
                    self.config.ownership_transition_position_tolerance_rad
                    if ownership_transition
                    else self.config.held_arm_position_tolerance_rad
                ),
                transition=ownership_transition,
            )
        except (TypeError, ValueError, RuntimeError) as error:
            self._enter_fault(str(error), now)
            return self.state

        if self.state is ExecutorState.ACQUIRING:
            assert self._command_q14 is not None
            measured_q14 = dual_arm_vector(sample.left_q, sample.right_q)
            acquisition_error = float(np.max(np.abs(measured_q14 - self._command_q14)))
            self._maximum_acquisition_position_change_rad = max(
                self._maximum_acquisition_position_change_rad,
                acquisition_error,
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
                self._opposite_hold.rebase_monitor(sample)
                self._transition(
                    ExecutorState.READY,
                    "acquisition ramp complete; hold monitor rebased at loaded equilibrium",
                    now,
                )
            return self.state

        if self.state is ExecutorState.RELEASING:
            self._tick_release(now, emergency=False)
            return self.state

        if self.state in {
            ExecutorState.HOLDING,
            ExecutorState.READY,
            ExecutorState.CAPTURING,
        }:
            self._send(now)
            return self.state

        assert self._command_q14 is not None
        assert self._goal_q14 is not None
        if self._mpc_command_buffer is not None:
            try:
                calibration_command = self._mpc_command_buffer.command(now_s=now)
            except (TypeError, ValueError, RuntimeError) as error:
                self._enter_fault(str(error), now)
                return self.state
            self._command_q14 = self._compose_command(calibration_command)
        elif self._trajectory_time_s is None:
            self._command_q14 = velocity_limited_step(
                self._command_q14,
                self._goal_q14,
                maximum_velocity_rad_s=self.config.maximum_joint_velocity_rad_s,
                duration_s=self.config.nominal_tick_period_s,
            )
        else:
            assert self._trajectory_command_q is not None
            assert self._motion_started_s is not None
            elapsed = min(
                max(now - self._motion_started_s, 0.0),
                float(self._trajectory_time_s[-1]),
            )
            calibration_command = np.asarray(
                [
                    np.interp(
                        elapsed,
                        self._trajectory_time_s,
                        self._trajectory_command_q[:, index],
                    )
                    for index in range(7)
                ],
                dtype=np.float64,
            )
            self._command_q14 = self._compose_command(calibration_command)
        self._send(now)
        assert self._motion_started_s is not None
        assert self._calibration_goal_q is not None
        measured_q = sample.arm_q(self.pose_set.calibration_arm)
        position_errors = np.abs(measured_q - self._calibration_goal_q)
        position_error = float(np.max(position_errors))
        commanded_q = (
            self._command_q14[:7]
            if self.pose_set.calibration_arm == "left"
            else self._command_q14[7:]
        )
        self._last_motion_elapsed_s = now - self._motion_started_s
        self._last_motion_measured_q = measured_q.copy()
        self._last_motion_position_errors = position_errors.copy()
        self._last_command_remaining_rad = float(
            np.max(np.abs(commanded_q - self._calibration_goal_q))
        )
        if self.state is ExecutorState.MOVING:
            streaming_complete = (
                self._mpc_command_buffer is not None
                and self._mpc_command_buffer.terminal
                and self._mpc_command_buffer.remaining_s(now_s=now)
                <= _COMMAND_COMPLETION_EPSILON_RAD
            )
            frozen_or_pose_complete = (
                self._mpc_command_buffer is None
                and self._last_command_remaining_rad <= _COMMAND_COMPLETION_EPSILON_RAD
            )
            if streaming_complete or frozen_or_pose_complete:
                self._reset_settle_window()
                self._transition(
                    ExecutorState.SETTLING,
                    "command complete; measured stationarity is now the completion "
                    "gate and target error is diagnostic only",
                    now,
                )
        elif self._last_command_remaining_rad > _COMMAND_COMPLETION_EPSILON_RAD:
            self._reset_settle_window()
            self._transition(
                ExecutorState.MOVING,
                "command became incomplete",
                now,
            )
        elif self._settle_started_s is None:
            self._settle_started_s = now
            self._settle_min_q = measured_q.copy()
            self._settle_max_q = measured_q.copy()
            self._last_settle_elapsed_s = 0.0
            self._last_settle_spread_rad = 0.0
        else:
            assert self._settle_min_q is not None and self._settle_max_q is not None
            self._settle_min_q = np.minimum(self._settle_min_q, measured_q)
            self._settle_max_q = np.maximum(self._settle_max_q, measured_q)
            maximum_spread = float(np.max(self._settle_max_q - self._settle_min_q))
            self._last_settle_elapsed_s = now - self._settle_started_s
            self._last_settle_spread_rad = maximum_spread
            if maximum_spread > self.config.settled_position_spread_rad:
                self._settle_started_s = now
                self._settle_min_q = measured_q.copy()
                self._settle_max_q = measured_q.copy()
                self._last_settle_elapsed_s = 0.0
            elif now - self._settle_started_s >= self.config.settle_dwell_s:
                if (
                    self.config.require_motion_endpoint_tolerance
                    and position_error > self.config.motion_position_tolerance_rad
                ):
                    self._enter_fault(
                        self.motion_diagnostic(
                            prefix=(
                                "motion settled outside the required endpoint "
                                f"tolerance {self.config.motion_position_tolerance_rad:.4f}rad"
                            )
                        ),
                        now,
                    )
                else:
                    self.current_pose_id = self._pending_pose_id
                    self._pending_pose_id = None
                    self._trajectory_time_s = None
                    self._trajectory_command_q = None
                    self._mpc_command_buffer = None
                    self._active_motion_timeout_s = self.config.motion_timeout_s
                    endpoint_evidence = (
                        f"endpoint error {position_error:.4f}rad passed"
                        if self.config.require_motion_endpoint_tolerance
                        else (
                            f"endpoint error {position_error:.4f}rad recorded; "
                            "endpoint tolerance disabled"
                        )
                    )
                    self._transition(
                        ExecutorState.READY,
                        "continuous measured position-spread settle passed; " + endpoint_evidence,
                        now,
                    )
        if self.state in {ExecutorState.MOVING, ExecutorState.SETTLING}:
            self._last_motion_phase = self.state
            if self._last_motion_elapsed_s > self._active_motion_timeout_s:
                self._enter_fault(self.motion_diagnostic(prefix="motion timed out"), now)
        return self.state

    def observe_state(self) -> RobotStateSample:
        """Return one fresh measured state without changing the command."""

        now = self.clock.monotonic()
        sample = self.transport.observe()
        self._validate_fresh_state(sample, now)
        return sample

    def motion_diagnostic(self, *, prefix: str = "motion status") -> str:
        """Describe the latest measured tracking/settling evidence."""

        if (
            self._pending_pose_id is None
            or self._last_motion_phase is None
            or self._last_motion_elapsed_s is None
            or self._last_motion_measured_q is None
            or self._last_motion_position_errors is None
            or self._last_command_remaining_rad is None
        ):
            return f"{prefix}: no active measured-motion diagnostic"
        worst_index = int(np.argmax(self._last_motion_position_errors))
        joint_name = arm_joint_names(self.pose_set.calibration_arm)[worst_index]
        error = float(self._last_motion_position_errors[worst_index])
        measured = float(self._last_motion_measured_q[worst_index])
        assert self._calibration_goal_q is not None
        target = float(self._calibration_goal_q[worst_index])
        settle_elapsed = self._last_settle_elapsed_s or 0.0
        spread = (
            "n/a"
            if self._last_settle_spread_rad is None
            else f"{self._last_settle_spread_rad:.4f}rad"
        )
        return (
            f"{prefix} after {self._last_motion_elapsed_s:.2f}s while "
            f"{self._last_motion_phase.value} for {self._pending_pose_id}: "
            f"{joint_name} has maximum position error {error:.4f}rad "
            f"(measured={measured:.4f}, target={target:.4f}, "
            f"limit={self.config.motion_position_tolerance_rad:.4f}rad); "
            f"command remaining={self._last_command_remaining_rad:.4f}rad; "
            f"settle window={settle_elapsed:.2f}/{self.config.settle_dwell_s:.2f}s, "
            f"position spread={spread} "
            f"(limit={self.config.settled_position_spread_rad:.4f}rad)"
        )

    def begin_capture(self) -> None:
        if self.state not in {ExecutorState.READY, ExecutorState.HOLDING}:
            raise RuntimeError("capture can only begin from ready or holding")
        self._transition(
            ExecutorState.CAPTURING,
            "stationary capture started",
            self.clock.monotonic(),
        )

    def finish_capture(self, *, outcome: str) -> None:
        if self.state is not ExecutorState.CAPTURING:
            raise RuntimeError("capture can only finish while capturing")
        if not outcome.strip():
            raise ValueError("capture outcome must be non-empty")
        self._transition(
            ExecutorState.HOLDING,
            f"capture finished: {outcome.strip()}",
            self.clock.monotonic(),
        )

    def begin_clean_release(
        self,
        *,
        operator_confirmed: bool,
    ) -> None:
        if self.state not in {ExecutorState.HOLDING, ExecutorState.READY}:
            raise RuntimeError("clean release requires holding at the measured handoff pose")
        if not operator_confirmed:
            raise ValueError("operator confirmation is required for clean release")
        if self.current_pose_id != HANDOFF_POSE_ID:
            raise ValueError("executor is not at the measured handoff pose")
        now = self.clock.monotonic()
        sample = self.transport.observe()
        self._validate_fresh_state(sample, now)
        self._phase_started_s = now
        self._transition(ExecutorState.RELEASING, "clean release approved", now)

    def emergency_stop(self, reason: str) -> None:
        if self.state in {ExecutorState.STOPPED, ExecutorState.OBSERVING}:
            if self.state is ExecutorState.OBSERVING:
                self.transport.close()
                self._transition(ExecutorState.STOPPED, reason, self.clock.monotonic())
            return
        self._enter_fault(reason, self.clock.monotonic())

    def confirm_external_damping(self, reason: str) -> None:
        """Terminate command ownership after the G1 accepted whole-body damping."""

        self._confirm_external_takeover(
            reason,
            event_prefix="external damping confirmed",
        )

    def confirm_external_takeover(self, reason: str) -> None:
        """Terminate after a verified external controller accepted ownership."""

        self._confirm_external_takeover(
            reason,
            event_prefix="external controller takeover confirmed",
        )

    def _confirm_external_takeover(self, reason: str, *, event_prefix: str) -> None:

        if self.state is ExecutorState.STOPPED:
            return
        if not reason.strip():
            raise ValueError("external takeover reason must be non-empty")
        now = self.clock.monotonic()
        self.fault_reason = reason.strip()
        self.transport.close_after_external_takeover()
        self._transition(
            ExecutorState.STOPPED,
            f"{event_prefix}: {reason.strip()}",
            now,
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
            np.zeros(14, dtype=np.float64)
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

    def _compose_command(self, calibration_q: np.ndarray) -> np.ndarray:
        return self._opposite_hold.compose(calibration_q)

    def _tick_release(self, now: float, *, emergency: bool) -> None:
        assert self._phase_started_s is not None
        duration = self.config.release_ramp_s
        elapsed = max(now - self._phase_started_s, 0.0)
        initial = self._fault_initial_weight if emergency else 1.0
        self._weight = initial * max(1.0 - elapsed / duration, 0.0)
        self._send(now, emergency=emergency)
        if self._weight <= 0:
            self.transport.close()
            reason = "emergency weight reached zero" if emergency else "clean release complete"
            self._transition(ExecutorState.STOPPED, reason, now)

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


def validation_report_sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()
