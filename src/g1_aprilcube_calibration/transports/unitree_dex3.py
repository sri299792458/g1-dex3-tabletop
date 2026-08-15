"""Minimal, fail-closed Unitree Dex3 calibration-posture control.

The calibration geometry is frozen at NVIDIA GR00T WholeBodyControl's
``middle close`` Dex3 posture.  This module therefore implements only that
fixed posture and the vendor timeout command; it is not a general hand
controller.  Importing the module does not load DDS or create a publisher.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np

from g1_aprilcube_calibration.clock import MonotonicClock, SystemClock

DEX3_MOTOR_COUNT = 7
DEX3_LEFT_COMMAND_TOPIC = "rt/dex3/left/cmd"
DEX3_RIGHT_COMMAND_TOPIC = "rt/dex3/right/cmd"
DEX3_LEFT_STATE_TOPIC = "rt/dex3/left/state"
DEX3_RIGHT_STATE_TOPIC = "rt/dex3/right/state"

# Unitree's documented Dex3 DDS message order.  The older
# ``Dex3_1_Right_JointIndex`` enum in xr_teleoperate swaps the right index and
# middle labels, but the newer retargeting-to-hardware mapping explicitly
# identifies this common order for both hands and the controller writes the
# resulting vector directly by motor ID.
DEX3_LEFT_MOTOR_JOINT_SUFFIXES = (
    "thumb_0",
    "thumb_1",
    "thumb_2",
    "middle_0",
    "middle_1",
    "index_0",
    "index_1",
)
DEX3_RIGHT_MOTOR_JOINT_SUFFIXES = (
    "thumb_0",
    "thumb_1",
    "thumb_2",
    "middle_0",
    "middle_1",
    "index_0",
    "index_1",
)
DEX3_MOTOR_JOINT_SUFFIXES = {
    "left": DEX3_LEFT_MOTOR_JOINT_SUFFIXES,
    "right": DEX3_RIGHT_MOTOR_JOINT_SUFFIXES,
}

# NVIDIA GR00T WholeBodyControl's symmetric ``middle close`` targets in each
# hand's Unitree DDS motor-ID order. Index and middle receive the same pair in
# this preset, but their motor-ID semantics are still kept explicit here.
# These are a named grasp preset, not the firmware's power-on homing endpoint
# and not every joint's mechanical limit.
NVIDIA_MIDDLE_CLOSE_LEFT_Q_RAD = (0.0, 0.7, 0.7, -1.0, -1.5, -1.0, -1.5)
NVIDIA_MIDDLE_CLOSE_RIGHT_Q_RAD = (0.0, -0.7, -0.7, 1.0, 1.5, 1.0, 1.5)
DEX3_CALIBRATION_POSTURE_SOURCE = (
    "NVlabs/GR00T-WholeBodyControl G1GripperInverseKinematicsSolver._get_middle_close_q_desired"
)


class Dex3GraspNotAcquiredError(RuntimeError):
    """The commanded close completed or timed out without stable object contact."""


class Dex3RetentionLostError(RuntimeError):
    """A contact-stalled finger no longer provides the commissioned evidence."""


def dex3_motor_mode(motor_id: int, *, timeout: bool) -> int:
    """Pack Unitree's four-bit ID, active status, and timeout bit."""

    if not 0 <= motor_id < DEX3_MOTOR_COUNT:
        raise ValueError(f"Dex3 motor ID must be within [0, 6], got {motor_id}")
    return (motor_id & 0x0F) | (0x01 << 4) | ((1 if timeout else 0) << 7)


def dex3_motor_joint_name(side: str, motor_id: int) -> str:
    """Return Unitree's side-specific joint name for one DDS motor ID."""

    if side not in DEX3_MOTOR_JOINT_SUFFIXES:
        raise ValueError(f"Dex3 side must be left or right, got {side!r}")
    if not 0 <= motor_id < DEX3_MOTOR_COUNT:
        raise ValueError(f"Dex3 motor ID must be within [0, 6], got {motor_id}")
    return f"{side}_hand_{DEX3_MOTOR_JOINT_SUFFIXES[side][motor_id]}_joint"


@dataclass(frozen=True, slots=True)
class Dex3ControlConfig:
    """Pinned gains, fixed targets, and calibration-specific safety bounds."""

    network_interface: str
    domain_id: int = 0
    left_command_topic: str = DEX3_LEFT_COMMAND_TOPIC
    right_command_topic: str = DEX3_RIGHT_COMMAND_TOPIC
    left_state_topic: str = DEX3_LEFT_STATE_TOPIC
    right_state_topic: str = DEX3_RIGHT_STATE_TOPIC
    command_rate_hz: float = 100.0
    kp: float = 1.5
    kd: float = 0.2
    left_target_q_rad: tuple[float, ...] = NVIDIA_MIDDLE_CLOSE_LEFT_Q_RAD
    right_target_q_rad: tuple[float, ...] = NVIDIA_MIDDLE_CLOSE_RIGHT_Q_RAD
    state_freshness_timeout_s: float = 0.1
    maximum_measured_command_delta_rad: float = 0.25
    posture_ramp_s: float = 2.0
    posture_position_tolerance_rad: float = 0.08
    posture_position_spread_rad: float = 0.01
    posture_settle_dwell_s: float = 0.5
    posture_timeout_s: float = 8.0
    subscriber_queue_length: int = 10
    timeout_repetitions: int = 3

    def __post_init__(self) -> None:
        if not self.network_interface.strip():
            raise ValueError("network_interface must be non-empty")
        if self.domain_id < 0:
            raise ValueError("domain_id must be non-negative")
        topics = (
            self.left_command_topic,
            self.right_command_topic,
            self.left_state_topic,
            self.right_state_topic,
        )
        if any(not topic.strip() for topic in topics) or len(set(topics)) != 4:
            raise ValueError("Dex3 DDS topics must be non-empty and distinct")
        for name in ("left_target_q_rad", "right_target_q_rad"):
            target = np.asarray(getattr(self, name), dtype=np.float64)
            if target.shape != (DEX3_MOTOR_COUNT,) or not np.all(np.isfinite(target)):
                raise ValueError(f"Dex3 {name} must contain seven finite values")
            object.__setattr__(self, name, tuple(float(value) for value in target))
        positive = (
            self.command_rate_hz,
            self.state_freshness_timeout_s,
            self.maximum_measured_command_delta_rad,
            self.posture_position_tolerance_rad,
            self.posture_position_spread_rad,
            self.posture_timeout_s,
        )
        if not all(np.isfinite(value) and value > 0.0 for value in positive):
            raise ValueError("Dex3 rates, timeouts, and tolerances must be positive")
        non_negative = (
            self.kp,
            self.kd,
            self.posture_ramp_s,
            self.posture_settle_dwell_s,
        )
        if not all(np.isfinite(value) and value >= 0.0 for value in non_negative):
            raise ValueError(
                "Dex3 gains, durations, and clearance requirements must be non-negative"
            )
        if self.posture_timeout_s < (self.posture_ramp_s + self.posture_settle_dwell_s):
            raise ValueError("Dex3 posture timeout is shorter than ramp plus settle")
        if self.subscriber_queue_length <= 0 or self.timeout_repetitions <= 0:
            raise ValueError("Dex3 queue length and timeout repetitions must be positive")


@dataclass(frozen=True, slots=True)
class Dex3SDKBindings:
    """Injectable Unitree SDK surface used by transport tests."""

    initialize: Callable[[int, str], None]
    publisher_type: type
    subscriber_type: type
    hand_command_type: type
    hand_state_type: type
    make_hand_command: Callable[[], Any]

    @classmethod
    def load(cls) -> Dex3SDKBindings:
        try:
            from unitree_sdk2py.core.channel import (
                ChannelFactoryInitialize,
                ChannelPublisher,
                ChannelSubscriber,
            )
            from unitree_sdk2py.idl.default import unitree_hg_msg_dds__HandCmd_
            from unitree_sdk2py.idl.unitree_hg.msg.dds_ import HandCmd_, HandState_
        except (ImportError, OSError) as error:
            raise RuntimeError(
                "Unitree Dex3 SDK unavailable; invoke hardware commands through "
                "tools/g1_calib_hardware.sh"
            ) from error
        return cls(
            initialize=ChannelFactoryInitialize,
            publisher_type=ChannelPublisher,
            subscriber_type=ChannelSubscriber,
            hand_command_type=HandCmd_,
            hand_state_type=HandState_,
            make_hand_command=unitree_hg_msg_dds__HandCmd_,
        )


@dataclass(frozen=True, slots=True)
class Dex3HandState:
    receipt_monotonic_s: float
    position: np.ndarray
    velocity: np.ndarray

    def __post_init__(self) -> None:
        for name in ("position", "velocity"):
            value = np.asarray(getattr(self, name), dtype=np.float64)
            if value.shape != (DEX3_MOTOR_COUNT,) or not np.all(np.isfinite(value)):
                raise ValueError(f"Dex3 {name} must contain seven finite values")
            value.setflags(write=False)
            object.__setattr__(self, name, value)


@dataclass(frozen=True, slots=True)
class Dex3StatePair:
    left: Dex3HandState
    right: Dex3HandState

    @property
    def maximum_abs_position_rad(self) -> float:
        return float(max(np.max(np.abs(self.left.position)), np.max(np.abs(self.right.position))))

    @property
    def maximum_abs_velocity_rad_s(self) -> float:
        return float(max(np.max(np.abs(self.left.velocity)), np.max(np.abs(self.right.velocity))))

    def maximum_target_error(
        self,
        left_target_q_rad: tuple[float, ...],
        right_target_q_rad: tuple[float, ...],
    ) -> tuple[str, int, float]:
        """Return side, motor index, and absolute error of the worst joint."""

        errors = {
            "left": np.abs(self.left.position - np.asarray(left_target_q_rad, dtype=np.float64)),
            "right": np.abs(
                self.right.position - np.asarray(right_target_q_rad, dtype=np.float64)
            ),
        }
        side = max(errors, key=lambda item: float(np.max(errors[item])))
        motor_index = int(np.argmax(errors[side]))
        return side, motor_index, float(errors[side][motor_index])

    def to_dict(self) -> dict:
        return {
            "left": {
                "receipt_monotonic_s": self.left.receipt_monotonic_s,
                "q_rad": self.left.position.tolist(),
                "dq_rad_s": self.left.velocity.tolist(),
            },
            "right": {
                "receipt_monotonic_s": self.right.receipt_monotonic_s,
                "q_rad": self.right.position.tolist(),
                "dq_rad_s": self.right.velocity.tolist(),
            },
            "maximum_abs_position_rad": self.maximum_abs_position_rad,
            "maximum_abs_velocity_rad_s": self.maximum_abs_velocity_rad_s,
        }


@dataclass(frozen=True, slots=True)
class Dex3GraspStallEvidence:
    """Stable active-hand blockage while a farther closing target remains commanded."""

    active_side: str
    target_q_rad: tuple[float, ...]
    contact_q_rad: tuple[float, ...]
    moved_motor_ids: tuple[int, ...]
    blocked_motor_ids: tuple[int, ...]
    remaining_error_rad: tuple[float, ...]
    settle_spread_rad: float
    settle_dwell_s: float

    def __post_init__(self) -> None:
        if self.active_side not in DEX3_MOTOR_JOINT_SUFFIXES:
            raise ValueError("Dex3 grasp-stall side must be left or right")
        for name in ("target_q_rad", "contact_q_rad", "remaining_error_rad"):
            value = np.asarray(getattr(self, name), dtype=np.float64).reshape(-1)
            if value.shape != (DEX3_MOTOR_COUNT,) or not np.all(np.isfinite(value)):
                raise ValueError(f"Dex3 grasp-stall {name} must contain seven finite values")
            object.__setattr__(self, name, tuple(float(item) for item in value))
        moved = tuple(int(value) for value in self.moved_motor_ids)
        if (
            not moved
            or len(set(moved)) != len(moved)
            or any(value < 0 or value >= DEX3_MOTOR_COUNT for value in moved)
        ):
            raise ValueError("Dex3 grasp-stall moved motor IDs are invalid")
        object.__setattr__(self, "moved_motor_ids", moved)
        blocked = tuple(int(value) for value in self.blocked_motor_ids)
        if (
            not blocked
            or len(set(blocked)) != len(blocked)
            or any(value < 0 or value >= DEX3_MOTOR_COUNT for value in blocked)
        ):
            raise ValueError("Dex3 grasp-stall blocked motor IDs are invalid")
        object.__setattr__(self, "blocked_motor_ids", blocked)
        for name in ("settle_spread_rad", "settle_dwell_s"):
            value = float(getattr(self, name))
            if not np.isfinite(value) or value < 0.0:
                raise ValueError(f"Dex3 grasp-stall {name} must be finite and non-negative")
            object.__setattr__(self, name, value)

    def to_dict(self) -> dict:
        return {
            "active_side": self.active_side,
            "target_q_rad": list(self.target_q_rad),
            "contact_q_rad": list(self.contact_q_rad),
            "moved_motor_ids": list(self.moved_motor_ids),
            "moved_joint_names": [
                dex3_motor_joint_name(self.active_side, value) for value in self.moved_motor_ids
            ],
            "blocked_motor_ids": list(self.blocked_motor_ids),
            "blocked_joint_names": [
                dex3_motor_joint_name(self.active_side, value) for value in self.blocked_motor_ids
            ],
            "remaining_error_rad": list(self.remaining_error_rad),
            "settle_spread_rad": self.settle_spread_rad,
            "settle_dwell_s": self.settle_dwell_s,
        }


@dataclass(frozen=True, slots=True)
class Dex3RetentionEvidence:
    """The same blocked joints remained blocked through a test-lift dwell."""

    grasp_stall: Dex3GraspStallEvidence
    verified_q_rad: tuple[float, ...]
    maximum_blocked_departure_rad: float
    verification_dwell_s: float

    def __post_init__(self) -> None:
        if not isinstance(self.grasp_stall, Dex3GraspStallEvidence):
            raise TypeError("Dex3 retention evidence requires grasp-stall evidence")
        q = np.asarray(self.verified_q_rad, dtype=np.float64).reshape(-1)
        if q.shape != (DEX3_MOTOR_COUNT,) or not np.all(np.isfinite(q)):
            raise ValueError("Dex3 retention verified posture must contain seven finite values")
        object.__setattr__(self, "verified_q_rad", tuple(float(value) for value in q))
        for name in ("maximum_blocked_departure_rad", "verification_dwell_s"):
            value = float(getattr(self, name))
            if not np.isfinite(value) or value < 0.0:
                raise ValueError(f"Dex3 retention {name} must be finite and non-negative")
            object.__setattr__(self, name, value)

    def to_dict(self) -> dict:
        return {
            "grasp_stall": self.grasp_stall.to_dict(),
            "verified_q_rad": list(self.verified_q_rad),
            "maximum_blocked_departure_rad": self.maximum_blocked_departure_rad,
            "verification_dwell_s": self.verification_dwell_s,
        }


class UnitreeDex3StateObserver:
    """Read both hand states without constructing either command publisher."""

    def __init__(
        self,
        config: Dex3ControlConfig,
        *,
        bindings: Dex3SDKBindings | None = None,
        clock: MonotonicClock | None = None,
        initialize_factory: bool = True,
    ) -> None:
        self.config = config
        self.bindings = bindings or Dex3SDKBindings.load()
        self.clock = clock or SystemClock()
        self._lock = threading.Lock()
        self._states: dict[str, Dex3HandState] = {}
        self._errors: dict[str, str] = {}
        self._closed = False
        if initialize_factory:
            self.bindings.initialize(config.domain_id, config.network_interface)
        self._left_subscriber = self.bindings.subscriber_type(
            config.left_state_topic, self.bindings.hand_state_type
        )
        self._right_subscriber = self.bindings.subscriber_type(
            config.right_state_topic, self.bindings.hand_state_type
        )
        self._left_subscriber.Init(
            lambda message: self._receive("left", message),
            config.subscriber_queue_length,
        )
        self._right_subscriber.Init(
            lambda message: self._receive("right", message),
            config.subscriber_queue_length,
        )

    def _receive(self, side: str, message: Any) -> None:
        try:
            motors = message.motor_state
            if len(motors) < DEX3_MOTOR_COUNT:
                raise ValueError(f"has {len(motors)} motors; expected at least {DEX3_MOTOR_COUNT}")
            state = Dex3HandState(
                receipt_monotonic_s=self.clock.monotonic(),
                position=np.asarray(
                    [motors[index].q for index in range(DEX3_MOTOR_COUNT)],
                    dtype=np.float64,
                ),
                velocity=np.asarray(
                    [motors[index].dq for index in range(DEX3_MOTOR_COUNT)],
                    dtype=np.float64,
                ),
            )
        except (AttributeError, TypeError, ValueError) as error:
            with self._lock:
                self._errors[side] = f"invalid {side} Dex3 state: {error}"
            return
        with self._lock:
            if self._closed:
                return
            self._states[side] = state
            self._errors.pop(side, None)

    def observe(self) -> Dex3StatePair:
        with self._lock:
            if self._closed:
                raise RuntimeError("Dex3 state observer is closed")
            missing = [side for side in ("left", "right") if side not in self._states]
            if missing:
                details = [self._errors[side] for side in missing if side in self._errors]
                suffix = "" if not details else ": " + "; ".join(details)
                raise RuntimeError(
                    "no valid Dex3 state received for " + ", ".join(missing) + suffix
                )
            pair = Dex3StatePair(self._states["left"], self._states["right"])
        now = self.clock.monotonic()
        ages = {
            "left": now - pair.left.receipt_monotonic_s,
            "right": now - pair.right.receipt_monotonic_s,
        }
        stale = [
            f"{side}={age:.3f}s"
            for side, age in ages.items()
            if age < 0.0 or age > self.config.state_freshness_timeout_s
        ]
        if stale:
            raise RuntimeError(
                "stale Dex3 state (" + ", ".join(stale) + "); limit is "
                f"{self.config.state_freshness_timeout_s:.3f}s"
            )
        return pair

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
        for subscriber in (self._left_subscriber, self._right_subscriber):
            close = getattr(subscriber, "Close", None)
            if callable(close):
                close()


class UnitreeDex3PostureController:
    """Ramp both hands to the configured posture, hold it, then timeout."""

    def __init__(
        self,
        config: Dex3ControlConfig,
        *,
        observer: UnitreeDex3StateObserver,
        clock: MonotonicClock | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        if observer.config != config:
            raise ValueError("existing Dex3 observer configuration does not match")
        self.config = config
        self.observer = observer
        self.bindings = observer.bindings
        self.clock = clock or observer.clock
        self._sleep = sleep
        self._left_publisher = self.bindings.publisher_type(
            config.left_command_topic, self.bindings.hand_command_type
        )
        self._right_publisher = self.bindings.publisher_type(
            config.right_command_topic, self.bindings.hand_command_type
        )
        self._left_publisher.Init()
        self._right_publisher.Init()
        self._last_publish_s: float | None = None
        self._timed_out = False
        self._closed = False
        self._initial_posture: Dex3StatePair | None = None
        self._active_left_target: np.ndarray | None = None
        self._active_right_target: np.ndarray | None = None
        self._active_grasp_stall: Dex3GraspStallEvidence | None = None
        self._retention_test_active = False
        self._retention_test_loss: Dex3RetentionLostError | None = None
        self.command_count = 0

    @property
    def timed_out(self) -> bool:
        return self._timed_out

    def acquire_measured_hold(
        self,
        *,
        safety_heartbeat: Callable[[], None] | None = None,
    ) -> Dex3StatePair:
        """Take hand ownership without changing either measured finger target."""

        self._require_active()
        if self._initial_posture is not None:
            raise RuntimeError("Dex3 measured posture was already acquired")
        if safety_heartbeat is not None:
            safety_heartbeat()
        initial = self.observer.observe()
        self._initial_posture = initial
        self._publish_targets(
            initial,
            {
                "left": initial.left.position,
                "right": initial.right.position,
            },
        )
        self._active_left_target = initial.left.position.copy()
        self._active_right_target = initial.right.position.copy()
        self._active_grasp_stall = None
        self._retention_test_active = False
        self._retention_test_loss = None
        return initial

    def acquire_posture(
        self,
        *,
        safety_heartbeat: Callable[[], None] | None = None,
    ) -> Dex3StatePair:
        """Smoothly command the targets and require measured settling."""

        initial = self.observer.observe()
        if self._initial_posture is None:
            self._initial_posture = initial
        result = self._move_to_targets(
            left_target_q_rad=np.asarray(self.config.left_target_q_rad, dtype=np.float64),
            right_target_q_rad=np.asarray(self.config.right_target_q_rad, dtype=np.float64),
            label="calibration-posture acquisition",
            safety_heartbeat=safety_heartbeat,
        )
        self._active_left_target = np.asarray(self.config.left_target_q_rad, dtype=np.float64)
        self._active_right_target = np.asarray(self.config.right_target_q_rad, dtype=np.float64)
        self._active_grasp_stall = None
        self._retention_test_active = False
        self._retention_test_loss = None
        return result

    def command_posture(
        self,
        *,
        left_target_q_rad,
        right_target_q_rad,
        label: str,
        safety_heartbeat: Callable[[], None] | None = None,
    ) -> Dex3StatePair:
        """Move to one explicit task posture using the commissioned limiter."""

        left = np.asarray(left_target_q_rad, dtype=np.float64).reshape(-1)
        right = np.asarray(right_target_q_rad, dtype=np.float64).reshape(-1)
        if left.shape != (DEX3_MOTOR_COUNT,) or right.shape != (DEX3_MOTOR_COUNT,):
            raise ValueError("explicit Dex3 posture must contain seven values per hand")
        if not np.all(np.isfinite(left)) or not np.all(np.isfinite(right)):
            raise ValueError("explicit Dex3 posture contains NaN or infinity")
        if not label.strip():
            raise ValueError("explicit Dex3 posture label must be non-empty")
        result = self._move_to_targets(
            left_target_q_rad=left,
            right_target_q_rad=right,
            label=label.strip(),
            safety_heartbeat=safety_heartbeat,
        )
        self._active_left_target = left.copy()
        self._active_right_target = right.copy()
        self._active_grasp_stall = None
        self._retention_test_active = False
        self._retention_test_loss = None
        return result

    def command_grasp_until_stall(
        self,
        *,
        active_side: str,
        left_target_q_rad,
        right_target_q_rad,
        label: str,
        safety_heartbeat: Callable[[], None] | None = None,
    ) -> Dex3GraspStallEvidence:
        """Close smoothly and accept only stable blockage before the target.

        The threshold and dwell deliberately reuse the physically commissioned
        empty-hand posture tolerance and settling limits. Raw ``dq``, pressure,
        and effort are not used by the live decision.
        """

        if active_side not in DEX3_MOTOR_JOINT_SUFFIXES:
            raise ValueError("Dex3 grasp side must be left or right")
        left = np.asarray(left_target_q_rad, dtype=np.float64).reshape(-1)
        right = np.asarray(right_target_q_rad, dtype=np.float64).reshape(-1)
        if left.shape != (DEX3_MOTOR_COUNT,) or right.shape != (DEX3_MOTOR_COUNT,):
            raise ValueError("explicit Dex3 grasp posture must contain seven values per hand")
        if not np.all(np.isfinite(left)) or not np.all(np.isfinite(right)):
            raise ValueError("explicit Dex3 grasp posture contains NaN or infinity")
        if not label.strip():
            raise ValueError("explicit Dex3 grasp label must be non-empty")
        evidence = self._move_to_grasp_stall(
            active_side=active_side,
            left_target_q_rad=left,
            right_target_q_rad=right,
            label=label.strip(),
            safety_heartbeat=safety_heartbeat,
        )
        self._active_left_target = left.copy()
        self._active_right_target = right.copy()
        self._active_grasp_stall = evidence
        self._retention_test_active = False
        self._retention_test_loss = None
        return evidence

    def begin_retention_test(self) -> None:
        """Treat contact loss during the small test lift as a task outcome."""

        if self._active_grasp_stall is None:
            raise RuntimeError("Dex3 grasp stall was not acquired before the test lift")
        self._retention_test_active = True
        self._retention_test_loss = None

    def finish_retention_test(self) -> None:
        """Return contact monitoring to fail-closed behavior after a passed test."""

        self.check_retention_test()
        self._retention_test_active = False

    def check_retention_test(self) -> None:
        """Raise in the task thread if the heartbeat observed contact loss."""

        if self._retention_test_loss is not None:
            raise self._retention_test_loss

    def restore_initial_posture(
        self,
        *,
        safety_heartbeat: Callable[[], None] | None = None,
    ) -> Dex3StatePair:
        """Return both hands to the measured pre-command posture."""

        if self._initial_posture is None:
            raise RuntimeError("Dex3 posture was not acquired before restoration")
        result = self._move_to_targets(
            left_target_q_rad=self._initial_posture.left.position,
            right_target_q_rad=self._initial_posture.right.position,
            label="initial-posture restoration",
            safety_heartbeat=safety_heartbeat,
        )
        self._active_left_target = self._initial_posture.left.position.copy()
        self._active_right_target = self._initial_posture.right.position.copy()
        self._active_grasp_stall = None
        self._retention_test_active = False
        self._retention_test_loss = None
        return result

    def _move_to_grasp_stall(
        self,
        *,
        active_side: str,
        left_target_q_rad: np.ndarray,
        right_target_q_rad: np.ndarray,
        label: str,
        safety_heartbeat: Callable[[], None] | None,
    ) -> Dex3GraspStallEvidence:
        self._require_active()
        initial = self.observer.observe()
        starts = {"left": initial.left.position, "right": initial.right.position}
        final_targets = {"left": left_target_q_rad, "right": right_target_q_rad}
        active_start = starts[active_side]
        active_target = final_targets[active_side]
        travel = active_target - active_start
        tolerance = self.config.posture_position_tolerance_rad
        moving_motor = np.abs(travel) > tolerance
        if not np.any(moving_motor):
            raise RuntimeError(f"Dex3 {label} has no active-hand closing travel above tolerance")

        started = self.clock.monotonic()
        deadline = started + self.config.posture_timeout_s
        settled_since: float | None = None
        settle_min_q: np.ndarray | None = None
        settle_max_q: np.ndarray | None = None
        stable_blocked: tuple[int, ...] = ()
        moved_motor = np.zeros(DEX3_MOTOR_COUNT, dtype=bool)
        last_spread_rad: float | None = None
        period_s = 1.0 / self.config.command_rate_hz
        while True:
            if safety_heartbeat is not None:
                safety_heartbeat()
            now = self.clock.monotonic()
            pair = self.observer.observe()
            elapsed = max(now - started, 0.0)
            fraction = (
                1.0
                if self.config.posture_ramp_s == 0.0
                else min(elapsed / self.config.posture_ramp_s, 1.0)
            )
            smooth = fraction * fraction * (3.0 - 2.0 * fraction)
            targets = {
                side: starts[side] * (1.0 - smooth) + final_targets[side] * smooth
                for side in ("left", "right")
            }
            self._publish_targets(pair, targets)

            active = pair.left if active_side == "left" else pair.right
            inactive_side = "right" if active_side == "left" else "left"
            inactive = pair.right if active_side == "left" else pair.left
            remaining = active_target - active.position
            progress = active.position - active_start
            moved_motor |= moving_motor & (
                np.sign(travel) * progress >= self.config.posture_position_spread_rad
            )
            on_commanded_segment = (progress * travel >= -(tolerance * np.abs(travel))) & (
                progress * travel <= travel * travel + tolerance * np.abs(travel)
            )
            still_before_target = remaining * travel > 0.0
            blocked_mask = (
                moving_motor
                & (np.abs(remaining) > tolerance)
                & on_commanded_segment
                & still_before_target
            )
            blocked = tuple(int(value) for value in np.flatnonzero(blocked_mask))
            unexplained_active_error = np.any((np.abs(remaining) > tolerance) & ~blocked_mask)
            inactive_error = float(
                np.max(np.abs(inactive.position - final_targets[inactive_side]))
            )
            acceptable = (
                fraction == 1.0
                and np.any(moved_motor)
                and not unexplained_active_error
                and inactive_error <= tolerance
            )
            measured_positions = np.concatenate((pair.left.position, pair.right.position))
            if acceptable:
                if settled_since is None or blocked != stable_blocked:
                    settled_since = now
                    stable_blocked = blocked
                    settle_min_q = measured_positions.copy()
                    settle_max_q = measured_positions.copy()
                    last_spread_rad = 0.0
                else:
                    assert settle_min_q is not None and settle_max_q is not None
                    settle_min_q = np.minimum(settle_min_q, measured_positions)
                    settle_max_q = np.maximum(settle_max_q, measured_positions)
                    last_spread_rad = float(np.max(settle_max_q - settle_min_q))
                    if last_spread_rad > self.config.posture_position_spread_rad:
                        settled_since = now
                        settle_min_q = measured_positions.copy()
                        settle_max_q = measured_positions.copy()
                        last_spread_rad = 0.0
                if now - settled_since >= self.config.posture_settle_dwell_s:
                    if not blocked:
                        raise Dex3GraspNotAcquiredError(
                            f"Dex3 {label} reached the complete empty-hand target without "
                            "a stable position stall; retention was not established"
                        )
                    return Dex3GraspStallEvidence(
                        active_side=active_side,
                        target_q_rad=tuple(active_target),
                        contact_q_rad=tuple(active.position),
                        moved_motor_ids=tuple(int(value) for value in np.flatnonzero(moved_motor)),
                        blocked_motor_ids=blocked,
                        remaining_error_rad=tuple(remaining),
                        settle_spread_rad=float(last_spread_rad or 0.0),
                        settle_dwell_s=self.config.posture_settle_dwell_s,
                    )
            else:
                settled_since = None
                settle_min_q = None
                settle_max_q = None
                stable_blocked = ()
                last_spread_rad = None

            if now >= deadline:
                motor_index = int(np.argmax(np.abs(remaining)))
                raise Dex3GraspNotAcquiredError(
                    f"Dex3 {label} timed out before a stable grasp stall: worst remaining "
                    f"error={abs(remaining[motor_index]):.4f}rad at {active_side} motor "
                    f"{motor_index} ({dex3_motor_joint_name(active_side, motor_index)}); "
                    f"stall residual threshold={tolerance:.4f}rad, position spread="
                    f"{'n/a' if last_spread_rad is None else f'{last_spread_rad:.4f}rad'} "
                    f"(limit={self.config.posture_position_spread_rad:.4f}rad); commanded "
                    f"motion observed on motors={list(np.flatnonzero(moved_motor))}"
                )
            self._sleep(period_s)

    def verify_grasp_stall_persistence(
        self,
        *,
        safety_heartbeat: Callable[[], None] | None = None,
    ) -> Dex3RetentionEvidence:
        """Require the blocked posture to persist for one commissioned dwell."""

        if self._active_grasp_stall is None:
            raise RuntimeError("Dex3 grasp stall was not acquired before retention verification")
        if self._retention_test_loss is not None:
            raise self._retention_test_loss
        started = self.clock.monotonic()
        maximum_departure = 0.0
        pair = self.observer.observe()
        while True:
            if safety_heartbeat is not None:
                safety_heartbeat()
            pair, departure = self._publish_and_check_grasp_stall()
            maximum_departure = max(maximum_departure, departure)
            now = self.clock.monotonic()
            if now - started >= self.config.posture_settle_dwell_s:
                active = (
                    pair.left if self._active_grasp_stall.active_side == "left" else pair.right
                )
                return Dex3RetentionEvidence(
                    grasp_stall=self._active_grasp_stall,
                    verified_q_rad=tuple(active.position),
                    maximum_blocked_departure_rad=maximum_departure,
                    verification_dwell_s=self.config.posture_settle_dwell_s,
                )
            self._sleep(1.0 / self.config.command_rate_hz)

    def _move_to_targets(
        self,
        *,
        left_target_q_rad: np.ndarray,
        right_target_q_rad: np.ndarray,
        label: str,
        safety_heartbeat: Callable[[], None] | None,
    ) -> Dex3StatePair:
        self._require_active()
        initial = self.observer.observe()
        starts = {"left": initial.left.position, "right": initial.right.position}
        started = self.clock.monotonic()
        deadline = started + self.config.posture_timeout_s
        settled_since: float | None = None
        settle_min_q: np.ndarray | None = None
        settle_max_q: np.ndarray | None = None
        last_spread_rad: float | None = None
        previous_now = started
        period_s = 1.0 / self.config.command_rate_hz
        while True:
            if safety_heartbeat is not None:
                safety_heartbeat()
            now = self.clock.monotonic()
            pair = self.observer.observe()
            elapsed = max(now - started, 0.0)
            fraction = (
                1.0
                if self.config.posture_ramp_s == 0.0
                else min(elapsed / self.config.posture_ramp_s, 1.0)
            )
            smooth = fraction * fraction * (3.0 - 2.0 * fraction)
            targets = {
                "left": (starts["left"] * (1.0 - smooth) + left_target_q_rad * smooth),
                "right": (starts["right"] * (1.0 - smooth) + right_target_q_rad * smooth),
            }
            self._publish_targets(pair, targets)
            _, _, maximum_error = pair.maximum_target_error(
                tuple(left_target_q_rad),
                tuple(right_target_q_rad),
            )
            in_position = (
                fraction == 1.0 and maximum_error <= self.config.posture_position_tolerance_rad
            )
            settle_window_broken = False
            measured_positions = np.concatenate((pair.left.position, pair.right.position))
            if in_position:
                if settled_since is None:
                    settled_since = now
                    settle_min_q = measured_positions.copy()
                    settle_max_q = measured_positions.copy()
                    last_spread_rad = 0.0
                else:
                    assert settle_min_q is not None and settle_max_q is not None
                    settle_min_q = np.minimum(settle_min_q, measured_positions)
                    settle_max_q = np.maximum(settle_max_q, measured_positions)
                    last_spread_rad = float(np.max(settle_max_q - settle_min_q))
                    if last_spread_rad > self.config.posture_position_spread_rad:
                        settle_window_broken = True
                        settled_since = now
                        settle_min_q = measured_positions.copy()
                        settle_max_q = measured_positions.copy()
                if now - settled_since >= self.config.posture_settle_dwell_s:
                    return pair
            else:
                settled_since = None
                settle_min_q = None
                settle_max_q = None
                last_spread_rad = None
            # The acquisition timeout bounds entry into the accepted posture,
            # not the verification dwell after entry. A polling sample that
            # straddles the deadline may start that final window. Once started,
            # the window must remain within the configured position spread.
            crossed_deadline_this_sample = previous_now < deadline <= now
            may_finish_existing_window = settled_since is not None and (
                settled_since < deadline or crossed_deadline_this_sample
            )
            if now >= deadline and (not may_finish_existing_window or settle_window_broken):
                side, motor_index, maximum_error = pair.maximum_target_error(
                    tuple(left_target_q_rad),
                    tuple(right_target_q_rad),
                )
                measured_q = (
                    pair.left.position[motor_index]
                    if side == "left"
                    else pair.right.position[motor_index]
                )
                target_q = (
                    left_target_q_rad[motor_index]
                    if side == "left"
                    else right_target_q_rad[motor_index]
                )
                raise RuntimeError(
                    f"Dex3 {label} timed out: worst target "
                    f"error={maximum_error:.4f}rad at {side} motor {motor_index} "
                    f"({dex3_motor_joint_name(side, motor_index)}) "
                    f"(measured={measured_q:.4f}rad, target={target_q:.4f}rad, "
                    f"limit={self.config.posture_position_tolerance_rad:.4f}), "
                    "position spread="
                    f"{'n/a' if last_spread_rad is None else f'{last_spread_rad:.4f}rad'} "
                    f"(limit={self.config.posture_position_spread_rad:.4f}rad), "
                    "maximum raw velocity="
                    f"{pair.maximum_abs_velocity_rad_s:.4f}rad/s (diagnostic only)"
                )
            previous_now = now
            self._sleep(period_s)

    def maintain_posture(self) -> None:
        """Publish the fixed posture at the configured rate using fresh state."""

        self._maintain_targets(
            left_target_q_rad=np.asarray(self.config.left_target_q_rad, dtype=np.float64),
            right_target_q_rad=np.asarray(self.config.right_target_q_rad, dtype=np.float64),
            label="fixed calibration posture",
        )

    def maintain_initial_posture(self) -> None:
        """Keep the exact pre-command finger posture during arm clearance."""

        if self._initial_posture is None:
            raise RuntimeError("Dex3 measured posture was not acquired")
        self._maintain_targets(
            left_target_q_rad=self._initial_posture.left.position,
            right_target_q_rad=self._initial_posture.right.position,
            label="measured initial posture",
        )

    def maintain_active_posture(self) -> None:
        """Maintain the last acquired/restored/task-specific posture."""

        if self._active_left_target is None or self._active_right_target is None:
            raise RuntimeError("Dex3 posture has not been acquired")
        if self._active_grasp_stall is not None:
            now = self.clock.monotonic()
            if (
                self._last_publish_s is not None
                and now - self._last_publish_s < 1.0 / self.config.command_rate_hz
            ):
                return
            try:
                self._publish_and_check_grasp_stall()
            except Dex3RetentionLostError as error:
                if not self._retention_test_active:
                    raise
                self._retention_test_loss = error
            return
        self._maintain_targets(
            left_target_q_rad=self._active_left_target,
            right_target_q_rad=self._active_right_target,
            label="active task posture",
        )

    def _publish_and_check_grasp_stall(self) -> tuple[Dex3StatePair, float]:
        self._require_active()
        evidence = self._active_grasp_stall
        if (
            evidence is None
            or self._active_left_target is None
            or self._active_right_target is None
        ):
            raise RuntimeError("Dex3 grasp-stall hold is not active")
        pair = self.observer.observe()
        active = pair.left if evidence.active_side == "left" else pair.right
        inactive = pair.right if evidence.active_side == "left" else pair.left
        active_target = (
            self._active_left_target
            if evidence.active_side == "left"
            else self._active_right_target
        )
        inactive_target = (
            self._active_right_target
            if evidence.active_side == "left"
            else self._active_left_target
        )
        blocked = np.asarray(evidence.blocked_motor_ids, dtype=np.int64)
        contact = np.asarray(evidence.contact_q_rad, dtype=np.float64)
        departure = float(np.max(np.abs(active.position[blocked] - contact[blocked])))
        if departure > self.config.posture_position_spread_rad:
            motor_index = int(
                blocked[np.argmax(np.abs(active.position[blocked] - contact[blocked]))]
            )
            error = Dex3RetentionLostError(
                "Dex3 grasp retention lost: blocked "
                f"{dex3_motor_joint_name(evidence.active_side, motor_index)} moved "
                f"{departure:.4f}rad from its contact posture; limit is "
                f"{self.config.posture_position_spread_rad:.4f}rad"
            )
            self._publish_targets(
                pair,
                {"left": self._active_left_target, "right": self._active_right_target},
            )
            raise error
        remaining = np.abs(active_target[blocked] - active.position[blocked])
        if np.any(remaining <= self.config.posture_position_tolerance_rad):
            motor_index = int(blocked[np.argmin(remaining)])
            error = Dex3RetentionLostError(
                "Dex3 grasp retention lost: blocked "
                f"{dex3_motor_joint_name(evidence.active_side, motor_index)} continued to "
                "the empty-hand target"
            )
            self._publish_targets(
                pair,
                {"left": self._active_left_target, "right": self._active_right_target},
            )
            raise error
        unblocked = np.ones(DEX3_MOTOR_COUNT, dtype=bool)
        unblocked[blocked] = False
        active_error = float(
            np.max(
                np.abs(active.position[unblocked] - active_target[unblocked]),
                initial=0.0,
            )
        )
        inactive_error = float(np.max(np.abs(inactive.position - inactive_target)))
        if max(active_error, inactive_error) > self.config.posture_position_tolerance_rad:
            raise RuntimeError(
                "Dex3 grasp hold departed a non-contact target: maximum error "
                f"{max(active_error, inactive_error):.4f}rad; limit is "
                f"{self.config.posture_position_tolerance_rad:.4f}rad"
            )
        self._publish_targets(
            pair,
            {"left": self._active_left_target, "right": self._active_right_target},
        )
        return pair, departure

    def _maintain_targets(
        self,
        *,
        left_target_q_rad: np.ndarray,
        right_target_q_rad: np.ndarray,
        label: str,
    ) -> None:
        """Publish one frozen target after checking fresh measured tracking."""

        self._require_active()
        now = self.clock.monotonic()
        if (
            self._last_publish_s is not None
            and now - self._last_publish_s < 1.0 / self.config.command_rate_hz
        ):
            return
        pair = self.observer.observe()
        side, motor_index, maximum_error = pair.maximum_target_error(
            tuple(left_target_q_rad),
            tuple(right_target_q_rad),
        )
        if maximum_error > self.config.posture_position_tolerance_rad:
            raise RuntimeError(
                f"Dex3 departed the {label}: worst target error is "
                f"{maximum_error:.4f}rad at {side} motor {motor_index} "
                f"({dex3_motor_joint_name(side, motor_index)}); limit "
                f"is {self.config.posture_position_tolerance_rad:.4f}rad"
            )
        self._publish_targets(
            pair,
            {
                "left": left_target_q_rad,
                "right": right_target_q_rad,
            },
        )

    def timeout(self) -> None:
        """Send Unitree's timeout=1 command to both hands, then close DDS."""

        if self._timed_out:
            return
        if self._closed:
            raise RuntimeError("cannot timeout a closed Dex3 controller")
        left = self._make_command(np.zeros(DEX3_MOTOR_COUNT), timeout=True)
        right = self._make_command(np.zeros(DEX3_MOTOR_COUNT), timeout=True)
        for index in range(self.config.timeout_repetitions):
            self._write(self._left_publisher, left, "left timeout")
            self._write(self._right_publisher, right, "right timeout")
            if index + 1 < self.config.timeout_repetitions:
                self._sleep(1.0 / self.config.command_rate_hz)
        self._timed_out = True

    def close(self) -> None:
        if self._closed:
            return
        if self.command_count and not self._timed_out:
            raise RuntimeError("refusing to close an active Dex3 controller before timeout")
        self._closed = True
        self._left_publisher.Close()
        self._right_publisher.Close()
        self.observer.close()

    def timeout_and_close(self) -> None:
        self.timeout()
        self.close()

    def close_after_external_timeout(self) -> None:
        """Close after the PC2 watchdog verified its own Dex3 timeout."""

        if self._closed:
            return
        self._timed_out = True
        self.close()

    def _publish_targets(
        self,
        pair: Dex3StatePair,
        targets: dict[str, np.ndarray],
    ) -> None:
        self._require_active()
        maximum_delta = self.config.maximum_measured_command_delta_rad
        left_q = np.clip(
            targets["left"],
            pair.left.position - maximum_delta,
            pair.left.position + maximum_delta,
        )
        right_q = np.clip(
            targets["right"],
            pair.right.position - maximum_delta,
            pair.right.position + maximum_delta,
        )
        self._write(
            self._left_publisher,
            self._make_command(left_q, timeout=False),
            "left posture",
        )
        self._write(
            self._right_publisher,
            self._make_command(right_q, timeout=False),
            "right posture",
        )
        self._last_publish_s = self.clock.monotonic()
        self.command_count += 1

    def _make_command(self, q: np.ndarray, *, timeout: bool):
        command = self.bindings.make_hand_command()
        if len(command.motor_cmd) != DEX3_MOTOR_COUNT:
            raise ValueError(f"Dex3 HandCmd has {len(command.motor_cmd)} motors; expected 7")
        for motor_id, value in enumerate(q):
            motor = command.motor_cmd[motor_id]
            motor.mode = dex3_motor_mode(motor_id, timeout=timeout)
            motor.q = 0.0 if timeout else float(value)
            motor.dq = 0.0
            motor.tau = 0.0
            motor.kp = 0.0 if timeout else self.config.kp
            motor.kd = 0.0 if timeout else self.config.kd
        return command

    @staticmethod
    def _write(publisher, message, label: str) -> None:
        if publisher.Write(message) is False:
            raise RuntimeError(f"Unitree Dex3 {label} publish failed")

    def _require_active(self) -> None:
        if self._closed:
            raise RuntimeError("Dex3 controller is closed")
        if self._timed_out:
            raise RuntimeError("Dex3 controller was timed out")
