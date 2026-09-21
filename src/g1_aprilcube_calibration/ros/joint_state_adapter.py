"""Optional complete named ROS JointState conversion for offline integration."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from g1_aprilcube_calibration.clock import MonotonicClock, SystemClock
from g1_aprilcube_calibration.joint_map import reorder_named_joint_state
from g1_aprilcube_calibration.models import RobotStateSample, utc_now_iso


def robot_state_from_joint_state(
    message: Any,
    *,
    receipt_monotonic_s: float,
    receipt_utc: str,
    mode_machine: int,
) -> RobotStateSample:
    """Require full positions and measured velocities in authoritative order."""
    positions, velocities, estimated_torques = reorder_named_joint_state(
        message.name, message.position, message.velocity, message.effort
    )
    return RobotStateSample(
        receipt_monotonic_s=receipt_monotonic_s,
        receipt_utc=receipt_utc,
        mode_machine=mode_machine,
        position=positions,
        velocity=velocities,
        estimated_torque=estimated_torques,
    )


class ROSJointStateSubscriber:
    """Read-only JointState subscriber; raw Unitree LowState remains preferred."""

    def __init__(
        self,
        node: Any,
        *,
        topic: str,
        verified_mode_machine: int,
        callback: Callable[[RobotStateSample], None],
        clock: MonotonicClock | None = None,
        utc_now: Callable[[], str] = utc_now_iso,
    ) -> None:
        try:
            from rclpy.qos import qos_profile_sensor_data
            from sensor_msgs.msg import JointState
        except ImportError as error:
            raise RuntimeError(
                "ROS 2 JointState unavailable; source the ROS Jazzy environment"
            ) from error
        self._node = node
        self._clock = clock or SystemClock()
        self._utc_now = utc_now
        self._mode = verified_mode_machine
        self._callback = callback
        self.last_error: str | None = None
        self._subscription = node.create_subscription(
            JointState, topic, self._receive, qos_profile_sensor_data
        )

    def _receive(self, message: Any) -> None:
        try:
            sample = robot_state_from_joint_state(
                message,
                receipt_monotonic_s=self._clock.monotonic(),
                receipt_utc=self._utc_now(),
                mode_machine=self._mode,
            )
            self._callback(sample)
            self.last_error = None
        except (AttributeError, TypeError, ValueError) as error:
            self.last_error = f"invalid complete JointState: {error}"

    def close(self) -> None:
        self._node.destroy_subscription(self._subscription)
