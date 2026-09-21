"""Deterministic in-memory mode-5 G1 arm transport."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from g1_aprilcube_calibration.clock import ManualClock
from g1_aprilcube_calibration.joint_map import LEFT_ARM_INDICES, RIGHT_ARM_INDICES
from g1_aprilcube_calibration.models import RobotStateSample
from g1_aprilcube_calibration.transports.base import ArmCommand


class FakeArmTransport:
    def __init__(
        self,
        *,
        clock: ManualClock,
        initial_full_q: Sequence[float] | np.ndarray,
        tracking_velocity_rad_s: float = 1.0,
        mode_machine: int = 5,
    ) -> None:
        position = np.asarray(initial_full_q, dtype=np.float64).reshape(-1).copy()
        if position.shape != (29,) or not np.all(np.isfinite(position)):
            raise ValueError("initial_full_q must be a finite 29-joint vector")
        if tracking_velocity_rad_s <= 0:
            raise ValueError("tracking_velocity_rad_s must be positive")
        self.clock = clock
        self.position = position
        self.velocity = np.zeros(29, dtype=np.float64)
        self.estimated_torque = np.zeros(29, dtype=np.float64)
        self.tracking_velocity_rad_s = tracking_velocity_rad_s
        self.mode_machine = mode_machine
        self.commands: list[ArmCommand] = []
        self.closed = False
        self.freeze_state_receipt = False
        self._last_receipt_s = clock.monotonic()
        self._sequence = 0

    def observe(self) -> RobotStateSample:
        receipt = self._last_receipt_s
        if not self.freeze_state_receipt:
            receipt = self.clock.monotonic()
            self._last_receipt_s = receipt
            self._sequence += 1
        return RobotStateSample(
            receipt_monotonic_s=receipt,
            receipt_utc="2026-08-02T12:00:00Z",
            mode_machine=self.mode_machine,
            position=self.position,
            velocity=self.velocity,
            estimated_torque=self.estimated_torque,
            source_sequence=self._sequence,
        )

    def send_command(self, command: ArmCommand) -> None:
        if self.closed:
            raise RuntimeError("fake transport is closed")
        if command.issued_monotonic_s != self.clock.monotonic():
            raise ValueError("fake command timestamp does not match the manual clock")
        self.commands.append(command)

    def step(self, duration_s: float) -> None:
        if duration_s <= 0:
            raise ValueError("fake step duration must be positive")
        self.clock.advance(duration_s)
        previous = self.position.copy()
        if self.commands and self.commands[-1].weight > 0:
            target = np.asarray(self.commands[-1].q14)
            current = np.concatenate(
                (
                    self.position[np.asarray(LEFT_ARM_INDICES)],
                    self.position[np.asarray(RIGHT_ARM_INDICES)],
                )
            )
            maximum_step = self.tracking_velocity_rad_s * duration_s
            achieved = current + np.clip(target - current, -maximum_step, maximum_step)
            self.position[np.asarray(LEFT_ARM_INDICES)] = achieved[:7]
            self.position[np.asarray(RIGHT_ARM_INDICES)] = achieved[7:]
        self.velocity = (self.position - previous) / duration_s

    def close(self) -> None:
        self.closed = True

    def close_after_external_takeover(self) -> None:
        self.closed = True
