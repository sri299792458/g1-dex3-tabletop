"""Shared zero-displacement hold for arm_sdk ownership transitions."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from g1_aprilcube_calibration.joint_map import (
    arm_joint_names,
    dual_arm_vector,
    opposite_arm,
    validate_arm_side,
    validate_arm_vector,
)


class OppositeArmHold:
    """Command one measured arm pose and monitor its loaded equilibrium."""

    def __init__(
        self,
        *,
        calibration_arm: str,
        command_q: Sequence[float] | np.ndarray,
    ) -> None:
        self.calibration_arm = validate_arm_side(calibration_arm)
        self.arm = opposite_arm(self.calibration_arm)
        self.command_q = validate_arm_vector(command_q, side=self.arm)
        self.monitor_q = self.command_q.copy()

    @classmethod
    def from_dual_arm_vector(
        cls,
        *,
        calibration_arm: str,
        q14: Sequence[float] | np.ndarray,
    ) -> OppositeArmHold:
        measured = np.asarray(q14, dtype=np.float64).reshape(-1)
        if measured.shape != (14,) or not np.all(np.isfinite(measured)):
            raise ValueError("q14 must contain 14 finite values")
        command_q = measured[7:] if calibration_arm == "left" else measured[:7]
        return cls(calibration_arm=calibration_arm, command_q=command_q)

    def seed_from_sample(self, sample) -> None:
        self.command_q = sample.arm_q(self.arm).copy()
        self.monitor_q = self.command_q.copy()

    def rebase_monitor(self, sample) -> None:
        self.monitor_q = sample.arm_q(self.arm).copy()

    def validate(
        self,
        sample,
        *,
        tolerance_rad: float,
        transition: bool = False,
    ) -> None:
        if not np.isfinite(tolerance_rad) or tolerance_rad <= 0:
            raise ValueError("hold tolerance must be finite and positive")
        measured = sample.arm_q(self.arm)
        errors = np.abs(measured - self.monitor_q)
        error = float(np.max(errors))
        if error > tolerance_rad:
            context = " moved during ownership transition" if transition else " drifted"
            joint_index = int(np.argmax(errors))
            raise ValueError(
                f"held {self.arm} arm{context} by {error:.4f}rad; limit is {tolerance_rad:.4f}rad; "
                f"joint={arm_joint_names(self.arm)[joint_index]}, "
                f"reference={self.monitor_q[joint_index]:.6f}rad, "
                f"measured={measured[joint_index]:.6f}rad"
            )

    def compose(self, calibration_q: Sequence[float] | np.ndarray) -> np.ndarray:
        calibration = validate_arm_vector(
            calibration_q,
            side=self.calibration_arm,
        )
        if self.calibration_arm == "left":
            return dual_arm_vector(calibration, self.command_q)
        return dual_arm_vector(self.command_q, calibration)
