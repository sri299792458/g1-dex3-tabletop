"""Narrow command boundary owned by one calibration arm controller at a time."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

import numpy as np

from g1_aprilcube_calibration.joint_map import DUAL_ARM_DOF
from g1_aprilcube_calibration.models import RobotStateSample


@dataclass(frozen=True, slots=True)
class ArmCommand:
    q14: tuple[float, ...]
    weight: float
    issued_monotonic_s: float
    emergency_release: bool = False
    kp_scale14: tuple[float, ...] = (1.0,) * DUAL_ARM_DOF
    kd_scale14: tuple[float, ...] = (1.0,) * DUAL_ARM_DOF
    tau_ff14: tuple[float, ...] = (0.0,) * DUAL_ARM_DOF

    def __post_init__(self) -> None:
        array = np.asarray(self.q14, dtype=np.float64).reshape(-1)
        if array.shape != (DUAL_ARM_DOF,):
            raise ValueError(f"arm command must contain {DUAL_ARM_DOF} joints")
        if not np.all(np.isfinite(array)):
            raise ValueError("arm command contains NaN or infinity")
        if not np.isfinite(self.weight) or not 0 <= self.weight <= 1:
            raise ValueError("arm command weight must be within [0, 1]")
        if not np.isfinite(self.issued_monotonic_s) or self.issued_monotonic_s < 0:
            raise ValueError("arm command timestamp must be finite and non-negative")
        gain_scales: dict[str, np.ndarray] = {}
        for name, label in (
            ("kp_scale14", "Kp gain scale"),
            ("kd_scale14", "Kd gain scale"),
        ):
            gain_scale = np.asarray(getattr(self, name), dtype=np.float64).reshape(-1)
            if gain_scale.shape != (DUAL_ARM_DOF,):
                raise ValueError(f"arm command {label} must contain {DUAL_ARM_DOF} joints")
            if not np.all(np.isfinite(gain_scale)) or np.any(gain_scale < 0):
                raise ValueError(f"arm command {label} must be finite and non-negative")
            gain_scales[name] = gain_scale
        torque = np.asarray(self.tau_ff14, dtype=np.float64).reshape(-1)
        if torque.shape != (DUAL_ARM_DOF,):
            raise ValueError(f"arm command torque feedforward must contain {DUAL_ARM_DOF} joints")
        if not np.all(np.isfinite(torque)):
            raise ValueError("arm command torque feedforward contains NaN or infinity")
        object.__setattr__(self, "q14", tuple(float(value) for value in array))
        for name, gain_scale in gain_scales.items():
            object.__setattr__(
                self,
                name,
                tuple(float(value) for value in gain_scale),
            )
        object.__setattr__(
            self,
            "tau_ff14",
            tuple(float(value) for value in torque),
        )

    @classmethod
    def create(
        cls,
        q14: Sequence[float] | np.ndarray,
        *,
        weight: float,
        issued_monotonic_s: float,
        emergency_release: bool = False,
        kp_scale14: Sequence[float] | np.ndarray | None = None,
        kd_scale14: Sequence[float] | np.ndarray | None = None,
        tau_ff14: Sequence[float] | np.ndarray | None = None,
    ) -> ArmCommand:
        kp_scales = (1.0,) * DUAL_ARM_DOF if kp_scale14 is None else tuple(kp_scale14)
        kd_scales = (1.0,) * DUAL_ARM_DOF if kd_scale14 is None else tuple(kd_scale14)
        torque = (0.0,) * DUAL_ARM_DOF if tau_ff14 is None else tuple(tau_ff14)
        return cls(
            q14=tuple(q14),
            weight=weight,
            issued_monotonic_s=issued_monotonic_s,
            emergency_release=emergency_release,
            kp_scale14=kp_scales,
            kd_scale14=kd_scales,
            tau_ff14=torque,
        )


class ArmTransport(Protocol):
    """The only object permitted to touch the arm command channel."""

    def observe(self) -> RobotStateSample: ...

    def send_command(self, command: ArmCommand) -> None: ...

    def close(self) -> None: ...

    def close_after_external_takeover(self) -> None:
        """Close after a verified external controller has taken ownership."""
        ...
