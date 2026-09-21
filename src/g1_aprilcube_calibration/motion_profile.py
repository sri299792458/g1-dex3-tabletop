"""Small independently testable joint-space motion primitives."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from g1_aprilcube_calibration.joint_map import DUAL_ARM_DOF


def velocity_limited_step(
    current: Sequence[float] | np.ndarray,
    goal: Sequence[float] | np.ndarray,
    *,
    maximum_velocity_rad_s: float,
    duration_s: float,
) -> np.ndarray:
    if maximum_velocity_rad_s <= 0:
        raise ValueError("maximum_velocity_rad_s must be positive")
    if not np.isfinite(duration_s) or duration_s < 0:
        raise ValueError("duration_s must be finite and non-negative")
    current_array = np.asarray(current, dtype=np.float64).reshape(-1)
    goal_array = np.asarray(goal, dtype=np.float64).reshape(-1)
    if current_array.shape != (DUAL_ARM_DOF,) or goal_array.shape != (DUAL_ARM_DOF,):
        raise ValueError(f"current and goal must each contain {DUAL_ARM_DOF} joints")
    if not np.all(np.isfinite(current_array)) or not np.all(np.isfinite(goal_array)):
        raise ValueError("motion profile inputs must be finite")
    limit = maximum_velocity_rad_s * duration_s
    result = current_array + np.clip(goal_array - current_array, -limit, limit)
    result.setflags(write=False)
    return result
