"""Bounded pairing of the minimal Unitree camera-state inputs.

This module is an adapter around the pure estimator.  It contains no ROS,
planner, controller, or robot-command code and deliberately ignores every
joint outside the three-DOF pelvis-to-torso chain.
"""

from __future__ import annotations

import threading
from collections import deque
from dataclasses import dataclass

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from g1_aprilcube_calibration.models import RobotStateSample
from g1_aprilcube_calibration.transports.unitree_arm_sdk import IMUOrientationSample
from g1_dex3_tabletop.state_estimation import (
    WAIST_JOINT_INDICES,
    ProprioceptiveSample,
    rotation_from_wxyz,
)


@dataclass(frozen=True, slots=True)
class SynchronizedCameraStateInput:
    """One interpolated estimator input with timing evidence."""

    sample: ProprioceptiveSample
    lowstate_nearest_gap_s: float
    lowstate_bracket_span_s: float
    torso_imu_nearest_gap_s: float
    torso_imu_bracket_span_s: float

    def to_dict(self) -> dict[str, object]:
        return {
            "timestamp_ns": self.sample.timestamp_ns,
            "waist_q_rad": self.sample.waist_q_rad.tolist(),
            "lowstate_nearest_gap_s": self.lowstate_nearest_gap_s,
            "lowstate_bracket_span_s": self.lowstate_bracket_span_s,
            "torso_imu_nearest_gap_s": self.torso_imu_nearest_gap_s,
            "torso_imu_bracket_span_s": self.torso_imu_bracket_span_s,
        }


def _interpolation_indices(
    timestamps: np.ndarray,
    target_s: float,
    *,
    stream: str,
    maximum_gap_s: float,
) -> tuple[int, int, float, float, float]:
    if timestamps.size < 2:
        raise RuntimeError(f"{stream} has fewer than two samples")
    upper = int(np.searchsorted(timestamps, target_s, side="left"))
    if upper < len(timestamps) and timestamps[upper] == target_s:
        lower = upper
    else:
        if upper == 0 or upper == len(timestamps):
            raise RuntimeError(f"{stream} does not bracket estimator time {target_s:.9f}s")
        lower = upper - 1
    if lower == upper:
        nearest_gap_s = 0.0
        bracket_span_s = 0.0
        alpha = 0.0
    else:
        bracket_span_s = float(timestamps[upper] - timestamps[lower])
        nearest_gap_s = float(min(target_s - timestamps[lower], timestamps[upper] - target_s))
        if bracket_span_s > maximum_gap_s:
            raise RuntimeError(
                f"{stream} bracket spans {bracket_span_s:.6f}s; limit is {maximum_gap_s:.6f}s"
            )
        alpha = float((target_s - timestamps[lower]) / bracket_span_s)
    return lower, upper, alpha, nearest_gap_s, bracket_span_s


def _slerp_wxyz(first: np.ndarray, second: np.ndarray, alpha: float) -> np.ndarray:
    if alpha == 0.0:
        return rotation_from_wxyz(first)
    first_rotation = Rotation.from_matrix(rotation_from_wxyz(first))
    second_rotation = Rotation.from_matrix(rotation_from_wxyz(second))
    rotations = Rotation.concatenate((first_rotation, second_rotation))
    return Slerp([0.0, 1.0], rotations)([alpha]).as_matrix()[0]


class CameraStateInputBuffer:
    """Thread-safe bounded buffers for the four required measured quantities."""

    def __init__(self, *, maximum_samples: int = 4096) -> None:
        if maximum_samples < 2:
            raise ValueError("camera-state input buffer needs at least two samples")
        self._lowstate: deque[RobotStateSample] = deque(maxlen=maximum_samples)
        self._torso_imu: deque[IMUOrientationSample] = deque(maxlen=maximum_samples)
        self._lock = threading.Lock()

    def add_lowstate(self, sample: RobotStateSample) -> None:
        with self._lock:
            if self._lowstate and sample.receipt_monotonic_s <= (
                self._lowstate[-1].receipt_monotonic_s
            ):
                raise ValueError("LowState receipt times must be strictly increasing")
            if sample.pelvis_imu_quaternion_wxyz is None:
                raise ValueError("LowState has no pelvis IMU orientation")
            self._lowstate.append(sample)

    def add_torso_imu(self, sample: IMUOrientationSample) -> None:
        with self._lock:
            if self._torso_imu and sample.receipt_monotonic_s <= (
                self._torso_imu[-1].receipt_monotonic_s
            ):
                raise ValueError("torso IMU receipt times must be strictly increasing")
            self._torso_imu.append(sample)

    def latest(
        self,
        *,
        now_monotonic_s: float,
        maximum_age_s: float,
        maximum_gap_s: float,
    ) -> SynchronizedCameraStateInput:
        """Interpolate at the newest timestamp bracketed by both streams."""

        if not np.isfinite(now_monotonic_s) or now_monotonic_s < 0.0:
            raise ValueError("current time must be finite and non-negative")
        if not np.isfinite(maximum_age_s) or maximum_age_s <= 0.0:
            raise ValueError("camera-state maximum age must be finite and positive")
        with self._lock:
            if not self._lowstate or not self._torso_imu:
                raise RuntimeError("camera-state inputs are incomplete")
            target_s = min(
                self._lowstate[-1].receipt_monotonic_s,
                self._torso_imu[-1].receipt_monotonic_s,
            )
        if now_monotonic_s < target_s:
            raise ValueError("current time precedes the newest camera-state input")
        age_s = now_monotonic_s - target_s
        if age_s > maximum_age_s:
            raise RuntimeError(
                f"camera-state input is {age_s:.6f}s old; limit is {maximum_age_s:.6f}s"
            )
        return self.sample_at(target_s, maximum_gap_s=maximum_gap_s)

    def sample_at(
        self,
        target_monotonic_s: float,
        *,
        maximum_gap_s: float,
    ) -> SynchronizedCameraStateInput:
        """Interpolate both streams at one caller-selected monotonic time."""

        if not np.isfinite(target_monotonic_s) or target_monotonic_s < 0.0:
            raise ValueError("camera-state target time must be finite and non-negative")
        if not np.isfinite(maximum_gap_s) or maximum_gap_s <= 0.0:
            raise ValueError("camera-state maximum gap must be finite and positive")
        with self._lock:
            lowstate = tuple(self._lowstate)
            torso_imu = tuple(self._torso_imu)
        low_times = np.asarray(
            [sample.receipt_monotonic_s for sample in lowstate], dtype=np.float64
        )
        torso_times = np.asarray(
            [sample.receipt_monotonic_s for sample in torso_imu], dtype=np.float64
        )
        low_lower, low_upper, low_alpha, low_nearest, low_span = _interpolation_indices(
            low_times,
            target_monotonic_s,
            stream="LowState",
            maximum_gap_s=maximum_gap_s,
        )
        torso_lower, torso_upper, torso_alpha, torso_nearest, torso_span = _interpolation_indices(
            torso_times,
            target_monotonic_s,
            stream="torso IMU",
            maximum_gap_s=maximum_gap_s,
        )
        low_first = lowstate[low_lower]
        low_second = lowstate[low_upper]
        first_waist = low_first.position[list(WAIST_JOINT_INDICES)]
        second_waist = low_second.position[list(WAIST_JOINT_INDICES)]
        waist = (1.0 - low_alpha) * first_waist + low_alpha * second_waist
        pelvis_rotation = _slerp_wxyz(
            low_first.pelvis_imu_quaternion_wxyz,
            low_second.pelvis_imu_quaternion_wxyz,
            low_alpha,
        )
        torso_rotation = _slerp_wxyz(
            torso_imu[torso_lower].quaternion_wxyz,
            torso_imu[torso_upper].quaternion_wxyz,
            torso_alpha,
        )
        return SynchronizedCameraStateInput(
            sample=ProprioceptiveSample(
                timestamp_ns=round(target_monotonic_s * 1.0e9),
                waist_q_rad=waist,
                navigation_R_pelvis_imu=pelvis_rotation,
                navigation_R_torso_imu=torso_rotation,
            ),
            lowstate_nearest_gap_s=low_nearest,
            lowstate_bracket_span_s=low_span,
            torso_imu_nearest_gap_s=torso_nearest,
            torso_imu_bracket_span_s=torso_span,
        )
