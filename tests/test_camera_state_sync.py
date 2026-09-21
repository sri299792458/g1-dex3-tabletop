from __future__ import annotations

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from g1_aprilcube_calibration.models import RobotStateSample
from g1_aprilcube_calibration.transports.unitree_arm_sdk import IMUOrientationSample
from g1_dex3_tabletop.camera_state_sync import CameraStateInputBuffer
from g1_dex3_tabletop.state_estimation import WAIST_JOINT_INDICES


def _wxyz(axis: str, degrees: float) -> np.ndarray:
    xyzw = Rotation.from_euler(axis, degrees, degrees=True).as_quat()
    return xyzw[[3, 0, 1, 2]]


def _lowstate(
    timestamp_s: float,
    *,
    waist: tuple[float, float, float],
    pelvis_yaw_deg: float,
) -> RobotStateSample:
    position = np.zeros(29, dtype=np.float64)
    position[list(WAIST_JOINT_INDICES)] = waist
    return RobotStateSample(
        receipt_monotonic_s=timestamp_s,
        receipt_utc="2026-08-17T00:00:00Z",
        mode_machine=5,
        position=position,
        velocity=np.zeros(29, dtype=np.float64),
        estimated_torque=np.zeros(29, dtype=np.float64),
        pelvis_imu_quaternion_wxyz=_wxyz("z", pelvis_yaw_deg),
    )


def _torso(timestamp_s: float, *, pitch_deg: float) -> IMUOrientationSample:
    return IMUOrientationSample(
        receipt_monotonic_s=timestamp_s,
        quaternion_wxyz=_wxyz("y", pitch_deg),
    )


def _rotation_error_deg(first: np.ndarray, second: np.ndarray) -> float:
    return float(np.degrees(Rotation.from_matrix(first.T @ second).magnitude()))


def test_camera_state_input_interpolates_all_selected_inputs_at_one_time() -> None:
    buffer = CameraStateInputBuffer()
    buffer.add_lowstate(_lowstate(10.0, waist=(0.0, 0.1, 0.2), pelvis_yaw_deg=0.0))
    buffer.add_lowstate(_lowstate(10.2, waist=(0.2, 0.3, 0.4), pelvis_yaw_deg=20.0))
    buffer.add_torso_imu(_torso(9.95, pitch_deg=-5.0))
    buffer.add_torso_imu(_torso(10.15, pitch_deg=15.0))

    paired = buffer.sample_at(10.1, maximum_gap_s=0.25)

    np.testing.assert_allclose(paired.sample.waist_q_rad, [0.1, 0.2, 0.3])
    assert _rotation_error_deg(
        paired.sample.navigation_R_pelvis_imu,
        Rotation.from_euler("z", 10.0, degrees=True).as_matrix(),
    ) == pytest.approx(0.0, abs=1.0e-10)
    assert _rotation_error_deg(
        paired.sample.navigation_R_torso_imu,
        Rotation.from_euler("y", 10.0, degrees=True).as_matrix(),
    ) == pytest.approx(0.0, abs=1.0e-10)
    assert paired.lowstate_bracket_span_s == pytest.approx(0.2)
    assert paired.torso_imu_bracket_span_s == pytest.approx(0.2)


def test_latest_uses_newest_time_available_from_both_streams() -> None:
    buffer = CameraStateInputBuffer()
    buffer.add_lowstate(_lowstate(1.0, waist=(0.0, 0.0, 0.0), pelvis_yaw_deg=0.0))
    buffer.add_lowstate(_lowstate(1.2, waist=(0.2, 0.2, 0.2), pelvis_yaw_deg=0.0))
    buffer.add_torso_imu(_torso(1.0, pitch_deg=0.0))
    buffer.add_torso_imu(_torso(1.1, pitch_deg=1.0))

    paired = buffer.latest(
        now_monotonic_s=1.15,
        maximum_age_s=0.1,
        maximum_gap_s=0.25,
    )

    assert paired.sample.timestamp_ns == 1_100_000_000
    np.testing.assert_allclose(paired.sample.waist_q_rad, [0.1, 0.1, 0.1])
    assert paired.torso_imu_bracket_span_s == 0.0


def test_pairing_rejects_stale_or_widely_bracketed_inputs() -> None:
    buffer = CameraStateInputBuffer()
    buffer.add_lowstate(_lowstate(1.0, waist=(0.0, 0.0, 0.0), pelvis_yaw_deg=0.0))
    buffer.add_lowstate(_lowstate(1.3, waist=(0.0, 0.0, 0.0), pelvis_yaw_deg=0.0))
    buffer.add_torso_imu(_torso(1.0, pitch_deg=0.0))
    buffer.add_torso_imu(_torso(1.3, pitch_deg=0.0))

    with pytest.raises(RuntimeError, match="bracket spans"):
        buffer.sample_at(1.1, maximum_gap_s=0.2)
    with pytest.raises(RuntimeError, match="old"):
        buffer.latest(
            now_monotonic_s=1.5,
            maximum_age_s=0.1,
            maximum_gap_s=0.4,
        )


def test_pairing_rejects_missing_pelvis_orientation_and_out_of_order_samples() -> None:
    buffer = CameraStateInputBuffer()
    sample = _lowstate(1.0, waist=(0.0, 0.0, 0.0), pelvis_yaw_deg=0.0)
    without_imu = RobotStateSample(
        receipt_monotonic_s=1.0,
        receipt_utc=sample.receipt_utc,
        mode_machine=sample.mode_machine,
        position=sample.position,
        velocity=sample.velocity,
        estimated_torque=sample.estimated_torque,
    )
    with pytest.raises(ValueError, match="no pelvis IMU"):
        buffer.add_lowstate(without_imu)

    buffer.add_lowstate(sample)
    with pytest.raises(ValueError, match="strictly increasing"):
        buffer.add_lowstate(sample)
