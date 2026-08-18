from __future__ import annotations

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from g1_dex3_tabletop.continuous_state_estimation_replay import (
    StateSeries,
    _interpolate_vector_and_quaternion,
    _motion_intervals,
    _slerp_wxyz,
)


def _wxyz(rotation: Rotation) -> np.ndarray:
    xyzw = rotation.as_quat()
    return xyzw[[3, 0, 1, 2]]


def test_quaternion_slerp_uses_shortest_arc() -> None:
    identity = np.asarray([1.0, 0.0, 0.0, 0.0])
    target = -_wxyz(Rotation.from_euler("z", 90.0, degrees=True))

    midpoint = _slerp_wxyz(identity, target, 0.5)

    angle_deg = Rotation.from_quat(midpoint[[1, 2, 3, 0]]).as_euler("zyx", degrees=True)[0]
    assert angle_deg == pytest.approx(45.0)


def test_state_interpolation_reports_nearest_sample_gap() -> None:
    times = np.asarray([0, 10_000_000], dtype=np.int64)
    vectors = np.asarray([[0.0, 2.0], [10.0, 4.0]])
    quaternions = np.stack(
        [
            _wxyz(Rotation.identity()),
            _wxyz(Rotation.from_euler("x", 20.0, degrees=True)),
        ]
    )

    vector, quaternion, gap_ms = _interpolate_vector_and_quaternion(
        times, vectors, quaternions, 2_500_000
    )

    assert vector == pytest.approx([2.5, 2.5])
    assert gap_ms == pytest.approx(2.5)
    angle_deg = Rotation.from_quat(quaternion[[1, 2, 3, 0]]).as_euler("xyz", degrees=True)[0]
    assert angle_deg == pytest.approx(5.0)


def test_motion_intervals_group_nearby_command_changes() -> None:
    changes = np.asarray([1_000_000_000, 1_050_000_000, 1_300_000_000], dtype=np.int64)

    intervals = _motion_intervals(changes)

    assert intervals == (
        (900_000_000, 1_150_000_000),
        (1_200_000_000, 1_400_000_000),
    )


def test_state_series_interpolates_both_imus() -> None:
    times = np.asarray([1_000_000, 3_000_000], dtype=np.int64)
    series = StateSeries(
        lowstate_time_ns=times,
        waist_q_rad=np.stack([np.zeros(3), np.ones(3)]),
        pelvis_quaternion_wxyz=np.stack(
            [_wxyz(Rotation.identity()), _wxyz(Rotation.from_euler("x", 0.2))]
        ),
        torso_time_ns=times,
        torso_quaternion_wxyz=np.stack(
            [_wxyz(Rotation.identity()), _wxyz(Rotation.from_euler("y", -0.4))]
        ),
        motion_intervals_ns=((1_400_000, 1_600_000),),
        statistics={},
    )

    sample, gaps = series.sample(2_000_000)

    assert sample.waist_q_rad == pytest.approx(np.full(3, 0.5))
    assert Rotation.from_matrix(
        sample.navigation_R_pelvis_imu.copy()
    ).magnitude() == pytest.approx(0.1)
    assert Rotation.from_matrix(sample.navigation_R_torso_imu.copy()).magnitude() == pytest.approx(
        0.2
    )
    assert gaps == {
        "lowstate_nearest_gap_ms": 1.0,
        "torso_imu_nearest_gap_ms": 1.0,
    }
    assert not series.command_is_changing(2_000_000)
    assert series.command_is_changing(1_500_000)
    assert series.contains(1_000_000)
    assert series.contains(3_000_000)
    assert not series.contains(999_999)
    assert not series.contains(3_000_001)
