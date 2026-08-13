from __future__ import annotations

import numpy as np
import pytest

from g1_aprilcube_calibration.activation_handoff import (
    build_activation_handoff,
    trailing_stationary_window,
)
from g1_aprilcube_calibration.models import RobotStateSample
from g1_aprilcube_calibration.pose_schema import PoseSet
from g1_aprilcube_calibration.readiness import RecordingGateConfig

UTC = "2026-08-03T12:00:00Z"


def pose_set() -> PoseSet:
    return PoseSet(
        robot_model="g1_29dof_rev_1_0",
        mode_machine=5,
        urdf_sha256="a" * 64,
        calibration_arm="left",
    )


def config() -> RecordingGateConfig:
    return RecordingGateConfig(
        calibration_arm="left",
        state_freshness_timeout_s=0.1,
        stationary_duration_s=0.5,
        maximum_state_gap_s=0.11,
        maximum_calibration_position_spread_rad=0.01,
        maximum_hold_position_spread_rad=0.01,
        minimum_samples=5,
    )


def sample(time_s: float, offset: float = 0.0) -> RobotStateSample:
    position = np.arange(29, dtype=np.float64) / 100.0 + offset
    return RobotStateSample(
        receipt_monotonic_s=time_s,
        receipt_utc=UTC,
        mode_machine=5,
        position=position,
        velocity=np.full(29, 0.05),
        estimated_torque=np.arange(29, dtype=np.float64) + offset,
        source_sequence=int(time_s * 10),
    )


def test_trailing_window_discards_older_motion() -> None:
    samples = [sample(index / 10, 0.2 if index < 3 else 0.0) for index in range(10)]

    window = trailing_stationary_window(samples, duration_s=0.5)

    assert window[0].receipt_monotonic_s == pytest.approx(0.4)
    assert window[-1].receipt_monotonic_s == pytest.approx(0.9)


def test_build_activation_replaces_only_run_handoff_from_window_median() -> None:
    source = pose_set()
    samples = [sample(1.0 + index / 10, index * 0.0001) for index in range(6)]

    result = build_activation_handoff(
        source,
        samples,
        now_monotonic_s=1.5,
        config=config(),
    )

    expected = np.arange(29, dtype=np.float64) / 100.0 + 0.00025
    np.testing.assert_allclose(result.handoff_q, expected[15:22])
    np.testing.assert_allclose(result.hold_q, expected[22:29])
    np.testing.assert_allclose(result.reference_state.position, expected)
    np.testing.assert_allclose(
        result.reference_state.estimated_torque,
        np.arange(29, dtype=np.float64) + 0.00025,
    )
    assert result.pose_set.poses == source.poses
    assert result.source_pose_set_sha256 == source.content_sha256
    assert result.pose_set.content_sha256 == source.content_sha256
    assert result.readiness.ready


def test_build_activation_rejects_a_moving_window() -> None:
    samples = [sample(1.0 + index / 10) for index in range(6)]
    samples[-1].position.setflags(write=True)
    samples[-1].position[15] += 0.02

    with pytest.raises(ValueError, match="not stationary"):
        build_activation_handoff(
            pose_set(),
            samples,
            now_monotonic_s=1.5,
            config=config(),
        )


def test_build_activation_retains_raw_dq_only_as_diagnostic() -> None:
    samples = [sample(1.0 + index / 10) for index in range(6)]
    samples[-1].velocity.setflags(write=True)
    samples[-1].velocity[15] = 10.0

    result = build_activation_handoff(
        pose_set(),
        samples,
        now_monotonic_s=1.5,
        config=config(),
    )

    assert result.readiness.ready
    assert result.readiness.maximum_calibration_velocity_rad_s == pytest.approx(10.0)
