import numpy as np
import pytest

from g1_aprilcube_calibration.models import RobotStateSample, validate_utc_iso


def make_sample() -> RobotStateSample:
    return RobotStateSample(
        receipt_monotonic_s=10.0,
        receipt_utc="2026-08-02T12:00:00Z",
        mode_machine=5,
        position=np.arange(29, dtype=float) / 10.0,
        velocity=np.zeros(29),
        estimated_torque=-np.arange(29, dtype=float),
        source_sequence=7,
    )


def test_robot_state_round_trip_and_arm_views() -> None:
    sample = make_sample()
    restored = RobotStateSample.from_dict(sample.to_dict())

    assert restored.is_mode5
    assert np.array_equal(restored.position, sample.position)
    assert np.array_equal(restored.right_q, sample.position[22:29])
    assert np.array_equal(restored.left_q, sample.position[15:22])
    assert np.array_equal(restored.left_tau_est, sample.estimated_torque[15:22])
    assert np.array_equal(restored.right_tau_est, sample.estimated_torque[22:29])
    assert restored.age_s(10.25) == pytest.approx(0.25)
    assert not restored.position.flags.writeable


def test_robot_state_requires_tau_est_in_serialized_input() -> None:
    serialized = make_sample().to_dict()
    del serialized["estimated_torque"]

    with pytest.raises(KeyError, match="estimated_torque"):
        RobotStateSample.from_dict(serialized)


def test_robot_state_rejects_bad_state() -> None:
    with pytest.raises(ValueError, match="exactly 29"):
        RobotStateSample(
            0.0,
            "2026-08-02T12:00:00Z",
            5,
            np.zeros(28),
            np.zeros(29),
            np.zeros(29),
        )
    with pytest.raises(ValueError, match="precedes"):
        make_sample().age_s(9.0)
    with pytest.raises(TypeError, match="integer"):
        RobotStateSample(
            0.0,
            "2026-08-02T12:00:00Z",
            True,
            np.zeros(29),
            np.zeros(29),
            np.zeros(29),
        )
    with pytest.raises(ValueError, match="estimated joint torque"):
        RobotStateSample(
            0.0,
            "2026-08-02T12:00:00Z",
            5,
            np.zeros(29),
            np.zeros(29),
            np.full(29, np.nan),
        )


def test_utc_validation_rejects_naive_and_non_utc_timestamps() -> None:
    assert validate_utc_iso("2026-08-02T12:00:00Z") == "2026-08-02T12:00:00Z"
    with pytest.raises(ValueError, match="timezone"):
        validate_utc_iso("2026-08-02T12:00:00")
    with pytest.raises(ValueError, match="zero UTC offset"):
        validate_utc_iso("2026-08-02T12:00:00-05:00")
