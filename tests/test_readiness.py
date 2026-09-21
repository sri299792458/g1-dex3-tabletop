import numpy as np
import pytest

from g1_aprilcube_calibration.models import RobotStateSample
from g1_aprilcube_calibration.readiness import (
    RecordingGateConfig,
    StateSampleBuffer,
    evaluate_recording_window,
)


def sample(time_s: float, *, mode: int = 5, dq: float = 0.0, q_offset: float = 0.0):
    q = np.arange(29, dtype=float) / 100.0
    q[15:22] += q_offset
    velocity = np.zeros(29)
    velocity[15:22] = dq
    return RobotStateSample(
        time_s,
        "2026-08-02T12:00:00Z",
        mode,
        q,
        velocity,
        np.zeros(29),
    )


def config() -> RecordingGateConfig:
    return RecordingGateConfig(
        calibration_arm="left",
        state_freshness_timeout_s=0.1,
        stationary_duration_s=0.4,
        maximum_state_gap_s=0.06,
        maximum_calibration_position_spread_rad=0.01,
        minimum_samples=5,
    )


def test_centered_stationary_window_is_ready() -> None:
    buffer = StateSampleBuffer()
    buffer.extend(sample(time_s) for time_s in np.arange(9.7, 10.31, 0.05))
    window = buffer.centered_window(center_monotonic_s=10.1, duration_s=0.4)
    report = evaluate_recording_window(window, now_monotonic_s=10.31, config=config())

    assert report.ready
    assert window[0].receipt_monotonic_s <= 9.9
    assert window[-1].receipt_monotonic_s >= 10.3
    assert report.calibration_arm == "left"
    assert report.maximum_calibration_velocity_rad_s == 0.0


@pytest.mark.parametrize(
    ("mutator", "message"),
    [
        (lambda index, time_s: sample(time_s, mode=4 if index == 4 else 5), "mode"),
        (
            lambda index, time_s: sample(time_s, q_offset=0.02 if index == 4 else 0.0),
            "spread",
        ),
    ],
)
def test_stationary_window_fails_closed(mutator, message: str) -> None:
    samples = [mutator(index, 10.0 + index * 0.05) for index in range(9)]
    report = evaluate_recording_window(samples, now_monotonic_s=10.41, config=config())
    assert not report.ready
    assert any(message in failure for failure in report.hard_failures)


def test_stale_state_and_large_gap_are_rejected() -> None:
    samples = [sample(10.0), sample(10.05), sample(10.1), sample(10.3), sample(10.4)]
    report = evaluate_recording_window(samples, now_monotonic_s=10.6, config=config())
    assert not report.ready
    assert any("stale" in reason for reason in report.hard_failures)
    assert any("gap" in reason for reason in report.hard_failures)


def test_raw_dq_spike_is_diagnostic_when_position_is_stationary() -> None:
    samples = [sample(10.0 + index * 0.05, dq=10.0 if index == 4 else 0.0) for index in range(9)]

    report = evaluate_recording_window(samples, now_monotonic_s=10.41, config=config())

    assert report.ready
    assert report.maximum_calibration_velocity_rad_s == pytest.approx(10.0)


def test_buffer_rejects_non_monotonic_receipt_times() -> None:
    buffer = StateSampleBuffer()
    buffer.add(sample(1.0))
    with pytest.raises(ValueError, match="strictly increasing"):
        buffer.add(sample(1.0))
