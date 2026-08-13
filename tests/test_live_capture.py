from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest
from aprilcube import CorrespondenceDetector
from aprilcube.generate import DICT_MAP

from g1_aprilcube_calibration.camera_models import RectifiedCameraInfo
from g1_aprilcube_calibration.clock import ManualClock
from g1_aprilcube_calibration.config import QualityThresholds
from g1_aprilcube_calibration.live_capture import LiveBurstConfig, LiveBurstFrameSource
from g1_aprilcube_calibration.models import RobotStateSample
from g1_aprilcube_calibration.quality import PoseQualityEvaluator
from g1_aprilcube_calibration.readiness import RecordingGateConfig, StateSampleBuffer
from g1_aprilcube_calibration.ros.camera_adapter import ROSFrameBuffer, ROSImageFrame
from g1_aprilcube_calibration.session_runner import RecoverableCaptureError
from g1_aprilcube_calibration.timestamp_pairing import ImageTiming, PairingConfig

ROOT = Path(__file__).parents[1]
TARGET = ROOT / "third_party" / "aprilcube" / "models" / "dex3_safe_cube" / "config.json"
UTC = "2026-08-02T12:00:00Z"


def marker_image():
    image = np.full((480, 640, 3), 220, dtype=np.uint8)
    dictionary = cv2.aruco.getPredefinedDictionary(DICT_MAP["4x4_100"])
    marker = cv2.aruco.generateImageMarker(dictionary, 0, 180)
    image[140:320, 230:410] = cv2.cvtColor(marker, cv2.COLOR_GRAY2BGR)
    return image


def info():
    return RectifiedCameraInfo(
        640,
        480,
        "color_optical",
        "head_color",
        "TEST",
        "plumb_bob",
        (0.0,) * 5,
        (600.0, 0.0, 319.5, 0.0, 600.0, 239.5, 0.0, 0.0, 1.0),
        (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0),
        (600.0, 0.0, 319.5, 0.0, 0.0, 600.0, 239.5, 0.0, 0.0, 0.0, 1.0, 0.0),
    )


def thresholds():
    return QualityThresholds.from_yaml(ROOT / "config" / "capture_quality_dex3_aruco.yaml")


def test_live_source_waits_for_new_stationary_confirmed_frames():
    frames = ROSFrameBuffer()
    states = StateSampleBuffer()
    for time_s in np.arange(0.5, 2.01, 0.02):
        states.add(RobotStateSample(time_s, UTC, 5, np.zeros(29), np.zeros(29), np.zeros(29)))
    scheduled = [
        ROSImageFrame(marker_image(), ImageTiming(time_s, UTC, index), info())
        for index, time_s in enumerate((1.0, 1.5), start=1)
    ]
    clock = ManualClock()

    def wait_once(duration):
        clock.advance(duration)
        if scheduled:
            frames.add(scheduled.pop(0))

    source = LiveBurstFrameSource(
        camera_frames=frames,
        robot_states=states,
        detector=CorrespondenceDetector(TARGET),
        quality_evaluator=PoseQualityEvaluator(thresholds()),
        recording_config=RecordingGateConfig(
            calibration_arm="left",
            state_freshness_timeout_s=0.1,
            stationary_duration_s=0.5,
            maximum_state_gap_s=0.1,
            maximum_calibration_position_spread_rad=0.01,
            minimum_samples=5,
        ),
        pairing_config=PairingConfig(0.05, 0.1),
        config=LiveBurstConfig(frame_count=2, timeout_s=1, poll_interval_s=0.01),
        clock=clock,
        wait_once=wait_once,
        accept_yellow=lambda _frame: True,
    )
    burst = source.capture_burst(pose_id="pose", capture_id="capture_001")
    assert [frame.frame_id for frame in burst] == [
        "capture_001_000",
        "capture_001_001",
    ]
    assert all(frame.correspondences.valid for frame in burst)
    assert all(len(frame.state_window) >= 25 for frame in burst)


def test_live_source_can_take_supported_burst_without_changing_view_history():
    frames = ROSFrameBuffer()
    states = StateSampleBuffer()
    for time_s in np.arange(0.5, 1.81, 0.02):
        states.add(RobotStateSample(time_s, UTC, 5, np.zeros(29), np.zeros(29), np.zeros(29)))
    clock = ManualClock()
    scheduled = [
        ROSImageFrame(marker_image(), ImageTiming(time_s, UTC, index), info())
        for index, time_s in enumerate((1.0, 1.2, 1.4), start=1)
    ]

    def wait_once(duration):
        clock.advance(duration)
        if scheduled:
            frames.add(scheduled.pop(0))

    source = LiveBurstFrameSource(
        camera_frames=frames,
        robot_states=states,
        detector=CorrespondenceDetector(TARGET),
        quality_evaluator=PoseQualityEvaluator(thresholds()),
        recording_config=RecordingGateConfig(
            calibration_arm="left",
            state_freshness_timeout_s=0.1,
            stationary_duration_s=0.5,
            maximum_state_gap_s=0.1,
            maximum_calibration_position_spread_rad=0.01,
            minimum_samples=5,
        ),
        pairing_config=PairingConfig(0.05, 0.1),
        config=LiveBurstConfig(frame_count=3, timeout_s=1, poll_interval_s=0.01),
        clock=clock,
        wait_once=wait_once,
        accept_yellow=lambda _frame: True,
    )

    burst = source.capture_burst(
        pose_id="pose",
        capture_id="capture_001_supported",
        remember_signature=False,
    )

    assert [frame.frame_id for frame in burst] == [
        "capture_001_supported_000",
        "capture_001_supported_001",
        "capture_001_supported_002",
    ]
    with pytest.raises(ValueError, match="empty live-capture history"):
        source.undo_last_signature()


def test_visual_red_timeout_is_recoverable_for_replay() -> None:
    frames = ROSFrameBuffer()
    states = StateSampleBuffer()
    for time_s in np.arange(0.5, 2.01, 0.02):
        states.add(RobotStateSample(time_s, UTC, 5, np.zeros(29), np.zeros(29), np.zeros(29)))
    scheduled = [
        ROSImageFrame(
            np.full((480, 640, 3), 220, dtype=np.uint8),
            ImageTiming(1.0, UTC, 1),
            info(),
        )
    ]
    clock = ManualClock()

    def wait_once(duration):
        clock.advance(duration)
        if scheduled:
            frames.add(scheduled.pop(0))

    source = LiveBurstFrameSource(
        camera_frames=frames,
        robot_states=states,
        detector=CorrespondenceDetector(TARGET),
        quality_evaluator=PoseQualityEvaluator(thresholds()),
        recording_config=RecordingGateConfig(
            calibration_arm="left",
            state_freshness_timeout_s=0.1,
            stationary_duration_s=0.5,
            maximum_state_gap_s=0.1,
            maximum_calibration_position_spread_rad=0.01,
            minimum_samples=5,
        ),
        pairing_config=PairingConfig(0.05, 0.1),
        config=LiveBurstConfig(frame_count=1, timeout_s=0.1, poll_interval_s=0.01),
        clock=clock,
        wait_once=wait_once,
        accept_yellow=lambda _frame: True,
    )

    with pytest.raises(RecoverableCaptureError, match="visual quality is red"):
        source.capture_burst(pose_id="pose", capture_id="capture_001")


def test_live_source_keeps_frame_until_future_state_bracket_arrives():
    frames = ROSFrameBuffer()
    states = StateSampleBuffer()
    for time_s in np.arange(0.5, 1.01, 0.02):
        states.add(RobotStateSample(time_s, UTC, 5, np.zeros(29), np.zeros(29), np.zeros(29)))
    clock = ManualClock()
    frame_added = [False]

    def wait_once(duration):
        clock.advance(duration)
        if not frame_added[0]:
            frames.add(ROSImageFrame(marker_image(), ImageTiming(1.0, UTC, 1), info()))
            frame_added[0] = True
            return
        next_time = states.latest.receipt_monotonic_s + 0.02
        states.add(RobotStateSample(next_time, UTC, 5, np.zeros(29), np.zeros(29), np.zeros(29)))

    source = LiveBurstFrameSource(
        camera_frames=frames,
        robot_states=states,
        detector=CorrespondenceDetector(TARGET),
        quality_evaluator=PoseQualityEvaluator(thresholds()),
        recording_config=RecordingGateConfig(
            calibration_arm="left",
            state_freshness_timeout_s=0.1,
            stationary_duration_s=0.5,
            maximum_state_gap_s=0.1,
            maximum_calibration_position_spread_rad=0.01,
            minimum_samples=5,
        ),
        pairing_config=PairingConfig(0.05, 0.1),
        config=LiveBurstConfig(frame_count=1, timeout_s=1, poll_interval_s=0.01),
        clock=clock,
        wait_once=wait_once,
        accept_yellow=lambda _frame: True,
    )
    burst = source.capture_burst(pose_id="pose", capture_id="capture_001")
    assert burst[0].frame_id == "capture_001_000"
    assert burst[0].state_window[-1].receipt_monotonic_s >= 1.25


def test_live_source_restarts_burst_after_excessive_duration_and_arm_move():
    frames = ROSFrameBuffer()
    states = StateSampleBuffer()
    for time_s in np.arange(0.5, 4.51, 0.01):
        position = np.zeros(29)
        if time_s >= 2.0:
            position[15] = 0.5
        states.add(RobotStateSample(time_s, UTC, 5, position, np.zeros(29), np.zeros(29)))
    scheduled = [
        ROSImageFrame(marker_image(), ImageTiming(time_s, UTC, index), info())
        for index, time_s in enumerate((1.0, 4.0, 4.1), start=1)
    ]
    clock = ManualClock()
    restarts = []

    def wait_once(duration):
        clock.advance(duration)
        if scheduled:
            frames.add(scheduled.pop(0))

    source = LiveBurstFrameSource(
        camera_frames=frames,
        robot_states=states,
        detector=CorrespondenceDetector(TARGET),
        quality_evaluator=PoseQualityEvaluator(thresholds()),
        recording_config=RecordingGateConfig(
            calibration_arm="left",
            state_freshness_timeout_s=0.1,
            stationary_duration_s=0.5,
            maximum_state_gap_s=0.1,
            maximum_calibration_position_spread_rad=0.01,
            minimum_samples=5,
        ),
        pairing_config=PairingConfig(0.05, 0.1),
        config=LiveBurstConfig(frame_count=2, timeout_s=1, poll_interval_s=0.01),
        clock=clock,
        wait_once=wait_once,
        accept_yellow=lambda _frame: True,
        report_burst_restart=lambda pose_id, capture_id, reason: restarts.append(
            (pose_id, capture_id, reason)
        ),
    )

    burst = source.capture_burst(pose_id="pose", capture_id="capture_001")

    assert [frame.image_timing.receipt_monotonic_s for frame in burst] == [4.0, 4.1]
    assert [frame.frame_id for frame in burst] == [
        "capture_001_000",
        "capture_001_001",
    ]
    assert restarts == [("pose", "capture_001", "burst duration is 3.000s; limit is 2.000s")]


def test_live_source_restarts_when_individually_stable_frames_span_arm_motion():
    frames = ROSFrameBuffer()
    states = StateSampleBuffer()
    for time_s in np.arange(0.5, 1.61, 0.01):
        position = np.zeros(29)
        if time_s >= 1.1:
            position[15] = 0.5
        states.add(RobotStateSample(time_s, UTC, 5, position, np.zeros(29), np.zeros(29)))
    scheduled = [
        ROSImageFrame(marker_image(), ImageTiming(time_s, UTC, index), info())
        for index, time_s in enumerate((1.0, 1.2, 1.3), start=1)
    ]
    clock = ManualClock()

    def wait_once(duration):
        clock.advance(duration)
        if scheduled:
            frames.add(scheduled.pop(0))

    source = LiveBurstFrameSource(
        camera_frames=frames,
        robot_states=states,
        detector=CorrespondenceDetector(TARGET),
        quality_evaluator=PoseQualityEvaluator(thresholds()),
        recording_config=RecordingGateConfig(
            calibration_arm="left",
            state_freshness_timeout_s=0.1,
            stationary_duration_s=0.1,
            maximum_state_gap_s=0.02,
            maximum_calibration_position_spread_rad=0.01,
            minimum_samples=5,
        ),
        pairing_config=PairingConfig(0.05, 0.1),
        config=LiveBurstConfig(frame_count=2, timeout_s=1, poll_interval_s=0.01),
        clock=clock,
        wait_once=wait_once,
        accept_yellow=lambda _frame: True,
    )

    burst = source.capture_burst(pose_id="pose", capture_id="capture_001")

    assert [frame.image_timing.receipt_monotonic_s for frame in burst] == [1.2, 1.3]


def test_live_source_restarts_when_total_burst_duration_is_too_long():
    frames = ROSFrameBuffer()
    states = StateSampleBuffer()
    for time_s in np.arange(0.5, 2.51, 0.01):
        states.add(RobotStateSample(time_s, UTC, 5, np.zeros(29), np.zeros(29), np.zeros(29)))
    scheduled = [
        ROSImageFrame(marker_image(), ImageTiming(time_s, UTC, index), info())
        for index, time_s in enumerate((1.0, 1.4, 1.8, 1.9, 2.0), start=1)
    ]
    clock = ManualClock()

    def wait_once(duration):
        clock.advance(duration)
        if scheduled:
            frames.add(scheduled.pop(0))

    source = LiveBurstFrameSource(
        camera_frames=frames,
        robot_states=states,
        detector=CorrespondenceDetector(TARGET),
        quality_evaluator=PoseQualityEvaluator(thresholds()),
        recording_config=RecordingGateConfig(
            calibration_arm="left",
            state_freshness_timeout_s=0.1,
            stationary_duration_s=0.5,
            maximum_state_gap_s=0.1,
            maximum_calibration_position_spread_rad=0.01,
            minimum_samples=5,
        ),
        pairing_config=PairingConfig(0.05, 0.1),
        config=LiveBurstConfig(
            frame_count=3,
            timeout_s=1,
            poll_interval_s=0.01,
            maximum_duration_s=0.7,
        ),
        clock=clock,
        wait_once=wait_once,
        accept_yellow=lambda _frame: True,
    )

    burst = source.capture_burst(pose_id="pose", capture_id="capture_001")

    assert [frame.image_timing.receipt_monotonic_s for frame in burst] == [
        1.8,
        1.9,
        2.0,
    ]


def test_live_source_accepts_stationary_frames_across_transport_gap():
    frames = ROSFrameBuffer()
    states = StateSampleBuffer()
    for time_s in np.arange(0.5, 2.51, 0.01):
        states.add(RobotStateSample(time_s, UTC, 5, np.zeros(29), np.zeros(29), np.zeros(29)))
    scheduled = [
        ROSImageFrame(marker_image(), ImageTiming(time_s, UTC, index), info())
        for index, time_s in enumerate((1.0, 2.0), start=1)
    ]
    clock = ManualClock()

    def wait_once(duration):
        clock.advance(duration)
        if scheduled:
            frames.add(scheduled.pop(0))

    source = LiveBurstFrameSource(
        camera_frames=frames,
        robot_states=states,
        detector=CorrespondenceDetector(TARGET),
        quality_evaluator=PoseQualityEvaluator(thresholds()),
        recording_config=RecordingGateConfig(
            calibration_arm="left",
            state_freshness_timeout_s=0.1,
            stationary_duration_s=0.5,
            maximum_state_gap_s=0.1,
            maximum_calibration_position_spread_rad=0.01,
            minimum_samples=5,
        ),
        pairing_config=PairingConfig(0.05, 0.1),
        config=LiveBurstConfig(frame_count=2, timeout_s=0.1, poll_interval_s=0.01),
        clock=clock,
        wait_once=wait_once,
        accept_yellow=lambda _frame: True,
    )

    burst = source.capture_burst(pose_id="pose", capture_id="capture_001")

    assert [frame.image_timing.receipt_monotonic_s for frame in burst] == [1.0, 2.0]
