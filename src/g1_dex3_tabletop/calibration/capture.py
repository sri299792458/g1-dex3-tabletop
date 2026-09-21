"""Hardware-neutral same-frame bilateral burst acquisition."""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace

import numpy as np
from aprilcube import CorrespondenceDetector, CorrespondenceResult

from g1_aprilcube_calibration.camera_models import RectifiedCameraInfo
from g1_aprilcube_calibration.clock import MonotonicClock, SystemClock
from g1_aprilcube_calibration.live_capture import LiveBurstConfig
from g1_aprilcube_calibration.models import RobotStateSample
from g1_aprilcube_calibration.quality import (
    CameraIntrinsics,
    PoseQualityEvaluator,
    QualityGrade,
    QualityReport,
    ViewSignature,
)
from g1_aprilcube_calibration.readiness import (
    RecordingGateConfig,
    StateSampleBuffer,
    evaluate_recording_window,
)
from g1_aprilcube_calibration.ros.camera_adapter import ROSFrameBuffer, ROSImageFrame
from g1_aprilcube_calibration.session_runner import RecoverableCaptureError
from g1_aprilcube_calibration.timestamp_pairing import (
    ImageTiming,
    PairingConfig,
    PairingResult,
    pair_state_to_image,
)

_SIDES = ("left", "right")


@dataclass(frozen=True, slots=True)
class BilateralFrameEvidence:
    """One raw image with two detections and exactly one paired state window."""

    frame_id: str
    image_bgr: np.ndarray
    image_timing: ImageTiming
    camera_info: RectifiedCameraInfo
    state_window: tuple[RobotStateSample, ...]
    pairing: PairingResult
    left_correspondences: CorrespondenceResult
    right_correspondences: CorrespondenceResult
    left_quality: QualityReport
    right_quality: QualityReport

    def __post_init__(self) -> None:
        if not self.frame_id:
            raise ValueError("bilateral frame ID must be non-empty")
        image = np.asarray(self.image_bgr)
        if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
            raise ValueError("raw bilateral image must be uint8 BGR")
        if image.shape[:2] != (self.camera_info.height, self.camera_info.width):
            raise ValueError("raw bilateral image dimensions do not match CameraInfo")
        expected_size = (self.camera_info.width, self.camera_info.height)
        for side in _SIDES:
            correspondences = getattr(self, f"{side}_correspondences")
            quality = getattr(self, f"{side}_quality")
            if correspondences.image_size_wh != expected_size:
                raise ValueError(f"{side} correspondence dimensions do not match CameraInfo")
            if not correspondences.valid:
                raise ValueError(f"{side} target does not have valid correspondences")
            if quality.grade is QualityGrade.RED:
                raise ValueError(f"{side} target quality is red")
        if self.pairing.image != self.image_timing:
            raise ValueError("bilateral pairing belongs to a different image")
        if not self.state_window:
            raise ValueError("bilateral frame requires its complete state window")
        image = image.copy()
        image.setflags(write=False)
        object.__setattr__(self, "image_bgr", image)
        object.__setattr__(self, "state_window", tuple(self.state_window))

    @property
    def correspondences(self) -> dict[str, CorrespondenceResult]:
        return {
            "left": self.left_correspondences,
            "right": self.right_correspondences,
        }

    @property
    def qualities(self) -> dict[str, QualityReport]:
        return {"left": self.left_quality, "right": self.right_quality}


class BilateralLiveBurstSource:
    """Collect new stationary frames only when both hand targets pass quality."""

    def __init__(
        self,
        *,
        camera_frames: ROSFrameBuffer,
        robot_states: StateSampleBuffer,
        detectors: Mapping[str, CorrespondenceDetector],
        quality_evaluators: Mapping[str, PoseQualityEvaluator],
        recording_config: RecordingGateConfig,
        pairing_config: PairingConfig,
        config: LiveBurstConfig | None = None,
        clock: MonotonicClock | None = None,
        wait_once: Callable[[float], None] = time.sleep,
        accept_yellow: Callable[[ROSImageFrame, tuple[str, ...]], bool] | None = None,
        preview: Callable[
            [
                ROSImageFrame,
                dict[str, CorrespondenceResult],
                dict[str, QualityReport],
            ],
            None,
        ]
        | None = None,
        cancelled: Callable[[], bool] | None = None,
        report_burst_restart: Callable[[str, str, str], None] | None = None,
        history: Mapping[str, Sequence[ViewSignature]] | None = None,
    ) -> None:
        if set(detectors) != set(_SIDES) or set(quality_evaluators) != set(_SIDES):
            raise ValueError("bilateral capture requires left and right detectors/evaluators")
        self.camera_frames = camera_frames
        self.robot_states = robot_states
        self.detectors = dict(detectors)
        self.quality_evaluators = dict(quality_evaluators)
        self.recording_config = recording_config
        self.pairing_config = pairing_config
        self.config = config or LiveBurstConfig()
        self.clock = clock or SystemClock()
        self.wait_once = wait_once
        self.accept_yellow = accept_yellow or (lambda _frame, _sides: False)
        self.preview = preview
        self.cancelled = cancelled or (lambda: False)
        self.report_burst_restart = report_burst_restart or (
            lambda _pose_id, _capture_id, _reason: None
        )
        supplied_history = history or {side: () for side in _SIDES}
        if set(supplied_history) != set(_SIDES):
            raise ValueError("bilateral capture history must contain left and right")
        self._history = {side: list(supplied_history[side]) for side in _SIDES}

    def capture_burst(
        self,
        *,
        pose_id: str,
        capture_id: str,
        remember_signatures: bool = True,
    ) -> tuple[BilateralFrameEvidence, ...]:
        seen = {self._frame_key(frame) for frame in self.camera_frames.snapshot()}
        accepted: list[BilateralFrameEvidence] = []
        deadline = self.clock.monotonic() + self.config.timeout_s
        last_rejection = "no new rectified image"
        pose_local_rejection_seen = False
        while len(accepted) < self.config.frame_count:
            if self.cancelled():
                raise RuntimeError("operator cancelled bilateral live burst")
            if self.clock.monotonic() >= deadline:
                error_type = RecoverableCaptureError if pose_local_rejection_seen else RuntimeError
                raise error_type(
                    f"timed out with {len(accepted)}/{self.config.frame_count} bilateral "
                    f"frames; last rejection: {last_rejection}"
                )
            for frame in self.camera_frames.snapshot():
                key = self._frame_key(frame)
                if key in seen:
                    continue
                latest_state = self.robot_states.latest
                required_state_time = (
                    frame.timing.receipt_monotonic_s
                    + self.recording_config.stationary_duration_s / 2.0
                )
                if latest_state is None or latest_state.receipt_monotonic_s < required_state_time:
                    continue
                seen.add(key)
                try:
                    candidate = self._evaluate_frame(
                        frame,
                        frame_id=f"{capture_id}_{len(accepted):03d}",
                    )
                except RecoverableCaptureError as error:
                    last_rejection = str(error)
                    pose_local_rejection_seen = True
                    continue
                rejection = self._burst_rejection_reason(accepted, candidate)
                if rejection is not None:
                    last_rejection = rejection
                    pose_local_rejection_seen = True
                    self.report_burst_restart(pose_id, capture_id, rejection)
                    accepted = [replace(candidate, frame_id=f"{capture_id}_000")]
                    continue
                accepted.append(candidate)
                if len(accepted) >= self.config.frame_count:
                    break
            if len(accepted) < self.config.frame_count:
                self.wait_once(self.config.poll_interval_s)
        selected = select_bilateral_medoid(tuple(accepted))
        if remember_signatures:
            for side in _SIDES:
                signature = getattr(selected, f"{side}_quality").signature
                if signature is not None:
                    self._history[side].append(signature)
        return tuple(accepted)

    def undo_last_signatures(self) -> None:
        if any(not self._history[side] for side in _SIDES):
            raise ValueError("cannot undo incomplete bilateral capture history")
        for side in _SIDES:
            self._history[side].pop()

    def _evaluate_frame(
        self,
        frame: ROSImageFrame,
        *,
        frame_id: str,
    ) -> BilateralFrameEvidence:
        results = {side: self.detectors[side].detect(frame.image_bgr) for side in _SIDES}
        intrinsics = CameraIntrinsics(
            frame.camera_info.rectified_camera_matrix,
            np.asarray(frame.camera_info.d),
        )
        reports = {
            side: self.quality_evaluators[side].evaluate(
                results[side],
                intrinsics=intrinsics,
                history=self._history[side],
            )
            for side in _SIDES
        }
        if self.preview is not None:
            self.preview(frame, results, reports)
        for side in _SIDES:
            if not results[side].valid:
                raise RecoverableCaptureError(f"{side} target is not detected unambiguously")
            if reports[side].grade is QualityGrade.RED:
                raise RecoverableCaptureError(
                    f"{side} visual quality is red: " + "; ".join(reports[side].hard_failures)
                )
        yellow_sides = tuple(side for side in _SIDES if reports[side].grade is QualityGrade.YELLOW)
        if yellow_sides and not self.accept_yellow(frame, yellow_sides):
            raise RecoverableCaptureError(
                "visual quality is yellow and was not confirmed for " + ", ".join(yellow_sides)
            )
        state_window = self.robot_states.centered_window(
            center_monotonic_s=frame.timing.receipt_monotonic_s,
            duration_s=self.recording_config.stationary_duration_s,
        )
        readiness = evaluate_recording_window(
            state_window,
            now_monotonic_s=(
                state_window[-1].receipt_monotonic_s
                if state_window
                else frame.timing.receipt_monotonic_s
            ),
            config=self.recording_config,
        )
        if not readiness.ready:
            raise RecoverableCaptureError(
                "robot state is not stationary: " + "; ".join(readiness.hard_failures)
            )
        pairing = pair_state_to_image(
            frame.timing,
            state_window,
            config=self.pairing_config,
        )
        return BilateralFrameEvidence(
            frame_id=frame_id,
            image_bgr=frame.image_bgr,
            image_timing=frame.timing,
            camera_info=frame.camera_info,
            state_window=state_window,
            pairing=pairing,
            left_correspondences=results["left"],
            right_correspondences=results["right"],
            left_quality=reports["left"],
            right_quality=reports["right"],
        )

    def _burst_rejection_reason(
        self,
        accepted: list[BilateralFrameEvidence],
        candidate: BilateralFrameEvidence,
    ) -> str | None:
        if not accepted:
            return None
        duration = (
            candidate.image_timing.receipt_monotonic_s
            - accepted[0].image_timing.receipt_monotonic_s
        )
        if duration > self.config.maximum_duration_s + 1e-12:
            return (
                f"bilateral burst duration is {duration:.3f}s; limit is "
                f"{self.config.maximum_duration_s:.3f}s"
            )
        state_start = accepted[0].state_window[0].receipt_monotonic_s
        state_end = candidate.state_window[-1].receipt_monotonic_s
        combined_window = tuple(
            sample
            for sample in self.robot_states.snapshot()
            if state_start <= sample.receipt_monotonic_s <= state_end
        )
        if not combined_window:
            return "bilateral burst has no retained state samples"
        readiness = evaluate_recording_window(
            combined_window,
            now_monotonic_s=combined_window[-1].receipt_monotonic_s,
            config=self.recording_config,
        )
        if not readiness.ready:
            return "; ".join(readiness.hard_failures)
        return None

    @staticmethod
    def _frame_key(frame: ROSImageFrame) -> tuple[float, int | None]:
        return frame.timing.receipt_monotonic_s, frame.timing.header_stamp_ns


def select_bilateral_medoid(
    frames: tuple[BilateralFrameEvidence, ...],
) -> BilateralFrameEvidence:
    """Choose the joint left/right corner medoid from the dominant signature."""

    if not frames:
        raise ValueError("cannot select a bilateral medoid from no frames")
    signatures: dict[
        tuple[tuple[int, ...], tuple[int, ...]],
        list[BilateralFrameEvidence],
    ] = {}
    for frame in frames:
        signature = (
            frame.left_correspondences.tag_ids,
            frame.right_correspondences.tag_ids,
        )
        signatures.setdefault(signature, []).append(frame)
    candidates = max(
        signatures.values(),
        key=lambda group: (
            len(group),
            len(group[0].left_correspondences.tag_ids)
            + len(group[0].right_correspondences.tag_ids),
        ),
    )
    vectors = np.asarray([_normalized_corner_vector(frame) for frame in candidates])
    median = np.median(vectors, axis=0)
    distances = np.linalg.norm(vectors - median, axis=1)
    index = min(range(len(candidates)), key=lambda item: (distances[item], item))
    return candidates[index]


def _normalized_corner_vector(frame: BilateralFrameEvidence) -> np.ndarray:
    scale = np.asarray([frame.camera_info.width, frame.camera_info.height])
    values: list[np.ndarray] = []
    for side in _SIDES:
        result = getattr(frame, f"{side}_correspondences")
        values.extend(observation.image_corners_px / scale for observation in result.observations)
    return np.vstack(values).reshape(-1)
