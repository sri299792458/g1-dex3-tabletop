"""Continuous fixed-board trajectory benchmark for the G1 camera observer."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from g1_aprilcube_calibration.calibration_bundle import CalibrationBundle
from g1_aprilcube_calibration.ros.camera_adapter import (
    camera_info_from_ros,
    image_bgr_from_ros,
)
from g1_aprilcube_calibration.table_accuracy import CharucoBoardPoseDetector
from g1_aprilcube_calibration.transforms import invert_transform
from g1_aprilcube_calibration.urdf_model import URDFModel
from g1_dex3_tabletop.state_estimation import (
    ESTIMATOR_NAMES,
    WAIST_JOINT_INDICES,
    AnchoredCameraPoseEstimators,
    AnchoredCameraStateEstimator,
    CameraPoseAnchor,
    ProprioceptiveSample,
    pose_error,
    rotation_from_wxyz,
)
from g1_dex3_tabletop.state_estimation_replay import (
    DEFAULT_BUNDLE,
    DEFAULT_URDF,
    _align_events_to_mcap,
    _atomic_write_json,
    _clock_model,
    _load_events,
    _map_header_to_record_time,
    _message_stamp_ns,
    _sha256,
)


@dataclass(frozen=True, slots=True)
class ContinuousBoardObservation:
    sequence: int
    header_stamp_ns: int
    mcap_record_ns: int
    board_T_camera: np.ndarray
    reprojection_error_px: float
    point_count: int
    marker_count: int


@dataclass(frozen=True, slots=True)
class StateSeries:
    lowstate_time_ns: np.ndarray
    waist_q_rad: np.ndarray
    pelvis_quaternion_wxyz: np.ndarray
    torso_time_ns: np.ndarray
    torso_quaternion_wxyz: np.ndarray
    motion_intervals_ns: tuple[tuple[int, int], ...]
    statistics: dict[str, Any]

    @property
    def common_start_ns(self) -> int:
        return max(int(self.lowstate_time_ns[0]), int(self.torso_time_ns[0]))

    @property
    def common_end_ns(self) -> int:
        return min(int(self.lowstate_time_ns[-1]), int(self.torso_time_ns[-1]))

    def contains(self, timestamp_ns: int) -> bool:
        return self.common_start_ns <= timestamp_ns <= self.common_end_ns

    def sample(self, timestamp_ns: int) -> tuple[ProprioceptiveSample, dict[str, float]]:
        q, pelvis_quaternion, pelvis_gap_ms = _interpolate_vector_and_quaternion(
            self.lowstate_time_ns,
            self.waist_q_rad,
            self.pelvis_quaternion_wxyz,
            timestamp_ns,
        )
        _unused, torso_quaternion, torso_gap_ms = _interpolate_vector_and_quaternion(
            self.torso_time_ns,
            np.zeros((len(self.torso_time_ns), 0), dtype=np.float64),
            self.torso_quaternion_wxyz,
            timestamp_ns,
        )
        return (
            ProprioceptiveSample(
                timestamp_ns=timestamp_ns,
                waist_q_rad=q,
                navigation_R_pelvis_imu=rotation_from_wxyz(pelvis_quaternion),
                navigation_R_torso_imu=rotation_from_wxyz(torso_quaternion),
            ),
            {
                "lowstate_nearest_gap_ms": pelvis_gap_ms,
                "torso_imu_nearest_gap_ms": torso_gap_ms,
            },
        )

    def command_is_changing(self, timestamp_ns: int) -> bool:
        return any(start <= timestamp_ns <= end for start, end in self.motion_intervals_ns)


def _normalize_wxyz(value: np.ndarray) -> np.ndarray:
    quaternion = np.asarray(value, dtype=np.float64).reshape(-1)
    if quaternion.shape != (4,) or not np.all(np.isfinite(quaternion)):
        raise ValueError("quaternion must contain four finite values")
    norm = float(np.linalg.norm(quaternion))
    if norm < 1.0e-9:
        raise ValueError("quaternion norm is zero")
    return quaternion / norm


def _slerp_wxyz(first: np.ndarray, second: np.ndarray, alpha: float) -> np.ndarray:
    first = _normalize_wxyz(first)
    second = _normalize_wxyz(second)
    dot = float(np.dot(first, second))
    if dot < 0.0:
        second = -second
        dot = -dot
    dot = min(max(dot, -1.0), 1.0)
    if dot > 0.9995:
        return _normalize_wxyz((1.0 - alpha) * first + alpha * second)
    angle = float(np.arccos(dot))
    denominator = float(np.sin(angle))
    return _normalize_wxyz(
        np.sin((1.0 - alpha) * angle) / denominator * first
        + np.sin(alpha * angle) / denominator * second
    )


def _interpolate_vector_and_quaternion(
    times_ns: np.ndarray,
    vectors: np.ndarray,
    quaternions_wxyz: np.ndarray,
    timestamp_ns: int,
) -> tuple[np.ndarray, np.ndarray, float]:
    if timestamp_ns < int(times_ns[0]) or timestamp_ns > int(times_ns[-1]):
        raise ValueError("requested state timestamp is outside the recorded stream")
    upper = int(np.searchsorted(times_ns, timestamp_ns, side="left"))
    if upper == 0:
        lower = upper = 0
    elif upper == len(times_ns):
        lower = upper = len(times_ns) - 1
    elif int(times_ns[upper]) == timestamp_ns:
        lower = upper
    else:
        lower = upper - 1
    nearest_gap_ms = (
        min(
            abs(timestamp_ns - int(times_ns[lower])),
            abs(timestamp_ns - int(times_ns[upper])),
        )
        / 1.0e6
    )
    if lower == upper:
        return vectors[lower].copy(), quaternions_wxyz[lower].copy(), nearest_gap_ms
    span = int(times_ns[upper]) - int(times_ns[lower])
    alpha = (timestamp_ns - int(times_ns[lower])) / span
    vector = (1.0 - alpha) * vectors[lower] + alpha * vectors[upper]
    quaternion = _slerp_wxyz(quaternions_wxyz[lower], quaternions_wxyz[upper], alpha)
    return vector, quaternion, nearest_gap_ms


def _motion_intervals(
    change_times_ns: np.ndarray,
    *,
    maximum_change_gap_ns: int = 100_000_000,
    padding_ns: int = 100_000_000,
) -> tuple[tuple[int, int], ...]:
    if change_times_ns.size == 0:
        return ()
    intervals: list[tuple[int, int]] = []
    start = previous = int(change_times_ns[0])
    for value in change_times_ns[1:]:
        current = int(value)
        if current - previous > maximum_change_gap_ns:
            intervals.append((start - padding_ns, previous + padding_ns))
            start = current
        previous = current
    intervals.append((start - padding_ns, previous + padding_ns))
    return tuple(intervals)


def _read_state_series(bag_directory: Path) -> StateSeries:
    try:
        import rosbag2_py
        from rclpy.serialization import deserialize_message
        from unitree_hg.msg import IMUState, LowCmd, LowState
    except ImportError as error:
        raise RuntimeError(
            "ROS 2/Unitree MCAP readers are unavailable; use "
            "./tools/g1_state_estimation_research.sh"
        ) from error
    reader = rosbag2_py.SequentialReader()
    reader.open(
        rosbag2_py.StorageOptions(uri=str(bag_directory), storage_id="mcap"),
        rosbag2_py.ConverterOptions("", ""),
    )
    reader.set_filter(rosbag2_py.StorageFilter(topics=["/lowstate", "/secondary_imu", "/lowcmd"]))
    low_times: list[int] = []
    low_ticks: list[int] = []
    waist_q: list[np.ndarray] = []
    pelvis_quaternion: list[np.ndarray] = []
    torso_times: list[int] = []
    torso_quaternion: list[np.ndarray] = []
    command_times: list[int] = []
    command_q: list[np.ndarray] = []
    previous_tick: int | None = None
    duplicate_ticks = 0
    while reader.has_next():
        topic, raw, record_time_ns = reader.read_next()
        if topic == "/lowstate":
            message = deserialize_message(raw, LowState)
            tick = int(message.tick)
            if tick == previous_tick:
                duplicate_ticks += 1
                continue
            previous_tick = tick
            low_times.append(record_time_ns)
            low_ticks.append(tick)
            waist_q.append(
                np.asarray(
                    [message.motor_state[index].q for index in WAIST_JOINT_INDICES],
                    dtype=np.float64,
                )
            )
            pelvis_quaternion.append(np.asarray(message.imu_state.quaternion, dtype=np.float64))
        elif topic == "/secondary_imu":
            message = deserialize_message(raw, IMUState)
            quaternion = np.asarray(message.quaternion, dtype=np.float64)
            if torso_quaternion and np.array_equal(quaternion, torso_quaternion[-1]):
                continue
            torso_times.append(record_time_ns)
            torso_quaternion.append(quaternion)
        else:
            message = deserialize_message(raw, LowCmd)
            command_times.append(record_time_ns)
            command_q.append(
                np.asarray([command.q for command in message.motor_cmd[:29]], dtype=np.float64)
            )
    if len(low_times) < 2 or len(torso_times) < 2 or len(command_times) < 2:
        raise ValueError("MCAP lacks a complete continuous Unitree state sequence")
    command_values = np.stack(command_q)
    command_changed = np.max(np.abs(np.diff(command_values, axis=0)), axis=1) > 1.0e-6
    change_times = np.asarray(command_times[1:], dtype=np.int64)[command_changed]
    low_time_values = np.asarray(low_times, dtype=np.int64)
    torso_time_values = np.asarray(torso_times, dtype=np.int64)
    return StateSeries(
        lowstate_time_ns=low_time_values,
        waist_q_rad=np.stack(waist_q),
        pelvis_quaternion_wxyz=np.stack(pelvis_quaternion),
        torso_time_ns=torso_time_values,
        torso_quaternion_wxyz=np.stack(torso_quaternion),
        motion_intervals_ns=_motion_intervals(change_times),
        statistics={
            "lowstate_unique_tick_count": len(low_times),
            "lowstate_consecutive_duplicate_count": duplicate_ticks,
            "lowstate_duration_s": (int(low_time_values[-1]) - int(low_time_values[0])) / 1.0e9,
            "torso_imu_unique_orientation_count": len(torso_times),
            "torso_imu_duration_s": (int(torso_time_values[-1]) - int(torso_time_values[0]))
            / 1.0e9,
            "lowcmd_count": len(command_times),
            "motion_interval_count": len(_motion_intervals(change_times)),
        },
    )


def _read_camera_info(bag_directory: Path):
    try:
        import rosbag2_py
        from rclpy.serialization import deserialize_message
        from sensor_msgs.msg import CameraInfo
    except ImportError as error:
        raise RuntimeError("ROS 2 camera MCAP readers are unavailable") from error
    reader = rosbag2_py.SequentialReader()
    reader.open(
        rosbag2_py.StorageOptions(uri=str(bag_directory), storage_id="mcap"),
        rosbag2_py.ConverterOptions("", ""),
    )
    reader.set_filter(rosbag2_py.StorageFilter(topics=["/camera/color/camera_info"]))
    profile = None
    profile_hash = None
    pairs: list[tuple[int, int]] = []
    count = 0
    while reader.has_next():
        _topic, raw, record_time_ns = reader.read_next()
        message = deserialize_message(raw, CameraInfo)
        current = camera_info_from_ros(
            message,
            camera_name="recorded_g1_head_color",
            serial_number="recorded",
        )
        if profile is None:
            profile = current
            profile_hash = current.profile_sha256
        elif current.profile_sha256 != profile_hash:
            raise ValueError("recorded color CameraInfo changed during the episode")
        pairs.append((_message_stamp_ns(message), record_time_ns))
        count += 1
    if profile is None:
        raise ValueError("recorded color CameraInfo is unavailable")
    return profile, _clock_model(pairs), count


def _rejection_category(reason: str) -> str:
    if "was not detected" in reason:
        return "board_not_detected"
    if "markers; need" in reason:
        return "too_few_markers"
    if "corners; need" in reason or "no ChArUco corners" in reason:
        return "too_few_corners"
    if "reprojection error" in reason:
        return "reprojection_error"
    if "ambiguous" in reason:
        return "planar_pose_ambiguous"
    return "other"


def _detect_continuous_board(
    bag_directory: Path,
    *,
    frame_stride: int,
) -> tuple[list[ContinuousBoardObservation], dict[str, Any], dict[str, Any]]:
    if frame_stride < 1:
        raise ValueError("frame stride must be positive")
    try:
        import rosbag2_py
        from rclpy.serialization import deserialize_message
        from sensor_msgs.msg import Image
    except ImportError as error:
        raise RuntimeError("ROS 2 image MCAP readers are unavailable") from error
    camera_info, camera_info_clock, camera_info_count = _read_camera_info(bag_directory)
    detector = CharucoBoardPoseDetector()
    reader = rosbag2_py.SequentialReader()
    reader.open(
        rosbag2_py.StorageOptions(uri=str(bag_directory), storage_id="mcap"),
        rosbag2_py.ConverterOptions("", ""),
    )
    reader.set_filter(rosbag2_py.StorageFilter(topics=["/camera/color/image_raw"]))
    seen = 0
    sampled = 0
    observations: list[ContinuousBoardObservation] = []
    rejections: Counter[str] = Counter()
    image_clock_pairs: list[tuple[int, int]] = []
    while reader.has_next():
        _topic, raw, record_time_ns = reader.read_next()
        sequence = seen
        seen += 1
        if sequence % frame_stride:
            continue
        sampled += 1
        message = deserialize_message(raw, Image)
        header_stamp_ns = _message_stamp_ns(message)
        image_clock_pairs.append((header_stamp_ns, record_time_ns))
        try:
            estimate = detector.detect(image_bgr_from_ros(message), camera_info)
        except ValueError as error:
            rejections[_rejection_category(str(error))] += 1
            continue
        observations.append(
            ContinuousBoardObservation(
                sequence=sequence,
                header_stamp_ns=header_stamp_ns,
                mcap_record_ns=record_time_ns,
                board_T_camera=invert_transform(estimate.camera_T_target),
                reprojection_error_px=estimate.reprojection_error_px,
                point_count=estimate.point_count,
                marker_count=len(estimate.marker_ids),
            )
        )
    if len(observations) < 2:
        raise ValueError("continuous ChArUco replay produced fewer than two poses")
    image_clock = _clock_model(image_clock_pairs)
    mapped_residual_ms = np.asarray(
        [
            (
                observation.mcap_record_ns
                - _map_header_to_record_time(image_clock, observation.header_stamp_ns)
            )
            / 1.0e6
            for observation in observations
        ],
        dtype=np.float64,
    )
    return (
        observations,
        {
            "image_message_count": seen,
            "frame_stride": frame_stride,
            "sampled_frame_count": sampled,
            "accepted_frame_count": len(observations),
            "accepted_fraction": len(observations) / sampled,
            "rejection_count_by_category": dict(sorted(rejections.items())),
            "camera_info_message_count": camera_info_count,
            "camera_profile": camera_info.to_dict(),
            "reprojection_error_px": _scalar_summary(
                [observation.reprojection_error_px for observation in observations]
            ),
            "image_record_minus_mapped_header_ms": {
                "mean": float(np.mean(mapped_residual_ms)),
                "p95_absolute": float(np.percentile(np.abs(mapped_residual_ms), 95)),
                "maximum_absolute": float(np.max(np.abs(mapped_residual_ms))),
            },
        },
        {"image": image_clock, "camera_info": camera_info_clock},
    )


def _scalar_summary(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        raise ValueError("cannot summarize an empty sequence")
    return {
        "mean": float(np.mean(array)),
        "rms": float(np.sqrt(np.mean(np.square(array)))),
        "p95": float(np.percentile(array, 95)),
        "maximum": float(np.max(array)),
    }


def _error_summary(errors: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "sample_count": len(errors),
        "translation_norm_mm": _scalar_summary(
            [float(error["translation_norm_mm"]) for error in errors]
        ),
        "rotation_deg": _scalar_summary([float(error["rotation_deg"]) for error in errors]),
        "mean_translation_reference_xyz_mm": np.mean(
            [error["translation_reference_xyz_mm"] for error in errors], axis=0
        ).tolist(),
    }


def _evaluate_continuous(
    *,
    observations: list[ContinuousBoardObservation],
    image_clock: dict[str, Any],
    state_series: StateSeries,
    estimators: AnchoredCameraPoseEstimators,
    baseline_time_ns: int,
    evaluation_end_time_ns: int,
    baseline_board_T_camera: np.ndarray,
    reanchor_intervals_s: tuple[float, ...],
) -> dict[str, Any]:
    baseline_sample, baseline_gaps = state_series.sample(baseline_time_ns)
    global_anchor = CameraPoseAnchor(
        reference_T_camera=baseline_board_T_camera,
        sample=baseline_sample,
    )
    global_observer = AnchoredCameraStateEstimator(estimators)
    global_observer.reset(global_anchor)
    evaluated: list[dict[str, Any]] = []
    outside_state_range_count = 0
    after_evaluation_window_count = 0
    for observation in observations:
        timestamp_ns = _map_header_to_record_time(image_clock, observation.header_stamp_ns)
        if timestamp_ns < baseline_time_ns:
            continue
        if timestamp_ns > evaluation_end_time_ns:
            after_evaluation_window_count += 1
            continue
        if not state_series.contains(timestamp_ns):
            outside_state_range_count += 1
            continue
        sample, gaps = state_series.sample(timestamp_ns)
        predicted = estimators.predict_all(global_anchor, sample)
        production_prediction = global_observer.estimate(sample).reference_T_camera
        if not np.allclose(
            production_prediction,
            predicted["hybrid_pelvis_position_torso_orientation"],
            atol=1.0e-12,
            rtol=0.0,
        ):
            raise RuntimeError("standalone camera observer diverged from the validated hybrid")
        evaluated.append(
            {
                "observation": observation,
                "timestamp_ns": timestamp_ns,
                "sample": sample,
                "state_gaps": gaps,
                "command_changing": state_series.command_is_changing(timestamp_ns),
                "global_errors": {
                    name: pose_error(value, observation.board_T_camera)
                    for name, value in predicted.items()
                },
            }
        )
    if len(evaluated) < 2:
        raise ValueError("fewer than two ChArUco poses follow the loaded baseline")
    global_summary: dict[str, Any] = {}
    for name in ESTIMATOR_NAMES:
        all_errors = [item["global_errors"][name] for item in evaluated]
        moving_errors = [
            item["global_errors"][name] for item in evaluated if item["command_changing"]
        ]
        global_summary[name] = {
            "all": _error_summary(all_errors),
            "while_command_changing": (
                None if not moving_errors else _error_summary(moving_errors)
            ),
        }

    reanchored: dict[str, Any] = {}
    for interval_s in reanchor_intervals_s:
        if interval_s <= 0:
            raise ValueError("re-anchor intervals must be positive")
        anchor_item = evaluated[0]
        anchor = CameraPoseAnchor(
            reference_T_camera=anchor_item["observation"].board_T_camera,
            sample=anchor_item["sample"],
        )
        observer = AnchoredCameraStateEstimator(estimators)
        observer.reset(anchor)
        period_start_ns = anchor_item["timestamp_ns"]
        all_errors: list[dict[str, Any]] = []
        boundary_errors: list[dict[str, Any]] = []
        realized_intervals_s: list[float] = []
        current_period: list[dict[str, Any]] = []
        for item in evaluated[1:]:
            prediction = observer.estimate(item["sample"]).reference_T_camera
            error = pose_error(prediction, item["observation"].board_T_camera)
            all_errors.append(error)
            current_period.append(error)
            if item["timestamp_ns"] - period_start_ns < interval_s * 1.0e9:
                continue
            boundary_errors.append(current_period[-1])
            realized_intervals_s.append((item["timestamp_ns"] - period_start_ns) / 1.0e9)
            anchor = CameraPoseAnchor(
                reference_T_camera=item["observation"].board_T_camera,
                sample=item["sample"],
            )
            observer.reset(anchor)
            period_start_ns = item["timestamp_ns"]
            current_period = []
        if current_period:
            boundary_errors.append(current_period[-1])
        key = f"{interval_s:g}s"
        reanchored[key] = {
            "visual_update_interval_s": interval_s,
            "realized_update_interval_s": _scalar_summary(realized_intervals_s),
            "all_propagated_frames": _error_summary(all_errors),
            "last_frame_before_each_update": _error_summary(boundary_errors),
        }

    low_gaps = [item["state_gaps"]["lowstate_nearest_gap_ms"] for item in evaluated]
    torso_gaps = [item["state_gaps"]["torso_imu_nearest_gap_ms"] for item in evaluated]
    trajectory = [
        {
            "sequence": item["observation"].sequence,
            "header_stamp_ns": item["observation"].header_stamp_ns,
            "mapped_mcap_time_ns": item["timestamp_ns"],
            "command_changing": item["command_changing"],
            "board_T_camera": item["observation"].board_T_camera.tolist(),
            "reprojection_error_px": item["observation"].reprojection_error_px,
            "point_count": item["observation"].point_count,
            "marker_count": item["observation"].marker_count,
            "hybrid_global_error": item["global_errors"][
                "hybrid_pelvis_position_torso_orientation"
            ],
        }
        for item in evaluated
    ]
    return {
        "baseline": {
            "timestamp_ns": baseline_time_ns,
            "board_T_camera": baseline_board_T_camera.tolist(),
            "state_pairing_gap_ms": baseline_gaps,
        },
        "evaluation_end_time_ns": evaluation_end_time_ns,
        "evaluated_frame_count": len(evaluated),
        "outside_state_range_frame_count": outside_state_range_count,
        "after_evaluation_window_frame_count": after_evaluation_window_count,
        "command_changing_frame_count": sum(int(item["command_changing"]) for item in evaluated),
        "state_pairing_gap_ms": {
            "lowstate": _scalar_summary(low_gaps),
            "torso_imu": _scalar_summary(torso_gaps),
        },
        "global_anchor": global_summary,
        "periodic_visual_reanchor": reanchored,
        "trajectory": trajectory,
    }


def analyze_continuous_run(
    run_directory: Path,
    *,
    output_path: Path,
    urdf_path: Path = DEFAULT_URDF,
    calibration_bundle_path: Path = DEFAULT_BUNDLE,
    frame_stride: int = 1,
    reanchor_intervals_s: tuple[float, ...] = (0.5, 1.0, 2.0, 5.0),
) -> dict[str, Any]:
    run_directory = run_directory.resolve()
    bag_directory = run_directory / "raw_episode/bag"
    observations_path = run_directory / "board_observations.json"
    if not bag_directory.is_dir() or not observations_path.is_file():
        raise FileNotFoundError("run lacks raw MCAP or board observations")
    observations, detection, clocks = _detect_continuous_board(
        bag_directory,
        frame_stride=frame_stride,
    )
    state_series = _read_state_series(bag_directory)
    _document, receipt_events = _load_events(observations_path)
    aligned_events, event_timing = _align_events_to_mcap(
        receipt_events,
        clocks["image"],
    )
    baselines = [event for event in aligned_events if event.phase == "loaded_baseline"]
    if len(baselines) != 1:
        raise ValueError("run must contain exactly one loaded baseline")
    baseline = baselines[0]
    model = URDFModel(urdf_path)
    bundle = CalibrationBundle.load(calibration_bundle_path)
    estimators = AnchoredCameraPoseEstimators(model=model, calibration_bundle=bundle)
    evaluation = _evaluate_continuous(
        observations=observations,
        image_clock=clocks["image"],
        state_series=state_series,
        estimators=estimators,
        baseline_time_ns=baseline.center_ns,
        evaluation_end_time_ns=max(event.end_ns for event in aligned_events),
        baseline_board_T_camera=baseline.board_T_camera,
        reanchor_intervals_s=reanchor_intervals_s,
    )
    report = {
        "schema_version": 1,
        "kind": "g1_continuous_camera_state_estimation_research_report",
        "commands_robot": False,
        "run_directory": str(run_directory),
        "provenance": {
            "board_observations_sha256": _sha256(observations_path),
            "urdf": str(urdf_path.resolve()),
            "urdf_sha256": model.sha256,
            "calibration_bundle": str(calibration_bundle_path.resolve()),
            "calibration_bundle_sha256": bundle.content_sha256,
        },
        "detection": detection,
        "clock_models": clocks,
        "board_event_timing": event_timing,
        "state_series": state_series.statistics,
        "evaluation": evaluation,
        "interpretation_limits": [
            "The fixed ChArUco board supplies evaluation ground truth and visual re-anchors.",
            "Periodic re-anchor results simulate bounded visual outages; they are not IMU-only global odometry.",
            "Command-changing labels come from lowcmd target changes and include 100 ms padding.",
            "The hybrid retains a fixed-pelvis-IMU-origin contact hypothesis between visual updates.",
        ],
    }
    _atomic_write_json(output_path, report)
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="g1-continuous-state-estimation-research")
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument(
        "--reanchor-interval-s",
        type=float,
        action="append",
        dest="reanchor_intervals_s",
    )
    parser.add_argument("--urdf", type=Path, default=DEFAULT_URDF)
    parser.add_argument("--calibration-bundle", type=Path, default=DEFAULT_BUNDLE)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    intervals = (
        (0.5, 1.0, 2.0, 5.0)
        if args.reanchor_intervals_s is None
        else tuple(args.reanchor_intervals_s)
    )
    try:
        report = analyze_continuous_run(
            args.run,
            output_path=args.output,
            urdf_path=args.urdf,
            calibration_bundle_path=args.calibration_bundle,
            frame_stride=args.frame_stride,
            reanchor_intervals_s=intervals,
        )
    except (FileNotFoundError, RuntimeError, TypeError, ValueError) as error:
        print(f"error: {error}")
        return 1
    hybrid = report["evaluation"]["global_anchor"]["hybrid_pelvis_position_torso_orientation"]
    print(
        json.dumps(
            {
                "commands_robot": False,
                "output": str(args.output.resolve()),
                "sampled_frames": report["detection"]["sampled_frame_count"],
                "accepted_frames": report["detection"]["accepted_frame_count"],
                "global_hybrid": hybrid,
                "periodic_visual_reanchor": report["evaluation"]["periodic_visual_reanchor"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
