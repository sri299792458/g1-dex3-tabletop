"""Offline MCAP replay and fixed-board evaluation of G1 camera estimators."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation

from g1_aprilcube_calibration.calibration_bundle import CalibrationBundle
from g1_aprilcube_calibration.transforms import invert_transform, validate_transform
from g1_aprilcube_calibration.urdf_model import URDFModel
from g1_dex3_tabletop.state_estimation import (
    ESTIMATOR_CONTRACTS,
    ESTIMATOR_NAMES,
    AnchoredCameraPoseEstimators,
    CameraPoseAnchor,
    ProprioceptiveSample,
    pose_error,
)

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_URDF = ROOT / "config/urdf/g1_29dof_rev_1_0_g1pilot_collision.urdf"
DEFAULT_BUNDLE = ROOT / "config/calibrations/dex3_shared_20260812_selected_free.json"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _atomic_write_json(path: Path, document: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(document, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _utc_ns(value: str) -> int:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    return round(parsed.timestamp() * 1_000_000_000.0)


def _message_stamp_ns(message: Any) -> int:
    return int(message.header.stamp.sec) * 1_000_000_000 + int(message.header.stamp.nanosec)


def _mean_rotation_from_wxyz(quaternions: list[np.ndarray]) -> np.ndarray:
    if not quaternions:
        raise ValueError("cannot average an empty quaternion sequence")
    values = np.stack(quaternions).astype(np.float64)
    norms = np.linalg.norm(values, axis=1)
    if np.any(~np.isfinite(values)) or np.any(norms < 1.0e-9):
        raise ValueError("IMU quaternion is invalid")
    values /= norms[:, None]
    return Rotation.from_quat(values[:, [1, 2, 3, 0]]).mean().as_matrix()


@dataclass(frozen=True, slots=True)
class BoardEvent:
    index: int
    repetition: int | None
    arm: str | None
    phase: str
    start_ns: int
    end_ns: int
    header_start_ns: int
    header_end_ns: int
    board_T_camera: np.ndarray

    @property
    def center_ns(self) -> int:
        return (self.start_ns + self.end_ns) // 2


@dataclass(slots=True)
class _EventSamples:
    lowstate: list[tuple[int, int, np.ndarray, np.ndarray, np.ndarray]]
    torso_imu: list[tuple[int, np.ndarray, np.ndarray, np.ndarray]]
    d435_accel: list[tuple[int, np.ndarray]]


def _load_events(path: Path) -> tuple[dict[str, Any], list[BoardEvent]]:
    document = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(document, dict) or not isinstance(document.get("events"), list):
        raise TypeError("board_observations.json has no event sequence")
    result: list[BoardEvent] = []
    for index, item in enumerate(document["events"]):
        timing = item["board"]["input_frame_timing"]
        if len(timing) < 3:
            raise ValueError(f"board event {index} has fewer than three frames")
        receipt_ns = [_utc_ns(frame["receipt_utc"]) for frame in timing]
        header_ns = [int(frame["header_stamp_ns"]) for frame in timing]
        camera_T_board = validate_transform(
            np.asarray(item["board"]["aggregate"]["camera_T_board"], dtype=np.float64)
        )
        result.append(
            BoardEvent(
                index=index,
                repetition=(None if item.get("repetition") is None else int(item["repetition"])),
                arm=item.get("arm"),
                phase=str(item["phase"]),
                start_ns=min(receipt_ns),
                end_ns=max(receipt_ns),
                header_start_ns=min(header_ns),
                header_end_ns=max(header_ns),
                board_T_camera=invert_transform(camera_T_board),
            )
        )
    return document, result


def _read_camera_clock_models(bag_directory: Path) -> dict[str, Any]:
    try:
        import rosbag2_py
        from rclpy.serialization import deserialize_message
        from sensor_msgs.msg import CameraInfo, Imu
    except ImportError as error:
        raise RuntimeError(
            "ROS 2 MCAP readers are unavailable; use ./tools/g1_state_estimation_research.sh"
        ) from error
    topics = (
        "/camera/color/camera_info",
        "/camera/gyro/sample",
        "/camera/accel/sample",
    )
    pairs: dict[str, list[tuple[int, int]]] = {topic: [] for topic in topics}
    reader = rosbag2_py.SequentialReader()
    reader.open(
        rosbag2_py.StorageOptions(uri=str(bag_directory), storage_id="mcap"),
        rosbag2_py.ConverterOptions("", ""),
    )
    reader.set_filter(rosbag2_py.StorageFilter(topics=list(topics)))
    while reader.has_next():
        topic, raw, record_time_ns = reader.read_next()
        message_type = CameraInfo if topic.endswith("camera_info") else Imu
        message = deserialize_message(raw, message_type)
        pairs[topic].append((_message_stamp_ns(message), record_time_ns))
    return {topic: _clock_model(values) for topic, values in pairs.items()}


def _map_header_to_record_time(clock_model: dict[str, Any], header_ns: int) -> int:
    mapping = clock_model["affine_mapping"]
    return round(
        int(mapping["record_anchor_ns"])
        + float(mapping["slope"]) * (header_ns - int(mapping["header_anchor_ns"]))
        + float(mapping["delta_offset_ns"])
    )


def _align_events_to_mcap(
    events: list[BoardEvent],
    color_clock_model: dict[str, Any],
) -> tuple[list[BoardEvent], dict[str, Any]]:
    aligned: list[BoardEvent] = []
    receipt_minus_mapped_ms: list[float] = []
    for event in events:
        mapped_start = _map_header_to_record_time(color_clock_model, event.header_start_ns)
        mapped_end = _map_header_to_record_time(color_clock_model, event.header_end_ns)
        aligned.append(replace(event, start_ns=mapped_start, end_ns=mapped_end))
        receipt_minus_mapped_ms.extend(
            [
                (event.start_ns - mapped_start) / 1.0e6,
                (event.end_ns - mapped_end) / 1.0e6,
            ]
        )
    values = np.asarray(receipt_minus_mapped_ms, dtype=np.float64)
    return aligned, {
        "state_pairing_clock": "fitted_color_header_to_mcap_record_time",
        "application_receipt_minus_mapped_record_ms": {
            "median": float(np.median(values)),
            "p95_absolute": float(np.percentile(np.abs(values), 95)),
            "maximum_absolute": float(np.max(np.abs(values))),
        },
    }


def _event_index_at_time(events: list[BoardEvent], timestamp_ns: int) -> int | None:
    for event in events:
        if event.start_ns <= timestamp_ns <= event.end_ns:
            return event.index
    return None


def _read_mcap(
    bag_directory: Path,
    events: list[BoardEvent],
) -> tuple[list[_EventSamples], dict[str, Any]]:
    try:
        import rosbag2_py
        from rclpy.serialization import deserialize_message
        from sensor_msgs.msg import CameraInfo, Imu
        from unitree_hg.msg import IMUState, LowCmd, LowState
    except ImportError as error:
        raise RuntimeError(
            "ROS 2/Unitree MCAP readers are unavailable; use "
            "./tools/g1_state_estimation_research.sh"
        ) from error

    samples = [_EventSamples(lowstate=[], torso_imu=[], d435_accel=[]) for _ in events]
    reader = rosbag2_py.SequentialReader()
    reader.open(
        rosbag2_py.StorageOptions(uri=str(bag_directory), storage_id="mcap"),
        rosbag2_py.ConverterOptions("", ""),
    )
    topics = {item.name: item.type for item in reader.get_all_topics_and_types()}
    required = {
        "/lowstate": "unitree_hg/msg/LowState",
        "/secondary_imu": "unitree_hg/msg/IMUState",
        "/lowcmd": "unitree_hg/msg/LowCmd",
        "/camera/color/camera_info": "sensor_msgs/msg/CameraInfo",
        "/camera/gyro/sample": "sensor_msgs/msg/Imu",
        "/camera/accel/sample": "sensor_msgs/msg/Imu",
    }
    problems = [
        f"{name}: expected {expected}, got {topics.get(name)!r}"
        for name, expected in required.items()
        if topics.get(name) != expected
    ]
    if problems:
        raise ValueError("MCAP state-estimation topic contract differs: " + "; ".join(problems))
    reader.set_filter(rosbag2_py.StorageFilter(topics=list(required)))

    topic_count = {name: 0 for name in required}
    lowstate_times: list[int] = []
    lowstate_ticks: list[int] = []
    torso_times: list[int] = []
    lowcmd_times: list[int] = []
    lowcmd_positions: list[np.ndarray] = []
    clock_pairs: dict[str, list[tuple[int, int]]] = {
        "/camera/color/camera_info": [],
        "/camera/gyro/sample": [],
        "/camera/accel/sample": [],
    }
    while reader.has_next():
        topic, raw, record_time_ns = reader.read_next()
        topic_count[topic] += 1
        event_index = _event_index_at_time(events, record_time_ns)
        if topic == "/lowstate":
            message = deserialize_message(raw, LowState)
            lowstate_times.append(record_time_ns)
            lowstate_ticks.append(int(message.tick))
            if event_index is not None:
                samples[event_index].lowstate.append(
                    (
                        record_time_ns,
                        int(message.tick),
                        np.asarray(
                            [state.q for state in message.motor_state[:29]],
                            dtype=np.float64,
                        ),
                        np.asarray(message.imu_state.quaternion, dtype=np.float64),
                        np.asarray(message.imu_state.accelerometer, dtype=np.float64),
                    )
                )
        elif topic == "/secondary_imu":
            message = deserialize_message(raw, IMUState)
            torso_times.append(record_time_ns)
            if event_index is not None:
                samples[event_index].torso_imu.append(
                    (
                        record_time_ns,
                        np.asarray(message.quaternion, dtype=np.float64),
                        np.asarray(message.accelerometer, dtype=np.float64),
                        np.asarray(message.gyroscope, dtype=np.float64),
                    )
                )
        elif topic == "/lowcmd":
            message = deserialize_message(raw, LowCmd)
            lowcmd_times.append(record_time_ns)
            lowcmd_positions.append(
                np.asarray([command.q for command in message.motor_cmd[:29]], dtype=np.float64)
            )
        else:
            message_type = CameraInfo if topic.endswith("camera_info") else Imu
            message = deserialize_message(raw, message_type)
            clock_pairs[topic].append((_message_stamp_ns(message), record_time_ns))
            if topic == "/camera/accel/sample" and event_index is not None:
                samples[event_index].d435_accel.append(
                    (
                        record_time_ns,
                        np.asarray(
                            [
                                message.linear_acceleration.x,
                                message.linear_acceleration.y,
                                message.linear_acceleration.z,
                            ],
                            dtype=np.float64,
                        ),
                    )
                )

    camera_clock_models = {topic: _clock_model(pairs) for topic, pairs in clock_pairs.items()}
    color_offset_s = camera_clock_models["/camera/color/camera_info"]["raw_record_minus_header_s"][
        "median"
    ]
    imu_offset_s = np.mean(
        [
            camera_clock_models[topic]["raw_record_minus_header_s"]["median"]
            for topic in ("/camera/gyro/sample", "/camera/accel/sample")
        ]
    )
    return samples, {
        "topic_message_count": topic_count,
        "lowstate": _stream_timing(lowstate_times),
        "secondary_imu": _stream_timing(torso_times),
        "lowstate_tick": _tick_summary(lowstate_ticks),
        "camera_clock_models": camera_clock_models,
        "camera_stream_header_offset_difference_ms": {
            "color_minus_mean_d435i_imu": 1000.0 * (color_offset_s - imu_offset_s),
            "interpretation": (
                "systematic producer-header difference; align each stream to MCAP "
                "record time independently rather than comparing raw header seconds"
            ),
        },
        "command_settling": _command_settling(events, lowcmd_times, lowcmd_positions),
    }


def _stream_timing(times_ns: list[int]) -> dict[str, Any]:
    if len(times_ns) < 2:
        raise ValueError("recorded stream contains fewer than two messages")
    gaps_ms = np.diff(np.asarray(times_ns, dtype=np.int64)) / 1.0e6
    duration_s = (times_ns[-1] - times_ns[0]) / 1.0e9
    return {
        "message_count": len(times_ns),
        "duration_s": duration_s,
        "mean_rate_hz": (len(times_ns) - 1) / duration_s,
        "gap_ms": {
            "median": float(np.median(gaps_ms)),
            "p99": float(np.percentile(gaps_ms, 99)),
            "maximum": float(np.max(gaps_ms)),
        },
    }


def _tick_summary(ticks: list[int]) -> dict[str, Any]:
    values = np.asarray(ticks, dtype=np.uint64)
    if values.size < 2:
        raise ValueError("lowstate has fewer than two ticks")
    changed = values[1:] != values[:-1]
    return {
        "raw_message_count": int(values.size),
        "changed_tick_count": int(np.count_nonzero(changed)) + 1,
        "consecutive_duplicate_count": int(np.count_nonzero(~changed)),
        "consecutive_duplicate_fraction": float(np.mean(~changed)),
    }


def _clock_model(pairs: list[tuple[int, int]]) -> dict[str, Any]:
    if len(pairs) < 2:
        raise ValueError("camera stream has fewer than two timestamp pairs")
    values = np.asarray(pairs, dtype=np.int64)
    header = values[:, 0]
    record = values[:, 1]
    header_anchor_ns = int(header[0])
    record_anchor_ns = int(record[0])
    header_delta_ns = (header - header_anchor_ns).astype(np.float64)
    record_delta_ns = (record - record_anchor_ns).astype(np.float64)
    centered_header = header_delta_ns - np.mean(header_delta_ns)
    centered_record = record_delta_ns - np.mean(record_delta_ns)
    slope = float(
        np.dot(centered_header, centered_record) / np.dot(centered_header, centered_header)
    )
    delta_offset_ns = float(np.mean(record_delta_ns - slope * header_delta_ns))
    residual_ms = (record_delta_ns - (slope * header_delta_ns + delta_offset_ns)) / 1.0e6
    raw_offset_s = (record - header).astype(np.float64) / 1.0e9
    return {
        "pair_count": len(pairs),
        "affine_mapping": {
            "header_anchor_ns": header_anchor_ns,
            "record_anchor_ns": record_anchor_ns,
            "equation": (
                "record_ns-record_anchor_ns = slope*(header_ns-header_anchor_ns) + delta_offset_ns"
            ),
            "slope": slope,
            "rate_error_ppm": (slope - 1.0) * 1.0e6,
            "delta_offset_ns": delta_offset_ns,
        },
        "raw_record_minus_header_s": {
            "median": float(np.median(raw_offset_s)),
            "minimum": float(np.min(raw_offset_s)),
            "maximum": float(np.max(raw_offset_s)),
        },
        "fit_residual_ms": {
            "standard_deviation": float(np.std(residual_ms)),
            "p95_absolute": float(np.percentile(np.abs(residual_ms), 95)),
            "maximum_absolute": float(np.max(np.abs(residual_ms))),
        },
    }


def _command_settling(
    events: list[BoardEvent],
    times_ns: list[int],
    positions: list[np.ndarray],
) -> dict[str, Any]:
    if len(times_ns) < 2:
        raise ValueError("lowcmd contains fewer than two messages")
    values = np.stack(positions)
    changed = np.max(np.abs(np.diff(values, axis=0)), axis=1) > 1.0e-6
    change_times = np.asarray(times_ns[1:], dtype=np.int64)[changed]
    per_lift: list[dict[str, Any]] = []
    for event in events:
        if event.phase != "lifted":
            continue
        preceding = change_times[change_times < event.start_ns]
        if not preceding.size:
            continue
        per_lift.append(
            {
                "event_index": event.index,
                "repetition": event.repetition,
                "arm": event.arm,
                "command_static_before_observation_s": (event.start_ns - int(preceding[-1]))
                / 1.0e9,
            }
        )
    stable = [
        item["command_static_before_observation_s"]
        for item in per_lift
        if item["repetition"] is not None and item["repetition"] >= 2
    ]
    return {
        "joint_command_change_threshold_rad": 1.0e-6,
        "per_lift": per_lift,
        "steady_repetitions_2_plus_s": (
            None
            if not stable
            else {
                "mean": float(np.mean(stable)),
                "minimum": float(np.min(stable)),
                "maximum": float(np.max(stable)),
            }
        ),
    }


def _aggregate_sample(event: BoardEvent, samples: _EventSamples) -> ProprioceptiveSample:
    if len(samples.lowstate) < 3 or len(samples.torso_imu) < 3:
        raise ValueError(
            f"event {event.index} lacks synchronized Unitree state: "
            f"lowstate={len(samples.lowstate)}, torso_imu={len(samples.torso_imu)}"
        )
    unique_lowstate: list[tuple[int, int, np.ndarray, np.ndarray, np.ndarray]] = []
    previous_tick: int | None = None
    for item in samples.lowstate:
        if item[1] != previous_tick:
            unique_lowstate.append(item)
            previous_tick = item[1]
    return ProprioceptiveSample(
        timestamp_ns=event.center_ns,
        q29_rad=np.median(np.stack([item[2] for item in unique_lowstate]), axis=0),
        navigation_R_pelvis_imu=_mean_rotation_from_wxyz([item[3] for item in unique_lowstate]),
        navigation_R_torso_imu=_mean_rotation_from_wxyz([item[1] for item in samples.torso_imu]),
    )


def _evaluation_pairs(events: list[BoardEvent]) -> tuple[str, list[tuple[BoardEvent, BoardEvent]]]:
    pre_lift = {
        (event.repetition, event.arm): event for event in events if event.phase == "pre_lift"
    }
    lifted = [event for event in events if event.phase == "lifted"]
    if pre_lift:
        pairs = [(pre_lift[(event.repetition, event.arm)], event) for event in lifted]
        return "explicit_same_cycle_pre_lift", pairs

    pairs: list[tuple[BoardEvent, BoardEvent]] = []
    for event in lifted:
        if event.repetition is None or event.repetition < 2:
            continue
        if event.index == 0 or events[event.index - 1].phase != "returned":
            raise ValueError("legacy lift is not preceded by a returned observation")
        pairs.append((events[event.index - 1], event))
    if not pairs:
        raise ValueError("no evaluable lift pairs were found")
    return "legacy_immediately_preceding_return_repetitions_2_plus", pairs


def _acceleration_noise(samples: list[_EventSamples]) -> dict[str, Any]:
    sources: dict[str, list[np.ndarray]] = {
        "pelvis_imu": [],
        "torso_imu": [],
        "d435i_accelerometer": [],
    }
    for event in samples:
        if len(event.lowstate) >= 3:
            sources["pelvis_imu"].append(
                np.std(np.stack([item[4] for item in event.lowstate]), axis=0)
            )
        if len(event.torso_imu) >= 3:
            sources["torso_imu"].append(
                np.std(np.stack([item[2] for item in event.torso_imu]), axis=0)
            )
        if len(event.d435_accel) >= 3:
            sources["d435i_accelerometer"].append(
                np.std(np.stack([item[1] for item in event.d435_accel]), axis=0)
            )
    result: dict[str, Any] = {}
    for name, values in sources.items():
        if not values:
            result[name] = None
            continue
        array = np.stack(values)
        result[name] = {
            "endpoint_window_count": int(array.shape[0]),
            "mean_axis_standard_deviation_m_s2": np.mean(array, axis=0).tolist(),
            "maximum_axis_standard_deviation_m_s2": np.max(array, axis=0).tolist(),
        }
    return result


def _metric_summary(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(np.mean(array)),
        "standard_deviation": float(np.std(array)),
        "rms": float(np.sqrt(np.mean(np.square(array)))),
        "maximum": float(np.max(array)),
    }


def _summarize_trials(trials: list[dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for arm in ("left", "right"):
        arm_trials = [trial for trial in trials if trial["arm"] == arm]
        if not arm_trials:
            continue
        result[arm] = {
            "trial_count": len(arm_trials),
            "observed_camera_motion": {
                "translation_norm_mm": _metric_summary(
                    [
                        trial["observed_camera_motion"]["translation_norm_mm"]
                        for trial in arm_trials
                    ]
                ),
                "rotation_deg": _metric_summary(
                    [trial["observed_camera_motion"]["rotation_deg"] for trial in arm_trials]
                ),
            },
            "estimators": {},
        }
        for estimator in ESTIMATOR_NAMES:
            result[arm]["estimators"][estimator] = {
                "translation_error_norm_mm": _metric_summary(
                    [trial["estimators"][estimator]["translation_norm_mm"] for trial in arm_trials]
                ),
                "rotation_error_deg": _metric_summary(
                    [trial["estimators"][estimator]["rotation_deg"] for trial in arm_trials]
                ),
                "mean_translation_error_reference_xyz_mm": np.mean(
                    [
                        trial["estimators"][estimator]["translation_reference_xyz_mm"]
                        for trial in arm_trials
                    ],
                    axis=0,
                ).tolist(),
            }
    return result


def analyze_run(
    run_directory: Path,
    *,
    urdf_path: Path = DEFAULT_URDF,
    calibration_bundle_path: Path = DEFAULT_BUNDLE,
) -> dict[str, Any]:
    run_directory = run_directory.resolve()
    observations_path = run_directory / "board_observations.json"
    bag_directory = run_directory / "raw_episode/bag"
    if not observations_path.is_file():
        raise FileNotFoundError(f"board observations are unavailable: {observations_path}")
    if not bag_directory.is_dir():
        raise FileNotFoundError(f"raw MCAP directory is unavailable: {bag_directory}")
    observation_document, receipt_timed_events = _load_events(observations_path)
    precomputed_clock_models = _read_camera_clock_models(bag_directory)
    events, event_timing = _align_events_to_mcap(
        receipt_timed_events,
        precomputed_clock_models["/camera/color/camera_info"],
    )
    raw_samples, recording = _read_mcap(bag_directory, events)
    recording["board_event_timing"] = event_timing
    state_samples = [
        _aggregate_sample(event, samples)
        for event, samples in zip(events, raw_samples, strict=True)
    ]
    model = URDFModel(urdf_path)
    bundle = CalibrationBundle.load(calibration_bundle_path)
    estimators = AnchoredCameraPoseEstimators(model=model, calibration_bundle=bundle)
    reference_policy, pairs = _evaluation_pairs(events)

    trials: list[dict[str, Any]] = []
    for reference, current in pairs:
        anchor = CameraPoseAnchor(
            reference_T_camera=reference.board_T_camera,
            sample=state_samples[reference.index],
        )
        observed_motion = pose_error(reference.board_T_camera, current.board_T_camera)
        estimates = estimators.predict_all(anchor, state_samples[current.index])
        trials.append(
            {
                "repetition": current.repetition,
                "arm": current.arm,
                "reference_event_index": reference.index,
                "current_event_index": current.index,
                "reference_phase": reference.phase,
                "current_phase": current.phase,
                "reference_time_ns": reference.center_ns,
                "current_time_ns": current.center_ns,
                "observed_camera_motion": observed_motion,
                "estimators": {
                    name: pose_error(predicted, current.board_T_camera)
                    for name, predicted in estimates.items()
                },
            }
        )

    return {
        "schema_version": 1,
        "kind": "g1_camera_state_estimation_research_report",
        "commands_robot": False,
        "run_directory": str(run_directory),
        "chair_condition": observation_document.get("chair_condition"),
        "provenance": {
            "board_observations_sha256": _sha256(observations_path),
            "urdf": str(urdf_path.resolve()),
            "urdf_sha256": model.sha256,
            "calibration_bundle": str(calibration_bundle_path.resolve()),
            "calibration_bundle_sha256": bundle.content_sha256,
        },
        "evaluation": {
            "reference_policy": reference_policy,
            "first_repetition_excluded": reference_policy.startswith("legacy_"),
            "trial_count": len(trials),
            "trials": trials,
            "summary": _summarize_trials(trials),
        },
        "recording": recording,
        "accelerometer_endpoint_noise": _acceleration_noise(raw_samples),
        "estimator_contracts": ESTIMATOR_CONTRACTS,
        "interpretation_limits": [
            "The ChArUco board is the external reference used only for evaluation and anchoring.",
            "The fixed-pelvis and fixed-IMU-origin hypotheses are contact assumptions, not globally observable odometry.",
            "IMU plus joint state does not observe absolute translation or absolute yaw without contact or visual/depth updates.",
            "Endpoint acceleration noise is reported; accelerometer position integration is intentionally not presented as a millimetre-scale estimate.",
            "The legacy physical runs lack same-cycle pre-lift images, so repetition 1 is excluded and every later lift is anchored to the immediately preceding returned observation.",
        ],
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="g1-state-estimation-research")
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--urdf", type=Path, default=DEFAULT_URDF)
    parser.add_argument("--calibration-bundle", type=Path, default=DEFAULT_BUNDLE)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        report = analyze_run(
            args.run,
            urdf_path=args.urdf,
            calibration_bundle_path=args.calibration_bundle,
        )
        _atomic_write_json(args.output, report)
    except (FileNotFoundError, RuntimeError, TypeError, ValueError) as error:
        print(f"error: {error}")
        return 1
    endpoint_mean = {}
    for arm, summary in report["evaluation"]["summary"].items():
        hybrid = summary["estimators"]["hybrid_pelvis_position_torso_orientation"]
        endpoint_mean[arm] = {
            "observed_translation_mm": summary["observed_camera_motion"]["translation_norm_mm"][
                "mean"
            ],
            "observed_rotation_deg": summary["observed_camera_motion"]["rotation_deg"]["mean"],
            "hybrid_residual_translation_mm": hybrid["translation_error_norm_mm"]["mean"],
            "hybrid_residual_rotation_deg": hybrid["rotation_error_deg"]["mean"],
        }
    print(
        json.dumps(
            {
                "commands_robot": False,
                "run": str(args.run.resolve()),
                "output": str(args.output.resolve()),
                "reference_policy": report["evaluation"]["reference_policy"],
                "trial_count": report["evaluation"]["trial_count"],
                "endpoint_mean": endpoint_mean,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
