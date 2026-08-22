"""Direct raw-MCAP to LeRobot conversion for G1/Dex3 tabletop episodes.

The live recorder deliberately writes plain MCAP.  This module is the offline
learning-data boundary: it reads the ROS 2 schemas embedded in MCAP without a
ROS installation, aligns the official Unitree state/command topics to the
recorded color frames, and delegates dataset/video writing to LeRobot.

The artifact and encoder conventions follow RPM-lab-UMN/spark-data-collection
``data_pipeline/convert_episode_bag_to_lerobot.py`` at commit
``a7fbd8de12be54c963026cd247bb42d34c7c6952``.  The G1 message adapter is kept
here because SPARK's production converter is intentionally UR-specific.
"""

from __future__ import annotations

import argparse
import bisect
import json
import math
import shutil
import sys
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from g1_aprilcube_calibration.joint_map import G1_29_JOINT_NAMES
from g1_aprilcube_calibration.transports.unitree_dex3 import DEX3_MOTOR_JOINT_SUFFIXES

FPS = 15
STATE_MAX_AGE_MS = 50.0
# LowCmd is a zero-order-held motor target.  Allow the commissioned 0.5 s
# watchdog window so the intentional damping-controller -> trajectory-controller
# handoff is represented by its last effective command rather than discarded.
ACTION_MAX_AGE_MS = 500.0
DEPTH_MAX_SKEW_MS = 50.0
DEPTH_SCALE_M_PER_UNIT = 0.001
DEFAULT_DEPTH_MIN_M = 0.15
DEFAULT_DEPTH_MAX_M = 2.0

LOWSTATE_TOPIC = "/lowstate"
LOWCMD_TOPIC = "/lowcmd"
SECONDARY_IMU_TOPIC = "/secondary_imu"
LEFT_STATE_TOPIC = "/dex3/left/state"
RIGHT_STATE_TOPIC = "/dex3/right/state"
LEFT_CMD_TOPIC = "/dex3/left/cmd"
RIGHT_CMD_TOPIC = "/dex3/right/cmd"
COLOR_TOPIC = "/camera/color/image_raw"
DEPTH_TOPIC = "/camera/depth/image_rect_raw"

REQUIRED_TOPICS = (
    LOWSTATE_TOPIC,
    LOWCMD_TOPIC,
    SECONDARY_IMU_TOPIC,
    LEFT_STATE_TOPIC,
    RIGHT_STATE_TOPIC,
    LEFT_CMD_TOPIC,
    RIGHT_CMD_TOPIC,
    COLOR_TOPIC,
    DEPTH_TOPIC,
)

G1_JOINT_NAMES = tuple(G1_29_JOINT_NAMES)
LEFT_HAND_JOINT_NAMES = tuple(
    f"left_hand_{suffix}_joint" for suffix in DEX3_MOTOR_JOINT_SUFFIXES["left"]
)
RIGHT_HAND_JOINT_NAMES = tuple(
    f"right_hand_{suffix}_joint" for suffix in DEX3_MOTOR_JOINT_SUFFIXES["right"]
)
ALL_JOINT_NAMES = (*G1_JOINT_NAMES, *LEFT_HAND_JOINT_NAMES, *RIGHT_HAND_JOINT_NAMES)

IMU_COMPONENT_NAMES = (
    "quaternion_x",
    "quaternion_y",
    "quaternion_z",
    "quaternion_w",
    "gyroscope_x",
    "gyroscope_y",
    "gyroscope_z",
    "accelerometer_x",
    "accelerometer_y",
    "accelerometer_z",
    "roll",
    "pitch",
    "yaw",
)
IMU_NAMES = tuple(
    f"{location}_{component}"
    for location in ("pelvis", "torso", "left_hand", "right_hand")
    for component in IMU_COMPONENT_NAMES
)
PRESSURE_NAMES = tuple(
    f"{side}_sensor_{sensor_index:02d}_taxel_{taxel_index:02d}"
    for side in ("left", "right")
    for sensor_index in range(9)
    for taxel_index in range(12)
)


@dataclass(frozen=True)
class ImageRef:
    file_index: int
    topic_index: int
    timestamp_ns: int


@dataclass(frozen=True)
class AlignedFrame:
    color_ref: ImageRef
    depth_ref: ImageRef
    values: dict[str, np.ndarray]


@dataclass(frozen=True)
class _RawSample:
    timestamp_ns: int
    schema: Any
    channel: Any
    message: Any


@dataclass(frozen=True)
class _ColorSnapshot:
    color_ref: ImageRef
    samples: dict[str, tuple[int, np.ndarray]]


def _float_array(values: Iterable[Any], expected: int, label: str) -> np.ndarray:
    result = np.asarray(tuple(float(value) for value in values), dtype=np.float32)
    if result.shape != (expected,):
        raise RuntimeError(f"{label} has shape {result.shape}, expected {(expected,)}")
    if not np.all(np.isfinite(result)):
        raise RuntimeError(f"{label} contains NaN or infinity")
    return result


def _motor_field(motors: Iterable[Any], field: str, expected: int, label: str) -> np.ndarray:
    motor_list = list(motors)
    if len(motor_list) < expected:
        raise RuntimeError(f"{label} contains {len(motor_list)} motors, expected at least {expected}")
    return _float_array(
        (getattr(motor, field) for motor in motor_list[:expected]), expected, label
    )


def _imu_vector(imu: Any, label: str) -> np.ndarray:
    return _float_array(
        (
            *imu.quaternion,
            *imu.gyroscope,
            *imu.accelerometer,
            *imu.rpy,
        ),
        13,
        label,
    )


def parse_lowstate(msg: Any) -> np.ndarray:
    return np.concatenate(
        (
            _motor_field(msg.motor_state, "q", 29, "LowState positions"),
            _motor_field(msg.motor_state, "dq", 29, "LowState velocities"),
            _motor_field(msg.motor_state, "tau_est", 29, "LowState efforts"),
            _imu_vector(msg.imu_state, "LowState pelvis IMU"),
        )
    ).astype(np.float32, copy=False)


def lowcmd_is_active(msg: Any) -> bool:
    motors = list(msg.motor_cmd)
    if len(motors) < 29:
        return False
    return any(float(motor.kp) > 0.0 or float(motor.kd) > 0.0 for motor in motors[15:29])


def parse_lowcmd(msg: Any) -> np.ndarray:
    return np.concatenate(
        (
            _motor_field(msg.motor_cmd, "q", 29, "LowCmd positions"),
            _motor_field(msg.motor_cmd, "dq", 29, "LowCmd velocities"),
            _motor_field(msg.motor_cmd, "tau", 29, "LowCmd efforts"),
            _motor_field(msg.motor_cmd, "kp", 29, "LowCmd kp"),
            _motor_field(msg.motor_cmd, "kd", 29, "LowCmd kd"),
        )
    ).astype(np.float32, copy=False)


def parse_hand_state(msg: Any, *, side: str) -> np.ndarray:
    pressure_sensors = list(msg.press_sensor_state)
    if len(pressure_sensors) != 9:
        raise RuntimeError(
            f"{side} HandState has {len(pressure_sensors)} pressure sensors, expected 9"
        )
    pressure = _float_array(
        (
            value
            for sensor in pressure_sensors
            for value in sensor.pressure
        ),
        108,
        f"{side} HandState pressure",
    )
    return np.concatenate(
        (
            _motor_field(msg.motor_state, "q", 7, f"{side} HandState positions"),
            _motor_field(msg.motor_state, "dq", 7, f"{side} HandState velocities"),
            _motor_field(msg.motor_state, "tau_est", 7, f"{side} HandState efforts"),
            pressure,
            _imu_vector(msg.imu_state, f"{side} HandState IMU"),
        )
    ).astype(np.float32, copy=False)


def handcmd_is_active(msg: Any) -> bool:
    motors = list(msg.motor_cmd)
    return len(motors) >= 7 and all((int(motor.mode) & 0x80) == 0 for motor in motors[:7])


def parse_handcmd(msg: Any, *, side: str) -> np.ndarray:
    return np.concatenate(
        (
            _motor_field(msg.motor_cmd, "q", 7, f"{side} HandCmd positions"),
            _motor_field(msg.motor_cmd, "dq", 7, f"{side} HandCmd velocities"),
            _motor_field(msg.motor_cmd, "tau", 7, f"{side} HandCmd efforts"),
            _motor_field(msg.motor_cmd, "kp", 7, f"{side} HandCmd kp"),
            _motor_field(msg.motor_cmd, "kd", 7, f"{side} HandCmd kd"),
        )
    ).astype(np.float32, copy=False)


def parse_secondary_imu(msg: Any) -> np.ndarray:
    return _imu_vector(msg, "secondary torso IMU")


def _message_parsers() -> dict[str, tuple[Callable[[Any], np.ndarray], Callable[[Any], bool]]]:
    return {
        LOWSTATE_TOPIC: (parse_lowstate, lambda _msg: True),
        LOWCMD_TOPIC: (parse_lowcmd, lowcmd_is_active),
        SECONDARY_IMU_TOPIC: (parse_secondary_imu, lambda _msg: True),
        LEFT_STATE_TOPIC: (lambda msg: parse_hand_state(msg, side="left"), lambda _msg: True),
        RIGHT_STATE_TOPIC: (lambda msg: parse_hand_state(msg, side="right"), lambda _msg: True),
        LEFT_CMD_TOPIC: (lambda msg: parse_handcmd(msg, side="left"), handcmd_is_active),
        RIGHT_CMD_TOPIC: (lambda msg: parse_handcmd(msg, side="right"), handcmd_is_active),
    }


def _bag_files(bag_directory: Path) -> tuple[Path, ...]:
    files = tuple(sorted(bag_directory.glob("*.mcap")))
    if not files:
        raise FileNotFoundError(f"no MCAP files found under {bag_directory}")
    return files


def index_and_align_episode(
    bag_files: tuple[Path, ...],
    *,
    state_max_age_ms: float = STATE_MAX_AGE_MS,
    action_max_age_ms: float = ACTION_MAX_AGE_MS,
    depth_max_skew_ms: float = DEPTH_MAX_SKEW_MS,
) -> tuple[list[AlignedFrame], dict[str, Any], dict[str, Any]]:
    """Index and align an episode without decoding every high-rate sample.

    The original implementation used ``read_ros2_messages`` over every topic.
    That dynamically decoded roughly one million 250-1000 Hz state messages in
    a typical run even though the published timeline is the 15 Hz color camera.
    MCAP already exposes raw messages in log-time order, so retain the latest
    state sample and only decode it when a color frame arrives.  Commands need
    their active/timeout predicate, so retain the messages since the previous
    color frame and scan newest-to-oldest until the latest active command is
    found.  This preserves latest-before semantics while reducing decoding to
    the samples that can actually enter the LeRobot episode.
    """

    try:
        from mcap.reader import make_reader
        from mcap_ros2.decoder import DecoderFactory
    except ImportError as error:
        raise RuntimeError(
            "mcap_ros2 is unavailable; run tools/setup_lerobot_conversion.sh"
        ) from error

    parsers = _message_parsers()
    state_topics = (LOWSTATE_TOPIC, SECONDARY_IMU_TOPIC, LEFT_STATE_TOPIC, RIGHT_STATE_TOPIC)
    action_topics = (LOWCMD_TOPIC, LEFT_CMD_TOPIC, RIGHT_CMD_TOPIC)
    factory = DecoderFactory()
    latest_state: dict[str, _RawSample] = {}
    pending_commands: dict[str, list[_RawSample]] = {topic: [] for topic in action_topics}
    latest_active_command: dict[str, tuple[int, np.ndarray]] = {}
    first_timestamps: dict[str, int] = {}
    last_timestamps: dict[str, int] = {}
    snapshots: list[_ColorSnapshot] = []
    color_refs: list[ImageRef] = []
    depth_refs: list[ImageRef] = []
    raw_counts = {topic: 0 for topic in REQUIRED_TOPICS}
    decoded_counts = {topic: 0 for topic in parsers}
    started = time.monotonic()

    def decode(sample: _RawSample) -> Any:
        decoder = factory.decoder_for(sample.channel.message_encoding, sample.schema)
        if decoder is None:
            raise RuntimeError(
                f"no ROS 2 decoder for {sample.channel.topic} "
                f"({sample.channel.message_encoding}, {sample.schema.encoding})"
            )
        return decoder(sample.message.data)

    def update_commands() -> None:
        for topic in action_topics:
            pending = pending_commands[topic]
            if not pending:
                continue
            parser, predicate = parsers[topic]
            for sample in reversed(pending):
                decoded_counts[topic] += 1
                message = decode(sample)
                if predicate(message):
                    latest_active_command[topic] = (sample.timestamp_ns, parser(message))
                    first_timestamps.setdefault(topic, sample.timestamp_ns)
                    last_timestamps[topic] = sample.timestamp_ns
                    break
            pending.clear()

    for file_index, bag_file in enumerate(bag_files):
        topic_indices = {COLOR_TOPIC: 0, DEPTH_TOPIC: 0}
        with bag_file.open("rb") as stream:
            reader = make_reader(stream)
            for schema, channel, message in reader.iter_messages(
                topics=REQUIRED_TOPICS,
                log_time_order=True,
            ):
                if schema is None:
                    raise RuntimeError(f"required topic {channel.topic} has no MCAP schema")
                topic = channel.topic
                timestamp_ns = int(message.log_time)
                raw_counts[topic] += 1
                if topic in topic_indices:
                    reference = ImageRef(
                        file_index=file_index,
                        topic_index=topic_indices[topic],
                        timestamp_ns=timestamp_ns,
                    )
                    topic_indices[topic] += 1
                    if topic == DEPTH_TOPIC:
                        depth_refs.append(reference)
                        continue

                    color_refs.append(reference)
                    update_commands()
                    samples: dict[str, tuple[int, np.ndarray]] = dict(latest_active_command)
                    for state_topic in state_topics:
                        sample = latest_state.get(state_topic)
                        if sample is None:
                            continue
                        parser, _predicate = parsers[state_topic]
                        decoded_counts[state_topic] += 1
                        samples[state_topic] = (sample.timestamp_ns, parser(decode(sample)))
                    snapshots.append(_ColorSnapshot(color_ref=reference, samples=samples))
                    continue

                sample = _RawSample(
                    timestamp_ns=timestamp_ns,
                    schema=schema,
                    channel=channel,
                    message=message,
                )
                if topic in state_topics:
                    latest_state[topic] = sample
                    first_timestamps.setdefault(topic, timestamp_ns)
                    last_timestamps[topic] = timestamp_ns
                elif topic in action_topics:
                    pending_commands[topic].append(sample)

    # Commands after the last color frame still define the original timeline's
    # terminal bound.  Decoding newest-to-oldest finds the only value that can
    # affect that bound without decoding the entire high-rate command stream.
    update_commands()

    missing = [topic for topic in (*state_topics, *action_topics) if topic not in first_timestamps]
    if not color_refs:
        missing.append(COLOR_TOPIC)
    if not depth_refs:
        missing.append(DEPTH_TOPIC)
    if missing:
        raise RuntimeError(f"raw episode is missing required data: {sorted(missing)}")

    t_start_ns = max(
        *(first_timestamps[topic] for topic in (*state_topics, *action_topics)),
        color_refs[0].timestamp_ns,
        depth_refs[0].timestamp_ns,
    )
    t_end_ns = min(
        *(last_timestamps[topic] for topic in (*state_topics, *action_topics)),
        color_refs[-1].timestamp_ns,
        depth_refs[-1].timestamp_ns,
    )
    state_limit_ns = round(state_max_age_ms * 1_000_000.0)
    action_limit_ns = round(action_max_age_ms * 1_000_000.0)
    depth_limit_ns = round(depth_max_skew_ms * 1_000_000.0)
    aligned: list[AlignedFrame] = []
    rejected: list[dict[str, Any]] = []
    ages_ms: dict[str, list[float]] = {topic: [] for topic in parsers}
    depth_skews_ms: list[float] = []

    for snapshot in snapshots:
        timestamp_ns = snapshot.color_ref.timestamp_ns
        if timestamp_ns < t_start_ns or timestamp_ns > t_end_ns:
            continue
        values: dict[str, np.ndarray] = {}
        failure: str | None = None
        for topic in (*state_topics, *action_topics):
            result = snapshot.samples.get(topic)
            if result is None:
                failure = f"no latest-before sample for {topic}"
                break
            sample_timestamp_ns, value = result
            age_ns = timestamp_ns - sample_timestamp_ns
            limit_ns = state_limit_ns if topic in state_topics else action_limit_ns
            if age_ns < 0:
                failure = f"latest-before sample for {topic} is from the future"
                break
            if age_ns > limit_ns:
                failure = f"{topic} age {age_ns / 1e6:.3f}ms exceeds {limit_ns / 1e6:.3f}ms"
                break
            values[topic] = value
            ages_ms[topic].append(age_ns / 1e6)

        depth_result = _nearest_ref(depth_refs, timestamp_ns) if failure is None else None
        if failure is None and depth_result is None:
            failure = "no depth sample"
        if failure is None and depth_result is not None:
            _depth_ref, skew_ns = depth_result
            if skew_ns > depth_limit_ns:
                failure = (
                    f"depth skew {skew_ns / 1e6:.3f}ms exceeds {depth_limit_ns / 1e6:.3f}ms"
                )
            else:
                depth_skews_ms.append(skew_ns / 1e6)
        if failure is not None:
            rejected.append({"timestamp_ns": timestamp_ns, "reason": failure})
            continue
        assert depth_result is not None
        aligned.append(
            AlignedFrame(
                color_ref=snapshot.color_ref,
                depth_ref=depth_result[0],
                values=values,
            )
        )

    if not aligned:
        raise RuntimeError("no RGB frames survived G1 state/action/depth alignment")
    if rejected:
        raise RuntimeError(
            "mid-episode alignment rejected recorded RGB frames; refusing a silently sparse dataset: "
            f"count={len(rejected)}, first={rejected[0]}"
        )

    def stats(values: list[float]) -> dict[str, float] | None:
        if not values:
            return None
        array = np.asarray(values, dtype=np.float64)
        return {
            "min": float(array.min()),
            "max": float(array.max()),
            "mean": float(array.mean()),
            "std": float(array.std()),
        }

    index_diagnostics = {
        "elapsed_s": time.monotonic() - started,
        "bag_files": [str(path) for path in bag_files],
        "topic_counts": raw_counts,
        "decoded_topic_counts": decoded_counts,
        "strategy": "raw_mcap_latest_before_camera",
    }
    alignment_diagnostics = {
        "timeline": "recorded_color_frames",
        "fps": FPS,
        "t_start_ns": t_start_ns,
        "t_end_ns": t_end_ns,
        "published_frame_count": len(aligned),
        "alignment_limits_ms": {
            "state": state_max_age_ms,
            "action": action_max_age_ms,
            "depth": depth_max_skew_ms,
        },
        "latest_before_age_ms": {topic: stats(values) for topic, values in ages_ms.items()},
        "depth_skew_ms": stats(depth_skews_ms),
    }
    return aligned, index_diagnostics, alignment_diagnostics


def _nearest_ref(refs: list[ImageRef], timestamp_ns: int) -> tuple[ImageRef, int] | None:
    index = bisect.bisect_left(refs, timestamp_ns, key=lambda ref: ref.timestamp_ns)
    candidates: list[ImageRef] = []
    if index < len(refs):
        candidates.append(refs[index])
    if index > 0:
        candidates.append(refs[index - 1])
    if not candidates:
        return None
    selected = min(candidates, key=lambda ref: abs(ref.timestamp_ns - timestamp_ns))
    return selected, abs(selected.timestamp_ns - timestamp_ns)


def _decode_rgb(msg: Any) -> np.ndarray:
    if str(msg.encoding).lower() not in {"rgb8", "bgr8"}:
        raise RuntimeError(f"unsupported RGB encoding {msg.encoding!r}")
    rows = np.frombuffer(msg.data, dtype=np.uint8).reshape(int(msg.height), int(msg.step))
    image = rows[:, : int(msg.width) * 3].reshape(int(msg.height), int(msg.width), 3)
    if str(msg.encoding).lower() == "bgr8":
        image = image[:, :, ::-1]
    return np.ascontiguousarray(image)


def _decode_depth(msg: Any) -> np.ndarray:
    encoding = str(msg.encoding).lower()
    if encoding not in {"16uc1", "mono16"}:
        raise RuntimeError(f"unsupported depth encoding {msg.encoding!r}")
    dtype = ">u2" if bool(msg.is_bigendian) else "<u2"
    rows = np.frombuffer(msg.data, dtype=dtype).reshape(int(msg.height), int(msg.step) // 2)
    return np.ascontiguousarray(rows[:, : int(msg.width)])


def extract_selected_images(
    bag_files: tuple[Path, ...],
    topic: str,
    references: Iterable[ImageRef],
    decoder: Callable[[Any], np.ndarray],
) -> dict[ImageRef, np.ndarray]:
    try:
        from mcap_ros2.reader import read_ros2_messages
    except ImportError as error:
        raise RuntimeError("mcap_ros2 is unavailable") from error
    wanted_by_file: dict[int, dict[int, ImageRef]] = {}
    for reference in references:
        wanted_by_file.setdefault(reference.file_index, {})[reference.topic_index] = reference
    result: dict[ImageRef, np.ndarray] = {}
    for file_index, wanted in wanted_by_file.items():
        for topic_index, item in enumerate(
            read_ros2_messages(
                bag_files[file_index], topics=[topic], log_time_order=False
            )
        ):
            reference = wanted.get(topic_index)
            if reference is not None:
                result[reference] = decoder(item.ros_msg)
                if all(candidate in result for candidate in wanted.values()):
                    break
        missing = set(wanted.values()) - set(result)
        if missing:
            raise RuntimeError(f"failed to extract {len(missing)} selected {topic} frames")
    return result


def _split_lowstate(value: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    return value[:29], value[29:58], value[58:87], value[87:100]


def _split_hand_state(
    value: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    return value[:7], value[7:14], value[14:21], value[21:129], value[129:142]


def _split_command(
    value: np.ndarray, count: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    return tuple(value[index * count : (index + 1) * count] for index in range(5))  # type: ignore[return-value]


def build_numeric_frame(aligned: AlignedFrame, first_timestamp_ns: int, task: str) -> dict[str, Any]:
    body_q, body_dq, body_effort, pelvis_imu = _split_lowstate(
        aligned.values[LOWSTATE_TOPIC]
    )
    torso_imu = aligned.values[SECONDARY_IMU_TOPIC]
    left_q, left_dq, left_effort, left_pressure, left_imu = _split_hand_state(
        aligned.values[LEFT_STATE_TOPIC]
    )
    right_q, right_dq, right_effort, right_pressure, right_imu = _split_hand_state(
        aligned.values[RIGHT_STATE_TOPIC]
    )
    body_cmd = _split_command(aligned.values[LOWCMD_TOPIC], 29)
    left_cmd = _split_command(aligned.values[LEFT_CMD_TOPIC], 7)
    right_cmd = _split_command(aligned.values[RIGHT_CMD_TOPIC], 7)

    action_parts = [
        np.concatenate((body_cmd[index], left_cmd[index], right_cmd[index])).astype(
            np.float32, copy=False
        )
        for index in range(5)
    ]
    return {
        "observation.state": np.concatenate((body_q, left_q, right_q)).astype(
            np.float32, copy=False
        ),
        "observation.velocity": np.concatenate((body_dq, left_dq, right_dq)).astype(
            np.float32, copy=False
        ),
        "observation.effort": np.concatenate(
            (body_effort, left_effort, right_effort)
        ).astype(np.float32, copy=False),
        "observation.pressure": np.concatenate((left_pressure, right_pressure)).astype(
            np.float32, copy=False
        ),
        "observation.imu": np.concatenate(
            (pelvis_imu, torso_imu, left_imu, right_imu)
        ).astype(np.float32, copy=False),
        "observation.source_time_s": np.asarray(
            [(aligned.color_ref.timestamp_ns - first_timestamp_ns) / 1_000_000_000.0],
            dtype=np.float32,
        ),
        "action": action_parts[0],
        "action.velocity": action_parts[1],
        "action.effort": action_parts[2],
        "action.kp": action_parts[3],
        "action.kd": action_parts[4],
        "task": task,
    }


def feature_schema(rgb_shape: tuple[int, ...], depth_shape: tuple[int, ...]) -> dict[str, Any]:
    joint_names = list(ALL_JOINT_NAMES)
    return {
        "observation.state": {"dtype": "float32", "shape": (43,), "names": joint_names},
        "observation.velocity": {
            "dtype": "float32",
            "shape": (43,),
            "names": joint_names,
        },
        "observation.effort": {
            "dtype": "float32",
            "shape": (43,),
            "names": joint_names,
        },
        "observation.pressure": {
            "dtype": "float32",
            "shape": (216,),
            "names": list(PRESSURE_NAMES),
        },
        "observation.imu": {
            "dtype": "float32",
            "shape": (52,),
            "names": list(IMU_NAMES),
        },
        "observation.source_time_s": {
            "dtype": "float32",
            "shape": (1,),
            "names": ["seconds_from_first_published_frame"],
        },
        "action": {"dtype": "float32", "shape": (43,), "names": joint_names},
        "action.velocity": {"dtype": "float32", "shape": (43,), "names": joint_names},
        "action.effort": {"dtype": "float32", "shape": (43,), "names": joint_names},
        "action.kp": {"dtype": "float32", "shape": (43,), "names": joint_names},
        "action.kd": {"dtype": "float32", "shape": (43,), "names": joint_names},
        "observation.images.head": {
            "dtype": "video",
            "shape": rgb_shape,
            "names": ["height", "width", "channels"],
        },
        "observation.depth.head": {
            "dtype": "video",
            "shape": depth_shape,
            "names": ["height", "width", "channels"],
            "info": {"is_depth_map": True},
        },
    }


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise TypeError(f"expected a JSON object in {path}")
    return value


def _episode_task(run_directory: Path, status: dict[str, Any]) -> str:
    task = str(status.get("task", "")).strip()
    return task.replace("_", " ") if task else run_directory.name.replace("_", " ")


def _compare_features(existing: dict[str, Any], expected: dict[str, Any]) -> None:
    ignored = {"index", "episode_index", "task_index", "timestamp", "frame_index"}
    existing_core = {
        key: value
        for key, value in existing.items()
        if key not in ignored and not key.startswith("meta/")
    }
    if set(existing_core) != set(expected):
        raise RuntimeError(
            f"existing dataset feature keys differ: {sorted(existing_core)} != {sorted(expected)}"
        )
    for key, expected_value in expected.items():
        existing_value = existing_core[key]
        if (
            existing_value["dtype"] != expected_value["dtype"]
            or tuple(existing_value["shape"]) != tuple(expected_value["shape"])
            or existing_value.get("names") != expected_value.get("names")
            or bool((existing_value.get("info") or {}).get("is_depth_map"))
            != bool((expected_value.get("info") or {}).get("is_depth_map"))
        ):
            raise RuntimeError(f"existing dataset feature mismatch for {key}")


def _validate_depth_encoder_bounds(
    features: dict[str, Any], depth_keys: Iterable[str], *, depth_min: float, depth_max: float
) -> None:
    for key in depth_keys:
        info = features[key].get("info") or {}
        existing_min = info.get("video.depth_min")
        existing_max = info.get("video.depth_max")
        if existing_min is None or existing_max is None:
            raise RuntimeError(f"existing depth feature {key} has no native encoder bounds")
        if not math.isclose(float(existing_min), depth_min) or not math.isclose(
            float(existing_max), depth_max
        ):
            raise RuntimeError(
                f"existing depth encoder bounds for {key} are "
                f"[{existing_min}, {existing_max}], not [{depth_min}, {depth_max}]"
            )


def _depth_report(
    histogram: np.ndarray,
    frame_count: int,
    *,
    scale: float,
    depth_min_m: float,
    depth_max_m: float,
) -> dict[str, Any]:
    total = int(histogram.sum())
    zero = int(histogram[0])
    valid_histogram = histogram[1:]
    valid = int(valid_histogram.sum())
    values_m = np.arange(1, 65536, dtype=np.float64) * scale
    cumulative = np.cumsum(valid_histogram)

    def percentile(p: float) -> float | None:
        if not valid:
            return None
        rank = max(1, math.ceil((p / 100.0) * valid))
        return float((np.searchsorted(cumulative, rank) + 1) * scale)

    return {
        "frame_count": frame_count,
        "depth_scale_meters_per_unit": scale,
        "zero_fraction": float(zero / total) if total else 0.0,
        "valid_depth_percentiles_m": {
            f"p{p:g}": percentile(p) for p in (0.0, 0.1, 1.0, 5.0, 50.0, 95.0, 99.0, 99.9, 100.0)
        },
        "encoder": {
            "depth_min_m": depth_min_m,
            "depth_max_m": depth_max_m,
            "valid_fraction_below_min": float(valid_histogram[values_m < depth_min_m].sum() / valid)
            if valid
            else 0.0,
            "valid_fraction_above_max": float(valid_histogram[values_m > depth_max_m].sum() / valid)
            if valid
            else 0.0,
            "invalid_zero_behavior": "decoded_as_depth_min",
        },
    }


def convert_episode(args: argparse.Namespace) -> int:
    try:
        from lerobot.configs import DepthEncoderConfig, RGBEncoderConfig
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
    except ImportError as error:
        raise RuntimeError(
            "LeRobot conversion environment is unavailable; run tools/setup_lerobot_conversion.sh"
        ) from error

    run_directory = args.run_directory.expanduser().resolve()
    episode_directory = run_directory / "raw_episode"
    manifest_path = episode_directory / "episode_manifest.json"
    status_path = run_directory / "status.json"
    if not manifest_path.is_file() or not status_path.is_file():
        raise FileNotFoundError(
            f"expected status.json and raw_episode/episode_manifest.json under {run_directory}"
        )
    manifest = _read_json(manifest_path)
    status = _read_json(status_path)
    if status.get("status") != "completed" and not args.allow_non_completed:
        raise RuntimeError(
            f"run status is {status.get('status')!r}, not 'completed'; pass --allow-non-completed explicitly"
        )
    if not bool((manifest.get("profile") or {}).get("camera_recording_enabled")):
        raise RuntimeError("episode was recorded without camera payloads")

    episode_id = str((manifest.get("episode") or {}).get("episode_id") or run_directory.name)
    dataset_root = args.published_root.expanduser().resolve() / args.dataset_id
    artifact_dir = dataset_root / "meta" / "g1_conversion" / episode_id
    if artifact_dir.exists():
        raise RuntimeError(f"episode already has conversion artifacts at {artifact_dir}")

    bag_files = _bag_files(episode_directory / "bag")
    aligned, index_diagnostics, alignment_diagnostics = index_and_align_episode(bag_files)
    first_depth_images = extract_selected_images(
        bag_files,
        DEPTH_TOPIC,
        [aligned[0].depth_ref],
        _decode_depth,
    )
    first_color_images = extract_selected_images(
        bag_files, COLOR_TOPIC, [aligned[0].color_ref], _decode_rgb
    )
    first_rgb = first_color_images[aligned[0].color_ref]
    first_depth = first_depth_images[aligned[0].depth_ref]
    features = feature_schema(first_rgb.shape, (*first_depth.shape, 1))
    rgb_encoder = RGBEncoderConfig(vcodec=args.vcodec)
    depth_encoder = DepthEncoderConfig(
        depth_min=args.depth_min_m,
        depth_max=args.depth_max_m,
    )

    info_path = dataset_root / "meta" / "info.json"
    if info_path.is_file():
        dataset = LeRobotDataset.resume(
            repo_id=args.dataset_id,
            root=dataset_root,
            rgb_encoder=rgb_encoder,
            depth_encoder=depth_encoder,
            streaming_encoding=True,
        )
        if int(dataset.fps) != FPS:
            raise RuntimeError(f"existing dataset fps is {dataset.fps}, expected {FPS}")
        _compare_features(dataset.meta.features, features)
        _validate_depth_encoder_bounds(
            dataset.meta.features,
            dataset.meta.depth_keys,
            depth_min=depth_encoder.depth_min,
            depth_max=depth_encoder.depth_max,
        )
    else:
        if dataset_root.exists() and any(dataset_root.iterdir()):
            raise RuntimeError(
                f"dataset target exists but is not initialized: {dataset_root}"
            )
        dataset = LeRobotDataset.create(
            repo_id=args.dataset_id,
            root=dataset_root,
            robot_type="unitree_g1_dual_dex3",
            fps=FPS,
            features=features,
            rgb_encoder=rgb_encoder,
            depth_encoder=depth_encoder,
            streaming_encoding=True,
        )

    dataset_episode_index = int(dataset.meta.total_episodes)
    task = args.task or _episode_task(run_directory, status)
    first_timestamp_ns = aligned[0].color_ref.timestamp_ns
    selected_color_by_file: dict[int, dict[int, int]] = {}
    selected_depth_by_file: dict[int, dict[int, list[int]]] = {}
    for frame_index, frame in enumerate(aligned):
        selected_color_by_file.setdefault(frame.color_ref.file_index, {})[
            frame.color_ref.topic_index
        ] = frame_index
        selected_depth_by_file.setdefault(frame.depth_ref.file_index, {}).setdefault(
            frame.depth_ref.topic_index, []
        ).append(frame_index)
    numeric_frames = [
        build_numeric_frame(frame, first_timestamp_ns, task) for frame in aligned
    ]

    started = time.monotonic()
    from mcap_ros2.reader import read_ros2_messages

    pending_images: dict[int, dict[str, np.ndarray]] = {}
    next_frame_index = 0
    depth_histogram = np.zeros(65536, dtype=np.int64)
    depth_frame_count = 0
    try:
        for file_index, bag_file in enumerate(bag_files):
            topic_indices = {COLOR_TOPIC: 0, DEPTH_TOPIC: 0}
            wanted_colors = selected_color_by_file.get(file_index, {})
            wanted_depths = selected_depth_by_file.get(file_index, {})
            for item in read_ros2_messages(
                bag_file,
                topics=[COLOR_TOPIC, DEPTH_TOPIC],
                log_time_order=True,
            ):
                topic = item.channel.topic
                topic_index = topic_indices[topic]
                topic_indices[topic] += 1
                if topic == COLOR_TOPIC:
                    frame_index = wanted_colors.get(topic_index)
                    if frame_index is not None:
                        pending_images.setdefault(frame_index, {})["rgb"] = _decode_rgb(
                            item.ros_msg
                        )
                else:
                    frame_indices = wanted_depths.get(topic_index)
                    if frame_indices:
                        depth = _decode_depth(item.ros_msg)
                        depth_histogram += np.bincount(depth.reshape(-1), minlength=65536)
                        depth_frame_count += 1
                        for frame_index in frame_indices:
                            pending_images.setdefault(frame_index, {})["depth"] = depth

                while next_frame_index in pending_images and set(
                    pending_images[next_frame_index]
                ) == {"rgb", "depth"}:
                    images = pending_images.pop(next_frame_index)
                    frame = numeric_frames[next_frame_index]
                    frame["observation.images.head"] = images["rgb"]
                    frame["observation.depth.head"] = (
                        images["depth"].astype(np.float32)
                        * np.float32(args.depth_scale_m_per_unit)
                    )[:, :, None]
                    dataset.add_frame(frame)
                    # StreamingVideoEncoder owns the queued arrays after
                    # add_frame().  Do not retain every multi-megabyte image
                    # through numeric_frames until reload verification.
                    del frame["observation.images.head"]
                    del frame["observation.depth.head"]
                    next_frame_index += 1
        if next_frame_index != len(aligned):
            raise RuntimeError(
                f"extracted images for only {next_frame_index}/{len(aligned)} aligned frames"
            )
        dataset.save_episode()
    finally:
        dataset.finalize()

    depth_report = _depth_report(
        depth_histogram,
        depth_frame_count,
        scale=args.depth_scale_m_per_unit,
        depth_min_m=args.depth_min_m,
        depth_max_m=args.depth_max_m,
    )

    reloaded = LeRobotDataset(
        repo_id=args.dataset_id,
        root=dataset_root,
        episodes=[dataset_episode_index],
        depth_output_unit="m",
    )
    try:
        if len(reloaded) != len(aligned):
            raise RuntimeError(
                f"reloaded episode has {len(reloaded)} frames, expected {len(aligned)}"
            )
        for index in (0, len(reloaded) // 2, len(reloaded) - 1):
            sample = reloaded[index]
            if tuple(sample["observation.images.head"].shape) != (
                first_rgb.shape[2],
                first_rgb.shape[0],
                first_rgb.shape[1],
            ):
                raise RuntimeError("reloaded RGB shape differs from the source")
            if tuple(sample["observation.depth.head"].shape) != (
                1,
                first_depth.shape[0],
                first_depth.shape[1],
            ):
                raise RuntimeError("reloaded depth shape differs from the source")
            for key, expected in numeric_frames[index].items():
                if key in {"task", "observation.images.head", "observation.depth.head"}:
                    continue
                actual = np.asarray(sample[key])
                expected_array = np.asarray(expected)
                if actual.size == expected_array.size == 1:
                    actual = actual.reshape(1)
                    expected_array = expected_array.reshape(1)
                if actual.shape != expected_array.shape or not np.allclose(
                    actual, expected_array, rtol=1e-6, atol=1e-6
                ):
                    raise RuntimeError(
                        f"reloaded numeric feature {key} differs at frame {index}"
                    )
    finally:
        reloaded.finalize()

    source_dir = dataset_root / "meta" / "g1_source" / episode_id
    source_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(manifest_path, source_dir / manifest_path.name)
    shutil.copy2(status_path, source_dir / status_path.name)
    notes_path = episode_directory / "notes.md"
    if notes_path.is_file():
        shutil.copy2(notes_path, source_dir / notes_path.name)

    diagnostics = {
        "schema_version": 1,
        "episode_id": episode_id,
        "source_run": str(run_directory),
        "dataset_id": args.dataset_id,
        "dataset_root": str(dataset_root),
        "dataset_episode_index": dataset_episode_index,
        "published_frame_count": len(aligned),
        "task": task,
        "source": "raw_mcap",
        "mcap_reader": "mcap_ros2 dynamic schema decoder",
        "index": index_diagnostics,
        "alignment": alignment_diagnostics,
        "depth": depth_report,
        "writer_elapsed_s": time.monotonic() - started,
        "video_encoding": "LeRobot StreamingVideoEncoder",
        "verified_reload_indices": [0, len(aligned) // 2, len(aligned) - 1],
        "source_snapshot": str(source_dir.relative_to(dataset_root)),
    }
    _write_json(artifact_dir / "diagnostics.json", diagnostics)
    _write_json(
        artifact_dir / "conversion_summary.json",
        {
            "episode_id": episode_id,
            "dataset_episode_index": dataset_episode_index,
            "published_frame_count": len(aligned),
            "status": "verified",
            "source": "raw_mcap",
        },
    )
    with (artifact_dir / "profile.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(
            {
                "profile_name": "g1_seated_tabletop_lerobot_v1",
                "fps": FPS,
                "timeline": "recorded_color_frames",
                "state_max_age_ms": STATE_MAX_AGE_MS,
                "action_max_age_ms": ACTION_MAX_AGE_MS,
                "depth_max_skew_ms": DEPTH_MAX_SKEW_MS,
                "depth_scale_meters_per_unit": args.depth_scale_m_per_unit,
                "depth_min_m": args.depth_min_m,
                "depth_max_m": args.depth_max_m,
            },
            handle,
            sort_keys=False,
        )

    print(f"converted {episode_id} -> {dataset_root}")
    print(f"dataset_episode_index={dataset_episode_index}")
    print(f"published_frames={len(aligned)}")
    print("status=verified")
    print(f"artifacts={artifact_dir}")
    return 0


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_directory", type=Path)
    parser.add_argument("--dataset-id", required=True)
    parser.add_argument(
        "--published-root",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "published",
    )
    parser.add_argument("--task", default="")
    parser.add_argument("--vcodec", default="h264")
    parser.add_argument("--depth-scale-m-per-unit", type=float, default=DEPTH_SCALE_M_PER_UNIT)
    parser.add_argument("--depth-min-m", type=float, default=DEFAULT_DEPTH_MIN_M)
    parser.add_argument("--depth-max-m", type=float, default=DEFAULT_DEPTH_MAX_M)
    parser.add_argument("--allow-non-completed", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    if not math.isfinite(args.depth_scale_m_per_unit) or args.depth_scale_m_per_unit <= 0:
        parser.error("--depth-scale-m-per-unit must be positive and finite")
    if not (
        math.isfinite(args.depth_min_m)
        and math.isfinite(args.depth_max_m)
        and 0.0 < args.depth_min_m < args.depth_max_m
    ):
        parser.error("depth bounds must be finite and satisfy 0 < min < max")
    return convert_episode(args)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
