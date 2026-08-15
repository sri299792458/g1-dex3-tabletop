"""No-robot benchmark for rosbag interference with the 250 Hz control loop."""

from __future__ import annotations

import argparse
import json
import math
import os
import resource
import signal
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import yaml

IMAGE_TOPIC = "/g1_recording_benchmark/camera/color/image_raw"
CAMERA_INFO_TOPIC = "/g1_recording_benchmark/camera/color/camera_info"
STATE_TOPIC = "/g1_recording_benchmark/lowstate"
COMMAND_TOPIC = "/g1_recording_benchmark/lowcmd"
WIDTH = 1280
HEIGHT = 720
IMAGE_RATE_HZ = 15.0
CONTROL_RATE_HZ = 250.0
MODULE_NAME = "g1_dex3_tabletop.recording_benchmark"


def summarize_tick_times(tick_times_ns: list[int], *, rate_hz: float) -> dict[str, Any]:
    """Summarize observed tick-to-tick gaps without inventing a pass threshold."""

    if rate_hz <= 0:
        raise ValueError("rate_hz must be positive")
    if len(tick_times_ns) < 2:
        raise ValueError("at least two tick timestamps are required")
    gaps_ms = np.diff(np.asarray(tick_times_ns, dtype=np.int64)) / 1_000_000.0
    period_ms = 1_000.0 / rate_hz
    conservative_missed_periods = sum(
        max(math.floor(float(gap) / period_ms) - 1, 0) for gap in gaps_ms
    )
    return {
        "tick_count": len(tick_times_ns),
        "interval_count": int(gaps_ms.size),
        "nominal_period_ms": period_ms,
        "mean_gap_ms": float(np.mean(gaps_ms)),
        "p50_gap_ms": float(np.percentile(gaps_ms, 50)),
        "p95_gap_ms": float(np.percentile(gaps_ms, 95)),
        "p99_gap_ms": float(np.percentile(gaps_ms, 99)),
        "p99_9_gap_ms": float(np.percentile(gaps_ms, 99.9)),
        "maximum_gap_ms": float(np.max(gaps_ms)),
        "gaps_over_5ms": int(np.count_nonzero(gaps_ms > 5.0)),
        "gaps_over_10ms": int(np.count_nonzero(gaps_ms > 10.0)),
        "gaps_over_20ms": int(np.count_nonzero(gaps_ms > 20.0)),
        "gaps_over_50ms": int(np.count_nonzero(gaps_ms > 50.0)),
        "conservative_missed_periods": int(conservative_missed_periods),
    }


def aggregate_results(results: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Aggregate repeated trials while retaining the worst observed tails."""

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for result in results:
        grouped[str(result["scenario"])].append(result)
    aggregate: dict[str, dict[str, Any]] = {}
    for scenario, trials in grouped.items():
        aggregate[scenario] = {
            "trials": len(trials),
            "median_p99_gap_ms": float(
                np.median([trial["timing"]["p99_gap_ms"] for trial in trials])
            ),
            "worst_p99_9_gap_ms": max(float(trial["timing"]["p99_9_gap_ms"]) for trial in trials),
            "worst_maximum_gap_ms": max(
                float(trial["timing"]["maximum_gap_ms"]) for trial in trials
            ),
            "total_gaps_over_10ms": sum(
                int(trial["timing"]["gaps_over_10ms"]) for trial in trials
            ),
            "total_gaps_over_20ms": sum(
                int(trial["timing"]["gaps_over_20ms"]) for trial in trials
            ),
            "total_gaps_over_50ms": sum(
                int(trial["timing"]["gaps_over_50ms"]) for trial in trials
            ),
            "minimum_camera_receive_rate_hz": min(
                float(trial["camera_receive_rate_hz"]) for trial in trials
            ),
            "mean_bag_write_mib_s": float(
                np.mean([trial["bag"]["write_mib_s"] for trial in trials])
            ),
        }
    return aggregate


def _atomic_write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _run_camera_publisher() -> int:
    import rclpy
    from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
    from sensor_msgs.msg import CameraInfo, Image

    rclpy.init()
    node = rclpy.create_node("g1_recording_benchmark_camera_publisher")
    qos = QoSProfile(
        history=HistoryPolicy.KEEP_LAST,
        depth=30,
        reliability=ReliabilityPolicy.RELIABLE,
        durability=DurabilityPolicy.VOLATILE,
    )
    image_publisher = node.create_publisher(Image, IMAGE_TOPIC, qos)
    info_publisher = node.create_publisher(CameraInfo, CAMERA_INFO_TOPIC, qos)
    image = Image()
    image.header.frame_id = "g1_benchmark_color_optical_frame"
    image.height = HEIGHT
    image.width = WIDTH
    image.encoding = "rgb8"
    image.is_bigendian = False
    image.step = WIDTH * 3
    image.data = bytes((index * 17 + 31) % 256 for index in range(WIDTH * HEIGHT * 3))
    info = CameraInfo()
    info.header.frame_id = image.header.frame_id
    info.width = WIDTH
    info.height = HEIGHT
    info.distortion_model = "plumb_bob"
    info.d = [0.0] * 5
    info.k = [900.0, 0.0, WIDTH / 2, 0.0, 900.0, HEIGHT / 2, 0.0, 0.0, 1.0]
    info.r = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
    info.p = [900.0, 0.0, WIDTH / 2, 0.0, 0.0, 900.0, HEIGHT / 2, 0.0, 0.0, 0.0, 1.0, 0.0]
    period_s = 1.0 / IMAGE_RATE_HZ
    deadline = time.monotonic()
    try:
        while rclpy.ok():
            stamp = node.get_clock().now().to_msg()
            image.header.stamp = stamp
            info.header.stamp = stamp
            info_publisher.publish(info)
            image_publisher.publish(image)
            deadline += period_s
            remaining = deadline - time.monotonic()
            if remaining > 0:
                time.sleep(remaining)
            else:
                deadline = time.monotonic()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    return 0


def _run_state_publisher() -> int:
    import rclpy
    from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
    from sensor_msgs.msg import JointState

    rclpy.init()
    node = rclpy.create_node("g1_recording_benchmark_state_publisher")
    qos = QoSProfile(
        history=HistoryPolicy.KEEP_LAST,
        depth=2,
        reliability=ReliabilityPolicy.RELIABLE,
        durability=DurabilityPolicy.VOLATILE,
    )
    state_publisher = node.create_publisher(JointState, STATE_TOPIC, qos)
    command_publisher = node.create_publisher(JointState, COMMAND_TOPIC, qos)
    names = [f"g1_joint_{index:02d}" for index in range(29)]
    state = JointState(name=names, position=[0.0] * 29, velocity=[0.0] * 29, effort=[0.0] * 29)
    command = JointState(name=names, position=[0.0] * 29, velocity=[], effort=[0.0] * 29)
    period_s = 1.0 / CONTROL_RATE_HZ
    deadline = time.monotonic()
    sequence = 0
    try:
        while rclpy.ok():
            stamp = node.get_clock().now().to_msg()
            state.header.stamp = stamp
            command.header.stamp = stamp
            phase = sequence * 0.001
            state.position[0] = phase
            command.position[0] = phase
            state_publisher.publish(state)
            command_publisher.publish(command)
            sequence += 1
            deadline += period_s
            remaining = deadline - time.monotonic()
            if remaining > 0:
                time.sleep(remaining)
            else:
                deadline = time.monotonic()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    return 0


def _run_workload(duration_s: float, output: Path) -> int:
    import rclpy

    from g1_aprilcube_calibration.executor_driver import (
        ExecutorControlDriver,
        SynchronizedPoseExecutor,
    )
    from g1_aprilcube_calibration.ros.camera_adapter import ROSCameraSubscriber

    class TimingExecutor:
        def __init__(self) -> None:
            self.tick_times_ns: list[int] = []

        def tick(self):
            self.tick_times_ns.append(time.monotonic_ns())

    class CountingCameraSubscriber(ROSCameraSubscriber):
        def __init__(self, *args, **kwargs) -> None:
            self.received_frames = 0
            super().__init__(*args, **kwargs)

        def _on_image(self, message: Any) -> None:
            super()._on_image(message)
            if self.last_error is None:
                self.received_frames += 1

    rclpy.init()
    node = rclpy.create_node("g1_recording_benchmark_control_workload")
    camera = CountingCameraSubscriber(
        node,
        image_topic=IMAGE_TOPIC,
        camera_info_topic=CAMERA_INFO_TOPIC,
        camera_name="g1_benchmark_color",
        serial_number="synthetic",
        reliability="reliable",
        qos_depth=2,
        maximum_frames=30,
    )
    timing_executor = TimingExecutor()
    driver = ExecutorControlDriver(
        SynchronizedPoseExecutor(timing_executor),
        rate_hz=CONTROL_RATE_HZ,
    )
    discovery_deadline_s = time.monotonic() + 10.0
    while camera.received_frames < 3 and time.monotonic() < discovery_deadline_s:
        rclpy.spin_once(node, timeout_sec=0.05)
    if camera.received_frames < 3:
        raise RuntimeError(
            "camera workload did not receive three valid warm-up frames: "
            f"{camera.last_error or 'no callback'}"
        )
    camera.received_frames = 0
    started_s = time.monotonic()
    driver.start()
    try:
        while time.monotonic() - started_s < duration_s:
            rclpy.spin_once(node, timeout_sec=0.01)
            driver.check()
    finally:
        driver.close()
        camera.close()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    elapsed_s = time.monotonic() - started_s
    result = {
        "duration_s": elapsed_s,
        "timing": summarize_tick_times(
            timing_executor.tick_times_ns,
            rate_hz=CONTROL_RATE_HZ,
        ),
        "camera_frames_received": camera.received_frames,
        "camera_receive_rate_hz": camera.received_frames / elapsed_s,
        "camera_last_error": camera.last_error,
    }
    _atomic_write_json(output, result)
    return 0


def _stop_process(process: subprocess.Popen, *, signal_number: int, timeout_s: float) -> None:
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal_number)
        process.wait(timeout=timeout_s)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait(timeout=5.0)


def _start_role(role: str, log_path: Path) -> tuple[subprocess.Popen, Any]:
    log_handle = log_path.open("w", encoding="utf-8")
    process = subprocess.Popen(
        [sys.executable, "-m", MODULE_NAME, role],
        stdout=log_handle,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    return process, log_handle


def _wait_for_topics(topics: set[str], timeout_s: float = 10.0) -> None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        result = subprocess.run(
            ["ros2", "topic", "list"],
            check=False,
            capture_output=True,
            text=True,
            timeout=5.0,
        )
        live = set(result.stdout.splitlines())
        if topics <= live:
            return
        time.sleep(0.25)
    raise RuntimeError(f"synthetic topics did not become live: {sorted(topics)}")


def _bag_metrics(bag_dir: Path, *, duration_s: float) -> dict[str, Any]:
    if not bag_dir.exists():
        return {
            "bytes": 0,
            "duration_s": 0.0,
            "write_mib_s": 0.0,
            "message_counts": {},
            "message_rates_hz": {},
        }
    total_bytes = sum(path.stat().st_size for path in bag_dir.rglob("*") if path.is_file())
    message_counts: dict[str, int] = {}
    bag_duration_s = duration_s
    metadata_path = bag_dir / "metadata.yaml"
    if metadata_path.exists():
        metadata = yaml.safe_load(metadata_path.read_text(encoding="utf-8"))
        info = metadata.get("rosbag2_bagfile_information", {})
        bag_duration_s = float(info.get("duration", {}).get("nanoseconds", 0)) / 1e9
        for entry in info.get("topics_with_message_count", []):
            topic = entry.get("topic_metadata", {}).get("name")
            if topic:
                message_counts[str(topic)] = int(entry.get("message_count", 0))
    return {
        "bytes": total_bytes,
        "duration_s": bag_duration_s,
        "write_mib_s": total_bytes / (1024 * 1024) / max(bag_duration_s, 1e-9),
        "message_counts": message_counts,
        "message_rates_hz": {
            topic: count / max(bag_duration_s, 1e-9) for topic, count in message_counts.items()
        },
    }


def _run_trial(
    *,
    scenario: str,
    trial: int,
    duration_s: float,
    trial_dir: Path,
) -> dict[str, Any]:
    trial_dir.mkdir(parents=True, exist_ok=False)
    camera_process = state_process = recorder_process = None
    camera_log = state_log = recorder_log = None
    usage_before = resource.getrusage(resource.RUSAGE_CHILDREN)
    try:
        camera_process, camera_log = _start_role("_camera", trial_dir / "camera.log")
        state_process, state_log = _start_role("_state", trial_dir / "state.log")
        _wait_for_topics({IMAGE_TOPIC, CAMERA_INFO_TOPIC, STATE_TOPIC, COMMAND_TOPIC})
        bag_dir = trial_dir / "bag"
        if scenario != "baseline":
            topics = [STATE_TOPIC, COMMAND_TOPIC]
            if scenario == "state_plus_raw_rgb":
                topics.extend([IMAGE_TOPIC, CAMERA_INFO_TOPIC])
            recorder_log = (trial_dir / "recorder.log").open("w", encoding="utf-8")
            recorder_process = subprocess.Popen(
                [
                    "ros2",
                    "bag",
                    "record",
                    "--output",
                    str(bag_dir),
                    "--storage",
                    "mcap",
                    *topics,
                ],
                stdout=recorder_log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            time.sleep(2.0)
            if recorder_process.poll() is not None:
                raise RuntimeError(
                    f"rosbag recorder exited with {recorder_process.returncode}; "
                    f"see {trial_dir / 'recorder.log'}"
                )
        workload_path = trial_dir / "workload.json"
        workload_log_path = trial_dir / "workload.log"
        with workload_log_path.open("w", encoding="utf-8") as workload_log:
            workload = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    MODULE_NAME,
                    "_workload",
                    "--duration-s",
                    str(duration_s),
                    "--output",
                    str(workload_path),
                ],
                stdout=workload_log,
                stderr=subprocess.STDOUT,
                check=False,
            )
        if workload.returncode != 0:
            raise RuntimeError(
                f"control workload exited with {workload.returncode}; see {workload_log_path}"
            )
    finally:
        if recorder_process is not None:
            _stop_process(recorder_process, signal_number=signal.SIGINT, timeout_s=20.0)
        for process in (camera_process, state_process):
            if process is not None:
                _stop_process(process, signal_number=signal.SIGTERM, timeout_s=5.0)
        for handle in (camera_log, state_log, recorder_log):
            if handle is not None:
                handle.close()
    usage_after = resource.getrusage(resource.RUSAGE_CHILDREN)
    workload_result = json.loads(workload_path.read_text(encoding="utf-8"))
    result = {
        "scenario": scenario,
        "trial": trial,
        **workload_result,
        "bag": _bag_metrics(trial_dir / "bag", duration_s=duration_s),
        "child_cpu_user_s": usage_after.ru_utime - usage_before.ru_utime,
        "child_cpu_system_s": usage_after.ru_stime - usage_before.ru_stime,
    }
    _atomic_write_json(trial_dir / "result.json", result)
    return result


def _print_summary(aggregate: dict[str, dict[str, Any]]) -> None:
    print("\nRecording interference summary")
    print(
        "scenario               p99 median   p99.9 worst   max worst   >10ms  >50ms  "
        "camera Hz  bag MiB/s"
    )
    for scenario in ("baseline", "state_only", "state_plus_raw_rgb"):
        value = aggregate[scenario]
        print(
            f"{scenario:22s} "
            f"{value['median_p99_gap_ms']:10.3f} "
            f"{value['worst_p99_9_gap_ms']:13.3f} "
            f"{value['worst_maximum_gap_ms']:11.3f} "
            f"{value['total_gaps_over_10ms']:7d} "
            f"{value['total_gaps_over_50ms']:6d} "
            f"{value['minimum_camera_receive_rate_hz']:9.2f} "
            f"{value['mean_bag_write_mib_s']:10.2f}"
        )


def _run_benchmark(args: argparse.Namespace) -> int:
    if args.duration_s < 5:
        raise ValueError("duration must be at least 5 seconds")
    if args.trials <= 0:
        raise ValueError("trials must be positive")
    timestamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    run_dir = Path(args.output_root).resolve() / timestamp
    run_dir.mkdir(parents=True, exist_ok=False)
    print("NO-ROBOT RECORDING BENCHMARK — ROS_LOCALHOST_ONLY must be 1; no Unitree topic or ")
    print("robot command publisher is created.")
    if os.environ.get("ROS_LOCALHOST_ONLY") != "1":
        raise RuntimeError("ROS_LOCALHOST_ONLY=1 is required for the no-robot benchmark")
    scenarios = ("baseline", "state_only", "state_plus_raw_rgb")
    results: list[dict[str, Any]] = []
    for trial_index in range(args.trials):
        rotated = (
            scenarios[trial_index % len(scenarios) :] + scenarios[: trial_index % len(scenarios)]
        )
        for scenario in rotated:
            print(f"running trial {trial_index + 1}/{args.trials}: {scenario}", flush=True)
            result = _run_trial(
                scenario=scenario,
                trial=trial_index + 1,
                duration_s=args.duration_s,
                trial_dir=run_dir / f"trial_{trial_index + 1:02d}_{scenario}",
            )
            results.append(result)
            print(
                f"  max gap={result['timing']['maximum_gap_ms']:.3f}ms; "
                f">50ms={result['timing']['gaps_over_50ms']}; "
                f"camera={result['camera_receive_rate_hz']:.2f}Hz; "
                f"bag={result['bag']['write_mib_s']:.2f}MiB/s; "
                "recorded image="
                f"{result['bag']['message_rates_hz'].get(IMAGE_TOPIC, 0.0):.2f}Hz",
                flush=True,
            )
    aggregate = aggregate_results(results)
    report = {
        "schema_version": 1,
        "commands_robot": False,
        "ros_localhost_only": os.environ.get("ROS_LOCALHOST_ONLY"),
        "duration_s": args.duration_s,
        "trials": args.trials,
        "image": {
            "width": WIDTH,
            "height": HEIGHT,
            "encoding": "rgb8",
            "rate_hz": IMAGE_RATE_HZ,
            "payload_mib_s": WIDTH * HEIGHT * 3 * IMAGE_RATE_HZ / (1024 * 1024),
        },
        "control_rate_hz": CONTROL_RATE_HZ,
        "results": results,
        "aggregate": aggregate,
        "interpretation_boundary": (
            "This detects laptop scheduling/DDS/disk contention without robot motion. "
            "It is not physical commissioning and cannot prove hardware safety."
        ),
    }
    _atomic_write_json(run_dir / "report.json", report)
    _print_summary(aggregate)
    print(f"\nreport: {run_dir / 'report.json'}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    run = subparsers.add_parser("run", help="run all no-robot recording scenarios")
    run.add_argument("--duration-s", type=float, default=15.0)
    run.add_argument("--trials", type=int, default=3)
    run.add_argument("--output-root", default="work/recording_benchmark")
    subparsers.add_parser("_camera", help=argparse.SUPPRESS)
    subparsers.add_parser("_state", help=argparse.SUPPRESS)
    workload = subparsers.add_parser("_workload", help=argparse.SUPPRESS)
    workload.add_argument("--duration-s", type=float, required=True)
    workload.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "run":
        return _run_benchmark(args)
    if args.command == "_camera":
        return _run_camera_publisher()
    if args.command == "_state":
        return _run_state_publisher()
    if args.command == "_workload":
        return _run_workload(args.duration_s, args.output)
    raise AssertionError(args.command)


if __name__ == "__main__":
    raise SystemExit(main())
