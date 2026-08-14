"""Single-approval seated cube pick/lift/replace hardware workflow.

Importing this module is inert. DDS publishers are constructed only inside
``run_tabletop`` after the explicit operator approval.
"""

from __future__ import annotations

import hashlib
import json
import select
import subprocess
import sys
import termios
import time
import tty
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

import cv2
from aprilcube import CorrespondenceDetector

from g1_aprilcube_calibration.activation_handoff import build_activation_handoff
from g1_aprilcube_calibration.calibration_bundle import CalibrationBundle
from g1_aprilcube_calibration.clock import SystemClock
from g1_aprilcube_calibration.config import QualityThresholds
from g1_aprilcube_calibration.executor_driver import (
    ExecutorControlDriver,
    SynchronizedPoseExecutor,
)
from g1_aprilcube_calibration.executor_state_machine import ExecutorState, PoseExecutor
from g1_aprilcube_calibration.pose_schema import PoseSet
from g1_aprilcube_calibration.process_lock import CommandOwnerLock
from g1_aprilcube_calibration.readiness import StateSampleBuffer
from g1_aprilcube_calibration.ros.camera_adapter import ROSCameraSubscriber, ROSImageFrame
from g1_aprilcube_calibration.transports.unitree_arm_sdk import UnitreeLowStateObserver
from g1_aprilcube_calibration.transports.unitree_debug_lowcmd import (
    UnitreeDebugLowCmdTransport,
)
from g1_aprilcube_calibration.transports.unitree_dex3 import (
    UnitreeDex3PostureController,
    UnitreeDex3StateObserver,
)
from g1_aprilcube_calibration.urdf_model import URDFModel
from g1_dex3_tabletop.calibration_candidates import camera_info_from_hardware
from g1_dex3_tabletop.execution_plan import pose_set_from_trajectories
from g1_dex3_tabletop.hardware_config import (
    debug_lowcmd_config,
    dex3_config,
    executor_config,
    gravity_feedforward,
    load_hardware,
    recording_configs,
    resolve_hardware_path,
    transport_config,
    watchdog,
)
from g1_dex3_tabletop.planning.contracts import RobotSnapshot, atomic_write_json
from g1_dex3_tabletop.tabletop_contracts import (
    SupportedEscapePlan,
    TabletopTaskPlan,
)
from g1_dex3_tabletop.tabletop_perception import observe_resting_cube
from g1_dex3_tabletop.tabletop_workflow import (
    assemble_execution_plan,
    build_tabletop_request,
    request_at_clearance,
)

ROOT = Path(__file__).resolve().parents[2]
MOTION_ACK = "I CONFIRM THE G1 IS SECURED BY THE LOAD-BEARING HARNESS AND THE WORKSPACE IS CLEAR"


def _run_id() -> str:
    return datetime.now(timezone.utc).strftime("tabletop_%Y%m%dT%H%M%SZ")


def _snapshot(state, hands) -> RobotSnapshot:
    return RobotSnapshot(
        measured_q29_rad=tuple(state.position),
        left_dex3_q_rad=tuple(hands.left.position),
        right_dex3_q_rad=tuple(hands.right.position),
    )


def _wait_for_state(observer, timeout_s: float = 5.0):
    deadline = time.monotonic() + timeout_s
    last = "no sample"
    while time.monotonic() < deadline:
        try:
            return observer.observe()
        except RuntimeError as error:
            last = str(error)
        time.sleep(0.02)
    raise RuntimeError("timed out waiting for rt/lowstate: " + last)


def _wait_for_hands(observer, timeout_s: float = 5.0):
    deadline = time.monotonic() + timeout_s
    last = "no sample"
    while time.monotonic() < deadline:
        try:
            return observer.observe()
        except RuntimeError as error:
            last = str(error)
        time.sleep(0.02)
    raise RuntimeError("timed out waiting for Dex3 state: " + last)


def _wait_for_activation(observer, states, pose_set, recording, timeout_s: float = 5.0):
    deadline = time.monotonic() + timeout_s
    last = "no stationary window"
    while time.monotonic() < deadline:
        try:
            observer.observe()
            return build_activation_handoff(
                pose_set,
                states.snapshot(),
                now_monotonic_s=time.monotonic(),
                config=recording,
            )
        except (RuntimeError, ValueError) as error:
            last = str(error)
        time.sleep(0.01)
    raise RuntimeError("timed out waiting for stationary supported handoff: " + last)


def _collect_frames(
    rclpy,
    node,
    camera: ROSCameraSubscriber,
    *,
    count: int,
    timeout_s: float,
    control_check=None,
) -> tuple[ROSImageFrame, ...]:
    baseline = {
        (item.timing.receipt_monotonic_s, item.timing.header_stamp_ns)
        for item in camera.frames.snapshot()
    }
    selected: list[ROSImageFrame] = []
    deadline = time.monotonic() + timeout_s
    while len(selected) < count and time.monotonic() < deadline:
        if control_check is not None:
            control_check()
        rclpy.spin_once(node, timeout_sec=0.01)
        for frame in camera.frames.snapshot():
            key = (frame.timing.receipt_monotonic_s, frame.timing.header_stamp_ns)
            if key in baseline:
                continue
            baseline.add(key)
            selected.append(frame)
            if len(selected) == count:
                break
    if len(selected) != count:
        detail = f"; last camera error: {camera.last_error}" if camera.last_error else ""
        raise RuntimeError(f"received only {len(selected)}/{count} new camera frames{detail}")
    if selected[-1].timing.receipt_monotonic_s - selected[0].timing.receipt_monotonic_s > 2.0:
        raise RuntimeError("camera observation burst exceeded 2.0 seconds")
    return tuple(selected)


def _wait_for_space_with_preview(rclpy, node, camera, *, no_window: bool) -> None:
    if not sys.stdin.isatty():
        raise RuntimeError("operator approval requires an interactive terminal")
    print(
        "START — G1 seated; both arms supported and still; AprilCube resting "
        "upright and visible; complete right-arm sweep clear. Press SPACE once: ",
        end="",
        flush=True,
    )
    descriptor = sys.stdin.fileno()
    previous = termios.tcgetattr(descriptor)
    try:
        tty.setraw(descriptor)
        while True:
            rclpy.spin_once(node, timeout_sec=0.01)
            if not no_window:
                frame = camera.frames.latest
                if frame is not None:
                    rendered = frame.image_bgr.copy()
                    cv2.putText(
                        rendered,
                        "READ-ONLY - SPACE starts complete task",
                        (24, 42),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.9,
                        (0, 220, 255),
                        2,
                    )
                    cv2.imshow("G1 tabletop", rendered)
                    cv2.waitKey(1)
            ready, _, _ = select.select([sys.stdin], [], [], 0.0)
            if ready:
                key = sys.stdin.read(1)
                if key == " ":
                    print("SPACE", flush=True)
                    return
                if key == "\x03":
                    raise KeyboardInterrupt
    finally:
        termios.tcsetattr(descriptor, termios.TCSADRAIN, previous)


def _wait_ready(executor, driver, *, timeout_s: float, label: str) -> None:
    deadline = time.monotonic() + timeout_s
    next_report = time.monotonic() + 1.0
    while time.monotonic() < deadline:
        driver.check()
        if executor.state is ExecutorState.READY:
            return
        if executor.state is ExecutorState.STOPPED:
            raise RuntimeError(f"controller stopped while waiting for {label}")
        if time.monotonic() >= next_report:
            print(executor.motion_diagnostic(prefix=f"motion status for {label}"), flush=True)
            next_report += 1.0
        time.sleep(0.01)
    raise RuntimeError(f"timed out waiting for {label}")


def _invoke_planner(command: str, request_path: Path, output_path: Path, driver=None) -> None:
    worker = ROOT / ".venv-planner/bin/g1-curobo-worker"
    if not worker.is_file():
        raise FileNotFoundError("planner environment missing; run ./tools/setup_planner_env.sh")
    completed = subprocess.run(
        [str(worker), command, "--request", str(request_path), "--output", str(output_path)],
        check=False,
    )
    if driver is not None:
        driver.check()
    if completed.returncode != 0:
        raise RuntimeError(f"CuRobo worker {command} exited with {completed.returncode}")


def _finger_heartbeat(watchdog_value, controller):
    def maintain() -> None:
        watchdog_value.pulse()
        controller.maintain_active_posture()

    return maintain


def _command_fingers(
    controller,
    driver,
    watchdog_value,
    *,
    left,
    right,
    label: str,
) -> None:
    driver.safety_heartbeat = watchdog_value.pulse

    def check() -> None:
        driver.check()
        watchdog_value.pulse()

    controller.command_posture(
        left_target_q_rad=left,
        right_target_q_rad=right,
        label=label,
        safety_heartbeat=check,
    )
    driver.safety_heartbeat = _finger_heartbeat(watchdog_value, controller)


def _save_frames(directory: Path, frames: tuple[ROSImageFrame, ...]) -> None:
    directory.mkdir(parents=True, exist_ok=False)
    manifest = []
    for index, frame in enumerate(frames):
        path = directory / f"frame_{index:03d}.png"
        if not cv2.imwrite(str(path), frame.image_bgr):
            raise RuntimeError(f"failed to write {path}")
        manifest.append(
            {
                "path": path.name,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "timing": {
                    "receipt_monotonic_s": frame.timing.receipt_monotonic_s,
                    "receipt_utc": frame.timing.receipt_utc,
                    "header_stamp_ns": frame.timing.header_stamp_ns,
                },
            }
        )
    atomic_write_json(directory / "manifest.json", {"frames": manifest})


def run_tabletop(args) -> int:
    """Run one complete seated right-Dex3 task after a single SPACE."""

    if args.confirm != MOTION_ACK:
        raise ValueError(f"--confirm must equal exactly: {MOTION_ACK}")
    hardware = load_hardware(args.hardware_config)
    quality = QualityThresholds.from_yaml(args.quality_config)
    bundle = CalibrationBundle.load(args.calibration_bundle)
    model = URDFModel(resolve_hardware_path(args.hardware_config, hardware["robot"]["urdf"]))
    expected_camera = camera_info_from_hardware(hardware)
    task_run = (args.output_root / _run_id()).resolve()
    if task_run.exists():
        raise FileExistsError(f"tabletop run already exists: {task_run}")
    task_run.mkdir(parents=True)
    hardware_bytes = args.hardware_config.read_bytes()
    bundle_bytes = args.calibration_bundle.read_bytes()
    quality_bytes = args.quality_config.read_bytes()
    detector = CorrespondenceDetector(args.cube_config)
    recording, _pairing = recording_configs(args.hardware_config)
    control_config, rate_hz = executor_config(args.hardware_config)
    control_config = replace(control_config, require_motion_endpoint_tolerance=False)
    empty_pose_set = PoseSet(
        robot_model=model.name,
        mode_machine=5,
        urdf_sha256=model.sha256,
        calibration_arm="right",
    )
    preflight_frames: tuple[ROSImageFrame, ...] = ()
    loaded_frames: tuple[ROSImageFrame, ...] = ()
    status: dict = {"status": "started", "commands_robot": False}
    primary_error: BaseException | None = None
    camera = observer = dex_observer = transport = dex_controller = None
    guard = synchronized = driver = None
    command_lock = CommandOwnerLock(args.lock_file)
    command_lock.acquire()
    try:
        try:
            import rclpy
        except ImportError as error:
            raise RuntimeError("rclpy unavailable; use ./tools/g1_tabletop_hardware.sh") from error
        rclpy.init(args=None)
        node = rclpy.create_node("g1_dex3_tabletop_task")
        try:
            camera = ROSCameraSubscriber(
                node,
                image_topic=hardware["ros"]["image_topic"],
                camera_info_topic=hardware["ros"]["camera_info_topic"],
                camera_name=hardware["camera"]["name"],
                serial_number=hardware["camera"]["serial_number"],
                reliability=hardware["ros"]["camera_reliability"],
                qos_depth=int(hardware["ros"]["camera_qos_depth"]),
                maximum_frames=30,
            )
            states = StateSampleBuffer()
            transport_cfg = transport_config(
                args.hardware_config,
                interface=args.network_interface,
                domain_id=args.domain_id,
            )
            observer = UnitreeLowStateObserver(transport_cfg, on_sample=states.add)
            _wait_for_state(observer)
            hand_cfg = dex3_config(
                args.hardware_config,
                interface=args.network_interface,
                domain_id=args.domain_id,
            )
            dex_observer = UnitreeDex3StateObserver(hand_cfg, initialize_factory=False)
            hands = _wait_for_hands(dex_observer)
            activation = _wait_for_activation(observer, states, empty_pose_set, recording)
            preflight_frames = _collect_frames(
                rclpy, node, camera, count=args.observation_frames, timeout_s=10.0
            )
            if preflight_frames[-1].camera_info.profile_sha256 != expected_camera.profile_sha256:
                raise ValueError("live camera profile differs from hardware.yaml")
            observe_resting_cube(
                [item.image_bgr for item in preflight_frames],
                camera_info=expected_camera,
                detector=detector,
                snapshot=_snapshot(activation.reference_state, hands),
                minimum_tag_short_side_px=quality.minimum_tag_short_side_px,
                maximum_reprojection_error_px=quality.pnp_reject_reprojection_px,
            )
            print(
                "READ-ONLY PREFLIGHT PASSED — seated stationary state, both Dex3 "
                "states, rectified camera profile, and resting AprilCube are valid; "
                "no command publisher exists",
                flush=True,
            )
            _wait_for_space_with_preview(rclpy, node, camera, no_window=args.no_window)
            if args.hardware_config.read_bytes() != hardware_bytes:
                raise RuntimeError("hardware configuration changed after preflight")
            if args.calibration_bundle.read_bytes() != bundle_bytes:
                raise RuntimeError("calibration bundle changed after preflight")
            if args.quality_config.read_bytes() != quality_bytes:
                raise RuntimeError("capture quality configuration changed after preflight")
            activation = _wait_for_activation(observer, states, empty_pose_set, recording)
            gravity = gravity_feedforward(
                args.hardware_config, activation.reference_state.position
            )
            guard = watchdog(
                args.hardware_config,
                host=args.pc2_host,
                ssh_identity=args.pc2_ssh_identity,
                initial_fsm_id=int(hardware["control"]["required_seated_fsm_id"]),
                restore_seated=True,
            )
            transport = UnitreeDebugLowCmdTransport(
                transport_cfg,
                debug_lowcmd_config(args.hardware_config),
                observer=observer,
                ownership_keepalive=guard.pulse,
            )
            observer = None
            dex_controller = UnitreeDex3PostureController(hand_cfg, observer=dex_observer)
            dex_observer = None
            raw = PoseExecutor(
                transport=transport,
                clock=SystemClock(),
                pose_set=empty_pose_set,
                handoff_q=activation.handoff_q,
                hold_q=activation.hold_q,
                approved_validation_report_sha256="0" * 64,
                config=control_config,
                gravity_feedforward=gravity,
            )
            synchronized = SynchronizedPoseExecutor(raw)
            driver = ExecutorControlDriver(
                synchronized,
                rate_hz=rate_hz,
                safety_heartbeat=guard.pulse,
            )
            guard.start()
            held_hands = dex_controller.acquire_measured_hold(safety_heartbeat=guard.pulse)
            driver.safety_heartbeat = _finger_heartbeat(guard, dex_controller)
            driver.start()
            synchronized.acquire(operator_confirmed=True)
            _wait_ready(
                synchronized,
                driver,
                timeout_s=control_config.acquisition_ramp_s + 5.0,
                label="loaded ownership",
            )
            print(
                "CONTROL ACQUIRED — exact measured 29-joint state held with dual-Dex3 "
                "gravity feedforward; observing the fixed cube and planning all motion",
                flush=True,
            )
            loaded_frames = _collect_frames(
                rclpy,
                node,
                camera,
                count=args.observation_frames,
                timeout_s=10.0,
                control_check=driver.check,
            )
            loaded_state = synchronized.observe_state()
            loaded_observation = observe_resting_cube(
                [item.image_bgr for item in loaded_frames],
                camera_info=expected_camera,
                detector=detector,
                snapshot=_snapshot(loaded_state, held_hands),
                minimum_tag_short_side_px=quality.minimum_tag_short_side_px,
                maximum_reprojection_error_px=quality.pnp_reject_reprojection_px,
            )
            loaded_request = build_tabletop_request(
                observation=loaded_observation,
                calibration_bundle=bundle,
                calibration_bundle_path=args.calibration_bundle,
                grasp_shortlist_path=args.grasp_shortlist,
                task_config_path=args.task_config,
            )
            loaded_request_path = task_run / "loaded_request.json"
            escape_path = task_run / "supported_escape.json"
            loaded_request.write_json(loaded_request_path)
            _invoke_planner("plan-supported-escape", loaded_request_path, escape_path, driver)
            escape = SupportedEscapePlan.from_json(escape_path)
            clearance_request = request_at_clearance(loaded_request, escape)
            clearance_request_path = task_run / "clearance_request.json"
            task_path = task_run / "task_plan.json"
            clearance_request.write_json(clearance_request_path)
            _invoke_planner("plan-tabletop-task", clearance_request_path, task_path, driver)
            task = TabletopTaskPlan.from_json(task_path)
            _clearance, execution = assemble_execution_plan(
                loaded_request=loaded_request,
                supported_escape=escape,
                task=task,
            )
            execution_path = task_run / "execution_plan.json"
            execution.write_json(execution_path)
            pose_set = pose_set_from_trajectories(
                arm="right",
                trajectories=execution.trajectories,
                reference_full_q=loaded_state.position,
                robot_model=model.name,
                urdf_sha256=model.sha256,
                source="NVlabs/curobo_complete_tabletop_lifecycle",
            )
            synchronized.install_validated_plan(
                pose_set=pose_set,
                approved_validation_report_sha256=execution.content_sha256,
                validated_reference_state=loaded_state,
            )
            print(
                f"COMPLETE PLAN FROZEN — grasp {task.selected_candidate_id}; eight "
                "connected CuRobo trajectories return exactly to the supported handoff",
                flush=True,
            )
            initial_left = held_hands.left.position
            initial_right = held_hands.right.position
            for index, trajectory in enumerate(execution.trajectories):
                synchronized.start_trajectory(
                    from_pose_id=trajectory.from_pose_id,
                    to_pose_id=trajectory.to_pose_id,
                    sample_time_s=trajectory.sample_time_s,
                    command_q_rad=trajectory.command_q_rad,
                    plan_sha256=execution.content_sha256,
                    operator_confirmed=True,
                )
                _wait_ready(
                    synchronized,
                    driver,
                    timeout_s=max(
                        control_config.motion_timeout_s, trajectory.sample_time_s[-1] + 5
                    ),
                    label=trajectory.to_pose_id,
                )
                if index == 0:
                    _command_fingers(
                        dex_controller,
                        driver,
                        guard,
                        left=initial_left,
                        right=task.open_right_dex3_q_rad,
                        label="right-hand pregrasp open",
                    )
                elif index == 2:
                    _command_fingers(
                        dex_controller,
                        driver,
                        guard,
                        left=initial_left,
                        right=task.closed_right_dex3_q_rad,
                        label="selected qualified cube grasp",
                    )
                elif index == 4:
                    _command_fingers(
                        dex_controller,
                        driver,
                        guard,
                        left=initial_left,
                        right=task.open_right_dex3_q_rad,
                        label="cube release after exact replacement",
                    )
                elif index == 6:
                    _command_fingers(
                        dex_controller,
                        driver,
                        guard,
                        left=initial_left,
                        right=initial_right,
                        label="initial finger posture restoration",
                    )
                print(f"completed {index + 1}/8: {trajectory.to_pose_id}", flush=True)
            driver.close()
            driver.check()
            dex_controller.timeout()
            guard.restore_seated()
            synchronized.confirm_external_takeover("PC2 verified AI FSM 0 -> 1 -> seated FSM 3")
            status = {
                "status": "completed",
                "commands_robot": True,
                "selected_candidate_id": task.selected_candidate_id,
                "execution_plan_sha256": execution.content_sha256,
                "terminal_action": guard.terminal_action,
                "table_collision_policy": "local_manipulation_geometry_plane",
                "calibration_validation_claim": False,
            }
            print(
                "TABLETOP TASK PASSED — cube picked, lifted 100 mm, replaced, arm "
                "returned to its supported start, and seated FSM 3 restored",
                flush=True,
            )
        finally:
            if not args.no_window:
                cv2.destroyAllWindows()
            if "node" in locals():
                if camera is not None:
                    camera.close()
                    camera = None
                node.destroy_node()
            if "rclpy" in locals() and rclpy.ok():
                rclpy.shutdown()
    except BaseException as error:
        primary_error = error
        status = {
            "status": "failed",
            "commands_robot": bool(transport is not None and transport.command_count),
            "error_type": type(error).__name__,
            "error": str(error),
        }
        raise
    finally:
        cleanup_errors: list[str] = []
        if driver is not None and driver.is_alive:
            try:
                driver.close()
            except BaseException as error:  # noqa: BLE001
                cleanup_errors.append(f"driver: {error}")
        if dex_controller is not None and not dex_controller.timed_out:
            try:
                dex_controller.timeout()
            except BaseException as error:  # noqa: BLE001
                cleanup_errors.append(f"Dex3 timeout: {error}")
        if (
            guard is not None
            and transport is not None
            and (guard.armed or transport.requires_external_takeover)
        ):
            try:
                if guard.armed:
                    guard.restore_zero_torque("tabletop task failed or was interrupted")
                synchronized.confirm_external_takeover(
                    "PC2 verified AI zero-torque takeover after tabletop failure"
                )
            except BaseException as error:  # noqa: BLE001
                cleanup_errors.append(f"PC2 takeover: {error}")
        if dex_controller is not None:
            try:
                if not dex_controller.timed_out and guard is not None and guard.terminal_action:
                    dex_controller.close_after_external_timeout()
                else:
                    dex_controller.close()
            except BaseException as error:  # noqa: BLE001
                cleanup_errors.append(f"Dex3 close: {error}")
        for name, value in (("lowstate observer", observer), ("Dex3 observer", dex_observer)):
            if value is not None:
                try:
                    value.close()
                except BaseException as error:  # noqa: BLE001
                    cleanup_errors.append(f"{name}: {error}")
        command_lock.release()
        status["cleanup_errors"] = cleanup_errors
        try:
            if preflight_frames:
                _save_frames(task_run / "preflight", preflight_frames)
            if loaded_frames:
                _save_frames(task_run / "loaded_observation", loaded_frames)
            atomic_write_json(task_run / "status.json", status)
        except BaseException as error:
            if primary_error is None:
                raise
            print(f"warning: failed to write complete failure artifacts: {error}", file=sys.stderr)
    print(json.dumps({"run": str(task_run), **status}, indent=2, sort_keys=True))
    return 0
