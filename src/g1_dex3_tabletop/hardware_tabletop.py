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
import numpy as np
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
from g1_aprilcube_calibration.joint_map import arm_indices, validate_arm_side
from g1_aprilcube_calibration.pose_schema import PoseSet
from g1_aprilcube_calibration.process_lock import CommandOwnerLock
from g1_aprilcube_calibration.readiness import StateSampleBuffer
from g1_aprilcube_calibration.ros.camera_adapter import ROSCameraSubscriber, ROSImageFrame
from g1_aprilcube_calibration.transforms import invert_transform
from g1_aprilcube_calibration.transports.unitree_arm_sdk import (
    UnitreeLowStateObserver,
    UnitreeTorsoIMUObserver,
)
from g1_aprilcube_calibration.transports.unitree_debug_lowcmd import (
    UnitreeDebugLowCmdTransport,
)
from g1_aprilcube_calibration.transports.unitree_dex3 import (
    Dex3GraspNotAcquiredError,
    Dex3RetentionLostError,
    UnitreeDex3PostureController,
    UnitreeDex3StateObserver,
)
from g1_aprilcube_calibration.urdf_model import URDFModel
from g1_dex3_tabletop.calibration_candidates import camera_info_from_hardware
from g1_dex3_tabletop.camera_state_sync import CameraStateInputBuffer
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
from g1_dex3_tabletop.mpc_command_buffer import MPCCommandWindow
from g1_dex3_tabletop.persistent_planner import (
    PersistentTabletopPlanner,
    PlannerRequestRejected,
)
from g1_dex3_tabletop.planning.contracts import (
    PlannedTrajectory,
    RobotSnapshot,
    atomic_write_json,
)
from g1_dex3_tabletop.planning.dex3_handedness import dex3_empty_close_reference
from g1_dex3_tabletop.raw_episode_recording import RawEpisodeRecorder, tabletop_raw_topics
from g1_dex3_tabletop.state_estimation import (
    AnchoredCameraPoseEstimators,
    AnchoredCameraStateEstimator,
    CameraPoseAnchor,
    pose_error,
)
from g1_dex3_tabletop.tabletop_contracts import (
    PregraspRemainingPlan,
    RetentionRouteValidationRequest,
    RetentionRouteValidationResult,
    SupportedEscapePlan,
    TabletopExecutionPlan,
    TabletopPregraspPlan,
    build_pregrasp_remaining_plan,
    pregrasp_to_clearance_return,
)
from g1_dex3_tabletop.tabletop_object import load_tabletop_object_profile
from g1_dex3_tabletop.tabletop_perception import (
    camera_motion_from_fixed_cube,
    observe_resting_cube,
)
from g1_dex3_tabletop.tabletop_presentation import (
    DIRECT_PRESENTATION_ID,
    load_tabletop_presentation,
)
from g1_dex3_tabletop.tabletop_workflow import (
    build_tabletop_request,
    load_task_config,
    request_at_clearance,
    request_at_clearance_observation,
    request_at_estimated_pregrasp,
)

ROOT = Path(__file__).resolve().parents[2]
MOTION_ACK = "I CONFIRM THE G1 IS SECURED BY THE LOAD-BEARING HARNESS AND THE WORKSPACE IS CLEAR"


class TabletopTaskRejected(RuntimeError):
    """The controller is healthy, but this pick attempt should return and stop."""


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


def _wait_for_torso_imu(observer, timeout_s: float = 5.0):
    deadline = time.monotonic() + timeout_s
    last = "no sample"
    while time.monotonic() < deadline:
        try:
            return observer.observe()
        except RuntimeError as error:
            last = str(error)
        time.sleep(0.02)
    raise RuntimeError("timed out waiting for rt/secondary_imu: " + last)


def _wait_for_camera_state_input(
    buffer: CameraStateInputBuffer,
    *,
    target_monotonic_s: float | None,
    maximum_age_s: float,
    maximum_gap_s: float,
    control_check=None,
    timeout_s: float = 1.0,
):
    deadline = time.monotonic() + timeout_s
    last = "camera-state inputs are incomplete"
    while time.monotonic() < deadline:
        if control_check is not None:
            control_check()
        try:
            if target_monotonic_s is None:
                return buffer.latest(
                    now_monotonic_s=time.monotonic(),
                    maximum_age_s=maximum_age_s,
                    maximum_gap_s=maximum_gap_s,
                )
            return buffer.sample_at(
                target_monotonic_s,
                maximum_gap_s=maximum_gap_s,
            )
        except RuntimeError as error:
            last = str(error)
        time.sleep(0.005)
    raise RuntimeError("timed out pairing camera-state inputs: " + last)


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


def _wait_for_space_with_preview(
    rclpy,
    node,
    camera,
    *,
    arm: str,
    no_window: bool,
) -> None:
    if not sys.stdin.isatty():
        raise RuntimeError("operator approval requires an interactive terminal")
    print(
        "START — G1 seated; both arms supported and still; AprilCube resting "
        f"upright and visible; complete {arm}-arm sweep clear. Press SPACE once: ",
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
    """Run a one-shot worker for the separate calibration workflow."""

    worker = ROOT / ".venv-planner/bin/g1-curobo-worker"
    if not worker.is_file():
        raise FileNotFoundError("planner environment missing; run ./tools/setup_planner_env.sh")
    log_path = output_path.with_suffix(".planner.log")
    process = subprocess.Popen(
        [str(worker), command, "--request", str(request_path), "--output", str(output_path)],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    tail: list[str] = []
    try:
        with log_path.open("w", encoding="utf-8") as log:
            assert process.stdout is not None
            for line in process.stdout:
                print(line, end="", flush=True)
                log.write(line)
                log.flush()
                if line.strip():
                    tail.append(line.strip())
                    tail = tail[-3:]
        returncode = process.wait()
    except BaseException:
        process.terminate()
        try:
            process.wait(timeout=2.0)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
        raise
    if driver is not None:
        driver.check()
    if returncode != 0:
        detail = tail[-1] if tail else "no diagnostic output"
        raise RuntimeError(
            f"CuRobo worker {command} exited with {returncode}: {detail}; "
            f"full planner log: {log_path}"
        )


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


def _command_retention_test_close(
    controller,
    driver,
    watchdog_value,
    *,
    active_side: str,
    left,
    right,
    empty_close_reference_q_rad,
    minimum_opposed_shortfall_rad: float,
    label: str,
):
    driver.safety_heartbeat = watchdog_value.pulse

    def check() -> None:
        driver.check()
        watchdog_value.pulse()

    try:
        return controller.command_close_for_retention_test(
            active_side=active_side,
            left_target_q_rad=left,
            right_target_q_rad=right,
            empty_close_reference_q_rad=empty_close_reference_q_rad,
            minimum_opposed_shortfall_rad=minimum_opposed_shortfall_rad,
            label=label,
            safety_heartbeat=check,
        )
    finally:
        driver.safety_heartbeat = _finger_heartbeat(watchdog_value, controller)


def _trajectory_maps(execution: TabletopExecutionPlan):
    normal = {trajectory.to_pose_id: trajectory for trajectory in execution.trajectories}
    recovery = {
        (trajectory.from_pose_id, trajectory.to_pose_id): trajectory
        for trajectory in execution.recovery_trajectories
    }
    if len(normal) != len(execution.trajectories):
        raise ValueError("tabletop execution plan has duplicate phase names")
    return normal, recovery


def _return_to_clearance_phases(normal_routes) -> tuple[str, ...]:
    """Return the exact post-retreat phase sequence installed by the planner."""

    if "return_to_clearance" not in normal_routes:
        raise ValueError("tabletop execution plan has no return-to-clearance phase")
    if "return_to_pregrasp" in normal_routes:
        return "return_to_pregrasp", "return_to_clearance"
    return ("return_to_clearance",)


def _execute_trajectory(
    synchronized,
    driver,
    trajectory,
    *,
    plan_sha256: str,
    control_config,
) -> None:
    synchronized.start_trajectory(
        from_pose_id=trajectory.from_pose_id,
        to_pose_id=trajectory.to_pose_id,
        sample_time_s=trajectory.sample_time_s,
        command_q_rad=trajectory.command_q_rad,
        plan_sha256=plan_sha256,
        operator_confirmed=True,
    )
    _wait_ready(
        synchronized,
        driver,
        timeout_s=max(control_config.motion_timeout_s, trajectory.sample_time_s[-1] + 5),
        label=trajectory.to_pose_id,
    )
    print(f"completed phase: {trajectory.to_pose_id}", flush=True)


def _trajectory_with_endpoints(
    trajectory: PlannedTrajectory,
    *,
    from_pose_id: str,
    to_pose_id: str,
) -> PlannedTrajectory:
    """Rename a frozen route without changing any command or timing sample."""

    return PlannedTrajectory(
        from_pose_id=from_pose_id,
        to_pose_id=to_pose_id,
        sample_time_s=trajectory.sample_time_s,
        command_q_rad=trajectory.command_q_rad,
        model_q_rad=trajectory.model_q_rad,
        planning_time_s=trajectory.planning_time_s,
    )


def _execute_mpc_phase(
    synchronized,
    driver,
    planner,
    *,
    arm: str,
    trajectory,
    plan_sha256: str,
    control_config,
    measured_active_dex3_q_rad=None,
    phase_record: dict | None = None,
) -> tuple[dict, list[dict]]:
    """Execute one normal frozen route through phase-aware rolling MPC."""

    phase = trajectory.to_pose_id
    try:
        preparation_event = planner.request_payload(
            "prepare-mpc-phase",
            payload={
                "phase": phase,
                "measured_active_dex3_q_rad": (
                    None
                    if measured_active_dex3_q_rad is None
                    else list(measured_active_dex3_q_rad)
                ),
            },
            control_check=driver.check,
            timeout_s=30.0,
        )
    except (PlannerRequestRejected, RuntimeError) as error:
        driver.check()
        raise RuntimeError(f"CuRobo MPC preparation failed for {phase}: {error}") from error
    preparation = dict(preparation_event["payload"])
    if phase_record is not None:
        phase_record["preparation"] = preparation
    if preparation["reused_warm_model"]:
        preparation_detail = (
            "reconfigured the one warmed solver in "
            f"{1000.0 * preparation['reconfiguration_time_s']:.1f}ms; "
            "no solver construction or CUDA graph rebuild"
        )
    else:
        preparation_detail = (
            f"built {preparation['physical_mode']} collision model in "
            f"{preparation['preparation_time_s']:.3f}s "
            f"(construction={preparation['build_time_s']:.3f}s, "
            f"CUDA setup={preparation['setup_time_s']:.3f}s)"
        )
    print(
        f"CUROBO MPC {phase} READY — {preparation_detail}; fixed-rate robot "
        "control remained active",
        flush=True,
    )

    def request_window() -> MPCCommandWindow:
        state, active_command = synchronized.observe_control_input()
        event = planner.request_payload(
            "step-mpc-phase",
            payload={
                "phase": phase,
                "measured_command_q_rad": state.arm_q(arm).tolist(),
                "measured_dq_rad_s": state.arm_dq(arm).tolist(),
                "active_command_q_rad": active_command.tolist(),
                "state_monotonic_s": state.receipt_monotonic_s,
            },
            control_check=driver.check,
            timeout_s=max(1.0, control_config.state_freshness_timeout_s * 10.0),
        )
        return MPCCommandWindow.from_dict(event["payload"])

    windows: list[dict] = []
    first = request_window()
    try:
        accepted = synchronized.start_streaming_trajectory(
            from_pose_id=trajectory.from_pose_id,
            to_pose_id=trajectory.to_pose_id,
            window=first,
            plan_sha256=plan_sha256,
            operator_confirmed=True,
        )
    except BaseException:
        if phase_record is not None:
            phase_record["rejected_window"] = first.to_dict()
        raise
    windows.append(accepted.to_dict())
    if phase_record is not None:
        phase_record["windows"].append(accepted.to_dict())
    replan_lead_s = control_config.state_freshness_timeout_s
    while synchronized.state is ExecutorState.MOVING:
        driver.check()
        status = synchronized.streaming_trajectory_status()
        if bool(status["terminal"]):
            break
        if float(status["remaining_s"]) > replan_lead_s:
            time.sleep(0.01)
            continue
        window = request_window()
        try:
            accepted = synchronized.update_streaming_trajectory(window=window)
        except BaseException:
            if phase_record is not None:
                phase_record["rejected_window"] = window.to_dict()
            raise
        windows.append(accepted.to_dict())
        if phase_record is not None:
            phase_record["windows"].append(accepted.to_dict())
        if accepted.generation % 10 == 0:
            print(
                "CuRobo MPC progress: "
                f"phase={phase}, window={accepted.generation}, "
                f"solve={accepted.solve_time_s:.3f}s, "
                f"remaining={accepted.duration_s:.3f}s",
                flush=True,
            )
    _wait_ready(
        synchronized,
        driver,
        timeout_s=control_config.motion_timeout_s,
        label=f"{phase} MPC terminal settle",
    )
    print(
        f"completed phase: {phase} through {len(windows)} validated MPC windows",
        flush=True,
    )
    if phase_record is not None:
        phase_record["completed"] = True
    return preparation, windows


def _restore_seated_control(*, driver, dex_controller, guard, synchronized) -> None:
    """Cleanly return a healthy held controller to Unitree seated FSM 3."""

    if driver.is_alive:
        driver.close()
    driver.check()
    if not dex_controller.timed_out:
        dex_controller.timeout()
    guard.restore_seated()
    synchronized.confirm_external_takeover("PC2 verified AI FSM 0 -> 1 -> seated FSM 3")


def _teardown_ros_runtime(
    *,
    no_window: bool,
    camera,
    node,
    rclpy_module,
    transport,
    synchronized,
) -> None:
    """Tear ROS down only after direct robot ownership has ended."""

    if (
        transport is not None
        and transport.requires_external_takeover
        and (synchronized is None or synchronized.state is not ExecutorState.STOPPED)
    ):
        raise RuntimeError("refusing ROS teardown before verified external robot-control takeover")
    if not no_window:
        cv2.destroyAllWindows()
    if camera is not None:
        camera.close()
    if node is not None:
        node.destroy_node()
    if rclpy_module is not None and rclpy_module.ok():
        rclpy_module.shutdown()


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
    """Run one complete seated selected-Dex3 task after a single SPACE."""

    if args.confirm != MOTION_ACK:
        raise ValueError(f"--confirm must equal exactly: {MOTION_ACK}")
    arm = validate_arm_side(args.arm)
    empty_close_reference_q_rad, minimum_opposed_shortfall_rad = dex3_empty_close_reference(arm)
    object_profile = load_tabletop_object_profile(args.object_profile)
    presentation = load_tabletop_presentation(
        args.presentation,
        direct_shortlist_override=(
            object_profile.direct_grasp_shortlist_path
            if args.presentation == DIRECT_PRESENTATION_ID
            else None
        ),
    )
    presentation.require_arm(arm)
    presentation.require_object_profile(object_profile.profile_id)
    hardware = load_hardware(args.hardware_config)
    configured_arms = {
        str(hardware["robot"]["calibration_arm"]),
        str(hardware["control"]["calibration_arm"]),
    }
    if configured_arms != {arm}:
        raise ValueError(
            f"hardware configuration selects {sorted(configured_arms)}, not requested {arm}"
        )
    quality = QualityThresholds.from_yaml(args.quality_config)
    bundle = CalibrationBundle.load(args.calibration_bundle)
    task_config = load_task_config(args.task_config)
    patch_dimensions = tuple(task_config["table"]["open_transit_patch_dimensions_m"])
    model = URDFModel(resolve_hardware_path(args.hardware_config, hardware["robot"]["urdf"]))
    expected_camera = camera_info_from_hardware(hardware)
    task_run = (args.output_root / _run_id()).resolve()
    if task_run.exists():
        raise FileExistsError(f"tabletop run already exists: {task_run}")
    task_run.mkdir(parents=True)
    hardware_bytes = args.hardware_config.read_bytes()
    bundle_bytes = args.calibration_bundle.read_bytes()
    quality_bytes = args.quality_config.read_bytes()
    task_config_bytes = args.task_config.read_bytes()
    object_profile_bytes = object_profile.config_path.read_bytes()
    detector_config_bytes = object_profile.detector_config_path.read_bytes()
    grasp_shortlist_bytes = object_profile.direct_grasp_shortlist_path.read_bytes()
    presentation_config_bytes = (
        None if presentation.config_path is None else presentation.config_path.read_bytes()
    )
    # The D435i tabletop stream is already high-contrast.  CLAHE amplified
    # foam/print texture and corrupted marker corners in retained hardware
    # frames, while raw grayscale passed every archived tabletop burst.
    detector = CorrespondenceDetector(
        object_profile.detector_config_path,
        preprocess=False,
    )
    recording, pairing = recording_configs(args.hardware_config)
    control_config, rate_hz = executor_config(args.hardware_config)
    control_config = replace(control_config, require_motion_endpoint_tolerance=False)
    task_velocity = float(task_config["motion"]["maximum_arm_velocity_rad_s"])
    if task_velocity > control_config.maximum_joint_velocity_rad_s:
        raise ValueError(
            f"tabletop arm velocity {task_velocity:.4f}rad/s exceeds the commissioned "
            f"controller ceiling {control_config.maximum_joint_velocity_rad_s:.4f}rad/s"
        )
    empty_pose_set = PoseSet(
        robot_model=model.name,
        mode_machine=5,
        urdf_sha256=model.sha256,
        calibration_arm=arm,
    )
    preflight_frames: tuple[ROSImageFrame, ...] = ()
    loaded_frames: tuple[ROSImageFrame, ...] = ()
    clearance_frames: tuple[ROSImageFrame, ...] = ()
    status: dict = {
        "status": "started",
        "commands_robot": False,
        "arm": arm,
        "object_profile_id": object_profile.profile_id,
        "presentation_id": presentation.presentation_id,
        "motion_controller": args.motion_controller,
    }
    primary_error: BaseException | None = None
    rejection_return_completed = False
    camera = observer = torso_observer = dex_observer = transport = dex_controller = None
    node = rclpy_module = None
    raw_recorder = None
    guard = synchronized = driver = planner = None
    mpc_phases: dict[str, dict] = {}
    cube_anchor_motion = None
    camera_state_inputs = CameraStateInputBuffer()
    camera_estimator = AnchoredCameraStateEstimator(
        AnchoredCameraPoseEstimators(model=model, calibration_bundle=bundle)
    )
    camera_state_anchor = None
    camera_state_estimate = None
    command_lock = CommandOwnerLock(args.lock_file)
    command_lock.acquire()
    try:
        try:
            import rclpy
        except ImportError as error:
            raise RuntimeError("rclpy unavailable; use ./tools/g1_tabletop_hardware.sh") from error
        rclpy_module = rclpy
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

            def receive_lowstate(sample) -> None:
                states.add(sample)
                camera_state_inputs.add_lowstate(sample)

            transport_cfg = transport_config(
                args.hardware_config,
                interface=args.network_interface,
                domain_id=args.domain_id,
            )
            observer = UnitreeLowStateObserver(transport_cfg, on_sample=receive_lowstate)
            _wait_for_state(observer)
            torso_observer = UnitreeTorsoIMUObserver(
                transport_cfg,
                lowstate_observer=observer,
                on_sample=camera_state_inputs.add_torso_imu,
            )
            _wait_for_torso_imu(torso_observer)
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
                f"object={object_profile.profile_id}; "
                f"presentation={presentation.presentation_id}; no command publisher exists",
                flush=True,
            )
            if presentation.fixture is not None:
                print(
                    "TRIPOD-H50 CONTRACT — fixture base fixed to the table; cube centred "
                    "and yaw-aligned on its three pads; the exact fixture mesh will be a "
                    "CuRobo obstacle",
                    flush=True,
                )
            planner = PersistentTabletopPlanner(
                executable=ROOT / ".venv-planner/bin/g1-curobo-worker",
                log_path=task_run / "planner.log",
            )
            planner.start()
            print(
                "PLANNER READY — one isolated CUDA worker is warm and will remain alive "
                "for lifecycle planning and measured-contact validation; no robot command "
                "publisher exists",
                flush=True,
            )
            _wait_for_space_with_preview(
                rclpy,
                node,
                camera,
                arm=arm,
                no_window=args.no_window,
            )
            if args.hardware_config.read_bytes() != hardware_bytes:
                raise RuntimeError("hardware configuration changed after preflight")
            if args.calibration_bundle.read_bytes() != bundle_bytes:
                raise RuntimeError("calibration bundle changed after preflight")
            if args.quality_config.read_bytes() != quality_bytes:
                raise RuntimeError("capture quality configuration changed after preflight")
            if args.task_config.read_bytes() != task_config_bytes:
                raise RuntimeError("tabletop task configuration changed after preflight")
            if object_profile.config_path.read_bytes() != object_profile_bytes:
                raise RuntimeError("tabletop object profile changed after preflight")
            if object_profile.detector_config_path.read_bytes() != detector_config_bytes:
                raise RuntimeError("object detector config changed after preflight")
            if object_profile.direct_grasp_shortlist_path.read_bytes() != grasp_shortlist_bytes:
                raise RuntimeError("object grasp shortlist changed after preflight")
            if (
                presentation.config_path is not None
                and presentation.config_path.read_bytes() != presentation_config_bytes
            ):
                raise RuntimeError("tabletop presentation configuration changed after preflight")
            raw_recorder = RawEpisodeRecorder(
                task_run / "raw_episode",
                repository=ROOT,
                topics=tabletop_raw_topics(record_camera=not args.skip_camera_recording),
            )
            raw_recorder.start()
            recording_content = (
                "state/command plus raw RGB and CameraInfo"
                if not args.skip_camera_recording
                else "state/command only; camera topics intentionally excluded"
            )
            print(
                "RAW EPISODE RECORDING — plain MCAP is active before command publisher "
                f"creation ({recording_content}); compression and conversion remain offline",
                flush=True,
            )
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
                "gravity feedforward; observing the fixed cube and planning the reversible "
                "supported escape",
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
                arm=arm,
                observation=loaded_observation,
                calibration_bundle=bundle,
                calibration_bundle_path=args.calibration_bundle,
                grasp_shortlist_path=presentation.grasp_shortlist_path,
                task_config_path=args.task_config,
                object_dimensions_m=object_profile.dimensions_m,
                presentation_id=presentation.presentation_id,
                fixture=presentation.fixture,
            )
            loaded_request_path = task_run / "loaded_request.json"
            escape_path = task_run / "supported_escape.json"
            loaded_request.write_json(loaded_request_path)
            try:
                planner.request(
                    "plan-supported-escape",
                    request_path=loaded_request_path,
                    output_path=escape_path,
                    control_check=driver.check,
                )
            except (PlannerRequestRejected, RuntimeError) as error:
                driver.check()
                raise TabletopTaskRejected(f"supported escape planning failed: {error}") from error
            escape = SupportedEscapePlan.from_json(escape_path)
            clearance_request = request_at_clearance(loaded_request, escape)
            clearance_request.write_json(task_run / "initial_clearance_request.json")
            pose_set = pose_set_from_trajectories(
                arm=arm,
                trajectories=(escape.outbound, escape.inbound),
                reference_full_q=loaded_state.position,
                robot_model=model.name,
                urdf_sha256=model.sha256,
                source="NVlabs/curobo_reversible_supported_escape",
            )
            synchronized.install_validated_plan(
                pose_set=pose_set,
                approved_validation_report_sha256=escape.content_sha256,
                validated_reference_state=loaded_state,
            )
            print(
                "REVERSIBLE SUPPORTED ESCAPE FROZEN — the table-normal lift and its "
                "exact reverse return to handoff are validated. The only executable "
                "grasp task will be planned from a fresh fixed-cube observation at clearance",
                flush=True,
            )
            initial_left = held_hands.left.position
            initial_right = held_hands.right.position
            grasp_close = None
            retention_evidence = None
            retention_route = None
            _execute_trajectory(
                synchronized,
                driver,
                escape.outbound,
                plan_sha256=escape.content_sha256,
                control_config=control_config,
            )
            try:
                clearance_frames = _collect_frames(
                    rclpy,
                    node,
                    camera,
                    count=args.observation_frames,
                    timeout_s=10.0,
                    control_check=driver.check,
                )
                boundary_state = synchronized.observe_state()
                boundary_hands = dex_controller.observer.observe()
                boundary_observation = observe_resting_cube(
                    [item.image_bgr for item in clearance_frames],
                    camera_info=expected_camera,
                    detector=detector,
                    snapshot=_snapshot(boundary_state, boundary_hands),
                    minimum_tag_short_side_px=quality.minimum_tag_short_side_px,
                    maximum_reprojection_error_px=quality.pnp_reject_reprojection_px,
                )
                cube_anchor_motion = camera_motion_from_fixed_cube(
                    loaded_observation.camera_T_object,
                    boundary_observation.camera_T_object,
                )
                clearance_request = request_at_clearance_observation(
                    loaded_request,
                    escape,
                    boundary_observation,
                )
                clearance_request_path = task_run / "clearance_request.json"
                clearance_request.write_json(clearance_request_path)
                # Preserve the state synchronized to the visual anchor before
                # planning. At the measured ~1 kHz input rate, the bounded
                # buffer can otherwise evict this sample during a GPU solve.
                anchor_time_s = float(
                    np.median([frame.timing.receipt_monotonic_s for frame in clearance_frames])
                )
                camera_state_anchor = _wait_for_camera_state_input(
                    camera_state_inputs,
                    target_monotonic_s=anchor_time_s,
                    maximum_age_s=control_config.state_freshness_timeout_s,
                    maximum_gap_s=pairing.maximum_bracket_span_s,
                    control_check=driver.check,
                )
                anchor_estimate = camera_estimator.reset(
                    CameraPoseAnchor(
                        reference_T_camera=invert_transform(
                            np.asarray(boundary_observation.camera_T_object)
                        ),
                        sample=camera_state_anchor.sample,
                    )
                )
                atomic_write_json(
                    task_run / "camera_state_anchor.json",
                    {
                        "estimate": anchor_estimate.to_dict(),
                        "input": camera_state_anchor.to_dict(),
                        "visual_observation_sha256": (
                            clearance_request.observation.content_sha256
                        ),
                    },
                )
                escape_return = _trajectory_with_endpoints(
                    escape.inbound,
                    from_pose_id="return_to_clearance",
                    to_pose_id="__handoff__",
                )
                if args.motion_controller == "trajectory":
                    pregrasp_path = task_run / "pregrasp_plan.json"
                    planner.request(
                        "plan-tabletop-pregrasp-at-clearance",
                        request_path=clearance_request_path,
                        output_path=pregrasp_path,
                        control_check=driver.check,
                    )
                    pregrasp_plan = TabletopPregraspPlan.from_json(pregrasp_path)
                    stage_trajectories = (
                        pregrasp_plan.outbound,
                        pregrasp_plan.inbound,
                        escape_return,
                    )
                    replanned_pose_set = pose_set_from_trajectories(
                        arm=arm,
                        trajectories=stage_trajectories,
                        reference_full_q=boundary_state.position,
                        robot_model=model.name,
                        urdf_sha256=model.sha256,
                        source="NVlabs/curobo_reversible_pregrasp_boundary",
                        initial_pose_id="clearance",
                        initial_command_q_rad=escape.outbound.command_q_rad[-1],
                    )
                    stage_plan_sha256 = pregrasp_plan.content_sha256
                else:
                    replanned_execution_path = task_run / "execution_plan.json"
                    planner.request(
                        "replan-tabletop-at-clearance",
                        request_path=clearance_request_path,
                        output_path=replanned_execution_path,
                        control_check=driver.check,
                    )
                    replanned_execution = TabletopExecutionPlan.from_json(replanned_execution_path)
                    replanned_pose_set = pose_set_from_trajectories(
                        arm=arm,
                        trajectories=replanned_execution.trajectories,
                        reference_full_q=boundary_state.position,
                        robot_model=model.name,
                        urdf_sha256=model.sha256,
                        source="NVlabs/curobo_fixed_cube_clearance_replan",
                        initial_pose_id="clearance",
                        initial_command_q_rad=escape.outbound.command_q_rad[-1],
                    )
                    stage_plan_sha256 = replanned_execution.content_sha256
                synchronized.replace_validated_remaining_plan(
                    pose_set=replanned_pose_set,
                    approved_validation_report_sha256=stage_plan_sha256,
                    validated_reference_state=boundary_state,
                )
            except (PlannerRequestRejected, RuntimeError, ValueError) as error:
                driver.check()
                try:
                    _execute_trajectory(
                        synchronized,
                        driver,
                        escape.inbound,
                        plan_sha256=escape.content_sha256,
                        control_config=control_config,
                    )
                except (RuntimeError, ValueError) as recovery_error:
                    raise RuntimeError(
                        f"clearance-boundary preparation failed: {error}; "
                        f"frozen reverse also failed: {recovery_error}"
                    ) from recovery_error
                rejection_return_completed = True
                raise TabletopTaskRejected(
                    f"fixed-cube clearance-boundary replan failed: {error}"
                ) from error

            if args.motion_controller == "trajectory":
                execution = pregrasp_plan
                task = None
                normal_routes = {
                    pregrasp_plan.outbound.to_pose_id: pregrasp_plan.outbound,
                    escape_return.to_pose_id: escape_return,
                }
                recovery_routes = {
                    (
                        pregrasp_plan.inbound.from_pose_id,
                        pregrasp_plan.inbound.to_pose_id,
                    ): pregrasp_plan.inbound
                }
                active_plan_sha256 = pregrasp_plan.content_sha256
                selected_candidate_id = pregrasp_plan.selected_candidate_id
                active_open = pregrasp_plan.open_active_dex3_q_rad
                print(
                    "CLEARANCE-TO-PREGRASP PLAN INSTALLED — fixed-cube anchor measured "
                    f"{cube_anchor_motion['translation_norm_mm']:.2f} mm / "
                    f"{cube_anchor_motion['rotation_deg']:.2f} deg camera motion; selected "
                    f"grasp {selected_candidate_id}; its unexecuted linear approach passed "
                    "strict validation, but only the reversible pregrasp route was installed; "
                    "the payload lifecycle will be planned once after the pregrasp state "
                    f"correction; planning={pregrasp_plan.planner_provenance['elapsed_s']:.2f}s",
                    flush=True,
                )
            else:
                execution = replanned_execution
                task = replanned_execution.task
                normal_routes, recovery_routes = _trajectory_maps(execution)
                active_plan_sha256 = execution.content_sha256
                task.write_json(task_run / "task_plan.json")
                selected_candidate_id = task.selected_candidate_id
                active_open = task.open_active_dex3_q_rad
                print(
                    "CLEARANCE-BOUNDARY REPLAN INSTALLED — fixed-cube anchor measured "
                    f"{cube_anchor_motion['translation_norm_mm']:.2f} mm / "
                    f"{cube_anchor_motion['rotation_deg']:.2f} deg camera motion; selected "
                    f"grasp {selected_candidate_id}; required hand/table execution "
                    f"margin={clearance_request.minimum_hand_plane_clearance_m * 1000.0:.1f} mm",
                    flush=True,
                )

            def execute_phase(
                name: str,
                *,
                measured_active_dex3_q_rad=None,
                use_mpc: bool = True,
            ) -> None:
                if args.motion_controller == "mpc" and use_mpc:
                    phase_record = {
                        "completed": False,
                        "preparation": None,
                        "windows": [],
                    }
                    mpc_phases[name] = phase_record
                    preparation, windows = _execute_mpc_phase(
                        synchronized,
                        driver,
                        planner,
                        arm=arm,
                        trajectory=normal_routes[name],
                        plan_sha256=active_plan_sha256,
                        control_config=control_config,
                        measured_active_dex3_q_rad=measured_active_dex3_q_rad,
                        phase_record=phase_record,
                    )
                    assert phase_record["preparation"] == preparation
                    assert phase_record["windows"] == windows
                    return
                _execute_trajectory(
                    synchronized,
                    driver,
                    normal_routes[name],
                    plan_sha256=active_plan_sha256,
                    control_config=control_config,
                )

            def execute_recovery(source: str, target: str) -> None:
                _execute_trajectory(
                    synchronized,
                    driver,
                    recovery_routes[(source, target)],
                    plan_sha256=active_plan_sha256,
                    control_config=control_config,
                )

            def execute_return_to_clearance(*, use_mpc: bool) -> None:
                for phase in _return_to_clearance_phases(normal_routes):
                    execute_phase(phase, use_mpc=use_mpc)

            def open_active_hand(label: str) -> None:
                _command_fingers(
                    dex_controller,
                    driver,
                    guard,
                    left=active_open if arm == "left" else initial_left,
                    right=active_open if arm == "right" else initial_right,
                    label=label,
                )

            def return_after_task_rejection(*, from_test_lift: bool) -> None:
                nonlocal rejection_return_completed
                if from_test_lift:
                    execute_recovery("retention_test_lift", "payload_replace")
                    open_active_hand("release after failed retention test")
                    execute_phase("grasp_retreat", use_mpc=False)
                else:
                    open_active_hand("open after rejected grasp attempt")
                    execute_recovery("grasp_approach", "grasp_retreat")
                execute_return_to_clearance(use_mpc=False)
                _command_fingers(
                    dex_controller,
                    driver,
                    guard,
                    left=initial_left,
                    right=initial_right,
                    label="initial finger posture restoration after task rejection",
                )
                execute_phase("__handoff__", use_mpc=False)
                rejection_return_completed = True

            open_active_hand(f"{arm}-hand pregrasp open")
            execute_phase("move_to_pregrasp")
            if args.motion_controller == "trajectory":
                original_pregrasp_plan = pregrasp_plan
                original_clearance_request = clearance_request
                boundary_return = pregrasp_to_clearance_return(original_pregrasp_plan)
                try:
                    pregrasp_state = synchronized.observe_state()
                    pregrasp_hands = dex_controller.observer.observe()
                    camera_state_current = _wait_for_camera_state_input(
                        camera_state_inputs,
                        target_monotonic_s=None,
                        maximum_age_s=control_config.state_freshness_timeout_s,
                        maximum_gap_s=pairing.maximum_bracket_span_s,
                        control_check=driver.check,
                    )
                    camera_state_estimate = camera_estimator.estimate(camera_state_current.sample)
                    q29 = np.asarray(pregrasp_state.position, dtype=np.float64).copy()
                    q29[np.asarray(arm_indices(arm))] = np.asarray(
                        original_pregrasp_plan.outbound.command_q_rad[-1]
                    )
                    estimated_request = request_at_estimated_pregrasp(
                        original_clearance_request,
                        snapshot=RobotSnapshot(
                            measured_q29_rad=tuple(q29),
                            left_dex3_q_rad=tuple(pregrasp_hands.left.position),
                            right_dex3_q_rad=tuple(pregrasp_hands.right.position),
                        ),
                        estimate=camera_state_estimate,
                        anchor_input=camera_state_anchor,
                        current_input=camera_state_current,
                    )
                    estimated_request_path = task_run / "pregrasp_estimated_request.json"
                    remaining_path = task_run / "pregrasp_remaining_plan.json"
                    estimated_request.write_json(estimated_request_path)
                    planner.request(
                        "replan-tabletop-at-pregrasp",
                        request_path=estimated_request_path,
                        output_path=remaining_path,
                        control_check=driver.check,
                    )
                    remaining = PregraspRemainingPlan.from_json(remaining_path)
                    if (
                        remaining.prior_pregrasp_plan_sha256
                        != original_pregrasp_plan.content_sha256
                    ):
                        raise RuntimeError(
                            "pregrasp correction belongs to a different prior pregrasp plan"
                        )
                    if remaining.selected_candidate_id != selected_candidate_id:
                        raise RuntimeError("pregrasp correction changed the selected grasp")
                    expected_remaining = build_pregrasp_remaining_plan(
                        prior_pregrasp_plan=original_pregrasp_plan,
                        estimated_request=estimated_request,
                        task=remaining.task,
                    )
                    if remaining.content_sha256 != expected_remaining.content_sha256:
                        raise RuntimeError(
                            "pregrasp correction differs from the deterministic remapping "
                            "of its hash-bound task and prior execution"
                        )
                    installation_input = _wait_for_camera_state_input(
                        camera_state_inputs,
                        target_monotonic_s=None,
                        maximum_age_s=control_config.state_freshness_timeout_s,
                        maximum_gap_s=pairing.maximum_bracket_span_s,
                        control_check=driver.check,
                    )
                    installation_estimate = camera_estimator.estimate(installation_input.sample)
                    installation_motion = pose_error(
                        camera_state_estimate.reference_T_camera,
                        installation_estimate.reference_T_camera,
                    )
                    if installation_motion["translation_norm_mm"] > float(
                        task_config["perception"]["maximum_translation_spread_mm"]
                    ) or installation_motion["rotation_deg"] > float(
                        task_config["perception"]["maximum_rotation_spread_deg"]
                    ):
                        raise RuntimeError(
                            "camera state changed while CuRobo replanned: "
                            f"{installation_motion['translation_norm_mm']:.3f}mm / "
                            f"{installation_motion['rotation_deg']:.3f}deg"
                        )
                    atomic_write_json(
                        task_run / "pregrasp_estimator_installation_check.json",
                        {
                            "planned_estimate": camera_state_estimate.to_dict(),
                            "installation_estimate": installation_estimate.to_dict(),
                            "installation_input": installation_input.to_dict(),
                            "change": installation_motion,
                            "limits": {
                                "translation_mm": task_config["perception"][
                                    "maximum_translation_spread_mm"
                                ],
                                "rotation_deg": task_config["perception"][
                                    "maximum_rotation_spread_deg"
                                ],
                            },
                        },
                    )
                    corrected_pose_set = pose_set_from_trajectories(
                        arm=arm,
                        trajectories=remaining.trajectories,
                        reference_full_q=pregrasp_state.position,
                        robot_model=model.name,
                        urdf_sha256=model.sha256,
                        source="NVlabs/curobo_proprioceptive_pregrasp_replan",
                        initial_pose_id="move_to_pregrasp",
                        initial_command_q_rad=(original_pregrasp_plan.outbound.command_q_rad[-1]),
                    )
                    synchronized.replace_validated_remaining_plan(
                        pose_set=corrected_pose_set,
                        approved_validation_report_sha256=remaining.content_sha256,
                        validated_reference_state=pregrasp_state,
                    )
                except (PlannerRequestRejected, RuntimeError, ValueError) as error:
                    driver.check()
                    _execute_trajectory(
                        synchronized,
                        driver,
                        boundary_return,
                        plan_sha256=original_pregrasp_plan.content_sha256,
                        control_config=control_config,
                    )
                    _command_fingers(
                        dex_controller,
                        driver,
                        guard,
                        left=initial_left,
                        right=initial_right,
                        label="initial finger posture restoration after pregrasp replan rejection",
                    )
                    _execute_trajectory(
                        synchronized,
                        driver,
                        escape_return,
                        plan_sha256=original_pregrasp_plan.content_sha256,
                        control_config=control_config,
                    )
                    rejection_return_completed = True
                    raise TabletopTaskRejected(
                        f"pregrasp camera-state correction failed: {error}"
                    ) from error

                clearance_request = estimated_request
                task = remaining.task
                execution = remaining
                active_plan_sha256 = remaining.content_sha256
                normal_routes = {
                    trajectory.to_pose_id: trajectory
                    for trajectory in (
                        *remaining.trajectories,
                        escape_return,
                    )
                }
                recovery_routes = {
                    (trajectory.from_pose_id, trajectory.to_pose_id): trajectory
                    for trajectory in remaining.recovery_trajectories
                }
                task.write_json(task_run / "pregrasp_corrected_task_plan.json")
                print(
                    "PREGRASP STATE CORRECTION INSTALLED — waist/pelvis/torso state "
                    f"propagated the fixed-cube camera anchor for "
                    f"{camera_state_estimate.anchor_age_s:.3f}s; grasp "
                    f"{task.selected_candidate_id} was preserved and every remaining "
                    "motion was planned from the exact active command; "
                    f"planning={task.planner_provenance['elapsed_s']:.2f}s, "
                    "open_optimizer_reused="
                    f"{task.planner_provenance['open_optimizer_reused']}",
                    flush=True,
                )
                execute_phase("estimated_pregrasp", use_mpc=False)
            else:
                print(
                    "PREGRASP STATE CORRECTION NOT APPLIED — the optional MPC execution "
                    "path retains its own frozen lifecycle; use the default trajectory "
                    "controller for the commissioned stationary-boundary correction",
                    flush=True,
                )
            if task is None:
                raise RuntimeError("tabletop remainder planning ended without a task plan")
            retention_test_lift_mm = 1000.0 * float(
                task.planner_provenance["retention_test_lift_actual_m"]
            )
            payload_lift_mm = 1000.0 * float(clearance_request.lift_m)
            print(
                "EMPTY-CLOSE REFERENCE READY — grasp evidence will require stable "
                "thumb and opposing-finger shortfall of at least "
                f"{minimum_opposed_shortfall_rad:.4f}rad on both sides; pressure and "
                "tau_est remain recorded diagnostics; beginning grasp approach",
                flush=True,
            )
            execute_phase("grasp_approach")
            active_close_target = task.close_target_active_dex3_q_rad
            try:
                grasp_close = _command_retention_test_close(
                    dex_controller,
                    driver,
                    guard,
                    active_side=arm,
                    left=active_close_target if arm == "left" else initial_left,
                    right=active_close_target if arm == "right" else initial_right,
                    empty_close_reference_q_rad=empty_close_reference_q_rad,
                    minimum_opposed_shortfall_rad=minimum_opposed_shortfall_rad,
                    label=f"descriptor-defined {arm}-hand cube close",
                )
            except Dex3GraspNotAcquiredError as error:
                driver.check()
                print(f"TASK REJECTED — no stable grasp close: {error}", flush=True)
                return_after_task_rejection(from_test_lift=False)
                raise TabletopTaskRejected(f"no stable grasp close: {error}") from error
            atomic_write_json(
                task_run / "grasp_close.json",
                grasp_close.to_dict(),
            )
            dex_controller.begin_retention_test()
            retention_request = RetentionRouteValidationRequest(
                tabletop_request=clearance_request,
                task_plan=task,
                measured_active_dex3_q_rad=grasp_close.close_q_rad,
                blocked_motor_ids=grasp_close.blocked_motor_ids,
            )
            retention_request_path = task_run / "retention_route_request.json"
            retention_route_path = task_run / "retention_route_validation.json"
            retention_request.write_json(retention_request_path)
            try:
                planner.request(
                    "validate-retention-route",
                    request_path=retention_request_path,
                    output_path=retention_route_path,
                    control_check=driver.check,
                )
            except (PlannerRequestRejected, RuntimeError) as error:
                # If this was actually a controller fault, preserve the fail-closed
                # path. Otherwise the frozen open-hand reverse route remains valid.
                driver.check()
                print(
                    f"TASK REJECTED — measured close route is unavailable: {error}",
                    flush=True,
                )
                return_after_task_rejection(from_test_lift=False)
                raise TabletopTaskRejected(
                    f"measured close route is unavailable: {error}"
                ) from error
            retention_route = RetentionRouteValidationResult.from_json(retention_route_path)
            if retention_route.request_sha256 != retention_request.content_sha256:
                raise RuntimeError("retention-route validation belongs to another request")
            try:
                dex_controller.check_retention_test()
            except Dex3RetentionLostError as error:
                print(f"TASK REJECTED — cube contact was lost before lift: {error}")
                return_after_task_rejection(from_test_lift=False)
                raise TabletopTaskRejected(
                    f"cube contact was lost before lift: {error}"
                ) from error
            print(
                "GRASP CLOSE STABILIZED — thumb and opposing-finger obstruction relative "
                "to commissioned empty close passed the frozen payload-route collision "
                "recheck; pressure does not decide retention; beginning the "
                f"{retention_test_lift_mm:.1f} mm retention checkpoint within the payload lift",
                flush=True,
            )
            execute_phase(
                "retention_test_lift",
                measured_active_dex3_q_rad=grasp_close.close_q_rad,
            )
            try:
                retention_evidence = dex_controller.verify_retention_at_lifted_checkpoint(
                    safety_heartbeat=lambda: (driver.check(), guard.pulse()),
                )
                dex_controller.finish_retention_test()
            except Dex3RetentionLostError as error:
                driver.check()
                print(
                    "TASK REJECTED — cube contact did not survive the lifted checkpoint; "
                    "holding the close target during the exact low-lift reverse and opening "
                    f"only after returning to support: {error}"
                )
                return_after_task_rejection(from_test_lift=True)
                raise TabletopTaskRejected(
                    f"cube contact did not survive lift checkpoint: {error}"
                ) from error
            atomic_write_json(
                task_run / "retention_evidence.json",
                retention_evidence.to_dict(),
            )
            print(
                "RETENTION TEST PASSED — stable thumb and opposing-finger obstruction "
                "relative to commissioned empty close remained after separation from "
                "the support; continuing the payload lift",
                flush=True,
            )
            execute_phase(
                "payload_lift",
                measured_active_dex3_q_rad=grasp_close.close_q_rad,
            )
            execute_phase(
                "payload_lower",
                measured_active_dex3_q_rad=grasp_close.close_q_rad,
            )
            execute_phase(
                "payload_replace",
                measured_active_dex3_q_rad=grasp_close.close_q_rad,
            )
            open_active_hand("cube release after exact replacement")
            execute_phase("grasp_retreat")
            execute_return_to_clearance(use_mpc=True)
            _command_fingers(
                dex_controller,
                driver,
                guard,
                left=initial_left,
                right=initial_right,
                label="initial finger posture restoration",
            )
            execute_phase("__handoff__")
            if grasp_close is None or retention_evidence is None or retention_route is None:
                raise RuntimeError("tabletop lifecycle ended without retention evidence")
            _restore_seated_control(
                driver=driver,
                dex_controller=dex_controller,
                guard=guard,
                synchronized=synchronized,
            )
            status = {
                "status": "completed",
                "commands_robot": True,
                "arm": arm,
                "presentation_id": presentation.presentation_id,
                "selected_candidate_id": task.selected_candidate_id,
                "supported_escape_plan_sha256": escape.content_sha256,
                "active_plan_sha256": execution.content_sha256,
                "active_plan_kind": execution.kind,
                "pregrasp_plan_sha256": (
                    pregrasp_plan.content_sha256
                    if args.motion_controller == "trajectory"
                    else None
                ),
                "cube_anchor_motion": cube_anchor_motion,
                "pregrasp_camera_state_estimate": (
                    None if camera_state_estimate is None else camera_state_estimate.to_dict()
                ),
                "motion_controller": args.motion_controller,
                "mpc_phase_count": len(mpc_phases),
                "mpc_window_count": sum(
                    len(document["windows"]) for document in mpc_phases.values()
                ),
                "grasp_close": grasp_close.to_dict(),
                "retention_evidence": retention_evidence.to_dict(),
                "retention_route_validation_sha256": retention_route.content_sha256,
                "terminal_action": guard.terminal_action,
                "table_collision_policy": "local_manipulation_geometry_plane",
                "open_transit_table_patch_dimensions_m": list(patch_dimensions),
                "minimum_hand_plane_clearance_m": (
                    clearance_request.minimum_hand_plane_clearance_m
                ),
                "calibration_validation_claim": False,
            }
            print(
                "TABLETOP TASK PASSED — lifted opposed joint obstruction survived a "
                f"{retention_test_lift_mm:.1f} mm retention checkpoint; cube then completed the "
                f"{payload_lift_mm:.1f} mm lift, was replaced, and the arm "
                "returned to its supported start, and seated FSM 3 restored",
                flush=True,
            )
        finally:
            # Robot ownership is resolved by the outer lifecycle handlers. ROS
            # teardown must remain later because participant destruction can
            # block Python callbacks long enough to stale the control state.
            pass
    except TabletopTaskRejected as rejection:
        try:
            _restore_seated_control(
                driver=driver,
                dex_controller=dex_controller,
                guard=guard,
                synchronized=synchronized,
            )
        except BaseException as error:
            primary_error = error
            status = {
                "status": "failed",
                "commands_robot": bool(transport is not None and transport.command_count),
                "arm": arm,
                "presentation_id": presentation.presentation_id,
                "motion_controller": args.motion_controller,
                "error_type": type(error).__name__,
                "error": f"task rejection recovery failed after {rejection}: {error}",
            }
            raise
        status = {
            "status": "task_rejected",
            "commands_robot": bool(transport is not None and transport.command_count),
            "arm": arm,
            "presentation_id": presentation.presentation_id,
            "motion_controller": args.motion_controller,
            "reason": str(rejection),
            "terminal_action": guard.terminal_action,
            "frozen_reverse_return_completed": rejection_return_completed,
            "calibration_validation_claim": False,
        }
        return_description = (
            "the arm returned through the frozen reverse route"
            if rejection_return_completed
            else "the arm remained at the supported ownership pose"
        )
        print(
            f"TABLETOP TASK REJECTED — {return_description} and seated FSM 3 was "
            "restored; no emergency zero-torque transition was requested. Reason: "
            f"{rejection}",
            flush=True,
        )
    except BaseException as error:
        primary_error = error
        status = {
            "status": "failed",
            "commands_robot": bool(transport is not None and transport.command_count),
            "arm": arm,
            "presentation_id": presentation.presentation_id,
            "motion_controller": args.motion_controller,
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
        if planner is not None:
            try:
                planner.close()
            except BaseException as error:  # noqa: BLE001
                cleanup_errors.append(f"planner: {error}")
        if dex_controller is not None:
            try:
                if not dex_controller.timed_out and guard is not None and guard.terminal_action:
                    dex_controller.close_after_external_timeout()
                else:
                    dex_controller.close()
            except BaseException as error:  # noqa: BLE001
                cleanup_errors.append(f"Dex3 close: {error}")
        for name, value in (
            ("torso IMU observer", torso_observer),
            ("lowstate observer", observer),
            ("Dex3 observer", dex_observer),
        ):
            if value is not None:
                try:
                    value.close()
                except BaseException as error:  # noqa: BLE001
                    cleanup_errors.append(f"{name}: {error}")
        try:
            _teardown_ros_runtime(
                no_window=args.no_window,
                camera=camera,
                node=node,
                rclpy_module=rclpy_module,
                transport=transport,
                synchronized=synchronized,
            )
            camera = None
            node = None
        except BaseException as error:  # noqa: BLE001
            cleanup_errors.append(f"ROS teardown: {error}")
        command_lock.release()
        if raw_recorder is not None and (raw_recorder.started or raw_recorder.summary is not None):
            try:
                status["recording"] = raw_recorder.stop()
            except BaseException as error:  # noqa: BLE001
                cleanup_errors.append(f"raw episode recorder: {error}")
                status["recording"] = {
                    "state": "finalization_failed",
                    "complete": False,
                    "episode_directory": str(raw_recorder.episode_directory),
                    "error": str(error),
                }
        status["cleanup_errors"] = cleanup_errors
        status.setdefault(
            "minimum_hand_plane_clearance_m",
            float(task_config["table"]["minimum_hand_plane_clearance_m"]),
        )
        if cube_anchor_motion is not None:
            status.setdefault("cube_anchor_motion", cube_anchor_motion)
        if camera_state_estimate is not None:
            status.setdefault(
                "pregrasp_camera_state_estimate",
                camera_state_estimate.to_dict(),
            )
        try:
            if preflight_frames:
                _save_frames(task_run / "preflight", preflight_frames)
            if loaded_frames:
                _save_frames(task_run / "loaded_observation", loaded_frames)
            if clearance_frames:
                _save_frames(task_run / "clearance_observation", clearance_frames)
            if cube_anchor_motion is not None:
                atomic_write_json(task_run / "cube_anchor_motion.json", cube_anchor_motion)
            if mpc_phases:
                phase_plan_sha256 = {
                    name: document["preparation"]["plan_sha256"]
                    for name, document in mpc_phases.items()
                    if document["preparation"] is not None
                }
                atomic_write_json(
                    task_run / "mpc_lifecycle.json",
                    {
                        "schema_version": 1,
                        "controller": "curobo_mpc",
                        "run_status": status["status"],
                        "plan_sha256s": sorted(set(phase_plan_sha256.values())),
                        "phase_plan_sha256": phase_plan_sha256,
                        "phase_order": list(mpc_phases),
                        "phases": mpc_phases,
                    },
                )
            atomic_write_json(task_run / "status.json", status)
        except BaseException as error:
            if primary_error is None:
                raise
            print(f"warning: failed to write complete failure artifacts: {error}", file=sys.stderr)
    print(json.dumps({"run": str(task_run), **status}, indent=2, sort_keys=True))
    return 0
