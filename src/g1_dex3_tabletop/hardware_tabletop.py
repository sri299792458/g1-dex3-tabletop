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
from g1_aprilcube_calibration.joint_map import validate_arm_side
from g1_aprilcube_calibration.pose_schema import PoseSet
from g1_aprilcube_calibration.process_lock import CommandOwnerLock
from g1_aprilcube_calibration.readiness import StateSampleBuffer
from g1_aprilcube_calibration.ros.camera_adapter import ROSCameraSubscriber, ROSImageFrame
from g1_aprilcube_calibration.transports.unitree_arm_sdk import UnitreeLowStateObserver
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
from g1_dex3_tabletop.planning.contracts import RobotSnapshot, atomic_write_json
from g1_dex3_tabletop.raw_episode_recording import RawEpisodeRecorder, tabletop_raw_topics
from g1_dex3_tabletop.tabletop_contracts import (
    RetentionRouteValidationRequest,
    RetentionRouteValidationResult,
    TabletopExecutionPlan,
)
from g1_dex3_tabletop.tabletop_perception import observe_resting_cube
from g1_dex3_tabletop.tabletop_presentation import load_tabletop_presentation
from g1_dex3_tabletop.tabletop_workflow import (
    build_tabletop_request,
    load_task_config,
    request_at_clearance,
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


def _command_grasp_fingers(
    controller,
    driver,
    watchdog_value,
    *,
    active_side: str,
    left,
    right,
    label: str,
):
    driver.safety_heartbeat = watchdog_value.pulse

    def check() -> None:
        driver.check()
        watchdog_value.pulse()

    try:
        return controller.command_grasp_until_stall(
            active_side=active_side,
            left_target_q_rad=left,
            right_target_q_rad=right,
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
    presentation = load_tabletop_presentation(
        args.presentation,
        direct_shortlist_override=args.grasp_shortlist,
    )
    presentation.require_arm(arm)
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
    presentation_config_bytes = (
        None if presentation.config_path is None else presentation.config_path.read_bytes()
    )
    detector = CorrespondenceDetector(args.cube_config)
    recording, _pairing = recording_configs(args.hardware_config)
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
    status: dict = {
        "status": "started",
        "commands_robot": False,
        "arm": arm,
        "presentation_id": presentation.presentation_id,
        "motion_controller": args.motion_controller,
    }
    primary_error: BaseException | None = None
    rejection_return_completed = False
    camera = observer = dex_observer = transport = dex_controller = None
    node = rclpy_module = None
    raw_recorder = None
    guard = synchronized = driver = planner = None
    mpc_phases: dict[str, dict] = {}
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
                arm=arm,
                observation=loaded_observation,
                calibration_bundle=bundle,
                calibration_bundle_path=args.calibration_bundle,
                grasp_shortlist_path=presentation.grasp_shortlist_path,
                task_config_path=args.task_config,
                presentation_id=presentation.presentation_id,
                fixture=presentation.fixture,
            )
            loaded_request_path = task_run / "loaded_request.json"
            execution_path = task_run / "execution_plan.json"
            loaded_request.write_json(loaded_request_path)
            try:
                planner.request(
                    "plan-tabletop-lifecycle",
                    request_path=loaded_request_path,
                    output_path=execution_path,
                    control_check=driver.check,
                )
            except (PlannerRequestRejected, RuntimeError) as error:
                driver.check()
                raise TabletopTaskRejected(
                    f"complete lifecycle planning failed: {error}"
                ) from error
            execution = TabletopExecutionPlan.from_json(execution_path)
            escape = execution.supported_escape
            task = execution.task
            # Keep the component artifacts independently inspectable even though the
            # worker planned them as one transaction.
            clearance_request = request_at_clearance(loaded_request, escape)
            clearance_request.write_json(task_run / "clearance_request.json")
            escape.write_json(task_run / "supported_escape.json")
            task.write_json(task_run / "task_plan.json")
            retention_test_lift_mm = 1000.0 * float(
                task.planner_provenance["retention_test_lift_actual_m"]
            )
            payload_lift_mm = 1000.0 * float(clearance_request.lift_m)
            pose_set = pose_set_from_trajectories(
                arm=arm,
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
                f"COMPLETE PLAN FROZEN — grasp {task.selected_candidate_id}; ten "
                "connected CuRobo trajectories return exactly to the supported handoff",
                flush=True,
            )
            initial_left = held_hands.left.position
            initial_right = held_hands.right.position
            grasp_stall = None
            retention_evidence = None
            retention_route = None
            normal_routes, recovery_routes = _trajectory_maps(execution)

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
                        plan_sha256=execution.content_sha256,
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
                    plan_sha256=execution.content_sha256,
                    control_config=control_config,
                )

            def execute_recovery(source: str, target: str) -> None:
                _execute_trajectory(
                    synchronized,
                    driver,
                    recovery_routes[(source, target)],
                    plan_sha256=execution.content_sha256,
                    control_config=control_config,
                )

            active_open = task.open_active_dex3_q_rad

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
                execute_phase("return_to_clearance", use_mpc=False)
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

            execute_phase("clearance")
            open_active_hand(f"{arm}-hand pregrasp open")
            execute_phase("move_to_pregrasp")
            execute_phase("grasp_approach")
            active_close_target = task.close_target_active_dex3_q_rad
            try:
                grasp_stall = _command_grasp_fingers(
                    dex_controller,
                    driver,
                    guard,
                    active_side=arm,
                    left=active_close_target if arm == "left" else initial_left,
                    right=active_close_target if arm == "right" else initial_right,
                    label=f"descriptor-defined {arm}-hand cube close",
                )
            except Dex3GraspNotAcquiredError as error:
                driver.check()
                print(f"TASK REJECTED — no stable cube contact: {error}", flush=True)
                return_after_task_rejection(from_test_lift=False)
                raise TabletopTaskRejected(f"no stable cube contact: {error}") from error
            atomic_write_json(task_run / "grasp_stall.json", grasp_stall.to_dict())
            dex_controller.begin_retention_test()
            retention_request = RetentionRouteValidationRequest(
                tabletop_request=clearance_request,
                task_plan=task,
                measured_active_dex3_q_rad=grasp_stall.contact_q_rad,
                blocked_motor_ids=grasp_stall.blocked_motor_ids,
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
                    f"TASK REJECTED — measured contact route is unavailable: {error}",
                    flush=True,
                )
                return_after_task_rejection(from_test_lift=False)
                raise TabletopTaskRejected(
                    f"measured contact route is unavailable: {error}"
                ) from error
            retention_route = RetentionRouteValidationResult.from_json(retention_route_path)
            if retention_route.request_sha256 != retention_request.content_sha256:
                raise RuntimeError("retention-route validation belongs to another request")
            try:
                dex_controller.check_retention_test()
            except Dex3RetentionLostError as error:
                print(f"TASK REJECTED — cube contact was lost before test lift: {error}")
                return_after_task_rejection(from_test_lift=False)
                raise TabletopTaskRejected(
                    f"cube contact was lost before test lift: {error}"
                ) from error
            print(
                "GRASP STALL ACQUIRED — measured stalled fingers passed the frozen "
                "payload-route collision recheck; beginning the "
                f"{retention_test_lift_mm:.1f} mm test lift",
                flush=True,
            )
            execute_phase(
                "retention_test_lift",
                measured_active_dex3_q_rad=grasp_stall.contact_q_rad,
            )
            try:
                retention_evidence = dex_controller.verify_grasp_stall_persistence(
                    safety_heartbeat=lambda: (driver.check(), guard.pulse()),
                )
                dex_controller.finish_retention_test()
            except Dex3RetentionLostError as error:
                driver.check()
                print(f"TASK REJECTED — cube contact did not survive test lift: {error}")
                return_after_task_rejection(from_test_lift=True)
                raise TabletopTaskRejected(
                    f"cube contact did not survive test lift: {error}"
                ) from error
            atomic_write_json(
                task_run / "retention_evidence.json",
                retention_evidence.to_dict(),
            )
            print(
                "RETENTION TEST PASSED — a fresh stable finger stall remained after "
                "the test lift; continuing the full payload lift",
                flush=True,
            )
            execute_phase(
                "payload_lift",
                measured_active_dex3_q_rad=grasp_stall.contact_q_rad,
            )
            execute_phase(
                "payload_lower",
                measured_active_dex3_q_rad=grasp_stall.contact_q_rad,
            )
            execute_phase(
                "payload_replace",
                measured_active_dex3_q_rad=grasp_stall.contact_q_rad,
            )
            open_active_hand("cube release after exact replacement")
            execute_phase("grasp_retreat")
            execute_phase("return_to_clearance")
            _command_fingers(
                dex_controller,
                driver,
                guard,
                left=initial_left,
                right=initial_right,
                label="initial finger posture restoration",
            )
            execute_phase("__handoff__")
            if grasp_stall is None or retention_evidence is None or retention_route is None:
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
                "execution_plan_sha256": execution.content_sha256,
                "motion_controller": args.motion_controller,
                "mpc_phase_count": len(mpc_phases),
                "mpc_window_count": sum(
                    len(document["windows"]) for document in mpc_phases.values()
                ),
                "grasp_stall": grasp_stall.to_dict(),
                "retention_evidence": retention_evidence.to_dict(),
                "retention_route_validation_sha256": retention_route.content_sha256,
                "terminal_action": guard.terminal_action,
                "table_collision_policy": "local_manipulation_geometry_plane",
                "open_transit_table_patch_dimensions_m": list(patch_dimensions),
                "calibration_validation_claim": False,
            }
            print(
                "TABLETOP TASK PASSED — grasp stall survived a "
                f"{retention_test_lift_mm:.1f} mm test lift; cube then completed the "
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
        for name, value in (("lowstate observer", observer), ("Dex3 observer", dex_observer)):
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
        try:
            if preflight_frames:
                _save_frames(task_run / "preflight", preflight_frames)
            if loaded_frames:
                _save_frames(task_run / "loaded_observation", loaded_frames)
            if mpc_phases:
                atomic_write_json(
                    task_run / "mpc_lifecycle.json",
                    {
                        "schema_version": 1,
                        "controller": "curobo_mpc",
                        "run_status": status["status"],
                        "plan_sha256": next(
                            (
                                document["preparation"]["plan_sha256"]
                                for document in mpc_phases.values()
                                if document["preparation"] is not None
                            ),
                            None,
                        ),
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
