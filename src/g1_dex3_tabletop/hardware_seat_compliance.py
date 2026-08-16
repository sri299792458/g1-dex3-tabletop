"""Both-arm seated chair-compliance diagnostic using a fixed ChArUco board."""

from __future__ import annotations

import hashlib
import json
import select
import sys
import termios
import tty
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

from g1_aprilcube_calibration.calibration_bundle import CalibrationBundle
from g1_aprilcube_calibration.clock import SystemClock
from g1_aprilcube_calibration.executor_driver import (
    ExecutorControlDriver,
    SynchronizedPoseExecutor,
)
from g1_aprilcube_calibration.executor_state_machine import PoseExecutor
from g1_aprilcube_calibration.joint_map import LEFT_ARM_INDICES, RIGHT_ARM_INDICES
from g1_aprilcube_calibration.pose_schema import PoseSet
from g1_aprilcube_calibration.process_lock import CommandOwnerLock
from g1_aprilcube_calibration.readiness import StateSampleBuffer
from g1_aprilcube_calibration.ros.camera_adapter import ROSCameraSubscriber, ROSImageFrame
from g1_aprilcube_calibration.table_accuracy import CharucoBoardPoseDetector
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
from g1_dex3_tabletop.hardware_calibration import (
    _execute_stage,
    _stage_executor,
    _stop_driver,
)
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
from g1_dex3_tabletop.hardware_tabletop import (
    MOTION_ACK,
    _collect_frames,
    _finger_heartbeat,
    _restore_seated_control,
    _save_frames,
    _snapshot,
    _teardown_ros_runtime,
    _wait_for_activation,
    _wait_for_hands,
    _wait_for_state,
    _wait_ready,
)
from g1_dex3_tabletop.persistent_planner import (
    PersistentTabletopPlanner,
    PlannerRequestRejected,
)
from g1_dex3_tabletop.planning.contracts import RobotSnapshot, atomic_write_json
from g1_dex3_tabletop.raw_episode_recording import RawEpisodeRecorder, tabletop_raw_topics
from g1_dex3_tabletop.seat_compliance import (
    build_charuco_escape_request,
    camera_motion_from_fixed_board,
    observe_charuco_board,
    summarize_camera_motion_cycles,
)
from g1_dex3_tabletop.tabletop_contracts import (
    CharucoBoardObservation,
    SupportedEscapePlan,
)
from g1_dex3_tabletop.tabletop_workflow import load_task_config

ROOT = Path(__file__).resolve().parents[2]
ARM_ORDER = ("left", "right")


def _run_id(condition: str) -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"seat_compliance_{condition}_{stamp}"


def _wait_for_space_with_board_preview(
    rclpy,
    node,
    camera,
    *,
    condition: str,
    repetitions: int,
    no_window: bool,
) -> None:
    if not sys.stdin.isatty():
        raise RuntimeError("operator approval requires an interactive terminal")
    print(
        "START — G1 seated; both arms supported and still on the table; fixed "
        "ChArUco board fully visible and immobile; both 100 mm arm sweeps clear. "
        f"Condition={condition}; {repetitions} left/right pairs. Press SPACE once: ",
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
                        "READ-ONLY - SPACE starts both-arm chair diagnostic",
                        (24, 42),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (0, 220, 255),
                        2,
                    )
                    cv2.imshow("G1 seat compliance", rendered)
                    cv2.waitKey(1)
            ready, _, _ = select.select([sys.stdin], [], [], 0.0)
            if not ready:
                continue
            key = sys.stdin.read(1)
            if key == " ":
                print("SPACE", flush=True)
                return
            if key == "\x03":
                raise KeyboardInterrupt
    finally:
        termios.tcsetattr(descriptor, termios.TCSADRAIN, previous)


def _frame_evidence(frames: tuple[ROSImageFrame, ...]) -> list[dict]:
    return [
        {
            "frame_index": index,
            "image_sha256": hashlib.sha256(frame.image_bgr.tobytes()).hexdigest(),
            "receipt_monotonic_s": frame.timing.receipt_monotonic_s,
            "receipt_utc": frame.timing.receipt_utc,
            "header_stamp_ns": frame.timing.header_stamp_ns,
        }
        for index, frame in enumerate(frames)
    ]


def _observe_board(
    rclpy,
    node,
    camera,
    *,
    camera_info,
    detector,
    snapshot: RobotSnapshot,
    frame_count: int,
    control_check=None,
) -> tuple[tuple[ROSImageFrame, ...], CharucoBoardObservation, dict]:
    frames = _collect_frames(
        rclpy,
        node,
        camera,
        count=frame_count,
        timeout_s=10.0,
        control_check=control_check,
    )
    observation, evidence = observe_charuco_board(
        [frame.image_bgr for frame in frames],
        camera_info=camera_info,
        snapshot=snapshot,
        detector=detector,
    )
    evidence["input_frame_timing"] = _frame_evidence(frames)
    return frames, observation, evidence


def _q14(q29: np.ndarray) -> np.ndarray:
    return np.concatenate((q29[np.asarray(LEFT_ARM_INDICES)], q29[np.asarray(RIGHT_ARM_INDICES)]))


def run_measure_seat_compliance(args) -> int:
    """Lift and exactly return each arm while recording fixed-board camera motion."""

    if args.confirm != MOTION_ACK:
        raise ValueError(f"--confirm must equal exactly: {MOTION_ACK}")
    if args.repetitions < 1:
        raise ValueError("--repetitions must be positive")
    hardware = load_hardware(args.hardware_config)
    bundle = CalibrationBundle.load(args.calibration_bundle)
    task = load_task_config(args.task_config)
    lift_m = float(task["motion"]["supported_escape_m"])
    if not np.isclose(lift_m, 0.100, atol=1.0e-12, rtol=0.0):
        raise ValueError("seat-compliance diagnostic requires the commissioned 0.100 m escape")
    model = URDFModel(resolve_hardware_path(args.hardware_config, hardware["robot"]["urdf"]))
    expected_camera = camera_info_from_hardware(hardware)
    recording, _pairing = recording_configs(args.hardware_config)
    control_config, rate_hz = executor_config(args.hardware_config)
    control_config = replace(control_config, require_motion_endpoint_tolerance=False)
    task_velocity = float(task["motion"]["maximum_arm_velocity_rad_s"])
    if task_velocity > control_config.maximum_joint_velocity_rad_s:
        raise ValueError(
            f"diagnostic velocity {task_velocity:.4f}rad/s exceeds the commissioned "
            f"controller ceiling {control_config.maximum_joint_velocity_rad_s:.4f}rad/s"
        )
    task_run = (args.output_root / _run_id(args.chair_condition)).resolve()
    if task_run.exists():
        raise FileExistsError(f"seat-compliance run already exists: {task_run}")
    task_run.mkdir(parents=True)
    hardware_bytes = args.hardware_config.read_bytes()
    bundle_bytes = args.calibration_bundle.read_bytes()
    task_bytes = args.task_config.read_bytes()
    detector = CharucoBoardPoseDetector()
    empty_pose_set = PoseSet(
        robot_model=model.name,
        mode_machine=5,
        urdf_sha256=model.sha256,
        calibration_arm="right",
    )
    status: dict = {
        "status": "started",
        "commands_robot": False,
        "chair_condition": args.chair_condition,
        "arms": list(ARM_ORDER),
        "repetitions": args.repetitions,
        "calibration_validation_claim": False,
    }
    observation_events: list[dict] = []
    preflight_frames: tuple[ROSImageFrame, ...] = ()
    loaded_frames: tuple[ROSImageFrame, ...] = ()
    primary_error: BaseException | None = None
    camera = observer = dex_observer = transport = dex_controller = None
    node = rclpy_module = None
    raw_recorder = None
    guard = synchronized = active_synchronized = driver = planner = None
    command_lock = CommandOwnerLock(args.lock_file)
    command_lock.acquire()
    try:
        try:
            import rclpy
        except ImportError as error:
            raise RuntimeError("rclpy unavailable; use ./tools/g1_tabletop_hardware.sh") from error
        rclpy_module = rclpy
        rclpy.init(args=None)
        node = rclpy.create_node("g1_dex3_seat_compliance")
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
        preflight_frames, _preflight_board, preflight_evidence = _observe_board(
            rclpy,
            node,
            camera,
            camera_info=expected_camera,
            detector=detector,
            snapshot=_snapshot(activation.reference_state, hands),
            frame_count=args.observation_frames,
        )
        if preflight_frames[-1].camera_info.profile_sha256 != expected_camera.profile_sha256:
            raise ValueError("live camera profile differs from hardware.yaml")
        print(
            "READ-ONLY PREFLIGHT PASSED — seated stationary state, both Dex3 states, "
            "rectified camera profile, and the fixed ChArUco board are valid; no "
            "command publisher exists",
            flush=True,
        )
        planner = PersistentTabletopPlanner(
            executable=ROOT / ".venv-planner/bin/g1-curobo-worker",
            log_path=task_run / "planner.log",
        )
        planner.start()
        print(
            "PLANNER READY — the isolated CUDA worker is warm; no robot command publisher exists",
            flush=True,
        )
        _wait_for_space_with_board_preview(
            rclpy,
            node,
            camera,
            condition=args.chair_condition,
            repetitions=args.repetitions,
            no_window=args.no_window,
        )
        if args.hardware_config.read_bytes() != hardware_bytes:
            raise RuntimeError("hardware configuration changed after preflight")
        if args.calibration_bundle.read_bytes() != bundle_bytes:
            raise RuntimeError("calibration bundle changed after preflight")
        if args.task_config.read_bytes() != task_bytes:
            raise RuntimeError("tabletop task configuration changed after preflight")
        raw_recorder = RawEpisodeRecorder(
            task_run / "raw_episode",
            repository=ROOT,
            topics=tabletop_raw_topics(record_camera=not args.skip_camera_recording),
        )
        raw_recorder.start()
        camera_recording = (
            "native depth/RGB and CameraInfo"
            if not args.skip_camera_recording
            else "camera images intentionally excluded"
        )
        print(
            "RAW EPISODE RECORDING — LowState, LowCmd, both Dex3 streams, pelvis/torso/"
            f"RealSense IMUs, TF, and {camera_recording} are active before command creation",
            flush=True,
        )
        activation = _wait_for_activation(observer, states, empty_pose_set, recording)
        gravity = gravity_feedforward(args.hardware_config, activation.reference_state.position)
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
        active_synchronized = synchronized
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
            "gravity feedforward; observing the loaded fixed board and planning both arms",
            flush=True,
        )
        loaded_state = synchronized.observe_state()
        loaded_frames, loaded_board, loaded_evidence = _observe_board(
            rclpy,
            node,
            camera,
            camera_info=expected_camera,
            detector=detector,
            snapshot=_snapshot(loaded_state, held_hands),
            frame_count=args.observation_frames,
            control_check=driver.check,
        )
        plans: dict[str, SupportedEscapePlan] = {}
        for arm in ARM_ORDER:
            request = build_charuco_escape_request(
                arm=arm,
                observation=loaded_board,
                calibration_bundle=bundle,
                calibration_bundle_path=args.calibration_bundle,
                task_config_path=args.task_config,
                random_seed=args.seed,
            )
            request_path = task_run / f"{arm}_escape_request.json"
            plan_path = task_run / f"{arm}_escape_plan.json"
            request.write_json(request_path)
            try:
                planner.request(
                    "plan-charuco-supported-escape",
                    request_path=request_path,
                    output_path=plan_path,
                    control_check=driver.check,
                )
            except (PlannerRequestRejected, RuntimeError) as error:
                driver.check()
                raise RuntimeError(f"{arm} supported-escape planning failed: {error}") from error
            plan = SupportedEscapePlan.from_json(plan_path)
            if plan.request_sha256 != request.content_sha256:
                raise RuntimeError(f"{arm} escape plan belongs to a different request")
            plans[arm] = plan
        combined_plan_sha256 = hashlib.sha256(
            "".join(plans[arm].content_sha256 for arm in ARM_ORDER).encode()
        ).hexdigest()
        synchronized.install_validated_plan(
            pose_set=PoseSet(
                robot_model=model.name,
                mode_machine=5,
                urdf_sha256=model.sha256,
                calibration_arm="right",
            ),
            approved_validation_report_sha256=combined_plan_sha256,
            validated_reference_state=loaded_state,
        )
        q29 = np.asarray(loaded_state.position, dtype=np.float64)
        q14 = _q14(q29)
        _stop_driver(driver, guard)
        driver = None
        observation_events.append(
            {
                "phase": "loaded_baseline",
                "robot_state": loaded_state.to_dict(),
                "board": loaded_evidence,
            }
        )
        print(
            "BOTH PLANS FROZEN — each arm has an independent CuRobo 100 mm "
            "table-normal escape and its byte-identical reverse; beginning one arm at a time",
            flush=True,
        )
        final_stage = (args.repetitions, ARM_ORDER[-1])
        for repetition in range(1, args.repetitions + 1):
            for arm in ARM_ORDER:
                plan = plans[arm]
                synchronized, driver = _stage_executor(
                    arm=arm,
                    trajectory=plan.outbound,
                    q29=q29,
                    q14=q14,
                    model=model,
                    transport=transport,
                    gravity=gravity,
                    control_config=control_config,
                    plan_sha256=plan.content_sha256,
                    acquire=False,
                    heartbeat=_finger_heartbeat(guard, dex_controller),
                    rate_hz=rate_hz,
                )
                active_synchronized = synchronized
                pre_lift_state = synchronized.observe_state()
                _frames, pre_lift_board, pre_lift_evidence = _observe_board(
                    rclpy,
                    node,
                    camera,
                    camera_info=expected_camera,
                    detector=detector,
                    snapshot=_snapshot(pre_lift_state, held_hands),
                    frame_count=args.observation_frames,
                    control_check=driver.check,
                )
                observation_events.append(
                    {
                        "repetition": repetition,
                        "arm": arm,
                        "phase": "pre_lift",
                        "robot_state": pre_lift_state.to_dict(),
                        "board": pre_lift_evidence,
                    }
                )
                _execute_stage(
                    synchronized,
                    driver,
                    plan.outbound,
                    plan_sha256=plan.content_sha256,
                    timeout_s=control_config.motion_timeout_s,
                )
                lifted_state = synchronized.observe_state()
                _frames, lifted_board, lifted_evidence = _observe_board(
                    rclpy,
                    node,
                    camera,
                    camera_info=expected_camera,
                    detector=detector,
                    snapshot=_snapshot(lifted_state, held_hands),
                    frame_count=args.observation_frames,
                    control_check=driver.check,
                )
                observation_events.append(
                    {
                        "repetition": repetition,
                        "arm": arm,
                        "phase": "lifted",
                        "robot_state": lifted_state.to_dict(),
                        "board": lifted_evidence,
                        "camera_motion_from_pre_lift": camera_motion_from_fixed_board(
                            pre_lift_board.camera_T_board,
                            lifted_board.camera_T_board,
                        ),
                    }
                )
                _execute_stage(
                    synchronized,
                    driver,
                    plan.inbound,
                    plan_sha256=plan.content_sha256,
                    timeout_s=control_config.motion_timeout_s,
                )
                returned_state = synchronized.observe_state()
                _frames, returned_board, returned_evidence = _observe_board(
                    rclpy,
                    node,
                    camera,
                    camera_info=expected_camera,
                    detector=detector,
                    snapshot=_snapshot(returned_state, held_hands),
                    frame_count=args.observation_frames,
                    control_check=driver.check,
                )
                observation_events.append(
                    {
                        "repetition": repetition,
                        "arm": arm,
                        "phase": "returned",
                        "robot_state": returned_state.to_dict(),
                        "board": returned_evidence,
                        "camera_return_residual_from_pre_lift": (
                            camera_motion_from_fixed_board(
                                pre_lift_board.camera_T_board,
                                returned_board.camera_T_board,
                            )
                        ),
                        "camera_motion_from_lifted": camera_motion_from_fixed_board(
                            lifted_board.camera_T_board,
                            returned_board.camera_T_board,
                        ),
                    }
                )
                print(
                    f"completed repetition {repetition}/{args.repetitions}: {arm} "
                    "pre-lift observation, 100 mm lift, lifted observation, exact reverse, "
                    "return observation",
                    flush=True,
                )
                if (repetition, arm) != final_stage:
                    _stop_driver(driver, guard)
                    driver = None
        _restore_seated_control(
            driver=driver,
            dex_controller=dex_controller,
            guard=guard,
            synchronized=synchronized,
        )
        status = {
            "status": "completed",
            "commands_robot": True,
            "chair_condition": args.chair_condition,
            "arms": list(ARM_ORDER),
            "repetitions": args.repetitions,
            "lift_m": lift_m,
            "combined_plan_sha256": combined_plan_sha256,
            "per_arm_plan_sha256": {arm: plans[arm].content_sha256 for arm in ARM_ORDER},
            "terminal_action": guard.terminal_action,
            "camera_motion_summary": summarize_camera_motion_cycles(observation_events),
            "calibration_validation_claim": False,
            "interpretation": (
                "fixed-board camera motion plus synchronized raw pelvis/torso/RealSense "
                "IMU, depth, waist, arm, command, and Dex3 evidence; compare matched "
                "cushion and rigid runs before assigning motion to seat compliance"
            ),
        }
        print(
            "SEAT-COMPLIANCE RUN PASSED — both arms completed every 100 mm lift and "
            "exact return; seated FSM 3 restored",
            flush=True,
        )
    except BaseException as error:
        primary_error = error
        status = {
            **status,
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
                    guard.restore_zero_torque(
                        "seat-compliance diagnostic failed or was interrupted"
                    )
                if active_synchronized is not None:
                    active_synchronized.confirm_external_takeover(
                        "PC2 verified AI zero-torque takeover after diagnostic failure"
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
                synchronized=active_synchronized,
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
            atomic_write_json(
                task_run / "board_observations.json",
                {
                    "chair_condition": args.chair_condition,
                    "preflight": locals().get("preflight_evidence"),
                    "events": observation_events,
                },
            )
            atomic_write_json(task_run / "status.json", status)
        except BaseException as error:
            if primary_error is None:
                raise
            print(f"warning: failed to write complete failure artifacts: {error}", file=sys.stderr)
    print(json.dumps({"run": str(task_run), **status}, indent=2, sort_keys=True))
    return 0
