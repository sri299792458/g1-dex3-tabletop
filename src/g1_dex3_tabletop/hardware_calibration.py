"""Single-approval standing automatic Dex3 calibration collection.

Importing this module is inert.  The complete CuRobo preparation and
calibration route is produced before the operator approval; Unitree publishers
are constructed only after SPACE.
"""

from __future__ import annotations

import json
import select
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

from g1_aprilcube_calibration.calibration_bundle import CalibrationBundle
from g1_aprilcube_calibration.clock import SystemClock
from g1_aprilcube_calibration.executor_driver import (
    ExecutorControlDriver,
    SynchronizedPoseExecutor,
)
from g1_aprilcube_calibration.executor_state_machine import ExecutorState, PoseExecutor
from g1_aprilcube_calibration.joint_map import LEFT_ARM_INDICES, RIGHT_ARM_INDICES
from g1_aprilcube_calibration.live_capture import LiveBurstConfig, LiveBurstFrameSource
from g1_aprilcube_calibration.models import utc_now_iso
from g1_aprilcube_calibration.pose_schema import HANDOFF_POSE_ID, PoseSet
from g1_aprilcube_calibration.preview import render_operator_preview
from g1_aprilcube_calibration.process_lock import CommandOwnerLock
from g1_aprilcube_calibration.quality import (
    CameraIntrinsics,
    PoseQualityEvaluator,
    QualityThresholds,
)
from g1_aprilcube_calibration.readiness import StateSampleBuffer
from g1_aprilcube_calibration.ros.camera_adapter import ROSCameraSubscriber
from g1_aprilcube_calibration.session_runner import CaptureSessionRunner
from g1_aprilcube_calibration.session_store import IsolatedSessionStore, SessionStore
from g1_aprilcube_calibration.transports.unitree_arm_sdk import (
    UnitreeArmSDKTransport,
    UnitreeLowStateObserver,
)
from g1_aprilcube_calibration.transports.unitree_dex3 import (
    UnitreeDex3PostureController,
    UnitreeDex3StateObserver,
)
from g1_aprilcube_calibration.urdf_model import URDFModel
from g1_dex3_tabletop.calibration_candidates import (
    CandidateDesignConfig,
    camera_info_from_hardware,
    generate_calibration_candidates,
)
from g1_dex3_tabletop.calibration_execution import (
    CuroboCalibrationOrchestrator,
    pose_set_from_curobo_plan,
)
from g1_dex3_tabletop.execution_plan import pose_set_from_trajectories
from g1_dex3_tabletop.hardware_config import (
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
    _finger_heartbeat,
    _invoke_planner,
    _wait_for_activation,
    _wait_for_hands,
    _wait_for_state,
    _wait_ready,
)
from g1_dex3_tabletop.planning.contracts import (
    CalibrationPlanRequest,
    CalibrationPlanResult,
    Dex3PreparationPlan,
    Dex3PreparationRequest,
    PlannedTrajectory,
    RobotSnapshot,
)

ROOT = Path(__file__).resolve().parents[2]


def _session_name(arm: str) -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"dex3_{arm}_curobo_{stamp}"


def _snapshot(state, hands) -> RobotSnapshot:
    return RobotSnapshot(
        tuple(state.position),
        tuple(hands.left.position),
        tuple(hands.right.position),
    )


def clearance_snapshot(
    source: RobotSnapshot,
    preparation: Dex3PreparationPlan,
    *,
    left_fingers: tuple[float, ...],
    right_fingers: tuple[float, ...],
) -> RobotSnapshot:
    """Return the exact state used to plan the post-preparation route."""

    q29 = np.asarray(source.measured_q29_rad, dtype=np.float64).copy()
    dual = np.asarray(preparation.dual_clearance_q14_rad, dtype=np.float64)
    q29[np.asarray(LEFT_ARM_INDICES)] = dual[:7]
    q29[np.asarray(RIGHT_ARM_INDICES)] = dual[7:]
    return RobotSnapshot(tuple(q29), left_fingers, right_fingers)


def _canonical_json_bytes(value) -> bytes:
    return json.dumps(value, indent=2, sort_keys=True, allow_nan=False).encode() + b"\n"


def _wait_for_space(rclpy, node, camera, *, arm: str, no_window: bool) -> None:
    if not sys.stdin.isatty():
        raise RuntimeError("operator approval requires an interactive terminal")
    print(
        "READ-ONLY PREFLIGHT COMPLETE — no command publisher exists. Confirm "
        "the G1 is standing in Ready under the load-bearing harness, both marker "
        "plates and the complete bilateral shoulder/finger/"
        f"{arm}-arm sweep are clear. Press SPACE once: ",
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
                        "READ-ONLY - SPACE starts complete calibration",
                        (24, 42),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.85,
                        (0, 220, 255),
                        2,
                    )
                    cv2.imshow("G1 Dex3 automatic calibration", rendered)
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


def _stage_trajectory(
    trajectory: PlannedTrajectory,
    *,
    target_id: str | None = None,
) -> PlannedTrajectory:
    """Rebase a separately planned stage onto a fresh ownership handoff."""

    return replace(
        trajectory,
        from_pose_id=HANDOFF_POSE_ID,
        to_pose_id=target_id or trajectory.to_pose_id,
    )


def _stage_executor(
    *,
    arm: str,
    trajectory: PlannedTrajectory | None,
    q29: np.ndarray,
    q14: np.ndarray,
    model: URDFModel,
    transport,
    gravity,
    control_config,
    plan_sha256: str,
    acquire: bool,
    heartbeat,
    rate_hz: float,
):
    trajectories = () if trajectory is None else (trajectory,)
    pose_set = pose_set_from_trajectories(
        arm=arm,
        trajectories=trajectories,
        reference_full_q=q29,
        robot_model=model.name,
        urdf_sha256=model.sha256,
        source="NVlabs/curobo_frozen_calibration_stage",
    )
    active = np.asarray(RIGHT_ARM_INDICES if arm == "right" else LEFT_ARM_INDICES)
    opposite = np.asarray(LEFT_ARM_INDICES if arm == "right" else RIGHT_ARM_INDICES)
    raw = PoseExecutor(
        transport=transport,
        clock=SystemClock(),
        pose_set=pose_set,
        handoff_q=q29[active],
        hold_q=q29[opposite],
        approved_validation_report_sha256=plan_sha256,
        config=control_config,
        gravity_feedforward=gravity,
    )
    synchronized = SynchronizedPoseExecutor(raw)
    driver = ExecutorControlDriver(
        synchronized,
        rate_hz=rate_hz,
        safety_heartbeat=heartbeat,
    )
    if acquire:
        driver.start()
        synchronized.acquire(operator_confirmed=True)
    else:
        synchronized.adopt_owned_control(previous_command_q14=q14)
        driver.start()
    _wait_ready(
        synchronized,
        driver,
        timeout_s=control_config.acquisition_ramp_s + 5.0,
        label=f"{arm} ownership stage",
    )
    return synchronized, driver


def _execute_stage(
    synchronized,
    driver,
    trajectory: PlannedTrajectory,
    *,
    plan_sha256: str,
    timeout_s: float,
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
        timeout_s=max(timeout_s, trajectory.sample_time_s[-1] + 5.0),
        label=trajectory.to_pose_id,
    )


def _stop_driver(driver, guard) -> None:
    driver.close()
    driver.check()
    guard.pulse()


def run_collect_calibration(args) -> int:
    """Plan and collect one finite CuRobo calibration route."""

    if args.confirm != MOTION_ACK:
        raise ValueError(f"--confirm must equal exactly: {MOTION_ACK}")
    hardware_path, target_path = args.hardware_config, args.target_config
    hardware = load_hardware(hardware_path)
    if hardware["robot"]["calibration_arm"] != args.arm:
        raise ValueError("hardware calibration arm differs from --arm")
    if args.session_directory is None:
        session_directory = (ROOT / "sessions" / _session_name(args.arm)).resolve()
    else:
        session_directory = args.session_directory.resolve()
    if session_directory.exists():
        raise FileExistsError(f"session directory already exists: {session_directory}")
    bundle = CalibrationBundle.load(args.calibration_bundle)
    model = URDFModel(resolve_hardware_path(hardware_path, hardware["robot"]["urdf"]))
    expected_camera = camera_info_from_hardware(hardware)
    target = json.loads(target_path.read_text(encoding="utf-8"))
    thresholds = QualityThresholds.from_yaml(args.quality_config)
    recording, pairing = recording_configs(hardware_path)
    control_config, rate_hz = executor_config(hardware_path)
    dex_cfg = dex3_config(
        hardware_path,
        interface=args.network_interface,
        domain_id=args.domain_id,
    )
    design = CandidateDesignConfig(
        target_count=args.target_count,
        candidate_count=args.candidate_count,
        seed=args.seed,
    )
    hardware_bytes = hardware_path.read_bytes()
    target_bytes = target_path.read_bytes()
    quality_bytes = args.quality_config.read_bytes()
    bundle_bytes = args.calibration_bundle.read_bytes()
    planning_directory = ROOT / "work" / f"{session_directory.name}_preflight"
    if planning_directory.exists():
        raise FileExistsError(f"preflight directory already exists: {planning_directory}")
    planning_directory.mkdir(parents=True)
    status_path = planning_directory / "status.json"
    status = {"status": "preflight", "commands_robot": False}
    command_lock = CommandOwnerLock(args.lock_file)
    camera = observer = dex_observer = transport = dex_controller = None
    guard = synchronized = driver = active_synchronized = None
    isolated_store = None
    preflight_snapshot = None
    command_lock.acquire()
    try:
        try:
            import rclpy
        except ImportError as error:
            raise RuntimeError("rclpy unavailable; use ./tools/g1_tabletop_hardware.sh") from error
        rclpy.init(args=None)
        node = rclpy.create_node("g1_dex3_curobo_calibration")
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
                hardware_path,
                interface=args.network_interface,
                domain_id=args.domain_id,
            )
            observer = UnitreeLowStateObserver(transport_cfg, on_sample=states.add)
            _wait_for_state(observer)
            hand_cfg = dex3_config(
                hardware_path,
                interface=args.network_interface,
                domain_id=args.domain_id,
            )
            dex_observer = UnitreeDex3StateObserver(hand_cfg, initialize_factory=False)
            hands = _wait_for_hands(dex_observer)
            empty_pose_set = PoseSet(
                robot_model=model.name,
                mode_machine=5,
                urdf_sha256=model.sha256,
                calibration_arm=args.arm,
            )
            activation = _wait_for_activation(
                observer, states, empty_pose_set, recording, timeout_s=8.0
            )
            preflight_snapshot = _snapshot(activation.reference_state, hands)
            # Wait for a real live rectified frame without requiring the marker
            # to be visible in the compact Ready posture.
            camera_deadline = time.monotonic() + args.camera_timeout_s
            while camera.frames.latest is None and time.monotonic() < camera_deadline:
                rclpy.spin_once(node, timeout_sec=0.02)
            if camera.frames.latest is None:
                raise RuntimeError("timed out waiting for rectified camera messages")
            if camera.frames.latest.camera_info.profile_sha256 != expected_camera.profile_sha256:
                raise ValueError("live CameraInfo differs from hardware.yaml")

            preparation_request = Dex3PreparationRequest(
                snapshot=preflight_snapshot,
                joint_position_offsets_rad=dict(bundle.joint_position_offsets_rad),
                left_target_q_rad=dex_cfg.left_target_q_rad,
                right_target_q_rad=dex_cfg.right_target_q_rad,
                random_seed=args.seed,
            )
            preparation_request_path = planning_directory / "preparation_request.json"
            preparation_plan_path = planning_directory / "preparation_plan.json"
            preparation_request.write_json(preparation_request_path)
            print(
                "READ-ONLY CUROBO PREPARATION — live Ready snapshot, complete "
                "G1/Dex3/marker geometry, and finger sweep; NO MOTION",
                flush=True,
            )
            _invoke_planner(
                "plan-dex3-preparation",
                preparation_request_path,
                preparation_plan_path,
            )
            preparation = Dex3PreparationPlan.from_json(preparation_plan_path)
            planned_snapshot = clearance_snapshot(
                preflight_snapshot,
                preparation,
                left_fingers=dex_cfg.left_target_q_rad,
                right_fingers=dex_cfg.right_target_q_rad,
            )
            candidates = generate_calibration_candidates(
                camera_info=expected_camera,
                target_config=target,
                torso_T_camera=bundle.torso_T_camera,
                palm_T_marker=np.asarray(
                    hardware["robot"]["calibration_target_modeled_hand_T_target"],
                    dtype=np.float64,
                ),
                exposed_marker_normal=np.asarray(
                    hardware["robot"]["calibration_target_exposed_face_normal_target"],
                    dtype=np.float64,
                ),
                config=design,
            )
            calibration_request = CalibrationPlanRequest(
                arm=args.arm,
                snapshot=planned_snapshot,
                torso_T_camera=tuple(
                    tuple(float(value) for value in row) for row in bundle.torso_T_camera
                ),
                palm_T_marker=tuple(
                    tuple(float(value) for value in row)
                    for row in hardware["robot"]["calibration_target_modeled_hand_T_target"]
                ),
                joint_position_offsets_rad=dict(bundle.joint_position_offsets_rad),
                candidates=candidates,
                selection_config=design.to_dict(),
                target_count=args.target_count,
                ik_batch_size=args.ik_batch_size,
                random_seed=args.seed,
            )
            request_path = planning_directory / "calibration_request.json"
            plan_path = planning_directory / "calibration_plan.json"
            calibration_request.write_json(request_path)
            print(
                f"READ-ONLY CUROBO CALIBRATION — selecting and connecting "
                f"{args.target_count} {args.arm}-arm views; NO MOTION",
                flush=True,
            )
            _invoke_planner("plan-calibration", request_path, plan_path)
            plan = CalibrationPlanResult.from_json(plan_path)
            if preparation.request_sha256 != preparation_request.content_sha256:
                raise ValueError("preparation plan/request binding changed")
            if plan.request_sha256 != calibration_request.content_sha256:
                raise ValueError("calibration plan/request binding changed")
            pose_set = pose_set_from_curobo_plan(
                calibration_request,
                plan,
                urdf_sha256=model.sha256,
                robot_model=model.name,
            )
            isolated_store = IsolatedSessionStore(
                session_directory,
                poll_interval_s=1.0 / rate_hz,
            )
            isolated_store.start()
            print(
                f"FINITE ROUTE READY — {len(plan.capture_pose_ids)} targets, "
                f"{len(plan.trajectories)} trajectories, exact return to the "
                f"live-derived handoff; Dex3 outward offset="
                f"{preparation.outward_offset_rad:.4f}rad",
                flush=True,
            )
            _wait_for_space(rclpy, node, camera, arm=args.arm, no_window=args.no_window)
            if hardware_path.read_bytes() != hardware_bytes:
                raise RuntimeError("hardware configuration changed after preflight")
            if target_path.read_bytes() != target_bytes:
                raise RuntimeError("target configuration changed after preflight")
            if args.quality_config.read_bytes() != quality_bytes:
                raise RuntimeError("quality configuration changed after preflight")
            if args.calibration_bundle.read_bytes() != bundle_bytes:
                raise RuntimeError("calibration bundle changed after preflight")
            latest_activation = _wait_for_activation(
                observer, states, empty_pose_set, recording, timeout_s=8.0
            )
            latest_hands = _wait_for_hands(dex_observer)
            latest_snapshot = _snapshot(latest_activation.reference_state, latest_hands)
            drift = max(
                float(
                    np.max(
                        np.abs(
                            np.asarray(latest_snapshot.measured_q29_rad)
                            - np.asarray(preflight_snapshot.measured_q29_rad)
                        )
                    )
                ),
                float(
                    np.max(
                        np.abs(
                            np.asarray(latest_snapshot.left_dex3_q_rad)
                            - np.asarray(preflight_snapshot.left_dex3_q_rad)
                        )
                    )
                ),
                float(
                    np.max(
                        np.abs(
                            np.asarray(latest_snapshot.right_dex3_q_rad)
                            - np.asarray(preflight_snapshot.right_dex3_q_rad)
                        )
                    )
                ),
            )
            if drift > float(hardware["control"]["activation_position_tolerance_rad"]):
                raise RuntimeError(
                    f"robot changed after read-only planning by {drift:.4f}rad; "
                    "no command publisher was created"
                )

            gravity = gravity_feedforward(
                hardware_path, latest_activation.reference_state.position
            )
            guard = watchdog(
                hardware_path,
                host=args.pc2_host,
                ssh_identity=args.pc2_ssh_identity,
                initial_fsm_id=int(hardware["control"]["required_regular_fsm_id"]),
                restore_seated=False,
            )
            guard.start()
            transport = UnitreeArmSDKTransport(transport_cfg, observer=observer)
            observer = None
            dex_controller = UnitreeDex3PostureController(
                hand_cfg,
                observer=dex_observer,
            )
            dex_observer = None
            held_hands = dex_controller.acquire_measured_hold(safety_heartbeat=guard.pulse)
            q29 = np.asarray(latest_activation.reference_state.position, dtype=np.float64)
            q14 = np.concatenate(
                (q29[np.asarray(LEFT_ARM_INDICES)], q29[np.asarray(RIGHT_ARM_INDICES)])
            )

            right_outbound = _stage_trajectory(preparation.right_outbound)
            synchronized, driver = _stage_executor(
                arm="right",
                trajectory=right_outbound,
                q29=q29,
                q14=q14,
                model=model,
                transport=transport,
                gravity=gravity,
                control_config=control_config,
                plan_sha256=preparation.content_sha256,
                acquire=True,
                heartbeat=_finger_heartbeat(guard, dex_controller),
                rate_hz=rate_hz,
            )
            active_synchronized = synchronized
            _execute_stage(
                synchronized,
                driver,
                right_outbound,
                plan_sha256=preparation.content_sha256,
                timeout_s=control_config.motion_timeout_s,
            )
            q29[np.asarray(RIGHT_ARM_INDICES)] = np.asarray(right_outbound.command_q_rad[-1])
            q14[7:] = np.asarray(right_outbound.command_q_rad[-1])
            _stop_driver(driver, guard)
            driver = None

            left_outbound = _stage_trajectory(preparation.left_outbound)
            synchronized, driver = _stage_executor(
                arm="left",
                trajectory=left_outbound,
                q29=q29,
                q14=q14,
                model=model,
                transport=transport,
                gravity=gravity,
                control_config=control_config,
                plan_sha256=preparation.content_sha256,
                acquire=False,
                heartbeat=_finger_heartbeat(guard, dex_controller),
                rate_hz=rate_hz,
            )
            active_synchronized = synchronized
            _execute_stage(
                synchronized,
                driver,
                left_outbound,
                plan_sha256=preparation.content_sha256,
                timeout_s=control_config.motion_timeout_s,
            )
            q29[np.asarray(LEFT_ARM_INDICES)] = np.asarray(left_outbound.command_q_rad[-1])
            q14[:7] = np.asarray(left_outbound.command_q_rad[-1])
            dex_controller.command_posture(
                left_target_q_rad=dex_cfg.left_target_q_rad,
                right_target_q_rad=dex_cfg.right_target_q_rad,
                label="NVIDIA middle-close calibration posture",
                safety_heartbeat=driver.check,
            )
            driver.safety_heartbeat = _finger_heartbeat(guard, dex_controller)
            settled_hands = _wait_for_hands(dex_controller.observer)
            _stop_driver(driver, guard)
            driver = None

            calibration_q29 = q29.copy()
            calibration_q14 = q14.copy()
            raw = PoseExecutor(
                transport=transport,
                clock=SystemClock(),
                pose_set=pose_set,
                handoff_q=plan.handoff_command_q_rad,
                hold_q=(q14[7:] if args.arm == "left" else q14[:7]),
                approved_validation_report_sha256=plan.content_sha256,
                config=control_config,
                gravity_feedforward=gravity,
            )
            synchronized = SynchronizedPoseExecutor(raw)
            active_synchronized = synchronized
            synchronized.adopt_owned_control(previous_command_q14=q14)
            driver = ExecutorControlDriver(
                synchronized,
                rate_hz=rate_hz,
                safety_heartbeat=_finger_heartbeat(guard, dex_controller),
            )
            driver.start()

            SessionStore(session_directory).create(
                session_id=session_directory.name,
                created_at_utc=utc_now_iso(),
                camera_info=expected_camera,
                pose_set_content_sha256=plan.content_sha256,
                artifacts={
                    "hardware.yaml": hardware_bytes,
                    "target.json": target_bytes,
                    "capture_quality.yaml": quality_bytes,
                    "calibration_request.json": _canonical_json_bytes(
                        calibration_request.to_dict()
                    ),
                    "calibration_plan.json": _canonical_json_bytes(plan.to_dict()),
                },
                pairing_config=pairing,
                recording_gate_config=recording,
                provenance={
                    "command": "g1-tabletop collect-calibration",
                    "arm": args.arm,
                    "requested_target_count": len(plan.capture_pose_ids),
                    "yellow_policy": "automatic_accept_with_recorded_warning",
                    "red_policy": "reject_target_and_continue",
                    "route_policy": "finite_curobo_route_exact_handoff_return",
                    "preparation_plan_sha256": preparation.content_sha256,
                    "preparation_outward_offset_rad": preparation.outward_offset_rad,
                    "calibration_bundle_sha256": bundle.content_sha256,
                    "marker_transform_policy": "fixed_CAD_palm_T_marker",
                    "settled_dex3_state": settled_hands.to_dict(),
                },
                collection_method="curobo_automatic",
            )
            isolated_store.set_health_check(driver.check)
            detector = CorrespondenceDetector(target_path)
            evaluator = PoseQualityEvaluator(thresholds)
            progress = {"message": "starting", "accepted": 0, "rejected": 0}
            last_preview_key = None

            def preview(frame, correspondences, quality):
                if args.no_window:
                    return
                intrinsics = CameraIntrinsics(
                    frame.camera_info.rectified_camera_matrix,
                    frame.camera_info.d,
                )
                rendered = render_operator_preview(
                    frame.image_bgr,
                    correspondences,
                    quality,
                    intrinsics=intrinsics,
                    saved_view_count=progress["accepted"],
                    footer_lines=(
                        f"{progress['message']} | CONTROL {synchronized.state.value.upper()}",
                        (
                            f"ACCEPTED {progress['accepted']}/"
                            f"{len(plan.capture_pose_ids)}  "
                            f"REJECTED {progress['rejected']}"
                        ),
                        "RED/NO MARKER = SKIP | CTRL+C = ABORT AND DAMP",
                    ),
                )
                cv2.imshow("G1 Dex3 automatic calibration", rendered)
                cv2.waitKey(1)

            def wait_once(duration_s: float) -> None:
                nonlocal last_preview_key
                rclpy.spin_once(node, timeout_sec=0.0)
                driver.check()
                latest = camera.frames.latest
                if not args.no_window and latest is not None:
                    key = (
                        latest.timing.receipt_monotonic_s,
                        latest.timing.header_stamp_ns,
                    )
                    if key != last_preview_key:
                        detections = detector.detect(latest.image_bgr)
                        quality = evaluator.evaluate(
                            detections,
                            intrinsics=CameraIntrinsics(
                                latest.camera_info.rectified_camera_matrix,
                                latest.camera_info.d,
                            ),
                            history=(),
                        )
                        preview(latest, detections, quality)
                        last_preview_key = key
                time.sleep(duration_s)

            source = LiveBurstFrameSource(
                camera_frames=camera.frames,
                robot_states=states,
                detector=detector,
                quality_evaluator=evaluator,
                recording_config=recording,
                pairing_config=pairing,
                config=LiveBurstConfig(
                    frame_count=thresholds.stationary_burst_frames,
                    timeout_s=args.burst_timeout_s,
                    poll_interval_s=1.0 / rate_hz,
                    maximum_duration_s=thresholds.stationary_burst_maximum_duration_s,
                ),
                wait_once=wait_once,
                accept_yellow=lambda _frame: True,
                preview=preview,
            )
            runner = CaptureSessionRunner(executor=synchronized, store=isolated_store)

            def report(message: str, accepted: int, rejected: int) -> None:
                progress.update(message=message, accepted=accepted, rejected=rejected)
                print(
                    f"{message}; accepted={accepted}/{len(plan.capture_pose_ids)}, "
                    f"rejected={rejected}",
                    flush=True,
                )

            orchestrator = CuroboCalibrationOrchestrator(
                executor=synchronized,
                plan=plan,
                capture_runner=runner,
                frame_source=source,
                wait_until_ready=lambda: _wait_ready(
                    synchronized,
                    driver,
                    timeout_s=control_config.motion_timeout_s + 5.0,
                    label="calibration trajectory",
                ),
                report_progress=report,
            )
            result = orchestrator.run()
            isolated_store.finalize()
            print(
                f"FINITE CALIBRATION ROUTE COMPLETE — accepted={result.accepted_count}, "
                f"rejected={result.rejected_count}; restoring fingers and shoulders",
                flush=True,
            )
            dex_controller.restore_initial_posture(safety_heartbeat=driver.check)
            driver.safety_heartbeat = _finger_heartbeat(guard, dex_controller)
            _stop_driver(driver, guard)
            driver = None

            left_return = _stage_trajectory(
                preparation.left_return,
                target_id="left_shoulder_restored",
            )
            synchronized, driver = _stage_executor(
                arm="left",
                trajectory=left_return,
                q29=calibration_q29,
                q14=calibration_q14,
                model=model,
                transport=transport,
                gravity=gravity,
                control_config=control_config,
                plan_sha256=preparation.content_sha256,
                acquire=False,
                heartbeat=_finger_heartbeat(guard, dex_controller),
                rate_hz=rate_hz,
            )
            active_synchronized = synchronized
            _execute_stage(
                synchronized,
                driver,
                left_return,
                plan_sha256=preparation.content_sha256,
                timeout_s=control_config.motion_timeout_s,
            )
            calibration_q29[np.asarray(LEFT_ARM_INDICES)] = np.asarray(
                left_return.command_q_rad[-1]
            )
            calibration_q14[:7] = np.asarray(left_return.command_q_rad[-1])
            _stop_driver(driver, guard)
            driver = None

            right_return = _stage_trajectory(
                preparation.right_return,
                target_id="right_shoulder_restored",
            )
            synchronized, driver = _stage_executor(
                arm="right",
                trajectory=right_return,
                q29=calibration_q29,
                q14=calibration_q14,
                model=model,
                transport=transport,
                gravity=gravity,
                control_config=control_config,
                plan_sha256=preparation.content_sha256,
                acquire=False,
                heartbeat=_finger_heartbeat(guard, dex_controller),
                rate_hz=rate_hz,
            )
            active_synchronized = synchronized
            _execute_stage(
                synchronized,
                driver,
                right_return,
                plan_sha256=preparation.content_sha256,
                timeout_s=control_config.motion_timeout_s,
            )
            calibration_q29[np.asarray(RIGHT_ARM_INDICES)] = np.asarray(
                right_return.command_q_rad[-1]
            )
            calibration_q14[7:] = np.asarray(right_return.command_q_rad[-1])
            _stop_driver(driver, guard)
            driver = None

            # Rebase the restored live command as the clean-release handoff.
            synchronized, driver = _stage_executor(
                arm="right",
                trajectory=None,
                q29=calibration_q29,
                q14=calibration_q14,
                model=model,
                transport=transport,
                gravity=gravity,
                control_config=control_config,
                plan_sha256="0" * 64,
                acquire=False,
                heartbeat=_finger_heartbeat(guard, dex_controller),
                rate_hz=rate_hz,
            )
            active_synchronized = synchronized
            synchronized.begin_clean_release(operator_confirmed=True)
            deadline = time.monotonic() + control_config.release_ramp_s + 5.0
            while synchronized.state is not ExecutorState.STOPPED:
                driver.check()
                if time.monotonic() >= deadline:
                    raise RuntimeError("clean arm_sdk release timed out")
                time.sleep(0.01)
            _stop_driver(driver, guard)
            driver = None
            dex_controller.timeout()
            guard.disarm()
            synchronized.confirm_external_takeover("clean arm_sdk weight-zero release completed")
            status = {
                "status": "completed",
                "commands_robot": True,
                "session": str(session_directory),
                "arm": args.arm,
                "accepted": result.accepted_count,
                "rejected": result.rejected_count,
                "attempted": result.attempted_count,
                "terminal_action": guard.terminal_action,
            }
            print(
                f"CALIBRATION COLLECTION PASSED — finalized {session_directory}; "
                f"accepted={result.accepted_count}, rejected={result.rejected_count}",
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
        status = {
            "status": "failed",
            "commands_robot": bool(transport is not None and transport.command_count),
            "error_type": type(error).__name__,
            "error": str(error),
            "session": str(session_directory),
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
        if guard is not None and guard.armed:
            try:
                if transport is None or transport.command_count == 0:
                    guard.disarm()
                else:
                    guard.damp("automatic CuRobo calibration failed or was interrupted")
                    if active_synchronized is not None:
                        active_synchronized.confirm_external_damping(
                            "PC2 verified Damp after calibration failure"
                        )
            except BaseException as error:  # noqa: BLE001
                cleanup_errors.append(f"PC2 safety: {error}")
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
        if transport is not None and transport.command_count == 0:
            try:
                transport.close()
            except BaseException as error:  # noqa: BLE001
                cleanup_errors.append(f"transport: {error}")
        if isolated_store is not None:
            try:
                isolated_store.close()
            except BaseException as error:  # noqa: BLE001
                cleanup_errors.append(f"writer: {error}")
        command_lock.release()
        status["cleanup_errors"] = cleanup_errors
        status_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            status_path.write_text(json.dumps(status, indent=2, sort_keys=True) + "\n")
        except OSError:
            pass
    print(json.dumps(status, indent=2, sort_keys=True))
    return 0
