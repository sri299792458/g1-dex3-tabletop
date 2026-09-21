"""Dedicated execution of one frozen same-frame bilateral calibration route.

Importing this module is inert. The command performs a read-only validation
and visibility preflight, then creates Unitree publishers only after SPACE.
It never generates poses or invokes CuRobo in the hardware process.
"""

from __future__ import annotations

import json
import time
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
from aprilcube import CorrespondenceDetector

from g1_aprilcube_calibration.clock import SystemClock
from g1_aprilcube_calibration.executor_driver import (
    ExecutorControlDriver,
    SynchronizedPoseExecutor,
)
from g1_aprilcube_calibration.executor_state_machine import ExecutorState, PoseExecutor
from g1_aprilcube_calibration.joint_map import arm_indices, opposite_arm
from g1_aprilcube_calibration.live_capture import LiveBurstConfig
from g1_aprilcube_calibration.models import utc_now_iso
from g1_aprilcube_calibration.process_lock import CommandOwnerLock
from g1_aprilcube_calibration.quality import (
    CameraIntrinsics,
    PoseQualityEvaluator,
    QualityGrade,
    QualityThresholds,
)
from g1_aprilcube_calibration.readiness import StateSampleBuffer
from g1_aprilcube_calibration.ros.camera_adapter import ROSCameraSubscriber
from g1_aprilcube_calibration.transports.unitree_arm_sdk import (
    UnitreeArmSDKTransport,
    UnitreeLowStateObserver,
)
from g1_aprilcube_calibration.transports.unitree_dex3 import (
    UnitreeDex3PostureController,
    UnitreeDex3StateObserver,
)
from g1_aprilcube_calibration.urdf_model import URDFModel
from g1_dex3_tabletop.calibration import (
    BILATERAL_SOURCE_ARTIFACTS,
    BilateralCollectionOrchestrator,
    BilateralExecutionPlan,
    BilateralLiveBurstSource,
    BilateralPoseDesignArtifact,
    BilateralSessionStore,
    CameraFrameArtifact,
    pose_sets_from_bilateral_plan,
)
from g1_dex3_tabletop.calibration_candidates import camera_info_from_hardware
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
    _wait_for_activation,
    _wait_for_hands,
    _wait_for_space_with_preview,
    _wait_for_state,
    _wait_ready,
)

ROOT = Path(__file__).resolve().parents[2]
_SIDES = ("left", "right")


def _session_name() -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"dex3_bilateral_curobo_{stamp}"


def validate_frozen_bilateral_inputs(
    *,
    design: BilateralPoseDesignArtifact,
    plan: BilateralExecutionPlan,
    model: URDFModel,
    camera_frames: CameraFrameArtifact,
    expected_camera,
) -> dict:
    """Validate all immutable route/model/camera bindings without hardware writes."""

    plan.validate_design(design)
    if plan.robot_model != model.name:
        raise ValueError("bilateral execution plan belongs to a different robot model")
    if plan.urdf_sha256 != model.sha256:
        raise ValueError("bilateral execution plan belongs to a different robot URDF")
    if camera_frames.camera_serial != expected_camera.serial_number:
        raise ValueError("camera-frame artifact belongs to a different RealSense serial")
    if camera_frames.optical_frame != expected_camera.frame_id:
        raise ValueError("camera-frame artifact optical frame differs from CameraInfo")
    if camera_frames.urdf_parent_link not in model.links:
        raise ValueError("camera-frame artifact parent link is absent from the robot URDF")
    return pose_sets_from_bilateral_plan(design, plan)


def _handoff_error_rad(design: BilateralPoseDesignArtifact, q29) -> float:
    handoff = np.asarray(
        design.waypoint_joint_positions_rad[design.schedule[0].candidate_id],
        dtype=np.float64,
    )
    measured = np.asarray(q29, dtype=np.float64).reshape(-1)
    if measured.shape != handoff.shape:
        raise ValueError("live state and bilateral handoff have different joint counts")
    return float(np.max(np.abs(measured - handoff)))


def _hand_posture_error_rad(hands, target_q_rad_by_arm) -> float:
    return max(
        float(
            np.max(
                np.abs(np.asarray(hands.left.position) - np.asarray(target_q_rad_by_arm["left"]))
            )
        ),
        float(
            np.max(
                np.abs(np.asarray(hands.right.position) - np.asarray(target_q_rad_by_arm["right"]))
            )
        ),
    )


def _wait_for_bilateral_visibility(
    rclpy,
    node,
    camera,
    *,
    detectors,
    evaluators,
    expected_camera,
    timeout_s: float,
):
    deadline = time.monotonic() + timeout_s
    last = "no rectified image"
    while time.monotonic() < deadline:
        rclpy.spin_once(node, timeout_sec=0.02)
        frame = camera.frames.latest
        if frame is None:
            continue
        if frame.camera_info.profile_sha256 != expected_camera.profile_sha256:
            raise ValueError("live CameraInfo differs from the frozen hardware profile")
        intrinsics = CameraIntrinsics(
            frame.camera_info.rectified_camera_matrix,
            frame.camera_info.d,
        )
        results = {side: detectors[side].detect(frame.image_bgr) for side in _SIDES}
        reports = {
            side: evaluators[side].evaluate(
                results[side],
                intrinsics=intrinsics,
                history=(),
            )
            for side in _SIDES
        }
        failures = []
        for side in _SIDES:
            if not results[side].valid:
                failures.append(f"{side} target is not detected unambiguously")
            elif reports[side].grade is QualityGrade.RED:
                failures.append(
                    f"{side} quality is red: " + "; ".join(reports[side].hard_failures)
                )
        if not failures:
            return frame, results, reports
        last = "; ".join(failures)
    raise RuntimeError("timed out waiting for both hand targets: " + last)


def _render_bilateral_preview(image, detections, reports, *, footer: tuple[str, ...]):
    canvas = np.asarray(image).copy()
    colors = {"left": (255, 180, 0), "right": (0, 220, 255)}
    for side in _SIDES:
        color = colors[side]
        for observation in detections[side].observations:
            corners = np.rint(observation.image_corners_px).astype(np.int32)
            cv2.polylines(canvas, [corners], True, color, 3, cv2.LINE_AA)
        cv2.putText(
            canvas,
            f"{side.upper()}: {reports[side].grade.value.upper()}",
            (24, 42 if side == "left" else 76),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            color,
            2,
            cv2.LINE_AA,
        )
    y = canvas.shape[0] - 24 * len(footer)
    for line in footer:
        cv2.putText(
            canvas,
            line,
            (24, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.58,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        y += 24
    return canvas


def _stop_driver(driver, guard) -> None:
    driver.close()
    driver.check()
    guard.pulse()


def run_collect_bilateral_calibration(args) -> int:
    """Execute and record one exact, prevalidated bilateral calibration route."""

    if args.confirm != MOTION_ACK:
        raise ValueError(f"--confirm must equal exactly: {MOTION_ACK}")
    if args.maximum_capture_attempts < 1:
        raise ValueError("--maximum-capture-attempts must be positive")

    hardware_path = args.hardware_config.resolve()
    hardware = load_hardware(hardware_path)
    design = BilateralPoseDesignArtifact.from_json(args.pose_design)
    plan = BilateralExecutionPlan.from_json(args.execution_plan)
    camera_frames = CameraFrameArtifact.from_json(args.camera_frames)
    model = URDFModel(resolve_hardware_path(hardware_path, hardware["robot"]["urdf"]))
    expected_camera = camera_info_from_hardware(hardware)
    pose_sets = validate_frozen_bilateral_inputs(
        design=design,
        plan=plan,
        model=model,
        camera_frames=camera_frames,
        expected_camera=expected_camera,
    )
    initial_arm = plan.transitions[0].arm
    recording, pairing = recording_configs(hardware_path)
    recording = replace(recording, calibration_arm=initial_arm)
    control_config, rate_hz = executor_config(hardware_path)
    dex_config_value = dex3_config(
        hardware_path,
        interface=args.network_interface,
        domain_id=args.domain_id,
    )
    configured_dex3_q = {
        "left": dex_config_value.left_target_q_rad,
        "right": dex_config_value.right_target_q_rad,
    }
    planned_posture_model_error = max(
        float(
            np.max(
                np.abs(
                    np.asarray(plan.dex3_joint_positions_rad[side])
                    - np.asarray(configured_dex3_q[side])
                )
            )
        )
        for side in _SIDES
    )
    if planned_posture_model_error > dex_config_value.posture_position_tolerance_rad:
        raise ValueError(
            "bilateral execution plan uses a Dex3 posture outside the commissioned "
            f"middle-close tolerance by {planned_posture_model_error:.4f}rad"
        )
    thresholds = QualityThresholds.from_yaml(args.quality_config)
    detectors = {
        "left": CorrespondenceDetector(args.left_target_config),
        "right": CorrespondenceDetector(args.right_target_config),
    }
    if detectors["left"].valid_ids & detectors["right"].valid_ids:
        raise ValueError("left and right target artifacts must use disjoint marker IDs")
    evaluators = {side: PoseQualityEvaluator(thresholds) for side in _SIDES}

    session_directory = (
        (ROOT / "sessions" / _session_name()).resolve()
        if args.session_directory is None
        else args.session_directory.resolve()
    )
    if session_directory.exists():
        raise FileExistsError(f"session directory already exists: {session_directory}")
    work_directory = ROOT / "work" / f"{session_directory.name}_collection"
    if work_directory.exists():
        raise FileExistsError(f"collection work directory already exists: {work_directory}")
    work_directory.mkdir(parents=True)
    status_path = work_directory / "status.json"
    status = {"status": "preflight", "commands_robot": False}

    paths = {
        "camera_frames.json": args.camera_frames.resolve(),
        "capture_quality.yaml": args.quality_config.resolve(),
        "execution_plan.json": args.execution_plan.resolve(),
        "left_target.json": args.left_target_config.resolve(),
        "pose_design.json": args.pose_design.resolve(),
        "right_target.json": args.right_target_config.resolve(),
        "robot.urdf": model.path,
    }
    if set(paths) != BILATERAL_SOURCE_ARTIFACTS:
        raise AssertionError("bilateral source path contract changed")
    source_artifacts = {name: path.read_bytes() for name, path in paths.items()}

    command_lock = CommandOwnerLock(args.lock_file)
    camera = observer = dex_observer = transport = dex_controller = None
    guard = synchronized = driver = None
    command_lock.acquire()
    try:
        try:
            import rclpy
        except ImportError as error:
            raise RuntimeError("rclpy unavailable; use ./tools/g1_tabletop_hardware.sh") from error
        rclpy.init(args=None)
        node = rclpy.create_node("g1_dex3_bilateral_calibration")
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
            dex_observer = UnitreeDex3StateObserver(
                dex_config_value,
                initialize_factory=False,
            )
            hands = _wait_for_hands(dex_observer)
            activation = _wait_for_activation(
                observer,
                states,
                pose_sets[initial_arm],
                recording,
                timeout_s=8.0,
            )
            handoff_error = _handoff_error_rad(design, activation.reference_state.position)
            if handoff_error > control_config.settled_position_spread_rad:
                raise RuntimeError(
                    f"live robot differs from the frozen bilateral handoff by "
                    f"{handoff_error:.4f}rad; limit is "
                    f"{control_config.settled_position_spread_rad:.4f}rad"
                )
            hand_error = _hand_posture_error_rad(hands, plan.dex3_joint_positions_rad)
            if hand_error > dex_config_value.posture_position_tolerance_rad:
                raise RuntimeError(
                    f"live Dex3 posture differs from the modeled marker posture by "
                    f"{hand_error:.4f}rad; limit is "
                    f"{dex_config_value.posture_position_tolerance_rad:.4f}rad"
                )
            _wait_for_bilateral_visibility(
                rclpy,
                node,
                camera,
                detectors=detectors,
                evaluators=evaluators,
                expected_camera=expected_camera,
                timeout_s=args.camera_timeout_s,
            )
            print(
                f"READ-ONLY BILATERAL PREFLIGHT PASSED — "
                f"{len(design.schedule)} same-frame captures, "
                f"{len(plan.transitions)} frozen trajectories, exact handoff return; "
                "NO MOTION",
                flush=True,
            )
            _wait_for_space_with_preview(
                rclpy,
                node,
                camera,
                arm="both",
                no_window=args.no_window,
                prompt=(
                    "START BILATERAL CALIBRATION — G1 standing in the frozen handoff "
                    "under the load-bearing harness; both hand plates visible; complete "
                    "two-arm route clear. Press SPACE once: "
                ),
                overlay="READ-ONLY - SPACE starts frozen bilateral route",
            )
            changed = [
                name for name, path in paths.items() if path.read_bytes() != source_artifacts[name]
            ]
            if changed:
                raise RuntimeError(
                    "bilateral source artifacts changed after preflight: " + ", ".join(changed)
                )
            latest_activation = _wait_for_activation(
                observer,
                states,
                pose_sets[initial_arm],
                recording,
                timeout_s=8.0,
            )
            latest_hands = _wait_for_hands(dex_observer)
            latest_handoff_error = _handoff_error_rad(
                design,
                latest_activation.reference_state.position,
            )
            latest_hand_error = _hand_posture_error_rad(
                latest_hands,
                plan.dex3_joint_positions_rad,
            )
            if latest_handoff_error > control_config.settled_position_spread_rad:
                raise RuntimeError("robot changed after bilateral preflight; no publisher created")
            if latest_hand_error > dex_config_value.posture_position_tolerance_rad:
                raise RuntimeError("Dex3 posture changed after preflight; no publisher created")

            store = BilateralSessionStore(session_directory)
            store.create(
                session_id=session_directory.name,
                created_at_utc=utc_now_iso(),
                day_group_id=args.day_group_id or datetime.now(timezone.utc).date().isoformat(),
                camera_info=expected_camera,
                camera_frames=camera_frames,
                pose_design_sha256=design.content_sha256,
                execution_plan_sha256=plan.content_sha256,
                source_artifacts=source_artifacts,
                pairing_config=pairing,
                recording_gate_config=recording,
                provenance={
                    "command": "g1-tabletop collect-bilateral-calibration",
                    "route_policy": "offline_frozen_curobo_edges_exact_handoff_return",
                    "capture_policy": "same_image_left_and_right_targets",
                    "yellow_policy": "accepted_with_recorded_quality_evidence",
                    "maximum_capture_attempts": args.maximum_capture_attempts,
                    "initial_arm": initial_arm,
                    "handoff_error_rad": latest_handoff_error,
                    "dex3_posture_error_rad": latest_hand_error,
                    "planned_dex3_to_commissioned_posture_error_rad": (
                        planned_posture_model_error
                    ),
                },
            )

            gravity = gravity_feedforward(
                hardware_path,
                latest_activation.reference_state.position,
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
                dex_config_value,
                observer=dex_observer,
            )
            dex_observer = None
            dex_controller.acquire_measured_hold(safety_heartbeat=guard.pulse)
            dex_controller.command_posture(
                left_target_q_rad=plan.dex3_joint_positions_rad["left"],
                right_target_q_rad=plan.dex3_joint_positions_rad["right"],
                label="frozen bilateral CuRobo posture",
                safety_heartbeat=guard.pulse,
            )

            handoff_q29 = np.asarray(
                design.waypoint_joint_positions_rad[design.schedule[0].candidate_id],
                dtype=np.float64,
            )
            raw = PoseExecutor(
                transport=transport,
                clock=SystemClock(),
                pose_set=pose_sets[initial_arm],
                handoff_q=handoff_q29[np.asarray(arm_indices(initial_arm))],
                hold_q=handoff_q29[np.asarray(arm_indices(opposite_arm(initial_arm)))],
                approved_validation_report_sha256=plan.content_sha256,
                config=control_config,
                gravity_feedforward=gravity,
            )
            synchronized = SynchronizedPoseExecutor(raw)
            driver = ExecutorControlDriver(
                synchronized,
                rate_hz=rate_hz,
                safety_heartbeat=_finger_heartbeat(guard, dex_controller),
            )
            driver.start()
            synchronized.acquire(operator_confirmed=True)
            _wait_ready(
                synchronized,
                driver,
                timeout_s=control_config.acquisition_ramp_s + 5.0,
                label="bilateral handoff ownership",
            )
            # Acquisition intentionally seeds the command from the latest
            # measurement. Rebase that command to the CuRobo-validated anchor
            # only through the executor's commissioned bounded plan-install
            # transition; this makes the first frozen trajectory sample exact.
            validated_reference = replace(
                latest_activation.reference_state,
                position=handoff_q29,
            )
            synchronized.install_validated_plan(
                pose_set=pose_sets[initial_arm],
                approved_validation_report_sha256=plan.content_sha256,
                validated_reference_state=validated_reference,
            )
            aligned = _wait_for_activation(
                transport,
                states,
                pose_sets[initial_arm],
                recording,
                timeout_s=8.0,
            )
            aligned_error = _handoff_error_rad(design, aligned.reference_state.position)
            if aligned_error > control_config.activation_position_tolerance_rad:
                raise RuntimeError(
                    f"loaded bilateral handoff error is {aligned_error:.4f}rad; "
                    f"limit is {control_config.activation_position_tolerance_rad:.4f}rad"
                )

            progress = {"message": "starting", "accepted": 0, "retries": 0}

            def preview(frame, detections, reports) -> None:
                if args.no_window:
                    return
                rendered = _render_bilateral_preview(
                    frame.image_bgr,
                    detections,
                    reports,
                    footer=(
                        f"{progress['message']} | {synchronized.state.value.upper()}",
                        (
                            f"ACCEPTED {progress['accepted']}/{len(design.schedule)}  "
                            f"RETRIES {progress['retries']}"
                        ),
                        "BOTH TARGETS REQUIRED IN THE SAME FRAME | CTRL+C DAMP",
                    ),
                )
                cv2.imshow("G1 Dex3 bilateral calibration", rendered)
                cv2.waitKey(1)

            def wait_once(duration_s: float) -> None:
                rclpy.spin_once(node, timeout_sec=0.0)
                driver.check()
                time.sleep(duration_s)

            source = BilateralLiveBurstSource(
                camera_frames=camera.frames,
                robot_states=states,
                detectors=detectors,
                quality_evaluators=evaluators,
                recording_config=recording,
                pairing_config=pairing,
                config=LiveBurstConfig(
                    frame_count=thresholds.stationary_burst_frames,
                    timeout_s=args.burst_timeout_s,
                    poll_interval_s=1.0 / rate_hz,
                    maximum_duration_s=thresholds.stationary_burst_maximum_duration_s,
                ),
                wait_once=wait_once,
                accept_yellow=lambda _frame, _sides: True,
                preview=preview,
            )

            def report(message: str, accepted: int, retries: int) -> None:
                progress.update(message=message, accepted=accepted, retries=retries)
                print(
                    f"{message}; accepted={accepted}/{len(design.schedule)}, retries={retries}",
                    flush=True,
                )

            maximum_transition_duration_s = max(
                transition.trajectory.sample_time_s[-1] for transition in plan.transitions
            )
            orchestrator = BilateralCollectionOrchestrator(
                executor=synchronized,
                design=design,
                plan=plan,
                pose_sets=pose_sets,
                store=store,
                frame_source=source,
                wait_until_ready=lambda: _wait_ready(
                    synchronized,
                    driver,
                    timeout_s=max(
                        control_config.motion_timeout_s,
                        maximum_transition_duration_s,
                    )
                    + 5.0,
                    label="bilateral calibration trajectory",
                ),
                maximum_capture_attempts=args.maximum_capture_attempts,
                report_progress=report,
            )
            result = orchestrator.run()
            synchronized.begin_clean_release(operator_confirmed=True)
            deadline = time.monotonic() + control_config.release_ramp_s + 5.0
            while synchronized.state is not ExecutorState.STOPPED:
                driver.check()
                if time.monotonic() >= deadline:
                    raise RuntimeError("clean bilateral arm_sdk release timed out")
                time.sleep(0.01)
            _stop_driver(driver, guard)
            driver = None
            dex_controller.timeout()
            guard.disarm()
            synchronized.confirm_external_takeover(
                "clean bilateral arm_sdk weight-zero release completed"
            )
            status = {
                "status": "completed",
                "commands_robot": True,
                "session": str(session_directory),
                "pose_design_sha256": design.content_sha256,
                "execution_plan_sha256": plan.content_sha256,
                "accepted": result.accepted_count,
                "retries": result.retry_count,
                "attempted": result.attempted_count,
                "terminal_action": guard.terminal_action,
            }
            print(
                f"BILATERAL CALIBRATION COLLECTION PASSED — finalized "
                f"{session_directory}; accepted={result.accepted_count}, "
                f"retries={result.retry_count}",
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
            "commands_robot": bool(
                (transport is not None and transport.command_count)
                or (dex_controller is not None and dex_controller.command_count)
            ),
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
                    guard.damp("bilateral calibration failed or was interrupted")
                    if synchronized is not None:
                        synchronized.confirm_external_damping(
                            "PC2 verified Damp after bilateral calibration failure"
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
        command_lock.release()
        status["cleanup_errors"] = cleanup_errors
        try:
            status_path.write_text(json.dumps(status, indent=2, sort_keys=True) + "\n")
        except OSError:
            pass
    print(json.dumps(status, indent=2, sort_keys=True))
    return 0
