"""Fixed two-cube stack hardware workflow.

This module deliberately coordinates exactly two existing pick/place operations:
move the 60 mm cube on the table, observe the result, then place the 40 mm cube
on it.  It is not a task language.
"""

from __future__ import annotations

import json
import sys
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from aprilcube import CorrespondenceDetector

from g1_aprilcube_calibration.calibration_bundle import CalibrationBundle
from g1_aprilcube_calibration.clock import SystemClock
from g1_aprilcube_calibration.config import QualityThresholds
from g1_aprilcube_calibration.executor_driver import (
    ExecutorControlDriver,
    SynchronizedPoseExecutor,
)
from g1_aprilcube_calibration.executor_state_machine import PoseExecutor
from g1_aprilcube_calibration.joint_map import arm_indices
from g1_aprilcube_calibration.pose_schema import PoseSet
from g1_aprilcube_calibration.process_lock import CommandOwnerLock
from g1_aprilcube_calibration.readiness import StateSampleBuffer
from g1_aprilcube_calibration.ros.camera_adapter import ROSCameraSubscriber, ROSImageFrame
from g1_aprilcube_calibration.transforms import invert_transform
from g1_aprilcube_calibration.transports.unitree_arm_sdk import UnitreeLowStateObserver
from g1_aprilcube_calibration.transports.unitree_debug_lowcmd import UnitreeDebugLowCmdTransport
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
from g1_dex3_tabletop.hardware_tabletop import (
    MOTION_ACK,
    TabletopTaskRejected,
    _collect_frames,
    _command_fingers,
    _command_retention_test_close,
    _execute_trajectory,
    _finger_heartbeat,
    _resolve_task_velocity,
    _restore_seated_control,
    _save_frames,
    _snapshot,
    _teardown_ros_runtime,
    _trajectory_with_endpoints,
    _wait_for_activation,
    _wait_for_hands,
    _wait_for_space_with_preview,
    _wait_for_state,
)
from g1_dex3_tabletop.persistent_planner import (
    PersistentTabletopPlanner,
    PlannerRequestRejected,
)
from g1_dex3_tabletop.planning.contracts import PlannedTrajectory, RobotSnapshot, atomic_write_json
from g1_dex3_tabletop.planning.dex3_handedness import (
    dex3_empty_close_reference,
    dex3_execution_profile,
)
from g1_dex3_tabletop.raw_episode_recording import RawEpisodeRecorder, tabletop_raw_topics
from g1_dex3_tabletop.stack_workflow import (
    build_observed_stack_second_stage_request,
    build_stack_stage_requests,
    request_base_T_camera,
    request_hand_positions,
    stack_arm_assignments,
    stack_placement_candidates,
)
from g1_dex3_tabletop.tabletop_contracts import (
    PickPlaceRetentionRouteValidationRequest,
    RetentionRouteValidationResult,
    SupportedEscapePlan,
    TabletopCuboid,
    TabletopObservation,
    TabletopPickPlacePlan,
    TabletopPickPlaceRequest,
    TabletopTaskRequest,
)
from g1_dex3_tabletop.tabletop_object import load_tabletop_object_profile
from g1_dex3_tabletop.tabletop_perception import observe_resting_cube_pair
from g1_dex3_tabletop.tabletop_workflow import build_tabletop_request, load_task_config

ROOT = Path(__file__).resolve().parents[2]


def _run_id() -> str:
    return datetime.now(timezone.utc).strftime("stack_%Y%m%dT%H%M%SZ")


def _tuple_transform(value: np.ndarray) -> tuple[tuple[float, ...], ...]:
    return tuple(tuple(float(item) for item in row) for row in value)


def _exact_clearance_snapshot(
    state,
    hands,
    *,
    left_escape: SupportedEscapePlan,
    right_escape: SupportedEscapePlan,
) -> RobotSnapshot:
    q29 = np.asarray(state.position, dtype=np.float64).copy()
    q29[np.asarray(arm_indices("left"))] = np.asarray(left_escape.outbound.command_q_rad[-1])
    q29[np.asarray(arm_indices("right"))] = np.asarray(right_escape.outbound.command_q_rad[-1])
    return RobotSnapshot(
        measured_q29_rad=tuple(q29),
        left_dex3_q_rad=tuple(hands.left.position),
        right_dex3_q_rad=tuple(hands.right.position),
    )


def _observe_pair(
    frames: tuple[ROSImageFrame, ...],
    *,
    expected_camera,
    cube40_detector,
    cube60_detector,
    snapshot: RobotSnapshot,
    quality: QualityThresholds,
) -> tuple[TabletopObservation, TabletopObservation]:
    return observe_resting_cube_pair(
        [item.image_bgr for item in frames],
        camera_info=expected_camera,
        cube40_detector=cube40_detector,
        cube60_detector=cube60_detector,
        snapshot=snapshot,
        minimum_tag_short_side_px=quality.minimum_tag_short_side_px,
        maximum_reprojection_error_px=quality.pnp_reject_reprojection_px,
    )


def _with_other_cube(
    request: TabletopTaskRequest,
    *,
    other_id: str,
    other_observation: TabletopObservation,
    other_dimensions_m: tuple[float, ...],
) -> TabletopTaskRequest:
    primary_T_other = invert_transform(
        np.asarray(request.observation.camera_T_object)
    ) @ np.asarray(other_observation.camera_T_object)
    return replace(
        request,
        environment_cuboids=(
            TabletopCuboid(
                object_id=other_id,
                object_T_cuboid=_tuple_transform(primary_T_other),
                dimensions_m=other_dimensions_m,
            ),
        ),
    )


def _install_plan_at_current_boundary(
    synchronized,
    *,
    arm: str,
    trajectories: tuple[PlannedTrajectory, ...],
    recovery_trajectories: tuple[PlannedTrajectory, ...] = (),
    plan_sha256: str,
    validated_reference_state,
    model: URDFModel,
) -> tuple[PlannedTrajectory, ...]:
    """Install one immutable arm plan without changing the 14-joint command."""

    if not trajectories:
        raise ValueError("cannot install an empty stack motion plan")
    current_id = synchronized.current_pose_id
    if current_id is None:
        raise RuntimeError("stack plan installation requires a named settled boundary")
    selected_is_current = synchronized.pose_set.calibration_arm == arm
    executable = trajectories
    if selected_is_current and trajectories[0].from_pose_id != current_id:
        executable = (
            _trajectory_with_endpoints(
                trajectories[0],
                from_pose_id=current_id,
                to_pose_id=trajectories[0].to_pose_id,
            ),
            *trajectories[1:],
        )
    boundary_id = current_id if selected_is_current else trajectories[0].from_pose_id
    all_routes = (*executable, *recovery_trajectories)
    pose_set = pose_set_from_trajectories(
        arm=arm,
        trajectories=all_routes,
        reference_full_q=validated_reference_state.position,
        robot_model=model.name,
        urdf_sha256=model.sha256,
        source="NVlabs/curobo_fixed_two_cube_stack",
        initial_pose_id=boundary_id if boundary_id != "__handoff__" else None,
        initial_command_q_rad=(
            executable[0].command_q_rad[0] if boundary_id != "__handoff__" else None
        ),
    )
    if selected_is_current:
        synchronized.replace_validated_remaining_plan(
            pose_set=pose_set,
            approved_validation_report_sha256=plan_sha256,
            validated_reference_state=validated_reference_state,
        )
    else:
        synchronized.switch_validated_arm_plan(
            pose_set=pose_set,
            approved_validation_report_sha256=plan_sha256,
            validated_reference_state=validated_reference_state,
            boundary_pose_id=boundary_id,
        )
    return executable


def _pick_place_recovery_routes(
    plan: TabletopPickPlacePlan,
) -> tuple[PlannedTrajectory, PlannedTrajectory, PlannedTrajectory, PlannedTrajectory]:
    source = plan.source_task.trajectories
    reverse_test_lift = _trajectory_with_endpoints(
        source[5],
        from_pose_id="retention_test_lift",
        to_pose_id="recovery_grasp",
    )
    retreat_from_grasp = _trajectory_with_endpoints(
        source[6],
        from_pose_id="grasp_approach",
        to_pose_id="recovery_pregrasp",
    )
    retreat_after_test = _trajectory_with_endpoints(
        source[6],
        from_pose_id="recovery_grasp",
        to_pose_id="recovery_pregrasp",
    )
    return_to_clearance = _trajectory_with_endpoints(
        source[7],
        from_pose_id="recovery_pregrasp",
        to_pose_id="return_to_clearance",
    )
    return reverse_test_lift, retreat_from_grasp, retreat_after_test, return_to_clearance


def _plan_pick_place(
    planner,
    *,
    request: TabletopPickPlaceRequest,
    directory: Path,
    driver,
) -> TabletopPickPlacePlan:
    directory.mkdir(parents=True, exist_ok=True)
    request_path = directory / "request.json"
    plan_path = directory / "plan.json"
    request.write_json(request_path)
    planner.request(
        "plan-tabletop-pick-place",
        request_path=request_path,
        output_path=plan_path,
        control_check=driver.check,
    )
    plan = TabletopPickPlacePlan.from_json(plan_path)
    if plan.request_sha256 != request.content_sha256:
        raise RuntimeError("pick-place plan belongs to another request")
    return plan


def _execute_pick_place(
    *,
    label: str,
    request: TabletopPickPlaceRequest,
    plan: TabletopPickPlacePlan,
    plan_directory: Path,
    synchronized,
    driver,
    planner,
    dex_controller,
    guard,
    model: URDFModel,
    control_config,
    measured_open_left,
    measured_open_right,
) -> dict[str, object]:
    """Execute one already planned fixed pick/place operation."""

    arm = plan.arm
    empty_close, minimum_shortfall = dex3_empty_close_reference(arm)
    left_open_target, _left_close = dex3_execution_profile("left")
    right_open_target, _right_close = dex3_execution_profile("right")
    recovery = _pick_place_recovery_routes(plan)
    boundary_state = synchronized.observe_state()
    trajectories = _install_plan_at_current_boundary(
        synchronized,
        arm=arm,
        trajectories=plan.trajectories,
        recovery_trajectories=recovery,
        plan_sha256=plan.content_sha256,
        validated_reference_state=boundary_state,
        model=model,
    )
    routes = {value.to_pose_id: value for value in trajectories}

    def execute(trajectory: PlannedTrajectory) -> None:
        _execute_trajectory(
            synchronized,
            driver,
            trajectory,
            plan_sha256=plan.content_sha256,
            control_config=control_config,
        )

    def open_active(open_label: str) -> None:
        _command_fingers(
            dex_controller,
            driver,
            guard,
            left=left_open_target,
            right=right_open_target,
            left_acceptance=measured_open_left,
            right_acceptance=measured_open_right,
            label=open_label,
        )

    def recover_from_grasp() -> None:
        open_active(f"{label}: release rejected grasp at source")
        execute(recovery[1])
        execute(recovery[3])

    execute(routes["move_to_pregrasp"])
    execute(routes["grasp_approach"])
    close_target = plan.source_task.close_target_active_dex3_q_rad
    try:
        close = _command_retention_test_close(
            dex_controller,
            driver,
            guard,
            active_side=arm,
            left=close_target if arm == "left" else measured_open_left,
            right=close_target if arm == "right" else measured_open_right,
            empty_close_reference_q_rad=empty_close,
            minimum_opposed_shortfall_rad=minimum_shortfall,
            label=f"{label}: descriptor-defined {arm}-hand close",
        )
    except Dex3GraspNotAcquiredError as error:
        driver.check()
        recover_from_grasp()
        raise TabletopTaskRejected(f"{label} did not acquire the cube: {error}") from error
    atomic_write_json(plan_directory / "grasp_close.json", close.to_dict())
    dex_controller.begin_retention_test()
    retention_request = PickPlaceRetentionRouteValidationRequest(
        pick_place_request=request,
        pick_place_plan=plan,
        measured_active_dex3_q_rad=close.close_q_rad,
        blocked_motor_ids=close.blocked_motor_ids,
    )
    retention_request_path = plan_directory / "retention_request.json"
    retention_result_path = plan_directory / "retention_validation.json"
    retention_request.write_json(retention_request_path)
    try:
        planner.request(
            "validate-pick-place-retention-route",
            request_path=retention_request_path,
            output_path=retention_result_path,
            control_check=driver.check,
        )
        retention = RetentionRouteValidationResult.from_json(retention_result_path)
        dex_controller.check_retention_test()
    except (PlannerRequestRejected, Dex3RetentionLostError, RuntimeError) as error:
        driver.check()
        recover_from_grasp()
        raise TabletopTaskRejected(f"{label} measured close did not validate: {error}") from error
    execute(routes["retention_test_lift"])
    try:
        evidence = dex_controller.verify_retention_at_lifted_checkpoint(
            safety_heartbeat=lambda: (driver.check(), guard.pulse())
        )
        dex_controller.finish_retention_test()
    except Dex3RetentionLostError as error:
        driver.check()
        execute(recovery[0])
        open_active(f"{label}: release after failed retention lift")
        execute(recovery[2])
        execute(recovery[3])
        raise TabletopTaskRejected(f"{label} lost retention: {error}") from error
    atomic_write_json(plan_directory / "retention_evidence.json", evidence.to_dict())
    for phase in ("payload_lift", "payload_transfer", "placement_lower", "placement_contact"):
        execute(routes[phase])
    open_active(f"{label}: release at destination")
    execute(routes["placement_retreat"])
    execute(routes["return_to_clearance"])
    return {
        "arm": arm,
        "plan_sha256": plan.content_sha256,
        "selected_candidate_id": plan.selected_candidate_id,
        "grasp_close": close.to_dict(),
        "retention_evidence": evidence.to_dict(),
        "retention_validation_sha256": retention.content_sha256,
    }


def _return_supported_arms(
    *,
    synchronized,
    driver,
    model: URDFModel,
    control_config,
    left_escape: SupportedEscapePlan | None,
    right_escape: SupportedEscapePlan | None,
    left_at_clearance: bool,
    right_at_clearance: bool,
) -> tuple[bool, bool]:
    """Return in reverse escape order: right first, then left."""

    for arm, escape, should_return in (
        ("right", right_escape, right_at_clearance),
        ("left", left_escape, left_at_clearance),
    ):
        if not should_return:
            continue
        if escape is None:
            raise RuntimeError(f"{arm} is at clearance without a frozen supported return")
        reference = synchronized.observe_state()
        executable = _install_plan_at_current_boundary(
            synchronized,
            arm=arm,
            trajectories=(escape.inbound,),
            plan_sha256=escape.content_sha256,
            validated_reference_state=reference,
            model=model,
        )
        _execute_trajectory(
            synchronized,
            driver,
            executable[0],
            plan_sha256=escape.content_sha256,
            control_config=control_config,
        )
        if arm == "right":
            right_at_clearance = False
        else:
            left_at_clearance = False
    return left_at_clearance, right_at_clearance


def run_stack(args) -> int:
    """Move the 60 mm cube, reobserve, then stack the 40 mm cube on it."""

    if args.confirm != MOTION_ACK:
        raise ValueError(f"--confirm must equal exactly: {MOTION_ACK}")
    cube40_profile = load_tabletop_object_profile("cube40-r3")
    cube60_profile = load_tabletop_object_profile("cube60-r3")
    hardware = load_hardware(args.hardware_config)
    if {
        str(hardware["robot"]["calibration_arm"]),
        str(hardware["control"]["calibration_arm"]),
    } != {"left"}:
        raise ValueError("the fixed stack runtime starts from the commissioned left-arm config")
    quality = QualityThresholds.from_yaml(args.quality_config)
    bundle = CalibrationBundle.load(args.calibration_bundle)
    task_config = load_task_config(args.task_config)
    model = URDFModel(resolve_hardware_path(args.hardware_config, hardware["robot"]["urdf"]))
    expected_camera = camera_info_from_hardware(hardware)
    recording, _pairing = recording_configs(args.hardware_config)
    control_config, rate_hz = executor_config(args.hardware_config)
    control_config = replace(control_config, require_motion_endpoint_tolerance=False)
    velocity = _resolve_task_velocity(
        float(task_config["motion"]["maximum_arm_velocity_rad_s"]),
        args.maximum_arm_velocity_rad_s,
        control_config.maximum_joint_velocity_rad_s,
    )
    run_directory = (args.output_root / _run_id()).resolve()
    if run_directory.exists():
        raise FileExistsError(f"stack run already exists: {run_directory}")
    run_directory.mkdir(parents=True)
    frozen_files = {
        "hardware": (args.hardware_config, args.hardware_config.read_bytes()),
        "calibration": (args.calibration_bundle, args.calibration_bundle.read_bytes()),
        "quality": (args.quality_config, args.quality_config.read_bytes()),
        "task": (args.task_config, args.task_config.read_bytes()),
        "cube40_profile": (cube40_profile.config_path, cube40_profile.config_path.read_bytes()),
        "cube60_profile": (cube60_profile.config_path, cube60_profile.config_path.read_bytes()),
        "cube40_detector": (
            cube40_profile.detector_config_path,
            cube40_profile.detector_config_path.read_bytes(),
        ),
        "cube60_detector": (
            cube60_profile.detector_config_path,
            cube60_profile.detector_config_path.read_bytes(),
        ),
        "cube40_grasps": (
            cube40_profile.direct_grasp_shortlist_path,
            cube40_profile.direct_grasp_shortlist_path.read_bytes(),
        ),
        "cube60_grasps": (
            cube60_profile.direct_grasp_shortlist_path,
            cube60_profile.direct_grasp_shortlist_path.read_bytes(),
        ),
    }
    cube40_detector = CorrespondenceDetector(
        cube40_profile.detector_config_path,
        preprocess=False,
    )
    cube60_detector = CorrespondenceDetector(
        cube60_profile.detector_config_path,
        preprocess=False,
    )
    empty_pose_set = PoseSet(
        robot_model=model.name,
        mode_machine=5,
        urdf_sha256=model.sha256,
        calibration_arm="left",
    )
    status: dict[str, object] = {
        "status": "started",
        "commands_robot": False,
        "task": "cube60_to_table_then_cube40_on_cube60",
        "maximum_arm_velocity_rad_s": velocity,
    }
    frame_sets: dict[str, tuple[ROSImageFrame, ...]] = {}
    primary_error: BaseException | None = None
    camera = observer = dex_observer = transport = dex_controller = None
    node = rclpy_module = raw_recorder = guard = synchronized = driver = planner = None
    left_escape = right_escape = None
    left_at_clearance = right_at_clearance = False
    initial_left = initial_right = None
    measured_open_left = measured_open_right = None
    command_lock = CommandOwnerLock(args.lock_file)
    command_lock.acquire()
    try:
        try:
            import rclpy
        except ImportError as error:
            raise RuntimeError("rclpy unavailable; use ./tools/g1_tabletop_hardware.sh") from error
        rclpy_module = rclpy
        rclpy.init(args=None)
        node = rclpy.create_node("g1_dex3_two_cube_stack")
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

        def receive_lowstate(sample) -> None:
            states.add(sample)

        observer = UnitreeLowStateObserver(transport_cfg, on_sample=receive_lowstate)
        _wait_for_state(observer)
        hand_cfg = dex3_config(
            args.hardware_config,
            interface=args.network_interface,
            domain_id=args.domain_id,
        )
        dex_observer = UnitreeDex3StateObserver(hand_cfg, initialize_factory=False)
        hands = _wait_for_hands(dex_observer)
        activation = _wait_for_activation(observer, states, empty_pose_set, recording)
        frame_sets["preflight"] = _collect_frames(
            rclpy,
            node,
            camera,
            count=args.observation_frames,
            timeout_s=10.0,
        )
        if frame_sets["preflight"][-1].camera_info.profile_sha256 != (
            expected_camera.profile_sha256
        ):
            raise ValueError("live camera profile differs from hardware configuration")
        _observe_pair(
            frame_sets["preflight"],
            expected_camera=expected_camera,
            cube40_detector=cube40_detector,
            cube60_detector=cube60_detector,
            snapshot=_snapshot(activation.reference_state, hands),
            quality=quality,
        )
        print(
            "READ-ONLY STACK PREFLIGHT PASSED — both uniquely tagged cubes are visible "
            "on the bare table and no command publisher exists",
            flush=True,
        )
        planner = PersistentTabletopPlanner(
            executable=ROOT / ".venv-planner/bin/g1-curobo-worker",
            log_path=run_directory / "planner.log",
        )
        planner.start()
        _wait_for_space_with_preview(
            rclpy,
            node,
            camera,
            arm="left",
            no_window=args.no_window,
        )
        for label, (path, content) in frozen_files.items():
            if path.read_bytes() != content:
                raise RuntimeError(f"{label} changed after read-only preflight")
        raw_recorder = RawEpisodeRecorder(
            run_directory / "raw_episode",
            repository=ROOT,
            topics=tabletop_raw_topics(record_camera=not args.skip_camera_recording),
        )
        raw_recorder.start()
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
        driver = ExecutorControlDriver(
            synchronized,
            rate_hz=rate_hz,
            safety_heartbeat=guard.pulse,
        )
        guard.start()
        held_hands = dex_controller.acquire_measured_hold(safety_heartbeat=guard.pulse)
        initial_left = held_hands.left.position.copy()
        initial_right = held_hands.right.position.copy()
        driver.safety_heartbeat = _finger_heartbeat(guard, dex_controller)
        driver.start()
        synchronized.acquire(operator_confirmed=True)
        from g1_dex3_tabletop.hardware_tabletop import _wait_ready

        _wait_ready(
            synchronized,
            driver,
            timeout_s=control_config.acquisition_ramp_s + 5.0,
            label="stack ownership",
        )
        print(
            "CONTROL ACQUIRED — both arms are held at the exact measured state; "
            "planning the left supported escape",
            flush=True,
        )

        def build_request(arm: str, observation, profile) -> TabletopTaskRequest:
            return build_tabletop_request(
                arm=arm,
                observation=observation,
                calibration_bundle=bundle,
                calibration_bundle_path=args.calibration_bundle,
                grasp_shortlist_path=profile.direct_grasp_shortlist_path,
                task_config_path=args.task_config,
                object_dimensions_m=profile.dimensions_m,
                maximum_arm_velocity_rad_s=velocity,
            )

        frame_sets["loaded"] = _collect_frames(
            rclpy,
            node,
            camera,
            count=args.observation_frames,
            timeout_s=10.0,
            control_check=driver.check,
        )
        loaded_state = synchronized.observe_state()
        loaded_hands = dex_controller.observer.observe()
        loaded40, loaded60 = _observe_pair(
            frame_sets["loaded"],
            expected_camera=expected_camera,
            cube40_detector=cube40_detector,
            cube60_detector=cube60_detector,
            snapshot=_snapshot(loaded_state, loaded_hands),
            quality=quality,
        )
        left_request = _with_other_cube(
            build_request("left", loaded40, cube40_profile),
            other_id="cube60",
            other_observation=loaded60,
            other_dimensions_m=cube60_profile.dimensions_m,
        )
        left_request.write_json(run_directory / "left_escape_request.json")
        planner.request(
            "plan-supported-escape",
            request_path=run_directory / "left_escape_request.json",
            output_path=run_directory / "left_escape.json",
            control_check=driver.check,
        )
        left_escape = SupportedEscapePlan.from_json(run_directory / "left_escape.json")
        left_pose_set = pose_set_from_trajectories(
            arm="left",
            trajectories=(left_escape.outbound, left_escape.inbound),
            reference_full_q=loaded_state.position,
            robot_model=model.name,
            urdf_sha256=model.sha256,
            source="NVlabs/curobo_stack_left_supported_escape",
        )
        synchronized.install_validated_plan(
            pose_set=left_pose_set,
            approved_validation_report_sha256=left_escape.content_sha256,
            validated_reference_state=loaded_state,
        )
        _execute_trajectory(
            synchronized,
            driver,
            left_escape.outbound,
            plan_sha256=left_escape.content_sha256,
            control_config=control_config,
        )
        left_at_clearance = True

        frame_sets["left_clearance"] = _collect_frames(
            rclpy,
            node,
            camera,
            count=args.observation_frames,
            timeout_s=10.0,
            control_check=driver.check,
        )
        left_clearance_state = synchronized.observe_state()
        left_clearance_hands = dex_controller.observer.observe()
        left_q29 = np.asarray(left_clearance_state.position, dtype=np.float64).copy()
        left_q29[np.asarray(arm_indices("left"))] = np.asarray(
            left_escape.outbound.command_q_rad[-1]
        )
        left_boundary_snapshot = RobotSnapshot(
            measured_q29_rad=tuple(left_q29),
            left_dex3_q_rad=tuple(left_clearance_hands.left.position),
            right_dex3_q_rad=tuple(left_clearance_hands.right.position),
        )
        after_left40, after_left60 = _observe_pair(
            frame_sets["left_clearance"],
            expected_camera=expected_camera,
            cube40_detector=cube40_detector,
            cube60_detector=cube60_detector,
            snapshot=left_boundary_snapshot,
            quality=quality,
        )
        right_request = _with_other_cube(
            build_request("right", after_left60, cube60_profile),
            other_id="cube40",
            other_observation=after_left40,
            other_dimensions_m=cube40_profile.dimensions_m,
        )
        right_request.write_json(run_directory / "right_escape_request.json")
        planner.request(
            "plan-supported-escape",
            request_path=run_directory / "right_escape_request.json",
            output_path=run_directory / "right_escape.json",
            control_check=driver.check,
        )
        right_escape = SupportedEscapePlan.from_json(run_directory / "right_escape.json")
        right_pose_set = pose_set_from_trajectories(
            arm="right",
            trajectories=(right_escape.outbound, right_escape.inbound),
            reference_full_q=left_clearance_state.position,
            robot_model=model.name,
            urdf_sha256=model.sha256,
            source="NVlabs/curobo_stack_right_supported_escape",
        )
        synchronized.switch_validated_arm_plan(
            pose_set=right_pose_set,
            approved_validation_report_sha256=right_escape.content_sha256,
            validated_reference_state=left_clearance_state,
            boundary_pose_id="__handoff__",
        )
        _execute_trajectory(
            synchronized,
            driver,
            right_escape.outbound,
            plan_sha256=right_escape.content_sha256,
            control_config=control_config,
        )
        right_at_clearance = True
        print(
            "BOTH CLEARANCES REACHED — opening both Dex3 hands and observing both cubes "
            "before any object motion",
            flush=True,
        )
        left_open, _left_close = dex3_execution_profile("left")
        right_open, _right_close = dex3_execution_profile("right")
        open_pair = _command_fingers(
            dex_controller,
            driver,
            guard,
            left=left_open,
            right=right_open,
            label="stack run-local empty-open acquisition",
        )
        measured_open_left = open_pair.left.position.copy()
        measured_open_right = open_pair.right.position.copy()
        atomic_write_json(
            run_directory / "dex3_run_local_open.json",
            {
                "left_command_q_rad": list(left_open),
                "right_command_q_rad": list(right_open),
                "left_measured_q_rad": measured_open_left.tolist(),
                "right_measured_q_rad": measured_open_right.tolist(),
            },
        )
        frame_sets["both_clearance"] = _collect_frames(
            rclpy,
            node,
            camera,
            count=args.observation_frames,
            timeout_s=10.0,
            control_check=driver.check,
        )
        clearance_state = synchronized.observe_state()
        clearance_hands = dex_controller.observer.observe()
        clearance_snapshot = _exact_clearance_snapshot(
            clearance_state,
            clearance_hands,
            left_escape=left_escape,
            right_escape=right_escape,
        )
        clearance40, clearance60 = _observe_pair(
            frame_sets["both_clearance"],
            expected_camera=expected_camera,
            cube40_detector=cube40_detector,
            cube60_detector=cube60_detector,
            snapshot=clearance_snapshot,
            quality=quality,
        )
        reference_request = build_request("left", clearance40, cube40_profile)
        base_T_camera = request_base_T_camera(reference_request, model)
        left_hand, right_hand = request_hand_positions(reference_request, model)
        assignments = stack_arm_assignments(
            cube40=clearance40,
            cube60=clearance60,
            base_T_camera=base_T_camera,
            base_left_hand_position=left_hand,
            base_right_hand_position=right_hand,
        )
        candidates = tuple(
            sorted(
                stack_placement_candidates(
                    cube40=clearance40,
                    cube60=clearance60,
                    base_T_camera=base_T_camera,
                ),
                key=lambda value: (value.cube60_displacement_m, value.candidate_id),
            )
        )
        search_results: list[dict[str, object]] = []
        selected = None
        search_root = run_directory / "feasibility_search"
        for cube60_arm, cube40_arm in assignments:
            request40 = build_request(cube40_arm, clearance40, cube40_profile)
            request60 = build_request(cube60_arm, clearance60, cube60_profile)
            for candidate in candidates:
                directory = search_root / f"{cube60_arm}60_{cube40_arm}40" / candidate.candidate_id
                stage1_request, stage2_request = build_stack_stage_requests(
                    cube40_request=request40,
                    cube60_request=request60,
                    candidate=candidate,
                )
                attempt = {
                    "cube60_arm": cube60_arm,
                    "cube40_arm": cube40_arm,
                    "candidate": candidate.to_dict(),
                }
                try:
                    stage1_plan = _plan_pick_place(
                        planner,
                        request=stage1_request,
                        directory=directory / "stage1",
                        driver=driver,
                    )
                    stage2_plan = _plan_pick_place(
                        planner,
                        request=stage2_request,
                        directory=directory / "stage2_nominal",
                        driver=driver,
                    )
                except (PlannerRequestRejected, RuntimeError, ValueError) as error:
                    driver.check()
                    attempt.update({"passed": False, "reason": str(error)})
                    search_results.append(attempt)
                    continue
                attempt.update(
                    {
                        "passed": True,
                        "stage1_plan_sha256": stage1_plan.content_sha256,
                        "stage2_nominal_plan_sha256": stage2_plan.content_sha256,
                    }
                )
                search_results.append(attempt)
                selected = (
                    candidate,
                    cube60_arm,
                    cube40_arm,
                    stage1_request,
                    stage1_plan,
                    directory,
                )
                break
            if selected is not None:
                break
        atomic_write_json(
            run_directory / "feasibility_search.json",
            {
                "attempts": search_results,
                "selected": None if selected is None else selected[0].to_dict(),
            },
        )
        if selected is None:
            raise TabletopTaskRejected(
                "no placement candidate and opposite-arm assignment produced both complete plans"
            )
        candidate, cube60_arm, cube40_arm, stage1_request, stage1_plan, selected_dir = selected
        print(
            "STACK PLAN SELECTED — "
            f"60 mm arm={cube60_arm}, 40 mm arm={cube40_arm}, "
            f"placement={candidate.candidate_id}; executing only the 60 mm transfer now",
            flush=True,
        )
        stage1_result = _execute_pick_place(
            label="stage 1: 60 mm cube",
            request=stage1_request,
            plan=stage1_plan,
            plan_directory=selected_dir / "stage1",
            synchronized=synchronized,
            driver=driver,
            planner=planner,
            dex_controller=dex_controller,
            guard=guard,
            model=model,
            control_config=control_config,
            measured_open_left=measured_open_left,
            measured_open_right=measured_open_right,
        )
        frame_sets["after_stage1"] = _collect_frames(
            rclpy,
            node,
            camera,
            count=args.observation_frames,
            timeout_s=10.0,
            control_check=driver.check,
        )
        stage1_state = synchronized.observe_state()
        stage1_hands = dex_controller.observer.observe()
        stage1_snapshot = _exact_clearance_snapshot(
            stage1_state,
            stage1_hands,
            left_escape=left_escape,
            right_escape=right_escape,
        )
        actual40, actual60 = _observe_pair(
            frame_sets["after_stage1"],
            expected_camera=expected_camera,
            cube40_detector=cube40_detector,
            cube60_detector=cube60_detector,
            snapshot=stage1_snapshot,
            quality=quality,
        )
        actual40_request = build_request(cube40_arm, actual40, cube40_profile)
        actual_base_T_camera = request_base_T_camera(actual40_request, model)
        final_stage2_request = build_observed_stack_second_stage_request(
            cube40_request=actual40_request,
            placed_cube60=actual60,
            base_T_camera=actual_base_T_camera,
        )
        try:
            final_stage2_plan = _plan_pick_place(
                planner,
                request=final_stage2_request,
                directory=selected_dir / "stage2_actual",
                driver=driver,
            )
        except (PlannerRequestRejected, RuntimeError, ValueError) as error:
            driver.check()
            raise TabletopTaskRejected(
                f"the actual 60 mm placement left no complete 40-on-60 plan: {error}"
            ) from error
        print(
            "STAGE-ONE RESULT REOBSERVED — the final 40-on-60 plan uses the actual "
            "placed 60 mm cube pose; beginning the 40 mm transfer",
            flush=True,
        )
        stage2_result = _execute_pick_place(
            label="stage 2: 40 mm cube onto 60 mm cube",
            request=final_stage2_request,
            plan=final_stage2_plan,
            plan_directory=selected_dir / "stage2_actual",
            synchronized=synchronized,
            driver=driver,
            planner=planner,
            dex_controller=dex_controller,
            guard=guard,
            model=model,
            control_config=control_config,
            measured_open_left=measured_open_left,
            measured_open_right=measured_open_right,
        )
        _command_fingers(
            dex_controller,
            driver,
            guard,
            left=initial_left,
            right=initial_right,
            label="restore initial finger postures after stack",
        )
        left_at_clearance, right_at_clearance = _return_supported_arms(
            synchronized=synchronized,
            driver=driver,
            model=model,
            control_config=control_config,
            left_escape=left_escape,
            right_escape=right_escape,
            left_at_clearance=left_at_clearance,
            right_at_clearance=right_at_clearance,
        )
        _restore_seated_control(
            driver=driver,
            dex_controller=dex_controller,
            guard=guard,
            synchronized=synchronized,
        )
        status = {
            "status": "completed",
            "commands_robot": True,
            "task": "cube60_to_table_then_cube40_on_cube60",
            "placement_candidate": candidate.to_dict(),
            "cube60_arm": cube60_arm,
            "cube40_arm": cube40_arm,
            "stage1": stage1_result,
            "stage2": stage2_result,
            "terminal_action": guard.terminal_action,
            "maximum_arm_velocity_rad_s": velocity,
        }
        print(
            "TWO-CUBE STACK PASSED — the 60 mm cube was placed and reobserved, the "
            "40 mm cube was placed on it, both arms returned to their supported starts, "
            "and seated FSM 3 was restored",
            flush=True,
        )
    except TabletopTaskRejected as rejection:
        try:
            if (
                driver is not None
                and dex_controller is not None
                and guard is not None
                and initial_left is not None
                and initial_right is not None
            ):
                _command_fingers(
                    dex_controller,
                    driver,
                    guard,
                    left=initial_left,
                    right=initial_right,
                    label="restore initial finger postures after stack rejection",
                )
            if synchronized is not None and driver is not None:
                left_at_clearance, right_at_clearance = _return_supported_arms(
                    synchronized=synchronized,
                    driver=driver,
                    model=model,
                    control_config=control_config,
                    left_escape=left_escape,
                    right_escape=right_escape,
                    left_at_clearance=left_at_clearance,
                    right_at_clearance=right_at_clearance,
                )
            _restore_seated_control(
                driver=driver,
                dex_controller=dex_controller,
                guard=guard,
                synchronized=synchronized,
            )
        except BaseException as error:
            primary_error = error
            raise RuntimeError(
                f"stack rejection recovery failed after {rejection}: {error}"
            ) from error
        status = {
            "status": "task_rejected",
            "commands_robot": bool(transport is not None and transport.command_count),
            "task": "cube60_to_table_then_cube40_on_cube60",
            "reason": str(rejection),
            "terminal_action": guard.terminal_action,
            "supported_return_completed": not left_at_clearance and not right_at_clearance,
        }
        print(
            "STACK TASK REJECTED — both arms returned through their frozen supported "
            f"routes and seated FSM 3 was restored. Reason: {rejection}",
            flush=True,
        )
    except BaseException as error:
        primary_error = error
        status = {
            "status": "failed",
            "commands_robot": bool(transport is not None and transport.command_count),
            "task": "cube60_to_table_then_cube40_on_cube60",
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
                    guard.restore_zero_torque("stack task failed or was interrupted")
                synchronized.confirm_external_takeover(
                    "PC2 verified AI zero-torque takeover after stack failure"
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
        except BaseException as error:  # noqa: BLE001
            cleanup_errors.append(f"ROS teardown: {error}")
        command_lock.release()
        if raw_recorder is not None and (raw_recorder.started or raw_recorder.summary is not None):
            try:
                status["recording"] = raw_recorder.stop()
            except BaseException as error:  # noqa: BLE001
                cleanup_errors.append(f"raw episode recorder: {error}")
        status["cleanup_errors"] = cleanup_errors
        try:
            for label, frames in frame_sets.items():
                _save_frames(run_directory / label, frames)
            atomic_write_json(run_directory / "status.json", status)
        except BaseException as error:
            if primary_error is None:
                raise
            print(f"warning: failed to write complete stack artifacts: {error}", file=sys.stderr)
    print(json.dumps({"run": str(run_directory), **status}, indent=2, sort_keys=True))
    return 0
