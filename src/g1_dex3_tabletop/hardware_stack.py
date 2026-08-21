"""Fixed one-pick two-cube stack hardware workflow.

One uniquely tagged 60 mm cube is picked and placed directly on the other.
It is not a task language.
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
    build_direct_stack_request,
    request_base_T_camera,
    request_hand_positions,
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


def _active_clearance_snapshot(
    state,
    hands,
    *,
    arm: str,
    escape: SupportedEscapePlan,
    inactive_command_q_rad,
) -> RobotSnapshot:
    endpoint = escape.outbound.command_q_rad[-1]
    return _dual_arm_command_snapshot(
        state,
        hands,
        left_command_q_rad=endpoint if arm == "left" else inactive_command_q_rad,
        right_command_q_rad=endpoint if arm == "right" else inactive_command_q_rad,
    )


def _dual_arm_command_snapshot(
    state,
    hands,
    *,
    left_command_q_rad,
    right_command_q_rad,
) -> RobotSnapshot:
    """Bind planning to the exact held commands, not loaded tracking offsets."""

    left = np.asarray(left_command_q_rad, dtype=np.float64).reshape(-1)
    right = np.asarray(right_command_q_rad, dtype=np.float64).reshape(-1)
    if left.shape != (7,) or right.shape != (7,):
        raise ValueError("dual-arm command snapshot requires two seven-joint commands")
    if not np.all(np.isfinite(left)) or not np.all(np.isfinite(right)):
        raise ValueError("dual-arm command snapshot contains NaN or infinity")
    q29 = np.asarray(state.position, dtype=np.float64).copy()
    q29[np.asarray(arm_indices("left"))] = left
    q29[np.asarray(arm_indices("right"))] = right
    return RobotSnapshot(
        measured_q29_rad=tuple(q29),
        left_dex3_q_rad=tuple(hands.left.position),
        right_dex3_q_rad=tuple(hands.right.position),
    )


def _observe_pair(
    frames: tuple[ROSImageFrame, ...],
    *,
    expected_camera,
    upper_detector,
    bottom_detector,
    snapshot: RobotSnapshot,
    quality: QualityThresholds,
) -> tuple[TabletopObservation, TabletopObservation]:
    return observe_resting_cube_pair(
        [item.image_bgr for item in frames],
        camera_info=expected_camera,
        first_detector=upper_detector,
        second_detector=bottom_detector,
        snapshot=snapshot,
        minimum_tag_short_side_px=quality.minimum_tag_short_side_px,
        maximum_reprojection_error_px=quality.pnp_reject_reprojection_px,
    )


def _nearest_cube_move(
    *,
    reference_request: TabletopTaskRequest,
    upper_observation: TabletopObservation,
    bottom_observation: TabletopObservation,
    model: URDFModel,
) -> dict[str, object]:
    """Choose the globally nearest cube/arm pair from the supported start."""

    base_T_camera = request_base_T_camera(reference_request, model)
    left_hand, right_hand = request_hand_positions(reference_request, model)
    hand_positions = {"left": left_hand, "right": right_hand}
    observations = {
        "secondary": upper_observation,
        "primary": bottom_observation,
    }
    choices: list[dict[str, object]] = []
    for moving_cube, observation in observations.items():
        cube_position = (
            base_T_camera @ np.asarray(observation.camera_T_object, dtype=np.float64)
        )[:3, 3]
        arm = min(
            ("left", "right"),
            key=lambda side: float(np.linalg.norm(hand_positions[side] - cube_position)),
        )
        choices.append(
            {
                "moving_cube": moving_cube,
                "support_cube": "primary" if moving_cube == "secondary" else "secondary",
                "arm": arm,
                "source_hand_distance_m": float(
                    np.linalg.norm(hand_positions[arm] - cube_position)
                ),
            }
        )
    return min(
        choices,
        key=lambda value: (float(value["source_hand_distance_m"]), str(value["moving_cube"])),
    )


class PickPlaceGraspRejected(TabletopTaskRejected):
    """A physically rejected grasp after the arm has recovered to clearance."""

    def __init__(self, message: str, *, candidate_id: str) -> None:
        super().__init__(message)
        self.candidate_id = str(candidate_id)


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


def _find_direct_stack_plan(
    *,
    planner,
    driver,
    options: tuple[tuple[dict[str, object], TabletopPickPlaceRequest], ...],
    directory: Path,
):
    """Return the first complete direct cube-on-cube plan."""

    attempts: list[dict[str, object]] = []
    endpoint_viable: list[
        tuple[
            dict[str, object],
            TabletopPickPlaceRequest,
            Path,
            dict[str, object],
        ]
    ] = []
    for metadata, request in options:
        option_directory = (
            directory
            / f"move_{metadata['moving_cube']}_with_{metadata['arm']}_arm"
            / f"yaw_quarter_turns_{metadata['yaw_quarter_turns']}"
        )
        option_directory.mkdir(parents=True, exist_ok=True)
        request_path = option_directory / "request.json"
        request.write_json(request_path)
        attempt = dict(metadata)
        try:
            event = planner.request_payload(
                "analyze-pick-place-endpoints",
                payload={"request": str(request_path.resolve())},
                control_check=driver.check,
                timeout_s=60.0,
            )
            endpoint = event["payload"]
            if endpoint.get("request_sha256") != request.content_sha256:
                raise RuntimeError("endpoint analysis belongs to another request")
            atomic_write_json(option_directory / "endpoint_feasibility.json", endpoint)
        except (PlannerRequestRejected, RuntimeError, ValueError) as error:
            driver.check()
            attempt.update({"passed": False, "reason": str(error)})
            attempts.append(attempt)
            continue
        attempt["endpoint_feasibility"] = endpoint
        if int(endpoint["common_candidate_count"]) == 0:
            attempt.update(
                {
                    "passed": False,
                    "reason": "no common strict endpoint-valid grasp candidate",
                }
            )
            attempts.append(attempt)
            continue
        attempts.append(attempt)
        endpoint_viable.append((metadata, request, option_directory, attempt))

    for metadata, request, option_directory, attempt in endpoint_viable:
        try:
            plan = _plan_pick_place(
                planner,
                request=request,
                directory=option_directory,
                driver=driver,
            )
        except (PlannerRequestRejected, RuntimeError, ValueError) as error:
            driver.check()
            attempt.update({"passed": False, "reason": str(error)})
            continue
        attempt.update(
            {
                "passed": True,
                "request_sha256": request.content_sha256,
                "plan_sha256": plan.content_sha256,
            }
        )
        return (metadata, request, plan, option_directory), attempts
    return None, attempts


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
    empty_close_reference_q_rad,
    minimum_opposed_shortfall_rad: float,
) -> dict[str, object]:
    """Execute one already planned fixed pick/place operation."""

    arm = plan.arm
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
            left=left_open_target if arm == "left" else measured_open_left,
            right=right_open_target if arm == "right" else measured_open_right,
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
            empty_close_reference_q_rad=empty_close_reference_q_rad,
            minimum_opposed_shortfall_rad=minimum_opposed_shortfall_rad,
            label=f"{label}: descriptor-defined {arm}-hand close",
        )
    except Dex3GraspNotAcquiredError as error:
        driver.check()
        recover_from_grasp()
        raise PickPlaceGraspRejected(
            f"{label} did not acquire the cube: {error}",
            candidate_id=plan.selected_candidate_id,
        ) from error
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
    except (PlannerRequestRejected, Dex3RetentionLostError) as error:
        driver.check()
        recover_from_grasp()
        raise PickPlaceGraspRejected(
            f"{label} measured close did not validate: {error}",
            candidate_id=plan.selected_candidate_id,
        ) from error
    except RuntimeError as error:
        driver.check()
        recover_from_grasp()
        raise TabletopTaskRejected(
            f"{label} retention-validation infrastructure failed: {error}"
        ) from error
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
        raise PickPlaceGraspRejected(
            f"{label} lost retention: {error}",
            candidate_id=plan.selected_candidate_id,
        ) from error
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
    """Pick either 60 mm cube and place it directly on the other."""

    if args.confirm != MOTION_ACK:
        raise ValueError(f"--confirm must equal exactly: {MOTION_ACK}")
    if args.grasp_retries < 0:
        raise ValueError("--grasp-retries must be non-negative")
    upper_profile = load_tabletop_object_profile("cube60-r3-secondary")
    bottom_profile = load_tabletop_object_profile("cube60-r3")
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
        "upper_profile": (upper_profile.config_path, upper_profile.config_path.read_bytes()),
        "bottom_profile": (bottom_profile.config_path, bottom_profile.config_path.read_bytes()),
        "upper_detector": (
            upper_profile.detector_config_path,
            upper_profile.detector_config_path.read_bytes(),
        ),
        "bottom_detector": (
            bottom_profile.detector_config_path,
            bottom_profile.detector_config_path.read_bytes(),
        ),
        "upper_grasps": (
            upper_profile.direct_grasp_shortlist_path,
            upper_profile.direct_grasp_shortlist_path.read_bytes(),
        ),
        "bottom_grasps": (
            bottom_profile.direct_grasp_shortlist_path,
            bottom_profile.direct_grasp_shortlist_path.read_bytes(),
        ),
    }
    upper_detector = CorrespondenceDetector(
        upper_profile.detector_config_path,
        preprocess=False,
    )
    bottom_detector = CorrespondenceDetector(
        bottom_profile.detector_config_path,
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
        "task": "one_pick_direct_cube60_on_cube60",
        "maximum_arm_velocity_rad_s": velocity,
        "requested_pregrasp_distance_m": args.pregrasp_distance_m,
        "grasp_retries": args.grasp_retries,
    }
    retry_events: list[dict[str, object]] = []
    frame_sets: dict[str, tuple[ROSImageFrame, ...]] = {}
    primary_error: BaseException | None = None
    camera = observer = dex_observer = transport = dex_controller = None
    node = rclpy_module = raw_recorder = guard = synchronized = driver = planner = None
    runtime_warmup = None
    left_escape = right_escape = None
    left_at_clearance = right_at_clearance = False
    initial_left = initial_right = None
    measured_open_left = measured_open_right = None
    command_lock = CommandOwnerLock(args.lock_file)
    command_lock.acquire()
    try:
        planner = PersistentTabletopPlanner(
            executable=ROOT / ".venv-planner/bin/g1-curobo-worker",
            log_path=run_directory / "planner.log",
        )
        planner.launch()
        print(
            "PLANNER STARTED — isolated CUDA initialization is running concurrently "
            "with the read-only two-cube preflight; no command publisher exists",
            flush=True,
        )
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
        preflight_snapshot = _snapshot(activation.reference_state, hands)
        preflight_upper, preflight_bottom = _observe_pair(
            frame_sets["preflight"],
            expected_camera=expected_camera,
            upper_detector=upper_detector,
            bottom_detector=bottom_detector,
            snapshot=preflight_snapshot,
            quality=quality,
        )
        print(
            "READ-ONLY STACK PREFLIGHT PASSED — both uniquely tagged cubes are visible "
            "on the bare table and no command publisher exists",
            flush=True,
        )
        preflight_reference = build_tabletop_request(
            arm="left",
            observation=preflight_upper,
            calibration_bundle=bundle,
            calibration_bundle_path=args.calibration_bundle,
            grasp_shortlist_path=upper_profile.direct_grasp_shortlist_path,
            task_config_path=args.task_config,
            object_dimensions_m=upper_profile.dimensions_m,
            maximum_arm_velocity_rad_s=velocity,
            pregrasp_distance_m=args.pregrasp_distance_m,
        )
        status["pregrasp_distance_m"] = preflight_reference.pregrasp_distance_m
        selected_move = _nearest_cube_move(
            reference_request=preflight_reference,
            upper_observation=preflight_upper,
            bottom_observation=preflight_bottom,
            model=model,
        )
        selected_arm = str(selected_move["arm"])
        moving_cube = str(selected_move["moving_cube"])
        # Grasp validation depends on a physically measured empty-close
        # reference for the selected hand. Resolve it before SPACE, ownership,
        # or any arm motion—not after a complete task plan has been found.
        selected_empty_close, selected_minimum_shortfall = dex3_empty_close_reference(
            selected_arm
        )
        if moving_cube == "secondary":
            preflight_moving = preflight_upper
            preflight_moving_profile = upper_profile
            preflight_support = preflight_bottom
            preflight_support_profile = bottom_profile
        else:
            preflight_moving = preflight_bottom
            preflight_moving_profile = bottom_profile
            preflight_support = preflight_upper
            preflight_support_profile = upper_profile
        preflight_warmup_request = _with_other_cube(
            build_tabletop_request(
                arm=selected_arm,
                observation=preflight_moving,
                calibration_bundle=bundle,
                calibration_bundle_path=args.calibration_bundle,
                grasp_shortlist_path=preflight_moving_profile.direct_grasp_shortlist_path,
                task_config_path=args.task_config,
                object_dimensions_m=preflight_moving_profile.dimensions_m,
                maximum_arm_velocity_rad_s=velocity,
                pregrasp_distance_m=args.pregrasp_distance_m,
            ),
            other_id="support_cube",
            other_observation=preflight_support,
            other_dimensions_m=preflight_support_profile.dimensions_m,
        )
        atomic_write_json(run_directory / "preflight_selection.json", selected_move)
        preflight_warmup_path = run_directory / "preflight_warmup_request.json"
        preflight_warmup_request.write_json(preflight_warmup_path)
        empty_pose_set = PoseSet(
            robot_model=model.name,
            mode_machine=5,
            urdf_sha256=model.sha256,
            calibration_arm=selected_arm,
        )
        # The read-only preflight necessarily starts from the commissioned
        # left-arm configuration. Once it selects the active arm, bind the
        # actual ownership handoff to the same arm as its pose set. This is the
        # same activation invariant used by the single-cube workflow.
        recording = replace(recording, calibration_arm=selected_arm)
        runtime_warmup = planner.begin_payload_request(
            "prewarm-tabletop-runtime",
            payload={
                "request": str(preflight_warmup_path.resolve()),
                "moving_grasp_mpc": False,
            },
        )
        print(
            "ACTIVE-ARM RUNTIME WARMUP QUEUED — "
            f"the {selected_arm} arm is nearest to the {moving_cube} cube; only its "
            "open-hand and attached-60-mm MotionGen models are warming during the "
            "read-only preview. SPACE still authorizes only later command-publisher creation",
            flush=True,
        )
        _wait_for_space_with_preview(
            rclpy,
            node,
            camera,
            arm=selected_arm,
            no_window=args.no_window,
        )
        print(
            "SPACE RECORDED — no command publisher exists; completing any remaining "
            "background active-arm warmup before ownership",
            flush=True,
        )
        warmup_event = planner.finish_request(runtime_warmup, timeout_s=300.0)
        runtime_warmup = None
        atomic_write_json(
            run_directory / "runtime_warmup.json",
            warmup_event["payload"],
        )
        print(
            f"ACTIVE-ARM RUNTIME WARMUP READY — reusable {selected_arm}-arm MotionGen "
            "topologies completed before robot ownership",
            flush=True,
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
            f"planning only the selected {selected_arm}-arm supported escape",
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
                pregrasp_distance_m=args.pregrasp_distance_m,
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
        loaded_upper, loaded_bottom = _observe_pair(
            frame_sets["loaded"],
            expected_camera=expected_camera,
            upper_detector=upper_detector,
            bottom_detector=bottom_detector,
            snapshot=_snapshot(loaded_state, loaded_hands),
            quality=quality,
        )
        if moving_cube == "secondary":
            loaded_moving = loaded_upper
            loaded_moving_profile = upper_profile
            loaded_support = loaded_bottom
            loaded_support_profile = bottom_profile
        else:
            loaded_moving = loaded_bottom
            loaded_moving_profile = bottom_profile
            loaded_support = loaded_upper
            loaded_support_profile = upper_profile
        active_request = _with_other_cube(
            build_request(selected_arm, loaded_moving, loaded_moving_profile),
            other_id="support_cube",
            other_observation=loaded_support,
            other_dimensions_m=loaded_support_profile.dimensions_m,
        )
        active_request.write_json(run_directory / "supported_escape_request.json")
        planner.request(
            "plan-supported-escape",
            request_path=run_directory / "supported_escape_request.json",
            output_path=run_directory / "supported_escape.json",
            control_check=driver.check,
        )
        active_escape = SupportedEscapePlan.from_json(run_directory / "supported_escape.json")
        if selected_arm == "left":
            left_escape = active_escape
            inactive_command_q_rad = loaded_state.right_q.copy()
        else:
            right_escape = active_escape
            inactive_command_q_rad = loaded_state.left_q.copy()
        active_pose_set = pose_set_from_trajectories(
            arm=selected_arm,
            trajectories=(active_escape.outbound, active_escape.inbound),
            reference_full_q=loaded_state.position,
            robot_model=model.name,
            urdf_sha256=model.sha256,
            source=f"NVlabs/curobo_stack_{selected_arm}_supported_escape",
        )
        synchronized.install_validated_plan(
            pose_set=active_pose_set,
            approved_validation_report_sha256=active_escape.content_sha256,
            validated_reference_state=loaded_state,
        )
        _execute_trajectory(
            synchronized,
            driver,
            active_escape.outbound,
            plan_sha256=active_escape.content_sha256,
            control_config=control_config,
        )
        left_at_clearance = selected_arm == "left"
        right_at_clearance = selected_arm == "right"
        print(
            f"{selected_arm.upper()} CLEARANCE REACHED — opening only the selected Dex3 "
            "hand; the unused arm and hand remain at their supported initial state",
            flush=True,
        )
        active_open, _active_close = dex3_execution_profile(selected_arm)
        open_pair = _command_fingers(
            dex_controller,
            driver,
            guard,
            left=active_open if selected_arm == "left" else initial_left,
            right=active_open if selected_arm == "right" else initial_right,
            label=f"{selected_arm}-hand stack empty-open acquisition",
        )
        measured_open_left = open_pair.left.position.copy()
        measured_open_right = open_pair.right.position.copy()
        atomic_write_json(
            run_directory / "dex3_run_local_open.json",
            {
                "active_side": selected_arm,
                "left_command_q_rad": list(
                    active_open if selected_arm == "left" else initial_left
                ),
                "right_command_q_rad": list(
                    active_open if selected_arm == "right" else initial_right
                ),
                "left_measured_q_rad": measured_open_left.tolist(),
                "right_measured_q_rad": measured_open_right.tolist(),
            },
        )
        frame_sets["active_clearance"] = _collect_frames(
            rclpy,
            node,
            camera,
            count=args.observation_frames,
            timeout_s=10.0,
            control_check=driver.check,
        )
        clearance_state = synchronized.observe_state()
        clearance_hands = dex_controller.observer.observe()
        clearance_snapshot = _active_clearance_snapshot(
            clearance_state,
            clearance_hands,
            arm=selected_arm,
            escape=active_escape,
            inactive_command_q_rad=inactive_command_q_rad,
        )
        clearance_upper, clearance_bottom = _observe_pair(
            frame_sets["active_clearance"],
            expected_camera=expected_camera,
            upper_detector=upper_detector,
            bottom_detector=bottom_detector,
            snapshot=clearance_snapshot,
            quality=quality,
        )

        def direct_options(
            observed_upper: TabletopObservation,
            observed_bottom: TabletopObservation,
            *,
            excluded_candidate_ids: tuple[str, ...] = (),
        ) -> tuple[tuple[dict[str, object], TabletopPickPlaceRequest], ...]:
            reference = build_request(selected_arm, observed_upper, upper_profile)
            current_base_T_camera = request_base_T_camera(reference, model)
            left_hand, right_hand = request_hand_positions(reference, model)
            hand_positions = {"left": left_hand, "right": right_hand}
            observed = {
                "secondary": (observed_upper, upper_profile),
                "primary": (observed_bottom, bottom_profile),
            }
            cube_positions = {
                name: (current_base_T_camera @ np.asarray(value[0].camera_T_object))[:3, 3]
                for name, value in observed.items()
            }
            support_cube = "primary" if moving_cube == "secondary" else "secondary"
            moving_observation, moving_profile = observed[moving_cube]
            support_observation = observed[support_cube][0]
            distance = float(
                np.linalg.norm(hand_positions[selected_arm] - cube_positions[moving_cube])
            )
            moving_request = build_request(selected_arm, moving_observation, moving_profile)
            options = []
            for yaw_quarter_turns in (0, 1, 3, 2):
                options.append(
                    (
                        {
                            "moving_cube": moving_cube,
                            "support_cube": support_cube,
                            "arm": selected_arm,
                            "source_hand_distance_m": distance,
                            "yaw_quarter_turns": yaw_quarter_turns,
                            "yaw_is_nominal_only": True,
                        },
                        build_direct_stack_request(
                            moving_request=moving_request,
                            support_cube=support_observation,
                            base_T_camera=current_base_T_camera,
                            yaw_quarter_turns=yaw_quarter_turns,
                            excluded_candidate_ids=excluded_candidate_ids,
                        ),
                    )
                )
            return tuple(options)

        selected, search_results = _find_direct_stack_plan(
            planner=planner,
            driver=driver,
            options=direct_options(clearance_upper, clearance_bottom),
            directory=run_directory / "feasibility_search",
        )
        atomic_write_json(
            run_directory / "feasibility_search.json",
            {
                "attempts": search_results,
                "selected": None if selected is None else selected[0],
            },
        )
        if selected is None:
            raise TabletopTaskRejected(
                f"the selected {selected_arm}-arm direct cube-on-cube transfer produced "
                "no complete plan"
            )
        selected_metadata, stack_request, stack_plan, selected_dir = selected
        print(
            "DIRECT STACK PLAN SELECTED — "
            f"{selected_arm} arm picks the {moving_cube} cube and places it directly "
            f"on the {selected_metadata['support_cube']} cube; nominal yaw quarter-turns="
            f"{selected_metadata['yaw_quarter_turns']}",
            flush=True,
        )
        failed_candidates: list[str] = []
        task_attempt = 1
        while True:
            try:
                stack_result = _execute_pick_place(
                    label=f"direct stack attempt {task_attempt}: {moving_cube} 60 mm cube",
                    request=stack_request,
                    plan=stack_plan,
                    plan_directory=selected_dir,
                    synchronized=synchronized,
                    driver=driver,
                    planner=planner,
                    dex_controller=dex_controller,
                    guard=guard,
                    model=model,
                    control_config=control_config,
                    measured_open_left=measured_open_left,
                    measured_open_right=measured_open_right,
                    empty_close_reference_q_rad=selected_empty_close,
                    minimum_opposed_shortfall_rad=selected_minimum_shortfall,
                )
                break
            except PickPlaceGraspRejected as rejection:
                driver.check()
                failed_candidates.append(rejection.candidate_id)
                retry_events.append(
                    {
                        "attempt": task_attempt,
                        "candidate_id": rejection.candidate_id,
                        "reason": str(rejection),
                        "recovered_to_clearance": True,
                    }
                )
                if task_attempt > args.grasp_retries:
                    raise TabletopTaskRejected(
                        f"direct stack exhausted {args.grasp_retries} grasp retries: {rejection}"
                    ) from rejection
                task_attempt += 1
                print(
                    "DIRECT STACK GRASP REJECTED — the active arm recovered to clearance; "
                    "reobserving both cubes and replanning the same one-pick task without "
                    f"{rejection.candidate_id} (attempt {task_attempt}/"
                    f"{args.grasp_retries + 1})",
                    flush=True,
                )
                key = f"retry_{task_attempt:02d}"
                frame_sets[key] = _collect_frames(
                    rclpy,
                    node,
                    camera,
                    count=args.observation_frames,
                    timeout_s=10.0,
                    control_check=driver.check,
                )
                retry_state = synchronized.observe_state()
                retry_hands = dex_controller.observer.observe()
                retry_snapshot = _active_clearance_snapshot(
                    retry_state,
                    retry_hands,
                    arm=selected_arm,
                    escape=active_escape,
                    inactive_command_q_rad=inactive_command_q_rad,
                )
                try:
                    retry_upper, retry_bottom = _observe_pair(
                        frame_sets[key],
                        expected_camera=expected_camera,
                        upper_detector=upper_detector,
                        bottom_detector=bottom_detector,
                        snapshot=retry_snapshot,
                        quality=quality,
                    )
                    retry_selected, retry_search = _find_direct_stack_plan(
                        planner=planner,
                        driver=driver,
                        options=direct_options(
                            retry_upper,
                            retry_bottom,
                            excluded_candidate_ids=tuple(failed_candidates),
                        ),
                        directory=(run_directory / "retries" / f"attempt_{task_attempt:02d}"),
                    )
                    atomic_write_json(
                        run_directory / "retries" / f"attempt_{task_attempt:02d}_search.json",
                        {
                            "excluded_candidate_ids": failed_candidates,
                            "attempts": retry_search,
                        },
                    )
                    if retry_selected is None:
                        raise RuntimeError(
                            "no direct stack plan remained after excluding the failed grasp"
                        )
                    selected_metadata, stack_request, stack_plan, selected_dir = retry_selected
                except (PlannerRequestRejected, RuntimeError, ValueError) as error:
                    driver.check()
                    raise TabletopTaskRejected(
                        f"direct stack retry could not produce a fresh plan: {error}"
                    ) from error
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
            "task": "one_pick_direct_cube60_on_cube60",
            "selection": selected_metadata,
            "result": stack_result,
            "retry_events": retry_events,
            "terminal_action": guard.terminal_action,
            "maximum_arm_velocity_rad_s": velocity,
        }
        print(
            "DIRECT TWO-CUBE STACK PASSED — one 60 mm cube was placed directly on "
            f"the other, the {selected_arm} arm returned to its supported start, "
            "the unused arm never moved, and seated FSM 3 was restored",
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
            "task": "one_pick_direct_cube60_on_cube60",
            "reason": str(rejection),
            "retry_events": retry_events,
            "terminal_action": guard.terminal_action,
            "supported_return_completed": not left_at_clearance and not right_at_clearance,
        }
        print(
            "STACK TASK REJECTED — the selected arm returned through its frozen "
            "supported route, the unused arm remained at its start, and seated FSM 3 "
            f"was restored. Reason: {rejection}",
            flush=True,
        )
    except BaseException as error:
        primary_error = error
        status = {
            "status": "failed",
            "commands_robot": bool(transport is not None and transport.command_count),
            "task": "one_pick_direct_cube60_on_cube60",
            "error_type": type(error).__name__,
            "error": str(error),
            "retry_events": retry_events,
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
