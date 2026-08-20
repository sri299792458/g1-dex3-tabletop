"""CuRobo-only planner for a selected-Dex3 cube pick/lift/replace task."""

from __future__ import annotations

import copy
import gc
import hashlib
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from scipy.spatial.transform import Rotation

from g1_aprilcube_calibration.joint_map import arm_indices, arm_joint_names
from g1_aprilcube_calibration.transforms import invert_transform
from g1_aprilcube_calibration.transports.unitree_dex3 import (
    DEX3_MOTOR_JOINT_SUFFIXES,
)
from g1_dex3_tabletop.planning.contracts import PlannedTrajectory, RobotSnapshot
from g1_dex3_tabletop.planning.curobo_backend import (
    COLLISION_ACTIVATION_DISTANCE_M,
    FINGER_SWEEP_MAXIMUM_JOINT_STEP_RAD,
    IK_SEEDS,
    OPEN_TRANSIT_OBJECT_CLEARANCE_M,
    TRAJECTORY_INTERPOLATION_DT_S,
    CuroboKinematicCollisionChecker,
    CuroboWorldCollisionChecker,
    _joint_state_dt,
    _reverse_trajectory,
    _self_collision_pair_penetrations,
    sample_linear_joint_sweep,
)
from g1_dex3_tabletop.planning.dex3_handedness import (
    dex3_execution_profile,
)
from g1_dex3_tabletop.planning.g1_model import (
    CUROBO_COMMIT,
    attachment_link,
    build_tabletop_robot_config,
    build_tabletop_route_validation_robot_config,
    command_from_model_q,
    grasp_frame,
    model_source_hashes,
)
from g1_dex3_tabletop.tabletop_contracts import (
    PICK_PLACE_PHASE_ORDER,
    CharucoSupportedEscapeRequest,
    MovingGraspContinuationRequest,
    PickPlaceRetentionRouteValidationRequest,
    RetentionRouteValidationRequest,
    RetentionRouteValidationResult,
    SupportedEscapePlan,
    TabletopPickPlacePlan,
    TabletopPickPlaceRequest,
    TabletopPregraspPlan,
    TabletopTaskPlan,
    TabletopTaskRequest,
)
from g1_dex3_tabletop.tabletop_geometry import canonical_resting_cube_pose
from g1_dex3_tabletop.tabletop_workflow import destination_request_for_pick_place

ROOT = Path(__file__).resolve().parents[3]
WORLD_COLLISION_DISABLE_RADIUS_EPSILON_M = 1.0e-6
IK_BRANCH_DUPLICATE_TOLERANCE_RAD = 1.0e-5
PREGRASP_IK_POSITION_TOLERANCE_M = 0.005
PREGRASP_IK_ORIENTATION_TOLERANCE_RAD = 0.05
FIXED_CLOSE_COLLISION_BATCH_SAMPLES = 4096
PLANE_CLEARANCE_NUMERICAL_TOLERANCE_M = 1.0e-6


@dataclass(frozen=True, slots=True)
class _PregraspBranch:
    """One collision-valid CuRobo IK solution for one pregrasp goal."""

    candidate_local_index: int
    solver_seed_index: int
    model_q_rad: np.ndarray
    position_error_m: float
    rotation_error_rad: float


@dataclass(frozen=True, slots=True)
class _OpenBranchPlan:
    approach: PlannedTrajectory
    grasp: PlannedTrajectory
    minimum_plane_clearance_m: float
    minimum_plane_link: str
    minimum_plane_sample: int


@dataclass(frozen=True, slots=True)
class _ApproachBranchPlan:
    approach: PlannedTrajectory
    minimum_plane_clearance_m: float
    minimum_plane_link: str
    minimum_plane_sample: int
    fixed_close_sweep: _FixedCloseSweepResult


@dataclass(frozen=True, slots=True)
class _StartRelativePlaneClearance:
    minimum_m: float
    minimum_link: str
    minimum_sample: int
    boundary_m: float
    first_full_margin_sample: int
    last_full_margin_sample: int


@dataclass(frozen=True, slots=True)
class _FixedCloseSweepResult:
    sample_count: int
    minimum_plane_clearance_m: float
    minimum_plane_link: str
    minimum_plane_sample: int
    minimum_fixture_clearance_m: float | None
    minimum_fixture_link: str | None
    minimum_fixture_sample: int | None


@dataclass(frozen=True, slots=True)
class _LiftBranchPlan:
    retention_test_lift: PlannedTrajectory
    payload_lift: PlannedTrajectory
    retention_test_lift_actual_m: float
    closed_hand_minimum_plane_clearance_m: float
    payload_minimum_plane_clearance_m: float
    payload_start_plane_clearance_m: float
    attachment_sphere_count: int


@dataclass(frozen=True, slots=True)
class _MovingGraspGeometry:
    base_T_torso: np.ndarray
    base_T_detected_object: np.ndarray
    base_T_object: np.ndarray
    plane_point: np.ndarray
    down: np.ndarray


class _BranchRejected(RuntimeError):
    """An expected geometric/planning rejection of one finite IK branch."""

    def __init__(self, stage: str, reason: str):
        super().__init__(reason)
        self.stage = stage
        self.reason = reason


def prewarm_tabletop_model_resolution() -> float:
    """Pay one-time CuRobo/CUDA model startup before robot ownership.

    The live joint and finger values still produce a freshly resolved model.
    This command-free warmup only moves CUDA module loading and the first G1
    topology construction ahead of the operator approval boundary.
    """

    import torch
    from curobo.types import DeviceCfg

    if not torch.cuda.is_available():
        raise RuntimeError("CuRobo tabletop model warmup requires a CUDA device")
    started = time.monotonic()
    snapshot = RobotSnapshot((0.0,) * 29, (0.0,) * 7, (0.0,) * 7)
    robot, _reference = build_tabletop_robot_config(
        arm="left",
        snapshot=snapshot,
        joint_position_offsets_rad={},
        active_finger_q_rad=dex3_execution_profile("left")[0],
    )
    checker = CuroboKinematicCollisionChecker(
        robot=robot,
        device_cfg=DeviceCfg(device=torch.device("cuda:0"), dtype=torch.float32),
    )
    torch.cuda.synchronize()
    del checker
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    return time.monotonic() - started


def _contact_links(arm: str) -> tuple[str, ...]:
    """Return distal links allowed to contact a presentation fixture."""

    return (
        f"{arm}_hand_thumb_2_link",
        f"{arm}_hand_middle_1_link",
        f"{arm}_hand_index_1_link",
    )


def _object_contact_links(arm: str) -> tuple[str, ...]:
    """Return movable finger links allowed to contact the grasped object.

    The final open-hand approach places the cube between the fingers.  Dex3's
    proximal finger collision spheres can therefore overlap the cube before a
    distal tip sphere does.  Treating only the three distal links as contact
    geometry makes the optimizer stop outside otherwise qualified grasps.  The
    palm, wrist, table, fixture, opposite arm, and complete self geometry remain
    enabled and are still checked independently on every route/window.
    """

    return tuple(
        f"{arm}_hand_{suffix}_link" for suffix in DEX3_MOTOR_JOINT_SUFFIXES[arm]
    )


def _local_table_plane_links(arm: str) -> tuple[str, ...]:
    return (
        f"{arm}_wrist_pitch_link",
        f"{arm}_wrist_yaw_link",
        f"{arm}_hand_palm_link",
        *(f"{arm}_hand_{suffix}_link" for suffix in DEX3_MOTOR_JOINT_SUFFIXES[arm]),
    )


def _local_wrist_plane_links(arm: str) -> tuple[str, ...]:
    return (f"{arm}_wrist_pitch_link", f"{arm}_wrist_yaw_link")


def _fixed_hand_links(arm: str) -> tuple[str, ...]:
    return (
        f"{arm}_hand_palm_link",
        *(f"{arm}_hand_{suffix}_link" for suffix in DEX3_MOTOR_JOINT_SUFFIXES[arm]),
    )


def _selected_open_transit_world_robot(
    strict_robot: dict[str, Any], *, arm: str
) -> dict[str, Any]:
    """Scope open-transit world obstacles to the selected wrist and hand.

    CuRobo applies one sphere set to world and self collision. Making unrelated
    radii negative therefore makes this optimizer model deliberately
    permissive; every generated route is checked afterward against the strict
    full-robot self-collision, cube, fixture, and table-plane guards.
    """

    robot = copy.deepcopy(strict_robot)
    kinematics = robot["kinematics"]
    selected_links = set(_local_table_plane_links(arm))
    source_buffer = kinematics.get("collision_sphere_buffer", 0.0)
    buffers: dict[str, float] = {}
    for link_name, spheres in kinematics["collision_spheres"].items():
        if link_name in selected_links:
            buffers[link_name] = (
                float(source_buffer.get(link_name, 0.0))
                if isinstance(source_buffer, dict)
                else float(source_buffer)
            )
            continue
        maximum_radius = max((float(item["radius"]) for item in spheres), default=0.0)
        buffers[link_name] = -maximum_radius - WORLD_COLLISION_DISABLE_RADIUS_EPSILON_M
    kinematics["collision_sphere_buffer"] = buffers
    return robot


def _resolved_motion_robot(
    *,
    checker: CuroboKinematicCollisionChecker,
    source_robot: dict[str, Any],
    arm: str,
    scope_world_to_selected_hand: bool,
):
    """Clone one parsed strict model into an independent optimizer model.

    Loading the G1 URDF and its 766 collision spheres costs roughly two
    seconds on the commissioned laptop.  The strict checker and motion
    optimizer need independent tensors, but they do not need to parse the
    identical kinematic tree twice.  Clone the already-resolved tensors,
    retain only the moving grasp query frame, and optionally reproduce the
    open-transit world-sphere policy exactly.
    """

    import torch
    from curobo._src.robot.kinematics.kinematics_cfg import KinematicsCfg
    from curobo._src.robot.types.self_collision_params import SelfCollisionKinematicsCfg
    from curobo._src.types.robot import RobotCfg

    source_config = checker.config
    params = source_config.kinematics_config.clone()
    frame = grasp_frame(arm)
    if frame not in params.tool_frames:
        raise ValueError(f"resolved CuRobo model lacks grasp frame {frame}")
    frame_count = len(params.tool_frames)
    frame_index = params.tool_frames.index(frame)
    params.tool_frames = [frame]
    params.tool_frame_map = (
        params.tool_frame_map[frame_index : frame_index + 1].clone().contiguous()
    )
    joint_affects = params.joint_affects_endeffector.reshape(-1, frame_count)
    params.joint_affects_endeffector = (
        joint_affects[:, frame_index : frame_index + 1].clone().contiguous().reshape(-1)
    )

    if scope_world_to_selected_hand:
        selected_links = set(_local_table_plane_links(arm))
        raw = source_robot["kinematics"]
        raw_spheres = raw["collision_spheres"]
        reserved_links = set((raw.get("extra_collision_spheres") or {}).keys())
        for link_name in params.link_name_to_idx_map:
            indices = params.get_sphere_index_from_link_name(link_name)
            if indices.numel() == 0 or link_name in selected_links:
                continue
            if link_name in reserved_links or link_name not in raw_spheres:
                # CuRobo reserves payload spheres that do not appear in the
                # source sphere dictionary.  Keep them disabled before grasp.
                radii = torch.full(
                    (indices.numel(),),
                    -100.0,
                    dtype=params.link_spheres.dtype,
                    device=params.link_spheres.device,
                )
            else:
                maximum_radius = max(
                    (float(item["radius"]) for item in raw_spheres[link_name]),
                    default=0.0,
                )
                buffer = -maximum_radius - WORLD_COLLISION_DISABLE_RADIUS_EPSILON_M
                radii = checker.device_cfg.to_device(
                    [float(item["radius"]) + buffer for item in raw_spheres[link_name]]
                )
                if radii.numel() != indices.numel():
                    raise RuntimeError(
                        f"resolved sphere count differs for {link_name}: "
                        f"{radii.numel()} != {indices.numel()}"
                    )
            params.link_spheres[:, indices, 3] = radii
            params.reference_link_spheres[:, indices, 3] = radii

    strict_self = source_config.self_collision_config
    independent_self = None
    if strict_self is not None:
        independent_self = SelfCollisionKinematicsCfg(
            num_spheres=strict_self.num_spheres,
            sphere_padding=strict_self.sphere_padding.clone(),
            collision_pairs=strict_self.collision_pairs.clone(),
            _num_checks_per_thread_large_collision_pairs=(
                strict_self._num_checks_per_thread_large_collision_pairs
            ),
            _max_threads_per_block_large_collision_pairs=(
                strict_self._max_threads_per_block_large_collision_pairs
            ),
            _max_threads_per_block_small_collision_pairs=(
                strict_self._max_threads_per_block_small_collision_pairs
            ),
            _num_checks_per_thread_small_collision_pairs=(
                strict_self._num_checks_per_thread_small_collision_pairs
            ),
        )
    resolved = KinematicsCfg(
        device_cfg=source_config.device_cfg,
        tool_frames=[frame],
        kinematics_config=params,
        self_collision_config=independent_self,
        kinematics_parser=source_config.kinematics_parser,
        generator_config=source_config.generator_config,
    )
    return RobotCfg(kinematics=resolved, device_cfg=checker.device_cfg)


def _validate_strict_supported_escape_self_collision(
    samples: list[dict[tuple[str, str], float]],
) -> None:
    """Reject every enabled CuRobo sphere overlap, including at the live start."""

    if not samples:
        raise RuntimeError("supported escape collision validation has no samples")
    for index, collisions in enumerate(samples):
        if not collisions:
            continue
        details = ", ".join(
            f"{first}/{second}={penetration_m * 1000.0:.3f}mm"
            for (first, second), penetration_m in sorted(collisions.items())
        )
        if index == 0:
            raise RuntimeError(
                "live supported-start state is in strict CuRobo self-collision: "
                f"{details}. Reposition the robot and rerun; no start-state "
                "collision exception is applied"
            )
        raise RuntimeError(
            "strict supported escape creates self-collision at sample "
            f"{index}/{len(samples) - 1}: {details}"
        )


def _pose_list(matrix: np.ndarray) -> list[float]:
    quaternion_xyzw = Rotation.from_matrix(matrix[:3, :3]).as_quat()
    return [
        *matrix[:3, 3].tolist(),
        float(quaternion_xyzw[3]),
        *quaternion_xyzw[:3].tolist(),
    ]


def _anchor_trajectory_start(
    trajectory: PlannedTrajectory,
    *,
    command_q_rad: np.ndarray,
    model_q_rad: np.ndarray,
) -> PlannedTrajectory:
    """Preserve the exact serialized boundary across isolated float32 planners."""

    expected_command = np.asarray(command_q_rad, dtype=np.float64)
    expected_model = np.asarray(model_q_rad, dtype=np.float64)
    actual_command = np.asarray(trajectory.command_q_rad[0], dtype=np.float64)
    actual_model = np.asarray(trajectory.model_q_rad[0], dtype=np.float64)
    scale = max(
        1.0,
        float(np.max(np.abs(expected_command))),
        float(np.max(np.abs(expected_model))),
    )
    tolerance = 4.0 * float(np.finfo(np.float32).eps) * scale
    error = max(
        float(np.max(np.abs(actual_command - expected_command))),
        float(np.max(np.abs(actual_model - expected_model))),
    )
    if error > tolerance:
        raise RuntimeError(
            "CuRobo trajectory does not begin at its serialized request state: "
            f"error={error:.9f}rad, float32 tolerance={tolerance:.9f}rad"
        )
    command_samples = list(trajectory.command_q_rad)
    model_samples = list(trajectory.model_q_rad)
    command_samples[0] = tuple(float(value) for value in expected_command)
    model_samples[0] = tuple(float(value) for value in expected_model)
    return PlannedTrajectory(
        from_pose_id=trajectory.from_pose_id,
        to_pose_id=trajectory.to_pose_id,
        sample_time_s=trajectory.sample_time_s,
        command_q_rad=tuple(command_samples),
        model_q_rad=tuple(model_samples),
        planning_time_s=trajectory.planning_time_s,
    )


def _anchor_trajectory_end(
    trajectory: PlannedTrajectory,
    *,
    command_q_rad: np.ndarray,
    model_q_rad: np.ndarray,
) -> PlannedTrajectory:
    """Preserve the exact next-phase boundary across float32 CuRobo output."""

    reversed_trajectory = _reverse_trajectory(trajectory)
    anchored = _anchor_trajectory_start(
        reversed_trajectory,
        command_q_rad=command_q_rad,
        model_q_rad=model_q_rad,
    )
    restored = _reverse_trajectory(anchored)
    return PlannedTrajectory(
        from_pose_id=trajectory.from_pose_id,
        to_pose_id=trajectory.to_pose_id,
        sample_time_s=restored.sample_time_s,
        command_q_rad=restored.command_q_rad,
        model_q_rad=restored.model_q_rad,
        planning_time_s=trajectory.planning_time_s,
    )


def _split_lift_trajectory(
    trajectory: PlannedTrajectory,
    *,
    split_index: int,
) -> tuple[PlannedTrajectory, PlannedTrajectory]:
    """Split one already validated lift without changing any sample or command."""

    sample_count = len(trajectory.sample_time_s)
    if split_index <= 0 or split_index >= sample_count - 1:
        raise ValueError("retention-test split must leave motion samples on both sides")
    split_time = trajectory.sample_time_s[split_index]
    test_lift = PlannedTrajectory(
        from_pose_id="grasp_approach",
        to_pose_id="retention_test_lift",
        sample_time_s=trajectory.sample_time_s[: split_index + 1],
        command_q_rad=trajectory.command_q_rad[: split_index + 1],
        model_q_rad=trajectory.model_q_rad[: split_index + 1],
        planning_time_s=trajectory.planning_time_s,
    )
    payload_lift = PlannedTrajectory(
        from_pose_id="retention_test_lift",
        to_pose_id="payload_lift",
        sample_time_s=tuple(
            value - split_time for value in trajectory.sample_time_s[split_index:]
        ),
        command_q_rad=trajectory.command_q_rad[split_index:],
        model_q_rad=trajectory.model_q_rad[split_index:],
        planning_time_s=0.0,
    )
    return test_lift, payload_lift


def _load_shortlist(request: TabletopTaskRequest) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    path = (ROOT / request.grasp_shortlist_path).resolve()
    if not path.is_relative_to(ROOT):
        raise ValueError("grasp shortlist must be inside the repository")
    content = path.read_bytes()
    if hashlib.sha256(content).hexdigest() != request.grasp_shortlist_sha256:
        raise ValueError("grasp shortlist SHA-256 differs from the request")
    document = yaml.safe_load(content)
    if document.get("format") != "g1_aprilcube_executable_grasp_shortlist":
        raise ValueError("unsupported grasp shortlist format")
    presentation = document.get("presentation")
    shortlist_presentation_id = (
        "direct" if presentation is None else str(presentation.get("id", ""))
    )
    if shortlist_presentation_id != request.presentation_id:
        raise ValueError(
            "grasp shortlist presentation differs from the task request: "
            f"{shortlist_presentation_id!r} != {request.presentation_id!r}"
        )
    if request.fixture is not None:
        assert presentation is not None
        expected_fixture = {
            "fixture_id": request.fixture.fixture_id,
            "fixture_mesh": request.fixture.mesh_path,
            "fixture_mesh_sha256": request.fixture.mesh_sha256,
            "fixture_mesh_scale": list(request.fixture.mesh_scale),
            "support_height_m": request.fixture.support_height_m,
            "cube_pose_contract": request.fixture.cube_pose_contract,
        }
        actual_fixture = {name: presentation.get(name) for name in expected_fixture}
        if actual_fixture != expected_fixture:
            raise ValueError(
                "grasp shortlist fixture contract differs from the task request: "
                f"expected={expected_fixture}, actual={actual_fixture}"
            )
    applicable = tuple(document.get("applicable_hand_sides", ()))
    if request.arm not in applicable or document.get("object_id") != "cube_head":
        raise ValueError(
            f"tabletop shortlist is not qualified for the selected {request.arm} hand"
        )
    if (
        document.get("qualification_source_hand_side") != "right"
        or document.get("canonical_grasp_frame") != "GraspGenX_G"
        or document.get("handedness_policy")
        != "preserve_object_T_G_and_apply_exact_dex3_side_adapter"
    ):
        raise ValueError("tabletop shortlist lacks the commissioned Dex3 handedness contract")
    mesh_path = (ROOT / str(document.get("object_mesh", ""))).resolve()
    if not mesh_path.is_relative_to(ROOT) or not mesh_path.is_file():
        raise ValueError("grasp shortlist object mesh must exist inside the repository")
    mesh_content = mesh_path.read_bytes()
    if hashlib.sha256(mesh_content).hexdigest() != document.get("object_mesh_sha256"):
        raise ValueError("grasp shortlist object mesh SHA-256 does not match the repository")
    vertices = []
    for line in mesh_content.decode("utf-8").splitlines():
        fields = line.split()
        if fields[:1] == ["v"] and len(fields) == 4:
            vertices.append(tuple(float(value) for value in fields[1:]))
    if not vertices:
        raise ValueError("grasp shortlist object mesh contains no OBJ vertices")
    points = np.asarray(vertices, dtype=np.float64)
    mesh_dimensions = points.max(axis=0) - points.min(axis=0)
    if not np.allclose(mesh_dimensions, request.object_dimensions_m, atol=1.0e-6, rtol=0.0):
        raise ValueError(
            "task object dimensions differ from the qualified grasp mesh: "
            f"task={list(request.object_dimensions_m)}, mesh={mesh_dimensions.tolist()}"
        )
    candidates = list(document.get("candidates", ()))
    if not candidates or len(candidates) != int(document.get("candidate_count", -1)):
        raise ValueError("qualified cube shortlist candidate count is invalid")
    approach_distance_m = float(document["execution_contract"]["approach_distance_m"])
    if request.fixture is None:
        contract = document["execution_contract"]
        if (
            document.get("format_version") != 2
            or contract.get("fixed_cube_during_qualification") is not True
            or contract.get("fixed_descriptor_close_target") is not True
            or contract.get("exact_fixed_close_sweep_table_clear") is not True
            or int(contract.get("fixed_close_sweep_samples", 0)) != 51
        ):
            raise ValueError(
                "direct-table shortlist lacks stationary-cube fixed-close qualification"
            )
    for candidate in candidates:
        evidence = candidate.get("execution_evidence", {})
        if evidence.get("intrinsic_retention_passed") is not True:
            raise ValueError(
                "qualified cube candidate lacks intrinsic retention evidence: "
                f"{candidate.get('candidate_id', '<missing ID>')}"
            )
        if request.fixture is None:
            exact_clearance = float(evidence.get("fixed_close_sweep_table_clearance_m", -1.0))
            if (
                evidence.get("qualification_model") != "stationary_cube_fixed_descriptor_close"
                or exact_clearance < request.minimum_hand_plane_clearance_m
            ):
                raise ValueError(
                    "direct-table candidate lacks the required exact fixed-close clearance: "
                    f"{candidate.get('candidate_id', '<missing ID>')}"
                )
        if request.fixture is not None and approach_distance_m not in candidate.get(
            "valid_approach_distances_m", ()
        ):
            raise ValueError(
                "fixture-qualified cube candidate does not admit the runtime approach: "
                f"{candidate.get('candidate_id', '<missing ID>')} at "
                f"{approach_distance_m:.3f}m"
            )
    return document, candidates


def _candidate_transform(entry: dict[str, Any]) -> np.ndarray:
    pose = entry["object_T_G"]
    orientation = pose["orientation"]
    rotation = Rotation.from_quat([*orientation["xyz"], orientation["w"]]).as_matrix()
    result = np.eye(4, dtype=np.float64)
    result[:3, :3] = rotation
    result[:3, 3] = np.asarray(pose["position"], dtype=np.float64)
    return result


def _canonical_resting_cube_pose(base_T_detected_object: np.ndarray) -> np.ndarray:
    return canonical_resting_cube_pose(base_T_detected_object)


def _table_from_resting_object(
    request: TabletopTaskRequest, base_T_torso: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Derive only the supporting plane from the resting cube.

    A single cube observation contains no evidence about the finite table
    footprint, its yaw, or its thickness.  The first return value is therefore
    a point on the top plane, not a fabricated table-frame origin.
    """

    base_T_camera = base_T_torso @ np.asarray(request.torso_T_camera)
    detected_object = base_T_camera @ np.asarray(request.planning_camera_T_object)
    base_T_object = _canonical_resting_cube_pose(detected_object)
    table_reference = request.table_reference_camera_T_object
    if table_reference is None:
        base_T_table_reference = base_T_object
        reference_extent = request.object_dimensions_m[2]
        fixture_height = 0.0 if request.fixture is None else request.fixture.support_height_m
    else:
        base_T_table_reference = _canonical_resting_cube_pose(
            base_T_camera @ np.asarray(table_reference)
        )
        assert request.table_reference_object_dimensions_m is not None
        reference_extent = request.table_reference_object_dimensions_m[2]
        fixture_height = 0.0
    object_up = base_T_table_reference[:3, 2]
    down = -object_up
    top_origin = base_T_table_reference[:3, 3] + (0.5 * reference_extent + fixture_height) * down
    return top_origin, base_T_object, down


def _base_T_detected_object(
    request: TabletopTaskRequest,
    base_T_torso: np.ndarray,
) -> np.ndarray:
    """Return the unpermuted detector frame used by inter-object contracts."""

    return (
        np.asarray(base_T_torso, dtype=np.float64)
        @ np.asarray(request.torso_T_camera, dtype=np.float64)
        @ np.asarray(request.planning_camera_T_object, dtype=np.float64)
    )


def _table_from_charuco_board(
    request: CharucoSupportedEscapeRequest,
    base_T_torso: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the physical top-plane point and down direction from the board."""

    base_T_camera = base_T_torso @ np.asarray(request.torso_T_camera)
    base_T_board = base_T_camera @ np.asarray(request.observation.camera_T_board)
    down = base_T_board[:3, 2]
    # The frozen OpenCV board frame has +Z into the backing.  Reuse the same
    # 20-degree face-up policy as the AprilCube table-plane implementation so
    # a flipped planar solution can never request a downward escape.
    if float(down[2]) > -np.cos(np.deg2rad(20.0)):
        raise RuntimeError(
            "ChArUco board normal is not face-up within 20 degrees; refusing "
            "to infer the lift direction"
        )
    return base_T_board[:3, 3], down


def _fixture_mesh_path(request: TabletopTaskRequest) -> Path | None:
    if request.fixture is None:
        return None
    path = (ROOT / request.fixture.mesh_path).resolve()
    if not path.is_relative_to(ROOT) or not path.is_file():
        raise ValueError("fixture mesh must exist inside the repository")
    if hashlib.sha256(path.read_bytes()).hexdigest() != request.fixture.mesh_sha256:
        raise ValueError("fixture mesh SHA-256 differs from the task request")
    return path


def _base_T_fixture(
    request: TabletopTaskRequest,
    base_T_object: np.ndarray,
    down: np.ndarray,
) -> np.ndarray | None:
    """Place the mesh base on the inferred table under its aligned cube."""

    if request.fixture is None:
        return None
    result = np.asarray(base_T_object, dtype=np.float64).copy()
    result[:3, 3] += (
        0.5 * request.object_dimensions_m[2] + request.fixture.support_height_m
    ) * down
    return result


def _base_scene(
    request: TabletopTaskRequest,
    base_T_torso: np.ndarray,
    *,
    include_cube: bool,
    include_open_transit_table_patch: bool = False,
    include_environment_cuboids: bool = True,
    include_placement_support: bool = True,
    base_T_object_override: np.ndarray | None = None,
    base_T_detected_object_override: np.ndarray | None = None,
    plane_point_override: np.ndarray | None = None,
    down_override: np.ndarray | None = None,
) -> dict[str, Any]:
    overrides = (
        base_T_object_override,
        base_T_detected_object_override,
        plane_point_override,
        down_override,
    )
    if any(value is not None for value in overrides):
        if not all(value is not None for value in overrides):
            raise ValueError("live tabletop geometry overrides must be supplied together")
        base_T_object = np.asarray(base_T_object_override, dtype=np.float64)
        base_T_detected_object = np.asarray(
            base_T_detected_object_override, dtype=np.float64
        )
        plane_point = np.asarray(plane_point_override, dtype=np.float64)
        down = np.asarray(down_override, dtype=np.float64)
    else:
        plane_point, base_T_object, down = _table_from_resting_object(request, base_T_torso)
        base_T_detected_object = _base_T_detected_object(request, base_T_torso)
    scene: dict[str, Any] = {"cuboid": {}}
    if include_cube:
        scene["cuboid"]["cube"] = {
            "dims": list(request.object_dimensions_m),
            "pose": _pose_list(base_T_object),
        }
    if include_environment_cuboids:
        for cuboid in request.environment_cuboids:
            if cuboid.role == "placement_support" and not include_placement_support:
                continue
            base_T_cuboid = base_T_detected_object @ np.asarray(cuboid.object_T_cuboid)
            scene["cuboid"][cuboid.object_id] = {
                "dims": list(cuboid.dimensions_m),
                "pose": _pose_list(base_T_cuboid),
            }
    if include_open_transit_table_patch:
        dimensions = request.open_transit_table_patch_dimensions_m
        base_T_patch = base_T_object.copy()
        # The cuboid is entirely below the inferred plane, with its top face
        # exactly coincident with the cube's supporting surface.
        base_T_patch[:3, 3] = plane_point + 0.5 * dimensions[2] * down
        scene["cuboid"]["open_transit_table_patch"] = {
            "dims": list(dimensions),
            "pose": _pose_list(base_T_patch),
        }
    fixture_path = _fixture_mesh_path(request)
    fixture_pose = _base_T_fixture(request, base_T_object, down)
    if fixture_path is not None and fixture_pose is not None:
        assert request.fixture is not None
        scene["mesh"] = {
            request.fixture.fixture_id: {
                "file_path": str(fixture_path),
                "pose": _pose_list(fixture_pose),
                "scale": list(request.fixture.mesh_scale),
            }
        }
    return scene


def _attached_lift_scene(
    request: TabletopTaskRequest,
    base_T_torso: np.ndarray,
    *,
    base_T_object_override: np.ndarray | None = None,
    base_T_detected_object_override: np.ndarray | None = None,
    plane_point_override: np.ndarray | None = None,
    down_override: np.ndarray | None = None,
) -> dict[str, Any]:
    """Scene for lifting an object away from its supporting fixture.

    The attached cube begins in deliberate contact with its presenter. CuRobo's
    world checker cannot disable only that object/fixture pair, so the support
    is omitted from this optimizer scene. The generated route is independently
    rechecked against the exact fixture using the robot spheres without the
    attached cube, preserving every hand/fixture collision rule.
    """

    scene = _base_scene(
        request,
        base_T_torso,
        include_cube=False,
        include_placement_support=False,
        base_T_object_override=base_T_object_override,
        base_T_detected_object_override=base_T_detected_object_override,
        plane_point_override=plane_point_override,
        down_override=down_override,
    )
    if request.fixture is not None:
        meshes = scene.get("mesh", {})
        meshes.pop(request.fixture.fixture_id, None)
        if not meshes:
            scene.pop("mesh", None)
    return scene


def _fixture_collision_checker(
    request: TabletopTaskRequest,
    base_T_object: np.ndarray,
    base_T_detected_object: np.ndarray,
    down: np.ndarray,
    *,
    device_cfg,
):
    """Load the hash-checked fixture once into CuRobo's CUDA collision world."""

    path = _fixture_mesh_path(request)
    fixture_pose = _base_T_fixture(request, base_T_object, down)
    cuboids = {
        cuboid.object_id: {
            "dims": list(cuboid.dimensions_m),
            "pose": _pose_list(base_T_detected_object @ np.asarray(cuboid.object_T_cuboid)),
        }
        for cuboid in request.environment_cuboids
    }
    if (path is None or fixture_pose is None) and not cuboids:
        return None
    scene: dict[str, Any] = {}
    if cuboids:
        scene["cuboid"] = cuboids
    if path is not None and fixture_pose is not None:
        assert request.fixture is not None
        scene["mesh"] = {
            request.fixture.fixture_id: {
                "file_path": str(path),
                "pose": _pose_list(fixture_pose),
                "scale": list(request.fixture.mesh_scale),
            }
        }
    return CuroboWorldCollisionChecker(
        scene=scene,
        device_cfg=device_cfg,
    )


def _local_plane_clearance(
    planner,
    model_q: np.ndarray,
    *,
    arm: str,
    plane_point: np.ndarray,
    down: np.ndarray,
    include_payload: bool,
    link_names: tuple[str, ...] | None = None,
) -> tuple[float, str, int]:
    """Return minimum signed clearance for local manipulation geometry.

    CuRobo still checks full-robot self-collision.  This independent guard is
    intentionally limited to the selected wrist/hand and, after grasping, the
    attached payload.  Applying an unbounded plane to the fixed torso, legs,
    opposite arm, or elbow would falsely assert that geometry outside the
    unseen table footprint must also be above the tabletop.
    """

    from curobo.types import JointState

    values = np.asarray(model_q, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 7:
        raise ValueError("plane validation requires an N x 7 trajectory")
    device_cfg = planner.device_cfg
    state = JointState.from_position(
        device_cfg.to_device(values), joint_names=list(arm_joint_names(arm))
    )
    spheres = planner.compute_kinematics(state).robot_spheres
    if spheres is None:
        raise RuntimeError("CuRobo kinematics did not return collision spheres")
    sphere_array = spheres.detach().cpu().numpy().reshape(values.shape[0], -1, 4)
    config = planner.kinematics.config.kinematics_config
    return _local_plane_clearance_from_spheres(
        sphere_array,
        config=config,
        arm=arm,
        plane_point=plane_point,
        down=down,
        include_payload=include_payload,
        link_names=link_names,
    )


def _local_plane_clearance_from_spheres(
    sphere_array: np.ndarray,
    *,
    config,
    arm: str,
    plane_point: np.ndarray,
    down: np.ndarray,
    include_payload: bool,
    link_names: tuple[str, ...] | None = None,
) -> tuple[float, str, int]:
    """Evaluate the table guard from already-computed CuRobo spheres."""

    clearances, links = _local_plane_clearance_samples_from_spheres(
        sphere_array,
        config=config,
        arm=arm,
        plane_point=plane_point,
        down=down,
        include_payload=include_payload,
        link_names=link_names,
    )
    minimum_sample = int(np.argmin(clearances))
    return float(clearances[minimum_sample]), links[minimum_sample], minimum_sample


def _local_plane_clearance_samples_from_spheres(
    sphere_array: np.ndarray,
    *,
    config,
    arm: str,
    plane_point: np.ndarray,
    down: np.ndarray,
    include_payload: bool,
    link_names: tuple[str, ...] | None = None,
) -> tuple[np.ndarray, tuple[str, ...]]:
    """Return the closest guarded link and signed clearance at every sample."""

    import torch

    values = np.asarray(sphere_array, dtype=np.float64)
    if values.ndim != 3 or values.shape[2] != 4:
        raise ValueError("table-plane sphere array must have shape N x S x 4")
    selected_links = list(_local_table_plane_links(arm) if link_names is None else link_names)
    if include_payload:
        selected_links.append(attachment_link(arm))
    minimum = np.full(values.shape[0], np.inf, dtype=np.float64)
    minimum_link = np.full(values.shape[0], "", dtype=object)
    up = -np.asarray(down, dtype=np.float64)
    point = np.asarray(plane_point, dtype=np.float64)
    for link_name in selected_links:
        try:
            indices = config.get_sphere_index_from_link_name(link_name)
        except BaseException as error:  # CuRobo raises for unknown names.
            raise RuntimeError(f"CuRobo model lacks table-guard link {link_name}") from error
        indices_np = torch.as_tensor(indices).detach().cpu().numpy().reshape(-1)
        if len(indices_np) == 0:
            raise RuntimeError(f"CuRobo model has no spheres for table-guard link {link_name}")
        selected = values[:, indices_np, :]
        valid = selected[..., 3] > 0.0
        clearance = np.einsum("...i,i->...", selected[..., :3] - point, up) - selected[..., 3]
        clearance = np.where(valid, clearance, np.inf)
        per_sample = np.min(clearance, axis=1)
        update = per_sample < minimum
        minimum[update] = per_sample[update]
        minimum_link[update] = link_name
    if not np.all(np.isfinite(minimum)) or np.any(minimum_link == ""):
        raise RuntimeError("table-plane guard found no enabled local collision spheres")
    return minimum, tuple(str(value) for value in minimum_link)


def _validate_start_relative_retention_clearance(
    clearances_m: np.ndarray,
    links: tuple[str, ...],
    *,
    required_m: float,
) -> _StartRelativePlaneClearance:
    """Validate an escape from, and exact return to, a positive grasp boundary."""

    values = np.asarray(clearances_m, dtype=np.float64).reshape(-1)
    if len(values) < 2 or len(links) != len(values) or not np.all(np.isfinite(values)):
        raise ValueError("retention table clearances must describe every finite route sample")
    if not np.isfinite(required_m) or required_m <= 0.0:
        raise ValueError("retention free-space table margin must be positive and finite")
    tolerance = PLANE_CLEARANCE_NUMERICAL_TOLERANCE_M
    start = float(values[0])
    end = float(values[-1])
    minimum_sample = int(np.argmin(values))
    minimum = float(values[minimum_sample])
    minimum_link = links[minimum_sample]
    if start <= 0.0 or end <= 0.0:
        boundary_sample = 0 if start <= end else len(values) - 1
        raise RuntimeError(
            "measured close Dex3 grasp boundary is not above the observed table plane: "
            f"clearance={values[boundary_sample]:.4f}m at {links[boundary_sample]} "
            f"sample {boundary_sample}/{len(values) - 1}"
        )
    if abs(start - end) > tolerance:
        raise RuntimeError(
            "measured close payload route does not return to the same hand/table "
            f"boundary: start={start:.6f}m, end={end:.6f}m"
        )

    full_margin = values >= required_m - tolerance
    if not np.any(full_margin):
        raise RuntimeError(
            "measured close payload route never reaches the required free-space "
            f"hand/table margin: maximum={np.max(values):.4f}m; required={required_m:.4f}m"
        )
    first_full = int(np.flatnonzero(full_margin)[0])
    last_full = int(np.flatnonzero(full_margin)[-1])
    boundary = min(start, end)

    outbound_minimum_sample = int(np.argmin(values[: first_full + 1]))
    if values[outbound_minimum_sample] < boundary - tolerance:
        raise RuntimeError(
            "measured close payload escape moves closer to the table than its "
            f"already-achieved grasp boundary: clearance={values[outbound_minimum_sample]:.4f}m "
            f"at {links[outbound_minimum_sample]} sample {outbound_minimum_sample}/"
            f"{len(values) - 1}; boundary={boundary:.4f}m"
        )

    interior = values[first_full : last_full + 1]
    interior_minimum_offset = int(np.argmin(interior))
    interior_minimum_sample = first_full + interior_minimum_offset
    if interior[interior_minimum_offset] < required_m - tolerance:
        raise RuntimeError(
            "measured close payload route drops below the required free-space "
            f"hand/table margin: clearance={values[interior_minimum_sample]:.4f}m at "
            f"{links[interior_minimum_sample]} sample {interior_minimum_sample}/"
            f"{len(values) - 1}; required={required_m:.4f}m"
        )

    return_minimum_offset = int(np.argmin(values[last_full:]))
    return_minimum_sample = last_full + return_minimum_offset
    if values[return_minimum_sample] < boundary - tolerance:
        raise RuntimeError(
            "measured close payload return moves closer to the table than its "
            f"grasp boundary: clearance={values[return_minimum_sample]:.4f}m at "
            f"{links[return_minimum_sample]} sample {return_minimum_sample}/"
            f"{len(values) - 1}; boundary={boundary:.4f}m"
        )

    return _StartRelativePlaneClearance(
        minimum_m=minimum,
        minimum_link=minimum_link,
        minimum_sample=minimum_sample,
        boundary_m=boundary,
        first_full_margin_sample=first_full,
        last_full_margin_sample=last_full,
    )


def _validate_pick_place_retention_clearance(
    clearances: np.ndarray,
    link_names: tuple[str, ...],
    *,
    required_m: float,
) -> _StartRelativePlaneClearance:
    """Validate contact→free-space→contact without assuming equal endpoints."""

    values = np.asarray(clearances, dtype=np.float64)
    links = tuple(link_names)
    if values.ndim != 1 or len(values) < 3 or len(links) != len(values):
        raise ValueError("pick-place clearance evidence must be one finite route")
    if not np.all(np.isfinite(values)):
        raise ValueError("pick-place clearance evidence must be finite")
    tolerance = PLANE_CLEARANCE_NUMERICAL_TOLERANCE_M
    if values[0] <= 0.0 or values[-1] <= 0.0:
        boundary_index = 0 if values[0] <= values[-1] else len(values) - 1
        raise RuntimeError(
            "measured close pick-place endpoint is not above its supporting plane: "
            f"clearance={values[boundary_index]:.4f}m at {links[boundary_index]} "
            f"sample {boundary_index}/{len(values) - 1}"
        )
    full = np.flatnonzero(values >= required_m - tolerance)
    if len(full) == 0:
        raise RuntimeError(
            "measured close pick-place route never reaches the required free-space "
            f"hand/table margin {required_m:.4f}m"
        )
    first_full = int(full[0])
    last_full = int(full[-1])
    outbound_minimum_sample = int(np.argmin(values[: first_full + 1]))
    if values[outbound_minimum_sample] < values[0] - tolerance:
        raise RuntimeError(
            "measured close pick-place escape moves below its achieved source contact "
            f"clearance: {values[outbound_minimum_sample]:.4f}m at sample "
            f"{outbound_minimum_sample}/{len(values) - 1}; source={values[0]:.4f}m"
        )
    interior = values[first_full : last_full + 1]
    interior_offset = int(np.argmin(interior))
    interior_sample = first_full + interior_offset
    if interior[interior_offset] < required_m - tolerance:
        raise RuntimeError(
            "measured close pick-place route drops below the required free-space "
            f"margin: {values[interior_sample]:.4f}m at sample "
            f"{interior_sample}/{len(values) - 1}; required={required_m:.4f}m"
        )
    inbound_offset = int(np.argmin(values[last_full:]))
    inbound_sample = last_full + inbound_offset
    if values[inbound_sample] < values[-1] - tolerance:
        raise RuntimeError(
            "measured close pick-place placement moves below its destination contact "
            f"clearance: {values[inbound_sample]:.4f}m at sample "
            f"{inbound_sample}/{len(values) - 1}; destination={values[-1]:.4f}m"
        )
    minimum_sample = int(np.argmin(values))
    return _StartRelativePlaneClearance(
        minimum_m=float(values[minimum_sample]),
        minimum_link=links[minimum_sample],
        minimum_sample=minimum_sample,
        boundary_m=min(float(values[0]), float(values[-1])),
        first_full_margin_sample=first_full,
        last_full_margin_sample=last_full,
    )


class _FixedCloseSweepValidator:
    """Check the commanded finger sweep at a candidate's fixed arm contact pose.

    Hardware sends the one descriptor close target, so candidate selection
    proves the complete fixed-cube sweep before moving to pregrasp. Physical
    contact may stop the fingers earlier; that measured posture still undergoes
    the independent frozen-route check before any lift.
    """

    def __init__(
        self,
        *,
        request: TabletopTaskRequest,
        base_T_object: np.ndarray,
        base_T_detected_object: np.ndarray,
        plane_point: np.ndarray,
        down: np.ndarray,
        open_q: np.ndarray,
        close_target_q: np.ndarray,
    ) -> None:
        import torch
        from curobo.types import DeviceCfg

        started = time.monotonic()
        self.request = request
        self.arm = request.arm
        self.plane_point = np.asarray(plane_point, dtype=np.float64)
        self.down = np.asarray(down, dtype=np.float64)
        self.finger_sweep = sample_linear_joint_sweep(open_q, close_target_q)
        robot, self.active_joint_names, reference = build_tabletop_route_validation_robot_config(
            arm=self.arm,
            snapshot=request.planning_snapshot,
            joint_position_offsets_rad=request.joint_position_offsets_rad,
        )
        self.device_cfg = DeviceCfg(device=torch.device("cuda:0"), dtype=torch.float32)
        self.checker = CuroboKinematicCollisionChecker(
            robot=robot,
            device_cfg=self.device_cfg,
        )
        self.fixture_checker = _fixture_collision_checker(
            request,
            base_T_object,
            base_T_detected_object,
            self.down,
            device_cfg=self.device_cfg,
        )
        self.relative_hand_sweep_spheres, self.hand_sphere_link_names = (
            self._build_relative_hand_sweep(reference)
        )
        self.cache_build_s = time.monotonic() - started

    def _build_relative_hand_sweep(
        self,
        reference: tuple[float, ...],
    ):
        """Express the commanded hand sweep once in the grasp frame."""

        import torch
        from curobo.types import JointState

        arm_q = np.asarray(reference[:7], dtype=np.float64)
        sweep_q = np.concatenate(
            (
                np.repeat(arm_q[None], len(self.finger_sweep), axis=0),
                self.finger_sweep,
            ),
            axis=1,
        )
        spheres = self.checker.robot_spheres(
            sweep_q,
            joint_names=self.active_joint_names,
        ).reshape(len(sweep_q), -1, 4)
        state = JointState.from_position(
            self.device_cfg.to_device(sweep_q[:1]),
            joint_names=list(self.active_joint_names),
        )
        grasp_pose = (
            self.checker.kinematics.compute_kinematics(state)
            .tool_poses[grasp_frame(self.arm)]
            .get_matrix()[0]
        )
        config = self.checker.config.kinematics_config
        indices: list[int] = []
        sphere_links: list[str] = []
        for link_name in _fixed_hand_links(self.arm):
            current = (
                torch.as_tensor(config.get_sphere_index_from_link_name(link_name))
                .detach()
                .cpu()
                .tolist()
            )
            indices.extend(int(value) for value in current)
            sphere_links.extend([link_name] * len(current))
        selected = spheres[:, indices]
        enabled = selected[0, :, 3] > 0.0
        selected = selected[:, enabled]
        sphere_links = [
            name
            for name, keep in zip(sphere_links, enabled.detach().cpu().tolist(), strict=True)
            if keep
        ]
        rotation = grasp_pose[:3, :3]
        translation = grasp_pose[:3, 3]
        local_centers = torch.matmul(
            selected[..., :3] - translation,
            rotation,
        )
        return (
            torch.cat((local_centers, selected[..., 3:4]), dim=-1),
            tuple(sphere_links),
        )

    def batch_candidate_rejections(
        self,
        grasp_matrices: list[np.ndarray],
        *,
        exact_table_evidence: list[tuple[float, str, int]] | None = None,
    ) -> dict[int, tuple[str, str]]:
        """Prune target-fixed hand/fixture and hand/table failures on CUDA."""

        import torch

        if self.fixture_checker is None:
            return {}
        matrices = np.asarray(grasp_matrices, dtype=np.float64)
        if matrices.ndim != 3 or matrices.shape[1:] != (4, 4):
            raise ValueError("fixed-close candidate poses must have shape N x 4 x 4")
        if self.request.fixture is None:
            if exact_table_evidence is None or len(exact_table_evidence) != len(matrices):
                raise ValueError(
                    "direct-table fixed-close pruning requires exact qualification evidence"
                )
        elif exact_table_evidence is not None:
            raise ValueError("fixture planning cannot use direct-table qualification evidence")
        sample_count = len(self.finger_sweep)
        candidates_per_batch = max(
            FIXED_CLOSE_COLLISION_BATCH_SAMPLES // sample_count,
            1,
        )
        relative = self.relative_hand_sweep_spheres
        rejections: dict[int, tuple[str, str]] = {}
        up = self.device_cfg.to_device(-self.down)
        plane_point = self.device_cfg.to_device(self.plane_point)
        for start in range(0, len(matrices), candidates_per_batch):
            stop = min(start + candidates_per_batch, len(matrices))
            transforms = self.device_cfg.to_device(matrices[start:stop])
            rotation = transforms[:, :3, :3]
            translation = transforms[:, :3, 3]
            centers = (
                torch.matmul(
                    relative[None, ..., :3],
                    rotation[:, None, :, :].transpose(-1, -2),
                )
                + translation[:, None, None, :]
            )
            radii = relative[None, ..., 3:4].expand(len(transforms), -1, -1, -1)
            world = torch.cat((centers, radii), dim=-1)
            flat = world.reshape(-1, world.shape[-2], 4)
            fixture_hits = self.fixture_checker.deepest_collisions(
                flat,
                sphere_link_names=self.hand_sphere_link_names,
            )
            clearance = torch.sum((centers - plane_point) * up, dim=-1) - radii[..., 0]
            minimum_clearance, minimum_flat = clearance.reshape(len(transforms), -1).min(dim=1)
            for local_index in range(len(transforms)):
                candidate_index = start + local_index
                sample_hits = fixture_hits[
                    local_index * sample_count : (local_index + 1) * sample_count
                ]
                hits = [value for value in sample_hits if value is not None]
                if hits:
                    penetration, link_name, flat_sample = max(
                        hits,
                        key=lambda value: value[0],
                    )
                    sample = flat_sample - local_index * sample_count
                    rejections[candidate_index] = (
                        "batched_fixed_close_fixture",
                        (
                            f"{link_name}={penetration * 1000.0:.3f}mm penetration at "
                            f"sample {sample}/{sample_count - 1}"
                        ),
                    )
                    continue
                if exact_table_evidence is None:
                    minimum = float(minimum_clearance[local_index].item())
                    flattened = int(minimum_flat[local_index].item())
                    sample = flattened // int(world.shape[-2])
                    sphere = flattened % int(world.shape[-2])
                    link_name = self.hand_sphere_link_names[sphere]
                else:
                    minimum, link_name, sample = exact_table_evidence[candidate_index]
                if minimum < self.request.minimum_hand_plane_clearance_m:
                    rejections[candidate_index] = (
                        "batched_fixed_close_table_plane",
                        (
                            f"clearance={minimum:.4f}m at "
                            f"{link_name} sample "
                            f"{sample}/{sample_count - 1}; required="
                            f"{self.request.minimum_hand_plane_clearance_m:.4f}m"
                        ),
                    )
        return rejections

    def validate(
        self,
        contact_model_q: np.ndarray,
        candidate: dict[str, Any],
    ) -> _FixedCloseSweepResult:
        arm_q = np.asarray(contact_model_q, dtype=np.float64).reshape(-1)
        if arm_q.shape != (7,) or not np.all(np.isfinite(arm_q)):
            raise ValueError("fixed-close sweep requires seven finite contact arm joints")
        sweep_q = np.concatenate(
            (
                np.repeat(arm_q[None], len(self.finger_sweep), axis=0),
                self.finger_sweep,
            ),
            axis=1,
        )
        spheres_tensor = self.checker.robot_spheres(
            sweep_q,
            joint_names=self.active_joint_names,
        )
        collision_samples = self.checker.self_collision_pair_penetrations_from_spheres(
            spheres_tensor
        )
        for sample_index, pairs in enumerate(collision_samples):
            if not pairs:
                continue
            pair, penetration = max(pairs.items(), key=lambda item: item[1])
            raise _BranchRejected(
                "fixed_close_sweep_strict_self_collision",
                f"{pair[0]}/{pair[1]}={penetration * 1000.0:.3f}mm at "
                f"sample {sample_index}/{len(sweep_q) - 1}",
            )

        sphere_array = spheres_tensor.detach().cpu().numpy().reshape(len(sweep_q), -1, 4)
        if self.request.fixture is None:
            evidence = candidate["execution_evidence"]
            exact_link = str(evidence["fixed_close_sweep_minimum_link"])
            if self.arm == "left":
                exact_link = exact_link.replace("right_", "left_", 1)
            exact = (
                float(evidence["fixed_close_sweep_table_clearance_m"]),
                exact_link,
                int(evidence["fixed_close_sweep_minimum_sample"]),
            )
            wrist = _local_plane_clearance_from_spheres(
                sphere_array,
                config=self.checker.config.kinematics_config,
                arm=self.arm,
                plane_point=self.plane_point,
                down=self.down,
                include_payload=False,
                link_names=_local_wrist_plane_links(self.arm),
            )
            hand_clearance, hand_link, hand_sample = min(exact, wrist, key=lambda item: item[0])
        else:
            hand_clearance, hand_link, hand_sample = _local_plane_clearance_from_spheres(
                sphere_array,
                config=self.checker.config.kinematics_config,
                arm=self.arm,
                plane_point=self.plane_point,
                down=self.down,
                include_payload=False,
            )
        if hand_clearance < self.request.minimum_hand_plane_clearance_m:
            raise _BranchRejected(
                "fixed_close_sweep_table_plane",
                f"clearance={hand_clearance:.4f}m at {hand_link} sample "
                f"{hand_sample}/{len(sweep_q) - 1}; required="
                f"{self.request.minimum_hand_plane_clearance_m:.4f}m",
            )

        if self.fixture_checker is not None:
            fixture_hit = self.fixture_checker.first_collision(
                spheres_tensor,
                kinematics_config=self.checker.config.kinematics_config,
            )
            if fixture_hit is not None:
                penetration, fixture_link, fixture_sample = fixture_hit
                raise _BranchRejected(
                    "fixed_close_sweep_fixture",
                    f"{fixture_link}={penetration * 1000.0:.3f}mm penetration at "
                    f"sample {fixture_sample}/{len(sweep_q) - 1}",
                )
        return _FixedCloseSweepResult(
            sample_count=len(sweep_q),
            minimum_plane_clearance_m=hand_clearance,
            minimum_plane_link=hand_link,
            minimum_plane_sample=hand_sample,
            minimum_fixture_clearance_m=None,
            minimum_fixture_link=None,
            minimum_fixture_sample=None,
        )

    def validate_closed_lift_fixture(self, arm_q: np.ndarray) -> None:
        """Recheck hand/fixture separation while the cube lifts off its support."""

        if self.fixture_checker is None:
            return
        values = np.asarray(arm_q, dtype=np.float64)
        if values.ndim != 2 or values.shape[1] != 7:
            raise ValueError("closed-lift fixture validation requires N x 7 arm joints")
        closed = np.repeat(self.finger_sweep[-1:, :], len(values), axis=0)
        route_q = np.concatenate((values, closed), axis=1)
        sphere_tensor = self.checker.robot_spheres(
            route_q,
            joint_names=self.active_joint_names,
        )
        fixture_hit = self.fixture_checker.first_collision(
            sphere_tensor,
            kinematics_config=self.checker.config.kinematics_config,
        )
        if fixture_hit is None:
            return
        penetration, fixture_link, fixture_sample = fixture_hit
        raise _BranchRejected(
            "closed_lift_fixture",
            f"{fixture_link}={penetration * 1000.0:.3f}mm penetration at "
            f"sample {fixture_sample}/{len(values) - 1}",
        )


def _world_cuboid_clearances(
    *,
    robot: dict[str, Any],
    q_samples: np.ndarray,
    scene: dict[str, Any],
    device_cfg,
    disabled_links: set[str],
    checker: CuroboKinematicCollisionChecker | None = None,
    sphere_array: np.ndarray | None = None,
) -> list[dict[tuple[str, str], float]]:
    """Return robot-sphere/cuboid pairs below the hard object margin."""

    values = np.asarray(q_samples, dtype=np.float64)
    active = checker or CuroboKinematicCollisionChecker(robot=robot, device_cfg=device_cfg)
    if sphere_array is None:
        spheres = active.robot_spheres(values).detach().cpu().numpy().reshape(len(values), -1, 4)
    else:
        spheres = np.asarray(sphere_array, dtype=np.float64)
        if spheres.ndim != 3 or spheres.shape[0] != len(values) or spheres.shape[2] != 4:
            raise ValueError("precomputed world-clearance spheres must have shape N x S x 4")
    sphere_link_indices = (
        active.config.kinematics_config.link_sphere_idx_map.detach().cpu().numpy().reshape(-1)
    )
    index_to_name = {
        value: name for name, value in active.config.kinematics_config.link_name_to_idx_map.items()
    }
    sphere_links = [index_to_name[int(index)] for index in sphere_link_indices]
    cuboids = []
    for object_name, document in scene.get("cuboid", {}).items():
        pose = np.asarray(document["pose"], dtype=np.float64)
        rotation = Rotation.from_quat([pose[4], pose[5], pose[6], pose[3]]).as_matrix()
        cuboids.append(
            (
                str(object_name),
                pose[:3],
                rotation,
                0.5 * np.asarray(document["dims"], dtype=np.float64),
            )
        )
    enabled_links = np.asarray(
        [name not in disabled_links for name in sphere_links],
        dtype=bool,
    )
    enabled = (spheres[..., 3] > 0.0) & enabled_links[None, :]
    result: list[dict[tuple[str, str], float]] = [{} for _ in values]
    for object_name, center, rotation, half_extents in cuboids:
        local = (spheres[..., :3] - center[None, None, :]) @ rotation
        offset = np.abs(local) - half_extents[None, None, :]
        signed_box_distance = np.linalg.norm(np.maximum(offset, 0.0), axis=-1) + np.minimum(
            np.max(offset, axis=-1),
            0.0,
        )
        clearances = signed_box_distance - spheres[..., 3]
        hits = np.argwhere(enabled & (clearances < OPEN_TRANSIT_OBJECT_CLEARANCE_M))
        for sample_index, sphere_index in hits:
            link_name = sphere_links[int(sphere_index)]
            key = (link_name, object_name)
            clearance = float(clearances[sample_index, sphere_index])
            grouped = result[int(sample_index)]
            grouped[key] = min(grouped.get(key, float("inf")), clearance)
    return result


def _planner(robot, scene: dict[str, Any], *, max_goalset: int, seed: int):
    import torch
    from curobo.motion_planner import MotionPlanner, MotionPlannerCfg
    from curobo.types import DeviceCfg

    if not torch.cuda.is_available():
        raise RuntimeError("CuRobo tabletop planning requires a CUDA device")
    device_cfg = DeviceCfg(device=torch.device("cuda:0"), dtype=torch.float32)
    config = MotionPlannerCfg.create(
        robot=robot,
        scene_model=scene,
        collision_cache={"cuboid": 4, "mesh": 1},
        max_goalset=max_goalset,
        device_cfg=device_cfg,
        num_ik_seeds=IK_SEEDS,
        num_trajopt_seeds=4,
        position_tolerance=PREGRASP_IK_POSITION_TOLERANCE_M,
        orientation_tolerance=PREGRASP_IK_ORIENTATION_TOLERANCE_RAD,
        self_collision_check=True,
        use_cuda_graph=True,
        random_seed=seed,
        optimizer_collision_activation_distance=COLLISION_ACTIVATION_DISTANCE_M,
        interpolation_dt=TRAJECTORY_INTERPOLATION_DT_S,
        interpolation_buffer_size=1000,
    )
    return MotionPlanner(config), device_cfg


def _batched_pregrasp_ik_solver(
    robot,
    scene: dict[str, Any],
    *,
    candidate_count: int,
    seed: int,
    device_cfg,
):
    """Build an IK-only GPU batch with one independent row per candidate."""

    from curobo.inverse_kinematics import InverseKinematics, InverseKinematicsCfg

    if candidate_count <= 0:
        raise ValueError("batched pregrasp IK requires at least one candidate")
    config = InverseKinematicsCfg.create(
        robot=robot,
        optimizer_configs=["ik/lbfgs_ik.yml"],
        scene_model=scene,
        collision_cache={"cuboid": 4, "mesh": 1},
        device_cfg=device_cfg,
        num_seeds=IK_SEEDS,
        position_tolerance=PREGRASP_IK_POSITION_TOLERANCE_M,
        orientation_tolerance=PREGRASP_IK_ORIENTATION_TOLERANCE_RAD,
        self_collision_check=True,
        max_batch_size=candidate_count,
        max_goalset=1,
        use_cuda_graph=True,
        random_seed=seed,
        optimizer_collision_activation_distance=COLLISION_ACTIVATION_DISTANCE_M,
    )
    return InverseKinematics(config)


class ReusableOpenPlanner:
    """Retain one fixed-shape open-hand MotionPlanner across task boundaries."""

    def __init__(self) -> None:
        self._planner = None
        self._device_cfg = None
        self._configuration_key: str | None = None
        self._seed: int | None = None

    @staticmethod
    def _robot_configs(planner) -> tuple[Any, ...]:
        configs: list[Any] = []
        seen: set[int] = set()
        rollouts = [
            *planner.ik_solver.core.get_all_rollout_instances(),
            *planner.trajopt_solver.core.get_all_rollout_instances(),
        ]
        if planner.graph_planner is not None:
            rollouts.extend(planner.graph_planner.get_all_rollout_instances())
        for rollout in rollouts:
            config = rollout.transition_model.robot_model.config
            if id(config) in seen:
                continue
            seen.add(id(config))
            configs.append(config)
        if not configs:
            raise RuntimeError("CuRobo MotionPlanner exposes no robot configuration buffers")
        return tuple(configs)

    @staticmethod
    def _assert_compatible(target, source) -> None:
        import torch

        left = target.kinematics_config
        right = source.kinematics_config
        if left.joint_names != right.joint_names:
            raise RuntimeError("reusable planner active joints changed")
        if left.tool_frames != right.tool_frames:
            raise RuntimeError("reusable planner tool frames changed")
        if left.lock_jointstate.joint_names != right.lock_jointstate.joint_names:
            raise RuntimeError("reusable planner locked joints changed")
        for name in (
            "fixed_transforms",
            "link_map",
            "joint_map",
            "joint_map_type",
            "tool_frame_map",
            "joint_offset_map",
            "link_spheres",
            "link_sphere_idx_map",
        ):
            if getattr(left, name).shape != getattr(right, name).shape:
                raise RuntimeError(f"reusable planner changed CuRobo {name} shape")
        if target.self_collision_config.sphere_padding.shape != (
            source.self_collision_config.sphere_padding.shape
        ):
            raise RuntimeError("reusable planner changed self-collision padding shape")
        if not torch.equal(
            target.self_collision_config.collision_pairs,
            source.self_collision_config.collision_pairs,
        ):
            raise RuntimeError("reusable planner changed self-collision pair topology")

    def acquire(
        self,
        robot,
        scene: dict[str, Any],
        *,
        seed: int,
        configuration_key: str,
    ) -> tuple[Any, Any, dict[str, Any]]:
        """Create or value-update one planner while preserving CUDA graph allocations."""

        started = time.monotonic()
        if self._planner is None or self._seed != seed:
            self.close()
            planner, device_cfg = _planner(
                robot,
                scene,
                max_goalset=1,
                seed=seed,
            )
            self._planner = planner
            self._device_cfg = device_cfg
            self._configuration_key = configuration_key
            self._seed = seed
            return (
                planner,
                device_cfg,
                {
                    "reused": False,
                    "configuration_changed": True,
                    "topology_rebuilt": False,
                    "elapsed_s": time.monotonic() - started,
                },
            )
        if self._configuration_key != configuration_key:
            from curobo._src.geom.types import SceneCfg

            source = robot.kinematics
            targets = self._robot_configs(self._planner)
            try:
                for target in targets:
                    self._assert_compatible(target, source)
            except RuntimeError:
                self.close()
                planner, device_cfg = _planner(
                    robot,
                    scene,
                    max_goalset=1,
                    seed=seed,
                )
                self._planner = planner
                self._device_cfg = device_cfg
                self._configuration_key = configuration_key
                self._seed = seed
                return (
                    planner,
                    device_cfg,
                    {
                        "reused": False,
                        "configuration_changed": True,
                        "topology_rebuilt": True,
                        "elapsed_s": time.monotonic() - started,
                    },
                )
            for target in targets:
                target.kinematics_config.copy_(source.kinematics_config)
                target.self_collision_config.sphere_padding.copy_(
                    source.self_collision_config.sphere_padding
                )
            self._planner.update_world(SceneCfg.create(scene))
            self._planner.reset_seed()
            self._configuration_key = configuration_key
            changed = True
        else:
            self._planner.reset_seed()
            changed = False
        return (
            self._planner,
            self._device_cfg,
            {
                "reused": True,
                "configuration_changed": changed,
                "topology_rebuilt": False,
                "elapsed_s": time.monotonic() - started,
            },
        )

    def close(self) -> None:
        planner = self._planner
        self._planner = None
        self._device_cfg = None
        self._configuration_key = None
        self._seed = None
        if planner is not None:
            _cleanup(planner)


def _joint_state(device_cfg, values: np.ndarray, names: tuple[str, ...]):
    from curobo.types import JointState

    return JointState.from_position(
        device_cfg.to_device(np.asarray(values, dtype=np.float64)).unsqueeze(0),
        joint_names=list(names),
    )


def _fresh_branch_start_state(device_cfg, reference_model_q: np.ndarray, *, arm: str):
    """Create an independent start state for one CuRobo branch attempt.

    CuRobo planning calls may mutate the supplied ``JointState`` in place.  A
    branch must therefore never reuse the state used for IK enumeration or by
    an earlier rejected branch.
    """

    return _joint_state(
        device_cfg,
        np.asarray(reference_model_q, dtype=np.float64).copy(),
        arm_joint_names(arm),
    )


def _base_T_torso(planner, state) -> np.ndarray:
    pose = planner.compute_kinematics(state).tool_poses["torso_link"]
    return pose.get_matrix()[0].detach().cpu().numpy()


def _base_T_grasp(planner, state, *, arm: str) -> np.ndarray:
    pose = planner.compute_kinematics(state).tool_poses[grasp_frame(arm)]
    return pose.get_matrix()[0].detach().cpu().numpy()


def _use_moving_grasp_frame_only(robot: dict[str, Any], *, arm: str) -> None:
    """Remove the static torso query frame before pose/grasp optimization.

    The first kinematics-only planner exposes ``torso_link`` so the locked
    seated torso pose can be read.  The subsequent planners optimize only the
    selected arm, so the torso cannot move and does not belong in their goal set.
    This is especially important for ``plan_grasp``, which applies approach
    offsets to every goal frame.
    """

    robot["kinematics"]["tool_frames"] = [grasp_frame(arm)]


def _goalset(matrices: list[np.ndarray], device_cfg, *, arm: str):
    import torch
    from curobo.types import GoalToolPose

    count = len(matrices)
    positions = torch.empty((1, 1, 1, count, 3), device=device_cfg.device)
    quaternions = torch.empty((1, 1, 1, count, 4), device=device_cfg.device)
    for index, matrix in enumerate(matrices):
        positions[0, 0, 0, index] = device_cfg.to_device(matrix[:3, 3])
        quaternion_xyzw = Rotation.from_matrix(matrix[:3, :3]).as_quat()
        quaternions[0, 0, 0, index] = device_cfg.to_device(
            [quaternion_xyzw[3], *quaternion_xyzw[:3]]
        )
    return GoalToolPose([grasp_frame(arm)], positions, quaternions)


def _batched_pose_goals(matrices: list[np.ndarray], device_cfg, *, arm: str):
    """Return one independent single-pose IK problem per candidate."""

    from curobo.types import GoalToolPose, Pose

    values = np.asarray(matrices, dtype=np.float64)
    if values.ndim != 3 or values.shape[1:] != (4, 4) or len(values) == 0:
        raise ValueError("batched pregrasp goals require a non-empty N x 4 x 4 array")
    pose = Pose.from_matrix(device_cfg.to_device(values))
    return GoalToolPose.from_poses(
        {grasp_frame(arm): pose},
        ordered_tool_frames=[grasp_frame(arm)],
    )


def _as_numpy(value) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def _pregrasp_matrix(grasp_matrix: np.ndarray, approach_distance_m: float) -> np.ndarray:
    offset = np.eye(4, dtype=np.float64)
    offset[2, 3] = -approach_distance_m
    return np.asarray(grasp_matrix, dtype=np.float64) @ offset


def _repeat_joint_state(state, count: int):
    from curobo.types import JointState

    return JointState.from_position(
        state.position.repeat(count, 1),
        joint_names=state.joint_names,
    )


def _enumerate_pregrasp_branches(
    solver,
    goals,
    state,
    *,
    candidate_count: int,
) -> tuple[list[_PregraspBranch], Any]:
    """Return up to 16 dedicated collision-valid IK branches per candidate."""

    if candidate_count <= 0:
        raise ValueError("pregrasp IK candidate count must be positive")
    batch_state = _repeat_joint_state(state, candidate_count)
    result = solver.solve_pose(
        goals,
        return_seeds=IK_SEEDS,
        current_state=batch_state,
    )
    success = _as_numpy(result.success).astype(bool)
    solutions = _as_numpy(result.solution)
    if success.ndim != 2 or success.shape[0] != candidate_count:
        raise RuntimeError(f"CuRobo pregrasp IK returned invalid success shape {success.shape}")
    if solutions.shape != (*success.shape, 7):
        raise RuntimeError(f"CuRobo pregrasp IK returned invalid solution shape {solutions.shape}")
    position_error = _as_numpy(result.position_error).reshape(*success.shape, -1).max(axis=2)
    rotation_error = _as_numpy(result.rotation_error).reshape(*success.shape, -1).max(axis=2)
    reference = _as_numpy(state.position).reshape(-1)

    branches: list[_PregraspBranch] = []
    for candidate_local_index in range(candidate_count):
        valid = np.flatnonzero(success[candidate_local_index])
        ordered = sorted(
            (int(index) for index in valid),
            key=lambda index: float(
                np.linalg.norm(solutions[candidate_local_index, index] - reference)
            ),
        )
        unique: list[np.ndarray] = []
        for solver_seed_index in ordered:
            solution = np.asarray(
                solutions[candidate_local_index, solver_seed_index],
                dtype=np.float64,
            )
            if any(
                float(np.max(np.abs(solution - previous))) <= IK_BRANCH_DUPLICATE_TOLERANCE_RAD
                for previous in unique
            ):
                continue
            unique.append(solution)
            branches.append(
                _PregraspBranch(
                    candidate_local_index=candidate_local_index,
                    solver_seed_index=solver_seed_index,
                    model_q_rad=solution,
                    position_error_m=float(
                        position_error[candidate_local_index, solver_seed_index]
                    ),
                    rotation_error_rad=float(
                        rotation_error[candidate_local_index, solver_seed_index]
                    ),
                )
            )
    branches.sort(key=lambda branch: float(np.linalg.norm(branch.model_q_rad - reference)))
    return branches, result


def _try_branch_pool(
    branches: list[_PregraspBranch],
    *,
    candidate_ids: list[str],
    attempt: Callable[[_PregraspBranch], Any],
    report: Callable[[str], None],
) -> tuple[
    _PregraspBranch | None,
    Any | None,
    list[dict[str, Any]],
]:
    """Try the finite IK branch pool without discarding a candidate early."""

    from collections import Counter

    failures: list[dict[str, Any]] = []
    totals = Counter(branch.candidate_local_index for branch in branches)
    attempted: Counter[int] = Counter()
    for pool_index, branch in enumerate(branches, start=1):
        if not 0 <= branch.candidate_local_index < len(candidate_ids):
            raise RuntimeError("CuRobo returned an invalid pregrasp candidate index")
        candidate_id = candidate_ids[branch.candidate_local_index]
        attempted[branch.candidate_local_index] += 1
        candidate_branch_index = attempted[branch.candidate_local_index]
        candidate_branch_count = totals[branch.candidate_local_index]
        report(
            f"trying {candidate_id} IK branch "
            f"{candidate_branch_index}/{candidate_branch_count} "
            f"(solver seed {branch.solver_seed_index})"
        )
        try:
            result = attempt(branch)
        except _BranchRejected as rejection:
            failure = {
                "candidate_id": candidate_id,
                "pool_branch_index": pool_index,
                "candidate_branch_index": candidate_branch_index,
                "candidate_branch_count": candidate_branch_count,
                "solver_seed_index": branch.solver_seed_index,
                "stage": rejection.stage,
                "reason": rejection.reason,
            }
            failures.append(failure)
            report(
                f"rejected {candidate_id} IK branch "
                f"{candidate_branch_index}/{candidate_branch_count} "
                f"at {rejection.stage}: {rejection.reason}"
            )
            continue
        return branch, result, failures
    return None, None, failures


def _pregrasp_endpoint_self_collision_reasons(
    branches: list[_PregraspBranch],
    *,
    checker: CuroboKinematicCollisionChecker,
) -> list[str | None]:
    """Name exact strict endpoint collisions before trajectory optimization.

    The open-transit optimizer deliberately scopes world spheres to the moving
    hand.  Its IK pool can therefore contain an endpoint that the independent
    strict full-robot checker must reject.  No trajectory can make an invalid
    terminal configuration valid, so detect those endpoints in one batched FK
    call instead of spending seconds planning a route that will be discarded.
    """

    if not branches:
        return []
    samples = checker.self_collision_pair_penetrations(
        np.stack([branch.model_q_rad for branch in branches], axis=0)
    )
    reasons: list[str | None] = []
    for pairs in samples:
        if not pairs:
            reasons.append(None)
            continue
        pair, penetration = max(pairs.items(), key=lambda item: item[1])
        reasons.append(f"{pair[0]}/{pair[1]}={penetration * 1000.0:.3f}mm")
    return reasons


def _batched_ik_failure_diagnostic(
    *,
    result,
    robot: dict[str, Any],
    scene: dict[str, Any],
    candidates: list[dict[str, Any]],
    device_cfg,
    disabled_collision_links: set[str],
) -> str:
    """Name physical constraints behind an otherwise opaque batched-IK failure."""

    success = _as_numpy(result.success).astype(bool)
    if success.ndim != 2 or success.shape[0] != len(candidates):
        return f"IK diagnostic returned invalid success shape {success.shape}"
    solutions = _as_numpy(result.solution)
    if solutions.shape != (*success.shape, 7):
        return f"IK diagnostic returned invalid solution shape {solutions.shape}"
    position_error = _as_numpy(result.position_error).reshape(*success.shape, -1).max(axis=2)
    rotation_error = _as_numpy(result.rotation_error).reshape(*success.shape, -1).max(axis=2)
    converged = np.argwhere(
        (position_error < PREGRASP_IK_POSITION_TOLERANCE_M)
        & (rotation_error < PREGRASP_IK_ORIENTATION_TOLERANCE_RAD)
    )
    best_position_mm = float(np.min(position_error) * 1000.0)
    best_rotation_deg = float(np.rad2deg(np.min(rotation_error)))
    if len(converged) == 0:
        return (
            "IK diagnostic found no Cartesian-converged branch: "
            f"best position error={best_position_mm:.3f}mm, "
            f"best rotation error={best_rotation_deg:.3f}deg"
        )

    converged_q = np.stack(
        [solutions[candidate_index, seed_index] for candidate_index, seed_index in converged]
    )
    pair_samples = _self_collision_pair_penetrations(
        robot=robot,
        q_samples=converged_q,
        device_cfg=device_cfg,
    )
    world_samples = _world_cuboid_clearances(
        robot=robot,
        q_samples=converged_q,
        scene=scene,
        device_cfg=device_cfg,
        disabled_links=disabled_collision_links,
    )
    branches: list[str] = []
    seen: set[tuple[str, tuple[tuple[str, str], ...], tuple[tuple[str, str], ...]]] = set()
    for indices, pair_depths, world_clearances in zip(
        converged, pair_samples, world_samples, strict=True
    ):
        candidate_index, _seed_index = (int(value) for value in indices)
        candidate_id = str(candidates[candidate_index]["candidate_id"])
        filtered = {
            pair: depth
            for pair, depth in pair_depths.items()
            if not disabled_collision_links.intersection(pair)
        }
        ordered = sorted(filtered.items(), key=lambda item: -item[1])
        ordered_world = sorted(world_clearances.items(), key=lambda item: item[1])
        key = (
            candidate_id,
            tuple(pair for pair, _depth in ordered),
            tuple(pair for pair, _clearance in ordered_world),
        )
        if key in seen:
            continue
        seen.add(key)
        if ordered:
            pair_text = ", ".join(
                f"{first}/{second}={depth * 1000.0:.3f}mm"
                for (first, second), depth in ordered[:4]
            )
            branches.append(f"{candidate_id} self-collision [{pair_text}]")
        elif ordered_world:
            world_text = ", ".join(
                f"{link}/{object_name}={clearance * 1000.0:+.3f}mm clearance"
                for (link, object_name), clearance in ordered_world[:4]
            )
            branches.append(f"{candidate_id} scene collision/proximity [{world_text}]")
        else:
            branches.append(
                f"{candidate_id} converged but was infeasible; "
                "no enabled self-collision pair or nearby scene cuboid was identified"
            )
        if len(branches) == 3:
            break
    return (
        f"IK diagnostic found {len(converged)}/{int(success.size)} "
        "Cartesian-converged but infeasible branches: " + "; ".join(branches)
    )


def _pose_failure_reason(planner, goal, state, result) -> str:
    """Describe a failed pose plan using CuRobo's lowest useful result level."""

    import torch

    if result is None:
        ik_result = planner.ik_solver.solve_pose(
            goal,
            return_seeds=IK_SEEDS,
            current_state=state,
        )
        success_count = int(torch.count_nonzero(ik_result.success).item())
        seed_count = int(ik_result.success.numel())
        position_error_mm = float(torch.min(ik_result.position_error).item() * 1000.0)
        rotation_error_deg = float(torch.rad2deg(torch.min(ik_result.rotation_error)).item())
        return (
            "pose planning returned no trajectory; collision-constrained IK "
            f"successes={success_count}/{seed_count}, best position error="
            f"{position_error_mm:.3f}mm, best rotation error={rotation_error_deg:.3f}deg"
        )
    success_count = int(torch.count_nonzero(result.success).item())
    position_error = getattr(result, "position_error", None)
    rotation_error = getattr(result, "rotation_error", None)
    details = [f"trajectory successes={success_count}/{int(result.success.numel())}"]
    if position_error is not None:
        details.append(
            f"best position error={float(torch.min(position_error).item() * 1000.0):.3f}mm"
        )
    if rotation_error is not None:
        details.append(
            f"best rotation error={float(torch.rad2deg(torch.min(rotation_error)).item()):.3f}deg"
        )
    return "pose trajectory optimization failed; " + ", ".join(details)


def _planned_trajectory(
    *,
    from_id: str,
    to_id: str,
    model_q: np.ndarray,
    native_dt: float,
    arm: str,
    offsets: dict[str, float],
    planning_time_s: float,
    maximum_velocity_rad_s: float,
) -> PlannedTrajectory:
    values = np.asarray(model_q, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 7 or len(values) < 2:
        raise RuntimeError(f"CuRobo returned invalid {from_id}->{to_id} trajectory {values.shape}")
    if not np.isfinite(maximum_velocity_rad_s) or maximum_velocity_rad_s <= 0.0:
        raise ValueError("maximum trajectory velocity must be positive and finite")
    peak = float(np.max(np.abs(np.diff(values, axis=0))) / native_dt)
    dt = native_dt * max(1.0, peak / maximum_velocity_rad_s)
    command = np.stack(
        [
            command_from_model_q(
                q,
                arm=arm,
                joint_position_offsets_rad=offsets,
            )
            for q in values
        ]
    )
    return PlannedTrajectory(
        from_pose_id=from_id,
        to_pose_id=to_id,
        sample_time_s=tuple(float(index * dt) for index in range(len(values))),
        command_q_rad=tuple(tuple(row) for row in command),
        model_q_rad=tuple(tuple(row) for row in values),
        planning_time_s=float(planning_time_s),
    )


def _result_trajectory(
    planner,
    result,
    *,
    arm: str,
    from_id: str,
    to_id: str,
    offsets: dict[str, float],
    maximum_velocity_rad_s: float,
):
    if result is None or not bool(result.success.any()):
        status = "none" if result is None else str(getattr(result, "status", "unsuccessful"))
        raise RuntimeError(f"CuRobo failed {from_id}->{to_id}: {status}")
    plan = result.get_interpolated_plan().reorder(list(arm_joint_names(arm)))
    values = np.asarray(plan.position.detach().cpu().numpy(), dtype=np.float64).squeeze()
    return _planned_trajectory(
        from_id=from_id,
        to_id=to_id,
        model_q=values,
        native_dt=_joint_state_dt(plan),
        arm=arm,
        offsets=offsets,
        planning_time_s=float(result.total_time),
        maximum_velocity_rad_s=maximum_velocity_rad_s,
    )


def _cleanup(planner) -> None:
    import torch

    if planner is not None:
        planner.destroy()
    gc.collect()
    torch.cuda.empty_cache()


def _validate_supported_escape_endpoint(
    *,
    start: np.ndarray,
    target: np.ndarray,
    endpoint: np.ndarray,
    down: np.ndarray,
    position_tolerance_m: float,
) -> tuple[float, float]:
    """Validate the FK endpoint against the Cartesian goal CuRobo accepted."""

    target_error_m = float(np.linalg.norm(endpoint[:3, 3] - target[:3, 3]))
    if target_error_m > position_tolerance_m:
        raise RuntimeError(
            "supported escape missed its requested Cartesian endpoint: "
            f"position error={target_error_m:.4f}m; "
            f"CuRobo tolerance={position_tolerance_m:.4f}m"
        )
    achieved_escape_m = -float(np.dot(down, endpoint[:3, 3] - start[:3, 3]))
    return achieved_escape_m, target_error_m


def plan_supported_escape(
    request: TabletopTaskRequest | CharucoSupportedEscapeRequest,
    *,
    progress: Callable[[str], None] | None = None,
) -> SupportedEscapePlan:
    """Lift the supported selected hand along the observed table normal."""

    report = progress or (lambda _message: None)
    arm = request.arm
    snapshot = request.observation.snapshot
    initial_fingers = snapshot.left_dex3_q_rad if arm == "left" else snapshot.right_dex3_q_rad
    strict_robot, reference_tuple = build_tabletop_robot_config(
        arm=arm,
        snapshot=snapshot,
        joint_position_offsets_rad=request.joint_position_offsets_rad,
        active_finger_q_rad=initial_fingers,
    )
    planner = None
    strict_checker = None
    started = time.monotonic()
    strict_model_resolution_s = 0.0
    optimizer_model_clone_s = 0.0
    optimizer_setup_s = 0.0
    pose_planning_s = 0.0
    try:
        import torch
        from curobo.types import DeviceCfg

        # Reading the fixed torso and live grasp transforms requires FK, not a
        # complete trajectory optimizer. Reuse the strict checker that is
        # required for the route validation below instead of cold-starting and
        # immediately destroying a throwaway MotionPlanner.
        device_cfg = DeviceCfg(device=torch.device("cuda:0"), dtype=torch.float32)
        stage_started = time.monotonic()
        strict_checker = CuroboKinematicCollisionChecker(robot=strict_robot, device_cfg=device_cfg)
        strict_model_resolution_s = time.monotonic() - stage_started
        names = arm_joint_names(arm)
        reference = np.asarray(reference_tuple)
        start_collision_samples = _self_collision_pair_penetrations(
            robot=strict_robot,
            q_samples=reference[None],
            device_cfg=device_cfg,
            checker=strict_checker,
        )
        _validate_strict_supported_escape_self_collision(start_collision_samples)
        state = _joint_state(device_cfg, reference, names)
        kinematics = strict_checker.kinematics.compute_kinematics(state)
        base_T_torso = kinematics.tool_poses["torso_link"].get_matrix()[0].detach().cpu().numpy()
        base_T_grasp = (
            kinematics.tool_poses[grasp_frame(arm)].get_matrix()[0].detach().cpu().numpy()
        )
        stage_started = time.monotonic()
        motion_robot = _resolved_motion_robot(
            checker=strict_checker,
            source_robot=strict_robot,
            arm=arm,
            scope_world_to_selected_hand=False,
        )
        optimizer_model_clone_s = time.monotonic() - stage_started
        if isinstance(request, CharucoSupportedEscapeRequest):
            scene = {}
            plane_point, down = _table_from_charuco_board(request, base_T_torso)
            plane_source = "fixed ChArUco board"
        else:
            scene = _base_scene(request, base_T_torso, include_cube=True)
            plane_point, _base_T_object, down = _table_from_resting_object(request, base_T_torso)
            plane_source = "resting AprilCube"
        stage_started = time.monotonic()
        planner, device_cfg = _planner(
            motion_robot,
            scene,
            max_goalset=1,
            seed=request.random_seed,
        )
        optimizer_setup_s = time.monotonic() - stage_started
        state = _joint_state(device_cfg, reference, names)
        target = base_T_grasp.copy()
        target[:3, 3] -= request.supported_escape_m * down
        from curobo.types import GoalToolPose, Pose, ToolPoseCriteria

        goal = GoalToolPose.from_poses(
            {grasp_frame(arm): Pose.from_matrix(device_cfg.to_device(target[None]))},
            ordered_tool_frames=[grasp_frame(arm)],
        )
        criterion = ToolPoseCriteria.linear_motion(
            axis="z", non_terminal_scale=1.0, project_distance_to_goal=False
        )
        planner.update_tool_pose_criteria({grasp_frame(arm): criterion})
        stage_started = time.monotonic()
        try:
            result = planner.plan_pose(goal, state, max_attempts=8)
        finally:
            pose_planning_s = time.monotonic() - stage_started
            planner.update_tool_pose_criteria({grasp_frame(arm): ToolPoseCriteria()})
        if result is None or not bool(result.success.any()):
            reason = _pose_failure_reason(planner, goal, state, result)
            raise RuntimeError(f"CuRobo failed __handoff__->clearance: {reason}")
        outbound = _result_trajectory(
            planner,
            result,
            arm=arm,
            from_id="__handoff__",
            to_id="clearance",
            offsets=request.joint_position_offsets_rad,
            maximum_velocity_rad_s=request.maximum_arm_velocity_rad_s,
        )
        outbound = _anchor_trajectory_start(
            outbound,
            command_q_rad=command_from_model_q(
                reference,
                arm=arm,
                joint_position_offsets_rad=request.joint_position_offsets_rad,
            ),
            model_q_rad=reference,
        )
        route_q = np.asarray(outbound.model_q_rad)
        strict_collision_samples = _self_collision_pair_penetrations(
            robot=strict_robot,
            q_samples=route_q,
            device_cfg=device_cfg,
            checker=strict_checker,
        )
        _validate_strict_supported_escape_self_collision(strict_collision_samples)
        endpoint_q = np.asarray(outbound.model_q_rad[-1])
        endpoint = _joint_state(device_cfg, endpoint_q, names)
        actual_target = _base_T_grasp(planner, endpoint, arm=arm)
        position_tolerance_m = float(planner.trajopt_solver.config.position_tolerance)
        achieved_escape_m, target_error_m = _validate_supported_escape_endpoint(
            start=base_T_grasp,
            target=target,
            endpoint=actual_target,
            down=down,
            position_tolerance_m=position_tolerance_m,
        )
        terminal_clearance = -float(np.dot(down, actual_target[:3, 3] - plane_point))
        route_clearance, route_link, route_sample = _local_plane_clearance(
            planner,
            route_q,
            arm=arm,
            plane_point=plane_point,
            down=down,
            include_payload=False,
        )
        start_clearance, _start_link, _start_sample = _local_plane_clearance(
            planner,
            route_q[:1],
            arm=arm,
            plane_point=plane_point,
            down=down,
            include_payload=False,
        )
        if route_clearance < start_clearance - COLLISION_ACTIVATION_DISTANCE_M:
            raise RuntimeError(
                f"supported escape drives local {arm}-hand geometry farther through "
                f"the observed table plane: minimum={route_clearance:.4f}m at "
                f"{route_link} sample {route_sample}, start={start_clearance:.4f}m"
            )
        inbound = _reverse_trajectory(outbound)
        report(
            f"CuRobo supported escape passed; requested lift={request.supported_escape_m:.4f}m, "
            f"achieved lift={achieved_escape_m:.4f}m, endpoint error={target_error_m:.4f}m "
            f"(CuRobo tolerance={position_tolerance_m:.4f}m), "
            f"G-frame plane clearance={terminal_clearance:.4f}m"
        )
        return SupportedEscapePlan(
            request_sha256=request.content_sha256,
            outbound=outbound,
            inbound=inbound,
            minimum_terminal_plane_clearance_m=terminal_clearance,
            planner_provenance={
                **model_source_hashes(),
                "curobo_commit": CUROBO_COMMIT,
                "elapsed_s": time.monotonic() - started,
                "stage_timings_s": {
                    "strict_model_resolution": strict_model_resolution_s,
                    "optimizer_model_clone": optimizer_model_clone_s,
                    "optimizer_setup": optimizer_setup_s,
                    "pose_planning": pose_planning_s,
                },
                "execution_maximum_arm_velocity_rad_s": (request.maximum_arm_velocity_rad_s),
                "supported_escape_requested_m": request.supported_escape_m,
                "supported_escape_achieved_m": achieved_escape_m,
                "supported_escape_endpoint_error_m": target_error_m,
                "curobo_position_tolerance_m": position_tolerance_m,
                "policy": (
                    f"straight table-normal escape from {plane_source}; "
                    "no fabricated table box; "
                    f"local {arm}-wrist/hand plane guard; exact reverse return"
                ),
                "local_plane_minimum_clearance_m": route_clearance,
                "local_plane_start_clearance_m": start_clearance,
                "local_plane_minimum_link": route_link,
                "local_plane_minimum_sample": route_sample,
                "self_collision_policy": (
                    "strict CuRobo self-collision from the exact measured start "
                    "through the complete supported escape; any enabled overlap "
                    "aborts with its link pair and penetration; exact reverse return"
                ),
                "arm": arm,
                "self_collision_start_state": "clear",
                "self_collision_strict_sample_count": len(strict_collision_samples),
            },
        )
    finally:
        _cleanup(planner)


def _snapshot_at_arm_q(
    request: TabletopTaskRequest, command_q: tuple[float, ...]
) -> RobotSnapshot:
    snapshot = request.planning_snapshot
    q29 = np.asarray(snapshot.measured_q29_rad).copy()
    q29[np.asarray(arm_indices(request.arm))] = np.asarray(command_q)
    return RobotSnapshot(
        measured_q29_rad=tuple(q29),
        left_dex3_q_rad=snapshot.left_dex3_q_rad,
        right_dex3_q_rad=snapshot.right_dex3_q_rad,
    )


def _cuboid_cover_spheres(dimensions_m: tuple[float, float, float]) -> np.ndarray:
    """Conservatively cover a cuboid with a deterministic 3 x 3 x 3 sphere grid."""

    dimensions = np.asarray(dimensions_m, dtype=np.float64)
    if dimensions.shape != (3,) or not np.all(np.isfinite(dimensions)):
        raise ValueError("attached cuboid dimensions must contain three finite values")
    if np.any(dimensions <= 0.0):
        raise ValueError("attached cuboid dimensions must be positive")
    cell_half_extents = dimensions / 6.0
    axes = [np.linspace(-dimension / 3.0, dimension / 3.0, 3) for dimension in dimensions]
    centers = np.asarray(
        [[x, y, z] for x in axes[0] for y in axes[1] for z in axes[2]],
        dtype=np.float64,
    )
    radius = float(np.linalg.norm(cell_half_extents))
    return np.column_stack((centers, np.full(len(centers), radius)))


def _attach_cube(
    planner,
    state,
    grasp_T_object: np.ndarray,
    dimensions_m: tuple[float, float, float],
    *,
    arm: str,
) -> int:
    import torch
    import trimesh

    source = torch.as_tensor(
        _cuboid_cover_spheres(dimensions_m),
        device=state.position.device,
        dtype=state.position.dtype,
    )
    centers = trimesh.transform_points(source[:, :3].detach().cpu().numpy(), grasp_T_object)
    source[:, :3] = torch.as_tensor(centers, device=source.device, dtype=source.dtype)
    for current in (
        planner.ik_solver.core.attachment_manager,
        planner.trajopt_solver.core.attachment_manager,
    ):
        current.update(
            source,
            state,
            link_name=attachment_link(arm),
            world_objects_pose_offset=None,
        )
    return len(source)


def _cspace_failure_reason(result) -> str:
    if result is None:
        return "CuRobo returned no joint-space trajectory"
    success = _as_numpy(result.success).astype(bool).reshape(-1)
    return (
        "CuRobo joint-space trajectory optimization failed; "
        f"successes={int(np.count_nonzero(success))}/{len(success)}"
    )


def _plan_approach_trajectory(
    *,
    planner,
    device_cfg,
    state,
    reference_command_q: np.ndarray,
    reference_model_q: np.ndarray,
    pregrasp_model_q: np.ndarray,
    request: TabletopTaskRequest,
    arm: str,
) -> PlannedTrajectory:
    """Plan one clearance-to-pregrasp IK branch without weakening validation."""

    pregrasp_state = _joint_state(device_cfg, pregrasp_model_q, arm_joint_names(arm))
    approach_result = planner.plan_cspace(
        goal_state=pregrasp_state,
        current_state=state,
        max_attempts=5,
        enable_graph_attempt=1,
    )
    if approach_result is None or not bool(approach_result.success.any()):
        raise _BranchRejected(
            "clearance_to_pregrasp",
            _cspace_failure_reason(approach_result),
        )
    approach = _result_trajectory(
        planner,
        approach_result,
        arm=arm,
        from_id="clearance",
        to_id="move_to_pregrasp",
        offsets=request.joint_position_offsets_rad,
        maximum_velocity_rad_s=request.maximum_arm_velocity_rad_s,
    )
    approach = _anchor_trajectory_start(
        approach,
        command_q_rad=reference_command_q,
        model_q_rad=reference_model_q,
    )

    return approach


def _validate_open_route(
    *,
    planner,
    device_cfg,
    route_q: np.ndarray,
    open_robot: dict[str, Any],
    strict_checker: CuroboKinematicCollisionChecker,
    request: TabletopTaskRequest,
    base_T_torso: np.ndarray,
    plane_point: np.ndarray,
    down: np.ndarray,
    arm: str,
    disabled_cube_links: set[str],
) -> tuple[float, str, int]:
    """Apply strict self, object, fixture, and table checks to an open-hand route."""

    open_route_q = np.asarray(route_q, dtype=np.float64)
    route_self_collisions = _self_collision_pair_penetrations(
        robot=open_robot,
        q_samples=open_route_q,
        device_cfg=device_cfg,
        checker=strict_checker,
    )
    for sample_index, pairs in enumerate(route_self_collisions):
        if not pairs:
            continue
        pair, penetration = max(pairs.items(), key=lambda item: item[1])
        raise _BranchRejected(
            "open_route_strict_self_collision",
            f"{pair[0]}/{pair[1]}={penetration * 1000.0:.3f}mm at sample {sample_index}",
        )

    cube_scene = _base_scene(
        request,
        base_T_torso,
        include_cube=True,
        include_environment_cuboids=False,
    )
    route_cube_clearances = _world_cuboid_clearances(
        robot=open_robot,
        q_samples=open_route_q,
        scene=cube_scene,
        device_cfg=device_cfg,
        disabled_links=disabled_cube_links,
        checker=strict_checker,
    )
    closest_cube = None
    for sample_index, clearances in enumerate(route_cube_clearances):
        for (link_name, object_name), clearance in clearances.items():
            if closest_cube is None or clearance < closest_cube[3]:
                closest_cube = (sample_index, link_name, object_name, clearance)
    if closest_cube is not None:
        sample_index, link_name, object_name, clearance = closest_cube
        raise _BranchRejected(
            "open_route_strict_cube_collision",
            f"{link_name}/{object_name}={clearance * 1000.0:+.3f}mm "
            f"clearance at sample {sample_index}; required "
            f"{OPEN_TRANSIT_OBJECT_CLEARANCE_M * 1000.0:.3f}mm",
        )

    environment_scene = _base_scene(
        request,
        base_T_torso,
        include_cube=False,
        include_environment_cuboids=True,
    )
    environment_scene.get("cuboid", {}).pop("open_transit_table_patch", None)
    route_environment_clearances = _world_cuboid_clearances(
        robot=open_robot,
        q_samples=open_route_q,
        scene=environment_scene,
        device_cfg=device_cfg,
        disabled_links=set(),
        checker=strict_checker,
    )
    closest_environment = None
    for sample_index, clearances in enumerate(route_environment_clearances):
        for (link_name, object_name), clearance in clearances.items():
            if closest_environment is None or clearance < closest_environment[3]:
                closest_environment = (sample_index, link_name, object_name, clearance)
    if closest_environment is not None:
        sample_index, link_name, object_name, clearance = closest_environment
        raise _BranchRejected(
            "open_route_environment_collision",
            f"{link_name}/{object_name}={clearance * 1000.0:+.3f}mm "
            f"clearance at sample {sample_index}; required "
            f"{OPEN_TRANSIT_OBJECT_CLEARANCE_M * 1000.0:.3f}mm",
        )

    open_clearance, open_link, open_sample = _local_plane_clearance(
        planner,
        open_route_q,
        arm=arm,
        plane_point=plane_point,
        down=down,
        include_payload=False,
    )
    if open_clearance < request.minimum_hand_plane_clearance_m:
        raise _BranchRejected(
            "open_route_table_plane",
            f"clearance={open_clearance:.4f}m at {open_link} sample {open_sample}; "
            f"required={request.minimum_hand_plane_clearance_m:.4f}m",
        )
    return open_clearance, open_link, open_sample


def _validate_open_route_segments(
    *,
    planner,
    device_cfg,
    transit_q: np.ndarray,
    grasp_q: np.ndarray,
    open_robot: dict[str, Any],
    strict_checker: CuroboKinematicCollisionChecker,
    request: TabletopTaskRequest,
    base_T_torso: np.ndarray,
    plane_point: np.ndarray,
    down: np.ndarray,
    arm: str,
) -> tuple[float, str, int]:
    """Keep grasp-contact exceptions out of clearance-to-pregrasp transit."""

    transit = np.asarray(transit_q, dtype=np.float64)
    grasp = np.asarray(grasp_q, dtype=np.float64)
    segments = (
        (transit, set(), 0),
        (grasp, set(_object_contact_links(arm)), len(transit) - 1),
    )
    clearances: list[tuple[float, str, int]] = []
    for route, disabled_cube_links, sample_offset in segments:
        clearance, link, sample = _validate_open_route(
            planner=planner,
            device_cfg=device_cfg,
            route_q=route,
            open_robot=open_robot,
            strict_checker=strict_checker,
            request=request,
            base_T_torso=base_T_torso,
            plane_point=plane_point,
            down=down,
            arm=arm,
            disabled_cube_links=disabled_cube_links,
        )
        clearances.append((clearance, link, sample + sample_offset))
    return min(clearances, key=lambda value: value[0])


def _plan_open_branch(
    *,
    planner,
    device_cfg,
    state,
    reference_command_q: np.ndarray,
    reference_model_q: np.ndarray,
    pregrasp_model_q: np.ndarray,
    grasp_matrix: np.ndarray,
    open_robot: dict[str, Any],
    strict_checker: CuroboKinematicCollisionChecker,
    request: TabletopTaskRequest,
    base_T_torso: np.ndarray,
    plane_point: np.ndarray,
    down: np.ndarray,
    arm: str,
) -> _OpenBranchPlan:
    """Plan and independently validate one exact open-hand IK branch."""

    approach = _plan_approach_trajectory(
        planner=planner,
        device_cfg=device_cfg,
        state=state,
        reference_command_q=reference_command_q,
        reference_model_q=reference_model_q,
        pregrasp_model_q=pregrasp_model_q,
        request=request,
        arm=arm,
    )
    actual_pregrasp_model_q = np.asarray(approach.model_q_rad[-1], dtype=np.float64)
    actual_pregrasp_state = _joint_state(device_cfg, actual_pregrasp_model_q, arm_joint_names(arm))
    grasp_goal = _goalset([grasp_matrix], device_cfg, arm=arm)
    from curobo.types import ToolPoseCriteria

    criterion = ToolPoseCriteria.linear_motion(
        axis="z",
        non_terminal_scale=1.0,
        project_distance_to_goal=True,
    )
    contact_links = list(_object_contact_links(arm))
    planner.update_tool_pose_criteria({grasp_frame(arm): criterion})
    planner.disable_link_collision(contact_links)
    try:
        grasp_result = planner.plan_pose(grasp_goal, actual_pregrasp_state, max_attempts=8)
    finally:
        planner.enable_link_collision(contact_links)
        planner.update_tool_pose_criteria({grasp_frame(arm): ToolPoseCriteria()})
    if grasp_result is None or not bool(grasp_result.success.any()):
        raise _BranchRejected(
            "linear_grasp_approach",
            _pose_failure_reason(planner, grasp_goal, actual_pregrasp_state, grasp_result),
        )
    grasp = _result_trajectory(
        planner,
        grasp_result,
        arm=arm,
        from_id="move_to_pregrasp",
        to_id="grasp_approach",
        offsets=request.joint_position_offsets_rad,
        maximum_velocity_rad_s=request.maximum_arm_velocity_rad_s,
    )
    grasp = _anchor_trajectory_start(
        grasp,
        command_q_rad=np.asarray(approach.command_q_rad[-1]),
        model_q_rad=actual_pregrasp_model_q,
    )

    # Fingertip/cube contact is intentional only on the final straight grasp
    # segment. The clearance-to-pregrasp transit must retain every cube link.
    open_clearance, open_link, open_sample = _validate_open_route_segments(
        planner=planner,
        device_cfg=device_cfg,
        transit_q=np.asarray(approach.model_q_rad),
        grasp_q=np.asarray(grasp.model_q_rad),
        open_robot=open_robot,
        strict_checker=strict_checker,
        request=request,
        base_T_torso=base_T_torso,
        plane_point=plane_point,
        down=down,
        arm=arm,
    )
    return _OpenBranchPlan(
        approach=approach,
        grasp=grasp,
        minimum_plane_clearance_m=open_clearance,
        minimum_plane_link=open_link,
        minimum_plane_sample=open_sample,
    )


def _plan_attached_lift(
    *,
    request: TabletopTaskRequest,
    selected: dict[str, Any],
    close_target_q: np.ndarray,
    contact_command_q: tuple[float, ...],
    contact_model_q: np.ndarray,
    base_T_torso: np.ndarray,
    plane_point: np.ndarray,
    down: np.ndarray,
    arm: str,
    fixed_close_validator: _FixedCloseSweepValidator,
    contact_snapshot: RobotSnapshot | None = None,
    base_T_object_override: np.ndarray | None = None,
    base_T_detected_object_override: np.ndarray | None = None,
) -> _LiftBranchPlan:
    """Plan with the descriptor close target; live measured fingers are rechecked later."""

    planner = None
    try:
        if contact_snapshot is None:
            contact_snapshot = _snapshot_at_arm_q(request, contact_command_q)
        close_target_robot, _ = build_tabletop_robot_config(
            arm=arm,
            snapshot=contact_snapshot,
            joint_position_offsets_rad=request.joint_position_offsets_rad,
            active_finger_q_rad=tuple(close_target_q),
        )
        _use_moving_grasp_frame_only(close_target_robot, arm=arm)
        attached_scene = _attached_lift_scene(
            request,
            base_T_torso,
            base_T_object_override=base_T_object_override,
            base_T_detected_object_override=base_T_detected_object_override,
            plane_point_override=(
                plane_point if base_T_object_override is not None else None
            ),
            down_override=down if base_T_object_override is not None else None,
        )
        planner, device_cfg = _planner(
            close_target_robot,
            attached_scene,
            max_goalset=1,
            seed=request.random_seed,
        )
        contact_state = _joint_state(device_cfg, contact_model_q, arm_joint_names(arm))
        object_T_grasp = _candidate_transform(selected)
        sphere_count = _attach_cube(
            planner,
            contact_state,
            invert_transform(object_T_grasp),
            request.object_dimensions_m,
            arm=arm,
        )
        contact_pose = _base_T_grasp(planner, contact_state, arm=arm)
        lift_pose = contact_pose.copy()
        lift_pose[:3, 3] -= request.lift_m * down
        from curobo.types import GoalToolPose, Pose, ToolPoseCriteria

        lift_goal = GoalToolPose.from_poses(
            {grasp_frame(arm): Pose.from_matrix(device_cfg.to_device(lift_pose[None]))},
            ordered_tool_frames=[grasp_frame(arm)],
        )
        criterion = ToolPoseCriteria.linear_motion(
            axis="z",
            non_terminal_scale=1.0,
            project_distance_to_goal=False,
        )
        planner.update_tool_pose_criteria({grasp_frame(arm): criterion})
        try:
            lift_result = planner.plan_pose(lift_goal, contact_state, max_attempts=8)
        finally:
            planner.update_tool_pose_criteria({grasp_frame(arm): ToolPoseCriteria()})
        if lift_result is None or not bool(lift_result.success.any()):
            raise _BranchRejected(
                "attached_payload_lift",
                _pose_failure_reason(planner, lift_goal, contact_state, lift_result),
            )
        lift = _result_trajectory(
            planner,
            lift_result,
            arm=arm,
            from_id="grasp_approach",
            to_id="payload_lift",
            offsets=request.joint_position_offsets_rad,
            maximum_velocity_rad_s=request.maximum_arm_velocity_rad_s,
        )
        lift = _anchor_trajectory_start(
            lift,
            command_q_rad=np.asarray(contact_command_q),
            model_q_rad=contact_model_q,
        )
        lift_q = np.asarray(lift.model_q_rad)
        fixed_close_validator.validate_closed_lift_fixture(lift_q)
        from curobo.types import JointState

        lift_state = JointState.from_position(
            device_cfg.to_device(lift_q),
            joint_names=list(arm_joint_names(arm)),
        )
        lift_pose = (
            planner.compute_kinematics(lift_state)
            .tool_poses[grasp_frame(arm)]
            .get_matrix()
            .detach()
            .cpu()
            .numpy()
        )
        lift_height = (lift_pose[:, :3, 3] - lift_pose[0, :3, 3]) @ (-down)
        split_candidates = np.flatnonzero(lift_height >= request.retention_test_lift_m)
        if len(split_candidates) == 0:
            raise _BranchRejected(
                "retention_test_lift",
                "planned payload route never reaches the configured retention-checkpoint height",
            )
        split_index = int(split_candidates[0])
        if split_index <= 0 or split_index >= len(lift_q) - 1:
            raise _BranchRejected(
                "retention_test_lift",
                "retention-checkpoint boundary falls at unusable sample "
                f"{split_index}/{len(lift_q) - 1}",
            )
        retention_test_lift, remaining_lift = _split_lift_trajectory(
            lift,
            split_index=split_index,
        )
        if request.fixture is None:
            evidence = selected["execution_evidence"]
            exact_link = str(evidence["fixed_close_sweep_minimum_link"])
            if arm == "left":
                exact_link = exact_link.replace("right_", "left_", 1)
            exact = (
                float(evidence["fixed_close_sweep_table_clearance_m"]),
                exact_link,
                int(evidence["fixed_close_sweep_minimum_sample"]),
            )
            wrist = _local_plane_clearance(
                planner,
                lift_q,
                arm=arm,
                plane_point=plane_point,
                down=down,
                include_payload=False,
                link_names=_local_wrist_plane_links(arm),
            )
            hand_clearance, hand_link, hand_sample = min(exact, wrist, key=lambda item: item[0])
        else:
            hand_clearance, hand_link, hand_sample = _local_plane_clearance(
                planner,
                lift_q,
                arm=arm,
                plane_point=plane_point,
                down=down,
                include_payload=False,
            )
        if hand_clearance < request.minimum_hand_plane_clearance_m:
            raise _BranchRejected(
                "closed_lift_table_plane",
                f"clearance={hand_clearance:.4f}m at {hand_link} sample {hand_sample}; "
                f"required={request.minimum_hand_plane_clearance_m:.4f}m",
            )
        payload_clearance, payload_link, payload_sample = _local_plane_clearance(
            planner,
            lift_q,
            arm=arm,
            plane_point=plane_point,
            down=down,
            include_payload=True,
        )
        payload_start_clearance, _payload_start_link, _payload_start_sample = (
            _local_plane_clearance(
                planner,
                lift_q[:1],
                arm=arm,
                plane_point=plane_point,
                down=down,
                include_payload=True,
            )
        )
        if payload_clearance < (payload_start_clearance - COLLISION_ACTIVATION_DISTANCE_M):
            raise _BranchRejected(
                "payload_table_plane",
                f"minimum={payload_clearance:.4f}m at {payload_link} sample "
                f"{payload_sample}, start={payload_start_clearance:.4f}m",
            )
        return _LiftBranchPlan(
            retention_test_lift=retention_test_lift,
            payload_lift=remaining_lift,
            retention_test_lift_actual_m=float(lift_height[split_index]),
            closed_hand_minimum_plane_clearance_m=hand_clearance,
            payload_minimum_plane_clearance_m=payload_clearance,
            payload_start_plane_clearance_m=payload_start_clearance,
            attachment_sphere_count=sphere_count,
        )
    finally:
        _cleanup(planner)


def _rename_trajectory(
    trajectory: PlannedTrajectory,
    *,
    from_pose_id: str,
    to_pose_id: str,
) -> PlannedTrajectory:
    return PlannedTrajectory(
        from_pose_id=from_pose_id,
        to_pose_id=to_pose_id,
        sample_time_s=trajectory.sample_time_s,
        command_q_rad=trajectory.command_q_rad,
        model_q_rad=trajectory.model_q_rad,
        planning_time_s=trajectory.planning_time_s,
    )


def _attached_transfer_scene(
    request: TabletopPickPlaceRequest,
    base_T_torso: np.ndarray,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    """Build one table/world scene covering the complete source-destination span."""

    source = request.source_request
    plane_point, base_T_object, down = _table_from_resting_object(source, base_T_torso)
    scene = _base_scene(
        source,
        base_T_torso,
        include_cube=False,
        include_open_transit_table_patch=False,
    )
    dimensions = np.asarray(source.open_transit_table_patch_dimensions_m, dtype=np.float64)
    relative = np.asarray(request.source_T_destination_object, dtype=np.float64)
    base_T_detected_source = _base_T_detected_object(source, base_T_torso)
    base_T_destination = base_T_detected_source @ relative
    displacement = base_T_destination[:3, 3] - base_T_object[:3, 3]
    displacement_xy = base_T_object[:3, :2].T @ displacement
    expanded = dimensions.copy()
    expanded[:2] += np.abs(displacement_xy)
    base_T_patch = base_T_object.copy()
    base_T_patch[:3, 3] = (
        plane_point + base_T_object[:3, :2] @ (0.5 * displacement_xy) + 0.5 * expanded[2] * down
    )
    scene.setdefault("cuboid", {})["open_transit_table_patch"] = {
        "dims": expanded.tolist(),
        "pose": _pose_list(base_T_patch),
    }
    return scene, plane_point, down


def _plan_attached_transfer(
    request: TabletopPickPlaceRequest,
    source_task: TabletopTaskPlan,
    destination_task: TabletopTaskPlan,
) -> tuple[PlannedTrajectory, dict[str, Any]]:
    """Connect the two already-validated lifted states with the payload attached."""

    import torch
    from curobo.types import DeviceCfg

    source = request.source_request
    arm = source.arm
    if source_task.arm != arm or destination_task.arm != arm:
        raise ValueError("pick-place transfer tasks use a different arm")
    if source_task.selected_candidate_id != destination_task.selected_candidate_id:
        raise ValueError("pick-place transfer changed the selected grasp")
    if not np.allclose(
        source_task.object_T_grasp,
        destination_task.object_T_grasp,
        atol=1.0e-9,
        rtol=0.0,
    ):
        raise ValueError("pick-place transfer changed the object-to-grasp transform")

    source_lift = source_task.trajectories[3]
    destination_lift = destination_task.trajectories[3]
    start_command = np.asarray(source_lift.command_q_rad[-1], dtype=np.float64)
    start_model = np.asarray(source_lift.model_q_rad[-1], dtype=np.float64)
    goal_command = np.asarray(destination_lift.command_q_rad[-1], dtype=np.float64)
    goal_model = np.asarray(destination_lift.model_q_rad[-1], dtype=np.float64)
    close_target = dex3_execution_profile(arm)[1]
    transfer_snapshot = _snapshot_at_arm_q(source, tuple(start_command))
    robot, _ = build_tabletop_robot_config(
        arm=arm,
        snapshot=transfer_snapshot,
        joint_position_offsets_rad=source.joint_position_offsets_rad,
        active_finger_q_rad=close_target,
    )
    device_cfg = DeviceCfg(device=torch.device("cuda:0"), dtype=torch.float32)
    checker = CuroboKinematicCollisionChecker(robot=robot, device_cfg=device_cfg)
    start_state = _joint_state(device_cfg, start_model, arm_joint_names(arm))
    base_T_torso = (
        checker.kinematics.compute_kinematics(start_state)
        .tool_poses["torso_link"]
        .get_matrix()[0]
        .detach()
        .cpu()
        .numpy()
    )
    _use_moving_grasp_frame_only(robot, arm=arm)
    scene, plane_point, down = _attached_transfer_scene(request, base_T_torso)
    planner = None
    started = time.monotonic()
    try:
        planner, planner_device = _planner(
            robot,
            scene,
            max_goalset=1,
            seed=source.random_seed,
        )
        start_state = _joint_state(planner_device, start_model, arm_joint_names(arm))
        goal_state = _joint_state(planner_device, goal_model, arm_joint_names(arm))
        sphere_count = _attach_cube(
            planner,
            start_state,
            invert_transform(np.asarray(source_task.object_T_grasp, dtype=np.float64)),
            source.object_dimensions_m,
            arm=arm,
        )
        result = planner.plan_cspace(
            goal_state=goal_state,
            current_state=start_state,
            max_attempts=10,
            enable_graph_attempt=1,
        )
        if result is None or not bool(result.success.any()):
            raise _BranchRejected(
                "attached_payload_transfer",
                _cspace_failure_reason(result),
            )
        transfer = _result_trajectory(
            planner,
            result,
            arm=arm,
            from_id="payload_lift",
            to_id="payload_transfer",
            offsets=source.joint_position_offsets_rad,
            maximum_velocity_rad_s=source.maximum_arm_velocity_rad_s,
        )
        transfer = _anchor_trajectory_start(
            transfer,
            command_q_rad=start_command,
            model_q_rad=start_model,
        )
        transfer = _anchor_trajectory_end(
            transfer,
            command_q_rad=goal_command,
            model_q_rad=goal_model,
        )
        transfer_q = np.asarray(transfer.model_q_rad, dtype=np.float64)
        hand_clearance, hand_link, hand_sample = _local_plane_clearance(
            planner,
            transfer_q,
            arm=arm,
            plane_point=plane_point,
            down=down,
            include_payload=False,
        )
        if hand_clearance < source.minimum_hand_plane_clearance_m:
            raise _BranchRejected(
                "attached_transfer_table_plane",
                f"hand clearance={hand_clearance:.4f}m at {hand_link} sample "
                f"{hand_sample}; required={source.minimum_hand_plane_clearance_m:.4f}m",
            )
        payload_clearance, payload_link, payload_sample = _local_plane_clearance(
            planner,
            transfer_q,
            arm=arm,
            plane_point=plane_point,
            down=down,
            include_payload=True,
        )
        if payload_clearance < source.minimum_hand_plane_clearance_m:
            raise _BranchRejected(
                "attached_transfer_payload_table_plane",
                f"payload clearance={payload_clearance:.4f}m at {payload_link} sample "
                f"{payload_sample}; required={source.minimum_hand_plane_clearance_m:.4f}m",
            )
        return transfer, {
            "elapsed_s": time.monotonic() - started,
            "attachment_sphere_count": sphere_count,
            "minimum_hand_plane_clearance_m": hand_clearance,
            "minimum_hand_plane_link": hand_link,
            "minimum_hand_plane_sample": hand_sample,
            "minimum_payload_plane_clearance_m": payload_clearance,
            "minimum_payload_plane_link": payload_link,
            "minimum_payload_plane_sample": payload_sample,
            "world_cuboid_ids": sorted(scene.get("cuboid", {})),
        }
    finally:
        _cleanup(planner)


def _payload_route(task: TabletopTaskPlan) -> np.ndarray:
    payload_phases = (
        "retention_test_lift",
        "payload_lift",
        "payload_lower",
        "payload_replace",
    )
    payload_trajectories = tuple(
        trajectory for trajectory in task.trajectories if trajectory.to_pose_id in payload_phases
    )
    if tuple(value.to_pose_id for value in payload_trajectories) != payload_phases:
        raise ValueError("retention-route task lacks the complete split payload lifecycle")
    return np.concatenate(
        (
            np.asarray(payload_trajectories[0].model_q_rad, dtype=np.float64),
            *(
                np.asarray(value.model_q_rad[1:], dtype=np.float64)
                for value in payload_trajectories[1:]
            ),
        ),
        axis=0,
    )


def plan_moving_grasp_continuation(
    continuation: MovingGraspContinuationRequest,
    *,
    progress: Callable[[str], None] | None = None,
) -> tuple[TabletopTaskPlan, _MovingGraspGeometry]:
    """Rebuild only the close/lift/replace lifecycle after moving-target MPC.

    The global clearance-to-pregrasp route is preserved.  The approach and
    its reverse are the immutable MPC windows actually accepted by the robot
    controller.  Only the payload lift is newly optimized at the reached
    object pose.
    """

    report = progress or (lambda _message: None)
    request = continuation.tabletop_request
    prior = continuation.prior_task_plan
    arm = request.arm
    terminal_command = np.asarray(continuation.terminal_command_q_rad, dtype=np.float64)
    snapshot = request.planning_snapshot
    q29 = np.asarray(snapshot.measured_q29_rad, dtype=np.float64).copy()
    q29[np.asarray(arm_indices(arm))] = terminal_command
    active_fingers = tuple(continuation.terminal_active_dex3_q_rad)
    contact_snapshot = RobotSnapshot(
        measured_q29_rad=tuple(q29),
        left_dex3_q_rad=(active_fingers if arm == "left" else snapshot.left_dex3_q_rad),
        right_dex3_q_rad=(active_fingers if arm == "right" else snapshot.right_dex3_q_rad),
    )
    contact_model = np.asarray(
        continuation.executed_grasp_approach.model_q_rad[-1],
        dtype=np.float64,
    )

    import torch
    from curobo.types import DeviceCfg

    device_cfg = DeviceCfg(device=torch.device("cuda:0"), dtype=torch.float32)
    robot, _ = build_tabletop_robot_config(
        arm=arm,
        snapshot=contact_snapshot,
        joint_position_offsets_rad=request.joint_position_offsets_rad,
        active_finger_q_rad=active_fingers,
    )
    checker = CuroboKinematicCollisionChecker(robot=robot, device_cfg=device_cfg)
    contact_state = _joint_state(device_cfg, contact_model, arm_joint_names(arm))
    base_T_torso = (
        checker.kinematics.compute_kinematics(contact_state)
        .tool_poses["torso_link"]
        .get_matrix()[0]
        .detach()
        .cpu()
        .numpy()
    )
    base_T_torso[:3, :3] = Rotation.from_matrix(base_T_torso[:3, :3]).as_matrix()
    reference_T_camera = np.asarray(continuation.reference_T_camera, dtype=np.float64)
    camera_T_object = np.asarray(continuation.camera_T_object, dtype=np.float64)
    reference_T_torso = reference_T_camera @ invert_transform(
        np.asarray(request.torso_T_camera, dtype=np.float64)
    )
    base_T_reference = base_T_torso @ invert_transform(reference_T_torso)
    base_T_detected_object = base_T_reference @ reference_T_camera @ camera_T_object
    base_T_object = canonical_resting_cube_pose(base_T_detected_object)
    down = -base_T_object[:3, 2]
    fixture_height = 0.0 if request.fixture is None else request.fixture.support_height_m
    plane_point = base_T_object[:3, 3] + (
        0.5 * request.object_dimensions_m[2] + fixture_height
    ) * down
    geometry = _MovingGraspGeometry(
        base_T_torso=base_T_torso,
        base_T_detected_object=base_T_detected_object,
        base_T_object=base_T_object,
        plane_point=plane_point,
        down=down,
    )

    _shortlist, candidates = _load_shortlist(request)
    selected = [
        candidate
        for candidate in candidates
        if str(candidate.get("candidate_id")) == prior.selected_candidate_id
    ]
    if len(selected) != 1:
        raise ValueError("moving-grasp selected candidate is not uniquely present")
    selected_candidate = selected[0]
    reached_pose = (
        checker.kinematics.compute_kinematics(contact_state)
        .tool_poses[grasp_frame(arm)]
        .get_matrix()[0]
        .detach()
        .cpu()
        .numpy()
    )
    expected_pose = base_T_object @ np.asarray(prior.object_T_grasp, dtype=np.float64)
    position_error = float(np.linalg.norm(reached_pose[:3, 3] - expected_pose[:3, 3]))
    rotation_error = float(
        Rotation.from_matrix(reached_pose[:3, :3].T @ expected_pose[:3, :3]).magnitude()
    )
    if (
        position_error > PREGRASP_IK_POSITION_TOLERANCE_M
        or rotation_error > PREGRASP_IK_ORIENTATION_TOLERANCE_RAD
    ):
        raise ValueError(
            "terminal MPC state does not realize the live selected grasp: "
            f"error={position_error * 1000.0:.3f}mm/"
            f"{np.degrees(rotation_error):.3f}deg"
        )
    _open_profile, close_profile = dex3_execution_profile(arm)
    close_target = np.asarray(close_profile, dtype=np.float64)
    fixed_close_validator = _FixedCloseSweepValidator(
        request=request,
        base_T_object=base_T_object,
        base_T_detected_object=base_T_detected_object,
        plane_point=plane_point,
        down=down,
        open_q=np.asarray(active_fingers, dtype=np.float64),
        close_target_q=close_target,
    )
    fixed_close_sweep = fixed_close_validator.validate(contact_model, selected_candidate)
    report("moving-target terminal fixed-close sweep passed; planning attached lift")
    started = time.monotonic()
    lift_plan = _plan_attached_lift(
        request=request,
        selected=selected_candidate,
        close_target_q=close_target,
        contact_command_q=tuple(terminal_command),
        contact_model_q=contact_model,
        base_T_torso=base_T_torso,
        plane_point=plane_point,
        down=down,
        arm=arm,
        fixed_close_validator=fixed_close_validator,
        contact_snapshot=contact_snapshot,
        base_T_object_override=base_T_object,
        base_T_detected_object_override=base_T_detected_object,
    )
    retention_test_lift = lift_plan.retention_test_lift
    payload_lift = lift_plan.payload_lift
    payload_lower = _rename_trajectory(
        _reverse_trajectory(payload_lift),
        from_pose_id="payload_lift",
        to_pose_id="payload_lower",
    )
    payload_replace = _rename_trajectory(
        _reverse_trajectory(retention_test_lift),
        from_pose_id="payload_lower",
        to_pose_id="payload_replace",
    )
    grasp_retreat = _rename_trajectory(
        _reverse_trajectory(continuation.executed_grasp_approach),
        from_pose_id="payload_replace",
        to_pose_id="grasp_retreat",
    )
    return_to_clearance = _rename_trajectory(
        _reverse_trajectory(prior.trajectories[0]),
        from_pose_id="grasp_retreat",
        to_pose_id="return_to_clearance",
    )
    task = TabletopTaskPlan(
        request_sha256=request.content_sha256,
        arm=arm,
        selected_candidate_id=prior.selected_candidate_id,
        object_T_grasp=prior.object_T_grasp,
        open_active_dex3_q_rad=prior.open_active_dex3_q_rad,
        close_target_active_dex3_q_rad=prior.close_target_active_dex3_q_rad,
        initial_active_dex3_q_rad=prior.initial_active_dex3_q_rad,
        trajectories=(
            prior.trajectories[0],
            continuation.executed_grasp_approach,
            retention_test_lift,
            payload_lift,
            payload_lower,
            payload_replace,
            grasp_retreat,
            return_to_clearance,
        ),
        phase_order=prior.phase_order,
        planner_provenance={
            **prior.planner_provenance,
            "moving_target_continuation": {
                "request_sha256": continuation.content_sha256,
                "terminal_mpc_window_sha256": (
                    continuation.terminal_mpc_window_sha256
                ),
                "target_provenance": continuation.target_provenance,
                "payload_replan_elapsed_s": time.monotonic() - started,
                "base_T_detected_object": base_T_detected_object.tolist(),
                "base_T_object": base_T_object.tolist(),
                "terminal_grasp_position_error_m": position_error,
                "terminal_grasp_rotation_error_rad": rotation_error,
                "plane_point": plane_point.tolist(),
                "down": down.tolist(),
                "fixed_close_sweep_sample_count": fixed_close_sweep.sample_count,
                "fixed_close_sweep_minimum_plane_clearance_m": (
                    fixed_close_sweep.minimum_plane_clearance_m
                ),
                "retention_test_lift_actual_m": (
                    lift_plan.retention_test_lift_actual_m
                ),
                "payload_start_plane_clearance_m": (
                    lift_plan.payload_start_plane_clearance_m
                ),
                "return_policy": (
                    "reverse_new_payload_lift_then_reverse_exact_accepted_mpc_approach_"
                    "then_reverse_original_clearance_to_pregrasp"
                ),
            },
            "retention_test_lift_actual_m": lift_plan.retention_test_lift_actual_m,
            "payload_start_plane_clearance_m": lift_plan.payload_start_plane_clearance_m,
        },
    )
    report(
        "moving-target payload continuation ready; exact accepted MPC approach "
        "is the open-hand reverse"
    )
    return task, geometry


class RetentionRouteValidator:
    """Cached FK/collision checker for one frozen task's measured close pose."""

    def __init__(
        self,
        tabletop: TabletopTaskRequest,
        task: TabletopTaskPlan,
        *,
        geometry: _MovingGraspGeometry | None = None,
    ) -> None:
        import torch
        from curobo.types import DeviceCfg, JointState

        if task.request_sha256 != tabletop.content_sha256:
            raise ValueError("retention validator task belongs to another request")
        if not torch.cuda.is_available():
            raise RuntimeError("CuRobo retention validation requires a CUDA device")
        self.tabletop = tabletop
        self.task = task
        self.arm = tabletop.arm
        self.route_q = _payload_route(task)
        started = time.monotonic()
        robot, self.active_joint_names, reference = build_tabletop_route_validation_robot_config(
            arm=self.arm,
            snapshot=tabletop.planning_snapshot,
            joint_position_offsets_rad=tabletop.joint_position_offsets_rad,
        )
        self.device_cfg = DeviceCfg(device=torch.device("cuda:0"), dtype=torch.float32)
        self.checker = CuroboKinematicCollisionChecker(
            robot=robot,
            device_cfg=self.device_cfg,
        )
        reference_state = JointState.from_position(
            self.device_cfg.to_device(np.asarray(reference, dtype=np.float64)[None]),
            joint_names=list(self.active_joint_names),
        )
        kinematics = self.checker.kinematics.compute_kinematics(reference_state)
        base_T_torso = kinematics.tool_poses["torso_link"].get_matrix()[0].detach().cpu().numpy()
        if geometry is None:
            self.plane_point, base_T_object, self.down = _table_from_resting_object(
                tabletop,
                base_T_torso,
            )
            base_T_detected_object = _base_T_detected_object(tabletop, base_T_torso)
        else:
            self.plane_point = geometry.plane_point
            base_T_object = geometry.base_T_object
            self.down = geometry.down
            base_T_detected_object = geometry.base_T_detected_object
        self.fixture_checker = _fixture_collision_checker(
            tabletop,
            base_T_object,
            base_T_detected_object,
            self.down,
            device_cfg=self.device_cfg,
        )
        self.cache_build_s = time.monotonic() - started

    def validate(
        self,
        request: RetentionRouteValidationRequest,
        *,
        progress: Callable[[str], None] | None = None,
    ) -> RetentionRouteValidationResult:
        report = progress or (lambda _message: None)
        if request.tabletop_request.content_sha256 != self.tabletop.content_sha256:
            raise ValueError("retention request differs from the cached tabletop request")
        if request.task_plan.content_sha256 != self.task.content_sha256:
            raise ValueError("retention request differs from the cached task plan")
        finger_q = np.asarray(request.measured_active_dex3_q_rad, dtype=np.float64)
        route_q = np.concatenate(
            (self.route_q, np.repeat(finger_q[None], len(self.route_q), axis=0)),
            axis=1,
        )
        started = time.monotonic()
        collision_samples = self.checker.self_collision_pair_penetrations(
            route_q,
            joint_names=self.active_joint_names,
        )
        for sample_index, pairs in enumerate(collision_samples):
            if not pairs:
                continue
            pair, penetration = max(pairs.items(), key=lambda item: item[1])
            raise RuntimeError(
                "measured close Dex3 posture invalidates the frozen payload route: "
                f"{pair[0]}/{pair[1]}={penetration * 1000.0:.3f}mm penetration "
                f"at sample {sample_index}/{len(route_q) - 1}"
            )
        sphere_tensor = self.checker.robot_spheres(
            route_q,
            joint_names=self.active_joint_names,
        )
        spheres = sphere_tensor.detach().cpu().numpy().reshape(len(route_q), -1, 4)
        hand_clearances, hand_links = _local_plane_clearance_samples_from_spheres(
            spheres,
            config=self.checker.config.kinematics_config,
            arm=self.arm,
            plane_point=self.plane_point,
            down=self.down,
            include_payload=False,
        )
        plane_policy = _validate_start_relative_retention_clearance(
            hand_clearances,
            hand_links,
            required_m=self.tabletop.minimum_hand_plane_clearance_m,
        )
        hand_clearance = plane_policy.minimum_m
        hand_link = plane_policy.minimum_link
        hand_sample = plane_policy.minimum_sample
        if self.fixture_checker is not None:
            fixture_hit = self.fixture_checker.first_collision(
                sphere_tensor,
                kinematics_config=self.checker.config.kinematics_config,
            )
            if fixture_hit is not None:
                penetration, fixture_link, fixture_sample = fixture_hit
                raise RuntimeError(
                    "measured close Dex3 posture invalidates the frozen payload route "
                    f"against the presentation fixture: {fixture_link} has "
                    f"{penetration * 1000.0:.3f}mm penetration at sample "
                    f"{fixture_sample}/{len(route_q) - 1}"
                )
        report(
            "measured close-hand retention route passed strict self-collision and "
            "start-relative table-plane checks; "
            f"boundary/minimum hand clearance={plane_policy.boundary_m:.4f}m/"
            f"{hand_clearance:.4f}m, full {self.tabletop.minimum_hand_plane_clearance_m:.4f}m "
            f"margin over samples {plane_policy.first_full_margin_sample}-"
            f"{plane_policy.last_full_margin_sample}"
            + (
                " and CUDA fixture collision check passed"
                if self.fixture_checker is not None
                else ""
            )
        )
        return RetentionRouteValidationResult(
            request_sha256=request.content_sha256,
            arm=self.arm,
            selected_candidate_id=self.task.selected_candidate_id,
            route_sample_count=len(route_q),
            minimum_hand_plane_clearance_m=hand_clearance,
            minimum_hand_plane_link=hand_link,
            minimum_hand_plane_sample=hand_sample,
            minimum_fixture_clearance_m=None,
            minimum_fixture_clearance_link=None,
            minimum_fixture_clearance_sample=None,
            planner_provenance={
                **model_source_hashes(),
                "curobo_commit": CUROBO_COMMIT,
                "elapsed_s": time.monotonic() - started,
                "cached_kinematics": True,
                "cache_build_s": self.cache_build_s,
                "required_hand_plane_clearance_m": (self.tabletop.minimum_hand_plane_clearance_m),
                "table_plane_policy": "positive_start_relative_escape_exact_reverse_return",
                "boundary_hand_plane_clearance_m": plane_policy.boundary_m,
                "first_full_margin_sample": plane_policy.first_full_margin_sample,
                "last_full_margin_sample": plane_policy.last_full_margin_sample,
                "policy": (
                    "frozen split payload arm route; measured stable-close active Dex3; "
                    "strict full-robot self-collision, selected wrist/hand table plane, "
                    "and optional exact presenter mesh"
                ),
                "presentation_id": self.tabletop.presentation_id,
                "fixture_mesh_rechecked_on_cuda": self.fixture_checker is not None,
                "blocked_motor_ids": list(request.blocked_motor_ids),
                "retention_evidence_policy": (
                    "commissioned_empty_close_opposed_joint_obstruction"
                ),
                "pressure_used_for_live_decision": False,
            },
        )


def _pick_place_payload_route(plan: TabletopPickPlacePlan) -> np.ndarray:
    payload_phases = PICK_PLACE_PHASE_ORDER[2:7]
    trajectories = tuple(
        value for value in plan.trajectories if value.to_pose_id in payload_phases
    )
    if tuple(value.to_pose_id for value in trajectories) != payload_phases:
        raise ValueError("pick-place plan lacks its complete closed-payload route")
    return np.concatenate(
        (
            np.asarray(trajectories[0].model_q_rad, dtype=np.float64),
            *(np.asarray(value.model_q_rad[1:], dtype=np.float64) for value in trajectories[1:]),
        ),
        axis=0,
    )


class PickPlaceRetentionRouteValidator:
    """Recheck the complete transfer using the actual contact-stopped Dex3 posture."""

    def __init__(
        self,
        request: TabletopPickPlaceRequest,
        plan: TabletopPickPlacePlan,
    ) -> None:
        import torch
        from curobo.types import DeviceCfg, JointState

        if plan.request_sha256 != request.content_sha256:
            raise ValueError("pick-place retention plan belongs to a different request")
        if not torch.cuda.is_available():
            raise RuntimeError("CuRobo retention validation requires a CUDA device")
        self.request = request
        self.plan = plan
        self.tabletop = request.source_request
        self.arm = self.tabletop.arm
        self.route_q = _pick_place_payload_route(plan)
        started = time.monotonic()
        robot, self.active_joint_names, reference = build_tabletop_route_validation_robot_config(
            arm=self.arm,
            snapshot=self.tabletop.planning_snapshot,
            joint_position_offsets_rad=self.tabletop.joint_position_offsets_rad,
        )
        self.device_cfg = DeviceCfg(device=torch.device("cuda:0"), dtype=torch.float32)
        self.checker = CuroboKinematicCollisionChecker(
            robot=robot,
            device_cfg=self.device_cfg,
        )
        reference_state = JointState.from_position(
            self.device_cfg.to_device(np.asarray(reference, dtype=np.float64)[None]),
            joint_names=list(self.active_joint_names),
        )
        kinematics = self.checker.kinematics.compute_kinematics(reference_state)
        base_T_torso = kinematics.tool_poses["torso_link"].get_matrix()[0].detach().cpu().numpy()
        self.plane_point, base_T_object, self.down = _table_from_resting_object(
            self.tabletop,
            base_T_torso,
        )
        self.environment_checker = _fixture_collision_checker(
            self.tabletop,
            base_T_object,
            _base_T_detected_object(self.tabletop, base_T_torso),
            self.down,
            device_cfg=self.device_cfg,
        )
        self.cache_build_s = time.monotonic() - started

    def validate(
        self,
        request: PickPlaceRetentionRouteValidationRequest,
        *,
        progress: Callable[[str], None] | None = None,
    ) -> RetentionRouteValidationResult:
        report = progress or (lambda _message: None)
        if request.pick_place_request.content_sha256 != self.request.content_sha256:
            raise ValueError("retention request differs from the cached pick-place request")
        if request.pick_place_plan.content_sha256 != self.plan.content_sha256:
            raise ValueError("retention request differs from the cached pick-place plan")
        finger_q = np.asarray(request.measured_active_dex3_q_rad, dtype=np.float64)
        route_q = np.concatenate(
            (self.route_q, np.repeat(finger_q[None], len(self.route_q), axis=0)),
            axis=1,
        )
        started = time.monotonic()
        collision_samples = self.checker.self_collision_pair_penetrations(
            route_q,
            joint_names=self.active_joint_names,
        )
        for sample_index, pairs in enumerate(collision_samples):
            if not pairs:
                continue
            pair, penetration = max(pairs.items(), key=lambda item: item[1])
            raise RuntimeError(
                "measured close Dex3 posture invalidates the pick-place payload route: "
                f"{pair[0]}/{pair[1]}={penetration * 1000.0:.3f}mm penetration "
                f"at sample {sample_index}/{len(route_q) - 1}"
            )
        sphere_tensor = self.checker.robot_spheres(
            route_q,
            joint_names=self.active_joint_names,
        )
        spheres = sphere_tensor.detach().cpu().numpy().reshape(len(route_q), -1, 4)
        hand_clearances, hand_links = _local_plane_clearance_samples_from_spheres(
            spheres,
            config=self.checker.config.kinematics_config,
            arm=self.arm,
            plane_point=self.plane_point,
            down=self.down,
            include_payload=False,
        )
        plane_policy = _validate_pick_place_retention_clearance(
            hand_clearances,
            hand_links,
            required_m=self.tabletop.minimum_hand_plane_clearance_m,
        )
        if self.environment_checker is not None:
            environment_hit = self.environment_checker.first_collision(
                sphere_tensor,
                kinematics_config=self.checker.config.kinematics_config,
            )
            if environment_hit is not None:
                penetration, link_name, sample = environment_hit
                raise RuntimeError(
                    "measured close Dex3 posture invalidates the pick-place route "
                    f"against the fixed world: {link_name} has "
                    f"{penetration * 1000.0:.3f}mm penetration at sample "
                    f"{sample}/{len(route_q) - 1}"
                )
        report(
            "measured close-hand pick-place route passed strict self-collision, "
            "fixed-world, and source-to-destination table-plane checks"
        )
        return RetentionRouteValidationResult(
            request_sha256=request.content_sha256,
            arm=self.arm,
            selected_candidate_id=self.plan.selected_candidate_id,
            route_sample_count=len(route_q),
            minimum_hand_plane_clearance_m=plane_policy.minimum_m,
            minimum_hand_plane_link=plane_policy.minimum_link,
            minimum_hand_plane_sample=plane_policy.minimum_sample,
            minimum_fixture_clearance_m=None,
            minimum_fixture_clearance_link=None,
            minimum_fixture_clearance_sample=None,
            planner_provenance={
                **model_source_hashes(),
                "curobo_commit": CUROBO_COMMIT,
                "elapsed_s": time.monotonic() - started,
                "cached_kinematics": True,
                "cache_build_s": self.cache_build_s,
                "required_hand_plane_clearance_m": (self.tabletop.minimum_hand_plane_clearance_m),
                "table_plane_policy": "positive_source_and_destination_contacts",
                "source_contact_hand_plane_clearance_m": float(hand_clearances[0]),
                "destination_contact_hand_plane_clearance_m": float(hand_clearances[-1]),
                "first_full_margin_sample": plane_policy.first_full_margin_sample,
                "last_full_margin_sample": plane_policy.last_full_margin_sample,
                "policy": (
                    "fixed pick-place closed-payload arm route; measured stable-close "
                    "Dex3; strict full-robot self-collision, fixed cuboids, and table plane"
                ),
                "blocked_motor_ids": list(request.blocked_motor_ids),
                "pressure_used_for_live_decision": False,
            },
        )


def validate_retention_route(
    request: RetentionRouteValidationRequest,
    *,
    progress: Callable[[str], None] | None = None,
) -> RetentionRouteValidationResult:
    """Standalone compatibility entry point using the same cached-checker code."""

    validator = RetentionRouteValidator(request.tabletop_request, request.task_plan)
    return validator.validate(request, progress=progress)


def _plan_tabletop(
    request: TabletopTaskRequest,
    *,
    required_candidate_id: str | None = None,
    pregrasp_only: bool,
    open_planner_cache: ReusableOpenPlanner | None = None,
    progress: Callable[[str], None] | None = None,
) -> TabletopTaskPlan | TabletopPregraspPlan:
    """Plan either the first boundary route or one complete task."""

    report = progress or (lambda _message: None)
    arm = request.arm
    shortlist, candidates = _load_shortlist(request)
    if required_candidate_id is not None:
        candidates = [
            candidate
            for candidate in candidates
            if str(candidate.get("candidate_id")) == required_candidate_id
        ]
        if len(candidates) != 1:
            raise ValueError(
                f"required grasp candidate {required_candidate_id!r} is not uniquely present"
            )
        report(f"preserving previously selected grasp {required_candidate_id}")
    open_target_profile, close_target_profile = dex3_execution_profile(arm)
    open_target_q = np.asarray(open_target_profile, dtype=np.float64)
    open_q = np.asarray(
        request.planning_snapshot.left_dex3_q_rad
        if arm == "left"
        else request.planning_snapshot.right_dex3_q_rad,
        dtype=np.float64,
    )
    close_target_q = np.asarray(close_target_profile, dtype=np.float64)
    reference = np.asarray(request.planning_snapshot.measured_q29_rad)[
        np.asarray(arm_indices(arm))
    ]
    # Model-space reference includes removable calibration joint offsets.
    reference_model = np.asarray(
        [
            value + request.joint_position_offsets_rad.get(name, 0.0)
            for name, value in zip(arm_joint_names(arm), reference, strict=True)
        ]
    )
    started = time.monotonic()
    planner = None
    strict_model_resolution_s = 0.0
    optimizer_model_clone_s = 0.0
    open_optimizer_setup_s = 0.0
    open_optimizer_reconfiguration_s = 0.0
    open_optimizer_reused = False
    open_optimizer_topology_rebuilt = False
    batched_ik_setup_s = 0.0
    batched_ik_s = 0.0
    batched_fixed_close_s = 0.0
    endpoint_precheck_s = 0.0
    fixed_close_sweep_validation_s = 0.0
    open_branch_planning_s = 0.0
    attached_lift_planning_s = 0.0
    try:
        import torch
        from curobo.types import DeviceCfg

        query_robot, _ = build_tabletop_robot_config(
            arm=arm,
            snapshot=request.planning_snapshot,
            joint_position_offsets_rad=request.joint_position_offsets_rad,
            active_finger_q_rad=tuple(open_q),
        )
        # The torso transform is an FK query. Building a full MotionPlanner for
        # this one read added a second cold CUDA initialization to every task.
        # This checker is also the strict full-robot validator used by every
        # candidate branch, so no temporary model is needed.
        device_cfg = DeviceCfg(device=torch.device("cuda:0"), dtype=torch.float32)
        stage_started = time.monotonic()
        strict_open_checker = CuroboKinematicCollisionChecker(
            robot=query_robot,
            device_cfg=device_cfg,
        )
        strict_model_resolution_s = time.monotonic() - stage_started
        state = _joint_state(device_cfg, reference_model, arm_joint_names(arm))
        kinematics = strict_open_checker.kinematics.compute_kinematics(state)
        base_T_torso = kinematics.tool_poses["torso_link"].get_matrix()[0].detach().cpu().numpy()

        plane_point, base_T_object, down = _table_from_resting_object(request, base_T_torso)
        fixed_close_validator = _FixedCloseSweepValidator(
            request=request,
            base_T_object=base_T_object,
            base_T_detected_object=_base_T_detected_object(request, base_T_torso),
            plane_point=plane_point,
            down=down,
            open_q=open_q,
            close_target_q=close_target_q,
        )
        grasp_matrices = [base_T_object @ _candidate_transform(item) for item in candidates]
        approach_distance_m = float(shortlist["execution_contract"]["approach_distance_m"])
        branch_rejections: list[dict[str, Any]] = []
        stage_started = time.monotonic()
        exact_table_evidence = None
        if request.fixture is None:
            exact_table_evidence = []
            for candidate in candidates:
                evidence = candidate["execution_evidence"]
                link_name = str(evidence["fixed_close_sweep_minimum_link"])
                if arm == "left":
                    link_name = link_name.replace("right_", "left_", 1)
                exact_table_evidence.append(
                    (
                        float(evidence["fixed_close_sweep_table_clearance_m"]),
                        link_name,
                        int(evidence["fixed_close_sweep_minimum_sample"]),
                    )
                )
        candidate_rejections = fixed_close_validator.batch_candidate_rejections(
            grasp_matrices,
            exact_table_evidence=exact_table_evidence,
        )
        batched_fixed_close_s = time.monotonic() - stage_started
        remaining_indices = [
            index for index in range(len(candidates)) if index not in candidate_rejections
        ]
        for index, (stage, reason) in candidate_rejections.items():
            branch_rejections.append(
                {
                    "candidate_id": str(candidates[index]["candidate_id"]),
                    "stage": stage,
                    "reason": reason,
                }
            )
        if candidate_rejections:
            report(
                "batched fixed-close GPU pruning removed "
                f"{len(candidate_rejections)}/{len(candidates)} candidates; "
                f"{len(remaining_indices)} remain for arm IK"
            )
        if not remaining_indices:
            raise RuntimeError(
                "every qualified grasp collides with the presentation fixture or table "
                f"during the fixed descriptor close sweep: {branch_rejections}"
            )
        selected: dict[str, Any] | None = None
        selected_branch: _PregraspBranch | None = None
        selected_round = 0
        approach_plan: _ApproachBranchPlan | None = None
        open_plan: _OpenBranchPlan | None = None
        lift_plan: _LiftBranchPlan | None = None
        fixed_close_sweep_plan: _FixedCloseSweepResult | None = None
        branch_attempt_count = 0
        search_round = 0
        open_robot = query_robot
        _use_moving_grasp_frame_only(open_robot, arm=arm)
        transit_robot = _selected_open_transit_world_robot(open_robot, arm=arm)
        stage_started = time.monotonic()
        resolved_transit_robot = _resolved_motion_robot(
            checker=strict_open_checker,
            source_robot=query_robot,
            arm=arm,
            scope_world_to_selected_hand=True,
        )
        optimizer_model_clone_s = time.monotonic() - stage_started
        scene = _base_scene(
            request,
            base_T_torso,
            include_cube=True,
            include_open_transit_table_patch=True,
        )

        while remaining_indices:
            search_round += 1
            subset_indices = remaining_indices.copy()
            subset_candidates = [candidates[index] for index in subset_indices]
            subset_matrices = [grasp_matrices[index] for index in subset_indices]
            candidate_ids = [str(item["candidate_id"]) for item in subset_candidates]
            pregrasp_matrices = [
                _pregrasp_matrix(matrix, approach_distance_m) for matrix in subset_matrices
            ]

            def create_open_planner(
                current_transit_robot=resolved_transit_robot,
                current_scene=scene,
            ):
                nonlocal open_optimizer_reconfiguration_s
                nonlocal open_optimizer_reused, open_optimizer_setup_s
                nonlocal open_optimizer_topology_rebuilt
                setup_started = time.monotonic()
                if open_planner_cache is None:
                    current_planner, current_device = _planner(
                        current_transit_robot,
                        current_scene,
                        max_goalset=1,
                        seed=request.random_seed,
                    )
                    cache_event = None
                else:
                    current_planner, current_device, cache_event = open_planner_cache.acquire(
                        current_transit_robot,
                        current_scene,
                        seed=request.random_seed,
                        configuration_key=request.content_sha256,
                    )
                    open_optimizer_reused = open_optimizer_reused or bool(cache_event["reused"])
                    open_optimizer_topology_rebuilt = open_optimizer_topology_rebuilt or bool(
                        cache_event["topology_rebuilt"]
                    )
                    if cache_event["reused"]:
                        open_optimizer_reconfiguration_s += float(cache_event["elapsed_s"])
                open_optimizer_setup_s += time.monotonic() - setup_started
                current_state = _joint_state(
                    current_device,
                    reference_model,
                    arm_joint_names(arm),
                )
                return current_planner, current_device, current_state

            stage_started = time.monotonic()
            ik_solver = _batched_pregrasp_ik_solver(
                resolved_transit_robot,
                scene,
                candidate_count=len(subset_candidates),
                seed=request.random_seed,
                device_cfg=device_cfg,
            )
            batched_ik_setup_s += time.monotonic() - stage_started
            pregrasp_goals = _batched_pose_goals(pregrasp_matrices, device_cfg, arm=arm)
            stage_started = time.monotonic()
            try:
                branches, ik_result = _enumerate_pregrasp_branches(
                    ik_solver,
                    pregrasp_goals,
                    state,
                    candidate_count=len(subset_candidates),
                )
                batched_ik_s += time.monotonic() - stage_started
                diagnostic = (
                    None
                    if branches
                    else _batched_ik_failure_diagnostic(
                        result=ik_result,
                        robot=transit_robot,
                        scene=scene,
                        candidates=subset_candidates,
                        device_cfg=device_cfg,
                        disabled_collision_links=set(),
                    )
                )
            finally:
                _cleanup(ik_solver)
            if not branches:
                raise RuntimeError(
                    "no collision-valid pregrasp IK branch remains for the qualified "
                    f"candidates; prior branch rejections={branch_rejections}; {diagnostic}"
                )
            ik_branch_counts = {
                local_index: sum(
                    branch.candidate_local_index == local_index for branch in branches
                )
                for local_index in range(len(subset_candidates))
            }
            stage_started = time.monotonic()
            endpoint_collision_reasons = _pregrasp_endpoint_self_collision_reasons(
                branches,
                checker=strict_open_checker,
            )
            endpoint_precheck_s += time.monotonic() - stage_started
            surviving_branches: list[_PregraspBranch] = []
            endpoint_pruned = 0
            for branch, reason in zip(branches, endpoint_collision_reasons, strict=True):
                if reason is None:
                    surviving_branches.append(branch)
                    continue
                endpoint_pruned += 1
                branch_rejections.append(
                    {
                        "candidate_id": candidate_ids[branch.candidate_local_index],
                        "solver_seed_index": branch.solver_seed_index,
                        "stage": "pregrasp_endpoint_strict_self_collision",
                        "reason": reason,
                    }
                )
            branches = surviving_branches
            branch_counts = {
                local_index: sum(
                    branch.candidate_local_index == local_index for branch in branches
                )
                for local_index in range(len(subset_candidates))
            }
            counts_text = ", ".join(
                f"{candidate_ids[index]}={ik_branch_counts[index]}"
                for index in range(len(candidate_ids))
            )
            report(
                f"CuRobo independent batched pregrasp IK: {len(branches)} unique "
                "strict-endpoint-valid branches after pruning "
                f"{endpoint_pruned} branches in one GPU pass; "
                f"{len(subset_candidates)} candidates with {IK_SEEDS} dedicated seeds "
                f"each produced: {counts_text}"
            )

            def attempt_branch(
                branch: _PregraspBranch,
                current_candidates=subset_candidates,
                current_matrices=subset_matrices,
                current_open_robot=open_robot,
                current_strict_checker=strict_open_checker,
            ) -> (
                _ApproachBranchPlan
                | tuple[
                    _OpenBranchPlan,
                    _LiftBranchPlan,
                    _FixedCloseSweepResult,
                ]
            ):
                nonlocal planner, device_cfg, state, branch_attempt_count
                nonlocal fixed_close_sweep_validation_s
                nonlocal open_branch_planning_s, attached_lift_planning_s
                branch_attempt_count += 1
                if planner is None:
                    planner, device_cfg, state = create_open_planner()
                branch_start_state = _fresh_branch_start_state(
                    device_cfg,
                    reference_model,
                    arm=arm,
                )
                selected_local = branch.candidate_local_index
                candidate = current_candidates[selected_local]
                branch_started = time.monotonic()
                if pregrasp_only:
                    try:
                        open_validation = _plan_open_branch(
                            planner=planner,
                            device_cfg=device_cfg,
                            state=branch_start_state,
                            reference_command_q=reference,
                            reference_model_q=reference_model,
                            pregrasp_model_q=branch.model_q_rad,
                            grasp_matrix=current_matrices[selected_local],
                            open_robot=current_open_robot,
                            strict_checker=current_strict_checker,
                            request=request,
                            base_T_torso=base_T_torso,
                            plane_point=plane_point,
                            down=down,
                            arm=arm,
                        )
                    finally:
                        open_branch_planning_s += time.monotonic() - branch_started
                    sweep_started = time.monotonic()
                    try:
                        fixed_close_sweep = fixed_close_validator.validate(
                            np.asarray(open_validation.grasp.model_q_rad[-1]),
                            candidate,
                        )
                    finally:
                        fixed_close_sweep_validation_s += time.monotonic() - sweep_started
                    return _ApproachBranchPlan(
                        approach=open_validation.approach,
                        minimum_plane_clearance_m=(open_validation.minimum_plane_clearance_m),
                        minimum_plane_link=open_validation.minimum_plane_link,
                        minimum_plane_sample=open_validation.minimum_plane_sample,
                        fixed_close_sweep=fixed_close_sweep,
                    )
                try:
                    branch_open = _plan_open_branch(
                        planner=planner,
                        device_cfg=device_cfg,
                        state=branch_start_state,
                        reference_command_q=reference,
                        reference_model_q=reference_model,
                        pregrasp_model_q=branch.model_q_rad,
                        grasp_matrix=current_matrices[selected_local],
                        open_robot=current_open_robot,
                        strict_checker=current_strict_checker,
                        request=request,
                        base_T_torso=base_T_torso,
                        plane_point=plane_point,
                        down=down,
                        arm=arm,
                    )
                finally:
                    open_branch_planning_s += time.monotonic() - branch_started
                sweep_started = time.monotonic()
                try:
                    branch_fixed_close_sweep = fixed_close_validator.validate(
                        np.asarray(branch_open.grasp.model_q_rad[-1]),
                        candidate,
                    )
                finally:
                    fixed_close_sweep_validation_s += time.monotonic() - sweep_started
                if open_planner_cache is None:
                    _cleanup(planner)
                planner = None
                lift_started = time.monotonic()
                try:
                    branch_lift = _plan_attached_lift(
                        request=request,
                        selected=candidate,
                        close_target_q=close_target_q,
                        contact_command_q=branch_open.grasp.command_q_rad[-1],
                        contact_model_q=np.asarray(branch_open.grasp.model_q_rad[-1]),
                        base_T_torso=base_T_torso,
                        plane_point=plane_point,
                        down=down,
                        arm=arm,
                        fixed_close_validator=fixed_close_validator,
                    )
                finally:
                    attached_lift_planning_s += time.monotonic() - lift_started
                return branch_open, branch_lift, branch_fixed_close_sweep

            branch, complete, failures = _try_branch_pool(
                branches,
                candidate_ids=candidate_ids,
                attempt=attempt_branch,
                report=report,
            )
            branch_rejections.extend(failures)
            if branch is not None and complete is not None:
                selected = subset_candidates[branch.candidate_local_index]
                selected_branch = branch
                selected_round = search_round
                if pregrasp_only:
                    if not isinstance(complete, _ApproachBranchPlan):
                        raise RuntimeError("pregrasp search returned an invalid branch result")
                    approach_plan = complete
                else:
                    if not isinstance(complete, tuple):
                        raise RuntimeError("task search returned an invalid branch result")
                    open_plan, lift_plan, fixed_close_sweep_plan = complete
                break

            if open_planner_cache is None:
                _cleanup(planner)
            planner = None
            for local_index in range(len(subset_candidates)):
                global_index = subset_indices[local_index]
                candidate_id = str(candidates[global_index]["candidate_id"])
                count = ik_branch_counts[local_index]
                if count == 0:
                    branch_rejections.append(
                        {
                            "candidate_id": candidate_id,
                            "stage": "batched_pregrasp_ik",
                            "reason": (
                                f"0 unique collision-valid solutions from {IK_SEEDS} "
                                "dedicated seeds"
                            ),
                        }
                    )
                remaining_indices.remove(global_index)
                report(
                    f"exhausted all {count} unique branches from {IK_SEEDS} dedicated "
                    f"IK seeds for {candidate_id}"
                )

        if selected is None or selected_branch is None:
            raise RuntimeError(
                "all qualified cube grasp IK branches failed route validation: "
                f"{branch_rejections}"
            )

        if pregrasp_only:
            if approach_plan is None:
                raise RuntimeError("pregrasp search ended without a selected route")
            object_T_grasp = _candidate_transform(selected)
            inbound = _reverse_trajectory(approach_plan.approach)
            inbound = PlannedTrajectory(
                from_pose_id="move_to_pregrasp",
                to_pose_id="return_to_clearance",
                sample_time_s=inbound.sample_time_s,
                command_q_rad=inbound.command_q_rad,
                model_q_rad=inbound.model_q_rad,
                planning_time_s=0.0,
            )
            report(
                f"selected {selected['candidate_id']}; planned only the reversible "
                "clearance-to-pregrasp boundary route"
            )
            return TabletopPregraspPlan(
                request_sha256=request.content_sha256,
                arm=arm,
                selected_candidate_id=str(selected["candidate_id"]),
                object_T_grasp=tuple(
                    tuple(float(value) for value in row) for row in object_T_grasp
                ),
                open_active_dex3_q_rad=tuple(open_target_q),
                outbound=approach_plan.approach,
                inbound=inbound,
                planner_provenance={
                    **model_source_hashes(),
                    "curobo_commit": CUROBO_COMMIT,
                    "arm": arm,
                    "presentation_id": request.presentation_id,
                    "fixture": None if request.fixture is None else request.fixture.to_dict(),
                    "elapsed_s": time.monotonic() - started,
                    "stage_timings_s": {
                        "strict_model_resolution": strict_model_resolution_s,
                        "optimizer_model_clone": optimizer_model_clone_s,
                        "open_optimizer_setup": open_optimizer_setup_s,
                        "open_optimizer_reconfiguration": (open_optimizer_reconfiguration_s),
                        "independent_batched_ik_setup": batched_ik_setup_s,
                        "independent_batched_ik": batched_ik_s,
                        "batched_fixed_close_candidate_pruning": (batched_fixed_close_s),
                        "strict_pregrasp_endpoint_check": endpoint_precheck_s,
                        "fixed_close_sweep_model_resolution": (
                            fixed_close_validator.cache_build_s
                        ),
                        "fixed_close_sweep_validation": (fixed_close_sweep_validation_s),
                        "pregrasp_route_planning_and_validation": open_branch_planning_s,
                    },
                    "grasp_shortlist_sha256": request.grasp_shortlist_sha256,
                    "candidate_count": len(candidates),
                    "candidate_count_after_fixed_close_pruning": len(remaining_indices),
                    "execution_maximum_arm_velocity_rad_s": (request.maximum_arm_velocity_rad_s),
                    "selection_policy": (
                        "independent_batched_curobo_pregrasp_ik_with_strict_"
                        "clearance_to_pregrasp_grasp_approach_and_fixed_close_"
                        "sweep_validation"
                    ),
                    "pregrasp_ik_batches": selected_round,
                    "pregrasp_ik_seeds_per_candidate": IK_SEEDS,
                    "pregrasp_ik_duplicate_tolerance_rad": (IK_BRANCH_DUPLICATE_TOLERANCE_RAD),
                    "pregrasp_ik_unique_branches_by_candidate": {
                        candidate_ids[index]: branch_counts[index]
                        for index in range(len(candidate_ids))
                    },
                    "pregrasp_ik_branches_tested": branch_attempt_count,
                    "selected_pregrasp_solver_seed_index": selected_branch.solver_seed_index,
                    "selected_pregrasp_position_error_m": selected_branch.position_error_m,
                    "selected_pregrasp_rotation_error_rad": selected_branch.rotation_error_rad,
                    "rejected_grasp_branches": branch_rejections,
                    "open_hand_minimum_plane_clearance_m": (
                        approach_plan.minimum_plane_clearance_m
                    ),
                    "open_hand_minimum_plane_clearance_link": (approach_plan.minimum_plane_link),
                    "open_hand_minimum_plane_clearance_sample": (
                        approach_plan.minimum_plane_sample
                    ),
                    "measured_empty_open_active_dex3_q_rad": list(open_q),
                    "fixed_close_sweep_sample_count": (
                        approach_plan.fixed_close_sweep.sample_count
                    ),
                    "fixed_close_sweep_maximum_joint_step_rad": (
                        FINGER_SWEEP_MAXIMUM_JOINT_STEP_RAD
                    ),
                    "fixed_close_sweep_minimum_plane_clearance_m": (
                        approach_plan.fixed_close_sweep.minimum_plane_clearance_m
                    ),
                    "fixed_close_sweep_minimum_plane_link": (
                        approach_plan.fixed_close_sweep.minimum_plane_link
                    ),
                    "fixed_close_sweep_minimum_plane_sample": (
                        approach_plan.fixed_close_sweep.minimum_plane_sample
                    ),
                    "fixed_close_sweep_minimum_fixture_clearance_m": (
                        approach_plan.fixed_close_sweep.minimum_fixture_clearance_m
                    ),
                    "fixed_close_sweep_minimum_fixture_link": (
                        approach_plan.fixed_close_sweep.minimum_fixture_link
                    ),
                    "fixed_close_sweep_minimum_fixture_sample": (
                        approach_plan.fixed_close_sweep.minimum_fixture_sample
                    ),
                    "planning_scope": "clearance_to_pregrasp_only",
                    "candidate_validation_scope": (
                        "clearance_to_pregrasp checks every hand link against the cube; "
                        "contact-tip exceptions apply only to the unexecuted linear grasp "
                        "approach; the complete fixed descriptor close sweep is checked "
                        "at contact; only the reversible pregrasp route is serialized"
                    ),
                    "open_optimizer_reused": open_optimizer_reused,
                    "open_optimizer_topology_rebuilt": open_optimizer_topology_rebuilt,
                },
            )

        if open_plan is None or lift_plan is None or fixed_close_sweep_plan is None:
            raise RuntimeError("complete task search ended without a complete branch")

        approach = open_plan.approach
        grasp = open_plan.grasp
        retention_test_lift = lift_plan.retention_test_lift
        lift = lift_plan.payload_lift
        object_T_grasp = _candidate_transform(selected)
        lower = _reverse_trajectory(lift)
        lower = PlannedTrajectory(
            from_pose_id="payload_lift",
            to_pose_id="payload_lower",
            sample_time_s=lower.sample_time_s,
            command_q_rad=lower.command_q_rad,
            model_q_rad=lower.model_q_rad,
            planning_time_s=lower.planning_time_s,
        )
        replace = _reverse_trajectory(retention_test_lift)
        replace = PlannedTrajectory(
            from_pose_id="payload_lower",
            to_pose_id="payload_replace",
            sample_time_s=replace.sample_time_s,
            command_q_rad=replace.command_q_rad,
            model_q_rad=replace.model_q_rad,
            planning_time_s=replace.planning_time_s,
        )
        reverse_grasp = _reverse_trajectory(grasp)
        retreat = PlannedTrajectory(
            from_pose_id="payload_replace",
            to_pose_id="grasp_retreat",
            sample_time_s=reverse_grasp.sample_time_s,
            command_q_rad=reverse_grasp.command_q_rad,
            model_q_rad=reverse_grasp.model_q_rad,
            planning_time_s=0.0,
        )
        reverse_approach = _reverse_trajectory(approach)
        return_clearance = PlannedTrajectory(
            from_pose_id="grasp_retreat",
            to_pose_id="return_to_clearance",
            sample_time_s=reverse_approach.sample_time_s,
            command_q_rad=reverse_approach.command_q_rad,
            model_q_rad=reverse_approach.model_q_rad,
            planning_time_s=0.0,
        )
        report(f"selected {selected['candidate_id']}; planned complete pick/lift/replace/return")
        return TabletopTaskPlan(
            request_sha256=request.content_sha256,
            arm=arm,
            selected_candidate_id=str(selected["candidate_id"]),
            object_T_grasp=tuple(tuple(float(v) for v in row) for row in object_T_grasp),
            open_active_dex3_q_rad=tuple(open_target_q),
            close_target_active_dex3_q_rad=tuple(close_target_q),
            initial_active_dex3_q_rad=(
                request.planning_snapshot.left_dex3_q_rad
                if arm == "left"
                else request.planning_snapshot.right_dex3_q_rad
            ),
            trajectories=(
                approach,
                grasp,
                retention_test_lift,
                lift,
                lower,
                replace,
                retreat,
                return_clearance,
            ),
            phase_order=(
                "move_to_pregrasp",
                "grasp_approach",
                "retention_test_lift",
                "payload_lift",
                "payload_lower",
                "payload_replace",
                "grasp_retreat",
                "return_to_clearance",
            ),
            planner_provenance={
                **model_source_hashes(),
                "curobo_commit": CUROBO_COMMIT,
                "arm": arm,
                "presentation_id": request.presentation_id,
                "fixture": (None if request.fixture is None else request.fixture.to_dict()),
                "elapsed_s": time.monotonic() - started,
                "stage_timings_s": {
                    "strict_model_resolution": strict_model_resolution_s,
                    "optimizer_model_clone": optimizer_model_clone_s,
                    "open_optimizer_setup": open_optimizer_setup_s,
                    "open_optimizer_reconfiguration": open_optimizer_reconfiguration_s,
                    "independent_batched_ik_setup": batched_ik_setup_s,
                    "independent_batched_ik": batched_ik_s,
                    "batched_fixed_close_candidate_pruning": batched_fixed_close_s,
                    "strict_pregrasp_endpoint_check": endpoint_precheck_s,
                    "fixed_close_sweep_model_resolution": (fixed_close_validator.cache_build_s),
                    "fixed_close_sweep_validation": fixed_close_sweep_validation_s,
                    "open_branch_planning_and_validation": open_branch_planning_s,
                    "attached_lift_planning_and_validation": attached_lift_planning_s,
                },
                "grasp_shortlist_sha256": request.grasp_shortlist_sha256,
                "candidate_count": len(candidates),
                "candidate_count_after_fixed_close_pruning": len(remaining_indices),
                "required_candidate_id": required_candidate_id,
                "execution_maximum_arm_velocity_rad_s": (request.maximum_arm_velocity_rad_s),
                "selection_policy": (
                    "independent_batched_curobo_pregrasp_ik_with_strict_endpoint_and_"
                    "complete_lifecycle_validation"
                ),
                "pregrasp_ik_batches": selected_round,
                "pregrasp_ik_seeds_per_candidate": IK_SEEDS,
                "pregrasp_ik_duplicate_tolerance_rad": IK_BRANCH_DUPLICATE_TOLERANCE_RAD,
                "pregrasp_ik_unique_branches_by_candidate": {
                    candidate_ids[index]: branch_counts[index]
                    for index in range(len(candidate_ids))
                },
                "pregrasp_ik_branches_tested": branch_attempt_count,
                "selected_pregrasp_solver_seed_index": (selected_branch.solver_seed_index),
                "selected_pregrasp_position_error_m": (selected_branch.position_error_m),
                "selected_pregrasp_rotation_error_rad": (selected_branch.rotation_error_rad),
                "rejected_grasp_branches": branch_rejections,
                "qualification": (
                    "GraspGenX pose + intrinsic retention + stationary-cube fixed-close "
                    "exact-mesh table qualification"
                ),
                "finger_close_command_policy": (
                    "one descriptor-defined target for every grasp; physical contact limits "
                    "measured travel; candidate PhysX endpoints are qualification evidence only"
                ),
                "measured_empty_open_active_dex3_q_rad": list(open_q),
                "fixed_close_sweep_sample_count": fixed_close_sweep_plan.sample_count,
                "fixed_close_sweep_maximum_joint_step_rad": (FINGER_SWEEP_MAXIMUM_JOINT_STEP_RAD),
                "fixed_close_sweep_minimum_plane_clearance_m": (
                    fixed_close_sweep_plan.minimum_plane_clearance_m
                ),
                "fixed_close_sweep_minimum_plane_link": (
                    fixed_close_sweep_plan.minimum_plane_link
                ),
                "fixed_close_sweep_minimum_plane_sample": (
                    fixed_close_sweep_plan.minimum_plane_sample
                ),
                "fixed_close_sweep_minimum_fixture_clearance_m": (
                    fixed_close_sweep_plan.minimum_fixture_clearance_m
                ),
                "fixed_close_sweep_minimum_fixture_link": (
                    fixed_close_sweep_plan.minimum_fixture_link
                ),
                "fixed_close_sweep_minimum_fixture_sample": (
                    fixed_close_sweep_plan.minimum_fixture_sample
                ),
                "attachment_policy": (
                    "CuRobo AttachmentManager deterministic conservative 3x3x3 cuboid cover"
                ),
                "attached_payload_support_policy": (
                    "cube/presentation-fixture contact is exempt only during the direct "
                    "separation lift; the exact generated route is independently checked "
                    "for every robot/fixture collision without the attached cube"
                ),
                "attachment_sphere_count": lift_plan.attachment_sphere_count,
                "visual_policy": (
                    "fresh stationary AprilCube observation after supported escape; "
                    "support plane inferred from its gravity-aligned bottom face; "
                    "configured local open-transit patch centred under the cube; "
                    "optional presenter fixed from the observed aligned cube pose"
                ),
                "open_transit_world_scope": (
                    f"{arm} wrist/hand only during patch steering; this optimizer "
                    "sphere set is deliberately permissive; every IK endpoint and "
                    "generated route is independently revalidated against strict "
                    "full-robot self/cube geometry"
                ),
                "table_plane_policy": f"local_{arm}_wrist_hand_payload_only",
                "required_hand_plane_clearance_m": (request.minimum_hand_plane_clearance_m),
                "open_transit_table_patch_dimensions_m": list(
                    request.open_transit_table_patch_dimensions_m
                ),
                "open_route_minimum_plane_clearance_m": (open_plan.minimum_plane_clearance_m),
                "open_route_minimum_plane_link": open_plan.minimum_plane_link,
                "open_route_minimum_plane_sample": open_plan.minimum_plane_sample,
                "closed_lift_hand_minimum_plane_clearance_m": (
                    lift_plan.closed_hand_minimum_plane_clearance_m
                ),
                "payload_lift_minimum_plane_clearance_m": (
                    lift_plan.payload_minimum_plane_clearance_m
                ),
                "payload_start_plane_clearance_m": (lift_plan.payload_start_plane_clearance_m),
                "retention_test_lift_requested_m": request.retention_test_lift_m,
                "retention_test_lift_actual_m": (lift_plan.retention_test_lift_actual_m),
                "return_policy": "exact reverse lift, grasp, and approach trajectories",
                "open_optimizer_reused": open_optimizer_reused,
                "open_optimizer_topology_rebuilt": open_optimizer_topology_rebuilt,
            },
        )
    finally:
        if open_planner_cache is None:
            _cleanup(planner)


def plan_tabletop_pregrasp(
    request: TabletopTaskRequest,
    *,
    open_planner_cache: ReusableOpenPlanner | None = None,
    progress: Callable[[str], None] | None = None,
) -> TabletopPregraspPlan:
    """Plan only the reversible clearance-to-pregrasp boundary route."""

    result = _plan_tabletop(
        request,
        pregrasp_only=True,
        open_planner_cache=open_planner_cache,
        progress=progress,
    )
    if not isinstance(result, TabletopPregraspPlan):
        raise TypeError("pregrasp planner returned a complete task")
    return result


def plan_tabletop_task(
    request: TabletopTaskRequest,
    *,
    required_candidate_id: str | None = None,
    open_planner_cache: ReusableOpenPlanner | None = None,
    progress: Callable[[str], None] | None = None,
) -> TabletopTaskPlan:
    """Plan a complete task while preserving alternate arm IK branches."""

    result = _plan_tabletop(
        request,
        required_candidate_id=required_candidate_id,
        pregrasp_only=False,
        open_planner_cache=open_planner_cache,
        progress=progress,
    )
    if not isinstance(result, TabletopTaskPlan):
        raise TypeError("complete task planner returned only a pregrasp route")
    return result


def plan_tabletop_pick_place(
    request: TabletopPickPlaceRequest,
    *,
    open_planner_cache: ReusableOpenPlanner | None = None,
    progress: Callable[[str], None] | None = None,
) -> TabletopPickPlacePlan:
    """Plan one fixed pick/place sequence without introducing a task graph."""

    report = progress or (lambda _message: None)
    started = time.monotonic()
    source = request.source_request
    report("planning the qualified source grasp and attached lift")
    first_source_task = plan_tabletop_task(
        source,
        open_planner_cache=open_planner_cache,
        progress=report,
    )
    destination = destination_request_for_pick_place(request)
    _shortlist, candidates = _load_shortlist(source)
    candidate_ids = [str(value["candidate_id"]) for value in candidates]
    candidate_order = [
        first_source_task.selected_candidate_id,
        *(value for value in candidate_ids if value != first_source_task.selected_candidate_id),
    ]
    pick_place_rejections: list[dict[str, str]] = []
    source_task = destination_task = transfer = transfer_provenance = None
    for candidate_id in candidate_order:
        if candidate_id == first_source_task.selected_candidate_id:
            current_source = first_source_task
        else:
            report(f"trying next source/destination grasp {candidate_id}")
            try:
                current_source = plan_tabletop_task(
                    source,
                    required_candidate_id=candidate_id,
                    open_planner_cache=open_planner_cache,
                    progress=report,
                )
            except RuntimeError as error:
                pick_place_rejections.append(
                    {
                        "candidate_id": candidate_id,
                        "stage": "source",
                        "reason": str(error),
                    }
                )
                continue
        report(
            "planning the destination contact and reverse retreat while preserving "
            f"grasp {candidate_id}"
        )
        try:
            current_destination = plan_tabletop_task(
                destination,
                required_candidate_id=candidate_id,
                open_planner_cache=open_planner_cache,
                progress=report,
            )
            report("planning the attached-payload bridge between validated lifted states")
            current_transfer, current_transfer_provenance = _plan_attached_transfer(
                request,
                current_source,
                current_destination,
            )
        except RuntimeError as error:
            pick_place_rejections.append(
                {
                    "candidate_id": candidate_id,
                    "stage": "destination_or_transfer",
                    "reason": str(error),
                }
            )
            continue
        source_task = current_source
        destination_task = current_destination
        transfer = current_transfer
        transfer_provenance = current_transfer_provenance
        break
    if (
        source_task is None
        or destination_task is None
        or transfer is None
        or transfer_provenance is None
    ):
        raise RuntimeError(
            "no qualified grasp passed the complete source, destination, and attached "
            f"transfer lifecycle: {pick_place_rejections}"
        )
    destination_lower = _rename_trajectory(
        destination_task.trajectories[4],
        from_pose_id="payload_transfer",
        to_pose_id="placement_lower",
    )
    destination_contact = _rename_trajectory(
        destination_task.trajectories[5],
        from_pose_id="placement_lower",
        to_pose_id="placement_contact",
    )
    destination_retreat = _rename_trajectory(
        destination_task.trajectories[6],
        from_pose_id="placement_contact",
        to_pose_id="placement_retreat",
    )
    destination_clearance = _rename_trajectory(
        destination_task.trajectories[7],
        from_pose_id="placement_retreat",
        to_pose_id="return_to_clearance",
    )
    trajectories = (
        *source_task.trajectories[:4],
        transfer,
        destination_lower,
        destination_contact,
        destination_retreat,
        destination_clearance,
    )
    report(
        f"pick-place plan ready with {len(trajectories)} fixed motion phases; "
        "the source and destination use the same qualified grasp"
    )
    return TabletopPickPlacePlan(
        request_sha256=request.content_sha256,
        arm=source.arm,
        selected_candidate_id=source_task.selected_candidate_id,
        source_task=source_task,
        destination_task=destination_task,
        trajectories=trajectories,
        phase_order=PICK_PLACE_PHASE_ORDER,
        planner_provenance={
            **model_source_hashes(),
            "curobo_commit": CUROBO_COMMIT,
            "elapsed_s": time.monotonic() - started,
            "source_task_sha256": source_task.content_sha256,
            "destination_task_sha256": destination_task.content_sha256,
            "transfer": transfer_provenance,
            "rejected_pick_place_grasps": pick_place_rejections,
            "selection_policy": ("first-grasp-passing-source-destination-and-attached-transfer"),
            "task_structure": "fixed_pick_place_sequence",
        },
    )
