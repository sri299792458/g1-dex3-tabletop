"""CuRobo-only planner for a selected-Dex3 cube pick/lift/replace task."""

from __future__ import annotations

import copy
import gc
import hashlib
import time
from collections.abc import Callable
from dataclasses import dataclass
from itertools import permutations, product
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
    IK_SEEDS,
    TRAJECTORY_INTERPOLATION_DT_S,
    CuroboKinematicCollisionChecker,
    _joint_state_dt,
    _reverse_trajectory,
    _self_collision_pair_penetrations,
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
    CharucoSupportedEscapeRequest,
    RetentionRouteValidationRequest,
    RetentionRouteValidationResult,
    SupportedEscapePlan,
    TabletopTaskPlan,
    TabletopTaskRequest,
)

ROOT = Path(__file__).resolve().parents[3]
WORLD_COLLISION_DISABLE_RADIUS_EPSILON_M = 1.0e-6


@dataclass(frozen=True, slots=True)
class _PregraspBranch:
    """One collision-valid CuRobo IK solution for one pregrasp goal."""

    goalset_local_index: int
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
class _LiftBranchPlan:
    retention_test_lift: PlannedTrajectory
    payload_lift: PlannedTrajectory
    retention_test_lift_actual_m: float
    closed_hand_minimum_plane_clearance_m: float
    payload_minimum_plane_clearance_m: float
    payload_start_plane_clearance_m: float
    attachment_sphere_count: int


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
    return tuple(f"{arm}_hand_{suffix}_link" for suffix in ("thumb_2", "middle_1", "index_1"))


def _local_table_plane_links(arm: str) -> tuple[str, ...]:
    return (
        f"{arm}_wrist_pitch_link",
        f"{arm}_wrist_yaw_link",
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
    for candidate in candidates:
        if candidate.get("execution_evidence", {}).get("intrinsic_retention_passed") is not True:
            raise ValueError(
                "qualified cube candidate lacks intrinsic retention evidence: "
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
    """Map the uppermost physical cube face to canonical object +Z."""

    detected = np.asarray(base_T_detected_object, dtype=np.float64)
    rotations = []
    for permutation in permutations(range(3)):
        for signs in product((-1.0, 1.0), repeat=3):
            symmetry = np.zeros((3, 3), dtype=np.float64)
            symmetry[list(permutation), range(3)] = signs
            if np.linalg.det(symmetry) > 0.0:
                rotations.append(symmetry)
    candidates = [
        symmetry
        for symmetry in rotations
        if float((detected[:3, :3] @ symmetry)[2, 2]) >= np.cos(np.deg2rad(20.0))
    ]
    if not candidates:
        raise RuntimeError(
            "AprilCube is not resting on a face: no face normal points upward within 20 degrees"
        )
    # The four rotations about the upward face are physically equivalent.
    # Select the smallest frame change deterministically; tabletop yaw remains
    # exactly whatever the detector observed.
    symmetry = max(candidates, key=lambda value: (float(np.trace(value)), *value.ravel()))
    canonical = detected.copy()
    canonical[:3, :3] = detected[:3, :3] @ symmetry
    return canonical


def _table_from_resting_object(
    request: TabletopTaskRequest, base_T_torso: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Derive only the supporting plane from the resting cube.

    A single cube observation contains no evidence about the finite table
    footprint, its yaw, or its thickness.  The first return value is therefore
    a point on the top plane, not a fabricated table-frame origin.
    """

    base_T_camera = base_T_torso @ np.asarray(request.torso_T_camera)
    detected_object = base_T_camera @ np.asarray(request.observation.camera_T_object)
    base_T_object = _canonical_resting_cube_pose(detected_object)
    object_up = base_T_object[:3, 2]
    down = -object_up
    extent = request.object_dimensions_m[2]
    support_height = 0.0 if request.fixture is None else request.fixture.support_height_m
    top_origin = base_T_object[:3, 3] + (0.5 * extent + support_height) * down
    return top_origin, base_T_object, down


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
) -> dict[str, Any]:
    plane_point, base_T_object, down = _table_from_resting_object(request, base_T_torso)
    scene: dict[str, Any] = {"cuboid": {}}
    if include_cube:
        scene["cuboid"]["cube"] = {
            "dims": list(request.object_dimensions_m),
            "pose": _pose_list(base_T_object),
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


def _fixture_collision_mesh(
    request: TabletopTaskRequest,
    base_T_object: np.ndarray,
    down: np.ndarray,
):
    """Load the hash-checked fixture once in base coordinates for CPU rechecks."""

    path = _fixture_mesh_path(request)
    fixture_pose = _base_T_fixture(request, base_T_object, down)
    if path is None or fixture_pose is None:
        return None
    assert request.fixture is not None
    import trimesh

    mesh = trimesh.load(path, force="mesh", process=True)
    if not isinstance(mesh, trimesh.Trimesh) or not mesh.is_watertight:
        raise ValueError("fixture collision mesh must be one watertight solid")
    mesh.apply_scale(np.asarray(request.fixture.mesh_scale, dtype=np.float64))
    mesh.apply_transform(fixture_pose)
    return mesh


def _fixture_clearance_from_spheres(
    sphere_array: np.ndarray,
    *,
    config,
    fixture_mesh,
) -> tuple[float, str, int]:
    """Return minimum exact-mesh clearance for already-computed robot spheres."""

    import trimesh

    values = np.asarray(sphere_array, dtype=np.float64)
    if values.ndim != 3 or values.shape[2] != 4:
        raise ValueError("fixture sphere array must have shape N x S x 4")
    valid_samples, valid_spheres = np.nonzero(values[..., 3] > 0.0)
    if len(valid_samples) == 0:
        raise RuntimeError("fixture collision recheck found no enabled robot spheres")
    selected = values[valid_samples, valid_spheres]
    # trimesh uses positive signed distance inside a watertight mesh and
    # negative distance outside. Sphere clearance is therefore -distance-r.
    signed = trimesh.proximity.signed_distance(fixture_mesh, selected[:, :3])
    clearances = -np.asarray(signed, dtype=np.float64) - selected[:, 3]
    if not np.all(np.isfinite(clearances)):
        raise RuntimeError("fixture collision recheck produced a non-finite distance")
    minimum_index = int(np.argmin(clearances))
    sphere_index = int(valid_spheres[minimum_index])
    link_index = int(config.link_sphere_idx_map.reshape(-1)[sphere_index].item())
    index_to_name = {value: name for name, value in config.link_name_to_idx_map.items()}
    if link_index not in index_to_name:
        raise RuntimeError("fixture collision recheck could not resolve a sphere link")
    return (
        float(clearances[minimum_index]),
        index_to_name[link_index],
        int(valid_samples[minimum_index]),
    )


def _local_plane_clearance(
    planner,
    model_q: np.ndarray,
    *,
    arm: str,
    plane_point: np.ndarray,
    down: np.ndarray,
    include_payload: bool,
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
    )


def _local_plane_clearance_from_spheres(
    sphere_array: np.ndarray,
    *,
    config,
    arm: str,
    plane_point: np.ndarray,
    down: np.ndarray,
    include_payload: bool,
) -> tuple[float, str, int]:
    """Evaluate the table guard from already-computed CuRobo spheres."""

    import torch

    values = np.asarray(sphere_array, dtype=np.float64)
    if values.ndim != 3 or values.shape[2] != 4:
        raise ValueError("table-plane sphere array must have shape N x S x 4")
    link_names = list(_local_table_plane_links(arm))
    if include_payload:
        link_names.append(attachment_link(arm))
    minimum = float("inf")
    minimum_link = ""
    minimum_sample = -1
    up = -np.asarray(down, dtype=np.float64)
    point = np.asarray(plane_point, dtype=np.float64)
    for link_name in link_names:
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
        flat_index = int(np.argmin(clearance))
        value = float(clearance.reshape(-1)[flat_index])
        if value < minimum:
            sample_index, _sphere_index = np.unravel_index(flat_index, clearance.shape)
            minimum = value
            minimum_link = link_name
            minimum_sample = int(sample_index)
    if not np.isfinite(minimum):
        raise RuntimeError("table-plane guard found no enabled local collision spheres")
    return minimum, minimum_link, minimum_sample


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
    """Return signed robot-sphere clearances to nearby named scene cuboids."""

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
        hits = np.argwhere(enabled & (clearances < COLLISION_ACTIVATION_DISTANCE_M))
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
        self_collision_check=True,
        use_cuda_graph=True,
        random_seed=seed,
        optimizer_collision_activation_distance=COLLISION_ACTIVATION_DISTANCE_M,
        interpolation_dt=TRAJECTORY_INTERPOLATION_DT_S,
        interpolation_buffer_size=1000,
    )
    return MotionPlanner(config), device_cfg


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


def _as_numpy(value) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def _pregrasp_matrix(grasp_matrix: np.ndarray, approach_distance_m: float) -> np.ndarray:
    offset = np.eye(4, dtype=np.float64)
    offset[2, 3] = -approach_distance_m
    return np.asarray(grasp_matrix, dtype=np.float64) @ offset


def _enumerate_pregrasp_branches(planner, goals, state) -> list[_PregraspBranch]:
    """Return every collision-valid pregrasp branch produced by CuRobo's IK seeds.

    A goal set is still used so CuRobo can prioritize all retained GraspGenX
    candidates together.  Unlike ``MotionPlanner.plan_grasp``, no branch is
    collapsed to one final arm configuration here.
    """

    result = planner.ik_solver.solve_pose(
        goals,
        return_seeds=IK_SEEDS,
        current_state=state,
    )
    success = _as_numpy(result.success).astype(bool).reshape(-1)
    solutions = _as_numpy(result.solution).reshape(len(success), -1)
    if solutions.shape[1] != 7:
        raise RuntimeError(f"CuRobo pregrasp IK returned invalid shape {solutions.shape}")
    position_error = _as_numpy(result.position_error).reshape(len(success), -1).max(axis=1)
    rotation_error = _as_numpy(result.rotation_error).reshape(len(success), -1).max(axis=1)
    if result.goalset_index is None:
        goalset_index = np.zeros(len(success), dtype=np.int64)
    else:
        goalset_index = _as_numpy(result.goalset_index).reshape(len(success), -1)[:, 0]

    branches: list[_PregraspBranch] = []
    for solver_seed_index in np.flatnonzero(success):
        branches.append(
            _PregraspBranch(
                goalset_local_index=int(goalset_index[solver_seed_index]),
                solver_seed_index=int(solver_seed_index),
                model_q_rad=np.asarray(solutions[solver_seed_index], dtype=np.float64),
                position_error_m=float(position_error[solver_seed_index]),
                rotation_error_rad=float(rotation_error[solver_seed_index]),
            )
        )
    return branches


def _try_branch_pool(
    branches: list[_PregraspBranch],
    *,
    candidate_ids: list[str],
    attempt: Callable[[_PregraspBranch], tuple[_OpenBranchPlan, _LiftBranchPlan]],
    report: Callable[[str], None],
) -> tuple[
    _PregraspBranch | None,
    tuple[_OpenBranchPlan, _LiftBranchPlan] | None,
    list[dict[str, Any]],
]:
    """Try the finite IK branch pool without discarding a candidate early."""

    failures: list[dict[str, Any]] = []
    for pool_index, branch in enumerate(branches, start=1):
        if not 0 <= branch.goalset_local_index < len(candidate_ids):
            raise RuntimeError("CuRobo returned an invalid pregrasp goal-set index")
        candidate_id = candidate_ids[branch.goalset_local_index]
        report(
            f"trying {candidate_id} IK branch {pool_index}/{len(branches)} "
            f"(solver seed {branch.solver_seed_index})"
        )
        try:
            result = attempt(branch)
        except _BranchRejected as rejection:
            failure = {
                "candidate_id": candidate_id,
                "pool_branch_index": pool_index,
                "solver_seed_index": branch.solver_seed_index,
                "stage": rejection.stage,
                "reason": rejection.reason,
            }
            failures.append(failure)
            report(
                f"rejected {candidate_id} IK branch {pool_index}/{len(branches)} "
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


def _goalset_ik_failure_diagnostic(
    *,
    planner,
    robot: dict[str, Any],
    scene: dict[str, Any],
    goals,
    state,
    candidates: list[dict[str, Any]],
    remaining: list[int],
    device_cfg,
    arm: str,
    disabled_collision_links: set[str],
) -> str:
    """Name physical constraints behind an otherwise opaque goal-set failure."""

    def array(value) -> np.ndarray:
        if hasattr(value, "detach"):
            value = value.detach().cpu().numpy()
        return np.asarray(value)

    planner.disable_link_collision(list(disabled_collision_links))
    try:
        result = planner.ik_solver.solve_pose(
            goals,
            return_seeds=IK_SEEDS,
            current_state=state,
        )
    finally:
        planner.enable_link_collision(list(disabled_collision_links))

    position_error = array(result.position_error).reshape(-1)
    rotation_error = array(result.rotation_error).reshape(-1)
    position_tolerance = float(planner.ik_solver.config.position_tolerance)
    rotation_tolerance = float(planner.ik_solver.config.orientation_tolerance)
    converged = np.flatnonzero(
        (position_error < position_tolerance) & (rotation_error < rotation_tolerance)
    )
    best_position_mm = float(np.min(position_error) * 1000.0)
    best_rotation_deg = float(np.rad2deg(np.min(rotation_error)))
    if len(converged) == 0:
        return (
            "IK diagnostic found no Cartesian-converged branch: "
            f"best position error={best_position_mm:.3f}mm, "
            f"best rotation error={best_rotation_deg:.3f}deg"
        )

    solutions = array(result.solution).reshape(-1, 7)
    goalset_indices = array(result.goalset_index).reshape(-1)
    pair_samples = _self_collision_pair_penetrations(
        robot=robot,
        q_samples=solutions[converged],
        device_cfg=device_cfg,
    )
    world_samples = _world_cuboid_clearances(
        robot=robot,
        q_samples=solutions[converged],
        scene=scene,
        device_cfg=device_cfg,
        disabled_links=disabled_collision_links,
    )
    branches: list[str] = []
    seen: set[tuple[str, tuple[tuple[str, str], ...], tuple[tuple[str, str], ...]]] = set()
    for seed_index, pair_depths, world_clearances in zip(
        converged, pair_samples, world_samples, strict=True
    ):
        local_goal_index = int(goalset_indices[seed_index])
        candidate_id = "unknown candidate"
        if 0 <= local_goal_index < len(remaining):
            candidate_id = str(candidates[remaining[local_goal_index]]["candidate_id"])
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
        f"IK diagnostic found {len(converged)}/{int(array(result.success).size)} "
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
        terminal_clearance = -float(np.dot(down, actual_target[:3, 3] - plane_point))
        if terminal_clearance < 0.05:
            raise RuntimeError(
                f"supported escape finishes only {terminal_clearance:.4f}m above the table plane"
            )
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
            f"CuRobo supported escape passed; G-frame plane clearance={terminal_clearance:.4f}m"
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
    q29 = np.asarray(request.observation.snapshot.measured_q29_rad).copy()
    q29[np.asarray(arm_indices(request.arm))] = np.asarray(command_q)
    return RobotSnapshot(
        measured_q29_rad=tuple(q29),
        left_dex3_q_rad=request.observation.snapshot.left_dex3_q_rad,
        right_dex3_q_rad=request.observation.snapshot.right_dex3_q_rad,
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

    # Start the Cartesian segment at the trajectory's actual terminal state,
    # rather than assuming the joint-space optimizer ended at bit-identical IK.
    actual_pregrasp_model_q = np.asarray(approach.model_q_rad[-1], dtype=np.float64)
    actual_pregrasp_state = _joint_state(device_cfg, actual_pregrasp_model_q, arm_joint_names(arm))
    grasp_goal = _goalset([grasp_matrix], device_cfg, arm=arm)
    from curobo.types import ToolPoseCriteria

    criterion = ToolPoseCriteria.linear_motion(
        axis="z",
        non_terminal_scale=1.0,
        project_distance_to_goal=True,
    )
    contact_links = list(_contact_links(arm))
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

    open_route_q = np.concatenate(
        (np.asarray(approach.model_q_rad), np.asarray(grasp.model_q_rad)[1:]),
        axis=0,
    )
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

    cube_scene = _base_scene(request, base_T_torso, include_cube=True)
    route_cube_clearances = _world_cuboid_clearances(
        robot=open_robot,
        q_samples=open_route_q,
        scene=cube_scene,
        device_cfg=device_cfg,
        disabled_links=set(_contact_links(arm)),
        checker=strict_checker,
    )
    for sample_index, clearances in enumerate(route_cube_clearances):
        if not clearances:
            continue
        (link_name, object_name), clearance = min(clearances.items(), key=lambda item: item[1])
        raise _BranchRejected(
            "open_route_strict_cube_collision",
            f"{link_name}/{object_name}={clearance * 1000.0:+.3f}mm "
            f"clearance at sample {sample_index}",
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
) -> _LiftBranchPlan:
    """Plan with the descriptor close target; live measured fingers are rechecked later."""

    planner = None
    try:
        contact_snapshot = _snapshot_at_arm_q(request, contact_command_q)
        close_target_robot, _ = build_tabletop_robot_config(
            arm=arm,
            snapshot=contact_snapshot,
            joint_position_offsets_rad=request.joint_position_offsets_rad,
            active_finger_q_rad=tuple(close_target_q),
        )
        _use_moving_grasp_frame_only(close_target_robot, arm=arm)
        attached_scene = _base_scene(request, base_T_torso, include_cube=False)
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
                "planned payload route never reaches the configured retention-test height",
            )
        split_index = int(split_candidates[0])
        if split_index <= 0 or split_index >= len(lift_q) - 1:
            raise _BranchRejected(
                "retention_test_lift",
                f"retention-test boundary falls at unusable sample {split_index}/{len(lift_q) - 1}",
            )
        retention_test_lift, remaining_lift = _split_lift_trajectory(
            lift,
            split_index=split_index,
        )
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


class RetentionRouteValidator:
    """Cached FK/collision checker for one frozen task's measured contact pose."""

    def __init__(self, tabletop: TabletopTaskRequest, task: TabletopTaskPlan) -> None:
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
            snapshot=tabletop.observation.snapshot,
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
        self.plane_point, base_T_object, self.down = _table_from_resting_object(
            tabletop,
            base_T_torso,
        )
        self.fixture_mesh = _fixture_collision_mesh(
            tabletop,
            base_T_object,
            self.down,
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
                "measured stalled Dex3 posture invalidates the frozen payload route: "
                f"{pair[0]}/{pair[1]}={penetration * 1000.0:.3f}mm penetration "
                f"at sample {sample_index}/{len(route_q) - 1}"
            )
        spheres = (
            self.checker.robot_spheres(route_q, joint_names=self.active_joint_names)
            .detach()
            .cpu()
            .numpy()
            .reshape(len(route_q), -1, 4)
        )
        hand_clearance, hand_link, hand_sample = _local_plane_clearance_from_spheres(
            spheres,
            config=self.checker.config.kinematics_config,
            arm=self.arm,
            plane_point=self.plane_point,
            down=self.down,
            include_payload=False,
        )
        if hand_clearance < self.tabletop.minimum_hand_plane_clearance_m:
            raise RuntimeError(
                "measured stalled Dex3 posture leaves insufficient hand/table execution "
                f"margin: clearance={hand_clearance:.4f}m at {hand_link} sample "
                f"{hand_sample}/{len(route_q) - 1}; required="
                f"{self.tabletop.minimum_hand_plane_clearance_m:.4f}m"
            )
        fixture_clearance = fixture_link = fixture_sample = None
        if self.fixture_mesh is not None:
            fixture_clearance, fixture_link, fixture_sample = _fixture_clearance_from_spheres(
                spheres,
                config=self.checker.config.kinematics_config,
                fixture_mesh=self.fixture_mesh,
            )
            if fixture_clearance < 0.0:
                raise RuntimeError(
                    "measured stalled Dex3 posture invalidates the frozen payload route "
                    f"against the presentation fixture: {fixture_link} has "
                    f"{fixture_clearance * 1000.0:.3f}mm clearance at sample "
                    f"{fixture_sample}/{len(route_q) - 1}"
                )
        fixture_report = (
            ""
            if fixture_clearance is None
            else f" and fixture clearance={fixture_clearance:.4f}m at {fixture_link}"
        )
        report(
            "measured stalled-hand retention route passed strict self-collision and "
            f"table-plane checks; minimum hand clearance={hand_clearance:.4f}m"
            f"{fixture_report}"
        )
        return RetentionRouteValidationResult(
            request_sha256=request.content_sha256,
            arm=self.arm,
            selected_candidate_id=self.task.selected_candidate_id,
            route_sample_count=len(route_q),
            minimum_hand_plane_clearance_m=hand_clearance,
            minimum_hand_plane_link=hand_link,
            minimum_hand_plane_sample=hand_sample,
            minimum_fixture_clearance_m=fixture_clearance,
            minimum_fixture_clearance_link=fixture_link,
            minimum_fixture_clearance_sample=fixture_sample,
            planner_provenance={
                **model_source_hashes(),
                "curobo_commit": CUROBO_COMMIT,
                "elapsed_s": time.monotonic() - started,
                "cached_kinematics": True,
                "cache_build_s": self.cache_build_s,
                "required_hand_plane_clearance_m": (self.tabletop.minimum_hand_plane_clearance_m),
                "policy": (
                    "frozen split payload arm route; measured contact-stalled active Dex3; "
                    "strict full-robot self-collision, selected wrist/hand table plane, "
                    "and optional exact presenter mesh"
                ),
                "presentation_id": self.tabletop.presentation_id,
                "fixture_mesh_rechecked": self.fixture_mesh is not None,
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


def plan_tabletop_task(
    request: TabletopTaskRequest,
    *,
    progress: Callable[[str], None] | None = None,
) -> TabletopTaskPlan:
    """Plan a complete task while preserving alternate arm IK branches."""

    report = progress or (lambda _message: None)
    arm = request.arm
    shortlist, candidates = _load_shortlist(request)
    open_profile, close_target_profile = dex3_execution_profile(arm)
    open_q = np.asarray(open_profile, dtype=np.float64)
    close_target_q = np.asarray(close_target_profile, dtype=np.float64)
    reference = np.asarray(request.observation.snapshot.measured_q29_rad)[
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
    goalset_ik_s = 0.0
    endpoint_precheck_s = 0.0
    open_branch_planning_s = 0.0
    attached_lift_planning_s = 0.0
    try:
        import torch
        from curobo.types import DeviceCfg

        query_robot, _ = build_tabletop_robot_config(
            arm=arm,
            snapshot=request.observation.snapshot,
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
        grasp_matrices = [base_T_object @ _candidate_transform(item) for item in candidates]
        approach_distance_m = float(shortlist["execution_contract"]["approach_distance_m"])
        remaining_indices = list(range(len(candidates)))
        branch_rejections: list[dict[str, Any]] = []
        selected: dict[str, Any] | None = None
        selected_branch: _PregraspBranch | None = None
        selected_round = 0
        open_plan: _OpenBranchPlan | None = None
        lift_plan: _LiftBranchPlan | None = None
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
            current_candidate_count = len(subset_candidates)

            def create_open_planner(
                current_transit_robot=resolved_transit_robot,
                current_scene=scene,
                candidate_count=current_candidate_count,
            ):
                nonlocal open_optimizer_setup_s
                setup_started = time.monotonic()
                current_planner, current_device = _planner(
                    current_transit_robot,
                    current_scene,
                    max_goalset=candidate_count,
                    seed=request.random_seed,
                )
                open_optimizer_setup_s += time.monotonic() - setup_started
                current_state = _joint_state(
                    current_device,
                    reference_model,
                    arm_joint_names(arm),
                )
                return current_planner, current_device, current_state

            planner, device_cfg, state = create_open_planner()
            pregrasp_goals = _goalset(pregrasp_matrices, device_cfg, arm=arm)
            stage_started = time.monotonic()
            branches = _enumerate_pregrasp_branches(planner, pregrasp_goals, state)
            goalset_ik_s += time.monotonic() - stage_started
            if not branches:
                diagnostic = _goalset_ik_failure_diagnostic(
                    planner=planner,
                    robot=transit_robot,
                    scene=scene,
                    goals=pregrasp_goals,
                    state=state,
                    candidates=subset_candidates,
                    remaining=list(range(len(subset_candidates))),
                    device_cfg=device_cfg,
                    arm=arm,
                    disabled_collision_links=set(),
                )
                raise RuntimeError(
                    "no collision-valid pregrasp IK branch remains for the qualified "
                    f"candidates; prior branch rejections={branch_rejections}; {diagnostic}"
                )
            represented_local_indices = sorted({branch.goalset_local_index for branch in branches})
            stage_started = time.monotonic()
            endpoint_collision_reasons = _pregrasp_endpoint_self_collision_reasons(
                branches,
                checker=strict_open_checker,
            )
            endpoint_precheck_s += time.monotonic() - stage_started
            endpoint_collision_by_seed = {
                branch.solver_seed_index: reason
                for branch, reason in zip(branches, endpoint_collision_reasons, strict=True)
            }
            report(
                f"CuRobo pregrasp IK round {search_round}: {len(branches)}/{IK_SEEDS} "
                f"collision-valid branches across {len(represented_local_indices)} "
                f"of {len(subset_candidates)} remaining candidates"
            )

            def attempt_branch(
                branch: _PregraspBranch,
                current_candidates=subset_candidates,
                current_matrices=subset_matrices,
                current_open_robot=open_robot,
                current_strict_checker=strict_open_checker,
                current_endpoint_collisions=endpoint_collision_by_seed,
            ) -> tuple[_OpenBranchPlan, _LiftBranchPlan]:
                nonlocal planner, device_cfg, state, branch_attempt_count
                nonlocal open_branch_planning_s, attached_lift_planning_s
                branch_attempt_count += 1
                endpoint_collision = current_endpoint_collisions[branch.solver_seed_index]
                if endpoint_collision is not None:
                    raise _BranchRejected(
                        "pregrasp_endpoint_strict_self_collision",
                        endpoint_collision,
                    )
                if planner is None:
                    planner, device_cfg, state = create_open_planner()
                branch_start_state = _fresh_branch_start_state(
                    device_cfg,
                    reference_model,
                    arm=arm,
                )
                selected_local = branch.goalset_local_index
                candidate = current_candidates[selected_local]
                branch_started = time.monotonic()
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
                    )
                finally:
                    attached_lift_planning_s += time.monotonic() - lift_started
                return branch_open, branch_lift

            branch, complete, failures = _try_branch_pool(
                branches,
                candidate_ids=candidate_ids,
                attempt=attempt_branch,
                report=report,
            )
            branch_rejections.extend(failures)
            if branch is not None and complete is not None:
                selected = subset_candidates[branch.goalset_local_index]
                selected_branch = branch
                selected_round = search_round
                open_plan, lift_plan = complete
                break

            _cleanup(planner)
            planner = None
            for local_index in represented_local_indices:
                global_index = subset_indices[local_index]
                candidate_id = str(candidates[global_index]["candidate_id"])
                count = sum(branch.goalset_local_index == local_index for branch in branches)
                remaining_indices.remove(global_index)
                report(
                    f"exhausted all {count} returned IK branches for {candidate_id}; "
                    "only now removing that grasp candidate"
                )

        if selected is None or selected_branch is None or open_plan is None or lift_plan is None:
            raise RuntimeError(
                "all qualified cube grasp IK branches failed complete-path validation: "
                f"{branch_rejections}"
            )

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
            open_active_dex3_q_rad=tuple(open_q),
            close_target_active_dex3_q_rad=tuple(close_target_q),
            initial_active_dex3_q_rad=(
                request.observation.snapshot.left_dex3_q_rad
                if arm == "left"
                else request.observation.snapshot.right_dex3_q_rad
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
                    "goalset_ik": goalset_ik_s,
                    "strict_pregrasp_endpoint_check": endpoint_precheck_s,
                    "open_branch_planning_and_validation": open_branch_planning_s,
                    "attached_lift_planning_and_validation": attached_lift_planning_s,
                },
                "grasp_shortlist_sha256": request.grasp_shortlist_sha256,
                "candidate_count": len(candidates),
                "execution_maximum_arm_velocity_rad_s": (request.maximum_arm_velocity_rad_s),
                "selection_policy": (
                    "bounded_curobo_pregrasp_ik_branch_search_with_strict_endpoint_and_"
                    "complete_lifecycle_validation"
                ),
                "pregrasp_ik_search_rounds": selected_round,
                "pregrasp_ik_branches_tested": branch_attempt_count,
                "selected_pregrasp_solver_seed_index": (selected_branch.solver_seed_index),
                "selected_pregrasp_position_error_m": (selected_branch.position_error_m),
                "selected_pregrasp_rotation_error_rad": (selected_branch.rotation_error_rad),
                "rejected_grasp_branches": branch_rejections,
                "qualification": "GraspGenX + Isaac/PhysX retained shortlist",
                "finger_close_command_policy": (
                    "one descriptor-defined target for every grasp; physical contact limits "
                    "measured travel; candidate PhysX endpoints are qualification evidence only"
                ),
                "attachment_policy": (
                    "CuRobo AttachmentManager deterministic conservative 3x3x3 cuboid cover"
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
            },
        )
    finally:
        _cleanup(planner)
