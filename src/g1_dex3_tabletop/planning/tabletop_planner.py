"""CuRobo-only planner for the right-Dex3 cube pick/lift/replace task."""

from __future__ import annotations

import gc
import hashlib
import json
import time
from collections.abc import Callable
from itertools import permutations, product
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from scipy.spatial.transform import Rotation

from g1_aprilcube_calibration.joint_map import RIGHT_ARM_INDICES, arm_joint_names
from g1_aprilcube_calibration.transforms import invert_transform
from g1_aprilcube_calibration.transports.unitree_dex3 import (
    DEX3_RIGHT_MOTOR_JOINT_SUFFIXES,
)
from g1_dex3_tabletop.planning.contracts import PlannedTrajectory, RobotSnapshot
from g1_dex3_tabletop.planning.curobo_backend import (
    COLLISION_ACTIVATION_DISTANCE_M,
    EXECUTION_MAXIMUM_VELOCITY_RAD_S,
    IK_SEEDS,
    TRAJECTORY_INTERPOLATION_DT_S,
    _joint_state_dt,
    _reverse_trajectory,
)
from g1_dex3_tabletop.planning.g1_model import (
    CUROBO_COMMIT,
    RIGHT_ATTACHMENT_LINK,
    RIGHT_GRASP_FRAME,
    build_tabletop_robot_config,
    command_from_model_q,
    model_source_hashes,
)
from g1_dex3_tabletop.tabletop_contracts import (
    SupportedEscapePlan,
    TabletopTaskPlan,
    TabletopTaskRequest,
)

ROOT = Path(__file__).resolve().parents[3]
RIGHT_PROFILE = ROOT / "config/tabletop/dex3_rev1_right_profile.json"
CONTACT_LINKS = (
    "right_hand_thumb_2_link",
    "right_hand_middle_1_link",
    "right_hand_index_1_link",
)
LOCAL_TABLE_PLANE_LINKS = (
    "right_wrist_pitch_link",
    "right_wrist_yaw_link",
    "right_hand_palm_link",
    "right_hand_thumb_0_link",
    "right_hand_thumb_1_link",
    "right_hand_thumb_2_link",
    "right_hand_middle_0_link",
    "right_hand_middle_1_link",
    "right_hand_index_0_link",
    "right_hand_index_1_link",
)


def _pose_list(matrix: np.ndarray) -> list[float]:
    quaternion_xyzw = Rotation.from_matrix(matrix[:3, :3]).as_quat()
    return [
        *matrix[:3, 3].tolist(),
        float(quaternion_xyzw[3]),
        *quaternion_xyzw[:3].tolist(),
    ]


def _load_profile() -> tuple[np.ndarray, np.ndarray]:
    document = json.loads(RIGHT_PROFILE.read_text(encoding="utf-8"))
    names = [f"right_hand_{suffix}_joint" for suffix in DEX3_RIGHT_MOTOR_JOINT_SUFFIXES]
    return (
        np.asarray([document["open"][name] for name in names], dtype=np.float64),
        np.asarray([document["close"][name] for name in names], dtype=np.float64),
    )


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
    if document.get("hand_side") != "right" or document.get("object_id") != "cube_head":
        raise ValueError("tabletop task requires the qualified right-hand cube shortlist")
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
    if len(candidates) != 15:
        raise ValueError("qualified cube shortlist must contain exactly 15 candidates")
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
            "AprilCube is not resting on a face: no face normal points upward "
            "within 20 degrees"
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
    top_origin = base_T_object[:3, 3] + 0.5 * extent * down
    return top_origin, base_T_object, down


def _base_scene(
    request: TabletopTaskRequest, base_T_torso: np.ndarray, *, include_cube: bool
) -> dict[str, Any]:
    _plane_point, base_T_object, _down = _table_from_resting_object(request, base_T_torso)
    scene: dict[str, Any] = {"cuboid": {}}
    if include_cube:
        scene["cuboid"]["cube"] = {
            "dims": list(request.object_dimensions_m),
            "pose": _pose_list(base_T_object),
        }
    return scene


def _local_plane_clearance(
    planner,
    model_q: np.ndarray,
    *,
    plane_point: np.ndarray,
    down: np.ndarray,
    include_payload: bool,
) -> tuple[float, str, int]:
    """Return minimum signed clearance for local manipulation geometry.

    CuRobo still checks full-robot self-collision.  This independent guard is
    intentionally limited to the right wrist/hand and, after grasping, the
    attached payload.  Applying an unbounded plane to the fixed torso, legs,
    opposite arm, or elbow would falsely assert that geometry outside the
    unseen table footprint must also be above the tabletop.
    """

    import torch
    from curobo.types import JointState

    values = np.asarray(model_q, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 7:
        raise ValueError("plane validation requires an N x 7 trajectory")
    device_cfg = planner.device_cfg
    state = JointState.from_position(
        device_cfg.to_device(values), joint_names=list(arm_joint_names("right"))
    )
    spheres = planner.compute_kinematics(state).robot_spheres
    if spheres is None:
        raise RuntimeError("CuRobo kinematics did not return collision spheres")
    sphere_array = spheres.detach().cpu().numpy()
    sphere_array = sphere_array.reshape(values.shape[0], -1, 4)
    config = planner.kinematics.config.kinematics_config
    link_names = list(LOCAL_TABLE_PLANE_LINKS)
    if include_payload:
        link_names.append(RIGHT_ATTACHMENT_LINK)
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
        selected = sphere_array[:, indices_np, :]
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


def _planner(robot: dict[str, Any], scene: dict[str, Any], *, max_goalset: int, seed: int):
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


def _base_T_torso(planner, state) -> np.ndarray:
    pose = planner.compute_kinematics(state).tool_poses["torso_link"]
    return pose.get_matrix()[0].detach().cpu().numpy()


def _base_T_grasp(planner, state) -> np.ndarray:
    pose = planner.compute_kinematics(state).tool_poses[RIGHT_GRASP_FRAME]
    return pose.get_matrix()[0].detach().cpu().numpy()


def _use_moving_grasp_frame_only(robot: dict[str, Any]) -> None:
    """Remove the static torso query frame before pose/grasp optimization.

    The first kinematics-only planner exposes ``torso_link`` so the locked
    seated torso pose can be read.  The subsequent planners optimize only the
    right arm, so the torso cannot move and does not belong in their goal set.
    This is especially important for ``plan_grasp``, which applies approach
    offsets to every goal frame.
    """

    robot["kinematics"]["tool_frames"] = [RIGHT_GRASP_FRAME]


def _goalset(matrices: list[np.ndarray], device_cfg):
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
    return GoalToolPose([RIGHT_GRASP_FRAME], positions, quaternions)


def _plan_grasp_with_backtracking(
    *,
    planner,
    state,
    candidates: list[dict[str, Any]],
    grasp_matrices: list[np.ndarray],
    device_cfg,
    approach_distance_m: float,
    report: Callable[[str], None],
):
    """Use native CuRobo grasp planning while trying alternate goal-set choices.

    ``MotionPlanner.plan_grasp`` selects one reachable final grasp from its
    goal set before planning that grasp's pregrasp approach.  If the chosen
    approach fails, CuRobo returns immediately instead of selecting another
    member.  Remove only that chosen member and resubmit the remaining native
    goal set so every qualified candidate gets a fair complete-path test.
    """

    remaining = list(range(len(candidates)))
    rejected: list[dict[str, str]] = []
    while remaining:
        report(
            f"submitting {len(remaining)} remaining PhysX-qualified cube grasps "
            "as one CuRobo goal set"
        )
        goals = _goalset([grasp_matrices[index] for index in remaining], device_cfg)
        result = planner.plan_grasp(
            goals,
            state,
            grasp_approach_axis="z",
            grasp_approach_offset=-approach_distance_m,
            grasp_approach_in_tool_frame=True,
            plan_grasp_to_lift=False,
            disable_collision_links=list(CONTACT_LINKS),
        )
        if result is not None and bool(result.success.any()):
            selected_local = int(result.goalset_index.reshape(-1)[0].item())
            return result, remaining[selected_local], rejected
        status = "no planner result" if result is None else str(result.status)
        goalset_index = None if result is None else result.goalset_index
        if goalset_index is None:
            raise RuntimeError(
                "no remaining qualified cube grasp had a reachable CuRobo final "
                f"grasp; prior rejections={rejected}; status={status}"
            )
        selected_local = int(goalset_index.reshape(-1)[0].item())
        if not 0 <= selected_local < len(remaining):
            raise RuntimeError("CuRobo returned an invalid grasp goal-set index")
        selected_index = remaining.pop(selected_local)
        candidate_id = str(candidates[selected_index]["candidate_id"])
        rejected.append({"candidate_id": candidate_id, "reason": status})
        report(f"rejected {candidate_id}: {status}; trying the remaining goal set")
    raise RuntimeError(f"all qualified cube grasp approaches failed: {rejected}")


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


def _trajectory_array(planner, state, last_tstep=None) -> tuple[np.ndarray, float]:
    active = planner.kinematics.get_active_js(state).reorder(list(arm_joint_names("right")))
    values = np.asarray(active.position.detach().cpu().numpy(), dtype=np.float64)
    while values.ndim > 2:
        values = values[0]
    if last_tstep is not None:
        import torch

        last = int(torch.as_tensor(last_tstep).reshape(-1)[0].item())
        values = values[: last + 1]
    return values, _joint_state_dt(active)


def _planned_trajectory(
    *,
    from_id: str,
    to_id: str,
    model_q: np.ndarray,
    native_dt: float,
    offsets: dict[str, float],
    planning_time_s: float,
) -> PlannedTrajectory:
    values = np.asarray(model_q, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 7 or len(values) < 2:
        raise RuntimeError(f"CuRobo returned invalid {from_id}->{to_id} trajectory {values.shape}")
    peak = float(np.max(np.abs(np.diff(values, axis=0))) / native_dt)
    dt = native_dt * max(1.0, peak / EXECUTION_MAXIMUM_VELOCITY_RAD_S)
    command = np.stack(
        [
            command_from_model_q(
                q,
                arm="right",
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


def _result_trajectory(planner, result, *, from_id: str, to_id: str, offsets: dict[str, float]):
    if result is None or not bool(result.success.any()):
        status = "none" if result is None else str(getattr(result, "status", "unsuccessful"))
        raise RuntimeError(f"CuRobo failed {from_id}->{to_id}: {status}")
    plan = result.get_interpolated_plan().reorder(list(arm_joint_names("right")))
    values = np.asarray(plan.position.detach().cpu().numpy(), dtype=np.float64).squeeze()
    return _planned_trajectory(
        from_id=from_id,
        to_id=to_id,
        model_q=values,
        native_dt=_joint_state_dt(plan),
        offsets=offsets,
        planning_time_s=float(result.total_time),
    )


def _grasp_subtrajectory(
    planner,
    result,
    attribute: str,
    *,
    from_id: str,
    to_id: str,
    offsets: dict[str, float],
) -> PlannedTrajectory:
    state = getattr(result, f"{attribute}_interpolated_trajectory")
    last = getattr(result, f"{attribute}_interpolated_last_tstep")
    if state is None:
        state = getattr(result, f"{attribute}_trajectory")
        last = None
    if state is None:
        raise RuntimeError(f"CuRobo grasp result lacks {attribute} trajectory")
    values, dt = _trajectory_array(planner, state, last)
    return _planned_trajectory(
        from_id=from_id,
        to_id=to_id,
        model_q=values,
        native_dt=dt,
        offsets=offsets,
        planning_time_s=float(result.planning_time),
    )


def _concatenate(
    first: PlannedTrajectory, second: PlannedTrajectory, *, from_id: str, to_id: str
) -> PlannedTrajectory:
    if first.to_pose_id != second.from_pose_id:
        raise ValueError("cannot concatenate trajectories with different endpoints")
    offset = first.sample_time_s[-1]
    return PlannedTrajectory(
        from_pose_id=from_id,
        to_pose_id=to_id,
        sample_time_s=first.sample_time_s
        + tuple(offset + value for value in second.sample_time_s[1:]),
        command_q_rad=first.command_q_rad + second.command_q_rad[1:],
        model_q_rad=first.model_q_rad + second.model_q_rad[1:],
        planning_time_s=first.planning_time_s + second.planning_time_s,
    )


def _cleanup(planner) -> None:
    import torch

    if planner is not None:
        planner.destroy()
    gc.collect()
    torch.cuda.empty_cache()


def plan_supported_escape(
    request: TabletopTaskRequest,
    *,
    progress: Callable[[str], None] | None = None,
) -> SupportedEscapePlan:
    """Lift the supported right hand 100 mm along the observed table normal."""

    report = progress or (lambda _message: None)
    initial_fingers = request.observation.snapshot.right_dex3_q_rad
    robot, reference_tuple = build_tabletop_robot_config(
        snapshot=request.observation.snapshot,
        joint_position_offsets_rad=request.joint_position_offsets_rad,
        right_finger_q_rad=initial_fingers,
    )
    planner = None
    started = time.monotonic()
    try:
        # The torso pose is independent of the active right-arm coordinates.
        planner, device_cfg = _planner(robot, {}, max_goalset=1, seed=request.random_seed)
        names = arm_joint_names("right")
        reference = np.asarray(reference_tuple)
        state = _joint_state(device_cfg, reference, names)
        base_T_torso = _base_T_torso(planner, state)
        base_T_grasp = _base_T_grasp(planner, state)
        planner.destroy()
        planner = None
        _use_moving_grasp_frame_only(robot)
        scene = _base_scene(request, base_T_torso, include_cube=True)
        planner, device_cfg = _planner(robot, scene, max_goalset=1, seed=request.random_seed)
        state = _joint_state(device_cfg, reference, names)
        plane_point, _base_T_object, down = _table_from_resting_object(request, base_T_torso)
        target = base_T_grasp.copy()
        target[:3, 3] -= request.supported_escape_m * down
        from curobo.types import GoalToolPose, Pose, ToolPoseCriteria

        goal = GoalToolPose.from_poses(
            {RIGHT_GRASP_FRAME: Pose.from_matrix(device_cfg.to_device(target[None]))},
            ordered_tool_frames=[RIGHT_GRASP_FRAME],
        )
        criterion = ToolPoseCriteria.linear_motion(
            axis="z", non_terminal_scale=1.0, project_distance_to_goal=False
        )
        planner.update_tool_pose_criteria({RIGHT_GRASP_FRAME: criterion})
        try:
            result = planner.plan_pose(goal, state, max_attempts=8)
        finally:
            planner.update_tool_pose_criteria({RIGHT_GRASP_FRAME: ToolPoseCriteria()})
        if result is None or not bool(result.success.any()):
            reason = _pose_failure_reason(planner, goal, state, result)
            raise RuntimeError(f"CuRobo failed __handoff__->clearance: {reason}")
        outbound = _result_trajectory(
            planner,
            result,
            from_id="__handoff__",
            to_id="clearance",
            offsets=request.joint_position_offsets_rad,
        )
        endpoint_q = np.asarray(outbound.model_q_rad[-1])
        endpoint = _joint_state(device_cfg, endpoint_q, names)
        actual_target = _base_T_grasp(planner, endpoint)
        terminal_clearance = -float(np.dot(down, actual_target[:3, 3] - plane_point))
        if terminal_clearance < 0.05:
            raise RuntimeError(
                f"supported escape finishes only {terminal_clearance:.4f}m above the table plane"
            )
        route_q = np.asarray(outbound.model_q_rad)
        route_clearance, route_link, route_sample = _local_plane_clearance(
            planner,
            route_q,
            plane_point=plane_point,
            down=down,
            include_payload=False,
        )
        start_clearance, _start_link, _start_sample = _local_plane_clearance(
            planner,
            route_q[:1],
            plane_point=plane_point,
            down=down,
            include_payload=False,
        )
        if route_clearance < start_clearance - COLLISION_ACTIVATION_DISTANCE_M:
            raise RuntimeError(
                "supported escape drives local right-hand geometry farther through "
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
                "policy": (
                    "straight 100mm table-normal escape; no fabricated table box; "
                    "local right-wrist/hand plane guard; exact reverse return"
                ),
                "local_plane_minimum_clearance_m": route_clearance,
                "local_plane_start_clearance_m": start_clearance,
                "local_plane_minimum_link": route_link,
                "local_plane_minimum_sample": route_sample,
            },
        )
    finally:
        _cleanup(planner)


def _snapshot_at_right_q(
    request: TabletopTaskRequest, command_q: tuple[float, ...]
) -> RobotSnapshot:
    q29 = np.asarray(request.observation.snapshot.measured_q29_rad).copy()
    q29[np.asarray(RIGHT_ARM_INDICES)] = np.asarray(command_q)
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
            link_name=RIGHT_ATTACHMENT_LINK,
            world_objects_pose_offset=None,
        )
    return len(source)


def plan_tabletop_task(
    request: TabletopTaskRequest,
    *,
    progress: Callable[[str], None] | None = None,
) -> TabletopTaskPlan:
    """Plan from the freshly observed elevated clearance pose and back."""

    report = progress or (lambda _message: None)
    shortlist, candidates = _load_shortlist(request)
    open_q, _profile_close_q = _load_profile()
    reference = np.asarray(request.observation.snapshot.measured_q29_rad)[
        np.asarray(RIGHT_ARM_INDICES)
    ]
    # Model-space reference includes removable calibration joint offsets.
    reference_model = np.asarray(
        [
            value + request.joint_position_offsets_rad.get(name, 0.0)
            for name, value in zip(arm_joint_names("right"), reference, strict=True)
        ]
    )
    started = time.monotonic()
    planner = None
    try:
        query_robot, _ = build_tabletop_robot_config(
            snapshot=request.observation.snapshot,
            joint_position_offsets_rad=request.joint_position_offsets_rad,
            right_finger_q_rad=tuple(open_q),
        )
        planner, device_cfg = _planner(query_robot, {}, max_goalset=1, seed=request.random_seed)
        state = _joint_state(device_cfg, reference_model, arm_joint_names("right"))
        base_T_torso = _base_T_torso(planner, state)
        planner.destroy()
        planner = None

        plane_point, base_T_object, down = _table_from_resting_object(request, base_T_torso)
        grasp_matrices = [base_T_object @ _candidate_transform(item) for item in candidates]
        approach_distance_m = float(shortlist["execution_contract"]["approach_distance_m"])
        remaining_indices = list(range(len(candidates)))
        grasp_rejections: list[dict[str, str]] = []

        while remaining_indices:
            subset_indices = remaining_indices.copy()
            subset_candidates = [candidates[index] for index in subset_indices]
            subset_matrices = [grasp_matrices[index] for index in subset_indices]
            open_robot, _ = build_tabletop_robot_config(
                snapshot=request.observation.snapshot,
                joint_position_offsets_rad=request.joint_position_offsets_rad,
                right_finger_q_rad=tuple(open_q),
            )
            _use_moving_grasp_frame_only(open_robot)
            scene = _base_scene(request, base_T_torso, include_cube=True)
            planner, device_cfg = _planner(
                open_robot,
                scene,
                max_goalset=len(subset_candidates),
                seed=request.random_seed,
            )
            state = _joint_state(device_cfg, reference_model, arm_joint_names("right"))
            grasp_result, selected_subset_index, approach_rejections = (
                _plan_grasp_with_backtracking(
                    planner=planner,
                    state=state,
                    candidates=subset_candidates,
                    grasp_matrices=subset_matrices,
                    device_cfg=device_cfg,
                    approach_distance_m=approach_distance_m,
                    report=report,
                )
            )
            for rejection in approach_rejections:
                candidate_id = rejection["candidate_id"]
                rejected_index = next(
                    index
                    for index in remaining_indices
                    if str(candidates[index]["candidate_id"]) == candidate_id
                )
                remaining_indices.remove(rejected_index)
                grasp_rejections.append(
                    {
                        "candidate_id": candidate_id,
                        "stage": "approach",
                        "reason": rejection["reason"],
                    }
                )
            selected_index = subset_indices[selected_subset_index]
            selected = candidates[selected_index]
            candidate_id = str(selected["candidate_id"])
            approach = _grasp_subtrajectory(
                planner,
                grasp_result,
                "approach",
                from_id="clearance",
                to_id="move_to_pregrasp",
                offsets=request.joint_position_offsets_rad,
            )
            grasp = _grasp_subtrajectory(
                planner,
                grasp_result,
                "grasp",
                from_id="move_to_pregrasp",
                to_id="grasp_approach",
                offsets=request.joint_position_offsets_rad,
            )
            open_route_q = np.concatenate(
                (np.asarray(approach.model_q_rad), np.asarray(grasp.model_q_rad)[1:]),
                axis=0,
            )
            open_clearance, open_link, open_sample = _local_plane_clearance(
                planner,
                open_route_q,
                plane_point=plane_point,
                down=down,
                include_payload=False,
            )
            if open_clearance < 0.0:
                reason = f"clearance={open_clearance:.4f}m at {open_link} sample {open_sample}"
                grasp_rejections.append(
                    {
                        "candidate_id": candidate_id,
                        "stage": "open_route_table_plane",
                        "reason": reason,
                    }
                )
                remaining_indices.remove(selected_index)
                report(
                    f"rejected {candidate_id} at open-route table-plane check: "
                    f"{reason}; trying the remaining goal set"
                )
                planner.destroy()
                planner = None
                continue
            planner.destroy()
            planner = None

            closed_mapping = selected["execution_evidence"]["isaac_closed_q"]
            closed_q = np.asarray(
                [
                    closed_mapping[f"right_hand_{suffix}_joint"]
                    for suffix in DEX3_RIGHT_MOTOR_JOINT_SUFFIXES
                ],
                dtype=np.float64,
            )
            contact_command_q = grasp.command_q_rad[-1]
            contact_model_q = np.asarray(grasp.model_q_rad[-1])
            contact_snapshot = _snapshot_at_right_q(request, contact_command_q)
            closed_robot, _ = build_tabletop_robot_config(
                snapshot=contact_snapshot,
                joint_position_offsets_rad=request.joint_position_offsets_rad,
                right_finger_q_rad=tuple(closed_q),
            )
            _use_moving_grasp_frame_only(closed_robot)
            attached_scene = _base_scene(request, base_T_torso, include_cube=False)
            planner, device_cfg = _planner(
                closed_robot, attached_scene, max_goalset=1, seed=request.random_seed
            )
            contact_state = _joint_state(device_cfg, contact_model_q, arm_joint_names("right"))
            object_T_grasp = _candidate_transform(selected)
            grasp_T_object = invert_transform(object_T_grasp)
            sphere_count = _attach_cube(
                planner,
                contact_state,
                grasp_T_object,
                request.object_dimensions_m,
            )
            contact_pose = _base_T_grasp(planner, contact_state)
            lift_pose = contact_pose.copy()
            lift_pose[:3, 3] -= request.lift_m * down
            from curobo.types import GoalToolPose, Pose, ToolPoseCriteria

            lift_goal = GoalToolPose.from_poses(
                {RIGHT_GRASP_FRAME: Pose.from_matrix(device_cfg.to_device(lift_pose[None]))},
                ordered_tool_frames=[RIGHT_GRASP_FRAME],
            )
            criterion = ToolPoseCriteria.linear_motion(
                axis="z", non_terminal_scale=1.0, project_distance_to_goal=False
            )
            planner.update_tool_pose_criteria({RIGHT_GRASP_FRAME: criterion})
            try:
                lift_result = planner.plan_pose(lift_goal, contact_state, max_attempts=8)
            finally:
                planner.update_tool_pose_criteria({RIGHT_GRASP_FRAME: ToolPoseCriteria()})
            if lift_result is None or not bool(lift_result.success.any()):
                reason = _pose_failure_reason(planner, lift_goal, contact_state, lift_result)
                grasp_rejections.append(
                    {
                        "candidate_id": candidate_id,
                        "stage": "attached_payload_lift",
                        "reason": reason,
                    }
                )
                remaining_indices.remove(selected_index)
                report(
                    f"rejected {candidate_id} at attached-payload lift: {reason}; "
                    "trying the remaining goal set"
                )
                planner.destroy()
                planner = None
                continue
            lift = _result_trajectory(
                planner,
                lift_result,
                from_id="grasp_approach",
                to_id="payload_lift",
                offsets=request.joint_position_offsets_rad,
            )
            lift_q = np.asarray(lift.model_q_rad)
            hand_clearance, hand_link, hand_sample = _local_plane_clearance(
                planner,
                lift_q,
                plane_point=plane_point,
                down=down,
                include_payload=False,
            )
            if hand_clearance < 0.0:
                reason = f"clearance={hand_clearance:.4f}m at {hand_link} sample {hand_sample}"
                grasp_rejections.append(
                    {
                        "candidate_id": candidate_id,
                        "stage": "closed_lift_table_plane",
                        "reason": reason,
                    }
                )
                remaining_indices.remove(selected_index)
                report(
                    f"rejected {candidate_id} at closed-lift table-plane check: "
                    f"{reason}; trying the remaining goal set"
                )
                planner.destroy()
                planner = None
                continue
            payload_clearance, payload_link, payload_sample = _local_plane_clearance(
                planner,
                lift_q,
                plane_point=plane_point,
                down=down,
                include_payload=True,
            )
            payload_start_clearance, _payload_start_link, _payload_start_sample = (
                _local_plane_clearance(
                    planner,
                    lift_q[:1],
                    plane_point=plane_point,
                    down=down,
                    include_payload=True,
                )
            )
            if payload_clearance < (payload_start_clearance - COLLISION_ACTIVATION_DISTANCE_M):
                reason = (
                    f"minimum={payload_clearance:.4f}m at {payload_link} sample "
                    f"{payload_sample}, start={payload_start_clearance:.4f}m"
                )
                grasp_rejections.append(
                    {
                        "candidate_id": candidate_id,
                        "stage": "payload_table_plane",
                        "reason": reason,
                    }
                )
                remaining_indices.remove(selected_index)
                report(
                    f"rejected {candidate_id} at payload table-plane check: {reason}; "
                    "trying the remaining goal set"
                )
                planner.destroy()
                planner = None
                continue

            # Every remaining phase is an exact reverse of a validated outbound
            # trajectory, so selecting the candidate here validates its complete
            # pick/lift/replace/retreat/return lifecycle before any motion begins.
            break
        else:
            raise RuntimeError(
                f"all qualified cube grasps failed complete-path validation: {grasp_rejections}"
            )

        replace = _reverse_trajectory(lift)
        replace = PlannedTrajectory(
            from_pose_id="payload_lift",
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
            selected_candidate_id=str(selected["candidate_id"]),
            object_T_grasp=tuple(tuple(float(v) for v in row) for row in object_T_grasp),
            open_right_dex3_q_rad=tuple(open_q),
            closed_right_dex3_q_rad=tuple(closed_q),
            initial_right_dex3_q_rad=request.observation.snapshot.right_dex3_q_rad,
            trajectories=(approach, grasp, lift, replace, retreat, return_clearance),
            phase_order=(
                "move_to_pregrasp",
                "grasp_approach",
                "payload_lift",
                "payload_replace",
                "grasp_retreat",
                "return_to_clearance",
            ),
            planner_provenance={
                **model_source_hashes(),
                "curobo_commit": CUROBO_COMMIT,
                "elapsed_s": time.monotonic() - started,
                "grasp_shortlist_sha256": request.grasp_shortlist_sha256,
                "candidate_count": len(candidates),
                "selection_policy": ("native_curobo_goalset_with_complete_lifecycle_backtracking"),
                "rejected_grasp_candidates": grasp_rejections,
                "qualification": "GraspGenX + Isaac/PhysX retained shortlist",
                "attachment_policy": (
                    "CuRobo AttachmentManager deterministic conservative 3x3x3 cuboid cover"
                ),
                "attachment_sphere_count": sphere_count,
                "visual_policy": (
                    "fresh stationary AprilCube observation after supported escape; "
                    "support plane inferred from its gravity-aligned bottom face; "
                    "no unobserved finite table footprint is invented"
                ),
                "table_plane_policy": "local_right_wrist_hand_payload_only",
                "open_route_minimum_plane_clearance_m": open_clearance,
                "open_route_minimum_plane_link": open_link,
                "closed_lift_hand_minimum_plane_clearance_m": hand_clearance,
                "payload_lift_minimum_plane_clearance_m": payload_clearance,
                "payload_start_plane_clearance_m": payload_start_clearance,
                "return_policy": "exact reverse lift, grasp, and approach trajectories",
            },
        )
    finally:
        _cleanup(planner)
