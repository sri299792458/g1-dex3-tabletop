"""CuRobo-only planner for the right-Dex3 cube pick/lift/replace task."""

from __future__ import annotations

import gc
import hashlib
import json
import time
from collections.abc import Callable
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
CUBE_MESH = ROOT / "generated/tabletop_cube/mujoco/cube.obj"
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


def _table_from_resting_object(
    request: TabletopTaskRequest, base_T_torso: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Derive only the supporting plane from the resting cube.

    A single cube observation contains no evidence about the finite table
    footprint, its yaw, or its thickness.  The first return value is therefore
    a point on the top plane, not a fabricated table-frame origin.
    """

    base_T_camera = base_T_torso @ np.asarray(request.torso_T_camera)
    base_T_object = base_T_camera @ np.asarray(request.observation.camera_T_object)
    axes = base_T_object[:3, :3]
    candidates = [
        (float(direction * axes[2, index]), index, direction)
        for index in range(3)
        for direction in (-1.0, 1.0)
    ]
    alignment, axis_index, direction = min(candidates, key=lambda item: item[0])
    if alignment > -np.cos(np.deg2rad(20.0)):
        raise RuntimeError(
            "AprilCube does not have one face sufficiently aligned with the "
            "modeled gravity direction; cannot infer the tabletop safely"
        )
    down = direction * axes[:, axis_index]
    extent = request.object_dimensions_m[axis_index]
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
        status = "none" if result is None else str(result.status)
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


def _attach_cube(planner, state, grasp_T_object: np.ndarray) -> int:
    import torch
    import trimesh
    from curobo._src.geom.sphere_fit.types import SphereFitType
    from curobo.scene import Mesh

    if not CUBE_MESH.is_file():
        raise FileNotFoundError(
            "generated AprilCube mesh is missing; run ./tools/generate_tabletop_cube.sh"
        )
    manager = planner.trajopt_solver.core.attachment_manager
    source = manager.fit_spheres(
        [Mesh(name="cube_payload", file_path=str(CUBE_MESH), pose=[0, 0, 0, 1, 0, 0, 0])],
        num_spheres=16,
        surface_radius=0.003,
        sphere_fit_type=SphereFitType.MORPHIT,
    ).clone()
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
        open_robot, _ = build_tabletop_robot_config(
            snapshot=request.observation.snapshot,
            joint_position_offsets_rad=request.joint_position_offsets_rad,
            right_finger_q_rad=tuple(open_q),
        )
        planner, device_cfg = _planner(open_robot, {}, max_goalset=15, seed=request.random_seed)
        state = _joint_state(device_cfg, reference_model, arm_joint_names("right"))
        base_T_torso = _base_T_torso(planner, state)
        planner.destroy()
        planner = None
        scene = _base_scene(request, base_T_torso, include_cube=True)
        planner, device_cfg = _planner(open_robot, scene, max_goalset=15, seed=request.random_seed)
        state = _joint_state(device_cfg, reference_model, arm_joint_names("right"))
        base_T_camera = base_T_torso @ np.asarray(request.torso_T_camera)
        base_T_object = base_T_camera @ np.asarray(request.observation.camera_T_object)
        grasp_matrices = [base_T_object @ _candidate_transform(item) for item in candidates]
        goals = _goalset(grasp_matrices, device_cfg)
        report("submitting all 15 PhysX-qualified cube grasps as one CuRobo goal set")
        grasp_result = planner.plan_grasp(
            goals,
            state,
            grasp_approach_axis="z",
            grasp_approach_offset=-float(shortlist["execution_contract"]["approach_distance_m"]),
            grasp_approach_in_tool_frame=True,
            plan_grasp_to_lift=False,
            disable_collision_links=list(CONTACT_LINKS),
        )
        if grasp_result is None or not bool(grasp_result.success.any()):
            status = "none" if grasp_result is None else grasp_result.status
            raise RuntimeError(f"no qualified cube grasp had a complete CuRobo approach: {status}")
        selected_index = int(grasp_result.goalset_index.reshape(-1)[0].item())
        selected = candidates[selected_index]
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
        plane_point, _plane_object, down = _table_from_resting_object(request, base_T_torso)
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
            raise RuntimeError(
                "selected grasp route crosses the observed table plane: "
                f"clearance={open_clearance:.4f}m at {open_link} sample {open_sample}"
            )
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
        attached_scene = _base_scene(request, base_T_torso, include_cube=False)
        planner, device_cfg = _planner(
            closed_robot, attached_scene, max_goalset=1, seed=request.random_seed
        )
        contact_state = _joint_state(device_cfg, contact_model_q, arm_joint_names("right"))
        object_T_grasp = _candidate_transform(selected)
        grasp_T_object = invert_transform(object_T_grasp)
        sphere_count = _attach_cube(planner, contact_state, grasp_T_object)
        contact_pose = _base_T_grasp(planner, contact_state)
        plane_point, _base_T_object, down = _table_from_resting_object(request, base_T_torso)
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
            raise RuntimeError(
                "closed-hand lift crosses the observed table plane: "
                f"clearance={hand_clearance:.4f}m at {hand_link} sample {hand_sample}"
            )
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
            raise RuntimeError(
                "payload lift moves local geometry farther through the observed "
                f"table plane: minimum={payload_clearance:.4f}m at {payload_link} "
                f"sample {payload_sample}, start={payload_start_clearance:.4f}m"
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
                "selection_policy": "all_15_candidates_one_native_curobo_goalset",
                "qualification": "GraspGenX + Isaac/PhysX retained shortlist",
                "attachment_policy": "CuRobo AttachmentManager 16-sphere MORPHIT fit",
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
