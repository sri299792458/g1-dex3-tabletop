"""CuRobo MPC for the local, visually updated grasp approach.

MotionGen owns global motion to the pregrasp and the discrete task state machine
owns finger contact, payload motion, release, and return.  MPC is deliberately
limited to the one segment for which a changing visual goal is useful:
pregrasp to grasp.  Every MPC window is independently checked with the same
full-robot, cube, fixture, and table-plane policies used by the frozen planner
before it may cross into the 250 Hz command process.
"""

from __future__ import annotations

import copy
import gc
import hashlib
import json
import time
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation

from g1_aprilcube_calibration.joint_map import arm_indices, arm_joint_names
from g1_aprilcube_calibration.transforms import invert_transform, validate_transform
from g1_aprilcube_calibration.transports.unitree_dex3 import DEX3_MOTOR_JOINT_SUFFIXES
from g1_dex3_tabletop.mpc_command_buffer import (
    MPCCommandWindow,
    command_sequence_from_measured_plan,
)
from g1_dex3_tabletop.planning.contracts import PlannedTrajectory
from g1_dex3_tabletop.planning.curobo_backend import (
    COLLISION_ACTIVATION_DISTANCE_M,
    OPEN_TRANSIT_OBJECT_CLEARANCE_M,
)
from g1_dex3_tabletop.planning.dex3_handedness import dex3_execution_profile
from g1_dex3_tabletop.planning.g1_model import (
    attachment_link,
    build_tabletop_robot_config,
    command_from_model_q,
    grasp_frame,
)
from g1_dex3_tabletop.planning.tabletop_planner import (
    WORLD_COLLISION_DISABLE_RADIUS_EPSILON_M,
    _base_scene,
    _base_T_detected_object,
    _contact_links,
    _cuboid_cover_spheres,
    _fixture_collision_checker,
    _local_plane_clearance_from_spheres,
    _object_contact_links,
    _selected_open_transit_world_robot,
    _table_from_resting_object,
    _use_moving_grasp_frame_only,
    _world_cuboid_clearances,
)
from g1_dex3_tabletop.tabletop_contracts import TabletopExecutionPlan, TabletopTaskRequest
from g1_dex3_tabletop.tabletop_geometry import canonical_resting_cube_pose

MPC_COMMAND_DT_S = 0.01
MPC_KNOT_DT_S = 0.04
MPC_EXECUTOR_DT_S = 0.004
MPC_INTERPOLATION_STEPS = 4
# The retained RTX 5090 replay contained one 180 ms solve. Six knots provide
# 240 ms to the immutable splice (at least 232 ms after the executor's two-tick
# installation guard) while leaving more than half of CuRobo's 0.8 s complete
# state rollout available as the certified fallback tail.
MPC_HANDOFF_INTERVAL_S = 6 * MPC_KNOT_DT_S
MPC_OPTIMIZER_VELOCITY_SCALE = 0.95
MPC_FLOAT32_CONSTRAINT_EPSILON = 10.0 * np.finfo(np.float32).eps
# CuRobo's state horizon is constructed to end at a zero-velocity goal.  Keep
# only a small numerical tolerance so every accepted full horizon is also a
# bounded continuation to rest if the next solve cannot be installed.
MPC_STOP_VELOCITY_TOLERANCE_RAD_S = 1.0e-4
if MPC_KNOT_DT_S != MPC_COMMAND_DT_S * MPC_INTERPOLATION_STEPS:
    raise RuntimeError("MPC command, knot, and interpolation timing disagree")
MPC_COLD_START_ITERATIONS = 200
# The pinned CuRobo optimizer advances in 25-iteration inner blocks. Retained
# replay keeps the 100-iteration warm solve within the 100 ms worker allowance.
MPC_WARM_START_ITERATIONS = 100

@dataclass(frozen=True, slots=True)
class MPCPhaseSpec:
    """The physical collision state for one normal tabletop motion."""

    phase: str
    mode: str
    request_state: str
    finger_state: str
    include_cube_in_optimizer: bool
    include_table_patch: bool
    include_fixture_in_optimizer: bool
    allow_fingertip_cube_contact: bool
    attached_payload: bool
    reference_fixed_goal: bool


def mpc_phase_spec(phase: str) -> MPCPhaseSpec:
    """Return the sole physical state owned by production MPC."""

    if phase != "grasp_approach":
        raise ValueError(f"unsupported moving-grasp MPC phase: {phase}")
    return MPCPhaseSpec(
        phase=phase,
        mode="open_contact",
        request_state="clearance",
        finger_state="open",
        # Contact links are disabled only against world geometry. The cube
        # remains active for every non-contact link, and the independent
        # strict check enforces the same policy on every returned window.
        include_cube_in_optimizer=True,
        include_table_patch=True,
        include_fixture_in_optimizer=False,
        allow_fingertip_cube_contact=True,
        attached_payload=False,
        reference_fixed_goal=True,
    )


def _fixture_excluded_links(spec: MPCPhaseSpec, *, arm: str) -> tuple[str, ...]:
    """Return only the links intentionally excluded from fixture checking."""

    links: list[str] = []
    if spec.allow_fingertip_cube_contact:
        links.extend(_contact_links(arm))
    if spec.attached_payload:
        links.append(attachment_link(arm))
    return tuple(links)


def _numpy(value) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=np.float64)


def _mpc_constraint_summary(metrics: Any) -> list[dict[str, Any]]:
    """Return compact named constraint evidence from one CuRobo rollout."""

    result: list[dict[str, Any]] = []
    collections = metrics.costs_and_constraints
    for kind, collection in (
        ("constraint", collections.constraints),
        ("hybrid", collections.hybrid_costs_constraints),
    ):
        for name, tensor in zip(collection.names, collection.values, strict=True):
            values = _numpy(tensor)
            result.append(
                {
                    "kind": kind,
                    "name": name,
                    "maximum": float(np.max(values)),
                    "positive_sample_count": int(np.count_nonzero(values > 0.0)),
                }
            )
    return result


def _mpc_constraints_numerically_feasible(summary: list[dict[str, Any]]) -> bool:
    """Ignore only float32-scale cspace residue; never soften collision gates."""

    return all(
        item["maximum"] <= (MPC_FLOAT32_CONSTRAINT_EPSILON if item["name"] == "cspace" else 0.0)
        for item in summary
    )


def _cspace_bound_diagnostics(
    cspace_cost,
    *,
    names: tuple[str, ...],
    position: np.ndarray,
    velocity: np.ndarray,
    acceleration: np.ndarray,
    jerk: np.ndarray,
) -> dict[str, dict[str, Any]]:
    """Name the exact joint-state derivative behind a CuRobo cspace failure."""

    cfg = cspace_cost.config
    activation = _numpy(cfg.activation_distance).reshape(-1)
    paths = {
        "position": position,
        "velocity": velocity,
        "acceleration": acceleration,
        "jerk": jerk,
    }
    result: dict[str, dict[str, Any]] = {}
    for component_index, (component, values) in enumerate(paths.items()):
        path = np.asarray(values, dtype=np.float64)
        if path.ndim != 2 or path.shape[1] != len(names):
            raise ValueError(f"CuRobo {component} path has an unexpected shape")
        limits = _numpy(getattr(cfg.joint_limits, component)).reshape(2, len(names))
        span = limits[1] - limits[0]
        lower = limits[0] + activation[component_index] * span
        upper = limits[1] - activation[component_index] * span
        violation = np.maximum(np.maximum(lower[None, :] - path, path - upper[None, :]), 0.0)
        flat_index = int(np.argmax(violation))
        sample_index, joint_index = np.unravel_index(flat_index, violation.shape)
        result[component] = {
            "maximum_violation": float(violation[sample_index, joint_index]),
            "sample": int(sample_index),
            "joint": names[joint_index],
            "value": float(path[sample_index, joint_index]),
            "active_lower_bound": float(lower[joint_index]),
            "active_upper_bound": float(upper[joint_index]),
        }
    return result


def _reserved_velocity_constraint_is_safe(
    summary: list[dict[str, Any]],
    cspace: dict[str, dict[str, Any]],
    *,
    full_state_peak_velocity_rad_s: float,
    physical_velocity_limit_rad_s: float,
) -> bool:
    """Accept only a tightened velocity-bound residual below the real limit."""

    if any(item["maximum"] > 0.0 for item in summary if item["name"] != "cspace"):
        return False
    if cspace["velocity"]["maximum_violation"] <= 0.0:
        return False
    if any(
        cspace[component]["maximum_violation"] > 0.0
        for component in ("position", "acceleration", "jerk")
    ):
        return False
    return full_state_peak_velocity_rad_s <= physical_velocity_limit_rad_s + 1.0e-6


def _bounded_route_goal(
    route_q: np.ndarray,
    *,
    current_q: np.ndarray,
    route_progress_index: int,
    maximum_distance_rad: float,
) -> tuple[np.ndarray, int, bool]:
    """Interpolate a bounded goal along the existing frozen joint path."""

    route = np.asarray(route_q, dtype=np.float64)
    current = np.asarray(current_q, dtype=np.float64).reshape(-1)
    if route.ndim != 2 or route.shape[1:] != current.shape:
        raise ValueError("MPC route and current state dimensions differ")
    if route_progress_index < 0 or route_progress_index >= len(route):
        raise ValueError("MPC route progress is outside the route")
    if not np.isfinite(maximum_distance_rad) or maximum_distance_rad <= 0.0:
        raise ValueError("MPC route lookahead must be positive and finite")
    goal = current.copy()
    remaining = float(maximum_distance_rad)
    segment_end_index = route_progress_index
    for index in range(route_progress_index + 1, len(route)):
        segment_end = route[index]
        segment_distance = float(np.max(np.abs(segment_end - goal)))
        segment_end_index = index
        if segment_distance > remaining:
            goal = goal + (remaining / segment_distance) * (segment_end - goal)
            return goal, segment_end_index, False
        goal = segment_end.copy()
        remaining -= segment_distance
    return goal, len(route) - 1, True


def _resample_to_executor_grid(
    sample_time_s: np.ndarray,
    *paths: np.ndarray,
) -> tuple[np.ndarray, ...]:
    """Resample every path onto the maximum 250 Hz validation spacing."""

    times = np.asarray(sample_time_s, dtype=np.float64).reshape(-1)
    if (
        len(times) < 2
        or times[0] != 0.0
        or not np.all(np.isfinite(times))
        or not np.all(np.diff(times) > 0.0)
    ):
        raise ValueError("coarse MPC times must start at zero and increase")
    duration = float(times[-1])
    dense_times = np.arange(0.0, duration + 0.5 * MPC_EXECUTOR_DT_S, MPC_EXECUTOR_DT_S)
    if dense_times[-1] < duration - 1.0e-12:
        dense_times = np.append(dense_times, duration)
    else:
        dense_times[-1] = duration
    result: list[np.ndarray] = [dense_times]
    for path in paths:
        values = np.asarray(path, dtype=np.float64)
        if values.shape != (len(times), 7) or not np.all(np.isfinite(values)):
            raise ValueError("MPC resampling requires finite N x 7 paths")
        result.append(
            np.stack(
                [np.interp(dense_times, times, values[:, index]) for index in range(7)],
                axis=1,
            )
        )
    return tuple(result)


def _joint_state(
    device_cfg,
    q: np.ndarray,
    dq: np.ndarray,
    ddq: np.ndarray,
    names: tuple[str, ...],
):
    from curobo.types import JointState

    return JointState.from_numpy(
        joint_names=list(names),
        position=np.asarray(q, dtype=np.float64)[None, :],
        velocity=np.asarray(dq, dtype=np.float64)[None, :],
        acceleration=np.asarray(ddq, dtype=np.float64)[None, :],
        # The configured cubic B-spline cannot independently constrain jerk.
        jerk=np.zeros_like(np.asarray(q, dtype=np.float64))[None, :],
        device_cfg=device_cfg,
    )


def _matrix_from_pose_list(value: Any) -> np.ndarray:
    pose = np.asarray(value, dtype=np.float64).reshape(-1)
    if pose.shape != (7,) or not np.all(np.isfinite(pose)):
        raise ValueError("CuRobo scene pose must contain seven finite values")
    result = np.eye(4, dtype=np.float64)
    result[:3, 3] = pose[:3]
    result[:3, :3] = Rotation.from_quat(pose[[4, 5, 6, 3]]).as_matrix()
    return validate_transform(result)


def _pose_list_from_matrix(value: Any) -> list[float]:
    """Return CuRobo's xyz+wxyz pose list for one rigid transform."""

    matrix = _rigid_transform(value)
    quaternion_xyzw = Rotation.from_matrix(matrix[:3, :3].copy()).as_quat()
    return [
        *[float(item) for item in matrix[:3, 3]],
        float(quaternion_xyzw[3]),
        *[float(item) for item in quaternion_xyzw[:3]],
    ]


def _scene_topology_key(scene: dict[str, Any]) -> str:
    """Hash obstacle identity and shape while deliberately ignoring live poses."""

    topology: dict[str, dict[str, Any]] = {}
    for group in ("cuboid", "mesh"):
        topology[group] = {}
        for name, obstacle in scene.get(group, {}).items():
            topology[group][name] = {
                key: value for key, value in obstacle.items() if key != "pose"
            }
    return hashlib.sha256(
        json.dumps(
            topology,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode()
    ).hexdigest()


def _rigid_transform(value: Any) -> np.ndarray:
    """Project accumulated float32 transform arithmetic back onto SE(3)."""

    result = np.asarray(value, dtype=np.float64).copy()
    if result.shape != (4, 4) or not np.all(np.isfinite(result)):
        raise ValueError("rigid transform must be a finite 4x4 matrix")
    result[:3, :3] = Rotation.from_matrix(result[:3, :3]).as_matrix()
    result[3] = (0.0, 0.0, 0.0, 1.0)
    return validate_transform(result)


def _nominal_base_T_live_base(
    *,
    base_T_torso0: np.ndarray,
    reference_T_torso0: np.ndarray,
    reference_T_torso: np.ndarray,
) -> np.ndarray:
    """Map live body-fixed model coordinates into the nominal scene frame.

    CuRobo keeps the pelvis/base and locked torso transform numerically frozen.
    The estimator instead tells us how that physical torso moved in the fixed
    object/table reference frame.  This rigid map lets the independent strict
    checker retain its one nominal cube/table/fixture scene while evaluating
    robot spheres at the live body pose.
    """

    base_T_torso0 = _rigid_transform(base_T_torso0)
    reference_T_torso0 = _rigid_transform(reference_T_torso0)
    reference_T_torso = _rigid_transform(reference_T_torso)
    return _rigid_transform(
        base_T_torso0
        @ invert_transform(reference_T_torso0)
        @ reference_T_torso
        @ invert_transform(base_T_torso0)
    )


def _resolved_robot_with_velocity_limit(
    robot: dict[str, Any],
    *,
    device_cfg,
    maximum_velocity_rad_s: float,
):
    """Resolve the CuRobo model once, then set the actual active-joint bound.

    CuRobo's dictionary ``velocity_scale`` is applied while the URDF is loaded
    and again while its reduced kinematic parameters are constructed in the
    pinned revision.  Mutating the resolved ``RobotCfg`` avoids relying on a
    square-root scale workaround and makes the effective limit directly
    inspectable as exactly the task's commissioned velocity ceiling.
    """

    from curobo._src.types.robot import RobotCfg

    if not np.isfinite(maximum_velocity_rad_s) or maximum_velocity_rad_s <= 0.0:
        raise ValueError("MPC maximum velocity must be positive and finite")
    resolved = RobotCfg.create(robot, device_cfg)
    limits = resolved.kinematics.kinematics_config.joint_limits
    limits.velocity[0, :].fill_(-float(maximum_velocity_rad_s))
    limits.velocity[1, :].fill_(float(maximum_velocity_rad_s))
    return resolved


def _disable_world_collision_links(
    robot: dict[str, Any],
    *,
    link_names: tuple[str, ...],
) -> dict[str, Any]:
    """Remove selected contact-link spheres from the permissive optimizer.

    CuRobo has one sphere set for world and self collision, so this cannot be a
    world-only exception inside the optimizer.  The strict full-robot checker
    independently restores and validates complete self geometry before any
    returned command window can be used.
    """

    result = copy.deepcopy(robot)
    kinematics = result["kinematics"]
    source_buffer = kinematics.get("collision_sphere_buffer", 0.0)
    if isinstance(source_buffer, dict):
        buffers = {str(name): float(value) for name, value in source_buffer.items()}
    else:
        buffers = {str(name): float(source_buffer) for name in kinematics["collision_spheres"]}
    for link_name in link_names:
        spheres = kinematics["collision_spheres"].get(link_name)
        if spheres is None:
            raise ValueError(f"CuRobo model lacks contact link {link_name}")
        maximum_radius = max((float(item["radius"]) for item in spheres), default=0.0)
        buffers[link_name] = -maximum_radius - WORLD_COLLISION_DISABLE_RADIUS_EPSILON_M
    kinematics["collision_sphere_buffer"] = buffers
    return result


def _payload_link_spheres(
    dimensions_m: tuple[float, ...],
    *,
    object_T_grasp: tuple[tuple[float, ...], ...],
) -> np.ndarray:
    """Return the frozen planner's 27 cube spheres in the grasp-link frame."""

    source = _cuboid_cover_spheres(tuple(float(value) for value in dimensions_m))
    grasp_T_object = invert_transform(np.asarray(object_T_grasp, dtype=np.float64))
    centers = source[:, :3] @ grasp_T_object[:3, :3].T + grasp_T_object[:3, 3][None, :]
    return np.column_stack((centers, source[:, 3]))


def _install_payload_spheres_on_params(config, *, arm: str, spheres: np.ndarray) -> None:
    """Install deterministic link-local payload spheres without reallocating buffers."""

    import torch

    existing = config.get_link_spheres(attachment_link(arm)).clone()
    if len(spheres) > len(existing):
        raise RuntimeError(
            f"payload needs {len(spheres)} spheres but {attachment_link(arm)} has "
            f"only {len(existing)} slots"
        )
    existing[:, :] = 0.0
    existing[:, 3] = -100.0
    existing[: len(spheres)] = torch.as_tensor(
        spheres,
        device=existing.device,
        dtype=existing.dtype,
    )
    config.update_link_spheres(attachment_link(arm), existing)


def _install_payload_spheres_on_checker(checker, *, arm: str, spheres: np.ndarray) -> None:
    """Install deterministic link-local payload spheres in a strict FK checker."""

    _install_payload_spheres_on_params(
        checker.config.kinematics_config,
        arm=arm,
        spheres=spheres,
    )


@dataclass(frozen=True, slots=True)
class _ResolvedPhaseKinematics:
    """Immutable GPU tensor values for one locked-finger posture.

    CuRobo folds locked joints into the fixed-transform tensors while loading
    the reduced seven-joint model.  Changing only ``lock_jointstate`` would
    therefore be wrong; every physical finger posture needs a resolved
    kinematic value set.  These clones are copied into the already-warmed MPC
    buffers without changing tensor shapes or CUDA graph addresses.
    """

    params: Any
    self_collision_padding: Any
    self_collision_pairs: Any


def _finger_joint_names(arm: str) -> tuple[str, ...]:
    return tuple(f"{arm}_hand_{suffix}_joint" for suffix in DEX3_MOTOR_JOINT_SUFFIXES[arm])


def _validated_finger_q(values: Any, *, label: str) -> np.ndarray:
    result = np.asarray(values, dtype=np.float64).reshape(-1)
    if result.shape != (7,) or not np.all(np.isfinite(result)):
        raise ValueError(f"{label} must contain seven finite values")
    return result


def _collision_buffer_value(buffer: Any, link_name: str) -> float:
    if isinstance(buffer, dict):
        return float(buffer.get(link_name, 0.0))
    return float(buffer or 0.0)


def _world_collision_buffer_deltas(
    strict_robot: dict[str, Any],
    *,
    arm: str,
    mode: str,
) -> dict[str, float]:
    """Return the resolved sphere-radius changes for one optimizer scope.

    This reproduces CuRobo's independently loaded physical-mode model.  The
    strict checker remains unchanged and rechecks complete self geometry after
    every optimized window.
    """

    if mode not in ("supported", "open_free", "open_contact", "attached"):
        raise ValueError(f"unsupported MPC physical mode: {mode}")
    if mode in ("supported", "attached"):
        return {}
    target = _selected_open_transit_world_robot(strict_robot, arm=arm)
    if mode == "open_contact":
        target = _disable_world_collision_links(
            target,
            link_names=_object_contact_links(arm),
        )
    source_buffer = strict_robot["kinematics"].get("collision_sphere_buffer", 0.0)
    target_buffer = target["kinematics"].get("collision_sphere_buffer", 0.0)
    return {
        link_name: _collision_buffer_value(target_buffer, link_name)
        - _collision_buffer_value(source_buffer, link_name)
        for link_name in strict_robot["kinematics"]["collision_spheres"]
        # CuRobo's loader leaves reserved, non-positive attachment slots at
        # their original disabled radius even when a dictionary buffer exists
        # for that link.  Do the same in the in-place representation.
        if any(
            float(sphere["radius"]) > 0.0
            for sphere in strict_robot["kinematics"]["collision_spheres"][link_name]
        )
        if _collision_buffer_value(target_buffer, link_name)
        != _collision_buffer_value(source_buffer, link_name)
    }


class MovingGraspMPC:
    """Persistent CuRobo MPC for the visually updated grasp approach."""

    def __init__(
        self,
        clearance_request: TabletopTaskRequest,
        execution: TabletopExecutionPlan | None,
        *,
        phase: str,
        loaded_request: TabletopTaskRequest | None = None,
        measured_active_dex3_q_rad: np.ndarray | None = None,
        reference_T_camera0: np.ndarray | None = None,
    ) -> None:
        import torch
        from curobo.model_predictive_control import (
            ModelPredictiveControl,
            ModelPredictiveControlCfg,
        )
        from curobo.types import DeviceCfg

        if execution is not None:
            if clearance_request.content_sha256 != execution.clearance_request_sha256:
                raise ValueError("MPC request differs from the frozen clearance request")
            if clearance_request.arm != execution.task.arm:
                raise ValueError("MPC request and execution plan select different arms")
            if loaded_request is not None:
                if loaded_request.content_sha256 != execution.loaded_request_sha256:
                    raise ValueError("MPC loaded request differs from the frozen execution plan")
                if loaded_request.arm != clearance_request.arm:
                    raise ValueError("loaded and clearance MPC requests select different arms")
            initial_fingers = _validated_finger_q(
                execution.task.initial_active_dex3_q_rad,
                label="initial active Dex3 posture",
            )
            initial_route = next(
                trajectory
                for trajectory in execution.trajectories
                if trajectory.to_pose_id == phase
            )
            object_T_grasp = np.asarray(execution.task.object_T_grasp, dtype=np.float64)
            plan_sha256 = execution.content_sha256
            payload_start_plane_clearance_m = float(
                execution.task.planner_provenance.get(
                    "payload_start_plane_clearance_m",
                    clearance_request.minimum_hand_plane_clearance_m,
                )
            )
        else:
            if phase != "grasp_approach" or loaded_request is not None:
                raise ValueError(
                    "provisional MPC construction supports only unloaded grasp_approach"
                )
            initial_fingers = _validated_finger_q(
                dex3_execution_profile(clearance_request.arm)[0],
                label="provisional empty-open active Dex3 posture",
            )
            command_q = np.asarray(clearance_request.planning_snapshot.measured_q29_rad)[
                np.asarray(arm_indices(clearance_request.arm))
            ]
            model_q = np.asarray(
                [
                    value + clearance_request.joint_position_offsets_rad.get(name, 0.0)
                    for name, value in zip(
                        arm_joint_names(clearance_request.arm),
                        command_q,
                        strict=True,
                    )
                ],
                dtype=np.float64,
            )
            initial_route = PlannedTrajectory(
                from_pose_id="move_to_pregrasp",
                to_pose_id="grasp_approach",
                sample_time_s=(0.0, MPC_EXECUTOR_DT_S),
                command_q_rad=(tuple(command_q), tuple(command_q)),
                model_q_rad=(tuple(model_q), tuple(model_q)),
                planning_time_s=0.0,
            )
            object_T_grasp = np.eye(4, dtype=np.float64)
            plan_sha256 = clearance_request.content_sha256
            payload_start_plane_clearance_m = (
                clearance_request.minimum_hand_plane_clearance_m
            )
        self.loaded_request = loaded_request
        self.clearance_request = clearance_request
        self.execution = execution
        self._provisional_route = initial_route if execution is None else None
        self._initial_active_dex3_q_rad = initial_fingers.copy()
        self._object_T_grasp = _rigid_transform(object_T_grasp)
        self._plan_sha256 = plan_sha256
        self._payload_start_plane_clearance_m = payload_start_plane_clearance_m
        self.arm = clearance_request.arm
        self.names = arm_joint_names(self.arm)
        self.device_cfg = DeviceCfg(device=torch.device("cuda:0"), dtype=torch.float32)

        strict_robot, _reference = build_tabletop_robot_config(
            arm=self.arm,
            snapshot=clearance_request.planning_snapshot,
            joint_position_offsets_rad=clearance_request.joint_position_offsets_rad,
            active_finger_q_rad=tuple(float(value) for value in initial_fingers),
        )
        from g1_dex3_tabletop.planning.curobo_backend import (
            CuroboKinematicCollisionChecker,
        )

        # The strict checker already owns the exact full-robot kinematics.
        # Reuse it for this fixed-torso FK query instead of resolving and
        # constructing a duplicate Kinematics model on every cold MPC start.
        self._strict_checker = CuroboKinematicCollisionChecker(
            robot=strict_robot,
            device_cfg=self.device_cfg,
        )
        torso_state = _joint_state(
            self.device_cfg,
            np.asarray(initial_route.model_q_rad[0], dtype=np.float64),
            np.zeros(7, dtype=np.float64),
            np.zeros(7, dtype=np.float64),
            self.names,
        )
        base_T_torso = (
            self._strict_checker.kinematics.compute_kinematics(torso_state)
            .tool_poses["torso_link"]
            .get_matrix()[0]
            .detach()
            .cpu()
            .numpy()
        )
        # CuRobo evaluates FK in float32 on the GPU. Project the tiny numerical
        # drift back onto SO(3) before crossing into the strict float64 geometry
        # contracts used by the controller process.
        base_T_torso[:3, :3] = Rotation.from_matrix(base_T_torso[:3, :3]).as_matrix()
        robot = copy.deepcopy(strict_robot)
        _use_moving_grasp_frame_only(robot, arm=self.arm)
        resolved_robot = _resolved_robot_with_velocity_limit(
            robot,
            device_cfg=self.device_cfg,
            # Leave a small numerical reserve inside the independently
            # enforced controller limit. CuRobo's float32 interpolation can
            # otherwise exceed an exactly equal limit by a few 1e-4 rad/s.
            maximum_velocity_rad_s=(
                clearance_request.maximum_arm_velocity_rad_s * MPC_OPTIMIZER_VELOCITY_SCALE
            ),
        )
        initial_kinematics = _ResolvedPhaseKinematics(
            params=resolved_robot.kinematics.kinematics_config.clone(),
            self_collision_padding=(
                resolved_robot.kinematics.self_collision_config.sphere_padding.clone()
            ),
            self_collision_pairs=(
                resolved_robot.kinematics.self_collision_config.collision_pairs.clone()
            ),
        )
        scene = _base_scene(
            clearance_request,
            base_T_torso,
            include_cube=True,
            include_open_transit_table_patch=True,
        )
        self._scene_topology_key = _scene_topology_key(scene)
        self._strict_robot = strict_robot
        self._plane_point, base_T_object, self._down = _table_from_resting_object(
            clearance_request,
            base_T_torso,
        )
        base_T_object = _rigid_transform(base_T_object)
        self._base_T_object0 = _rigid_transform(base_T_object)
        self._fixture_checker = _fixture_collision_checker(
            clearance_request,
            base_T_object,
            _base_T_detected_object(clearance_request, base_T_torso),
            self._down,
            device_cfg=self.device_cfg,
        )
        self._cube_scene = _base_scene(
            clearance_request,
            base_T_torso,
            include_cube=True,
            include_open_transit_table_patch=False,
        )
        self._object0_T_table_patch: np.ndarray | None = None
        table_patch = scene.get("cuboid", {}).get("open_transit_table_patch")
        if table_patch is not None:
            self._object0_T_table_patch = _rigid_transform(
                invert_transform(base_T_object)
                @ _matrix_from_pose_list(table_patch["pose"])
            )
        self._base_T_torso0 = _rigid_transform(base_T_torso)
        # ``reference`` is a genuinely fixed table frame for moving-target
        # operation.  Legacy command-free replays may omit it and retain the
        # original stationary-cube frame.
        self._reference_T_camera0 = _rigid_transform(
            invert_transform(
                np.asarray(clearance_request.planning_camera_T_object, dtype=np.float64)
            )
            if reference_T_camera0 is None
            else reference_T_camera0
        )
        self._reference_T_torso0 = _rigid_transform(
            self._reference_T_camera0
            @ invert_transform(np.asarray(clearance_request.torso_T_camera, dtype=np.float64))
        )
        nominal_base_T_reference = _rigid_transform(
            self._base_T_torso0 @ invert_transform(self._reference_T_torso0)
        )
        self._reference_T_object0 = _rigid_transform(
            invert_transform(nominal_base_T_reference) @ base_T_object
        )
        torso0_T_base = invert_transform(self._base_T_torso0)
        self._reference_T_obstacles: dict[str, np.ndarray] = {}
        for obstacle_group in ("cuboid", "mesh"):
            for name, obstacle in scene.get(obstacle_group, {}).items():
                # The detected object is updated independently from every
                # live image.  It must never be propagated as a fixed board
                # obstacle when the table frame and object frame are separate.
                if name in ("cube", "open_transit_table_patch"):
                    continue
                base_T_obstacle = _matrix_from_pose_list(obstacle["pose"])
                self._reference_T_obstacles[name] = _rigid_transform(
                    self._reference_T_torso0 @ torso0_T_base @ base_T_obstacle
                )
        self._strict_nominal_base_T_live_base = np.eye(4, dtype=np.float64)
        self._active_world_correction: dict[str, Any] = {
            "camera_translation_from_anchor_m": 0.0,
            "camera_rotation_from_anchor_deg": 0.0,
            "strict_scene_frame_translation_m": 0.0,
            "strict_scene_frame_rotation_deg": 0.0,
        }
        cfg = ModelPredictiveControlCfg.create(
            robot=resolved_robot,
            scene_model=scene,
            collision_cache={"cuboid": 4, "mesh": 1},
            device_cfg=self.device_cfg,
            use_cuda_graph=True,
            # In the pinned CuRobo implementation this is the interpolated
            # state/command period.  The B-spline knot period is four times
            # this value; retained output proves sequence.dt follows it.
            optimization_dt=MPC_COMMAND_DT_S,
            interpolation_steps=MPC_INTERPOLATION_STEPS,
            optimizer_collision_activation_distance=COLLISION_ACTIVATION_DISTANCE_M,
            position_tolerance=0.005,
            orientation_tolerance=0.05,
            cold_start_optimization_num_iters=MPC_COLD_START_ITERATIONS,
            warm_start_optimization_num_iters=MPC_WARM_START_ITERATIONS,
            use_deceleration_on_failure=True,
            random_seed=clearance_request.random_seed,
        )
        self.mpc = ModelPredictiveControl(cfg)
        self._solver_kinematics_cfgs = self._collect_solver_kinematics_cfgs()
        self._assert_compatible_kinematics(initial_kinematics)
        self._resolved_finger_kinematics: dict[bytes, _ResolvedPhaseKinematics] = {
            initial_fingers.tobytes(): initial_kinematics,
        }
        open_fingers = _validated_finger_q(
            self._initial_active_dex3_q_rad,
            label="measured empty-open active Dex3 posture",
        )
        self._resolved_finger_kinematics[open_fingers.tobytes()] = self._resolve_finger_kinematics(
            open_fingers
        )
        self._world_collision_deltas = {
            "open_contact": _world_collision_buffer_deltas(
                strict_robot,
                arm=self.arm,
                mode="open_contact",
            )
        }
        self._payload_spheres = _payload_link_spheres(
            clearance_request.object_dimensions_m,
            object_T_grasp=self._object_T_grasp,
        )
        self._payload_sphere_count = 0
        effective = _numpy(self.mpc.kinematics.get_joint_limits().velocity[1])
        if not np.allclose(
            effective,
            clearance_request.maximum_arm_velocity_rad_s * MPC_OPTIMIZER_VELOCITY_SCALE,
            atol=1.0e-7,
            rtol=0.0,
        ):
            raise RuntimeError(f"CuRobo MPC effective velocity limits are {effective.tolist()}")
        self._setup = False
        # One solver replaces the former four physical-mode solvers.  Preserve
        # the last optimized action seed for each mode so returning to a mode
        # (notably open-contact for grasp retreat) has the same useful warm
        # start that its dedicated solver used to retain.
        self._geometry_action_seed_cache: dict[tuple[str, bytes], Any] = {}
        self._state_correction_warmed_geometry: set[tuple[str, bytes]] = set()
        self._last_state_correction_prewarm_s = 0.0
        self._generation = 0
        self._last_window_valid_from_s: float | None = None
        self._last_window_content_sha256: str | None = None
        self._committed_action_seed = None
        self._active_goal_pose: np.ndarray | None = None
        self._active_goal_model_q: np.ndarray | None = None
        self._active_goal_is_corrected = False
        self._active_reference_T_object: np.ndarray | None = None
        self._lookahead_rad = (
            clearance_request.maximum_arm_velocity_rad_s
            * MPC_OPTIMIZER_VELOCITY_SCALE
            * self.mpc.action_horizon
            * MPC_KNOT_DT_S
            # Every certified horizon must finish at zero velocity.  The
            # largest symmetric accelerate/decelerate displacement at a
            # bounded peak speed is v*T/2, not v*T.
            * 0.5
        )
        self.spec = mpc_phase_spec(phase)
        self.request = clearance_request
        self.route = initial_route
        self.path_model_q = np.asarray(initial_route.model_q_rad, dtype=np.float64)
        self.active_finger_q_rad = initial_fingers
        self._activate_phase_geometry(
            phase,
            measured_active_dex3_q_rad=measured_active_dex3_q_rad,
            reset_optimizer=False,
        )

    def bind_moving_grasp_execution(
        self,
        clearance_request: TabletopTaskRequest,
        execution: TabletopExecutionPlan,
        *,
        loaded_request: TabletopTaskRequest | None,
        reference_T_camera0: np.ndarray,
    ) -> dict[str, Any]:
        """Bind fresh route and scene values into an already-warm MPC topology."""

        import torch
        from curobo.types import Pose

        started = time.perf_counter()
        if clearance_request.content_sha256 != execution.clearance_request_sha256:
            raise ValueError("MPC request differs from the frozen clearance request")
        if clearance_request.arm != self.arm or execution.task.arm != self.arm:
            raise ValueError("warmed MPC belongs to another arm")
        if loaded_request is not None:
            if loaded_request.content_sha256 != execution.loaded_request_sha256:
                raise ValueError("MPC loaded request differs from the frozen execution plan")
            if loaded_request.arm != self.arm:
                raise ValueError("loaded MPC request belongs to another arm")
        route = next(
            (
                trajectory
                for trajectory in execution.trajectories
                if trajectory.to_pose_id == "grasp_approach"
            ),
            None,
        )
        if route is None:
            raise ValueError("frozen execution has no grasp_approach route")
        initial_fingers = _validated_finger_q(
            execution.task.initial_active_dex3_q_rad,
            label="measured empty-open active Dex3 posture",
        )
        strict_robot, _reference = build_tabletop_robot_config(
            arm=self.arm,
            snapshot=clearance_request.planning_snapshot,
            joint_position_offsets_rad=clearance_request.joint_position_offsets_rad,
            active_finger_q_rad=tuple(float(value) for value in initial_fingers),
        )
        robot = copy.deepcopy(strict_robot)
        _use_moving_grasp_frame_only(robot, arm=self.arm)
        resolved_robot = _resolved_robot_with_velocity_limit(
            robot,
            device_cfg=self.device_cfg,
            maximum_velocity_rad_s=(
                clearance_request.maximum_arm_velocity_rad_s
                * MPC_OPTIMIZER_VELOCITY_SCALE
            ),
        )
        kinematics = _ResolvedPhaseKinematics(
            params=resolved_robot.kinematics.kinematics_config.clone(),
            self_collision_padding=(
                resolved_robot.kinematics.self_collision_config.sphere_padding.clone()
            ),
            self_collision_pairs=(
                resolved_robot.kinematics.self_collision_config.collision_pairs.clone()
            ),
        )
        self._assert_compatible_kinematics(kinematics)
        for config in self._solver_kinematics_cfgs:
            config.kinematics_config.copy_(kinematics.params)
            config.self_collision_config.sphere_padding.copy_(
                kinematics.self_collision_padding
            )
        self._apply_strict_checker_kinematics(kinematics)

        route_start = np.asarray(route.model_q_rad[0], dtype=np.float64)
        torso_state = _joint_state(
            self.device_cfg,
            route_start,
            np.zeros(7, dtype=np.float64),
            np.zeros(7, dtype=np.float64),
            self.names,
        )
        base_T_torso = (
            self._strict_checker.kinematics.compute_kinematics(torso_state)
            .tool_poses["torso_link"]
            .get_matrix()[0]
            .detach()
            .cpu()
            .numpy()
        )
        base_T_torso[:3, :3] = Rotation.from_matrix(base_T_torso[:3, :3]).as_matrix()
        scene = _base_scene(
            clearance_request,
            base_T_torso,
            include_cube=True,
            include_open_transit_table_patch=True,
        )
        if _scene_topology_key(scene) != self._scene_topology_key:
            raise RuntimeError("live tabletop scene changes the warmed MPC obstacle topology")
        for group in ("cuboid", "mesh"):
            for name, obstacle in scene.get(group, {}).items():
                self.mpc.scene_collision_checker.update_obstacle_pose(
                    name,
                    Pose.from_matrix(
                        self.device_cfg.to_device(
                            _matrix_from_pose_list(obstacle["pose"])[None]
                        )
                    ),
                )

        self.loaded_request = loaded_request
        self.clearance_request = clearance_request
        self.execution = execution
        self._provisional_route = None
        self._initial_active_dex3_q_rad = initial_fingers.copy()
        self._object_T_grasp = _rigid_transform(execution.task.object_T_grasp)
        self._plan_sha256 = execution.content_sha256
        self._payload_start_plane_clearance_m = float(
            execution.task.planner_provenance.get(
                "payload_start_plane_clearance_m",
                clearance_request.minimum_hand_plane_clearance_m,
            )
        )
        self.request = clearance_request
        self.route = route
        self.path_model_q = np.asarray(route.model_q_rad, dtype=np.float64)
        self.active_finger_q_rad = initial_fingers
        self._strict_robot = strict_robot
        self._plane_point, base_T_object, self._down = _table_from_resting_object(
            clearance_request,
            base_T_torso,
        )
        base_T_object = _rigid_transform(base_T_object)
        self._base_T_object0 = _rigid_transform(base_T_object)
        self._fixture_checker = _fixture_collision_checker(
            clearance_request,
            base_T_object,
            _base_T_detected_object(clearance_request, base_T_torso),
            self._down,
            device_cfg=self.device_cfg,
        )
        self._cube_scene = _base_scene(
            clearance_request,
            base_T_torso,
            include_cube=True,
            include_open_transit_table_patch=False,
        )
        self._object0_T_table_patch = None
        table_patch = scene.get("cuboid", {}).get("open_transit_table_patch")
        if table_patch is not None:
            self._object0_T_table_patch = _rigid_transform(
                invert_transform(base_T_object)
                @ _matrix_from_pose_list(table_patch["pose"])
            )
        self._base_T_torso0 = _rigid_transform(base_T_torso)
        self._reference_T_camera0 = _rigid_transform(reference_T_camera0)
        self._reference_T_torso0 = _rigid_transform(
            self._reference_T_camera0
            @ invert_transform(np.asarray(clearance_request.torso_T_camera, dtype=np.float64))
        )
        nominal_base_T_reference = _rigid_transform(
            self._base_T_torso0 @ invert_transform(self._reference_T_torso0)
        )
        self._reference_T_object0 = _rigid_transform(
            invert_transform(nominal_base_T_reference) @ base_T_object
        )
        torso0_T_base = invert_transform(self._base_T_torso0)
        self._reference_T_obstacles = {}
        for group in ("cuboid", "mesh"):
            for name, obstacle in scene.get(group, {}).items():
                if name in ("cube", "open_transit_table_patch"):
                    continue
                self._reference_T_obstacles[name] = _rigid_transform(
                    self._reference_T_torso0
                    @ torso0_T_base
                    @ _matrix_from_pose_list(obstacle["pose"])
                )
        self._strict_nominal_base_T_live_base = np.eye(4, dtype=np.float64)
        self._active_world_correction = {
            "camera_translation_from_anchor_m": 0.0,
            "camera_rotation_from_anchor_deg": 0.0,
            "strict_scene_frame_translation_m": 0.0,
            "strict_scene_frame_rotation_deg": 0.0,
        }
        self._resolved_finger_kinematics = {
            initial_fingers.tobytes(): kinematics,
        }
        self._world_collision_deltas = {
            "open_contact": _world_collision_buffer_deltas(
                strict_robot,
                arm=self.arm,
                mode="open_contact",
            )
        }
        self._payload_spheres = _payload_link_spheres(
            clearance_request.object_dimensions_m,
            object_T_grasp=self._object_T_grasp,
        )
        self._payload_sphere_count = 0
        self.spec = mpc_phase_spec("grasp_approach")
        self._apply_world_collision_scope(self.spec.mode)
        self._set_optimizer_obstacles(self.spec)
        self._update_live_world(self._reference_T_camera0)
        self._geometry_action_seed_cache.clear()
        self._state_correction_warmed_geometry.clear()
        self._last_state_correction_prewarm_s = 0.0
        self._generation = 0
        self._last_window_valid_from_s = None
        self._last_window_content_sha256 = None
        self._committed_action_seed = None
        self._active_goal_pose = None
        self._active_goal_model_q = None
        self._active_goal_is_corrected = False
        self._active_reference_T_object = None
        self._lookahead_rad = (
            clearance_request.maximum_arm_velocity_rad_s
            * MPC_OPTIMIZER_VELOCITY_SCALE
            * self.mpc.action_horizon
            * MPC_KNOT_DT_S
            * 0.5
        )
        self._setup = False
        setup_started = time.perf_counter()
        setup_s = self.setup_at_frozen_route_start()
        torch.cuda.synchronize()
        return {
            "rebind_time_s": setup_started - started,
            "setup_time_s": setup_s,
            "total_time_s": time.perf_counter() - started,
            "plan_sha256": self._plan_sha256,
            "reused_warm_model": True,
        }

    def _collect_solver_kinematics_cfgs(self) -> tuple[Any, ...]:
        """Return every distinct MPC/IK kinematics buffer owner.

        The pinned CuRobo revision shares one tensor set across all MPC and IK
        rollouts.  Iterating defensively over unique owners keeps the update
        correct if a later revision stops sharing them.
        """

        result: list[Any] = []
        seen: set[int] = set()
        for core in (self.mpc.core, self.mpc.ik_solver.core):
            for rollout in core.get_all_rollout_instances():
                config = rollout.transition_model.robot_model.config
                key = id(config.kinematics_config)
                if key in seen:
                    continue
                seen.add(key)
                result.append(config)
        if not result:
            raise RuntimeError("CuRobo MPC exposes no kinematics buffers")
        return tuple(result)

    def _assert_compatible_kinematics(self, state: _ResolvedPhaseKinematics) -> None:
        """Require value-only phase switching with invariant tensor shapes."""

        import torch

        source = state.params
        for config in self._solver_kinematics_cfgs:
            target = config.kinematics_config
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
                if getattr(target, name).shape != getattr(source, name).shape:
                    raise RuntimeError(f"phase kinematics changes CuRobo {name} shape")
            if target.joint_names != source.joint_names:
                raise RuntimeError("phase kinematics changes active CuRobo joints")
            if target.lock_jointstate.joint_names != source.lock_jointstate.joint_names:
                raise RuntimeError("phase kinematics changes locked CuRobo joints")
            if config.self_collision_config.sphere_padding.shape != (
                state.self_collision_padding.shape
            ):
                raise RuntimeError("phase kinematics changes self-collision padding shape")
            if not torch.equal(
                config.self_collision_config.collision_pairs, state.self_collision_pairs
            ):
                raise RuntimeError("phase kinematics changes self-collision pair topology")
        strict = self._strict_checker.config.kinematics_config
        if strict.fixed_transforms.shape != source.fixed_transforms.shape:
            raise RuntimeError("strict checker and MPC have different link-tree shapes")
        if strict.link_spheres.shape != source.link_spheres.shape:
            raise RuntimeError("strict checker and MPC have different collision-sphere shapes")
        if strict.lock_jointstate.joint_names != source.lock_jointstate.joint_names:
            raise RuntimeError("strict checker and MPC have different locked joints")
        if not torch.equal(
            self._strict_checker.config.self_collision_config.collision_pairs,
            state.self_collision_pairs,
        ):
            raise RuntimeError("strict checker and MPC have different self-collision pairs")

    def _resolve_finger_kinematics(
        self,
        active_finger_q_rad: np.ndarray,
    ) -> _ResolvedPhaseKinematics:
        """Resolve folded fixed transforms once for a new locked-finger posture."""

        finger = _validated_finger_q(
            active_finger_q_rad,
            label="active Dex3 phase posture",
        )
        robot, _reference = build_tabletop_robot_config(
            arm=self.arm,
            snapshot=self.clearance_request.planning_snapshot,
            joint_position_offsets_rad=self.clearance_request.joint_position_offsets_rad,
            active_finger_q_rad=tuple(float(value) for value in finger),
        )
        _use_moving_grasp_frame_only(robot, arm=self.arm)
        resolved = _resolved_robot_with_velocity_limit(
            robot,
            device_cfg=self.device_cfg,
            maximum_velocity_rad_s=self.clearance_request.maximum_arm_velocity_rad_s,
        )
        state = _ResolvedPhaseKinematics(
            params=resolved.kinematics.kinematics_config.clone(),
            self_collision_padding=(
                resolved.kinematics.self_collision_config.sphere_padding.clone()
            ),
            self_collision_pairs=(
                resolved.kinematics.self_collision_config.collision_pairs.clone()
            ),
        )
        self._assert_compatible_kinematics(state)
        return state

    def _finger_q_for_phase(
        self,
        spec: MPCPhaseSpec,
        measured_active_dex3_q_rad: np.ndarray | None,
    ) -> np.ndarray:
        if spec.finger_state == "initial":
            return _validated_finger_q(
                self._initial_active_dex3_q_rad,
                label="initial active Dex3 posture",
            )
        if spec.finger_state == "open":
            return _validated_finger_q(
                self._initial_active_dex3_q_rad,
                label="measured empty-open active Dex3 posture",
            )
        if measured_active_dex3_q_rad is None:
            raise ValueError(f"MPC phase {spec.phase} requires measured close fingers")
        return _validated_finger_q(
            measured_active_dex3_q_rad,
            label="measured MPC contact fingers",
        )

    def _apply_strict_checker_kinematics(
        self,
        state: _ResolvedPhaseKinematics,
    ) -> None:
        """Update only posture-dependent strict-checker values.

        The strict checker keeps an extra ``torso_link`` query frame, so its
        tool-frame map intentionally differs from the MPC model.  The folded
        link transforms, locked joint witness, and collision spheres are the
        posture-dependent values that must match.
        """

        target = self._strict_checker.config.kinematics_config
        source = state.params
        target.fixed_transforms.copy_(source.fixed_transforms)
        target.lock_jointstate.copy_(source.lock_jointstate)
        target.link_spheres.copy_(source.link_spheres)
        if target.reference_link_spheres is not None and source.reference_link_spheres is not None:
            target.reference_link_spheres.copy_(source.reference_link_spheres)
        self._strict_checker.config.self_collision_config.sphere_padding.copy_(
            state.self_collision_padding
        )

    def _apply_world_collision_scope(self, mode: str) -> None:
        """Apply the exact radii from the independently loaded mode model."""

        deltas = self._world_collision_deltas[mode]
        if not deltas:
            return
        for config in self._solver_kinematics_cfgs:
            params = config.kinematics_config
            for link_name, delta in deltas.items():
                indices = params.get_sphere_index_from_link_name(link_name)
                params.link_spheres[:, indices, 3] += delta

    def _set_optimizer_obstacles(self, spec: MPCPhaseSpec) -> None:
        checker = self.mpc.scene_collision_checker
        checker.enable_obstacle("cube", enable=spec.include_cube_in_optimizer)
        checker.enable_obstacle(
            "open_transit_table_patch",
            enable=spec.include_table_patch,
        )
        if self.clearance_request.fixture is not None:
            checker.enable_obstacle(
                self.clearance_request.fixture.fixture_id,
                enable=spec.include_fixture_in_optimizer,
            )

    def _apply_phase_payload(self, *, attached: bool) -> None:
        if not attached:
            self._payload_sphere_count = 0
            return
        for config in self._solver_kinematics_cfgs:
            _install_payload_spheres_on_params(
                config.kinematics_config,
                arm=self.arm,
                spheres=self._payload_spheres,
            )
        _install_payload_spheres_on_checker(
            self._strict_checker,
            arm=self.arm,
            spheres=self._payload_spheres,
        )
        self._payload_sphere_count = len(self._payload_spheres)

    def _activate_phase_geometry(
        self,
        phase: str,
        *,
        measured_active_dex3_q_rad: np.ndarray | None,
        reset_optimizer: bool,
    ) -> dict[str, Any]:
        """Switch one warmed solver to another fixed-size physical phase."""

        import torch

        started = time.perf_counter()
        spec = mpc_phase_spec(phase)
        previous_geometry = (
            (self.spec.mode, self.active_finger_q_rad.tobytes()) if self._setup else None
        )
        if spec.request_state == "loaded":
            if self.loaded_request is None:
                raise ValueError(f"MPC phase {phase} requires the loaded tabletop request")
            request = self.loaded_request
        else:
            request = self.clearance_request
        fingers = self._finger_q_for_phase(spec, measured_active_dex3_q_rad)
        key = fingers.tobytes()
        geometry = (spec.mode, key)
        geometry_changed = previous_geometry != geometry
        if reset_optimizer and previous_geometry is not None and geometry_changed:
            self._geometry_action_seed_cache[previous_geometry] = (
                self.mpc.trajectory_execution_manager.get_action_buffer().clone()
            )
        kinematics_cache_hit = key in self._resolved_finger_kinematics
        resolve_started = time.perf_counter()
        if not kinematics_cache_hit:
            self._resolved_finger_kinematics[key] = self._resolve_finger_kinematics(fingers)
        resolve_s = time.perf_counter() - resolve_started
        state = self._resolved_finger_kinematics[key]
        self._assert_compatible_kinematics(state)
        for config in self._solver_kinematics_cfgs:
            config.kinematics_config.copy_(state.params)
            config.self_collision_config.sphere_padding.copy_(state.self_collision_padding)
        self._apply_strict_checker_kinematics(state)
        self._apply_world_collision_scope(spec.mode)
        self._set_optimizer_obstacles(spec)
        self._apply_phase_payload(attached=spec.attached_payload)

        self.spec = spec
        self.request = request
        self.route = self._route_for_phase(phase)
        self.path_model_q = np.asarray(self.route.model_q_rad, dtype=np.float64)
        self.active_finger_q_rad = fingers
        # Phase preparation validates and prewarms the immutable nominal route.
        # The first live step immediately reapplies the newest estimator pose.
        # Resetting here also prevents the preceding phase's scene transform
        # from leaking across a physical-mode switch.
        self._update_live_world(self._reference_T_camera0)
        self._active_goal_pose = None
        self._active_goal_model_q = None
        self._active_goal_is_corrected = False
        self._last_window_valid_from_s = None
        self._last_window_content_sha256 = None
        self._committed_action_seed = None
        prewarm_s = 0.0
        state_correction_prewarm_s = 0.0
        if reset_optimizer and self._setup and geometry_changed:
            route_start = self.path_model_q[0]
            cached_seed = self._geometry_action_seed_cache.get(geometry)
            if cached_seed is None:
                route_start_state = _joint_state(
                    self.device_cfg,
                    route_start,
                    np.zeros(7, dtype=np.float64),
                    np.zeros(7, dtype=np.float64),
                    self.names,
                )
                self.mpc.reset_robot(route_start_state)
                self.update_nominal_goal(route_start)
                # Spend the first 200-iteration solve during phase preparation,
                # before a live LowState freshness timestamp exists.  The first
                # requested command window can then use the 100-iteration warm
                # path instead of consuming nearly the complete 100 ms age
                # allowance in the planner worker.
                prewarm_started = time.perf_counter()
                self.mpc.optimize_action_sequence(route_start_state)
                torch.cuda.synchronize()
                prewarm_s = time.perf_counter() - prewarm_started
            else:
                # Re-evaluate the retained mode seed against the newly active
                # fixed-size collision/kinematics values.  Clearing optimizer
                # history avoids carrying quasi-Newton state across physical
                # models, while keeping the mode's trajectory seed recreates
                # the useful per-mode warm start without another CUDA graph.
                self.update_nominal_goal(route_start)
                self.mpc.trajectory_execution_manager.update_action_buffer(cached_seed.clone())
                self.mpc.optimizer.reinitialize(cached_seed.clone())
                self.mpc._mpc_warm_start_available = True
            state_correction_prewarm_s = self._prewarm_state_corrected_goal(geometry)
            strict = self._strict_window_diagnostics(np.repeat(route_start[None, :], 2, axis=0))
            if not bool(strict["strict_valid"]):
                raise RuntimeError(
                    f"frozen {phase} start fails strict MPC validation after phase switch: "
                    f"{strict['strict_failure']}"
                )
        torch.cuda.synchronize()
        if self._setup:
            self._committed_action_seed = (
                self.mpc.trajectory_execution_manager.get_action_buffer().clone()
            )
        return {
            "phase": phase,
            "physical_mode": spec.mode,
            "kinematics_cache_hit": kinematics_cache_hit,
            "kinematics_resolve_time_s": resolve_s,
            "optimizer_prewarm_time_s": prewarm_s,
            "state_correction_prewarm_time_s": state_correction_prewarm_s,
            "reconfiguration_time_s": time.perf_counter() - started,
        }

    def _prewarm_state_corrected_goal(self, geometry: tuple[str, bytes]) -> float:
        """Warm local-branch IK before a live state freshness clock exists."""

        if not self.spec.reference_fixed_goal or geometry in (
            self._state_correction_warmed_geometry
        ):
            return 0.0
        started = time.perf_counter()
        route_start = self.path_model_q[0]
        self.update_anchored_goal(
            route_start,
            reference_T_camera=self._reference_T_camera0,
        )
        self.update_nominal_goal(route_start)
        self._state_correction_warmed_geometry.add(geometry)
        elapsed = time.perf_counter() - started
        self._last_state_correction_prewarm_s = elapsed
        return elapsed

    def _route_for_phase(self, phase: str):
        if self.execution is None:
            if phase != "grasp_approach" or self._provisional_route is None:
                raise ValueError("provisional MPC exposes only grasp_approach")
            return self._provisional_route
        matches = tuple(
            trajectory
            for trajectory in self.execution.trajectories
            if trajectory.to_pose_id == phase
        )
        if len(matches) != 1:
            raise ValueError(f"frozen plan has {len(matches)} routes ending at {phase}")
        return matches[0]

    def _strict_world_spheres(self, sphere_tensor):
        """Express live-body robot spheres in the nominal strict-scene frame."""

        transform = self._strict_nominal_base_T_live_base
        result = sphere_tensor.clone()
        rotation = result.new_tensor(transform[:3, :3])
        translation = result.new_tensor(transform[:3, 3])
        result[..., :3] = sphere_tensor[..., :3] @ rotation.T + translation
        return result

    def _strict_window_diagnostics(self, model_q_rad: np.ndarray) -> dict[str, Any]:
        """Reapply the frozen planner's strict checks to one returned window."""

        values = np.asarray(model_q_rad, dtype=np.float64)
        if values.ndim != 2 or values.shape[1] != 7 or not np.all(np.isfinite(values)):
            raise ValueError("strict MPC validation requires a finite N x 7 route")
        diagnostics: dict[str, Any] = {
            "strict_valid": True,
            "strict_sample_count": len(values),
            "strict_failure": None,
            **self._active_world_correction,
        }

        limits = _numpy(self._strict_checker.kinematics.get_joint_limits().position)
        if limits.shape != (2, 7):
            raise RuntimeError(f"strict MPC joint limits have unexpected shape {limits.shape}")
        violations = np.argwhere((values < limits[0][None, :]) | (values > limits[1][None, :]))
        if len(violations):
            sample_index, joint_index = (int(item) for item in violations[0])
            diagnostics.update(
                {
                    "strict_valid": False,
                    "strict_failure": "joint_limit",
                    "strict_failure_sample": sample_index,
                    "strict_failure_links": [self.names[joint_index]],
                    "strict_failure_position_rad": float(values[sample_index, joint_index]),
                    "strict_failure_limits_rad": [
                        float(limits[0, joint_index]),
                        float(limits[1, joint_index]),
                    ],
                }
            )
            return diagnostics

        sphere_tensor = self._strict_checker.robot_spheres(values)
        collisions = self._strict_checker.self_collision_pair_penetrations_from_spheres(
            sphere_tensor
        )
        for sample_index, pairs in enumerate(collisions):
            if not pairs:
                continue
            pair, penetration = max(pairs.items(), key=lambda item: item[1])
            diagnostics.update(
                {
                    "strict_valid": False,
                    "strict_failure": "self_collision",
                    "strict_failure_sample": sample_index,
                    "strict_failure_links": list(pair),
                    "strict_failure_penetration_m": float(penetration),
                }
            )
            return diagnostics

        world_sphere_tensor = self._strict_world_spheres(sphere_tensor)
        spheres = world_sphere_tensor.detach().cpu().numpy().reshape(len(values), -1, 4)
        config = self._strict_checker.config.kinematics_config
        hand_clearance, hand_link, hand_sample = _local_plane_clearance_from_spheres(
            spheres,
            config=config,
            arm=self.arm,
            plane_point=self._plane_point,
            down=self._down,
            include_payload=False,
        )
        diagnostics.update(
            {
                "minimum_hand_plane_clearance_m": float(hand_clearance),
                "minimum_hand_plane_link": hand_link,
                "minimum_hand_plane_sample": hand_sample,
            }
        )
        if self.spec.mode == "supported":
            hand_floor = (
                float(
                    self.execution.supported_escape.planner_provenance[
                        "local_plane_start_clearance_m"
                    ]
                )
                - COLLISION_ACTIVATION_DISTANCE_M
            )
        else:
            hand_floor = self.clearance_request.minimum_hand_plane_clearance_m
        diagnostics["required_hand_plane_clearance_m"] = hand_floor
        if hand_clearance < hand_floor:
            diagnostics.update(
                {
                    "strict_valid": False,
                    "strict_failure": "hand_table_plane",
                    "strict_failure_sample": hand_sample,
                    "strict_failure_links": [hand_link, "table_plane"],
                }
            )
            return diagnostics

        if self.spec.attached_payload:
            payload_clearance, payload_link, payload_sample = _local_plane_clearance_from_spheres(
                spheres,
                config=config,
                arm=self.arm,
                plane_point=self._plane_point,
                down=self._down,
                include_payload=True,
            )
            payload_floor = (
                self._payload_start_plane_clearance_m
                - COLLISION_ACTIVATION_DISTANCE_M
            )
            diagnostics.update(
                {
                    "minimum_payload_plane_clearance_m": float(payload_clearance),
                    "minimum_payload_plane_link": payload_link,
                    "minimum_payload_plane_sample": payload_sample,
                    "required_payload_plane_clearance_m": payload_floor,
                    "attachment_sphere_count": self._payload_sphere_count,
                }
            )
            if payload_clearance < payload_floor:
                diagnostics.update(
                    {
                        "strict_valid": False,
                        "strict_failure": "payload_table_plane",
                        "strict_failure_sample": payload_sample,
                        "strict_failure_links": [payload_link, "table_plane"],
                    }
                )
                return diagnostics
        else:
            disabled_cube_links = (
                set(_object_contact_links(self.arm))
                if self.spec.mode in ("open_free", "open_contact")
                else set()
            )
            cube_samples = _world_cuboid_clearances(
                robot=self._strict_robot,
                q_samples=values,
                scene=self._cube_scene,
                device_cfg=self.device_cfg,
                disabled_links=disabled_cube_links,
                checker=self._strict_checker,
                sphere_array=spheres,
            )
            closest_cube = None
            for sample_index, clearances in enumerate(cube_samples):
                if not clearances:
                    continue
                pair, clearance = min(clearances.items(), key=lambda item: item[1])
                if closest_cube is None or clearance < closest_cube[2]:
                    closest_cube = (sample_index, pair, float(clearance))
            if closest_cube is not None:
                sample_index, pair, clearance = closest_cube
                diagnostics.update(
                    {
                        "strict_valid": False,
                        "strict_failure": "cube_clearance_below_required_margin",
                        "strict_failure_sample": sample_index,
                        "strict_failure_links": list(pair),
                        "strict_failure_clearance_m": clearance,
                        "required_cube_clearance_m": OPEN_TRANSIT_OBJECT_CLEARANCE_M,
                    }
                )
                return diagnostics

        if self._fixture_checker is not None:
            required_fixture_clearance = (
                0.0
                if self.spec.attached_payload
                else float(np.float32(COLLISION_ACTIVATION_DISTANCE_M))
            )
            fixture_spheres = world_sphere_tensor
            excluded_fixture_links = _fixture_excluded_links(self.spec, arm=self.arm)
            if excluded_fixture_links:
                fixture_spheres = world_sphere_tensor.clone()
                for link_name in excluded_fixture_links:
                    indices = config.get_sphere_index_from_link_name(link_name).reshape(-1)
                    fixture_spheres[..., indices, 3] = -100.0
            fixture_clearance, fixture_link, fixture_sample = (
                self._fixture_checker.minimum_clearance(
                    fixture_spheres,
                    kinematics_config=config,
                    activation_distance_m=COLLISION_ACTIVATION_DISTANCE_M,
                )
            )
            diagnostics.update(
                {
                    "minimum_fixture_clearance_m": fixture_clearance,
                    "minimum_fixture_clearance_link": fixture_link,
                    "minimum_fixture_clearance_sample": fixture_sample,
                    "fixture_clearance_search_distance_m": (COLLISION_ACTIVATION_DISTANCE_M),
                    "required_fixture_clearance_m": (required_fixture_clearance),
                }
            )
            if fixture_clearance < required_fixture_clearance:
                diagnostics.update(
                    {
                        "strict_valid": False,
                        "strict_failure": "fixture_collision_or_activation_distance",
                        "strict_failure_sample": fixture_sample,
                        "strict_failure_links": [fixture_link, self.request.fixture.fixture_id],
                        "strict_failure_clearance_m": fixture_clearance,
                    }
                )
        return diagnostics

    def setup(
        self,
        *,
        model_q_rad: np.ndarray,
        model_dq_rad_s: np.ndarray,
        validate_strict_start: bool = True,
    ) -> float:
        state = _joint_state(
            self.device_cfg,
            model_q_rad,
            model_dq_rad_s,
            np.zeros(7, dtype=np.float64),
            self.names,
        )
        started = time.perf_counter()
        self.mpc.setup(state)
        # ``setup`` captures/warmups CUDA and then deliberately resets CuRobo's
        # action buffer, which would otherwise make the first live request pay
        # for another 200-iteration cold solve.  Complete that solve now, while
        # no LowState freshness timestamp exists.
        self.mpc.optimize_action_sequence(state)
        import torch

        geometry = (self.spec.mode, self.active_finger_q_rad.tobytes())
        self._prewarm_state_corrected_goal(geometry)

        warm_route = np.repeat(
            np.asarray(model_q_rad, dtype=np.float64)[None, :],
            2,
            axis=0,
        )
        if validate_strict_start:
            strict = self._strict_window_diagnostics(warm_route)
            if not bool(strict["strict_valid"]):
                raise RuntimeError(
                    f"frozen {self.spec.phase} start fails strict MPC validation: "
                    f"{strict['strict_failure']}"
                )
        torch.cuda.synchronize()
        self._committed_action_seed = (
            self.mpc.trajectory_execution_manager.get_action_buffer().clone()
        )
        self._setup = True
        return time.perf_counter() - started

    def setup_at_frozen_route_start(self, *, validate_strict_start: bool = True) -> float:
        """Build CUDA graphs before any live state freshness clock starts."""

        return self.setup(
            model_q_rad=self.path_model_q[0],
            model_dq_rad_s=np.zeros(7, dtype=np.float64),
            validate_strict_start=validate_strict_start,
        )

    def tool_pose(self, model_q_rad: np.ndarray) -> np.ndarray:
        state = _joint_state(
            self.device_cfg,
            model_q_rad,
            np.zeros(7, dtype=np.float64),
            np.zeros(7, dtype=np.float64),
            self.names,
        )
        result = (
            self.mpc.compute_kinematics(state)
            .tool_poses[grasp_frame(self.arm)]
            .get_matrix()[0]
            .detach()
            .cpu()
            .numpy()
        )
        result[:3, :3] = Rotation.from_matrix(result[:3, :3]).as_matrix()
        return result

    def update_nominal_goal(self, model_q_rad: np.ndarray) -> None:
        """Track one local pose/joint waypoint from the validated route."""

        from curobo.types import GoalToolPose, Pose

        q = np.asarray(model_q_rad, dtype=np.float64).reshape(-1)
        if q.shape != (7,) or not np.all(np.isfinite(q)):
            raise ValueError("MPC goal must contain seven finite model coordinates")
        goal_state = _joint_state(
            self.device_cfg,
            q,
            np.zeros(7),
            np.zeros(7),
            self.names,
        )
        goal_matrix = _rigid_transform(self.tool_pose(q))
        goal_pose = Pose.from_matrix(self.device_cfg.to_device(goal_matrix[None]))
        goals = GoalToolPose.from_poses(
            {grasp_frame(self.arm): goal_pose},
            ordered_tool_frames=[grasp_frame(self.arm)],
            num_goalset=1,
        )
        if not self.mpc.update_goal_tool_poses(goals, run_ik=False):
            raise RuntimeError("CuRobo MPC rejected the nominal tool-pose goal")
        self.mpc.update_goal_state(goal_state)
        # The route already fixes all seven arm coordinates and therefore its
        # hand pose.  A simultaneous Cartesian reach cost gives the redundant
        # arm freedom to leave that collision-validated branch.  Use CuRobo's
        # joint-goal MPC for transit; Cartesian accuracy is still checked by
        # FK at the corrected terminal state.
        self.mpc.disable_tool_pose_tracking()
        self.mpc.enable_joint_position_tracking()
        self._active_goal_pose = goal_matrix
        self._active_goal_model_q = q.copy()
        self._active_goal_is_corrected = False

    def _update_live_world(self, reference_T_camera: np.ndarray) -> dict[str, Any]:
        """Express the fixed task scene in the estimator's live body frame."""

        from curobo.types import Pose

        reference_T_camera = _rigid_transform(reference_T_camera)
        reference_T_torso = _rigid_transform(
            reference_T_camera
            @ invert_transform(np.asarray(self.request.torso_T_camera, dtype=np.float64))
        )
        base_T_reference = _rigid_transform(
            self._base_T_torso0 @ invert_transform(reference_T_torso)
        )
        for name, reference_T_obstacle in self._reference_T_obstacles.items():
            corrected_base_T_obstacle = _rigid_transform(base_T_reference @ reference_T_obstacle)
            self.mpc.scene_collision_checker.update_obstacle_pose(
                name,
                Pose.from_matrix(self.device_cfg.to_device(corrected_base_T_obstacle[None])),
            )

        strict_transform = _nominal_base_T_live_base(
            base_T_torso0=self._base_T_torso0,
            reference_T_torso0=self._reference_T_torso0,
            reference_T_torso=reference_T_torso,
        )
        self._strict_nominal_base_T_live_base = strict_transform
        camera_delta = reference_T_camera @ invert_transform(self._reference_T_camera0)
        self._active_world_correction = {
            "camera_translation_from_anchor_m": float(
                np.linalg.norm(reference_T_camera[:3, 3] - self._reference_T_camera0[:3, 3])
            ),
            "camera_rotation_from_anchor_deg": float(
                np.degrees(Rotation.from_matrix(camera_delta[:3, :3]).magnitude())
            ),
            "strict_scene_frame_translation_m": float(np.linalg.norm(strict_transform[:3, 3])),
            "strict_scene_frame_rotation_deg": float(
                np.degrees(Rotation.from_matrix(strict_transform[:3, :3].copy()).magnitude())
            ),
        }
        return dict(self._active_world_correction)

    def update_anchored_goal(
        self,
        model_q_rad: np.ndarray,
        *,
        reference_T_camera: np.ndarray,
    ) -> dict[str, Any]:
        """Move the local goal and scene with the estimated live torso pose."""

        from curobo.types import GoalToolPose, Pose

        reference_T_camera = _rigid_transform(reference_T_camera)
        reference_T_torso = _rigid_transform(
            reference_T_camera
            @ invert_transform(np.asarray(self.request.torso_T_camera, dtype=np.float64))
        )
        nominal_base_T_goal = _rigid_transform(self.tool_pose(model_q_rad))
        reference_T_goal = _rigid_transform(
            self._reference_T_torso0 @ invert_transform(self._base_T_torso0) @ nominal_base_T_goal
        )
        corrected_base_T_goal = _rigid_transform(
            self._base_T_torso0 @ invert_transform(reference_T_torso) @ reference_T_goal
        )
        state_correction = self._update_live_world(reference_T_camera)
        goal_pose = Pose.from_matrix(self.device_cfg.to_device(corrected_base_T_goal[None]))
        goals = GoalToolPose.from_poses(
            {grasp_frame(self.arm): goal_pose},
            ordered_tool_frames=[grasp_frame(self.arm)],
            num_goalset=1,
        )
        # Preserve the frozen route's local IK branch explicitly. CuRobo's
        # convenience ``update_goal_tool_poses(run_ik=True)`` seeds from the
        # previous MPC goal, which can lag far behind the rolling lookahead and
        # stall progress. The nominal waypoint is already a validated solution
        # for the uncorrected pose, so it is the correct one-seed initializer
        # for this small frame correction.
        nominal_goal_state = _joint_state(
            self.device_cfg,
            np.asarray(model_q_rad, dtype=np.float64),
            np.zeros(7, dtype=np.float64),
            np.zeros(7, dtype=np.float64),
            self.names,
        )
        ik_result = self.mpc.ik_solver.solve_pose(
            goal_tool_poses=goals,
            current_state=nominal_goal_state,
            seed_config=nominal_goal_state.position.view(1, 1, 7).clone(),
            return_seeds=1,
        )
        if ik_result is None or not bool(ik_result.success.reshape(-1)[0].item()):
            raise RuntimeError("CuRobo MPC could not solve the state-corrected local goal")
        corrected_goal_q = _numpy(ik_result.solution).reshape(-1, 7)[0]
        corrected_goal_state = _joint_state(
            self.device_cfg,
            corrected_goal_q,
            np.zeros(7, dtype=np.float64),
            np.zeros(7, dtype=np.float64),
            self.names,
        )
        if not self.mpc.update_goal_tool_poses(goals, run_ik=False):
            raise RuntimeError("CuRobo MPC rejected the state-corrected Cartesian goal")
        self.mpc.update_goal_state(corrected_goal_state)
        self.mpc.disable_tool_pose_tracking()
        self.mpc.enable_joint_position_tracking()
        self._active_goal_pose = corrected_base_T_goal
        self._active_goal_model_q = corrected_goal_q.copy()
        self._active_goal_is_corrected = True
        correction = corrected_base_T_goal @ invert_transform(nominal_base_T_goal)
        return {
            **state_correction,
            "goal_translation_correction_m": float(np.linalg.norm(correction[:3, 3])),
            "goal_rotation_correction_deg": float(
                np.degrees(Rotation.from_matrix(correction[:3, :3]).magnitude())
            ),
        }

    def _moving_object_geometry(
        self,
        *,
        reference_T_camera: np.ndarray,
        camera_T_object: np.ndarray,
    ) -> tuple[dict[str, Any], np.ndarray, np.ndarray, np.ndarray]:
        """Map one board/camera/object observation into the MPC base frame."""

        reference_T_camera = _rigid_transform(reference_T_camera)
        camera_T_object = _rigid_transform(camera_T_object)
        state_correction = self._update_live_world(reference_T_camera)
        reference_T_torso = _rigid_transform(
            reference_T_camera
            @ invert_transform(np.asarray(self.request.torso_T_camera, dtype=np.float64))
        )
        live_base_T_reference = _rigid_transform(
            self._base_T_torso0 @ invert_transform(reference_T_torso)
        )
        nominal_base_T_reference = _rigid_transform(
            self._base_T_torso0 @ invert_transform(self._reference_T_torso0)
        )
        reference_T_detected_object = _rigid_transform(
            reference_T_camera @ camera_T_object
        )
        nominal_base_T_detected_object = _rigid_transform(
            nominal_base_T_reference @ reference_T_detected_object
        )
        nominal_base_T_object = _rigid_transform(
            canonical_resting_cube_pose(nominal_base_T_detected_object)
        )
        reference_T_object = _rigid_transform(
            invert_transform(nominal_base_T_reference) @ nominal_base_T_object
        )
        live_base_T_object = _rigid_transform(
            live_base_T_reference @ reference_T_object
        )
        return (
            state_correction,
            reference_T_object,
            nominal_base_T_object,
            live_base_T_object,
        )

    def update_moving_grasp_goal(
        self,
        *,
        reference_T_camera: np.ndarray,
        camera_T_object: np.ndarray,
        nominal_goal_model_q_rad: np.ndarray,
        terminal_goal: bool,
        geometry: tuple[dict[str, Any], np.ndarray, np.ndarray, np.ndarray] | None = None,
    ) -> dict[str, Any]:
        """Install one live object pose and one reachable approach-line goal.

        ``reference`` is the fixed table-board frame.  The cube is observed in
        the camera frame on every update, while the existing proprioceptive
        estimator supplies the board-to-camera transform.  CuRobo therefore
        receives both the live cube obstacle and the corresponding short
        look-ahead on the already validated object-relative grasp line.  The
        global route is not replayed by MPC; it supplies the selected arm
        branch and the geometric grasp line only.
        """

        from curobo.types import GoalToolPose, Pose

        if self.spec.phase != "grasp_approach":
            raise RuntimeError("moving-target MPC is only valid for grasp_approach")
        if geometry is None:
            geometry = self._moving_object_geometry(
                reference_T_camera=reference_T_camera,
                camera_T_object=camera_T_object,
            )
        (
            state_correction,
            reference_T_object,
            nominal_base_T_object,
            live_base_T_object,
        ) = geometry
        nominal_goal_q = np.asarray(nominal_goal_model_q_rad, dtype=np.float64).reshape(-1)
        if nominal_goal_q.shape != (7,) or not np.all(np.isfinite(nominal_goal_q)):
            raise ValueError("moving-target nominal goal must contain seven finite values")
        if terminal_goal:
            object0_T_goal = _rigid_transform(
                self._object_T_grasp
            )
        else:
            object0_T_goal = _rigid_transform(
                invert_transform(self._base_T_object0) @ self.tool_pose(nominal_goal_q)
            )
        live_base_T_goal = _rigid_transform(
            live_base_T_object @ object0_T_goal
        )

        self.mpc.scene_collision_checker.update_obstacle_pose(
            "cube",
            Pose.from_matrix(self.device_cfg.to_device(live_base_T_object[None])),
        )
        self._cube_scene["cuboid"]["cube"]["pose"] = _pose_list_from_matrix(
            nominal_base_T_object
        )
        if self._object0_T_table_patch is not None:
            live_base_T_patch = _rigid_transform(
                live_base_T_object @ self._object0_T_table_patch
            )
            self.mpc.scene_collision_checker.update_obstacle_pose(
                "open_transit_table_patch",
                Pose.from_matrix(self.device_cfg.to_device(live_base_T_patch[None])),
            )
        goal_pose = Pose.from_matrix(self.device_cfg.to_device(live_base_T_goal[None]))
        goals = GoalToolPose.from_poses(
            {grasp_frame(self.arm): goal_pose},
            ordered_tool_frames=[grasp_frame(self.arm)],
            num_goalset=1,
        )
        # Track the exact live Cartesian pose while using the matching frozen
        # approach sample as redundant-arm posture regularization.  The route
        # sample is already exact when the observed object has not moved.  For
        # a moved object, solve only the local retargeting IK from that branch.
        nominal_goal_pose = self.tool_pose(nominal_goal_q)
        goal_translation_delta = float(
            np.linalg.norm(nominal_goal_pose[:3, 3] - live_base_T_goal[:3, 3])
        )
        goal_rotation_delta = float(
            Rotation.from_matrix(
                nominal_goal_pose[:3, :3].T @ live_base_T_goal[:3, :3]
            ).magnitude()
        )
        goal_model_q = nominal_goal_q.copy()
        if goal_translation_delta > 1.0e-5 or goal_rotation_delta > 1.0e-4:
            nominal_goal_state = _joint_state(
                self.device_cfg,
                nominal_goal_q,
                np.zeros(7, dtype=np.float64),
                np.zeros(7, dtype=np.float64),
                self.names,
            )
            ik_result = self.mpc.ik_solver.solve_pose(
                goal_tool_poses=goals,
                current_state=nominal_goal_state,
                seed_config=nominal_goal_state.position.view(1, 1, 7).clone(),
                return_seeds=1,
            )
            if ik_result is None or not bool(ik_result.success.reshape(-1)[0].item()):
                evidence = "no result"
                if ik_result is not None:
                    evidence = (
                        "position_error="
                        f"{float(_numpy(ik_result.position_error).reshape(-1)[0]):.6f}m, "
                        "rotation_error="
                        f"{float(_numpy(ik_result.rotation_error).reshape(-1)[0]):.6f}rad, "
                        f"feasible={bool(_numpy(ik_result.feasible).reshape(-1)[0])}"
                    )
                raise RuntimeError(
                    f"CuRobo MPC could not retarget the live approach branch: {evidence}"
                )
            goal_model_q = _numpy(ik_result.solution).reshape(-1, 7)[0]
        if not self.mpc.update_goal_tool_poses(goals, run_ik=False):
            raise RuntimeError("CuRobo MPC rejected the live Cartesian grasp goal")
        self.mpc.update_goal_state(
            _joint_state(
                self.device_cfg,
                goal_model_q,
                np.zeros(7, dtype=np.float64),
                np.zeros(7, dtype=np.float64),
                self.names,
            )
        )
        self.mpc.enable_joint_position_tracking()
        self.mpc.enable_tool_pose_tracking()
        self._active_goal_pose = live_base_T_goal
        self._active_goal_model_q = goal_model_q.copy()
        self._active_goal_is_corrected = True
        self._active_reference_T_object = reference_T_object

        object_delta = reference_T_object @ invert_transform(self._reference_T_object0)
        return {
            **state_correction,
            "goal_source": "live_cube_pose_in_fixed_table_board_frame",
            "goal_tracking": "object_relative_approach_line_lookahead",
            "terminal_grasp_goal": terminal_goal,
            "nominal_goal_model_q_rad": nominal_goal_q.tolist(),
            "retargeted_goal_model_q_rad": goal_model_q.tolist(),
            "goal_translation_from_nominal_m": goal_translation_delta,
            "goal_rotation_from_nominal_deg": float(np.degrees(goal_rotation_delta)),
            "object_translation_from_plan_m": float(
                np.linalg.norm(reference_T_object[:3, 3] - self._reference_T_object0[:3, 3])
            ),
            "object_rotation_from_plan_deg": float(
                np.degrees(Rotation.from_matrix(object_delta[:3, :3]).magnitude())
            ),
            "reference_T_object": reference_T_object.tolist(),
            "live_base_T_object": live_base_T_object.tolist(),
            "live_base_T_grasp": live_base_T_goal.tolist(),
        }

    def next_moving_target_window(
        self,
        *,
        handoff_predicted_q_rad: np.ndarray,
        handoff_predicted_dq_rad_s: np.ndarray,
        handoff_predicted_ddq_rad_s2: np.ndarray,
        handoff_command_q_rad: np.ndarray,
        source_state_monotonic_s: float,
        valid_from_monotonic_s: float,
        predecessor_sha256: str | None,
        reference_T_camera: np.ndarray,
        camera_T_object: np.ndarray,
        target_provenance: dict[str, Any],
        committed_route_progress_index: int,
    ) -> MPCCommandWindow:
        """Optimize one immutable window toward the latest visual grasp pose."""

        window_started = time.perf_counter()
        predicted_command = np.asarray(handoff_predicted_q_rad, dtype=np.float64).reshape(-1)
        if predicted_command.shape != (7,) or not np.all(np.isfinite(predicted_command)):
            raise ValueError("predicted MPC handoff position must contain seven finite values")
        if not isinstance(target_provenance, dict):
            raise TypeError("moving-target provenance must be a dictionary")
        recorded_camera = _rigid_transform(target_provenance.get("reference_T_camera"))
        recorded_object = _rigid_transform(target_provenance.get("camera_T_object"))
        if not np.allclose(recorded_camera, _rigid_transform(reference_T_camera), atol=1.0e-12):
            raise ValueError("moving-target provenance and reference camera pose differ")
        if not np.allclose(recorded_object, _rigid_transform(camera_T_object), atol=1.0e-12):
            raise ValueError("moving-target provenance and cube pose differ")

        model_q = np.asarray(
            [
                value + self.request.joint_position_offsets_rad.get(name, 0.0)
                for name, value in zip(self.names, predicted_command, strict=True)
            ],
            dtype=np.float64,
        )
        if committed_route_progress_index < 0 or committed_route_progress_index >= len(
            self.path_model_q
        ):
            raise ValueError("committed moving-target route progress is outside the approach")
        geometry = self._moving_object_geometry(
            reference_T_camera=reference_T_camera,
            camera_T_object=camera_T_object,
        )
        live_base_T_object = geometry[3]
        object_T_current = invert_transform(live_base_T_object) @ self.tool_pose(model_q)
        route_progress_index = committed_route_progress_index
        remaining_indices = range(route_progress_index, len(self.path_model_q))
        object_relative_distances = np.asarray(
            [
                np.linalg.norm(
                    (
                        invert_transform(self._base_T_object0)
                        @ self.tool_pose(self.path_model_q[index])
                    )[:3, 3]
                    - object_T_current[:3, 3]
                )
                for index in remaining_indices
            ],
            dtype=np.float64,
        )
        route_progress_index += int(np.argmin(object_relative_distances))
        goal_q, waypoint_index, requested_terminal = _bounded_route_goal(
            self.path_model_q,
            current_q=self.path_model_q[route_progress_index],
            route_progress_index=route_progress_index,
            maximum_distance_rad=self._lookahead_rad,
        )
        goal_update_started = time.perf_counter()
        target_update = self.update_moving_grasp_goal(
            reference_T_camera=reference_T_camera,
            camera_T_object=camera_T_object,
            nominal_goal_model_q_rad=goal_q,
            terminal_goal=requested_terminal,
            geometry=geometry,
        )
        goal_update_time_s = time.perf_counter() - goal_update_started
        window = self.solve_window(
            model_q_rad=model_q,
            model_dq_rad_s=np.asarray(handoff_predicted_dq_rad_s, dtype=np.float64),
            model_ddq_rad_s2=np.asarray(handoff_predicted_ddq_rad_s2, dtype=np.float64),
            active_command_q_rad=np.asarray(handoff_command_q_rad, dtype=np.float64),
            source_state_monotonic_s=source_state_monotonic_s,
            valid_from_monotonic_s=valid_from_monotonic_s,
            predecessor_sha256=predecessor_sha256,
            terminal=requested_terminal,
        )
        if self._active_goal_pose is None:
            raise RuntimeError("moving-target MPC has no active Cartesian goal")
        terminal_model = np.asarray(
            window.diagnostics["predicted_terminal_model_q_rad"], dtype=np.float64
        )
        terminal_pose = self.tool_pose(terminal_model)
        translation_error = float(
            np.linalg.norm(terminal_pose[:3, 3] - self._active_goal_pose[:3, 3])
        )
        rotation_error = float(
            Rotation.from_matrix(
                terminal_pose[:3, :3].T @ self._active_goal_pose[:3, :3]
            ).magnitude()
        )
        terminal = (
            requested_terminal
            and translation_error <= 0.005
            and rotation_error <= 0.05
        )
        values = window.to_dict(include_hash=False)
        values["terminal"] = bool(terminal and window.feasible)
        total_window_time_s = time.perf_counter() - window_started
        values["solve_time_s"] = total_window_time_s
        values["diagnostics"] = {
            **window.diagnostics,
            **target_update,
            "moving_target": dict(target_provenance),
            "terminal_translation_error_m": translation_error,
            "terminal_rotation_error_rad": rotation_error,
            "terminal_corrected_joint_error_rad": None,
            "committed_route_progress_index": committed_route_progress_index,
            "proposed_route_progress_index": route_progress_index,
            "route_progress_index": route_progress_index,
            "route_waypoint_index": waypoint_index,
            "object_relative_route_distance_m": float(
                object_relative_distances[route_progress_index - committed_route_progress_index]
            ),
            "goal_update_time_s": goal_update_time_s,
            "mpc_core_window_time_s": window.solve_time_s,
            "worker_total_window_time_s": total_window_time_s,
        }
        window = MPCCommandWindow.from_dict(values)
        if window.feasible:
            self._last_window_valid_from_s = window.valid_from_monotonic_s
            self._last_window_content_sha256 = window.content_sha256
            self._committed_action_seed = (
                self.mpc.trajectory_execution_manager.get_action_buffer().clone()
            )
        elif self._committed_action_seed is not None:
            self.mpc.update_seed_trajectory(self._committed_action_seed)
        return window

    def _align_warm_start_to_handoff(
        self,
        *,
        valid_from_monotonic_s: float,
        predecessor_sha256: str | None,
    ) -> int:
        """Shift CuRobo's retained knot seed by the committed handoff interval."""

        if predecessor_sha256 is None:
            if self._last_window_content_sha256 is not None:
                raise RuntimeError("MPC predecessor chain restarted inside one phase")
            return 1
        if predecessor_sha256 != self._last_window_content_sha256:
            raise RuntimeError("MPC worker predecessor does not match its retained warm seed")
        assert self._last_window_valid_from_s is not None
        knot_delta = (
            float(valid_from_monotonic_s) - self._last_window_valid_from_s
        ) / MPC_KNOT_DT_S
        elapsed_knots = round(knot_delta)
        if elapsed_knots < 1 or not np.isclose(
            knot_delta,
            elapsed_knots,
            atol=1.0e-6,
            rtol=0.0,
        ):
            raise RuntimeError(
                "successive MPC handoffs are not separated by whole positive knot periods"
            )
        # warm_start_solve() always rolls its seed and optimizer history by one
        # knot. Pre-shift only the additional elapsed knots here.
        additional_shift = elapsed_knots - 1
        if additional_shift:
            seed = self.mpc.trajectory_execution_manager.get_action_buffer().clone()
            if additional_shift >= seed.shape[-2]:
                seed[..., :, :] = seed[..., -1:, :]
            else:
                seed = seed.roll(-additional_shift, dims=-2)
                seed[..., -additional_shift:, :] = seed[
                    ..., -additional_shift - 1 : -additional_shift, :
                ]
            self.mpc.update_seed_trajectory(seed)
        return elapsed_knots

    def solve_window(
        self,
        *,
        model_q_rad: np.ndarray,
        model_dq_rad_s: np.ndarray,
        model_ddq_rad_s2: np.ndarray,
        active_command_q_rad: np.ndarray,
        source_state_monotonic_s: float,
        valid_from_monotonic_s: float,
        predecessor_sha256: str | None,
        terminal: bool,
    ) -> MPCCommandWindow:
        """Optimize and certify the exact dense trajectory exposed to control."""

        import torch

        if not self._setup:
            raise RuntimeError("MPC must be set up before solving")
        current = _joint_state(
            self.device_cfg,
            model_q_rad,
            model_dq_rad_s,
            model_ddq_rad_s2,
            self.names,
        )
        elapsed_warm_start_knots = self._align_warm_start_to_handoff(
            valid_from_monotonic_s=valid_from_monotonic_s,
            predecessor_sha256=predecessor_sha256,
        )
        started = time.perf_counter()
        result = self.mpc.optimize_action_sequence(current)
        torch.cuda.synchronize()
        optimizer_wall_s = time.perf_counter() - started
        sequence = result.action_sequence
        full_robot_sequence = result.robot_state_sequence
        if sequence is None or full_robot_sequence is None:
            raise RuntimeError("CuRobo MPC returned no complete predicted state trajectory")
        full_sequence = full_robot_sequence.joint_state
        returned_dt = _numpy(full_sequence.dt).reshape(-1)
        if len(returned_dt) == 0 or not np.all(np.isfinite(returned_dt)):
            raise RuntimeError("CuRobo MPC returned no finite JointState dt")
        if not np.allclose(returned_dt, returned_dt[0], atol=1.0e-9, rtol=0.0):
            raise RuntimeError(
                f"CuRobo MPC returned nonuniform JointState dt {returned_dt.tolist()}"
            )
        returned_state_dt_s = float(returned_dt[0])
        if not np.isclose(
            returned_state_dt_s,
            MPC_COMMAND_DT_S,
            atol=1.0e-9,
            rtol=0.0,
        ):
            raise RuntimeError(
                "CuRobo MPC returned command dt "
                f"{returned_state_dt_s:.12f}s; expected {MPC_COMMAND_DT_S:.12f}s"
            )
        full_model_positions = _numpy(full_sequence.position).reshape(-1, 7)
        full_model_velocities = _numpy(full_sequence.velocity).reshape(-1, 7)
        full_model_accelerations = _numpy(full_sequence.acceleration).reshape(-1, 7)
        full_model_jerks = _numpy(full_sequence.jerk).reshape(-1, 7)
        # Use the complete state rollout, including CuRobo's terminal support
        # samples.  The B-spline command slice omits both the exact boundary
        # state and the natural deceleration tail; neither omission is valid
        # for an asynchronous future handoff.
        model_commands = full_model_positions
        derivative_fields = {
            "velocity": full_sequence.velocity,
            "acceleration": full_sequence.acceleration,
            "jerk": full_sequence.jerk,
        }
        missing_derivatives = [name for name, value in derivative_fields.items() if value is None]
        if missing_derivatives:
            raise RuntimeError(
                "CuRobo MPC full state trajectory omitted required boundary derivatives: "
                + ", ".join(missing_derivatives)
            )
        model_velocities = full_model_velocities
        model_accelerations = full_model_accelerations
        command_start = np.asarray(active_command_q_rad, dtype=np.float64).reshape(-1)
        if command_start.shape != (7,) or not np.all(np.isfinite(command_start)):
            raise ValueError("active MPC command must contain seven finite values")
        measured_command = command_from_model_q(
            model_q_rad,
            arm=self.arm,
            joint_position_offsets_rad=self.request.joint_position_offsets_rad,
        )
        measured_plan_sequence = np.stack(
            [
                command_from_model_q(
                    row,
                    arm=self.arm,
                    joint_position_offsets_rad=self.request.joint_position_offsets_rad,
                )
                for row in model_commands
            ]
        )
        boundary_position_error = float(
            np.max(np.abs(measured_plan_sequence[0] - measured_command))
        )
        if boundary_position_error > 1.0e-6:
            raise RuntimeError(
                "CuRobo full predicted trajectory does not begin at its supplied "
                f"handoff state: error={boundary_position_error:.9f}rad"
            )
        boundary_derivative_errors = {
            "velocity_rad_s": float(
                np.max(np.abs(model_velocities[0] - np.asarray(model_dq_rad_s, dtype=np.float64)))
            ),
            "acceleration_rad_s2": float(
                np.max(
                    np.abs(model_accelerations[0] - np.asarray(model_ddq_rad_s2, dtype=np.float64))
                )
            ),
        }
        if (
            boundary_derivative_errors["velocity_rad_s"] > 1.0e-5
            or boundary_derivative_errors["acceleration_rad_s2"] > 1.0e-4
        ):
            raise RuntimeError(
                "CuRobo full predicted trajectory does not preserve its supplied "
                f"handoff derivatives: {boundary_derivative_errors}"
            )
        coarse_times = (
            np.arange(len(measured_plan_sequence), dtype=np.float64) * returned_state_dt_s
        )
        command_sequence, tracking_offset = command_sequence_from_measured_plan(
            measured_plan_sequence[1:],
            measured_q_rad=measured_command,
            active_command_q_rad=command_start,
            future_sample_time_s=coarse_times[1:],
        )
        coarse_commands = np.concatenate((command_start[None, :], command_sequence), axis=0)
        coarse_predicted = np.concatenate(
            (measured_command[None, :], measured_plan_sequence[1:]), axis=0
        )
        coarse_predicted_dq = model_velocities
        coarse_predicted_ddq = model_accelerations
        (
            times,
            commands,
            predicted_commands,
            predicted_dq,
            predicted_ddq,
        ) = _resample_to_executor_grid(
            coarse_times,
            coarse_commands,
            coarse_predicted,
            coarse_predicted_dq,
            coarse_predicted_ddq,
        )
        curobo_feasible = bool(
            result.success is not None and bool(result.success.reshape(-1)[0].item())
        )
        curobo_constraints = _mpc_constraint_summary(
            self.mpc.trajectory_execution_manager.get_current_metrics()
        )
        cspace_bound_diagnostics = _cspace_bound_diagnostics(
            self.mpc.metrics_rollout.constraint_manager.get_cost("cspace"),
            names=self.names,
            position=full_model_positions,
            velocity=full_model_velocities,
            acceleration=full_model_accelerations,
            jerk=full_model_jerks,
        )
        full_state_peak_velocity = float(np.max(np.abs(full_model_velocities)))
        curobo_numerically_feasible = _mpc_constraints_numerically_feasible(curobo_constraints)
        curobo_reserved_velocity_feasible = _reserved_velocity_constraint_is_safe(
            curobo_constraints,
            cspace_bound_diagnostics,
            full_state_peak_velocity_rad_s=full_state_peak_velocity,
            physical_velocity_limit_rad_s=self.request.maximum_arm_velocity_rad_s,
        )
        feasible = (
            curobo_feasible or curobo_numerically_feasible or curobo_reserved_velocity_feasible
        )
        uncompensated_coarse_commands = np.concatenate(
            (command_start[None, :], measured_plan_sequence[1:]), axis=0
        )
        planned_measured_peak_velocity = float(
            np.max(np.abs(np.diff(coarse_predicted, axis=0)) / np.diff(coarse_times)[:, None])
        )
        uncompensated_peak_velocity = float(
            np.max(
                np.abs(np.diff(uncompensated_coarse_commands, axis=0))
                / np.diff(coarse_times)[:, None]
            )
        )
        peak_velocity = float(np.max(np.abs(np.diff(commands, axis=0)) / np.diff(times)[:, None]))
        terminal_predicted_velocity = float(np.max(np.abs(predicted_dq[-1])))
        if peak_velocity > self.request.maximum_arm_velocity_rad_s + 1.0e-6:
            feasible = False
        if terminal_predicted_velocity > MPC_STOP_VELOCITY_TOLERANCE_RAD_S:
            feasible = False
        diagnostics = {
            "phase": self.spec.phase,
            "physical_mode": self.spec.mode,
            "curobo_feasible": curobo_feasible,
            "curobo_numerically_feasible": curobo_numerically_feasible,
            "curobo_reserved_velocity_feasible": (curobo_reserved_velocity_feasible),
            "curobo_float32_constraint_epsilon": MPC_FLOAT32_CONSTRAINT_EPSILON,
            "curobo_constraints": curobo_constraints,
            "curobo_cspace_bound_diagnostics": cspace_bound_diagnostics,
            "curobo_full_state_peak_velocity_rad_s": full_state_peak_velocity,
            "optimizer_wall_time_s": optimizer_wall_s,
            "curobo_reported_solve_time_s": float(result.solve_time),
            "peak_velocity_rad_s": peak_velocity,
            "terminal_predicted_velocity_rad_s": terminal_predicted_velocity,
            "terminal_velocity_tolerance_rad_s": (MPC_STOP_VELOCITY_TOLERANCE_RAD_S),
            "planned_measured_peak_velocity_rad_s": planned_measured_peak_velocity,
            "uncompensated_peak_velocity_rad_s": uncompensated_peak_velocity,
            "command_tracking_offset_rad": tracking_offset.tolist(),
            "command_tracking_offset_policy": "hold_complete_window_remeasure_each_update",
            "command_dt_s": MPC_COMMAND_DT_S,
            "knot_dt_s": MPC_KNOT_DT_S,
            "executor_validation_dt_s": MPC_EXECUTOR_DT_S,
            "coarse_sample_count": len(coarse_times),
            "dense_sample_count": len(times),
            "curobo_full_state_boundary_error_rad": boundary_position_error,
            "curobo_full_state_boundary_derivative_errors": (boundary_derivative_errors),
            "curobo_command_start_index": (
                self.mpc.trajectory_execution_manager.command_start_idx
            ),
            "curobo_state_source": "full_robot_state_sequence_including_support",
            "warm_start_elapsed_knots": elapsed_warm_start_knots,
            "maximum_command_tracking_offset_rad": float(np.max(np.abs(tracking_offset))),
            "maximum_command_tracking_offset_joint": self.names[
                int(np.argmax(np.abs(tracking_offset)))
            ],
            "predicted_terminal_model_q_rad": model_commands[-1].tolist(),
            "predicted_model_q_rad": model_commands.tolist(),
            "reported_peak_velocity_rad_s": (
                None
                if sequence is None or sequence.velocity is None
                else float(np.max(np.abs(_numpy(sequence.velocity))))
            ),
            "reported_sequence_dt": (
                None
                if sequence is None or sequence.dt is None
                else _numpy(sequence.dt).reshape(-1).tolist()
            ),
            "position_error_m": (
                None
                if result.position_error is None
                else float(result.position_error.reshape(-1)[0].item())
            ),
            "rotation_error_rad": (
                None
                if result.rotation_error is None
                else float(result.rotation_error.reshape(-1)[0].item())
            ),
        }
        strict_model_commands = np.stack(
            [
                np.asarray(
                    [
                        value + self.request.joint_position_offsets_rad.get(name, 0.0)
                        for name, value in zip(self.names, row, strict=True)
                    ],
                    dtype=np.float64,
                )
                for row in commands
            ]
        )
        strict_started = time.perf_counter()
        predicted_model_route = np.stack(
            [
                np.asarray(
                    [
                        value + self.request.joint_position_offsets_rad.get(name, 0.0)
                        for name, value in zip(self.names, row, strict=True)
                    ],
                    dtype=np.float64,
                )
                for row in predicted_commands
            ]
        )
        predicted_strict_diagnostics = self._strict_window_diagnostics(predicted_model_route)
        command_strict_diagnostics = self._strict_window_diagnostics(strict_model_commands)
        strict_validation_s = time.perf_counter() - strict_started
        diagnostics.update(command_strict_diagnostics)
        diagnostics["predicted_state_strict_validation"] = predicted_strict_diagnostics
        diagnostics["command_target_strict_validation"] = command_strict_diagnostics
        if bool(command_strict_diagnostics["strict_valid"]) and not bool(
            predicted_strict_diagnostics["strict_valid"]
        ):
            diagnostics.update(predicted_strict_diagnostics)
        diagnostics["strict_validation_time_s"] = strict_validation_s
        if not (
            bool(predicted_strict_diagnostics["strict_valid"])
            and bool(command_strict_diagnostics["strict_valid"])
        ):
            feasible = False
        wall_s = time.perf_counter() - started
        diagnostics["worker_wall_time_s"] = wall_s
        window = MPCCommandWindow(
            generation=self._generation,
            plan_sha256=self._plan_sha256,
            source_state_monotonic_s=source_state_monotonic_s,
            valid_from_monotonic_s=valid_from_monotonic_s,
            sample_time_s=tuple(times),
            command_q_rad=tuple(tuple(float(value) for value in row) for row in commands),
            predicted_q_rad=tuple(
                tuple(float(value) for value in row) for row in predicted_commands
            ),
            predicted_dq_rad_s=tuple(tuple(float(value) for value in row) for row in predicted_dq),
            predicted_ddq_rad_s2=tuple(
                tuple(float(value) for value in row) for row in predicted_ddq
            ),
            predecessor_sha256=predecessor_sha256,
            feasible=feasible,
            terminal=bool(terminal and feasible),
            solve_time_s=wall_s,
            diagnostics=diagnostics,
        )
        self._generation += 1
        return window

    def close(self) -> None:
        self.mpc.destroy()
        gc.collect()
