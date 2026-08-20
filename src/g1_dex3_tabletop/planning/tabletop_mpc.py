"""Phase-aware CuRobo MPC for one frozen tabletop lifecycle.

CuRobo continuously replans only arm motion.  The commissioned task state
machine still owns the discrete physical transitions: open fingers, establish
and validate contact, attach the cube model, release it, and restore the
supported handoff.  Every MPC window is independently checked with the same
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
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation

from g1_aprilcube_calibration.joint_map import arm_joint_names
from g1_aprilcube_calibration.transforms import invert_transform, validate_transform
from g1_aprilcube_calibration.transports.unitree_dex3 import DEX3_MOTOR_JOINT_SUFFIXES
from g1_dex3_tabletop.mpc_command_buffer import (
    MPCCommandWindow,
    command_sequence_from_measured_plan,
)
from g1_dex3_tabletop.planning.curobo_backend import (
    COLLISION_ACTIVATION_DISTANCE_M,
    OPEN_TRANSIT_OBJECT_CLEARANCE_M,
)
from g1_dex3_tabletop.planning.g1_model import (
    CUROBO_COMMIT,
    attachment_link,
    build_tabletop_robot_config,
    command_from_model_q,
    grasp_frame,
    model_source_hashes,
)
from g1_dex3_tabletop.planning.tabletop_planner import (
    WORLD_COLLISION_DISABLE_RADIUS_EPSILON_M,
    _base_scene,
    _base_T_detected_object,
    _contact_links,
    _cuboid_cover_spheres,
    _fixture_collision_checker,
    _local_plane_clearance_from_spheres,
    _selected_open_transit_world_robot,
    _table_from_resting_object,
    _use_moving_grasp_frame_only,
    _world_cuboid_clearances,
)
from g1_dex3_tabletop.tabletop_contracts import TabletopExecutionPlan, TabletopTaskRequest

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
MPC_PHASE_ORDER = (
    "move_to_pregrasp",
    "grasp_approach",
    "retention_test_lift",
    "payload_lift",
    "payload_lower",
    "payload_replace",
    "grasp_retreat",
    "return_to_clearance",
)
MPC_ATTACHED_PHASES = frozenset(
    ("retention_test_lift", "payload_lift", "payload_lower", "payload_replace")
)


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
    """Map a motion endpoint to its already-commissioned physical state."""

    if phase not in MPC_PHASE_ORDER:
        raise ValueError(f"unsupported tabletop MPC phase: {phase}")
    if phase == "move_to_pregrasp":
        return MPCPhaseSpec(
            phase=phase,
            mode="open_free",
            request_state="clearance",
            finger_state="open",
            include_cube_in_optimizer=True,
            include_table_patch=True,
            # Exact fixture-mesh distance dominates rolling optimization even
            # though this controller follows an already fixture-validated
            # route. Keep the mesh in the independent strict window check.
            include_fixture_in_optimizer=False,
            allow_fingertip_cube_contact=False,
            attached_payload=False,
            reference_fixed_goal=True,
        )
    if phase == "return_to_clearance":
        return MPCPhaseSpec(
            phase=phase,
            mode="open_free",
            request_state="clearance",
            finger_state="open",
            include_cube_in_optimizer=True,
            include_table_patch=True,
            include_fixture_in_optimizer=False,
            allow_fingertip_cube_contact=False,
            attached_payload=False,
            # Clearance is the exact joint-space junction with the frozen
            # supported return.  Keep this endpoint body-relative while the
            # live table/cube/fixture scene remains state-corrected.
            reference_fixed_goal=False,
        )
    if phase in ("grasp_approach", "grasp_retreat"):
        return MPCPhaseSpec(
            phase=phase,
            mode="open_contact",
            request_state="clearance",
            finger_state="open",
            # Contact links are disabled only against world geometry. The
            # cube therefore remains active for every non-contact link, and
            # the strict post-check independently enforces the same policy.
            include_cube_in_optimizer=True,
            include_table_patch=True,
            include_fixture_in_optimizer=False,
            allow_fingertip_cube_contact=True,
            attached_payload=False,
            reference_fixed_goal=True,
        )
    return MPCPhaseSpec(
        phase=phase,
        mode="attached",
        request_state="clearance",
        finger_state="measured_contact",
        include_cube_in_optimizer=False,
        include_table_patch=False,
        # The attached cube starts and ends in deliberate contact with its
        # presenter. The frozen payload planner therefore omits the fixture
        # from optimization and independently checks every robot sphere (but
        # not the attached-object proxy) against its exact mesh.
        include_fixture_in_optimizer=False,
        allow_fingertip_cube_contact=False,
        attached_payload=True,
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
        target = _disable_world_collision_links(target, link_names=_contact_links(arm))
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


@dataclass(frozen=True, slots=True)
class MPCBenchmarkConfig:
    maximum_steps: int = 300
    waypoint_tolerance_rad: float = 0.005
    handoff_interval_s: float = MPC_HANDOFF_INTERVAL_S
    simulated_tracking_offset_rad: tuple[float, ...] | None = None

    def __post_init__(self) -> None:
        if self.maximum_steps <= 0:
            raise ValueError("MPC benchmark step count must be positive")
        if not np.isfinite(self.waypoint_tolerance_rad) or self.waypoint_tolerance_rad <= 0.0:
            raise ValueError("MPC waypoint tolerance must be positive and finite")
        if (
            not np.isfinite(self.handoff_interval_s)
            or self.handoff_interval_s <= 0.0
            or not np.isclose(
                self.handoff_interval_s / MPC_KNOT_DT_S,
                round(self.handoff_interval_s / MPC_KNOT_DT_S),
                atol=1.0e-9,
                rtol=0.0,
            )
        ):
            raise ValueError("MPC handoff interval must be a positive whole knot period")
        if self.simulated_tracking_offset_rad is not None:
            offset = np.asarray(self.simulated_tracking_offset_rad, dtype=np.float64)
            if offset.shape != (7,) or not np.all(np.isfinite(offset)):
                raise ValueError("simulated MPC tracking offset must contain seven finite values")


class TabletopPhaseMPC:
    """Warm-start MPC bound to one physical mode of a frozen lifecycle."""

    def __init__(
        self,
        clearance_request: TabletopTaskRequest,
        execution: TabletopExecutionPlan,
        *,
        phase: str,
        loaded_request: TabletopTaskRequest | None = None,
        measured_active_dex3_q_rad: np.ndarray | None = None,
    ) -> None:
        import torch
        from curobo.model_predictive_control import (
            ModelPredictiveControl,
            ModelPredictiveControlCfg,
        )
        from curobo.types import DeviceCfg

        if clearance_request.content_sha256 != execution.clearance_request_sha256:
            raise ValueError("MPC request differs from the frozen clearance request")
        if clearance_request.arm != execution.task.arm:
            raise ValueError("MPC request and execution plan select different arms")
        if loaded_request is not None:
            if loaded_request.content_sha256 != execution.loaded_request_sha256:
                raise ValueError("MPC loaded request differs from the frozen execution plan")
            if loaded_request.arm != clearance_request.arm:
                raise ValueError("loaded and clearance MPC requests select different arms")
        self.loaded_request = loaded_request
        self.clearance_request = clearance_request
        self.execution = execution
        self.arm = clearance_request.arm
        self.names = arm_joint_names(self.arm)
        self.device_cfg = DeviceCfg(device=torch.device("cuda:0"), dtype=torch.float32)

        initial_fingers = _validated_finger_q(
            execution.task.initial_active_dex3_q_rad,
            label="initial active Dex3 posture",
        )
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
        initial_route = self._route_for_phase(phase)
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
        self._strict_robot = strict_robot
        self._plane_point, base_T_object, self._down = _table_from_resting_object(
            clearance_request,
            base_T_torso,
        )
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
        self._base_T_torso0 = _rigid_transform(base_T_torso)
        self._reference_T_camera0 = _rigid_transform(
            invert_transform(
                np.asarray(clearance_request.planning_camera_T_object, dtype=np.float64)
            )
        )
        self._reference_T_torso0 = _rigid_transform(
            self._reference_T_camera0
            @ invert_transform(np.asarray(clearance_request.torso_T_camera, dtype=np.float64))
        )
        torso0_T_base = invert_transform(self._base_T_torso0)
        self._reference_T_obstacles: dict[str, np.ndarray] = {}
        for obstacle_group in ("cuboid", "mesh"):
            for name, obstacle in scene.get(obstacle_group, {}).items():
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
            execution.task.initial_active_dex3_q_rad,
            label="measured empty-open active Dex3 posture",
        )
        self._resolved_finger_kinematics[open_fingers.tobytes()] = self._resolve_finger_kinematics(
            open_fingers
        )
        self._world_collision_deltas = {
            mode: _world_collision_buffer_deltas(
                strict_robot,
                arm=self.arm,
                mode=mode,
            )
            for mode in ("supported", "open_free", "open_contact", "attached")
        }
        self._payload_spheres = _payload_link_spheres(
            clearance_request.object_dimensions_m,
            object_T_grasp=execution.task.object_T_grasp,
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
        self._lookahead_rad = (
            clearance_request.maximum_arm_velocity_rad_s
            * MPC_OPTIMIZER_VELOCITY_SCALE
            * self.mpc.action_horizon
            * MPC_KNOT_DT_S
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
                self.execution.task.initial_active_dex3_q_rad,
                label="initial active Dex3 posture",
            )
        if spec.finger_state == "open":
            return _validated_finger_q(
                self.execution.task.initial_active_dex3_q_rad,
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
        matches = tuple(
            trajectory
            for trajectory in self.execution.trajectories
            if trajectory.to_pose_id == phase
        )
        if len(matches) != 1:
            raise ValueError(f"frozen plan has {len(matches)} routes ending at {phase}")
        return matches[0]

    def can_select_phase(
        self,
        phase: str,
        *,
        measured_active_dex3_q_rad: np.ndarray | None,
    ) -> bool:
        """Return whether the warmed fixed-shape solver can represent a phase."""

        candidate = mpc_phase_spec(phase)
        if candidate.request_state == "loaded" and self.loaded_request is None:
            return False
        if candidate.attached_payload:
            if measured_active_dex3_q_rad is None:
                return False
            measured = np.asarray(measured_active_dex3_q_rad, dtype=np.float64).reshape(-1)
            return measured.shape == (7,) and np.all(np.isfinite(measured))
        return True

    def select_phase(
        self,
        phase: str,
        *,
        measured_active_dex3_q_rad: np.ndarray | None = None,
    ) -> dict[str, Any]:
        """Switch the one warmed solver to the next physical phase in place."""

        return self._activate_phase_geometry(
            phase,
            measured_active_dex3_q_rad=measured_active_dex3_q_rad,
            reset_optimizer=True,
        )

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
                float(self.execution.task.planner_provenance["payload_start_plane_clearance_m"])
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
                set(_contact_links(self.arm))
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

    def setup(self, *, model_q_rad: np.ndarray, model_dq_rad_s: np.ndarray) -> float:
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

    def setup_at_frozen_route_start(self) -> float:
        """Build CUDA graphs before any live state freshness clock starts."""

        return self.setup(
            model_q_rad=self.path_model_q[0],
            model_dq_rad_s=np.zeros(7, dtype=np.float64),
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

    def next_nominal_window(
        self,
        *,
        handoff_predicted_q_rad: np.ndarray,
        handoff_predicted_dq_rad_s: np.ndarray,
        handoff_predicted_ddq_rad_s2: np.ndarray,
        handoff_command_q_rad: np.ndarray,
        source_state_monotonic_s: float,
        valid_from_monotonic_s: float,
        predecessor_sha256: str | None,
        committed_route_progress_index: int,
        reference_T_camera: np.ndarray | None = None,
        camera_state_provenance: dict[str, Any] | None = None,
    ) -> MPCCommandWindow:
        """Plan from one immutable future handoff on the frozen phase route."""

        window_started = time.perf_counter()
        predicted_command = np.asarray(handoff_predicted_q_rad, dtype=np.float64).reshape(-1)
        if predicted_command.shape != (7,) or not np.all(np.isfinite(predicted_command)):
            raise ValueError("predicted MPC handoff position must contain seven finite values")
        if committed_route_progress_index < 0 or committed_route_progress_index >= len(
            self.path_model_q
        ):
            raise ValueError("committed MPC route progress is outside the frozen route")
        model_q = np.asarray(
            [
                value + self.request.joint_position_offsets_rad.get(name, 0.0)
                for name, value in zip(self.names, predicted_command, strict=True)
            ],
            dtype=np.float64,
        )
        # Follow the collision-validated route monotonically.  The measured
        # state selects the nearest remaining sample, and the MPC goal is one
        # action horizon farther along that same route.  This preserves the
        # commissioned IK branch while still replanning every rolling window.
        route_progress_index = committed_route_progress_index
        remaining = self.path_model_q[route_progress_index:]
        route_progress_index += int(
            np.argmin(np.max(np.abs(remaining - model_q[None, :]), axis=1))
        )
        goal_q, waypoint_index, requested_terminal = _bounded_route_goal(
            self.path_model_q,
            current_q=model_q,
            route_progress_index=route_progress_index,
            maximum_distance_rad=self._lookahead_rad,
        )
        if camera_state_provenance is not None:
            if reference_T_camera is None:
                raise ValueError("camera-state provenance requires a reference camera pose")
            recorded_pose = _rigid_transform(camera_state_provenance.get("reference_T_camera"))
            if not np.allclose(
                recorded_pose,
                _rigid_transform(reference_T_camera),
                atol=1.0e-12,
                rtol=0.0,
            ):
                raise ValueError("camera-state provenance and correction pose differ")

        goal_update_started = time.perf_counter()
        state_correction: dict[str, Any] = {}
        if reference_T_camera is None:
            self._update_live_world(self._reference_T_camera0)
            self.update_nominal_goal(goal_q)
        elif self.spec.reference_fixed_goal:
            state_correction = self.update_anchored_goal(
                goal_q,
                reference_T_camera=reference_T_camera,
            )
        else:
            # The final clearance target must remain the exact joint-space
            # junction with the frozen supported return.  Only its surrounding
            # fixed world is moved from the live camera/body estimate.
            state_correction = self._update_live_world(reference_T_camera)
            self.update_nominal_goal(goal_q)
            state_correction.update(
                {
                    "goal_translation_correction_m": 0.0,
                    "goal_rotation_correction_deg": 0.0,
                }
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
        if requested_terminal:
            terminal_model = np.asarray(
                window.diagnostics["predicted_terminal_model_q_rad"],
                dtype=np.float64,
            )
            if terminal_model.shape != (7,) or not np.all(np.isfinite(terminal_model)):
                raise RuntimeError("MPC window has no valid predicted terminal model state")
            if self._active_goal_is_corrected:
                if self._active_goal_pose is None or self._active_goal_model_q is None:
                    raise RuntimeError("MPC terminal check has no active corrected goal")
                terminal_pose = self.tool_pose(terminal_model)
                translation_error = float(
                    np.linalg.norm(terminal_pose[:3, 3] - self._active_goal_pose[:3, 3])
                )
                rotation_error = float(
                    Rotation.from_matrix(
                        terminal_pose[:3, :3].T @ self._active_goal_pose[:3, :3]
                    ).magnitude()
                )
                terminal_joint_error = float(
                    np.max(np.abs(terminal_model - self._active_goal_model_q))
                )
                terminal = (
                    translation_error <= 0.005
                    and rotation_error <= 0.05
                    and terminal_joint_error <= 0.005
                )
            else:
                terminal = float(np.max(np.abs(terminal_model - self.path_model_q[-1]))) <= 0.005
                translation_error = None
                rotation_error = None
                terminal_joint_error = None
            if terminal != window.terminal:
                values = window.to_dict(include_hash=False)
                values["terminal"] = terminal and window.feasible
                values["diagnostics"] = {
                    **window.diagnostics,
                    "terminal_translation_error_m": translation_error,
                    "terminal_rotation_error_rad": rotation_error,
                    "terminal_corrected_joint_error_rad": terminal_joint_error,
                }
                window = MPCCommandWindow.from_dict(values)
        if state_correction:
            values = window.to_dict(include_hash=False)
            diagnostics = {**window.diagnostics, **state_correction}
            if camera_state_provenance is not None:
                diagnostics["camera_state_correction"] = dict(camera_state_provenance)
            values["diagnostics"] = diagnostics
            window = MPCCommandWindow.from_dict(values)
        values = window.to_dict(include_hash=False)
        diagnostics = dict(window.diagnostics)
        diagnostics.update(
            {
                "committed_route_progress_index": committed_route_progress_index,
                "proposed_route_progress_index": route_progress_index,
                "route_progress_index": route_progress_index,
                "route_waypoint_index": waypoint_index,
                "route_waypoint_model_q_rad": goal_q.tolist(),
            }
        )
        diagnostics["goal_update_time_s"] = goal_update_time_s
        diagnostics["mpc_core_window_time_s"] = window.solve_time_s
        total_window_time_s = time.perf_counter() - window_started
        diagnostics["worker_total_window_time_s"] = total_window_time_s
        values["diagnostics"] = diagnostics
        values["solve_time_s"] = total_window_time_s
        window = MPCCommandWindow.from_dict(values)
        if window.feasible:
            self._last_window_valid_from_s = window.valid_from_monotonic_s
            self._last_window_content_sha256 = window.content_sha256
            self._committed_action_seed = (
                self.mpc.trajectory_execution_manager.get_action_buffer().clone()
            )
        elif self._committed_action_seed is not None:
            # CuRobo installs every optimizer result in its execution manager,
            # including an infeasible one.  Keep the warm start tied to the
            # last trajectory that the controller actually accepted so a
            # retry cannot be seeded from a rejected path.
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
            plan_sha256=self.execution.content_sha256,
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


def benchmark_open_approach_mpc(
    request: TabletopTaskRequest,
    execution: TabletopExecutionPlan,
    *,
    config: MPCBenchmarkConfig | None = None,
) -> dict[str, Any]:
    """Run a deterministic, command-free MPC replay of the retained approach."""

    import torch

    config = config or MPCBenchmarkConfig()
    started = time.perf_counter()
    controller = TabletopPhaseMPC(
        request,
        execution,
        phase="move_to_pregrasp",
    )
    route_q = controller.path_model_q
    q = route_q[0].copy()
    dq = np.zeros(7, dtype=np.float64)
    ddq = np.zeros(7, dtype=np.float64)
    command_q = command_from_model_q(
        q,
        arm=request.arm,
        joint_position_offsets_rad=request.joint_position_offsets_rad,
    )
    setup_s = controller.setup(model_q_rad=q, model_dq_rad_s=dq)
    accepted = 0
    rejected = 0
    latencies: list[float] = []
    peak_velocities: list[float] = []
    first_rejection: dict[str, Any] | None = None
    simulated_time_s = 0.0
    predecessor_sha256: str | None = None
    committed_route_progress_index = 0
    benchmark_origin_s = time.monotonic()
    try:
        for _step in range(config.maximum_steps):
            source_time = benchmark_origin_s + simulated_time_s
            window = controller.next_nominal_window(
                handoff_predicted_q_rad=command_q,
                handoff_predicted_dq_rad_s=dq,
                handoff_predicted_ddq_rad_s2=ddq,
                handoff_command_q_rad=command_q,
                source_state_monotonic_s=source_time,
                valid_from_monotonic_s=(source_time + config.handoff_interval_s),
                predecessor_sha256=predecessor_sha256,
                committed_route_progress_index=committed_route_progress_index,
            )
            latencies.append(window.solve_time_s)
            peak_velocities.append(window.peak_velocity_rad_s())
            if not window.feasible:
                rejected += 1
                if first_rejection is None:
                    first_rejection = window.to_dict()
                break
            accepted += 1
            predecessor_sha256 = window.content_sha256
            committed_route_progress_index = int(
                window.diagnostics["proposed_route_progress_index"]
            )
            execution_time_s = window.duration_s if window.terminal else config.handoff_interval_s
            window_times = np.asarray(window.sample_time_s, dtype=np.float64)
            window_commands = np.asarray(window.command_q_rad, dtype=np.float64)
            window_predicted = np.asarray(window.predicted_q_rad, dtype=np.float64)
            window_predicted_dq = np.asarray(window.predicted_dq_rad_s, dtype=np.float64)
            window_predicted_ddq = np.asarray(window.predicted_ddq_rad_s2, dtype=np.float64)
            command_q = np.asarray(
                [
                    np.interp(execution_time_s, window_times, window_commands[:, index])
                    for index in range(7)
                ],
                dtype=np.float64,
            )
            predicted_q = np.asarray(
                [
                    np.interp(execution_time_s, window_times, window_predicted[:, index])
                    for index in range(7)
                ],
                dtype=np.float64,
            )
            q = np.asarray(
                [
                    value + request.joint_position_offsets_rad.get(name, 0.0)
                    for name, value in zip(controller.names, predicted_q, strict=True)
                ],
                dtype=np.float64,
            )
            dq = np.asarray(
                [
                    np.interp(
                        execution_time_s,
                        window_times,
                        window_predicted_dq[:, index],
                    )
                    for index in range(7)
                ],
                dtype=np.float64,
            )
            ddq = np.asarray(
                [
                    np.interp(
                        execution_time_s,
                        window_times,
                        window_predicted_ddq[:, index],
                    )
                    for index in range(7)
                ],
                dtype=np.float64,
            )
            simulated_time_s += execution_time_s
            if window.terminal:
                break
    finally:
        controller.close()
        torch.cuda.empty_cache()
    reached = float(np.max(np.abs(q - route_q[-1]))) <= config.waypoint_tolerance_rad
    latency = np.asarray(latencies, dtype=np.float64)
    return {
        "schema_version": 1,
        "kind": "g1_tabletop_open_approach_mpc_benchmark",
        "commands_robot": False,
        "request_sha256": request.content_sha256,
        "execution_plan_sha256": execution.content_sha256,
        "arm": request.arm,
        "phase": "move_to_pregrasp",
        "reached_terminal": reached,
        "accepted_windows": accepted,
        "rejected_windows": rejected,
        "first_rejection": first_rejection,
        "maximum_joint_error_to_terminal_rad": float(np.max(np.abs(q - route_q[-1]))),
        "simulated_time_s": simulated_time_s,
        "setup_time_s": setup_s,
        "benchmark_wall_time_s": time.perf_counter() - started,
        "solve_latency_s": {
            "count": len(latencies),
            "mean": float(np.mean(latency)) if len(latency) else None,
            "p95": float(np.percentile(latency, 95)) if len(latency) else None,
            "maximum": float(np.max(latency)) if len(latency) else None,
        },
        "maximum_window_velocity_rad_s": (
            float(np.max(peak_velocities)) if peak_velocities else None
        ),
        "configuration": {
            "command_dt_s": MPC_COMMAND_DT_S,
            "knot_dt_s": MPC_KNOT_DT_S,
            "interpolation_steps": MPC_INTERPOLATION_STEPS,
            "certified_horizon_source": "complete_curobo_robot_state_sequence",
            "executor_validation_dt_s": MPC_EXECUTOR_DT_S,
            "cold_start_iterations": MPC_COLD_START_ITERATIONS,
            "warm_start_iterations": MPC_WARM_START_ITERATIONS,
            "maximum_steps": config.maximum_steps,
            "waypoint_tolerance_rad": config.waypoint_tolerance_rad,
            "handoff_interval_s": config.handoff_interval_s,
            "maximum_arm_velocity_rad_s": request.maximum_arm_velocity_rad_s,
            "route_tracking": "monotonic_frozen_route_lookahead",
            "route_lookahead_rad": controller._lookahead_rad,
            "collision_activation_distance_m": COLLISION_ACTIVATION_DISTANCE_M,
            "open_transit_object_clearance_m": OPEN_TRANSIT_OBJECT_CLEARANCE_M,
        },
        "provenance": {**model_source_hashes(), "curobo_commit": CUROBO_COMMIT},
    }


def _simulate_phase(
    controller: TabletopPhaseMPC,
    *,
    command_q_rad: np.ndarray,
    model_dq_rad_s: np.ndarray,
    config: MPCBenchmarkConfig,
    reference_T_camera: np.ndarray | None = None,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    """Replay one phase without ROS or a robot and return its terminal state."""

    measured_q = np.asarray(command_q_rad, dtype=np.float64).copy()
    dq = np.asarray(model_dq_rad_s, dtype=np.float64).copy()
    ddq = np.zeros(7, dtype=np.float64)
    tracking_offset = (
        np.zeros(7, dtype=np.float64)
        if config.simulated_tracking_offset_rad is None
        else np.asarray(config.simulated_tracking_offset_rad, dtype=np.float64)
    )
    route_q = controller.path_model_q
    start_model = np.asarray(
        [
            value + controller.request.joint_position_offsets_rad.get(name, 0.0)
            for name, value in zip(controller.names, measured_q, strict=True)
        ],
        dtype=np.float64,
    )
    accepted = 0
    rejected = 0
    windows: list[dict[str, Any]] = []
    simulated_time_s = 0.0
    terminal_received = False
    stopped_at_certified_horizon = False
    install_rejection: str | None = None
    predecessor_sha256: str | None = None
    committed_route_progress_index = 0
    benchmark_origin_s = time.monotonic()
    now_s = benchmark_origin_s
    active_window: MPCCommandWindow | None = None

    def sample_active(at_s: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        assert active_window is not None
        return (
            active_window.sample_command(monotonic_s=at_s),
            active_window.sample_predicted_q(monotonic_s=at_s),
            active_window.sample_predicted_dq(monotonic_s=at_s),
            active_window.sample_predicted_ddq(monotonic_s=at_s),
        )

    for _step in range(config.maximum_steps):
        if active_window is None:
            valid_from_s = now_s + config.handoff_interval_s
            boundary_command = measured_q + tracking_offset
            boundary_q = measured_q
            boundary_dq = dq
            boundary_ddq = ddq
        else:
            relative_s = max(
                now_s + config.handoff_interval_s - active_window.valid_from_monotonic_s,
                0.0,
            )
            handoff_knots = int(np.ceil((relative_s - 1.0e-12) / MPC_KNOT_DT_S))
            valid_from_s = active_window.valid_from_monotonic_s + handoff_knots * MPC_KNOT_DT_S
            if valid_from_s > active_window.expiration_monotonic_s + 1.0e-12:
                stop_s = active_window.expiration_monotonic_s
                command_q, measured_q, dq, ddq = sample_active(stop_s)
                simulated_time_s += max(stop_s - now_s, 0.0)
                now_s = stop_s
                stopped_at_certified_horizon = True
                break
            boundary_command, boundary_q, boundary_dq, boundary_ddq = sample_active(valid_from_s)
        window = controller.next_nominal_window(
            handoff_predicted_q_rad=boundary_q,
            handoff_predicted_dq_rad_s=boundary_dq,
            handoff_predicted_ddq_rad_s2=boundary_ddq,
            handoff_command_q_rad=boundary_command,
            source_state_monotonic_s=now_s,
            valid_from_monotonic_s=valid_from_s,
            predecessor_sha256=predecessor_sha256,
            committed_route_progress_index=committed_route_progress_index,
            reference_T_camera=reference_T_camera,
        )
        windows.append(window.to_dict())
        if not window.feasible:
            rejected += 1
            if active_window is None:
                break
            solve_end_s = min(
                now_s + window.solve_time_s,
                active_window.expiration_monotonic_s,
            )
            command_q, measured_q, dq, ddq = sample_active(solve_end_s)
            simulated_time_s += max(solve_end_s - now_s, 0.0)
            now_s = solve_end_s
            if now_s >= active_window.expiration_monotonic_s - 1.0e-12:
                stopped_at_certified_horizon = True
                break
            continue
        if now_s + window.solve_time_s >= valid_from_s:
            rejected += 1
            install_rejection = "future handoff deadline missed"
            if active_window is not None:
                stop_s = active_window.expiration_monotonic_s
                command_q, measured_q, dq, ddq = sample_active(stop_s)
                simulated_time_s += max(stop_s - now_s, 0.0)
                now_s = stop_s
                stopped_at_certified_horizon = True
            break
        accepted += 1
        if active_window is not None:
            simulated_time_s += max(valid_from_s - now_s, 0.0)
        now_s = valid_from_s
        active_window = window
        command_q, measured_q, dq, ddq = sample_active(now_s)
        predecessor_sha256 = active_window.content_sha256
        committed_route_progress_index = int(
            active_window.diagnostics["proposed_route_progress_index"]
        )
        if not np.allclose(
            command_q - measured_q,
            tracking_offset,
            atol=1.0e-7,
            rtol=0.0,
        ):
            raise RuntimeError("simulated MPC command/model offset changed inside a window")
        if active_window.terminal:
            stop_s = active_window.expiration_monotonic_s
            command_q, measured_q, dq, ddq = sample_active(stop_s)
            simulated_time_s += stop_s - now_s
            now_s = stop_s
            terminal_received = True
            break
    else:
        if active_window is not None:
            stop_s = active_window.expiration_monotonic_s
            command_q, measured_q, dq, ddq = sample_active(stop_s)
            simulated_time_s += max(stop_s - now_s, 0.0)
            stopped_at_certified_horizon = True
    terminal_model = np.asarray(
        [
            value + controller.request.joint_position_offsets_rad.get(name, 0.0)
            for name, value in zip(controller.names, measured_q, strict=True)
        ],
        dtype=np.float64,
    )
    terminal_joint_error = float(np.max(np.abs(terminal_model - route_q[-1])))
    if reference_T_camera is not None and controller.spec.reference_fixed_goal:
        reached = terminal_received
        terminal_pose = controller.tool_pose(terminal_model)
        if controller._active_goal_pose is None:
            raise RuntimeError("corrected MPC replay ended without an active pose goal")
        terminal_translation_error_m = float(
            np.linalg.norm(terminal_pose[:3, 3] - controller._active_goal_pose[:3, 3])
        )
        terminal_rotation_error_rad = float(
            Rotation.from_matrix(
                terminal_pose[:3, :3].T @ controller._active_goal_pose[:3, :3]
            ).magnitude()
        )
    else:
        reached = terminal_joint_error <= config.waypoint_tolerance_rad
        terminal_translation_error_m = None
        terminal_rotation_error_rad = None
    solve_times = np.asarray([item["solve_time_s"] for item in windows], dtype=np.float64)
    slowest_window = (
        max(windows, key=lambda item: float(item["solve_time_s"])) if windows else None
    )
    fixture_windows = [
        item
        for item in windows
        if item["diagnostics"].get("minimum_fixture_clearance_m") is not None
    ]
    closest_fixture_window = (
        min(
            fixture_windows,
            key=lambda item: float(item["diagnostics"]["minimum_fixture_clearance_m"]),
        )
        if fixture_windows
        else None
    )

    def latency_summary(diagnostic_key: str) -> dict[str, float | int] | None:
        samples = [
            (int(item["generation"]), float(item["diagnostics"][diagnostic_key]))
            for item in windows
            if item["diagnostics"].get(diagnostic_key) is not None
        ]
        if not samples:
            return None
        values = np.asarray([value for _generation, value in samples], dtype=np.float64)
        maximum_index = int(np.argmax(values))
        return {
            "mean": float(np.mean(values)),
            "p95": float(np.percentile(values, 95)),
            "maximum": float(values[maximum_index]),
            "maximum_generation": samples[maximum_index][0],
        }

    return (
        {
            "phase": controller.spec.phase,
            "physical_mode": controller.spec.mode,
            "reached_terminal": reached,
            "accepted_windows": accepted,
            "rejected_windows": rejected,
            "stopped_at_certified_horizon": stopped_at_certified_horizon,
            "install_rejection": install_rejection,
            "first_rejection": next(
                (item for item in windows if not bool(item["feasible"])),
                None,
            ),
            "maximum_start_error_from_frozen_route_rad": float(
                np.max(np.abs(start_model - route_q[0]))
            ),
            "simulated_tracking_offset_rad": tracking_offset.tolist(),
            "maximum_terminal_error_rad": terminal_joint_error,
            "terminal_translation_error_m": terminal_translation_error_m,
            "terminal_rotation_error_rad": terminal_rotation_error_rad,
            "simulated_time_s": simulated_time_s,
            "window_count": len(windows),
            "last_window_route_state": (
                None
                if not windows
                else {
                    key: windows[-1]["diagnostics"].get(key)
                    for key in (
                        "route_progress_index",
                        "route_waypoint_index",
                        "route_waypoint_model_q_rad",
                        "predicted_terminal_model_q_rad",
                    )
                }
            ),
            "solve_latency_s": {
                "mean": float(np.mean(solve_times)) if len(solve_times) else None,
                "p95": (float(np.percentile(solve_times, 95)) if len(solve_times) else None),
                "maximum": float(np.max(solve_times)) if len(solve_times) else None,
            },
            "slowest_window": (
                None
                if slowest_window is None
                else {
                    "generation": slowest_window["generation"],
                    "total_s": slowest_window["solve_time_s"],
                    "goal_update_s": slowest_window["diagnostics"].get("goal_update_time_s"),
                    "mpc_core_s": slowest_window["diagnostics"].get("mpc_core_window_time_s"),
                    "optimizer_s": slowest_window["diagnostics"].get("optimizer_wall_time_s"),
                    "strict_validation_s": slowest_window["diagnostics"].get(
                        "strict_validation_time_s"
                    ),
                }
            ),
            "latency_components_s": {
                key: latency_summary(key)
                for key in (
                    "goal_update_time_s",
                    "optimizer_wall_time_s",
                    "strict_validation_time_s",
                    "mpc_core_window_time_s",
                    "worker_total_window_time_s",
                )
            },
            "fixture_clearance": (
                None
                if closest_fixture_window is None
                else {
                    "minimum_m": closest_fixture_window["diagnostics"][
                        "minimum_fixture_clearance_m"
                    ],
                    "link": closest_fixture_window["diagnostics"][
                        "minimum_fixture_clearance_link"
                    ],
                    "generation": closest_fixture_window["generation"],
                    "search_distance_m": closest_fixture_window["diagnostics"][
                        "fixture_clearance_search_distance_m"
                    ],
                }
            ),
            "maximum_window_velocity_rad_s": (
                max(MPCCommandWindow.from_dict(item).peak_velocity_rad_s() for item in windows)
                if windows
                else None
            ),
        },
        measured_q,
        dq,
    )


def benchmark_tabletop_lifecycle_mpc(
    loaded_request: TabletopTaskRequest,
    clearance_request: TabletopTaskRequest,
    execution: TabletopExecutionPlan,
    *,
    config: MPCBenchmarkConfig | None = None,
    measured_active_dex3_q_rad: np.ndarray | None = None,
    reference_T_camera: np.ndarray | None = None,
) -> dict[str, Any]:
    """Replay every normal phase with its exact physical MPC model, command-free."""

    import torch

    config = config or MPCBenchmarkConfig()
    if loaded_request.content_sha256 != execution.loaded_request_sha256:
        raise ValueError("lifecycle benchmark loaded request differs from the plan")
    if clearance_request.content_sha256 != execution.clearance_request_sha256:
        raise ValueError("lifecycle benchmark clearance request differs from the plan")
    started = time.perf_counter()
    if reference_T_camera is not None:
        reference_T_camera = _rigid_transform(reference_T_camera)
    controller: TabletopPhaseMPC | None = None
    phase_results: list[dict[str, Any]] = []
    setup_events: list[dict[str, Any]] = []
    first_phase = MPC_PHASE_ORDER[0]
    first_route = next(
        trajectory for trajectory in execution.trajectories if trajectory.to_pose_id == first_phase
    )
    command_q = np.asarray(first_route.command_q_rad[0], dtype=np.float64)
    dq = np.zeros(7, dtype=np.float64)
    if measured_active_dex3_q_rad is None:
        contact_fingers = np.asarray(
            execution.task.close_target_active_dex3_q_rad,
            dtype=np.float64,
        )
        measured_close_source = "descriptor_target"
    else:
        contact_fingers = np.asarray(measured_active_dex3_q_rad, dtype=np.float64).reshape(-1)
        if contact_fingers.shape != (7,) or not np.all(np.isfinite(contact_fingers)):
            raise ValueError("measured MPC close posture must contain seven finite values")
        measured_close_source = "retained_hardware_grasp_close"
    try:
        for phase in MPC_PHASE_ORDER:
            measured = contact_fingers if phase in MPC_ATTACHED_PHASES else None
            reused = controller is not None
            switch = None
            if controller is None:
                build_started = time.perf_counter()
                controller = TabletopPhaseMPC(
                    clearance_request,
                    execution,
                    phase=phase,
                    loaded_request=loaded_request,
                    measured_active_dex3_q_rad=measured,
                )
                build_s = time.perf_counter() - build_started
                setup_s = controller.setup_at_frozen_route_start()
                preparation_s = build_s + setup_s
            else:
                switch = controller.select_phase(
                    phase,
                    measured_active_dex3_q_rad=measured,
                )
                build_s = 0.0
                setup_s = 0.0
                preparation_s = switch["reconfiguration_time_s"]
            setup_events.append(
                {
                    "phase": phase,
                    "physical_mode": controller.spec.mode,
                    "reused_warm_model": reused,
                    "build_time_s": build_s,
                    "preparation_time_s": preparation_s,
                    "setup_time_s": setup_s,
                    "kinematics_cache_hit": (
                        True if switch is None else switch["kinematics_cache_hit"]
                    ),
                    "kinematics_resolve_time_s": (
                        0.0 if switch is None else switch["kinematics_resolve_time_s"]
                    ),
                    "optimizer_prewarm_time_s": (
                        0.0 if switch is None else switch["optimizer_prewarm_time_s"]
                    ),
                    "state_correction_prewarm_time_s": (
                        controller._last_state_correction_prewarm_s
                        if switch is None
                        else switch["state_correction_prewarm_time_s"]
                    ),
                    "reconfiguration_time_s": (
                        0.0 if switch is None else switch["reconfiguration_time_s"]
                    ),
                }
            )
            phase_result, command_q, dq = _simulate_phase(
                controller,
                command_q_rad=command_q,
                model_dq_rad_s=dq,
                config=config,
                reference_T_camera=reference_T_camera,
            )
            phase_results.append(phase_result)
            if not phase_result["reached_terminal"]:
                break
    finally:
        if controller is not None:
            controller.close()
        torch.cuda.empty_cache()
    return {
        "schema_version": 1,
        "kind": "g1_tabletop_phase_aware_mpc_lifecycle_benchmark",
        "commands_robot": False,
        "loaded_request_sha256": loaded_request.content_sha256,
        "clearance_request_sha256": clearance_request.content_sha256,
        "execution_plan_sha256": execution.content_sha256,
        "arm": execution.task.arm,
        "measured_close_source": measured_close_source,
        "active_dex3_close_q_rad": contact_fingers.tolist(),
        "camera_state_correction": {
            "enabled": reference_T_camera is not None,
            "reference_T_camera": (
                None if reference_T_camera is None else reference_T_camera.tolist()
            ),
        },
        "complete": (
            len(phase_results) == len(MPC_PHASE_ORDER)
            and all(bool(item["reached_terminal"]) for item in phase_results)
        ),
        "completed_phase_count": sum(bool(item["reached_terminal"]) for item in phase_results),
        "phase_count": len(MPC_PHASE_ORDER),
        "setup_events": setup_events,
        "phases": phase_results,
        "total_accepted_windows": sum(item["accepted_windows"] for item in phase_results),
        "total_rejected_windows": sum(item["rejected_windows"] for item in phase_results),
        "benchmark_wall_time_s": time.perf_counter() - started,
        "configuration": {
            "command_dt_s": MPC_COMMAND_DT_S,
            "knot_dt_s": MPC_KNOT_DT_S,
            "interpolation_steps": MPC_INTERPOLATION_STEPS,
            "certified_horizon_source": "complete_curobo_robot_state_sequence",
            "cold_start_iterations": MPC_COLD_START_ITERATIONS,
            "warm_start_iterations": MPC_WARM_START_ITERATIONS,
            "maximum_steps_per_phase": config.maximum_steps,
            "waypoint_tolerance_rad": config.waypoint_tolerance_rad,
            "handoff_interval_s": config.handoff_interval_s,
            "simulated_tracking_offset_rad": (
                None
                if config.simulated_tracking_offset_rad is None
                else list(config.simulated_tracking_offset_rad)
            ),
            "maximum_arm_velocity_rad_s": clearance_request.maximum_arm_velocity_rad_s,
            "route_tracking": "monotonic_frozen_route_lookahead",
            "route_lookahead_rad": controller._lookahead_rad,
            "collision_activation_distance_m": COLLISION_ACTIVATION_DISTANCE_M,
            "open_transit_object_clearance_m": OPEN_TRANSIT_OBJECT_CLEARANCE_M,
        },
        "provenance": {**model_source_hashes(), "curobo_commit": CUROBO_COMMIT},
    }


def benchmark_from_paths(
    loaded_request_path: Path,
    clearance_request_path: Path,
    execution_path: Path,
    output_path: Path,
    *,
    maximum_steps: int,
    grasp_close_path: Path | None = None,
    camera_state_estimate_path: Path | None = None,
) -> dict[str, Any]:
    from g1_dex3_tabletop.planning.contracts import atomic_write_json

    loaded_request = TabletopTaskRequest.from_json(loaded_request_path)
    clearance_request = TabletopTaskRequest.from_json(clearance_request_path)
    execution = TabletopExecutionPlan.from_json(execution_path)
    measured_close = None
    grasp_close_provenance = None
    if grasp_close_path is not None:
        raw = grasp_close_path.read_bytes()
        document = json.loads(raw)
        if document.get("active_side") != execution.task.arm:
            raise ValueError("retained grasp close selects a different arm")
        measured_close = np.asarray(document.get("close_q_rad"), dtype=np.float64)
        grasp_close_provenance = {
            "path": str(grasp_close_path.resolve()),
            "sha256": hashlib.sha256(raw).hexdigest(),
        }
    reference_T_camera = None
    camera_state_provenance = None
    if camera_state_estimate_path is not None:
        raw = camera_state_estimate_path.read_bytes()
        document = json.loads(raw)
        estimate = document.get(
            "installation_estimate",
            document.get("estimate", document),
        )
        if not isinstance(estimate, dict):
            raise ValueError("camera-state estimate JSON has no estimate object")
        reference_T_camera = _rigid_transform(estimate.get("reference_T_camera"))
        camera_state_provenance = {
            "path": str(camera_state_estimate_path.resolve()),
            "sha256": hashlib.sha256(raw).hexdigest(),
            "timestamp_ns": estimate.get("timestamp_ns"),
            "anchor_timestamp_ns": estimate.get("anchor_timestamp_ns"),
        }
    result = benchmark_tabletop_lifecycle_mpc(
        loaded_request,
        clearance_request,
        execution,
        config=MPCBenchmarkConfig(maximum_steps=maximum_steps),
        measured_active_dex3_q_rad=measured_close,
        reference_T_camera=reference_T_camera,
    )
    result["grasp_close_provenance"] = grasp_close_provenance
    result["camera_state_estimate_provenance"] = camera_state_provenance
    atomic_write_json(output_path, result)
    return result
