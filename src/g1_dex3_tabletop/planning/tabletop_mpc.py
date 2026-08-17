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
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation

from g1_aprilcube_calibration.joint_map import arm_joint_names
from g1_aprilcube_calibration.transforms import invert_transform, validate_transform
from g1_aprilcube_calibration.transports.unitree_dex3 import DEX3_MOTOR_JOINT_SUFFIXES
from g1_dex3_tabletop.mpc_command_buffer import MPCCommandWindow
from g1_dex3_tabletop.planning.curobo_backend import COLLISION_ACTIVATION_DISTANCE_M
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
    _contact_links,
    _cuboid_cover_spheres,
    _fixture_clearance_from_spheres,
    _fixture_collision_mesh,
    _local_plane_clearance_from_spheres,
    _selected_open_transit_world_robot,
    _table_from_resting_object,
    _use_moving_grasp_frame_only,
    _world_cuboid_clearances,
)
from g1_dex3_tabletop.tabletop_contracts import TabletopExecutionPlan, TabletopTaskRequest

MPC_OPTIMIZATION_DT_S = 0.04
MPC_INTERPOLATION_STEPS = 4
MPC_DOCUMENTED_COMMAND_DT_S = MPC_OPTIMIZATION_DT_S / MPC_INTERPOLATION_STEPS
MPC_COLD_START_ITERATIONS = 200
MPC_WARM_START_ITERATIONS = 100
MPC_EXPOSED_INTERPOLATION_WINDOWS = 3
MPC_PHASE_ORDER = (
    "clearance",
    "move_to_pregrasp",
    "grasp_approach",
    "retention_test_lift",
    "payload_lift",
    "payload_lower",
    "payload_replace",
    "grasp_retreat",
    "return_to_clearance",
    "__handoff__",
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
    allow_fingertip_cube_contact: bool
    attached_payload: bool


def mpc_phase_spec(phase: str) -> MPCPhaseSpec:
    """Map a motion endpoint to its already-commissioned physical state."""

    if phase not in MPC_PHASE_ORDER:
        raise ValueError(f"unsupported tabletop MPC phase: {phase}")
    if phase in ("clearance", "__handoff__"):
        return MPCPhaseSpec(
            phase=phase,
            mode="supported",
            request_state="loaded",
            finger_state="initial",
            include_cube_in_optimizer=True,
            include_table_patch=False,
            allow_fingertip_cube_contact=False,
            attached_payload=False,
        )
    if phase in ("move_to_pregrasp", "return_to_clearance"):
        return MPCPhaseSpec(
            phase=phase,
            mode="open_free",
            request_state="clearance",
            finger_state="open",
            include_cube_in_optimizer=True,
            include_table_patch=True,
            allow_fingertip_cube_contact=False,
            attached_payload=False,
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
            allow_fingertip_cube_contact=True,
            attached_payload=False,
        )
    return MPCPhaseSpec(
        phase=phase,
        mode="attached",
        request_state="clearance",
        finger_state="measured_contact",
        include_cube_in_optimizer=False,
        include_table_patch=False,
        allow_fingertip_cube_contact=False,
        attached_payload=True,
    )


def _numpy(value) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=np.float64)


def _joint_state(device_cfg, q: np.ndarray, dq: np.ndarray, names: tuple[str, ...]):
    import torch
    from curobo.types import JointState

    state = JointState.from_position(
        device_cfg.to_device(np.asarray(q, dtype=np.float64)).unsqueeze(0),
        joint_names=list(names),
    )
    state.velocity = device_cfg.to_device(np.asarray(dq, dtype=np.float64)).unsqueeze(0)
    state.acceleration = torch.zeros_like(state.position)
    return state


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
    replan_lead_s: float = 0.1

    def __post_init__(self) -> None:
        if self.maximum_steps <= 0:
            raise ValueError("MPC benchmark step count must be positive")
        if not np.isfinite(self.waypoint_tolerance_rad) or self.waypoint_tolerance_rad <= 0.0:
            raise ValueError("MPC waypoint tolerance must be positive and finite")
        if not np.isfinite(self.replan_lead_s) or self.replan_lead_s <= 0.0:
            raise ValueError("MPC replan lead must be positive and finite")


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
            snapshot=clearance_request.observation.snapshot,
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
            maximum_velocity_rad_s=clearance_request.maximum_arm_velocity_rad_s,
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
        self._fixture_mesh = _fixture_collision_mesh(
            clearance_request,
            base_T_object,
            self._down,
        )
        self._cube_scene = _base_scene(
            clearance_request,
            base_T_torso,
            include_cube=True,
            include_open_transit_table_patch=False,
        )
        self._base_T_torso0 = _rigid_transform(base_T_torso)
        self._reference_T_torso0 = _rigid_transform(
            invert_transform(
                np.asarray(clearance_request.observation.camera_T_object, dtype=np.float64)
            )
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
        cfg = ModelPredictiveControlCfg.create(
            robot=resolved_robot,
            scene_model=scene,
            collision_cache={"cuboid": 4, "mesh": 1},
            device_cfg=self.device_cfg,
            use_cuda_graph=True,
            optimization_dt=MPC_OPTIMIZATION_DT_S,
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
            execution.task.open_active_dex3_q_rad,
            label="open active Dex3 posture",
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
        execution_manager = self.mpc.trajectory_execution_manager
        execution_manager.command_end_idx = (
            execution_manager.command_start_idx
            + MPC_EXPOSED_INTERPOLATION_WINDOWS * MPC_INTERPOLATION_STEPS
        )
        effective = _numpy(self.mpc.kinematics.get_joint_limits().velocity[1])
        if not np.allclose(
            effective,
            clearance_request.maximum_arm_velocity_rad_s,
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
        self._generation = 0
        self._route_progress_index = 0
        self._active_goal_pose: np.ndarray | None = None
        self._active_goal_is_corrected = False
        self._lookahead_rad = (
            clearance_request.maximum_arm_velocity_rad_s
            * self.mpc.action_horizon
            * MPC_OPTIMIZATION_DT_S
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
            snapshot=self.clearance_request.observation.snapshot,
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
                self.execution.task.open_active_dex3_q_rad,
                label="open active Dex3 posture",
            )
        if measured_active_dex3_q_rad is None:
            raise ValueError(f"MPC phase {spec.phase} requires measured contact fingers")
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
        self._route_progress_index = 0
        self._active_goal_pose = None
        self._active_goal_is_corrected = False
        prewarm_s = 0.0
        if reset_optimizer and self._setup and geometry_changed:
            route_start = self.path_model_q[0]
            cached_seed = self._geometry_action_seed_cache.get(geometry)
            if cached_seed is None:
                route_start_state = _joint_state(
                    self.device_cfg,
                    route_start,
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
            strict = self._strict_window_diagnostics(np.repeat(route_start[None, :], 2, axis=0))
            if not bool(strict["strict_valid"]):
                raise RuntimeError(
                    f"frozen {phase} start fails strict MPC validation after phase switch: "
                    f"{strict['strict_failure']}"
                )
        torch.cuda.synchronize()
        return {
            "phase": phase,
            "physical_mode": spec.mode,
            "kinematics_cache_hit": kinematics_cache_hit,
            "kinematics_resolve_time_s": resolve_s,
            "optimizer_prewarm_time_s": prewarm_s,
            "reconfiguration_time_s": time.perf_counter() - started,
        }

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

    def _strict_window_diagnostics(self, model_q_rad: np.ndarray) -> dict[str, Any]:
        """Reapply the frozen planner's strict checks to one returned window."""

        values = np.asarray(model_q_rad, dtype=np.float64)
        if values.ndim != 2 or values.shape[1] != 7 or not np.all(np.isfinite(values)):
            raise ValueError("strict MPC validation requires a finite N x 7 route")
        diagnostics: dict[str, Any] = {
            "strict_valid": True,
            "strict_sample_count": len(values),
            "strict_failure": None,
        }

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

        spheres = sphere_tensor.detach().cpu().numpy().reshape(len(values), -1, 4)
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
                        "strict_failure": "cube_collision_or_activation_distance",
                        "strict_failure_sample": sample_index,
                        "strict_failure_links": list(pair),
                        "strict_failure_clearance_m": clearance,
                    }
                )
                return diagnostics

        if self._fixture_mesh is not None:
            fixture_spheres = spheres
            if self.spec.allow_fingertip_cube_contact:
                # The frozen linear-contact planner applies the same exception
                # to these three links while the fingers enter the grasp.
                fixture_spheres = spheres.copy()
                for link_name in _contact_links(self.arm):
                    indices = (
                        config.get_sphere_index_from_link_name(link_name)
                        .detach()
                        .cpu()
                        .numpy()
                        .reshape(-1)
                    )
                    fixture_spheres[:, indices, 3] = -100.0
            fixture_clearance, fixture_link, fixture_sample = _fixture_clearance_from_spheres(
                fixture_spheres,
                config=config,
                fixture_mesh=self._fixture_mesh,
            )
            diagnostics.update(
                {
                    "minimum_fixture_clearance_m": float(fixture_clearance),
                    "minimum_fixture_clearance_link": fixture_link,
                    "minimum_fixture_clearance_sample": fixture_sample,
                }
            )
            if fixture_clearance < 0.0:
                diagnostics.update(
                    {
                        "strict_valid": False,
                        "strict_failure": "fixture_collision",
                        "strict_failure_sample": fixture_sample,
                        "strict_failure_links": [fixture_link, self.request.fixture.fixture_id],
                    }
                )
        return diagnostics

    def setup(self, *, model_q_rad: np.ndarray, model_dq_rad_s: np.ndarray) -> float:
        state = _joint_state(self.device_cfg, model_q_rad, model_dq_rad_s, self.names)
        started = time.perf_counter()
        self.mpc.setup(state)
        # ``setup`` captures/warmups CUDA and then deliberately resets CuRobo's
        # action buffer, which would otherwise make the first live request pay
        # for another 200-iteration cold solve.  Complete that solve now, while
        # no LowState freshness timestamp exists.
        self.mpc.optimize_action_sequence(state)
        import torch

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
        goal_state = _joint_state(self.device_cfg, q, np.zeros(7), self.names)
        goal_matrix = self.tool_pose(q)
        goal_pose = Pose.from_matrix(self.device_cfg.to_device(goal_matrix[None]))
        goals = GoalToolPose.from_poses(
            {grasp_frame(self.arm): goal_pose},
            ordered_tool_frames=[grasp_frame(self.arm)],
            num_goalset=1,
        )
        if not self.mpc.update_goal_tool_poses(goals, run_ik=False):
            raise RuntimeError("CuRobo MPC rejected the nominal tool-pose goal")
        self.mpc.update_goal_state(goal_state)
        self.mpc.enable_joint_position_tracking()
        self._active_goal_pose = goal_matrix
        self._active_goal_is_corrected = False

    def update_anchored_goal(
        self,
        model_q_rad: np.ndarray,
        *,
        reference_T_camera: np.ndarray,
    ) -> dict[str, float]:
        """Move the local goal and scene with the estimated live torso pose."""

        from curobo.types import GoalToolPose, Pose

        reference_T_camera = validate_transform(np.asarray(reference_T_camera, dtype=np.float64))
        reference_T_torso = _rigid_transform(
            reference_T_camera
            @ invert_transform(np.asarray(self.request.torso_T_camera, dtype=np.float64))
        )
        nominal_base_T_goal = self.tool_pose(model_q_rad)
        reference_T_goal = _rigid_transform(
            self._reference_T_torso0 @ invert_transform(self._base_T_torso0) @ nominal_base_T_goal
        )
        corrected_base_T_goal = _rigid_transform(
            self._base_T_torso0 @ invert_transform(reference_T_torso) @ reference_T_goal
        )
        for name, reference_T_obstacle in self._reference_T_obstacles.items():
            corrected_base_T_obstacle = _rigid_transform(
                self._base_T_torso0 @ invert_transform(reference_T_torso) @ reference_T_obstacle
            )
            self.mpc.scene_collision_checker.update_obstacle_pose(
                name,
                Pose.from_matrix(self.device_cfg.to_device(corrected_base_T_obstacle[None])),
            )
        goal_pose = Pose.from_matrix(self.device_cfg.to_device(corrected_base_T_goal[None]))
        goals = GoalToolPose.from_poses(
            {grasp_frame(self.arm): goal_pose},
            ordered_tool_frames=[grasp_frame(self.arm)],
            num_goalset=1,
        )
        if not self.mpc.update_goal_tool_poses(goals, run_ik=True):
            raise RuntimeError("CuRobo MPC could not solve the state-corrected local goal")
        self._active_goal_pose = corrected_base_T_goal
        self._active_goal_is_corrected = True
        correction = corrected_base_T_goal @ invert_transform(nominal_base_T_goal)
        return {
            "goal_translation_correction_m": float(np.linalg.norm(correction[:3, 3])),
            "goal_rotation_correction_deg": float(
                np.degrees(Rotation.from_matrix(correction[:3, :3]).magnitude())
            ),
        }

    def next_nominal_window(
        self,
        *,
        measured_command_q_rad: np.ndarray,
        measured_dq_rad_s: np.ndarray,
        active_command_q_rad: np.ndarray,
        state_monotonic_s: float,
        reference_T_camera: np.ndarray | None = None,
    ) -> MPCCommandWindow:
        """Advance along the current frozen phase by one checked MPC horizon."""

        measured_command = np.asarray(measured_command_q_rad, dtype=np.float64).reshape(-1)
        if measured_command.shape != (7,) or not np.all(np.isfinite(measured_command)):
            raise ValueError("measured MPC arm position must contain seven finite values")
        model_q = np.asarray(
            [
                value + self.request.joint_position_offsets_rad.get(name, 0.0)
                for name, value in zip(self.names, measured_command, strict=True)
            ],
            dtype=np.float64,
        )
        remaining = self.path_model_q[self._route_progress_index :]
        self._route_progress_index += int(
            np.argmin(np.max(np.abs(remaining - model_q[None, :]), axis=1))
        )
        waypoint_index = len(self.path_model_q) - 1
        for index in range(self._route_progress_index + 1, len(self.path_model_q)):
            if float(np.max(np.abs(self.path_model_q[index] - model_q))) >= (self._lookahead_rad):
                waypoint_index = index
                break
        goal_q = self.path_model_q[waypoint_index]
        state_correction: dict[str, float] = {}
        if reference_T_camera is None:
            self.update_nominal_goal(goal_q)
        else:
            state_correction = self.update_anchored_goal(
                goal_q,
                reference_T_camera=reference_T_camera,
            )
        requested_terminal = waypoint_index == len(self.path_model_q) - 1
        window = self.solve_window(
            model_q_rad=model_q,
            model_dq_rad_s=np.asarray(measured_dq_rad_s, dtype=np.float64),
            active_command_q_rad=np.asarray(active_command_q_rad, dtype=np.float64),
            state_monotonic_s=state_monotonic_s,
            terminal=requested_terminal,
        )
        if requested_terminal:
            terminal_command = np.asarray(window.command_q_rad[-1], dtype=np.float64)
            terminal_model = np.asarray(
                [
                    value + self.request.joint_position_offsets_rad.get(name, 0.0)
                    for name, value in zip(self.names, terminal_command, strict=True)
                ]
            )
            if self._active_goal_is_corrected:
                if self._active_goal_pose is None:
                    raise RuntimeError("MPC terminal check has no active tool-pose goal")
                terminal_pose = self.tool_pose(terminal_model)
                translation_error = float(
                    np.linalg.norm(terminal_pose[:3, 3] - self._active_goal_pose[:3, 3])
                )
                rotation_error = float(
                    Rotation.from_matrix(
                        terminal_pose[:3, :3].T @ self._active_goal_pose[:3, :3]
                    ).magnitude()
                )
                terminal = translation_error <= 0.005 and rotation_error <= 0.05
            else:
                terminal = float(np.max(np.abs(terminal_model - self.path_model_q[-1]))) <= 0.005
                translation_error = None
                rotation_error = None
            if terminal != window.terminal:
                values = window.to_dict(include_hash=False)
                values["terminal"] = terminal and window.feasible
                values["diagnostics"] = {
                    **window.diagnostics,
                    "terminal_translation_error_m": translation_error,
                    "terminal_rotation_error_rad": rotation_error,
                }
                window = MPCCommandWindow.from_dict(values)
        if state_correction:
            values = window.to_dict(include_hash=False)
            values["diagnostics"] = {**window.diagnostics, **state_correction}
            window = MPCCommandWindow.from_dict(values)
        return window

    def solve_window(
        self,
        *,
        model_q_rad: np.ndarray,
        model_dq_rad_s: np.ndarray,
        active_command_q_rad: np.ndarray,
        state_monotonic_s: float,
        terminal: bool,
    ) -> MPCCommandWindow:
        """Optimize one window and reject, rather than expose, infeasible output."""

        import torch

        if not self._setup:
            raise RuntimeError("MPC must be set up before solving")
        current = _joint_state(self.device_cfg, model_q_rad, model_dq_rad_s, self.names)
        started = time.perf_counter()
        result = self.mpc.optimize_action_sequence(current)
        torch.cuda.synchronize()
        optimizer_wall_s = time.perf_counter() - started
        sequence = result.action_sequence
        if sequence is None:
            model_commands = np.asarray(model_q_rad, dtype=np.float64)[None, :]
            returned_state_dt_s = MPC_OPTIMIZATION_DT_S
        else:
            model_commands = _numpy(sequence.position).reshape(-1, 7)
            returned_dt = _numpy(sequence.dt).reshape(-1)
            if len(returned_dt) == 0 or not np.all(np.isfinite(returned_dt)):
                raise RuntimeError("CuRobo MPC returned no finite JointState dt")
            if not np.allclose(returned_dt, returned_dt[0], atol=1.0e-9, rtol=0.0):
                raise RuntimeError(
                    f"CuRobo MPC returned nonuniform JointState dt {returned_dt.tolist()}"
                )
            returned_state_dt_s = float(returned_dt[0])
            if returned_state_dt_s <= 0.0:
                raise RuntimeError("CuRobo MPC returned a non-positive JointState dt")
        command_start = np.asarray(active_command_q_rad, dtype=np.float64).reshape(-1)
        if command_start.shape != (7,) or not np.all(np.isfinite(command_start)):
            raise ValueError("active MPC command must contain seven finite values")
        command_sequence = np.stack(
            [
                command_from_model_q(
                    row,
                    arm=self.arm,
                    joint_position_offsets_rad=self.request.joint_position_offsets_rad,
                )
                for row in model_commands
            ]
        )
        commands = np.concatenate((command_start[None, :], command_sequence), axis=0)
        first_command_index = self.mpc.trajectory_execution_manager.command_start_idx
        times = np.concatenate(
            (
                np.asarray([0.0], dtype=np.float64),
                (first_command_index + np.arange(len(command_sequence), dtype=np.float64))
                * returned_state_dt_s,
            )
        )
        feasible = bool(result.success is not None and bool(result.success.reshape(-1)[0].item()))
        curobo_feasible = feasible
        peak_velocity = float(np.max(np.abs(np.diff(commands, axis=0)) / np.diff(times)[:, None]))
        if peak_velocity > self.request.maximum_arm_velocity_rad_s + 1.0e-6:
            feasible = False
        diagnostics = {
            "phase": self.spec.phase,
            "physical_mode": self.spec.mode,
            "curobo_feasible": curobo_feasible,
            "optimizer_wall_time_s": optimizer_wall_s,
            "curobo_reported_solve_time_s": float(result.solve_time),
            "peak_velocity_rad_s": peak_velocity,
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
        strict_diagnostics = self._strict_window_diagnostics(strict_model_commands)
        strict_validation_s = time.perf_counter() - strict_started
        diagnostics.update(strict_diagnostics)
        diagnostics["strict_validation_time_s"] = strict_validation_s
        if not bool(strict_diagnostics["strict_valid"]):
            feasible = False
        wall_s = time.perf_counter() - started
        diagnostics["worker_wall_time_s"] = wall_s
        window = MPCCommandWindow(
            generation=self._generation,
            plan_sha256=self.execution.content_sha256,
            state_monotonic_s=state_monotonic_s,
            sample_time_s=tuple(times),
            command_q_rad=tuple(tuple(float(value) for value in row) for row in commands),
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
    command_q = command_from_model_q(
        q,
        arm=request.arm,
        joint_position_offsets_rad=request.joint_position_offsets_rad,
    )
    setup_s = controller.setup(model_q_rad=q, model_dq_rad_s=dq)
    lookahead_rad = controller._lookahead_rad
    accepted = 0
    rejected = 0
    latencies: list[float] = []
    peak_velocities: list[float] = []
    first_rejection: dict[str, Any] | None = None
    simulated_time_s = 0.0
    try:
        for _step in range(config.maximum_steps):
            window = controller.next_nominal_window(
                measured_command_q_rad=command_q,
                measured_dq_rad_s=dq,
                active_command_q_rad=command_q,
                state_monotonic_s=time.monotonic(),
            )
            latencies.append(window.solve_time_s)
            peak_velocities.append(window.peak_velocity_rad_s())
            if not window.feasible:
                rejected += 1
                if first_rejection is None:
                    first_rejection = window.to_dict()
                break
            accepted += 1
            execution_time_s = (
                window.duration_s
                if window.terminal
                else max(window.duration_s - config.replan_lead_s, 0.0)
            )
            window_times = np.asarray(window.sample_time_s, dtype=np.float64)
            window_commands = np.asarray(window.command_q_rad, dtype=np.float64)
            command_q = np.asarray(
                [
                    np.interp(execution_time_s, window_times, window_commands[:, index])
                    for index in range(7)
                ],
                dtype=np.float64,
            )
            q = np.asarray(
                [
                    value + request.joint_position_offsets_rad.get(name, 0.0)
                    for name, value in zip(controller.names, command_q, strict=True)
                ],
                dtype=np.float64,
            )
            upper = int(np.searchsorted(window_times, execution_time_s, side="right"))
            upper = min(max(upper, 1), len(window_times) - 1)
            lower = upper - 1
            dq = (window_commands[upper] - window_commands[lower]) / (
                window_times[upper] - window_times[lower]
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
            "optimization_dt_s": MPC_OPTIMIZATION_DT_S,
            "interpolation_steps": MPC_INTERPOLATION_STEPS,
            "exposed_interpolation_windows": MPC_EXPOSED_INTERPOLATION_WINDOWS,
            "documented_command_dt_s": MPC_DOCUMENTED_COMMAND_DT_S,
            "cold_start_iterations": MPC_COLD_START_ITERATIONS,
            "warm_start_iterations": MPC_WARM_START_ITERATIONS,
            "maximum_steps": config.maximum_steps,
            "waypoint_tolerance_rad": config.waypoint_tolerance_rad,
            "replan_lead_s": config.replan_lead_s,
            "maximum_arm_velocity_rad_s": request.maximum_arm_velocity_rad_s,
            "route_lookahead_rad": lookahead_rad,
            "collision_activation_distance_m": COLLISION_ACTIVATION_DISTANCE_M,
        },
        "provenance": {**model_source_hashes(), "curobo_commit": CUROBO_COMMIT},
    }


def _simulate_phase(
    controller: TabletopPhaseMPC,
    *,
    command_q_rad: np.ndarray,
    model_dq_rad_s: np.ndarray,
    config: MPCBenchmarkConfig,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    """Replay one phase without ROS or a robot and return its terminal state."""

    command_q = np.asarray(command_q_rad, dtype=np.float64).copy()
    dq = np.asarray(model_dq_rad_s, dtype=np.float64).copy()
    route_q = controller.path_model_q
    start_model = np.asarray(
        [
            value + controller.request.joint_position_offsets_rad.get(name, 0.0)
            for name, value in zip(controller.names, command_q, strict=True)
        ],
        dtype=np.float64,
    )
    accepted = 0
    rejected = 0
    windows: list[dict[str, Any]] = []
    simulated_time_s = 0.0
    for _step in range(config.maximum_steps):
        window = controller.next_nominal_window(
            measured_command_q_rad=command_q,
            measured_dq_rad_s=dq,
            active_command_q_rad=command_q,
            state_monotonic_s=time.monotonic(),
        )
        windows.append(window.to_dict())
        if not window.feasible:
            rejected += 1
            break
        accepted += 1
        execution_time_s = (
            window.duration_s
            if window.terminal
            else max(window.duration_s - config.replan_lead_s, 0.0)
        )
        window_times = np.asarray(window.sample_time_s, dtype=np.float64)
        window_commands = np.asarray(window.command_q_rad, dtype=np.float64)
        command_q = np.asarray(
            [
                np.interp(execution_time_s, window_times, window_commands[:, index])
                for index in range(7)
            ],
            dtype=np.float64,
        )
        upper = int(np.searchsorted(window_times, execution_time_s, side="right"))
        upper = min(max(upper, 1), len(window_times) - 1)
        lower = upper - 1
        dq = (window_commands[upper] - window_commands[lower]) / (
            window_times[upper] - window_times[lower]
        )
        simulated_time_s += execution_time_s
        if window.terminal:
            dq = np.zeros(7, dtype=np.float64)
            break
    terminal_model = np.asarray(
        [
            value + controller.request.joint_position_offsets_rad.get(name, 0.0)
            for name, value in zip(controller.names, command_q, strict=True)
        ],
        dtype=np.float64,
    )
    reached = float(np.max(np.abs(terminal_model - route_q[-1]))) <= config.waypoint_tolerance_rad
    solve_times = np.asarray([item["solve_time_s"] for item in windows], dtype=np.float64)
    return (
        {
            "phase": controller.spec.phase,
            "physical_mode": controller.spec.mode,
            "reached_terminal": reached,
            "accepted_windows": accepted,
            "rejected_windows": rejected,
            "first_rejection": next(
                (item for item in windows if not bool(item["feasible"])),
                None,
            ),
            "maximum_start_error_from_frozen_route_rad": float(
                np.max(np.abs(start_model - route_q[0]))
            ),
            "maximum_terminal_error_rad": float(np.max(np.abs(terminal_model - route_q[-1]))),
            "simulated_time_s": simulated_time_s,
            "window_count": len(windows),
            "solve_latency_s": {
                "mean": float(np.mean(solve_times)) if len(solve_times) else None,
                "p95": (float(np.percentile(solve_times, 95)) if len(solve_times) else None),
                "maximum": float(np.max(solve_times)) if len(solve_times) else None,
            },
            "maximum_window_velocity_rad_s": (
                max(MPCCommandWindow.from_dict(item).peak_velocity_rad_s() for item in windows)
                if windows
                else None
            ),
        },
        command_q,
        dq,
    )


def benchmark_tabletop_lifecycle_mpc(
    loaded_request: TabletopTaskRequest,
    clearance_request: TabletopTaskRequest,
    execution: TabletopExecutionPlan,
    *,
    config: MPCBenchmarkConfig | None = None,
) -> dict[str, Any]:
    """Replay every normal phase with its exact physical MPC model, command-free."""

    import torch

    config = config or MPCBenchmarkConfig()
    if loaded_request.content_sha256 != execution.loaded_request_sha256:
        raise ValueError("lifecycle benchmark loaded request differs from the plan")
    if clearance_request.content_sha256 != execution.clearance_request_sha256:
        raise ValueError("lifecycle benchmark clearance request differs from the plan")
    started = time.perf_counter()
    controller: TabletopPhaseMPC | None = None
    phase_results: list[dict[str, Any]] = []
    setup_events: list[dict[str, Any]] = []
    command_q = np.asarray(execution.trajectories[0].command_q_rad[0], dtype=np.float64)
    dq = np.zeros(7, dtype=np.float64)
    contact_fingers = np.asarray(
        execution.task.close_target_active_dex3_q_rad,
        dtype=np.float64,
    )
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
            "optimization_dt_s": MPC_OPTIMIZATION_DT_S,
            "interpolation_steps": MPC_INTERPOLATION_STEPS,
            "exposed_interpolation_windows": MPC_EXPOSED_INTERPOLATION_WINDOWS,
            "cold_start_iterations": MPC_COLD_START_ITERATIONS,
            "warm_start_iterations": MPC_WARM_START_ITERATIONS,
            "maximum_steps_per_phase": config.maximum_steps,
            "waypoint_tolerance_rad": config.waypoint_tolerance_rad,
            "replan_lead_s": config.replan_lead_s,
            "maximum_arm_velocity_rad_s": clearance_request.maximum_arm_velocity_rad_s,
            "collision_activation_distance_m": COLLISION_ACTIVATION_DISTANCE_M,
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
) -> dict[str, Any]:
    from g1_dex3_tabletop.planning.contracts import atomic_write_json

    loaded_request = TabletopTaskRequest.from_json(loaded_request_path)
    clearance_request = TabletopTaskRequest.from_json(clearance_request_path)
    execution = TabletopExecutionPlan.from_json(execution_path)
    result = benchmark_tabletop_lifecycle_mpc(
        loaded_request,
        clearance_request,
        execution,
        config=MPCBenchmarkConfig(maximum_steps=maximum_steps),
    )
    atomic_write_json(output_path, result)
    return result
