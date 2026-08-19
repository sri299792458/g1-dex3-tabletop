"""Native CuRobo batched IK and complete calibration-route planning."""

from __future__ import annotations

import gc
import time
from collections.abc import Callable
from dataclasses import dataclass
from itertools import pairwise
from typing import Any

import numpy as np

from g1_aprilcube_calibration.joint_map import (
    LEFT_ARM_INDICES,
    RIGHT_ARM_INDICES,
    arm_joint_names,
)
from g1_aprilcube_calibration.pose_schema import HANDOFF_POSE_ID
from g1_aprilcube_calibration.transforms import invert_transform
from g1_aprilcube_calibration.transports.unitree_dex3 import (
    DEX3_MOTOR_JOINT_SUFFIXES,
)
from g1_dex3_tabletop.calibration_candidates import (
    CandidateDesignConfig,
    select_information_candidates,
)
from g1_dex3_tabletop.planning.contracts import (
    CalibrationPlanRequest,
    CalibrationPlanResult,
    Dex3PreparationPlan,
    Dex3PreparationRequest,
    PlannedCalibrationPose,
    PlannedTrajectory,
    RobotSnapshot,
)
from g1_dex3_tabletop.planning.g1_model import (
    CUROBO_COMMIT,
    build_locked_robot_config,
    build_robot_config_for_active_joints,
    command_from_model_q,
    model_source_hashes,
    palm_link,
)


def _clearance_from_activation_cost(cost: np.ndarray, activation_distance_m: float) -> np.ndarray:
    """Invert CuRobo's exact smooth collision-activation function."""

    values = np.asarray(cost, dtype=np.float64)
    activation = float(activation_distance_m)
    if not np.isfinite(activation) or activation <= 0.0:
        raise ValueError("collision activation distance must be positive")
    return np.where(
        values <= 0.0,
        activation,
        np.where(
            values <= 0.5 * activation,
            activation - np.sqrt(2.0 * activation * np.maximum(values, 0.0)),
            0.5 * activation - values,
        ),
    )

IK_SEEDS = 16
IK_RETURN_SEEDS = 4
IK_POSITION_TOLERANCE_M = 0.002
IK_ROTATION_TOLERANCE_RAD = 0.02
TRAJECTORY_INTERPOLATION_DT_S = 0.025
EXECUTION_MAXIMUM_VELOCITY_RAD_S = 0.2
COLLISION_ACTIVATION_DISTANCE_M = 0.01
# Keep CuRobo's optimizer activation band separate from the hard open-hand
# object margin. Entering the activation band should shape the optimizer cost;
# it is not itself a physical collision.
OPEN_TRANSIT_OBJECT_CLEARANCE_M = 0.005
FINGER_SWEEP_MAXIMUM_JOINT_STEP_RAD = 0.02


def sample_linear_joint_sweep(
    start: np.ndarray,
    target: np.ndarray,
    *,
    maximum_joint_step_rad: float = FINGER_SWEEP_MAXIMUM_JOINT_STEP_RAD,
) -> np.ndarray:
    """Sample one direct joint-space sweep with a bounded per-sample step."""

    initial = np.asarray(start, dtype=np.float64).reshape(-1)
    final = np.asarray(target, dtype=np.float64).reshape(-1)
    if initial.shape != final.shape or len(initial) == 0:
        raise ValueError("joint sweep endpoints must have the same non-empty shape")
    if not np.all(np.isfinite(initial)) or not np.all(np.isfinite(final)):
        raise ValueError("joint sweep endpoints must be finite")
    if not np.isfinite(maximum_joint_step_rad) or maximum_joint_step_rad <= 0.0:
        raise ValueError("maximum joint sweep step must be positive and finite")
    intervals = max(
        int(np.ceil(np.max(np.abs(final - initial)) / maximum_joint_step_rad)),
        1,
    )
    alpha = np.linspace(0.0, 1.0, intervals + 1, dtype=np.float64)[:, None]
    return initial[None] + alpha * (final - initial)[None]


@dataclass(frozen=True, slots=True)
class _FeasibleIK:
    source_index: int
    model_q: np.ndarray
    position_error_m: float
    rotation_error_rad: float


def plan_dex3_preparation(
    request: Dex3PreparationRequest,
    *,
    progress: Callable[[str], None] | None = None,
) -> Dex3PreparationPlan:
    """Plan the commissioned right-then-left shoulder clearance with CuRobo."""

    report = progress or (lambda _message: None)
    offsets = np.arange(
        request.initial_outward_offset_rad,
        request.maximum_outward_offset_rad + 0.5 * request.outward_search_step_rad,
        request.outward_search_step_rad,
    )
    rejected: list[dict[str, object]] = []
    for index, offset in enumerate(offsets, start=1):
        report(
            f"CuRobo Dex3 preparation candidate {index}/{len(offsets)}: "
            f"outward offset={offset:.4f}rad"
        )
        try:
            right_outbound, after_right = _plan_clearance_arm(
                request=request,
                snapshot=request.snapshot,
                arm="right",
                shoulder_roll_delta=-float(offset),
                from_pose_id=HANDOFF_POSE_ID,
                to_pose_id="right_shoulder_clearance",
            )
            left_outbound, after_dual = _plan_clearance_arm(
                request=request,
                snapshot=after_right,
                arm="left",
                shoulder_roll_delta=float(offset),
                from_pose_id="right_shoulder_clearance",
                to_pose_id="dual_shoulder_clearance",
            )
            sweep_count = _validate_curobo_finger_sweep(
                request=request,
                snapshot=after_dual,
            )
        except (RuntimeError, ValueError) as error:
            message = str(error).lower()
            if isinstance(error, RuntimeError) and any(
                token in message
                for token in (
                    "cuda",
                    "kernel",
                    "device-side",
                    "launch failure",
                    "out of memory",
                )
            ):
                # A CUDA/runtime failure is not a geometric rejection.  The
                # context may be unusable after such a fault, so continuing
                # the offset search would hide the real failure.
                raise
            rejected.append({"outward_offset_rad": float(offset), "reason": str(error)})
            report(f"CuRobo Dex3 preparation rejected {offset:.4f}rad: {error}")
            continue
        final_q = np.asarray(after_dual.measured_q29_rad, dtype=np.float64)
        dual_q14 = tuple(final_q[np.asarray((*LEFT_ARM_INDICES, *RIGHT_ARM_INDICES))])
        return Dex3PreparationPlan(
            request_sha256=request.content_sha256,
            outward_offset_rad=float(offset),
            right_outbound=right_outbound,
            left_outbound=left_outbound,
            left_return=_reverse_trajectory(left_outbound),
            right_return=_reverse_trajectory(right_outbound),
            dual_clearance_q14_rad=dual_q14,
            finger_sweep_sample_count=sweep_count,
            planner_provenance={
                **model_source_hashes(),
                "route_policy": "right_shoulder_then_left_shoulder_then_linear_finger_sweep",
                "execution_maximum_velocity_rad_s": EXECUTION_MAXIMUM_VELOCITY_RAD_S,
                "self_collision_activation_distance_m": (COLLISION_ACTIVATION_DISTANCE_M),
                "finger_sweep_maximum_joint_step_rad": (FINGER_SWEEP_MAXIMUM_JOINT_STEP_RAD),
                "rejected_candidates": rejected,
            },
        )
    detail = " | ".join(
        f"{item['outward_offset_rad']:.4f}: {item['reason']}" for item in rejected[-6:]
    )
    raise RuntimeError(
        "no CuRobo-validated Dex3 shoulder/finger preparation route was found: " + detail
    )


def _plan_clearance_arm(
    *,
    request: Dex3PreparationRequest,
    snapshot: RobotSnapshot,
    arm: str,
    shoulder_roll_delta: float,
    from_pose_id: str,
    to_pose_id: str,
) -> tuple[PlannedTrajectory, RobotSnapshot]:
    import torch
    from curobo.motion_planner import MotionPlanner, MotionPlannerCfg
    from curobo.types import DeviceCfg

    robot, reference_values = build_locked_robot_config(
        arm=arm,
        snapshot=snapshot,
        joint_position_offsets_rad=request.joint_position_offsets_rad,
    )
    reference = np.asarray(reference_values, dtype=np.float64)
    target = reference.copy()
    target[1] += shoulder_roll_delta
    device_cfg = DeviceCfg(device=torch.device("cuda:0"), dtype=torch.float32)
    source_collision = _self_collision_pair_penetrations(
        robot=robot,
        q_samples=reference[None],
        device_cfg=device_cfg,
    )[0]
    if source_collision:
        # A physical Ready posture can begin with a compact Dex3 finger already
        # touching the hip. No collision planner may claim that state is free.
        # The commissioned recovery is the one-joint outward shoulder motion;
        # validate its complete sampled sweep with CuRobo and permit only the
        # collision pairs that already exist at sample zero to monotonically
        # disappear. This exception cannot admit a new collision.
        trajectory, diagnostic = _monotonic_collision_recovery_trajectory(
            robot=robot,
            device_cfg=device_cfg,
            source_q=reference,
            target_q=target,
            arm=arm,
            source_id=from_pose_id,
            target_id=to_pose_id,
            joint_position_offsets_rad=request.joint_position_offsets_rad,
            source_collision=source_collision,
        )
    else:
        planner = MotionPlanner(
            MotionPlannerCfg.create(
                robot=robot,
                device_cfg=device_cfg,
                num_ik_seeds=IK_SEEDS,
                num_trajopt_seeds=4,
                self_collision_check=True,
                use_cuda_graph=True,
                random_seed=request.random_seed,
                optimizer_collision_activation_distance=(COLLISION_ACTIVATION_DISTANCE_M),
                interpolation_dt=TRAJECTORY_INTERPOLATION_DT_S,
                interpolation_buffer_size=1000,
            )
        )
        try:
            trajectory, diagnostic = _plan_edge(
                planner=planner,
                device_cfg=device_cfg,
                names=list(arm_joint_names(arm)),
                arm=arm,
                source_id=from_pose_id,
                target_id=to_pose_id,
                source_q=reference,
                target_q=target,
                joint_position_offsets_rad=request.joint_position_offsets_rad,
            )
        finally:
            planner.destroy()
    if trajectory is None:
        raise RuntimeError(f"{arm} shoulder route unavailable: {diagnostic}")
    updated_q29 = np.asarray(snapshot.measured_q29_rad, dtype=np.float64).copy()
    indices = LEFT_ARM_INDICES if arm == "left" else RIGHT_ARM_INDICES
    updated_q29[np.asarray(indices)] = np.asarray(trajectory.command_q_rad[-1])
    return (
        trajectory,
        RobotSnapshot(
            measured_q29_rad=tuple(updated_q29),
            left_dex3_q_rad=snapshot.left_dex3_q_rad,
            right_dex3_q_rad=snapshot.right_dex3_q_rad,
        ),
    )


def _monotonic_collision_recovery_trajectory(
    *,
    robot: dict,
    device_cfg,
    source_q: np.ndarray,
    target_q: np.ndarray,
    arm: str,
    source_id: str,
    target_id: str,
    joint_position_offsets_rad: dict[str, float],
    source_collision: dict[tuple[str, str], float],
) -> tuple[PlannedTrajectory | None, str]:
    maximum_step_rad = 0.005
    sample_count = max(
        int(np.ceil(np.max(np.abs(target_q - source_q)) / maximum_step_rad)) + 1,
        2,
    )
    alpha = np.linspace(0.0, 1.0, sample_count)[:, None]
    model_q = source_q[None] + alpha * (target_q - source_q)[None]
    collisions = _self_collision_pair_penetrations(
        robot=robot,
        q_samples=model_q,
        device_cfg=device_cfg,
    )
    initial_pairs = set(source_collision)
    moving_prefix = f"{arm}_"
    numerical_slack_m = 2.5e-4
    for index, values in enumerate(collisions[1:], start=1):
        new_pairs = set(values) - initial_pairs
        if new_pairs:
            return None, (
                f"commissioned recovery creates new self collision at sample "
                f"{index}/{sample_count - 1}: {sorted(new_pairs)}"
            )
        for pair, penetration_m in values.items():
            if penetration_m > source_collision[pair] + numerical_slack_m:
                return None, (
                    f"commissioned recovery worsens {pair[0]}/{pair[1]} from "
                    f"{source_collision[pair]:.6f}m to {penetration_m:.6f}m"
                )
    unresolved = {
        pair: value
        for pair, value in collisions[-1].items()
        if pair[0].startswith(moving_prefix) or pair[1].startswith(moving_prefix)
    }
    if unresolved:
        return None, f"moving {arm} remains in self collision: {unresolved}"
    duration_s = float(np.max(np.abs(target_q - source_q))) / (EXECUTION_MAXIMUM_VELOCITY_RAD_S)
    times = np.linspace(0.0, duration_s, sample_count)
    command_q = np.stack(
        [
            command_from_model_q(
                value,
                arm=arm,
                joint_position_offsets_rad=joint_position_offsets_rad,
            )
            for value in model_q
        ]
    )
    return (
        PlannedTrajectory(
            from_pose_id=source_id,
            to_pose_id=target_id,
            sample_time_s=tuple(times),
            command_q_rad=tuple(tuple(value) for value in command_q),
            model_q_rad=tuple(tuple(value) for value in model_q),
            planning_time_s=0.0,
        ),
        "CuRobo-checked monotonic recovery from measured initial contact",
    )


class CuroboKinematicCollisionChecker:
    """Reusable CuRobo FK and strict self-collision state for one robot model."""

    def __init__(self, *, robot: dict | None = None, device_cfg, kinematics_config=None) -> None:
        from curobo._src.cost.cost_self_collision import SelfCollisionCost
        from curobo._src.cost.cost_self_collision_cfg import SelfCollisionCostCfg
        from curobo._src.robot.kinematics.kinematics import Kinematics
        from curobo._src.robot.kinematics.kinematics_cfg import KinematicsCfg

        if (robot is None) == (kinematics_config is None):
            raise ValueError("provide exactly one of robot or kinematics_config")
        self.device_cfg = device_cfg
        self.config = (
            kinematics_config
            if kinematics_config is not None
            else KinematicsCfg.from_data_dict(robot["kinematics"], device_cfg=device_cfg)
        )
        self.kinematics = Kinematics(self.config)
        self.cost = SelfCollisionCost(
            SelfCollisionCostCfg(
                weight=device_cfg.to_device([1.0]),
                device_cfg=device_cfg,
                self_collision_kin_config=self.config.self_collision_config,
                store_pair_distance=True,
            )
        )

    def robot_spheres(
        self,
        q_samples: np.ndarray,
        *,
        joint_names: tuple[str, ...] | None = None,
    ):
        from curobo._src.state.state_joint import JointState

        values = np.asarray(q_samples, dtype=np.float64)
        names = list(joint_names or tuple(self.kinematics.joint_names))
        state = JointState.from_position(
            self.device_cfg.to_device(values),
            joint_names=names,
        )
        return self.kinematics.compute_kinematics(state).robot_spheres

    def self_collision_pair_penetrations(
        self,
        q_samples: np.ndarray,
        *,
        joint_names: tuple[str, ...] | None = None,
    ) -> list[dict[tuple[str, str], float]]:
        """Return positive sphere penetration grouped by physical links.

        CuRobo's CUDA kernel identifies the sparse set of colliding sphere
        pairs. Only those hits cross the CPU boundary.
        """

        values = np.asarray(q_samples, dtype=np.float64)
        spheres = self.robot_spheres(values, joint_names=joint_names)
        return self.self_collision_pair_penetrations_from_spheres(spheres)

    def self_collision_pair_penetrations_from_spheres(
        self,
        spheres,
    ) -> list[dict[tuple[str, str], float]]:
        """Group strict self overlaps from one already-computed FK sphere tensor."""

        import torch

        sample_count = int(spheres.shape[0])
        self.cost.setup_batch_tensors(sample_count, 1)
        self.cost.forward(spheres)
        pair_hits = torch.nonzero(self.cost._pair_distance[:, 0] > 0.0, as_tuple=False)
        result: list[dict[tuple[str, str], float]] = [{} for _ in range(sample_count)]
        if pair_hits.numel() == 0:
            return result

        collision_pairs = self.config.self_collision_config.collision_pairs
        sample_indices = pair_hits[:, 0]
        pair_indices = pair_hits[:, 1]
        hit_sphere_pairs = collision_pairs[pair_indices].to(dtype=torch.long)
        sphere_values = spheres.reshape(sample_count, -1, 4)
        first = sphere_values[sample_indices, hit_sphere_pairs[:, 0]]
        second = sphere_values[sample_indices, hit_sphere_pairs[:, 1]]
        padding = self.config.self_collision_config.sphere_padding.reshape(-1)
        penetration = (
            first[:, 3]
            + padding[hit_sphere_pairs[:, 0]]
            + second[:, 3]
            + padding[hit_sphere_pairs[:, 1]]
            - torch.linalg.vector_norm(first[:, :3] - second[:, :3], dim=1)
        )
        sphere_links = self.config.kinematics_config.link_sphere_idx_map[
            hit_sphere_pairs.to(dtype=torch.int32)
        ]
        index_to_name = {
            value: name
            for name, value in self.config.kinematics_config.link_name_to_idx_map.items()
        }
        hit_samples = sample_indices.detach().cpu().tolist()
        hit_links = sphere_links.detach().cpu().tolist()
        hit_penetrations = penetration.detach().cpu().tolist()
        for sample_index, (first_link, second_link), penetration_m in zip(
            hit_samples, hit_links, hit_penetrations, strict=True
        ):
            if penetration_m <= 0.0:
                continue
            pair = tuple(sorted((index_to_name[first_link], index_to_name[second_link])))
            result[sample_index][pair] = max(
                result[sample_index].get(pair, 0.0), float(penetration_m)
            )
        return result


class CuroboWorldCollisionChecker:
    """Reusable CUDA sphere-to-world collision queries for one frozen scene."""

    def __init__(self, *, scene: dict[str, Any], device_cfg) -> None:
        from curobo._src.geom.collision.collision_scene import (
            SceneCollision,
            SceneCollisionCfg,
        )
        from curobo._src.geom.types import SceneCfg

        cache = {
            "cuboid": len(scene.get("cuboid", {})),
            "mesh": len(scene.get("mesh", {})),
        }
        self.device_cfg = device_cfg
        self.scene = SceneCollision.from_config(
            SceneCollisionCfg(
                device_cfg=device_cfg,
                scene_model=SceneCfg.create(scene),
                cache=cache,
            )
        )
        self._buffers: dict[tuple[int, ...], Any] = {}

    @staticmethod
    def _query_shape(spheres):
        if spheres.ndim == 3:
            return spheres.unsqueeze(0)
        if spheres.ndim != 4:
            raise ValueError("world collision spheres must have three or four dimensions")
        if spheres.shape[0] == 1:
            return spheres
        if spheres.shape[1] == 1:
            return spheres.transpose(0, 1)
        raise ValueError("world collision sphere tensor has an ambiguous batch shape")

    def deepest_collisions(
        self,
        spheres,
        *,
        kinematics_config=None,
        sphere_link_names: tuple[str, ...] | None = None,
    ) -> list[tuple[float, str, int] | None]:
        """Return the deepest collision for every trajectory sample on CUDA."""

        from curobo._src.geom.collision.buffer_collision import CollisionBuffer

        query = self._query_shape(spheres)
        shape = tuple(int(value) for value in query.shape)
        if shape not in self._buffers:
            self._buffers[shape] = CollisionBuffer.from_shape(shape, self.device_cfg)
        distance = self.scene.get_sphere_distance_raw(
            query,
            self._buffers[shape],
            self.device_cfg.to_device([1.0]),
            self.device_cfg.to_device([0.0]),
        )[0]
        penetration, sphere_index = distance.max(dim=1)
        penetration_values = penetration.detach().cpu().tolist()
        sphere_indices = sphere_index.detach().cpu().tolist()
        if sphere_link_names is not None:
            if len(sphere_link_names) != int(distance.shape[1]):
                raise ValueError("world collision sphere-link list has the wrong length")
            names = sphere_link_names
        else:
            if kinematics_config is None:
                raise ValueError("world collision query requires sphere link metadata")
            link_indices = (
                kinematics_config.link_sphere_idx_map.reshape(-1).detach().cpu().tolist()
            )
            index_to_name = {
                value: name for name, value in kinematics_config.link_name_to_idx_map.items()
            }
            try:
                names = tuple(index_to_name[int(value)] for value in link_indices)
            except KeyError as error:
                raise RuntimeError(
                    "CUDA world collision query could not resolve a sphere link"
                ) from error
        return [
            None if float(value) <= 0.0 else (float(value), names[int(index)], sample)
            for sample, (value, index) in enumerate(
                zip(penetration_values, sphere_indices, strict=True)
            )
        ]

    def first_collision(self, spheres, *, kinematics_config):
        """Return the deepest CUDA-detected world collision, or ``None``."""

        hits = [
            value
            for value in self.deepest_collisions(
                spheres,
                kinematics_config=kinematics_config,
            )
            if value is not None
        ]
        return max(hits, key=lambda value: value[0]) if hits else None

    def minimum_clearance(
        self,
        spheres,
        *,
        kinematics_config,
        activation_distance_m: float,
    ) -> tuple[float, str, int]:
        """Return the closest sphere/mesh clearance within a finite search band.

        CuRobo's raw collision output is the smooth activation cost rather
        than signed distance.  Invert that exact piecewise function here.  A
        returned value equal to ``activation_distance_m`` means the true
        clearance is at least that large.
        """

        from curobo._src.geom.collision.buffer_collision import CollisionBuffer

        if not np.isfinite(activation_distance_m) or activation_distance_m <= 0.0:
            raise ValueError("world-clearance activation distance must be positive")
        query = self._query_shape(spheres)
        shape = tuple(int(value) for value in query.shape)
        if shape not in self._buffers:
            self._buffers[shape] = CollisionBuffer.from_shape(shape, self.device_cfg)
        activation = float(activation_distance_m)
        cost = self.scene.get_sphere_distance_raw(
            query,
            self._buffers[shape],
            self.device_cfg.to_device([1.0]),
            self.device_cfg.to_device([activation]),
        )[0]
        maximum_cost, sphere_index = cost.max(dim=1)
        values = maximum_cost.detach().cpu().numpy()
        clearance = _clearance_from_activation_cost(values, activation)
        sample_index = int(np.argmin(clearance))
        selected_sphere = int(sphere_index[sample_index].item())
        link_index = int(
            kinematics_config.link_sphere_idx_map.reshape(-1)[selected_sphere].item()
        )
        index_to_name = {
            value: name for name, value in kinematics_config.link_name_to_idx_map.items()
        }
        return float(clearance[sample_index]), index_to_name[link_index], sample_index


def _self_collision_pair_penetrations(
    *,
    robot: dict,
    q_samples: np.ndarray,
    device_cfg,
    checker: CuroboKinematicCollisionChecker | None = None,
) -> list[dict[tuple[str, str], float]]:
    """Return strict collision results, optionally reusing parsed CUDA state."""

    active = checker or CuroboKinematicCollisionChecker(robot=robot, device_cfg=device_cfg)
    return active.self_collision_pair_penetrations(q_samples)


def _validate_curobo_finger_sweep(
    *, request: Dex3PreparationRequest, snapshot: RobotSnapshot
) -> int:
    """Use CuRobo's full articulated model on the exact direct finger sweep."""

    import torch
    from curobo.collision_checking import (
        RobotCollisionChecker,
        RobotCollisionCheckerCfg,
    )
    from curobo.types import DeviceCfg

    names = tuple(
        f"{side}_hand_{suffix}_joint"
        for side in ("left", "right")
        for suffix in DEX3_MOTOR_JOINT_SUFFIXES[side]
    )
    robot, start_values = build_robot_config_for_active_joints(
        active_joint_names=names,
        snapshot=snapshot,
        joint_position_offsets_rad=request.joint_position_offsets_rad,
    )
    start_by_name = dict(zip(names, start_values, strict=True))
    target_by_name = dict(
        zip(
            names,
            (*request.left_target_q_rad, *request.right_target_q_rad),
            strict=True,
        )
    )
    device_cfg = DeviceCfg(device=torch.device("cuda:0"), dtype=torch.float32)
    checker = RobotCollisionChecker(
        RobotCollisionCheckerCfg.load_from_config(
            robot_config=robot,
            device_cfg=device_cfg,
            self_collision_activation_distance=COLLISION_ACTIVATION_DISTANCE_M,
        )
    )
    # CuRobo exposes active joints in kinematic-tree order, which is not the
    # Unitree Dex3 DDS motor order above (the index fingers are first in this
    # model).  Every raw tensor passed to RobotCollisionChecker must therefore
    # be assembled by name in CuRobo's own order.
    curobo_names = tuple(checker.kinematics.joint_names)
    if len(curobo_names) != len(names) or set(curobo_names) != set(names):
        raise RuntimeError("CuRobo Dex3 active-joint set differs from the requested model")
    start = np.asarray([start_by_name[name] for name in curobo_names], dtype=np.float64)
    target = np.asarray([target_by_name[name] for name in curobo_names], dtype=np.float64)
    sweep = sample_linear_joint_sweep(start, target)
    intervals = len(sweep) - 1
    q = device_cfg.to_device(sweep).unsqueeze(0)
    state = checker.get_kinematics(q)
    collision_cost = checker.get_self_collision_distance(state.robot_spheres)
    colliding = collision_cost.detach().cpu().numpy().reshape(len(sweep), -1).sum(axis=1) > 1e-8
    if np.any(colliding):
        first = int(np.flatnonzero(colliding)[0])
        raise RuntimeError(f"finger sweep self-collision at sample {first}/{intervals}")

    limits = checker.kinematics.get_joint_limits().position.detach().cpu().numpy()
    lower, upper = limits[0], limits[1]
    if np.any(target < lower - 1e-6) or np.any(target > upper + 1e-6):
        raise RuntimeError("NVIDIA middle-close target is outside CuRobo joint limits")
    violations = np.maximum(np.maximum(lower[None] - sweep, sweep - upper[None]), 0.0)
    for joint_index, joint_name in enumerate(curobo_names):
        values = violations[:, joint_index]
        if values[0] <= 1e-6 and np.any(values > 1e-6):
            raise RuntimeError(f"finger sweep leaves the hard limits at {joint_name}")
        if np.any(np.diff(values) > 1e-6) or values[-1] > 1e-6:
            raise RuntimeError(
                f"finger sweep does not monotonically recover {joint_name} into limits"
            )
    del checker
    gc.collect()
    torch.cuda.empty_cache()
    return len(sweep)


def inspect_model(request: CalibrationPlanRequest) -> dict:
    robot, reference = build_locked_robot_config(
        arm=request.arm,
        snapshot=request.snapshot,
        joint_position_offsets_rad=request.joint_position_offsets_rad,
    )
    spheres = robot["kinematics"]["collision_spheres"]
    return {
        "commands_robot": False,
        "backend": "NVlabs/curobo",
        "curobo_commit": CUROBO_COMMIT,
        "arm": request.arm,
        "active_joint_names": list(arm_joint_names(request.arm)),
        "active_reference_model_q_rad": list(reference),
        "locked_joint_count": len(robot["kinematics"]["lock_joints"]),
        "collision_link_count": len(spheres),
        "collision_sphere_count": sum(len(values) for values in spheres.values()),
        "tool_frames": list(robot["kinematics"]["tool_frames"]),
        "model_sources": model_source_hashes(),
        "request_sha256": request.content_sha256,
    }


def plan_calibration(
    request: CalibrationPlanRequest,
    *,
    progress: Callable[[str], None] | None = None,
) -> CalibrationPlanResult:
    """Filter candidates with batched IK, select 80, and plan every route edge."""

    report = progress or (lambda _message: None)
    design = CandidateDesignConfig.from_dict(request.selection_config)
    if design.target_count != request.target_count:
        raise ValueError("request target count differs from the selection configuration")
    robot, reference_tuple = build_locked_robot_config(
        arm=request.arm,
        snapshot=request.snapshot,
        joint_position_offsets_rad=request.joint_position_offsets_rad,
    )
    reference = np.asarray(reference_tuple, dtype=np.float64)
    feasible, ik_seconds, device_name = _batched_ik(
        request=request,
        robot=robot,
        reference=reference,
        progress=report,
    )
    if len(feasible) < request.target_count:
        raise RuntimeError(
            f"CuRobo found only {len(feasible)} collision-free IK poses; "
            f"requested {request.target_count}"
        )
    feasible_candidates = [request.candidates[item.source_index] for item in feasible]
    ranked_candidates, ranking_steps = select_information_candidates(
        feasible_candidates,
        count=len(feasible_candidates),
        config=design,
    )
    ik_by_id = {request.candidates[item.source_index].candidate_id: item for item in feasible}
    ranked_poses = [
        PlannedCalibrationPose(
            candidate_id=candidate.candidate_id,
            camera_T_marker=candidate.camera_T_marker,
            model_q_rad=tuple(ik_by_id[candidate.candidate_id].model_q),
            command_q_rad=tuple(
                command_from_model_q(
                    ik_by_id[candidate.candidate_id].model_q,
                    arm=request.arm,
                    joint_position_offsets_rad=request.joint_position_offsets_rad,
                )
            ),
            ik_position_error_m=ik_by_id[candidate.candidate_id].position_error_m,
            ik_rotation_error_rad=ik_by_id[candidate.candidate_id].rotation_error_rad,
            selection_metadata={
                **candidate.selection_metadata,
                "selection_step": ranking_steps[index],
            },
        )
        for index, candidate in enumerate(ranked_candidates)
    ]
    ordered, trajectories, route_seconds, route_skips, fallback_edges = _plan_route(
        robot=robot,
        arm=request.arm,
        reference=reference,
        ranked=ranked_poses,
        target_count=request.target_count,
        joint_position_offsets_rad=request.joint_position_offsets_rad,
        random_seed=request.random_seed,
        progress=report,
    )
    report(
        f"selected and connected {len(ordered)} fixed-marker information/coverage "
        f"poses from {len(feasible)} CuRobo-feasible candidates"
    )
    selected_ids = {item.candidate_id for item in ordered}
    selection_steps = tuple(step for step in ranking_steps if step["candidate_id"] in selected_ids)
    route_ids = (
        (HANDOFF_POSE_ID,) + tuple(item.candidate_id for item in ordered) + (HANDOFF_POSE_ID,)
    )
    return CalibrationPlanResult(
        request_sha256=request.content_sha256,
        arm=request.arm,
        active_joint_names=arm_joint_names(request.arm),
        handoff_model_q_rad=tuple(reference),
        handoff_command_q_rad=tuple(
            command_from_model_q(
                reference,
                arm=request.arm,
                joint_position_offsets_rad=request.joint_position_offsets_rad,
            )
        ),
        poses=tuple(ordered),
        route_pose_ids=route_ids,
        capture_pose_ids=route_ids[1:-1],
        trajectories=tuple(trajectories),
        selection_steps=tuple(selection_steps),
        planner_provenance={
            **model_source_hashes(),
            "device": device_name,
            "candidate_count": len(request.candidates),
            "feasible_ik_count": len(feasible),
            "selected_count": len(ordered),
            "ik_elapsed_s": ik_seconds,
            "route_elapsed_s": route_seconds,
            "route_candidate_skips": route_skips,
            "route_via_handoff_fallback_edges": fallback_edges,
            "ik_batch_size": min(request.ik_batch_size, len(request.candidates)),
            "ik_seeds": IK_SEEDS,
            "ik_return_seeds": IK_RETURN_SEEDS,
            "ik_position_tolerance_m": IK_POSITION_TOLERANCE_M,
            "ik_rotation_tolerance_rad": IK_ROTATION_TOLERANCE_RAD,
            "self_collision_check": True,
            "collision_activation_distance_m": COLLISION_ACTIVATION_DISTANCE_M,
            "trajectory_interpolation_dt_s": TRAJECTORY_INTERPOLATION_DT_S,
            "execution_maximum_velocity_rad_s": EXECUTION_MAXIMUM_VELOCITY_RAD_S,
            "mounted_plate_spheres_per_hand": 30,
            "route_policy": (
                "information_ranked_handoff_connected_targets_nearest_neighbor; "
                "direct_edges_with_frozen_via_handoff_fallback; direct_return"
            ),
        },
    )


def _batched_ik(
    *,
    request: CalibrationPlanRequest,
    robot: dict,
    reference: np.ndarray,
    progress: Callable[[str], None],
) -> tuple[list[_FeasibleIK], float, str]:
    import torch
    from curobo.inverse_kinematics import InverseKinematics, InverseKinematicsCfg
    from curobo.types import DeviceCfg, GoalToolPose, JointState, Pose

    if not torch.cuda.is_available():
        raise RuntimeError("CuRobo calibration planning requires a CUDA device")
    device_cfg = DeviceCfg(device=torch.device("cuda:0"), dtype=torch.float32)
    batch_capacity = min(request.ik_batch_size, len(request.candidates))
    config = InverseKinematicsCfg.create(
        robot=robot,
        device_cfg=device_cfg,
        num_seeds=IK_SEEDS,
        self_collision_check=True,
        max_batch_size=batch_capacity,
        use_cuda_graph=True,
        position_tolerance=IK_POSITION_TOLERANCE_M,
        orientation_tolerance=IK_ROTATION_TOLERANCE_RAD,
        optimizer_collision_activation_distance=COLLISION_ACTIVATION_DISTANCE_M,
        random_seed=request.random_seed,
    )
    solver = InverseKinematics(config)
    reference_tensor = device_cfg.to_device(reference).unsqueeze(0)
    reference_state = JointState.from_position(reference_tensor, joint_names=solver.joint_names)
    kinematics = solver.compute_kinematics(reference_state)
    base_T_torso = kinematics.tool_poses["torso_link"].get_matrix()[0].detach().cpu().numpy()
    torso_T_camera = np.asarray(request.torso_T_camera, dtype=np.float64)
    marker_T_palm = invert_transform(np.asarray(request.palm_T_marker, dtype=np.float64))
    base_T_palm = np.stack(
        [
            base_T_torso
            @ torso_T_camera
            @ np.asarray(candidate.camera_T_marker, dtype=np.float64)
            @ marker_T_palm
            for candidate in request.candidates
        ]
    )
    feasible: list[_FeasibleIK] = []
    started = time.monotonic()
    for lower in range(0, len(request.candidates), batch_capacity):
        upper = min(lower + batch_capacity, len(request.candidates))
        count = upper - lower
        batch_palm = base_T_palm[lower:upper]
        if count < batch_capacity:
            batch_palm = np.concatenate(
                (
                    batch_palm,
                    np.repeat(batch_palm[-1][None], batch_capacity - count, axis=0),
                )
            )
        palm_goal = Pose.from_matrix(device_cfg.to_device(batch_palm))
        torso_goal = Pose.from_matrix(
            device_cfg.to_device(np.repeat(base_T_torso[None], batch_capacity, axis=0))
        )
        goals = GoalToolPose.from_poses(
            {palm_link(request.arm): palm_goal, "torso_link": torso_goal},
            ordered_tool_frames=solver.tool_frames,
        )
        current = JointState.from_position(
            reference_tensor.repeat(batch_capacity, 1), joint_names=solver.joint_names
        )
        result = solver.solve_pose(
            goal_tool_poses=goals,
            return_seeds=IK_RETURN_SEEDS,
            current_state=current,
        )
        success = result.success.detach().cpu().numpy().astype(bool)
        solutions = result.solution.detach().cpu().numpy()
        position_errors = result.position_error.detach().cpu().numpy()
        rotation_errors = result.rotation_error.detach().cpu().numpy()
        for local_index in range(count):
            valid = np.flatnonzero(success[local_index])
            if not len(valid):
                continue
            best = int(
                min(
                    valid,
                    key=lambda index: float(
                        np.linalg.norm(solutions[local_index, index] - reference)
                    ),
                )
            )
            feasible.append(
                _FeasibleIK(
                    source_index=lower + local_index,
                    model_q=np.asarray(solutions[local_index, best], dtype=np.float64),
                    position_error_m=float(np.max(position_errors[local_index, best])),
                    rotation_error_rad=float(np.max(rotation_errors[local_index, best])),
                )
            )
        progress(
            f"CuRobo batched IK: {upper}/{len(request.candidates)} evaluated; "
            f"{len(feasible)} feasible"
        )
    elapsed = time.monotonic() - started
    device_name = torch.cuda.get_device_name(device_cfg.device)
    if hasattr(solver, "destroy"):
        solver.destroy()
    del solver
    gc.collect()
    torch.cuda.empty_cache()
    return feasible, elapsed, device_name


def _nearest_neighbor_order(
    poses: list[PlannedCalibrationPose], reference: np.ndarray
) -> list[PlannedCalibrationPose]:
    remaining = list(poses)
    current = np.asarray(reference, dtype=np.float64)
    ordered: list[PlannedCalibrationPose] = []
    while remaining:
        selected = min(
            remaining,
            key=lambda item: (
                float(np.linalg.norm(np.asarray(item.model_q_rad) - current)),
                item.candidate_id,
            ),
        )
        ordered.append(selected)
        remaining.remove(selected)
        current = np.asarray(selected.model_q_rad, dtype=np.float64)
    return ordered


def _plan_route(
    *,
    robot: dict,
    arm: str,
    reference: np.ndarray,
    ranked: list[PlannedCalibrationPose],
    target_count: int,
    joint_position_offsets_rad: dict[str, float],
    random_seed: int,
    progress: Callable[[str], None],
) -> tuple[
    list[PlannedCalibrationPose],
    list[PlannedTrajectory],
    float,
    list[dict[str, str]],
    int,
]:
    import torch
    from curobo.motion_planner import MotionPlanner, MotionPlannerCfg
    from curobo.types import DeviceCfg

    device_cfg = DeviceCfg(device=torch.device("cuda:0"), dtype=torch.float32)
    config = MotionPlannerCfg.create(
        robot=robot,
        device_cfg=device_cfg,
        num_ik_seeds=IK_SEEDS,
        num_trajopt_seeds=4,
        self_collision_check=True,
        use_cuda_graph=True,
        random_seed=random_seed,
        optimizer_collision_activation_distance=COLLISION_ACTIVATION_DISTANCE_M,
        interpolation_dt=TRAJECTORY_INTERPOLATION_DT_S,
        interpolation_buffer_size=1000,
    )
    planner = MotionPlanner(config)
    names = list(arm_joint_names(arm))
    started = time.monotonic()
    handoff = np.asarray(reference, dtype=np.float64)
    accepted: list[PlannedCalibrationPose] = []
    star_paths: dict[str, PlannedTrajectory] = {}
    route_skips: list[dict[str, str]] = []
    for candidate in ranked:
        candidate_q = np.asarray(candidate.model_q_rad, dtype=np.float64)
        star, diagnostics = _plan_edge(
            planner=planner,
            device_cfg=device_cfg,
            names=names,
            arm=arm,
            source_id=HANDOFF_POSE_ID,
            target_id=candidate.candidate_id,
            source_q=handoff,
            target_q=candidate_q,
            joint_position_offsets_rad=joint_position_offsets_rad,
        )
        if star is None:
            route_skips.append({"candidate_id": candidate.candidate_id, "reason": diagnostics})
            progress(
                f"CuRobo handoff connectivity rejected {candidate.candidate_id}: {diagnostics}"
            )
            continue
        accepted.append(candidate)
        star_paths[candidate.candidate_id] = star
        progress(
            f"CuRobo handoff connectivity: {len(accepted)}/{target_count} accepted; "
            f"last={candidate.candidate_id}"
        )
        if len(accepted) == target_count:
            break
    if len(accepted) != target_count:
        planner.destroy()
        raise RuntimeError(
            f"only {len(accepted)} information-ranked IK poses had a CuRobo path "
            f"to the measured handoff; requested {target_count}"
        )

    ordered = _nearest_neighbor_order(accepted, handoff)
    trajectories: list[PlannedTrajectory] = [star_paths[ordered[0].candidate_id]]
    fallback_edges = 0
    for edge_index, (source_pose, target_pose) in enumerate(pairwise(ordered), start=2):
        direct, diagnostics = _plan_edge(
            planner=planner,
            device_cfg=device_cfg,
            names=names,
            arm=arm,
            source_id=source_pose.candidate_id,
            target_id=target_pose.candidate_id,
            source_q=np.asarray(source_pose.model_q_rad, dtype=np.float64),
            target_q=np.asarray(target_pose.model_q_rad, dtype=np.float64),
            joint_position_offsets_rad=joint_position_offsets_rad,
        )
        if direct is None:
            direct = _concatenate_trajectories(
                _reverse_trajectory(star_paths[source_pose.candidate_id]),
                star_paths[target_pose.candidate_id],
                from_pose_id=source_pose.candidate_id,
                to_pose_id=target_pose.candidate_id,
            )
            fallback_edges += 1
            progress(
                f"CuRobo direct edge {source_pose.candidate_id}->{target_pose.candidate_id} "
                f"was unavailable ({diagnostics}); froze its validated via-handoff path"
            )
        trajectories.append(direct)
        progress(
            f"CuRobo route: {edge_index}/{target_count + 1} frozen; "
            f"{source_pose.candidate_id}->{target_pose.candidate_id}; "
            f"duration={direct.sample_time_s[-1]:.2f}s"
        )
    final_return = _reverse_trajectory(star_paths[ordered[-1].candidate_id])
    trajectories.append(final_return)
    progress(
        f"CuRobo route: {target_count + 1}/{target_count + 1} frozen; "
        f"{ordered[-1].candidate_id}->{HANDOFF_POSE_ID}; "
        f"duration={final_return.sample_time_s[-1]:.2f}s"
    )
    elapsed = time.monotonic() - started
    planner.destroy()
    return ordered, trajectories, elapsed, route_skips, fallback_edges


def _plan_edge(
    *,
    planner,
    device_cfg,
    names: list[str],
    arm: str,
    source_id: str,
    target_id: str,
    source_q: np.ndarray,
    target_q: np.ndarray,
    joint_position_offsets_rad: dict[str, float],
) -> tuple[PlannedTrajectory | None, str]:
    import torch
    from curobo.types import JointState

    source = JointState.from_position(
        device_cfg.to_device(source_q).unsqueeze(0), joint_names=names
    )
    target = JointState.from_position(
        device_cfg.to_device(target_q).unsqueeze(0), joint_names=names
    )
    result = planner.plan_cspace(
        goal_state=target,
        current_state=source,
        max_attempts=5,
        enable_graph_attempt=1,
    )
    if result is None or not bool(torch.any(result.success)):
        if result is None:
            return None, "no planner result"
        return None, (
            f"success={result.success.detach().cpu().tolist()}, "
            f"position_error="
            f"{None if result.position_error is None else result.position_error.detach().cpu().tolist()}, "
            f"rotation_error="
            f"{None if result.rotation_error is None else result.rotation_error.detach().cpu().tolist()}"
        )
    plan = result.get_interpolated_plan().reorder(names)
    model_q = np.asarray(plan.position.detach().cpu().numpy(), dtype=np.float64).squeeze()
    if model_q.ndim != 2 or model_q.shape[1] != 7 or len(model_q) < 2:
        raise RuntimeError(
            f"CuRobo returned an invalid trajectory shape for {source_id}->{target_id}: "
            f"{model_q.shape}"
        )
    if np.linalg.norm(model_q[0] - source_q) > 1e-3:
        raise RuntimeError(f"CuRobo trajectory does not begin at {source_id}")
    if np.linalg.norm(model_q[-1] - target_q) > 1e-3:
        raise RuntimeError(f"CuRobo trajectory does not end at {target_id}")
    model_q[0] = source_q
    model_q[-1] = target_q
    native_dt = _joint_state_dt(plan)
    maximum_step_velocity = float(np.max(np.abs(np.diff(model_q, axis=0))) / native_dt)
    time_scale = max(1.0, maximum_step_velocity / EXECUTION_MAXIMUM_VELOCITY_RAD_S)
    dt = native_dt * time_scale
    command_q = np.stack(
        [
            command_from_model_q(
                value,
                arm=arm,
                joint_position_offsets_rad=joint_position_offsets_rad,
            )
            for value in model_q
        ]
    )
    return (
        PlannedTrajectory(
            from_pose_id=source_id,
            to_pose_id=target_id,
            sample_time_s=tuple(float(index * dt) for index in range(len(model_q))),
            command_q_rad=tuple(tuple(value) for value in command_q),
            model_q_rad=tuple(tuple(value) for value in model_q),
            planning_time_s=float(result.total_time),
        ),
        "success",
    )


def _reverse_trajectory(trajectory: PlannedTrajectory) -> PlannedTrajectory:
    duration = trajectory.sample_time_s[-1]
    times = tuple(duration - value for value in reversed(trajectory.sample_time_s))
    return PlannedTrajectory(
        from_pose_id=trajectory.to_pose_id,
        to_pose_id=trajectory.from_pose_id,
        sample_time_s=times,
        command_q_rad=tuple(reversed(trajectory.command_q_rad)),
        model_q_rad=tuple(reversed(trajectory.model_q_rad)),
        planning_time_s=0.0,
    )


def _concatenate_trajectories(
    first: PlannedTrajectory,
    second: PlannedTrajectory,
    *,
    from_pose_id: str,
    to_pose_id: str,
) -> PlannedTrajectory:
    if first.to_pose_id != second.from_pose_id:
        raise ValueError("trajectory concatenation endpoints differ")
    offset = first.sample_time_s[-1]
    times = first.sample_time_s + tuple(offset + value for value in second.sample_time_s[1:])
    return PlannedTrajectory(
        from_pose_id=from_pose_id,
        to_pose_id=to_pose_id,
        sample_time_s=times,
        command_q_rad=first.command_q_rad + second.command_q_rad[1:],
        model_q_rad=first.model_q_rad + second.model_q_rad[1:],
        planning_time_s=first.planning_time_s + second.planning_time_s,
    )


def _joint_state_dt(plan) -> float:
    value = plan.dt
    if value is None:
        return TRAJECTORY_INTERPOLATION_DT_S
    if hasattr(value, "detach"):
        array = np.asarray(value.detach().cpu().numpy(), dtype=np.float64).reshape(-1)
    else:
        array = np.asarray(value, dtype=np.float64).reshape(-1)
    positive = array[np.isfinite(array) & (array > 0)]
    if not len(positive):
        return TRAJECTORY_INTERPOLATION_DT_S
    return float(positive[0])
