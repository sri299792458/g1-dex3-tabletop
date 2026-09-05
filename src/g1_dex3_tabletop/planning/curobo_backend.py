"""Native CuRobo batched IK and complete calibration-route planning."""

from __future__ import annotations

import gc
import heapq
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from itertools import pairwise
from typing import Any

import numpy as np

from g1_aprilcube_calibration.joint_map import (
    LEFT_ARM_INDICES,
    RIGHT_ARM_INDICES,
    arm_indices,
    arm_joint_names,
)
from g1_aprilcube_calibration.pose_schema import HANDOFF_POSE_ID
from g1_aprilcube_calibration.transforms import invert_transform
from g1_aprilcube_calibration.transports.unitree_dex3 import (
    DEX3_MOTOR_JOINT_SUFFIXES,
)
from g1_dex3_tabletop.calibration.design import BilateralDesignSelection
from g1_dex3_tabletop.calibration.execution import BilateralPlannedTransition
from g1_dex3_tabletop.calibration.planning import (
    BilateralCalibrationPlanningRequest,
    BilateralDesignPool,
    BilateralFeasiblePose,
    BilateralIKResult,
    BilateralRoutePlanningRequest,
    BilateralRoutePlanningResult,
)
from g1_dex3_tabletop.calibration_candidates import (
    CandidateDesignConfig,
    select_information_candidates,
)
from g1_dex3_tabletop.planning.contracts import (
    BilateralCalibrationAdapterPlan,
    BilateralCalibrationAdapterRequest,
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
    Dex3FingerTargetLimitError,
    build_locked_robot_config,
    build_robot_config_for_active_joints,
    command_from_model_q,
    model_source_hashes,
    palm_link,
    validate_dex3_finger_targets,
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
# Calibration routes must retain a measured geometric margin, not merely avoid
# positive penetration. Pairs already closer than this at the commissioned
# phase reference are allowed only the small, explicit degradation below.
CALIBRATION_SELF_CLEARANCE_M = 0.010
CALIBRATION_PREPARATION_CLEARANCE_M = 0.005
CALIBRATION_PREEXISTING_CLEARANCE_DEGRADATION_M = 0.00025
CALIBRATION_CLEARANCE_NUMERICAL_TOLERANCE_M = 1.0e-6
# Keep CuRobo's optimizer activation band separate from the hard open-hand
# object margin. Entering the activation band should shape the optimizer cost;
# it is not itself a physical collision.
OPEN_TRANSIT_OBJECT_CLEARANCE_M = 0.005
FINGER_SWEEP_MAXIMUM_JOINT_STEP_RAD = 0.02
FINGER_SWEEP_MINIMUM_CLEARANCE_M = 0.005


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


@dataclass(frozen=True, slots=True)
class BilateralRouteConnectivity:
    """Anchor-rooted collision trees over the full visible candidate pool."""

    connected_candidate_ids_by_arm: dict[str, tuple[str, ...]]
    parent_candidate_id_by_arm: dict[str, dict[str, str]]
    provenance: dict[str, Any]


def plan_dex3_preparation(
    request: Dex3PreparationRequest,
    *,
    progress: Callable[[str], None] | None = None,
) -> Dex3PreparationPlan:
    """Plan the commissioned right-then-left shoulder clearance with CuRobo."""

    validate_dex3_finger_targets(
        left_q_rad=request.left_target_q_rad,
        right_q_rad=request.right_target_q_rad,
        label="calibration close target",
    )
    if request.left_return_target_q_rad is not None:
        validate_dex3_finger_targets(
            left_q_rad=request.left_return_target_q_rad,
            right_q_rad=request.right_return_target_q_rad,
            label="starting-posture restoration target",
        )
    import torch
    from curobo.types import DeviceCfg

    report = progress or (lambda _message: None)
    device_cfg = DeviceCfg(device=torch.device("cuda:0"), dtype=torch.float32)
    if request.left_return_target_q_rad is not None:
        assert request.right_return_target_q_rad is not None
        return_ready = RobotSnapshot(
            measured_q29_rad=request.snapshot.measured_q29_rad,
            left_dex3_q_rad=request.left_return_target_q_rad,
            right_dex3_q_rad=request.right_return_target_q_rad,
        )
        robot, reference = build_robot_config_for_active_joints(
            active_joint_names=(*arm_joint_names("left"), *arm_joint_names("right")),
            snapshot=return_ready,
            joint_position_offsets_rad=request.joint_position_offsets_rad,
            ignore_internal_hand_collisions=True,
            ignore_adjacent_shoulder_collisions=True,
            ignore_static_body_collisions=True,
        )
        checker = CuroboKinematicCollisionChecker(robot=robot, device_cfg=device_cfg)
        joint_names = tuple(checker.kinematics.joint_names)
        reference = _full_arm_model_reference(
            return_ready.measured_q29_rad,
            joint_position_offsets_rad=request.joint_position_offsets_rad,
            joint_names=joint_names,
        )
        collisions = checker.self_collision_pair_penetrations(
            reference[None], joint_names=joint_names
        )[0]
        del checker
        if collisions:
            detail = ", ".join(
                f"{left}/{right} ({value * 1000.0:.2f}mm modeled overlap)"
                for (left, right), value in sorted(collisions.items())
            )
            raise ValueError(
                "requested return hand posture collides at the measured Ready "
                f"arm/body state: {detail}. Increasing shoulder clearance cannot "
                "fix this fixed return endpoint; the hand-return policy must be corrected."
            )
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
            arm_clearance_certificate = _certify_bilateral_arm_segments(
                snapshot=request.snapshot,
                reference_q29=request.snapshot.measured_q29_rad,
                joint_position_offsets_rad=request.joint_position_offsets_rad,
                segments=(
                    (
                        right_outbound.from_pose_id + "->" + right_outbound.to_pose_id,
                        "right",
                        right_outbound,
                        request.snapshot.measured_q29_rad,
                    ),
                    (
                        left_outbound.from_pose_id + "->" + left_outbound.to_pose_id,
                        "left",
                        left_outbound,
                        after_right.measured_q29_rad,
                    ),
                ),
                device_cfg=device_cfg,
                phase="ready_hand_preparation",
            )
            if not arm_clearance_certificate["passed"]:
                raise ValueError(
                    "shoulder preparation violates the hard self-clearance policy: "
                    f"{arm_clearance_certificate['minimum_margin_segment_id']} "
                    f"has margin "
                    f"{arm_clearance_certificate['minimum_margin_to_required_clearance_m']:.6f}m"
                )
            return_sweep_count = 0
            return_arm_certificate = None
            if request.left_return_target_q_rad is not None:
                assert request.right_return_target_q_rad is not None
                assert request.left_settled_target_q_rad is not None
                assert request.right_settled_target_q_rad is not None
                return_snapshot = RobotSnapshot(
                    measured_q29_rad=after_dual.measured_q29_rad,
                    left_dex3_q_rad=request.left_settled_target_q_rad,
                    right_dex3_q_rad=request.right_settled_target_q_rad,
                )
                return_request = Dex3PreparationRequest(
                    snapshot=return_snapshot,
                    joint_position_offsets_rad=request.joint_position_offsets_rad,
                    left_target_q_rad=request.left_return_target_q_rad,
                    right_target_q_rad=request.right_return_target_q_rad,
                    random_seed=request.random_seed,
                )
                return_sweep_count = _validate_curobo_finger_sweep(
                    request=return_request,
                    snapshot=return_snapshot,
                )
                # A reverse arm path is only equivalent to the outbound path
                # when the fingers have the same geometry. Certify the exact
                # return samples with the fingers commanded during return.
                return_arm_certificate = _certify_bilateral_arm_segments(
                    snapshot=return_ready,
                    reference_q29=return_ready.measured_q29_rad,
                    joint_position_offsets_rad=request.joint_position_offsets_rad,
                    segments=(
                        (
                            "left_shoulder_return",
                            "left",
                            _reverse_trajectory(left_outbound),
                            after_dual.measured_q29_rad,
                        ),
                        (
                            "right_shoulder_return",
                            "right",
                            _reverse_trajectory(right_outbound),
                            after_right.measured_q29_rad,
                        ),
                    ),
                    device_cfg=device_cfg,
                    phase="return_hand_shoulder_preparation",
                )
                if not return_arm_certificate["passed"]:
                    raise ValueError(
                        "shoulder return with the requested hand posture violates "
                        "the hard self-clearance policy: minimum margin "
                        f"{return_arm_certificate['minimum_margin_to_required_clearance_m']:.6f}m"
                    )
        except (RuntimeError, ValueError) as error:
            if isinstance(error, Dex3FingerTargetLimitError):
                raise
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
            return_sweep_sample_count=return_sweep_count,
            planner_provenance={
                **model_source_hashes(),
                "route_policy": (
                    "right_shoulder_then_left_shoulder_then_close_sweep"
                    + ("_and_return_sweep" if return_sweep_count else "")
                ),
                "execution_maximum_velocity_rad_s": EXECUTION_MAXIMUM_VELOCITY_RAD_S,
                "self_collision_activation_distance_m": (COLLISION_ACTIVATION_DISTANCE_M),
                "finger_sweep_maximum_joint_step_rad": (FINGER_SWEEP_MAXIMUM_JOINT_STEP_RAD),
                "arm_self_clearance_certificate": arm_clearance_certificate,
                "return_arm_self_clearance_certificate": return_arm_certificate,
                "internal_hand_pair_policy": "commissioned_same_hand_exclusion",
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
                robot=robot,
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
        import torch

        self._clearance_sphere_pairs = self.config.self_collision_config.collision_pairs.to(
            dtype=torch.long
        )
        sphere_links = self.config.kinematics_config.link_sphere_idx_map[
            self._clearance_sphere_pairs.to(dtype=torch.int32)
        ].to(dtype=torch.long)
        ordered_link_pairs = torch.sort(sphere_links, dim=1).values
        self._clearance_unique_link_pairs, self._clearance_pair_groups = torch.unique(
            ordered_link_pairs,
            dim=0,
            return_inverse=True,
        )
        index_to_name = {
            value: name
            for name, value in self.config.kinematics_config.link_name_to_idx_map.items()
        }
        self._clearance_link_pairs = tuple(
            tuple(index_to_name[index] for index in pair)
            for pair in self._clearance_unique_link_pairs.detach().cpu().tolist()
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

    def self_collision_link_pair_clearances(
        self,
        q_samples: np.ndarray,
        *,
        joint_names: tuple[str, ...] | None = None,
    ):
        """Return minimum signed sphere clearance for every physical link pair."""

        import torch

        values = np.asarray(q_samples, dtype=np.float64)
        spheres = self.robot_spheres(values, joint_names=joint_names).reshape(
            len(values),
            -1,
            4,
        )
        sphere_pairs = self._clearance_sphere_pairs
        padding = self.config.self_collision_config.sphere_padding.reshape(-1)
        first = spheres[:, sphere_pairs[:, 0]]
        second = spheres[:, sphere_pairs[:, 1]]
        sphere_clearance = torch.linalg.vector_norm(
            first[:, :, :3] - second[:, :, :3],
            dim=2,
        ) - (
            first[:, :, 3]
            + padding[sphere_pairs[:, 0]]
            + second[:, :, 3]
            + padding[sphere_pairs[:, 1]]
        )

        link_clearance = torch.full(
            (len(values), len(self._clearance_unique_link_pairs)),
            float("inf"),
            device=sphere_clearance.device,
            dtype=sphere_clearance.dtype,
        )
        link_clearance.scatter_reduce_(
            1,
            self._clearance_pair_groups[None].expand(len(values), -1),
            sphere_clearance,
            reduce="amin",
            include_self=True,
        )
        return link_clearance, self._clearance_link_pairs


def _self_clearance_certificate(
    *,
    checker: CuroboKinematicCollisionChecker,
    joint_names: tuple[str, ...],
    reference_q: np.ndarray,
    segments: Sequence[tuple[str, np.ndarray]],
    phase: str,
    preparation: bool = False,
) -> dict[str, Any]:
    """Certify sampled route clearance against one fixed phase reference."""

    import torch

    reference, link_pairs = checker.self_collision_link_pair_clearances(
        np.asarray(reference_q, dtype=np.float64)[None],
        joint_names=joint_names,
    )
    reference = reference[0]
    clearance_floor = (
        CALIBRATION_PREPARATION_CLEARANCE_M if preparation else CALIBRATION_SELF_CLEARANCE_M
    )
    hard = torch.full_like(reference, clearance_floor)
    required = (
        torch.where(
            reference >= clearance_floor,
            hard,
            torch.clamp(reference - CALIBRATION_PREEXISTING_CLEARANCE_DEGRADATION_M, min=0.0),
        )
        if preparation
        else hard
    )
    evidence: list[dict[str, Any]] = []
    for segment_id, q_samples in segments:
        samples = np.asarray(q_samples, dtype=np.float64)
        clearances, actual_pairs = checker.self_collision_link_pair_clearances(
            samples,
            joint_names=joint_names,
        )
        if actual_pairs != link_pairs:
            raise RuntimeError("self-clearance link-pair ordering changed")
        minimum, minimum_flat = torch.min(clearances.reshape(-1), dim=0)
        margin_values = clearances - required[None]
        margin, margin_flat = torch.min(margin_values.reshape(-1), dim=0)
        pair_count = clearances.shape[1]
        minimum_sample = int(minimum_flat // pair_count)
        minimum_pair = int(minimum_flat % pair_count)
        margin_sample = int(margin_flat // pair_count)
        margin_pair = int(margin_flat % pair_count)
        evidence.append(
            {
                "segment_id": segment_id,
                "sample_count": len(samples),
                "minimum_clearance_m": float(minimum.detach().cpu()),
                "minimum_clearance_sample_index": minimum_sample,
                "minimum_clearance_link_pair": list(link_pairs[minimum_pair]),
                "minimum_margin_to_required_clearance_m": float(margin.detach().cpu()),
                "minimum_margin_sample_index": margin_sample,
                "minimum_margin_link_pair": list(link_pairs[margin_pair]),
                "required_clearance_at_minimum_margin_m": float(
                    required[margin_pair].detach().cpu()
                ),
                "actual_clearance_at_minimum_margin_m": float(
                    clearances[margin_sample, margin_pair].detach().cpu()
                ),
            }
        )
    if not evidence:
        raise ValueError("self-clearance certificate requires sampled segments")
    worst_clearance = min(evidence, key=lambda item: item["minimum_clearance_m"])
    worst_margin = min(
        evidence,
        key=lambda item: item["minimum_margin_to_required_clearance_m"],
    )
    tolerance = CALIBRATION_CLEARANCE_NUMERICAL_TOLERANCE_M
    return {
        "phase": phase,
        "policy": "ready_5mm_or_reference_bounded" if preparation else "strict_core_10mm",
        "hard_clearance_m": clearance_floor,
        "preexisting_clearance_maximum_degradation_m": (
            CALIBRATION_PREEXISTING_CLEARANCE_DEGRADATION_M if preparation else 0.0
        ),
        "numerical_tolerance_m": tolerance,
        "reference_minimum_clearance_m": float(torch.min(reference).detach().cpu()),
        "minimum_clearance_m": worst_clearance["minimum_clearance_m"],
        "minimum_clearance_segment_id": worst_clearance["segment_id"],
        "minimum_clearance_sample_index": worst_clearance["minimum_clearance_sample_index"],
        "minimum_clearance_link_pair": worst_clearance["minimum_clearance_link_pair"],
        "minimum_margin_to_required_clearance_m": worst_margin[
            "minimum_margin_to_required_clearance_m"
        ],
        "minimum_margin_segment_id": worst_margin["segment_id"],
        "minimum_margin_sample_index": worst_margin["minimum_margin_sample_index"],
        "minimum_margin_link_pair": worst_margin["minimum_margin_link_pair"],
        "passed": (
            worst_clearance["minimum_clearance_m"] >= -tolerance
            and worst_margin["minimum_margin_to_required_clearance_m"] >= -tolerance
        ),
        "segments": evidence,
    }


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
        link_index = int(kinematics_config.link_sphere_idx_map.reshape(-1)[selected_sphere].item())
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
    return validate_dex3_finger_sweep(replace(request, snapshot=snapshot))["sample_count"]


def validate_dex3_finger_sweep(request: Dex3PreparationRequest) -> dict[str, Any]:
    """Shared preflight/loaded-state check of the exact articulated finger sweep."""

    validate_dex3_finger_targets(
        left_q_rad=request.left_target_q_rad,
        right_q_rad=request.right_target_q_rad,
        label="finger sweep target",
    )
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
        ignore_internal_hand_collisions=True,
        ignore_adjacent_shoulder_collisions=True,
        ignore_static_body_collisions=True,
        active_joint_names=names,
        snapshot=request.snapshot,
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
    strict_checker = CuroboKinematicCollisionChecker(robot=robot, device_cfg=device_cfg)
    pair_hits = strict_checker.self_collision_pair_penetrations(
        sweep,
        joint_names=curobo_names,
    )
    colliding = np.asarray([bool(value) for value in pair_hits], dtype=bool)
    if np.any(colliding):
        first = int(np.flatnonzero(colliding)[0])
        pairs = ", ".join(
            f"{left}<->{right} ({penetration * 1000.0:.2f}mm)"
            for (left, right), penetration in sorted(pair_hits[first].items())
        )
        raise RuntimeError(f"finger sweep self-collision at sample {first}/{intervals}: {pairs}")

    clearances, link_pairs = strict_checker.self_collision_link_pair_clearances(
        sweep,
        joint_names=curobo_names,
    )
    hand_pair_indices = [
        index for index, pair in enumerate(link_pairs) if any("_hand_" in link for link in pair)
    ]
    if not hand_pair_indices:
        raise RuntimeError("finger sweep model has no external hand collision pairs")
    hand_clearances = clearances[:, hand_pair_indices]
    minimum, minimum_flat = torch.min(hand_clearances.reshape(-1), dim=0)
    minimum_clearance = float(minimum.detach().cpu())
    minimum_pair = link_pairs[hand_pair_indices[int(minimum_flat % len(hand_pair_indices))]]
    if (
        minimum_clearance
        < FINGER_SWEEP_MINIMUM_CLEARANCE_M - CALIBRATION_CLEARANCE_NUMERICAL_TOLERANCE_M
    ):
        raise RuntimeError(
            f"finger sweep clearance {minimum_clearance:.6f}m at "
            f"{minimum_pair[0]}/{minimum_pair[1]} is below the commissioned "
            f"{FINGER_SWEEP_MINIMUM_CLEARANCE_M:.3f}m floor"
        )

    limits = checker.kinematics.get_joint_limits().position.detach().cpu().numpy()
    lower, upper = limits[0], limits[1]
    if np.any(target < lower - 1e-6) or np.any(target > upper + 1e-6):
        invalid = (target < lower - 1e-6) | (target > upper + 1e-6)
        detail = "; ".join(
            f"{curobo_names[index]}={target[index]:.9f}rad outside "
            f"[{lower[index]:.9f}, {upper[index]:.9f}]rad"
            for index in np.flatnonzero(invalid)
        )
        raise Dex3FingerTargetLimitError(f"finger sweep target: {detail}")
    violations = np.maximum(np.maximum(lower[None] - sweep, sweep - upper[None]), 0.0)
    for joint_index, joint_name in enumerate(curobo_names):
        values = violations[:, joint_index]
        if values[0] <= 1e-6 and np.any(values > 1e-6):
            raise RuntimeError(f"finger sweep leaves the hard limits at {joint_name}")
        if np.any(np.diff(values) > 1e-6) or values[-1] > 1e-6:
            raise RuntimeError(
                f"finger sweep does not monotonically recover {joint_name} into limits"
            )
    del strict_checker
    del checker
    gc.collect()
    torch.cuda.empty_cache()
    return {
        "schema_version": 1,
        "operation": "validate_dex3_finger_sweep",
        "commands_robot": False,
        "request_sha256": request.content_sha256,
        "passed": True,
        "sample_count": len(sweep),
        "minimum_clearance_m": minimum_clearance,
        "minimum_clearance_link_pair": list(minimum_pair),
        "required_clearance_m": FINGER_SWEEP_MINIMUM_CLEARANCE_M,
        "model_sources": model_source_hashes(),
    }


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


def solve_bilateral_calibration_ik(
    request: BilateralCalibrationPlanningRequest,
    *,
    progress: Callable[[str], None] | None = None,
) -> BilateralIKResult:
    """Run collision-aware batched IK for both active-arm candidate pools."""

    report = progress or (lambda _message: None)
    poses: list[BilateralFeasiblePose] = []
    provenance: dict[str, Any] = {
        **model_source_hashes(),
        "self_collision_check": True,
        "collision_activation_distance_m": COLLISION_ACTIVATION_DISTANCE_M,
        "ik_batch_size": request.ik_batch_size,
        "ik_seeds": IK_SEEDS,
        "ik_return_seeds": IK_RETURN_SEEDS,
        "ik_position_tolerance_m": IK_POSITION_TOLERANCE_M,
        "ik_rotation_tolerance_rad": IK_ROTATION_TOLERANCE_RAD,
        "candidate_count_by_arm": {},
        "feasible_ik_count_by_arm": {},
        "ik_elapsed_s_by_arm": {},
    }
    devices: set[str] = set()
    for side in ("left", "right"):
        side_snapshot = request.clearance_snapshot
        candidates = request.candidates_by_arm[side]
        side_request = CalibrationPlanRequest(
            arm=side,
            snapshot=side_snapshot,
            torso_T_camera=request.nominal_torso_T_camera,
            palm_T_marker=request.nominal_hand_T_targets[side],
            joint_position_offsets_rad=request.joint_position_offsets_rad,
            candidates=candidates,
            selection_config={"selector": "bilateral_full_model_in_parent"},
            target_count=1,
            ik_batch_size=request.ik_batch_size,
            random_seed=request.random_seed,
        )
        robot, reference_tuple = build_locked_robot_config(
            arm=side,
            snapshot=side_snapshot,
            joint_position_offsets_rad=request.joint_position_offsets_rad,
        )
        feasible, elapsed_s, device_name = _batched_ik(
            request=side_request,
            robot=robot,
            reference=np.asarray(reference_tuple, dtype=np.float64),
            progress=lambda message, selected=side: report(f"{selected}: {message}"),
        )
        devices.add(device_name)
        provenance["candidate_count_by_arm"][side] = len(candidates)
        provenance["feasible_ik_count_by_arm"][side] = len(feasible)
        provenance["ik_elapsed_s_by_arm"][side] = elapsed_s
        indices = np.asarray(arm_indices(side), dtype=np.int64)
        for item in feasible:
            candidate = candidates[item.source_index]
            command_q = command_from_model_q(
                item.model_q,
                arm=side,
                joint_position_offsets_rad=request.joint_position_offsets_rad,
            )
            full_q = np.asarray(
                request.clearance_snapshot.measured_q29_rad,
                dtype=np.float64,
            ).copy()
            full_q[indices] = command_q
            poses.append(
                BilateralFeasiblePose(
                    candidate_id=candidate.candidate_id,
                    active_arm=side,
                    active_model_q_rad=tuple(item.model_q),
                    active_command_q_rad=tuple(command_q),
                    full_command_q29_rad=tuple(full_q),
                    ik_position_error_m=item.position_error_m,
                    ik_rotation_error_rad=item.rotation_error_rad,
                    candidate_metadata=candidate.selection_metadata,
                )
            )
        report(f"{side}: retained {len(feasible)}/{len(candidates)} collision-free IK poses")
    provenance["device"] = ",".join(sorted(devices))
    return BilateralIKResult(
        request_sha256=request.content_sha256,
        poses=tuple(poses),
        planner_provenance=provenance,
    )


def _nearest_neighbor_edges(
    q_by_id: dict[str, np.ndarray],
    *,
    neighbor_count: int,
) -> tuple[tuple[float, str, str], ...]:
    """Return deterministic undirected k-nearest edges."""

    node_ids = tuple(sorted(q_by_id))
    if not 1 <= neighbor_count < len(node_ids):
        raise ValueError("nearest-neighbor edge count is outside the node set")
    edges: dict[tuple[str, str], float] = {}
    for source_id in node_ids:
        source = np.asarray(q_by_id[source_id], dtype=np.float64)
        nearest = sorted(
            (
                (float(np.linalg.norm(source - np.asarray(q_by_id[target_id]))), target_id)
                for target_id in node_ids
                if target_id != source_id
            ),
            key=lambda item: (item[0], item[1]),
        )[:neighbor_count]
        for distance, target_id in nearest:
            pair = tuple(sorted((source_id, target_id)))
            edges[pair] = min(edges.get(pair, float("inf")), distance)
    return tuple(
        (distance, source_id, target_id)
        for (source_id, target_id), distance in sorted(
            edges.items(),
            key=lambda item: (item[1], item[0][0], item[0][1]),
        )
    )


def _rooted_shortest_valid_edge_tree(
    *,
    root_id: str,
    node_ids: Sequence[str],
    passed_edges: Sequence[tuple[float, str, str]],
) -> dict[str, str]:
    """Grow the same shortest-valid-edge tree used by the prior calibration."""

    nodes = {str(value) for value in node_ids}
    if root_id not in nodes:
        raise ValueError("collision-tree root is absent from its node set")
    adjacency: dict[str, list[tuple[float, str]]] = {node_id: [] for node_id in nodes}
    for distance, first, second in passed_edges:
        if first not in nodes or second not in nodes:
            raise ValueError("collision-tree edge references an unknown node")
        adjacency[first].append((float(distance), second))
        adjacency[second].append((float(distance), first))
    connected = {root_id}
    parents: dict[str, str] = {}
    heap = [(distance, root_id, target_id) for distance, target_id in adjacency[root_id]]
    heapq.heapify(heap)
    while heap:
        _distance, parent, target_id = heapq.heappop(heap)
        if target_id in connected or parent not in connected:
            continue
        connected.add(target_id)
        parents[target_id] = parent
        for distance, remaining_id in adjacency[target_id]:
            if remaining_id not in connected:
                heapq.heappush(heap, (distance, target_id, remaining_id))
    return parents


def _edge_clearance_passes(
    *,
    checker: CuroboKinematicCollisionChecker,
    joint_names: tuple[str, ...],
    required_clearance,
    edges: Sequence[tuple[float, str, str]],
    q_by_id: dict[str, np.ndarray],
    maximum_batch_samples: int = 512,
) -> tuple[tuple[float, str, str], ...]:
    """Batch straight-edge sweeps through one reusable CUDA checker."""

    import torch

    passed: list[tuple[float, str, str]] = []
    pending: list[tuple[tuple[float, str, str], np.ndarray]] = []
    pending_samples = 0

    def flush() -> None:
        nonlocal pending_samples
        if not pending:
            return
        samples = np.concatenate([item[1] for item in pending], axis=0)
        clearances, _pairs = checker.self_collision_link_pair_clearances(
            samples,
            joint_names=joint_names,
        )
        offset = 0
        tolerance = CALIBRATION_CLEARANCE_NUMERICAL_TOLERANCE_M
        for edge, edge_samples in pending:
            count = len(edge_samples)
            values = clearances[offset : offset + count]
            minimum_clearance = torch.min(values)
            minimum_margin = torch.min(values - required_clearance[None])
            if (
                float(minimum_clearance.detach().cpu()) >= -tolerance
                and float(minimum_margin.detach().cpu()) >= -tolerance
            ):
                passed.append(edge)
            offset += count
        pending.clear()
        pending_samples = 0

    for edge in edges:
        _distance, source_id, target_id = edge
        samples = sample_linear_joint_sweep(
            q_by_id[source_id],
            q_by_id[target_id],
            maximum_joint_step_rad=0.005,
        )
        if pending and pending_samples + len(samples) > maximum_batch_samples:
            flush()
        pending.append((edge, samples))
        pending_samples += len(samples)
    flush()
    return tuple(passed)


def clearance_certified_bilateral_anchor_candidates(
    request: BilateralCalibrationPlanningRequest,
    ik_result: BilateralIKResult,
) -> tuple[dict[str, tuple[str, ...]], dict[str, Any]]:
    """Filter anchor sources against the real closed-phase clearance policy."""

    import torch
    from curobo.types import DeviceCfg

    if not torch.cuda.is_available():
        raise RuntimeError("CuRobo bilateral anchor filtering requires a CUDA device")
    ik_result.validate_request(request)
    started = time.monotonic()
    device_cfg = DeviceCfg(device=torch.device("cuda:0"), dtype=torch.float32)
    active_names = (*arm_joint_names("left"), *arm_joint_names("right"))
    robot, _reference = build_robot_config_for_active_joints(
        ignore_internal_hand_collisions=True,
        ignore_adjacent_shoulder_collisions=True,
        ignore_static_body_collisions=True,
        active_joint_names=active_names,
        snapshot=request.clearance_snapshot,
        joint_position_offsets_rad=request.joint_position_offsets_rad,
    )
    checker = CuroboKinematicCollisionChecker(robot=robot, device_cfg=device_cfg)
    joint_names = tuple(checker.kinematics.joint_names)
    reference_q = _full_arm_model_reference(
        request.clearance_snapshot.measured_q29_rad,
        joint_position_offsets_rad=request.joint_position_offsets_rad,
        joint_names=joint_names,
    )
    reference_clearance, _pairs = checker.self_collision_link_pair_clearances(
        reference_q[None],
        joint_names=joint_names,
    )
    reference_clearance = reference_clearance[0]
    required = torch.full_like(reference_clearance, CALIBRATION_SELF_CLEARANCE_M)
    accepted: dict[str, tuple[str, ...]] = {}
    diagnostics: dict[str, Any] = {}
    for side in ("left", "right"):
        poses = tuple(item for item in ik_result.poses if item.active_arm == side)
        q_by_id = {"dual_shoulder_clearance": reference_q}
        edges: list[tuple[float, str, str]] = []
        for pose in poses:
            q29 = np.asarray(request.clearance_snapshot.measured_q29_rad, dtype=np.float64).copy()
            q29[np.asarray(arm_indices(side), dtype=np.int64)] = np.asarray(
                pose.active_command_q_rad,
                dtype=np.float64,
            )
            q_by_id[pose.candidate_id] = _full_arm_model_reference(
                q29,
                joint_position_offsets_rad=request.joint_position_offsets_rad,
                joint_names=joint_names,
            )
            edges.append(
                (
                    float(np.linalg.norm(q_by_id[pose.candidate_id] - reference_q)),
                    "dual_shoulder_clearance",
                    pose.candidate_id,
                )
            )
        passed_edges = _edge_clearance_passes(
            checker=checker,
            joint_names=joint_names,
            required_clearance=required,
            edges=edges,
            q_by_id=q_by_id,
        )
        passed_ids = {edge[2] for edge in passed_edges}
        accepted[side] = tuple(
            pose.candidate_id for pose in poses if pose.candidate_id in passed_ids
        )
        diagnostics[side] = {
            "ik_candidate_count": len(poses),
            "clearance_connected_candidate_count": len(accepted[side]),
        }
        if not accepted[side]:
            raise RuntimeError(
                f"no {side} IK candidate satisfies the closed-phase self-clearance policy"
            )
    del checker
    return accepted, {
        **model_source_hashes(),
        "device": torch.cuda.get_device_name(device_cfg.device),
        "elapsed_s": time.monotonic() - started,
        "policy": (
            "filter_anchor_sources_by_full_clearance_to_candidate_sweep_before_"
            "bilateral_pair_selection"
        ),
        "arms": diagnostics,
    }


def _bilateral_core_reference_clearance(
    *,
    checker: CuroboKinematicCollisionChecker,
    joint_names: tuple[str, ...],
    request: BilateralCalibrationPlanningRequest,
    pool: BilateralDesignPool,
):
    """Keep the stricter pairwise reference used by preparation or core replay."""

    references = np.stack(
        [
            _full_arm_model_reference(
                q29,
                joint_position_offsets_rad=request.joint_position_offsets_rad,
                joint_names=joint_names,
            )
            for q29 in (request.clearance_snapshot.measured_q29_rad, pool.anchor_q29_rad)
        ]
    )
    clearances, _pairs = checker.self_collision_link_pair_clearances(
        references,
        joint_names=joint_names,
    )
    # The required-clearance function is monotone in its reference gap, so
    # this enforces both existing policies without weakening either one.
    return clearances.amax(dim=0)


def connect_bilateral_design_pool(
    request: BilateralCalibrationPlanningRequest,
    pool: BilateralDesignPool,
    *,
    progress: Callable[[str], None] | None = None,
) -> BilateralRouteConnectivity:
    """Root all visible poses with batched CuRobo edge-clearance checks."""

    import torch
    from curobo.types import DeviceCfg

    if not torch.cuda.is_available():
        raise RuntimeError("CuRobo bilateral connectivity requires a CUDA device")
    report = progress or (lambda _message: None)
    started = time.monotonic()
    device_cfg = DeviceCfg(device=torch.device("cuda:0"), dtype=torch.float32)
    active_names = (*arm_joint_names("left"), *arm_joint_names("right"))
    robot, _reference = build_robot_config_for_active_joints(
        ignore_internal_hand_collisions=True,
        ignore_adjacent_shoulder_collisions=True,
        ignore_static_body_collisions=True,
        active_joint_names=active_names,
        snapshot=request.clearance_snapshot,
        joint_position_offsets_rad=request.joint_position_offsets_rad,
    )
    checker = CuroboKinematicCollisionChecker(robot=robot, device_cfg=device_cfg)
    joint_names = tuple(checker.kinematics.joint_names)
    if len(joint_names) != len(active_names) or set(joint_names) != set(active_names):
        raise RuntimeError("CuRobo bilateral connectivity joint set differs from the request")
    reference_clearance = _bilateral_core_reference_clearance(
        checker=checker,
        joint_names=joint_names,
        request=request,
        pool=pool,
    )
    required = torch.full_like(reference_clearance, CALIBRATION_SELF_CLEARANCE_M)
    parents_by_arm: dict[str, dict[str, str]] = {}
    connected_by_arm: dict[str, tuple[str, ...]] = {}
    diagnostics: dict[str, Any] = {}
    root_id = "bilateral_anchor"
    for side in ("left", "right"):
        candidate_ids = tuple(
            sorted(item.candidate_id for item in pool.candidates if item.active_arm == side)
        )
        q29_by_id = {
            root_id: pool.full_q29_rad_by_candidate_id[root_id],
            **{
                candidate_id: pool.full_q29_rad_by_candidate_id[candidate_id]
                for candidate_id in candidate_ids
            },
        }
        q_by_id = {
            candidate_id: _full_arm_model_reference(
                q29,
                joint_position_offsets_rad=request.joint_position_offsets_rad,
                joint_names=joint_names,
            )
            for candidate_id, q29 in q29_by_id.items()
        }
        passed_by_pair: dict[tuple[str, str], tuple[float, str, str]] = {}
        evaluated_pairs: set[tuple[str, str]] = set()
        parent: dict[str, str] = {}
        used_neighbors = 0
        node_count = len(q_by_id)
        required_count = getattr(request.design_config, f"{side}_excitation_count")
        desired_connected = min(len(candidate_ids), max(required_count * 3, required_count + 8))
        neighbor_counts = tuple(
            dict.fromkeys(
                min(node_count - 1, value)
                for value in (8, 16, 32, 64, node_count - 1)
                if node_count > 1
            )
        )
        previous_connected_count = -1
        for neighbor_count in neighbor_counts:
            edges = _nearest_neighbor_edges(q_by_id, neighbor_count=neighbor_count)
            new_edges = tuple(
                edge for edge in edges if tuple(sorted((edge[1], edge[2]))) not in evaluated_pairs
            )
            for edge in new_edges:
                evaluated_pairs.add(tuple(sorted((edge[1], edge[2]))))
            passed = _edge_clearance_passes(
                checker=checker,
                joint_names=joint_names,
                required_clearance=required,
                edges=new_edges,
                q_by_id=q_by_id,
            )
            for edge in passed:
                passed_by_pair[tuple(sorted((edge[1], edge[2])))] = edge
            parent = _rooted_shortest_valid_edge_tree(
                root_id=root_id,
                node_ids=tuple(q_by_id),
                passed_edges=tuple(passed_by_pair.values()),
            )
            used_neighbors = neighbor_count
            report(
                f"{side}: anchor-rooted collision tree contains {len(parent)}/"
                f"{len(candidate_ids)} visible candidates after {len(evaluated_pairs)} "
                "batched edge checks"
            )
            if len(parent) >= desired_connected:
                break
            if neighbor_count >= 16 and len(parent) == previous_connected_count:
                break
            previous_connected_count = len(parent)
        if len(parent) < required_count:
            raise RuntimeError(
                f"CuRobo rooted only {len(parent)} {side} calibration candidates at "
                f"the bilateral anchor; {required_count} required"
            )
        parents_by_arm[side] = parent
        connected_by_arm[side] = tuple(
            candidate_id for candidate_id in candidate_ids if candidate_id in parent
        )
        diagnostics[side] = {
            "visible_candidate_count": len(candidate_ids),
            "connected_candidate_count": len(parent),
            "evaluated_edge_count": len(evaluated_pairs),
            "passed_edge_count": len(passed_by_pair),
            "nearest_neighbor_count": used_neighbors,
        }
    del checker
    return BilateralRouteConnectivity(
        connected_candidate_ids_by_arm=connected_by_arm,
        parent_candidate_id_by_arm=parents_by_arm,
        provenance={
            **model_source_hashes(),
            "device": torch.cuda.get_device_name(device_cfg.device),
            "elapsed_s": time.monotonic() - started,
            "maximum_joint_step_rad": 0.005,
            "self_clearance_policy": "strict_core_10mm",
            "tree_policy": "shortest_valid_edge_tree_rooted_at_bilateral_anchor",
            "arms": diagnostics,
        },
    )


def connect_selected_bilateral_design(
    request: BilateralCalibrationPlanningRequest,
    pool: BilateralDesignPool,
    selection: BilateralDesignSelection,
    *,
    progress: Callable[[str], None] | None = None,
) -> tuple[dict[str, tuple[tuple[float, str, str], ...]], dict[str, Any]]:
    """Certify the complete selected-pose graph for efficient route search."""

    import torch
    from curobo.types import DeviceCfg

    if not torch.cuda.is_available():
        raise RuntimeError("CuRobo selected bilateral graph planning requires a CUDA device")
    report = progress or (lambda _message: None)
    started = time.monotonic()
    device_cfg = DeviceCfg(device=torch.device("cuda:0"), dtype=torch.float32)
    active_names = (*arm_joint_names("left"), *arm_joint_names("right"))
    robot, _reference = build_robot_config_for_active_joints(
        ignore_internal_hand_collisions=True,
        ignore_adjacent_shoulder_collisions=True,
        ignore_static_body_collisions=True,
        active_joint_names=active_names,
        snapshot=request.clearance_snapshot,
        joint_position_offsets_rad=request.joint_position_offsets_rad,
    )
    checker = CuroboKinematicCollisionChecker(robot=robot, device_cfg=device_cfg)
    joint_names = tuple(checker.kinematics.joint_names)
    reference_clearance = _bilateral_core_reference_clearance(
        checker=checker,
        joint_names=joint_names,
        request=request,
        pool=pool,
    )
    required = torch.full_like(reference_clearance, CALIBRATION_SELF_CLEARANCE_M)
    root_id = "bilateral_anchor"
    valid_edges_by_arm: dict[str, tuple[tuple[float, str, str], ...]] = {}
    diagnostics: dict[str, Any] = {}
    pool_ids = {item.candidate_id for item in pool.candidates}
    for side in ("left", "right"):
        selected_ids = tuple(
            item.candidate_id for item in selection.candidates if item.active_arm == side
        )
        if any(candidate_id not in pool_ids for candidate_id in selected_ids):
            raise ValueError("selected bilateral design contains a candidate outside its pool")
        q29_by_id = {
            root_id: pool.full_q29_rad_by_candidate_id[root_id],
            **{
                candidate_id: pool.full_q29_rad_by_candidate_id[candidate_id]
                for candidate_id in selected_ids
            },
        }
        q_by_id = {
            candidate_id: _full_arm_model_reference(
                q29,
                joint_position_offsets_rad=request.joint_position_offsets_rad,
                joint_names=joint_names,
            )
            for candidate_id, q29 in q29_by_id.items()
        }
        edges = _nearest_neighbor_edges(
            q_by_id,
            neighbor_count=len(q_by_id) - 1,
        )
        passed = _edge_clearance_passes(
            checker=checker,
            joint_names=joint_names,
            required_clearance=required,
            edges=edges,
            q_by_id=q_by_id,
        )
        parents = _rooted_shortest_valid_edge_tree(
            root_id=root_id,
            node_ids=tuple(q_by_id),
            passed_edges=passed,
        )
        if set(parents) != set(selected_ids):
            disconnected = sorted(set(selected_ids) - set(parents))
            raise RuntimeError(
                f"selected {side} bilateral design is not graph-connected: "
                + ", ".join(disconnected)
            )
        valid_edges_by_arm[side] = tuple(
            (
                float(np.max(np.abs(q_by_id[first] - q_by_id[second]))),
                first,
                second,
            )
            for _distance, first, second in passed
        )
        diagnostics[side] = {
            "selected_candidate_count": len(selected_ids),
            "evaluated_edge_count": len(edges),
            "passed_edge_count": len(passed),
        }
        report(f"{side}: selected valid-edge graph connected all {len(selected_ids)} poses")
    del checker
    return valid_edges_by_arm, {
        **model_source_hashes(),
        "device": torch.cuda.get_device_name(device_cfg.device),
        "elapsed_s": time.monotonic() - started,
        "maximum_joint_step_rad": 0.005,
        "graph_policy": "complete_selected_graph_with_maximum_joint_delta_edge_cost",
        "self_clearance_policy": "strict_core_10mm",
        "arms": diagnostics,
    }


def plan_bilateral_calibration_adapter(
    request: BilateralCalibrationAdapterRequest,
    *,
    progress: Callable[[str], None] | None = None,
) -> BilateralCalibrationAdapterPlan:
    """Plan only the live Ready-to-fixed-anchor reversible boundary adapter."""

    import torch
    from curobo.types import DeviceCfg

    if not torch.cuda.is_available():
        raise RuntimeError("CuRobo bilateral adapter planning requires a CUDA device")
    report = progress or (lambda _message: None)
    started = time.monotonic()
    arm_index_array = np.asarray(
        (*arm_indices("left"), *arm_indices("right")),
        dtype=np.int64,
    )
    locked_indices = np.asarray(
        [index for index in range(29) if index not in set(arm_index_array)],
        dtype=np.int64,
    )
    live_q29 = np.asarray(request.snapshot.measured_q29_rad, dtype=np.float64)
    anchor_q29 = np.asarray(request.anchor_q29_rad, dtype=np.float64)
    locked_error = float(np.max(np.abs(live_q29[locked_indices] - anchor_q29[locked_indices])))
    # The reusable core freezes arm commands, not the old Ready leg/waist pose.
    # Plan the boundary with today's complete measured snapshot and certify
    # every core segment with that same live body geometry below. The old/live
    # difference is retained as provenance, not used as a readiness tolerance.

    preparation_request = Dex3PreparationRequest(
        snapshot=request.snapshot,
        joint_position_offsets_rad=request.joint_position_offsets_rad,
        left_target_q_rad=request.left_close_command_q_rad,
        right_target_q_rad=request.right_close_command_q_rad,
        left_settled_target_q_rad=request.left_close_model_q_rad,
        right_settled_target_q_rad=request.right_close_model_q_rad,
        left_return_target_q_rad=request.snapshot.left_dex3_q_rad,
        right_return_target_q_rad=request.snapshot.right_dex3_q_rad,
        random_seed=request.random_seed,
    )
    report("planning live Ready-to-shoulder-clearance preparation")
    preparation = plan_dex3_preparation(preparation_request, progress=report)
    clearance_q29 = live_q29.copy()
    clearance_q14 = np.asarray(preparation.dual_clearance_q14_rad, dtype=np.float64)
    clearance_q29[np.asarray(arm_indices("left"), dtype=np.int64)] = clearance_q14[:7]
    clearance_q29[np.asarray(arm_indices("right"), dtype=np.int64)] = clearance_q14[7:]
    clearance_snapshot = RobotSnapshot(
        measured_q29_rad=tuple(clearance_q29),
        left_dex3_q_rad=request.left_close_model_q_rad,
        right_dex3_q_rad=request.right_close_model_q_rad,
    )
    device_cfg = DeviceCfg(device=torch.device("cuda:0"), dtype=torch.float32)

    def model_target(side: str) -> np.ndarray:
        indices = np.asarray(arm_indices(side), dtype=np.int64)
        command = anchor_q29[indices]
        return np.asarray(
            [
                value + request.joint_position_offsets_rad.get(name, 0.0)
                for name, value in zip(arm_joint_names(side), command, strict=True)
            ],
            dtype=np.float64,
        )

    report("planning right shoulder-clearance to fixed visual anchor")
    right_outbound, right_diagnostic = _plan_bilateral_preparation_edge(
        joint_position_offsets_rad=request.joint_position_offsets_rad,
        random_seed=request.random_seed,
        device_cfg=device_cfg,
        snapshot=clearance_snapshot,
        side="right",
        source_id="dual_shoulder_clearance",
        target_id="right_anchor_preparation",
        target_q=model_target("right"),
    )
    if right_outbound is None:
        raise RuntimeError("right live anchor adapter failed: " + right_diagnostic)
    right_anchor_q29 = clearance_q29.copy()
    right_anchor_q29[np.asarray(arm_indices("right"), dtype=np.int64)] = anchor_q29[
        np.asarray(arm_indices("right"), dtype=np.int64)
    ]
    right_anchor_snapshot = RobotSnapshot(
        measured_q29_rad=tuple(right_anchor_q29),
        left_dex3_q_rad=request.left_close_model_q_rad,
        right_dex3_q_rad=request.right_close_model_q_rad,
    )
    report("planning left shoulder-clearance to fixed visual anchor")
    left_outbound, left_diagnostic = _plan_bilateral_preparation_edge(
        joint_position_offsets_rad=request.joint_position_offsets_rad,
        random_seed=request.random_seed,
        device_cfg=device_cfg,
        snapshot=right_anchor_snapshot,
        side="left",
        source_id="right_anchor_preparation",
        target_id=HANDOFF_POSE_ID,
        target_q=model_target("left"),
    )
    if left_outbound is None:
        raise RuntimeError("left live anchor adapter failed: " + left_diagnostic)
    live_anchor_q29 = anchor_q29.copy()
    live_anchor_q29[locked_indices] = live_q29[locked_indices]
    live_anchor_snapshot = RobotSnapshot(
        measured_q29_rad=tuple(live_anchor_q29),
        left_dex3_q_rad=request.left_close_model_q_rad,
        right_dex3_q_rad=request.right_close_model_q_rad,
    )
    core_segments = []
    for item in request.core_transitions:
        side = str(item["arm"])
        trajectory = PlannedTrajectory.from_dict(item["trajectory"])
        source_q29 = live_anchor_q29.copy()
        source_q29[np.asarray(arm_indices(side), dtype=np.int64)] = np.asarray(
            trajectory.command_q_rad[0],
            dtype=np.float64,
        )
        core_segments.append(
            (
                f"{trajectory.from_pose_id}->{trajectory.to_pose_id}",
                side,
                trajectory,
                tuple(source_q29),
            )
        )
    report("batch-validating the reusable core under the live locked-body posture")
    live_core_clearance = _certify_bilateral_arm_segments(
        snapshot=live_anchor_snapshot,
        reference_q29=tuple(live_anchor_q29),
        joint_position_offsets_rad=request.joint_position_offsets_rad,
        segments=tuple(core_segments),
        device_cfg=device_cfg,
        phase="live_locked_body_closed_core",
    )
    live_core_certificate = _combine_self_clearance_certificates((live_core_clearance,))
    if not live_core_certificate["passed"]:
        raise ValueError(
            "reusable calibration core failed live-body self-clearance replay: "
            f"minimum margin "
            f"{live_core_certificate['minimum_margin_to_required_clearance_m']:.6f}m"
        )
    left_return = _rename_trajectory(
        _reverse_trajectory(left_outbound),
        from_pose_id=HANDOFF_POSE_ID,
        to_pose_id="right_anchor_preparation",
    )
    right_return = _rename_trajectory(
        _reverse_trajectory(right_outbound),
        from_pose_id="right_anchor_preparation",
        to_pose_id="dual_shoulder_clearance",
    )
    anchor_q14 = np.concatenate(
        (
            anchor_q29[np.asarray(arm_indices("left"), dtype=np.int64)],
            anchor_q29[np.asarray(arm_indices("right"), dtype=np.int64)],
        )
    )
    return BilateralCalibrationAdapterPlan(
        request_sha256=request.content_sha256,
        preparation=preparation,
        right_anchor_outbound=right_outbound,
        left_anchor_outbound=left_outbound,
        left_anchor_return=left_return,
        right_anchor_return=right_return,
        anchor_q14_rad=tuple(anchor_q14),
        maximum_locked_joint_error_rad_observed=locked_error,
        live_core_self_clearance_certificate=live_core_certificate,
        planner_provenance={
            **model_source_hashes(),
            "device": torch.cuda.get_device_name(device_cfg.device),
            "elapsed_s": time.monotonic() - started,
            "execution_plan_sha256": request.execution_plan_sha256,
            "policy": "live_ready_clearance_fixed_anchor_exact_reverse",
            "locked_body_policy": "measured_live_body_with_full_core_clearance_replay",
            "hand_return_policy": "restore_measured_start_before_reversing_shoulders",
        },
    )


def plan_bilateral_calibration_route(
    request: BilateralRoutePlanningRequest,
    *,
    progress: Callable[[str], None] | None = None,
) -> BilateralRoutePlanningResult:
    """Freeze the already-selected valid-graph traversal and preparation."""

    import torch
    from curobo.types import DeviceCfg

    if not torch.cuda.is_available():
        raise RuntimeError("CuRobo bilateral route planning requires a CUDA device")
    report = progress or (lambda _message: None)
    device_cfg = DeviceCfg(device=torch.device("cuda:0"), dtype=torch.float32)
    planner_robots: dict[str, dict] = {}
    strict_checkers: dict[str, CuroboKinematicCollisionChecker] = {}
    planning_started = time.monotonic()
    try:
        active_arm_names = (*arm_joint_names("left"), *arm_joint_names("right"))
        anchor_robot, anchor_reference = build_robot_config_for_active_joints(
            ignore_internal_hand_collisions=True,
            ignore_adjacent_shoulder_collisions=True,
            ignore_static_body_collisions=True,
            active_joint_names=active_arm_names,
            snapshot=request.calibration_snapshot,
            joint_position_offsets_rad=request.joint_position_offsets_rad,
        )
        anchor_checker = CuroboKinematicCollisionChecker(
            robot=anchor_robot,
            device_cfg=device_cfg,
        )
        anchor_by_name = dict(zip(active_arm_names, anchor_reference, strict=True))
        checker_arm_names = tuple(anchor_checker.kinematics.joint_names)
        if len(checker_arm_names) != len(active_arm_names) or set(checker_arm_names) != set(
            active_arm_names
        ):
            raise RuntimeError("CuRobo bilateral active-arm set differs from the request")
        ordered_anchor_reference = np.asarray(
            [anchor_by_name[name] for name in checker_arm_names],
            dtype=np.float64,
        )
        anchor_collisions = anchor_checker.self_collision_pair_penetrations(
            ordered_anchor_reference[None],
            joint_names=checker_arm_names,
        )[0]
        del anchor_checker
        if anchor_collisions:
            rejected_anchor_ids = tuple(sorted(request.anchor_candidate_ids_by_arm.values()))
            return BilateralRoutePlanningResult(
                request_sha256=request.content_sha256,
                transitions=(),
                disconnected_candidate_ids=rejected_anchor_ids,
                finger_sweep_sample_count=0,
                restoration_sweep_sample_count=0,
                planner_provenance={
                    **model_source_hashes(),
                    "device": torch.cuda.get_device_name(device_cfg.device),
                    "route_elapsed_s": time.monotonic() - planning_started,
                    "anchor_collision_pairs": [
                        {"links": list(pair), "penetration_m": penetration}
                        for pair, penetration in sorted(anchor_collisions.items())
                    ],
                    "route_policy": "reject_and_reselect_colliding_bilateral_anchor",
                },
            )

        finger_sweep_sample_count = request.dex3_preparation_plan.finger_sweep_sample_count
        restoration_sweep_sample_count = request.dex3_preparation_plan.return_sweep_sample_count
        report("CuRobo is independently replaying the closed-hand calibration core")

        # The design phase already certified the complete selected-pose edge
        # graphs and chose short anchor-to-anchor walks. Recheck only those
        # scheduled graph edges here so the frozen artifact is independently
        # certified without constructing trajectory optimizers per capture.
        for side in ("left", "right"):
            robot, _reference_tuple = build_locked_robot_config(
                arm=side,
                snapshot=request.calibration_snapshot,
                joint_position_offsets_rad=request.joint_position_offsets_rad,
            )
            planner_robots[side] = robot
            strict_checkers[side] = CuroboKinematicCollisionChecker(
                robot=robot,
                device_cfg=device_cfg,
            )

        if (
            request.schedule[0].capture_role != "anchor"
            or request.schedule[-1].capture_role != "anchor"
            or any(item.hand_action is not None for item in request.schedule)
        ):
            raise ValueError("bilateral route is not a closed-hand anchor-to-anchor core")

        transitions: list[BilateralPlannedTransition] = []
        for edge_index in range(len(request.schedule) - 1):
            start = request.schedule[edge_index]
            end = request.schedule[edge_index + 1]
            side = request.transition_arm(edge_index)
            trajectory, diagnostics = _strict_linear_joint_trajectory(
                robot=planner_robots[side],
                device_cfg=device_cfg,
                source_q=_route_model_q(request, start.candidate_id, side),
                target_q=_route_model_q(request, end.candidate_id, side),
                arm=side,
                source_id=start.occurrence_id,
                target_id=end.occurrence_id,
                joint_position_offsets_rad=request.joint_position_offsets_rad,
                checker=strict_checkers[side],
            )
            if trajectory is None:
                rejected = tuple(
                    sorted(
                        {
                            candidate_id
                            for candidate_id in (start.candidate_id, end.candidate_id)
                            if candidate_id
                            not in {
                                "bilateral_anchor",
                            }
                        }
                    )
                ) or tuple(sorted(request.anchor_candidate_ids_by_arm.values()))
                return BilateralRoutePlanningResult(
                    request_sha256=request.content_sha256,
                    transitions=(),
                    disconnected_candidate_ids=rejected,
                    finger_sweep_sample_count=0,
                    restoration_sweep_sample_count=0,
                    planner_provenance={
                        **model_source_hashes(),
                        "device": torch.cuda.get_device_name(device_cfg.device),
                        "route_elapsed_s": time.monotonic() - planning_started,
                        "graph_edge_rejection": {
                            "transition": f"{start.occurrence_id}->{end.occurrence_id}",
                            "reason": diagnostics,
                        },
                        "route_policy": "reject_if_frozen_valid_graph_replay_changes",
                    },
                )
            transitions.append(BilateralPlannedTransition(arm=side, trajectory=trajectory))
            report(
                f"bilateral route {edge_index + 1}/{len(request.schedule) - 1}: "
                f"{start.occurrence_id}->{end.occurrence_id}; "
                f"duration={trajectory.sample_time_s[-1]:.2f}s"
            )
        closed_segments = tuple(
            (
                transition.transition_id,
                transition.arm,
                transition.trajectory,
                request.waypoint_joint_positions_rad[request.schedule[index].candidate_id],
            )
            for index, transition in enumerate(transitions)
        )
        closed_clearance = _certify_bilateral_arm_segments(
            snapshot=request.calibration_snapshot,
            reference_q29=request.waypoint_joint_positions_rad[request.schedule[0].candidate_id],
            joint_position_offsets_rad=request.joint_position_offsets_rad,
            segments=closed_segments,
            device_cfg=device_cfg,
            phase="closed_hands",
        )
        clearance_certificate = _combine_self_clearance_certificates((closed_clearance,))
        if not clearance_certificate["passed"]:
            rejected_candidate_ids = _route_clearance_rejection_candidate_ids(
                request,
                transitions,
                clearance_certificate,
            )
            return BilateralRoutePlanningResult(
                request_sha256=request.content_sha256,
                transitions=(),
                disconnected_candidate_ids=rejected_candidate_ids,
                finger_sweep_sample_count=0,
                restoration_sweep_sample_count=0,
                planner_provenance={
                    **model_source_hashes(),
                    "device": torch.cuda.get_device_name(device_cfg.device),
                    "route_elapsed_s": time.monotonic() - planning_started,
                    "self_clearance_certificate": clearance_certificate,
                    "rejected_candidate_or_anchor_ids": list(rejected_candidate_ids),
                    "route_policy": "reject_and_reselect_self_clearance_violation",
                },
            )
        return BilateralRoutePlanningResult(
            request_sha256=request.content_sha256,
            transitions=tuple(transitions),
            disconnected_candidate_ids=(),
            finger_sweep_sample_count=finger_sweep_sample_count,
            restoration_sweep_sample_count=restoration_sweep_sample_count,
            planner_provenance={
                **model_source_hashes(),
                "device": torch.cuda.get_device_name(device_cfg.device),
                "route_elapsed_s": time.monotonic() - planning_started,
                "route_edge_count": len(transitions),
                "trajectory_interpolation_dt_s": TRAJECTORY_INTERPOLATION_DT_S,
                "execution_maximum_velocity_rad_s": EXECUTION_MAXIMUM_VELOCITY_RAD_S,
                "self_collision_check": True,
                "collision_activation_distance_m": COLLISION_ACTIVATION_DISTANCE_M,
                "self_clearance_certificate": clearance_certificate,
                "mounted_plate_spheres_per_hand": 30,
                "route_policy": (
                    "fixed_closed_anchor; minimum_motion_cost_valid_graph_walks_with_"
                    "interleaved_capture_blocks; return_to_identical_anchor"
                ),
            },
        )
    finally:
        strict_checkers.clear()
        planner_robots.clear()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def _plan_bilateral_preparation_edge(
    *,
    joint_position_offsets_rad: dict[str, float],
    random_seed: int,
    device_cfg,
    snapshot: RobotSnapshot,
    side: str,
    source_id: str,
    target_id: str,
    target_q: np.ndarray,
) -> tuple[PlannedTrajectory | None, str]:
    """Plan one Ready-envelope arm edge, including measured-contact recovery."""

    from curobo.motion_planner import MotionPlanner, MotionPlannerCfg

    robot, reference_values = build_locked_robot_config(
        arm=side,
        snapshot=snapshot,
        joint_position_offsets_rad=joint_position_offsets_rad,
    )
    reference = np.asarray(reference_values, dtype=np.float64)
    source_collision = _self_collision_pair_penetrations(
        robot=robot,
        q_samples=reference[None],
        device_cfg=device_cfg,
    )[0]
    if source_collision:
        try:
            return _monotonic_collision_recovery_trajectory(
                robot=robot,
                device_cfg=device_cfg,
                source_q=reference,
                target_q=np.asarray(target_q, dtype=np.float64),
                arm=side,
                source_id=source_id,
                target_id=target_id,
                joint_position_offsets_rad=joint_position_offsets_rad,
                source_collision=source_collision,
            )
        except (RuntimeError, ValueError) as error:
            return None, str(error)
    planner = MotionPlanner(
        MotionPlannerCfg.create(
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
    )
    try:
        return _plan_edge(
            planner=planner,
            robot=robot,
            device_cfg=device_cfg,
            names=list(arm_joint_names(side)),
            arm=side,
            source_id=source_id,
            target_id=target_id,
            source_q=reference,
            target_q=np.asarray(target_q, dtype=np.float64),
            joint_position_offsets_rad=joint_position_offsets_rad,
        )
    finally:
        planner.destroy()


def _route_model_q(
    request: BilateralRoutePlanningRequest,
    candidate_id: str,
    side: str,
) -> np.ndarray:
    full_q = np.asarray(request.waypoint_joint_positions_rad[candidate_id], dtype=np.float64)
    command_q = full_q[np.asarray(arm_indices(side), dtype=np.int64)]
    return np.asarray(
        [
            value + request.joint_position_offsets_rad.get(name, 0.0)
            for name, value in zip(arm_joint_names(side), command_q, strict=True)
        ],
        dtype=np.float64,
    )


def _full_arm_model_reference(
    q29: Sequence[float],
    *,
    joint_position_offsets_rad: dict[str, float],
    joint_names: tuple[str, ...],
) -> np.ndarray:
    command = np.asarray(q29, dtype=np.float64)
    by_name: dict[str, float] = {}
    for side in ("left", "right"):
        side_q = command[np.asarray(arm_indices(side), dtype=np.int64)]
        for name, value in zip(arm_joint_names(side), side_q, strict=True):
            by_name[name] = float(value + joint_position_offsets_rad.get(name, 0.0))
    return np.asarray([by_name[name] for name in joint_names], dtype=np.float64)


def _full_arm_trajectory_samples(
    *,
    start_q29: Sequence[float],
    arm: str,
    trajectory: PlannedTrajectory,
    joint_position_offsets_rad: dict[str, float],
    joint_names: tuple[str, ...],
) -> np.ndarray:
    reference = _full_arm_model_reference(
        start_q29,
        joint_position_offsets_rad=joint_position_offsets_rad,
        joint_names=joint_names,
    )
    active = np.asarray(trajectory.model_q_rad, dtype=np.float64)
    samples = np.repeat(reference[None], len(active), axis=0)
    active_indices = [joint_names.index(name) for name in arm_joint_names(arm)]
    samples[:, active_indices] = active
    return samples


def _certify_bilateral_arm_segments(
    *,
    snapshot: RobotSnapshot,
    reference_q29: Sequence[float],
    joint_position_offsets_rad: dict[str, float],
    segments: Sequence[tuple[str, str, PlannedTrajectory, Sequence[float]]],
    device_cfg,
    phase: str,
) -> dict[str, Any]:
    active_names = (*arm_joint_names("left"), *arm_joint_names("right"))
    robot, _reference = build_robot_config_for_active_joints(
        ignore_internal_hand_collisions=True,
        ignore_adjacent_shoulder_collisions=True,
        ignore_static_body_collisions=True,
        active_joint_names=active_names,
        snapshot=snapshot,
        joint_position_offsets_rad=joint_position_offsets_rad,
    )
    checker = CuroboKinematicCollisionChecker(robot=robot, device_cfg=device_cfg)
    joint_names = tuple(checker.kinematics.joint_names)
    if len(joint_names) != len(active_names) or set(joint_names) != set(active_names):
        raise RuntimeError("CuRobo bilateral clearance joint set differs from the request")
    certificate = _self_clearance_certificate(
        checker=checker,
        joint_names=joint_names,
        reference_q=_full_arm_model_reference(
            reference_q29,
            joint_position_offsets_rad=joint_position_offsets_rad,
            joint_names=joint_names,
        ),
        segments=tuple(
            (
                segment_id,
                _full_arm_trajectory_samples(
                    start_q29=start_q29,
                    arm=arm,
                    trajectory=trajectory,
                    joint_position_offsets_rad=joint_position_offsets_rad,
                    joint_names=joint_names,
                ),
            )
            for segment_id, arm, trajectory, start_q29 in segments
        ),
        phase=phase,
        preparation=phase in {"ready_hand_preparation", "return_hand_shoulder_preparation"},
    )
    del checker
    return certificate


def _combine_self_clearance_certificates(
    certificates: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    if not certificates:
        raise ValueError("combined self-clearance certificate requires phases")
    minimum = min(certificates, key=lambda item: item["minimum_clearance_m"])
    margin = min(
        certificates,
        key=lambda item: item["minimum_margin_to_required_clearance_m"],
    )
    return {
        "policy": (
            "strict_core_10mm"
            if all(item["policy"] == "strict_core_10mm" for item in certificates)
            else "strict_core_10mm_with_separate_ready_preparation"
        ),
        "hard_clearance_m": min(item["hard_clearance_m"] for item in certificates),
        "preexisting_clearance_maximum_degradation_m": max(
            item["preexisting_clearance_maximum_degradation_m"] for item in certificates
        ),
        "minimum_clearance_m": minimum["minimum_clearance_m"],
        "minimum_clearance_phase": minimum["phase"],
        "minimum_clearance_segment_id": minimum["minimum_clearance_segment_id"],
        "minimum_clearance_sample_index": minimum["minimum_clearance_sample_index"],
        "minimum_clearance_link_pair": minimum["minimum_clearance_link_pair"],
        "minimum_margin_to_required_clearance_m": margin["minimum_margin_to_required_clearance_m"],
        "minimum_margin_phase": margin["phase"],
        "minimum_margin_segment_id": margin["minimum_margin_segment_id"],
        "minimum_margin_sample_index": margin["minimum_margin_sample_index"],
        "minimum_margin_link_pair": margin["minimum_margin_link_pair"],
        "passed": all(item["passed"] for item in certificates),
        "phases": list(certificates),
    }


def _route_clearance_rejection_candidate_ids(
    request: BilateralRoutePlanningRequest,
    transitions: Sequence[BilateralPlannedTransition],
    certificate: dict[str, Any],
) -> tuple[str, ...]:
    segment_id = str(certificate["minimum_margin_segment_id"])
    edge_index = next(
        (
            index
            for index, transition in enumerate(transitions)
            if transition.transition_id == segment_id
        ),
        None,
    )
    if edge_index is None:
        raise RuntimeError("self-clearance certificate names an unknown transition")
    transition = transitions[edge_index]
    selected_ids = {item.candidate_id for item in request.selection.candidates}
    endpoint_ids = {
        request.schedule[edge_index].candidate_id,
        request.schedule[edge_index + 1].candidate_id,
    }
    rejected = sorted(endpoint_ids & selected_ids)
    if not rejected:
        rejected = [request.anchor_candidate_ids_by_arm[transition.arm]]
    return tuple(rejected)


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
            robot=robot,
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
            robot=robot,
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
    robot: dict,
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
        optimizer_diagnostic = (
            "no planner result"
            if result is None
            else (
                f"success={result.success.detach().cpu().tolist()}, "
                f"position_error="
                f"{None if result.position_error is None else result.position_error.detach().cpu().tolist()}, "
                f"rotation_error="
                f"{None if result.rotation_error is None else result.rotation_error.detach().cpu().tolist()}"
            )
        )
        direct, direct_diagnostic = _strict_linear_joint_trajectory(
            robot=robot,
            device_cfg=device_cfg,
            source_q=source_q,
            target_q=target_q,
            arm=arm,
            source_id=source_id,
            target_id=target_id,
            joint_position_offsets_rad=joint_position_offsets_rad,
        )
        if direct is not None:
            return (
                direct,
                f"strict linear fallback after optimizer failure: {optimizer_diagnostic}",
            )
        return None, f"{optimizer_diagnostic}; {direct_diagnostic}"
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


def _strict_linear_joint_trajectory(
    *,
    robot: dict,
    device_cfg,
    source_q: np.ndarray,
    target_q: np.ndarray,
    arm: str,
    source_id: str,
    target_id: str,
    joint_position_offsets_rad: dict[str, float],
    checker: CuroboKinematicCollisionChecker | None = None,
) -> tuple[PlannedTrajectory | None, str]:
    """Certify a deterministic straight edge with CuRobo's strict checker."""

    maximum_step_rad = 0.005
    sample_count = max(
        int(np.ceil(np.max(np.abs(target_q - source_q)) / maximum_step_rad)) + 1,
        2,
    )
    alpha = np.linspace(0.0, 1.0, sample_count, dtype=np.float64)[:, None]
    model_q = source_q[None] + alpha * (target_q - source_q)[None]
    collisions = _self_collision_pair_penetrations(
        robot=robot,
        q_samples=model_q,
        device_cfg=device_cfg,
        checker=checker,
    )
    first_collision = next(
        ((index, values) for index, values in enumerate(collisions) if values),
        None,
    )
    if first_collision is not None:
        index, values = first_collision
        return None, (
            f"strict linear edge self-collides at sample {index}/{sample_count - 1}: {values}"
        )
    maximum_delta = float(np.max(np.abs(target_q - source_q)))
    duration_s = max(maximum_delta / EXECUTION_MAXIMUM_VELOCITY_RAD_S, 1e-6)
    times = np.linspace(0.0, duration_s, sample_count, dtype=np.float64)
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
            sample_time_s=tuple(float(value) for value in times),
            command_q_rad=tuple(tuple(float(item) for item in row) for row in command_q),
            model_q_rad=tuple(tuple(float(item) for item in row) for row in model_q),
            planning_time_s=0.0,
        ),
        "strict linear edge passed",
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
