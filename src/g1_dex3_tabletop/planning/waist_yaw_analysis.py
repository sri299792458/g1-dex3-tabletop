"""Read-only CuRobo comparison of locked and bounded waist-yaw planning.

This module deliberately stops at collision-constrained pregrasp IK.  Its
eight-coordinate results are analysis artifacts, not controller trajectories.
That boundary prevents an uncommissioned waist command from entering the
existing seven-arm-joint execution path.
"""

from __future__ import annotations

import copy
import gc
import time
from pathlib import Path
from typing import Any

import numpy as np

from g1_dex3_tabletop.planning.contracts import atomic_write_json
from g1_dex3_tabletop.planning.curobo_backend import (
    IK_SEEDS,
    CuroboKinematicCollisionChecker,
)
from g1_dex3_tabletop.planning.g1_model import (
    CUROBO_COMMIT,
    WAIST_YAW_JOINT_NAME,
    build_tabletop_robot_config,
    model_source_hashes,
    tabletop_motion_joint_names,
)
from g1_dex3_tabletop.planning.tabletop_planner import (
    _as_numpy,
    _base_scene,
    _base_T_torso,
    _candidate_transform,
    _contact_links,
    _goalset,
    _joint_state,
    _load_shortlist,
    _local_plane_clearance_from_spheres,
    _planner,
    _pregrasp_matrix,
    _selected_open_transit_world_robot,
    _table_from_resting_object,
    _use_moving_grasp_frame_only,
    _world_cuboid_clearances,
)
from g1_dex3_tabletop.tabletop_contracts import TabletopTaskRequest


def _validated_ranges(values: tuple[float, ...]) -> tuple[float, ...]:
    ranges = tuple(float(value) for value in values)
    if not ranges:
        raise ValueError("waist-yaw analysis requires at least one bounded range")
    if any(not np.isfinite(value) or value <= 0.0 for value in ranges):
        raise ValueError("waist-yaw half ranges must be positive and finite")
    if tuple(sorted(set(ranges))) != ranges:
        raise ValueError("waist-yaw half ranges must be unique and increasing")
    return ranges


def _resolve_robot(
    robot: dict[str, Any],
    *,
    device_cfg,
    start_model_waist_yaw_rad: float,
    waist_half_range_rad: float | None,
):
    from curobo._src.types.robot import RobotCfg

    resolved = RobotCfg.create(robot, device_cfg)
    config = resolved.kinematics.kinematics_config
    names = tuple(config.cspace.joint_names)
    limits = config.joint_limits.position
    hard_lower = hard_upper = applied_lower = applied_upper = None
    if waist_half_range_rad is not None:
        if WAIST_YAW_JOINT_NAME not in names:
            raise RuntimeError("waist-yaw study model did not expose waist yaw")
        index = names.index(WAIST_YAW_JOINT_NAME)
        hard_lower = float(limits[0, index])
        hard_upper = float(limits[1, index])
        applied_lower = max(hard_lower, start_model_waist_yaw_rad - waist_half_range_rad)
        applied_upper = min(hard_upper, start_model_waist_yaw_rad + waist_half_range_rad)
        if not applied_lower < applied_upper:
            raise RuntimeError("measured waist yaw lies outside the CuRobo hard limits")
        limits[0, index] = applied_lower
        limits[1, index] = applied_upper
    return resolved, names, {
        "hard_lower_rad": hard_lower,
        "hard_upper_rad": hard_upper,
        "applied_lower_rad": applied_lower,
        "applied_upper_rad": applied_upper,
    }


def _solve_configuration(
    *,
    request: TabletopTaskRequest,
    open_q: tuple[float, ...],
    candidate_ids: list[str],
    pregrasp_matrices: list[np.ndarray],
    planning_scene: dict[str, Any],
    strict_cube_scene: dict[str, Any],
    plane_point: np.ndarray,
    down: np.ndarray,
    waist_half_range_rad: float | None,
) -> dict[str, Any]:
    import torch
    from curobo.types import DeviceCfg

    include_waist = waist_half_range_rad is not None
    requested_names = tabletop_motion_joint_names(
        request.arm,
        include_waist_yaw=include_waist,
    )
    strict_robot, reference_values = build_tabletop_robot_config(
        arm=request.arm,
        snapshot=request.planning_snapshot,
        joint_position_offsets_rad=request.joint_position_offsets_rad,
        active_finger_q_rad=open_q,
        include_waist_yaw=include_waist,
    )
    reference_by_name = dict(zip(requested_names, reference_values, strict=True))
    transit_robot = _selected_open_transit_world_robot(
        copy.deepcopy(strict_robot),
        arm=request.arm,
    )
    _use_moving_grasp_frame_only(transit_robot, arm=request.arm)
    device_cfg = DeviceCfg(device=torch.device("cuda:0"), dtype=torch.float32)
    measured_waist_command = float(request.planning_snapshot.measured_q29_rad[12])
    start_model_waist = float(
        reference_by_name.get(WAIST_YAW_JOINT_NAME, measured_waist_command)
    )
    resolved, solver_names, limit_record = _resolve_robot(
        transit_robot,
        device_cfg=device_cfg,
        start_model_waist_yaw_rad=start_model_waist,
        waist_half_range_rad=waist_half_range_rad,
    )
    if set(solver_names) != set(requested_names):
        raise RuntimeError(
            "CuRobo active joints differ from the requested waist/arm coordinates: "
            f"requested={requested_names}, actual={solver_names}"
        )
    start = np.asarray([reference_by_name[name] for name in solver_names], dtype=np.float64)
    planner = None
    started = time.monotonic()
    try:
        planner, planner_device = _planner(
            resolved,
            planning_scene,
            max_goalset=len(candidate_ids),
            seed=request.random_seed,
        )
        goals = _goalset(pregrasp_matrices, planner_device, arm=request.arm)
        result = planner.ik_solver.solve_pose(
            goals,
            return_seeds=IK_SEEDS,
            current_state=_joint_state(planner_device, start, solver_names),
        )
        raw_success = _as_numpy(result.success).astype(bool).reshape(-1)
        solutions = _as_numpy(result.solution).reshape(len(raw_success), -1)
        if solutions.shape[1] != len(solver_names):
            raise RuntimeError(f"CuRobo waist-yaw IK returned invalid shape {solutions.shape}")
        if result.goalset_index is None:
            goal_indices = np.zeros(len(raw_success), dtype=np.int64)
        else:
            goal_indices = _as_numpy(result.goalset_index).reshape(len(raw_success), -1)[:, 0]

        successful_indices = np.flatnonzero(raw_success)
        if len(successful_indices) == 0:
            return {
                "mode": "waist_yaw_bounded" if include_waist else "waist_yaw_locked",
                "waist_yaw_half_range_rad": waist_half_range_rad,
                "active_joint_names": list(solver_names),
                "measured_waist_yaw_command_rad": measured_waist_command,
                "model_start_waist_yaw_rad": start_model_waist,
                "waist_yaw_limits": limit_record,
                "ik_seed_count": len(raw_success),
                "raw_collision_constrained_success_count": 0,
                "strictly_valid_solution_count": 0,
                "strictly_valid_candidate_count": 0,
                "strictly_valid_candidate_ids": [],
                "strict_rejection_counts": {
                    "strict_self_collision": 0,
                    "strict_cube_collision": 0,
                    "table_plane": 0,
                },
                "solutions": [],
                "elapsed_s": time.monotonic() - started,
            }
        strict_checker = CuroboKinematicCollisionChecker(
            robot=strict_robot,
            device_cfg=planner_device,
        )
        successful_q = solutions[successful_indices]
        self_collisions = strict_checker.self_collision_pair_penetrations(
            successful_q,
            joint_names=solver_names,
        )
        world_collisions = _world_cuboid_clearances(
            robot=strict_robot,
            q_samples=successful_q,
            scene=strict_cube_scene,
            device_cfg=planner_device,
            disabled_links=set(_contact_links(request.arm)),
            checker=strict_checker,
        )
        spheres = (
            strict_checker.robot_spheres(successful_q, joint_names=solver_names)
            .detach()
            .cpu()
            .numpy()
            .reshape(len(successful_q), -1, 4)
        )
        resolved_limits = resolved.kinematics.kinematics_config.joint_limits.position
        lower = resolved_limits[0].detach().cpu().numpy()
        upper = resolved_limits[1].detach().cpu().numpy()
        arm_solver_indices = [
            solver_names.index(name)
            for name in tabletop_motion_joint_names(request.arm, include_waist_yaw=False)
        ]
        waist_solver_index = (
            solver_names.index(WAIST_YAW_JOINT_NAME) if include_waist else None
        )
        solutions_record: list[dict[str, Any]] = []
        rejection_counts = {
            "strict_self_collision": 0,
            "strict_cube_collision": 0,
            "table_plane": 0,
        }
        for local_index, seed_index in enumerate(successful_indices):
            clearance, clearance_link, _sample = _local_plane_clearance_from_spheres(
                spheres[local_index : local_index + 1],
                config=strict_checker.config.kinematics_config,
                arm=request.arm,
                plane_point=plane_point,
                down=down,
                include_payload=False,
            )
            reasons: list[str] = []
            if self_collisions[local_index]:
                rejection_counts["strict_self_collision"] += 1
                reasons.append("strict_self_collision")
            if world_collisions[local_index]:
                rejection_counts["strict_cube_collision"] += 1
                reasons.append("strict_cube_collision")
            if clearance < 0.0:
                rejection_counts["table_plane"] += 1
                reasons.append("table_plane")
            q = solutions[seed_index]
            goal_index = int(goal_indices[seed_index])
            if not 0 <= goal_index < len(candidate_ids):
                raise RuntimeError("CuRobo waist-yaw IK returned an invalid goal index")
            arm_delta = q[arm_solver_indices] - start[arm_solver_indices]
            hard_margin = np.minimum(q - lower, upper - q)
            solutions_record.append(
                {
                    "solver_seed_index": int(seed_index),
                    "candidate_id": candidate_ids[goal_index],
                    "strictly_valid": not reasons,
                    "strict_rejections": reasons,
                    "strict_self_collisions": [
                        {
                            "links": list(pair),
                            "penetration_m": float(penetration),
                        }
                        for pair, penetration in sorted(self_collisions[local_index].items())
                    ],
                    "strict_scene_collisions": [
                        {
                            "link": pair[0],
                            "object": pair[1],
                            "clearance_m": float(clearance_value),
                        }
                        for pair, clearance_value in sorted(
                            world_collisions[local_index].items()
                        )
                    ],
                    "model_waist_yaw_rad": (
                        start_model_waist
                        if waist_solver_index is None
                        else float(q[waist_solver_index])
                    ),
                    "waist_yaw_change_rad": (
                        0.0
                        if waist_solver_index is None
                        else float(q[waist_solver_index] - start[waist_solver_index])
                    ),
                    "maximum_arm_change_rad": float(np.max(np.abs(arm_delta))),
                    "arm_change_l2_rad": float(np.linalg.norm(arm_delta)),
                    "minimum_active_joint_limit_margin_rad": float(np.min(hard_margin)),
                    "minimum_table_plane_clearance_m": clearance,
                    "minimum_table_plane_link": clearance_link,
                }
            )
        valid = [item for item in solutions_record if item["strictly_valid"]]
        valid_candidates = sorted({str(item["candidate_id"]) for item in valid})
        return {
            "mode": "waist_yaw_bounded" if include_waist else "waist_yaw_locked",
            "waist_yaw_half_range_rad": waist_half_range_rad,
            "active_joint_names": list(solver_names),
            "measured_waist_yaw_command_rad": measured_waist_command,
            "model_start_waist_yaw_rad": start_model_waist,
            "waist_yaw_limits": limit_record,
            "ik_seed_count": len(raw_success),
            "raw_collision_constrained_success_count": int(np.count_nonzero(raw_success)),
            "strictly_valid_solution_count": len(valid),
            "strictly_valid_candidate_count": len(valid_candidates),
            "strictly_valid_candidate_ids": valid_candidates,
            "strict_rejection_counts": rejection_counts,
            "solutions": solutions_record,
            "elapsed_s": time.monotonic() - started,
        }
    finally:
        if planner is not None:
            planner.destroy()
        gc.collect()
        torch.cuda.empty_cache()


def analyze_waist_yaw(
    request: TabletopTaskRequest,
    *,
    waist_half_ranges_rad: tuple[float, ...],
    candidate_ids: tuple[str, ...] = (),
    progress=None,
) -> dict[str, Any]:
    """Compare locked-arm IK with bounded waist-yaw plus arm IK."""

    import torch
    report = progress or (lambda _message: None)
    ranges = _validated_ranges(waist_half_ranges_rad)
    shortlist, candidates = _load_shortlist(request)
    requested_candidate_ids = tuple(str(value) for value in candidate_ids)
    if len(requested_candidate_ids) != len(set(requested_candidate_ids)):
        raise ValueError("waist-yaw candidate filter must contain unique IDs")
    if requested_candidate_ids:
        by_id = {str(item["candidate_id"]): item for item in candidates}
        missing = sorted(set(requested_candidate_ids) - set(by_id))
        if missing:
            raise ValueError(f"waist-yaw candidate filter contains unknown IDs: {missing}")
        candidates = [by_id[value] for value in requested_candidate_ids]
    candidate_ids = [str(item["candidate_id"]) for item in candidates]
    open_q = (
        request.planning_snapshot.left_dex3_q_rad
        if request.arm == "left"
        else request.planning_snapshot.right_dex3_q_rad
    )
    locked_robot, reference = build_tabletop_robot_config(
        arm=request.arm,
        snapshot=request.planning_snapshot,
        joint_position_offsets_rad=request.joint_position_offsets_rad,
        active_finger_q_rad=open_q,
    )
    query_planner = None
    started = time.monotonic()
    try:
        query_planner, device_cfg = _planner(
            copy.deepcopy(locked_robot),
            {},
            max_goalset=1,
            seed=request.random_seed,
        )
        names = tabletop_motion_joint_names(request.arm)
        state = _joint_state(device_cfg, np.asarray(reference), names)
        base_T_torso = _base_T_torso(query_planner, state)
    finally:
        if query_planner is not None:
            query_planner.destroy()
        gc.collect()
        torch.cuda.empty_cache()

    plane_point, base_T_object, down = _table_from_resting_object(request, base_T_torso)
    approach_distance = float(shortlist["execution_contract"]["approach_distance_m"])
    pregrasp_matrices = [
        _pregrasp_matrix(base_T_object @ _candidate_transform(item), approach_distance)
        for item in candidates
    ]
    planning_scene = _base_scene(
        request,
        base_T_torso,
        include_cube=True,
        include_open_transit_table_patch=True,
    )
    strict_cube_scene = _base_scene(
        request,
        base_T_torso,
        include_cube=True,
    )
    configurations = []
    for half_range in (None, *ranges):
        label = "locked" if half_range is None else f"±{half_range:.3f}rad"
        report(f"analyzing waist yaw {label}; no robot command")
        configurations.append(
            _solve_configuration(
                request=request,
                open_q=open_q,
                candidate_ids=candidate_ids,
                pregrasp_matrices=pregrasp_matrices,
                planning_scene=planning_scene,
                strict_cube_scene=strict_cube_scene,
                plane_point=plane_point,
                down=down,
                waist_half_range_rad=half_range,
            )
        )
    locked_candidates = set(configurations[0]["strictly_valid_candidate_ids"])
    for item in configurations[1:]:
        item["additional_candidate_ids_vs_locked"] = sorted(
            set(item["strictly_valid_candidate_ids"]) - locked_candidates
        )
    return {
        "schema_version": 1,
        "operation": "analyze_tabletop_waist_yaw",
        "commands_robot": False,
        "analysis_scope": (
            "collision-constrained pregrasp IK plus independent strict full-robot "
            "self/cube and selected-hand table-plane validation; not a trajectory "
            "or hardware-execution contract"
        ),
        "request_sha256": request.content_sha256,
        "arm": request.arm,
        "candidate_count": len(candidates),
        "candidate_filter": list(requested_candidate_ids),
        "requested_waist_yaw_half_ranges_rad": list(ranges),
        "configurations": configurations,
        "planner_provenance": {
            **model_source_hashes(),
            "curobo_commit": CUROBO_COMMIT,
            "ik_seed_count": IK_SEEDS,
            "elapsed_s": time.monotonic() - started,
        },
    }


def analyze_waist_yaw_from_paths(
    request_path: str | Path,
    output_path: str | Path,
    *,
    waist_half_ranges_rad: tuple[float, ...],
    candidate_ids: tuple[str, ...] = (),
    progress=None,
) -> dict[str, Any]:
    request = TabletopTaskRequest.from_json(request_path)
    result = analyze_waist_yaw(
        request,
        waist_half_ranges_rad=waist_half_ranges_rad,
        candidate_ids=candidate_ids,
        progress=progress,
    )
    result["request_path"] = str(Path(request_path).resolve())
    atomic_write_json(output_path, result)
    return result
