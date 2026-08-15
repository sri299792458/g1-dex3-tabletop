from __future__ import annotations

import json
from pathlib import Path

import pytest

from g1_dex3_tabletop.planning.contracts import PlannedTrajectory, RobotSnapshot
from g1_dex3_tabletop.tabletop_contracts import (
    RetentionRouteValidationRequest,
    RetentionRouteValidationResult,
    SupportedEscapePlan,
    TabletopExecutionPlan,
    TabletopObservation,
    TabletopTaskPlan,
    TabletopTaskRequest,
    combine_tabletop_plans,
)


def identity():
    return (
        (1.0, 0.0, 0.0, 0.0),
        (0.0, 1.0, 0.0, 0.0),
        (0.0, 0.0, 1.0, 0.0),
        (0.0, 0.0, 0.0, 1.0),
    )


def observation() -> TabletopObservation:
    return TabletopObservation(
        snapshot=RobotSnapshot((0.0,) * 29, (0.0,) * 7, (0.0,) * 7),
        camera_T_object=identity(),
        camera_profile_sha256="a" * 64,
        source_frame_sha256=("b" * 64, "c" * 64, "d" * 64),
        object_translation_spread_mm=0.2,
        object_rotation_spread_deg=0.1,
    )


def request() -> TabletopTaskRequest:
    return TabletopTaskRequest(
        observation=observation(),
        arm="right",
        torso_T_camera=identity(),
        joint_position_offsets_rad={"right_shoulder_roll_joint": -0.05},
        calibration_bundle_sha256="e" * 64,
        grasp_shortlist_path="config/tabletop/shortlist.yaml",
        grasp_shortlist_sha256="f" * 64,
    )


def trajectory(source: str, target: str, start: float, end: float) -> PlannedTrajectory:
    return PlannedTrajectory(
        source,
        target,
        (0.0, 1.0),
        ((start,) * 7, (end,) * 7),
        ((start,) * 7, (end,) * 7),
        0.1,
    )


def test_tabletop_request_round_trip_and_hash_guard(tmp_path: Path) -> None:
    value = request()
    path = tmp_path / "request.json"
    value.write_json(path)
    assert TabletopTaskRequest.from_json(path) == value
    document = json.loads(path.read_text())
    document["lift_m"] = 0.2
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        TabletopTaskRequest.from_dict(document)


def test_tabletop_request_rejects_invalid_open_transit_patch() -> None:
    values = request().to_dict(include_hash=False)
    values["open_transit_table_patch_dimensions_m"] = [0.4, 0.0, 0.02]
    with pytest.raises(ValueError, match="patch dimensions must be positive"):
        TabletopTaskRequest.from_dict(values)


def test_tabletop_request_rejects_invalid_arm_velocity() -> None:
    values = request().to_dict(include_hash=False)
    values["maximum_arm_velocity_rad_s"] = 0.0
    with pytest.raises(ValueError, match="maximum_arm_velocity_rad_s"):
        TabletopTaskRequest.from_dict(values)


def test_supported_escape_requires_exact_reverse() -> None:
    outbound = trajectory("__handoff__", "clearance", 0.0, 0.1)
    inbound = trajectory("clearance", "__handoff__", 0.1, 0.0)
    plan = SupportedEscapePlan("a" * 64, outbound, inbound, 0.08, {})
    assert SupportedEscapePlan.from_dict(plan.to_dict()) == plan
    bad = trajectory("clearance", "__handoff__", 0.09, 0.0)
    with pytest.raises(ValueError, match="exact reverse"):
        SupportedEscapePlan("a" * 64, outbound, bad, 0.08, {})


def test_task_plan_requires_complete_finite_lifecycle() -> None:
    phases = (
        "move_to_pregrasp",
        "grasp_approach",
        "retention_test_lift",
        "payload_lift",
        "payload_lower",
        "payload_replace",
        "grasp_retreat",
        "return_to_clearance",
    )
    trajectories = []
    source = "clearance"
    for index, phase in enumerate(phases):
        trajectories.append(trajectory(source, phase, index / 100, (index + 1) / 100))
        source = phase
    plan = TabletopTaskPlan(
        request_sha256="a" * 64,
        arm="right",
        selected_candidate_id="cube_1",
        object_T_grasp=identity(),
        open_active_dex3_q_rad=(0.0,) * 7,
        closed_active_dex3_q_rad=(0.5,) * 7,
        initial_active_dex3_q_rad=(0.1,) * 7,
        trajectories=tuple(trajectories),
        phase_order=phases,
        planner_provenance={},
    )
    assert TabletopTaskPlan.from_dict(plan.to_dict()) == plan
    with pytest.raises(ValueError, match="complete eight-motion lifecycle"):
        TabletopTaskPlan(
            request_sha256="a" * 64,
            arm="right",
            selected_candidate_id="cube_1",
            object_T_grasp=identity(),
            open_active_dex3_q_rad=(0.0,) * 7,
            closed_active_dex3_q_rad=(0.5,) * 7,
            initial_active_dex3_q_rad=(0.1,) * 7,
            trajectories=tuple(trajectories[:-1]),
            phase_order=phases[:-1],
            planner_provenance={},
        )


def test_complete_execution_binds_escape_task_and_exact_return() -> None:
    loaded_request = request()
    clearance_observation = TabletopObservation(
        snapshot=RobotSnapshot((0.0,) * 22 + (0.1,) * 7, (0.0,) * 7, (0.0,) * 7),
        camera_T_object=identity(),
        camera_profile_sha256="a" * 64,
        source_frame_sha256=("b" * 64, "c" * 64, "d" * 64),
        object_translation_spread_mm=0.2,
        object_rotation_spread_deg=0.1,
    )
    clearance_request = TabletopTaskRequest(
        observation=clearance_observation,
        arm="right",
        torso_T_camera=identity(),
        joint_position_offsets_rad=loaded_request.joint_position_offsets_rad,
        calibration_bundle_sha256="e" * 64,
        grasp_shortlist_path="config/tabletop/shortlist.yaml",
        grasp_shortlist_sha256="f" * 64,
    )
    outbound = trajectory("__handoff__", "clearance", 0.0, 0.1)
    escape = SupportedEscapePlan(
        loaded_request.content_sha256,
        outbound,
        trajectory("clearance", "__handoff__", 0.1, 0.0),
        0.08,
        {},
    )
    phases = (
        "move_to_pregrasp",
        "grasp_approach",
        "retention_test_lift",
        "payload_lift",
        "payload_lower",
        "payload_replace",
        "grasp_retreat",
        "return_to_clearance",
    )
    motions = []
    source = "clearance"
    start = 0.1
    endpoints = (0.20, 0.21, 0.22, 0.23, 0.22, 0.21, 0.20, 0.10)
    for phase, end in zip(phases, endpoints, strict=True):
        motions.append(trajectory(source, phase, start, end))
        source = phase
        start = end
    task = TabletopTaskPlan(
        clearance_request.content_sha256,
        "right",
        "cube_1",
        identity(),
        (0.0,) * 7,
        (0.5,) * 7,
        (0.1,) * 7,
        tuple(motions),
        phases,
        {},
    )
    plan = combine_tabletop_plans(
        loaded_request=loaded_request,
        clearance_request=clearance_request,
        supported_escape=escape,
        task=task,
    )
    assert TabletopExecutionPlan.from_dict(plan.to_dict()) == plan
    assert plan.trajectories[-1].from_pose_id == "return_to_clearance"
    assert plan.trajectories[-1].to_pose_id == "__handoff__"
    assert tuple((item.from_pose_id, item.to_pose_id) for item in plan.recovery_trajectories) == (
        ("grasp_approach", "grasp_retreat"),
        ("retention_test_lift", "payload_replace"),
    )
    assert plan.recovery_trajectories[0].command_q_rad == plan.trajectories[7].command_q_rad
    assert plan.recovery_trajectories[1].command_q_rad == plan.trajectories[6].command_q_rad


def test_retention_route_contract_binds_measured_fingers_to_task(tmp_path: Path) -> None:
    tabletop_request = request()
    phases = (
        "move_to_pregrasp",
        "grasp_approach",
        "retention_test_lift",
        "payload_lift",
        "payload_lower",
        "payload_replace",
        "grasp_retreat",
        "return_to_clearance",
    )
    motions = []
    source = "clearance"
    for index, phase in enumerate(phases):
        motions.append(trajectory(source, phase, index / 100, (index + 1) / 100))
        source = phase
    task = TabletopTaskPlan(
        tabletop_request.content_sha256,
        "right",
        "cube_1",
        identity(),
        (0.0,) * 7,
        (0.5,) * 7,
        (0.1,) * 7,
        tuple(motions),
        phases,
        {},
    )
    validation_request = RetentionRouteValidationRequest(
        tabletop_request=tabletop_request,
        task_plan=task,
        measured_active_dex3_q_rad=(0.25,) * 7,
        blocked_motor_ids=(3,),
    )
    request_path = tmp_path / "retention_request.json"
    validation_request.write_json(request_path)
    assert RetentionRouteValidationRequest.from_json(request_path) == validation_request

    result = RetentionRouteValidationResult(
        request_sha256=validation_request.content_sha256,
        arm="right",
        selected_candidate_id="cube_1",
        route_sample_count=20,
        minimum_hand_plane_clearance_m=0.004,
        minimum_hand_plane_link="right_hand_middle_1_link",
        minimum_hand_plane_sample=0,
        planner_provenance={"pressure_used_for_live_decision": False},
    )
    result_path = tmp_path / "retention_result.json"
    result.write_json(result_path)
    assert RetentionRouteValidationResult.from_json(result_path) == result
