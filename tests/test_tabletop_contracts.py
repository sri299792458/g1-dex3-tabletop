from __future__ import annotations

import json
from pathlib import Path

import pytest

from g1_dex3_tabletop.planning.contracts import PlannedTrajectory, RobotSnapshot
from g1_dex3_tabletop.planning.dex3_handedness import dex3_execution_profile
from g1_dex3_tabletop.tabletop_contracts import (
    PICK_PLACE_PHASE_ORDER,
    CharucoBoardObservation,
    CharucoSupportedEscapeRequest,
    EstimatedCameraPlanningState,
    PregraspRemainingPlan,
    RetentionRouteValidationRequest,
    RetentionRouteValidationResult,
    SupportedEscapePlan,
    TabletopCuboid,
    TabletopExecutionPlan,
    TabletopObservation,
    TabletopPickPlacePlan,
    TabletopPickPlaceRequest,
    TabletopPregraspPlan,
    TabletopTaskPlan,
    TabletopTaskRequest,
    build_pregrasp_remaining_plan,
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


def charuco_observation() -> CharucoBoardObservation:
    return CharucoBoardObservation(
        snapshot=RobotSnapshot((0.0,) * 29, (0.0,) * 7, (0.0,) * 7),
        camera_T_board=identity(),
        camera_profile_sha256="a" * 64,
        source_frame_sha256=("b" * 64, "c" * 64, "d" * 64),
        translation_spread_mm=0.2,
        rotation_spread_deg=0.1,
        board_spec={
            "squares_x": 6,
            "squares_y": 9,
            "square_length_mm": 30.0,
            "marker_length_mm": 22.0,
            "dictionary_name": "DICT_5X5_50",
            "legacy_pattern": False,
            "active_dimensions_mm": [180.0, 270.0],
            "marker_count": 27,
            "charuco_corner_count": 40,
        },
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


def test_tabletop_request_hash_binds_pregrasp_distance() -> None:
    original = request()
    changed = TabletopTaskRequest.from_dict(
        {**original.to_dict(include_hash=False), "pregrasp_distance_m": 0.075}
    )

    assert changed.pregrasp_distance_m == 0.075
    assert changed.content_sha256 != original.content_sha256
    with pytest.raises(ValueError, match="positive and finite"):
        TabletopTaskRequest.from_dict(
            {**original.to_dict(include_hash=False), "pregrasp_distance_m": 0.0}
        )


def test_estimated_planning_state_is_separate_from_visual_observation() -> None:
    original = request()
    object_T_camera = [list(row) for row in identity()]
    object_T_camera[0][3] = 0.012
    estimated = EstimatedCameraPlanningState(
        snapshot=RobotSnapshot((0.1,) * 29, (0.2,) * 7, (0.3,) * 7),
        object_T_camera=object_T_camera,
        anchor_observation_sha256=original.observation.content_sha256,
        anchor_timestamp_ns=1_000_000_000,
        timestamp_ns=2_000_000_000,
        anchor_input_timing={"timestamp_ns": 1_000_000_000},
        current_input_timing={"timestamp_ns": 2_000_000_000},
    )
    document = original.to_dict(include_hash=False)
    document["estimated_planning_state"] = estimated.to_dict()
    propagated = TabletopTaskRequest.from_dict(document)

    assert propagated.observation == original.observation
    assert propagated.planning_snapshot == estimated.snapshot
    assert propagated.planning_camera_T_object[0][3] == pytest.approx(-0.012)
    assert TabletopTaskRequest.from_dict(propagated.to_dict()) == propagated

    bad = original.to_dict(include_hash=False)
    bad["estimated_planning_state"] = {
        **estimated.to_dict(),
        "anchor_observation_sha256": "0" * 64,
    }
    with pytest.raises(ValueError, match="different visual anchor"):
        TabletopTaskRequest.from_dict(bad)


def test_charuco_escape_request_round_trip_and_frozen_board(tmp_path: Path) -> None:
    value = CharucoSupportedEscapeRequest(
        observation=charuco_observation(),
        arm="left",
        torso_T_camera=identity(),
        joint_position_offsets_rad={"left_shoulder_roll_joint": 0.02},
        calibration_bundle_sha256="e" * 64,
    )
    path = tmp_path / "charuco_request.json"
    value.write_json(path)
    assert CharucoSupportedEscapeRequest.from_json(path) == value

    document = value.to_dict(include_hash=False)
    document["observation"]["board_spec"]["dictionary_name"] = "DICT_4X4_50"
    with pytest.raises(ValueError, match="different frozen table board"):
        CharucoSupportedEscapeRequest.from_dict(document)


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


def test_table_reference_pose_and_dimensions_are_atomic() -> None:
    values = request().to_dict(include_hash=False)
    values["table_reference_camera_T_object"] = identity()
    with pytest.raises(ValueError, match="pose and dimensions"):
        TabletopTaskRequest.from_dict(values)


def test_pick_place_request_binds_relative_destination_and_support(tmp_path: Path) -> None:
    source = request()
    values = source.to_dict(include_hash=False)
    values["environment_cuboids"] = [
        TabletopCuboid(
            object_id="cube60",
            object_T_cuboid=identity(),
            dimensions_m=(0.06, 0.06, 0.06),
        ).to_dict()
    ]
    source = TabletopTaskRequest.from_dict(values)
    destination = [list(row) for row in identity()]
    destination[0][3] = 0.1
    pick_place = TabletopPickPlaceRequest(
        source_request=source,
        source_T_destination_objects=(destination,),
        destination_support_object_id="cube60",
        excluded_candidate_ids=("failed_grasp",),
    )
    path = tmp_path / "pick_place_request.json"
    pick_place.write_json(path)
    assert TabletopPickPlaceRequest.from_json(path) == pick_place
    assert pick_place.excluded_candidate_ids == ("failed_grasp",)

    bad = pick_place.to_dict(include_hash=False)
    bad["destination_support_object_id"] = "missing"
    with pytest.raises(ValueError, match="absent from the source world"):
        TabletopPickPlaceRequest.from_dict(bad)


def test_pick_place_plan_has_one_fixed_continuous_sequence() -> None:
    open_q, close_q = dex3_execution_profile("right")
    lifecycle_phases = (
        "move_to_pregrasp",
        "grasp_approach",
        "retention_test_lift",
        "payload_lift",
        "payload_lower",
        "payload_replace",
        "grasp_retreat",
        "return_to_clearance",
    )
    lifecycle = []
    source = "clearance"
    for index, phase in enumerate(lifecycle_phases):
        lifecycle.append(trajectory(source, phase, index / 100, (index + 1) / 100))
        source = phase
    task = TabletopTaskPlan(
        request_sha256="a" * 64,
        arm="right",
        selected_candidate_id="candidate",
        object_T_grasp=identity(),
        open_active_dex3_q_rad=open_q,
        close_target_active_dex3_q_rad=close_q,
        initial_active_dex3_q_rad=(0.1,) * 7,
        trajectories=tuple(lifecycle),
        phase_order=lifecycle_phases,
        planner_provenance={},
    )
    pick_place_trajectories = []
    source = "clearance"
    for index, phase in enumerate(PICK_PLACE_PHASE_ORDER):
        pick_place_trajectories.append(trajectory(source, phase, index / 100, (index + 1) / 100))
        source = phase
    plan = TabletopPickPlacePlan(
        request_sha256="b" * 64,
        arm="right",
        selected_candidate_id="candidate",
        selected_destination_index=0,
        source_task=task,
        destination_task=task,
        trajectories=tuple(pick_place_trajectories),
        phase_order=PICK_PLACE_PHASE_ORDER,
        planner_provenance={},
    )
    assert TabletopPickPlacePlan.from_dict(plan.to_dict()) == plan

    discontinuous = list(pick_place_trajectories)
    discontinuous[4] = trajectory("payload_lift", "payload_transfer", 0.2, 0.3)
    with pytest.raises(ValueError, match="trajectory discontinuity"):
        TabletopPickPlacePlan(
            request_sha256="b" * 64,
            arm="right",
            selected_candidate_id="candidate",
            selected_destination_index=0,
            source_task=task,
            destination_task=task,
            trajectories=tuple(discontinuous),
            phase_order=PICK_PLACE_PHASE_ORDER,
            planner_provenance={},
        )


def test_supported_escape_requires_exact_reverse() -> None:
    outbound = trajectory("__handoff__", "clearance", 0.0, 0.1)
    inbound = trajectory("clearance", "__handoff__", 0.1, 0.0)
    plan = SupportedEscapePlan("a" * 64, outbound, inbound, 0.08, {})
    assert SupportedEscapePlan.from_dict(plan.to_dict()) == plan
    bad = trajectory("clearance", "__handoff__", 0.09, 0.0)
    with pytest.raises(ValueError, match="exact reverse"):
        SupportedEscapePlan("a" * 64, outbound, bad, 0.08, {})


def test_task_plan_requires_complete_finite_lifecycle() -> None:
    open_q, close_q = dex3_execution_profile("right")
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
        open_active_dex3_q_rad=open_q,
        close_target_active_dex3_q_rad=close_q,
        initial_active_dex3_q_rad=(0.1,) * 7,
        trajectories=tuple(trajectories),
        phase_order=phases,
        planner_provenance={},
    )
    assert TabletopTaskPlan.from_dict(plan.to_dict()) == plan
    serialized = plan.to_dict(include_hash=False)
    assert serialized["close_target_active_dex3_q_rad"] == list(close_q)
    assert "closed_active_dex3_q_rad" not in serialized
    serialized["close_target_active_dex3_q_rad"] = [0.5] * 7
    with pytest.raises(ValueError, match="differs from the Dex3 descriptor"):
        TabletopTaskPlan.from_dict(serialized)
    with pytest.raises(ValueError, match="complete eight-motion lifecycle"):
        TabletopTaskPlan(
            request_sha256="a" * 64,
            arm="right",
            selected_candidate_id="cube_1",
            object_T_grasp=identity(),
            open_active_dex3_q_rad=open_q,
            close_target_active_dex3_q_rad=close_q,
            initial_active_dex3_q_rad=(0.1,) * 7,
            trajectories=tuple(trajectories[:-1]),
            phase_order=phases[:-1],
            planner_provenance={},
        )


def test_complete_execution_binds_escape_task_and_exact_return() -> None:
    open_q, close_q = dex3_execution_profile("right")
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
        open_q,
        close_q,
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
        ("move_to_pregrasp", "return_to_clearance"),
    )
    assert plan.recovery_trajectories[0].command_q_rad == plan.trajectories[7].command_q_rad
    assert plan.recovery_trajectories[1].command_q_rad == plan.trajectories[6].command_q_rad
    assert plan.recovery_trajectories[2].command_q_rad == tuple(
        reversed(plan.trajectories[1].command_q_rad)
    )

    estimated_state = EstimatedCameraPlanningState(
        snapshot=RobotSnapshot((0.0,) * 22 + (0.20,) * 7, (0.0,) * 7, (0.0,) * 7),
        object_T_camera=identity(),
        anchor_observation_sha256=clearance_request.observation.content_sha256,
        anchor_timestamp_ns=1,
        timestamp_ns=2,
        anchor_input_timing={"timestamp_ns": 1},
        current_input_timing={"timestamp_ns": 2},
    )
    estimated_document = clearance_request.to_dict(include_hash=False)
    estimated_document["estimated_planning_state"] = estimated_state.to_dict()
    estimated_request = TabletopTaskRequest.from_dict(estimated_document)
    new_endpoints = (0.25, 0.26, 0.27, 0.28, 0.27, 0.26, 0.25, 0.20)
    new_motions = []
    source = "clearance"
    start = 0.20
    for phase, end in zip(phases, new_endpoints, strict=True):
        new_motions.append(trajectory(source, phase, start, end))
        source = phase
        start = end
    new_task = TabletopTaskPlan(
        estimated_request.content_sha256,
        "right",
        "cube_1",
        identity(),
        open_q,
        close_q,
        (0.1,) * 7,
        tuple(new_motions),
        phases,
        {},
    )
    pregrasp = TabletopPregraspPlan(
        request_sha256=clearance_request.content_sha256,
        arm="right",
        selected_candidate_id="cube_1",
        object_T_grasp=identity(),
        open_active_dex3_q_rad=open_q,
        outbound=plan.task.trajectories[0],
        inbound=PlannedTrajectory(
            "move_to_pregrasp",
            "return_to_clearance",
            (0.0, 1.0),
            ((0.20,) * 7, (0.10,) * 7),
            ((0.20,) * 7, (0.10,) * 7),
            0.0,
        ),
        planner_provenance={},
    )
    assert TabletopPregraspPlan.from_dict(pregrasp.to_dict()) == pregrasp
    remaining = build_pregrasp_remaining_plan(
        prior_pregrasp_plan=pregrasp,
        estimated_request=estimated_request,
        task=new_task,
    )
    assert PregraspRemainingPlan.from_dict(remaining.to_dict()) == remaining
    assert remaining.trajectories[0].from_pose_id == "move_to_pregrasp"
    assert remaining.trajectories[-1].to_pose_id == "return_to_clearance"
    assert remaining.trajectories[-1].command_q_rad == tuple(
        reversed(pregrasp.outbound.command_q_rad)
    )
    assert tuple(
        (item.from_pose_id, item.to_pose_id) for item in remaining.recovery_trajectories[-2:]
    ) == (
        ("estimated_pregrasp", "move_to_pregrasp"),
        ("move_to_pregrasp", "return_to_clearance"),
    )
    assert remaining.recovery_trajectories[-2].command_q_rad == tuple(
        reversed(remaining.trajectories[0].command_q_rad)
    )


def test_retention_route_contract_binds_measured_fingers_to_task(tmp_path: Path) -> None:
    open_q, close_q = dex3_execution_profile("right")
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
        open_q,
        close_q,
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
