from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from g1_dex3_tabletop.planning import tabletop_planner, tabletop_session
from g1_dex3_tabletop.planning.contracts import PlannedTrajectory, RobotSnapshot
from g1_dex3_tabletop.planning.dex3_handedness import dex3_execution_profile
from g1_dex3_tabletop.planning.tabletop_planner import (
    _table_from_resting_object,
    plan_tabletop_pick_place,
)
from g1_dex3_tabletop.planning.tabletop_session import TabletopPlanningSession
from g1_dex3_tabletop.stack_workflow import (
    build_observed_stack_second_stage_request,
    build_stack_stage_requests,
    stack_arm_assignments,
    stack_placement_candidates,
)
from g1_dex3_tabletop.tabletop_contracts import (
    TabletopObservation,
    TabletopPickPlaceRequest,
    TabletopTaskPlan,
    TabletopTaskRequest,
)
from g1_dex3_tabletop.tabletop_workflow import destination_request_for_pick_place


def _transform(x: float, y: float, z: float):
    value = np.eye(4)
    value[:3, 3] = (x, y, z)
    return tuple(tuple(float(item) for item in row) for row in value)


def _observation(x: float, y: float, z: float) -> TabletopObservation:
    return TabletopObservation(
        snapshot=RobotSnapshot((0.0,) * 29, (0.0,) * 7, (0.0,) * 7),
        camera_T_object=_transform(x, y, z),
        camera_profile_sha256="a" * 64,
        source_frame_sha256=("b" * 64, "c" * 64, "d" * 64),
        object_translation_spread_mm=0.1,
        object_rotation_spread_deg=0.1,
    )


def _request(arm: str, observation: TabletopObservation, size: float) -> TabletopTaskRequest:
    return TabletopTaskRequest(
        observation=observation,
        arm=arm,
        torso_T_camera=_transform(0.0, 0.0, 0.0),
        joint_position_offsets_rad={},
        calibration_bundle_sha256="e" * 64,
        grasp_shortlist_path="config/tabletop/shortlist.yaml",
        grasp_shortlist_sha256="f" * 64,
        object_dimensions_m=(size, size, size),
    )


def _trajectory(source: str, target: str, start: float, end: float) -> PlannedTrajectory:
    return PlannedTrajectory(
        from_pose_id=source,
        to_pose_id=target,
        sample_time_s=(0.0, 1.0),
        command_q_rad=((start,) * 7, (end,) * 7),
        model_q_rad=((start,) * 7, (end,) * 7),
        planning_time_s=0.1,
    )


def _task(
    request: TabletopTaskRequest,
    *,
    candidate_id: str,
    endpoints: tuple[float, ...],
) -> TabletopTaskPlan:
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
    routes = []
    source = "clearance"
    start = 0.0
    for phase, end in zip(phases, endpoints, strict=True):
        routes.append(_trajectory(source, phase, start, end))
        source = phase
        start = end
    open_q, close_q = dex3_execution_profile(request.arm)
    return TabletopTaskPlan(
        request_sha256=request.content_sha256,
        arm=request.arm,
        selected_candidate_id=candidate_id,
        object_T_grasp=_transform(0.0, 0.0, 0.0),
        open_active_dex3_q_rad=open_q,
        close_target_active_dex3_q_rad=close_q,
        initial_active_dex3_q_rad=(0.0,) * 7,
        trajectories=tuple(routes),
        phase_order=phases,
        planner_provenance={},
    )


def test_stack_candidates_span_only_proven_nonoverlapping_table_segment() -> None:
    cube40 = _observation(0.20, 0.0, 0.020)
    cube60 = _observation(-0.20, 0.0, 0.030)

    candidates = stack_placement_candidates(
        cube40=cube40,
        cube60=cube60,
        base_T_camera=np.eye(4),
        count=5,
    )

    assert len(candidates) == 5
    assert all(item.cube60_displacement_m > 0.0 for item in candidates)
    assert all(item.placed_center_separation_m > 0.075 for item in candidates)
    assert [item.placed_center_separation_m for item in candidates] == sorted(
        item.placed_center_separation_m for item in candidates
    )
    assert candidates[0].table_plane_disagreement_mm == pytest.approx(0.0)
    assert candidates[0].table_normal_disagreement_deg == pytest.approx(0.0)


def test_stack_stage_requests_keep_real_table_and_finite_60mm_support() -> None:
    cube40 = _observation(0.20, 0.0, 0.020)
    cube60 = _observation(-0.20, 0.0, 0.030)
    candidate = stack_placement_candidates(
        cube40=cube40,
        cube60=cube60,
        base_T_camera=np.eye(4),
        count=1,
    )[0]
    stage1, stage2 = build_stack_stage_requests(
        cube40_request=_request("right", cube40, 0.040),
        cube60_request=_request("left", cube60, 0.060),
        candidate=candidate,
    )

    assert stage1.source_request.environment_cuboids[0].object_id == "cube40"
    assert stage1.destination_support_object_id is None
    assert stage2.source_request.environment_cuboids[0].object_id == "cube60"
    destination = destination_request_for_pick_place(stage2)
    assert destination.environment_cuboids[0].role == "placement_support"
    camera_T_destination = np.asarray(destination.observation.camera_T_object)
    assert camera_T_destination[2, 3] == pytest.approx(0.080)
    np.testing.assert_allclose(
        destination.table_reference_camera_T_object,
        stage2.source_request.observation.camera_T_object,
    )
    plane_point, _destination_object, _down = _table_from_resting_object(
        destination,
        np.eye(4),
    )
    assert plane_point[2] == pytest.approx(0.0)


def test_observed_second_stage_uses_actual_support_pose() -> None:
    cube40 = _observation(0.20, 0.0, 0.020)
    placed60 = _observation(-0.05, 0.10, 0.030)

    stage2 = build_observed_stack_second_stage_request(
        cube40_request=_request("right", cube40, 0.040),
        placed_cube60=placed60,
        base_T_camera=np.eye(4),
    )

    support = stage2.source_request.environment_cuboids[0]
    assert support.object_id == "cube60"
    np.testing.assert_allclose(
        np.asarray(support.object_T_cuboid)[:3, 3],
        (-0.25, 0.10, 0.01),
    )
    destination = destination_request_for_pick_place(stage2)
    assert destination.environment_cuboids[0].role == "placement_support"
    np.testing.assert_allclose(
        np.asarray(destination.observation.camera_T_object)[:3, 3],
        (-0.05, 0.10, 0.080),
    )


def test_nearest_arm_assignment_is_only_an_ordering_heuristic() -> None:
    cube40 = _observation(0.30, 0.20, 0.020)
    cube60 = _observation(0.30, -0.20, 0.030)
    ordered = stack_arm_assignments(
        cube40=cube40,
        cube60=cube60,
        base_T_camera=np.eye(4),
        base_left_hand_position=np.asarray((0.30, 0.15, 0.10)),
        base_right_hand_position=np.asarray((0.30, -0.15, 0.10)),
    )

    assert ordered == (("right", "left"), ("left", "right"))
    assert set(ordered) == {("left", "right"), ("right", "left")}


def test_stack_requires_opposite_arms() -> None:
    cube40 = _observation(0.20, 0.0, 0.020)
    cube60 = _observation(-0.20, 0.0, 0.030)
    candidate = stack_placement_candidates(
        cube40=cube40,
        cube60=cube60,
        base_T_camera=np.eye(4),
        count=1,
    )[0]
    request40 = _request("left", cube40, 0.040)
    request60 = replace(_request("right", cube60, 0.060), arm="left")
    with pytest.raises(ValueError, match="different arm"):
        build_stack_stage_requests(
            cube40_request=request40,
            cube60_request=request60,
            candidate=candidate,
        )


def test_pick_place_tries_next_grasp_when_first_cannot_place(monkeypatch) -> None:
    source = _request("left", _observation(0.20, 0.0, 0.020), 0.040)
    source_T_destination = np.eye(4)
    source_T_destination[0, 3] = 0.10
    request = TabletopPickPlaceRequest(
        source_request=source,
        source_T_destination_object=source_T_destination,
    )
    destination = destination_request_for_pick_place(request)
    source_endpoints = (0.10, 0.20, 0.30, 0.40, 0.30, 0.20, 0.10, 0.0)
    destination_endpoints = (0.11, 0.21, 0.31, 0.41, 0.31, 0.21, 0.11, 0.0)
    calls = []

    def fake_plan(current, *, required_candidate_id=None, **_kwargs):
        candidate_id = required_candidate_id or "candidate_a"
        calls.append((current.content_sha256, required_candidate_id))
        if current.content_sha256 == destination.content_sha256 and candidate_id == "candidate_a":
            raise RuntimeError("candidate_a cannot reach destination")
        endpoints = (
            source_endpoints
            if current.content_sha256 == source.content_sha256
            else destination_endpoints
        )
        return _task(current, candidate_id=candidate_id, endpoints=endpoints)

    def fake_transfer(_request, source_task, destination_task, **_kwargs):
        return (
            _trajectory(
                "payload_lift",
                "payload_transfer",
                source_task.trajectories[3].command_q_rad[-1][0],
                destination_task.trajectories[3].command_q_rad[-1][0],
            ),
            {"test": True},
        )

    monkeypatch.setattr(tabletop_planner, "plan_tabletop_task", fake_plan)
    monkeypatch.setattr(
        tabletop_planner,
        "_load_shortlist",
        lambda _request: ({}, [{"candidate_id": "candidate_a"}, {"candidate_id": "candidate_b"}]),
    )
    monkeypatch.setattr(tabletop_planner, "_plan_attached_transfer", fake_transfer)

    plan = plan_tabletop_pick_place(request)

    assert plan.selected_candidate_id == "candidate_b"
    assert [value[1] for value in calls] == [None, "candidate_a", "candidate_b", "candidate_b"]
    assert plan.planner_provenance["rejected_pick_place_grasps"][0]["candidate_id"] == (
        "candidate_a"
    )


def test_pick_place_checker_is_built_only_for_the_selected_execution(monkeypatch) -> None:
    planned_request = SimpleNamespace(content_sha256="a" * 64)
    planned_plan = SimpleNamespace(content_sha256="b" * 64)
    validation_request = SimpleNamespace(
        pick_place_request=planned_request,
        pick_place_plan=planned_plan,
    )
    builds = []

    class FakeValidator:
        def __init__(self, request, plan):
            builds.append((request, plan))

        def validate(self, request, *, progress=None):
            return (request, progress)

    monkeypatch.setattr(
        tabletop_session,
        "plan_tabletop_pick_place",
        lambda request, **_kwargs: planned_plan,
    )
    monkeypatch.setattr(tabletop_session, "PickPlaceRetentionRouteValidator", FakeValidator)
    session = TabletopPlanningSession()

    assert session.plan_pick_place(planned_request) is planned_plan
    assert builds == []
    assert session.validate_pick_place_retention_route(validation_request) == (
        validation_request,
        None,
    )
    assert builds == [(planned_request, planned_plan)]
