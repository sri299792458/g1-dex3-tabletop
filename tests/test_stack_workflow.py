from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from g1_aprilcube_calibration.joint_map import arm_indices
from g1_dex3_tabletop import hardware_stack
from g1_dex3_tabletop.hardware_stack import (
    _active_clearance_snapshot,
    _dual_arm_command_snapshot,
    _nearest_cube_move,
)
from g1_dex3_tabletop.planning import tabletop_planner, tabletop_session
from g1_dex3_tabletop.planning.contracts import PlannedTrajectory, RobotSnapshot
from g1_dex3_tabletop.planning.dex3_handedness import dex3_execution_profile
from g1_dex3_tabletop.planning.tabletop_planner import (
    _table_from_resting_object,
    plan_tabletop_pick_place,
)
from g1_dex3_tabletop.planning.tabletop_session import TabletopPlanningSession
from g1_dex3_tabletop.stack_workflow import (
    build_direct_stack_request,
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


def test_right_escape_snapshot_uses_held_command_instead_of_tracking_measurement() -> None:
    measured = np.zeros(29)
    right_indices = np.asarray(arm_indices("right"))
    measured[right_indices[3]] = -0.020277314
    state = SimpleNamespace(position=measured)
    hands = SimpleNamespace(
        left=SimpleNamespace(position=np.zeros(7)),
        right=SimpleNamespace(position=np.zeros(7)),
    )
    left_command = np.full(7, 0.25)
    held_right_command = np.zeros(7)

    snapshot = _dual_arm_command_snapshot(
        state,
        hands,
        left_command_q_rad=left_command,
        right_command_q_rad=held_right_command,
    )

    q29 = np.asarray(snapshot.measured_q29_rad)
    np.testing.assert_array_equal(q29[right_indices], held_right_command)
    np.testing.assert_array_equal(q29[np.asarray(arm_indices("left"))], left_command)


def test_active_clearance_snapshot_keeps_unused_arm_at_supported_command() -> None:
    state = SimpleNamespace(position=np.full(29, 0.9))
    hands = SimpleNamespace(
        left=SimpleNamespace(position=np.zeros(7)),
        right=SimpleNamespace(position=np.ones(7)),
    )
    escape = SimpleNamespace(
        outbound=SimpleNamespace(command_q_rad=((0.0,) * 7, (0.25,) * 7))
    )

    snapshot = _active_clearance_snapshot(
        state,
        hands,
        arm="right",
        escape=escape,
        inactive_command_q_rad=np.full(7, -0.1),
    )

    q29 = np.asarray(snapshot.measured_q29_rad)
    np.testing.assert_array_equal(q29[np.asarray(arm_indices("left"))], -0.1)
    np.testing.assert_array_equal(q29[np.asarray(arm_indices("right"))], 0.25)


def test_nearest_cube_move_selects_one_global_cube_arm_pair() -> None:
    class FakeModel:
        @staticmethod
        def transform(_parent, child, _positions):
            value = np.eye(4)
            if child == "left_rubber_hand":
                value[0, 3] = -0.30
            elif child == "right_rubber_hand":
                value[0, 3] = 0.30
            return value

    upper = _observation(0.24, 0.0, 0.030)
    bottom = _observation(-0.05, 0.0, 0.030)

    selected = _nearest_cube_move(
        reference_request=_request("left", upper, 0.060),
        upper_observation=upper,
        bottom_observation=bottom,
        model=FakeModel(),
    )

    assert selected["moving_cube"] == "secondary"
    assert selected["support_cube"] == "primary"
    assert selected["arm"] == "right"
    assert selected["source_hand_distance_m"] == pytest.approx(np.hypot(0.06, 0.03))


def _task(
    request: TabletopTaskRequest,
    *,
    candidate_id: str,
    endpoints: tuple[float, ...],
    branch_counts: dict[str, int] | None = None,
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
        planner_provenance=(
            {}
            if branch_counts is None
            else {"pregrasp_ik_unique_branches_by_candidate": branch_counts}
        ),
    )


def test_direct_stack_uses_observed_support_cube_and_center_on_center_target() -> None:
    moving = _observation(0.20, 0.0, 0.030)
    support = _observation(-0.20, 0.10, 0.030)

    request = build_direct_stack_request(
        moving_request=_request("left", moving, 0.060),
        support_cube=support,
        base_T_camera=np.eye(4),
        excluded_candidate_ids=("failed_grasp",),
    )

    obstacle = request.source_request.environment_cuboids[0]
    assert obstacle.object_id == "support_cube"
    np.testing.assert_allclose(
        np.asarray(obstacle.object_T_cuboid)[:3, 3],
        (-0.40, 0.10, 0.0),
    )
    assert request.excluded_candidate_ids == ("failed_grasp",)
    destination = destination_request_for_pick_place(request)
    assert destination.environment_cuboids[0].role == "placement_support"
    np.testing.assert_allclose(
        np.asarray(destination.observation.camera_T_object)[:3, 3],
        (-0.20, 0.10, 0.090),
    )
    plane_point, _destination_object, _down = _table_from_resting_object(
        destination,
        np.eye(4),
    )
    assert plane_point[2] == pytest.approx(0.0)


def test_direct_stack_yaw_is_only_a_nominal_wrist_path_choice() -> None:
    moving = _observation(0.20, 0.0, 0.030)
    support = _observation(-0.20, 0.10, 0.030)

    request = build_direct_stack_request(
        moving_request=_request("right", moving, 0.060),
        support_cube=support,
        base_T_camera=np.eye(4),
        yaw_quarter_turns=1,
    )
    source_T_destination = np.asarray(request.source_T_destination_object)

    np.testing.assert_allclose(
        source_T_destination[:3, :3],
        Rotation.from_euler("z", 90.0, degrees=True).as_matrix(),
        atol=1.0e-12,
    )
    with pytest.raises(ValueError, match="0, 1, 2, or 3"):
        build_direct_stack_request(
            moving_request=_request("right", moving, 0.060),
            support_cube=support,
            base_T_camera=np.eye(4),
            yaw_quarter_turns=4,
        )


def test_attached_stack_transfer_does_not_turn_supported_unused_arm_into_table_collision(
    monkeypatch,
) -> None:
    moving = _observation(0.20, 0.0, 0.030)
    support = _observation(-0.20, 0.10, 0.030)
    request = build_direct_stack_request(
        moving_request=_request("right", moving, 0.060),
        support_cube=support,
        base_T_camera=np.eye(4),
    )
    base_scene_calls = []

    def fake_base_scene(_request, _base_T_torso, **kwargs):
        base_scene_calls.append(kwargs)
        return {"cuboid": {"support_cube": {"dims": [0.06, 0.06, 0.06]}}}

    monkeypatch.setattr(tabletop_planner, "_base_scene", fake_base_scene)
    monkeypatch.setattr(
        tabletop_planner,
        "_table_from_resting_object",
        lambda _request, _base_T_torso: (
            np.array([0.0, 0.0, 0.0]),
            np.eye(4),
            np.array([0.0, 0.0, -1.0]),
        ),
    )

    scene, _plane_point, _down = tabletop_planner._attached_transfer_scene(
        request,
        np.eye(4),
    )

    assert base_scene_calls == [
        {"include_cube": False, "include_open_transit_table_patch": False}
    ]
    assert set(scene["cuboid"]) == {"support_cube"}


def test_direct_stack_intersects_every_endpoint_before_route_planning(
    monkeypatch,
    tmp_path,
) -> None:
    moving = _observation(0.20, 0.0, 0.030)
    support = _observation(-0.20, 0.10, 0.030)
    requests = tuple(
        build_direct_stack_request(
            moving_request=_request("left", moving, 0.060),
            support_cube=support,
            base_T_camera=np.eye(4),
            yaw_quarter_turns=yaw,
        )
        for yaw in (0, 1)
    )
    events = []

    class FakePlanner:
        def request_payload(self, _operation, *, payload, **_kwargs):
            request = TabletopPickPlaceRequest.from_json(payload["request"])
            events.append(("endpoint", request.content_sha256))
            common = 0 if request.content_sha256 == requests[0].content_sha256 else 2
            return {
                "payload": {
                    "request_sha256": request.content_sha256,
                    "common_candidate_count": common,
                }
            }

    plan = SimpleNamespace(content_sha256="c" * 64)

    def fake_plan(_planner, *, request, **_kwargs):
        events.append(("plan", request.content_sha256))
        return plan

    monkeypatch.setattr(hardware_stack, "_plan_pick_place", fake_plan)
    result, attempts = hardware_stack._find_direct_stack_plan(
        planner=FakePlanner(),
        driver=SimpleNamespace(check=lambda: None),
        options=(
            ({"moving_cube": "first", "arm": "left", "yaw_quarter_turns": 0}, requests[0]),
            (
                {"moving_cube": "first", "arm": "left", "yaw_quarter_turns": 1},
                requests[1],
            ),
        ),
        directory=tmp_path,
    )

    assert [event[0] for event in events] == ["endpoint", "endpoint", "plan"]
    assert result is not None
    assert result[1] == requests[1]
    assert result[2] is plan
    assert attempts[0]["passed"] is False
    assert attempts[1]["passed"] is True


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
        candidate_id = required_candidate_id or (
            "candidate_b"
            if current.content_sha256 == destination.content_sha256
            else "candidate_a"
        )
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
        "_tabletop_endpoint_feasibility",
        lambda _request, **_kwargs: tabletop_planner._EndpointFeasibility(
            candidate_branch_counts={"candidate_a": 2, "candidate_b": 2},
            fixed_close_viable_candidate_count=2,
            rejections=(),
            elapsed_s=0.1,
        ),
    )
    monkeypatch.setattr(
        tabletop_planner,
        "_load_shortlist",
        lambda _request: ({}, [{"candidate_id": "candidate_a"}, {"candidate_id": "candidate_b"}]),
    )
    monkeypatch.setattr(tabletop_planner, "_plan_attached_transfer", fake_transfer)

    plan = plan_tabletop_pick_place(request)

    assert plan.selected_candidate_id == "candidate_b"
    assert [value[1] for value in calls] == ["candidate_a", "candidate_b", "candidate_b"]
    assert plan.planner_provenance["rejected_pick_place_grasps"][0]["candidate_id"] == (
        "candidate_a"
    )


def test_pick_place_skips_candidates_pruned_by_batched_endpoint_intersection(
    monkeypatch,
) -> None:
    source = _request("left", _observation(0.20, 0.0, 0.020), 0.040)
    source_T_destination = np.eye(4)
    source_T_destination[0, 3] = 0.10
    request = TabletopPickPlaceRequest(
        source_request=source,
        source_T_destination_object=source_T_destination,
    )
    endpoints = (0.10, 0.20, 0.30, 0.40, 0.30, 0.20, 0.10, 0.0)
    calls = []

    def fake_plan(current, *, required_candidate_id=None, **_kwargs):
        calls.append((current.content_sha256, required_candidate_id))
        return _task(current, candidate_id=required_candidate_id, endpoints=endpoints)

    def fake_feasibility(current, **_kwargs):
        counts = (
            {"candidate_a": 4, "candidate_b": 0, "candidate_c": 3}
            if current.content_sha256 == source.content_sha256
            else {"candidate_a": 0, "candidate_b": 5, "candidate_c": 2}
        )
        return tabletop_planner._EndpointFeasibility(
            candidate_branch_counts=counts,
            fixed_close_viable_candidate_count=2,
            rejections=(),
            elapsed_s=0.1,
        )

    monkeypatch.setattr(tabletop_planner, "plan_tabletop_task", fake_plan)
    monkeypatch.setattr(
        tabletop_planner,
        "_tabletop_endpoint_feasibility",
        fake_feasibility,
    )
    monkeypatch.setattr(
        tabletop_planner,
        "_load_shortlist",
        lambda _request: (
            {},
            [
                {"candidate_id": "candidate_a"},
                {"candidate_id": "candidate_b"},
                {"candidate_id": "candidate_c"},
            ],
        ),
    )
    monkeypatch.setattr(
        tabletop_planner,
        "_plan_attached_transfer",
        lambda _request, source_task, destination_task, **_kwargs: (
            _trajectory(
                "payload_lift",
                "payload_transfer",
                source_task.trajectories[3].command_q_rad[-1][0],
                destination_task.trajectories[3].command_q_rad[-1][0],
            ),
            {"test": True},
        ),
    )

    plan = plan_tabletop_pick_place(request)

    assert plan.selected_candidate_id == "candidate_c"
    assert [required for _request_hash, required in calls] == ["candidate_c", "candidate_c"]
    assert plan.planner_provenance["common_endpoint_viable_candidate_count"] == 1


def test_pick_place_retry_excludes_failed_physical_grasp(monkeypatch) -> None:
    source = _request("left", _observation(0.20, 0.0, 0.030), 0.060)
    source_T_destination = np.eye(4)
    source_T_destination[0, 3] = 0.10
    request = TabletopPickPlaceRequest(
        source_request=source,
        source_T_destination_object=source_T_destination,
        excluded_candidate_ids=("candidate_a",),
    )
    calls = []
    feasibility_calls = []

    def fake_plan(
        current,
        *,
        required_candidate_id=None,
        excluded_candidate_ids=(),
        **_kwargs,
    ):
        calls.append((required_candidate_id, excluded_candidate_ids))
        candidate_id = required_candidate_id or "candidate_b"
        return _task(
            current,
            candidate_id=candidate_id,
            endpoints=(0.10, 0.20, 0.30, 0.40, 0.30, 0.20, 0.10, 0.0),
        )

    monkeypatch.setattr(tabletop_planner, "plan_tabletop_task", fake_plan)

    def fake_feasibility(_current, *, excluded_candidate_ids=(), **_kwargs):
        feasibility_calls.append(excluded_candidate_ids)
        return tabletop_planner._EndpointFeasibility(
            candidate_branch_counts={"candidate_b": 2},
            fixed_close_viable_candidate_count=1,
            rejections=(),
            elapsed_s=0.1,
        )

    monkeypatch.setattr(
        tabletop_planner,
        "_tabletop_endpoint_feasibility",
        fake_feasibility,
    )
    monkeypatch.setattr(
        tabletop_planner,
        "_load_shortlist",
        lambda _request: ({}, [{"candidate_id": "candidate_a"}, {"candidate_id": "candidate_b"}]),
    )
    monkeypatch.setattr(
        tabletop_planner,
        "_plan_attached_transfer",
        lambda _request, source_task, destination_task, **_kwargs: (
            _trajectory(
                "payload_lift",
                "payload_transfer",
                source_task.trajectories[3].command_q_rad[-1][0],
                destination_task.trajectories[3].command_q_rad[-1][0],
            ),
            {"test": True},
        ),
    )

    plan = plan_tabletop_pick_place(request)

    assert plan.selected_candidate_id == "candidate_b"
    assert feasibility_calls == [("candidate_a",), ("candidate_a",)]
    assert all(required != "candidate_a" for required, _excluded in calls)


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


def test_stack_runtime_warmup_populates_both_arm_pools_without_planning(
    monkeypatch,
) -> None:
    left = SimpleNamespace(arm="left")
    right = SimpleNamespace(arm="right")
    calls = []
    events = []

    def prewarm(request, *, planner_pool):
        calls.append((request.arm, planner_pool))
        return {"arm": request.arm, "task_feasibility_planning_performed": False}

    monkeypatch.setattr(tabletop_session, "prewarm_tabletop_runtime_models", prewarm)
    session = TabletopPlanningSession()

    result = session.prewarm_stack_runtime(
        (right, left),
        progress=events.append,
    )

    assert calls == [
        ("left", session._planner_pool),
        ("right", session._planner_pool),
    ]
    assert set(result["arms"]) == {"left", "right"}
    assert result["task_feasibility_planning_performed"] is False
    assert result["robot_command_authorized"] is False
    assert session._active_task is None
    assert session._execution is None
    assert events == [
        (
            "warming persistent left-arm open-hand and attached-payload MotionGen "
            "models; no robot command or task solve"
        ),
        (
            "warming persistent right-arm open-hand and attached-payload MotionGen "
            "models; no robot command or task solve"
        ),
        (
            "command-free dual-arm stack warmup complete; live loaded and clearance "
            "observations remain the only source of executable task feasibility"
        ),
    ]

    with pytest.raises(ValueError, match="duplicate arm"):
        session.prewarm_stack_runtime((left, left))
