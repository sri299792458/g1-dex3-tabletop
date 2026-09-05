from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from g1_aprilcube_calibration.joint_map import arm_indices
from g1_dex3_tabletop import hardware_stack, tabletop_perception
from g1_dex3_tabletop.hardware_stack import (
    _command_bound_snapshot,
    _dual_arm_command_snapshot,
    _install_plan_at_current_boundary,
    _ordered_arm_choices,
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
    DIRECT_STACK_YAW_QUARTER_TURNS,
    build_direct_stack_request,
)
from g1_dex3_tabletop.tabletop_contracts import (
    TabletopObservation,
    TabletopPickPlaceRequest,
    TabletopTaskPlan,
    TabletopTaskRequest,
)
from g1_dex3_tabletop.tabletop_workflow import destination_requests_for_pick_place


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


def test_command_bound_snapshot_reproduces_latest_right_elbow_regression() -> None:
    measured = np.zeros(29)
    right_indices = np.asarray(arm_indices("right"))
    measured[right_indices] = (
        -0.038637143,
        0.002565425,
        -0.151228935,
        -0.022518359,
        0.124492131,
        -0.177773997,
        0.164135948,
    )
    state = SimpleNamespace(position=measured)
    hands = SimpleNamespace(
        left=SimpleNamespace(position=np.zeros(7)),
        right=SimpleNamespace(position=np.ones(7)),
    )
    left_command = np.full(7, 0.25)
    right_command = np.asarray(
        (
            -0.033471942,
            0.004554804,
            -0.151420683,
            -0.001498028,
            0.124492131,
            -0.171242595,
            0.163572684,
        )
    )

    snapshot = _command_bound_snapshot(
        state,
        hands,
        np.concatenate((left_command, right_command)),
    )

    q29 = np.asarray(snapshot.measured_q29_rad)
    np.testing.assert_array_equal(q29[np.asarray(arm_indices("left"))], left_command)
    np.testing.assert_array_equal(q29[right_indices], right_command)
    assert abs(measured[right_indices[3]] - right_command[3]) == pytest.approx(0.021020331)


def test_new_episode_installs_same_arm_plan_at_retained_handoff(monkeypatch) -> None:
    replacement = object()
    monkeypatch.setattr(
        hardware_stack,
        "pose_set_from_trajectories",
        lambda **_kwargs: replacement,
    )
    calls = []

    class FakeExecutor:
        current_pose_id = "__handoff__"
        pose_set = SimpleNamespace(calibration_arm="left")
        dual_arm_command_q = np.zeros(14)

        @staticmethod
        def install_validated_plan(**kwargs) -> None:
            calls.append(("install", kwargs))

        @staticmethod
        def replace_validated_remaining_plan(**kwargs) -> None:
            calls.append(("replace", kwargs))

        @staticmethod
        def switch_validated_arm_plan(**kwargs) -> None:
            calls.append(("switch", kwargs))

    outbound = SimpleNamespace(
        from_pose_id="__handoff__",
        to_pose_id="clearance",
        command_q_rad=((0.0,) * 7, (0.1,) * 7),
    )
    result = _install_plan_at_current_boundary(
        FakeExecutor(),
        arm="left",
        trajectories=(outbound,),
        plan_sha256="a" * 64,
        validated_reference_state=SimpleNamespace(position=np.zeros(29)),
        model=SimpleNamespace(name="g1", sha256="b" * 64),
    )

    assert result == (outbound,)
    assert [name for name, _kwargs in calls] == ["install"]
    assert calls[0][1]["pose_set"] is replacement
    assert calls[0][1]["preserve_current_command"] is True


def test_new_episode_can_switch_selected_arm_at_retained_handoff(monkeypatch) -> None:
    monkeypatch.setattr(
        hardware_stack,
        "pose_set_from_trajectories",
        lambda **_kwargs: object(),
    )
    calls = []

    class FakeExecutor:
        current_pose_id = "__handoff__"
        pose_set = SimpleNamespace(calibration_arm="left")
        dual_arm_command_q = np.zeros(14)

        @staticmethod
        def switch_validated_arm_plan(**kwargs) -> None:
            calls.append(kwargs)

    outbound = SimpleNamespace(
        from_pose_id="__handoff__",
        to_pose_id="clearance",
        command_q_rad=((0.0,) * 7, (0.1,) * 7),
    )
    _install_plan_at_current_boundary(
        FakeExecutor(),
        arm="right",
        trajectories=(outbound,),
        plan_sha256="a" * 64,
        validated_reference_state=SimpleNamespace(position=np.zeros(29)),
        model=SimpleNamespace(name="g1", sha256="b" * 64),
    )

    assert len(calls) == 1
    assert calls[0]["boundary_pose_id"] == "__handoff__"


def test_stack_plan_install_rejects_latest_measured_right_elbow_start() -> None:
    command_right = np.asarray(
        (
            -0.033471942,
            0.004554804,
            -0.151420683,
            -0.001498028,
            0.124492131,
            -0.171242595,
            0.163572684,
        )
    )
    measured_right = command_right.copy()
    measured_right[3] = -0.022518359

    class FakeExecutor:
        current_pose_id = "__handoff__"
        pose_set = SimpleNamespace(calibration_arm="left")
        dual_arm_command_q = np.concatenate((np.zeros(7), command_right))

    outbound = SimpleNamespace(
        from_pose_id="__handoff__",
        to_pose_id="clearance",
        command_q_rad=(tuple(measured_right), tuple(measured_right + 0.1)),
    )

    with pytest.raises(ValueError, match=r"active command by 0\.021020331rad"):
        _install_plan_at_current_boundary(
            FakeExecutor(),
            arm="right",
            trajectories=(outbound,),
            plan_sha256="a" * 64,
            validated_reference_state=SimpleNamespace(position=np.zeros(29)),
            model=SimpleNamespace(name="g1", sha256="b" * 64),
        )


def test_clearance_perception_failure_is_recoverable_only_with_healthy_control() -> None:
    checks = []

    class HealthyDriver:
        @staticmethod
        def check() -> None:
            checks.append("healthy")

    rejection = hardware_stack._clearance_perception_rejection(
        HealthyDriver(),
        ValueError("primary cube marker is hidden"),
    )

    assert isinstance(rejection, hardware_stack.TabletopTaskRejected)
    assert checks == ["healthy"]
    assert "primary cube marker is hidden" in str(rejection)

    class FaultedDriver:
        @staticmethod
        def check() -> None:
            raise RuntimeError("controller fault")

    with pytest.raises(RuntimeError, match="controller fault"):
        hardware_stack._clearance_perception_rejection(
            FaultedDriver(),
            ValueError("camera failed"),
        )


def test_supported_perception_failure_rejects_episode_only_with_healthy_control() -> None:
    checks = []

    class HealthyDriver:
        @staticmethod
        def check() -> None:
            checks.append("healthy")

    rejection = hardware_stack._supported_perception_rejection(
        HealthyDriver(),
        ValueError("secondary cube passed only 2/5 frames"),
    )

    assert isinstance(rejection, hardware_stack.StackPerceptionBlocked)
    assert isinstance(rejection, hardware_stack.TabletopTaskRejected)
    assert checks == ["healthy"]
    assert str(rejection) == (
        "PERCEPTION BLOCKED — both arms remain supported. Remove any occlusion, "
        "reposition the cubes, and press SPACE to try again."
    )
    assert rejection.detail == "secondary cube passed only 2/5 frames"

    class FaultedDriver:
        @staticmethod
        def check() -> None:
            raise RuntimeError("controller fault")

    with pytest.raises(RuntimeError, match="controller fault"):
        hardware_stack._supported_perception_rejection(
            FaultedDriver(),
            ValueError("camera failed"),
        )


def test_paired_cube_observation_names_the_failed_detector(monkeypatch) -> None:
    def fail_first(_images, *, detector, **_kwargs):
        if detector == "secondary-detector":
            raise ValueError("marker is hidden")
        raise AssertionError("the second detector must not run after the first fails")

    monkeypatch.setattr(tabletop_perception, "observe_resting_cube", fail_first)

    with pytest.raises(
        ValueError,
        match=r"secondary cube \(tag IDs 20-25\): marker is hidden",
    ):
        tabletop_perception.observe_resting_cube_pair(
            (),
            camera_info=object(),
            first_detector="secondary-detector",
            second_detector="primary-detector",
            snapshot=RobotSnapshot((0.0,) * 29, (0.0,) * 7, (0.0,) * 7),
            base_T_camera=np.eye(4),
            first_label="secondary cube (tag IDs 20-25)",
            second_label="primary cube (tag IDs 10-15)",
        )


def test_ordered_arm_choices_keep_both_cube_directions_for_each_arm() -> None:
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

    selected = _ordered_arm_choices(
        reference_request=_request("left", upper, 0.060),
        upper_observation=upper,
        bottom_observation=bottom,
        model=FakeModel(),
    )

    assert selected[0]["arm"] == "right"
    assert selected[0]["moving_cube_order"] == ("secondary", "primary")
    assert selected[0]["source_hand_distances_m"]["secondary"] == pytest.approx(
        np.hypot(0.06, 0.03)
    )
    assert selected[1]["arm"] == "left"
    assert selected[1]["moving_cube_order"] == ("primary", "secondary")
    assert set(selected[0]["moving_cube_order"]) == {"primary", "secondary"}
    assert set(selected[1]["moving_cube_order"]) == {"primary", "secondary"}


def _task(
    request: TabletopTaskRequest,
    *,
    candidate_id: str,
    endpoints: tuple[float, ...],
    branch_counts: dict[str, int] | None = None,
    goalset_index: int = 0,
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
        planner_provenance={
            "selected_goalset_index": goalset_index,
            **(
                {}
                if branch_counts is None
                else {"pregrasp_ik_unique_branches_by_candidate": branch_counts}
            ),
        },
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
    destinations = destination_requests_for_pick_place(request)
    assert len(destinations) == 4
    destination = destinations[0]
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
    )
    assert DIRECT_STACK_YAW_QUARTER_TURNS == (0, 1, 3, 2)
    assert len(request.source_T_destination_objects) == 4
    for transform, quarter_turns in zip(
        request.source_T_destination_objects,
        DIRECT_STACK_YAW_QUARTER_TURNS,
        strict=True,
    ):
        np.testing.assert_allclose(
            np.asarray(transform)[:3, :3],
            Rotation.from_euler("z", 90.0 * quarter_turns, degrees=True).as_matrix(),
            atol=1.0e-12,
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

    assert base_scene_calls == [{"include_cube": False, "include_open_transit_table_patch": False}]
    assert set(scene["cuboid"]) == {"support_cube"}


def test_direct_stack_submits_one_yaw_goalset_before_route_planning(
    monkeypatch,
    tmp_path,
) -> None:
    moving = _observation(0.20, 0.0, 0.030)
    support = _observation(-0.20, 0.10, 0.030)
    request = build_direct_stack_request(
        moving_request=_request("left", moving, 0.060),
        support_cube=support,
        base_T_camera=np.eye(4),
    )
    events = []

    class FakePlanner:
        def request_payload(self, _operation, *, payload, **_kwargs):
            request = TabletopPickPlaceRequest.from_json(payload["request"])
            events.append(("endpoint", request.content_sha256))
            return {
                "payload": {
                    "request_sha256": request.content_sha256,
                    "common_candidate_count": 2,
                }
            }

    plan = SimpleNamespace(content_sha256="c" * 64, selected_destination_index=2)

    def fake_plan(_planner, *, request, **_kwargs):
        events.append(("plan", request.content_sha256))
        return plan

    monkeypatch.setattr(hardware_stack, "_plan_pick_place", fake_plan)
    result, attempts = hardware_stack._find_direct_stack_plan(
        planner=FakePlanner(),
        driver=SimpleNamespace(check=lambda: None),
        metadata={"moving_cube": "first", "arm": "left"},
        request=request,
        directory=tmp_path,
    )

    assert [event[0] for event in events] == ["endpoint", "plan"]
    assert result is not None
    assert result[1] == request
    assert result[2] is plan
    assert len(attempts) == 1
    assert attempts[0]["passed"] is True
    assert attempts[0]["selected_destination_index"] == 2


def test_pick_place_session_consumes_matching_endpoint_analysis_once(monkeypatch) -> None:
    request = SimpleNamespace(content_sha256="a" * 64)
    analysis = SimpleNamespace(request_sha256=request.content_sha256)
    plan = object()
    received = []

    monkeypatch.setattr(
        tabletop_session,
        "_pick_place_endpoint_analysis",
        lambda received_request, **_kwargs: analysis if received_request is request else None,
    )
    monkeypatch.setattr(
        tabletop_session,
        "_pick_place_endpoint_analysis_report",
        lambda received_analysis: {"request_sha256": received_analysis.request_sha256},
    )

    def fake_plan(received_request, *, endpoint_analysis, **_kwargs):
        received.append((received_request, endpoint_analysis))
        return plan

    monkeypatch.setattr(tabletop_session, "plan_tabletop_pick_place", fake_plan)
    session = TabletopPlanningSession()

    assert session.analyze_pick_place_endpoints(request) == {
        "request_sha256": request.content_sha256
    }
    assert session.plan_pick_place(request) is plan
    assert session._pending_pick_place_endpoint_analysis is None
    assert session.plan_pick_place(request) is plan
    assert received == [(request, analysis), (request, None)]


def test_pick_place_session_never_reuses_another_request_analysis(monkeypatch) -> None:
    analyzed_request = SimpleNamespace(content_sha256="a" * 64)
    planned_request = SimpleNamespace(content_sha256="b" * 64)
    analysis = SimpleNamespace(request_sha256=analyzed_request.content_sha256)
    received = []

    monkeypatch.setattr(
        tabletop_session,
        "_pick_place_endpoint_analysis",
        lambda _request, **_kwargs: analysis,
    )
    monkeypatch.setattr(
        tabletop_session,
        "_pick_place_endpoint_analysis_report",
        lambda _analysis: {},
    )
    monkeypatch.setattr(
        tabletop_session,
        "plan_tabletop_pick_place",
        lambda _request, *, endpoint_analysis, **_kwargs: received.append(endpoint_analysis),
    )
    session = TabletopPlanningSession()

    session.analyze_pick_place_endpoints(analyzed_request)
    session.plan_pick_place(planned_request)

    assert received == [None]
    assert session._pending_pick_place_endpoint_analysis is None


def test_pick_place_session_clears_endpoint_analysis_when_planning_fails(
    monkeypatch,
) -> None:
    request = SimpleNamespace(content_sha256="a" * 64)
    analysis = SimpleNamespace(request_sha256=request.content_sha256)
    monkeypatch.setattr(
        tabletop_session,
        "_pick_place_endpoint_analysis",
        lambda _request, **_kwargs: analysis,
    )
    monkeypatch.setattr(
        tabletop_session,
        "_pick_place_endpoint_analysis_report",
        lambda _analysis: {},
    )

    def fail(*_args, **_kwargs):
        raise RuntimeError("planning failed")

    monkeypatch.setattr(tabletop_session, "plan_tabletop_pick_place", fail)
    session = TabletopPlanningSession()
    session.analyze_pick_place_endpoints(request)

    with pytest.raises(RuntimeError, match="planning failed"):
        session.plan_pick_place(request)

    assert session._pending_pick_place_endpoint_analysis is None


def test_pick_place_tries_next_grasp_when_first_cannot_place(monkeypatch) -> None:
    source = _request("left", _observation(0.20, 0.0, 0.020), 0.040)
    source_T_destination = np.eye(4)
    source_T_destination[0, 3] = 0.10
    alternate_destination = source_T_destination.copy()
    alternate_destination[1, 3] = 0.20
    request = TabletopPickPlaceRequest(
        source_request=source,
        source_T_destination_objects=(source_T_destination, alternate_destination),
    )
    source_endpoints = (0.10, 0.20, 0.30, 0.40, 0.30, 0.20, 0.10, 0.0)
    destination_endpoints = (0.11, 0.21, 0.31, 0.41, 0.31, 0.21, 0.11, 0.0)
    calls = []

    def fake_source_plan(current, *, required_candidate_id=None, **_kwargs):
        calls.append(("source", required_candidate_id))
        return _task(
            current,
            candidate_id=required_candidate_id,
            endpoints=source_endpoints,
        )

    def fake_destination_plan(requests, *, required_candidate_id, **_kwargs):
        calls.append(("destination", required_candidate_id))
        if required_candidate_id == "candidate_a":
            raise RuntimeError("candidate_a cannot reach destination")
        return _task(
            requests[0],
            candidate_id=required_candidate_id,
            endpoints=destination_endpoints,
            goalset_index=0,
        )

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

    monkeypatch.setattr(tabletop_planner, "plan_tabletop_task", fake_source_plan)
    monkeypatch.setattr(
        tabletop_planner,
        "_plan_tabletop_task_goalset",
        fake_destination_plan,
    )
    monkeypatch.setattr(
        tabletop_planner,
        "_tabletop_endpoint_feasibility",
        lambda _request, **_kwargs: tabletop_planner._EndpointFeasibility(
            candidate_branch_counts={"candidate_a": 2, "candidate_b": 2},
            candidate_best_joint_distance_rad={"candidate_a": 1.0, "candidate_b": 1.0},
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
    assert calls == [
        ("destination", "candidate_a"),
        ("destination", "candidate_b"),
        ("source", "candidate_b"),
    ]
    assert plan.planner_provenance["rejected_pick_place_grasps"][0]["candidate_id"] == (
        "candidate_a"
    )


def test_pick_place_orders_common_grasps_by_endpoint_joint_distance(monkeypatch) -> None:
    source = _request("left", _observation(0.20, 0.0, 0.020), 0.040)
    source_T_destination = np.eye(4)
    source_T_destination[0, 3] = 0.10
    request = TabletopPickPlaceRequest(
        source_request=source,
        source_T_destination_objects=(source_T_destination,),
    )
    endpoints = (0.10, 0.20, 0.30, 0.40, 0.30, 0.20, 0.10, 0.0)
    calls = []

    def fake_source_plan(current, *, required_candidate_id=None, **_kwargs):
        calls.append(("source", required_candidate_id))
        return _task(current, candidate_id=required_candidate_id, endpoints=endpoints)

    def fake_destination_plan(requests, *, required_candidate_id, **_kwargs):
        calls.append(("destination", required_candidate_id))
        return _task(
            requests[0],
            candidate_id=required_candidate_id,
            endpoints=endpoints,
            goalset_index=0,
        )

    feasibility_calls = 0

    def fake_feasibility(_request, **_kwargs):
        nonlocal feasibility_calls
        feasibility_calls += 1
        best = (
            {"candidate_a": 2.0, "candidate_b": 0.4}
            if feasibility_calls == 1
            else {"candidate_a": 1.5, "candidate_b": 0.5}
        )
        return tabletop_planner._EndpointFeasibility(
            candidate_branch_counts={"candidate_a": 2, "candidate_b": 2},
            candidate_best_joint_distance_rad=best,
            fixed_close_viable_candidate_count=2,
            rejections=(),
            elapsed_s=0.1,
        )

    monkeypatch.setattr(tabletop_planner, "plan_tabletop_task", fake_source_plan)
    monkeypatch.setattr(
        tabletop_planner,
        "_plan_tabletop_task_goalset",
        fake_destination_plan,
    )
    monkeypatch.setattr(tabletop_planner, "_tabletop_endpoint_feasibility", fake_feasibility)
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

    assert calls == [("destination", "candidate_b"), ("source", "candidate_b")]
    assert plan.selected_candidate_id == "candidate_b"
    assert plan.planner_provenance["ranked_common_candidate_ids"] == [
        "candidate_b",
        "candidate_a",
    ]
    assert plan.planner_provenance["common_candidate_joint_distance_rad"] == {
        "candidate_a": {"source": 2.0, "destination": 1.5, "total": 3.5},
        "candidate_b": {"source": 0.4, "destination": 0.5, "total": 0.9},
    }


def test_pick_place_reuses_matching_analysis_and_its_endpoint_branches(monkeypatch) -> None:
    source = _request("left", _observation(0.20, 0.0, 0.020), 0.040)
    source_T_destination = np.eye(4)
    source_T_destination[0, 3] = 0.10
    request = TabletopPickPlaceRequest(
        source_request=source,
        source_T_destination_objects=(source_T_destination,),
    )
    source_feasibility = tabletop_planner._EndpointFeasibility(
        candidate_branch_counts={"candidate_a": 1},
        candidate_best_joint_distance_rad={"candidate_a": 1.0},
        fixed_close_viable_candidate_count=1,
        rejections=(),
        elapsed_s=0.1,
    )
    destination_feasibility = tabletop_planner._EndpointFeasibility(
        candidate_branch_counts={"candidate_a": 1},
        candidate_best_joint_distance_rad={"candidate_a": 2.0},
        fixed_close_viable_candidate_count=1,
        rejections=(),
        elapsed_s=0.1,
    )
    analysis = tabletop_planner._PickPlaceEndpointAnalysis(
        request_sha256=request.content_sha256,
        candidate_ids=("candidate_a",),
        source=source_feasibility,
        destination=destination_feasibility,
    )
    endpoints = (0.10, 0.20, 0.30, 0.40, 0.30, 0.20, 0.10, 0.0)
    received = []

    monkeypatch.setattr(
        tabletop_planner,
        "_pick_place_endpoint_analysis",
        lambda *_args, **_kwargs: pytest.fail("matching endpoint analysis was recomputed"),
    )
    monkeypatch.setattr(
        tabletop_planner,
        "plan_tabletop_task",
        lambda current, *, endpoint_feasibility, **_kwargs: (
            received.append(("source", endpoint_feasibility))
            or _task(current, candidate_id="candidate_a", endpoints=endpoints)
        ),
    )
    monkeypatch.setattr(
        tabletop_planner,
        "_plan_tabletop_task_goalset",
        lambda requests, *, endpoint_feasibility, **_kwargs: (
            received.append(("destination", endpoint_feasibility))
            or _task(
                requests[0],
                candidate_id="candidate_a",
                endpoints=endpoints,
                goalset_index=0,
            )
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

    plan = plan_tabletop_pick_place(request, endpoint_analysis=analysis)

    assert received == [
        ("destination", destination_feasibility),
        ("source", source_feasibility),
    ]
    assert plan.planner_provenance["endpoint_analysis_reused"] is True


def test_pick_place_skips_candidates_pruned_by_batched_endpoint_intersection(
    monkeypatch,
) -> None:
    source = _request("left", _observation(0.20, 0.0, 0.020), 0.040)
    source_T_destination = np.eye(4)
    source_T_destination[0, 3] = 0.10
    alternate_destination = source_T_destination.copy()
    alternate_destination[1, 3] = 0.20
    request = TabletopPickPlaceRequest(
        source_request=source,
        source_T_destination_objects=(source_T_destination, alternate_destination),
    )
    endpoints = (0.10, 0.20, 0.30, 0.40, 0.30, 0.20, 0.10, 0.0)
    calls = []

    def fake_source_plan(current, *, required_candidate_id=None, **_kwargs):
        calls.append(("source", required_candidate_id))
        return _task(current, candidate_id=required_candidate_id, endpoints=endpoints)

    def fake_destination_plan(requests, *, required_candidate_id, **_kwargs):
        calls.append(("destination", required_candidate_id))
        return _task(
            requests[1],
            candidate_id=required_candidate_id,
            endpoints=endpoints,
            goalset_index=1,
        )

    def fake_feasibility(current, **_kwargs):
        counts = (
            {"candidate_a": 4, "candidate_b": 0, "candidate_c": 3}
            if current.content_sha256 == source.content_sha256
            else {"candidate_a": 0, "candidate_b": 5, "candidate_c": 2}
        )
        return tabletop_planner._EndpointFeasibility(
            candidate_branch_counts=counts,
            candidate_best_joint_distance_rad={
                candidate_id: 1.0 for candidate_id, count in counts.items() if count > 0
            },
            fixed_close_viable_candidate_count=2,
            rejections=(),
            elapsed_s=0.1,
        )

    monkeypatch.setattr(tabletop_planner, "plan_tabletop_task", fake_source_plan)
    monkeypatch.setattr(
        tabletop_planner,
        "_plan_tabletop_task_goalset",
        fake_destination_plan,
    )
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
    transfer_requests = []

    def fake_transfer(selected_request, source_task, destination_task, **_kwargs):
        transfer_requests.append(selected_request)
        return (
            _trajectory(
                "payload_lift",
                "payload_transfer",
                source_task.trajectories[3].command_q_rad[-1][0],
                destination_task.trajectories[3].command_q_rad[-1][0],
            ),
            {"test": True},
        )

    monkeypatch.setattr(tabletop_planner, "_plan_attached_transfer", fake_transfer)

    plan = plan_tabletop_pick_place(request)

    assert plan.selected_candidate_id == "candidate_c"
    assert plan.selected_destination_index == 1
    assert len(transfer_requests) == 1
    assert transfer_requests[0].source_T_destination_objects == (
        request.source_T_destination_objects[1],
    )
    assert calls == [("destination", "candidate_c"), ("source", "candidate_c")]
    assert plan.planner_provenance["common_endpoint_viable_candidate_count"] == 1


def test_pick_place_retry_excludes_failed_physical_grasp(monkeypatch) -> None:
    source = _request("left", _observation(0.20, 0.0, 0.030), 0.060)
    source_T_destination = np.eye(4)
    source_T_destination[0, 3] = 0.10
    request = TabletopPickPlaceRequest(
        source_request=source,
        source_T_destination_objects=(source_T_destination,),
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

    def fake_destination_plan(requests, *, required_candidate_id, **_kwargs):
        calls.append((required_candidate_id, ()))
        return _task(
            requests[0],
            candidate_id=required_candidate_id,
            endpoints=(0.10, 0.20, 0.30, 0.40, 0.30, 0.20, 0.10, 0.0),
            goalset_index=0,
        )

    monkeypatch.setattr(tabletop_planner, "plan_tabletop_task", fake_plan)
    monkeypatch.setattr(
        tabletop_planner,
        "_plan_tabletop_task_goalset",
        fake_destination_plan,
    )

    def fake_feasibility(_current, *, excluded_candidate_ids=(), **_kwargs):
        feasibility_calls.append(excluded_candidate_ids)
        return tabletop_planner._EndpointFeasibility(
            candidate_branch_counts={"candidate_b": 2},
            candidate_best_joint_distance_rad={"candidate_b": 1.0},
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
    session = TabletopPlanningSession()
    monkeypatch.setattr(
        session._planner_pool,
        "pick_place_retention_validator",
        lambda request, plan: FakeValidator(request, plan),
    )

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
