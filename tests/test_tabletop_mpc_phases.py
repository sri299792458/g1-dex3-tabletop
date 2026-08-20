from types import SimpleNamespace

import numpy as np
import pytest

from g1_aprilcube_calibration.transforms import invert_transform
from g1_dex3_tabletop.planning.curobo_backend import _clearance_from_activation_cost
from g1_dex3_tabletop.planning.tabletop_mpc import (
    MovingGraspMPC,
    _bounded_route_goal,
    _fixture_excluded_links,
    _nominal_base_T_live_base,
    _payload_link_spheres,
    _reserved_velocity_constraint_is_safe,
    _world_collision_buffer_deltas,
    mpc_phase_spec,
)
from g1_dex3_tabletop.planning.tabletop_planner import _world_cuboid_clearances
from g1_dex3_tabletop.planning.tabletop_session import TabletopPlanningSession


class _ArrayValue:
    def __init__(self, value) -> None:
        self._value = np.asarray(value)

    def detach(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return self._value


def test_curobo_activation_cost_is_inverted_to_signed_clearance() -> None:
    activation = 0.01
    outside_clearance = 0.004
    outside_cost = 0.5 * (activation - outside_clearance) ** 2 / activation
    penetration = 0.002
    penetration_cost = activation + penetration - 0.5 * activation

    clearance = _clearance_from_activation_cost(
        np.asarray([0.0, outside_cost, penetration_cost]),
        activation,
    )

    np.testing.assert_allclose(clearance, [activation, outside_clearance, -penetration])


def test_tightened_velocity_residual_requires_all_real_constraints_to_pass() -> None:
    summary = [
        {"name": "cspace", "maximum": 0.001},
        {"name": "self_collision", "maximum": 0.0},
        {"name": "scene_collision", "maximum": 0.0},
    ]
    cspace = {
        name: {"maximum_violation": 0.0005 if name == "velocity" else 0.0}
        for name in ("position", "velocity", "acceleration", "jerk")
    }

    assert _reserved_velocity_constraint_is_safe(
        summary,
        cspace,
        full_state_peak_velocity_rad_s=0.0955,
        physical_velocity_limit_rad_s=0.1,
    )
    assert not _reserved_velocity_constraint_is_safe(
        summary,
        cspace,
        full_state_peak_velocity_rad_s=0.101,
        physical_velocity_limit_rad_s=0.1,
    )
    summary[1]["maximum"] = 1.0e-9
    assert not _reserved_velocity_constraint_is_safe(
        summary,
        cspace,
        full_state_peak_velocity_rad_s=0.0955,
        physical_velocity_limit_rad_s=0.1,
    )


def test_route_lookahead_interpolates_without_leaving_the_frozen_path() -> None:
    route = np.asarray([[0.0] * 7, [0.02] * 7, [0.04] * 7, [0.07] * 7])

    goal, segment_end, terminal = _bounded_route_goal(
        route,
        current_q=np.zeros(7),
        route_progress_index=0,
        maximum_distance_rad=0.05,
    )
    np.testing.assert_allclose(goal, 0.05)
    assert segment_end == 3
    assert not terminal

    goal, segment_end, terminal = _bounded_route_goal(
        route,
        current_q=np.zeros(7),
        route_progress_index=0,
        maximum_distance_rad=0.08,
    )
    np.testing.assert_allclose(goal, 0.07)
    assert segment_end == 3
    assert terminal


def test_strict_mpc_check_rejects_translated_command_outside_hard_limits() -> None:
    limits = _ArrayValue(np.vstack((np.full(7, -1.0), np.full(7, 1.0))))
    checker = SimpleNamespace(
        kinematics=SimpleNamespace(
            get_joint_limits=lambda: SimpleNamespace(position=limits),
        )
    )
    controller = object.__new__(MovingGraspMPC)
    controller._strict_checker = checker
    controller._active_world_correction = {}
    controller.names = tuple(f"joint_{index}" for index in range(7))

    diagnostics = controller._strict_window_diagnostics(
        np.asarray([[0.0] * 6 + [1.01]], dtype=np.float64)
    )

    assert not diagnostics["strict_valid"]
    assert diagnostics["strict_failure"] == "joint_limit"
    assert diagnostics["strict_failure_sample"] == 0
    assert diagnostics["strict_failure_links"] == ["joint_6"]
    assert diagnostics["strict_failure_position_rad"] == pytest.approx(1.01)
    assert diagnostics["strict_failure_limits_rad"] == [-1.0, 1.0]


def test_every_normal_tabletop_motion_has_one_physical_mpc_state() -> None:
    spec = mpc_phase_spec("grasp_approach")

    assert spec.mode == "open_contact"
    assert not spec.attached_payload
    assert not spec.include_fixture_in_optimizer
    assert spec.reference_fixed_goal


def test_live_body_frame_maps_into_the_frozen_strict_scene() -> None:
    base_T_torso0 = np.eye(4)
    base_T_torso0[:3, 3] = [0.1, -0.2, 0.6]
    reference_T_torso0 = np.eye(4)
    reference_T_torso0[:3, 3] = [-0.3, 0.4, 0.8]
    reference_T_torso = np.eye(4)
    reference_T_torso[:3, :3] = np.asarray(
        [
            [0.999390827, 0.0, 0.034899497],
            [0.0, 1.0, 0.0],
            [-0.034899497, 0.0, 0.999390827],
        ]
    )
    reference_T_torso[:3, 3] = [-0.29, 0.395, 0.804]
    nominal_base_T_live_base = _nominal_base_T_live_base(
        base_T_torso0=base_T_torso0,
        reference_T_torso0=reference_T_torso0,
        reference_T_torso=reference_T_torso,
    )

    reference_point = np.asarray([0.2, -0.1, 0.05, 1.0])
    nominal_base_T_reference = base_T_torso0 @ invert_transform(reference_T_torso0)
    live_base_T_reference = base_T_torso0 @ invert_transform(reference_T_torso)
    live_point = live_base_T_reference @ reference_point

    np.testing.assert_allclose(
        nominal_base_T_live_base @ live_point,
        nominal_base_T_reference @ reference_point,
        atol=1.0e-9,
    )


def test_planning_session_rejects_a_window_without_a_live_target() -> None:
    session = TabletopPlanningSession()
    session._active_phase_mpc = object()

    with pytest.raises(TypeError, match="requires one live target"):
        session.step_moving_grasp_mpc({})


def test_planning_session_forwards_live_cartesian_target_and_route_progress() -> None:
    received = {}
    result = object()

    class FakeMPC:
        spec = SimpleNamespace(phase="grasp_approach")

        def next_moving_target_window(self, **kwargs):
            received.update(kwargs)
            return result

    target = {
        "reference_T_camera": np.eye(4).tolist(),
        "camera_T_object": np.eye(4).tolist(),
        "source_monotonic_s": 1.0,
        "source_frame_sha256": "f" * 64,
    }
    session = TabletopPlanningSession()
    session._active_phase_mpc = FakeMPC()

    actual = session.step_moving_grasp_mpc(
        {
            "phase": "grasp_approach",
            "handoff_predicted_q_rad": [0.0] * 7,
            "handoff_predicted_dq_rad_s": [0.0] * 7,
            "handoff_predicted_ddq_rad_s2": [0.0] * 7,
            "handoff_command_q_rad": [0.0] * 7,
            "source_state_monotonic_s": 1.0,
            "valid_from_monotonic_s": 1.2,
            "predecessor_sha256": None,
            "committed_route_progress_index": 99,
            "moving_target": target,
        }
    )

    assert actual is result
    assert received["committed_route_progress_index"] == 99
    np.testing.assert_allclose(received["reference_T_camera"], np.eye(4))
    np.testing.assert_allclose(received["camera_T_object"], np.eye(4))
    assert received["target_provenance"] == target


def test_supported_routes_remain_frozen_instead_of_entering_mpc() -> None:
    for phase in ("clearance", "__handoff__"):
        with pytest.raises(ValueError, match="unsupported moving-grasp MPC phase"):
            mpc_phase_spec(phase)


def test_fixture_exclusions_match_contact_and_attached_payload_policies() -> None:
    contact = _fixture_excluded_links(mpc_phase_spec("grasp_approach"), arm="left")

    assert contact == (
        "left_hand_thumb_2_link",
        "left_hand_middle_1_link",
        "left_hand_index_1_link",
    )


def test_contact_routes_keep_cube_for_noncontact_links_without_attaching_it() -> None:
    spec = mpc_phase_spec("grasp_approach")
    assert spec.allow_fingertip_cube_contact
    assert spec.include_cube_in_optimizer
    assert not spec.attached_payload
    assert spec.include_table_patch


def test_unknown_mpc_phase_is_rejected() -> None:
    with pytest.raises(ValueError, match="unsupported moving-grasp MPC phase"):
        mpc_phase_spec("invented_transition")


def test_planning_session_binds_the_warmed_moving_grasp_solver() -> None:
    class WarmedMPC:
        def __init__(self) -> None:
            self.spec = mpc_phase_spec("grasp_approach")
            self.binding = None

        def bind_moving_grasp_execution(
            self,
            clearance_request,
            execution,
            *,
            loaded_request,
            reference_T_camera0,
        ) -> dict:
            self.binding = (
                clearance_request,
                execution,
                loaded_request,
                np.asarray(reference_T_camera0),
            )
            return {
                "rebind_time_s": 0.2,
                "setup_time_s": 0.3,
                "total_time_s": 0.5,
            }

    session = TabletopPlanningSession()
    loaded = object()
    clearance = object()
    execution = SimpleNamespace(
        content_sha256="plan",
        task=SimpleNamespace(content_sha256="task"),
    )
    controller = WarmedMPC()
    session._loaded_request = loaded
    session._clearance_request = clearance
    session._execution = execution
    session._phase_mpc = controller
    reference = np.eye(4)

    result = session.prepare_moving_grasp_mpc(
        reference_T_camera0=reference,
    )

    assert result["reused_warm_model"]
    assert result["preparation_time_s"] == pytest.approx(0.5)
    assert controller.binding[:3] == (clearance, execution, loaded)
    assert np.array_equal(controller.binding[3], reference)
    assert session._active_phase_mpc is controller


def test_clearance_replan_retains_warmed_mpc_for_live_binding(monkeypatch) -> None:
    events = []
    request = SimpleNamespace(content_sha256="fresh", observation=object())
    task = object()
    execution = SimpleNamespace(supported_escape=object(), task=object())
    replacement = SimpleNamespace(task=task)

    class Validator:
        cache_build_s = 0.25

        def __init__(self, received_request, received_task) -> None:
            assert received_request is request
            assert received_task is task

    class Controller:
        def close(self) -> None:
            events.append("closed")

    monkeypatch.setattr(
        "g1_dex3_tabletop.planning.tabletop_session.request_at_clearance_observation",
        lambda loaded, escape, observation: request,
    )
    monkeypatch.setattr(
        "g1_dex3_tabletop.planning.tabletop_session.plan_tabletop_task",
        lambda received, planner_pool, progress: task,
    )
    monkeypatch.setattr(
        "g1_dex3_tabletop.planning.tabletop_session.combine_tabletop_plans",
        lambda **kwargs: replacement,
    )
    session = TabletopPlanningSession()
    monkeypatch.setattr(
        session._planner_pool,
        "retention_validator",
        lambda received_request, received_task: Validator(received_request, received_task),
    )
    session._loaded_request = object()
    session._clearance_request = object()
    session._supported_escape = execution.supported_escape
    session._execution = execution
    session._retention_validator = object()
    session._phase_mpc = Controller()
    session._active_phase_mpc = session._phase_mpc

    assert session.replan_at_clearance(request, progress=events.append) is replacement
    assert session._clearance_request is request
    assert session._execution is replacement
    assert isinstance(session._retention_validator, Validator)
    assert session._phase_mpc is not None
    assert session._active_phase_mpc is None
    assert "closed" not in events


def test_clearance_pregrasp_stage_does_not_create_a_complete_task(monkeypatch) -> None:
    events = []
    request = SimpleNamespace(content_sha256="fresh", observation=object())
    pregrasp = object()

    class Controller:
        def close(self) -> None:
            events.append("closed")

    monkeypatch.setattr(
        "g1_dex3_tabletop.planning.tabletop_session.request_at_clearance_observation",
        lambda loaded, escape, observation: request,
    )

    def plan(received, *, planner_pool, progress):
        assert received is request
        assert planner_pool is session._planner_pool
        progress("pregrasp ready")
        return pregrasp

    monkeypatch.setattr(
        "g1_dex3_tabletop.planning.tabletop_session.plan_tabletop_pregrasp",
        plan,
    )
    session = TabletopPlanningSession()
    session._loaded_request = object()
    session._supported_escape = object()
    session._execution = object()
    session._active_task = object()
    session._retention_validator = object()
    session._phase_mpc = Controller()
    session._active_phase_mpc = session._phase_mpc

    assert session.plan_pregrasp_at_clearance(request, progress=events.append) is pregrasp
    assert session._clearance_request is request
    assert session._pregrasp_plan is pregrasp
    assert session._execution is None
    assert session._active_task is None
    assert session._retention_validator is None
    assert session._phase_mpc is None
    assert session._active_phase_mpc is None
    assert events == ["pregrasp ready", "closed"]


def test_escape_only_session_retains_exact_reverse_for_later_boundary_replan(
    monkeypatch,
) -> None:
    request = object()
    escape = object()
    clearance = object()
    monkeypatch.setattr(
        "g1_dex3_tabletop.planning.tabletop_session.plan_supported_escape",
        lambda received, progress: escape,
    )
    monkeypatch.setattr(
        "g1_dex3_tabletop.planning.tabletop_session.request_at_clearance",
        lambda received, received_escape: clearance,
    )
    session = TabletopPlanningSession()

    assert session.plan_escape(request) is escape
    assert session._loaded_request is request
    assert session._supported_escape is escape
    assert session._clearance_request is clearance
    assert session._execution is None
    assert session._retention_validator is None


def test_runtime_warmup_populates_pool_without_installing_nominal_task(
    monkeypatch,
) -> None:
    request = object()
    result = {"planning_performed": False}
    events = []

    def prewarm(received, *, planner_pool):
        assert received is request
        assert planner_pool is session._planner_pool
        return result

    monkeypatch.setattr(
        "g1_dex3_tabletop.planning.tabletop_session.prewarm_tabletop_runtime_models",
        prewarm,
    )

    class WarmedMPC:
        def __init__(self, received, execution, *, phase) -> None:
            assert received is request
            assert execution is None
            assert phase == "grasp_approach"
            self.closed = False

        def setup_at_frozen_route_start(self, *, validate_strict_start) -> float:
            assert not validate_strict_start
            return 0.4

        def close(self) -> None:
            self.closed = True

    monkeypatch.setattr(
        "g1_dex3_tabletop.planning.tabletop_session.MovingGraspMPC",
        WarmedMPC,
    )
    session = TabletopPlanningSession()

    warmed = session.prewarm_runtime(
        request,
        moving_grasp_mpc=True,
        progress=events.append,
    )
    assert warmed["planning_performed"] is False
    assert warmed["moving_grasp_mpc"]["retained_for_live_binding"] is True
    assert session._phase_mpc is not None
    assert session._active_task is None
    assert session._execution is None
    assert events == [
        (
            "constructing persistent open-hand and attached-payload MotionGen models; "
            "no task solve or robot command is being used as a gate"
        ),
        (
            "command-free runtime warmup complete; fresh loaded and clearance "
            "observations remain the only source of executable task feasibility"
        ),
    ]


def test_payload_uses_frozen_planner_cuboid_cover_in_grasp_frame() -> None:
    object_T_grasp = np.eye(4)
    object_T_grasp[:3, 3] = (0.01, -0.02, 0.03)
    spheres = _payload_link_spheres(
        (0.04, 0.04, 0.04),
        object_T_grasp=tuple(tuple(float(value) for value in row) for row in object_T_grasp),
    )

    assert spheres.shape == (27, 4)
    # object_T_grasp is inverted, so the cuboid centre is translated by the
    # negative offset when expressed in the grasp frame.
    np.testing.assert_allclose(np.mean(spheres[:, :3], axis=0), (-0.01, 0.02, -0.03))
    assert np.all(spheres[:, 3] > 0.0)


def test_world_scope_does_not_activate_reserved_attachment_spheres() -> None:
    robot = {
        "kinematics": {
            "collision_spheres": {
                "torso_link": [{"center": [0.0, 0.0, 0.0], "radius": 0.1}],
                "left_attached_object": [{"center": [0.0, 0.0, 0.0], "radius": -100.0}],
            },
            "collision_sphere_buffer": {
                "torso_link": 0.0,
                "left_attached_object": 0.0,
            },
        }
    }

    deltas = _world_collision_buffer_deltas(robot, arm="left", mode="open_free")

    assert deltas["torso_link"] == pytest.approx(-0.100001)
    assert "left_attached_object" not in deltas


def test_precomputed_spheres_preserve_named_cuboid_clearance_policy() -> None:
    checker = SimpleNamespace(
        config=SimpleNamespace(
            kinematics_config=SimpleNamespace(
                link_sphere_idx_map=_ArrayValue([0, 1]),
                link_name_to_idx_map={"contact": 0, "wrist": 1},
            )
        )
    )
    scene = {
        "cuboid": {
            "cube": {
                "dims": [2.0, 2.0, 2.0],
                "pose": [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
            }
        }
    }
    spheres = np.asarray([[[0.0, 0.0, 0.0, 0.1], [2.0, 0.0, 0.0, 0.1]]])

    collisions = _world_cuboid_clearances(
        robot={},
        q_samples=np.zeros((1, 7)),
        scene=scene,
        device_cfg=None,
        disabled_links=set(),
        checker=checker,
        sphere_array=spheres,
    )
    assert collisions == [{("contact", "cube"): pytest.approx(-1.1)}]

    disabled = _world_cuboid_clearances(
        robot={},
        q_samples=np.zeros((1, 7)),
        scene=scene,
        device_cfg=None,
        disabled_links={"contact"},
        checker=checker,
        sphere_array=spheres,
    )
    assert disabled == [{}]


def test_named_cuboid_clearance_uses_five_mm_hard_margin() -> None:
    checker = SimpleNamespace(
        config=SimpleNamespace(
            kinematics_config=SimpleNamespace(
                link_sphere_idx_map=_ArrayValue([0]),
                link_name_to_idx_map={"wrist": 0},
            )
        )
    )
    scene = {
        "cuboid": {
            "cube": {
                "dims": [2.0, 2.0, 2.0],
                "pose": [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
            }
        }
    }
    q_samples = np.zeros((2, 7))
    # A 0.1 m radius sphere centered at x=1.104 m has 4 mm clearance;
    # x=1.106 m has 6 mm clearance. Only the former violates the 5 mm gate.
    spheres = np.asarray([[[1.104, 0.0, 0.0, 0.1]], [[1.106, 0.0, 0.0, 0.1]]])

    clearances = _world_cuboid_clearances(
        robot={},
        q_samples=q_samples,
        scene=scene,
        device_cfg=None,
        disabled_links=set(),
        checker=checker,
        sphere_array=spheres,
    )

    assert clearances[0] == {("wrist", "cube"): pytest.approx(0.004)}
    assert clearances[1] == {}
