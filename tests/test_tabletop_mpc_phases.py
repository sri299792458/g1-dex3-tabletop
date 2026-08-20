from types import SimpleNamespace

import numpy as np
import pytest

from g1_aprilcube_calibration.transforms import invert_transform
from g1_dex3_tabletop.mpc_command_buffer import MPCCommandWindow
from g1_dex3_tabletop.planning.curobo_backend import _clearance_from_activation_cost
from g1_dex3_tabletop.planning.tabletop_mpc import (
    MPC_ATTACHED_PHASES,
    MPC_PHASE_ORDER,
    MPCBenchmarkConfig,
    TabletopPhaseMPC,
    _bounded_route_goal,
    _fixture_excluded_links,
    _nominal_base_T_live_base,
    _payload_link_spheres,
    _reserved_velocity_constraint_is_safe,
    _simulate_phase,
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


def test_lifecycle_benchmark_preserves_configured_tracking_offset() -> None:
    offset = np.asarray([0.005, -0.002, 0.001, -0.008, 0.004, 0.0, 0.001])
    terminal = np.full(7, 0.01)

    class Controller:
        names = tuple(f"joint_{index}" for index in range(7))
        request = SimpleNamespace(joint_position_offsets_rad={})
        spec = SimpleNamespace(
            phase="move_to_pregrasp",
            mode="open_free",
            reference_fixed_goal=False,
        )
        path_model_q = np.stack((np.zeros(7), terminal))

        def next_nominal_window(
            self,
            *,
            handoff_predicted_q_rad,
            handoff_predicted_dq_rad_s,
            handoff_predicted_ddq_rad_s2,
            handoff_command_q_rad,
            source_state_monotonic_s,
            valid_from_monotonic_s,
            predecessor_sha256,
            committed_route_progress_index,
            reference_T_camera,
        ):
            del (
                handoff_predicted_dq_rad_s,
                handoff_predicted_ddq_rad_s2,
                reference_T_camera,
            )
            np.testing.assert_allclose(
                np.asarray(handoff_command_q_rad) - np.asarray(handoff_predicted_q_rad),
                offset,
            )
            return MPCCommandWindow(
                generation=0,
                plan_sha256="a" * 64,
                source_state_monotonic_s=source_state_monotonic_s,
                valid_from_monotonic_s=valid_from_monotonic_s,
                sample_time_s=(0.0, 0.2),
                command_q_rad=(
                    tuple(handoff_command_q_rad),
                    tuple(terminal + offset),
                ),
                predicted_q_rad=(tuple(handoff_predicted_q_rad), tuple(terminal)),
                predicted_dq_rad_s=(tuple(np.full(7, 0.05)), (0.0,) * 7),
                predicted_ddq_rad_s2=((0.0,) * 7,) * 2,
                predecessor_sha256=predecessor_sha256,
                feasible=True,
                terminal=True,
                solve_time_s=0.01,
                diagnostics={"proposed_route_progress_index": committed_route_progress_index + 1},
            )

    phase, measured, velocity = _simulate_phase(
        Controller(),
        command_q_rad=np.zeros(7),
        model_dq_rad_s=np.zeros(7),
        config=MPCBenchmarkConfig(
            maximum_steps=2,
            simulated_tracking_offset_rad=tuple(offset),
        ),
    )

    assert phase["reached_terminal"]
    assert phase["accepted_windows"] == 1
    np.testing.assert_allclose(measured, terminal)
    np.testing.assert_allclose(velocity, np.zeros(7))


def test_lifecycle_benchmark_retries_without_committing_a_rejected_window() -> None:
    terminal = np.full(7, 0.03)

    class Controller:
        names = tuple(f"joint_{index}" for index in range(7))
        request = SimpleNamespace(joint_position_offsets_rad={})
        spec = SimpleNamespace(
            phase="move_to_pregrasp",
            mode="open_free",
            reference_fixed_goal=False,
        )
        path_model_q = np.stack((np.zeros(7), terminal))

        def __init__(self) -> None:
            self.calls = 0
            self.committed_progress: list[int] = []

        def next_nominal_window(
            self,
            *,
            handoff_predicted_q_rad,
            handoff_predicted_dq_rad_s,
            handoff_predicted_ddq_rad_s2,
            handoff_command_q_rad,
            source_state_monotonic_s,
            valid_from_monotonic_s,
            predecessor_sha256,
            committed_route_progress_index,
            reference_T_camera,
        ):
            del handoff_predicted_ddq_rad_s2, reference_T_camera
            generation = self.calls
            self.calls += 1
            self.committed_progress.append(committed_route_progress_index)
            feasible = generation != 1
            is_terminal = generation == 2
            end = terminal if is_terminal else np.full(7, 0.02 + 0.005 * generation)
            return MPCCommandWindow(
                generation=generation,
                plan_sha256="a" * 64,
                source_state_monotonic_s=source_state_monotonic_s,
                valid_from_monotonic_s=valid_from_monotonic_s,
                sample_time_s=(0.0, 0.8),
                command_q_rad=(tuple(handoff_command_q_rad), tuple(end)),
                predicted_q_rad=(tuple(handoff_predicted_q_rad), tuple(end)),
                predicted_dq_rad_s=(tuple(handoff_predicted_dq_rad_s), (0.0,) * 7),
                predicted_ddq_rad_s2=((0.0,) * 7,) * 2,
                predecessor_sha256=predecessor_sha256,
                feasible=feasible,
                terminal=is_terminal,
                solve_time_s=0.01,
                diagnostics={"proposed_route_progress_index": committed_route_progress_index + 1},
            )

    controller = Controller()
    phase, measured, _velocity = _simulate_phase(
        controller,
        command_q_rad=np.zeros(7),
        model_dq_rad_s=np.zeros(7),
        config=MPCBenchmarkConfig(maximum_steps=4),
    )

    assert phase["reached_terminal"]
    assert phase["accepted_windows"] == 2
    assert phase["rejected_windows"] == 1
    assert controller.committed_progress == [0, 1, 1]
    np.testing.assert_allclose(measured, terminal)


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
    controller = object.__new__(TabletopPhaseMPC)
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
    specs = {phase: mpc_phase_spec(phase) for phase in MPC_PHASE_ORDER}

    assert "clearance" not in specs
    assert "__handoff__" not in specs
    assert specs["move_to_pregrasp"].mode == "open_free"
    assert specs["return_to_clearance"].mode == "open_free"
    assert specs["grasp_approach"].mode == "open_contact"
    assert specs["grasp_retreat"].mode == "open_contact"
    assert {phase for phase, spec in specs.items() if spec.attached_payload} == MPC_ATTACHED_PHASES
    assert all(
        spec.finger_state == "measured_contact" for spec in specs.values() if spec.attached_payload
    )
    assert all(not spec.include_fixture_in_optimizer for spec in specs.values())
    assert not specs["return_to_clearance"].reference_fixed_goal
    assert all(
        spec.reference_fixed_goal
        for phase, spec in specs.items()
        if phase != "return_to_clearance"
    )


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


def test_planning_session_forwards_hash_bound_camera_state_correction() -> None:
    received = {}
    result = object()

    class FakeMPC:
        spec = SimpleNamespace(phase="move_to_pregrasp")

        def next_nominal_window(self, **kwargs):
            received.update(kwargs)
            return result

    correction = {
        "reference_T_camera": np.eye(4).tolist(),
        "timestamp_ns": 12,
        "anchor_timestamp_ns": 10,
        "source_monotonic_s": 1.0,
    }
    session = TabletopPlanningSession()
    session._active_phase_mpc = FakeMPC()

    actual = session.step_mpc_phase(
        {
            "phase": "move_to_pregrasp",
            "handoff_predicted_q_rad": [0.0] * 7,
            "handoff_predicted_dq_rad_s": [0.0] * 7,
            "handoff_predicted_ddq_rad_s2": [0.0] * 7,
            "handoff_command_q_rad": [0.0] * 7,
            "source_state_monotonic_s": 1.0,
            "valid_from_monotonic_s": 1.2,
            "predecessor_sha256": None,
            "committed_route_progress_index": 4,
            "camera_state_correction": correction,
        }
    )

    assert actual is result
    np.testing.assert_allclose(received["reference_T_camera"], np.eye(4))
    assert received["camera_state_provenance"] == correction
    assert received["valid_from_monotonic_s"] == pytest.approx(1.2)
    assert received["committed_route_progress_index"] == 4


def test_supported_routes_remain_frozen_instead_of_entering_mpc() -> None:
    for phase in ("clearance", "__handoff__"):
        with pytest.raises(ValueError, match="unsupported tabletop MPC phase"):
            mpc_phase_spec(phase)


def test_fixture_exclusions_match_contact_and_attached_payload_policies() -> None:
    contact = _fixture_excluded_links(mpc_phase_spec("grasp_approach"), arm="left")
    attached = _fixture_excluded_links(mpc_phase_spec("retention_test_lift"), arm="left")

    assert contact == (
        "left_hand_thumb_2_link",
        "left_hand_middle_1_link",
        "left_hand_index_1_link",
    )
    assert attached == ("left_attached_object",)


def test_contact_routes_keep_cube_for_noncontact_links_without_attaching_it() -> None:
    for phase in ("grasp_approach", "grasp_retreat"):
        spec = mpc_phase_spec(phase)
        assert spec.allow_fingertip_cube_contact
        assert spec.include_cube_in_optimizer
        assert not spec.attached_payload
        assert spec.include_table_patch


def test_unknown_mpc_phase_is_rejected() -> None:
    with pytest.raises(ValueError, match="unsupported tabletop MPC phase"):
        mpc_phase_spec("invented_transition")


def test_planning_session_reuses_one_warmed_solver_across_every_mode(monkeypatch) -> None:
    built = []

    class FakePhaseMPC:
        def __init__(
            self,
            clearance_request,
            execution,
            *,
            phase,
            loaded_request,
            measured_active_dex3_q_rad,
        ) -> None:
            del clearance_request, execution, loaded_request
            self.spec = mpc_phase_spec(phase)
            self.measured = (
                None
                if measured_active_dex3_q_rad is None
                else np.asarray(measured_active_dex3_q_rad).copy()
            )
            self.closed = False
            built.append(self)

        def setup_at_frozen_route_start(self) -> float:
            return 1.0

        def can_select_phase(self, phase, *, measured_active_dex3_q_rad) -> bool:
            return not mpc_phase_spec(phase).attached_payload or (
                measured_active_dex3_q_rad is not None
            )

        def select_phase(self, phase, *, measured_active_dex3_q_rad) -> dict:
            self.spec = mpc_phase_spec(phase)
            self.measured = measured_active_dex3_q_rad
            return {
                "kinematics_cache_hit": True,
                "kinematics_resolve_time_s": 0.0,
                "optimizer_prewarm_time_s": 0.0,
                "reconfiguration_time_s": 0.01,
            }

        def close(self) -> None:
            self.closed = True

    monkeypatch.setattr(
        "g1_dex3_tabletop.planning.tabletop_session.TabletopPhaseMPC",
        FakePhaseMPC,
    )
    session = TabletopPlanningSession()
    session._loaded_request = object()
    session._clearance_request = object()
    session._execution = SimpleNamespace(content_sha256="plan")

    phases = MPC_PHASE_ORDER
    contact = np.zeros(7)
    for index, phase in enumerate(phases):
        result = session.prepare_mpc_phase(
            phase,
            measured_active_dex3_q_rad=(
                contact if mpc_phase_spec(phase).attached_payload else None
            ),
        )
        assert result["reused_warm_model"] == (index > 0)
        if index == 0:
            assert result["preparation_time_s"] == pytest.approx(
                result["build_time_s"] + result["setup_time_s"]
            )
        else:
            assert result["build_time_s"] == 0.0
            assert result["setup_time_s"] == 0.0
            assert result["preparation_time_s"] == pytest.approx(0.01)

    assert len(built) == 1
    assert session._phase_mpc is built[0]
    assert not built[0].closed

    session.close()
    assert all(controller.closed for controller in built)


def test_planning_session_updates_measured_contact_without_rebuilding_solver(monkeypatch) -> None:
    built = []

    class FakeAttachedMPC:
        def __init__(self, *args, phase, measured_active_dex3_q_rad, **kwargs) -> None:
            del args, kwargs
            self.spec = mpc_phase_spec(phase)
            self.measured = np.asarray(measured_active_dex3_q_rad).copy()
            self.closed = False
            built.append(self)

        def setup_at_frozen_route_start(self) -> float:
            return 1.0

        def can_select_phase(self, phase, *, measured_active_dex3_q_rad) -> bool:
            return measured_active_dex3_q_rad is not None

        def select_phase(self, phase, *, measured_active_dex3_q_rad) -> dict:
            self.spec = mpc_phase_spec(phase)
            cache_hit = np.array_equal(measured_active_dex3_q_rad, self.measured)
            self.measured = np.asarray(measured_active_dex3_q_rad).copy()
            return {
                "kinematics_cache_hit": cache_hit,
                "kinematics_resolve_time_s": 0.0 if cache_hit else 1.0,
                "optimizer_prewarm_time_s": 0.0,
                "reconfiguration_time_s": 0.01 if cache_hit else 1.01,
            }

        def close(self) -> None:
            self.closed = True

    monkeypatch.setattr(
        "g1_dex3_tabletop.planning.tabletop_session.TabletopPhaseMPC",
        FakeAttachedMPC,
    )
    session = TabletopPlanningSession()
    session._loaded_request = object()
    session._clearance_request = object()
    session._execution = SimpleNamespace(content_sha256="plan")

    first = np.zeros(7)
    second = np.ones(7)
    session.prepare_mpc_phase("retention_test_lift", measured_active_dex3_q_rad=first)
    reused = session.prepare_mpc_phase("payload_lift", measured_active_dex3_q_rad=first)
    replaced = session.prepare_mpc_phase("payload_lower", measured_active_dex3_q_rad=second)

    assert reused["reused_warm_model"]
    assert reused["kinematics_cache_hit"]
    assert replaced["reused_warm_model"]
    assert not replaced["kinematics_cache_hit"]
    assert len(built) == 1
    assert not built[0].closed


def test_clearance_replan_atomically_replaces_task_and_resets_old_mpc(monkeypatch) -> None:
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
        lambda received, progress: task,
    )
    monkeypatch.setattr(
        "g1_dex3_tabletop.planning.tabletop_session.combine_tabletop_plans",
        lambda **kwargs: replacement,
    )
    monkeypatch.setattr(
        "g1_dex3_tabletop.planning.tabletop_session.RetentionRouteValidator",
        Validator,
    )
    session = TabletopPlanningSession()
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
    assert session._phase_mpc is None
    assert session._active_phase_mpc is None
    assert "closed" in events


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

    def plan(received, *, open_planner_cache, progress):
        assert received is request
        assert open_planner_cache is session._open_planner
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
