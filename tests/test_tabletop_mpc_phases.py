from types import SimpleNamespace

import numpy as np
import pytest

from g1_dex3_tabletop.planning.tabletop_mpc import (
    MPC_ATTACHED_PHASES,
    MPC_PHASE_ORDER,
    _payload_link_spheres,
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


def test_every_normal_tabletop_motion_has_one_physical_mpc_state() -> None:
    specs = {phase: mpc_phase_spec(phase) for phase in MPC_PHASE_ORDER}

    assert specs["clearance"].mode == "supported"
    assert specs["__handoff__"].mode == "supported"
    assert specs["move_to_pregrasp"].mode == "open_free"
    assert specs["return_to_clearance"].mode == "open_free"
    assert specs["grasp_approach"].mode == "open_contact"
    assert specs["grasp_retreat"].mode == "open_contact"
    assert {phase for phase, spec in specs.items() if spec.attached_payload} == MPC_ATTACHED_PHASES
    assert all(
        spec.finger_state == "measured_contact" for spec in specs.values() if spec.attached_payload
    )


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

    phases = (
        "clearance",
        "move_to_pregrasp",
        "grasp_approach",
        "retention_test_lift",
        "payload_lift",
        "payload_lower",
        "payload_replace",
        "grasp_retreat",
        "return_to_clearance",
        "__handoff__",
    )
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
