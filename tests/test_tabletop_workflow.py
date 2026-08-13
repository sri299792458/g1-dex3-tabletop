from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np

from g1_aprilcube_calibration.calibration_bundle import CalibrationBundle
from g1_dex3_tabletop.planning.contracts import PlannedTrajectory, RobotSnapshot
from g1_dex3_tabletop.planning.tabletop_planner import (
    LOCAL_TABLE_PLANE_LINKS,
    _base_scene,
    _local_plane_clearance,
    _table_from_resting_object,
)
from g1_dex3_tabletop.tabletop_contracts import SupportedEscapePlan, TabletopObservation
from g1_dex3_tabletop.tabletop_workflow import (
    build_tabletop_request,
    request_at_clearance,
)


def _identity():
    return np.eye(4)


def _observation() -> TabletopObservation:
    return TabletopObservation(
        RobotSnapshot((0.0,) * 29, (0.0,) * 7, (0.0,) * 7),
        tuple(tuple(row) for row in _identity()),
        "a" * 64,
        ("b" * 64, "c" * 64, "d" * 64),
        0.1,
        0.1,
    )


def test_task_config_does_not_invent_unobserved_table_footprint() -> None:
    bundle_path = (
        Path(__file__).resolve().parents[1]
        / "config/calibrations/dex3_shared_20260812_selected_free.json"
    )
    shortlist = Path(__file__).resolve().parents[1] / (
        "config/tabletop/cube_right_executable_v1/shortlist.yaml"
    )
    task = Path(__file__).resolve().parents[1] / "config/tabletop/task.yaml"
    bundle = CalibrationBundle.load(bundle_path)
    request = build_tabletop_request(
        observation=_observation(),
        calibration_bundle=bundle,
        calibration_bundle_path=bundle_path,
        grasp_shortlist_path=shortlist,
        task_config_path=task,
    )
    assert "physical_table_dimensions_m" not in request.to_dict()
    assert "physical_table_thickness_m" not in request.to_dict()


def test_cube_observation_defines_plane_but_no_table_box() -> None:
    from g1_dex3_tabletop.tabletop_contracts import TabletopTaskRequest

    loaded = TabletopTaskRequest(
        _observation(),
        tuple(tuple(row) for row in _identity()),
        {},
        "e" * 64,
        "config/tabletop/x.yaml",
        "f" * 64,
    )
    point, object_pose, down = _table_from_resting_object(loaded, _identity())
    np.testing.assert_allclose(object_pose, np.eye(4))
    np.testing.assert_allclose(point, [0.0, 0.0, -0.0225])
    np.testing.assert_allclose(down, [0.0, 0.0, -1.0])
    assert _base_scene(loaded, _identity(), include_cube=False) == {"cuboid": {}}
    assert set(_base_scene(loaded, _identity(), include_cube=True)["cuboid"]) == {"cube"}


def test_local_plane_guard_excludes_unlocated_table_geometry() -> None:
    assert all(name.startswith("right_") for name in LOCAL_TABLE_PLANE_LINKS)
    assert not any("elbow" in name for name in LOCAL_TABLE_PLANE_LINKS)
    assert not any("torso" in name or "hip" in name for name in LOCAL_TABLE_PLANE_LINKS)


def test_local_plane_clearance_uses_sphere_surface_not_center() -> None:
    torch = __import__("pytest").importorskip("torch")
    __import__("pytest").importorskip("curobo")
    from curobo.types import DeviceCfg

    class KinematicsConfig:
        @staticmethod
        def get_sphere_index_from_link_name(_name):
            return torch.tensor([0])

    class Planner:
        device_cfg = DeviceCfg(device=torch.device("cpu"), dtype=torch.float32)
        kinematics = SimpleNamespace(config=SimpleNamespace(kinematics_config=KinematicsConfig()))

        @staticmethod
        def compute_kinematics(state):
            count = state.position.shape[0]
            spheres = torch.zeros((count, 1, 4), dtype=torch.float32)
            spheres[:, 0, 2] = torch.tensor([0.10, 0.02])[:count]
            spheres[:, 0, 3] = 0.01
            return SimpleNamespace(robot_spheres=spheres)

    clearance, link, sample = _local_plane_clearance(
        Planner(),
        np.zeros((2, 7)),
        plane_point=np.zeros(3),
        down=np.array([0.0, 0.0, -1.0]),
        include_payload=False,
    )
    assert np.isclose(clearance, 0.01)
    assert link in LOCAL_TABLE_PLANE_LINKS
    assert sample == 1


def test_clearance_request_uses_exact_escape_endpoint() -> None:
    source = _observation()
    from g1_dex3_tabletop.tabletop_contracts import TabletopTaskRequest

    loaded = TabletopTaskRequest(
        source,
        tuple(tuple(row) for row in _identity()),
        {},
        "e" * 64,
        "config/tabletop/x.yaml",
        "f" * 64,
    )
    outbound = PlannedTrajectory(
        "__handoff__",
        "clearance",
        (0.0, 1.0),
        ((0.0,) * 7, (0.2,) * 7),
        ((0.0,) * 7, (0.2,) * 7),
        1.0,
    )
    inbound = PlannedTrajectory(
        "clearance",
        "__handoff__",
        (0.0, 1.0),
        ((0.2,) * 7, (0.0,) * 7),
        ((0.2,) * 7, (0.0,) * 7),
        0.0,
    )
    escape = SupportedEscapePlan(loaded.content_sha256, outbound, inbound, 0.1, {})
    result = request_at_clearance(loaded, escape)
    assert result.observation.snapshot.measured_q29_rad[22:29] == (0.2,) * 7
