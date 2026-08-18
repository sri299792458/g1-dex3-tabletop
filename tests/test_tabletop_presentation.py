from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pytest

from g1_dex3_tabletop.planning.contracts import RobotSnapshot
from g1_dex3_tabletop.planning.tabletop_planner import (
    _attached_lift_scene,
    _base_scene,
    _load_shortlist,
    _table_from_resting_object,
)
from g1_dex3_tabletop.tabletop_contracts import TabletopObservation, TabletopTaskRequest
from g1_dex3_tabletop.tabletop_presentation import load_tabletop_presentation


def _identity() -> tuple[tuple[float, ...], ...]:
    return tuple(tuple(float(value) for value in row) for row in np.eye(4))


def _request(presentation_id: str) -> TabletopTaskRequest:
    presentation = load_tabletop_presentation(presentation_id)
    observation = TabletopObservation(
        snapshot=RobotSnapshot((0.0,) * 29, (0.0,) * 7, (0.0,) * 7),
        camera_T_object=_identity(),
        camera_profile_sha256="a" * 64,
        source_frame_sha256=("b" * 64, "c" * 64, "d" * 64),
        object_translation_spread_mm=0.1,
        object_rotation_spread_deg=0.1,
    )
    root = Path(__file__).resolve().parents[1]
    return TabletopTaskRequest(
        observation=observation,
        arm="right",
        torso_T_camera=_identity(),
        joint_position_offsets_rad={},
        calibration_bundle_sha256="e" * 64,
        grasp_shortlist_path=str(presentation.grasp_shortlist_path.relative_to(root)),
        grasp_shortlist_sha256=hashlib.sha256(
            presentation.grasp_shortlist_path.read_bytes()
        ).hexdigest(),
        presentation_id=presentation.presentation_id,
        fixture=presentation.fixture,
    )


def test_direct_presentation_preserves_the_existing_scene_and_shortlist() -> None:
    presentation = load_tabletop_presentation("direct")
    assert presentation.fixture is None
    assert presentation.applicable_hand_sides == ("left", "right")
    assert presentation.grasp_shortlist_path.name == "shortlist.yaml"
    presentation.require_object_profile("cube40-r3")
    presentation.require_object_profile("cube60-r3")

    request = _request("direct")
    shortlist, candidates = _load_shortlist(request)
    assert shortlist["shortlist_id"] == "cube_dex3_executable_v1"
    assert len(candidates) == 5
    assert shortlist["execution_contract"]["fixed_cube_during_qualification"] is True
    assert _base_scene(request, np.eye(4), include_cube=False) == {"cuboid": {}}


def test_tripod_h50_is_an_opt_in_hash_bound_fixture_presentation() -> None:
    presentation = load_tabletop_presentation("tripod-h50")
    presentation.require_arm("left")
    presentation.require_arm("right")
    presentation.require_object_profile("cube40-r3")
    with pytest.raises(ValueError, match="not qualified for object profile"):
        presentation.require_object_profile("cube60-r3")
    assert presentation.fixture is not None
    assert presentation.fixture.support_height_m == 0.050
    assert presentation.fixture.mesh_scale == (0.001, 0.001, 0.001)

    request = _request("tripod-h50")
    assert TabletopTaskRequest.from_dict(request.to_dict()) == request
    shortlist, candidates = _load_shortlist(request)
    assert shortlist["presentation"]["id"] == "tripod-h50"
    assert len(candidates) == 372
    assert shortlist["execution_contract"]["approach_distance_m"] == 0.07
    assert all(0.07 in item["valid_approach_distances_m"] for item in candidates)
    assert all("isaac_closed_q" in item["execution_evidence"] for item in candidates)


def test_tripod_h50_moves_the_table_plane_below_the_presented_cube() -> None:
    request = _request("tripod-h50")
    plane_point, object_pose, down = _table_from_resting_object(request, np.eye(4))
    np.testing.assert_allclose(object_pose, np.eye(4))
    np.testing.assert_allclose(down, [0.0, 0.0, -1.0])
    np.testing.assert_allclose(plane_point, [0.0, 0.0, -0.070])

    scene = _base_scene(
        request,
        np.eye(4),
        include_cube=True,
        include_open_transit_table_patch=True,
    )
    assert set(scene["cuboid"]) == {"cube", "open_transit_table_patch"}
    assert set(scene["mesh"]) == {"cube_tripod_presenter_h50"}
    fixture = scene["mesh"]["cube_tripod_presenter_h50"]
    assert Path(fixture["file_path"]).is_file()
    assert fixture["scale"] == [0.001, 0.001, 0.001]
    np.testing.assert_allclose(fixture["pose"][:3], [0.0, 0.0, -0.070])
    np.testing.assert_allclose(
        scene["cuboid"]["open_transit_table_patch"]["pose"][:3],
        [0.0, 0.0, -0.080],
    )


def test_attached_lift_exempts_only_the_cube_support_fixture() -> None:
    request = _request("tripod-h50")

    scene = _attached_lift_scene(request, np.eye(4))

    assert scene == {"cuboid": {}}
    assert request.fixture is not None
    assert request.fixture.fixture_id not in scene.get("mesh", {})
