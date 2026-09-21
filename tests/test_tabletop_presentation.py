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
from g1_dex3_tabletop.tabletop_object import load_tabletop_object_profile
from g1_dex3_tabletop.tabletop_presentation import load_tabletop_presentation


def _identity() -> tuple[tuple[float, ...], ...]:
    return tuple(tuple(float(value) for value in row) for row in np.eye(4))


def _request(
    presentation_id: str,
    object_profile_id: str = "cube40-r3",
) -> TabletopTaskRequest:
    profile = load_tabletop_object_profile(object_profile_id)
    presentation = load_tabletop_presentation(
        presentation_id,
        direct_object_profile_id=(
            profile.profile_id if presentation_id == "direct" else None
        ),
        direct_shortlist_override=(
            profile.direct_grasp_shortlist_path
            if presentation_id == "direct"
            else None
        ),
    )
    shortlist_path = presentation.grasp_shortlist_for(profile.profile_id)
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
        grasp_shortlist_path=str(shortlist_path.relative_to(root)),
        grasp_shortlist_sha256=hashlib.sha256(shortlist_path.read_bytes()).hexdigest(),
        object_dimensions_m=profile.dimensions_m,
        presentation_id=presentation.presentation_id,
        fixture=presentation.fixture,
    )


def test_direct_presentation_preserves_the_existing_scene_and_shortlist() -> None:
    profile = load_tabletop_object_profile("cube40-r3")
    presentation = load_tabletop_presentation(
        "direct",
        direct_object_profile_id=profile.profile_id,
        direct_shortlist_override=profile.direct_grasp_shortlist_path,
    )
    assert presentation.fixture is None
    assert presentation.applicable_hand_sides == ("left", "right")
    assert presentation.grasp_shortlist_for("cube40-r3").name == "shortlist.yaml"
    presentation.require_object_profile("cube40-r3")
    with pytest.raises(ValueError, match="not qualified for object profile"):
        presentation.require_object_profile("cube60-r3")

    request = _request("direct")
    shortlist, candidates = _load_shortlist(request)
    assert shortlist["shortlist_id"] == "cube_dex3_executable_v1"
    assert len(candidates) == 5
    assert shortlist["execution_contract"]["fixed_cube_during_qualification"] is True
    assert _base_scene(request, np.eye(4), include_cube=False) == {"cuboid": {}}


@pytest.mark.parametrize(
    ("object_profile_id", "expected_candidates"),
    (("cube40-r3", 113), ("cube60-r3", 312)),
)
def test_prime_tower_is_hash_bound_for_both_cube_profiles(
    object_profile_id: str,
    expected_candidates: int,
) -> None:
    presentation = load_tabletop_presentation("prime-tower")
    presentation.require_arm("left")
    presentation.require_arm("right")
    presentation.require_object_profile(object_profile_id)
    assert presentation.fixture is not None
    assert presentation.fixture.support_height_m == 0.060
    assert presentation.fixture.mesh_scale == (0.001, 0.001, 0.001)

    request = _request("prime-tower", object_profile_id)
    assert TabletopTaskRequest.from_dict(request.to_dict()) == request
    shortlist, candidates = _load_shortlist(request)
    assert shortlist["presentation"]["id"] == "prime-tower"
    assert len(candidates) == expected_candidates
    assert shortlist["execution_contract"]["approach_distance_m"] == 0.07
    assert all(0.07 in item["valid_approach_distances_m"] for item in candidates)
    assert all(
        item["execution_evidence"]["qualification_model"]
        == "stationary_cube_fixed_descriptor_close_on_prime_tower"
        for item in candidates
    )


@pytest.mark.parametrize(
    ("object_profile_id", "table_z"),
    (("cube40-r3", -0.080), ("cube60-r3", -0.090)),
)
def test_prime_tower_moves_the_table_plane_below_each_cube(
    object_profile_id: str,
    table_z: float,
) -> None:
    request = _request("prime-tower", object_profile_id)
    plane_point, object_pose, down = _table_from_resting_object(request, np.eye(4))
    np.testing.assert_allclose(object_pose, np.eye(4))
    np.testing.assert_allclose(down, [0.0, 0.0, -1.0])
    np.testing.assert_allclose(plane_point, [0.0, 0.0, table_z])

    scene = _base_scene(
        request,
        np.eye(4),
        include_cube=True,
        include_open_transit_table_patch=True,
    )
    assert set(scene["cuboid"]) == {"cube", "open_transit_table_patch"}
    assert set(scene["mesh"]) == {"cube_prime_tower_h60"}
    fixture = scene["mesh"]["cube_prime_tower_h60"]
    assert Path(fixture["file_path"]).is_file()
    assert fixture["scale"] == [0.001, 0.001, 0.001]
    np.testing.assert_allclose(fixture["pose"][:3], [0.0, 0.0, table_z])
    np.testing.assert_allclose(
        scene["cuboid"]["open_transit_table_patch"]["pose"][:3],
        [0.0, 0.0, table_z - 0.010],
    )


def test_attached_lift_exempts_only_the_cube_support_fixture() -> None:
    request = _request("prime-tower")

    scene = _attached_lift_scene(request, np.eye(4))

    assert scene == {"cuboid": {}}
    assert request.fixture is not None
    assert request.fixture.fixture_id not in scene.get("mesh", {})
