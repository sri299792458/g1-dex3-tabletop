from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import trimesh

from g1_aprilcube_calibration.joint_map import arm_joint_names
from g1_aprilcube_calibration.urdf_model import URDFModel
from g1_dex3_tabletop.calibration.visibility import BilateralSightlineChecker, _MeshSegments


@pytest.mark.parametrize("z, blocked", [(0.5, True), (1.5, False), (-0.5, False)])
def test_sightlines_are_finite_camera_to_marker_segments(z, blocked):
    mesh = trimesh.creation.box(extents=(0.1, 0.1, 0.1))
    mesh.apply_translation((0, 0, z))
    assert (
        _MeshSegments(mesh.triangles).blocks(
            np.zeros(3), np.array([[0, 0, 1.0], [0.01, 0.01, 1.0]]), epsilon_m=0.0001
        )
        is blocked
    )


def visibility_fixture(tmp_path: Path, *, blocker_side: str, finger: bool = False):
    parts = ['<robot name="visibility"><link name="torso_link"/>']
    for side in ("left", "right"):
        parent = "torso_link"
        for name in arm_joint_names(side):
            child = name.removesuffix("_joint") + "_link"
            parts.append(
                f'<link name="{child}"/><joint name="{name}" type="revolute">'
                f'<parent link="{parent}"/><child link="{child}"/>'
                '<axis xyz="0 0 1"/><limit lower="-3" upper="3" effort="1" velocity="1"/></joint>'
            )
            parent = child
        blocker = f"{side}_hand_thumb_0_link" if finger else f"{side}_hand_palm_link"
        visual = (
            '<visual><origin xyz="0 0 0.5"/><geometry><box size="0.08 0.08 0.08"/></geometry></visual>'
            if side == blocker_side
            else ""
        )
        parts.append(f'<link name="{blocker}">{visual}</link>')
        if finger:
            parts.append(
                f'<joint name="{side}_hand_thumb_0_joint" type="revolute">'
                f'<parent link="{parent}"/><child link="{blocker}"/>'
                '<axis xyz="0 1 0"/><limit lower="-3" upper="3" effort="1" velocity="1"/></joint>'
            )
        else:
            parts.append(
                f'<joint name="{side}_palm_fixed" type="fixed"><parent link="{parent}"/><child link="{blocker}"/></joint>'
            )
        x = 0 if side == "left" else 0.6
        parts.append(
            f'<link name="{side}_rubber_hand"/><joint name="{side}_target_fixed" type="fixed">'
            f'<parent link="{parent}"/><child link="{side}_rubber_hand"/><origin xyz="{x} 0 1"/></joint>'
        )
    parts.append("</robot>")
    path = tmp_path / "robot.urdf"
    path.write_text("".join(parts))
    model = URDFModel(path)
    corners = np.array([[-0.02, -0.02, 0], [0.02, -0.02, 0], [0.02, 0.02, 0], [-0.02, 0.02, 0]])
    request = SimpleNamespace(
        nominal_torso_T_camera=np.eye(4),
        nominal_hand_T_targets={side: np.eye(4) for side in ("left", "right")},
        target_object_points_m_by_arm={side: corners for side in ("left", "right")},
        joint_position_offsets_rad={},
        dex3_model_positions_rad={side: [0.0] * 7 for side in ("left", "right")},
    )
    return BilateralSightlineChecker(
        request,
        model,
        geometry_model=model,
        physical_hand_T_targets=request.nominal_hand_T_targets,
    )


@pytest.mark.parametrize("blocker_side", ["left", "right"])
def test_required_marker_detects_own_or_opposite_hand_blocking(tmp_path, blocker_side):
    checker = visibility_fixture(tmp_path, blocker_side=blocker_side)
    with pytest.raises(ValueError, match=f"left marker.*{blocker_side}_hand_palm_link"):
        checker.check(np.zeros(29), required_sides=("left",))
    checker.check(np.zeros(29), required_sides=("right",))
    with pytest.raises(ValueError, match="left marker"):
        checker.check(np.zeros(29), required_sides=("left", "right"))


def test_sightlines_use_the_modeled_finger_posture(tmp_path):
    checker = visibility_fixture(tmp_path, blocker_side="left", finger=True)
    with pytest.raises(ValueError, match="left_hand_thumb_0_link"):
        checker.check(np.zeros(29), required_sides=("left",))
    checker.request.dex3_model_positions_rad["left"][0] = np.pi / 2
    checker.check(np.zeros(29), required_sides=("left",))


def test_unmodeled_marker_surface_can_block_other_marker(tmp_path):
    checker = visibility_fixture(tmp_path, blocker_side="left")
    # Move the synthetic hand occluder away; put the right marker in front of the left.
    link, transform, mesh = checker.meshes[0]
    transform = transform.copy()
    transform[0, 3] = 2.0
    checker.meshes = [(link, transform, mesh)]
    checker.request.nominal_hand_T_targets["right"][:3, 3] = (-0.6, 0, -0.5)
    with pytest.raises(ValueError, match="blocked by right marker"):
        checker.check(np.zeros(29), required_sides=("left",))
