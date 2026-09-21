import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

from g1_aprilcube_calibration.calibration_bundle import CalibrationBundle
from g1_aprilcube_calibration.joint_map import arm_joint_names
from g1_dex3_tabletop.planning.contracts import RobotSnapshot
from g1_dex3_tabletop.planning.g1_model import (
    _mounted_plate_collision_spheres,
    build_locked_robot_config,
    build_robot_config_for_active_joints,
    build_tabletop_robot_config,
    command_from_model_q,
    corrected_joint_positions,
    grasp_T_palm,
    model_source_hashes,
    tabletop_motion_joint_names,
)

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("active_fingers", [False, True])
def test_calibration_hand_policy_keeps_external_pairs_and_geometry(monkeypatch, active_fingers):
    def load_yaml(path):
        return yaml.safe_load(Path(path).read_text())

    monkeypatch.setitem(sys.modules, "curobo.config_io", SimpleNamespace(load_yaml=load_yaml))
    snapshot = RobotSnapshot((0.0,) * 29, (0.0,) * 7, (0.0,) * 7)
    active = ("right_hand_thumb_1_joint",) if active_fingers else ("right_shoulder_roll_joint",)
    strict, _ = build_robot_config_for_active_joints(
        active_joint_names=active, snapshot=snapshot, joint_position_offsets_rad={}
    )
    configured, _ = build_robot_config_for_active_joints(
        active_joint_names=active,
        snapshot=snapshot,
        joint_position_offsets_rad={},
        ignore_internal_hand_collisions=True,
    )
    original = strict["kinematics"]
    actual = configured["kinematics"]
    # Only the same-hand pair policy changes. Every sphere, body pair, and
    # cross-hand pair retains exactly its original configuration.
    for key in set(original) - {"self_collision_ignore"}:
        assert actual[key] == original[key]
    links = actual["collision_link_names"]
    for first in links:
        for second in links:
            if first == second:
                continue
            before = second in original["self_collision_ignore"].get(first, [])
            after = second in actual["self_collision_ignore"].get(first, [])
            same_hand = any(
                first.startswith(f"{side}_hand_") and second.startswith(f"{side}_hand_")
                for side in ("left", "right")
            )
            assert after == (True if same_hand else before)
    if not active_fingers:
        arm_robot, _ = build_locked_robot_config(
            arm="right", snapshot=snapshot, joint_position_offsets_rad={}
        )
        arm_ignore = arm_robot["kinematics"]["self_collision_ignore"]
        additions = {
            (first, second)
            for first, others in arm_ignore.items()
            for second in others
            if second not in actual["self_collision_ignore"].get(first, [])
        }
        adjacent = {
            ("torso_link", "left_shoulder_roll_link"),
            ("left_shoulder_roll_link", "torso_link"),
            ("torso_link", "right_shoulder_roll_link"),
            ("right_shoulder_roll_link", "torso_link"),
        }
        assert adjacent <= additions
        assert ("pelvis", "left_hip_roll_link") in additions
        moving = {
            name.removesuffix("_joint") + "_link"
            for side in ("left", "right")
            for name in arm_joint_names(side)
        }
        moving.update(name for name in links if "_hand_" in name)
        assert all(
            first not in moving and second not in moving for first, second in additions - adjacent
        )
        assert "right_shoulder_yaw_link" not in arm_ignore["torso_link"]
        assert "left_shoulder_yaw_link" not in arm_ignore["torso_link"]


def test_both_cad_mounts_create_the_same_segmented_sphere_count() -> None:
    left = _mounted_plate_collision_spheres("left")
    right = _mounted_plate_collision_spheres("right")

    assert len(left) == len(right) == 30
    assert all(item["radius"] > 0 for item in left + right)
    assert model_source_hashes()["curobo_commit"] == ("8e734f3ced1df898990bcd92de40abce475907db")


def test_calibration_static_pair_exclusion_rejects_active_waist(monkeypatch):
    monkeypatch.setitem(
        sys.modules,
        "curobo.config_io",
        SimpleNamespace(load_yaml=lambda path: yaml.safe_load(Path(path).read_text())),
    )
    with pytest.raises(ValueError, match="locked"):
        build_robot_config_for_active_joints(
            active_joint_names=("waist_yaw_joint", "right_shoulder_roll_joint"),
            snapshot=RobotSnapshot((0.0,) * 29, (0.0,) * 7, (0.0,) * 7),
            joint_position_offsets_rad={},
            ignore_static_body_collisions=True,
        )


@pytest.mark.parametrize("gap, passed", [(0.009, False), (0.0101, True)])
def test_core_clearance_does_not_inherit_a_close_reference_exception(gap, passed):
    torch = pytest.importorskip("torch")
    from g1_dex3_tabletop.planning.curobo_backend import (
        _combine_self_clearance_certificates,
        _self_clearance_certificate,
    )

    class FixedGapChecker:
        def self_collision_link_pair_clearances(self, samples, *, joint_names):
            return torch.full((len(samples), 1), gap), (("torso", "hand"),)

    certificate = _self_clearance_certificate(
        checker=FixedGapChecker(),
        joint_names=("shoulder",),
        reference_q=np.zeros(1),
        segments=(("core_edge", np.zeros((2, 1))),),
        phase="closed_hand_core",
    )
    assert certificate["passed"] is passed
    assert certificate["hard_clearance_m"] == 0.010
    assert certificate["preexisting_clearance_maximum_degradation_m"] == 0.0
    combined = _combine_self_clearance_certificates([certificate])
    assert combined["passed"] is passed
    assert combined["policy"] == "strict_core_10mm"
    assert combined["preexisting_clearance_maximum_degradation_m"] == 0.0


def test_model_offsets_are_added_for_fk_and_removed_for_commands() -> None:
    bundle = CalibrationBundle.load(
        ROOT / "config/calibrations/dex3_shared_20260812_selected_free.json"
    )
    snapshot = RobotSnapshot((0.0,) * 29, (0.0,) * 7, (0.0,) * 7)
    corrected = corrected_joint_positions(snapshot, dict(bundle.joint_position_offsets_rad))
    names = (
        "right_shoulder_pitch_joint",
        "right_shoulder_roll_joint",
        "right_shoulder_yaw_joint",
        "right_elbow_joint",
        "right_wrist_roll_joint",
        "right_wrist_pitch_joint",
        "right_wrist_yaw_joint",
    )
    model_q = np.asarray([corrected[name] for name in names])

    np.testing.assert_allclose(
        command_from_model_q(
            model_q,
            arm="right",
            joint_position_offsets_rad=dict(bundle.joint_position_offsets_rad),
        ),
        np.zeros(7),
        atol=1e-15,
    )
    manifest = json.loads((ROOT / "cad/dex3_dorsal_aruco_mount/design_manifest.json").read_text())
    assert manifest["hand_side"] == "right"


def test_collision_only_model_preserves_nvidia_tool_frames() -> None:
    pytest.importorskip("curobo")
    snapshot = RobotSnapshot((0.0,) * 29, (0.0,) * 7, (0.0,) * 7)
    active = (
        "left_hand_thumb_0_joint",
        "left_hand_thumb_1_joint",
        "left_hand_thumb_2_joint",
        "left_hand_middle_0_joint",
        "left_hand_middle_1_joint",
        "left_hand_index_0_joint",
        "left_hand_index_1_joint",
        "right_hand_thumb_0_joint",
        "right_hand_thumb_1_joint",
        "right_hand_thumb_2_joint",
        "right_hand_middle_0_joint",
        "right_hand_middle_1_joint",
        "right_hand_index_0_joint",
        "right_hand_index_1_joint",
    )

    robot, reference = build_robot_config_for_active_joints(
        active_joint_names=active,
        snapshot=snapshot,
        joint_position_offsets_rad={},
    )

    assert len(reference) == 14
    assert robot["kinematics"]["tool_frames"] == [
        "right_hand_index_1_link",
        "left_hand_index_1_link",
        "right_ankle_roll_link",
        "left_ankle_roll_link",
    ]


@pytest.mark.parametrize("arm", ("left", "right"))
def test_tabletop_model_reserves_attached_payload_spheres(arm: str) -> None:
    pytest.importorskip("curobo")
    snapshot = RobotSnapshot((0.0,) * 29, (0.0,) * 7, (0.0,) * 7)

    robot, reference = build_tabletop_robot_config(
        arm=arm,
        snapshot=snapshot,
        joint_position_offsets_rad={},
        active_finger_q_rad=(0.0,) * 7,
    )

    assert len(reference) == 7
    assert robot["kinematics"]["extra_collision_spheres"][f"{arm}_attached_object"] == 32

    ignore = robot["kinematics"]["self_collision_ignore"]
    collision_links = robot["kinematics"]["collision_link_names"]
    for side in ("left", "right"):
        hand_links = sorted(name for name in collision_links if name.startswith(f"{side}_hand_"))
        for index, link in enumerate(hand_links):
            for other in hand_links[index + 1 :]:
                assert other in ignore[link]
                assert link in ignore[other]

    assert f"{arm}_hip_yaw_link" not in ignore[f"{arm}_hand_palm_link"]
    opposite = "right" if arm == "left" else "left"
    assert f"{opposite}_hand_palm_link" not in ignore[f"{arm}_hand_palm_link"]
    # Links outside the selected arm/hand subtree are locked to one measured
    # snapshot. Their mutual collision state cannot change during planning.
    assert f"{opposite}_hand_palm_link" in ignore["left_ankle_roll_link"]
    assert "left_ankle_roll_link" in ignore[f"{opposite}_hand_palm_link"]
    # Every selected-arm/body and selected-arm/opposite-arm pair stays active.
    assert "torso_link" not in ignore[f"{arm}_hand_palm_link"]
    assert f"{opposite}_hand_palm_link" not in ignore[f"{arm}_hand_palm_link"]
    for side in ("left", "right"):
        assert f"{side}_shoulder_roll_link" in ignore["torso_link"]
        assert "torso_link" in ignore[f"{side}_shoulder_roll_link"]
    assert f"{arm}_shoulder_yaw_link" not in ignore["torso_link"]
    assert f"{opposite}_shoulder_yaw_link" in ignore["torso_link"]
    assert "torso_link" in ignore[f"{opposite}_shoulder_yaw_link"]
    assert robot["kinematics"]["self_collision_buffer"][
        f"{arm}_shoulder_yaw_link"
    ] == pytest.approx(0.0)


@pytest.mark.parametrize("arm", ("left", "right"))
def test_offline_tabletop_model_can_expose_waist_yaw_with_one_arm(arm: str) -> None:
    pytest.importorskip("curobo")
    q29 = tuple(0.01 * index for index in range(29))
    snapshot = RobotSnapshot(q29, (0.0,) * 7, (0.0,) * 7)

    robot, reference = build_tabletop_robot_config(
        arm=arm,
        snapshot=snapshot,
        joint_position_offsets_rad={},
        active_finger_q_rad=(0.0,) * 7,
        include_waist_yaw=True,
    )

    names = tabletop_motion_joint_names(arm, include_waist_yaw=True)
    assert names == ("waist_yaw_joint", *tabletop_motion_joint_names(arm))
    assert len(reference) == 8
    assert reference[0] == pytest.approx(q29[12])
    assert "waist_yaw_joint" not in robot["kinematics"]["lock_joints"]
    opposite = "right" if arm == "left" else "left"
    assert f"{opposite}_shoulder_pitch_joint" in robot["kinematics"]["lock_joints"]
    # Static-pair pruning is intentionally disabled when waist yaw is active;
    # the opposite hand can then move relative to the locked legs.
    ignore = robot["kinematics"]["self_collision_ignore"]
    assert f"{opposite}_hand_palm_link" not in ignore.get("left_ankle_roll_link", [])


def test_graspgenx_g_to_palm_transform_is_side_specific() -> None:
    left = grasp_T_palm("left")
    right = grasp_T_palm("right")

    np.testing.assert_allclose(left[:3, 3], right[:3, 3], atol=1e-15)
    assert not np.allclose(left[:3, :3], right[:3, :3])
    assert np.linalg.det(left[:3, :3]) == pytest.approx(1.0)
    assert np.linalg.det(right[:3, :3]) == pytest.approx(1.0)
