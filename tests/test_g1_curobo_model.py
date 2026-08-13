import json
from pathlib import Path

import numpy as np
import pytest

from g1_aprilcube_calibration.calibration_bundle import CalibrationBundle
from g1_dex3_tabletop.planning.contracts import RobotSnapshot
from g1_dex3_tabletop.planning.g1_model import (
    _mounted_plate_collision_spheres,
    build_robot_config_for_active_joints,
    build_tabletop_robot_config,
    command_from_model_q,
    corrected_joint_positions,
    model_source_hashes,
)

ROOT = Path(__file__).resolve().parents[1]


def test_both_cad_mounts_create_the_same_segmented_sphere_count() -> None:
    left = _mounted_plate_collision_spheres("left")
    right = _mounted_plate_collision_spheres("right")

    assert len(left) == len(right) == 30
    assert all(item["radius"] > 0 for item in left + right)
    assert model_source_hashes()["curobo_commit"] == ("8e734f3ced1df898990bcd92de40abce475907db")


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


def test_tabletop_model_reserves_attached_payload_spheres() -> None:
    pytest.importorskip("curobo")
    snapshot = RobotSnapshot((0.0,) * 29, (0.0,) * 7, (0.0,) * 7)

    robot, reference = build_tabletop_robot_config(
        snapshot=snapshot,
        joint_position_offsets_rad={},
        right_finger_q_rad=(0.0,) * 7,
    )

    assert len(reference) == 7
    assert robot["kinematics"]["extra_collision_spheres"]["right_attached_object"] == 32
