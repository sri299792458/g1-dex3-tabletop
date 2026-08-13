from pathlib import Path

import numpy as np
import pytest
import yaml

pytest.importorskip("pinocchio")

from g1_aprilcube_calibration.gravity_compensation import (
    G1PinocchioGravityFeedforward,
)
from g1_aprilcube_calibration.transports.unitree_dex3 import (
    DEX3_MOTOR_JOINT_SUFFIXES,
)
from g1_dex3_tabletop.control.gravity import prepare_gravity_feedforward

ROOT = Path(__file__).resolve().parents[1]
URDF = ROOT / "third_party/unitree_ros/robots/g1_description/g1_29dof_rev_1_0.urdf"
DEX3_URDF = ROOT / "third_party/unitree_ros/robots/g1_description/g1_29dof_with_hand_rev_1_0.urdf"


def test_exact_g1_urdf_zero_pose_matches_pinned_rnea_result() -> None:
    gravity = G1PinocchioGravityFeedforward(URDF)
    gravity.seed_reference(np.zeros(29))

    torque = gravity.torque_for(np.zeros(14))

    np.testing.assert_allclose(
        torque,
        (
            -2.093391042880,
            0.194095098873,
            0.000131641485,
            -1.881420850182,
            -0.004373849931,
            -0.400972316718,
            0.000010638908,
            -2.093391042880,
            -0.194095098873,
            -0.000131641485,
            -1.881420850182,
            0.004373849931,
            -0.400972316718,
            -0.000010638908,
        ),
        atol=1e-10,
    )
    assert not torque.flags.writeable


def test_commanded_arms_replace_measured_reference_but_body_stays_fixed() -> None:
    gravity = G1PinocchioGravityFeedforward(URDF)
    reference = np.linspace(-0.2, 0.2, 29)
    command = np.linspace(-0.4, 0.5, 14)
    gravity.seed_reference(reference)

    first = gravity.torque_for(command)
    cached = gravity.torque_for(command)
    modified_body = reference.copy()
    modified_body[14] += 0.2
    gravity.seed_reference(modified_body)
    body_changed = gravity.torque_for(command)

    np.testing.assert_allclose(cached, first)
    assert not np.allclose(body_changed, first)


def test_gravity_feedforward_requires_reference_and_valid_command() -> None:
    gravity = G1PinocchioGravityFeedforward(URDF)

    with pytest.raises(RuntimeError, match="no measured reference"):
        gravity.torque_for(np.zeros(14))
    gravity.seed_reference(np.zeros(29))
    with pytest.raises(ValueError, match="14 joints"):
        gravity.torque_for(np.zeros(13))
    with pytest.raises(ValueError, match="NaN"):
        gravity.torque_for(np.full(14, np.nan))


def test_hardware_gravity_preflight_selects_both_seven_joint_arms() -> None:
    gravity = prepare_gravity_feedforward(
        ROOT / "config/hardware_dex3_aruco.yaml",
        np.zeros(29),
    )

    assert gravity.torque_for(np.zeros(14)).shape == (14,)
    assert gravity.urdf_sha256 == (
        "97da67732d067c3147fc5fb7b7bafc8982718f4e7f8c92ff82266a4d9c07200d"
    )


def test_full_dual_dex3_model_is_reduced_like_unitree_xr() -> None:
    gravity = G1PinocchioGravityFeedforward(DEX3_URDF)
    gravity.seed_reference(np.zeros(29))

    torque = gravity.torque_for(np.zeros(14))

    assert gravity.full_model_nq == 43
    assert gravity.reduced_model_nq == 29
    assert len(gravity.locked_joint_positions_rad) == 14
    assert all(value == 0.0 for value in gravity.locked_joint_positions_rad.values())
    np.testing.assert_allclose(
        torque,
        (
            -3.643280927745,
            0.201704393763,
            0.000227031299,
            -3.412327516517,
            -0.038808629922,
            -1.219054392731,
            0.000043575128,
            -3.643280927745,
            -0.201704393763,
            -0.000227031299,
            -3.412327516517,
            0.038808629922,
            -1.219054392731,
            -0.000043575128,
        ),
        atol=1e-10,
    )


def test_full_dex3_model_requires_complete_explicit_finger_posture() -> None:
    with pytest.raises(ValueError, match="specify every non-G1 joint"):
        G1PinocchioGravityFeedforward(
            DEX3_URDF,
            locked_joint_positions_rad={"left_hand_index_0_joint": 0.1},
        )


def test_dex3_hardware_profile_names_every_locked_finger_joint() -> None:
    hardware = yaml.safe_load((ROOT / "config/hardware_dex3_aruco.yaml").read_text())
    locked = hardware["control"]["gravity_locked_joint_positions_rad"]

    gravity = G1PinocchioGravityFeedforward(
        DEX3_URDF,
        locked_joint_positions_rad=locked,
    )

    assert gravity.reduced_model_nq == 29
    assert gravity.locked_joint_positions_rad == locked
    assert locked == {
        "left_hand_thumb_0_joint": 0.0,
        "left_hand_thumb_1_joint": 0.7,
        "left_hand_thumb_2_joint": 0.7,
        "left_hand_middle_0_joint": -1.0,
        "left_hand_middle_1_joint": -1.5,
        "left_hand_index_0_joint": -1.0,
        "left_hand_index_1_joint": -1.5,
        "right_hand_thumb_0_joint": 0.0,
        "right_hand_thumb_1_joint": -0.7,
        "right_hand_thumb_2_joint": -0.7,
        "right_hand_middle_0_joint": 1.0,
        "right_hand_middle_1_joint": 1.5,
        "right_hand_index_0_joint": 1.0,
        "right_hand_index_1_joint": 1.5,
    }
    dex3 = hardware["dex3_control"]
    assert dex3["posture_name"] == hardware["robot"]["calibration_finger_posture_model"]
    for side in ("left", "right"):
        target = dex3[f"{side}_target_q_rad"]
        assert target == [
            locked[f"{side}_hand_{suffix}_joint"] for suffix in DEX3_MOTOR_JOINT_SUFFIXES[side]
        ]
