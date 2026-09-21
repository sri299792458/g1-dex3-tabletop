import numpy as np
import pytest

from g1_aprilcube_calibration.joint_map import (
    G1_29_JOINT_NAMES,
    LEFT_ARM_INDICES,
    RIGHT_ARM_INDICES,
    arm_hand_link,
    arm_indices,
    arm_joint_names,
    dual_arm_vector,
    extract_left_arm,
    extract_right_arm,
    opposite_arm,
    reorder_named_joint_state,
    validate_full_joint_vector,
)


def test_mode5_arm_indices_and_names_are_authoritative() -> None:
    assert LEFT_ARM_INDICES == tuple(range(15, 22))
    assert RIGHT_ARM_INDICES == tuple(range(22, 29))
    assert G1_29_JOINT_NAMES[22:] == (
        "right_shoulder_pitch_joint",
        "right_shoulder_roll_joint",
        "right_shoulder_yaw_joint",
        "right_elbow_joint",
        "right_wrist_roll_joint",
        "right_wrist_pitch_joint",
        "right_wrist_yaw_joint",
    )


def test_extract_and_combine_arm_vectors() -> None:
    full = np.arange(29, dtype=np.float64)
    assert np.array_equal(extract_left_arm(full), np.arange(15, 22))
    assert np.array_equal(extract_right_arm(full), np.arange(22, 29))
    assert np.array_equal(dual_arm_vector(full[15:22], full[22:]), full[15:])


@pytest.mark.parametrize(
    ("side", "indices", "hand_link"),
    [
        ("left", tuple(range(15, 22)), "left_rubber_hand"),
        ("right", tuple(range(22, 29)), "right_rubber_hand"),
    ],
)
def test_calibration_side_is_a_runtime_argument(side, indices, hand_link) -> None:
    assert arm_indices(side) == indices
    assert arm_joint_names(side) == tuple(G1_29_JOINT_NAMES[index] for index in indices)
    assert arm_hand_link(side) == hand_link
    assert opposite_arm(opposite_arm(side)) == side


def test_validated_vectors_are_copied_and_read_only() -> None:
    original = np.arange(29, dtype=np.float64)
    validated = validate_full_joint_vector(original)
    original[0] = 100.0
    assert validated[0] == 0.0
    with pytest.raises(ValueError):
        validated[0] = 1.0


@pytest.mark.parametrize("bad", [np.zeros(28), np.zeros(30), [0.0] * 28 + [np.nan]])
def test_invalid_full_vectors_are_rejected(bad: object) -> None:
    with pytest.raises(ValueError):
        validate_full_joint_vector(bad)  # type: ignore[arg-type]


def test_named_joint_state_is_reordered_and_allows_extra_names() -> None:
    names = ["gripper"] + list(reversed(G1_29_JOINT_NAMES))
    positions = [999.0] + [float(G1_29_JOINT_NAMES.index(name)) for name in names[1:]]
    velocities = [-999.0] + [-value for value in positions[1:]]
    estimated_torques = [999.0] + [value / 10 for value in positions[1:]]

    ordered_q, ordered_dq, ordered_tau_est = reorder_named_joint_state(
        names, positions, velocities, estimated_torques
    )

    assert np.array_equal(ordered_q, np.arange(29))
    assert np.array_equal(ordered_dq, -np.arange(29))
    assert np.array_equal(ordered_tau_est, np.arange(29) / 10)


def test_named_joint_state_rejects_missing_and_duplicate_names() -> None:
    names = list(G1_29_JOINT_NAMES)
    values = list(np.arange(29, dtype=float))
    with pytest.raises(ValueError, match="missing"):
        reorder_named_joint_state(names[:-1], values[:-1], values[:-1], values[:-1])
    with pytest.raises(ValueError, match="duplicate"):
        reorder_named_joint_state(
            [*names, names[-1]],
            [*values, values[-1]],
            [*values, values[-1]],
            [*values, values[-1]],
        )
