"""Authoritative Unitree G1 mode-5 joint ordering and extraction helpers."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

import numpy as np

G1_29_JOINT_NAMES: tuple[str, ...] = (
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
)

LEFT_ARM_INDICES: tuple[int, ...] = tuple(range(15, 22))
RIGHT_ARM_INDICES: tuple[int, ...] = tuple(range(22, 29))
LEFT_ARM_JOINT_NAMES: tuple[str, ...] = tuple(
    G1_29_JOINT_NAMES[index] for index in LEFT_ARM_INDICES
)
RIGHT_ARM_JOINT_NAMES: tuple[str, ...] = tuple(
    G1_29_JOINT_NAMES[index] for index in RIGHT_ARM_INDICES
)

ArmSide = Literal["left", "right"]
ARM_INDICES: dict[ArmSide, tuple[int, ...]] = {
    "left": LEFT_ARM_INDICES,
    "right": RIGHT_ARM_INDICES,
}
ARM_JOINT_NAMES: dict[ArmSide, tuple[str, ...]] = {
    "left": LEFT_ARM_JOINT_NAMES,
    "right": RIGHT_ARM_JOINT_NAMES,
}
ARM_HAND_LINKS: dict[ArmSide, str] = {
    "left": "left_rubber_hand",
    "right": "right_rubber_hand",
}

G1_MODE_MACHINE = 5
G1_DOF = len(G1_29_JOINT_NAMES)
DUAL_ARM_DOF = len(LEFT_ARM_INDICES) + len(RIGHT_ARM_INDICES)

_NAME_TO_INDEX = {name: index for index, name in enumerate(G1_29_JOINT_NAMES)}


def validate_full_joint_vector(
    values: Sequence[float] | np.ndarray,
    *,
    name: str = "joint vector",
) -> np.ndarray:
    """Return a finite read-only 29-vector, rejecting partial state."""
    array = np.asarray(values, dtype=np.float64).reshape(-1).copy()
    if array.shape != (G1_DOF,):
        raise ValueError(f"{name} must contain exactly {G1_DOF} values, got {len(array)}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} contains NaN or infinity")
    array.setflags(write=False)
    return array


def validate_arm_vector(
    values: Sequence[float] | np.ndarray,
    *,
    side: str,
) -> np.ndarray:
    """Return a finite read-only seven-joint arm vector."""
    if side not in {"left", "right"}:
        raise ValueError(f"side must be 'left' or 'right', got {side!r}")
    array = np.asarray(values, dtype=np.float64).reshape(-1).copy()
    if array.shape != (7,):
        raise ValueError(f"{side} arm vector must contain exactly 7 values, got {len(array)}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{side} arm vector contains NaN or infinity")
    array.setflags(write=False)
    return array


def validate_arm_side(side: str) -> ArmSide:
    if side not in ARM_INDICES:
        raise ValueError(f"arm side must be 'left' or 'right', got {side!r}")
    return side


def opposite_arm(side: str) -> ArmSide:
    return "right" if validate_arm_side(side) == "left" else "left"


def arm_indices(side: str) -> tuple[int, ...]:
    return ARM_INDICES[validate_arm_side(side)]


def arm_joint_names(side: str) -> tuple[str, ...]:
    return ARM_JOINT_NAMES[validate_arm_side(side)]


def arm_hand_link(side: str) -> str:
    return ARM_HAND_LINKS[validate_arm_side(side)]


def extract_arm(
    values: Sequence[float] | np.ndarray,
    *,
    side: str,
) -> np.ndarray:
    full = validate_full_joint_vector(values)
    result = full[np.asarray(arm_indices(side))].copy()
    result.setflags(write=False)
    return result


def extract_left_arm(values: Sequence[float] | np.ndarray) -> np.ndarray:
    return extract_arm(values, side="left")


def extract_right_arm(values: Sequence[float] | np.ndarray) -> np.ndarray:
    return extract_arm(values, side="right")


def dual_arm_vector(
    left: Sequence[float] | np.ndarray,
    right: Sequence[float] | np.ndarray,
) -> np.ndarray:
    left_array = validate_arm_vector(left, side="left")
    right_array = validate_arm_vector(right, side="right")
    result = np.concatenate((left_array, right_array))
    result.setflags(write=False)
    return result


def reorder_named_joint_state(
    names: Sequence[str],
    positions: Sequence[float],
    velocities: Sequence[float],
    estimated_torques: Sequence[float],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert a complete named state into authoritative mode-5 order."""
    if (
        len(names) != len(positions)
        or len(names) != len(velocities)
        or len(names) != len(estimated_torques)
    ):
        raise ValueError(
            "joint names, positions, velocities, and estimated torques must have equal length"
        )
    if len(set(names)) != len(names):
        raise ValueError("joint state contains duplicate names")
    provided = set(names)
    required = set(G1_29_JOINT_NAMES)
    missing = sorted(required - provided)
    if missing:
        raise ValueError(f"joint state is missing required names: {', '.join(missing)}")
    by_name = {
        name: (float(position), float(velocity), float(estimated_torque))
        for name, position, velocity, estimated_torque in zip(
            names, positions, velocities, estimated_torques, strict=True
        )
        if name in _NAME_TO_INDEX
    }
    ordered_positions = validate_full_joint_vector(
        [by_name[name][0] for name in G1_29_JOINT_NAMES],
        name="ordered joint positions",
    )
    ordered_velocities = validate_full_joint_vector(
        [by_name[name][1] for name in G1_29_JOINT_NAMES],
        name="ordered joint velocities",
    )
    ordered_estimated_torques = validate_full_joint_vector(
        [by_name[name][2] for name in G1_29_JOINT_NAMES],
        name="ordered estimated joint torques",
    )
    return ordered_positions, ordered_velocities, ordered_estimated_torques
