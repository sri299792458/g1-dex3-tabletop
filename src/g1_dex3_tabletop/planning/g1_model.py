"""Construct the official CuRobo G1 model from one complete measured snapshot."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation

from g1_aprilcube_calibration.joint_map import (
    G1_29_JOINT_NAMES,
    arm_joint_names,
    validate_arm_side,
)
from g1_aprilcube_calibration.transports.unitree_dex3 import (
    DEX3_MOTOR_JOINT_SUFFIXES,
)
from g1_dex3_tabletop.planning.contracts import RobotSnapshot

CUROBO_COMMIT = "8e734f3ced1df898990bcd92de40abce475907db"
CUROBO_G1_CONFIG_RELATIVE = Path("curobo/content/configs/robot/unitree_g1.yml")
CUROBO_G1_URDF_RELATIVE = Path("curobo/content/assets/robot/g1/g1_29dof_with_hand_rev_1_0.urdf")
MOUNT_MANIFESTS = {
    "left": Path("cad/dex3_dorsal_aruco_mount_id5/design_manifest.json"),
    "right": Path("cad/dex3_dorsal_aruco_mount/design_manifest.json"),
}
VIRTUAL_BASE_JOINT_NAMES = (
    "base_j_x",
    "base_j_y",
    "base_j_z",
    "base_j_xtheta",
    "base_j_ytheta",
    "base_j_ztheta",
)
RIGHT_GRASP_FRAME = "right_hand_grasp_frame"
RIGHT_ATTACHMENT_LINK = "right_attached_object"
# Exact GraspGenX dex3_rev1_right descriptor joint.  The shortlist's G frame
# is the descriptor's root/world frame, so this is G_T_palm.
RIGHT_G_T_PALM_XYZ_M = (-0.06158248156116279, 0.0, 0.0)
RIGHT_G_T_PALM_RPY_RAD = (-np.pi / 2.0, -np.pi / 2.0, 0.0)
# The commissioned G1Pilot primitives and NVIDIA's CuRobo spheres agree that
# the retained seated handoff starts 1.19--1.20 mm inside the checked
# right-shoulder-yaw/torso proxy. The older supported-escape policy permits
# existing start penetration while moving out of it. CuRobo requires a free
# start state, so encode the same fixed 1.5 mm proxy-fit allowance here.
RIGHT_SHOULDER_YAW_START_FIT_MARGIN_M = 0.0015


def palm_link(arm: str) -> str:
    return f"{validate_arm_side(arm)}_hand_palm_link"


def curobo_checkout_root() -> Path:
    return Path(__file__).resolve().parents[3] / "third_party" / "curobo"


def model_source_hashes() -> dict[str, str]:
    checkout = curobo_checkout_root()
    config = checkout / CUROBO_G1_CONFIG_RELATIVE
    urdf = checkout / CUROBO_G1_URDF_RELATIVE
    result = {
        "curobo_commit": CUROBO_COMMIT,
        "robot_config_sha256": hashlib.sha256(config.read_bytes()).hexdigest(),
        "urdf_sha256": hashlib.sha256(urdf.read_bytes()).hexdigest(),
    }
    repository = Path(__file__).resolve().parents[3]
    for side, relative in MOUNT_MANIFESTS.items():
        result[f"{side}_mount_manifest_sha256"] = hashlib.sha256(
            (repository / relative).read_bytes()
        ).hexdigest()
    return result


def _mounted_plate_collision_spheres(side: str) -> list[dict[str, Any]]:
    """Cover the complete printed carrier with conservative palm-frame spheres.

    The CAD carrier is 50 x 60 mm and occupies marker-frame z in
    [-4.0, +2.675] mm. A 10 mm square tiling with one sphere per tile covers
    the plate, mounting tab, bosses, and measured print height without adding
    a hand-sized box around the articulated fingers.
    """

    selected_side = validate_arm_side(side)
    repository = Path(__file__).resolve().parents[3]
    manifest_path = repository / MOUNT_MANIFESTS[selected_side]
    with manifest_path.open(encoding="utf-8") as stream:
        manifest = json.load(stream)
    if manifest["hand_side"] != selected_side:
        raise ValueError(f"mount manifest side mismatch: {manifest_path}")
    palm_T_marker = np.asarray(manifest["nominal_palm_T_marker_face_m"], dtype=np.float64)
    if palm_T_marker.shape != (4, 4):
        raise ValueError(f"invalid palm-to-marker transform: {manifest_path}")
    radius_m = float(np.sqrt(0.005**2 + 0.005**2 + 0.0033375**2))
    spheres: list[dict[str, Any]] = []
    for marker_x_m in np.linspace(-0.020, 0.020, 5):
        for marker_y_m in np.linspace(-0.020, 0.030, 6):
            marker_center = np.asarray([marker_x_m, marker_y_m, -0.0006625, 1.0], dtype=np.float64)
            palm_center = palm_T_marker @ marker_center
            spheres.append(
                {
                    "center": [float(value) for value in palm_center[:3]],
                    "radius": radius_m,
                }
            )
    return spheres


def add_mounted_plate_collision_spheres(robot: dict[str, Any]) -> None:
    """Add both physical dorsal marker plates to NVIDIA's detailed hand model."""

    kinematics = robot["kinematics"]
    collision_spheres = kinematics["collision_spheres"]
    for side in ("left", "right"):
        link = palm_link(side)
        if link not in collision_spheres:
            raise ValueError(f"CuRobo model has no collision spheres for {link}")
        collision_spheres[link].extend(_mounted_plate_collision_spheres(side))


def corrected_joint_positions(
    snapshot: RobotSnapshot,
    joint_position_offsets_rad: dict[str, float],
) -> dict[str, float]:
    result = {
        name: float(value) + float(joint_position_offsets_rad.get(name, 0.0))
        for name, value in zip(G1_29_JOINT_NAMES, snapshot.measured_q29_rad, strict=True)
    }
    for side, values in (
        ("left", snapshot.left_dex3_q_rad),
        ("right", snapshot.right_dex3_q_rad),
    ):
        for suffix, value in zip(DEX3_MOTOR_JOINT_SUFFIXES[side], values, strict=True):
            result[f"{side}_hand_{suffix}_joint"] = float(value)
    result.update({name: 0.0 for name in VIRTUAL_BASE_JOINT_NAMES})
    return result


def command_from_model_q(
    model_q: np.ndarray,
    *,
    arm: str,
    joint_position_offsets_rad: dict[str, float],
) -> np.ndarray:
    """Convert CuRobo's corrected kinematic coordinates to Unitree commands."""

    names = arm_joint_names(arm)
    values = np.asarray(model_q, dtype=np.float64).reshape(-1)
    if values.shape != (7,) or not np.all(np.isfinite(values)):
        raise ValueError("CuRobo arm state must contain seven finite values")
    return np.asarray(
        [
            value - float(joint_position_offsets_rad.get(name, 0.0))
            for name, value in zip(names, values, strict=True)
        ],
        dtype=np.float64,
    )


def build_locked_robot_config(
    *,
    arm: str,
    snapshot: RobotSnapshot,
    joint_position_offsets_rad: dict[str, float],
) -> tuple[dict[str, Any], tuple[float, ...]]:
    """Load NVIDIA's complete G1 model and leave only one arm active."""

    selected_arm = validate_arm_side(arm)
    active_names = tuple(arm_joint_names(selected_arm))
    robot, reference = build_robot_config_for_active_joints(
        active_joint_names=active_names,
        snapshot=snapshot,
        joint_position_offsets_rad=joint_position_offsets_rad,
        tool_frames=(palm_link(selected_arm), "torso_link"),
    )
    return robot, reference


def build_robot_config_for_active_joints(
    *,
    active_joint_names: tuple[str, ...],
    snapshot: RobotSnapshot,
    joint_position_offsets_rad: dict[str, float],
    tool_frames: tuple[str, ...] = (),
) -> tuple[dict[str, Any], tuple[float, ...]]:
    """Load the full model while exposing only the explicitly named joints."""

    # CuRobo imports stay inside the planner process by design.
    from curobo.config_io import load_yaml

    if not active_joint_names or len(set(active_joint_names)) != len(active_joint_names):
        raise ValueError("active CuRobo joints must be unique and non-empty")
    checkout = curobo_checkout_root()
    robot = load_yaml(str(checkout / CUROBO_G1_CONFIG_RELATIVE))
    kinematics = robot["kinematics"]
    add_mounted_plate_collision_spheres(robot)
    corrected = corrected_joint_positions(snapshot, joint_position_offsets_rad)
    active_names = tuple(active_joint_names)
    configured_names = tuple(kinematics["cspace"]["joint_names"])
    missing = sorted(set(configured_names) - set(corrected))
    if missing:
        raise ValueError(f"measured snapshot lacks CuRobo joints: {missing}")
    if not set(active_names).issubset(configured_names):
        raise ValueError("selected G1 arm is absent from the CuRobo configuration")
    kinematics["lock_joints"] = {
        name: corrected[name] for name in configured_names if name not in active_names
    }
    # Collision-only consumers do not need a project-specific end-effector,
    # but CuRobo's kinematics loader still requires at least one tool frame to
    # seed the tree.  Preserve NVIDIA's complete-model tool frames when the
    # caller does not override them.  Arm/task planners pass explicit frames.
    if tool_frames:
        kinematics["tool_frames"] = list(tool_frames)
    elif not kinematics.get("tool_frames"):
        raise ValueError("CuRobo robot configuration has no kinematics tool frame")
    reference = tuple(corrected[name] for name in active_names)
    return robot, reference


def right_grasp_T_palm() -> np.ndarray:
    result = np.eye(4, dtype=np.float64)
    result[:3, :3] = Rotation.from_euler("xyz", RIGHT_G_T_PALM_RPY_RAD).as_matrix()
    result[:3, 3] = RIGHT_G_T_PALM_XYZ_M
    return result


def build_tabletop_robot_config(
    *,
    snapshot: RobotSnapshot,
    joint_position_offsets_rad: dict[str, float],
    right_finger_q_rad: tuple[float, ...],
) -> tuple[dict[str, Any], tuple[float, ...]]:
    """Right-arm G1/Dex3 model with GraspGenX G and attachment frames.

    All legs, waist, the left arm, and both hands are locked to the measured
    snapshot except the seven right-arm joints.  The right finger posture is
    explicit because each motion stage is planned against its actual hand
    geometry (initial, open, or closed).
    """

    finger = np.asarray(right_finger_q_rad, dtype=np.float64).reshape(-1)
    if finger.shape != (7,) or not np.all(np.isfinite(finger)):
        raise ValueError("right Dex3 posture must contain seven finite values")
    adjusted = RobotSnapshot(
        measured_q29_rad=snapshot.measured_q29_rad,
        left_dex3_q_rad=snapshot.left_dex3_q_rad,
        right_dex3_q_rad=tuple(float(value) for value in finger),
    )
    robot, reference = build_robot_config_for_active_joints(
        active_joint_names=tuple(arm_joint_names("right")),
        snapshot=adjusted,
        joint_position_offsets_rad=joint_position_offsets_rad,
        tool_frames=(RIGHT_GRASP_FRAME, "torso_link"),
    )
    kinematics = robot["kinematics"]
    palm_T_grasp = np.linalg.inv(right_grasp_T_palm())
    quaternion_xyzw = Rotation.from_matrix(palm_T_grasp[:3, :3]).as_quat()
    palm_T_grasp_pose = [
        *palm_T_grasp[:3, 3].tolist(),
        float(quaternion_xyzw[3]),
        *quaternion_xyzw[:3].tolist(),
    ]
    extra_links = kinematics.setdefault("extra_links", {})
    extra_links[RIGHT_GRASP_FRAME] = {
        "link_name": RIGHT_GRASP_FRAME,
        "joint_name": "right_hand_palm_to_grasp_frame",
        "joint_type": "FIXED",
        "parent_link_name": palm_link("right"),
        "fixed_transform": palm_T_grasp_pose,
    }
    extra_links[RIGHT_ATTACHMENT_LINK] = {
        "link_name": RIGHT_ATTACHMENT_LINK,
        "joint_name": "right_attachment_joint",
        "joint_type": "FIXED",
        "parent_link_name": RIGHT_GRASP_FRAME,
        "fixed_transform": [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
    }
    links = kinematics["collision_link_names"]
    if RIGHT_ATTACHMENT_LINK not in links:
        links.append(RIGHT_ATTACHMENT_LINK)
    # NVIDIA's G1 YAML carries this optional field explicitly as null.  Python
    # ``setdefault`` does not replace an existing None value, so normalize it
    # before reserving the payload spheres used after grasp closure.
    extra_collision_spheres = kinematics.get("extra_collision_spheres")
    if extra_collision_spheres is None:
        extra_collision_spheres = {}
        kinematics["extra_collision_spheres"] = extra_collision_spheres
    extra_collision_spheres[RIGHT_ATTACHMENT_LINK] = 32
    ignore = kinematics.setdefault("self_collision_ignore", {})
    # Every Dex3 joint is locked while CuRobo plans an arm trajectory.  Contact
    # between links within one locked hand is therefore constant and cannot be
    # created or resolved by any active planning coordinate.  NVIDIA's sphere
    # model leaves a few sub-millimetre cross-finger overlaps at valid closed
    # postures; retaining those invariant pairs makes every arm IK seed
    # infeasible.  Ignore only each locked hand's internal pairs.  Hand-to-arm,
    # hand-to-body, hand-to-world, and left-to-right-hand checks stay enabled.
    for side in ("left", "right"):
        locked_hand_links = sorted(
            name for name in kinematics["collision_link_names"] if name.startswith(f"{side}_hand_")
        )
        for index, link in enumerate(locked_hand_links):
            for other in locked_hand_links[index + 1 :]:
                ignore.setdefault(link, [])
                ignore.setdefault(other, [])
                if other not in ignore[link]:
                    ignore[link].append(other)
                if link not in ignore[other]:
                    ignore[other].append(link)
    # G1Pilot's commissioned collision policy starts at shoulder-yaw; the
    # shoulder-roll link is part of the proximal shoulder assembly and is not
    # checked against its adjacent torso geometry. Preserve that relation in
    # CuRobo while keeping shoulder-yaw and every distal arm/body pair active.
    ignore.setdefault("torso_link", [])
    ignore.setdefault("right_shoulder_roll_link", [])
    if "right_shoulder_roll_link" not in ignore["torso_link"]:
        ignore["torso_link"].append("right_shoulder_roll_link")
    if "torso_link" not in ignore["right_shoulder_roll_link"]:
        ignore["right_shoulder_roll_link"].append("torso_link")
    buffers = kinematics.setdefault("self_collision_buffer", {})
    buffers["right_shoulder_yaw_link"] = (
        float(buffers.get("right_shoulder_yaw_link", 0.0))
        - RIGHT_SHOULDER_YAW_START_FIT_MARGIN_M
    )
    hand_links = [name for name in links if name.startswith("right_hand_")]
    ignore[RIGHT_ATTACHMENT_LINK] = sorted(set(hand_links))
    for name in hand_links:
        ignore.setdefault(name, [])
        if RIGHT_ATTACHMENT_LINK not in ignore[name]:
            ignore[name].append(RIGHT_ATTACHMENT_LINK)
    kinematics.setdefault("self_collision_buffer", {})[RIGHT_ATTACHMENT_LINK] = 0.0
    return robot, reference
