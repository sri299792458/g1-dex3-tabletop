"""Construct the official CuRobo G1 model from one complete measured snapshot."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation

from g1_aprilcube_calibration.joint_map import (
    G1_29_JOINT_NAMES,
    arm_joint_names,
    opposite_arm,
    validate_arm_side,
)
from g1_aprilcube_calibration.transports.unitree_dex3 import (
    DEX3_MOTOR_JOINT_SUFFIXES,
)
from g1_aprilcube_calibration.urdf_model import URDFModel
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
DEX3_CANONICAL_PROFILE = Path("config/tabletop/dex3_rev1_canonical_profile.json")
WAIST_YAW_JOINT_NAME = "waist_yaw_joint"


def palm_link(arm: str) -> str:
    return f"{validate_arm_side(arm)}_hand_palm_link"


def grasp_frame(arm: str) -> str:
    return f"{validate_arm_side(arm)}_hand_grasp_frame"


def attachment_link(arm: str) -> str:
    return f"{validate_arm_side(arm)}_attached_object"


def tabletop_motion_joint_names(
    arm: str,
    *,
    include_waist_yaw: bool = False,
) -> tuple[str, ...]:
    """Return the exact active-coordinate order for tabletop planning.

    Hardware execution remains seven-arm-joint only.  The optional eighth
    coordinate exists so retained observations can be studied with waist yaw
    unlocked before the full-body command contract is commissioned.
    """

    selected = validate_arm_side(arm)
    prefix = (WAIST_YAW_JOINT_NAME,) if include_waist_yaw else ()
    return (*prefix, *arm_joint_names(selected))


def curobo_checkout_root() -> Path:
    return Path(__file__).resolve().parents[3] / "third_party" / "curobo"


class Dex3FingerTargetLimitError(ValueError):
    """A fixed finger target cannot be repaired by moving the shoulders."""


def validate_dex3_finger_targets(*, left_q_rad, right_q_rad, label: str) -> None:
    """Check named motor-order targets using the existing URDF limit reader."""

    model = URDFModel(curobo_checkout_root() / CUROBO_G1_URDF_RELATIVE)
    violations = []
    for side, values in (("left", left_q_rad), ("right", right_q_rad)):
        names = tuple(f"{side}_hand_{suffix}_joint" for suffix in DEX3_MOTOR_JOINT_SUFFIXES[side])
        positions = np.asarray(values, dtype=np.float64).reshape(-1)
        if positions.shape != (7,) or not np.all(np.isfinite(positions)):
            raise ValueError(f"{label}: {side} finger target requires seven finite values")
        for name, value, limit in zip(names, positions, model.joint_limits(names), strict=True):
            if value < limit.lower - 1e-6 or value > limit.upper + 1e-6:
                violations.append(
                    f"{name}={value:.9f}rad outside [{limit.lower:.9f}, {limit.upper:.9f}]rad"
                )
    if violations:
        raise Dex3FingerTargetLimitError(
            f"{label}: "
            + "; ".join(violations)
            + ". Increasing shoulder offset cannot fix a fixed finger target."
        )


def model_source_hashes() -> dict[str, str]:
    checkout = curobo_checkout_root()
    config = checkout / CUROBO_G1_CONFIG_RELATIVE
    urdf = checkout / CUROBO_G1_URDF_RELATIVE
    result = {
        "curobo_commit": CUROBO_COMMIT,
        "robot_config_sha256": hashlib.sha256(config.read_bytes()).hexdigest(),
        "urdf_sha256": hashlib.sha256(urdf.read_bytes()).hexdigest(),
        "dex3_canonical_profile_sha256": hashlib.sha256(
            (Path(__file__).resolve().parents[3] / DEX3_CANONICAL_PROFILE).read_bytes()
        ).hexdigest(),
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


def _ignore_invariant_static_collision_pairs(
    *,
    collision_links: list[str],
    self_collision_ignore: dict[str, list[str]],
    arm: str | None,
) -> None:
    """Remove collision work that no active arm coordinate can change.

    The hardware tabletop planner exposes exactly seven joints from one arm.
    Bilateral calibration passes ``arm=None`` to retain both arm/hand subtrees.
    Every collision link outside the retained subtrees is locked to the live
    measured snapshot, so the relative transform of any two such links is
    constant throughout IK and trajectory optimization. Keep every pair with
    at least one retained arm link; remove only locked-link/locked-link pairs.

    This optimization is deliberately not used by the offline waist-yaw model:
    unlocking the waist makes upper-body-to-leg relationships variable.
    """

    sides = (validate_arm_side(arm),) if arm is not None else ("left", "right")
    collision_link_set = set(collision_links)
    moving_links = set()
    for selected in sides:
        moving_links.update(
            name.removesuffix("_joint") + "_link" for name in arm_joint_names(selected)
        )
        moving_links.update(
            name for name in collision_links if name.startswith(f"{selected}_hand_")
        )
        if arm is not None or attachment_link(selected) in collision_link_set:
            moving_links.add(attachment_link(selected))
    missing = sorted(moving_links - collision_link_set)
    if missing:
        raise ValueError(f"selected-arm collision subtree is incomplete: {missing}")

    static_links = sorted(collision_link_set - moving_links)
    for index, link in enumerate(static_links):
        self_collision_ignore.setdefault(link, [])
        for other in static_links[index + 1 :]:
            self_collision_ignore.setdefault(other, [])
            if other not in self_collision_ignore[link]:
                self_collision_ignore[link].append(other)
            if link not in self_collision_ignore[other]:
                self_collision_ignore[other].append(link)
    for link in static_links:
        self_collision_ignore[link] = sorted(set(self_collision_ignore[link]))


def _ignore_internal_hand_collision_pairs(kinematics: dict[str, Any]) -> None:
    """Apply the commissioned same-hand exclusion without removing geometry.

    The earlier calibration's selected-pair profile excludes same-hand contacts
    for both arm motion and articulated finger sweeps. Tabletop arm planning
    uses this policy for its locked hands. Cross-hand and hand/body pairs, and
    all hand spheres used by world collision checks, remain untouched.
    """

    ignore = kinematics.setdefault("self_collision_ignore", {})
    for side in ("left", "right"):
        hand_links = sorted(
            name for name in kinematics["collision_link_names"] if name.startswith(f"{side}_hand_")
        )
        for index, link in enumerate(hand_links):
            for other in hand_links[index + 1 :]:
                ignore.setdefault(link, [])
                ignore.setdefault(other, [])
                if other not in ignore[link]:
                    ignore[link].append(other)
                if link not in ignore[other]:
                    ignore[other].append(link)


def _ignore_adjacent_shoulder_collision_pairs(kinematics: dict[str, Any]) -> None:
    """Preserve G1Pilot's proximal assembly exclusion, retaining shoulder yaw."""

    ignore = kinematics.setdefault("self_collision_ignore", {})
    ignore.setdefault("torso_link", [])
    for side in ("left", "right"):
        link = f"{side}_shoulder_roll_link"
        ignore.setdefault(link, [])
        if link not in ignore["torso_link"]:
            ignore["torso_link"].append(link)
        if "torso_link" not in ignore[link]:
            ignore[link].append("torso_link")


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
        ignore_internal_hand_collisions=True,
        ignore_adjacent_shoulder_collisions=True,
        ignore_static_body_collisions=True,
    )
    return robot, reference


def build_robot_config_for_active_joints(
    *,
    active_joint_names: tuple[str, ...],
    snapshot: RobotSnapshot,
    joint_position_offsets_rad: dict[str, float],
    tool_frames: tuple[str, ...] = (),
    ignore_internal_hand_collisions: bool = False,
    ignore_adjacent_shoulder_collisions: bool = False,
    ignore_static_body_collisions: bool = False,
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
    if ignore_internal_hand_collisions:
        _ignore_internal_hand_collision_pairs(kinematics)
    if ignore_adjacent_shoulder_collisions:
        _ignore_adjacent_shoulder_collision_pairs(kinematics)
    if ignore_static_body_collisions:
        allowed = {*arm_joint_names("left"), *arm_joint_names("right")}
        allowed.update(
            f"{side}_hand_{suffix}_joint"
            for side in ("left", "right")
            for suffix in DEX3_MOTOR_JOINT_SUFFIXES[side]
        )
        if not set(active_joint_names).issubset(allowed):
            raise ValueError("static-body exclusion requires locked legs and waist")
        _ignore_invariant_static_collision_pairs(
            collision_links=kinematics["collision_link_names"],
            self_collision_ignore=kinematics.setdefault("self_collision_ignore", {}),
            arm=None,
        )
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


def grasp_T_palm(arm: str) -> np.ndarray:
    """Return the selected GraspGenX descriptor's exact G_T_palm."""

    selected = validate_arm_side(arm)
    repository = Path(__file__).resolve().parents[3]
    profile = json.loads((repository / DEX3_CANONICAL_PROFILE).read_text(encoding="utf-8"))
    adapter = profile["side_adapter"][selected]
    result = np.eye(4, dtype=np.float64)
    result[:3, :3] = Rotation.from_euler("xyz", adapter["G_T_palm_rpy_rad"]).as_matrix()
    result[:3, 3] = np.asarray(adapter["G_T_palm_xyz_m"], dtype=np.float64)
    return result


def build_tabletop_robot_config(
    *,
    arm: str,
    snapshot: RobotSnapshot,
    joint_position_offsets_rad: dict[str, float],
    active_finger_q_rad: tuple[float, ...],
    include_waist_yaw: bool = False,
) -> tuple[dict[str, Any], tuple[float, ...]]:
    """Selected-arm G1/Dex3 model with GraspGenX G and attachment frames.

    By default all legs, waist, the opposite arm, and both hands are locked to
    the measured snapshot except the seven selected-arm joints.  Offline waist
    studies may additionally expose waist yaw; no hardware path enables that
    option yet.  The selected finger posture is explicit because each motion
    stage is planned against its actual hand geometry (initial, open, or
    closed).
    """

    selected = validate_arm_side(arm)
    finger = np.asarray(active_finger_q_rad, dtype=np.float64).reshape(-1)
    if finger.shape != (7,) or not np.all(np.isfinite(finger)):
        raise ValueError(f"{selected} Dex3 posture must contain seven finite values")
    left_fingers = snapshot.left_dex3_q_rad
    right_fingers = snapshot.right_dex3_q_rad
    if selected == "left":
        left_fingers = tuple(float(value) for value in finger)
    else:
        right_fingers = tuple(float(value) for value in finger)
    adjusted = RobotSnapshot(
        measured_q29_rad=snapshot.measured_q29_rad,
        left_dex3_q_rad=left_fingers,
        right_dex3_q_rad=right_fingers,
    )
    selected_grasp_frame = grasp_frame(selected)
    selected_attachment_link = attachment_link(selected)
    robot, reference = build_robot_config_for_active_joints(
        active_joint_names=tabletop_motion_joint_names(
            selected,
            include_waist_yaw=include_waist_yaw,
        ),
        snapshot=adjusted,
        joint_position_offsets_rad=joint_position_offsets_rad,
        tool_frames=(selected_grasp_frame, "torso_link"),
    )
    kinematics = robot["kinematics"]
    palm_T_grasp = np.linalg.inv(grasp_T_palm(selected))
    quaternion_xyzw = Rotation.from_matrix(palm_T_grasp[:3, :3]).as_quat()
    palm_T_grasp_pose = [
        *palm_T_grasp[:3, 3].tolist(),
        float(quaternion_xyzw[3]),
        *quaternion_xyzw[:3].tolist(),
    ]
    extra_links = kinematics.setdefault("extra_links", {})
    extra_links[selected_grasp_frame] = {
        "link_name": selected_grasp_frame,
        "joint_name": f"{selected}_hand_palm_to_grasp_frame",
        "joint_type": "FIXED",
        "parent_link_name": palm_link(selected),
        "fixed_transform": palm_T_grasp_pose,
    }
    extra_links[selected_attachment_link] = {
        "link_name": selected_attachment_link,
        "joint_name": f"{selected}_attachment_joint",
        "joint_type": "FIXED",
        "parent_link_name": selected_grasp_frame,
        "fixed_transform": [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
    }
    links = kinematics["collision_link_names"]
    if selected_attachment_link not in links:
        links.append(selected_attachment_link)
    # NVIDIA's G1 YAML carries this optional field explicitly as null.  Python
    # ``setdefault`` does not replace an existing None value, so normalize it
    # before reserving the payload spheres used after grasp closure.
    extra_collision_spheres = kinematics.get("extra_collision_spheres")
    if extra_collision_spheres is None:
        extra_collision_spheres = {}
        kinematics["extra_collision_spheres"] = extra_collision_spheres
    extra_collision_spheres[selected_attachment_link] = 32
    ignore = kinematics.setdefault("self_collision_ignore", {})
    _ignore_internal_hand_collision_pairs(kinematics)
    # G1Pilot's commissioned collision policy starts at shoulder-yaw; the
    # shoulder-roll link is part of the proximal shoulder assembly and is not
    # checked against its adjacent torso geometry. Preserve that relation in
    # CuRobo while keeping shoulder-yaw and every distal arm/body pair active.
    # Both shoulder-roll links are part of their adjacent proximal shoulder
    # assemblies, independent of which arm is active.
    _ignore_adjacent_shoulder_collision_pairs(kinematics)
    # The opposite arm and torso are both locked at the measured takeover
    # state. Their relative geometry is invariant under every active planning
    # coordinate, just like each locked hand's internal geometry above. Keep
    # the selected shoulder-yaw/torso pair strict; exclude only the invariant
    # opposite pair so a measured-start sphere-proxy overlap cannot make every
    # selected-arm IK seed infeasible.
    fixed_shoulder_yaw_link = f"{opposite_arm(selected)}_shoulder_yaw_link"
    ignore.setdefault(fixed_shoulder_yaw_link, [])
    if fixed_shoulder_yaw_link not in ignore["torso_link"]:
        ignore["torso_link"].append(fixed_shoulder_yaw_link)
    if "torso_link" not in ignore[fixed_shoulder_yaw_link]:
        ignore[fixed_shoulder_yaw_link].append("torso_link")
    hand_links = [name for name in links if name.startswith(f"{selected}_hand_")]
    ignore[selected_attachment_link] = sorted(set(hand_links))
    for name in hand_links:
        ignore.setdefault(name, [])
        if selected_attachment_link not in ignore[name]:
            ignore[name].append(selected_attachment_link)
    if not include_waist_yaw:
        _ignore_invariant_static_collision_pairs(
            collision_links=links,
            self_collision_ignore=ignore,
            arm=selected,
        )
    kinematics.setdefault("self_collision_buffer", {})[selected_attachment_link] = 0.0
    return robot, reference


def build_tabletop_route_validation_robot_config(
    *,
    arm: str,
    snapshot: RobotSnapshot,
    joint_position_offsets_rad: dict[str, float],
) -> tuple[dict[str, Any], tuple[str, ...], tuple[float, ...]]:
    """Expose one arm and its fingers for cached measured-contact checking.

    The structural model is independent of the eventual contact posture. The
    worker can therefore parse and upload it once while planning, then insert
    the seven measured finger coordinates after physical contact.
    """

    selected = validate_arm_side(arm)
    finger_names = tuple(
        f"{selected}_hand_{suffix}_joint" for suffix in DEX3_MOTOR_JOINT_SUFFIXES[selected]
    )
    active_names = (*arm_joint_names(selected), *finger_names)
    robot, reference = build_robot_config_for_active_joints(
        active_joint_names=active_names,
        snapshot=snapshot,
        joint_position_offsets_rad=joint_position_offsets_rad,
        tool_frames=(grasp_frame(selected), "torso_link"),
    )
    # Reuse the exact task model additions and collision policy. Only the set
    # of active coordinates differs from arm-trajectory planning.
    task_robot, _ = build_tabletop_robot_config(
        arm=selected,
        snapshot=snapshot,
        joint_position_offsets_rad=joint_position_offsets_rad,
        active_finger_q_rad=(
            snapshot.left_dex3_q_rad if selected == "left" else snapshot.right_dex3_q_rad
        ),
    )
    target = robot["kinematics"]
    source = task_robot["kinematics"]
    for key in (
        "extra_links",
        "collision_link_names",
        "extra_collision_spheres",
        "self_collision_ignore",
        "self_collision_buffer",
    ):
        target[key] = copy.deepcopy(source[key])
    return robot, active_names, reference
