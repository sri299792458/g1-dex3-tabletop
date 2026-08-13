"""Authored camera-space targets and a reversible calibration motion route."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from g1_aprilcube_calibration.joint_map import (
    G1_29_JOINT_NAMES,
    G1_MODE_MACHINE,
    arm_joint_names,
    validate_arm_side,
)
from g1_aprilcube_calibration.pose_schema import HANDOFF_POSE_ID
from g1_aprilcube_calibration.transforms import validate_transform

AUTHORED_COLLECTION_SCHEMA_VERSION = 1
_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")
_SHA_PATTERN = re.compile(r"^[0-9a-f]{64}$")


def _json_copy(value: Any, *, name: str) -> Any:
    try:
        return json.loads(json.dumps(value, sort_keys=True, allow_nan=False))
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be finite JSON-compatible data") from error


def exposed_target_normal_from_hardware(hardware: dict) -> np.ndarray:
    robot = hardware["robot"]
    key = "calibration_target_exposed_face_normal_target"
    raw = np.asarray(
        robot[key],
        dtype=np.float64,
    ).reshape(-1)
    if raw.shape != (3,) or not np.all(np.isfinite(raw)):
        raise ValueError(f"{key} must contain three finite values")
    norm = float(np.linalg.norm(raw))
    if norm <= 0:
        raise ValueError("calibration target exposed-face normal cannot be zero")
    return raw / norm


def modeled_hand_T_target_from_hardware(hardware: dict) -> np.ndarray:
    """Return the configured nominal palm-to-optical-target transform."""

    return validate_transform(
        np.asarray(
            hardware["robot"]["calibration_target_modeled_hand_T_target"],
            dtype=np.float64,
        )
    )


def validate_hardware_target_profile(hardware: dict, target: dict) -> None:
    """Bind an arm-specific hardware profile to exactly one optical marker."""

    if not isinstance(hardware, dict) or not isinstance(target, dict):
        raise TypeError("hardware and target profiles must be mappings")
    robot = hardware["robot"]
    control = hardware["control"]
    arm = validate_arm_side(str(robot["calibration_arm"]))
    if str(control["calibration_arm"]) != arm:
        raise ValueError("hardware robot/control calibration arms disagree")
    if str(robot["calibration_hand_frame"]) != f"{arm}_rubber_hand":
        raise ValueError("hardware calibration hand frame disagrees with arm")
    if str(robot["physical_hand_frame"]) != f"{arm}_hand_palm_link":
        raise ValueError("hardware physical palm frame disagrees with arm")

    marker_id = robot["calibration_target_id"]
    if isinstance(marker_id, bool) or not isinstance(marker_id, int):
        raise TypeError("hardware calibration marker ID must be an integer")
    opposite_marker_id = robot["opposite_hand_calibration_target_id"]
    if isinstance(opposite_marker_id, bool) or not isinstance(opposite_marker_id, int):
        raise TypeError("opposite-hand marker ID must be an integer")
    if marker_id == opposite_marker_id:
        raise ValueError("calibration and opposite-hand marker IDs must differ")
    tag_ids = target.get("tag_ids")
    marker_entries = target.get("markers")
    if tag_ids != [marker_id] or not isinstance(marker_entries, list):
        raise ValueError("target profile must contain exactly the hardware marker ID")
    if [entry.get("id") for entry in marker_entries] != [marker_id]:
        raise ValueError("target marker geometry differs from the hardware marker ID")

    hardware_dictionary = str(robot["calibration_target_dictionary"]).upper()
    target_dictionary = f"DICT_{str(target['dict']).upper()}"
    if hardware_dictionary != target_dictionary:
        raise ValueError("hardware and target ArUco dictionaries disagree")
    if str(target["target"]["mount"]) != str(robot["calibration_target_mount"]):
        raise ValueError("hardware and target mount revisions disagree")
    active_size_m = float(target["tag_size_mm"]) / 1000.0
    if not np.isclose(
        active_size_m,
        float(robot["calibration_target_active_size_m"]),
        atol=1e-12,
    ):
        raise ValueError("hardware and target active marker sizes disagree")


@dataclass(frozen=True, slots=True)
class AuthoredPoseTarget:
    """A command target, explicitly distinct from an encoder measurement."""

    id: str
    authored_calibration_q: tuple[float, ...]
    desired_camera_T_cube: np.ndarray
    ik_diagnostics: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not _ID_PATTERN.fullmatch(self.id) or self.id == HANDOFF_POSE_ID:
            raise ValueError(f"invalid authored target ID: {self.id!r}")
        q = np.asarray(self.authored_calibration_q, dtype=np.float64).reshape(-1)
        if q.shape != (7,) or not np.all(np.isfinite(q)):
            raise ValueError("authored target must contain seven finite joints")
        object.__setattr__(self, "authored_calibration_q", tuple(float(v) for v in q))
        object.__setattr__(
            self,
            "desired_camera_T_cube",
            validate_transform(self.desired_camera_T_cube),
        )
        object.__setattr__(
            self,
            "ik_diagnostics",
            _json_copy(self.ik_diagnostics, name="IK diagnostics"),
        )

    @property
    def command_calibration_q(self) -> tuple[float, ...]:
        return self.authored_calibration_q

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "authored_calibration_q": list(self.authored_calibration_q),
            "desired_camera_T_cube": self.desired_camera_T_cube.tolist(),
            "ik_diagnostics": self.ik_diagnostics,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AuthoredPoseTarget:
        if set(data) != {
            "id",
            "authored_calibration_q",
            "desired_camera_T_cube",
            "ik_diagnostics",
        }:
            raise ValueError("authored target fields do not match schema version 1")
        return cls(
            id=str(data["id"]),
            authored_calibration_q=tuple(data["authored_calibration_q"]),
            desired_camera_T_cube=np.asarray(data["desired_camera_T_cube"], dtype=np.float64),
            ik_diagnostics=dict(data["ik_diagnostics"]),
        )


@dataclass(frozen=True, slots=True)
class AuthoredCollectionPlan:
    """Immutable authored targets plus a collision-validated reversible route."""

    robot_model: str
    mode_machine: int
    urdf_sha256: str
    calibration_arm: str
    camera_profile_sha256: str
    reference_full_q_sha256: str
    targets: tuple[AuthoredPoseTarget, ...]
    route_pose_ids: tuple[str, ...]
    capture_pose_ids: tuple[str, ...]
    generation_config: dict[str, Any]
    schema_version: int = AUTHORED_COLLECTION_SCHEMA_VERSION
    joint_order: tuple[str, ...] = ()
    full_joint_order: tuple[str, ...] = G1_29_JOINT_NAMES

    def __post_init__(self) -> None:
        if self.schema_version != AUTHORED_COLLECTION_SCHEMA_VERSION:
            raise ValueError("unsupported authored collection schema version")
        if not self.robot_model:
            raise ValueError("robot_model must be non-empty")
        if self.mode_machine != G1_MODE_MACHINE:
            raise ValueError(f"authored collection requires mode_machine={G1_MODE_MACHINE}")
        for value in (
            self.urdf_sha256,
            self.camera_profile_sha256,
            self.reference_full_q_sha256,
        ):
            if not _SHA_PATTERN.fullmatch(value):
                raise ValueError("authored collection hashes must be lowercase SHA-256")
        arm = validate_arm_side(self.calibration_arm)
        expected_joint_order = arm_joint_names(arm)
        resolved_joint_order = self.joint_order or expected_joint_order
        if tuple(resolved_joint_order) != expected_joint_order:
            raise ValueError("authored target joint order does not match calibration arm")
        if tuple(self.full_joint_order) != G1_29_JOINT_NAMES:
            raise ValueError("authored full joint order does not match G1 mode 5")
        targets = tuple(self.targets)
        target_ids = [target.id for target in targets]
        if not targets or len(target_ids) != len(set(target_ids)):
            raise ValueError("authored collection requires unique non-empty targets")
        route = tuple(self.route_pose_ids)
        captures = tuple(self.capture_pose_ids)
        if len(route) < 3 or route[0] != HANDOFF_POSE_ID or route[-1] != HANDOFF_POSE_ID:
            raise ValueError("authored route must start and end at the live handoff")
        known = {HANDOFF_POSE_ID, *target_ids}
        if any(item not in known for item in route):
            raise ValueError("authored route references an unknown target")
        if not captures or len(captures) != len(set(captures)):
            raise ValueError("authored capture targets must be unique and non-empty")
        if set(captures) != set(target_ids):
            raise ValueError("every authored target must be captured exactly once")
        first_visit = tuple(dict.fromkeys(item for item in route if item != HANDOFF_POSE_ID))
        if captures != first_visit:
            raise ValueError("capture order must match the route's first target visits")
        object.__setattr__(self, "calibration_arm", arm)
        object.__setattr__(self, "joint_order", tuple(resolved_joint_order))
        object.__setattr__(self, "full_joint_order", tuple(self.full_joint_order))
        object.__setattr__(self, "targets", targets)
        object.__setattr__(self, "route_pose_ids", route)
        object.__setattr__(self, "capture_pose_ids", captures)
        object.__setattr__(
            self,
            "generation_config",
            _json_copy(self.generation_config, name="generation config"),
        )

    @property
    def poses(self) -> tuple[AuthoredPoseTarget, ...]:
        """Command-set compatibility used by the validator and executor."""

        return self.targets

    @property
    def content_sha256(self) -> str:
        encoded = json.dumps(
            self.to_dict(include_hash=False),
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode()
        return hashlib.sha256(encoded).hexdigest()

    def to_dict(self, *, include_hash: bool = True) -> dict[str, Any]:
        result = {
            "schema_version": self.schema_version,
            "robot_model": self.robot_model,
            "mode_machine": self.mode_machine,
            "urdf_sha256": self.urdf_sha256,
            "calibration_arm": self.calibration_arm,
            "joint_order": list(self.joint_order),
            "full_joint_order": list(self.full_joint_order),
            "camera_profile_sha256": self.camera_profile_sha256,
            "reference_full_q_sha256": self.reference_full_q_sha256,
            "generation_config": self.generation_config,
            "targets": [target.to_dict() for target in self.targets],
            "route_pose_ids": list(self.route_pose_ids),
            "capture_pose_ids": list(self.capture_pose_ids),
        }
        if include_hash:
            result["content_sha256"] = self.content_sha256
        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AuthoredCollectionPlan:
        expected = {
            "schema_version",
            "robot_model",
            "mode_machine",
            "urdf_sha256",
            "calibration_arm",
            "joint_order",
            "full_joint_order",
            "camera_profile_sha256",
            "reference_full_q_sha256",
            "generation_config",
            "targets",
            "route_pose_ids",
            "capture_pose_ids",
            "content_sha256",
        }
        if set(data) != expected:
            raise ValueError("authored collection fields do not match schema version 1")
        result = cls(
            schema_version=int(data["schema_version"]),
            robot_model=str(data["robot_model"]),
            mode_machine=int(data["mode_machine"]),
            urdf_sha256=str(data["urdf_sha256"]),
            calibration_arm=str(data["calibration_arm"]),
            joint_order=tuple(data["joint_order"]),
            full_joint_order=tuple(data["full_joint_order"]),
            camera_profile_sha256=str(data["camera_profile_sha256"]),
            reference_full_q_sha256=str(data["reference_full_q_sha256"]),
            generation_config=dict(data["generation_config"]),
            targets=tuple(AuthoredPoseTarget.from_dict(item) for item in data["targets"]),
            route_pose_ids=tuple(data["route_pose_ids"]),
            capture_pose_ids=tuple(data["capture_pose_ids"]),
        )
        if data["content_sha256"] != result.content_sha256:
            raise ValueError("authored collection content SHA-256 does not match")
        return result

    @classmethod
    def from_json(cls, path: str) -> AuthoredCollectionPlan:
        with open(path, encoding="utf-8") as stream:
            return cls.from_dict(json.load(stream))


def validate_exposed_camera_views(
    plan: AuthoredCollectionPlan,
    *,
    exposed_target_normal: np.ndarray,
) -> None:
    """Reject plans that view the mounted target through its palm-side half-space."""

    normal = np.asarray(exposed_target_normal, dtype=np.float64).reshape(-1)
    if normal.shape != (3,) or not np.all(np.isfinite(normal)):
        raise ValueError("exposed cube normal must contain three finite values")
    norm = float(np.linalg.norm(normal))
    if norm <= 0:
        raise ValueError("exposed cube normal cannot be zero")
    normal = normal / norm
    planned = np.asarray(
        plan.generation_config.get("exposed_target_normal", []), dtype=np.float64
    ).reshape(-1)
    if planned.shape != (3,) or not np.all(np.isfinite(planned)):
        raise ValueError("authored plan does not record its exposed cube normal")
    planned_norm = float(np.linalg.norm(planned))
    if planned_norm <= 0 or not np.allclose(planned / planned_norm, normal, atol=1e-12):
        raise ValueError("authored plan exposed cube normal differs from hardware")
    for target in plan.targets:
        camera_T_cube = target.desired_camera_T_cube
        cube_to_camera = camera_T_cube[:3, :3].T @ -camera_T_cube[:3, 3]
        direction_norm = float(np.linalg.norm(cube_to_camera))
        if direction_norm <= 0 or not np.isfinite(direction_norm):
            raise ValueError(f"authored target {target.id} has no camera direction")
        exposed_dot = float(cube_to_camera @ normal / direction_norm)
        if exposed_dot <= 0:
            raise ValueError(
                f"authored target {target.id} views the cube through the "
                f"palm-side hemisphere ({exposed_dot=:.6f})"
            )


def validate_modeled_hand_target_binding(
    plan: AuthoredCollectionPlan,
    *,
    modeled_hand_T_target: np.ndarray,
) -> None:
    """Reject an authored route generated for different mounted geometry."""

    configured = validate_transform(np.asarray(modeled_hand_T_target, dtype=np.float64))
    planned = np.asarray(
        plan.generation_config.get("modeled_hand_T_target", []),
        dtype=np.float64,
    )
    if planned.shape != (4, 4):
        raise ValueError("authored plan does not record its modeled hand-to-target pose")
    planned = validate_transform(planned)
    if not np.allclose(planned, configured, atol=1e-12, rtol=0.0):
        raise ValueError("authored plan modeled hand-to-target pose differs from hardware")
