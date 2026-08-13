"""Versioned pose-set records and content-hash validation."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field, replace
from importlib.resources import files
from typing import Any

import numpy as np
from jsonschema import Draft202012Validator

from g1_aprilcube_calibration.joint_map import (
    G1_29_JOINT_NAMES,
    G1_MODE_MACHINE,
    arm_indices,
    arm_joint_names,
    validate_arm_side,
    validate_full_joint_vector,
)
from g1_aprilcube_calibration.models import validate_utc_iso

POSE_SET_SCHEMA_VERSION = 3
HANDOFF_POSE_ID = "__handoff__"
_POSE_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


def _finite_tuple(values: Any, *, expected: int, name: str) -> tuple[float, ...]:
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    if array.shape != (expected,):
        raise ValueError(f"{name} must contain {expected} values, got {len(array)}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} contains NaN or infinity")
    return tuple(float(value) for value in array)


def _json_safe_copy(value: Any, *, name: str) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise TypeError(f"{name} must be a dictionary")
    try:
        encoded = json.dumps(value, allow_nan=False, sort_keys=True)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must contain finite JSON-compatible values") from error
    return json.loads(encoded)


@dataclass(frozen=True, slots=True)
class PoseRecord:
    id: str
    group: str
    measured_calibration_q: tuple[float, ...]
    measured_full_q: tuple[float, ...]
    calibration_q_spread: tuple[float, ...]
    recorded_at_utc: str
    recorded_monotonic_s: float
    replay_calibration_q: tuple[float, ...] | None = None
    source: str = "manual_teaching"
    anchor: bool = False
    preview_path: str | None = None
    visual_quality: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not _POSE_ID_PATTERN.fullmatch(self.id):
            raise ValueError(f"invalid pose ID: {self.id!r}")
        if not _POSE_ID_PATTERN.fullmatch(self.group):
            raise ValueError(f"invalid pose group: {self.group!r}")
        calibration = _finite_tuple(
            self.measured_calibration_q,
            expected=7,
            name="measured_calibration_q",
        )
        full = validate_full_joint_vector(self.measured_full_q, name="measured_full_q")
        spread = _finite_tuple(
            self.calibration_q_spread,
            expected=7,
            name="calibration_q_spread",
        )
        if any(value < 0 for value in spread):
            raise ValueError("calibration_q_spread cannot be negative")
        validate_utc_iso(self.recorded_at_utc)
        if not np.isfinite(self.recorded_monotonic_s) or self.recorded_monotonic_s < 0:
            raise ValueError("recorded_monotonic_s must be finite and non-negative")
        if not self.source:
            raise ValueError("pose source must be non-empty")
        if self.preview_path is not None and not self.preview_path:
            raise ValueError("preview_path cannot be an empty string")
        visual_quality = _json_safe_copy(self.visual_quality, name="visual_quality")
        replay = (
            None
            if self.replay_calibration_q is None
            else _finite_tuple(
                self.replay_calibration_q,
                expected=7,
                name="replay_calibration_q",
            )
        )
        object.__setattr__(self, "measured_calibration_q", calibration)
        object.__setattr__(self, "measured_full_q", tuple(float(v) for v in full))
        object.__setattr__(self, "calibration_q_spread", spread)
        object.__setattr__(self, "replay_calibration_q", replay)
        object.__setattr__(self, "visual_quality", visual_quality)

    @property
    def command_calibration_q(self) -> tuple[float, ...]:
        """Return the replay target without changing the recorded measurement."""

        if self.replay_calibration_q is not None:
            return self.replay_calibration_q
        return self.measured_calibration_q

    def to_dict(self) -> dict[str, Any]:
        result = {
            "id": self.id,
            "group": self.group,
            "measured_calibration_q": list(self.measured_calibration_q),
            "measured_full_q": list(self.measured_full_q),
            "calibration_q_spread": list(self.calibration_q_spread),
            "recorded_at_utc": self.recorded_at_utc,
            "recorded_monotonic_s": self.recorded_monotonic_s,
            "source": self.source,
            "anchor": self.anchor,
            "preview_path": self.preview_path,
            "visual_quality": self.visual_quality,
        }
        if self.replay_calibration_q is not None:
            result["replay_calibration_q"] = list(self.replay_calibration_q)
        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PoseRecord:
        return cls(
            id=str(data["id"]),
            group=str(data["group"]),
            measured_calibration_q=tuple(data["measured_calibration_q"]),
            measured_full_q=tuple(data["measured_full_q"]),
            calibration_q_spread=tuple(data["calibration_q_spread"]),
            recorded_at_utc=str(data["recorded_at_utc"]),
            recorded_monotonic_s=float(data["recorded_monotonic_s"]),
            replay_calibration_q=(
                None
                if data.get("replay_calibration_q") is None
                else tuple(data["replay_calibration_q"])
            ),
            source=str(data.get("source", "manual_teaching")),
            anchor=bool(data.get("anchor", False)),
            preview_path=data.get("preview_path"),
            visual_quality=dict(data.get("visual_quality", {})),
        )


@dataclass(frozen=True, slots=True)
class PoseAuditEvent:
    action: str
    pose_id: str
    occurred_at_utc: str
    details: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.action not in {"add", "undo"}:
            raise ValueError(f"unsupported pose audit action: {self.action}")
        if not _POSE_ID_PATTERN.fullmatch(self.pose_id):
            raise ValueError(f"invalid audit pose ID: {self.pose_id!r}")
        validate_utc_iso(self.occurred_at_utc)
        object.__setattr__(
            self,
            "details",
            _json_safe_copy(self.details, name="audit details"),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "action": self.action,
            "pose_id": self.pose_id,
            "occurred_at_utc": self.occurred_at_utc,
            "details": self.details,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PoseAuditEvent:
        return cls(
            action=str(data["action"]),
            pose_id=str(data["pose_id"]),
            occurred_at_utc=str(data["occurred_at_utc"]),
            details=dict(data.get("details", {})),
        )


@dataclass(frozen=True, slots=True)
class PoseSet:
    robot_model: str
    mode_machine: int
    urdf_sha256: str
    calibration_arm: str
    poses: tuple[PoseRecord, ...] = ()
    audit_log: tuple[PoseAuditEvent, ...] = ()
    schema_version: int = POSE_SET_SCHEMA_VERSION
    joint_order: tuple[str, ...] = ()
    full_joint_order: tuple[str, ...] = G1_29_JOINT_NAMES

    def __post_init__(self) -> None:
        if self.schema_version != POSE_SET_SCHEMA_VERSION:
            raise ValueError(
                f"unsupported pose schema version {self.schema_version}; "
                f"expected {POSE_SET_SCHEMA_VERSION}"
            )
        if not self.robot_model:
            raise ValueError("robot_model must be non-empty")
        if self.mode_machine != G1_MODE_MACHINE:
            raise ValueError(f"pose set requires mode_machine={G1_MODE_MACHINE}")
        if not _SHA256_PATTERN.fullmatch(self.urdf_sha256):
            raise ValueError("urdf_sha256 must be 64 lowercase hexadecimal characters")
        calibration_arm = validate_arm_side(self.calibration_arm)
        expected_joint_order = arm_joint_names(calibration_arm)
        joint_order = tuple(self.joint_order) or expected_joint_order
        if joint_order != expected_joint_order:
            raise ValueError(
                f"{calibration_arm}-arm joint order does not match authoritative mapping"
            )
        if tuple(self.full_joint_order) != G1_29_JOINT_NAMES:
            raise ValueError("full joint order does not match authoritative mapping")
        pose_ids = [pose.id for pose in self.poses]
        if len(pose_ids) != len(set(pose_ids)):
            raise ValueError("pose set contains duplicate pose IDs")
        calibration_indices = np.asarray(arm_indices(calibration_arm))
        for pose in self.poses:
            derived = np.asarray(pose.measured_full_q)[calibration_indices]
            if not np.allclose(
                np.asarray(pose.measured_calibration_q),
                derived,
                atol=1e-12,
                rtol=0,
            ):
                raise ValueError(
                    "measured_calibration_q does not match measured_full_q "
                    f"for the {calibration_arm} arm"
                )
        audit_stack: list[str] = []
        for event in self.audit_log:
            if event.action == "add":
                audit_stack.append(event.pose_id)
            elif not audit_stack or audit_stack.pop() != event.pose_id:
                raise ValueError("pose audit log contains an invalid undo sequence")
        if audit_stack != pose_ids:
            raise ValueError("pose audit log does not reproduce the active pose order")
        object.__setattr__(self, "calibration_arm", calibration_arm)
        object.__setattr__(self, "poses", tuple(self.poses))
        object.__setattr__(self, "audit_log", tuple(self.audit_log))
        object.__setattr__(self, "joint_order", joint_order)
        object.__setattr__(self, "full_joint_order", tuple(self.full_joint_order))

    @property
    def content_sha256(self) -> str:
        canonical = json.dumps(
            self.to_dict(include_hash=False),
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
        return hashlib.sha256(canonical).hexdigest()

    def to_dict(self, *, include_hash: bool = True) -> dict[str, Any]:
        data = {
            "schema_version": self.schema_version,
            "robot": {
                "model": self.robot_model,
                "mode_machine": self.mode_machine,
                "urdf_sha256": self.urdf_sha256,
            },
            "calibration_arm": self.calibration_arm,
            "joint_order": list(self.joint_order),
            "full_joint_order": list(self.full_joint_order),
            "poses": [pose.to_dict() for pose in self.poses],
            "audit_log": [event.to_dict() for event in self.audit_log],
        }
        if include_hash:
            data["content_sha256"] = self.content_sha256
        return data

    def with_pose(self, pose: PoseRecord, event: PoseAuditEvent) -> PoseSet:
        if event.action != "add" or event.pose_id != pose.id:
            raise ValueError("add event must reference the appended pose")
        return replace(
            self,
            poses=(*self.poses, pose),
            audit_log=(*self.audit_log, event),
        )

    def without_last_pose(self, event: PoseAuditEvent) -> PoseSet:
        if not self.poses:
            raise ValueError("cannot undo an empty pose set")
        last = self.poses[-1]
        if event.action != "undo" or event.pose_id != last.id:
            raise ValueError("undo event must reference the final pose")
        return replace(
            self,
            poses=self.poses[:-1],
            audit_log=(*self.audit_log, event),
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any], *, verify_hash: bool = True) -> PoseSet:
        validate_pose_set_document(data)
        robot = data["robot"]
        result = cls(
            schema_version=int(data["schema_version"]),
            robot_model=str(robot["model"]),
            mode_machine=int(robot["mode_machine"]),
            urdf_sha256=str(robot["urdf_sha256"]),
            calibration_arm=str(data["calibration_arm"]),
            joint_order=tuple(data["joint_order"]),
            full_joint_order=tuple(data["full_joint_order"]),
            poses=tuple(PoseRecord.from_dict(item) for item in data["poses"]),
            audit_log=tuple(PoseAuditEvent.from_dict(item) for item in data.get("audit_log", [])),
        )
        if verify_hash and data["content_sha256"] != result.content_sha256:
            raise ValueError("pose-set content SHA-256 does not match its contents")
        return result


def validate_pose_set_document(data: dict[str, Any]) -> None:
    schema_path = files("g1_aprilcube_calibration.schemas").joinpath("pose_set.schema.json")
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    Draft202012Validator(schema).validate(data)
