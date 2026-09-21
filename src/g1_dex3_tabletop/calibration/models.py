"""Hash-bound contracts for simultaneous left/right target observations."""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np

from g1_aprilcube_calibration.joint_map import (
    G1_29_JOINT_NAMES,
    arm_joint_names,
    validate_full_joint_vector,
)
from g1_aprilcube_calibration.transforms import validate_transform

BILATERAL_DATASET_SCHEMA_VERSION = 1
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_CAPTURE_ROLES = frozenset({"anchor", "excitation"})
_CAMERA_COMPONENTS = ("x", "y", "z", "roll", "pitch", "yaw")


def _validate_sha256(value: str, name: str) -> str:
    if not _SHA256_PATTERN.fullmatch(value):
        raise ValueError(f"{name} must be lowercase SHA-256")
    return value


def _canonical_mapping(value: dict[str, Any], name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise TypeError(f"{name} must be a mapping")
    result = json.loads(json.dumps(value, sort_keys=True, allow_nan=False))
    if not isinstance(result, dict):
        raise TypeError(f"{name} must be a mapping")
    return result


@dataclass(frozen=True, slots=True)
class CameraFrameArtifact:
    """Measured serial-specific RealSense color-frame chain."""

    camera_serial: str
    urdf_parent_link: str
    parent_frame: str
    color_frame: str
    optical_frame: str
    parent_T_color: tuple[tuple[float, ...], ...]
    color_T_optical: tuple[tuple[float, ...], ...]
    measured_at_utc: str
    acquisition: dict[str, Any]
    schema_version: int = 1

    def __post_init__(self) -> None:
        if self.schema_version != 1:
            raise ValueError("unsupported camera-frame artifact schema version")
        if not self.camera_serial.strip():
            raise ValueError("camera serial must be non-empty")
        if (
            not self.urdf_parent_link
            or not self.parent_frame
            or not self.color_frame
            or not self.optical_frame
        ):
            raise ValueError("camera artifact frame names must be non-empty")
        if not self.measured_at_utc:
            raise ValueError("camera artifact measurement time must be non-empty")
        parent_T_color = validate_transform(self.parent_T_color)
        color_T_optical = validate_transform(self.color_T_optical)
        object.__setattr__(
            self,
            "parent_T_color",
            tuple(tuple(float(value) for value in row) for row in parent_T_color),
        )
        object.__setattr__(
            self,
            "color_T_optical",
            tuple(tuple(float(value) for value in row) for row in color_T_optical),
        )
        object.__setattr__(
            self,
            "acquisition",
            _canonical_mapping(self.acquisition, "camera artifact acquisition"),
        )

    @property
    def parent_T_optical(self) -> np.ndarray:
        return validate_transform(
            np.asarray(self.parent_T_color) @ np.asarray(self.color_T_optical)
        )

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
            "camera_serial": self.camera_serial,
            "urdf_parent_link": self.urdf_parent_link,
            "parent_frame": self.parent_frame,
            "color_frame": self.color_frame,
            "optical_frame": self.optical_frame,
            "parent_T_color": [list(row) for row in self.parent_T_color],
            "color_T_optical": [list(row) for row in self.color_T_optical],
            "measured_at_utc": self.measured_at_utc,
            "acquisition": self.acquisition,
        }
        if include_hash:
            result["content_sha256"] = self.content_sha256
        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CameraFrameArtifact:
        expected = {
            "schema_version",
            "camera_serial",
            "urdf_parent_link",
            "parent_frame",
            "color_frame",
            "optical_frame",
            "parent_T_color",
            "color_T_optical",
            "measured_at_utc",
            "acquisition",
            "content_sha256",
        }
        if set(data) != expected:
            raise ValueError("camera-frame artifact fields do not match schema version 1")
        content_hash = data["content_sha256"]
        result = cls(
            schema_version=int(data["schema_version"]),
            camera_serial=data["camera_serial"],
            urdf_parent_link=data["urdf_parent_link"],
            parent_frame=data["parent_frame"],
            color_frame=data["color_frame"],
            optical_frame=data["optical_frame"],
            parent_T_color=tuple(tuple(row) for row in data["parent_T_color"]),
            color_T_optical=tuple(tuple(row) for row in data["color_T_optical"]),
            measured_at_utc=data["measured_at_utc"],
            acquisition=dict(data["acquisition"]),
        )
        if result.content_sha256 != content_hash:
            raise ValueError("camera-frame artifact content SHA-256 mismatch")
        return result

    @classmethod
    def from_json(cls, path: str | Path) -> CameraFrameArtifact:
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


@dataclass(frozen=True, slots=True)
class TargetObservation:
    """One hand target detected in the shared camera frame."""

    side: Literal["left", "right"]
    target_artifact_sha256: str
    visible_tag_ids: tuple[int, ...]
    corner_tag_ids: tuple[int, ...]
    image_points_px: tuple[tuple[float, float], ...]
    object_points_m: tuple[tuple[float, float, float], ...]
    correspondence_sha256: str

    def __post_init__(self) -> None:
        if self.side not in {"left", "right"}:
            raise ValueError("target observation side must be left or right")
        _validate_sha256(self.target_artifact_sha256, "target artifact hash")
        _validate_sha256(self.correspondence_sha256, "correspondence hash")
        image_points = np.asarray(self.image_points_px, dtype=np.float64)
        object_points = np.asarray(self.object_points_m, dtype=np.float64)
        if image_points.ndim != 2 or image_points.shape[1:] != (2,):
            raise ValueError("image points must have shape (N, 2)")
        if len(image_points) < 4 or len(image_points) % 4:
            raise ValueError("a target observation must contain complete tag corners")
        if object_points.shape != (len(image_points), 3):
            raise ValueError("object points must have shape (N, 3) and match image points")
        if not np.all(np.isfinite(image_points)) or not np.all(np.isfinite(object_points)):
            raise ValueError("target observation points must be finite")
        corner_tag_ids = tuple(int(value) for value in self.corner_tag_ids)
        visible_tag_ids = tuple(int(value) for value in self.visible_tag_ids)
        if len(corner_tag_ids) != len(image_points):
            raise ValueError("corner tag IDs must match the point count")
        if tuple(sorted(set(corner_tag_ids))) != visible_tag_ids:
            raise ValueError("visible tag IDs must be sorted and match corner tag IDs")
        object.__setattr__(self, "visible_tag_ids", visible_tag_ids)
        object.__setattr__(self, "corner_tag_ids", corner_tag_ids)
        object.__setattr__(
            self,
            "image_points_px",
            tuple(tuple(float(value) for value in point) for point in image_points),
        )
        object.__setattr__(
            self,
            "object_points_m",
            tuple(tuple(float(value) for value in point) for point in object_points),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "side": self.side,
            "target_artifact_sha256": self.target_artifact_sha256,
            "visible_tag_ids": list(self.visible_tag_ids),
            "corner_tag_ids": list(self.corner_tag_ids),
            "image_points_px": [list(point) for point in self.image_points_px],
            "object_points_m": [list(point) for point in self.object_points_m],
            "correspondence_sha256": self.correspondence_sha256,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TargetObservation:
        expected = {
            "side",
            "target_artifact_sha256",
            "visible_tag_ids",
            "corner_tag_ids",
            "image_points_px",
            "object_points_m",
            "correspondence_sha256",
        }
        if set(data) != expected:
            raise ValueError("target observation fields do not match schema version 1")
        return cls(
            side=data["side"],
            target_artifact_sha256=data["target_artifact_sha256"],
            visible_tag_ids=tuple(data["visible_tag_ids"]),
            corner_tag_ids=tuple(data["corner_tag_ids"]),
            image_points_px=tuple(tuple(point) for point in data["image_points_px"]),
            object_points_m=tuple(tuple(point) for point in data["object_points_m"]),
            correspondence_sha256=data["correspondence_sha256"],
        )


@dataclass(frozen=True, slots=True)
class BilateralCalibrationSample:
    """Two hand targets observed in one raw frame and one paired full state."""

    source_session_id: str
    capture_id: str
    pose_group_id: str
    day_group_id: str
    frame_id: str
    capture_role: Literal["anchor", "excitation"]
    raw_image_path: str
    raw_image_sha256: str
    camera_info: dict[str, Any]
    joint_positions_rad: tuple[float, ...]
    joint_velocities_rad_s: tuple[float, ...]
    pairing: dict[str, Any]
    left: TargetObservation
    right: TargetObservation

    def __post_init__(self) -> None:
        if (
            not self.source_session_id
            or not self.capture_id
            or not self.pose_group_id
            or not self.day_group_id
            or not self.frame_id
        ):
            raise ValueError("bilateral sample IDs must be non-empty")
        if self.capture_role not in _CAPTURE_ROLES:
            raise ValueError(f"unsupported capture role: {self.capture_role}")
        if not self.raw_image_path:
            raise ValueError("raw image path must be non-empty")
        _validate_sha256(self.raw_image_sha256, "raw image hash")
        left = (
            self.left
            if isinstance(self.left, TargetObservation)
            else TargetObservation.from_dict(self.left)
        )
        right = (
            self.right
            if isinstance(self.right, TargetObservation)
            else TargetObservation.from_dict(self.right)
        )
        if left.side != "left" or right.side != "right":
            raise ValueError("bilateral sample must contain left and right observations")
        positions = validate_full_joint_vector(
            self.joint_positions_rad,
            name="bilateral sample joint positions",
        )
        velocities = validate_full_joint_vector(
            self.joint_velocities_rad_s,
            name="bilateral sample joint velocities",
        )
        camera_info = _canonical_mapping(self.camera_info, "camera_info")
        pairing = _canonical_mapping(self.pairing, "pairing")
        object.__setattr__(self, "left", left)
        object.__setattr__(self, "right", right)
        object.__setattr__(self, "camera_info", camera_info)
        object.__setattr__(self, "pairing", pairing)
        object.__setattr__(self, "joint_positions_rad", tuple(float(value) for value in positions))
        object.__setattr__(
            self,
            "joint_velocities_rad_s",
            tuple(float(value) for value in velocities),
        )

    @property
    def observations(self) -> tuple[TargetObservation, TargetObservation]:
        return self.left, self.right

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_session_id": self.source_session_id,
            "capture_id": self.capture_id,
            "pose_group_id": self.pose_group_id,
            "day_group_id": self.day_group_id,
            "frame_id": self.frame_id,
            "capture_role": self.capture_role,
            "raw_image_path": self.raw_image_path,
            "raw_image_sha256": self.raw_image_sha256,
            "camera_info": self.camera_info,
            "joint_names": list(G1_29_JOINT_NAMES),
            "joint_positions_rad": list(self.joint_positions_rad),
            "joint_velocities_rad_s": list(self.joint_velocities_rad_s),
            "pairing": self.pairing,
            "observations": {
                "left": self.left.to_dict(),
                "right": self.right.to_dict(),
            },
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> BilateralCalibrationSample:
        expected = {
            "source_session_id",
            "capture_id",
            "pose_group_id",
            "day_group_id",
            "frame_id",
            "capture_role",
            "raw_image_path",
            "raw_image_sha256",
            "camera_info",
            "joint_names",
            "joint_positions_rad",
            "joint_velocities_rad_s",
            "pairing",
            "observations",
        }
        if set(data) != expected:
            raise ValueError("bilateral sample fields do not match schema version 1")
        if tuple(data["joint_names"]) != G1_29_JOINT_NAMES:
            raise ValueError("bilateral sample joint names do not match G1 mode-5 order")
        observations = data["observations"]
        if not isinstance(observations, dict) or set(observations) != {"left", "right"}:
            raise ValueError("bilateral sample requires exactly left and right observations")
        return cls(
            source_session_id=data["source_session_id"],
            capture_id=data["capture_id"],
            pose_group_id=data["pose_group_id"],
            day_group_id=data["day_group_id"],
            frame_id=data["frame_id"],
            capture_role=data["capture_role"],
            raw_image_path=data["raw_image_path"],
            raw_image_sha256=data["raw_image_sha256"],
            camera_info=dict(data["camera_info"]),
            joint_positions_rad=tuple(data["joint_positions_rad"]),
            joint_velocities_rad_s=tuple(data["joint_velocities_rad_s"]),
            pairing=dict(data["pairing"]),
            left=TargetObservation.from_dict(observations["left"]),
            right=TargetObservation.from_dict(observations["right"]),
        )


@dataclass(frozen=True, slots=True)
class BilateralCalibrationDataset:
    """Immutable solver input derived from a same-frame bilateral raw session."""

    dataset_id: str
    session_manifest_sha256_by_id: dict[str, str]
    pose_design_sha256: str
    execution_plan_sha256: str
    urdf_sha256: str
    rgb_optical_transform_sha256: str
    left_target_artifact_sha256: str
    right_target_artifact_sha256: str
    samples: tuple[BilateralCalibrationSample, ...]
    provenance: dict[str, Any]
    schema_version: int = BILATERAL_DATASET_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != BILATERAL_DATASET_SCHEMA_VERSION:
            raise ValueError("unsupported bilateral dataset schema version")
        if not self.dataset_id:
            raise ValueError("bilateral dataset ID must be non-empty")
        source_sessions = {
            str(name): str(value) for name, value in self.session_manifest_sha256_by_id.items()
        }
        if not source_sessions or any(
            not name or not _SHA256_PATTERN.fullmatch(value)
            for name, value in source_sessions.items()
        ):
            raise ValueError("bilateral dataset source sessions must map IDs to SHA-256")
        for name in (
            "pose_design_sha256",
            "execution_plan_sha256",
            "urdf_sha256",
            "rgb_optical_transform_sha256",
            "left_target_artifact_sha256",
            "right_target_artifact_sha256",
        ):
            _validate_sha256(getattr(self, name), name)
        samples = tuple(
            value
            if isinstance(value, BilateralCalibrationSample)
            else BilateralCalibrationSample.from_dict(value)
            for value in self.samples
        )
        if not samples:
            raise ValueError("bilateral dataset must contain at least one sample")
        sample_source_ids = {sample.source_session_id for sample in samples}
        if sample_source_ids != set(source_sessions):
            raise ValueError("bilateral samples and source-session manifests differ")
        capture_ids = [(sample.source_session_id, sample.capture_id) for sample in samples]
        frame_ids = [(sample.source_session_id, sample.frame_id) for sample in samples]
        if len(capture_ids) != len(set(capture_ids)):
            raise ValueError("bilateral dataset contains duplicate capture IDs")
        if len(frame_ids) != len(set(frame_ids)):
            raise ValueError("bilateral dataset contains duplicate frame IDs")
        for sample in samples:
            if sample.left.target_artifact_sha256 != self.left_target_artifact_sha256:
                raise ValueError("left observation target hash differs from dataset")
            if sample.right.target_artifact_sha256 != self.right_target_artifact_sha256:
                raise ValueError("right observation target hash differs from dataset")
        object.__setattr__(self, "samples", samples)
        object.__setattr__(
            self,
            "session_manifest_sha256_by_id",
            dict(sorted(source_sessions.items())),
        )
        object.__setattr__(self, "provenance", _canonical_mapping(self.provenance, "provenance"))

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
            "dataset_id": self.dataset_id,
            "session_manifest_sha256_by_id": self.session_manifest_sha256_by_id,
            "pose_design_sha256": self.pose_design_sha256,
            "execution_plan_sha256": self.execution_plan_sha256,
            "urdf_sha256": self.urdf_sha256,
            "rgb_optical_transform_sha256": self.rgb_optical_transform_sha256,
            "left_target_artifact_sha256": self.left_target_artifact_sha256,
            "right_target_artifact_sha256": self.right_target_artifact_sha256,
            "samples": [sample.to_dict() for sample in self.samples],
            "provenance": self.provenance,
        }
        if include_hash:
            result["content_sha256"] = self.content_sha256
        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> BilateralCalibrationDataset:
        expected = {
            "schema_version",
            "dataset_id",
            "session_manifest_sha256_by_id",
            "pose_design_sha256",
            "execution_plan_sha256",
            "urdf_sha256",
            "rgb_optical_transform_sha256",
            "left_target_artifact_sha256",
            "right_target_artifact_sha256",
            "samples",
            "provenance",
            "content_sha256",
        }
        if set(data) != expected:
            raise ValueError("bilateral dataset fields do not match schema version 1")
        content_hash = data["content_sha256"]
        result = cls(
            schema_version=int(data["schema_version"]),
            dataset_id=data["dataset_id"],
            session_manifest_sha256_by_id=dict(data["session_manifest_sha256_by_id"]),
            pose_design_sha256=data["pose_design_sha256"],
            execution_plan_sha256=data["execution_plan_sha256"],
            urdf_sha256=data["urdf_sha256"],
            rgb_optical_transform_sha256=data["rgb_optical_transform_sha256"],
            left_target_artifact_sha256=data["left_target_artifact_sha256"],
            right_target_artifact_sha256=data["right_target_artifact_sha256"],
            samples=tuple(BilateralCalibrationSample.from_dict(item) for item in data["samples"]),
            provenance=dict(data["provenance"]),
        )
        if result.content_sha256 != content_hash:
            raise ValueError("bilateral dataset content SHA-256 mismatch")
        return result

    @classmethod
    def from_json(cls, path: str | Path) -> BilateralCalibrationDataset:
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))

    def write_json(self, path: str | Path) -> Path:
        destination = Path(path).resolve()
        destination.parent.mkdir(parents=True, exist_ok=True)
        data = (
            json.dumps(self.to_dict(), indent=2, sort_keys=True, allow_nan=False) + "\n"
        ).encode()
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{destination.name}.",
            suffix=".tmp",
            dir=destination.parent,
        )
        temporary = Path(temporary_name)
        try:
            with os.fdopen(descriptor, "wb") as stream:
                stream.write(data)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, destination)
        finally:
            temporary.unlink(missing_ok=True)
        return destination


@dataclass(frozen=True, slots=True)
class BilateralModelSpec:
    """One declared, auditable Ferguson model to fit and cross-validate."""

    name: str
    left_joint_offsets: tuple[str, ...] = ()
    right_joint_offsets: tuple[str, ...] = ()
    camera_components: tuple[str, ...] = _CAMERA_COMPONENTS
    optimize_hand_targets: bool = True
    joint_offset_prior_sigma_deg: float = 5.0

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("bilateral model name must be non-empty")
        if not isinstance(self.optimize_hand_targets, bool):
            raise TypeError("optimize_hand_targets must be boolean")
        camera_components = tuple(dict.fromkeys(self.camera_components))
        invalid_components = sorted(set(camera_components) - set(_CAMERA_COMPONENTS))
        if invalid_components:
            raise ValueError("unsupported camera components: " + ", ".join(invalid_components))
        if not camera_components:
            raise ValueError("at least one camera component must be free")
        left = tuple(dict.fromkeys(self.left_joint_offsets))
        right = tuple(dict.fromkeys(self.right_joint_offsets))
        invalid_left = sorted(set(left) - set(arm_joint_names("left")))
        invalid_right = sorted(set(right) - set(arm_joint_names("right")))
        if invalid_left or invalid_right:
            raise ValueError(
                f"joint offsets do not match their arms: left={invalid_left}, right={invalid_right}"
            )
        sigma = float(self.joint_offset_prior_sigma_deg)
        if not np.isfinite(sigma) or sigma <= 0.0:
            raise ValueError("joint offset prior sigma must be positive and finite")
        object.__setattr__(self, "camera_components", camera_components)
        object.__setattr__(self, "left_joint_offsets", left)
        object.__setattr__(self, "right_joint_offsets", right)
        object.__setattr__(self, "joint_offset_prior_sigma_deg", sigma)

    @property
    def joint_offsets(self) -> tuple[str, ...]:
        return self.left_joint_offsets + self.right_joint_offsets

    @property
    def content_sha256(self) -> str:
        encoded = json.dumps(
            self.to_dict(),
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode()
        return hashlib.sha256(encoded).hexdigest()

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "left_joint_offsets": list(self.left_joint_offsets),
            "right_joint_offsets": list(self.right_joint_offsets),
            "camera_components": list(self.camera_components),
            "optimize_hand_targets": self.optimize_hand_targets,
            "joint_offset_prior_sigma_deg": self.joint_offset_prior_sigma_deg,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> BilateralModelSpec:
        expected = {
            "name",
            "left_joint_offsets",
            "right_joint_offsets",
            "camera_components",
            "optimize_hand_targets",
            "joint_offset_prior_sigma_deg",
        }
        if set(data) != expected:
            raise ValueError("bilateral model fields do not match schema version 1")
        return cls(
            name=data["name"],
            left_joint_offsets=tuple(data["left_joint_offsets"]),
            right_joint_offsets=tuple(data["right_joint_offsets"]),
            camera_components=tuple(data["camera_components"]),
            optimize_hand_targets=data["optimize_hand_targets"],
            joint_offset_prior_sigma_deg=float(data["joint_offset_prior_sigma_deg"]),
        )
