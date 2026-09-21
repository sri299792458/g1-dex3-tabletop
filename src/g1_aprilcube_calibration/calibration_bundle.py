"""Versioned, removable deployment overlays for calibrated G1 geometry."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import xml.etree.ElementTree as ET
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from g1_aprilcube_calibration.camera_initialization import (
    realsense_link_T_color_optical,
)
from g1_aprilcube_calibration.joint_map import G1_29_JOINT_NAMES
from g1_aprilcube_calibration.transforms import validate_transform
from g1_aprilcube_calibration.urdf_model import transform_from_xyz_rpy

BUNDLE_KIND = "g1_calibration_bundle"
BUNDLE_SCHEMA_VERSION = 1
CAMERA_PARENT_FRAME = "torso_link"
CAMERA_CHILD_FRAME = "camera_color_optical_frame"
CAMERA_MOUNT_JOINT = "d435_joint"


def _canonical_sha256(document: Mapping) -> str:
    return hashlib.sha256(
        json.dumps(document, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _matrix(value: object, *, name: str) -> np.ndarray:
    try:
        matrix = validate_transform(np.asarray(value, dtype=np.float64))
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} is not a rigid 4x4 transform") from error
    return matrix


@dataclass(frozen=True, slots=True)
class CalibrationTarget:
    hand_frame: str
    target_frame: str
    hand_T_target: np.ndarray
    target_artifact_sha256: str

    def __post_init__(self) -> None:
        if not self.hand_frame or not self.target_frame:
            raise ValueError("calibration target frames must be non-empty")
        object.__setattr__(
            self,
            "hand_T_target",
            _matrix(self.hand_T_target, name=f"{self.target_frame} transform"),
        )
        if len(self.target_artifact_sha256) != 64:
            raise ValueError("target artifact SHA-256 must contain 64 characters")

    @classmethod
    def from_dict(cls, data: Mapping) -> CalibrationTarget:
        return cls(
            hand_frame=str(data["hand_frame"]),
            target_frame=str(data["target_frame"]),
            hand_T_target=np.asarray(data["hand_T_target"], dtype=np.float64),
            target_artifact_sha256=str(data["target_artifact_sha256"]),
        )

    def to_dict(self) -> dict:
        return {
            "hand_frame": self.hand_frame,
            "target_frame": self.target_frame,
            "hand_T_target": self.hand_T_target.tolist(),
            "target_artifact_sha256": self.target_artifact_sha256,
        }


@dataclass(frozen=True, slots=True)
class CalibrationBundle:
    bundle_id: str
    base_urdf_sha256: str
    torso_T_camera: np.ndarray
    joint_position_offsets_rad: Mapping[str, float]
    targets: Mapping[str, CalibrationTarget]
    provenance: Mapping
    validation: Mapping

    def __post_init__(self) -> None:
        if not self.bundle_id:
            raise ValueError("calibration bundle ID must be non-empty")
        if len(self.base_urdf_sha256) != 64:
            raise ValueError("base URDF SHA-256 must contain 64 characters")
        object.__setattr__(
            self,
            "torso_T_camera",
            _matrix(self.torso_T_camera, name="torso_T_camera"),
        )
        offsets = {
            str(name): float(value) for name, value in self.joint_position_offsets_rad.items()
        }
        unknown = sorted(set(offsets) - set(G1_29_JOINT_NAMES))
        if unknown:
            raise ValueError(
                "calibration bundle contains unknown G1 joints: " + ", ".join(unknown)
            )
        if not all(np.isfinite(value) for value in offsets.values()):
            raise ValueError("joint offsets must be a finite mapping")
        object.__setattr__(self, "joint_position_offsets_rad", offsets)
        targets = {str(side): target for side, target in self.targets.items()}
        if set(targets) != {"left", "right"}:
            raise ValueError("calibration bundle must contain left and right targets")
        object.__setattr__(self, "targets", targets)
        if not isinstance(self.provenance, Mapping) or not self.provenance:
            raise ValueError("calibration bundle provenance must be non-empty")
        if not isinstance(self.validation, Mapping) or not self.validation:
            raise ValueError("calibration bundle validation must be non-empty")

    @classmethod
    def from_dict(cls, data: Mapping, *, verify_hash: bool = True) -> CalibrationBundle:
        required = {
            "schema_version",
            "kind",
            "bundle_id",
            "base_urdf_sha256",
            "camera",
            "joint_position_offsets_rad",
            "targets",
            "provenance",
            "validation",
            "content_sha256",
        }
        if set(data) != required:
            raise ValueError(
                "calibration bundle fields differ: "
                f"missing={sorted(required - set(data))}, "
                f"extra={sorted(set(data) - required)}"
            )
        if data["schema_version"] != BUNDLE_SCHEMA_VERSION:
            raise ValueError("unsupported calibration bundle schema version")
        if data["kind"] != BUNDLE_KIND:
            raise ValueError("document is not a G1 calibration bundle")
        camera = data["camera"]
        if not isinstance(camera, Mapping):
            raise TypeError("calibration bundle camera must be a mapping")
        if set(camera) != {"parent_frame", "child_frame", "parent_T_child"}:
            raise ValueError("calibration bundle camera fields differ")
        if camera["parent_frame"] != CAMERA_PARENT_FRAME:
            raise ValueError(f"camera parent must be {CAMERA_PARENT_FRAME}")
        if camera["child_frame"] != CAMERA_CHILD_FRAME:
            raise ValueError(f"camera child must be {CAMERA_CHILD_FRAME}")
        canonical = dict(data)
        expected = str(canonical.pop("content_sha256"))
        if verify_hash and expected != _canonical_sha256(canonical):
            raise ValueError("calibration bundle content SHA-256 mismatch")
        targets = data["targets"]
        if not isinstance(targets, Mapping):
            raise TypeError("calibration bundle targets must be a mapping")
        return cls(
            bundle_id=str(data["bundle_id"]),
            base_urdf_sha256=str(data["base_urdf_sha256"]),
            torso_T_camera=np.asarray(camera["parent_T_child"], dtype=np.float64),
            joint_position_offsets_rad=data["joint_position_offsets_rad"],
            targets={
                str(side): CalibrationTarget.from_dict(target) for side, target in targets.items()
            },
            provenance=data["provenance"],
            validation=data["validation"],
        )

    @classmethod
    def load(cls, path: str | Path) -> CalibrationBundle:
        source = Path(path)
        with source.open(encoding="utf-8") as stream:
            data = json.load(stream)
        if not isinstance(data, dict):
            raise TypeError("calibration bundle must contain a JSON object")
        return cls.from_dict(data)

    @property
    def content_sha256(self) -> str:
        return _canonical_sha256(self.to_dict(include_hash=False))

    def target(self, side: str) -> CalibrationTarget:
        try:
            return self.targets[side]
        except KeyError as error:
            raise ValueError("calibration target side must be left or right") from error

    def to_dict(self, *, include_hash: bool = True) -> dict:
        document = {
            "schema_version": BUNDLE_SCHEMA_VERSION,
            "kind": BUNDLE_KIND,
            "bundle_id": self.bundle_id,
            "base_urdf_sha256": self.base_urdf_sha256,
            "camera": {
                "parent_frame": CAMERA_PARENT_FRAME,
                "child_frame": CAMERA_CHILD_FRAME,
                "parent_T_child": self.torso_T_camera.tolist(),
            },
            "joint_position_offsets_rad": dict(sorted(self.joint_position_offsets_rad.items())),
            "targets": {side: self.targets[side].to_dict() for side in ("left", "right")},
            "provenance": dict(self.provenance),
            "validation": dict(self.validation),
        }
        if include_hash:
            document["content_sha256"] = _canonical_sha256(document)
        return document

    def materialize_urdf(self, base_urdf: str | Path, output: str | Path) -> Path:
        """Write a calibrated copy while leaving the base URDF untouched."""

        source = Path(base_urdf).resolve()
        destination = Path(output).resolve()
        if destination == source:
            raise ValueError("calibrated URDF output must not replace the base URDF")
        actual_hash = _sha256_file(source)
        if actual_hash != self.base_urdf_sha256:
            raise ValueError(
                "calibration bundle belongs to a different base URDF: "
                f"expected={self.base_urdf_sha256}, actual={actual_hash}"
            )
        root = ET.parse(source).getroot()
        joints = {element.attrib["name"]: element for element in root.findall("joint")}
        for name, offset in self.joint_position_offsets_rad.items():
            element = joints.get(name)
            if element is None:
                raise ValueError(f"base URDF is missing calibrated joint {name}")
            _apply_joint_zero_offset(element, offset)
        camera_joint = joints.get(CAMERA_MOUNT_JOINT)
        if camera_joint is None:
            raise ValueError(f"base URDF is missing {CAMERA_MOUNT_JOINT}")
        parent = camera_joint.find("parent")
        if parent is None or parent.attrib.get("link") != CAMERA_PARENT_FRAME:
            raise ValueError(
                f"{CAMERA_MOUNT_JOINT} must be parented directly to {CAMERA_PARENT_FRAME}"
            )
        torso_T_d435 = self.torso_T_camera @ np.linalg.inv(realsense_link_T_color_optical())
        _set_origin(camera_joint, torso_T_d435)
        _rebase_mesh_paths(
            root,
            source_directory=source.parent,
            output_directory=destination.parent,
        )
        destination.parent.mkdir(parents=True, exist_ok=True)
        fd, temporary_name = tempfile.mkstemp(
            prefix=f".{destination.name}.", dir=destination.parent
        )
        os.close(fd)
        temporary = Path(temporary_name)
        try:
            ET.ElementTree(root).write(temporary, encoding="utf-8", xml_declaration=True)
            os.replace(temporary, destination)
        finally:
            temporary.unlink(missing_ok=True)
        return destination


def _origin_transform(joint: ET.Element) -> np.ndarray:
    origin = joint.find("origin")
    if origin is None:
        return np.eye(4)
    xyz = np.fromstring(origin.attrib.get("xyz", "0 0 0"), sep=" ")
    rpy = np.fromstring(origin.attrib.get("rpy", "0 0 0"), sep=" ")
    if xyz.shape != (3,) or rpy.shape != (3,):
        raise ValueError(f"joint {joint.attrib['name']} has an invalid origin")
    return transform_from_xyz_rpy(xyz, rpy)


def _apply_joint_zero_offset(joint: ET.Element, offset: float) -> None:
    if joint.attrib.get("type") not in {"revolute", "continuous"}:
        raise ValueError(f"calibrated joint {joint.attrib['name']} is not revolute")
    axis_element = joint.find("axis")
    axis = np.fromstring(
        "1 0 0" if axis_element is None else axis_element.attrib.get("xyz", "1 0 0"),
        sep=" ",
    )
    if axis.shape != (3,) or not np.isfinite(axis).all() or np.linalg.norm(axis) <= 0:
        raise ValueError(f"joint {joint.attrib['name']} has an invalid axis")
    axis = axis / np.linalg.norm(axis)
    delta = np.eye(4)
    delta[:3, :3] = Rotation.from_rotvec(axis * offset).as_matrix()
    _set_origin(joint, _origin_transform(joint) @ delta)


def _set_origin(joint: ET.Element, transform: np.ndarray) -> None:
    matrix = validate_transform(transform)
    origin = joint.find("origin")
    if origin is None:
        origin = ET.SubElement(joint, "origin")
    rpy = Rotation.from_matrix(matrix[:3, :3].copy()).as_euler("xyz")
    origin.set("xyz", " ".join(f"{value:.17g}" for value in matrix[:3, 3]))
    origin.set("rpy", " ".join(f"{value:.17g}" for value in rpy))


def _rebase_mesh_paths(
    root: ET.Element, *, source_directory: Path, output_directory: Path
) -> None:
    """Keep relative mesh references valid when the overlay is written elsewhere."""

    for mesh in root.findall(".//mesh"):
        filename = mesh.attrib.get("filename")
        if not filename or filename.startswith("package://") or Path(filename).is_absolute():
            continue
        source_mesh = (source_directory / filename).resolve()
        mesh.set("filename", os.path.relpath(source_mesh, output_directory))
