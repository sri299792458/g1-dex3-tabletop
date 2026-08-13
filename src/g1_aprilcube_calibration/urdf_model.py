"""Minimal strict URDF kinematics and collision-geometry loader."""

from __future__ import annotations

import hashlib
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import trimesh
from scipy.spatial.transform import Rotation


def _vector(text: str | None, *, size: int, default: tuple[float, ...]) -> np.ndarray:
    if text is None:
        values = np.asarray(default, dtype=np.float64)
    else:
        values = np.fromstring(text, sep=" ", dtype=np.float64)
    if values.shape != (size,) or not np.all(np.isfinite(values)):
        raise ValueError(f"URDF vector must contain {size} finite values: {text!r}")
    return values


def transform_from_xyz_rpy(xyz: np.ndarray, rpy: np.ndarray) -> np.ndarray:
    result = np.eye(4, dtype=np.float64)
    result[:3, :3] = Rotation.from_euler("xyz", rpy).as_matrix()
    result[:3, 3] = xyz
    return result


def _origin(element: ET.Element | None) -> np.ndarray:
    if element is None:
        return np.eye(4)
    return transform_from_xyz_rpy(
        _vector(element.get("xyz"), size=3, default=(0.0, 0.0, 0.0)),
        _vector(element.get("rpy"), size=3, default=(0.0, 0.0, 0.0)),
    )


@dataclass(frozen=True, slots=True)
class JointLimit:
    lower: float
    upper: float
    velocity: float | None


@dataclass(frozen=True, slots=True)
class URDFJoint:
    name: str
    joint_type: str
    parent: str
    child: str
    origin: np.ndarray
    axis: np.ndarray
    limit: JointLimit | None

    def transform(self, position: float) -> np.ndarray:
        if not np.isfinite(position):
            raise ValueError(f"joint {self.name} position is not finite")
        motion = np.eye(4)
        if self.joint_type in {"revolute", "continuous"}:
            motion[:3, :3] = Rotation.from_rotvec(self.axis * position).as_matrix()
        elif self.joint_type == "prismatic":
            motion[:3, 3] = self.axis * position
        elif self.joint_type != "fixed":
            raise ValueError(f"unsupported URDF joint type: {self.joint_type}")
        return self.origin @ motion


@dataclass(frozen=True, slots=True)
class LinkGeometry:
    link_name: str
    local_transform: np.ndarray
    mesh: trimesh.Trimesh
    source: str


class URDFModel:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path).resolve()
        self.sha256 = hashlib.sha256(self.path.read_bytes()).hexdigest()
        root = ET.parse(self.path).getroot()
        if root.tag != "robot":
            raise ValueError("URDF root must be <robot>")
        self.name = root.get("name", "")
        self.links = tuple(link.attrib["name"] for link in root.findall("link"))
        self._link_elements = {link.attrib["name"]: link for link in root.findall("link")}
        joints: dict[str, URDFJoint] = {}
        child_joint: dict[str, URDFJoint] = {}
        for element in root.findall("joint"):
            name = element.attrib["name"]
            joint_type = element.attrib["type"]
            parent = element.find("parent")
            child = element.find("child")
            if parent is None or child is None:
                raise ValueError(f"joint {name} is missing parent or child")
            axis = _vector(
                None if element.find("axis") is None else element.find("axis").get("xyz"),
                size=3,
                default=(1.0, 0.0, 0.0),
            )
            norm = float(np.linalg.norm(axis))
            if norm <= 0:
                raise ValueError(f"joint {name} has a zero axis")
            axis = axis / norm
            limit_element = element.find("limit")
            limit = None
            if joint_type in {"revolute", "prismatic"}:
                if limit_element is None:
                    raise ValueError(f"joint {name} is missing limits")
                limit = JointLimit(
                    lower=float(limit_element.attrib["lower"]),
                    upper=float(limit_element.attrib["upper"]),
                    velocity=(
                        None
                        if "velocity" not in limit_element.attrib
                        else float(limit_element.attrib["velocity"])
                    ),
                )
            joint = URDFJoint(
                name=name,
                joint_type=joint_type,
                parent=parent.attrib["link"],
                child=child.attrib["link"],
                origin=_origin(element.find("origin")),
                axis=axis,
                limit=limit,
            )
            if name in joints or joint.child in child_joint:
                raise ValueError("URDF contains duplicate joint or child ownership")
            joints[name] = joint
            child_joint[joint.child] = joint
        self.joints = joints
        self._child_joint = child_joint
        child_links = set(child_joint)
        roots = sorted(set(self.links) - child_links)
        if len(roots) != 1:
            raise ValueError(f"URDF must have exactly one root link, found {roots}")
        self.root_link = roots[0]

    def chain(self, root_link: str, target_link: str) -> tuple[URDFJoint, ...]:
        if root_link not in self.links or target_link not in self.links:
            raise ValueError("kinematic chain references an unknown link")
        reverse: list[URDFJoint] = []
        current = target_link
        while current != root_link:
            joint = self._child_joint.get(current)
            if joint is None:
                raise ValueError(f"{root_link} is not an ancestor of {target_link}")
            reverse.append(joint)
            current = joint.parent
        return tuple(reversed(reverse))

    def transform(
        self,
        root_link: str,
        target_link: str,
        positions: dict[str, float],
    ) -> np.ndarray:
        result = np.eye(4)
        for joint in self.chain(root_link, target_link):
            value = 0.0 if joint.joint_type == "fixed" else positions.get(joint.name)
            if value is None:
                raise ValueError(f"missing position for movable joint {joint.name}")
            result = result @ joint.transform(float(value))
        return result

    def forward_kinematics(
        self,
        positions: dict[str, float],
        *,
        root_link: str | None = None,
    ) -> dict[str, np.ndarray]:
        root_link = root_link or self.root_link
        transforms = {root_link: np.eye(4)}
        pending = list(self.joints.values())
        while pending:
            progress = False
            for joint in pending.copy():
                if joint.parent not in transforms:
                    continue
                value = 0.0 if joint.joint_type == "fixed" else positions.get(joint.name)
                if value is None:
                    raise ValueError(f"missing position for movable joint {joint.name}")
                transforms[joint.child] = transforms[joint.parent] @ joint.transform(float(value))
                pending.remove(joint)
                progress = True
            if not progress:
                break
        return transforms

    def joint_limits(self, joint_names: tuple[str, ...]) -> tuple[JointLimit, ...]:
        result: list[JointLimit] = []
        for name in joint_names:
            joint = self.joints.get(name)
            if joint is None or joint.limit is None:
                raise ValueError(f"joint has no finite URDF limit: {name}")
            result.append(joint.limit)
        return tuple(result)

    def link_geometries(
        self,
        link_name: str,
        *,
        visual_fallback: bool = False,
    ) -> tuple[LinkGeometry, ...]:
        element = self._link_elements.get(link_name)
        if element is None:
            raise ValueError(f"unknown URDF link: {link_name}")
        geometries = element.findall("collision")
        source = "collision"
        if not geometries and visual_fallback:
            geometries = element.findall("visual")
            source = "visual_fallback"
        return tuple(
            self._load_geometry(link_name, geometry, source=source) for geometry in geometries
        )

    def _load_geometry(self, link_name: str, element: ET.Element, *, source: str) -> LinkGeometry:
        geometry = element.find("geometry")
        if geometry is None or len(geometry) != 1:
            raise ValueError(f"{link_name} has invalid {source} geometry")
        shape = geometry[0]
        if shape.tag == "mesh":
            filename = shape.attrib["filename"]
            if filename.startswith("package://"):
                raise ValueError("package:// mesh paths require an explicit resolver")
            mesh_path = (self.path.parent / filename).resolve()
            loaded = trimesh.load_mesh(mesh_path, process=False)
            if isinstance(loaded, trimesh.Scene):
                mesh = trimesh.util.concatenate(tuple(loaded.geometry.values()))
            else:
                mesh = loaded
            scale = _vector(shape.get("scale"), size=3, default=(1.0, 1.0, 1.0))
            mesh = mesh.copy()
            mesh.apply_scale(scale)
        elif shape.tag == "box":
            mesh = trimesh.creation.box(
                extents=_vector(shape.get("size"), size=3, default=(0.0, 0.0, 0.0))
            )
        elif shape.tag == "sphere":
            mesh = trimesh.creation.icosphere(radius=float(shape.attrib["radius"]))
        elif shape.tag == "cylinder":
            mesh = trimesh.creation.cylinder(
                radius=float(shape.attrib["radius"]),
                height=float(shape.attrib["length"]),
                sections=24,
            )
        else:
            raise ValueError(f"unsupported URDF geometry type: {shape.tag}")
        if not isinstance(mesh, trimesh.Trimesh) or mesh.is_empty:
            raise ValueError(f"failed to load {source} geometry for {link_name}")
        return LinkGeometry(
            link_name=link_name,
            local_transform=_origin(element.find("origin")),
            mesh=mesh,
            source=source,
        )
