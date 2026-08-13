"""Selected-pair FCL collision and clearance checks."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import trimesh
import yaml
from trimesh.collision import CollisionManager

from g1_aprilcube_calibration.urdf_model import (
    LinkGeometry,
    URDFModel,
    transform_from_xyz_rpy,
)


@dataclass(frozen=True, slots=True)
class CollisionPair:
    first: str
    second: str

    def __post_init__(self) -> None:
        if not self.first or not self.second or self.first == self.second:
            raise ValueError("collision pair must contain two distinct link names")


@dataclass(frozen=True, slots=True)
class AttachedBox:
    name: str
    parent_link: str
    size_m: tuple[float, float, float]
    xyz_m: tuple[float, float, float]
    rpy_rad: tuple[float, float, float]

    def __post_init__(self) -> None:
        if not self.name or not self.parent_link:
            raise ValueError("attached collision object requires a name and parent link")
        if len(self.size_m) != 3 or any(value <= 0 for value in self.size_m):
            raise ValueError("attached box size must contain three positive values")
        if len(self.xyz_m) != 3 or len(self.rpy_rad) != 3:
            raise ValueError("attached box pose must contain three xyz and rpy values")
        if not np.all(np.isfinite((*self.size_m, *self.xyz_m, *self.rpy_rad))):
            raise ValueError("attached box geometry must be finite")


@dataclass(frozen=True, slots=True)
class CollisionConfig:
    pairs: tuple[CollisionPair, ...]
    visual_fallback_links: tuple[str, ...] = ()
    attached_boxes: tuple[AttachedBox, ...] = ()
    required_attached_boxes: tuple[str, ...] = ()
    hardware_ready: bool = False
    blocking_reasons: tuple[str, ...] = ()
    schema_version: int = 1

    def __post_init__(self) -> None:
        names = [item.name for item in self.attached_boxes]
        if len(names) != len(set(names)):
            raise ValueError("collision configuration contains duplicate attachments")
        if self.hardware_ready and self.blocking_reasons:
            raise ValueError("hardware-ready collision config cannot have blocking reasons")
        if not self.hardware_ready and not self.blocking_reasons:
            raise ValueError("non-ready collision config must explain what is blocking")
        missing = sorted(set(self.required_attached_boxes) - set(names))
        if missing:
            raise ValueError(
                "collision configuration is missing required attachments: " + ", ".join(missing)
            )
        paired_names = {link for pair in self.pairs for link in (pair.first, pair.second)}
        unpaired = sorted(set(self.required_attached_boxes) - paired_names)
        if unpaired:
            raise ValueError(
                "required collision attachments are not used by any pair: " + ", ".join(unpaired)
            )
        if self.hardware_ready and self.required_attached_boxes and not self.attached_boxes:
            raise ValueError("hardware-ready collision config has no attached geometry")

    @classmethod
    def from_yaml(cls, path: str | Path) -> CollisionConfig:
        with Path(path).open(encoding="utf-8") as stream:
            data = yaml.safe_load(stream)
        return cls.from_mapping(data)

    @classmethod
    def from_mapping(cls, data: Mapping) -> CollisionConfig:
        if not isinstance(data, dict) or data.get("schema_version") != 1:
            raise ValueError("unsupported collision configuration")
        if not isinstance(data.get("hardware_ready", False), bool):
            raise TypeError("collision hardware_ready must be boolean")
        pairs = tuple(CollisionPair(*item) for item in data["pairs"])
        if len({tuple(sorted((item.first, item.second))) for item in pairs}) != len(pairs):
            raise ValueError("collision configuration contains duplicate pairs")
        return cls(
            pairs=pairs,
            visual_fallback_links=tuple(data.get("visual_fallback_links", [])),
            attached_boxes=tuple(
                AttachedBox(
                    name=item["name"],
                    parent_link=item["parent_link"],
                    size_m=tuple(item["size_m"]),
                    xyz_m=tuple(item["xyz_m"]),
                    rpy_rad=tuple(item.get("rpy_rad", [0.0, 0.0, 0.0])),
                )
                for item in data.get("attached_boxes", [])
            ),
            required_attached_boxes=tuple(data.get("required_attached_boxes", [])),
            hardware_ready=data.get("hardware_ready", False),
            blocking_reasons=tuple(data.get("blocking_reasons", [])),
        )

    @property
    def content_sha256(self) -> str:
        data = {
            "schema_version": self.schema_version,
            "pairs": [[item.first, item.second] for item in self.pairs],
            "visual_fallback_links": list(self.visual_fallback_links),
            "attached_boxes": [
                {
                    "name": item.name,
                    "parent_link": item.parent_link,
                    "size_m": list(item.size_m),
                    "xyz_m": list(item.xyz_m),
                    "rpy_rad": list(item.rpy_rad),
                }
                for item in self.attached_boxes
            ],
            "required_attached_boxes": list(self.required_attached_boxes),
            "hardware_ready": self.hardware_ready,
            "blocking_reasons": list(self.blocking_reasons),
        }
        return hashlib.sha256(
            json.dumps(data, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()


@dataclass(frozen=True, slots=True)
class CollisionResult:
    minimum_clearance_m: float
    minimum_pair: CollisionPair | None
    colliding_pairs: tuple[CollisionPair, ...]


class FCLCollisionChecker:
    def __init__(self, model: URDFModel, config: CollisionConfig) -> None:
        self.model = model
        self.config = config
        needed_links = {link for pair in config.pairs for link in (pair.first, pair.second)}
        attachments = {item.name: item for item in config.attached_boxes}
        self._managers: dict[str, CollisionManager] = {}
        self._geometries: dict[str, tuple[LinkGeometry, ...]] = {}
        self._parents: dict[str, str] = {}
        for link in needed_links:
            if link in attachments:
                attached = attachments[link]
                if attached.parent_link not in model.links:
                    raise ValueError(
                        f"attachment parent is not a URDF link: {attached.parent_link}"
                    )
                geometries = (
                    LinkGeometry(
                        link_name=attached.name,
                        local_transform=transform_from_xyz_rpy(
                            np.asarray(attached.xyz_m), np.asarray(attached.rpy_rad)
                        ),
                        mesh=trimesh.creation.box(extents=attached.size_m),
                        source="attached_box",
                    ),
                )
                self._parents[link] = attached.parent_link
            else:
                geometries = model.link_geometries(
                    link, visual_fallback=link in config.visual_fallback_links
                )
                self._parents[link] = link
            if not geometries:
                raise ValueError(f"selected collision link has no geometry: {link}")
            manager = CollisionManager()
            for index, geometry in enumerate(geometries):
                manager.add_object(f"{link}:{index}", geometry.mesh)
            self._managers[link] = manager
            self._geometries[link] = geometries

    def check(self, transforms: dict[str, np.ndarray]) -> CollisionResult:
        for link, geometries in self._geometries.items():
            parent = self._parents[link]
            if parent not in transforms:
                raise ValueError(f"FK result is missing collision link: {parent}")
            for index, geometry in enumerate(geometries):
                self._managers[link].set_transform(
                    f"{link}:{index}", transforms[parent] @ geometry.local_transform
                )
        minimum = float("inf")
        minimum_pair = None
        colliding: list[CollisionPair] = []
        for pair in self.config.pairs:
            distance = float(
                self._managers[pair.first].min_distance_other(self._managers[pair.second])
            )
            if distance < minimum:
                minimum = distance
                minimum_pair = pair
            if distance <= 0:
                colliding.append(pair)
        return CollisionResult(minimum, minimum_pair, tuple(colliding))
