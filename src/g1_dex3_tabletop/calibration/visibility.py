"""Camera sightlines through the existing detailed arm and Dex3 meshes."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from g1_aprilcube_calibration.joint_map import G1_29_JOINT_NAMES, arm_hand_link, arm_joint_names
from g1_aprilcube_calibration.transforms import transform_points
from g1_aprilcube_calibration.transports.unitree_dex3 import DEX3_MOTOR_JOINT_SUFFIXES
from g1_aprilcube_calibration.urdf_model import URDFModel
from g1_dex3_tabletop.planning.g1_model import (
    CUROBO_G1_URDF_RELATIVE,
    MOUNT_MANIFESTS,
    curobo_checkout_root,
)


@dataclass
class _MeshSegments:
    """Cache triangle data; test finite segments without an extra ray-library dependency."""

    triangles: np.ndarray

    def __post_init__(self) -> None:
        self.triangles = np.asarray(self.triangles, dtype=np.float64)
        self.lower = self.triangles.min(axis=1)
        self.upper = self.triangles.max(axis=1)
        self.bounds = np.array([self.lower.min(axis=0), self.upper.max(axis=0)])
        self.first = self.triangles[:, 0]
        self.edge1 = self.triangles[:, 1] - self.first
        self.edge2 = self.triangles[:, 2] - self.first

    def blocks(self, origin: np.ndarray, endpoints: np.ndarray, *, epsilon_m: float) -> bool:
        directions = endpoints - origin
        lengths = np.linalg.norm(directions, axis=1)
        if np.any(lengths <= 2 * epsilon_m):
            raise ValueError("marker sightline is too short")
        # Slab test against the entire mesh before touching its triangles.
        parallel = np.abs(directions) < 1e-14
        outside = parallel & ((origin < self.bounds[0]) | (origin > self.bounds[1]))
        first = np.divide(
            self.bounds[0] - origin,
            directions,
            out=np.full_like(directions, -np.inf),
            where=~parallel,
        )
        second = np.divide(
            self.bounds[1] - origin,
            directions,
            out=np.full_like(directions, np.inf),
            where=~parallel,
        )
        enter = np.maximum(np.minimum(first, second).max(axis=1), 0.0)
        leave = np.minimum(np.maximum(first, second).min(axis=1), 1.0)
        possible = ~outside.any(axis=1) & (enter <= leave)
        for endpoint, direction, length in zip(
            endpoints[possible], directions[possible], lengths[possible], strict=True
        ):
            lower, upper = np.minimum(origin, endpoint), np.maximum(origin, endpoint)
            mask = np.all(self.upper >= lower, axis=1) & np.all(self.lower <= upper, axis=1)
            if not np.any(mask):
                continue
            e1, e2 = self.edge1[mask], self.edge2[mask]
            h = np.cross(direction, e2)
            determinant = np.einsum("ij,ij->i", e1, h)
            nonparallel = np.abs(determinant) > 1e-14
            inverse = np.divide(
                1.0, determinant, out=np.zeros_like(determinant), where=nonparallel
            )
            offset = origin - self.first[mask]
            u = inverse * np.einsum("ij,ij->i", offset, h)
            cross = np.cross(offset, e1)
            v = inverse * (cross @ direction)
            t = inverse * np.einsum("ij,ij->i", e2, cross)
            if np.any(
                nonparallel
                & (u >= -1e-9)
                & (v >= -1e-9)
                & (u + v <= 1.0 + 1e-9)
                & (t > epsilon_m / length)
                & (t < 1.0 - epsilon_m / length)
            ):
                return True
        return False


def _marker_grid(corners: np.ndarray, *, size: int = 5) -> np.ndarray:
    points = []
    for quad in np.asarray(corners).reshape(-1, 4, 3):
        for u in np.linspace(0.0, 1.0, size):
            for v in np.linspace(0.0, 1.0, size):
                points.append(
                    (1 - u) * (1 - v) * quad[0]
                    + u * (1 - v) * quad[1]
                    + u * v * quad[2]
                    + (1 - u) * v * quad[3]
                )
    return np.asarray(points)


class BilateralSightlineChecker:
    """Reject occluded required markers at held poses; motion collision checks are separate."""

    def __init__(
        self,
        request,
        projection_model: URDFModel,
        *,
        geometry_model: URDFModel | None = None,
        physical_hand_T_targets=None,
    ) -> None:
        self.request = request
        self.projection_model = projection_model
        self.geometry_model = geometry_model or URDFModel(
            curobo_checkout_root() / CUROBO_G1_URDF_RELATIVE
        )
        self.origin = np.asarray(request.nominal_torso_T_camera)[:3, 3]
        if physical_hand_T_targets is None:
            repository = Path(__file__).resolve().parents[3]
            physical_hand_T_targets = {
                side: json.loads((repository / path).read_text())["nominal_palm_T_marker_face_m"]
                for side, path in MOUNT_MANIFESTS.items()
            }
        # Fitted target transforms are effective calibration parameters and can
        # place the optical face inside the CAD palm. Use the physical mount CAD
        # with the body meshes for obstruction, keeping fitted targets for pixels.
        self.physical_hand_T_targets = {
            side: np.asarray(value) for side, value in physical_hand_T_targets.items()
        }
        self.epsilon_m = 0.0001
        self.meshes = []
        hashes = hashlib.sha256()
        for side in ("left", "right"):
            # Both models must describe the same arm before applying fitted offsets.
            for name in arm_joint_names(side):
                first, second = projection_model.joints[name], self.geometry_model.joints[name]
                if not np.allclose(
                    first.origin, second.origin, atol=1e-10, rtol=0.0
                ) or not np.allclose(first.axis, second.axis, atol=1e-10, rtol=0.0):
                    raise ValueError(f"sightline and projection arm geometry differ at {name}")
            for link in self.geometry_model.links:
                if not link.startswith(f"{side}_"):
                    continue
                chain = (
                    self.geometry_model.chain("torso_link", link)
                    if any(key in link for key in ("shoulder", "elbow", "wrist", "hand"))
                    else ()
                )
                if not any(joint.name == arm_joint_names(side)[0] for joint in chain):
                    continue
                for geometry in self.geometry_model.link_geometries(link, prefer_visual=True):
                    triangles = np.asarray(geometry.mesh.triangles)
                    self.meshes.append((link, geometry.local_transform, _MeshSegments(triangles)))
                    hashes.update(link.encode())
                    hashes.update(geometry.local_transform.tobytes())
                    hashes.update(triangles.tobytes())
        if not self.meshes:
            raise ValueError("no arm/hand geometry is available for sightline checks")
        self.provenance = {
            "method": "finite_camera_to_marker_segments_against_visual_meshes",
            "geometry_urdf_sha256": self.geometry_model.sha256,
            "mesh_set_sha256": hashes.hexdigest(),
            "checked_links": sorted({link for link, _, _ in self.meshes}),
            "grid_samples_per_tag": 25,
            "endpoint_epsilon_m": self.epsilon_m,
            "physical_hand_T_targets": {
                side: value.tolist() for side, value in self.physical_hand_T_targets.items()
            },
            "limitations": "nominal geometry; cables and unsampled small occlusions require live detection",
        }

    def check(self, full_q, *, required_sides: tuple[str, ...]) -> None:
        positions = {
            name: float(value) + self.request.joint_position_offsets_rad.get(name, 0.0)
            for name, value in zip(G1_29_JOINT_NAMES, full_q, strict=True)
        }
        for side in ("left", "right"):
            positions.update(
                {
                    f"{side}_hand_{suffix}_joint": float(value)
                    for suffix, value in zip(
                        DEX3_MOTOR_JOINT_SUFFIXES[side],
                        self.request.dex3_model_positions_rad[side],
                        strict=True,
                    )
                }
            )
        transforms = self.geometry_model.forward_kinematics(positions, root_link="torso_link")
        target_corners = {}
        for side in ("left", "right"):
            torso_T_target = (
                self.projection_model.transform("torso_link", arm_hand_link(side), positions)
                @ self.physical_hand_T_targets[side]
            )
            target_corners[side] = transform_points(
                torso_T_target, np.asarray(self.request.target_object_points_m_by_arm[side])
            )
        for side in required_sides:
            endpoints = _marker_grid(target_corners[side])
            for link, link_T_mesh, mesh in self.meshes:
                torso_T_mesh = transforms[link] @ link_T_mesh
                rotation = torso_T_mesh[:3, :3]
                local_origin = (self.origin - torso_T_mesh[:3, 3]) @ rotation
                local_endpoints = (endpoints - torso_T_mesh[:3, 3]) @ rotation
                if mesh.blocks(local_origin, local_endpoints, epsilon_m=self.epsilon_m):
                    raise ValueError(f"{side} marker camera sightline is blocked by {link}")
            # The other optical marker is also opaque, even though its carrier
            # is an attachment outside the manufacturer URDF.
            other = "right" if side == "left" else "left"
            quads = target_corners[other].reshape(-1, 4, 3)
            triangles = np.concatenate((quads[:, [0, 1, 2]], quads[:, [0, 2, 3]]))
            if _MeshSegments(triangles).blocks(self.origin, endpoints, epsilon_m=self.epsilon_m):
                raise ValueError(f"{side} marker camera sightline is blocked by {other} marker")
