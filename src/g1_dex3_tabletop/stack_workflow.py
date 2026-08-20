"""Fixed two-cube stack geometry; this is intentionally not a task language."""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from g1_aprilcube_calibration.joint_map import G1_29_JOINT_NAMES
from g1_aprilcube_calibration.transforms import invert_transform, validate_transform
from g1_aprilcube_calibration.urdf_model import URDFModel
from g1_dex3_tabletop.tabletop_contracts import (
    TabletopCuboid,
    TabletopObservation,
    TabletopPickPlaceRequest,
    TabletopTaskRequest,
)
from g1_dex3_tabletop.tabletop_geometry import canonical_resting_cube_pose


@dataclass(frozen=True, slots=True)
class StackPlacementCandidate:
    """One concrete 60 mm table placement and resulting 40 mm stack pose."""

    candidate_id: str
    cube60_T_destination: tuple[tuple[float, ...], ...]
    cube40_T_placed_cube60: tuple[tuple[float, ...], ...]
    cube40_T_stack_destination: tuple[tuple[float, ...], ...]
    cube60_displacement_m: float
    placed_center_separation_m: float
    table_plane_disagreement_mm: float
    table_normal_disagreement_deg: float

    def to_dict(self) -> dict[str, object]:
        return {
            "candidate_id": self.candidate_id,
            "cube60_T_destination": [list(row) for row in self.cube60_T_destination],
            "cube40_T_placed_cube60": [list(row) for row in self.cube40_T_placed_cube60],
            "cube40_T_stack_destination": [list(row) for row in self.cube40_T_stack_destination],
            "cube60_displacement_m": self.cube60_displacement_m,
            "placed_center_separation_m": self.placed_center_separation_m,
            "table_plane_disagreement_mm": self.table_plane_disagreement_mm,
            "table_normal_disagreement_deg": self.table_normal_disagreement_deg,
        }


def _tuple_transform(value: np.ndarray) -> tuple[tuple[float, ...], ...]:
    transform = validate_transform(np.asarray(value, dtype=np.float64))
    return tuple(tuple(float(item) for item in row) for row in transform)


def _base_poses(
    *,
    cube40: TabletopObservation,
    cube60: TabletopObservation,
    base_T_camera: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    camera = validate_transform(np.asarray(base_T_camera, dtype=np.float64))
    raw40 = camera @ np.asarray(cube40.camera_T_object, dtype=np.float64)
    raw60 = camera @ np.asarray(cube60.camera_T_object, dtype=np.float64)
    return (
        raw40,
        raw60,
        canonical_resting_cube_pose(raw40),
        canonical_resting_cube_pose(raw60),
    )


def request_base_T_camera(
    request: TabletopTaskRequest,
    model: URDFModel,
) -> np.ndarray:
    """Resolve the calibrated camera pose from the request's measured body state."""

    positions = {
        name: float(value) + request.joint_position_offsets_rad.get(name, 0.0)
        for name, value in zip(
            G1_29_JOINT_NAMES,
            request.planning_snapshot.measured_q29_rad,
            strict=True,
        )
    }
    return validate_transform(
        model.transform("pelvis", "torso_link", positions)
        @ np.asarray(request.torso_T_camera, dtype=np.float64)
    )


def request_hand_positions(
    request: TabletopTaskRequest,
    model: URDFModel,
) -> tuple[np.ndarray, np.ndarray]:
    """Return live left/right rubber-hand origins for assignment ordering."""

    positions = {
        name: float(value) + request.joint_position_offsets_rad.get(name, 0.0)
        for name, value in zip(
            G1_29_JOINT_NAMES,
            request.planning_snapshot.measured_q29_rad,
            strict=True,
        )
    }
    return (
        model.transform("pelvis", "left_rubber_hand", positions)[:3, 3],
        model.transform("pelvis", "right_rubber_hand", positions)[:3, 3],
    )


def stack_placement_candidates(
    *,
    cube40: TabletopObservation,
    cube60: TabletopObservation,
    base_T_camera: np.ndarray,
    count: int = 5,
    object_clearance_m: float = 0.005,
) -> tuple[StackPlacementCandidate, ...]:
    """Sample the known-on-table segment; full CuRobo feasibility chooses one.

    Both observed cube centers prove that the segment between them lies on the
    same ordinary convex tabletop.  No fabricated table boundary or preferred
    midpoint is introduced.  The finite samples span the non-overlapping part
    of that segment and are only proposals for complete two-stage planning.
    """

    if count < 1:
        raise ValueError("stack placement candidate count must be positive")
    if not np.isfinite(object_clearance_m) or object_clearance_m < 0.0:
        raise ValueError("stack object clearance must be finite and non-negative")
    raw40, raw60, canonical40, canonical60 = _base_poses(
        cube40=cube40,
        cube60=cube60,
        base_T_camera=base_T_camera,
    )
    up40 = canonical40[:3, 2]
    up60 = canonical60[:3, 2]
    normal_dot = float(np.clip(np.dot(up40, up60), -1.0, 1.0))
    normal_disagreement_deg = float(np.rad2deg(np.arccos(normal_dot)))
    up = up40 + up60
    up_norm = float(np.linalg.norm(up))
    if up_norm <= 1.0e-9:
        raise ValueError("the two resting cubes imply opposite tabletop normals")
    up /= up_norm
    plane40 = canonical40[:3, 3] - 0.020 * up40
    plane60 = canonical60[:3, 3] - 0.030 * up60
    plane_disagreement_mm = 1000.0 * abs(float(np.dot(plane60 - plane40, up)))

    center_delta = raw60[:3, 3] - raw40[:3, 3]
    planar_delta = center_delta - np.dot(center_delta, up) * up
    separation = float(np.linalg.norm(planar_delta))
    if separation <= 1.0e-9:
        raise ValueError("the two cube centers do not define a tabletop placement segment")
    direction_40_to_60 = planar_delta / separation
    # Circumscribed footprint radii are orientation-independent.  CuRobo's
    # exact finite cuboids still make the final decision.
    minimum_separation = (
        0.5 * np.hypot(0.040, 0.040) + 0.5 * np.hypot(0.060, 0.060) + object_clearance_m
    )
    if separation <= minimum_separation:
        raise ValueError(
            "the observed cube centers leave no non-overlapping placement segment: "
            f"separation={separation:.4f}m, required>{minimum_separation:.4f}m"
        )
    # Exclude both endpoints: every candidate genuinely repositions the 60 mm
    # cube and remains strictly separated from the stationary 40 mm cube.
    distances_from_40 = np.linspace(minimum_separation, separation, count + 2)[1:-1]
    symmetry40 = invert_transform(raw40) @ canonical40
    candidates = []
    for index, distance_from_40 in enumerate(distances_from_40):
        placed60 = raw60.copy()
        placed60[:3, 3] = plane40 + direction_40_to_60 * distance_from_40 + 0.030 * up
        placed_canonical60 = placed60 @ (invert_transform(raw60) @ canonical60)

        destination40 = np.eye(4, dtype=np.float64)
        destination40[:3, :3] = placed_canonical60[:3, :3] @ symmetry40[:3, :3].T
        destination40[:3, 3] = placed60[:3, 3] + 0.050 * up
        candidate_id = f"segment_{index + 1:02d}_of_{count:02d}"
        candidates.append(
            StackPlacementCandidate(
                candidate_id=candidate_id,
                cube60_T_destination=_tuple_transform(invert_transform(raw60) @ placed60),
                cube40_T_placed_cube60=_tuple_transform(invert_transform(raw40) @ placed60),
                cube40_T_stack_destination=_tuple_transform(
                    invert_transform(raw40) @ destination40
                ),
                cube60_displacement_m=float(np.linalg.norm(placed60[:3, 3] - raw60[:3, 3])),
                placed_center_separation_m=float(distance_from_40),
                table_plane_disagreement_mm=plane_disagreement_mm,
                table_normal_disagreement_deg=normal_disagreement_deg,
            )
        )
    return tuple(candidates)


def stack_arm_assignments(
    *,
    cube40: TabletopObservation,
    cube60: TabletopObservation,
    base_T_camera: np.ndarray,
    base_left_hand_position: np.ndarray,
    base_right_hand_position: np.ndarray,
) -> tuple[tuple[str, str], ...]:
    """Order the two possible (60-arm, 40-arm) assignments by live proximity."""

    camera = validate_transform(np.asarray(base_T_camera, dtype=np.float64))
    center40 = (camera @ np.asarray(cube40.camera_T_object))[:3, 3]
    center60 = (camera @ np.asarray(cube60.camera_T_object))[:3, 3]
    left = np.asarray(base_left_hand_position, dtype=np.float64).reshape(3)
    right = np.asarray(base_right_hand_position, dtype=np.float64).reshape(3)
    assignments = (("left", "right"), ("right", "left"))
    scores = {
        ("left", "right"): float(
            np.linalg.norm(left - center60) + np.linalg.norm(right - center40)
        ),
        ("right", "left"): float(
            np.linalg.norm(right - center60) + np.linalg.norm(left - center40)
        ),
    }
    return tuple(sorted(assignments, key=lambda value: (scores[value], value)))


def build_stack_stage_requests(
    *,
    cube40_request: TabletopTaskRequest,
    cube60_request: TabletopTaskRequest,
    candidate: StackPlacementCandidate,
) -> tuple[TabletopPickPlaceRequest, TabletopPickPlaceRequest]:
    """Build the fixed 60-to-table then 40-on-60 plan requests."""

    if cube40_request.arm == cube60_request.arm:
        raise ValueError("the fixed stack workflow requires one different arm per cube")
    if tuple(cube40_request.object_dimensions_m) != (0.040, 0.040, 0.040):
        raise ValueError("the small-cube stack request must use the 40 mm profile")
    if tuple(cube60_request.object_dimensions_m) != (0.060, 0.060, 0.060):
        raise ValueError("the support-cube stack request must use the 60 mm profile")
    cube60_T_cube40 = invert_transform(
        np.asarray(cube60_request.observation.camera_T_object)
    ) @ np.asarray(cube40_request.observation.camera_T_object)
    stage1_source = replace(
        cube60_request,
        environment_cuboids=(
            TabletopCuboid(
                object_id="cube40",
                object_T_cuboid=_tuple_transform(cube60_T_cube40),
                dimensions_m=cube40_request.object_dimensions_m,
            ),
        ),
    )
    stage1 = TabletopPickPlaceRequest(
        source_request=stage1_source,
        source_T_destination_object=candidate.cube60_T_destination,
    )
    stage2_source = replace(
        cube40_request,
        environment_cuboids=(
            TabletopCuboid(
                object_id="cube60",
                object_T_cuboid=candidate.cube40_T_placed_cube60,
                dimensions_m=cube60_request.object_dimensions_m,
            ),
        ),
    )
    stage2 = TabletopPickPlaceRequest(
        source_request=stage2_source,
        source_T_destination_object=candidate.cube40_T_stack_destination,
        destination_support_object_id="cube60",
    )
    return stage1, stage2


def build_observed_stack_second_stage_request(
    *,
    cube40_request: TabletopTaskRequest,
    placed_cube60: TabletopObservation,
    base_T_camera: np.ndarray,
) -> TabletopPickPlaceRequest:
    """Build the final 40-on-60 transfer from the observed stage-one result."""

    if tuple(cube40_request.object_dimensions_m) != (0.040, 0.040, 0.040):
        raise ValueError("the small-cube stack request must use the 40 mm profile")
    raw40, raw60, canonical40, canonical60 = _base_poses(
        cube40=cube40_request.observation,
        cube60=placed_cube60,
        base_T_camera=base_T_camera,
    )
    normal = canonical40[:3, 2] + canonical60[:3, 2]
    normal_norm = float(np.linalg.norm(normal))
    if normal_norm <= 1.0e-9:
        raise ValueError("the two observed cubes imply opposite tabletop normals")
    up = normal / normal_norm
    symmetry40 = invert_transform(raw40) @ canonical40
    destination40 = np.eye(4, dtype=np.float64)
    destination40[:3, :3] = canonical60[:3, :3] @ symmetry40[:3, :3].T
    destination40[:3, 3] = raw60[:3, 3] + 0.050 * up
    cube40_T_cube60 = invert_transform(raw40) @ raw60
    source = replace(
        cube40_request,
        environment_cuboids=(
            TabletopCuboid(
                object_id="cube60",
                object_T_cuboid=_tuple_transform(cube40_T_cube60),
                dimensions_m=(0.060, 0.060, 0.060),
            ),
        ),
    )
    return TabletopPickPlaceRequest(
        source_request=source,
        source_T_destination_object=_tuple_transform(invert_transform(raw40) @ destination40),
        destination_support_object_id="cube60",
    )
