"""Direct two-cube stack geometry; this is intentionally not a task language."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
from scipy.spatial.transform import Rotation

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


def _tuple_transform(value: np.ndarray) -> tuple[tuple[float, ...], ...]:
    transform = validate_transform(np.asarray(value, dtype=np.float64))
    return tuple(tuple(float(item) for item in row) for row in transform)


def _base_poses(
    *,
    upper_cube: TabletopObservation,
    bottom_cube: TabletopObservation,
    base_T_camera: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    camera = validate_transform(np.asarray(base_T_camera, dtype=np.float64))
    raw_upper = camera @ np.asarray(upper_cube.camera_T_object, dtype=np.float64)
    raw_bottom = camera @ np.asarray(bottom_cube.camera_T_object, dtype=np.float64)
    return (
        raw_upper,
        raw_bottom,
        canonical_resting_cube_pose(raw_upper),
        canonical_resting_cube_pose(raw_bottom),
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


def build_direct_stack_request(
    *,
    moving_request: TabletopTaskRequest,
    support_cube: TabletopObservation,
    base_T_camera: np.ndarray,
    yaw_quarter_turns: int = 0,
    excluded_candidate_ids: tuple[str, ...] = (),
) -> TabletopPickPlaceRequest:
    """Place one observed 60 mm cube directly on the other observed cube.

    The quarter-turn is a nominal hand-path choice among the cube's four
    upright symmetries.  It is not evidence that the physical cube retains an
    exact yaw inside the Dex3 grasp.
    """

    if tuple(moving_request.object_dimensions_m) != (0.060, 0.060, 0.060):
        raise ValueError("the moving direct-stack cube must use a 60 mm profile")
    if yaw_quarter_turns not in (0, 1, 2, 3):
        raise ValueError("direct-stack yaw quarter-turns must be 0, 1, 2, or 3")
    raw_moving, raw_support, canonical_moving, canonical_support = _base_poses(
        upper_cube=moving_request.observation,
        bottom_cube=support_cube,
        base_T_camera=base_T_camera,
    )
    normal = canonical_moving[:3, 2] + canonical_support[:3, 2]
    normal_norm = float(np.linalg.norm(normal))
    if normal_norm <= 1.0e-9:
        raise ValueError("the two observed cubes imply opposite tabletop normals")
    up = normal / normal_norm
    destination = np.eye(4, dtype=np.float64)
    destination[:3, :3] = (
        Rotation.from_rotvec(yaw_quarter_turns * 0.5 * np.pi * up).as_matrix() @ raw_moving[:3, :3]
    )
    destination[:3, 3] = raw_support[:3, 3] + 0.060 * up
    moving_T_support = invert_transform(raw_moving) @ raw_support
    source = replace(
        moving_request,
        environment_cuboids=(
            TabletopCuboid(
                object_id="support_cube",
                object_T_cuboid=_tuple_transform(moving_T_support),
                dimensions_m=(0.060, 0.060, 0.060),
            ),
        ),
    )
    return TabletopPickPlaceRequest(
        source_request=source,
        source_T_destination_object=_tuple_transform(invert_transform(raw_moving) @ destination),
        destination_support_object_id="support_cube",
        excluded_candidate_ids=excluded_candidate_ids,
    )
