"""AprilCube-only tabletop observation used immediately before task planning."""

from __future__ import annotations

import hashlib
from collections.abc import Sequence

import numpy as np
from aprilcube import CorrespondenceDetector
from scipy.spatial.transform import Rotation

from g1_aprilcube_calibration.camera_models import RectifiedCameraInfo
from g1_aprilcube_calibration.table_accuracy import detect_hand_target_pose
from g1_dex3_tabletop.planning.contracts import RobotSnapshot
from g1_dex3_tabletop.tabletop_contracts import TabletopObservation


def _average(transforms: Sequence[np.ndarray]) -> np.ndarray:
    result = np.eye(4, dtype=np.float64)
    result[:3, 3] = np.mean([item[:3, 3] for item in transforms], axis=0)
    result[:3, :3] = (
        Rotation.from_matrix(np.stack([item[:3, :3] for item in transforms])).mean().as_matrix()
    )
    return result


def _rotation_error_deg(first: np.ndarray, second: np.ndarray) -> float:
    relative = first[:3, :3].T @ second[:3, :3]
    return float(np.rad2deg(Rotation.from_matrix(relative).magnitude()))


def observe_resting_cube(
    images_bgr: Sequence[np.ndarray],
    *,
    camera_info: RectifiedCameraInfo,
    detector: CorrespondenceDetector,
    snapshot: RobotSnapshot,
    minimum_frames: int = 3,
    maximum_translation_spread_mm: float = 5.0,
    maximum_rotation_spread_deg: float = 2.0,
) -> TabletopObservation:
    """Estimate one robust cube pose; its bottom face becomes the table plane."""

    if len(images_bgr) < minimum_frames:
        raise ValueError(f"need at least {minimum_frames} cube frames")
    transforms: list[np.ndarray] = []
    hashes: list[str] = []
    rejections: list[str] = []
    for index, image in enumerate(images_bgr):
        value = np.asarray(image)
        if value.ndim != 3 or value.shape[2] != 3 or value.dtype != np.uint8:
            rejections.append(f"frame {index}: image must be uint8 BGR")
            continue
        digest = hashlib.sha256(value.tobytes()).hexdigest()
        if digest in hashes:
            rejections.append(f"frame {index}: duplicate image")
            continue
        try:
            estimate = detect_hand_target_pose(
                value,
                camera_info,
                detector,
                minimum_visible_faces=1,
                minimum_tag_short_side_px=30.0,
                maximum_reprojection_error_px=1.5,
            )
        except ValueError as error:
            rejections.append(f"frame {index}: {error}")
            continue
        transforms.append(estimate.camera_T_target)
        hashes.append(digest)
    if len(transforms) < minimum_frames:
        raise ValueError(
            f"only {len(transforms)}/{len(images_bgr)} cube frames passed; "
            + "; ".join(rejections)
        )
    center = _average(transforms)
    translation_spread = max(
        1000.0 * float(np.linalg.norm(item[:3, 3] - center[:3, 3])) for item in transforms
    )
    rotation_spread = max(_rotation_error_deg(center, item) for item in transforms)
    if translation_spread > maximum_translation_spread_mm:
        raise ValueError(
            f"cube translation spread is {translation_spread:.3f}mm; limit is "
            f"{maximum_translation_spread_mm:.3f}mm"
        )
    if rotation_spread > maximum_rotation_spread_deg:
        raise ValueError(
            f"cube rotation spread is {rotation_spread:.3f}deg; limit is "
            f"{maximum_rotation_spread_deg:.3f}deg"
        )
    return TabletopObservation(
        snapshot=snapshot,
        camera_T_object=tuple(tuple(float(value) for value in row) for row in center),
        camera_profile_sha256=camera_info.profile_sha256,
        source_frame_sha256=tuple(hashes),
        object_translation_spread_mm=translation_spread,
        object_rotation_spread_deg=rotation_spread,
    )
