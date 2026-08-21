"""AprilCube-only tabletop observation used immediately before task planning."""

from __future__ import annotations

import hashlib
from collections.abc import Sequence

import numpy as np
from aprilcube import CorrespondenceDetector
from scipy.spatial.transform import Rotation

from g1_aprilcube_calibration.camera_models import RectifiedCameraInfo
from g1_aprilcube_calibration.table_accuracy import detect_hand_target_pose
from g1_aprilcube_calibration.transforms import invert_transform, validate_transform
from g1_dex3_tabletop.planning.contracts import RobotSnapshot
from g1_dex3_tabletop.tabletop_contracts import TabletopObservation


def observe_live_cube_frame(
    image_bgr: np.ndarray,
    *,
    camera_info: RectifiedCameraInfo,
    detector: CorrespondenceDetector,
    minimum_tag_short_side_px: float = 25.0,
    maximum_reprojection_error_px: float = 3.0,
) -> dict[str, object]:
    """Decode one fresh cube frame for reactive control.

    Stationary planning continues to use a multi-frame aggregate.  A reactive
    target must instead retain the timestamp and pose of one actual frame;
    averaging a burst would deliberately lag a moving object.
    """

    value = np.asarray(image_bgr)
    if value.ndim != 3 or value.shape[2] != 3 or value.dtype != np.uint8:
        raise ValueError("image must be uint8 BGR")
    estimate = detect_hand_target_pose(
        value,
        camera_info,
        detector,
        target_label="live tabletop AprilCube",
        minimum_visible_faces=1,
        minimum_tag_short_side_px=minimum_tag_short_side_px,
        maximum_reprojection_error_px=maximum_reprojection_error_px,
        single_best_face=True,
    )
    return {
        "camera_T_object": estimate.camera_T_target.tolist(),
        "source_frame_sha256": hashlib.sha256(value.tobytes()).hexdigest(),
        "pose_evidence": estimate.to_dict(),
    }


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


def camera_motion_from_fixed_cube(
    reference_camera_T_object,
    current_camera_T_object,
) -> dict[str, object]:
    """Measure camera motion in the AprilCube frame, assuming the cube stayed fixed."""

    reference_object_T_camera = invert_transform(
        validate_transform(np.asarray(reference_camera_T_object, dtype=np.float64))
    )
    current_object_T_camera = invert_transform(
        validate_transform(np.asarray(current_camera_T_object, dtype=np.float64))
    )
    translation_mm = 1000.0 * (current_object_T_camera[:3, 3] - reference_object_T_camera[:3, 3])
    return {
        "translation_object_xyz_mm": translation_mm.tolist(),
        "translation_norm_mm": float(np.linalg.norm(translation_mm)),
        "rotation_deg": _rotation_error_deg(
            reference_object_T_camera,
            current_object_T_camera,
        ),
        "anchor": "fixed_tabletop_aprilcube",
        "sign_convention": "current_camera_minus_reference_camera_in_fixed_object_frame",
    }


def observe_resting_cube(
    images_bgr: Sequence[np.ndarray],
    *,
    camera_info: RectifiedCameraInfo,
    detector: CorrespondenceDetector,
    snapshot: RobotSnapshot,
    minimum_frames: int = 3,
    maximum_translation_spread_mm: float = 5.0,
    maximum_rotation_spread_deg: float = 2.0,
    minimum_tag_short_side_px: float = 25.0,
    maximum_reprojection_error_px: float = 3.0,
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
                target_label="tabletop AprilCube",
                minimum_visible_faces=1,
                minimum_tag_short_side_px=minimum_tag_short_side_px,
                maximum_reprojection_error_px=maximum_reprojection_error_px,
                single_best_face=True,
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


def observe_resting_cube_pair(
    images_bgr: Sequence[np.ndarray],
    *,
    camera_info: RectifiedCameraInfo,
    first_detector: CorrespondenceDetector,
    second_detector: CorrespondenceDetector,
    snapshot: RobotSnapshot,
    minimum_frames: int = 3,
    maximum_translation_spread_mm: float = 5.0,
    maximum_rotation_spread_deg: float = 2.0,
    minimum_tag_short_side_px: float = 25.0,
    maximum_reprojection_error_px: float = 3.0,
) -> tuple[TabletopObservation, TabletopObservation]:
    """Observe both uniquely tagged cubes from the same accepted image frames."""

    common = set(
        observe_resting_cube(
            images_bgr,
            camera_info=camera_info,
            detector=first_detector,
            snapshot=snapshot,
            minimum_frames=minimum_frames,
            maximum_translation_spread_mm=maximum_translation_spread_mm,
            maximum_rotation_spread_deg=maximum_rotation_spread_deg,
            minimum_tag_short_side_px=minimum_tag_short_side_px,
            maximum_reprojection_error_px=maximum_reprojection_error_px,
        ).source_frame_sha256
    )
    common.intersection_update(
        observe_resting_cube(
            images_bgr,
            camera_info=camera_info,
            detector=second_detector,
            snapshot=snapshot,
            minimum_frames=minimum_frames,
            maximum_translation_spread_mm=maximum_translation_spread_mm,
            maximum_rotation_spread_deg=maximum_rotation_spread_deg,
            minimum_tag_short_side_px=minimum_tag_short_side_px,
            maximum_reprojection_error_px=maximum_reprojection_error_px,
        ).source_frame_sha256
    )
    paired_images = tuple(
        image
        for image in images_bgr
        if hashlib.sha256(np.asarray(image).tobytes()).hexdigest() in common
    )
    if len(paired_images) < minimum_frames:
        raise ValueError(
            "the two cubes were not jointly detected in enough frames: "
            f"{len(paired_images)}/{len(images_bgr)} common, need {minimum_frames}"
        )
    arguments = {
        "camera_info": camera_info,
        "snapshot": snapshot,
        "minimum_frames": minimum_frames,
        "maximum_translation_spread_mm": maximum_translation_spread_mm,
        "maximum_rotation_spread_deg": maximum_rotation_spread_deg,
        "minimum_tag_short_side_px": minimum_tag_short_side_px,
        "maximum_reprojection_error_px": maximum_reprojection_error_px,
    }
    return (
        observe_resting_cube(
            paired_images,
            detector=first_detector,
            **arguments,
        ),
        observe_resting_cube(
            paired_images,
            detector=second_detector,
            **arguments,
        ),
    )
