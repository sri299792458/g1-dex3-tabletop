"""AprilCube-only tabletop observation used immediately before task planning."""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from itertools import combinations

import numpy as np
from aprilcube import CorrespondenceDetector
from scipy.spatial.transform import Rotation

from g1_aprilcube_calibration.camera_models import RectifiedCameraInfo
from g1_aprilcube_calibration.table_accuracy import detect_hand_target_pose
from g1_aprilcube_calibration.transforms import invert_transform, validate_transform
from g1_dex3_tabletop.planning.contracts import RobotSnapshot
from g1_dex3_tabletop.tabletop_contracts import TabletopObservation

_MAXIMUM_RESTING_FACE_TILT_DEG = 20.0


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


def _transform_spread(transforms: Sequence[np.ndarray]) -> tuple[np.ndarray, float, float]:
    center = _average(transforms)
    translation_mm = max(
        1000.0 * float(np.linalg.norm(item[:3, 3] - center[:3, 3])) for item in transforms
    )
    rotation_deg = max(_rotation_error_deg(center, item) for item in transforms)
    return center, translation_mm, rotation_deg


def _largest_transform_consensus(
    transforms: Sequence[np.ndarray],
    *,
    minimum_frames: int,
    maximum_translation_spread_mm: float,
    maximum_rotation_spread_deg: float,
) -> tuple[tuple[int, ...], np.ndarray, float, float]:
    """Select the largest pose-consistent set grown from every minimum-size seed."""

    passing: list[tuple[int, float, tuple[int, ...], np.ndarray, float, float]] = []
    best_seed: tuple[float, float, float] | None = None
    for seed in combinations(range(len(transforms)), minimum_frames):
        center, translation_mm, rotation_deg = _transform_spread(
            [transforms[index] for index in seed]
        )
        score = max(
            translation_mm / maximum_translation_spread_mm,
            rotation_deg / maximum_rotation_spread_deg,
        )
        seed_diagnostic = (score, translation_mm, rotation_deg)
        if best_seed is None or seed_diagnostic < best_seed:
            best_seed = seed_diagnostic
        if score > 1.0:
            continue

        indices = list(seed)
        remaining = [index for index in range(len(transforms)) if index not in seed]
        while remaining:
            additions = []
            for index in remaining:
                candidate_indices = tuple(sorted((*indices, index)))
                candidate_center, candidate_translation, candidate_rotation = _transform_spread(
                    [transforms[item] for item in candidate_indices]
                )
                candidate_score = max(
                    candidate_translation / maximum_translation_spread_mm,
                    candidate_rotation / maximum_rotation_spread_deg,
                )
                if candidate_score <= 1.0:
                    additions.append(
                        (
                            candidate_score,
                            index,
                            candidate_indices,
                            candidate_center,
                            candidate_translation,
                            candidate_rotation,
                        )
                    )
            if not additions:
                break
            _score, added, candidate_indices, center, translation_mm, rotation_deg = min(
                additions, key=lambda item: item[:2]
            )
            indices = list(candidate_indices)
            remaining.remove(added)
        final_indices = tuple(indices)
        center, translation_mm, rotation_deg = _transform_spread(
            [transforms[index] for index in final_indices]
        )
        final_score = max(
            translation_mm / maximum_translation_spread_mm,
            rotation_deg / maximum_rotation_spread_deg,
        )
        passing.append(
            (
                -len(final_indices),
                final_score,
                final_indices,
                center,
                translation_mm,
                rotation_deg,
            )
        )

    if not passing:
        assert best_seed is not None
        _score, translation_mm, rotation_deg = best_seed
        raise ValueError(
            f"no {minimum_frames}-frame cube pose consensus passed: best translation spread "
            f"is {translation_mm:.3f}mm (limit {maximum_translation_spread_mm:.3f}mm), "
            f"best rotation spread is {rotation_deg:.3f}deg "
            f"(limit {maximum_rotation_spread_deg:.3f}deg)"
        )

    _count, _score, indices, center, translation_mm, rotation_deg = min(
        passing, key=lambda item: item[:3]
    )
    return indices, center, translation_mm, rotation_deg


def _resting_face_tilt_deg(
    camera_T_object: np.ndarray,
    *,
    base_T_camera: np.ndarray,
) -> float:
    base_T_object = base_T_camera @ camera_T_object
    best_alignment = float(np.max(np.abs(base_T_object[:3, :3][2, :])))
    return float(np.degrees(np.arccos(np.clip(best_alignment, -1.0, 1.0))))


def _detect_resting_camera_T_object(
    image_bgr: np.ndarray,
    *,
    camera_info: RectifiedCameraInfo,
    detector: CorrespondenceDetector,
    base_T_camera: np.ndarray,
    minimum_tag_short_side_px: float,
    maximum_reprojection_error_px: float,
) -> np.ndarray:
    failures = []
    for label, single_best_face in (("single-face", True), ("multi-face", False)):
        try:
            estimate = detect_hand_target_pose(
                image_bgr,
                camera_info,
                detector,
                target_label="tabletop AprilCube",
                minimum_visible_faces=1,
                minimum_tag_short_side_px=minimum_tag_short_side_px,
                maximum_reprojection_error_px=maximum_reprojection_error_px,
                single_best_face=single_best_face,
            )
        except ValueError as error:
            failures.append(f"{label} solve failed: {error}")
            continue
        tilt = _resting_face_tilt_deg(
            estimate.camera_T_target,
            base_T_camera=base_T_camera,
        )
        if tilt <= _MAXIMUM_RESTING_FACE_TILT_DEG:
            return estimate.camera_T_target
        failures.append(f"{label} resting tilt is {tilt:.3f} degrees")
    raise ValueError(
        "; ".join(failures)
        + f"; limit is {_MAXIMUM_RESTING_FACE_TILT_DEG:.0f} degrees"
    )


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
    base_T_camera: np.ndarray,
    minimum_frames: int = 3,
    maximum_translation_spread_mm: float = 5.0,
    maximum_rotation_spread_deg: float = 2.0,
    minimum_tag_short_side_px: float = 25.0,
    maximum_reprojection_error_px: float = 3.0,
) -> TabletopObservation:
    """Estimate one robust cube pose; its bottom face becomes the table plane."""

    if len(images_bgr) < minimum_frames:
        raise ValueError(f"need at least {minimum_frames} cube frames")
    base_T_camera = validate_transform(np.asarray(base_T_camera, dtype=np.float64))
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
            transform = _detect_resting_camera_T_object(
                value,
                camera_info=camera_info,
                detector=detector,
                base_T_camera=base_T_camera,
                minimum_tag_short_side_px=minimum_tag_short_side_px,
                maximum_reprojection_error_px=maximum_reprojection_error_px,
            )
        except ValueError as error:
            rejections.append(f"frame {index}: {error}")
            continue
        transforms.append(transform)
        hashes.append(digest)
    if len(transforms) < minimum_frames:
        raise ValueError(
            f"only {len(transforms)}/{len(images_bgr)} cube frames passed; "
            + "; ".join(rejections)
        )
    indices, center, translation_spread, rotation_spread = _largest_transform_consensus(
        transforms,
        minimum_frames=minimum_frames,
        maximum_translation_spread_mm=maximum_translation_spread_mm,
        maximum_rotation_spread_deg=maximum_rotation_spread_deg,
    )
    hashes = [hashes[index] for index in indices]
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
    base_T_camera: np.ndarray,
    minimum_frames: int = 3,
    maximum_translation_spread_mm: float = 5.0,
    maximum_rotation_spread_deg: float = 2.0,
    minimum_tag_short_side_px: float = 25.0,
    maximum_reprojection_error_px: float = 3.0,
    first_label: str = "first cube",
    second_label: str = "second cube",
) -> tuple[TabletopObservation, TabletopObservation]:
    """Observe both uniquely tagged cubes from the same accepted image frames."""

    arguments = {
        "camera_info": camera_info,
        "snapshot": snapshot,
        "base_T_camera": base_T_camera,
        "minimum_frames": minimum_frames,
        "maximum_translation_spread_mm": maximum_translation_spread_mm,
        "maximum_rotation_spread_deg": maximum_rotation_spread_deg,
        "minimum_tag_short_side_px": minimum_tag_short_side_px,
        "maximum_reprojection_error_px": maximum_reprojection_error_px,
    }

    def observe_labeled(images, detector, label: str) -> TabletopObservation:
        try:
            return observe_resting_cube(images, detector=detector, **arguments)
        except ValueError as error:
            raise ValueError(f"{label}: {error}") from error

    first = observe_labeled(images_bgr, first_detector, first_label)
    second = observe_labeled(images_bgr, second_detector, second_label)
    common = set(first.source_frame_sha256)
    common.intersection_update(second.source_frame_sha256)
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
    return (
        observe_labeled(paired_images, first_detector, first_label),
        observe_labeled(paired_images, second_detector, second_label),
    )
