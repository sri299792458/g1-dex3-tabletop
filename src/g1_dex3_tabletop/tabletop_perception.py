"""AprilCube-only tabletop observation used immediately before task planning."""

from __future__ import annotations

import hashlib
import heapq
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from itertools import combinations

import cv2
import numpy as np
from aprilcube import (
    CorrespondenceDetector,
    CorrespondenceResult,
    estimate_pose_hypotheses,
)
from scipy.spatial.transform import Rotation

from g1_aprilcube_calibration.camera_models import RectifiedCameraInfo
from g1_aprilcube_calibration.table_accuracy import detect_hand_target_pose
from g1_aprilcube_calibration.transforms import invert_transform, validate_transform
from g1_dex3_tabletop.planning.contracts import RobotSnapshot
from g1_dex3_tabletop.tabletop_contracts import TabletopObservation

_MAXIMUM_RESTING_FACE_TILT_DEG = 20.0


@dataclass(frozen=True, slots=True)
class _RestingPoseHypothesis:
    camera_T_object: np.ndarray
    reprojection_error_px: float
    source_tag_ids: tuple[int, ...]


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


def _hypothesis_products_by_error(
    candidate_sets: Sequence[Sequence[_RestingPoseHypothesis]],
) -> Iterator[tuple[_RestingPoseHypothesis, ...]]:
    """Yield the Cartesian product in increasing total reprojection error."""

    ordered = [
        tuple(sorted(items, key=lambda item: item.reprojection_error_px))
        for items in candidate_sets
    ]
    start = (0,) * len(ordered)

    def error(indices: tuple[int, ...]) -> float:
        return float(
            sum(
                ordered[axis][index].reprojection_error_px
                for axis, index in enumerate(indices)
            )
        )

    queue = [(error(start), start)]
    visited = {start}
    while queue:
        _error, indices = heapq.heappop(queue)
        yield tuple(ordered[axis][index] for axis, index in enumerate(indices))
        for axis, index in enumerate(indices):
            next_index = index + 1
            if next_index >= len(ordered[axis]):
                continue
            neighbor = (*indices[:axis], next_index, *indices[axis + 1 :])
            if neighbor in visited:
                continue
            visited.add(neighbor)
            heapq.heappush(queue, (error(neighbor), neighbor))


def _largest_hypothesis_consensus(
    hypotheses_by_frame: Sequence[Sequence[_RestingPoseHypothesis]],
    *,
    base_T_camera: np.ndarray,
    minimum_frames: int,
    maximum_translation_spread_mm: float,
    maximum_rotation_spread_deg: float,
) -> tuple[tuple[int, ...], np.ndarray, float, float]:
    """Select one ordered hypothesis per frame in the largest consistent subset."""

    best_seed: tuple[float, float, float] | None = None
    frame_range = range(len(hypotheses_by_frame))
    for frame_count in range(len(hypotheses_by_frame), minimum_frames - 1, -1):
        passing = []
        for frame_indices in combinations(frame_range, frame_count):
            candidate_sets = [hypotheses_by_frame[index] for index in frame_indices]
            for selected in _hypothesis_products_by_error(candidate_sets):
                transforms = [item.camera_T_object for item in selected]
                # A set within a radius limit cannot contain a pair separated
                # by more than twice that limit.  This inexpensive necessary
                # check removes most opposing planar branches before averaging.
                pairwise_valid = all(
                    1000.0
                    * float(
                        np.linalg.norm(
                            transforms[first][:3, 3] - transforms[second][:3, 3]
                        )
                    )
                    <= 2.0 * maximum_translation_spread_mm
                    and _rotation_error_deg(transforms[first], transforms[second])
                    <= 2.0 * maximum_rotation_spread_deg
                    for first, second in combinations(range(len(transforms)), 2)
                )
                if not pairwise_valid:
                    if frame_count == minimum_frames:
                        _, translation_mm, rotation_deg = _transform_spread(transforms)
                        score = max(
                            translation_mm / maximum_translation_spread_mm,
                            rotation_deg / maximum_rotation_spread_deg,
                        )
                        diagnostic = (score, translation_mm, rotation_deg)
                        if best_seed is None or diagnostic < best_seed:
                            best_seed = diagnostic
                    continue
                center, translation_mm, rotation_deg = _transform_spread(transforms)
                score = max(
                    translation_mm / maximum_translation_spread_mm,
                    rotation_deg / maximum_rotation_spread_deg,
                )
                diagnostic = (score, translation_mm, rotation_deg)
                if frame_count == minimum_frames and (
                    best_seed is None or diagnostic < best_seed
                ):
                    best_seed = diagnostic
                if score <= 1.0:
                    center_tilt = _resting_face_tilt_deg(
                        center,
                        base_T_camera=base_T_camera,
                    )
                    if center_tilt > _MAXIMUM_RESTING_FACE_TILT_DEG:
                        continue
                    # AprilCube orders each frame's candidates by reprojection
                    # error. The first passing product is therefore the
                    # lowest-total-error assignment for this frame subset.
                    passing.append(
                        (
                            float(
                                np.mean(
                                    [item.reprojection_error_px for item in selected]
                                )
                            ),
                            score,
                            frame_indices,
                            center,
                            translation_mm,
                            rotation_deg,
                        )
                    )
                    break
        if passing:
            (
                _error,
                _score,
                frame_indices,
                center,
                translation_mm,
                rotation_deg,
            ) = min(passing, key=lambda item: item[:3])
            return frame_indices, center, translation_mm, rotation_deg

    if best_seed is None:
        raise ValueError("no cube pose hypotheses were available for consensus")
    _score, translation_mm, rotation_deg = best_seed
    raise ValueError(
        f"no {minimum_frames}-frame cube pose consensus passed: best translation spread "
        f"is {translation_mm:.3f}mm (limit {maximum_translation_spread_mm:.3f}mm), "
        f"best rotation spread is {rotation_deg:.3f}deg "
        f"(limit {maximum_rotation_spread_deg:.3f}deg)"
    )


def _resting_face_tilt_deg(
    camera_T_object: np.ndarray,
    *,
    base_T_camera: np.ndarray,
) -> float:
    base_T_object = base_T_camera @ camera_T_object
    best_alignment = float(np.max(np.abs(base_T_object[:3, :3][2, :])))
    return float(np.degrees(np.arccos(np.clip(best_alignment, -1.0, 1.0))))


def _detect_resting_pose_hypotheses(
    image_bgr: np.ndarray,
    *,
    camera_info: RectifiedCameraInfo,
    detector: CorrespondenceDetector,
    base_T_camera: np.ndarray,
    minimum_tag_short_side_px: float,
    maximum_reprojection_error_px: float,
) -> tuple[_RestingPoseHypothesis, ...]:
    result: CorrespondenceResult = detector.detect(image_bgr)
    if not result.valid:
        if result.duplicate_tag_ids:
            raise ValueError(
                "tabletop AprilCube has duplicate tag IDs: "
                f"{list(result.duplicate_tag_ids)}"
            )
        raise ValueError("tabletop AprilCube was not detected")
    observations = tuple(
        item
        for item in result.observations
        if item.shortest_side_px >= minimum_tag_short_side_px
    )
    if not observations:
        largest = max(item.shortest_side_px for item in result.observations)
        raise ValueError(
            f"tabletop AprilCube marker is only {largest:.1f}px; need "
            f"{minimum_tag_short_side_px:.1f}px"
        )
    pose_result = CorrespondenceResult(
        image_size_wh=result.image_size_wh,
        observations=observations,
        ignored_tag_ids=result.ignored_tag_ids,
        opencv_rejected_candidates=result.opencv_rejected_candidates,
        quality_rejected_detections=result.quality_rejected_detections,
    )
    diagnostics = estimate_pose_hypotheses(
        pose_result,
        camera_info.rectified_camera_matrix,
        np.asarray(camera_info.d, dtype=np.float64),
    )
    accepted = []
    best_reprojection = float("inf")
    best_tilt = float("inf")
    for diagnostic in diagnostics:
        best_reprojection = min(
            best_reprojection,
            diagnostic.reprojection_error_px,
        )
        if diagnostic.reprojection_error_px > maximum_reprojection_error_px:
            continue
        transform = np.eye(4, dtype=np.float64)
        transform[:3, :3], _ = cv2.Rodrigues(diagnostic.rvec)
        transform[:3, 3] = diagnostic.tvec_mm.reshape(3) / 1000.0
        tilt = _resting_face_tilt_deg(
            transform,
            base_T_camera=base_T_camera,
        )
        best_tilt = min(best_tilt, tilt)
        if tilt <= _MAXIMUM_RESTING_FACE_TILT_DEG:
            accepted.append(
                _RestingPoseHypothesis(
                    camera_T_object=transform,
                    reprojection_error_px=diagnostic.reprojection_error_px,
                    source_tag_ids=diagnostic.source_tag_ids,
                )
            )
    if accepted:
        return tuple(accepted)
    reprojection_text = (
        "n/a" if not np.isfinite(best_reprojection) else f"{best_reprojection:.3f}px"
    )
    tilt_text = "n/a" if not np.isfinite(best_tilt) else f"{best_tilt:.3f} degrees"
    raise ValueError(
        f"no resting pose hypothesis passed: {len(diagnostics)} candidates; "
        f"best reprojection error {reprojection_text} "
        f"(limit {maximum_reprojection_error_px:.3f}px); best resting tilt "
        f"{tilt_text} (limit {_MAXIMUM_RESTING_FACE_TILT_DEG:.0f} degrees)"
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
    hypotheses_by_frame: list[tuple[_RestingPoseHypothesis, ...]] = []
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
            hypotheses = _detect_resting_pose_hypotheses(
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
        hypotheses_by_frame.append(hypotheses)
        hashes.append(digest)
    if len(hypotheses_by_frame) < minimum_frames:
        raise ValueError(
            f"only {len(hypotheses_by_frame)}/{len(images_bgr)} cube frames passed; "
            + "; ".join(rejections)
        )
    indices, center, translation_spread, rotation_spread = _largest_hypothesis_consensus(
        hypotheses_by_frame,
        base_T_camera=base_T_camera,
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
