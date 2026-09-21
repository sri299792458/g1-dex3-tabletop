"""Deterministic conversion of verified bilateral evidence into solver samples."""

from __future__ import annotations

import hashlib
from dataclasses import replace
from typing import Literal

from aprilcube import CorrespondenceResult

from g1_aprilcube_calibration.correspondence import correspondence_sha256
from g1_dex3_tabletop.calibration.capture import BilateralFrameEvidence
from g1_dex3_tabletop.calibration.models import (
    BilateralCalibrationDataset,
    BilateralCalibrationSample,
    TargetObservation,
)


def target_observation_from_correspondences(
    result: CorrespondenceResult,
    *,
    side: Literal["left", "right"],
    target_artifact_sha256: str,
) -> TargetObservation:
    """Flatten one re-detected target while preserving canonical corner order."""

    if not result.valid:
        raise ValueError(f"{side} target correspondences are not valid")
    image_points: list[tuple[float, float]] = []
    object_points: list[tuple[float, float, float]] = []
    corner_tag_ids: list[int] = []
    for observation in result.observations:
        for image_point, object_point_mm in zip(
            observation.image_corners_px,
            observation.object_corners_mm,
            strict=True,
        ):
            corner_tag_ids.append(observation.tag_id)
            image_points.append(tuple(float(value) for value in image_point))
            object_points.append(tuple(float(value) / 1000.0 for value in object_point_mm))
    return TargetObservation(
        side=side,
        target_artifact_sha256=target_artifact_sha256,
        visible_tag_ids=result.tag_ids,
        corner_tag_ids=tuple(corner_tag_ids),
        image_points_px=tuple(image_points),
        object_points_m=tuple(object_points),
        correspondence_sha256=correspondence_sha256(result),
    )


def sample_from_frame_evidence(
    frame: BilateralFrameEvidence,
    *,
    source_session_id: str,
    capture_id: str,
    pose_group_id: str,
    day_group_id: str,
    capture_role: Literal["anchor", "excitation"],
    raw_image_path: str,
    raw_image_sha256: str,
    left_target_artifact_sha256: str,
    right_target_artifact_sha256: str,
) -> BilateralCalibrationSample:
    """Build one same-frame sample after raw-image re-detection has passed."""

    paired_state = frame.pairing.nearest
    return BilateralCalibrationSample(
        source_session_id=source_session_id,
        capture_id=capture_id,
        pose_group_id=pose_group_id,
        day_group_id=day_group_id,
        frame_id=frame.frame_id,
        capture_role=capture_role,
        raw_image_path=raw_image_path,
        raw_image_sha256=raw_image_sha256,
        camera_info=frame.camera_info.to_dict(),
        joint_positions_rad=tuple(float(value) for value in paired_state.position),
        joint_velocities_rad_s=tuple(float(value) for value in paired_state.velocity),
        pairing=frame.pairing.to_dict(),
        left=target_observation_from_correspondences(
            frame.left_correspondences,
            side="left",
            target_artifact_sha256=left_target_artifact_sha256,
        ),
        right=target_observation_from_correspondences(
            frame.right_correspondences,
            side="right",
            target_artifact_sha256=right_target_artifact_sha256,
        ),
    )


def merge_bilateral_datasets(
    datasets: tuple[BilateralCalibrationDataset, ...],
    *,
    dataset_id: str | None = None,
) -> BilateralCalibrationDataset:
    """Merge compatible source sessions without erasing their manifest identities."""

    values = tuple(datasets)
    if len(values) < 2:
        raise ValueError("bilateral merge requires at least two datasets")
    reference = values[0]
    compatibility_fields = (
        "pose_design_sha256",
        "execution_plan_sha256",
        "urdf_sha256",
        "rgb_optical_transform_sha256",
        "left_target_artifact_sha256",
        "right_target_artifact_sha256",
    )
    for dataset in values[1:]:
        differing = [
            name
            for name in compatibility_fields
            if getattr(dataset, name) != getattr(reference, name)
        ]
        if differing:
            raise ValueError(
                "bilateral datasets are not source-compatible: " + ", ".join(differing)
            )
    source_sessions: dict[str, str] = {}
    for dataset in values:
        for session_id, manifest_hash in dataset.session_manifest_sha256_by_id.items():
            existing = source_sessions.get(session_id)
            if existing is not None and existing != manifest_hash:
                raise ValueError(
                    f"bilateral source session {session_id} has conflicting manifests"
                )
            if existing is not None:
                raise ValueError(f"bilateral source session is repeated: {session_id}")
            source_sessions[session_id] = manifest_hash
    content_seed = ":".join(dataset.content_sha256 for dataset in values)
    resolved_id = (
        dataset_id or f"bilateral_merge_{hashlib.sha256(content_seed.encode()).hexdigest()[:16]}"
    )
    return replace(
        reference,
        dataset_id=resolved_id,
        session_manifest_sha256_by_id=source_sessions,
        samples=tuple(sample for dataset in values for sample in dataset.samples),
        provenance={
            "merge_policy": ("strict_same_design_execution_urdf_camera_and_targets"),
            "source_dataset_sha256": [dataset.content_sha256 for dataset in values],
            "source_provenance": [dataset.provenance for dataset in values],
        },
    )
