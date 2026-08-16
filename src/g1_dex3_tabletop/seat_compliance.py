"""Pure ChArUco observation and request construction for the chair A/B test."""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation

from g1_aprilcube_calibration.calibration_bundle import CalibrationBundle
from g1_aprilcube_calibration.camera_models import RectifiedCameraInfo
from g1_aprilcube_calibration.joint_map import validate_arm_side
from g1_aprilcube_calibration.table_accuracy import (
    CharucoBoardPoseDetector,
    CharucoBoardSpec,
)
from g1_aprilcube_calibration.transforms import invert_transform, validate_transform
from g1_dex3_tabletop.planning.contracts import RobotSnapshot
from g1_dex3_tabletop.tabletop_contracts import (
    CharucoBoardObservation,
    CharucoSupportedEscapeRequest,
)
from g1_dex3_tabletop.tabletop_workflow import load_task_config


def _average_transforms(transforms: Sequence[np.ndarray]) -> np.ndarray:
    if not transforms:
        raise ValueError("cannot average an empty transform sequence")
    values = [validate_transform(item) for item in transforms]
    result = np.eye(4, dtype=np.float64)
    result[:3, 3] = np.median(np.stack([item[:3, 3] for item in values]), axis=0)
    result[:3, :3] = (
        Rotation.from_matrix(np.stack([item[:3, :3] for item in values])).mean().as_matrix()
    )
    return validate_transform(result)


def _rotation_error_deg(first: np.ndarray, second: np.ndarray) -> float:
    relative = first[:3, :3].T @ second[:3, :3]
    return float(np.degrees(Rotation.from_matrix(relative.copy()).magnitude()))


def observe_charuco_board(
    images_bgr: Sequence[np.ndarray],
    *,
    camera_info: RectifiedCameraInfo,
    snapshot: RobotSnapshot,
    detector: CharucoBoardPoseDetector | None = None,
    minimum_accepted_frames: int = 3,
) -> tuple[CharucoBoardObservation, dict[str, Any]]:
    """Aggregate a fixed-board burst without inventing a spread rejection gate."""

    if len(images_bgr) < minimum_accepted_frames:
        raise ValueError(
            f"need at least {minimum_accepted_frames} ChArUco frames, got {len(images_bgr)}"
        )
    active = detector or CharucoBoardPoseDetector()
    estimates = []
    hashes: list[str] = []
    rejected: list[dict[str, Any]] = []
    for index, image in enumerate(images_bgr):
        value = np.asarray(image)
        try:
            if value.ndim != 3 or value.shape[2] != 3 or value.dtype != np.uint8:
                raise ValueError("image must be uint8 BGR")
            digest = hashlib.sha256(value.tobytes()).hexdigest()
            if digest in hashes:
                raise ValueError("duplicate image content")
            estimate = active.detect(value, camera_info)
        except ValueError as error:
            rejected.append({"frame_index": index, "reason": str(error)})
            continue
        hashes.append(digest)
        estimates.append(estimate)
    if len(estimates) < minimum_accepted_frames:
        detail = "; ".join(f"frame {item['frame_index']}: {item['reason']}" for item in rejected)
        raise ValueError(
            f"only {len(estimates)}/{len(images_bgr)} ChArUco frames passed; "
            f"need {minimum_accepted_frames}. {detail}"
        )
    center = _average_transforms([item.camera_T_target for item in estimates])
    translation_spread_mm = max(
        1000.0 * float(np.linalg.norm(item.camera_T_target[:3, 3] - center[:3, 3]))
        for item in estimates
    )
    rotation_spread_deg = max(
        _rotation_error_deg(center, item.camera_T_target) for item in estimates
    )
    observation = CharucoBoardObservation(
        snapshot=snapshot,
        camera_T_board=tuple(tuple(float(value) for value in row) for row in center),
        camera_profile_sha256=camera_info.profile_sha256,
        source_frame_sha256=tuple(hashes),
        translation_spread_mm=translation_spread_mm,
        rotation_spread_deg=rotation_spread_deg,
        board_spec=CharucoBoardSpec().to_dict(),
    )
    evidence = {
        "accepted_frame_count": len(estimates),
        "input_frame_count": len(images_bgr),
        "rejected_frames": rejected,
        "per_frame": [item.to_dict() for item in estimates],
        "aggregate": observation.to_dict(),
        "spread_policy": (
            "recorded_not_gated; each accepted frame independently passed the frozen "
            "ChArUco geometry, corner-count, positive-depth, ambiguity, and reprojection gates"
        ),
    }
    return observation, evidence


def build_charuco_escape_request(
    *,
    arm: str,
    observation: CharucoBoardObservation,
    calibration_bundle: CalibrationBundle,
    calibration_bundle_path: str | Path,
    task_config_path: str | Path,
    random_seed: int,
) -> CharucoSupportedEscapeRequest:
    """Bind the observed board plane to the existing calibrated torso camera model."""

    if CalibrationBundle.load(calibration_bundle_path).content_sha256 != (
        calibration_bundle.content_sha256
    ):
        raise ValueError("calibration bundle object differs from its source file")
    task = load_task_config(task_config_path)
    return CharucoSupportedEscapeRequest(
        observation=observation,
        arm=validate_arm_side(arm),
        torso_T_camera=tuple(
            tuple(float(value) for value in row) for row in calibration_bundle.torso_T_camera
        ),
        joint_position_offsets_rad=dict(calibration_bundle.joint_position_offsets_rad),
        calibration_bundle_sha256=calibration_bundle.content_sha256,
        supported_escape_m=float(task["motion"]["supported_escape_m"]),
        maximum_arm_velocity_rad_s=float(task["motion"]["maximum_arm_velocity_rad_s"]),
        random_seed=int(random_seed),
    )


def camera_motion_from_fixed_board(
    reference_camera_T_board: Sequence[Sequence[float]],
    current_camera_T_board: Sequence[Sequence[float]],
) -> dict[str, Any]:
    """Report current camera motion relative to an explicit fixed-board reference."""

    reference_board_T_camera = invert_transform(
        validate_transform(np.asarray(reference_camera_T_board, dtype=np.float64))
    )
    current_board_T_camera = invert_transform(
        validate_transform(np.asarray(current_camera_T_board, dtype=np.float64))
    )
    translation_mm = 1000.0 * (current_board_T_camera[:3, 3] - reference_board_T_camera[:3, 3])
    return {
        "translation_board_xyz_mm": translation_mm.tolist(),
        "translation_norm_mm": float(np.linalg.norm(translation_mm)),
        "rotation_deg": _rotation_error_deg(
            reference_board_T_camera,
            current_board_T_camera,
        ),
        "sign_convention": "current_camera_minus_reference_camera_in_fixed_board_frame",
    }


def summarize_camera_motion_cycles(events: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Summarize only explicit same-cycle pre-lift, lifted, and returned observations."""

    result: dict[str, Any] = {
        "reference_policy": "explicit_same_cycle_pre_lift",
        "left": {"cycle_count": 0},
        "right": {"cycle_count": 0},
    }
    metric_names = (
        "pre_lift_to_lifted",
        "pre_lift_to_returned",
        "lifted_to_returned",
    )
    for arm in ("left", "right"):
        for name in metric_names:
            result[arm][name] = {
                "translation_norm_mm": [],
                "rotation_deg": [],
            }

    cycle_events = [item for item in events if item.get("phase") != "loaded_baseline"]
    if len(cycle_events) % 3:
        raise ValueError("camera-motion events do not contain complete three-stage cycles")
    for index in range(0, len(cycle_events), 3):
        pre_lift, lifted, returned = cycle_events[index : index + 3]
        if [item.get("phase") for item in (pre_lift, lifted, returned)] != [
            "pre_lift",
            "lifted",
            "returned",
        ]:
            raise ValueError("camera-motion cycle must be pre_lift, lifted, returned")
        identity = (pre_lift.get("repetition"), pre_lift.get("arm"))
        if any(
            (item.get("repetition"), item.get("arm")) != identity for item in (lifted, returned)
        ):
            raise ValueError("camera-motion cycle stages have different identities")
        arm = validate_arm_side(identity[1])

        def camera_T_board(item: dict[str, Any]) -> Sequence[Sequence[float]]:
            try:
                return item["board"]["aggregate"]["camera_T_board"]
            except (KeyError, TypeError) as error:
                raise ValueError("camera-motion event is missing its board transform") from error

        motions = {
            "pre_lift_to_lifted": camera_motion_from_fixed_board(
                camera_T_board(pre_lift), camera_T_board(lifted)
            ),
            "pre_lift_to_returned": camera_motion_from_fixed_board(
                camera_T_board(pre_lift), camera_T_board(returned)
            ),
            "lifted_to_returned": camera_motion_from_fixed_board(
                camera_T_board(lifted), camera_T_board(returned)
            ),
        }
        result[arm]["cycle_count"] += 1
        for name, motion in motions.items():
            result[arm][name]["translation_norm_mm"].append(motion["translation_norm_mm"])
            result[arm][name]["rotation_deg"].append(motion["rotation_deg"])
    return result
