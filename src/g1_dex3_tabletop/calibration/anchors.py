"""Repeated bilateral-anchor drift diagnostics for a fitted static model."""

from __future__ import annotations

import hashlib
import itertools
import json
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

from g1_aprilcube_calibration.camera_models import RectifiedCameraInfo
from g1_aprilcube_calibration.joint_map import G1_29_JOINT_NAMES, arm_hand_link
from g1_aprilcube_calibration.transforms import invert_transform, validate_transform
from g1_aprilcube_calibration.urdf_model import URDFModel
from g1_dex3_tabletop.calibration.models import (
    BilateralCalibrationDataset,
    BilateralCalibrationSample,
)
from g1_dex3_tabletop.calibration.solver import BilateralSolverResult
from g1_dex3_tabletop.calibration.validation import BilateralValidationConfig

_SIDES = ("left", "right")


@dataclass(frozen=True, slots=True)
class BilateralAnchorDriftReport:
    dataset_sha256: str
    model_sha256: str
    group_sample_counts: dict[str, int]
    maximum_temporal_translation_m_by_arm: dict[str, float]
    maximum_temporal_rotation_deg_by_arm: dict[str, float]
    maximum_bilateral_translation_disagreement_m: float
    maximum_bilateral_rotation_disagreement_deg: float
    passed: bool
    failures: tuple[str, ...]
    thresholds: dict[str, float]
    schema_version: int = 1

    @property
    def content_sha256(self) -> str:
        return hashlib.sha256(
            json.dumps(
                self.to_dict(include_hash=False),
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ).encode()
        ).hexdigest()

    def to_dict(self, *, include_hash: bool = True) -> dict[str, Any]:
        result = {
            "schema_version": self.schema_version,
            "dataset_sha256": self.dataset_sha256,
            "model_sha256": self.model_sha256,
            "group_sample_counts": self.group_sample_counts,
            "maximum_temporal_translation_m_by_arm": (self.maximum_temporal_translation_m_by_arm),
            "maximum_temporal_rotation_deg_by_arm": (self.maximum_temporal_rotation_deg_by_arm),
            "maximum_bilateral_translation_disagreement_m": (
                self.maximum_bilateral_translation_disagreement_m
            ),
            "maximum_bilateral_rotation_disagreement_deg": (
                self.maximum_bilateral_rotation_disagreement_deg
            ),
            "passed": self.passed,
            "failures": list(self.failures),
            "thresholds": self.thresholds,
        }
        if include_hash:
            result["content_sha256"] = self.content_sha256
        return result


def evaluate_bilateral_anchor_drift(
    dataset: BilateralCalibrationDataset,
    urdf_model: URDFModel,
    result: BilateralSolverResult,
    *,
    config: BilateralValidationConfig,
) -> BilateralAnchorDriftReport:
    """Infer camera pose independently from both hands at repeated anchor poses."""

    groups: dict[str, list[BilateralCalibrationSample]] = {}
    for sample in dataset.samples:
        if sample.capture_role == "anchor":
            groups.setdefault(sample.pose_group_id, []).append(sample)
    repeated = {name: values for name, values in groups.items() if len(values) >= 3}
    if not repeated:
        raise ValueError("anchor drift requires one anchor pose captured at least three times")
    temporal_translation = {side: 0.0 for side in _SIDES}
    temporal_rotation = {side: 0.0 for side in _SIDES}
    bilateral_translation = 0.0
    bilateral_rotation = 0.0
    for samples in repeated.values():
        inferred = {
            side: [
                _inferred_torso_T_camera(
                    sample,
                    side=side,
                    urdf_model=urdf_model,
                    result=result,
                )
                for sample in samples
            ]
            for side in _SIDES
        }
        for side in _SIDES:
            for first, second in itertools.combinations(inferred[side], 2):
                translation, rotation = _transform_difference(first, second)
                temporal_translation[side] = max(temporal_translation[side], translation)
                temporal_rotation[side] = max(temporal_rotation[side], rotation)
        for left, right in zip(inferred["left"], inferred["right"], strict=True):
            translation, rotation = _transform_difference(left, right)
            bilateral_translation = max(bilateral_translation, translation)
            bilateral_rotation = max(bilateral_rotation, rotation)
    thresholds = {
        "maximum_anchor_translation_drift_m": (config.maximum_anchor_translation_drift_m),
        "maximum_anchor_rotation_drift_deg": config.maximum_anchor_rotation_drift_deg,
        "maximum_bilateral_anchor_translation_disagreement_m": (
            config.maximum_bilateral_anchor_translation_disagreement_m
        ),
        "maximum_bilateral_anchor_rotation_disagreement_deg": (
            config.maximum_bilateral_anchor_rotation_disagreement_deg
        ),
    }
    failures: list[str] = []
    for side in _SIDES:
        if temporal_translation[side] > config.maximum_anchor_translation_drift_m:
            failures.append(f"{side} anchor translation drift exceeds its limit")
        if temporal_rotation[side] > config.maximum_anchor_rotation_drift_deg:
            failures.append(f"{side} anchor rotation drift exceeds its limit")
    if bilateral_translation > config.maximum_bilateral_anchor_translation_disagreement_m:
        failures.append("bilateral anchor translation disagreement exceeds its limit")
    if bilateral_rotation > config.maximum_bilateral_anchor_rotation_disagreement_deg:
        failures.append("bilateral anchor rotation disagreement exceeds its limit")
    return BilateralAnchorDriftReport(
        dataset_sha256=dataset.content_sha256,
        model_sha256=result.model.content_sha256,
        group_sample_counts={name: len(values) for name, values in sorted(repeated.items())},
        maximum_temporal_translation_m_by_arm=temporal_translation,
        maximum_temporal_rotation_deg_by_arm=temporal_rotation,
        maximum_bilateral_translation_disagreement_m=bilateral_translation,
        maximum_bilateral_rotation_disagreement_deg=bilateral_rotation,
        passed=not failures,
        failures=tuple(failures),
        thresholds=thresholds,
    )


def _inferred_torso_T_camera(
    sample: BilateralCalibrationSample,
    *,
    side: str,
    urdf_model: URDFModel,
    result: BilateralSolverResult,
) -> np.ndarray:
    observation = sample.left if side == "left" else sample.right
    object_points = np.asarray(observation.object_points_m, dtype=np.float64)
    image_points = np.asarray(observation.image_points_px, dtype=np.float64)
    positions = {
        name: float(value) + result.parameters.get(name, 0.0)
        for name, value in zip(
            G1_29_JOINT_NAMES,
            sample.joint_positions_rad,
            strict=True,
        )
    }
    torso_T_hand = urdf_model.transform(
        "torso_link",
        arm_hand_link(side),
        positions,
    )
    torso_T_target = torso_T_hand @ result.hand_T_targets[side]
    predicted_camera_T_target = invert_transform(result.torso_T_camera) @ torso_T_target
    initial_rvec = (
        Rotation.from_matrix(predicted_camera_T_target[:3, :3]).as_rotvec().reshape(3, 1)
    )
    initial_tvec = predicted_camera_T_target[:3, 3].reshape(3, 1).copy()
    camera_info = RectifiedCameraInfo.from_dict(sample.camera_info)
    success, rvec, tvec = cv2.solvePnP(
        object_points,
        image_points,
        camera_info.rectified_camera_matrix,
        np.zeros(5),
        rvec=initial_rvec,
        tvec=initial_tvec,
        useExtrinsicGuess=True,
        flags=cv2.SOLVEPNP_ITERATIVE,
    )
    if not success:
        raise ValueError(f"{side} repeated-anchor PnP failed for {sample.frame_id}")
    camera_T_target = np.eye(4)
    camera_T_target[:3, :3] = cv2.Rodrigues(rvec)[0]
    camera_T_target[:3, 3] = np.asarray(tvec).reshape(3)
    return validate_transform(torso_T_target @ invert_transform(camera_T_target))


def _transform_difference(first: np.ndarray, second: np.ndarray) -> tuple[float, float]:
    delta = invert_transform(first) @ second
    translation = float(np.linalg.norm(delta[:3, 3]))
    rotation_deg = float(np.rad2deg(Rotation.from_matrix(delta[:3, :3]).magnitude()))
    return translation, rotation_deg
