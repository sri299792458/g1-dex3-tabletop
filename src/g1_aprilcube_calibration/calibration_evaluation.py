"""Projection and diagnostics for native robot_calibration solutions.

This module never adjusts parameters.  Mike Ferguson's Ceres optimizer is the
only calibration backend; the code here only evaluates its returned offsets on
training or held-out observations.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from g1_aprilcube_calibration.camera_initialization import (
    realsense_link_T_color_optical,
)
from g1_aprilcube_calibration.camera_models import RectifiedCameraInfo
from g1_aprilcube_calibration.dataset_builder import CalibrationSample
from g1_aprilcube_calibration.joint_map import (
    G1_29_JOINT_NAMES,
    arm_hand_link,
    validate_arm_side,
)
from g1_aprilcube_calibration.transforms import (
    invert_transform,
    pose_vector_to_transform,
    transform_points,
    validate_transform,
)
from g1_aprilcube_calibration.urdf_model import URDFModel

CAMERA_MOUNT_JOINT = "d435_joint"
CAMERA_OFFSET_NAMES = tuple(
    f"{CAMERA_MOUNT_JOINT}_{suffix}" for suffix in ("x", "y", "z", "a", "b", "c")
)
TARGET_FRAME = "calibration_target"
TARGET_OFFSET_NAMES = tuple(
    f"{TARGET_FRAME}_{suffix}" for suffix in ("x", "y", "z", "a", "b", "c")
)


@dataclass(frozen=True, slots=True)
class ObservabilityReport:
    rank: int
    parameter_count: int
    singular_values: tuple[float, ...]
    condition_number: float | None
    observable: bool
    parameter_stddev: tuple[float | None, ...]

    def to_dict(self) -> dict:
        return {
            "rank": self.rank,
            "parameter_count": self.parameter_count,
            "singular_values": list(self.singular_values),
            "condition_number": self.condition_number,
            "observable": self.observable,
            "parameter_stddev": list(self.parameter_stddev),
        }


@dataclass(frozen=True, slots=True)
class NativeCalibrationSolution:
    torso_T_camera: np.ndarray
    hand_T_target: np.ndarray
    offsets: Mapping[str, float]
    success: bool
    message: str
    evaluations: int
    cost: float
    optimization_rms_px: float
    observability: ObservabilityReport
    backend: str = "mikeferguson/robot_calibration:Ceres"

    def __post_init__(self) -> None:
        object.__setattr__(self, "torso_T_camera", validate_transform(self.torso_T_camera))
        object.__setattr__(self, "hand_T_target", validate_transform(self.hand_T_target))
        offsets = {str(name): float(value) for name, value in self.offsets.items()}
        if not offsets or not all(np.isfinite(value) for value in offsets.values()):
            raise ValueError("native calibration offsets must be finite and non-empty")
        object.__setattr__(self, "offsets", offsets)
        if self.observability.parameter_count != len(offsets):
            raise ValueError("observability parameter count does not match offsets")

    @property
    def parameter_names(self) -> tuple[str, ...]:
        return tuple(self.offsets)

    @property
    def parameters(self) -> tuple[float, ...]:
        return tuple(self.offsets.values())

    def to_dict(self) -> dict:
        return {
            "backend": self.backend,
            "torso_T_camera": self.torso_T_camera.tolist(),
            "hand_T_target": self.hand_T_target.tolist(),
            "parameters": dict(self.offsets),
            "success": self.success,
            "message": self.message,
            "evaluations": self.evaluations,
            "cost": self.cost,
            "optimization_rms_px": self.optimization_rms_px,
            "observability": self.observability.to_dict(),
        }


class NativeCalibrationProjection:
    """Evaluate Ferguson offset semantics without performing optimization."""

    def __init__(
        self,
        model: URDFModel,
        *,
        calibration_arm: str,
        fixed_hand_T_target: np.ndarray | None,
    ) -> None:
        self.model = model
        self.calibration_arm = validate_arm_side(calibration_arm)
        self.hand_link = arm_hand_link(self.calibration_arm)
        self.fixed_hand_T_target = (
            None if fixed_hand_T_target is None else validate_transform(fixed_hand_T_target)
        )
        self._torso_T_d435 = model.transform("torso_link", "d435_link", {})

    def transforms(self, offsets: Mapping[str, float]) -> tuple[np.ndarray, np.ndarray]:
        values = {str(name): float(value) for name, value in offsets.items()}
        camera_delta = pose_vector_to_transform(
            [values.get(name, 0.0) for name in CAMERA_OFFSET_NAMES]
        )
        torso_T_camera = self._torso_T_d435 @ camera_delta @ realsense_link_T_color_optical()
        if self.fixed_hand_T_target is None:
            missing = [name for name in TARGET_OFFSET_NAMES if name not in values]
            if missing:
                raise ValueError(
                    "native solution is missing optimized target offsets: " + ", ".join(missing)
                )
            hand_T_target = pose_vector_to_transform(
                [values[name] for name in TARGET_OFFSET_NAMES]
            )
        else:
            hand_T_target = self.fixed_hand_T_target
        return validate_transform(torso_T_camera), validate_transform(hand_T_target)

    def project_sample(
        self,
        sample: CalibrationSample,
        solution: NativeCalibrationSolution,
    ) -> tuple[np.ndarray, np.ndarray]:
        return self.project_sample_with_offsets(sample, solution.offsets)

    def project_sample_with_offsets(
        self,
        sample: CalibrationSample,
        offsets: Mapping[str, float],
    ) -> tuple[np.ndarray, np.ndarray]:
        torso_T_camera, hand_T_target = self.transforms(offsets)
        position = np.asarray(sample.measured_state["position"], dtype=np.float64)
        if position.shape != (len(G1_29_JOINT_NAMES),):
            raise ValueError(f"sample {sample.frame_id} has an invalid joint state")
        positions = {
            name: float(value) + float(offsets.get(name, 0.0))
            for name, value in zip(G1_29_JOINT_NAMES, position, strict=True)
        }
        torso_T_hand = self.model.transform("torso_link", self.hand_link, positions)
        torso_points = transform_points(
            torso_T_hand @ hand_T_target,
            np.asarray(sample.object_points_m, dtype=np.float64),
        )
        camera_points = transform_points(invert_transform(torso_T_camera), torso_points)
        depths = camera_points[:, 2]
        intrinsics = RectifiedCameraInfo.from_dict(sample.camera_info)
        camera_matrix = intrinsics.rectified_camera_matrix
        projected = np.column_stack(
            (
                camera_matrix[0, 0] * camera_points[:, 0] / depths + camera_matrix[0, 2],
                camera_matrix[1, 1] * camera_points[:, 1] / depths + camera_matrix[1, 2],
            )
        )
        return projected, depths

    def pixel_residuals(
        self,
        offsets: Mapping[str, float],
        samples: Sequence[CalibrationSample],
    ) -> np.ndarray:
        residuals: list[np.ndarray] = []
        for sample in samples:
            predicted, depths = self.project_sample_with_offsets(sample, offsets)
            if np.any(depths <= 0):
                raise ValueError("native solution projects calibration points behind camera")
            residuals.append(
                (predicted - np.asarray(sample.image_points_px, dtype=np.float64)).reshape(-1)
            )
        if not residuals:
            raise ValueError("at least one sample is required for residual evaluation")
        return np.concatenate(residuals)

    def observability(
        self,
        offsets: Mapping[str, float],
        samples: Sequence[CalibrationSample],
        *,
        relative_rank_threshold: float = 1e-7,
        maximum_condition_number: float = 1e8,
    ) -> ObservabilityReport:
        names = tuple(offsets)
        center = np.asarray([offsets[name] for name in names], dtype=np.float64)
        columns: list[np.ndarray] = []
        for index, name in enumerate(names):
            step = 1e-6
            plus = center.copy()
            minus = center.copy()
            plus[index] += step
            minus[index] -= step
            plus_offsets = dict(zip(names, plus, strict=True))
            minus_offsets = dict(zip(names, minus, strict=True))
            columns.append(
                (
                    self.pixel_residuals(plus_offsets, samples)
                    - self.pixel_residuals(minus_offsets, samples)
                )
                / (2.0 * step)
            )
        jacobian = np.column_stack(columns)
        singular_values = np.linalg.svd(jacobian, compute_uv=False)
        threshold = (
            relative_rank_threshold * singular_values[0]
            if len(singular_values) and singular_values[0] > 0
            else np.inf
        )
        rank = int(np.count_nonzero(singular_values > threshold))
        condition = (
            float(singular_values[0] / singular_values[-1])
            if len(singular_values) and singular_values[-1] > 0
            else None
        )
        residual = self.pixel_residuals(offsets, samples)
        degrees_of_freedom = max(len(residual) - len(names), 1)
        variance = float(residual @ residual) / degrees_of_freedom
        covariance = variance * np.linalg.pinv(jacobian.T @ jacobian)
        standard_deviation = np.sqrt(np.maximum(np.diag(covariance), 0.0))
        observable = rank == len(names) and (
            condition is not None and condition <= maximum_condition_number
        )
        return ObservabilityReport(
            rank=rank,
            parameter_count=len(names),
            singular_values=tuple(float(value) for value in singular_values),
            condition_number=condition,
            observable=observable,
            parameter_stddev=tuple(float(value) for value in standard_deviation),
        )
