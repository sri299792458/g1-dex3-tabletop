"""Independent bilateral projection and normalized observability diagnostics."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from g1_aprilcube_calibration.calibration_evaluation import CAMERA_MOUNT_JOINT
from g1_aprilcube_calibration.camera_models import RectifiedCameraInfo
from g1_aprilcube_calibration.joint_map import G1_29_JOINT_NAMES, arm_hand_link
from g1_aprilcube_calibration.transforms import (
    invert_transform,
    pose_vector_to_transform,
    transform_points,
    transform_to_pose_vector,
    validate_transform,
)
from g1_aprilcube_calibration.urdf_model import URDFModel
from g1_dex3_tabletop.calibration.models import (
    BilateralCalibrationDataset,
    BilateralCalibrationSample,
    BilateralModelSpec,
    CameraFrameArtifact,
)

_SIDES = ("left", "right")
_CAMERA_NAMES = {
    "x": f"{CAMERA_MOUNT_JOINT}_x",
    "y": f"{CAMERA_MOUNT_JOINT}_y",
    "z": f"{CAMERA_MOUNT_JOINT}_z",
    "roll": f"{CAMERA_MOUNT_JOINT}_a",
    "pitch": f"{CAMERA_MOUNT_JOINT}_b",
    "yaw": f"{CAMERA_MOUNT_JOINT}_c",
}
_POSE_SUFFIXES = ("x", "y", "z", "a", "b", "c")


def target_parameter_names(side: str) -> tuple[str, ...]:
    if side not in _SIDES:
        raise ValueError("target side must be left or right")
    return tuple(f"{side}_calibration_target_{suffix}" for suffix in _POSE_SUFFIXES)


@dataclass(frozen=True, slots=True)
class BilateralObservabilityReport:
    parameter_names: tuple[str, ...]
    rank: int
    singular_values: tuple[float, ...]
    condition_number: float | None
    observable: bool
    normalized_null_vectors: tuple[tuple[float, ...], ...]

    @property
    def parameter_count(self) -> int:
        return len(self.parameter_names)

    def to_dict(self) -> dict:
        return {
            "parameter_names": list(self.parameter_names),
            "rank": self.rank,
            "parameter_count": self.parameter_count,
            "singular_values": list(self.singular_values),
            "condition_number": self.condition_number,
            "observable": self.observable,
            "normalized_null_vectors": [list(row) for row in self.normalized_null_vectors],
        }


class BilateralCalibrationProjection:
    """Evaluate the declared Ferguson model without adjusting its parameters."""

    def __init__(
        self,
        urdf_model: URDFModel,
        *,
        camera_frames: CameraFrameArtifact,
        model: BilateralModelSpec,
        initial_hand_T_targets: Mapping[str, np.ndarray],
    ) -> None:
        if set(initial_hand_T_targets) != set(_SIDES):
            raise ValueError("initial hand-target transforms must contain left and right")
        self.urdf_model = urdf_model
        self.camera_frames = camera_frames
        self.model = model
        self.initial_hand_T_targets = {
            side: validate_transform(initial_hand_T_targets[side]) for side in _SIDES
        }
        self._torso_T_camera_parent = urdf_model.transform(
            "torso_link",
            camera_frames.urdf_parent_link,
            {},
        )

    @property
    def parameter_names(self) -> tuple[str, ...]:
        names = list(self.model.joint_offsets)
        names.extend(_CAMERA_NAMES[item] for item in self.model.camera_components)
        if self.model.optimize_hand_targets:
            for side in _SIDES:
                names.extend(target_parameter_names(side))
        return tuple(names)

    @property
    def parameter_scales(self) -> np.ndarray:
        """Return physical scales used to compare heterogeneous Jacobian columns."""

        joint_names = set(self.model.joint_offsets)
        camera_translation_names = {_CAMERA_NAMES[item] for item in ("x", "y", "z")}
        target_translation_names = {
            target_parameter_names(side)[index] for side in _SIDES for index in range(3)
        }
        return np.asarray(
            [
                np.deg2rad(5.0)
                if name in joint_names
                else 0.01
                if name in camera_translation_names
                else 0.005
                if name in target_translation_names
                else np.deg2rad(5.0)
                for name in self.parameter_names
            ],
            dtype=np.float64,
        )

    def initial_parameters(self) -> dict[str, float]:
        values = {name: 0.0 for name in self.model.joint_offsets}
        values.update({_CAMERA_NAMES[item]: 0.0 for item in self.model.camera_components})
        if self.model.optimize_hand_targets:
            for side in _SIDES:
                pose = transform_to_pose_vector(self.initial_hand_T_targets[side])
                values.update(dict(zip(target_parameter_names(side), pose, strict=True)))
        return values

    def parameters_for_nominal_model(
        self,
        *,
        torso_T_camera: np.ndarray,
        hand_T_targets: Mapping[str, np.ndarray],
        joint_position_offsets_rad: Mapping[str, float],
    ) -> dict[str, float]:
        """Express a physical nominal model in this parameterization.

        Experiment design should be linearized around the best model currently
        available, even though the later solver remains free to estimate a new
        camera, hand-target transforms, and declared joint offsets.
        """

        if set(hand_T_targets) != set(_SIDES):
            raise ValueError("nominal hand targets must contain left and right")
        values = self.initial_parameters()
        correction = validate_transform(
            invert_transform(self._torso_T_camera_parent)
            @ validate_transform(torso_T_camera)
            @ invert_transform(self.camera_frames.parent_T_optical)
        )
        correction_pose = transform_to_pose_vector(correction)
        for index, component in enumerate(("x", "y", "z", "roll", "pitch", "yaw")):
            name = _CAMERA_NAMES[component]
            if name in values:
                values[name] = float(correction_pose[index])
        for name in self.model.joint_offsets:
            values[name] = float(joint_position_offsets_rad.get(name, 0.0))
        if self.model.optimize_hand_targets:
            for side in _SIDES:
                pose = transform_to_pose_vector(validate_transform(hand_T_targets[side]))
                values.update(dict(zip(target_parameter_names(side), pose, strict=True)))
        return self._validated_parameters(values)

    def validate_dataset_sources(self, dataset: BilateralCalibrationDataset) -> None:
        if dataset.urdf_sha256 != self.urdf_model.sha256:
            raise ValueError("bilateral dataset belongs to a different URDF")
        if dataset.rgb_optical_transform_sha256 != self.camera_frames.content_sha256:
            raise ValueError("bilateral dataset belongs to a different camera-frame artifact")

    def transforms(
        self,
        parameters: Mapping[str, float],
        *,
        side: str,
    ) -> tuple[np.ndarray, np.ndarray]:
        if side not in _SIDES:
            raise ValueError("projection side must be left or right")
        values = self._validated_parameters(parameters)
        camera_pose = [
            values.get(_CAMERA_NAMES[item], 0.0)
            for item in ("x", "y", "z", "roll", "pitch", "yaw")
        ]
        torso_T_camera = (
            self._torso_T_camera_parent
            @ pose_vector_to_transform(camera_pose)
            @ self.camera_frames.parent_T_optical
        )
        if self.model.optimize_hand_targets:
            hand_T_target = pose_vector_to_transform(
                [values[name] for name in target_parameter_names(side)]
            )
        else:
            hand_T_target = self.initial_hand_T_targets[side]
        return validate_transform(torso_T_camera), validate_transform(hand_T_target)

    def project_side(
        self,
        sample: BilateralCalibrationSample,
        *,
        side: str,
        parameters: Mapping[str, float],
    ) -> tuple[np.ndarray, np.ndarray]:
        values = self._validated_parameters(parameters)
        torso_T_camera, hand_T_target = self.transforms(values, side=side)
        positions = {
            name: float(position) + values.get(name, 0.0)
            for name, position in zip(
                G1_29_JOINT_NAMES,
                sample.joint_positions_rad,
                strict=True,
            )
        }
        torso_T_hand = self.urdf_model.transform(
            "torso_link",
            arm_hand_link(side),
            positions,
        )
        observation = sample.left if side == "left" else sample.right
        torso_points = transform_points(
            torso_T_hand @ hand_T_target,
            np.asarray(observation.object_points_m, dtype=np.float64),
        )
        camera_points = transform_points(invert_transform(torso_T_camera), torso_points)
        depths = camera_points[:, 2]
        intrinsics = RectifiedCameraInfo.from_dict(sample.camera_info)
        matrix = intrinsics.rectified_camera_matrix
        projected = np.column_stack(
            (
                matrix[0, 0] * camera_points[:, 0] / depths + matrix[0, 2],
                matrix[1, 1] * camera_points[:, 1] / depths + matrix[1, 2],
            )
        )
        return projected, depths

    def pixel_residuals(
        self,
        parameters: Mapping[str, float],
        samples: Sequence[BilateralCalibrationSample],
    ) -> np.ndarray:
        residuals: list[np.ndarray] = []
        for sample in samples:
            for side in _SIDES:
                predicted, depths = self.project_side(
                    sample,
                    side=side,
                    parameters=parameters,
                )
                if np.any(depths <= 0.0):
                    raise ValueError("bilateral projection contains points behind the camera")
                observation = sample.left if side == "left" else sample.right
                residuals.append(
                    (
                        predicted - np.asarray(observation.image_points_px, dtype=np.float64)
                    ).reshape(-1)
                )
        if not residuals:
            raise ValueError("bilateral projection requires at least one sample")
        return np.concatenate(residuals)

    def normalized_jacobian(
        self,
        parameters: Mapping[str, float],
        samples: Sequence[BilateralCalibrationSample],
    ) -> np.ndarray:
        values = self._validated_parameters(parameters)
        names = self.parameter_names
        center = np.asarray([values[name] for name in names], dtype=np.float64)
        columns: list[np.ndarray] = []
        for index, scale in enumerate(self.parameter_scales):
            step = float(scale) * 1e-5
            plus = center.copy()
            minus = center.copy()
            plus[index] += step
            minus[index] -= step
            plus_values = dict(zip(names, plus, strict=True))
            minus_values = dict(zip(names, minus, strict=True))
            derivative = (
                self.pixel_residuals(plus_values, samples)
                - self.pixel_residuals(minus_values, samples)
            ) / (2.0 * step)
            columns.append(derivative * scale)
        return np.column_stack(columns)

    def observability(
        self,
        parameters: Mapping[str, float],
        samples: Sequence[BilateralCalibrationSample],
        *,
        relative_rank_threshold: float = 1e-7,
        maximum_condition_number: float = 1e8,
    ) -> BilateralObservabilityReport:
        if not 0.0 < relative_rank_threshold < 1.0:
            raise ValueError("relative rank threshold must lie in (0, 1)")
        if maximum_condition_number <= 1.0:
            raise ValueError("maximum condition number must exceed one")
        jacobian = self.normalized_jacobian(parameters, samples)
        _u, singular_values, right_vectors = np.linalg.svd(
            jacobian,
            full_matrices=False,
        )
        threshold = (
            relative_rank_threshold * singular_values[0]
            if len(singular_values) and singular_values[0] > 0.0
            else np.inf
        )
        rank = int(np.count_nonzero(singular_values > threshold))
        condition = (
            float(singular_values[0] / singular_values[-1])
            if len(singular_values) and singular_values[-1] > 0.0
            else None
        )
        null_vectors = tuple(
            tuple(float(value) for value in right_vectors[index])
            for index, singular in enumerate(singular_values)
            if singular <= threshold
        )
        return BilateralObservabilityReport(
            parameter_names=self.parameter_names,
            rank=rank,
            singular_values=tuple(float(value) for value in singular_values),
            condition_number=condition,
            observable=(
                rank == len(self.parameter_names)
                and condition is not None
                and condition <= maximum_condition_number
            ),
            normalized_null_vectors=null_vectors,
        )

    def _validated_parameters(
        self,
        parameters: Mapping[str, float],
    ) -> dict[str, float]:
        if set(parameters) != set(self.parameter_names):
            missing = sorted(set(self.parameter_names) - set(parameters))
            unexpected = sorted(set(parameters) - set(self.parameter_names))
            raise ValueError(
                f"bilateral parameter set differs from model: missing={missing}, "
                f"unexpected={unexpected}"
            )
        values = {name: float(parameters[name]) for name in self.parameter_names}
        if not np.all(np.isfinite(tuple(values.values()))):
            raise ValueError("bilateral parameters must be finite")
        return values
