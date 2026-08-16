"""Pure anchored camera-pose estimators for offline G1 research.

The estimators in this module intentionally expose their contact assumptions.
They do not claim globally observable odometry and they have no ROS or robot-
command dependency.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from scipy.spatial.transform import Rotation

from g1_aprilcube_calibration.calibration_bundle import CalibrationBundle
from g1_aprilcube_calibration.joint_map import (
    G1_29_JOINT_NAMES,
    validate_full_joint_vector,
)
from g1_aprilcube_calibration.transforms import invert_transform, validate_transform
from g1_aprilcube_calibration.urdf_model import URDFModel

EstimatorName = Literal[
    "fixed_camera",
    "fixed_pelvis_fk",
    "fixed_torso_imu_origin",
    "fixed_pelvis_imu_origin_fk",
    "hybrid_pelvis_position_torso_orientation",
]

ESTIMATOR_NAMES: tuple[EstimatorName, ...] = (
    "fixed_camera",
    "fixed_pelvis_fk",
    "fixed_torso_imu_origin",
    "fixed_pelvis_imu_origin_fk",
    "hybrid_pelvis_position_torso_orientation",
)


def rotation_from_wxyz(value: np.ndarray) -> np.ndarray:
    """Convert one Unitree scalar-first quaternion to a rotation matrix."""

    quaternion = np.asarray(value, dtype=np.float64).reshape(-1)
    if (
        quaternion.shape != (4,)
        or not np.all(np.isfinite(quaternion))
        or np.linalg.norm(quaternion) <= 0.0
    ):
        raise ValueError("IMU quaternion must contain four finite values")
    quaternion = quaternion / np.linalg.norm(quaternion)
    return Rotation.from_quat(quaternion[[1, 2, 3, 0]]).as_matrix()


def proprioceptive_sample(state, torso_imu) -> ProprioceptiveSample:
    """Build the tested estimator input from fresh live Unitree observations."""

    if state.pelvis_imu_quaternion_wxyz is None:
        raise ValueError("LowState has no pelvis IMU quaternion")
    return ProprioceptiveSample(
        timestamp_ns=int(max(state.receipt_monotonic_s, torso_imu.receipt_monotonic_s) * 1e9),
        q29_rad=state.position,
        navigation_R_pelvis_imu=rotation_from_wxyz(state.pelvis_imu_quaternion_wxyz),
        navigation_R_torso_imu=rotation_from_wxyz(torso_imu.quaternion_wxyz),
    )


def _rotation(value: np.ndarray, *, name: str) -> np.ndarray:
    matrix = np.asarray(value, dtype=np.float64).copy()
    if matrix.shape != (3, 3) or not np.all(np.isfinite(matrix)):
        raise ValueError(f"{name} must be a finite 3x3 matrix")
    if not np.allclose(matrix.T @ matrix, np.eye(3), atol=1.0e-7, rtol=0.0):
        raise ValueError(f"{name} is not orthonormal")
    if not np.isclose(np.linalg.det(matrix), 1.0, atol=1.0e-7, rtol=0.0):
        raise ValueError(f"{name} is not a proper rotation")
    matrix.setflags(write=False)
    return matrix


def _transform(rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    result = np.eye(4, dtype=np.float64)
    result[:3, :3] = _rotation(rotation, name="transform rotation")
    vector = np.asarray(translation, dtype=np.float64).reshape(-1)
    if vector.shape != (3,) or not np.all(np.isfinite(vector)):
        raise ValueError("transform translation must contain three finite values")
    result[:3, 3] = vector
    return validate_transform(result)


@dataclass(frozen=True, slots=True)
class ProprioceptiveSample:
    """One synchronized measured-body sample.

    Unitree reports both IMU orientations in an arbitrary navigation frame.  An
    anchored estimator uses only their relative change, so that frame need not
    coincide with the table or robot model frame.
    """

    timestamp_ns: int
    q29_rad: np.ndarray
    navigation_R_pelvis_imu: np.ndarray
    navigation_R_torso_imu: np.ndarray

    def __post_init__(self) -> None:
        if self.timestamp_ns < 0:
            raise ValueError("sample timestamp must be non-negative")
        object.__setattr__(self, "q29_rad", validate_full_joint_vector(self.q29_rad))
        object.__setattr__(
            self,
            "navigation_R_pelvis_imu",
            _rotation(self.navigation_R_pelvis_imu, name="pelvis IMU orientation"),
        )
        object.__setattr__(
            self,
            "navigation_R_torso_imu",
            _rotation(self.navigation_R_torso_imu, name="torso IMU orientation"),
        )


@dataclass(frozen=True, slots=True)
class CameraPoseAnchor:
    """A visual table/world anchor paired with simultaneous robot state."""

    reference_T_camera: np.ndarray
    sample: ProprioceptiveSample

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "reference_T_camera",
            validate_transform(np.asarray(self.reference_T_camera, dtype=np.float64)),
        )


def pose_error(predicted: np.ndarray, observed: np.ndarray) -> dict[str, object]:
    """Return translation and rotation error without hiding error direction."""

    predicted = validate_transform(np.asarray(predicted, dtype=np.float64))
    observed = validate_transform(np.asarray(observed, dtype=np.float64))
    translation_mm = 1000.0 * (observed[:3, 3] - predicted[:3, 3])
    rotation_deg = float(
        np.degrees(Rotation.from_matrix(predicted[:3, :3].T @ observed[:3, :3]).magnitude())
    )
    return {
        "translation_reference_xyz_mm": translation_mm.tolist(),
        "translation_norm_mm": float(np.linalg.norm(translation_mm)),
        "rotation_deg": rotation_deg,
    }


class AnchoredCameraPoseEstimators:
    """Comparable camera-pose hypotheses sharing one visual anchor."""

    def __init__(
        self,
        *,
        model: URDFModel,
        calibration_bundle: CalibrationBundle,
    ) -> None:
        if model.sha256 != calibration_bundle.base_urdf_sha256:
            raise ValueError("calibration bundle belongs to a different base URDF")
        self.model = model
        self.bundle = calibration_bundle
        self.pelvis_T_pelvis_imu = model.transform("pelvis", "imu_in_pelvis", {})
        self.torso_T_torso_imu = model.transform("torso_link", "imu_in_torso", {})

    def pelvis_T_camera(self, sample: ProprioceptiveSample) -> np.ndarray:
        positions = {
            name: float(value) + float(self.bundle.joint_position_offsets_rad.get(name, 0.0))
            for name, value in zip(G1_29_JOINT_NAMES, sample.q29_rad, strict=True)
        }
        return validate_transform(
            self.model.transform("pelvis", "torso_link", positions) @ self.bundle.torso_T_camera
        )

    def predict(
        self,
        anchor: CameraPoseAnchor,
        current: ProprioceptiveSample,
        estimator: EstimatorName,
    ) -> np.ndarray:
        if estimator not in ESTIMATOR_NAMES:
            raise ValueError(f"unknown camera estimator: {estimator}")
        reference_T_camera = anchor.reference_T_camera
        reference_pelvis_T_camera = self.pelvis_T_camera(anchor.sample)
        current_pelvis_T_camera = self.pelvis_T_camera(current)
        reference_T_pelvis = reference_T_camera @ invert_transform(reference_pelvis_T_camera)

        if estimator == "fixed_camera":
            return reference_T_camera.copy()
        if estimator == "fixed_pelvis_fk":
            return validate_transform(reference_T_pelvis @ current_pelvis_T_camera)

        pelvis_prediction = self._fixed_pelvis_imu_origin_prediction(
            anchor,
            current,
            reference_T_pelvis=reference_T_pelvis,
            current_pelvis_T_camera=current_pelvis_T_camera,
        )
        if estimator == "fixed_pelvis_imu_origin_fk":
            return pelvis_prediction

        torso_prediction = self._fixed_torso_imu_origin_prediction(anchor, current)
        if estimator == "fixed_torso_imu_origin":
            return torso_prediction

        hybrid = pelvis_prediction.copy()
        hybrid[:3, :3] = torso_prediction[:3, :3]
        return validate_transform(hybrid)

    def predict_all(
        self,
        anchor: CameraPoseAnchor,
        current: ProprioceptiveSample,
    ) -> dict[EstimatorName, np.ndarray]:
        return {name: self.predict(anchor, current, name) for name in ESTIMATOR_NAMES}

    def _fixed_pelvis_imu_origin_prediction(
        self,
        anchor: CameraPoseAnchor,
        current: ProprioceptiveSample,
        *,
        reference_T_pelvis: np.ndarray,
        current_pelvis_T_camera: np.ndarray,
    ) -> np.ndarray:
        reference_R_pelvis = reference_T_pelvis[:3, :3]
        reference_R_navigation = reference_R_pelvis @ anchor.sample.navigation_R_pelvis_imu.T
        current_R_pelvis = reference_R_navigation @ current.navigation_R_pelvis_imu

        pelvis_p_imu = self.pelvis_T_pelvis_imu[:3, 3]
        reference_p_imu = reference_T_pelvis[:3, 3] + reference_R_pelvis @ pelvis_p_imu
        current_p_pelvis = reference_p_imu - current_R_pelvis @ pelvis_p_imu
        reference_T_current_pelvis = _transform(current_R_pelvis, current_p_pelvis)
        return validate_transform(reference_T_current_pelvis @ current_pelvis_T_camera)

    def _fixed_torso_imu_origin_prediction(
        self,
        anchor: CameraPoseAnchor,
        current: ProprioceptiveSample,
    ) -> np.ndarray:
        reference_T_torso = anchor.reference_T_camera @ invert_transform(
            self.bundle.torso_T_camera
        )
        reference_R_torso = reference_T_torso[:3, :3]
        reference_R_navigation = reference_R_torso @ anchor.sample.navigation_R_torso_imu.T
        current_R_torso = reference_R_navigation @ current.navigation_R_torso_imu

        torso_p_imu = self.torso_T_torso_imu[:3, 3]
        reference_p_imu = reference_T_torso[:3, 3] + reference_R_torso @ torso_p_imu
        current_p_torso = reference_p_imu - current_R_torso @ torso_p_imu
        reference_T_current_torso = _transform(current_R_torso, current_p_torso)
        return validate_transform(reference_T_current_torso @ self.bundle.torso_T_camera)


ESTIMATOR_CONTRACTS: dict[EstimatorName, dict[str, object]] = {
    "fixed_camera": {
        "measurements": [],
        "assumption": "camera did not move after the visual anchor",
        "purpose": "null baseline",
    },
    "fixed_pelvis_fk": {
        "measurements": ["measured waist joints"],
        "assumption": "pelvis frame is fixed in the reference frame",
        "purpose": "isolate the correction available from waist encoders",
    },
    "fixed_torso_imu_origin": {
        "measurements": ["torso IMU orientation"],
        "assumption": "torso IMU origin is fixed; only its orientation changes",
        "purpose": "measure torso-orientation correction and camera lever-arm motion",
    },
    "fixed_pelvis_imu_origin_fk": {
        "measurements": ["pelvis IMU orientation", "measured waist joints"],
        "assumption": "pelvis IMU origin is fixed in the reference frame",
        "purpose": "contact-anchored pelvis orientation plus articulated waist FK",
    },
    "hybrid_pelvis_position_torso_orientation": {
        "measurements": [
            "pelvis IMU orientation",
            "measured waist joints",
            "torso IMU orientation",
        ],
        "assumption": (
            "pelvis IMU origin is fixed; camera position comes from pelvis/waist "
            "kinematics and camera orientation comes directly from the torso IMU"
        ),
        "purpose": "best observed rigid-chair endpoint hypothesis",
    },
}
