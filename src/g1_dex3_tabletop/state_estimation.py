"""Pure anchored camera-pose estimation for the seated G1.

The estimator intentionally exposes its contact and observability assumptions.
It does not claim globally observable odometry and has no ROS, CuRobo, MPC, or
robot-command dependency.  Its input is deliberately limited to the three
waist joints, the pelvis and torso orientations, and a full visual anchor; it
accepts no additional sensor streams.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from scipy.spatial.transform import Rotation

from g1_aprilcube_calibration.calibration_bundle import CalibrationBundle
from g1_aprilcube_calibration.joint_map import G1_29_JOINT_NAMES
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
SELECTED_ESTIMATOR: EstimatorName = "hybrid_pelvis_position_torso_orientation"

WAIST_JOINT_NAMES: tuple[str, str, str] = (
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
)
WAIST_JOINT_INDICES: tuple[int, int, int] = tuple(
    G1_29_JOINT_NAMES.index(name) for name in WAIST_JOINT_NAMES
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


def _vector(value: np.ndarray, *, size: int, name: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64).reshape(-1).copy()
    if result.shape != (size,) or not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must contain {size} finite values")
    result.setflags(write=False)
    return result


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
    waist_q_rad: np.ndarray
    navigation_R_pelvis_imu: np.ndarray
    navigation_R_torso_imu: np.ndarray

    def __post_init__(self) -> None:
        if self.timestamp_ns < 0:
            raise ValueError("sample timestamp must be non-negative")
        object.__setattr__(
            self,
            "waist_q_rad",
            _vector(self.waist_q_rad, size=3, name="waist joint vector"),
        )
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


@dataclass(frozen=True, slots=True)
class CameraStateEstimate:
    """One propagated camera-pose estimate and its explicit provenance."""

    timestamp_ns: int
    anchor_timestamp_ns: int
    reference_T_camera: np.ndarray

    def __post_init__(self) -> None:
        if self.timestamp_ns < self.anchor_timestamp_ns:
            raise ValueError("estimate predates its visual anchor")
        object.__setattr__(
            self,
            "reference_T_camera",
            validate_transform(np.asarray(self.reference_T_camera, dtype=np.float64)),
        )

    @property
    def anchor_age_s(self) -> float:
        return (self.timestamp_ns - self.anchor_timestamp_ns) / 1.0e9

    def to_dict(self) -> dict[str, object]:
        return {
            "timestamp_ns": self.timestamp_ns,
            "anchor_timestamp_ns": self.anchor_timestamp_ns,
            "anchor_age_s": self.anchor_age_s,
            "reference_T_camera": self.reference_T_camera.tolist(),
            "estimator": SELECTED_ESTIMATOR,
        }


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
            for name, value in zip(WAIST_JOINT_NAMES, sample.waist_q_rad, strict=True)
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


class AnchoredCameraStateEstimator:
    """The one retained hybrid observer, independent of any control consumer.

    A full six-degree-of-freedom visual measurement establishes the reference
    frame.  Between visual updates, position is propagated with the measured
    pelvis orientation and three waist joints, while orientation comes from the
    torso IMU.  The fixed-pelvis-IMU-origin assumption is intentionally not
    hidden behind a generic odometry interface.
    """

    def __init__(self, pose_estimators: AnchoredCameraPoseEstimators) -> None:
        self._pose_estimators = pose_estimators
        self._anchor: CameraPoseAnchor | None = None

    @property
    def anchor(self) -> CameraPoseAnchor | None:
        return self._anchor

    def reset(self, anchor: CameraPoseAnchor) -> CameraStateEstimate:
        """Install one synchronized full-pose visual anchor."""

        self._anchor = anchor
        return CameraStateEstimate(
            timestamp_ns=anchor.sample.timestamp_ns,
            anchor_timestamp_ns=anchor.sample.timestamp_ns,
            reference_T_camera=anchor.reference_T_camera,
        )

    def clear(self) -> None:
        """Discard the external reference; estimates are impossible afterward."""

        self._anchor = None

    def estimate(self, sample: ProprioceptiveSample) -> CameraStateEstimate:
        """Propagate the current camera pose from the latest visual anchor."""

        anchor = self._anchor
        if anchor is None:
            raise RuntimeError("camera estimator has no visual anchor")
        if sample.timestamp_ns < anchor.sample.timestamp_ns:
            raise ValueError("camera-state sample predates the visual anchor")
        prediction = self._pose_estimators.predict(
            anchor,
            sample,
            SELECTED_ESTIMATOR,
        )
        return CameraStateEstimate(
            timestamp_ns=sample.timestamp_ns,
            anchor_timestamp_ns=anchor.sample.timestamp_ns,
            reference_T_camera=prediction,
        )


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
