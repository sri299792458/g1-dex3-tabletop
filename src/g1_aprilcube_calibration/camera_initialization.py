"""Nominal RealSense optical-frame initial values from the official G1 URDF."""

from __future__ import annotations

import cv2
import numpy as np

from g1_aprilcube_calibration.camera_models import RectifiedCameraInfo
from g1_aprilcube_calibration.dataset_builder import CalibrationSample
from g1_aprilcube_calibration.joint_map import G1_29_JOINT_NAMES, arm_hand_link
from g1_aprilcube_calibration.transforms import invert_transform
from g1_aprilcube_calibration.urdf_model import URDFModel


def realsense_link_T_color_optical() -> np.ndarray:
    """ROS camera-link axes to REP-103 optical axes, as a parent-from-child matrix."""
    result = np.eye(4)
    result[:3, :3] = np.asarray(
        [
            [0.0, 0.0, 1.0],
            [-1.0, 0.0, 0.0],
            [0.0, -1.0, 0.0],
        ]
    )
    return result


def nominal_torso_T_color_optical(model: URDFModel) -> np.ndarray:
    return model.transform("torso_link", "d435_link", {}) @ realsense_link_T_color_optical()


def estimate_hand_T_target_from_sample(
    model: URDFModel,
    sample: CalibrationSample,
    *,
    initial_torso_T_camera: np.ndarray,
    calibration_arm: str = "left",
) -> np.ndarray:
    """Use one rectified-frame PnP result only to initialize the batch solve."""
    object_points = np.asarray(sample.object_points_m, dtype=np.float64)
    image_points = np.asarray(sample.image_points_px, dtype=np.float64)
    if len(object_points) < 4:
        raise ValueError("at least four correspondences are required for PnP initialization")
    camera_info = RectifiedCameraInfo.from_dict(sample.camera_info)
    success, rotation_vector, translation_vector = cv2.solvePnP(
        object_points,
        image_points,
        camera_info.rectified_camera_matrix,
        np.zeros(5),
        flags=cv2.SOLVEPNP_SQPNP,
    )
    if not success:
        raise RuntimeError("single-frame target PnP initialization failed")
    camera_T_target = np.eye(4)
    camera_T_target[:3, :3], _ = cv2.Rodrigues(rotation_vector)
    camera_T_target[:3, 3] = translation_vector.reshape(3)
    if camera_T_target[2, 3] <= 0:
        raise ValueError("PnP target initialization lies behind the camera")
    position = np.asarray(sample.measured_state["position"], dtype=np.float64)
    if position.shape != (29,):
        raise ValueError("PnP initialization sample has an invalid joint state")
    torso_T_hand = model.transform(
        "torso_link",
        arm_hand_link(calibration_arm),
        dict(zip(G1_29_JOINT_NAMES, position, strict=True)),
    )
    return invert_transform(torso_T_hand) @ initial_torso_T_camera @ camera_T_target
