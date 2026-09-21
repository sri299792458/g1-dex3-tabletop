"""Stock Ferguson optimizer mapping for same-frame bilateral observations."""

from __future__ import annotations

import math
import warnings
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation

from g1_aprilcube_calibration.calibration_evaluation import CAMERA_MOUNT_JOINT
from g1_aprilcube_calibration.joint_map import G1_29_JOINT_NAMES, arm_hand_link
from g1_aprilcube_calibration.transforms import transform_points, validate_transform
from g1_dex3_tabletop.calibration.models import (
    BilateralCalibrationSample,
    BilateralModelSpec,
    CameraFrameArtifact,
    TargetObservation,
)

CAMERA_OPTICAL_FRAME = "camera_color_optical_frame"
CAMERA_PARAMETER_NAME = "camera"
CAMERA_COLOR_JOINT = "camera_color_joint"
CAMERA_OPTICAL_JOINT = "camera_color_optical_joint"
_SIDES = ("left", "right")
_CAMERA_COMPONENTS = ("x", "y", "z", "roll", "pitch", "yaw")


def _target_frame(side: str) -> str:
    return f"{side}_calibration_target"


def _camera_model(side: str) -> str:
    return f"{side}_camera"


def _arm_model(side: str) -> str:
    return f"{side}_arm"


def _target_initial_values(transform: np.ndarray) -> dict[str, float]:
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Gimbal lock detected.*")
        rpy = Rotation.from_matrix(transform[:3, :3].copy()).as_euler("xyz")
    return {
        "x": float(transform[0, 3]),
        "y": float(transform[1, 3]),
        "z": float(transform[2, 3]),
        "roll": float(rpy[0]),
        "pitch": float(rpy[1]),
        "yaw": float(rpy[2]),
    }


def add_measured_color_frames(
    urdf_text: str,
    *,
    camera_frames: CameraFrameArtifact,
) -> str:
    """Append the measured serial-specific RGB and optical fixed joints."""

    root = ET.fromstring(urdf_text)
    if root.tag != "robot":
        raise ValueError("URDF root must be <robot>")
    link_names = {element.attrib["name"] for element in root.findall("link")}
    joint_names = {element.attrib["name"] for element in root.findall("joint")}
    if camera_frames.urdf_parent_link not in link_names:
        raise ValueError(
            f"URDF does not contain camera parent link {camera_frames.urdf_parent_link}"
        )
    new_links = (camera_frames.color_frame, camera_frames.optical_frame)
    new_joints = (CAMERA_COLOR_JOINT, CAMERA_OPTICAL_JOINT)
    conflicts = sorted((set(new_links) & link_names) | (set(new_joints) & joint_names))
    if conflicts:
        raise ValueError("URDF already contains measured camera frames: " + ", ".join(conflicts))

    def append_fixed_joint(
        *,
        name: str,
        parent: str,
        child: str,
        parent_T_child: np.ndarray,
    ) -> None:
        ET.SubElement(root, "link", {"name": child})
        joint = ET.SubElement(root, "joint", {"name": name, "type": "fixed"})
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="Gimbal lock detected.*")
            rpy = Rotation.from_matrix(parent_T_child[:3, :3]).as_euler("xyz")
        ET.SubElement(
            joint,
            "origin",
            {
                "xyz": " ".join(f"{value:.17g}" for value in parent_T_child[:3, 3]),
                "rpy": " ".join(f"{value:.17g}" for value in rpy),
            },
        )
        ET.SubElement(joint, "parent", {"link": parent})
        ET.SubElement(joint, "child", {"link": child})

    append_fixed_joint(
        name=CAMERA_COLOR_JOINT,
        parent=camera_frames.urdf_parent_link,
        child=camera_frames.color_frame,
        parent_T_child=np.asarray(camera_frames.parent_T_color),
    )
    append_fixed_joint(
        name=CAMERA_OPTICAL_JOINT,
        parent=camera_frames.color_frame,
        child=camera_frames.optical_frame,
        parent_T_child=np.asarray(camera_frames.color_T_optical),
    )
    return ET.tostring(root, encoding="unicode") + "\n"


def build_bilateral_optimizer_config(
    *,
    model: BilateralModelSpec,
    sample_count: int,
    initial_hand_T_targets: dict[str, np.ndarray],
) -> dict[str, Any]:
    """Build a two-arm/two-camera-alias configuration for stock Ferguson."""

    if sample_count <= 0:
        raise ValueError("bilateral optimizer requires at least one sample")
    if set(initial_hand_T_targets) != set(_SIDES):
        raise ValueError("initial hand-target transforms must contain left and right")
    targets = {side: validate_transform(initial_hand_T_targets[side]) for side in _SIDES}
    model_names = [
        _arm_model("left"),
        _arm_model("right"),
        _camera_model("left"),
        _camera_model("right"),
    ]
    step: dict[str, Any] = {
        "max_num_iterations": 1000,
        "models": model_names,
        "free_frames": [CAMERA_MOUNT_JOINT],
        CAMERA_MOUNT_JOINT: {
            component: component in model.camera_components for component in _CAMERA_COMPONENTS
        },
        "error_blocks": ["left_reprojection", "right_reprojection"],
    }
    for side in _SIDES:
        step[_arm_model(side)] = {
            "type": "chain3d",
            "frame": arm_hand_link(side),
        }
        # Sensor names must differ because the two pixel feature arrays differ.
        # The shared frame and param_name retain one camera parameter block.
        step[_camera_model(side)] = {
            "type": "camera2d",
            "frame": CAMERA_OPTICAL_FRAME,
            "param_name": CAMERA_PARAMETER_NAME,
        }
        step[f"{side}_reprojection"] = {
            "type": "chain3d_to_camera2d",
            "model_3d": _arm_model(side),
            "model_2d": _camera_model(side),
            "scale": 1.0,
        }
    if model.optimize_hand_targets:
        step["free_frames_initial_values"] = []
        for side in _SIDES:
            frame = _target_frame(side)
            step["free_frames"].append(frame)
            step[frame] = {
                "x": True,
                "y": True,
                "z": True,
                "roll": True,
                "pitch": True,
                "yaw": True,
            }
            step["free_frames_initial_values"].append(frame)
            step[f"{frame}_initial_values"] = _target_initial_values(targets[side])
    if model.joint_offsets:
        step["free_params"] = list(model.joint_offsets)
        sigma_rad = math.radians(model.joint_offset_prior_sigma_deg)
        joint_scale = 1.0 / (sigma_rad * math.sqrt(sample_count))
        for index, joint_name in enumerate(model.joint_offsets):
            name = f"joint_offset_prior_{index:02d}"
            step["error_blocks"].append(name)
            step[name] = {
                "type": "outrageous",
                "param": joint_name,
                "joint_scale": joint_scale,
                "position_scale": 0.0,
                "rotation_scale": 0.0,
            }
    return {
        "robot_calibration": {
            "ros__parameters": {
                "verbose": True,
                "base_link": "torso_link",
                "calibration_steps": ["bilateral_calibration"],
                "bilateral_calibration": step,
            }
        }
    }


@dataclass(frozen=True, slots=True)
class FergusonObservation:
    sensor_name: str
    feature_frame: str
    points: tuple[tuple[float, float, float], ...]
    camera_info: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        if not self.sensor_name or not self.feature_frame:
            raise ValueError("Ferguson observation names must be non-empty")
        points = np.asarray(self.points, dtype=np.float64)
        if points.ndim != 2 or points.shape[1:] != (3,) or not np.all(np.isfinite(points)):
            raise ValueError("Ferguson observation points must have finite shape (N, 3)")
        if len(points) < 4:
            raise ValueError("Ferguson observation requires at least four points")
        object.__setattr__(
            self,
            "points",
            tuple(tuple(float(value) for value in point) for point in points),
        )
        if self.camera_info is not None:
            object.__setattr__(self, "camera_info", dict(self.camera_info))


@dataclass(frozen=True, slots=True)
class BilateralFergusonRecord:
    capture_id: str
    frame_id: str
    joint_names: tuple[str, ...]
    joint_positions: tuple[float, ...]
    observations: tuple[FergusonObservation, ...]
    stamp_ns: int | None = None

    def __post_init__(self) -> None:
        if not self.capture_id or not self.frame_id:
            raise ValueError("bilateral Ferguson record IDs must be non-empty")
        if self.joint_names != G1_29_JOINT_NAMES:
            raise ValueError("bilateral Ferguson record must use G1 mode-5 joint order")
        positions = np.asarray(self.joint_positions, dtype=np.float64).reshape(-1)
        if positions.shape != (len(G1_29_JOINT_NAMES),) or not np.all(np.isfinite(positions)):
            raise ValueError("bilateral Ferguson record has invalid joint positions")
        observations = tuple(self.observations)
        expected = ("left_arm", "left_camera", "right_arm", "right_camera")
        if tuple(item.sensor_name for item in observations) != expected:
            raise ValueError("bilateral Ferguson record must contain four ordered sensors")
        if len(observations[0].points) != len(observations[1].points):
            raise ValueError("left Ferguson feature counts differ")
        if len(observations[2].points) != len(observations[3].points):
            raise ValueError("right Ferguson feature counts differ")
        if self.stamp_ns is not None and self.stamp_ns < 0:
            raise ValueError("bilateral Ferguson timestamp must be non-negative")
        object.__setattr__(self, "joint_positions", tuple(float(value) for value in positions))
        object.__setattr__(self, "observations", observations)


def _side_observations(
    sample: BilateralCalibrationSample,
    observation: TargetObservation,
    *,
    optimize_hand_targets: bool,
    initial_hand_T_target: np.ndarray,
) -> tuple[FergusonObservation, FergusonObservation]:
    object_points = np.asarray(observation.object_points_m, dtype=np.float64)
    feature_frame = _target_frame(observation.side)
    if not optimize_hand_targets:
        object_points = transform_points(initial_hand_T_target, object_points)
        feature_frame = arm_hand_link(observation.side)
    arm = FergusonObservation(
        sensor_name=_arm_model(observation.side),
        feature_frame=feature_frame,
        points=tuple(tuple(float(value) for value in point) for point in object_points),
    )
    pixels = tuple(
        (float(point[0]), float(point[1]), 0.0) for point in observation.image_points_px
    )
    camera = FergusonObservation(
        sensor_name=_camera_model(observation.side),
        feature_frame=CAMERA_OPTICAL_FRAME,
        points=pixels,
        camera_info=sample.camera_info,
    )
    return arm, camera


def sample_to_ferguson_record(
    sample: BilateralCalibrationSample,
    *,
    model: BilateralModelSpec,
    initial_hand_T_targets: dict[str, np.ndarray],
) -> BilateralFergusonRecord:
    """Map one simultaneous frame to four unambiguous sensor observations."""

    if set(initial_hand_T_targets) != set(_SIDES):
        raise ValueError("initial hand-target transforms must contain left and right")
    targets = {side: validate_transform(initial_hand_T_targets[side]) for side in _SIDES}
    observations: list[FergusonObservation] = []
    for target_observation in sample.observations:
        observations.extend(
            _side_observations(
                sample,
                target_observation,
                optimize_hand_targets=model.optimize_hand_targets,
                initial_hand_T_target=targets[target_observation.side],
            )
        )
    return BilateralFergusonRecord(
        capture_id=sample.capture_id,
        frame_id=sample.frame_id,
        joint_names=G1_29_JOINT_NAMES,
        joint_positions=sample.joint_positions_rad,
        observations=tuple(observations),
        stamp_ns=_sample_stamp_ns(sample),
    )


def _sample_stamp_ns(sample: BilateralCalibrationSample) -> int | None:
    value = sample.pairing.get("image_header_stamp_ns")
    if value is None:
        value = sample.pairing.get("image", {}).get("header_stamp_ns")
    if value is None:
        return None
    stamp = int(value)
    return stamp if stamp > 0 else None
