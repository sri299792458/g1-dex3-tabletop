"""Synthetic mode-5 G1 corner datasets for pre-hardware recovery tests."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from aprilcube import CorrespondenceDetector

from g1_aprilcube_calibration.camera_models import RectifiedCameraInfo
from g1_aprilcube_calibration.dataset_builder import (
    CalibrationDataset,
    CalibrationSample,
)
from g1_aprilcube_calibration.joint_map import (
    G1_29_JOINT_NAMES,
    arm_hand_link,
    arm_indices,
    validate_arm_side,
)
from g1_aprilcube_calibration.transforms import (
    invert_transform,
    pose_vector_to_transform,
    transform_points,
)
from g1_aprilcube_calibration.urdf_model import URDFModel


@dataclass(frozen=True, slots=True)
class SyntheticTruth:
    torso_T_camera: np.ndarray
    hand_T_target: np.ndarray


def make_synthetic_dataset(
    model: URDFModel,
    target_config: str | Path,
    *,
    pose_count: int = 40,
    pixel_noise_stddev: float = 0.0,
    seed: int = 7,
    calibration_arm: str = "left",
) -> tuple[CalibrationDataset, SyntheticTruth]:
    if pose_count < 3:
        raise ValueError("synthetic dataset requires at least three poses")
    if pixel_noise_stddev < 0:
        raise ValueError("pixel noise standard deviation cannot be negative")
    calibration_arm = validate_arm_side(calibration_arm)
    hand_link = arm_hand_link(calibration_arm)
    calibration_indices = np.asarray(arm_indices(calibration_arm))
    detector = CorrespondenceDetector(target_config)
    tag_ids = tuple(sorted(detector.tag_corner_map))
    object_points = np.vstack([detector.tag_corner_map[tag_id] for tag_id in tag_ids]) / 1000.0
    corner_tag_ids = tuple(tag_id for tag_id in tag_ids for _ in range(4))
    camera_info = RectifiedCameraInfo(
        width=848,
        height=480,
        frame_id="synthetic_color_optical_frame",
        camera_name="synthetic_head_color",
        serial_number="SYNTHETIC",
        distortion_model="plumb_bob",
        d=(0.0,) * 5,
        k=(620.0, 0.0, 423.5, 0.0, 620.0, 239.5, 0.0, 0.0, 1.0),
        r=(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0),
        p=(620.0, 0.0, 423.5, 0.0, 0.0, 620.0, 239.5, 0.0, 0.0, 0.0, 1.0, 0.0),
    )
    hand_T_target = pose_vector_to_transform([0.075, 0.005, 0.045, 0.12, -0.08, 0.18])
    zero_positions = dict.fromkeys(G1_29_JOINT_NAMES, 0.0)
    torso_T_hand_zero = model.transform("torso_link", hand_link, zero_positions)
    target_center = transform_points(torso_T_hand_zero @ hand_T_target, np.zeros((1, 3)))[0]
    camera_position = np.asarray([0.0576, 0.0175, 0.4299])
    torso_T_camera = _look_at_transform(camera_position, target_center)
    truth = SyntheticTruth(torso_T_camera, hand_T_target)

    rng = np.random.default_rng(seed)
    samples: list[CalibrationSample] = []
    attempts = 0
    while len(samples) < pose_count and attempts < pose_count * 200:
        attempts += 1
        calibration = np.asarray(
            [
                rng.uniform(-0.55, 0.45),
                rng.uniform(-0.45, 0.45),
                rng.uniform(-0.7, 0.7),
                rng.uniform(-0.1, 0.9),
                rng.uniform(-0.5, 0.5),
                rng.uniform(-0.45, 0.45),
                rng.uniform(-0.5, 0.5),
            ]
        )
        full = np.zeros(29)
        full[calibration_indices] = calibration
        positions = dict(zip(G1_29_JOINT_NAMES, full, strict=True))
        torso_T_hand = model.transform("torso_link", hand_link, positions)
        torso_points = transform_points(torso_T_hand @ hand_T_target, object_points)
        camera_points = transform_points(invert_transform(torso_T_camera), torso_points)
        if np.min(camera_points[:, 2]) < 0.2:
            continue
        matrix = camera_info.rectified_camera_matrix
        pixels = np.column_stack(
            (
                matrix[0, 0] * camera_points[:, 0] / camera_points[:, 2] + matrix[0, 2],
                matrix[1, 1] * camera_points[:, 1] / camera_points[:, 2] + matrix[1, 2],
            )
        )
        if (
            np.min(pixels[:, 0]) < 15
            or np.max(pixels[:, 0]) >= camera_info.width - 15
            or np.min(pixels[:, 1]) < 15
            or np.max(pixels[:, 1]) >= camera_info.height - 15
        ):
            continue
        if pixel_noise_stddev:
            pixels = pixels + rng.normal(0.0, pixel_noise_stddev, pixels.shape)
        index = len(samples)
        digest = hashlib.sha256(f"synthetic-{seed}-{index}".encode()).hexdigest()
        samples.append(
            CalibrationSample(
                capture_id=f"synthetic_capture_{index:03d}",
                pose_id=f"synthetic_pose_{index:03d}",
                frame_id=f"synthetic_frame_{index:03d}",
                raw_image_path=f"synthetic/{index:03d}.png",
                raw_image_sha256=digest,
                camera_info=camera_info.to_dict(),
                measured_state={
                    "receipt_monotonic_s": float(index),
                    "receipt_utc": "2026-08-02T12:00:00Z",
                    "mode_machine": 5,
                    "position": full.tolist(),
                    "velocity": np.zeros(29).tolist(),
                    "source_sequence": index,
                },
                pairing={"nearest_delta_s": 0.0},
                visible_tag_ids=tag_ids,
                corner_tag_ids=corner_tag_ids,
                image_points_px=tuple(tuple(float(value) for value in point) for point in pixels),
                object_points_m=tuple(
                    tuple(float(value) for value in point) for point in object_points
                ),
                correspondence_sha256=digest,
            )
        )
    if len(samples) != pose_count:
        raise RuntimeError(
            f"could only generate {len(samples)}/{pose_count} in-frame synthetic poses"
        )
    dataset = CalibrationDataset(
        session_id=f"synthetic_{seed}",
        session_manifest_sha256=hashlib.sha256(f"manifest-{seed}".encode()).hexdigest(),
        target_artifact_sha256=hashlib.sha256(Path(target_config).read_bytes()).hexdigest(),
        pose_set_sha256=hashlib.sha256(f"pose-set-{seed}".encode()).hexdigest(),
        urdf_sha256=model.sha256,
        calibration_arm=calibration_arm,
        observation_phase="held",
        samples=tuple(samples),
    )
    return dataset, truth


def perturbed_initial_transforms(
    truth: SyntheticTruth,
) -> tuple[np.ndarray, np.ndarray]:
    camera_delta = pose_vector_to_transform([0.015, -0.012, 0.018, 0.04, -0.03, 0.025])
    target_delta = pose_vector_to_transform([-0.012, 0.01, -0.008, -0.035, 0.025, -0.03])
    return truth.torso_T_camera @ camera_delta, truth.hand_T_target @ target_delta


def _look_at_transform(position: np.ndarray, target: np.ndarray) -> np.ndarray:
    z_axis = target - position
    z_axis = z_axis / np.linalg.norm(z_axis)
    down_hint = np.asarray([0.0, 0.0, -1.0])
    x_axis = np.cross(down_hint, z_axis)
    x_axis = x_axis / np.linalg.norm(x_axis)
    y_axis = np.cross(z_axis, x_axis)
    result = np.eye(4)
    result[:3, :3] = np.column_stack((x_axis, y_axis, z_axis))
    result[:3, 3] = position
    return result
