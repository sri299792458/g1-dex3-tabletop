import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
import pytest
from aprilcube import CorrespondenceDetector
from scipy.spatial.transform import Rotation

from g1_aprilcube_calibration.camera_models import RectifiedCameraInfo
from g1_aprilcube_calibration.table_accuracy import (
    CharucoBoardPoseDetector,
    CharucoBoardSpec,
    build_table_target,
    create_evaluation_document,
    detect_hand_target_pose,
    predicted_board_T_cube_from_model,
    task_error,
    validate_plan_document,
)
from g1_aprilcube_calibration.transforms import invert_transform

ROOT = Path(__file__).resolve().parents[1]


def _transform(translation=(0.0, 0.0, 0.0), rotation_vector=(0.0, 0.0, 0.0)) -> np.ndarray:
    result = np.eye(4)
    result[:3, :3] = Rotation.from_rotvec(rotation_vector).as_matrix()
    result[:3, 3] = translation
    return result


def _camera_info(width: int = 1000, height: int = 1100) -> RectifiedCameraInfo:
    return RectifiedCameraInfo(
        width=width,
        height=height,
        frame_id="camera_color_optical_frame",
        camera_name="synthetic",
        serial_number="TEST",
        distortion_model="plumb_bob",
        d=(0.0, 0.0, 0.0, 0.0, 0.0),
        k=(1000.0, 0.0, 500.0, 0.0, 1000.0, 550.0, 0.0, 0.0, 1.0),
        r=(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0),
        p=(
            1000.0,
            0.0,
            500.0,
            0.0,
            0.0,
            1000.0,
            550.0,
            0.0,
            0.0,
            0.0,
            1.0,
            0.0,
        ),
    )


def _hashed(document: dict) -> dict:
    result = dict(document)
    result["content_sha256"] = hashlib.sha256(
        json.dumps(result, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()
    return result


def _plan_document() -> dict:
    info = _camera_info()
    identity = np.eye(4).tolist()
    initial = _transform((0.0, 0.0, 0.1)).tolist()
    burst = {
        "aggregate": {
            "camera_T_board": identity,
            "board_T_hand_cube": initial,
        }
    }
    return _hashed(
        {
            "schema_version": 2,
            "kind": "g1_table_accuracy_plan",
            "sources": {
                "dataset_path": "/tmp/dataset.json",
                "dataset_sha256": "a" * 64,
                "result_path": "/tmp/result.json",
                "result_sha256": "b" * 64,
                "hand_cube_config_path": "/tmp/cube.json",
                "hand_cube_config_sha256": "c" * 64,
                "urdf_sha256": "d" * 64,
            },
            "calibration_arm": "left",
            "hand_link": "left_rubber_hand",
            "camera_info": info.to_dict(),
            "camera_profile_sha256": info.profile_sha256,
            "board_spec": CharucoBoardSpec().to_dict(),
            "calibration": {
                "torso_T_camera": identity,
                "hand_T_cube": identity,
            },
            "planning_burst": burst,
            "target": {
                "board_xy_mm": [0.0, 0.0],
                "lift_above_initial_cube_mm": 100.0,
                "initial_board_T_hand_cube": initial,
                "desired_board_T_hand_cube": identity,
                "desired_torso_T_hand": identity,
                "orientation_policy": "preserve_initial_board_T_hand_cube_orientation",
                "height_policy": "initial_board_z_minus_relative_lift",
                "board_z_convention": "negative_z_is_above_printed_face",
            },
            "execution": {
                "commands_robot": False,
                "requires_separate_ik_collision_and_motion_validation": True,
            },
        }
    )


def test_exact_printed_charuco_geometry_and_frame() -> None:
    spec = CharucoBoardSpec()

    assert spec.width_mm == 180.0
    assert spec.height_mm == 270.0
    assert spec.expected_marker_count == 27
    assert spec.expected_corner_count == 40
    assert spec.dictionary_name == "DICT_5X5_50"


def test_charuco_pose_detector_recovers_synthetic_front_view() -> None:
    detector = CharucoBoardPoseDetector()
    board_image = detector.board.generateImage((600, 900), marginSize=0, borderBits=1)
    image = np.full((1100, 1000), 255, dtype=np.uint8)
    image[100:1000, 200:800] = board_image

    estimate = detector.detect(cv2.cvtColor(image, cv2.COLOR_GRAY2BGR), _camera_info())

    expected = _transform((-0.09, -0.135, 0.3))
    assert np.allclose(estimate.camera_T_target, expected, atol=3e-5)
    assert estimate.reprojection_error_px < 0.01
    assert estimate.point_count == 40
    assert estimate.marker_ids == tuple(range(27))


def test_dex3_single_planar_marker_detector_recovers_front_view() -> None:
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_50)
    marker = cv2.aruco.generateImageMarker(dictionary, 4, 400, borderBits=1)
    image = np.full((1100, 1000), 255, dtype=np.uint8)
    image[350:750, 300:700] = marker
    detector = CorrespondenceDetector(
        ROOT / "config/dex3_dorsal_aruco_target.json",
        preprocess=False,
    )

    estimate = detect_hand_target_pose(
        cv2.cvtColor(image, cv2.COLOR_GRAY2BGR),
        _camera_info(),
        detector,
    )

    expected = _transform((0.0, 0.0, 0.1), (np.pi, 0.0, 0.0))
    assert np.allclose(estimate.camera_T_target, expected, atol=1e-4)
    assert estimate.reprojection_error_px < 0.01
    assert estimate.point_count == 4
    assert estimate.marker_ids == (4,)
    assert estimate.visible_faces == ("dorsal",)


def test_target_transform_equation_and_relative_lift_axis() -> None:
    torso_T_camera = _transform((0.05, 0.01, 0.42), (0.1, -0.2, 0.05))
    hand_T_cube = _transform((0.03, -0.01, 0.02), (0.2, 0.1, -0.1))
    camera_T_board = _transform((-0.2, 0.1, 0.8), (0.3, -0.1, 0.2))
    current_board_T_cube = _transform((0.03, 0.04, -0.2), (0.4, 0.2, -0.3))

    desired_board_T_cube, desired_torso_T_hand = build_table_target(
        torso_T_camera=torso_T_camera,
        hand_T_cube=hand_T_cube,
        camera_T_board=camera_T_board,
        current_board_T_cube=current_board_T_cube,
        board_spec=CharucoBoardSpec(),
        lift_mm=100.0,
    )

    assert np.allclose(desired_board_T_cube[:3, 3], [0.0, 0.0, -0.3])
    assert np.allclose(desired_board_T_cube[:3, :3], current_board_T_cube[:3, :3])
    assert np.allclose(
        desired_torso_T_hand @ hand_T_cube,
        torso_T_camera @ camera_T_board @ desired_board_T_cube,
    )
    assert np.allclose(
        invert_transform(camera_T_board)
        @ invert_transform(torso_T_camera)
        @ desired_torso_T_hand
        @ hand_T_cube,
        desired_board_T_cube,
    )


def test_task_error_reports_board_axis_millimetres() -> None:
    desired = _transform((0.09, 0.135, -0.1))
    actual = _transform((0.093, 0.131, -0.088), (0.0, 0.0, np.deg2rad(2.0)))

    error = task_error(desired, actual)

    assert error["translation_error_board_xyz_mm"] == pytest.approx([3.0, -4.0, 12.0])
    assert error["planar_xy_error_mm"] == pytest.approx(5.0)
    assert error["translation_error_norm_mm"] == pytest.approx(13.0)
    assert error["orientation_error_deg"] == pytest.approx(2.0)


def test_plan_hash_tampering_is_rejected() -> None:
    plan = _plan_document()
    validate_plan_document(plan)

    plan["target"]["desired_board_T_hand_cube"][0][3] = 0.001

    with pytest.raises(ValueError, match="SHA-256"):
        validate_plan_document(plan)


def test_evaluation_does_not_use_calibrated_extrinsics() -> None:
    plan = _plan_document()
    achieved = {
        "aggregate": {
            "camera_T_board": _transform((0.01, 0.0, 0.0)).tolist(),
            "board_T_hand_cube": _transform((0.003, -0.004, 0.012)).tolist(),
            "relative_translation_spread_mm": 3.5,
            "relative_rotation_spread_deg": 0.5,
        }
    }

    report = create_evaluation_document(plan=plan, achieved_burst=achieved)

    assert report["task_error"]["translation_error_norm_mm"] == pytest.approx(13.0)
    assert report["camera_motion_between_bursts"]["translation_mm"] == pytest.approx(10.0)
    assert set(report) == {
        "schema_version",
        "kind",
        "plan_sha256",
        "achieved_burst",
        "desired_board_T_hand_cube",
        "actual_board_T_hand_cube",
        "task_error",
        "measurement_quality",
        "camera_motion_between_bursts",
        "content_sha256",
    }
    assert not report["measurement_quality"]["passed"]


def test_model_diagnostic_is_separate_from_direct_task_error() -> None:
    plan = _plan_document()
    achieved = {
        "aggregate": {
            "camera_T_board": np.eye(4).tolist(),
            "board_T_hand_cube": _transform((0.012, 0.0, 0.0)).tolist(),
            "relative_translation_spread_mm": 1.0,
            "relative_rotation_spread_deg": 0.2,
        }
    }
    predicted = _transform((0.005, 0.0, 0.0))

    report = create_evaluation_document(
        plan=plan,
        achieved_burst=achieved,
        predicted_board_T_cube=predicted,
    )

    assert report["task_error"]["translation_error_norm_mm"] == pytest.approx(12.0)
    assert report["model_implied_tracking_error"]["translation_error_norm_mm"] == pytest.approx(
        5.0
    )
    assert report["held_out_composite_model_error"]["translation_error_norm_mm"] == pytest.approx(
        7.0
    )


def test_model_prediction_uses_measured_fk_and_calibrated_chain() -> None:
    torso_T_camera = _transform((0.1, 0.0, 0.2))
    camera_T_board = _transform((0.0, 0.0, 0.5))
    torso_T_hand = _transform((0.2, 0.1, 0.3))
    hand_T_cube = _transform((0.03, 0.0, 0.0))

    predicted = predicted_board_T_cube_from_model(
        torso_T_camera=torso_T_camera,
        hand_T_cube=hand_T_cube,
        camera_T_board=camera_T_board,
        torso_T_hand_at_measured_q=torso_T_hand,
    )

    assert np.allclose(
        camera_T_board @ predicted,
        invert_transform(torso_T_camera) @ torso_T_hand @ hand_T_cube,
    )
