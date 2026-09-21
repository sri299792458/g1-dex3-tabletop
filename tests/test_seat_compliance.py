from __future__ import annotations

import cv2
import numpy as np

from g1_aprilcube_calibration.camera_models import RectifiedCameraInfo
from g1_aprilcube_calibration.table_accuracy import CharucoBoardPoseDetector
from g1_dex3_tabletop.planning.contracts import RobotSnapshot
from g1_dex3_tabletop.planning.tabletop_planner import _table_from_charuco_board
from g1_dex3_tabletop.seat_compliance import (
    camera_motion_from_fixed_board,
    observe_charuco_board,
    summarize_camera_motion_cycles,
)
from g1_dex3_tabletop.tabletop_contracts import (
    CharucoBoardObservation,
    CharucoSupportedEscapeRequest,
)


def _camera_info() -> RectifiedCameraInfo:
    return RectifiedCameraInfo(
        width=1000,
        height=1100,
        frame_id="camera_color_optical_frame",
        camera_name="synthetic",
        serial_number="TEST",
        distortion_model="plumb_bob",
        d=(0.0,) * 5,
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


def _board_images() -> list[np.ndarray]:
    detector = CharucoBoardPoseDetector()
    board = detector.board.generateImage((600, 900), marginSize=0, borderBits=1)
    gray = np.full((1100, 1000), 255, dtype=np.uint8)
    gray[100:1000, 200:800] = board
    result = []
    for index in range(3):
        image = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        image[index, index] = index
        result.append(image)
    return result


def test_board_burst_is_aggregated_without_a_spread_gate() -> None:
    observation, evidence = observe_charuco_board(
        _board_images(),
        camera_info=_camera_info(),
        snapshot=RobotSnapshot((0.0,) * 29, (0.0,) * 7, (0.0,) * 7),
    )

    assert evidence["accepted_frame_count"] == 3
    assert evidence["rejected_frames"] == []
    assert observation.board_spec["dictionary_name"] == "DICT_5X5_50"
    assert np.allclose(
        np.asarray(observation.camera_T_board)[:3, 3],
        [-0.09, -0.135, 0.3],
        atol=3e-5,
    )
    assert "recorded_not_gated" in evidence["spread_policy"]


def test_fixed_board_reports_camera_translation_in_board_coordinates() -> None:
    baseline = np.eye(4)
    current = np.eye(4)
    current[0, 3] = -0.012

    motion = camera_motion_from_fixed_board(baseline, current)

    assert np.allclose(motion["translation_board_xyz_mm"], [12.0, 0.0, 0.0])
    assert np.isclose(motion["translation_norm_mm"], 12.0)
    assert motion["rotation_deg"] == 0.0


def test_motion_summary_uses_explicit_same_cycle_pre_lift_reference() -> None:
    loaded = np.eye(4)
    loaded[0, 3] = 0.100
    pre_lift = np.eye(4)
    lifted = np.eye(4)
    lifted[0, 3] = -0.010
    returned = np.eye(4)
    returned[0, 3] = -0.001

    def event(phase: str, transform: np.ndarray, *, arm: str | None = None) -> dict:
        result = {
            "phase": phase,
            "board": {"aggregate": {"camera_T_board": transform.tolist()}},
        }
        if arm is not None:
            result.update({"repetition": 1, "arm": arm})
        return result

    summary = summarize_camera_motion_cycles(
        [
            event("loaded_baseline", loaded),
            event("pre_lift", pre_lift, arm="left"),
            event("lifted", lifted, arm="left"),
            event("returned", returned, arm="left"),
        ]
    )

    assert summary["reference_policy"] == "explicit_same_cycle_pre_lift"
    assert summary["left"]["cycle_count"] == 1
    assert summary["right"]["cycle_count"] == 0
    assert np.allclose(summary["left"]["pre_lift_to_lifted"]["translation_norm_mm"], [10.0])
    assert np.allclose(summary["left"]["pre_lift_to_returned"]["translation_norm_mm"], [1.0])
    assert np.allclose(summary["left"]["lifted_to_returned"]["translation_norm_mm"], [9.0])


def test_charuco_plane_uses_printed_board_positive_z_as_down() -> None:
    camera_T_board = np.eye(4)
    camera_T_board[:3, :3] = np.diag([1.0, -1.0, -1.0])
    camera_T_board[:3, 3] = [0.2, -0.1, 0.7]
    observation = CharucoBoardObservation(
        snapshot=RobotSnapshot((0.0,) * 29, (0.0,) * 7, (0.0,) * 7),
        camera_T_board=tuple(tuple(value for value in row) for row in camera_T_board),
        camera_profile_sha256="a" * 64,
        source_frame_sha256=("b" * 64, "c" * 64, "d" * 64),
        translation_spread_mm=0.2,
        rotation_spread_deg=0.1,
        board_spec=CharucoBoardPoseDetector().spec.to_dict(),
    )
    request = CharucoSupportedEscapeRequest(
        observation=observation,
        arm="left",
        torso_T_camera=tuple(tuple(value for value in row) for row in np.eye(4)),
        joint_position_offsets_rad={},
        calibration_bundle_sha256="e" * 64,
    )

    point, down = _table_from_charuco_board(request, np.eye(4))

    assert np.allclose(point, [0.2, -0.1, 0.7])
    assert np.allclose(down, [0.0, 0.0, -1.0])
