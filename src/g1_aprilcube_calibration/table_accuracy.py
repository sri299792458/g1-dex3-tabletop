"""Independent tabletop task-accuracy planning and measurement.

The calibration result is used only to construct a desired torso-from-hand
transform.  The achieved error is measured directly between a fixed ChArUco
board and the hand-mounted cube, so evaluation does not reuse the calibrated
camera or hand extrinsics.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from aprilcube import (
    CorrespondenceDetector,
    CorrespondenceResult,
    estimate_pose_diagnostic,
)
from scipy.spatial.transform import Rotation

from g1_aprilcube_calibration.camera_models import RectifiedCameraInfo
from g1_aprilcube_calibration.dataset_builder import CalibrationDataset
from g1_aprilcube_calibration.joint_map import arm_hand_link
from g1_aprilcube_calibration.residual_report import load_exported_result
from g1_aprilcube_calibration.transforms import invert_transform, validate_transform

PLAN_SCHEMA_VERSION = 2
EVALUATION_SCHEMA_VERSION = 2
DEFAULT_MAXIMUM_RELATIVE_TRANSLATION_SPREAD_MM = 3.0
DEFAULT_MAXIMUM_RELATIVE_ROTATION_SPREAD_DEG = 1.0


def _canonical_sha256(document: dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(document, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _transform_to_list(transform: np.ndarray) -> list[list[float]]:
    return validate_transform(transform).tolist()


def _transform_from_document(data: Any, *, name: str) -> np.ndarray:
    try:
        return validate_transform(np.asarray(data, dtype=np.float64))
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} is not a valid rigid transform: {error}") from error


def _rotation_error_deg(first: np.ndarray, second: np.ndarray) -> float:
    relative = first[:3, :3].T @ second[:3, :3]
    return float(np.degrees(Rotation.from_matrix(relative.copy()).magnitude()))


def _average_transforms(transforms: Sequence[np.ndarray]) -> np.ndarray:
    if not transforms:
        raise ValueError("cannot average an empty transform sequence")
    matrices = [validate_transform(item) for item in transforms]
    result = np.eye(4)
    result[:3, :3] = (
        Rotation.from_matrix(np.stack([item[:3, :3] for item in matrices])).mean().as_matrix()
    )
    result[:3, 3] = np.median(np.stack([item[:3, 3] for item in matrices]), axis=0)
    return validate_transform(result)


def _transform_spread(transforms: Sequence[np.ndarray], center: np.ndarray) -> tuple[float, float]:
    translations_mm = [
        1000.0 * float(np.linalg.norm(item[:3, 3] - center[:3, 3])) for item in transforms
    ]
    rotations_deg = [_rotation_error_deg(center, item) for item in transforms]
    return max(translations_mm, default=0.0), max(rotations_deg, default=0.0)


@dataclass(frozen=True, slots=True)
class CharucoBoardSpec:
    """Exact geometry of the rigid table reference.

    OpenCV's board frame starts at the upper-left outer board corner when the
    print is viewed upright.  +X follows the six-square edge, +Y follows the
    nine-square edge, and +Z points into the backing.  A point above a
    face-up board therefore has a negative Z coordinate.
    """

    squares_x: int = 6
    squares_y: int = 9
    square_length_mm: float = 30.0
    marker_length_mm: float = 22.0
    dictionary_name: str = "DICT_5X5_50"
    legacy_pattern: bool = False

    def __post_init__(self) -> None:
        if self.squares_x < 2 or self.squares_y < 2:
            raise ValueError("ChArUco board must have at least 2x2 squares")
        if self.square_length_mm <= 0 or self.marker_length_mm <= 0:
            raise ValueError("ChArUco dimensions must be positive")
        if self.marker_length_mm >= self.square_length_mm:
            raise ValueError("ChArUco marker must be smaller than its square")
        if self.dictionary_name != "DICT_5X5_50":
            raise ValueError("table test requires the frozen DICT_5X5_50 ChArUco board")

    @property
    def width_mm(self) -> float:
        return self.squares_x * self.square_length_mm

    @property
    def height_mm(self) -> float:
        return self.squares_y * self.square_length_mm

    @property
    def expected_marker_count(self) -> int:
        return (self.squares_x * self.squares_y) // 2

    @property
    def expected_corner_count(self) -> int:
        return (self.squares_x - 1) * (self.squares_y - 1)

    def to_dict(self) -> dict[str, Any]:
        return {
            "squares_x": self.squares_x,
            "squares_y": self.squares_y,
            "square_length_mm": self.square_length_mm,
            "marker_length_mm": self.marker_length_mm,
            "dictionary_name": self.dictionary_name,
            "legacy_pattern": self.legacy_pattern,
            "active_dimensions_mm": [self.width_mm, self.height_mm],
            "marker_count": self.expected_marker_count,
            "charuco_corner_count": self.expected_corner_count,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CharucoBoardSpec:
        spec = cls(
            squares_x=int(data["squares_x"]),
            squares_y=int(data["squares_y"]),
            square_length_mm=float(data["square_length_mm"]),
            marker_length_mm=float(data["marker_length_mm"]),
            dictionary_name=str(data["dictionary_name"]),
            legacy_pattern=bool(data["legacy_pattern"]),
        )
        if data != spec.to_dict():
            raise ValueError("ChArUco board specification contains inconsistent fields")
        return spec


@dataclass(frozen=True, slots=True)
class TargetPoseEstimate:
    camera_T_target: np.ndarray
    reprojection_error_px: float
    point_count: int
    marker_ids: tuple[int, ...]
    visible_faces: tuple[str, ...] = ()
    second_solution_error_px: float | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "camera_T_target", validate_transform(self.camera_T_target))
        if self.reprojection_error_px < 0 or not np.isfinite(self.reprojection_error_px):
            raise ValueError("reprojection error must be finite and non-negative")
        if self.point_count < 4:
            raise ValueError("pose estimate must contain at least four points")

    def to_dict(self) -> dict[str, Any]:
        return {
            "camera_T_target": _transform_to_list(self.camera_T_target),
            "reprojection_error_px": self.reprojection_error_px,
            "second_solution_error_px": self.second_solution_error_px,
            "point_count": self.point_count,
            "marker_ids": list(self.marker_ids),
            "visible_faces": list(self.visible_faces),
        }


class CharucoBoardPoseDetector:
    """Stateless pose detector for the exact printed table board."""

    def __init__(
        self,
        spec: CharucoBoardSpec | None = None,
        *,
        minimum_markers: int = 6,
        minimum_corners: int = 12,
        maximum_reprojection_error_px: float = 1.5,
    ) -> None:
        self.spec = spec or CharucoBoardSpec()
        if minimum_markers < 2 or minimum_corners < 4:
            raise ValueError("ChArUco minimum counts are too small")
        if maximum_reprojection_error_px <= 0:
            raise ValueError("maximum reprojection error must be positive")
        self.minimum_markers = int(minimum_markers)
        self.minimum_corners = int(minimum_corners)
        self.maximum_reprojection_error_px = float(maximum_reprojection_error_px)
        dictionary_id = getattr(cv2.aruco, self.spec.dictionary_name)
        self.dictionary = cv2.aruco.getPredefinedDictionary(dictionary_id)
        self.board = cv2.aruco.CharucoBoard(
            (self.spec.squares_x, self.spec.squares_y),
            self.spec.square_length_mm,
            self.spec.marker_length_mm,
            self.dictionary,
        )
        if hasattr(self.board, "setLegacyPattern"):
            self.board.setLegacyPattern(self.spec.legacy_pattern)
        parameters = cv2.aruco.DetectorParameters()
        parameters.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
        self.detector = cv2.aruco.CharucoDetector(
            self.board,
            cv2.aruco.CharucoParameters(),
            parameters,
        )

    def detect(
        self, image_bgr: np.ndarray, camera_info: RectifiedCameraInfo
    ) -> TargetPoseEstimate:
        image = np.asarray(image_bgr)
        expected_shape = (camera_info.height, camera_info.width)
        if image.shape[:2] != expected_shape:
            raise ValueError(
                f"image is {image.shape[1]}x{image.shape[0]}; camera profile is "
                f"{camera_info.width}x{camera_info.height}"
            )
        charuco_corners, charuco_ids, _marker_corners, marker_ids = self.detector.detectBoard(
            image
        )
        if marker_ids is None:
            raise ValueError("table ChArUco board was not detected")
        flat_marker_ids = tuple(int(value) for value in np.asarray(marker_ids).reshape(-1))
        duplicates = sorted(
            {value for value in flat_marker_ids if flat_marker_ids.count(value) > 1}
        )
        if duplicates:
            raise ValueError(f"duplicate ChArUco marker IDs detected: {duplicates}")
        if len(flat_marker_ids) < self.minimum_markers:
            raise ValueError(
                f"table board has only {len(flat_marker_ids)} markers; need {self.minimum_markers}"
            )
        if charuco_ids is None or charuco_corners is None:
            raise ValueError("table board produced no ChArUco corners")
        ids = np.asarray(charuco_ids, dtype=np.int32).reshape(-1)
        image_points = np.asarray(charuco_corners, dtype=np.float64).reshape(-1, 2)
        if len(ids) < self.minimum_corners:
            raise ValueError(
                f"table board has only {len(ids)} ChArUco corners; need {self.minimum_corners}"
            )
        if len({int(value) for value in ids}) != len(ids):
            raise ValueError("duplicate interpolated ChArUco corner IDs detected")
        all_object_points = np.asarray(self.board.getChessboardCorners(), dtype=np.float64)
        if np.any(ids < 0) or np.any(ids >= len(all_object_points)):
            raise ValueError("ChArUco corner ID is outside the configured board")
        object_points = all_object_points[ids]
        camera_matrix = camera_info.rectified_camera_matrix
        distortion = np.asarray(camera_info.d, dtype=np.float64)
        count, rvecs, tvecs, _ = cv2.solvePnPGeneric(
            object_points,
            image_points,
            camera_matrix,
            distortion,
            flags=cv2.SOLVEPNP_IPPE,
        )
        candidates: list[tuple[float, np.ndarray, np.ndarray]] = []
        if count:
            for rvec, tvec in zip(rvecs, tvecs, strict=True):
                rotation, _ = cv2.Rodrigues(rvec)
                camera_points = (rotation @ object_points.T + np.asarray(tvec).reshape(3, 1)).T
                if np.min(camera_points[:, 2]) <= 0:
                    continue
                projected, _ = cv2.projectPoints(
                    object_points, rvec, tvec, camera_matrix, distortion
                )
                delta = projected.reshape(-1, 2) - image_points
                rms = float(np.sqrt(np.mean(np.sum(np.square(delta), axis=1))))
                candidates.append(
                    (
                        rms,
                        np.asarray(rvec, dtype=np.float64).reshape(3, 1),
                        np.asarray(tvec, dtype=np.float64).reshape(3, 1),
                    )
                )
        if not candidates:
            raise ValueError("could not obtain a positive-depth table-board pose")
        candidates.sort(key=lambda item: item[0])
        _, rvec, tvec = candidates[0]
        rvec, tvec = cv2.solvePnPRefineLM(
            object_points,
            image_points,
            camera_matrix,
            distortion,
            rvec,
            tvec,
        )
        projected, _ = cv2.projectPoints(object_points, rvec, tvec, camera_matrix, distortion)
        delta = projected.reshape(-1, 2) - image_points
        rms = float(np.sqrt(np.mean(np.sum(np.square(delta), axis=1))))
        if rms > self.maximum_reprojection_error_px:
            raise ValueError(
                f"table-board reprojection error is {rms:.3f}px; limit is "
                f"{self.maximum_reprojection_error_px:.3f}px"
            )
        transform = np.eye(4)
        transform[:3, :3], _ = cv2.Rodrigues(rvec)
        transform[:3, 3] = np.asarray(tvec).reshape(3) / 1000.0
        second_error = candidates[1][0] if len(candidates) > 1 else None
        if second_error is not None and second_error - rms < 0.1:
            second = np.eye(4)
            second[:3, :3], _ = cv2.Rodrigues(candidates[1][1])
            second[:3, 3] = candidates[1][2].reshape(3) / 1000.0
            translation_delta_mm = 1000.0 * float(np.linalg.norm(second[:3, 3] - transform[:3, 3]))
            rotation_delta_deg = _rotation_error_deg(transform, second)
            if translation_delta_mm > 2.0 or rotation_delta_deg > 0.5:
                raise ValueError(
                    "table-board planar pose is ambiguous: two materially "
                    "different IPPE solutions have nearly equal reprojection error"
                )
        return TargetPoseEstimate(
            camera_T_target=transform,
            reprojection_error_px=rms,
            second_solution_error_px=second_error,
            point_count=len(ids),
            marker_ids=tuple(sorted(flat_marker_ids)),
        )


def detect_hand_target_pose(
    image_bgr: np.ndarray,
    camera_info: RectifiedCameraInfo,
    detector: CorrespondenceDetector,
    *,
    minimum_visible_faces: int = 1,
    minimum_tag_short_side_px: float = 30.0,
    maximum_reprojection_error_px: float = 1.5,
) -> TargetPoseEstimate:
    result: CorrespondenceResult = detector.detect(image_bgr)
    if not result.valid:
        if result.duplicate_tag_ids:
            raise ValueError(
                f"hand target has duplicate tag IDs: {list(result.duplicate_tag_ids)}"
            )
        raise ValueError("hand target was not detected")
    if len(result.visible_faces) < minimum_visible_faces:
        raise ValueError(
            f"hand target has only {len(result.visible_faces)} visible face; need "
            f"{minimum_visible_faces}"
        )
    minimum_side = min(item.shortest_side_px for item in result.observations)
    if minimum_side < minimum_tag_short_side_px:
        raise ValueError(
            f"hand-target marker is only {minimum_side:.1f}px; need "
            f"{minimum_tag_short_side_px:.1f}px"
        )
    object_points = np.vstack([item.object_corners_mm for item in result.observations]).astype(
        np.float64
    )
    image_points = np.vstack([item.image_corners_px for item in result.observations]).astype(
        np.float64
    )
    centered = object_points - np.mean(object_points, axis=0)
    planar = np.linalg.matrix_rank(centered, tol=1e-8) <= 2
    second_error: float | None = None
    if planar:
        transform, reprojection_error, second_error = _planar_target_pose(
            object_points,
            image_points,
            camera_info,
        )
    else:
        diagnostic = estimate_pose_diagnostic(
            result,
            camera_info.rectified_camera_matrix,
            np.asarray(camera_info.d, dtype=np.float64),
        )
        if diagnostic is None:
            raise ValueError("could not estimate a consistent hand-target pose")
        transform = np.eye(4)
        transform[:3, :3], _ = cv2.Rodrigues(diagnostic.rvec)
        transform[:3, 3] = diagnostic.tvec_mm.reshape(3) / 1000.0
        reprojection_error = diagnostic.reprojection_error_px
    if reprojection_error > maximum_reprojection_error_px:
        raise ValueError(
            f"hand-target reprojection error is "
            f"{reprojection_error:.3f}px; limit is "
            f"{maximum_reprojection_error_px:.3f}px"
        )
    return TargetPoseEstimate(
        camera_T_target=transform,
        reprojection_error_px=reprojection_error,
        second_solution_error_px=second_error,
        point_count=4 * len(result.observations),
        marker_ids=result.tag_ids,
        visible_faces=result.visible_faces,
    )


def _planar_target_pose(
    object_points: np.ndarray,
    image_points: np.ndarray,
    camera_info: RectifiedCameraInfo,
) -> tuple[np.ndarray, float, float | None]:
    """Solve a planar hand target and reject a materially ambiguous mirror pose."""

    square = len(object_points) == 4 and np.allclose(
        np.linalg.norm(np.roll(object_points, -1, axis=0) - object_points, axis=1),
        np.linalg.norm(object_points[1] - object_points[0]),
        rtol=1e-6,
        atol=1e-6,
    )
    flag = cv2.SOLVEPNP_IPPE_SQUARE if square else cv2.SOLVEPNP_IPPE
    count, rvecs, tvecs, _ = cv2.solvePnPGeneric(
        object_points,
        image_points,
        camera_info.rectified_camera_matrix,
        np.asarray(camera_info.d, dtype=np.float64),
        flags=flag,
    )
    candidates: list[tuple[float, np.ndarray, np.ndarray, np.ndarray]] = []
    if count:
        for rvec, tvec in zip(rvecs, tvecs, strict=True):
            rotation, _ = cv2.Rodrigues(rvec)
            camera_points = (rotation @ object_points.T + np.asarray(tvec).reshape(3, 1)).T
            if np.min(camera_points[:, 2]) <= 0:
                continue
            projected, _ = cv2.projectPoints(
                object_points,
                rvec,
                tvec,
                camera_info.rectified_camera_matrix,
                np.asarray(camera_info.d, dtype=np.float64),
            )
            delta = projected.reshape(-1, 2) - image_points
            rms = float(np.sqrt(np.mean(np.sum(np.square(delta), axis=1))))
            transform = np.eye(4)
            transform[:3, :3] = rotation
            transform[:3, 3] = np.asarray(tvec).reshape(3) / 1000.0
            candidates.append((rms, transform, np.asarray(rvec), np.asarray(tvec)))
    if not candidates:
        raise ValueError("could not obtain a positive-depth planar hand-target pose")
    candidates.sort(key=lambda item: item[0])
    _, transform, rvec, tvec = candidates[0]
    rvec, tvec = cv2.solvePnPRefineLM(
        object_points,
        image_points,
        camera_info.rectified_camera_matrix,
        np.asarray(camera_info.d, dtype=np.float64),
        rvec,
        tvec,
    )
    projected, _ = cv2.projectPoints(
        object_points,
        rvec,
        tvec,
        camera_info.rectified_camera_matrix,
        np.asarray(camera_info.d, dtype=np.float64),
    )
    delta = projected.reshape(-1, 2) - image_points
    rms = float(np.sqrt(np.mean(np.sum(np.square(delta), axis=1))))
    transform = np.eye(4)
    transform[:3, :3], _ = cv2.Rodrigues(rvec)
    transform[:3, 3] = np.asarray(tvec).reshape(3) / 1000.0
    second_error = candidates[1][0] if len(candidates) > 1 else None
    if second_error is not None and second_error - rms < 0.1:
        second = candidates[1][1]
        translation_delta_mm = 1000.0 * float(np.linalg.norm(second[:3, 3] - transform[:3, 3]))
        rotation_delta_deg = _rotation_error_deg(transform, second)
        if translation_delta_mm > 2.0 or rotation_delta_deg > 0.5:
            raise ValueError(
                "hand-target planar pose is ambiguous: two materially different "
                "IPPE solutions have nearly equal reprojection error"
            )
    return validate_transform(transform), rms, second_error


# Existing table-plan documents retain this callable name; new code uses the
# target-neutral entry point above.
detect_hand_cube_pose = detect_hand_target_pose


def build_table_target(
    *,
    torso_T_camera: np.ndarray,
    hand_T_cube: np.ndarray,
    camera_T_board: np.ndarray,
    current_board_T_cube: np.ndarray,
    board_spec: CharucoBoardSpec,
    lift_mm: float,
    board_xy_mm: tuple[float, float] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return desired board-from-cube and torso-from-hand transforms."""
    torso_T_camera = validate_transform(torso_T_camera)
    hand_T_cube = validate_transform(hand_T_cube)
    camera_T_board = validate_transform(camera_T_board)
    current_board_T_cube = validate_transform(current_board_T_cube)
    if lift_mm <= 0 or not np.isfinite(lift_mm):
        raise ValueError("relative lift must be positive and finite")
    if board_xy_mm is None:
        # OpenCV places the board origin at the upper-left outer corner.  Keep
        # the cube center on that corner so almost the complete board remains
        # visible during the achieved capture.
        board_xy_mm = (0.0, 0.0)
    xy = np.asarray(board_xy_mm, dtype=np.float64).reshape(-1)
    if xy.shape != (2,) or not np.all(np.isfinite(xy)):
        raise ValueError("board target XY must contain two finite millimetre values")
    if not (0.0 <= xy[0] <= board_spec.width_mm) or not (0.0 <= xy[1] <= board_spec.height_mm):
        raise ValueError("board target XY is outside the active board dimensions")
    desired_board_T_cube = np.eye(4)
    desired_board_T_cube[:3, :3] = current_board_T_cube[:3, :3]
    # OpenCV's +Z points into a face-up board.  Lift relative to the observed
    # supported cube pose instead of assuming that its center starts on the
    # table plane.
    desired_board_T_cube[:3, 3] = [
        xy[0] / 1000.0,
        xy[1] / 1000.0,
        float(current_board_T_cube[2, 3]) - lift_mm / 1000.0,
    ]
    desired_torso_T_hand = hand_target_from_board(
        torso_T_camera=torso_T_camera,
        hand_T_cube=hand_T_cube,
        camera_T_board=camera_T_board,
        desired_board_T_cube=desired_board_T_cube,
    )
    return (
        validate_transform(desired_board_T_cube),
        validate_transform(desired_torso_T_hand),
    )


def hand_target_from_board(
    *,
    torso_T_camera: np.ndarray,
    hand_T_cube: np.ndarray,
    camera_T_board: np.ndarray,
    desired_board_T_cube: np.ndarray,
) -> np.ndarray:
    """Convert a live board observation into the corresponding hand target."""
    target = (
        validate_transform(torso_T_camera)
        @ validate_transform(camera_T_board)
        @ validate_transform(desired_board_T_cube)
        @ invert_transform(hand_T_cube)
    )
    return validate_transform(target)


def task_error(
    desired_board_T_cube: np.ndarray, actual_board_T_cube: np.ndarray
) -> dict[str, Any]:
    desired = validate_transform(desired_board_T_cube)
    actual = validate_transform(actual_board_T_cube)
    error_mm = 1000.0 * (actual[:3, 3] - desired[:3, 3])
    return {
        "translation_error_board_xyz_mm": error_mm.tolist(),
        "planar_xy_error_mm": float(np.linalg.norm(error_mm[:2])),
        "vertical_z_error_mm": float(error_mm[2]),
        "translation_error_norm_mm": float(np.linalg.norm(error_mm)),
        "orientation_error_deg": _rotation_error_deg(desired, actual),
        "sign_convention": "actual_minus_desired_in_board_frame",
    }


def predicted_board_T_cube_from_model(
    *,
    torso_T_camera: np.ndarray,
    hand_T_cube: np.ndarray,
    camera_T_board: np.ndarray,
    torso_T_hand_at_measured_q: np.ndarray,
) -> np.ndarray:
    """Predict board-from-cube from measured joints and the fitted composite model."""

    return validate_transform(
        invert_transform(validate_transform(camera_T_board))
        @ invert_transform(validate_transform(torso_T_camera))
        @ validate_transform(torso_T_hand_at_measured_q)
        @ validate_transform(hand_T_cube)
    )


def observe_burst(
    image_paths: Sequence[str | Path],
    *,
    camera_info: RectifiedCameraInfo,
    hand_cube_detector: CorrespondenceDetector,
    board_detector: CharucoBoardPoseDetector,
    minimum_accepted_frames: int = 3,
    maximum_relative_translation_spread_mm: float = (
        DEFAULT_MAXIMUM_RELATIVE_TRANSLATION_SPREAD_MM
    ),
    maximum_relative_rotation_spread_deg: float = (DEFAULT_MAXIMUM_RELATIVE_ROTATION_SPREAD_DEG),
    reject_unstable: bool = True,
) -> dict[str, Any]:
    if minimum_accepted_frames < 1:
        raise ValueError("minimum accepted frame count must be positive")
    if not isinstance(reject_unstable, bool):
        raise TypeError("reject_unstable must be boolean")
    paths = [Path(item).resolve() for item in image_paths]
    if len(paths) < minimum_accepted_frames:
        raise ValueError(f"need at least {minimum_accepted_frames} input images, got {len(paths)}")
    accepted: list[dict[str, Any]] = []
    rejected: list[dict[str, str]] = []
    seen_hashes: set[str] = set()
    for path in paths:
        try:
            digest = _file_sha256(path)
            if digest in seen_hashes:
                raise ValueError("duplicate image content in burst")
            seen_hashes.add(digest)
            image = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if image is None:
                raise ValueError("could not decode image")
            board = board_detector.detect(image, camera_info)
            cube = detect_hand_target_pose(image, camera_info, hand_cube_detector)
            board_T_cube = invert_transform(board.camera_T_target) @ cube.camera_T_target
            accepted.append(
                {
                    "path": str(path),
                    "sha256": digest,
                    "board": board,
                    "cube": cube,
                    "board_T_cube": validate_transform(board_T_cube),
                }
            )
        except (OSError, ValueError) as error:
            rejected.append({"path": str(path), "reason": str(error)})
    if len(accepted) < minimum_accepted_frames:
        details = "; ".join(f"{Path(item['path']).name}: {item['reason']}" for item in rejected)
        raise ValueError(
            f"only {len(accepted)}/{len(paths)} paired frames passed; need "
            f"{minimum_accepted_frames}. {details}"
        )
    camera_T_board = _average_transforms([item["board"].camera_T_target for item in accepted])
    board_T_cube = _average_transforms([item["board_T_cube"] for item in accepted])
    relative_translation_spread_mm, relative_rotation_spread_deg = _transform_spread(
        [item["board_T_cube"] for item in accepted], board_T_cube
    )
    if reject_unstable and relative_translation_spread_mm > maximum_relative_translation_spread_mm:
        raise ValueError(
            f"board-to-hand translation spread is "
            f"{relative_translation_spread_mm:.3f}mm; limit is "
            f"{maximum_relative_translation_spread_mm:.3f}mm"
        )
    if reject_unstable and relative_rotation_spread_deg > maximum_relative_rotation_spread_deg:
        raise ValueError(
            f"board-to-hand rotation spread is "
            f"{relative_rotation_spread_deg:.3f}deg; limit is "
            f"{maximum_relative_rotation_spread_deg:.3f}deg"
        )
    board_translation_spread_mm, board_rotation_spread_deg = _transform_spread(
        [item["board"].camera_T_target for item in accepted], camera_T_board
    )
    return {
        "input_frame_count": len(paths),
        "accepted_frame_count": len(accepted),
        "rejected_frames": rejected,
        "accepted_frames": [
            {
                "path": item["path"],
                "sha256": item["sha256"],
                "board": item["board"].to_dict(),
                "hand_cube": item["cube"].to_dict(),
                "board_T_hand_cube": _transform_to_list(item["board_T_cube"]),
            }
            for item in accepted
        ],
        "aggregate": {
            "camera_T_board": _transform_to_list(camera_T_board),
            "board_T_hand_cube": _transform_to_list(board_T_cube),
            "relative_translation_spread_mm": relative_translation_spread_mm,
            "relative_rotation_spread_deg": relative_rotation_spread_deg,
            "camera_T_board_translation_spread_mm": board_translation_spread_mm,
            "camera_T_board_rotation_spread_deg": board_rotation_spread_deg,
        },
    }


def camera_info_from_dataset(dataset: CalibrationDataset) -> RectifiedCameraInfo:
    if not dataset.samples:
        raise ValueError("calibration dataset contains no samples")
    infos = [RectifiedCameraInfo.from_dict(item.camera_info) for item in dataset.samples]
    expected = infos[0].profile_sha256
    if any(item.profile_sha256 != expected for item in infos[1:]):
        raise ValueError("calibration dataset mixes multiple camera profiles")
    return infos[0]


def create_plan_document(
    *,
    dataset_path: str | Path,
    result_path: str | Path,
    hand_cube_config_path: str | Path,
    planning_burst: dict[str, Any],
    board_spec: CharucoBoardSpec,
    lift_mm: float,
    board_xy_mm: tuple[float, float] | None = None,
) -> dict[str, Any]:
    dataset_path = Path(dataset_path).resolve()
    result_path = Path(result_path).resolve()
    cube_path = Path(hand_cube_config_path).resolve()
    dataset = CalibrationDataset.from_json(dataset_path)
    result = load_exported_result(result_path)
    if result["dataset_sha256"] != dataset.content_sha256:
        raise ValueError("calibration result belongs to a different dataset")
    cube_hash = _file_sha256(cube_path)
    if cube_hash != dataset.target_artifact_sha256:
        raise ValueError(
            "hand-cube config differs from the target used by the calibration dataset"
        )
    camera_info = camera_info_from_dataset(dataset)
    camera_T_board = _transform_from_document(
        planning_burst["aggregate"]["camera_T_board"], name="camera_T_board"
    )
    current_board_T_cube = _transform_from_document(
        planning_burst["aggregate"]["board_T_hand_cube"],
        name="board_T_hand_cube",
    )
    torso_T_camera = _transform_from_document(
        result["solution"]["torso_T_camera"], name="torso_T_camera"
    )
    hand_T_cube = _transform_from_document(
        result["solution"]["hand_T_target"], name="hand_T_target"
    )
    desired_board_T_cube, desired_torso_T_hand = build_table_target(
        torso_T_camera=torso_T_camera,
        hand_T_cube=hand_T_cube,
        camera_T_board=camera_T_board,
        current_board_T_cube=current_board_T_cube,
        board_spec=board_spec,
        lift_mm=lift_mm,
        board_xy_mm=board_xy_mm,
    )
    target_xy = [0.0, 0.0] if board_xy_mm is None else [float(value) for value in board_xy_mm]
    document: dict[str, Any] = {
        "schema_version": PLAN_SCHEMA_VERSION,
        "kind": "g1_table_accuracy_plan",
        "sources": {
            "dataset_path": str(dataset_path),
            "dataset_sha256": dataset.content_sha256,
            "result_path": str(result_path),
            "result_sha256": result["content_sha256"],
            "hand_cube_config_path": str(cube_path),
            "hand_cube_config_sha256": cube_hash,
            "urdf_sha256": dataset.urdf_sha256,
        },
        "calibration_arm": dataset.calibration_arm,
        "hand_link": arm_hand_link(dataset.calibration_arm),
        "camera_info": camera_info.to_dict(),
        "camera_profile_sha256": camera_info.profile_sha256,
        "board_spec": board_spec.to_dict(),
        "calibration": {
            "torso_T_camera": _transform_to_list(torso_T_camera),
            "hand_T_cube": _transform_to_list(hand_T_cube),
        },
        "planning_burst": planning_burst,
        "target": {
            "board_xy_mm": target_xy,
            "lift_above_initial_cube_mm": float(lift_mm),
            "initial_board_T_hand_cube": _transform_to_list(current_board_T_cube),
            "desired_board_T_hand_cube": _transform_to_list(desired_board_T_cube),
            "desired_torso_T_hand": _transform_to_list(desired_torso_T_hand),
            "orientation_policy": "preserve_initial_board_T_hand_cube_orientation",
            "height_policy": "initial_board_z_minus_relative_lift",
            "board_z_convention": "negative_z_is_above_printed_face",
        },
        "execution": {
            "commands_robot": False,
            "requires_separate_ik_collision_and_motion_validation": True,
        },
    }
    document["content_sha256"] = _canonical_sha256(document)
    validate_plan_document(document)
    return document


def validate_plan_document(document: dict[str, Any]) -> None:
    required = {
        "schema_version",
        "kind",
        "sources",
        "calibration_arm",
        "hand_link",
        "camera_info",
        "camera_profile_sha256",
        "board_spec",
        "calibration",
        "planning_burst",
        "target",
        "execution",
        "content_sha256",
    }
    if set(document) != required:
        raise ValueError("table-accuracy plan fields do not match schema version 2")
    if document["schema_version"] != PLAN_SCHEMA_VERSION:
        raise ValueError("unsupported table-accuracy plan schema version")
    if document["kind"] != "g1_table_accuracy_plan":
        raise ValueError("document is not a G1 table-accuracy plan")
    source_fields = {
        "dataset_path",
        "dataset_sha256",
        "result_path",
        "result_sha256",
        "hand_cube_config_path",
        "hand_cube_config_sha256",
        "urdf_sha256",
    }
    if set(document["sources"]) != source_fields:
        raise ValueError("table-accuracy plan source fields are incomplete")
    for name in (
        "dataset_sha256",
        "result_sha256",
        "hand_cube_config_sha256",
        "urdf_sha256",
    ):
        value = document["sources"][name]
        if (
            not isinstance(value, str)
            or len(value) != 64
            or any(character not in "0123456789abcdef" for character in value)
        ):
            raise ValueError(f"table-accuracy plan {name} is not SHA-256")
    canonical = dict(document)
    expected = canonical.pop("content_sha256")
    if expected != _canonical_sha256(canonical):
        raise ValueError("table-accuracy plan content SHA-256 does not match")
    info = RectifiedCameraInfo.from_dict(document["camera_info"])
    if info.profile_sha256 != document["camera_profile_sha256"]:
        raise ValueError("table-accuracy plan camera profile hash does not match")
    CharucoBoardSpec.from_dict(document["board_spec"])
    _transform_from_document(document["calibration"]["torso_T_camera"], name="torso_T_camera")
    _transform_from_document(document["calibration"]["hand_T_cube"], name="hand_T_cube")
    target_fields = {
        "board_xy_mm",
        "lift_above_initial_cube_mm",
        "initial_board_T_hand_cube",
        "desired_board_T_hand_cube",
        "desired_torso_T_hand",
        "orientation_policy",
        "height_policy",
        "board_z_convention",
    }
    if set(document["target"]) != target_fields:
        raise ValueError("table-accuracy target fields are incomplete")
    initial = _transform_from_document(
        document["target"]["initial_board_T_hand_cube"],
        name="initial_board_T_hand_cube",
    )
    desired = _transform_from_document(
        document["target"]["desired_board_T_hand_cube"],
        name="desired_board_T_hand_cube",
    )
    _transform_from_document(
        document["target"]["desired_torso_T_hand"],
        name="desired_torso_T_hand",
    )
    xy = np.asarray(document["target"]["board_xy_mm"], dtype=np.float64)
    lift_mm = float(document["target"]["lift_above_initial_cube_mm"])
    if xy.shape != (2,) or not np.all(np.isfinite(xy)):
        raise ValueError("table-accuracy target XY is invalid")
    if not np.isfinite(lift_mm) or lift_mm <= 0:
        raise ValueError("table-accuracy relative lift is invalid")
    expected_translation = np.asarray(
        [xy[0] / 1000.0, xy[1] / 1000.0, initial[2, 3] - lift_mm / 1000.0]
    )
    if not np.allclose(desired[:3, 3], expected_translation, atol=1e-12):
        raise ValueError("table-accuracy desired target is inconsistent with relative lift")
    if not np.allclose(desired[:3, :3], initial[:3, :3], atol=1e-12):
        raise ValueError("table-accuracy desired target does not preserve orientation")
    if document["target"]["orientation_policy"] != (
        "preserve_initial_board_T_hand_cube_orientation"
    ):
        raise ValueError("table-accuracy orientation policy was modified")
    if document["target"]["height_policy"] != "initial_board_z_minus_relative_lift":
        raise ValueError("table-accuracy height policy was modified")
    if document["execution"] != {
        "commands_robot": False,
        "requires_separate_ik_collision_and_motion_validation": True,
    }:
        raise ValueError("table-accuracy plan execution boundary was modified")


def load_plan_document(path: str | Path) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as stream:
        document = json.load(stream)
    if not isinstance(document, dict):
        raise TypeError("table-accuracy plan must contain a JSON object")
    validate_plan_document(document)
    return document


def create_evaluation_document(
    *,
    plan: dict[str, Any],
    achieved_burst: dict[str, Any],
    predicted_board_T_cube: np.ndarray | None = None,
) -> dict[str, Any]:
    validate_plan_document(plan)
    desired = _transform_from_document(
        plan["target"]["desired_board_T_hand_cube"],
        name="desired_board_T_hand_cube",
    )
    actual = _transform_from_document(
        achieved_burst["aggregate"]["board_T_hand_cube"],
        name="actual_board_T_hand_cube",
    )
    planning_camera_T_board = _transform_from_document(
        plan["planning_burst"]["aggregate"]["camera_T_board"],
        name="planning_camera_T_board",
    )
    achieved_camera_T_board = _transform_from_document(
        achieved_burst["aggregate"]["camera_T_board"],
        name="achieved_camera_T_board",
    )
    translation_spread_mm = float(achieved_burst["aggregate"]["relative_translation_spread_mm"])
    rotation_spread_deg = float(achieved_burst["aggregate"]["relative_rotation_spread_deg"])
    document: dict[str, Any] = {
        "schema_version": EVALUATION_SCHEMA_VERSION,
        "kind": "g1_table_accuracy_evaluation",
        "plan_sha256": plan["content_sha256"],
        "achieved_burst": achieved_burst,
        "desired_board_T_hand_cube": _transform_to_list(desired),
        "actual_board_T_hand_cube": _transform_to_list(actual),
        "task_error": task_error(desired, actual),
        "measurement_quality": {
            "passed": (
                translation_spread_mm <= DEFAULT_MAXIMUM_RELATIVE_TRANSLATION_SPREAD_MM
                and rotation_spread_deg <= DEFAULT_MAXIMUM_RELATIVE_ROTATION_SPREAD_DEG
            ),
            "relative_translation_spread_mm": translation_spread_mm,
            "maximum_relative_translation_spread_mm": (
                DEFAULT_MAXIMUM_RELATIVE_TRANSLATION_SPREAD_MM
            ),
            "relative_rotation_spread_deg": rotation_spread_deg,
            "maximum_relative_rotation_spread_deg": (DEFAULT_MAXIMUM_RELATIVE_ROTATION_SPREAD_DEG),
            "interpretation": (
                "the task error remains an estimate when this quality gate fails; "
                "the spread is its directly observed repeatability warning"
            ),
        },
        "camera_motion_between_bursts": {
            "translation_mm": 1000.0
            * float(
                np.linalg.norm(achieved_camera_T_board[:3, 3] - planning_camera_T_board[:3, 3])
            ),
            "rotation_deg": _rotation_error_deg(planning_camera_T_board, achieved_camera_T_board),
            "interpretation": ("camera relative to fixed table board; includes torso/head motion"),
        },
    }
    if predicted_board_T_cube is not None:
        predicted = validate_transform(predicted_board_T_cube)
        document["predicted_board_T_hand_cube_from_measured_joints"] = _transform_to_list(
            predicted
        )
        document["model_implied_tracking_error"] = task_error(desired, predicted)
        document["held_out_composite_model_error"] = task_error(predicted, actual)
        document["model_diagnostic_interpretation"] = (
            "the tracking term is FK/model-implied and the composite residual "
            "contains camera, hand-mount, FK, elasticity, and visual error; only "
            "task_error is the direct board-relative landing measurement"
        )
    document["content_sha256"] = _canonical_sha256(document)
    return document
