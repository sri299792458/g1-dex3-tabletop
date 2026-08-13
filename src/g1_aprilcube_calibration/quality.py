"""Calibration-view quality and dataset-novelty evaluation."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Any

import cv2
import numpy as np
from aprilcube import CorrespondenceResult, PoseDiagnostic, estimate_pose_diagnostic

from g1_aprilcube_calibration.config import QualityThresholds


class QualityGrade(str, Enum):
    RED = "red"
    YELLOW = "yellow"
    GREEN = "green"


@dataclass(frozen=True, slots=True)
class CameraIntrinsics:
    camera_matrix: np.ndarray
    dist_coeffs: np.ndarray

    def __post_init__(self) -> None:
        matrix = np.asarray(self.camera_matrix, dtype=np.float64).copy()
        distortion = np.asarray(self.dist_coeffs, dtype=np.float64).reshape(-1).copy()
        if matrix.shape != (3, 3):
            raise ValueError(f"camera_matrix must be 3x3, got {matrix.shape}")
        if not np.all(np.isfinite(matrix)) or not np.all(np.isfinite(distortion)):
            raise ValueError("camera intrinsics must be finite")
        if matrix[0, 0] <= 0 or matrix[1, 1] <= 0:
            raise ValueError("camera focal lengths must be positive")
        matrix.setflags(write=False)
        distortion.setflags(write=False)
        object.__setattr__(self, "camera_matrix", matrix)
        object.__setattr__(self, "dist_coeffs", distortion)

    @classmethod
    def from_parameters(
        cls,
        fx: float,
        fy: float,
        cx: float,
        cy: float,
        dist_coeffs: Sequence[float] = (0, 0, 0, 0, 0),
    ) -> CameraIntrinsics:
        return cls(
            camera_matrix=np.array(
                [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]],
                dtype=np.float64,
            ),
            dist_coeffs=np.asarray(dist_coeffs, dtype=np.float64),
        )


@dataclass(frozen=True, slots=True)
class ViewSignature:
    image_cell: tuple[int, int]
    normalized_centroid: tuple[float, float]
    tag_ids: tuple[int, ...]
    visible_faces: tuple[str, ...]
    rvec: tuple[float, float, float] | None
    tvec_m: tuple[float, float, float] | None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ViewSignature:
        return cls(
            image_cell=tuple(data["image_cell"]),
            normalized_centroid=tuple(data["normalized_centroid"]),
            tag_ids=tuple(data["tag_ids"]),
            visible_faces=tuple(data["visible_faces"]),
            rvec=None if data["rvec"] is None else tuple(data["rvec"]),
            tvec_m=None if data["tvec_m"] is None else tuple(data["tvec_m"]),
        )


@dataclass(frozen=True, slots=True)
class QualityReport:
    grade: QualityGrade
    hard_failures: tuple[str, ...]
    warnings: tuple[str, ...]
    metrics: dict[str, Any]
    signature: ViewSignature | None
    pose_diagnostic: PoseDiagnostic | None

    @property
    def save_allowed(self) -> bool:
        return self.grade is not QualityGrade.RED

    def to_dict(self) -> dict[str, Any]:
        signature = None
        if self.signature is not None:
            signature = {
                "image_cell": list(self.signature.image_cell),
                "normalized_centroid": list(self.signature.normalized_centroid),
                "tag_ids": list(self.signature.tag_ids),
                "visible_faces": list(self.signature.visible_faces),
                "rvec": None if self.signature.rvec is None else list(self.signature.rvec),
                "tvec_m": (None if self.signature.tvec_m is None else list(self.signature.tvec_m)),
            }
        return {
            "grade": self.grade.value,
            "save_allowed": self.save_allowed,
            "hard_failures": list(self.hard_failures),
            "warnings": list(self.warnings),
            "metrics": self.metrics,
            "signature": signature,
        }


class PoseQualityEvaluator:
    """Apply hard visual gates, preferred gates, and novelty guidance."""

    def __init__(self, thresholds: QualityThresholds) -> None:
        self.thresholds = thresholds

    def evaluate(
        self,
        result: CorrespondenceResult,
        *,
        intrinsics: CameraIntrinsics | None = None,
        history: Sequence[ViewSignature] = (),
    ) -> QualityReport:
        hard_failures: list[str] = []
        warnings: list[str] = []
        observations = result.observations

        if self.thresholds.reject_duplicate_tag_ids and result.duplicate_tag_ids:
            ids = ", ".join(str(item) for item in result.duplicate_tag_ids)
            hard_failures.append(f"duplicate tag ID(s): {ids}")
        if len(observations) < self.thresholds.minimum_decoded_tags:
            hard_failures.append(
                f"need at least {self.thresholds.minimum_decoded_tags} decoded tag"
            )

        minimum_side = min((item.shortest_side_px for item in observations), default=None)
        minimum_margin = min((item.image_margin_px for item in observations), default=None)
        minimum_quad_quality = min((item.quad_quality for item in observations), default=None)
        if minimum_side is not None:
            if minimum_side < self.thresholds.minimum_tag_short_side_px:
                hard_failures.append(
                    f"tag too small: {minimum_side:.1f}px < "
                    f"{self.thresholds.minimum_tag_short_side_px:.1f}px"
                )
            elif minimum_side < self.thresholds.preferred_tag_short_side_px:
                warnings.append(
                    f"tag size is marginal: {minimum_side:.1f}px; prefer "
                    f"{self.thresholds.preferred_tag_short_side_px:.1f}px"
                )
        if minimum_margin is not None:
            if minimum_margin < self.thresholds.minimum_corner_image_margin_px:
                hard_failures.append(f"corner too close to image edge: {minimum_margin:.1f}px")
            elif minimum_margin < self.thresholds.preferred_corner_image_margin_px:
                warnings.append(
                    f"image margin is small: {minimum_margin:.1f}px; prefer "
                    f"{self.thresholds.preferred_corner_image_margin_px:.1f}px"
                )

        visible_faces = result.visible_faces
        if observations and len(visible_faces) < self.thresholds.preferred_visible_faces:
            warnings.append(
                f"only {len(visible_faces)} cube face visible; prefer "
                f"{self.thresholds.preferred_visible_faces}"
            )

        diagnostic = None
        if observations and not result.duplicate_tag_ids and intrinsics is not None:
            diagnostic = estimate_pose_diagnostic(
                result,
                intrinsics.camera_matrix,
                intrinsics.dist_coeffs,
            )
            if diagnostic is None:
                if len(visible_faces) >= 2:
                    hard_failures.append("multi-face PnP consistency check failed")
                else:
                    warnings.append("PnP diagnostic unavailable")
            elif len(visible_faces) >= 2:
                error = diagnostic.reprojection_error_px
                if error > self.thresholds.pnp_reject_reprojection_px:
                    hard_failures.append(
                        f"multi-face PnP error is {error:.2f}px; reject above "
                        f"{self.thresholds.pnp_reject_reprojection_px:.2f}px"
                    )
                elif error > self.thresholds.pnp_warning_reprojection_px:
                    warnings.append(
                        f"multi-face PnP error is {error:.2f}px; prefer at most "
                        f"{self.thresholds.pnp_warning_reprojection_px:.2f}px"
                    )
        elif observations:
            warnings.append("camera intrinsics unavailable; PnP/depth checks skipped")

        signature = self._make_signature(result, diagnostic)
        novel = signature is not None and self._is_novel(signature, history)
        if signature is not None and not novel:
            warnings.append("view is redundant with an already saved pose")

        metrics: dict[str, Any] = {
            "tag_count": len(observations),
            "tag_ids": list(result.tag_ids),
            "visible_faces": list(visible_faces),
            "minimum_tag_short_side_px": minimum_side,
            "minimum_corner_image_margin_px": minimum_margin,
            "minimum_quad_quality": minimum_quad_quality,
            "duplicate_tag_ids": list(result.duplicate_tag_ids),
            "ignored_tag_ids": list(result.ignored_tag_ids),
            "opencv_rejected_candidates": result.opencv_rejected_candidates,
            "quality_rejected_detections": result.quality_rejected_detections,
            "pnp_reprojection_error_px": (
                None if diagnostic is None else diagnostic.reprojection_error_px
            ),
            "pnp_depth_m": (
                None if diagnostic is None else float(diagnostic.tvec_mm[2, 0] / 1000.0)
            ),
            "novel_view": novel,
        }

        if hard_failures:
            grade = QualityGrade.RED
        elif warnings:
            grade = QualityGrade.YELLOW
        else:
            grade = QualityGrade.GREEN
        return QualityReport(
            grade=grade,
            hard_failures=tuple(hard_failures),
            warnings=tuple(warnings),
            metrics=metrics,
            signature=signature,
            pose_diagnostic=diagnostic,
        )

    def _make_signature(
        self,
        result: CorrespondenceResult,
        diagnostic: PoseDiagnostic | None,
    ) -> ViewSignature | None:
        if not result.observations:
            return None
        all_corners = np.vstack([item.image_corners_px for item in result.observations])
        centroid = np.mean(all_corners, axis=0)
        width, height = result.image_size_wh
        nx = float(centroid[0] / max(width - 1, 1))
        ny = float(centroid[1] / max(height - 1, 1))
        col = min(
            int(nx * self.thresholds.image_grid_columns),
            self.thresholds.image_grid_columns - 1,
        )
        row = min(
            int(ny * self.thresholds.image_grid_rows),
            self.thresholds.image_grid_rows - 1,
        )
        rvec = None
        tvec_m = None
        if diagnostic is not None:
            rvec = tuple(float(value) for value in diagnostic.rvec.reshape(3))
            tvec_m = tuple(float(value / 1000.0) for value in diagnostic.tvec_mm.reshape(3))
        return ViewSignature(
            image_cell=(col, row),
            normalized_centroid=(nx, ny),
            tag_ids=result.tag_ids,
            visible_faces=result.visible_faces,
            rvec=rvec,
            tvec_m=tvec_m,
        )

    def _is_novel(
        self,
        signature: ViewSignature,
        history: Sequence[ViewSignature],
    ) -> bool:
        if not history:
            return True
        for previous in history:
            if signature.image_cell != previous.image_cell:
                continue
            if signature.rvec is None or signature.tvec_m is None:
                return False
            if previous.rvec is None or previous.tvec_m is None:
                continue
            translation = float(
                np.linalg.norm(np.asarray(signature.tvec_m) - np.asarray(previous.tvec_m))
            )
            rotation = _rotation_distance_deg(signature.rvec, previous.rvec)
            if (
                translation < self.thresholds.novelty_translation_m
                and rotation < self.thresholds.novelty_rotation_deg
            ):
                return False
        return True


def _rotation_distance_deg(
    rvec_a: Sequence[float],
    rvec_b: Sequence[float],
) -> float:
    rotation_a, _ = cv2.Rodrigues(np.asarray(rvec_a, dtype=np.float64))
    rotation_b, _ = cv2.Rodrigues(np.asarray(rvec_b, dtype=np.float64))
    cosine = np.clip(
        (np.trace(rotation_a.T @ rotation_b) - 1.0) / 2.0,
        -1.0,
        1.0,
    )
    return float(np.degrees(np.arccos(cosine)))
