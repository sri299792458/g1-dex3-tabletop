"""OpenCV rendering for the laptop operator preview."""

from __future__ import annotations

import textwrap
from collections.abc import Sequence

import cv2
import numpy as np
from aprilcube import CorrespondenceResult

from g1_aprilcube_calibration.quality import (
    CameraIntrinsics,
    QualityGrade,
    QualityReport,
)

GRADE_COLORS = {
    QualityGrade.RED: (40, 40, 230),
    QualityGrade.YELLOW: (0, 210, 255),
    QualityGrade.GREEN: (35, 205, 70),
}


def render_operator_preview(
    image: np.ndarray,
    detections: CorrespondenceResult,
    report: QualityReport,
    *,
    intrinsics: CameraIntrinsics | None = None,
    saved_view_count: int = 0,
    footer_lines: Sequence[str] | None = None,
) -> np.ndarray:
    """Render detection geometry and a compact quality side panel."""
    frame = np.asarray(image)
    if frame.ndim == 2:
        canvas = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    elif frame.shape[2] == 4:
        canvas = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
    else:
        canvas = frame.copy()

    color = GRADE_COLORS[report.grade]
    for observation in detections.observations:
        corners = np.rint(observation.image_corners_px).astype(np.int32)
        cv2.polylines(canvas, [corners], True, color, 3, cv2.LINE_AA)
        for index, corner in enumerate(corners):
            cv2.circle(canvas, tuple(corner), 5, color, -1, cv2.LINE_AA)
            cv2.putText(
                canvas,
                str(index),
                (int(corner[0]) + 6, int(corner[1]) - 6),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                color,
                1,
                cv2.LINE_AA,
            )
        center = np.mean(corners, axis=0).astype(int)
        face = "?" if observation.face_name is None else observation.face_name
        cv2.putText(
            canvas,
            f"ID {observation.tag_id}  {face}",
            (int(center[0]) - 35, int(center[1])),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2,
            cv2.LINE_AA,
        )

    if intrinsics is not None and report.pose_diagnostic is not None:
        diagnostic = report.pose_diagnostic
        cv2.drawFrameAxes(
            canvas,
            intrinsics.camera_matrix,
            intrinsics.dist_coeffs,
            diagnostic.rvec,
            diagnostic.tvec_mm,
            20.0,
            3,
        )

    panel_width = 430
    panel = np.full((canvas.shape[0], panel_width, 3), 24, dtype=np.uint8)
    cv2.rectangle(panel, (0, 0), (panel_width - 1, 76), color, -1)
    cv2.putText(
        panel,
        report.grade.value.upper(),
        (22, 48),
        cv2.FONT_HERSHEY_DUPLEX,
        1.25,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        panel,
        "VISUAL QUALITY ONLY",
        (205, 45),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )

    metrics = report.metrics
    detail_lines = [
        f"Tags: {metrics['tag_ids']}   Faces: {metrics['visible_faces']}",
        _metric_line("Shortest side", metrics["minimum_tag_short_side_px"], "px"),
        _metric_line("Image margin", metrics["minimum_corner_image_margin_px"], "px"),
        _metric_line("PnP reprojection", metrics["pnp_reprojection_error_px"], "px"),
        _metric_line("Depth", metrics["pnp_depth_m"], "m"),
        f"Novel view: {'yes' if metrics['novel_view'] else 'no'}",
        f"Saved visual views: {saved_view_count}",
    ]
    y = 112
    for line in detail_lines:
        cv2.putText(
            panel,
            line,
            (18, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            (225, 225, 225),
            1,
            cv2.LINE_AA,
        )
        y += 28

    issues = list(report.hard_failures) + list(report.warnings)
    if issues:
        y += 8
        cv2.putText(
            panel,
            "WHY",
            (18, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.58,
            color,
            2,
            cv2.LINE_AA,
        )
        y += 26
        for issue in issues:
            for line in textwrap.wrap(f"- {issue}", width=48):
                cv2.putText(
                    panel,
                    line,
                    (18, y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.43,
                    (210, 210, 210),
                    1,
                    cv2.LINE_AA,
                )
                y += 22
            y += 3

    lines = (
        tuple(footer_lines)
        if footer_lines is not None
        else (
            "S save visual view   U undo   Q quit",
            "Robot readiness/collision status: not connected",
        )
    )
    controls_y = max(canvas.shape[0] - (25 * len(lines) + 15), y + 12)
    if controls_y + 25 * len(lines) < canvas.shape[0] + 12:
        for index, line in enumerate(lines):
            color = (100, 180, 255) if index == len(lines) - 1 else (155, 155, 155)
            cv2.putText(
                panel,
                line,
                (18, controls_y + index * 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.43,
                color,
                1,
                cv2.LINE_AA,
            )
    return np.hstack((canvas, panel))


def _metric_line(label: str, value: float | None, unit: str) -> str:
    if value is None:
        return f"{label}: unavailable"
    return f"{label}: {value:.2f} {unit}"
