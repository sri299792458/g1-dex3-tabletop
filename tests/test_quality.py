from pathlib import Path

import cv2
import numpy as np
import pytest
from aprilcube import (
    CorrespondenceDetector,
    CorrespondenceResult,
    PoseDiagnostic,
    TagCorrespondence,
)

from g1_aprilcube_calibration.config import QualityThresholds
from g1_aprilcube_calibration.preview import render_operator_preview
from g1_aprilcube_calibration.quality import (
    CameraIntrinsics,
    PoseQualityEvaluator,
    QualityGrade,
)

ROOT = Path(__file__).parents[1]
TARGET = ROOT / "third_party" / "aprilcube" / "models" / "dex3_safe_cube" / "config.json"
QUALITY_CONFIG = ROOT / "config" / "capture_quality_dex3_aruco.yaml"


@pytest.fixture
def thresholds() -> QualityThresholds:
    return QualityThresholds.from_yaml(QUALITY_CONFIG)


@pytest.fixture
def intrinsics() -> CameraIntrinsics:
    return CameraIntrinsics.from_parameters(600, 600, 319.5, 239.5)


def _result(*, one_face: bool = False, side_px: float = 80) -> CorrespondenceResult:
    detector = CorrespondenceDetector(TARGET)
    first = np.array(
        [
            [120, 100],
            [120 + side_px, 100],
            [120 + side_px, 100 + side_px],
            [120, 100 + side_px],
        ],
        dtype=np.float64,
    )
    observations = [
        TagCorrespondence(
            tag_id=0,
            face_name="+X",
            image_corners_px=first,
            object_corners_mm=detector.tag_corner_map[0],
            quad_quality=0.95,
            shortest_side_px=side_px,
            image_margin_px=100.0,
        )
    ]
    if not one_face:
        second = first + np.array([160, 30], dtype=np.float64)
        observations.append(
            TagCorrespondence(
                tag_id=5,
                face_name="-Z",
                image_corners_px=second,
                object_corners_mm=detector.tag_corner_map[5],
                quad_quality=0.9,
                shortest_side_px=side_px,
                image_margin_px=100.0,
            )
        )
    return CorrespondenceResult(
        image_size_wh=(640, 480),
        observations=tuple(observations),
    )


def _diagnostic(error_px: float = 0.5) -> PoseDiagnostic:
    return PoseDiagnostic(
        rvec=np.array([0.1, -0.2, 0.3]),
        tvec_mm=np.array([20.0, -10.0, 500.0]),
        reprojection_error_px=error_px,
        inlier_count=8,
    )


def test_two_face_novel_view_is_green(
    monkeypatch: pytest.MonkeyPatch,
    thresholds: QualityThresholds,
    intrinsics: CameraIntrinsics,
) -> None:
    monkeypatch.setattr(
        "g1_aprilcube_calibration.quality.estimate_pose_diagnostic",
        lambda *_args, **_kwargs: _diagnostic(),
    )
    report = PoseQualityEvaluator(thresholds).evaluate(_result(), intrinsics=intrinsics)

    assert report.grade is QualityGrade.GREEN
    assert report.save_allowed
    assert report.metrics["novel_view"]


def test_one_face_is_green_for_planar_dex3_target(
    monkeypatch: pytest.MonkeyPatch,
    thresholds: QualityThresholds,
    intrinsics: CameraIntrinsics,
) -> None:
    monkeypatch.setattr(
        "g1_aprilcube_calibration.quality.estimate_pose_diagnostic",
        lambda *_args, **_kwargs: _diagnostic(),
    )
    report = PoseQualityEvaluator(thresholds).evaluate(
        _result(one_face=True), intrinsics=intrinsics
    )

    assert report.grade is QualityGrade.GREEN
    assert not report.warnings


def test_small_tag_is_red(
    thresholds: QualityThresholds,
    intrinsics: CameraIntrinsics,
) -> None:
    report = PoseQualityEvaluator(thresholds).evaluate(_result(side_px=20), intrinsics=intrinsics)

    assert report.grade is QualityGrade.RED
    assert any("tag too small" in failure for failure in report.hard_failures)


def test_repeated_view_becomes_yellow(
    monkeypatch: pytest.MonkeyPatch,
    thresholds: QualityThresholds,
    intrinsics: CameraIntrinsics,
) -> None:
    monkeypatch.setattr(
        "g1_aprilcube_calibration.quality.estimate_pose_diagnostic",
        lambda *_args, **_kwargs: _diagnostic(),
    )
    evaluator = PoseQualityEvaluator(thresholds)
    first = evaluator.evaluate(_result(), intrinsics=intrinsics)
    repeated = evaluator.evaluate(
        _result(),
        intrinsics=intrinsics,
        history=[first.signature],
    )

    assert repeated.grade is QualityGrade.YELLOW
    assert not repeated.metrics["novel_view"]
    assert any("redundant" in warning for warning in repeated.warnings)


def test_multiface_bad_pnp_is_red(
    monkeypatch: pytest.MonkeyPatch,
    thresholds: QualityThresholds,
    intrinsics: CameraIntrinsics,
) -> None:
    monkeypatch.setattr(
        "g1_aprilcube_calibration.quality.estimate_pose_diagnostic",
        lambda *_args, **_kwargs: _diagnostic(error_px=4.0),
    )
    report = PoseQualityEvaluator(thresholds).evaluate(_result(), intrinsics=intrinsics)

    assert report.grade is QualityGrade.RED
    assert any("PnP error" in failure for failure in report.hard_failures)


def test_preview_renderer_adds_side_panel(
    monkeypatch: pytest.MonkeyPatch,
    thresholds: QualityThresholds,
    intrinsics: CameraIntrinsics,
) -> None:
    monkeypatch.setattr(
        "g1_aprilcube_calibration.quality.estimate_pose_diagnostic",
        lambda *_args, **_kwargs: _diagnostic(),
    )
    result = _result()
    report = PoseQualityEvaluator(thresholds).evaluate(result, intrinsics=intrinsics)
    frame = np.full((480, 640, 3), 180, dtype=np.uint8)

    rendered = render_operator_preview(frame, result, report, intrinsics=intrinsics)

    assert rendered.shape == (480, 1070, 3)
    assert rendered.dtype == np.uint8
    assert cv2.countNonZero(cv2.cvtColor(rendered, cv2.COLOR_BGR2GRAY)) > 0
