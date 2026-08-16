from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from g1_aprilcube_calibration.calibration_bundle import CalibrationBundle
from g1_aprilcube_calibration.urdf_model import URDFModel
from g1_dex3_tabletop.state_estimation import (
    ESTIMATOR_NAMES,
    AnchoredCameraPoseEstimators,
    CameraPoseAnchor,
    ProprioceptiveSample,
    pose_error,
)
from g1_dex3_tabletop.state_estimation_replay import (
    BoardEvent,
    _clock_model,
    _evaluation_pairs,
)

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def estimators() -> AnchoredCameraPoseEstimators:
    return AnchoredCameraPoseEstimators(
        model=URDFModel(ROOT / "config/urdf/g1_29dof_rev_1_0_g1pilot_collision.urdf"),
        calibration_bundle=CalibrationBundle.load(
            ROOT / "config/calibrations/dex3_shared_20260812_selected_free.json"
        ),
    )


def _sample(
    *,
    timestamp_ns: int = 1,
    q29: np.ndarray | None = None,
    pelvis_rotation: np.ndarray | None = None,
    torso_rotation: np.ndarray | None = None,
) -> ProprioceptiveSample:
    return ProprioceptiveSample(
        timestamp_ns=timestamp_ns,
        q29_rad=np.zeros(29) if q29 is None else q29,
        navigation_R_pelvis_imu=(np.eye(3) if pelvis_rotation is None else pelvis_rotation),
        navigation_R_torso_imu=(np.eye(3) if torso_rotation is None else torso_rotation),
    )


def test_every_estimator_reproduces_its_anchor(
    estimators: AnchoredCameraPoseEstimators,
) -> None:
    sample = _sample()
    reference_T_camera = np.eye(4)
    reference_T_camera[:3, :3] = Rotation.from_euler("xyz", [0.2, -0.1, 0.3]).as_matrix()
    reference_T_camera[:3, 3] = [0.2, -0.3, 0.8]
    anchor = CameraPoseAnchor(reference_T_camera=reference_T_camera, sample=sample)

    for name in ESTIMATOR_NAMES:
        assert np.allclose(
            estimators.predict(anchor, sample, name),
            reference_T_camera,
            atol=1.0e-10,
            rtol=0.0,
        )


def test_fixed_pelvis_fk_uses_measured_waist(
    estimators: AnchoredCameraPoseEstimators,
) -> None:
    reference = _sample()
    current_q = np.zeros(29)
    current_q[14] = 0.1
    current = _sample(timestamp_ns=2, q29=current_q)
    anchor = CameraPoseAnchor(reference_T_camera=np.eye(4), sample=reference)

    prediction = estimators.predict(anchor, current, "fixed_pelvis_fk")
    expected = np.linalg.inv(estimators.pelvis_T_camera(reference)) @ estimators.pelvis_T_camera(
        current
    )
    assert np.allclose(prediction, expected, atol=1.0e-10, rtol=0.0)


def test_hybrid_takes_position_from_pelvis_and_orientation_from_torso(
    estimators: AnchoredCameraPoseEstimators,
) -> None:
    reference = _sample()
    current_q = np.zeros(29)
    current_q[13] = 0.04
    current_q[14] = -0.06
    current = _sample(
        timestamp_ns=2,
        q29=current_q,
        pelvis_rotation=Rotation.from_euler("x", 0.03).as_matrix(),
        torso_rotation=Rotation.from_euler("y", -0.08).as_matrix(),
    )
    anchor = CameraPoseAnchor(reference_T_camera=np.eye(4), sample=reference)

    pelvis = estimators.predict(anchor, current, "fixed_pelvis_imu_origin_fk")
    torso = estimators.predict(anchor, current, "fixed_torso_imu_origin")
    hybrid = estimators.predict(anchor, current, "hybrid_pelvis_position_torso_orientation")

    assert np.allclose(hybrid[:3, 3], pelvis[:3, 3], atol=1.0e-12, rtol=0.0)
    assert np.allclose(hybrid[:3, :3], torso[:3, :3], atol=1.0e-12, rtol=0.0)


def test_pose_error_reports_direction_and_norm() -> None:
    predicted = np.eye(4)
    observed = np.eye(4)
    observed[:3, 3] = [0.001, -0.002, 0.002]
    observed[:3, :3] = Rotation.from_euler("z", 1.0, degrees=True).as_matrix()

    result = pose_error(predicted, observed)

    assert result["translation_reference_xyz_mm"] == pytest.approx([1.0, -2.0, 2.0])
    assert result["translation_norm_mm"] == pytest.approx(3.0)
    assert result["rotation_deg"] == pytest.approx(1.0)


def _event(index: int, *, repetition: int, arm: str, phase: str) -> BoardEvent:
    return BoardEvent(
        index=index,
        repetition=repetition,
        arm=arm,
        phase=phase,
        start_ns=index * 100,
        end_ns=index * 100 + 50,
        header_start_ns=index * 100,
        header_end_ns=index * 100 + 50,
        board_T_camera=np.eye(4),
    )


def test_legacy_evaluation_uses_immediately_preceding_return() -> None:
    events = [
        _event(0, repetition=1, arm="left", phase="lifted"),
        _event(1, repetition=1, arm="left", phase="returned"),
        _event(2, repetition=1, arm="right", phase="lifted"),
        _event(3, repetition=1, arm="right", phase="returned"),
        _event(4, repetition=2, arm="left", phase="lifted"),
        _event(5, repetition=2, arm="left", phase="returned"),
        _event(6, repetition=2, arm="right", phase="lifted"),
        _event(7, repetition=2, arm="right", phase="returned"),
    ]

    policy, pairs = _evaluation_pairs(events)

    assert policy == "legacy_immediately_preceding_return_repetitions_2_plus"
    assert [(reference.index, current.index) for reference, current in pairs] == [
        (3, 4),
        (5, 6),
    ]


def test_clock_fit_keeps_precision_with_epoch_sized_timestamps() -> None:
    header_anchor = 1_786_882_000_000_000_000
    record_anchor = header_anchor + 128_752_000_000
    slope = 1.0 - 3.0e-6
    pairs = []
    for index in range(1000):
        header = header_anchor + index * 5_000_000
        record = record_anchor + round(slope * (header - header_anchor))
        pairs.append((header, record))

    result = _clock_model(pairs)

    assert result["affine_mapping"]["rate_error_ppm"] == pytest.approx(-3.0, abs=1.0e-6)
    assert result["fit_residual_ms"]["maximum_absolute"] < 1.0e-6
