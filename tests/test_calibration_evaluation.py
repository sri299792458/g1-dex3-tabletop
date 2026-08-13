from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import yaml

from g1_aprilcube_calibration.calibration_evaluation import (
    CAMERA_OFFSET_NAMES,
    TARGET_OFFSET_NAMES,
    NativeCalibrationProjection,
    NativeCalibrationSolution,
)
from g1_aprilcube_calibration.camera_initialization import (
    realsense_link_T_color_optical,
)
from g1_aprilcube_calibration.dataset_builder import deterministic_holdout_split
from g1_aprilcube_calibration.residual_report import (
    CalibrationRunExporter,
    build_residual_report,
    load_exported_result,
)
from g1_aprilcube_calibration.synthetic import make_synthetic_dataset
from g1_aprilcube_calibration.transforms import (
    invert_transform,
    transform_to_pose_vector,
)
from g1_aprilcube_calibration.urdf_model import URDFModel

ROOT = Path(__file__).parents[1]
URDF = ROOT / "third_party/unitree_ros/robots/g1_description/g1_29dof_rev_1_0.urdf"
TARGET = ROOT / "third_party/aprilcube/models/dex3_safe_cube/config.json"


def _truth_offsets(model: URDFModel, truth) -> dict[str, float]:
    torso_T_d435 = model.transform("torso_link", "d435_link", {})
    camera_delta = (
        invert_transform(torso_T_d435)
        @ truth.torso_T_camera
        @ invert_transform(realsense_link_T_color_optical())
    )
    return {
        **dict(
            zip(
                CAMERA_OFFSET_NAMES,
                transform_to_pose_vector(camera_delta),
                strict=True,
            )
        ),
        **dict(
            zip(
                TARGET_OFFSET_NAMES,
                transform_to_pose_vector(truth.hand_T_target),
                strict=True,
            )
        ),
    }


def test_native_projection_reproduces_noiseless_truth_without_optimizing() -> None:
    model = URDFModel(URDF)
    dataset, truth = make_synthetic_dataset(model, TARGET, pose_count=20, seed=5)
    projection = NativeCalibrationProjection(
        model, calibration_arm="left", fixed_hand_T_target=None
    )
    offsets = _truth_offsets(model, truth)
    residual = projection.pixel_residuals(offsets, dataset.samples)
    assert np.max(np.abs(residual)) < 1e-8
    torso_T_camera, hand_T_target = projection.transforms(offsets)
    np.testing.assert_allclose(torso_T_camera, truth.torso_T_camera, atol=1e-12)
    np.testing.assert_allclose(hand_T_target, truth.hand_T_target, atol=1e-12)
    observability = projection.observability(offsets, dataset.samples)
    assert observability.rank == 12
    assert observability.observable


def test_fixed_target_projection_uses_native_joint_offsets() -> None:
    model = URDFModel(URDF)
    dataset, truth = make_synthetic_dataset(model, TARGET, pose_count=5, seed=9)
    full_offsets = _truth_offsets(model, truth)
    camera_offsets = {name: full_offsets[name] for name in CAMERA_OFFSET_NAMES}
    projection = NativeCalibrationProjection(
        model,
        calibration_arm="left",
        fixed_hand_T_target=truth.hand_T_target,
    )
    baseline = projection.pixel_residuals(camera_offsets, dataset.samples)
    assert np.max(np.abs(baseline)) < 1e-8
    with_joint = dict(camera_offsets)
    with_joint["left_elbow_joint"] = 0.05
    assert np.linalg.norm(projection.pixel_residuals(with_joint, dataset.samples)) > 1.0


def test_native_result_export_remains_usable_by_downstream_tools(
    tmp_path: Path,
) -> None:
    model = URDFModel(URDF)
    dataset, truth = make_synthetic_dataset(model, TARGET, pose_count=20, seed=11)
    training, holdout = deterministic_holdout_split(dataset)
    projection = NativeCalibrationProjection(
        model, calibration_arm="left", fixed_hand_T_target=None
    )
    offsets = _truth_offsets(model, truth)
    residual = projection.pixel_residuals(offsets, training)
    solution = NativeCalibrationSolution(
        torso_T_camera=truth.torso_T_camera,
        hand_T_target=truth.hand_T_target,
        offsets=offsets,
        success=True,
        message="Ceres termination: CONVERGENCE",
        evaluations=8,
        cost=0.0,
        optimization_rms_px=float(np.sqrt(np.mean(np.square(residual)))),
        observability=projection.observability(offsets, training),
    )
    report = build_residual_report(
        projection,
        solution,
        training_samples=training,
        holdout_samples=holdout,
    )
    output = tmp_path / "native_run"
    CalibrationRunExporter().export(
        output,
        dataset=dataset,
        result=solution,
        residuals=report,
        training_capture_ids=[sample.capture_id for sample in training],
        holdout_capture_ids=[sample.capture_id for sample in holdout],
        provenance={"optimizer_backend": solution.backend},
        bootstrap={},
    )
    exported = json.loads((output / "result.json").read_text())
    extrinsics = yaml.safe_load((output / "calibrated_extrinsics.yaml").read_text())
    assert exported["solution"]["backend"] == solution.backend
    assert extrinsics["optimizer_backend"] == solution.backend
    assert len(extrinsics["native_offsets"]) == 12
    assert (
        load_exported_result(output / "result.json")["content_sha256"]
        == (exported["content_sha256"])
    )
