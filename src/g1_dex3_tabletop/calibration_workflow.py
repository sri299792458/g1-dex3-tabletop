"""Focused fixed-marker calibration dataset, solve, and bundle workflow."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import yaml

from g1_aprilcube_calibration.authored_collection import (
    modeled_hand_T_target_from_hardware,
    validate_hardware_target_profile,
)
from g1_aprilcube_calibration.calibration_bundle import (
    CalibrationBundle,
    CalibrationTarget,
)
from g1_aprilcube_calibration.calibration_pipeline import (
    CalibrationPipeline,
    PipelineConfig,
    PipelineResult,
)
from g1_aprilcube_calibration.dataset_builder import CalibrationDataset
from g1_aprilcube_calibration.joint_map import arm_hand_link
from g1_aprilcube_calibration.robot_calibration_bridge import (
    ROBOT_CALIBRATION_REVISION,
)
from g1_aprilcube_calibration.urdf_model import URDFModel


def load_fixed_marker_profile(
    dataset: CalibrationDataset,
    *,
    hardware_path: Path,
    target_path: Path,
) -> tuple[dict, np.ndarray]:
    """Bind a dataset to the exact arm, target bytes, and CAD mount transform."""

    hardware = yaml.safe_load(hardware_path.read_text(encoding="utf-8"))
    if not isinstance(hardware, dict):
        raise TypeError("hardware configuration must contain a mapping")
    configured_arm = str(hardware["robot"]["calibration_arm"])
    if configured_arm != str(hardware["control"]["calibration_arm"]):
        raise ValueError("hardware robot/control calibration arms disagree")
    if configured_arm != dataset.calibration_arm:
        raise ValueError("dataset calibration arm differs from its hardware profile")
    target_bytes = target_path.read_bytes()
    if hashlib.sha256(target_bytes).hexdigest() != dataset.target_artifact_sha256:
        raise ValueError("dataset target differs from the selected target profile")
    target = json.loads(target_bytes)
    validate_hardware_target_profile(hardware, target)
    return hardware, modeled_hand_T_target_from_hardware(hardware)


def solve_fixed_marker_calibration(
    dataset: CalibrationDataset,
    *,
    hardware_path: Path,
    target_path: Path,
    urdf_path: Path,
    output_directory: Path,
    robot_calibration_directory: Path,
    runner_path: Path,
    bootstrap_trials: int = 50,
    bootstrap_seed: int = 17,
    timeout_s: float = 300.0,
) -> tuple[PipelineResult, dict]:
    """Run the sole Ferguson backend with the physical marker mount held fixed."""

    hardware, palm_T_marker = load_fixed_marker_profile(
        dataset,
        hardware_path=hardware_path,
        target_path=target_path,
    )
    model = URDFModel(urdf_path)
    pipeline = CalibrationPipeline(
        model,
        calibration_arm=dataset.calibration_arm,
        robot_calibration_directory=robot_calibration_directory,
        runner_path=runner_path,
        config=PipelineConfig(
            holdout_fraction=0.2,
            bootstrap_trials=bootstrap_trials,
            bootstrap_seed=bootstrap_seed,
            optimize_hand_target=False,
            free_joint_offsets=(),
            native_timeout_s=timeout_s,
        ),
    )
    provenance = {
        "command": "g1-tabletop solve-calibration",
        "optimizer_backend": "mikeferguson/robot_calibration:Ceres",
        "robot_calibration_revision": ROBOT_CALIBRATION_REVISION,
        "dataset_sha256": dataset.content_sha256,
        "target_transform_policy": "fixed_CAD_palm_T_marker",
        "hardware_config_sha256": hashlib.sha256(hardware_path.read_bytes()).hexdigest(),
        "target_config_sha256": dataset.target_artifact_sha256,
        "target_mount": hardware["robot"]["calibration_target_mount"],
        "calibration_arm": dataset.calibration_arm,
    }
    result = pipeline.run(
        dataset,
        initial_hand_T_target=palm_T_marker,
        output_directory=output_directory,
        provenance=provenance,
    )
    return result, provenance


def write_calibration_bundle(
    *,
    result: PipelineResult,
    dataset: CalibrationDataset,
    output_directory: Path,
    urdf_path: Path,
    left_hardware_path: Path,
    left_target_path: Path,
    right_hardware_path: Path,
    right_target_path: Path,
    provenance: dict,
) -> Path:
    """Emit the removable camera overlay plus both known plate transforms."""

    targets: dict[str, CalibrationTarget] = {}
    for side, hardware_path, target_path in (
        ("left", left_hardware_path, left_target_path),
        ("right", right_hardware_path, right_target_path),
    ):
        hardware = yaml.safe_load(hardware_path.read_text(encoding="utf-8"))
        target = json.loads(target_path.read_text(encoding="utf-8"))
        validate_hardware_target_profile(hardware, target)
        targets[side] = CalibrationTarget(
            hand_frame=arm_hand_link(side),
            target_frame=f"{side}_calibration_target",
            hand_T_target=modeled_hand_T_target_from_hardware(hardware),
            target_artifact_sha256=hashlib.sha256(target_path.read_bytes()).hexdigest(),
        )

    bundle = CalibrationBundle(
        bundle_id=output_directory.name,
        base_urdf_sha256=hashlib.sha256(urdf_path.read_bytes()).hexdigest(),
        torso_T_camera=result.solution.torso_T_camera,
        joint_position_offsets_rad={
            name: value
            for name, value in result.solution.offsets.items()
            if name.endswith("_joint")
        },
        targets=targets,
        provenance={
            **provenance,
            "dataset_session_id": dataset.session_id,
            "source_result": "result.json",
        },
        validation={
            "status": "heldout_calibration_validation",
            "training_radial_rms_px": result.residuals.training.rms_px,
            "holdout_radial_rms_px": result.residuals.holdout.rms_px,
            "observable": result.solution.observability.observable,
            "rank": result.solution.observability.rank,
            "parameter_count": result.solution.observability.parameter_count,
            "bootstrap_successful": result.bootstrap.successful_trials,
            "bootstrap_failed": result.bootstrap.failed_trials,
        },
    )
    path = output_directory / "calibration_bundle.json"
    path.write_text(
        json.dumps(bundle.to_dict(), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return path
