"""Direct, hash-bound export of a selected bilateral calibration solution."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from g1_aprilcube_calibration.calibration_bundle import (
    CalibrationBundle,
    CalibrationTarget,
)
from g1_aprilcube_calibration.joint_map import arm_hand_link
from g1_dex3_tabletop.calibration.anchors import BilateralAnchorDriftReport
from g1_dex3_tabletop.calibration.models import (
    BilateralCalibrationDataset,
    CameraFrameArtifact,
)
from g1_dex3_tabletop.calibration.solver import BilateralSolverResult
from g1_dex3_tabletop.calibration.validation import (
    BilateralValidationReport,
    solution_physical_violations,
)

_SIDES = ("left", "right")


def write_bilateral_calibration_bundle(
    *,
    result: BilateralSolverResult,
    dataset: BilateralCalibrationDataset,
    validation_report: BilateralValidationReport,
    anchor_drift_report: BilateralAnchorDriftReport,
    camera_frames: CameraFrameArtifact,
    initial_hand_T_targets: Mapping[str, np.ndarray],
    base_urdf_path: str | Path,
    destination: str | Path,
    bundle_id: str,
    provenance: Mapping[str, Any],
) -> Path:
    """Write the production overlay without any manual parameter transcription."""

    source_urdf = Path(base_urdf_path).resolve()
    output = Path(destination).resolve()
    if not source_urdf.is_file():
        raise FileNotFoundError(f"base URDF is missing: {source_urdf}")
    if output.exists():
        raise FileExistsError(f"calibration bundle already exists: {output}")
    if not bundle_id.strip():
        raise ValueError("bilateral calibration bundle ID must be non-empty")
    if not provenance:
        raise ValueError("bilateral bundle provenance must be non-empty")
    actual_urdf_sha256 = hashlib.sha256(source_urdf.read_bytes()).hexdigest()
    if actual_urdf_sha256 != dataset.urdf_sha256:
        raise ValueError("bilateral dataset belongs to a different base URDF")
    if dataset.rgb_optical_transform_sha256 != camera_frames.content_sha256:
        raise ValueError("bilateral dataset belongs to a different camera-frame artifact")
    if validation_report.dataset_sha256 != dataset.content_sha256:
        raise ValueError("validation report belongs to a different bilateral dataset")
    if not validation_report.passed:
        raise ValueError("bilateral validation did not select a model")
    if result.model.content_sha256 != validation_report.selected_model_sha256:
        raise ValueError("full-dataset result does not use the selected validation model")
    if anchor_drift_report.dataset_sha256 != dataset.content_sha256:
        raise ValueError("anchor drift report belongs to a different bilateral dataset")
    if anchor_drift_report.model_sha256 != result.model.content_sha256:
        raise ValueError("anchor drift report belongs to a different bilateral model")
    if not anchor_drift_report.passed:
        raise ValueError(
            "repeated bilateral anchors failed: " + "; ".join(anchor_drift_report.failures)
        )
    if not result.observability.observable:
        raise ValueError("cannot export an unobservable bilateral solution")
    if set(initial_hand_T_targets) != set(_SIDES):
        raise ValueError("initial hand-target transforms must contain left and right")
    physical_violations = solution_physical_violations(
        result,
        initial_hand_T_targets=initial_hand_T_targets,
        config=validation_report.config,
    )
    if physical_violations:
        raise ValueError(
            "full-dataset solution is not physically plausible: " + "; ".join(physical_violations)
        )
    expected_parameters = set(result.observability.parameter_names)
    if set(result.parameters) != expected_parameters:
        raise ValueError("full-dataset result parameter set differs from observability report")
    if not set(result.model.joint_offsets).issubset(result.parameters):
        raise ValueError("full-dataset result is missing selected joint offsets")

    target_hashes = {
        "left": dataset.left_target_artifact_sha256,
        "right": dataset.right_target_artifact_sha256,
    }
    targets = {
        side: CalibrationTarget(
            hand_frame=arm_hand_link(side),
            target_frame=f"{side}_calibration_target",
            hand_T_target=result.hand_T_targets[side],
            target_artifact_sha256=target_hashes[side],
        )
        for side in _SIDES
    }
    selected_candidate = next(
        candidate
        for candidate in validation_report.candidates
        if candidate.model.content_sha256 == validation_report.selected_model_sha256
    )
    bundle = CalibrationBundle(
        bundle_id=bundle_id,
        base_urdf_sha256=actual_urdf_sha256,
        torso_T_camera=result.torso_T_camera,
        joint_position_offsets_rad={
            name: result.parameters[name] for name in result.model.joint_offsets
        },
        targets=targets,
        provenance={
            **dict(provenance),
            "workflow": "same_frame_bilateral_aprilcube_v1",
            "optimizer_backend": "mikeferguson/robot_calibration:Ceres",
            "dataset_sha256": dataset.content_sha256,
            "source_session_manifest_sha256_by_id": (dataset.session_manifest_sha256_by_id),
            "pose_design_sha256": dataset.pose_design_sha256,
            "model_sha256": result.model.content_sha256,
            "validation_report_sha256": validation_report.content_sha256,
            "camera_frame_artifact": camera_frames.to_dict(),
            "target_artifact_sha256": target_hashes,
            "fitted_parameters": dict(sorted(result.parameters.items())),
            "fitted_parameter_units": {
                name: _parameter_unit(name) for name in sorted(result.parameters)
            },
        },
        validation={
            "status": "bilateral_grouped_heldout_validation",
            "passed": True,
            "report": validation_report.to_dict(),
            "anchor_drift": anchor_drift_report.to_dict(),
            "selected_pose_holdout": (
                None
                if selected_candidate.pose_holdout is None
                else selected_candidate.pose_holdout.to_dict()
            ),
            "selected_day_holdout": (
                None
                if selected_candidate.day_holdout is None
                else selected_candidate.day_holdout.to_dict()
            ),
            "selected_anchor_holdout": (
                None
                if selected_candidate.pose_anchor_holdout is None
                else selected_candidate.pose_anchor_holdout.to_dict()
            ),
            "full_dataset_fit": {
                "combined_radial_rms_px": result.combined_radial_rms_px,
                "arm_radial_rms_px": result.arm_radial_rms_px,
                "observability": result.observability.to_dict(),
                "iterations": result.iterations,
                "final_cost": result.final_cost,
                "termination": result.termination,
            },
        },
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    document = (
        json.dumps(
            bundle.to_dict(),
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
    )
    file_descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output.name}.",
        dir=output.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(file_descriptor, "w", encoding="utf-8") as stream:
            stream.write(document)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)
    CalibrationBundle.load(output)
    return output


def _parameter_unit(name: str) -> str:
    if name.endswith("_joint"):
        return "rad"
    if name.startswith("d435_joint_"):
        return "m" if name.rsplit("_", 1)[-1] in {"x", "y", "z"} else "rad"
    if name.startswith(("left_calibration_target_", "right_calibration_target_")):
        return "m" if name.rsplit("_", 1)[-1] in {"x", "y", "z"} else "rad"
    raise ValueError(f"unknown bilateral fitted parameter: {name}")
