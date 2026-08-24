"""Grouped held-out validation and conservative bilateral model selection."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Literal

import numpy as np
from scipy.spatial.transform import Rotation

from g1_aprilcube_calibration.transforms import invert_transform
from g1_aprilcube_calibration.urdf_model import URDFModel
from g1_dex3_tabletop.calibration.models import (
    BilateralCalibrationDataset,
    BilateralCalibrationSample,
    BilateralModelSpec,
    CameraFrameArtifact,
)
from g1_dex3_tabletop.calibration.projection import BilateralCalibrationProjection
from g1_dex3_tabletop.calibration.solver import BilateralSolverResult

_SIDES = ("left", "right")
_GROUPINGS = ("pose", "day")

BilateralFit = Callable[
    [BilateralCalibrationDataset, BilateralModelSpec, Path],
    BilateralSolverResult,
]


@dataclass(frozen=True, slots=True)
class BilateralValidationConfig:
    pose_fold_count: int = 5
    split_seed: int = 41
    maximum_combined_rms_px: float = 8.0
    maximum_arm_rms_px: float = 8.0
    maximum_anchor_rms_px: float = 8.0
    minimum_relative_improvement: float = 0.02
    maximum_arm_regression_px: float = 0.25
    maximum_arm_regression_fraction: float = 0.05
    maximum_joint_offset_deg: float = 10.0
    maximum_joint_fold_std_deg: float = 2.0
    maximum_camera_translation_correction_m: float = 0.10
    maximum_camera_rotation_correction_deg: float = 15.0
    maximum_target_translation_correction_m: float = 0.05
    maximum_target_rotation_correction_deg: float = 30.0
    maximum_anchor_translation_drift_m: float = 0.005
    maximum_anchor_rotation_drift_deg: float = 1.0
    maximum_bilateral_anchor_translation_disagreement_m: float = 0.010
    maximum_bilateral_anchor_rotation_disagreement_deg: float = 2.0
    bootstrap_trials: int = 20
    minimum_bootstrap_success_fraction: float = 0.8
    minimum_joint_sign_agreement: float = 0.8
    minimum_effective_joint_offset_deg: float = 0.1
    require_repeated_anchors: bool = True
    require_multiple_days: bool = False

    def __post_init__(self) -> None:
        if self.pose_fold_count < 2:
            raise ValueError("pose fold count must be at least two")
        if self.split_seed < 0:
            raise ValueError("validation split seed must be non-negative")
        if self.bootstrap_trials < 1:
            raise ValueError("bilateral validation requires at least one bootstrap trial")
        positive = (
            "maximum_combined_rms_px",
            "maximum_arm_rms_px",
            "maximum_anchor_rms_px",
            "maximum_arm_regression_px",
            "maximum_arm_regression_fraction",
            "maximum_joint_offset_deg",
            "maximum_joint_fold_std_deg",
            "maximum_camera_translation_correction_m",
            "maximum_camera_rotation_correction_deg",
            "maximum_target_translation_correction_m",
            "maximum_target_rotation_correction_deg",
            "maximum_anchor_translation_drift_m",
            "maximum_anchor_rotation_drift_deg",
            "maximum_bilateral_anchor_translation_disagreement_m",
            "maximum_bilateral_anchor_rotation_disagreement_deg",
            "minimum_effective_joint_offset_deg",
        )
        for name in positive:
            value = float(getattr(self, name))
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be positive and finite")
        if not 0.0 <= self.minimum_relative_improvement < 1.0:
            raise ValueError("minimum relative improvement must lie in [0, 1)")
        for name in (
            "minimum_bootstrap_success_fraction",
            "minimum_joint_sign_agreement",
        ):
            if not 0.0 < getattr(self, name) <= 1.0:
                raise ValueError(f"{name} must lie in (0, 1]")
        if not isinstance(self.require_repeated_anchors, bool):
            raise TypeError("require_repeated_anchors must be boolean")
        if not isinstance(self.require_multiple_days, bool):
            raise TypeError("require_multiple_days must be boolean")

    def to_dict(self) -> dict[str, Any]:
        return {name: getattr(self, name) for name in self.__dataclass_fields__}


@dataclass(frozen=True, slots=True)
class BilateralResidualMetrics:
    sample_count: int
    corner_count_by_arm: dict[str, int]
    radial_squared_sum_px2_by_arm: dict[str, float]

    def __post_init__(self) -> None:
        if self.sample_count <= 0:
            raise ValueError("residual metrics require at least one sample")
        counts = {str(side): int(value) for side, value in self.corner_count_by_arm.items()}
        sums = {
            str(side): float(value) for side, value in self.radial_squared_sum_px2_by_arm.items()
        }
        if set(counts) != set(_SIDES) or set(sums) != set(_SIDES):
            raise ValueError("residual metrics must contain left and right arms")
        if any(value <= 0 for value in counts.values()):
            raise ValueError("residual metrics require corners for both arms")
        if any(not np.isfinite(value) or value < 0.0 for value in sums.values()):
            raise ValueError("residual squared sums must be finite and non-negative")
        object.__setattr__(self, "corner_count_by_arm", counts)
        object.__setattr__(self, "radial_squared_sum_px2_by_arm", sums)

    @property
    def arm_rms_px(self) -> dict[str, float]:
        return {
            side: float(
                np.sqrt(self.radial_squared_sum_px2_by_arm[side] / self.corner_count_by_arm[side])
            )
            for side in _SIDES
        }

    @property
    def combined_rms_px(self) -> float:
        return float(
            np.sqrt(
                sum(self.radial_squared_sum_px2_by_arm.values())
                / sum(self.corner_count_by_arm.values())
            )
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "sample_count": self.sample_count,
            "corner_count_by_arm": self.corner_count_by_arm,
            "radial_squared_sum_px2_by_arm": self.radial_squared_sum_px2_by_arm,
            "arm_rms_px": self.arm_rms_px,
            "combined_rms_px": self.combined_rms_px,
        }


@dataclass(frozen=True, slots=True)
class BilateralFoldValidation:
    grouping: Literal["pose", "day"]
    fold_index: int
    training_groups: tuple[str, ...]
    holdout_groups: tuple[str, ...]
    training_dataset_sha256: str
    training: BilateralResidualMetrics
    holdout: BilateralResidualMetrics
    holdout_anchors: BilateralResidualMetrics | None
    parameters: dict[str, float]
    observable: bool
    rank: int
    parameter_count: int
    condition_number: float | None
    physical_violations: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "grouping": self.grouping,
            "fold_index": self.fold_index,
            "training_groups": list(self.training_groups),
            "holdout_groups": list(self.holdout_groups),
            "training_dataset_sha256": self.training_dataset_sha256,
            "training": self.training.to_dict(),
            "holdout": self.holdout.to_dict(),
            "holdout_anchors": (
                None if self.holdout_anchors is None else self.holdout_anchors.to_dict()
            ),
            "parameters": self.parameters,
            "observability": {
                "observable": self.observable,
                "rank": self.rank,
                "parameter_count": self.parameter_count,
                "condition_number": self.condition_number,
            },
            "physical_violations": list(self.physical_violations),
        }


@dataclass(frozen=True, slots=True)
class BilateralCandidateValidation:
    model: BilateralModelSpec
    pose_folds: tuple[BilateralFoldValidation, ...]
    day_folds: tuple[BilateralFoldValidation, ...]
    pose_holdout: BilateralResidualMetrics | None
    day_holdout: BilateralResidualMetrics | None
    pose_anchor_holdout: BilateralResidualMetrics | None
    joint_parameter_mean_rad: dict[str, float]
    joint_parameter_std_rad: dict[str, float]
    eligibility_failures: tuple[str, ...]
    selection_decision: str

    @property
    def eligible(self) -> bool:
        return not self.eligibility_failures

    def to_dict(self) -> dict[str, Any]:
        return {
            "model": self.model.to_dict(),
            "model_sha256": self.model.content_sha256,
            "pose_folds": [fold.to_dict() for fold in self.pose_folds],
            "day_folds": [fold.to_dict() for fold in self.day_folds],
            "pose_holdout": (None if self.pose_holdout is None else self.pose_holdout.to_dict()),
            "day_holdout": (None if self.day_holdout is None else self.day_holdout.to_dict()),
            "pose_anchor_holdout": (
                None if self.pose_anchor_holdout is None else self.pose_anchor_holdout.to_dict()
            ),
            "joint_parameter_mean_rad": self.joint_parameter_mean_rad,
            "joint_parameter_std_rad": self.joint_parameter_std_rad,
            "eligible": self.eligible,
            "eligibility_failures": list(self.eligibility_failures),
            "selection_decision": self.selection_decision,
        }


@dataclass(frozen=True, slots=True)
class BilateralBootstrapTrial:
    trial_index: int
    dataset_sha256: str
    successful: bool
    parameters: dict[str, float]
    observable: bool
    physical_violations: tuple[str, ...]
    failure: str | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "trial_index": self.trial_index,
            "dataset_sha256": self.dataset_sha256,
            "successful": self.successful,
            "parameters": self.parameters,
            "observable": self.observable,
            "physical_violations": list(self.physical_violations),
            "failure": self.failure,
        }


@dataclass(frozen=True, slots=True)
class BilateralBootstrapValidation:
    model_sha256: str
    requested_trials: int
    successful_trials: int
    failed_trials: int
    parameter_mean: dict[str, float]
    parameter_std: dict[str, float]
    joint_sign_agreement: dict[str, float]
    passed: bool
    failures: tuple[str, ...]
    trials: tuple[BilateralBootstrapTrial, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_sha256": self.model_sha256,
            "requested_trials": self.requested_trials,
            "successful_trials": self.successful_trials,
            "failed_trials": self.failed_trials,
            "parameter_mean": self.parameter_mean,
            "parameter_std": self.parameter_std,
            "joint_sign_agreement": self.joint_sign_agreement,
            "passed": self.passed,
            "failures": list(self.failures),
            "trials": [trial.to_dict() for trial in self.trials],
        }


@dataclass(frozen=True, slots=True)
class BilateralValidationReport:
    dataset_sha256: str
    config: BilateralValidationConfig
    candidates: tuple[BilateralCandidateValidation, ...]
    bootstrap_attempts: tuple[BilateralBootstrapValidation, ...]
    bootstrap: BilateralBootstrapValidation | None
    selected_model_sha256: str | None
    selected_model_name: str | None
    schema_version: int = 1

    @property
    def passed(self) -> bool:
        return (
            self.selected_model_sha256 is not None
            and self.bootstrap is not None
            and self.bootstrap.passed
        )

    @property
    def selected_model(self) -> BilateralModelSpec:
        for candidate in self.candidates:
            if candidate.model.content_sha256 == self.selected_model_sha256:
                return candidate.model
        raise RuntimeError("bilateral validation did not select a model")

    @property
    def content_sha256(self) -> str:
        return hashlib.sha256(
            json.dumps(
                self.to_dict(include_hash=False),
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ).encode()
        ).hexdigest()

    def to_dict(self, *, include_hash: bool = True) -> dict[str, Any]:
        result = {
            "schema_version": self.schema_version,
            "dataset_sha256": self.dataset_sha256,
            "config": self.config.to_dict(),
            "candidates": [candidate.to_dict() for candidate in self.candidates],
            "bootstrap_attempts": [attempt.to_dict() for attempt in self.bootstrap_attempts],
            "bootstrap": None if self.bootstrap is None else self.bootstrap.to_dict(),
            "passed": self.passed,
            "selected_model_sha256": self.selected_model_sha256,
            "selected_model_name": self.selected_model_name,
        }
        if include_hash:
            result["content_sha256"] = self.content_sha256
        return result


def validate_and_select_bilateral_model(
    dataset: BilateralCalibrationDataset,
    urdf_model: URDFModel,
    *,
    camera_frames: CameraFrameArtifact,
    initial_hand_T_targets: Mapping[str, np.ndarray],
    models: Sequence[BilateralModelSpec],
    fit: BilateralFit,
    output_directory: str | Path,
    config: BilateralValidationConfig | None = None,
) -> BilateralValidationReport:
    """Fit nested candidates on grouped folds and choose the simplest winner."""

    settings = config or BilateralValidationConfig()
    candidates = tuple(models)
    _validate_model_hierarchy(candidates)
    _validate_dataset_groups(dataset, settings)
    output = Path(output_directory).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"bilateral validation output is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)

    validated: list[BilateralCandidateValidation] = []
    for model_index, model in enumerate(candidates):
        try:
            candidate = _validate_candidate(
                dataset,
                urdf_model,
                camera_frames=camera_frames,
                initial_hand_T_targets=initial_hand_T_targets,
                model=model,
                fit=fit,
                output_directory=output / f"model_{model_index:02d}_{model.name}",
                config=settings,
            )
        except (OSError, RuntimeError, TypeError, ValueError) as error:
            candidate = BilateralCandidateValidation(
                model=model,
                pose_folds=(),
                day_folds=(),
                pose_holdout=None,
                day_holdout=None,
                pose_anchor_holdout=None,
                joint_parameter_mean_rad={},
                joint_parameter_std_rad={},
                eligibility_failures=(f"validation failed: {type(error).__name__}: {error}",),
                selection_decision="ineligible because validation did not complete",
            )
        validated.append(candidate)

    selected_index: int | None = None
    for index, candidate in enumerate(validated):
        if not candidate.eligible or candidate.pose_holdout is None:
            continue
        if selected_index is None:
            selected_index = index
            validated[index] = replace(
                candidate,
                selection_decision="selected as the first eligible model",
            )
            continue
        incumbent = validated[selected_index]
        assert incumbent.pose_holdout is not None
        incumbent_rms = incumbent.pose_holdout.combined_rms_px
        candidate_rms = candidate.pose_holdout.combined_rms_px
        improvement = (
            0.0
            if incumbent_rms <= np.finfo(np.float64).eps
            else (incumbent_rms - candidate_rms) / incumbent_rms
        )
        regressions = []
        incumbent_arm = incumbent.pose_holdout.arm_rms_px
        candidate_arm = candidate.pose_holdout.arm_rms_px
        for side in _SIDES:
            allowed = max(
                settings.maximum_arm_regression_px,
                settings.maximum_arm_regression_fraction * incumbent_arm[side],
            )
            if candidate_arm[side] - incumbent_arm[side] > allowed:
                regressions.append(side)
        if improvement >= settings.minimum_relative_improvement and not regressions:
            validated[selected_index] = replace(
                incumbent,
                selection_decision=(
                    f"superseded by {candidate.model.name}: held-out improvement {improvement:.3%}"
                ),
            )
            validated[index] = replace(
                candidate,
                selection_decision=(
                    f"selected: held-out improvement {improvement:.3%} without an arm regression"
                ),
            )
            selected_index = index
        else:
            reason = (
                f"not selected: held-out improvement {improvement:.3%} is below "
                f"{settings.minimum_relative_improvement:.3%}"
                if improvement < settings.minimum_relative_improvement
                else "not selected: held-out regression on " + ", ".join(regressions)
            )
            validated[index] = replace(candidate, selection_decision=reason)

    bootstrap: BilateralBootstrapValidation | None = None
    bootstrap_attempts: list[BilateralBootstrapValidation] = []
    while selected_index is not None:
        candidate = validated[selected_index]
        bootstrap = _bootstrap_candidate(
            dataset,
            model=candidate.model,
            fit=fit,
            output_directory=(
                output / f"model_{selected_index:02d}_{candidate.model.name}" / "bootstrap"
            ),
            initial_hand_T_targets=initial_hand_T_targets,
            config=settings,
        )
        bootstrap_attempts.append(bootstrap)
        if bootstrap.passed:
            break
        validated[selected_index] = replace(
            candidate,
            selection_decision=("not selected after bootstrap: " + "; ".join(bootstrap.failures)),
        )
        selected_index = next(
            (
                index
                for index in range(selected_index - 1, -1, -1)
                if validated[index].eligible and validated[index].pose_holdout is not None
            ),
            None,
        )
        if selected_index is not None:
            validated[selected_index] = replace(
                validated[selected_index],
                selection_decision="selected as bootstrap-stable fallback",
            )
    selected = None if selected_index is None else validated[selected_index].model
    report = BilateralValidationReport(
        dataset_sha256=dataset.content_sha256,
        config=settings,
        candidates=tuple(validated),
        bootstrap_attempts=tuple(bootstrap_attempts),
        bootstrap=bootstrap,
        selected_model_sha256=(None if selected is None else selected.content_sha256),
        selected_model_name=(None if selected is None else selected.name),
    )
    (output / "validation_report.json").write_text(
        json.dumps(report.to_dict(), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return report


def _bootstrap_candidate(
    dataset: BilateralCalibrationDataset,
    *,
    model: BilateralModelSpec,
    fit: BilateralFit,
    output_directory: Path,
    initial_hand_T_targets: Mapping[str, np.ndarray],
    config: BilateralValidationConfig,
) -> BilateralBootstrapValidation:
    trials: list[BilateralBootstrapTrial] = []
    successful_parameters: list[dict[str, float]] = []
    for trial_index in range(config.bootstrap_trials):
        resampled = _pose_group_bootstrap_dataset(
            dataset,
            seed=config.split_seed + 1009 * (trial_index + 1),
            trial_index=trial_index,
        )
        try:
            result = fit(
                resampled,
                model,
                output_directory / f"trial_{trial_index:03d}",
            )
            if result.model.content_sha256 != model.content_sha256:
                raise ValueError("bootstrap fit returned a different model")
            physical = solution_physical_violations(
                result,
                initial_hand_T_targets=initial_hand_T_targets,
                config=config,
            )
            if not result.observability.observable:
                raise ValueError("bootstrap fit is not observable")
            if physical:
                raise ValueError("; ".join(physical))
            parameters = dict(result.parameters)
            successful_parameters.append(parameters)
            trials.append(
                BilateralBootstrapTrial(
                    trial_index=trial_index,
                    dataset_sha256=resampled.content_sha256,
                    successful=True,
                    parameters=parameters,
                    observable=True,
                    physical_violations=(),
                    failure=None,
                )
            )
        except (OSError, RuntimeError, TypeError, ValueError) as error:
            trials.append(
                BilateralBootstrapTrial(
                    trial_index=trial_index,
                    dataset_sha256=resampled.content_sha256,
                    successful=False,
                    parameters={},
                    observable=False,
                    physical_violations=(),
                    failure=f"{type(error).__name__}: {error}",
                )
            )
    successful = len(successful_parameters)
    parameter_mean: dict[str, float] = {}
    parameter_std: dict[str, float] = {}
    sign_agreement: dict[str, float] = {}
    if successful_parameters:
        for name in model_parameter_names(model):
            values = np.asarray([parameters[name] for parameters in successful_parameters])
            parameter_mean[name] = float(np.mean(values))
            parameter_std[name] = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
            if name in model.joint_offsets:
                sign_agreement[name] = float(max(np.mean(values >= 0.0), np.mean(values <= 0.0)))
    failures: list[str] = []
    success_fraction = successful / config.bootstrap_trials
    if success_fraction < config.minimum_bootstrap_success_fraction:
        failures.append(
            f"bootstrap success fraction {success_fraction:.3f} is below "
            f"{config.minimum_bootstrap_success_fraction:.3f}"
        )
    for name in model.joint_offsets:
        if name not in parameter_mean:
            continue
        if np.rad2deg(parameter_std[name]) > config.maximum_joint_fold_std_deg:
            failures.append(f"{name} bootstrap variation exceeds its limit")
        if (
            abs(np.rad2deg(parameter_mean[name])) >= config.minimum_effective_joint_offset_deg
            and sign_agreement[name] < config.minimum_joint_sign_agreement
        ):
            failures.append(f"{name} bootstrap sign is unstable")
    return BilateralBootstrapValidation(
        model_sha256=model.content_sha256,
        requested_trials=config.bootstrap_trials,
        successful_trials=successful,
        failed_trials=config.bootstrap_trials - successful,
        parameter_mean=parameter_mean,
        parameter_std=parameter_std,
        joint_sign_agreement=sign_agreement,
        passed=not failures,
        failures=tuple(failures),
        trials=tuple(trials),
    )


def model_parameter_names(model: BilateralModelSpec) -> tuple[str, ...]:
    names = list(model.joint_offsets)
    camera_names = {
        "x": "d435_joint_x",
        "y": "d435_joint_y",
        "z": "d435_joint_z",
        "roll": "d435_joint_a",
        "pitch": "d435_joint_b",
        "yaw": "d435_joint_c",
    }
    names.extend(camera_names[component] for component in model.camera_components)
    if model.optimize_hand_targets:
        for side in _SIDES:
            names.extend(
                f"{side}_calibration_target_{suffix}" for suffix in ("x", "y", "z", "a", "b", "c")
            )
    return tuple(names)


def _pose_group_bootstrap_dataset(
    dataset: BilateralCalibrationDataset,
    *,
    seed: int,
    trial_index: int,
) -> BilateralCalibrationDataset:
    grouped: dict[str, list[BilateralCalibrationSample]] = {}
    for sample in dataset.samples:
        grouped.setdefault(sample.pose_group_id, []).append(sample)
    group_ids = tuple(sorted(grouped))
    generator = np.random.default_rng(seed)
    draws = generator.choice(group_ids, size=len(group_ids), replace=True)
    samples: list[BilateralCalibrationSample] = []
    for draw_index, group_id in enumerate(draws):
        for sample_index, sample in enumerate(grouped[str(group_id)]):
            suffix = f"bootstrap_{trial_index:03d}_{draw_index:03d}_{sample_index:03d}"
            samples.append(
                replace(
                    sample,
                    capture_id=f"{sample.capture_id}_{suffix}",
                    pose_group_id=f"{sample.pose_group_id}_{suffix}",
                    frame_id=f"{sample.frame_id}_{suffix}",
                )
            )
    return replace(
        dataset,
        samples=tuple(samples),
        provenance={
            **dataset.provenance,
            "bootstrap_parent_dataset_sha256": dataset.content_sha256,
            "bootstrap_trial_index": trial_index,
            "bootstrap_seed": seed,
        },
    )


def evaluate_bilateral_solution(
    projection: BilateralCalibrationProjection,
    parameters: Mapping[str, float],
    samples: Sequence[BilateralCalibrationSample],
) -> BilateralResidualMetrics:
    """Evaluate radial pixel error independently from the Ferguson objective."""

    values = tuple(samples)
    if not values:
        raise ValueError("cannot evaluate a bilateral solution without samples")
    counts = {side: 0 for side in _SIDES}
    sums = {side: 0.0 for side in _SIDES}
    for sample in values:
        for side in _SIDES:
            predicted, depths = projection.project_side(
                sample,
                side=side,
                parameters=parameters,
            )
            if np.any(depths <= 0.0):
                raise ValueError("bilateral solution projects target points behind camera")
            observation = sample.left if side == "left" else sample.right
            residual = predicted - np.asarray(observation.image_points_px)
            counts[side] += len(residual)
            sums[side] += float(np.sum(np.square(residual)))
    return BilateralResidualMetrics(
        sample_count=len(values),
        corner_count_by_arm=counts,
        radial_squared_sum_px2_by_arm=sums,
    )


def solution_physical_violations(
    result: BilateralSolverResult,
    *,
    initial_hand_T_targets: Mapping[str, np.ndarray],
    config: BilateralValidationConfig,
) -> tuple[str, ...]:
    """Return explicit plausibility failures for one fitted parameter vector."""

    violations: list[str] = []
    for name in result.model.joint_offsets:
        value = abs(np.rad2deg(result.parameters[name]))
        if value > config.maximum_joint_offset_deg:
            violations.append(
                f"{name} offset {value:.3f}deg exceeds {config.maximum_joint_offset_deg:.3f}deg"
            )
    camera_translation = np.asarray(
        [result.parameters.get(f"d435_joint_{axis}", 0.0) for axis in "xyz"]
    )
    if np.linalg.norm(camera_translation) > config.maximum_camera_translation_correction_m:
        violations.append("camera translation correction exceeds its physical limit")
    camera_rotation = np.asarray(
        [result.parameters.get(f"d435_joint_{axis}", 0.0) for axis in "abc"]
    )
    if np.linalg.norm(np.rad2deg(camera_rotation)) > config.maximum_camera_rotation_correction_deg:
        violations.append("camera rotation correction exceeds its physical limit")
    for side in _SIDES:
        delta = invert_transform(initial_hand_T_targets[side]) @ result.hand_T_targets[side]
        translation = float(np.linalg.norm(delta[:3, 3]))
        rotation = float(np.rad2deg(Rotation.from_matrix(delta[:3, :3]).magnitude()))
        if translation > config.maximum_target_translation_correction_m:
            violations.append(f"{side} target translation correction exceeds its limit")
        if rotation > config.maximum_target_rotation_correction_deg:
            violations.append(f"{side} target rotation correction exceeds its limit")
    return tuple(violations)


def _validate_candidate(
    dataset: BilateralCalibrationDataset,
    urdf_model: URDFModel,
    *,
    camera_frames: CameraFrameArtifact,
    initial_hand_T_targets: Mapping[str, np.ndarray],
    model: BilateralModelSpec,
    fit: BilateralFit,
    output_directory: Path,
    config: BilateralValidationConfig,
) -> BilateralCandidateValidation:
    pose_assignments = _fold_assignments(
        dataset.samples,
        grouping="pose",
        fold_count=config.pose_fold_count,
        seed=config.split_seed,
    )
    day_groups = {sample.day_group_id for sample in dataset.samples}
    day_assignments = (
        _fold_assignments(
            dataset.samples,
            grouping="day",
            fold_count=len(day_groups),
            seed=config.split_seed,
        )
        if len(day_groups) >= 2
        else ()
    )
    pose_folds = _fit_folds(
        dataset,
        urdf_model,
        camera_frames=camera_frames,
        initial_hand_T_targets=initial_hand_T_targets,
        model=model,
        fit=fit,
        assignments=pose_assignments,
        output_directory=output_directory / "pose_grouped",
        config=config,
    )
    day_folds = _fit_folds(
        dataset,
        urdf_model,
        camera_frames=camera_frames,
        initial_hand_T_targets=initial_hand_T_targets,
        model=model,
        fit=fit,
        assignments=day_assignments,
        output_directory=output_directory / "day_grouped",
        config=config,
    )
    pose_holdout = _aggregate_metrics(tuple(fold.holdout for fold in pose_folds))
    day_holdout = (
        _aggregate_metrics(tuple(fold.holdout for fold in day_folds)) if day_folds else None
    )
    anchor_metrics = tuple(
        fold.holdout_anchors for fold in pose_folds if fold.holdout_anchors is not None
    )
    pose_anchor_holdout = _aggregate_metrics(anchor_metrics) if anchor_metrics else None
    joint_mean, joint_std = _joint_parameter_statistics(model, pose_folds)
    failures = _eligibility_failures(
        pose_folds=pose_folds,
        day_folds=day_folds,
        pose_holdout=pose_holdout,
        day_holdout=day_holdout,
        pose_anchor_holdout=pose_anchor_holdout,
        joint_std=joint_std,
        config=config,
    )
    return BilateralCandidateValidation(
        model=model,
        pose_folds=pose_folds,
        day_folds=day_folds,
        pose_holdout=pose_holdout,
        day_holdout=day_holdout,
        pose_anchor_holdout=pose_anchor_holdout,
        joint_parameter_mean_rad=joint_mean,
        joint_parameter_std_rad=joint_std,
        eligibility_failures=failures,
        selection_decision=(
            "eligible; awaiting nested-model comparison"
            if not failures
            else "ineligible: " + "; ".join(failures)
        ),
    )


def _fit_folds(
    dataset: BilateralCalibrationDataset,
    urdf_model: URDFModel,
    *,
    camera_frames: CameraFrameArtifact,
    initial_hand_T_targets: Mapping[str, np.ndarray],
    model: BilateralModelSpec,
    fit: BilateralFit,
    assignments: tuple[tuple[str, ...], ...],
    output_directory: Path,
    config: BilateralValidationConfig,
) -> tuple[BilateralFoldValidation, ...]:
    folds: list[BilateralFoldValidation] = []
    grouping: Literal["pose", "day"] = "day" if output_directory.name == "day_grouped" else "pose"
    group_attribute = "day_group_id" if grouping == "day" else "pose_group_id"
    all_groups = {getattr(sample, group_attribute) for sample in dataset.samples}
    for fold_index, holdout_groups in enumerate(assignments):
        holdout_set = set(holdout_groups)
        training_samples = tuple(
            sample
            for sample in dataset.samples
            if getattr(sample, group_attribute) not in holdout_set
        )
        holdout_samples = tuple(
            sample for sample in dataset.samples if getattr(sample, group_attribute) in holdout_set
        )
        training_dataset = replace(
            dataset,
            samples=training_samples,
            provenance={
                **dataset.provenance,
                "validation_parent_dataset_sha256": dataset.content_sha256,
                "validation_grouping": grouping,
                "validation_fold_index": fold_index,
                "validation_holdout_groups": list(holdout_groups),
            },
        )
        result = fit(
            training_dataset,
            model,
            output_directory / f"fold_{fold_index:02d}",
        )
        if result.model.content_sha256 != model.content_sha256:
            raise ValueError("fit callback returned a result for a different model")
        projection = BilateralCalibrationProjection(
            urdf_model,
            camera_frames=camera_frames,
            model=model,
            initial_hand_T_targets=initial_hand_T_targets,
        )
        training = evaluate_bilateral_solution(
            projection,
            result.parameters,
            training_samples,
        )
        holdout = evaluate_bilateral_solution(
            projection,
            result.parameters,
            holdout_samples,
        )
        anchors = tuple(sample for sample in holdout_samples if sample.capture_role == "anchor")
        folds.append(
            BilateralFoldValidation(
                grouping=grouping,
                fold_index=fold_index,
                training_groups=tuple(sorted(all_groups - holdout_set)),
                holdout_groups=holdout_groups,
                training_dataset_sha256=training_dataset.content_sha256,
                training=training,
                holdout=holdout,
                holdout_anchors=(
                    evaluate_bilateral_solution(projection, result.parameters, anchors)
                    if anchors
                    else None
                ),
                parameters=result.parameters,
                observable=result.observability.observable,
                rank=result.observability.rank,
                parameter_count=result.observability.parameter_count,
                condition_number=result.observability.condition_number,
                physical_violations=solution_physical_violations(
                    result,
                    initial_hand_T_targets=initial_hand_T_targets,
                    config=config,
                ),
            )
        )
    return tuple(folds)


def _fold_assignments(
    samples: Sequence[BilateralCalibrationSample],
    *,
    grouping: Literal["pose", "day"],
    fold_count: int,
    seed: int,
) -> tuple[tuple[str, ...], ...]:
    if grouping not in _GROUPINGS:
        raise ValueError("validation grouping must be pose or day")
    attribute = "pose_group_id" if grouping == "pose" else "day_group_id"
    groups = sorted(
        {getattr(sample, attribute) for sample in samples},
        key=lambda value: hashlib.sha256(f"{seed}:{value}".encode()).hexdigest(),
    )
    actual_fold_count = min(fold_count, len(groups))
    if actual_fold_count < 2:
        raise ValueError(f"{grouping}-group validation requires at least two groups")
    assignments = [[] for _ in range(actual_fold_count)]
    for index, group in enumerate(groups):
        assignments[index % actual_fold_count].append(group)
    return tuple(tuple(sorted(values)) for values in assignments)


def _aggregate_metrics(
    values: Sequence[BilateralResidualMetrics],
) -> BilateralResidualMetrics:
    metrics = tuple(values)
    if not metrics:
        raise ValueError("cannot aggregate empty residual metrics")
    return BilateralResidualMetrics(
        sample_count=sum(item.sample_count for item in metrics),
        corner_count_by_arm={
            side: sum(item.corner_count_by_arm[side] for item in metrics) for side in _SIDES
        },
        radial_squared_sum_px2_by_arm={
            side: sum(item.radial_squared_sum_px2_by_arm[side] for item in metrics)
            for side in _SIDES
        },
    )


def _joint_parameter_statistics(
    model: BilateralModelSpec,
    folds: Sequence[BilateralFoldValidation],
) -> tuple[dict[str, float], dict[str, float]]:
    means: dict[str, float] = {}
    deviations: dict[str, float] = {}
    for name in model.joint_offsets:
        values = np.asarray([fold.parameters[name] for fold in folds])
        means[name] = float(np.mean(values))
        deviations[name] = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
    return means, deviations


def _eligibility_failures(
    *,
    pose_folds: Sequence[BilateralFoldValidation],
    day_folds: Sequence[BilateralFoldValidation],
    pose_holdout: BilateralResidualMetrics,
    day_holdout: BilateralResidualMetrics | None,
    pose_anchor_holdout: BilateralResidualMetrics | None,
    joint_std: Mapping[str, float],
    config: BilateralValidationConfig,
) -> tuple[str, ...]:
    failures: list[str] = []
    all_folds = tuple(pose_folds) + tuple(day_folds)
    if any(not fold.observable for fold in all_folds):
        failures.append("one or more training folds are not observable")
    physical = [item for fold in all_folds for item in fold.physical_violations]
    if physical:
        failures.append("physical plausibility failed: " + "; ".join(sorted(set(physical))))
    if pose_holdout.combined_rms_px > config.maximum_combined_rms_px:
        failures.append("pose-grouped combined RMS exceeds its limit")
    for side, value in pose_holdout.arm_rms_px.items():
        if value > config.maximum_arm_rms_px:
            failures.append(f"pose-grouped {side} RMS exceeds its limit")
    if (
        pose_anchor_holdout is not None
        and pose_anchor_holdout.combined_rms_px > config.maximum_anchor_rms_px
    ):
        failures.append("repeated-anchor RMS exceeds its limit")
    if day_holdout is not None:
        if day_holdout.combined_rms_px > config.maximum_combined_rms_px:
            failures.append("day-grouped combined RMS exceeds its limit")
        for side, value in day_holdout.arm_rms_px.items():
            if value > config.maximum_arm_rms_px:
                failures.append(f"day-grouped {side} RMS exceeds its limit")
    for name, value in joint_std.items():
        if np.rad2deg(value) > config.maximum_joint_fold_std_deg:
            failures.append(f"{name} varies too much across pose folds")
    return tuple(failures)


def _validate_model_hierarchy(models: Sequence[BilateralModelSpec]) -> None:
    if not models:
        raise ValueError("bilateral validation requires at least one model")
    if len({model.content_sha256 for model in models}) != len(models):
        raise ValueError("bilateral model hierarchy contains duplicates")
    baseline = models[0]
    previous_offsets: set[str] = set()
    for model in models:
        if model.camera_components != baseline.camera_components:
            raise ValueError("nested models must use the same camera components")
        if model.optimize_hand_targets != baseline.optimize_hand_targets:
            raise ValueError("nested models must use the same hand-target policy")
        offsets = set(model.joint_offsets)
        if not previous_offsets.issubset(offsets):
            raise ValueError("joint-offset candidates must be nested in declaration order")
        previous_offsets = offsets


def _validate_dataset_groups(
    dataset: BilateralCalibrationDataset,
    config: BilateralValidationConfig,
) -> None:
    pose_groups = {sample.pose_group_id for sample in dataset.samples}
    if len(pose_groups) < 2:
        raise ValueError("pose-grouped validation requires at least two pose groups")
    day_groups = {sample.day_group_id for sample in dataset.samples}
    if config.require_multiple_days and len(day_groups) < 2:
        raise ValueError("commissioning validation requires at least two capture days")
    if config.require_repeated_anchors:
        anchor_counts: dict[str, int] = {}
        for sample in dataset.samples:
            if sample.capture_role == "anchor":
                anchor_counts[sample.pose_group_id] = (
                    anchor_counts.get(sample.pose_group_id, 0) + 1
                )
        if not anchor_counts or max(anchor_counts.values()) < 3:
            raise ValueError(
                "validation requires the same bilateral anchor pose at least three times"
            )
