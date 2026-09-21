"""Train/holdout orchestration around the sole Ferguson/Ceres backend."""

from __future__ import annotations

import tempfile
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

from g1_aprilcube_calibration.calibration_evaluation import (
    NativeCalibrationProjection,
    NativeCalibrationSolution,
)
from g1_aprilcube_calibration.dataset_builder import (
    CalibrationDataset,
    CalibrationSample,
    deterministic_holdout_split,
)
from g1_aprilcube_calibration.residual_report import (
    CalibrationRunExporter,
    ResidualReport,
    build_residual_report,
)
from g1_aprilcube_calibration.robot_calibration_bridge import (
    solve_robot_calibration_dataset,
)
from g1_aprilcube_calibration.transforms import validate_transform
from g1_aprilcube_calibration.urdf_model import URDFModel


@dataclass(frozen=True, slots=True)
class PipelineConfig:
    holdout_fraction: float = 0.2
    bootstrap_trials: int = 50
    bootstrap_seed: int = 17
    optimize_hand_target: bool = False
    free_joint_offsets: tuple[str, ...] = ()
    joint_offset_prior_sigma_deg: float = 5.0
    native_timeout_s: float = 300.0

    def __post_init__(self) -> None:
        if not 0 < self.holdout_fraction < 1:
            raise ValueError("holdout fraction must lie strictly between zero and one")
        if self.bootstrap_trials < 0:
            raise ValueError("bootstrap trials cannot be negative")
        if self.bootstrap_seed < 0:
            raise ValueError("bootstrap seed cannot be negative")
        if not isinstance(self.optimize_hand_target, bool):
            raise TypeError("optimize_hand_target must be boolean")
        if self.joint_offset_prior_sigma_deg <= 0:
            raise ValueError("joint offset prior sigma must be positive")
        if self.native_timeout_s <= 0:
            raise ValueError("native optimizer timeout must be positive")
        object.__setattr__(
            self,
            "free_joint_offsets",
            tuple(dict.fromkeys(str(name) for name in self.free_joint_offsets)),
        )


@dataclass(frozen=True, slots=True)
class BootstrapReport:
    requested_trials: int
    successful_trials: int
    failed_trials: int
    parameter_names: tuple[str, ...]
    parameter_stddev: tuple[float | None, ...]

    def to_dict(self) -> dict:
        return {
            "requested_trials": self.requested_trials,
            "successful_trials": self.successful_trials,
            "failed_trials": self.failed_trials,
            "parameter_names": list(self.parameter_names),
            "parameter_stddev": list(self.parameter_stddev),
        }


@dataclass(frozen=True, slots=True)
class PipelineResult:
    solution: NativeCalibrationSolution
    residuals: ResidualReport
    bootstrap: BootstrapReport
    training_samples: tuple[CalibrationSample, ...]
    holdout_samples: tuple[CalibrationSample, ...]


class CalibrationPipeline:
    def __init__(
        self,
        model: URDFModel,
        *,
        calibration_arm: str,
        robot_calibration_directory: str | Path,
        runner_path: str | Path,
        config: PipelineConfig | None = None,
    ) -> None:
        self.model = model
        self.calibration_arm = calibration_arm
        self.robot_calibration_directory = Path(robot_calibration_directory)
        self.runner_path = Path(runner_path)
        self.config = config or PipelineConfig()

    def run(
        self,
        dataset: CalibrationDataset,
        *,
        initial_hand_T_target: np.ndarray,
        output_directory: str | Path,
        provenance: dict | None = None,
    ) -> PipelineResult:
        if dataset.urdf_sha256 != self.model.sha256:
            raise ValueError("dataset and optimizer URDF hashes do not match")
        if dataset.calibration_arm != self.calibration_arm:
            raise ValueError("dataset and optimizer calibration arms do not match")
        target = validate_transform(initial_hand_T_target)
        output = Path(output_directory).resolve()
        if output.exists():
            raise FileExistsError(f"calibration run already exists: {output}")
        output.mkdir(parents=True)
        training, holdout = deterministic_holdout_split(
            dataset, holdout_fraction=self.config.holdout_fraction
        )
        training_dataset = replace(dataset, samples=training)
        solution = self._solve(
            training_dataset,
            target,
            output / "native_training_fit",
        )
        projection = NativeCalibrationProjection(
            self.model,
            calibration_arm=self.calibration_arm,
            fixed_hand_T_target=(None if self.config.optimize_hand_target else target),
        )
        residuals = build_residual_report(
            projection,
            solution,
            training_samples=training,
            holdout_samples=holdout,
        )
        bootstrap = self._bootstrap(training_dataset, solution, target)
        result = PipelineResult(solution, residuals, bootstrap, training, holdout)
        CalibrationRunExporter().export(
            output,
            dataset=dataset,
            result=solution,
            residuals=residuals,
            training_capture_ids=[sample.capture_id for sample in training],
            holdout_capture_ids=[sample.capture_id for sample in holdout],
            provenance={} if provenance is None else provenance,
            bootstrap=bootstrap.to_dict(),
        )
        return result

    def _solve(
        self,
        dataset: CalibrationDataset,
        initial_hand_T_target: np.ndarray,
        output_directory: Path,
    ) -> NativeCalibrationSolution:
        return solve_robot_calibration_dataset(
            dataset,
            self.model,
            output_directory,
            initial_hand_T_target=initial_hand_T_target,
            initial_target_source="configured_nominal_palm_T_marker",
            optimize_hand_target=self.config.optimize_hand_target,
            free_joint_offsets=self.config.free_joint_offsets,
            joint_offset_prior_sigma_deg=self.config.joint_offset_prior_sigma_deg,
            robot_calibration_directory=self.robot_calibration_directory,
            runner_path=self.runner_path,
            timeout_s=self.config.native_timeout_s,
        )

    def _bootstrap(
        self,
        training_dataset: CalibrationDataset,
        solution: NativeCalibrationSolution,
        initial_hand_T_target: np.ndarray,
    ) -> BootstrapReport:
        names = solution.parameter_names
        if self.config.bootstrap_trials == 0:
            return BootstrapReport(0, 0, 0, names, (0.0,) * len(names))
        rng = np.random.default_rng(self.config.bootstrap_seed)
        by_pose: dict[str, list[CalibrationSample]] = {}
        for sample in training_dataset.samples:
            by_pose.setdefault(sample.pose_id, []).append(sample)
        pose_ids = tuple(sorted(by_pose))
        parameters: list[np.ndarray] = []
        failed = 0
        with tempfile.TemporaryDirectory(prefix="g1_native_bootstrap_") as temporary:
            root = Path(temporary)
            for trial_index in range(self.config.bootstrap_trials):
                indices = rng.integers(0, len(pose_ids), size=len(pose_ids))
                selected = tuple(
                    sample for index in indices for sample in by_pose[pose_ids[int(index)]]
                )
                unique_samples = tuple(
                    replace(
                        sample,
                        capture_id=f"bootstrap_{trial_index:04d}_{index:05d}",
                        frame_id=f"bootstrap_{trial_index:04d}_{index:05d}",
                    )
                    for index, sample in enumerate(selected)
                )
                trial_dataset = replace(training_dataset, samples=unique_samples)
                try:
                    trial = self._solve(
                        trial_dataset,
                        (
                            solution.hand_T_target
                            if self.config.optimize_hand_target
                            else initial_hand_T_target
                        ),
                        root / f"trial_{trial_index:04d}",
                    )
                except RuntimeError:
                    failed += 1
                    continue
                parameters.append(
                    np.asarray([trial.offsets[name] for name in names], dtype=np.float64)
                )
        standard_deviation = (
            np.std(np.vstack(parameters), axis=0, ddof=0)
            if parameters
            else np.full(len(names), np.nan)
        )
        return BootstrapReport(
            requested_trials=self.config.bootstrap_trials,
            successful_trials=len(parameters),
            failed_trials=failed,
            parameter_names=names,
            parameter_stddev=tuple(
                None if not np.isfinite(value) else float(value) for value in standard_deviation
            ),
        )
