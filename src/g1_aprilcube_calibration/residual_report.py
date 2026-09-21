"""Held-out, per-corner, per-tag, and joint-correlated residual diagnostics."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import tempfile
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from importlib.resources import files
from pathlib import Path

import numpy as np
import yaml
from jsonschema import Draft202012Validator

from g1_aprilcube_calibration.calibration_evaluation import (
    NativeCalibrationProjection,
    NativeCalibrationSolution,
)
from g1_aprilcube_calibration.dataset_builder import (
    CalibrationDataset,
    CalibrationSample,
)
from g1_aprilcube_calibration.joint_map import (
    arm_hand_link,
    arm_indices,
    arm_joint_names,
)


@dataclass(frozen=True, slots=True)
class CornerResidual:
    split: str
    capture_id: str
    pose_id: str
    frame_id: str
    tag_id: int
    corner_index: int
    observed_u_px: float
    observed_v_px: float
    predicted_u_px: float
    predicted_v_px: float
    residual_u_px: float
    residual_v_px: float
    radial_error_px: float
    depth_m: float

    def to_dict(self) -> dict:
        return {name: getattr(self, name) for name in self.__dataclass_fields__}


@dataclass(frozen=True, slots=True)
class ResidualMetrics:
    corner_count: int
    rms_px: float
    median_px: float
    p95_px: float
    maximum_px: float

    def to_dict(self) -> dict:
        return {name: getattr(self, name) for name in self.__dataclass_fields__}


@dataclass(frozen=True, slots=True)
class ResidualReport:
    training: ResidualMetrics
    holdout: ResidualMetrics
    per_capture_rms_px: dict[str, float]
    per_tag_rms_px: dict[str, float]
    calibration_joint_correlations: dict[str, dict[str, float | None]]
    corner_residuals: tuple[CornerResidual, ...]

    def to_dict(self) -> dict:
        return {
            "training": self.training.to_dict(),
            "holdout": self.holdout.to_dict(),
            "per_capture_rms_px": self.per_capture_rms_px,
            "per_tag_rms_px": self.per_tag_rms_px,
            "calibration_joint_correlations": self.calibration_joint_correlations,
            "corner_residuals": [item.to_dict() for item in self.corner_residuals],
        }


def build_residual_report(
    solver: NativeCalibrationProjection,
    result: NativeCalibrationSolution,
    *,
    training_samples: Sequence[CalibrationSample],
    holdout_samples: Sequence[CalibrationSample],
) -> ResidualReport:
    records: list[CornerResidual] = []
    for split, samples in (
        ("training", training_samples),
        ("holdout", holdout_samples),
    ):
        for sample in samples:
            predicted, depths = solver.project_sample(sample, result)
            observed = np.asarray(sample.image_points_px)
            residual = predicted - observed
            for index, (prediction, observation, delta, depth, tag_id) in enumerate(
                zip(
                    predicted,
                    observed,
                    residual,
                    depths,
                    sample.corner_tag_ids,
                    strict=True,
                )
            ):
                records.append(
                    CornerResidual(
                        split=split,
                        capture_id=sample.capture_id,
                        pose_id=sample.pose_id,
                        frame_id=sample.frame_id,
                        tag_id=tag_id,
                        corner_index=index % 4,
                        observed_u_px=float(observation[0]),
                        observed_v_px=float(observation[1]),
                        predicted_u_px=float(prediction[0]),
                        predicted_v_px=float(prediction[1]),
                        residual_u_px=float(delta[0]),
                        residual_v_px=float(delta[1]),
                        radial_error_px=float(np.linalg.norm(delta)),
                        depth_m=float(depth),
                    )
                )
    training = tuple(item for item in records if item.split == "training")
    holdout = tuple(item for item in records if item.split == "holdout")
    per_capture = _group_rms(records, lambda item: item.capture_id)
    per_tag = _group_rms(records, lambda item: str(item.tag_id))
    correlations = _joint_correlations(
        records,
        training_samples,
        holdout_samples,
        calibration_arm=solver.calibration_arm,
    )
    return ResidualReport(
        training=_metrics(training),
        holdout=_metrics(holdout),
        per_capture_rms_px=per_capture,
        per_tag_rms_px=per_tag,
        calibration_joint_correlations=correlations,
        corner_residuals=tuple(records),
    )


def _metrics(records: Sequence[CornerResidual]) -> ResidualMetrics:
    errors = np.asarray([item.radial_error_px for item in records], dtype=np.float64)
    if not errors.size:
        return ResidualMetrics(0, float("nan"), float("nan"), float("nan"), float("nan"))
    return ResidualMetrics(
        corner_count=len(errors),
        rms_px=float(np.sqrt(np.mean(np.square(errors)))),
        median_px=float(np.median(errors)),
        p95_px=float(np.percentile(errors, 95)),
        maximum_px=float(np.max(errors)),
    )


def _group_rms(records, key_function) -> dict[str, float]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for item in records:
        grouped[str(key_function(item))].append(item.radial_error_px)
    return {
        key: float(np.sqrt(np.mean(np.square(values)))) for key, values in sorted(grouped.items())
    }


def _joint_correlations(
    records: Sequence[CornerResidual],
    training_samples: Sequence[CalibrationSample],
    holdout_samples: Sequence[CalibrationSample],
    *,
    calibration_arm: str,
) -> dict[str, dict[str, float | None]]:
    by_capture: dict[str, list[CornerResidual]] = defaultdict(list)
    for item in records:
        by_capture[item.capture_id].append(item)
    samples = {item.capture_id: item for item in (*training_samples, *holdout_samples)}
    capture_ids = sorted(by_capture)
    mean_u = np.asarray(
        [np.mean([item.residual_u_px for item in by_capture[key]]) for key in capture_ids]
    )
    mean_v = np.asarray(
        [np.mean([item.residual_v_px for item in by_capture[key]]) for key in capture_ids]
    )
    radial = np.asarray(
        [
            np.sqrt(np.mean([item.radial_error_px**2 for item in by_capture[key]]))
            for key in capture_ids
        ]
    )
    result: dict[str, dict[str, float | None]] = {}
    for offset, name in zip(
        arm_indices(calibration_arm),
        arm_joint_names(calibration_arm),
        strict=True,
    ):
        joint = np.asarray(
            [samples[key].measured_state["position"][offset] for key in capture_ids]
        )
        result[name] = {
            "mean_u": _correlation(joint, mean_u),
            "mean_v": _correlation(joint, mean_v),
            "rms_radial": _correlation(joint, radial),
        }
    return result


def _correlation(first: np.ndarray, second: np.ndarray) -> float | None:
    if len(first) < 3 or np.std(first) < 1e-12 or np.std(second) < 1e-12:
        return None
    return float(np.corrcoef(first, second)[0, 1])


class CalibrationRunExporter:
    def export(
        self,
        directory: str | Path,
        *,
        dataset: CalibrationDataset,
        result: NativeCalibrationSolution,
        residuals: ResidualReport,
        training_capture_ids: Sequence[str],
        holdout_capture_ids: Sequence[str],
        provenance: dict,
        bootstrap: dict | None = None,
    ) -> Path:
        output = Path(directory)
        output.mkdir(parents=True, exist_ok=True)
        managed_names = {
            "result.json",
            "provenance.json",
            "calibrated_extrinsics.yaml",
            "corner_residuals.csv",
            "report.md",
        }
        conflicts = sorted(name for name in managed_names if (output / name).exists())
        if conflicts:
            raise FileExistsError(
                f"calibration run already contains result artifacts: {conflicts}"
            )
        if not residuals.corner_residuals:
            raise ValueError("cannot export a calibration run without corner residuals")
        payload = {
            "schema_version": 1,
            "dataset_sha256": dataset.content_sha256,
            "training_capture_ids": list(training_capture_ids),
            "holdout_capture_ids": list(holdout_capture_ids),
            "solution": result.to_dict(),
            "residual_summary": {
                key: value
                for key, value in residuals.to_dict().items()
                if key != "corner_residuals"
            },
            "provenance": json.loads(json.dumps(provenance, allow_nan=False)),
            "bootstrap": {} if bootstrap is None else bootstrap,
        }
        payload["content_sha256"] = hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
        ).hexdigest()
        validate_exported_result(payload)
        self._write_json(output / "result.json", payload)
        self._write_json(output / "provenance.json", payload["provenance"])
        self._write_yaml(
            output / "calibrated_extrinsics.yaml",
            {
                "torso_T_color_camera": result.torso_T_camera.tolist(),
                f"{arm_hand_link(dataset.calibration_arm)}_T_aprilcube": (
                    result.hand_T_target.tolist()
                ),
                "calibration_arm": dataset.calibration_arm,
                "optimizer_backend": result.backend,
                "native_offsets": dict(result.offsets),
                "dataset_sha256": dataset.content_sha256,
                "result_sha256": payload["content_sha256"],
            },
        )
        self._write_residual_csv(output / "corner_residuals.csv", residuals)
        self._write_text(output / "report.md", self._markdown(payload, residuals))
        return output

    @staticmethod
    def _markdown(payload: dict, report: ResidualReport) -> str:
        observable = payload["solution"]["observability"]
        condition = observable["condition_number"]
        condition_text = "unbounded" if condition is None else f"{condition:.6g}"
        lines = [
            "# G1 camera calibration report",
            "",
            f"- Dataset SHA-256: `{payload['dataset_sha256']}`",
            f"- Result SHA-256: `{payload['content_sha256']}`",
            f"- Training radial RMS: {report.training.rms_px:.4f} px",
            f"- Holdout radial RMS: {report.holdout.rms_px:.4f} px",
            f"- Optimizer backend: `{payload['solution']['backend']}`",
            f"- Jacobian rank: {observable['rank']}/{observable['parameter_count']}",
            f"- Jacobian condition number: {condition_text}",
            "",
            "## Residual-first decision",
            "",
            (
                "Inspect `corner_residuals.csv`, per-pose RMS, tag grouping, holdout "
                "error, rank, conditioning, bootstrap spread, and calibration-joint "
                "correlations in `result.json`. Compare staged native models on "
                "held-out poses; lower training RMS alone does not justify retaining "
                "additional joint offsets."
            ),
            "",
        ]
        return "\n".join(lines)

    @staticmethod
    def _write_json(path: Path, data: dict) -> None:
        CalibrationRunExporter._write_bytes(
            path, json.dumps(data, indent=2, sort_keys=True).encode() + b"\n"
        )

    @staticmethod
    def _write_yaml(path: Path, data: dict) -> None:
        CalibrationRunExporter._write_bytes(path, yaml.safe_dump(data, sort_keys=False).encode())

    @staticmethod
    def _write_text(path: Path, data: str) -> None:
        CalibrationRunExporter._write_bytes(path, data.encode())

    @staticmethod
    def _write_residual_csv(path: Path, report: ResidualReport) -> None:
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{path.name}.", suffix=".tmp", dir=path.parent, text=True
        )
        temporary = Path(temporary_name)
        try:
            with os.fdopen(descriptor, "w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(
                    stream, fieldnames=list(report.corner_residuals[0].to_dict())
                )
                writer.writeheader()
                writer.writerows(item.to_dict() for item in report.corner_residuals)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)

    @staticmethod
    def _write_bytes(path: Path, data: bytes) -> None:
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
        )
        temporary = Path(temporary_name)
        try:
            with os.fdopen(descriptor, "wb") as stream:
                stream.write(data)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)


def validate_exported_result(data: dict) -> None:
    schema_path = files("g1_aprilcube_calibration.schemas").joinpath("result.schema.json")
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    Draft202012Validator(schema).validate(data)
    canonical = dict(data)
    expected = canonical.pop("content_sha256")
    actual = hashlib.sha256(
        json.dumps(canonical, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()
    if expected != actual:
        raise ValueError("calibration result content SHA-256 does not match")


def load_exported_result(path: str | Path) -> dict:
    with Path(path).open(encoding="utf-8") as stream:
        data = json.load(stream)
    validate_exported_result(data)
    return data
