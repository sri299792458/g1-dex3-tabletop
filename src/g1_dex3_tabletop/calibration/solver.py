"""Export and execute the pinned bilateral Ferguson/Ceres model."""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from g1_aprilcube_calibration.transforms import validate_transform
from g1_aprilcube_calibration.urdf_model import URDFModel
from g1_dex3_tabletop.calibration.ferguson import (
    BilateralFergusonRecord,
    add_measured_color_frames,
    build_bilateral_optimizer_config,
    sample_to_ferguson_record,
)
from g1_dex3_tabletop.calibration.models import (
    BilateralCalibrationDataset,
    BilateralModelSpec,
    CameraFrameArtifact,
)
from g1_dex3_tabletop.calibration.projection import (
    BilateralCalibrationProjection,
    BilateralObservabilityReport,
)

ROBOT_CALIBRATION_REVISION = "db991b040d1dc28af09d8865fc72f09720e12b73"
_CERES_REPORT = re.compile(
    r"Ceres Solver Report: Iterations:\s*(?P<iterations>\d+),\s*"
    r"Initial cost:\s*(?P<initial>[+\-0-9.eE]+),\s*"
    r"Final cost:\s*(?P<final>[+\-0-9.eE]+),\s*"
    r"Termination:\s*(?P<termination>[A-Z_]+)"
)


@dataclass(frozen=True, slots=True)
class BilateralSolverArtifacts:
    output_directory: Path
    bag_directory: Path
    robot_description_path: Path
    optimizer_config_path: Path
    provenance_path: Path
    sample_count: int


@dataclass(frozen=True, slots=True)
class BilateralSolverResult:
    model: BilateralModelSpec
    parameters: dict[str, float]
    torso_T_camera: np.ndarray
    hand_T_targets: dict[str, np.ndarray]
    combined_radial_rms_px: float
    arm_radial_rms_px: dict[str, float]
    observability: BilateralObservabilityReport
    iterations: int
    final_cost: float
    termination: str

    def __post_init__(self) -> None:
        parameters = {str(name): float(value) for name, value in self.parameters.items()}
        if not parameters or not np.all(np.isfinite(tuple(parameters.values()))):
            raise ValueError("bilateral solver parameters must be finite and non-empty")
        if set(self.hand_T_targets) != {"left", "right"}:
            raise ValueError("bilateral solution must contain left and right target transforms")
        targets = {
            side: validate_transform(transform) for side, transform in self.hand_T_targets.items()
        }
        arm_rms = {str(side): float(value) for side, value in self.arm_radial_rms_px.items()}
        if set(arm_rms) != {"left", "right"} or not all(
            np.isfinite(value) and value >= 0.0 for value in arm_rms.values()
        ):
            raise ValueError("bilateral solution requires finite non-negative arm RMS")
        if not np.isfinite(self.combined_radial_rms_px) or self.combined_radial_rms_px < 0.0:
            raise ValueError("combined radial RMS must be finite and non-negative")
        if not isinstance(self.iterations, int) or self.iterations < 1:
            raise ValueError("bilateral solver iteration count must be a positive integer")
        if not np.isfinite(self.final_cost) or self.final_cost < 0.0:
            raise ValueError("bilateral solver final cost must be finite and non-negative")
        if not self.termination.strip():
            raise ValueError("bilateral solver termination must be non-empty")
        object.__setattr__(self, "parameters", parameters)
        object.__setattr__(self, "torso_T_camera", validate_transform(self.torso_T_camera))
        object.__setattr__(self, "hand_T_targets", targets)
        object.__setattr__(self, "arm_radial_rms_px", arm_rms)

    def to_dict(self) -> dict[str, Any]:
        return {
            "model": self.model.to_dict(),
            "model_sha256": self.model.content_sha256,
            "parameters": self.parameters,
            "torso_T_camera": self.torso_T_camera.tolist(),
            "hand_T_targets": {
                side: transform.tolist() for side, transform in self.hand_T_targets.items()
            },
            "combined_radial_rms_px": self.combined_radial_rms_px,
            "arm_radial_rms_px": self.arm_radial_rms_px,
            "observability": self.observability.to_dict(),
            "iterations": self.iterations,
            "final_cost": self.final_cost,
            "termination": self.termination,
        }


def export_bilateral_solver_inputs(
    dataset: BilateralCalibrationDataset,
    urdf_model: URDFModel,
    output_directory: str | Path,
    *,
    camera_frames: CameraFrameArtifact,
    model: BilateralModelSpec,
    initial_hand_T_targets: dict[str, np.ndarray],
    robot_calibration_directory: str | Path | None = None,
) -> BilateralSolverArtifacts:
    """Write the augmented URDF, optimizer configuration, bag, and provenance."""

    projection = BilateralCalibrationProjection(
        urdf_model,
        camera_frames=camera_frames,
        model=model,
        initial_hand_T_targets=initial_hand_T_targets,
    )
    projection.validate_dataset_sources(dataset)
    if robot_calibration_directory is not None:
        verify_robot_calibration_revision(Path(robot_calibration_directory))
    output = Path(output_directory).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"bilateral solver output is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    robot_description = add_measured_color_frames(
        urdf_model.path.read_text(encoding="utf-8"),
        camera_frames=camera_frames,
    )
    robot_description_path = output / "robot_description.urdf"
    robot_description_path.write_text(robot_description, encoding="utf-8")
    optimizer_config = build_bilateral_optimizer_config(
        model=model,
        sample_count=len(dataset.samples),
        initial_hand_T_targets=initial_hand_T_targets,
    )
    optimizer_config_path = output / "calibrate.yaml"
    optimizer_config_path.write_text(
        yaml.safe_dump(optimizer_config, sort_keys=False),
        encoding="utf-8",
    )
    bag_directory = output / "calibration_data"
    records = tuple(
        sample_to_ferguson_record(
            sample,
            model=model,
            initial_hand_T_targets=initial_hand_T_targets,
        )
        for sample in dataset.samples
    )
    _write_bilateral_rosbag(
        records,
        robot_description=robot_description,
        bag_directory=bag_directory,
    )
    provenance = {
        "schema_version": 1,
        "dataset_sha256": dataset.content_sha256,
        "dataset_id": dataset.dataset_id,
        "session_manifest_sha256_by_id": dataset.session_manifest_sha256_by_id,
        "urdf_sha256": dataset.urdf_sha256,
        "augmented_urdf_sha256": hashlib.sha256(robot_description.encode()).hexdigest(),
        "camera_frame_artifact_sha256": camera_frames.content_sha256,
        "camera_serial": camera_frames.camera_serial,
        "model": model.to_dict(),
        "model_sha256": model.content_sha256,
        "sample_count": len(dataset.samples),
        "robot_calibration_revision": ROBOT_CALIBRATION_REVISION,
        "optimizer_backend": "mikeferguson/robot_calibration:Ceres",
        "observation_policy": (
            "one same-frame CalibrationData record with left_arm, left_camera, "
            "right_arm, right_camera"
        ),
    }
    provenance_path = output / "provenance.json"
    provenance_path.write_text(
        json.dumps(provenance, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return BilateralSolverArtifacts(
        output_directory=output,
        bag_directory=bag_directory,
        robot_description_path=robot_description_path,
        optimizer_config_path=optimizer_config_path,
        provenance_path=provenance_path,
        sample_count=len(dataset.samples),
    )


def solve_bilateral_dataset(
    dataset: BilateralCalibrationDataset,
    urdf_model: URDFModel,
    output_directory: str | Path,
    *,
    camera_frames: CameraFrameArtifact,
    model: BilateralModelSpec,
    initial_hand_T_targets: dict[str, np.ndarray],
    robot_calibration_directory: str | Path,
    runner_path: str | Path,
    timeout_s: float = 300.0,
) -> BilateralSolverResult:
    """Run one declared bilateral model and independently evaluate its result."""

    if not np.isfinite(timeout_s) or timeout_s <= 0.0:
        raise ValueError("bilateral solver timeout must be positive and finite")
    artifacts = export_bilateral_solver_inputs(
        dataset,
        urdf_model,
        output_directory,
        camera_frames=camera_frames,
        model=model,
        initial_hand_T_targets=initial_hand_T_targets,
        robot_calibration_directory=robot_calibration_directory,
    )
    runner = Path(runner_path).resolve()
    if not runner.is_file():
        raise FileNotFoundError(f"robot_calibration runner is missing: {runner}")
    command = [
        str(runner),
        "ros2",
        "run",
        "robot_calibration",
        "calibrate",
        "--from-bag",
        str(artifacts.bag_directory),
        "--ros-args",
        "--params-file",
        str(artifacts.optimizer_config_path),
    ]
    try:
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout_s,
        )
    except subprocess.TimeoutExpired as error:
        raise RuntimeError(
            f"bilateral Ferguson solver timed out after {timeout_s:.1f}s"
        ) from error
    output = completed.stdout
    if completed.stderr:
        output += "\n--- stderr ---\n" + completed.stderr
    (artifacts.output_directory / "solver.log").write_text(output, encoding="utf-8")
    if completed.returncode != 0:
        raise RuntimeError(
            "bilateral Ferguson solver failed with status "
            f"{completed.returncode}; see {artifacts.output_directory / 'solver.log'}"
        )
    parameters, iterations, final_cost, termination = parse_ferguson_output(completed.stdout)
    projection = BilateralCalibrationProjection(
        urdf_model,
        camera_frames=camera_frames,
        model=model,
        initial_hand_T_targets=initial_hand_T_targets,
    )
    expected = projection.parameter_names
    missing = sorted(set(expected) - set(parameters))
    unexpected = sorted(set(parameters) - set(expected))
    if missing or unexpected:
        raise RuntimeError(
            "Ferguson returned a parameter set that differs from the declared model: "
            f"missing={missing}, unexpected={unexpected}"
        )
    ordered = {name: parameters[name] for name in expected}
    residual_by_arm: dict[str, list[np.ndarray]] = {"left": [], "right": []}
    for sample in dataset.samples:
        for side in ("left", "right"):
            predicted, depths = projection.project_side(
                sample,
                side=side,
                parameters=ordered,
            )
            if np.any(depths <= 0.0):
                raise RuntimeError("bilateral solution projects target points behind camera")
            observation = sample.left if side == "left" else sample.right
            residual_by_arm[side].append(
                predicted - np.asarray(observation.image_points_px, dtype=np.float64)
            )
    arm_rms = {side: _radial_rms(np.vstack(residual_by_arm[side])) for side in ("left", "right")}
    combined = _radial_rms(
        np.vstack([item for side in ("left", "right") for item in residual_by_arm[side]])
    )
    observability = projection.observability(ordered, dataset.samples)
    if not observability.observable:
        raise RuntimeError(
            "bilateral solver converged to an unobservable model: "
            f"rank={observability.rank}/{observability.parameter_count}, "
            f"condition={observability.condition_number}"
        )
    torso_T_camera, left_target = projection.transforms(ordered, side="left")
    right_camera, right_target = projection.transforms(ordered, side="right")
    if not np.allclose(torso_T_camera, right_camera, atol=1e-12, rtol=0.0):
        raise RuntimeError("bilateral projection produced two different shared camera transforms")
    result = BilateralSolverResult(
        model=model,
        parameters=ordered,
        torso_T_camera=torso_T_camera,
        hand_T_targets={"left": left_target, "right": right_target},
        combined_radial_rms_px=combined,
        arm_radial_rms_px=arm_rms,
        observability=observability,
        iterations=iterations,
        final_cost=final_cost,
        termination=termination,
    )
    (artifacts.output_directory / "solution.json").write_text(
        json.dumps(result.to_dict(), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return result


def parse_ferguson_output(output: str) -> tuple[dict[str, float], int, float, str]:
    reports = list(_CERES_REPORT.finditer(output))
    if not reports:
        raise RuntimeError("Ferguson optimizer output contains no Ceres report")
    report = reports[-1]
    termination = report.group("termination")
    if termination != "CONVERGENCE":
        raise RuntimeError(f"Ferguson optimizer did not converge: {termination}")
    marker = "Parameter Offsets:"
    start = output.rfind(marker)
    if start < 0:
        raise RuntimeError("Ferguson optimizer did not print parameter offsets")
    offsets: dict[str, float] = {}
    for line in output[start + len(marker) :].splitlines():
        match = re.fullmatch(
            r"\s*([A-Za-z0-9_]+):\s*"
            r"([+\-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+\-]?\d+)?)\s*",
            line,
        )
        if match:
            offsets[match.group(1)] = float(match.group(2))
        elif offsets:
            break
    if not offsets:
        raise RuntimeError("Ferguson optimizer printed an empty offset set")
    return (
        offsets,
        int(report.group("iterations")),
        float(report.group("final")),
        termination,
    )


def verify_robot_calibration_revision(directory: Path) -> None:
    # Pinned submodules store .git as a file pointing into the parent repo.
    if not (directory / ".git").exists():
        raise FileNotFoundError(f"robot_calibration is not a Git checkout: {directory}")
    completed = subprocess.run(
        ["git", "-C", str(directory), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    actual = completed.stdout.strip()
    if actual != ROBOT_CALIBRATION_REVISION:
        raise ValueError(
            "robot_calibration revision mismatch: "
            f"expected {ROBOT_CALIBRATION_REVISION}, got {actual}"
        )


def _write_bilateral_rosbag(
    records: tuple[BilateralFergusonRecord, ...],
    *,
    robot_description: str,
    bag_directory: Path,
) -> None:
    if not records:
        raise ValueError("cannot write a bilateral solver bag without records")
    try:
        import rosbag2_py
        from geometry_msgs.msg import PointStamped
        from rclpy.serialization import serialize_message
        from robot_calibration_msgs.msg import CalibrationData, Observation
        from std_msgs.msg import String
    except ImportError as error:
        raise RuntimeError(
            "ROS 2 and robot_calibration_msgs must be sourced before bilateral export"
        ) from error

    writer = rosbag2_py.SequentialWriter()
    writer.open(
        rosbag2_py.StorageOptions(uri=str(bag_directory), storage_id="sqlite3"),
        rosbag2_py.ConverterOptions("", ""),
    )
    writer.create_topic(
        rosbag2_py.TopicMetadata(
            name="/robot_description",
            type="std_msgs/msg/String",
            serialization_format="cdr",
        )
    )
    writer.create_topic(
        rosbag2_py.TopicMetadata(
            name="/calibration_data",
            type="robot_calibration_msgs/msg/CalibrationData",
            serialization_format="cdr",
        )
    )
    writer.write(
        "/robot_description",
        serialize_message(String(data=robot_description)),
        1,
    )
    previous_stamp_ns = 1
    for index, record in enumerate(records, start=1):
        message = CalibrationData()
        message.joint_states.name = list(record.joint_names)
        message.joint_states.position = list(record.joint_positions)
        observations = []
        for item in record.observations:
            observation = Observation(sensor_name=item.sensor_name)
            for point in item.points:
                feature = PointStamped()
                feature.header.frame_id = item.feature_frame
                feature.point.x = float(point[0])
                feature.point.y = float(point[1])
                feature.point.z = float(point[2])
                observation.features.append(feature)
            if item.camera_info is not None:
                _fill_camera_info(observation.ext_camera_info.camera_info, item.camera_info)
            observations.append(observation)
        message.observations = observations
        stamp_ns = max(record.stamp_ns or index + 1, previous_stamp_ns + 1)
        writer.write("/calibration_data", serialize_message(message), stamp_ns)
        previous_stamp_ns = stamp_ns


def _fill_camera_info(message: Any, data: dict[str, Any]) -> None:
    message.header.frame_id = str(data["frame_id"])
    message.height = int(data["height"])
    message.width = int(data["width"])
    message.distortion_model = str(data["distortion_model"])
    message.d = [float(value) for value in data["d"]]
    message.k = [float(value) for value in data["k"]]
    message.r = [float(value) for value in data["r"]]
    message.p = [float(value) for value in data["p"]]


def _radial_rms(residual: np.ndarray) -> float:
    values = np.asarray(residual, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 2 or not len(values):
        raise ValueError("radial RMS requires a non-empty N x 2 residual")
    return float(np.sqrt(np.mean(np.sum(np.square(values), axis=1))))
