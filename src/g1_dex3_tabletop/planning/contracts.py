"""Strict JSON contracts across the ROS/control and CuRobo process boundary."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from g1_aprilcube_calibration.joint_map import validate_arm_side
from g1_aprilcube_calibration.pose_schema import HANDOFF_POSE_ID

PLANNER_SCHEMA_VERSION = 1
PLANNER_BACKEND = "NVlabs/curobo"


def _finite_vector(value: Any, length: int, name: str) -> tuple[float, ...]:
    array = np.asarray(value, dtype=np.float64).reshape(-1)
    if array.shape != (length,) or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain {length} finite values")
    return tuple(float(item) for item in array)


def _finite_transform(value: Any, name: str) -> tuple[tuple[float, ...], ...]:
    matrix = np.asarray(value, dtype=np.float64)
    if matrix.shape != (4, 4) or not np.all(np.isfinite(matrix)):
        raise ValueError(f"{name} must be a finite 4x4 transform")
    if not np.allclose(matrix[3], [0.0, 0.0, 0.0, 1.0], atol=1e-9):
        raise ValueError(f"{name} has an invalid homogeneous final row")
    rotation = matrix[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-6):
        raise ValueError(f"{name} rotation is not orthonormal")
    if not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-6):
        raise ValueError(f"{name} rotation determinant is not one")
    return tuple(tuple(float(item) for item in row) for row in matrix)


@dataclass(frozen=True, slots=True)
class RobotSnapshot:
    """Complete stationary state needed to construct the locked CuRobo model."""

    measured_q29_rad: tuple[float, ...]
    left_dex3_q_rad: tuple[float, ...]
    right_dex3_q_rad: tuple[float, ...]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "measured_q29_rad",
            _finite_vector(self.measured_q29_rad, 29, "measured_q29_rad"),
        )
        object.__setattr__(
            self,
            "left_dex3_q_rad",
            _finite_vector(self.left_dex3_q_rad, 7, "left_dex3_q_rad"),
        )
        object.__setattr__(
            self,
            "right_dex3_q_rad",
            _finite_vector(self.right_dex3_q_rad, 7, "right_dex3_q_rad"),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "measured_q29_rad": list(self.measured_q29_rad),
            "left_dex3_q_rad": list(self.left_dex3_q_rad),
            "right_dex3_q_rad": list(self.right_dex3_q_rad),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RobotSnapshot:
        if set(data) != {
            "measured_q29_rad",
            "left_dex3_q_rad",
            "right_dex3_q_rad",
        }:
            raise ValueError("robot snapshot fields do not match schema version 1")
        return cls(**data)


@dataclass(frozen=True, slots=True)
class Dex3PreparationRequest:
    """Live Ready snapshot for CuRobo shoulder clearance and finger sweep."""

    snapshot: RobotSnapshot
    joint_position_offsets_rad: dict[str, float]
    left_target_q_rad: tuple[float, ...]
    right_target_q_rad: tuple[float, ...]
    initial_outward_offset_rad: float = 0.08
    outward_search_step_rad: float = 0.02
    maximum_outward_offset_rad: float = 0.50
    random_seed: int = 17
    schema_version: int = PLANNER_SCHEMA_VERSION
    operation: str = "plan_dex3_preparation"

    def __post_init__(self) -> None:
        if self.schema_version != PLANNER_SCHEMA_VERSION:
            raise ValueError("unsupported planner request schema version")
        if self.operation != "plan_dex3_preparation":
            raise ValueError("unsupported planner request operation")
        object.__setattr__(
            self,
            "snapshot",
            self.snapshot
            if isinstance(self.snapshot, RobotSnapshot)
            else RobotSnapshot.from_dict(self.snapshot),
        )
        object.__setattr__(
            self,
            "left_target_q_rad",
            _finite_vector(self.left_target_q_rad, 7, "left_target_q_rad"),
        )
        object.__setattr__(
            self,
            "right_target_q_rad",
            _finite_vector(self.right_target_q_rad, 7, "right_target_q_rad"),
        )
        offsets: dict[str, float] = {}
        for name, value in self.joint_position_offsets_rad.items():
            numeric = float(value)
            if not name or not np.isfinite(numeric):
                raise ValueError("joint offsets must have names and finite values")
            offsets[str(name)] = numeric
        object.__setattr__(self, "joint_position_offsets_rad", offsets)
        if not (0 < self.initial_outward_offset_rad <= self.maximum_outward_offset_rad):
            raise ValueError("initial outward offset is outside the search range")
        if self.outward_search_step_rad <= 0:
            raise ValueError("outward search step must be positive")

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
        document = {
            "schema_version": self.schema_version,
            "operation": self.operation,
            "snapshot": self.snapshot.to_dict(),
            "joint_position_offsets_rad": self.joint_position_offsets_rad,
            "left_target_q_rad": list(self.left_target_q_rad),
            "right_target_q_rad": list(self.right_target_q_rad),
            "initial_outward_offset_rad": self.initial_outward_offset_rad,
            "outward_search_step_rad": self.outward_search_step_rad,
            "maximum_outward_offset_rad": self.maximum_outward_offset_rad,
            "random_seed": self.random_seed,
        }
        if include_hash:
            document["content_sha256"] = self.content_sha256
        return document

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Dex3PreparationRequest:
        expected = {
            "schema_version",
            "operation",
            "snapshot",
            "joint_position_offsets_rad",
            "left_target_q_rad",
            "right_target_q_rad",
            "initial_outward_offset_rad",
            "outward_search_step_rad",
            "maximum_outward_offset_rad",
            "random_seed",
            "content_sha256",
        }
        if set(data) != expected:
            raise ValueError("Dex3 preparation request fields do not match schema")
        expected_hash = data["content_sha256"]
        result = cls(**{key: value for key, value in data.items() if key != "content_sha256"})
        if expected_hash != result.content_sha256:
            raise ValueError("Dex3 preparation request content SHA-256 mismatch")
        return result

    @classmethod
    def from_json(cls, path: str | Path) -> Dex3PreparationRequest:
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))

    def write_json(self, path: str | Path) -> None:
        atomic_write_json(path, self.to_dict())


@dataclass(frozen=True, slots=True)
class CalibrationCandidate:
    """One calibration marker pose sampled by the experiment-specific selector."""

    candidate_id: str
    camera_T_marker: tuple[tuple[float, ...], ...]
    selection_metadata: dict[str, Any]

    def __post_init__(self) -> None:
        if not self.candidate_id.strip():
            raise ValueError("candidate_id must be non-empty")
        object.__setattr__(
            self,
            "camera_T_marker",
            _finite_transform(self.camera_T_marker, "camera_T_marker"),
        )
        canonical = json.loads(
            json.dumps(self.selection_metadata, sort_keys=True, allow_nan=False)
        )
        object.__setattr__(self, "selection_metadata", canonical)

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "camera_T_marker": [list(row) for row in self.camera_T_marker],
            "selection_metadata": self.selection_metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CalibrationCandidate:
        if set(data) != {"candidate_id", "camera_T_marker", "selection_metadata"}:
            raise ValueError("calibration candidate fields do not match schema version 1")
        return cls(**data)


@dataclass(frozen=True, slots=True)
class CalibrationPlanRequest:
    """Complete read-only request for batched IK and route synthesis."""

    arm: str
    snapshot: RobotSnapshot
    torso_T_camera: tuple[tuple[float, ...], ...]
    palm_T_marker: tuple[tuple[float, ...], ...]
    joint_position_offsets_rad: dict[str, float]
    candidates: tuple[CalibrationCandidate, ...]
    selection_config: dict[str, Any]
    target_count: int = 80
    ik_batch_size: int = 256
    random_seed: int = 17
    schema_version: int = PLANNER_SCHEMA_VERSION
    operation: str = "plan_calibration"

    def __post_init__(self) -> None:
        if self.schema_version != PLANNER_SCHEMA_VERSION:
            raise ValueError("unsupported planner request schema version")
        if self.operation != "plan_calibration":
            raise ValueError("unsupported planner request operation")
        object.__setattr__(self, "arm", validate_arm_side(self.arm))
        object.__setattr__(
            self,
            "snapshot",
            self.snapshot
            if isinstance(self.snapshot, RobotSnapshot)
            else RobotSnapshot.from_dict(self.snapshot),
        )
        object.__setattr__(
            self,
            "torso_T_camera",
            _finite_transform(self.torso_T_camera, "torso_T_camera"),
        )
        object.__setattr__(
            self,
            "palm_T_marker",
            _finite_transform(self.palm_T_marker, "palm_T_marker"),
        )
        offsets: dict[str, float] = {}
        for name, value in self.joint_position_offsets_rad.items():
            numeric = float(value)
            if not name or not np.isfinite(numeric):
                raise ValueError("joint offsets must have names and finite values")
            offsets[str(name)] = numeric
        object.__setattr__(self, "joint_position_offsets_rad", offsets)
        selection_config = json.loads(
            json.dumps(self.selection_config, sort_keys=True, allow_nan=False)
        )
        if not isinstance(selection_config, dict) or not selection_config:
            raise ValueError("selection_config must be a non-empty mapping")
        object.__setattr__(self, "selection_config", selection_config)
        object.__setattr__(
            self,
            "candidates",
            tuple(
                item
                if isinstance(item, CalibrationCandidate)
                else CalibrationCandidate.from_dict(item)
                for item in self.candidates
            ),
        )
        if self.target_count < 1 or self.target_count > len(self.candidates):
            raise ValueError("target_count must lie within the candidate count")
        if self.ik_batch_size < 1:
            raise ValueError("ik_batch_size must be positive")

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
            "operation": self.operation,
            "arm": self.arm,
            "snapshot": self.snapshot.to_dict(),
            "torso_T_camera": [list(row) for row in self.torso_T_camera],
            "palm_T_marker": [list(row) for row in self.palm_T_marker],
            "joint_position_offsets_rad": self.joint_position_offsets_rad,
            "candidates": [item.to_dict() for item in self.candidates],
            "selection_config": self.selection_config,
            "target_count": self.target_count,
            "ik_batch_size": self.ik_batch_size,
            "random_seed": self.random_seed,
        }
        if include_hash:
            result["content_sha256"] = self.content_sha256
        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CalibrationPlanRequest:
        expected = {
            "schema_version",
            "operation",
            "arm",
            "snapshot",
            "torso_T_camera",
            "palm_T_marker",
            "joint_position_offsets_rad",
            "candidates",
            "selection_config",
            "target_count",
            "ik_batch_size",
            "random_seed",
            "content_sha256",
        }
        if set(data) != expected:
            raise ValueError("planner request fields do not match schema version 1")
        content_hash = data["content_sha256"]
        result = cls(**{key: value for key, value in data.items() if key != "content_sha256"})
        if content_hash != result.content_sha256:
            raise ValueError("planner request content SHA-256 mismatch")
        return result

    @classmethod
    def from_json(cls, path: str | Path) -> CalibrationPlanRequest:
        with Path(path).open(encoding="utf-8") as stream:
            return cls.from_dict(json.load(stream))

    def write_json(self, path: str | Path) -> None:
        atomic_write_json(path, self.to_dict())


@dataclass(frozen=True, slots=True)
class PlannedCalibrationPose:
    candidate_id: str
    camera_T_marker: tuple[tuple[float, ...], ...]
    model_q_rad: tuple[float, ...]
    command_q_rad: tuple[float, ...]
    ik_position_error_m: float
    ik_rotation_error_rad: float
    selection_metadata: dict[str, Any]

    def __post_init__(self) -> None:
        if not self.candidate_id or self.candidate_id == HANDOFF_POSE_ID:
            raise ValueError("planned candidate ID is invalid")
        object.__setattr__(
            self,
            "camera_T_marker",
            _finite_transform(self.camera_T_marker, "camera_T_marker"),
        )
        object.__setattr__(self, "model_q_rad", _finite_vector(self.model_q_rad, 7, "model_q_rad"))
        object.__setattr__(
            self, "command_q_rad", _finite_vector(self.command_q_rad, 7, "command_q_rad")
        )
        for name in ("ik_position_error_m", "ik_rotation_error_rad"):
            value = float(getattr(self, name))
            if not np.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and non-negative")
            object.__setattr__(self, name, value)
        metadata = json.loads(json.dumps(self.selection_metadata, sort_keys=True, allow_nan=False))
        object.__setattr__(self, "selection_metadata", metadata)

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "camera_T_marker": [list(row) for row in self.camera_T_marker],
            "model_q_rad": list(self.model_q_rad),
            "command_q_rad": list(self.command_q_rad),
            "ik_position_error_m": self.ik_position_error_m,
            "ik_rotation_error_rad": self.ik_rotation_error_rad,
            "selection_metadata": self.selection_metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PlannedCalibrationPose:
        if set(data) != {
            "candidate_id",
            "camera_T_marker",
            "model_q_rad",
            "command_q_rad",
            "ik_position_error_m",
            "ik_rotation_error_rad",
            "selection_metadata",
        }:
            raise ValueError("planned calibration pose fields do not match schema version 1")
        return cls(**data)


@dataclass(frozen=True, slots=True)
class PlannedTrajectory:
    from_pose_id: str
    to_pose_id: str
    sample_time_s: tuple[float, ...]
    command_q_rad: tuple[tuple[float, ...], ...]
    model_q_rad: tuple[tuple[float, ...], ...]
    planning_time_s: float

    def __post_init__(self) -> None:
        if not self.from_pose_id or not self.to_pose_id or self.from_pose_id == self.to_pose_id:
            raise ValueError("trajectory endpoints must be different non-empty IDs")
        times = np.asarray(self.sample_time_s, dtype=np.float64).reshape(-1)
        command = np.asarray(self.command_q_rad, dtype=np.float64)
        model = np.asarray(self.model_q_rad, dtype=np.float64)
        if (
            len(times) < 2
            or times[0] != 0.0
            or not np.all(np.isfinite(times))
            or not np.all(np.diff(times) > 0.0)
        ):
            raise ValueError("trajectory times must start at zero and increase strictly")
        if command.shape != (len(times), 7) or model.shape != command.shape:
            raise ValueError("trajectory joint arrays must be N x 7 and match timestamps")
        if not np.all(np.isfinite(command)) or not np.all(np.isfinite(model)):
            raise ValueError("trajectory joint arrays contain NaN or infinity")
        planning_time = float(self.planning_time_s)
        if not np.isfinite(planning_time) or planning_time < 0:
            raise ValueError("planning_time_s must be finite and non-negative")
        object.__setattr__(self, "sample_time_s", tuple(float(value) for value in times))
        object.__setattr__(
            self, "command_q_rad", tuple(tuple(float(v) for v in row) for row in command)
        )
        object.__setattr__(
            self, "model_q_rad", tuple(tuple(float(v) for v in row) for row in model)
        )
        object.__setattr__(self, "planning_time_s", planning_time)

    def to_dict(self) -> dict[str, Any]:
        return {
            "from_pose_id": self.from_pose_id,
            "to_pose_id": self.to_pose_id,
            "sample_time_s": list(self.sample_time_s),
            "command_q_rad": [list(row) for row in self.command_q_rad],
            "model_q_rad": [list(row) for row in self.model_q_rad],
            "planning_time_s": self.planning_time_s,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PlannedTrajectory:
        if set(data) != {
            "from_pose_id",
            "to_pose_id",
            "sample_time_s",
            "command_q_rad",
            "model_q_rad",
            "planning_time_s",
        }:
            raise ValueError("planned trajectory fields do not match schema version 1")
        return cls(**data)


@dataclass(frozen=True, slots=True)
class Dex3PreparationPlan:
    """Frozen right-then-left clearance route and validated finger sweep."""

    request_sha256: str
    outward_offset_rad: float
    right_outbound: PlannedTrajectory
    left_outbound: PlannedTrajectory
    left_return: PlannedTrajectory
    right_return: PlannedTrajectory
    dual_clearance_q14_rad: tuple[float, ...]
    finger_sweep_sample_count: int
    planner_provenance: dict[str, Any]
    schema_version: int = PLANNER_SCHEMA_VERSION
    backend: str = PLANNER_BACKEND

    def __post_init__(self) -> None:
        if len(self.request_sha256) != 64:
            raise ValueError("request SHA-256 must contain 64 characters")
        if self.schema_version != PLANNER_SCHEMA_VERSION or self.backend != PLANNER_BACKEND:
            raise ValueError("unsupported Dex3 preparation backend or schema")
        if self.outward_offset_rad <= 0:
            raise ValueError("outward offset must be positive")
        for name in ("right_outbound", "left_outbound", "left_return", "right_return"):
            value = getattr(self, name)
            if not isinstance(value, PlannedTrajectory):
                object.__setattr__(self, name, PlannedTrajectory.from_dict(value))
        expected_edges = (
            ("__handoff__", "right_shoulder_clearance"),
            ("right_shoulder_clearance", "dual_shoulder_clearance"),
            ("dual_shoulder_clearance", "right_shoulder_clearance"),
            ("right_shoulder_clearance", "__handoff__"),
        )
        actual_edges = tuple(
            (item.from_pose_id, item.to_pose_id)
            for item in (
                self.right_outbound,
                self.left_outbound,
                self.left_return,
                self.right_return,
            )
        )
        if actual_edges != expected_edges:
            raise ValueError("Dex3 preparation trajectories have invalid endpoints")
        object.__setattr__(
            self,
            "dual_clearance_q14_rad",
            _finite_vector(self.dual_clearance_q14_rad, 14, "dual_clearance_q14_rad"),
        )
        if self.finger_sweep_sample_count < 2:
            raise ValueError("finger sweep requires at least two samples")
        provenance = json.loads(
            json.dumps(self.planner_provenance, sort_keys=True, allow_nan=False)
        )
        if not provenance:
            raise ValueError("planner provenance must be non-empty")
        object.__setattr__(self, "planner_provenance", provenance)

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
        document = {
            "schema_version": self.schema_version,
            "backend": self.backend,
            "request_sha256": self.request_sha256,
            "outward_offset_rad": self.outward_offset_rad,
            "right_outbound": self.right_outbound.to_dict(),
            "left_outbound": self.left_outbound.to_dict(),
            "left_return": self.left_return.to_dict(),
            "right_return": self.right_return.to_dict(),
            "dual_clearance_q14_rad": list(self.dual_clearance_q14_rad),
            "finger_sweep_sample_count": self.finger_sweep_sample_count,
            "planner_provenance": self.planner_provenance,
        }
        if include_hash:
            document["content_sha256"] = self.content_sha256
        return document

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Dex3PreparationPlan:
        expected_hash = data.get("content_sha256")
        expected = {
            "schema_version",
            "backend",
            "request_sha256",
            "outward_offset_rad",
            "right_outbound",
            "left_outbound",
            "left_return",
            "right_return",
            "dual_clearance_q14_rad",
            "finger_sweep_sample_count",
            "planner_provenance",
            "content_sha256",
        }
        if set(data) != expected:
            raise ValueError("Dex3 preparation plan fields do not match schema")
        result = cls(**{key: value for key, value in data.items() if key != "content_sha256"})
        if expected_hash != result.content_sha256:
            raise ValueError("Dex3 preparation plan content SHA-256 mismatch")
        return result

    @classmethod
    def from_json(cls, path: str | Path) -> Dex3PreparationPlan:
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))

    def write_json(self, path: str | Path) -> None:
        atomic_write_json(path, self.to_dict())


@dataclass(frozen=True, slots=True)
class CalibrationPlanResult:
    request_sha256: str
    arm: str
    active_joint_names: tuple[str, ...]
    handoff_model_q_rad: tuple[float, ...]
    handoff_command_q_rad: tuple[float, ...]
    poses: tuple[PlannedCalibrationPose, ...]
    route_pose_ids: tuple[str, ...]
    capture_pose_ids: tuple[str, ...]
    trajectories: tuple[PlannedTrajectory, ...]
    selection_steps: tuple[dict[str, Any], ...]
    planner_provenance: dict[str, Any]
    schema_version: int = PLANNER_SCHEMA_VERSION
    backend: str = PLANNER_BACKEND

    def __post_init__(self) -> None:
        if len(self.request_sha256) != 64:
            raise ValueError("request SHA-256 must contain 64 characters")
        if self.schema_version != PLANNER_SCHEMA_VERSION or self.backend != PLANNER_BACKEND:
            raise ValueError("unsupported calibration plan backend or schema")
        object.__setattr__(self, "arm", validate_arm_side(self.arm))
        if len(self.active_joint_names) != 7 or len(set(self.active_joint_names)) != 7:
            raise ValueError("calibration plan must name seven unique active joints")
        object.__setattr__(
            self,
            "handoff_model_q_rad",
            _finite_vector(self.handoff_model_q_rad, 7, "handoff_model_q_rad"),
        )
        object.__setattr__(
            self,
            "handoff_command_q_rad",
            _finite_vector(self.handoff_command_q_rad, 7, "handoff_command_q_rad"),
        )
        object.__setattr__(
            self,
            "poses",
            tuple(
                value
                if isinstance(value, PlannedCalibrationPose)
                else PlannedCalibrationPose.from_dict(value)
                for value in self.poses
            ),
        )
        object.__setattr__(
            self,
            "trajectories",
            tuple(
                value
                if isinstance(value, PlannedTrajectory)
                else PlannedTrajectory.from_dict(value)
                for value in self.trajectories
            ),
        )
        pose_ids = tuple(item.candidate_id for item in self.poses)
        if len(pose_ids) != len(set(pose_ids)):
            raise ValueError("planned pose IDs must be unique")
        if (
            len(self.route_pose_ids) < 3
            or self.route_pose_ids[0] != HANDOFF_POSE_ID
            or self.route_pose_ids[-1] != HANDOFF_POSE_ID
        ):
            raise ValueError("route must start and end at the measured handoff")
        if set(self.route_pose_ids[1:-1]) != set(pose_ids):
            raise ValueError("route and planned pose IDs differ")
        if tuple(self.capture_pose_ids) != tuple(self.route_pose_ids[1:-1]):
            raise ValueError("every route target must be captured exactly once")
        if len(self.trajectories) != len(self.route_pose_ids) - 1:
            raise ValueError("one planned trajectory is required for every route edge")
        for index, trajectory in enumerate(self.trajectories):
            if (trajectory.from_pose_id, trajectory.to_pose_id) != (
                self.route_pose_ids[index],
                self.route_pose_ids[index + 1],
            ):
                raise ValueError("trajectory endpoints do not follow the frozen route")
        object.__setattr__(
            self,
            "selection_steps",
            tuple(
                json.loads(json.dumps(value, sort_keys=True, allow_nan=False))
                for value in self.selection_steps
            ),
        )
        provenance = json.loads(
            json.dumps(self.planner_provenance, sort_keys=True, allow_nan=False)
        )
        if not provenance:
            raise ValueError("planner provenance must be non-empty")
        object.__setattr__(self, "planner_provenance", provenance)

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
        document = {
            "schema_version": self.schema_version,
            "backend": self.backend,
            "request_sha256": self.request_sha256,
            "arm": self.arm,
            "active_joint_names": list(self.active_joint_names),
            "handoff_model_q_rad": list(self.handoff_model_q_rad),
            "handoff_command_q_rad": list(self.handoff_command_q_rad),
            "poses": [item.to_dict() for item in self.poses],
            "route_pose_ids": list(self.route_pose_ids),
            "capture_pose_ids": list(self.capture_pose_ids),
            "trajectories": [item.to_dict() for item in self.trajectories],
            "selection_steps": list(self.selection_steps),
            "planner_provenance": self.planner_provenance,
        }
        if include_hash:
            document["content_sha256"] = self.content_sha256
        return document

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CalibrationPlanResult:
        content_hash = data.get("content_sha256")
        expected = {
            "schema_version",
            "backend",
            "request_sha256",
            "arm",
            "active_joint_names",
            "handoff_model_q_rad",
            "handoff_command_q_rad",
            "poses",
            "route_pose_ids",
            "capture_pose_ids",
            "trajectories",
            "selection_steps",
            "planner_provenance",
            "content_sha256",
        }
        if set(data) != expected:
            raise ValueError("calibration plan fields do not match schema version 1")
        result = cls(**{key: value for key, value in data.items() if key != "content_sha256"})
        if content_hash != result.content_sha256:
            raise ValueError("calibration plan content SHA-256 mismatch")
        return result

    @classmethod
    def from_json(cls, path: str | Path) -> CalibrationPlanResult:
        with Path(path).open(encoding="utf-8") as stream:
            return cls.from_dict(json.load(stream))

    def write_json(self, path: str | Path) -> None:
        atomic_write_json(path, self.to_dict())


def atomic_write_json(path: str | Path, document: dict[str, Any]) -> None:
    """Write one canonical JSON artifact atomically with an fsync boundary."""

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(document, indent=2, sort_keys=True, allow_nan=False).encode() + b"\n"
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output.name}.", suffix=".tmp", dir=output.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)
