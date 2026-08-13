"""Immutable contracts for the visually corrected tabletop cube task.

The ROS/control process owns observations and execution.  The CUDA process
receives one complete, hash-bound request and returns only frozen joint
trajectories plus the exact finger and attachment state transitions.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from g1_dex3_tabletop.planning.contracts import (
    PLANNER_SCHEMA_VERSION,
    PlannedTrajectory,
    RobotSnapshot,
    _finite_transform,
    _finite_vector,
    atomic_write_json,
)


def _hash(document: dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(
            document,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode()
    ).hexdigest()


@dataclass(frozen=True, slots=True)
class TabletopObservation:
    """One stationary AprilCube observation paired with the complete robot state."""

    snapshot: RobotSnapshot
    camera_T_object: tuple[tuple[float, ...], ...]
    camera_profile_sha256: str
    source_frame_sha256: tuple[str, ...]
    object_translation_spread_mm: float
    object_rotation_spread_deg: float

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "snapshot",
            self.snapshot
            if isinstance(self.snapshot, RobotSnapshot)
            else RobotSnapshot.from_dict(self.snapshot),
        )
        object.__setattr__(
            self,
            "camera_T_object",
            _finite_transform(self.camera_T_object, "camera_T_object"),
        )
        if len(self.camera_profile_sha256) != 64:
            raise ValueError("camera profile SHA-256 must contain 64 characters")
        object.__setattr__(
            self,
            "source_frame_sha256",
            tuple(str(value) for value in self.source_frame_sha256),
        )
        if len(self.source_frame_sha256) < 3 or any(
            len(value) != 64 for value in self.source_frame_sha256
        ):
            raise ValueError("tabletop observation requires at least three frame hashes")
        spreads = (
            self.object_translation_spread_mm,
            self.object_rotation_spread_deg,
        )
        if any(not np.isfinite(value) or value < 0 for value in spreads):
            raise ValueError("observation spreads must be finite and non-negative")

    def to_dict(self) -> dict[str, Any]:
        return {
            "snapshot": self.snapshot.to_dict(),
            "camera_T_object": [list(row) for row in self.camera_T_object],
            "camera_profile_sha256": self.camera_profile_sha256,
            "source_frame_sha256": list(self.source_frame_sha256),
            "object_translation_spread_mm": self.object_translation_spread_mm,
            "object_rotation_spread_deg": self.object_rotation_spread_deg,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TabletopObservation:
        return cls(**data)


@dataclass(frozen=True, slots=True)
class TabletopTaskRequest:
    """Complete scene, calibration, and robot state for one right-hand task."""

    observation: TabletopObservation
    torso_T_camera: tuple[tuple[float, ...], ...]
    joint_position_offsets_rad: dict[str, float]
    calibration_bundle_sha256: str
    grasp_shortlist_path: str
    grasp_shortlist_sha256: str
    object_dimensions_m: tuple[float, ...] = (0.045, 0.045, 0.045)
    supported_escape_m: float = 0.100
    lift_m: float = 0.100
    random_seed: int = 17
    schema_version: int = PLANNER_SCHEMA_VERSION
    operation: str = "plan_tabletop_pick_lift_replace"

    def __post_init__(self) -> None:
        if self.schema_version != PLANNER_SCHEMA_VERSION:
            raise ValueError("unsupported tabletop request schema version")
        if self.operation != "plan_tabletop_pick_lift_replace":
            raise ValueError("unsupported tabletop request operation")
        object.__setattr__(
            self,
            "observation",
            self.observation
            if isinstance(self.observation, TabletopObservation)
            else TabletopObservation.from_dict(self.observation),
        )
        object.__setattr__(
            self,
            "torso_T_camera",
            _finite_transform(self.torso_T_camera, "torso_T_camera"),
        )
        object.__setattr__(
            self,
            "object_dimensions_m",
            _finite_vector(self.object_dimensions_m, 3, "object_dimensions_m"),
        )
        if any(value <= 0 for value in self.object_dimensions_m):
            raise ValueError("object dimensions must be positive")
        for name in (
            "supported_escape_m",
            "lift_m",
        ):
            if not np.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive and finite")
        for name in ("calibration_bundle_sha256", "grasp_shortlist_sha256"):
            if len(getattr(self, name)) != 64:
                raise ValueError(f"{name} must contain 64 characters")
        offsets = {str(k): float(v) for k, v in self.joint_position_offsets_rad.items()}
        if any(not name or not np.isfinite(value) for name, value in offsets.items()):
            raise ValueError("joint offsets must have names and finite values")
        object.__setattr__(self, "joint_position_offsets_rad", offsets)

    @property
    def content_sha256(self) -> str:
        return _hash(self.to_dict(include_hash=False))

    def to_dict(self, *, include_hash: bool = True) -> dict[str, Any]:
        result = {
            "schema_version": self.schema_version,
            "operation": self.operation,
            "observation": self.observation.to_dict(),
            "torso_T_camera": [list(row) for row in self.torso_T_camera],
            "joint_position_offsets_rad": self.joint_position_offsets_rad,
            "calibration_bundle_sha256": self.calibration_bundle_sha256,
            "grasp_shortlist_path": self.grasp_shortlist_path,
            "grasp_shortlist_sha256": self.grasp_shortlist_sha256,
            "object_dimensions_m": list(self.object_dimensions_m),
            "supported_escape_m": self.supported_escape_m,
            "lift_m": self.lift_m,
            "random_seed": self.random_seed,
        }
        if include_hash:
            result["content_sha256"] = self.content_sha256
        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TabletopTaskRequest:
        values = dict(data)
        expected_hash = values.pop("content_sha256", None)
        request = cls(**values)
        if expected_hash is not None and expected_hash != request.content_sha256:
            raise ValueError("tabletop request SHA-256 mismatch")
        return request

    @classmethod
    def from_json(cls, path: str | Path) -> TabletopTaskRequest:
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))

    def write_json(self, path: str | Path) -> None:
        atomic_write_json(path, self.to_dict())


@dataclass(frozen=True, slots=True)
class TabletopTaskPlan:
    """Frozen task from an elevated clearance state and back to that state."""

    request_sha256: str
    selected_candidate_id: str
    object_T_grasp: tuple[tuple[float, ...], ...]
    open_right_dex3_q_rad: tuple[float, ...]
    closed_right_dex3_q_rad: tuple[float, ...]
    initial_right_dex3_q_rad: tuple[float, ...]
    trajectories: tuple[PlannedTrajectory, ...]
    phase_order: tuple[str, ...]
    planner_provenance: dict[str, Any]
    schema_version: int = PLANNER_SCHEMA_VERSION
    kind: str = "g1_tabletop_pick_lift_replace_plan"

    def __post_init__(self) -> None:
        if len(self.request_sha256) != 64:
            raise ValueError("request SHA-256 must contain 64 characters")
        object.__setattr__(
            self, "object_T_grasp", _finite_transform(self.object_T_grasp, "object_T_grasp")
        )
        for name in (
            "open_right_dex3_q_rad",
            "closed_right_dex3_q_rad",
            "initial_right_dex3_q_rad",
        ):
            object.__setattr__(self, name, _finite_vector(getattr(self, name), 7, name))
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
        expected = (
            "move_to_pregrasp",
            "grasp_approach",
            "payload_lift",
            "payload_replace",
            "grasp_retreat",
            "return_to_clearance",
        )
        object.__setattr__(self, "phase_order", tuple(self.phase_order))
        if self.phase_order != expected or len(self.trajectories) != len(expected):
            raise ValueError("tabletop plan must contain the complete six-motion lifecycle")
        if tuple(item.to_pose_id for item in self.trajectories) != expected:
            raise ValueError("trajectory endpoints differ from tabletop phase order")
        if self.trajectories[0].from_pose_id != "clearance":
            raise ValueError("tabletop plan must begin at the elevated clearance pose")
        if self.trajectories[-1].to_pose_id != "return_to_clearance":
            raise ValueError("tabletop plan must finish at elevated clearance")

    @property
    def content_sha256(self) -> str:
        return _hash(self.to_dict(include_hash=False))

    def to_dict(self, *, include_hash: bool = True) -> dict[str, Any]:
        result = {
            "schema_version": self.schema_version,
            "kind": self.kind,
            "request_sha256": self.request_sha256,
            "selected_candidate_id": self.selected_candidate_id,
            "object_T_grasp": [list(row) for row in self.object_T_grasp],
            "open_right_dex3_q_rad": list(self.open_right_dex3_q_rad),
            "closed_right_dex3_q_rad": list(self.closed_right_dex3_q_rad),
            "initial_right_dex3_q_rad": list(self.initial_right_dex3_q_rad),
            "trajectories": [item.to_dict() for item in self.trajectories],
            "phase_order": list(self.phase_order),
            "planner_provenance": self.planner_provenance,
        }
        if include_hash:
            result["content_sha256"] = self.content_sha256
        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TabletopTaskPlan:
        values = dict(data)
        expected_hash = values.pop("content_sha256", None)
        plan = cls(**values)
        if expected_hash is not None and expected_hash != plan.content_sha256:
            raise ValueError("tabletop plan SHA-256 mismatch")
        return plan

    @classmethod
    def from_json(cls, path: str | Path) -> TabletopTaskPlan:
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))

    def write_json(self, path: str | Path) -> None:
        atomic_write_json(path, self.to_dict())


@dataclass(frozen=True, slots=True)
class SupportedEscapePlan:
    """Reversible table-contact escape planned from the loaded supported state."""

    request_sha256: str
    outbound: PlannedTrajectory
    inbound: PlannedTrajectory
    minimum_terminal_plane_clearance_m: float
    planner_provenance: dict[str, Any]
    schema_version: int = PLANNER_SCHEMA_VERSION
    kind: str = "g1_supported_table_escape_plan"

    def __post_init__(self) -> None:
        if len(self.request_sha256) != 64:
            raise ValueError("request SHA-256 must contain 64 characters")
        if not isinstance(self.outbound, PlannedTrajectory):
            object.__setattr__(self, "outbound", PlannedTrajectory.from_dict(self.outbound))
        if not isinstance(self.inbound, PlannedTrajectory):
            object.__setattr__(self, "inbound", PlannedTrajectory.from_dict(self.inbound))
        if self.outbound.from_pose_id != "__handoff__" or self.outbound.to_pose_id != "clearance":
            raise ValueError("supported escape outbound endpoint IDs are invalid")
        if self.inbound.from_pose_id != "clearance" or self.inbound.to_pose_id != "__handoff__":
            raise ValueError("supported escape inbound endpoint IDs are invalid")
        if self.inbound.command_q_rad != tuple(reversed(self.outbound.command_q_rad)):
            raise ValueError("supported return must be the exact reverse command path")
        if self.minimum_terminal_plane_clearance_m <= 0:
            raise ValueError("supported escape must finish above the table plane")

    @property
    def content_sha256(self) -> str:
        return _hash(self.to_dict(include_hash=False))

    def to_dict(self, *, include_hash: bool = True) -> dict[str, Any]:
        result = {
            "schema_version": self.schema_version,
            "kind": self.kind,
            "request_sha256": self.request_sha256,
            "outbound": self.outbound.to_dict(),
            "inbound": self.inbound.to_dict(),
            "minimum_terminal_plane_clearance_m": self.minimum_terminal_plane_clearance_m,
            "planner_provenance": self.planner_provenance,
        }
        if include_hash:
            result["content_sha256"] = self.content_sha256
        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SupportedEscapePlan:
        values = dict(data)
        expected_hash = values.pop("content_sha256", None)
        plan = cls(**values)
        if expected_hash is not None and expected_hash != plan.content_sha256:
            raise ValueError("supported escape plan SHA-256 mismatch")
        return plan

    @classmethod
    def from_json(cls, path: str | Path) -> SupportedEscapePlan:
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))

    def write_json(self, path: str | Path) -> None:
        atomic_write_json(path, self.to_dict())


@dataclass(frozen=True, slots=True)
class TabletopExecutionPlan:
    """One controller-ready lifecycle from supported handoff and back."""

    loaded_request_sha256: str
    clearance_request_sha256: str
    supported_escape: SupportedEscapePlan
    task: TabletopTaskPlan
    trajectories: tuple[PlannedTrajectory, ...]
    schema_version: int = PLANNER_SCHEMA_VERSION
    kind: str = "g1_tabletop_complete_execution_plan"

    def __post_init__(self) -> None:
        for name in ("loaded_request_sha256", "clearance_request_sha256"):
            if len(getattr(self, name)) != 64:
                raise ValueError(f"{name} must contain 64 characters")
        if not isinstance(self.supported_escape, SupportedEscapePlan):
            object.__setattr__(
                self,
                "supported_escape",
                SupportedEscapePlan.from_dict(self.supported_escape),
            )
        if not isinstance(self.task, TabletopTaskPlan):
            object.__setattr__(self, "task", TabletopTaskPlan.from_dict(self.task))
        if self.supported_escape.request_sha256 != self.loaded_request_sha256:
            raise ValueError("supported escape belongs to a different loaded request")
        if self.task.request_sha256 != self.clearance_request_sha256:
            raise ValueError("tabletop task belongs to a different clearance request")
        object.__setattr__(
            self,
            "trajectories",
            tuple(
                item if isinstance(item, PlannedTrajectory) else PlannedTrajectory.from_dict(item)
                for item in self.trajectories
            ),
        )
        expected_edges = (
            ("__handoff__", "clearance"),
            ("clearance", "move_to_pregrasp"),
            ("move_to_pregrasp", "grasp_approach"),
            ("grasp_approach", "payload_lift"),
            ("payload_lift", "payload_replace"),
            ("payload_replace", "grasp_retreat"),
            ("grasp_retreat", "return_to_clearance"),
            ("return_to_clearance", "__handoff__"),
        )
        actual_edges = tuple((item.from_pose_id, item.to_pose_id) for item in self.trajectories)
        if actual_edges != expected_edges:
            raise ValueError("complete tabletop trajectory lifecycle is disconnected")
        if self.trajectories[0] != self.supported_escape.outbound:
            raise ValueError("complete lifecycle does not start with the supported escape")
        if self.trajectories[1:7] != self.task.trajectories:
            raise ValueError("complete lifecycle task motions differ from the task plan")
        inbound = self.trajectories[-1]
        if (
            inbound.sample_time_s != self.supported_escape.inbound.sample_time_s
            or inbound.command_q_rad != self.supported_escape.inbound.command_q_rad
            or inbound.model_q_rad != self.supported_escape.inbound.model_q_rad
        ):
            raise ValueError("complete lifecycle return is not the exact supported return")
        clearance_error = float(
            np.max(
                np.abs(
                    np.asarray(self.trajectories[0].command_q_rad[-1])
                    - np.asarray(self.trajectories[1].command_q_rad[0])
                )
            )
        )
        return_error = float(
            np.max(
                np.abs(
                    np.asarray(self.trajectories[6].command_q_rad[-1])
                    - np.asarray(self.trajectories[7].command_q_rad[0])
                )
            )
        )
        if max(clearance_error, return_error) > 1e-8:
            raise ValueError("tabletop plans disagree at the shared clearance state")

    @property
    def content_sha256(self) -> str:
        return _hash(self.to_dict(include_hash=False))

    def to_dict(self, *, include_hash: bool = True) -> dict[str, Any]:
        result = {
            "schema_version": self.schema_version,
            "kind": self.kind,
            "loaded_request_sha256": self.loaded_request_sha256,
            "clearance_request_sha256": self.clearance_request_sha256,
            "supported_escape": self.supported_escape.to_dict(),
            "task": self.task.to_dict(),
            "trajectories": [item.to_dict() for item in self.trajectories],
        }
        if include_hash:
            result["content_sha256"] = self.content_sha256
        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TabletopExecutionPlan:
        values = dict(data)
        expected_hash = values.pop("content_sha256", None)
        plan = cls(**values)
        if expected_hash is not None and expected_hash != plan.content_sha256:
            raise ValueError("tabletop execution plan SHA-256 mismatch")
        return plan

    @classmethod
    def from_json(cls, path: str | Path) -> TabletopExecutionPlan:
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))

    def write_json(self, path: str | Path) -> None:
        atomic_write_json(path, self.to_dict())


def combine_tabletop_plans(
    *,
    loaded_request: TabletopTaskRequest,
    clearance_request: TabletopTaskRequest,
    supported_escape: SupportedEscapePlan,
    task: TabletopTaskPlan,
) -> TabletopExecutionPlan:
    """Bind the two isolated CuRobo results into one connected execution."""

    inbound = PlannedTrajectory(
        from_pose_id="return_to_clearance",
        to_pose_id="__handoff__",
        sample_time_s=supported_escape.inbound.sample_time_s,
        command_q_rad=supported_escape.inbound.command_q_rad,
        model_q_rad=supported_escape.inbound.model_q_rad,
        planning_time_s=supported_escape.inbound.planning_time_s,
    )
    return TabletopExecutionPlan(
        loaded_request_sha256=loaded_request.content_sha256,
        clearance_request_sha256=clearance_request.content_sha256,
        supported_escape=supported_escape,
        task=task,
        trajectories=(supported_escape.outbound, *task.trajectories, inbound),
    )
