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

from g1_aprilcube_calibration.joint_map import validate_arm_side
from g1_dex3_tabletop.planning.contracts import (
    PLANNER_SCHEMA_VERSION,
    PlannedTrajectory,
    RobotSnapshot,
    _finite_transform,
    _finite_vector,
    atomic_write_json,
)
from g1_dex3_tabletop.planning.dex3_handedness import dex3_execution_profile


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
class CharucoBoardObservation:
    """One fixed-table ChArUco observation paired with the complete robot state."""

    snapshot: RobotSnapshot
    camera_T_board: tuple[tuple[float, ...], ...]
    camera_profile_sha256: str
    source_frame_sha256: tuple[str, ...]
    translation_spread_mm: float
    rotation_spread_deg: float
    board_spec: dict[str, Any]

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
            "camera_T_board",
            _finite_transform(self.camera_T_board, "camera_T_board"),
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
            raise ValueError("ChArUco observation requires at least three frame hashes")
        for name in ("translation_spread_mm", "rotation_spread_deg"):
            value = float(getattr(self, name))
            if not np.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative")
        expected_spec = {
            "squares_x": 6,
            "squares_y": 9,
            "square_length_mm": 30.0,
            "marker_length_mm": 22.0,
            "dictionary_name": "DICT_5X5_50",
            "legacy_pattern": False,
            "active_dimensions_mm": [180.0, 270.0],
            "marker_count": 27,
            "charuco_corner_count": 40,
        }
        if self.board_spec != expected_spec:
            raise ValueError("ChArUco observation uses a different frozen table board")
        object.__setattr__(self, "board_spec", dict(self.board_spec))

    def to_dict(self) -> dict[str, Any]:
        return {
            "snapshot": self.snapshot.to_dict(),
            "camera_T_board": [list(row) for row in self.camera_T_board],
            "camera_profile_sha256": self.camera_profile_sha256,
            "source_frame_sha256": list(self.source_frame_sha256),
            "translation_spread_mm": self.translation_spread_mm,
            "rotation_spread_deg": self.rotation_spread_deg,
            "board_spec": {
                key: list(value) if isinstance(value, list) else value
                for key, value in self.board_spec.items()
            },
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CharucoBoardObservation:
        return cls(**data)


@dataclass(frozen=True, slots=True)
class CharucoSupportedEscapeRequest:
    """Hash-bound 100 mm supported escape using a fixed ChArUco table plane."""

    observation: CharucoBoardObservation
    arm: str
    torso_T_camera: tuple[tuple[float, ...], ...]
    joint_position_offsets_rad: dict[str, float]
    calibration_bundle_sha256: str
    supported_escape_m: float = 0.100
    maximum_arm_velocity_rad_s: float = 0.100
    random_seed: int = 17
    schema_version: int = PLANNER_SCHEMA_VERSION
    operation: str = "plan_charuco_supported_escape"

    def __post_init__(self) -> None:
        if self.schema_version != PLANNER_SCHEMA_VERSION:
            raise ValueError("unsupported ChArUco escape request schema version")
        if self.operation != "plan_charuco_supported_escape":
            raise ValueError("unsupported ChArUco escape request operation")
        if not isinstance(self.observation, CharucoBoardObservation):
            object.__setattr__(
                self,
                "observation",
                CharucoBoardObservation.from_dict(self.observation),
            )
        object.__setattr__(self, "arm", validate_arm_side(self.arm))
        object.__setattr__(
            self,
            "torso_T_camera",
            _finite_transform(self.torso_T_camera, "torso_T_camera"),
        )
        offsets = {
            str(name): float(value) for name, value in self.joint_position_offsets_rad.items()
        }
        if any(not np.isfinite(value) for value in offsets.values()):
            raise ValueError("joint position offsets must be finite")
        object.__setattr__(self, "joint_position_offsets_rad", offsets)
        if len(self.calibration_bundle_sha256) != 64:
            raise ValueError("calibration bundle SHA-256 must contain 64 characters")
        for name in ("supported_escape_m", "maximum_arm_velocity_rad_s"):
            value = float(getattr(self, name))
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be positive and finite")
        if not isinstance(self.random_seed, int):
            raise TypeError("random_seed must be an integer")

    @property
    def content_sha256(self) -> str:
        return _hash(self.to_dict(include_hash=False))

    def to_dict(self, *, include_hash: bool = True) -> dict[str, Any]:
        result = {
            "schema_version": self.schema_version,
            "operation": self.operation,
            "observation": self.observation.to_dict(),
            "arm": self.arm,
            "torso_T_camera": [list(row) for row in self.torso_T_camera],
            "joint_position_offsets_rad": self.joint_position_offsets_rad,
            "calibration_bundle_sha256": self.calibration_bundle_sha256,
            "supported_escape_m": self.supported_escape_m,
            "maximum_arm_velocity_rad_s": self.maximum_arm_velocity_rad_s,
            "random_seed": self.random_seed,
        }
        if include_hash:
            result["content_sha256"] = self.content_sha256
        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CharucoSupportedEscapeRequest:
        values = dict(data)
        expected_hash = values.pop("content_sha256", None)
        request = cls(**values)
        if expected_hash is not None and expected_hash != request.content_sha256:
            raise ValueError("ChArUco escape request SHA-256 mismatch")
        return request

    @classmethod
    def from_json(cls, path: str | Path) -> CharucoSupportedEscapeRequest:
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))

    def write_json(self, path: str | Path) -> None:
        atomic_write_json(path, self.to_dict())


@dataclass(frozen=True, slots=True)
class TabletopFixture:
    """One immutable object presenter whose pose is derived from the cube."""

    fixture_id: str
    mesh_path: str
    mesh_sha256: str
    mesh_scale: tuple[float, ...]
    support_height_m: float
    cube_pose_contract: str = "centred_and_yaw_aligned"

    def __post_init__(self) -> None:
        if not self.fixture_id.strip():
            raise ValueError("fixture ID must be non-empty")
        path = Path(self.mesh_path)
        if path.is_absolute() or ".." in path.parts:
            raise ValueError("fixture mesh path must be repository-relative")
        if len(self.mesh_sha256) != 64:
            raise ValueError("fixture mesh SHA-256 must contain 64 characters")
        object.__setattr__(
            self,
            "mesh_scale",
            _finite_vector(self.mesh_scale, 3, "fixture mesh scale"),
        )
        if any(value <= 0.0 for value in self.mesh_scale):
            raise ValueError("fixture mesh scale must be positive")
        if not np.isfinite(self.support_height_m) or self.support_height_m <= 0.0:
            raise ValueError("fixture support height must be positive and finite")
        if self.cube_pose_contract != "centred_and_yaw_aligned":
            raise ValueError("unsupported fixture-to-cube pose contract")

    def to_dict(self) -> dict[str, Any]:
        return {
            "fixture_id": self.fixture_id,
            "mesh_path": self.mesh_path,
            "mesh_sha256": self.mesh_sha256,
            "mesh_scale": list(self.mesh_scale),
            "support_height_m": self.support_height_m,
            "cube_pose_contract": self.cube_pose_contract,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TabletopFixture:
        return cls(**data)


@dataclass(frozen=True, slots=True)
class TabletopTaskRequest:
    """Complete scene, calibration, and robot state for one selected-arm task."""

    observation: TabletopObservation
    arm: str
    torso_T_camera: tuple[tuple[float, ...], ...]
    joint_position_offsets_rad: dict[str, float]
    calibration_bundle_sha256: str
    grasp_shortlist_path: str
    grasp_shortlist_sha256: str
    presentation_id: str = "direct"
    fixture: TabletopFixture | None = None
    object_dimensions_m: tuple[float, ...] = (0.040, 0.040, 0.040)
    open_transit_table_patch_dimensions_m: tuple[float, ...] = (0.400, 0.400, 0.020)
    minimum_hand_plane_clearance_m: float = 0.005
    supported_escape_m: float = 0.100
    retention_test_lift_m: float = 0.010
    lift_m: float = 0.100
    maximum_arm_velocity_rad_s: float = 0.100
    random_seed: int = 17
    schema_version: int = PLANNER_SCHEMA_VERSION
    operation: str = "plan_tabletop_pick_lift_replace"

    def __post_init__(self) -> None:
        if self.schema_version != PLANNER_SCHEMA_VERSION:
            raise ValueError("unsupported tabletop request schema version")
        if self.operation != "plan_tabletop_pick_lift_replace":
            raise ValueError("unsupported tabletop request operation")
        if not self.presentation_id.strip():
            raise ValueError("tabletop presentation ID must be non-empty")
        if self.fixture is not None and not isinstance(self.fixture, TabletopFixture):
            object.__setattr__(self, "fixture", TabletopFixture.from_dict(self.fixture))
        if self.presentation_id == "direct" and self.fixture is not None:
            raise ValueError("direct tabletop presentation cannot contain a fixture")
        if self.presentation_id != "direct" and self.fixture is None:
            raise ValueError("non-direct tabletop presentation requires a fixture")
        object.__setattr__(self, "arm", validate_arm_side(self.arm))
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
        object.__setattr__(
            self,
            "open_transit_table_patch_dimensions_m",
            _finite_vector(
                self.open_transit_table_patch_dimensions_m,
                3,
                "open_transit_table_patch_dimensions_m",
            ),
        )
        if any(value <= 0 for value in self.open_transit_table_patch_dimensions_m):
            raise ValueError("open-transit table patch dimensions must be positive")
        if (
            not np.isfinite(self.minimum_hand_plane_clearance_m)
            or self.minimum_hand_plane_clearance_m <= 0.0
        ):
            raise ValueError("minimum hand-plane clearance must be positive and finite")
        for name in (
            "supported_escape_m",
            "retention_test_lift_m",
            "lift_m",
            "maximum_arm_velocity_rad_s",
        ):
            if not np.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive and finite")
        if self.retention_test_lift_m >= self.lift_m:
            raise ValueError("retention test lift must be smaller than the complete payload lift")
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
            "arm": self.arm,
            "torso_T_camera": [list(row) for row in self.torso_T_camera],
            "joint_position_offsets_rad": self.joint_position_offsets_rad,
            "calibration_bundle_sha256": self.calibration_bundle_sha256,
            "grasp_shortlist_path": self.grasp_shortlist_path,
            "grasp_shortlist_sha256": self.grasp_shortlist_sha256,
            "presentation_id": self.presentation_id,
            "fixture": None if self.fixture is None else self.fixture.to_dict(),
            "object_dimensions_m": list(self.object_dimensions_m),
            "open_transit_table_patch_dimensions_m": list(
                self.open_transit_table_patch_dimensions_m
            ),
            "minimum_hand_plane_clearance_m": self.minimum_hand_plane_clearance_m,
            "supported_escape_m": self.supported_escape_m,
            "retention_test_lift_m": self.retention_test_lift_m,
            "lift_m": self.lift_m,
            "maximum_arm_velocity_rad_s": self.maximum_arm_velocity_rad_s,
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
    arm: str
    selected_candidate_id: str
    object_T_grasp: tuple[tuple[float, ...], ...]
    open_active_dex3_q_rad: tuple[float, ...]
    close_target_active_dex3_q_rad: tuple[float, ...]
    initial_active_dex3_q_rad: tuple[float, ...]
    trajectories: tuple[PlannedTrajectory, ...]
    phase_order: tuple[str, ...]
    planner_provenance: dict[str, Any]
    schema_version: int = PLANNER_SCHEMA_VERSION
    kind: str = "g1_tabletop_pick_lift_replace_plan"

    def __post_init__(self) -> None:
        if len(self.request_sha256) != 64:
            raise ValueError("request SHA-256 must contain 64 characters")
        object.__setattr__(self, "arm", validate_arm_side(self.arm))
        object.__setattr__(
            self, "object_T_grasp", _finite_transform(self.object_T_grasp, "object_T_grasp")
        )
        for name in (
            "open_active_dex3_q_rad",
            "close_target_active_dex3_q_rad",
            "initial_active_dex3_q_rad",
        ):
            object.__setattr__(self, name, _finite_vector(getattr(self, name), 7, name))
        expected_open, expected_close = dex3_execution_profile(self.arm)
        if self.open_active_dex3_q_rad != expected_open:
            raise ValueError("tabletop plan open target differs from the Dex3 descriptor")
        if self.close_target_active_dex3_q_rad != expected_close:
            raise ValueError("tabletop plan close target differs from the Dex3 descriptor")
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
            "retention_test_lift",
            "payload_lift",
            "payload_lower",
            "payload_replace",
            "grasp_retreat",
            "return_to_clearance",
        )
        object.__setattr__(self, "phase_order", tuple(self.phase_order))
        if self.phase_order != expected or len(self.trajectories) != len(expected):
            raise ValueError("tabletop plan must contain the complete eight-motion lifecycle")
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
            "arm": self.arm,
            "selected_candidate_id": self.selected_candidate_id,
            "object_T_grasp": [list(row) for row in self.object_T_grasp],
            "open_active_dex3_q_rad": list(self.open_active_dex3_q_rad),
            "close_target_active_dex3_q_rad": list(self.close_target_active_dex3_q_rad),
            "initial_active_dex3_q_rad": list(self.initial_active_dex3_q_rad),
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
class RetentionRouteValidationRequest:
    """Collision-check the frozen payload route at the measured stalled hand posture."""

    tabletop_request: TabletopTaskRequest
    task_plan: TabletopTaskPlan
    measured_active_dex3_q_rad: tuple[float, ...]
    blocked_motor_ids: tuple[int, ...]
    schema_version: int = PLANNER_SCHEMA_VERSION
    operation: str = "validate_retention_route"

    def __post_init__(self) -> None:
        if self.schema_version != PLANNER_SCHEMA_VERSION:
            raise ValueError("unsupported retention-route request schema version")
        if self.operation != "validate_retention_route":
            raise ValueError("unsupported retention-route request operation")
        if not isinstance(self.tabletop_request, TabletopTaskRequest):
            object.__setattr__(
                self,
                "tabletop_request",
                TabletopTaskRequest.from_dict(self.tabletop_request),
            )
        if not isinstance(self.task_plan, TabletopTaskPlan):
            object.__setattr__(self, "task_plan", TabletopTaskPlan.from_dict(self.task_plan))
        if self.task_plan.request_sha256 != self.tabletop_request.content_sha256:
            raise ValueError("retention-route task belongs to a different tabletop request")
        if self.task_plan.arm != self.tabletop_request.arm:
            raise ValueError("retention-route task and request select different arms")
        object.__setattr__(
            self,
            "measured_active_dex3_q_rad",
            _finite_vector(
                self.measured_active_dex3_q_rad,
                7,
                "measured_active_dex3_q_rad",
            ),
        )
        blocked = tuple(int(value) for value in self.blocked_motor_ids)
        if (
            not blocked
            or len(set(blocked)) != len(blocked)
            or any(value < 0 or value >= 7 for value in blocked)
        ):
            raise ValueError("retention-route blocked motor IDs are invalid")
        object.__setattr__(self, "blocked_motor_ids", blocked)

    @property
    def content_sha256(self) -> str:
        return _hash(self.to_dict(include_hash=False))

    def to_dict(self, *, include_hash: bool = True) -> dict[str, Any]:
        result = {
            "schema_version": self.schema_version,
            "operation": self.operation,
            "tabletop_request": self.tabletop_request.to_dict(),
            "task_plan": self.task_plan.to_dict(),
            "measured_active_dex3_q_rad": list(self.measured_active_dex3_q_rad),
            "blocked_motor_ids": list(self.blocked_motor_ids),
        }
        if include_hash:
            result["content_sha256"] = self.content_sha256
        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RetentionRouteValidationRequest:
        values = dict(data)
        expected_hash = values.pop("content_sha256", None)
        request = cls(**values)
        if expected_hash is not None and expected_hash != request.content_sha256:
            raise ValueError("retention-route request SHA-256 mismatch")
        return request

    @classmethod
    def from_json(cls, path: str | Path) -> RetentionRouteValidationRequest:
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))

    def write_json(self, path: str | Path) -> None:
        atomic_write_json(path, self.to_dict())


@dataclass(frozen=True, slots=True)
class RetentionRouteValidationResult:
    """Proof that measured stalled fingers do not invalidate the payload arm route."""

    request_sha256: str
    arm: str
    selected_candidate_id: str
    route_sample_count: int
    minimum_hand_plane_clearance_m: float
    minimum_hand_plane_link: str
    minimum_hand_plane_sample: int
    planner_provenance: dict[str, Any]
    minimum_fixture_clearance_m: float | None = None
    minimum_fixture_clearance_link: str | None = None
    minimum_fixture_clearance_sample: int | None = None
    schema_version: int = PLANNER_SCHEMA_VERSION
    kind: str = "g1_retention_route_validation"

    def __post_init__(self) -> None:
        if len(self.request_sha256) != 64:
            raise ValueError("retention-route request SHA-256 must contain 64 characters")
        object.__setattr__(self, "arm", validate_arm_side(self.arm))
        if not self.selected_candidate_id:
            raise ValueError("retention-route candidate ID must be non-empty")
        if self.route_sample_count < 2:
            raise ValueError("retention route must contain at least two samples")
        if not np.isfinite(self.minimum_hand_plane_clearance_m):
            raise ValueError("retention-route hand clearance must be finite")
        if not self.minimum_hand_plane_link or self.minimum_hand_plane_sample < 0:
            raise ValueError("retention-route minimum hand location is invalid")
        fixture_values = (
            self.minimum_fixture_clearance_m,
            self.minimum_fixture_clearance_link,
            self.minimum_fixture_clearance_sample,
        )
        if any(value is not None for value in fixture_values):
            if not all(value is not None for value in fixture_values):
                raise ValueError("retention-route fixture clearance fields must be complete")
            assert self.minimum_fixture_clearance_m is not None
            assert self.minimum_fixture_clearance_sample is not None
            if not np.isfinite(self.minimum_fixture_clearance_m):
                raise ValueError("retention-route fixture clearance must be finite")
            if (
                not self.minimum_fixture_clearance_link
                or self.minimum_fixture_clearance_sample < 0
            ):
                raise ValueError("retention-route minimum fixture location is invalid")
        provenance = json.loads(
            json.dumps(self.planner_provenance, sort_keys=True, allow_nan=False)
        )
        object.__setattr__(self, "planner_provenance", provenance)

    @property
    def content_sha256(self) -> str:
        return _hash(self.to_dict(include_hash=False))

    def to_dict(self, *, include_hash: bool = True) -> dict[str, Any]:
        result = {
            "schema_version": self.schema_version,
            "kind": self.kind,
            "request_sha256": self.request_sha256,
            "arm": self.arm,
            "selected_candidate_id": self.selected_candidate_id,
            "route_sample_count": self.route_sample_count,
            "minimum_hand_plane_clearance_m": self.minimum_hand_plane_clearance_m,
            "minimum_hand_plane_link": self.minimum_hand_plane_link,
            "minimum_hand_plane_sample": self.minimum_hand_plane_sample,
            "minimum_fixture_clearance_m": self.minimum_fixture_clearance_m,
            "minimum_fixture_clearance_link": self.minimum_fixture_clearance_link,
            "minimum_fixture_clearance_sample": self.minimum_fixture_clearance_sample,
            "planner_provenance": self.planner_provenance,
        }
        if include_hash:
            result["content_sha256"] = self.content_sha256
        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RetentionRouteValidationResult:
        values = dict(data)
        expected_hash = values.pop("content_sha256", None)
        result = cls(**values)
        if expected_hash is not None and expected_hash != result.content_sha256:
            raise ValueError("retention-route result SHA-256 mismatch")
        return result

    @classmethod
    def from_json(cls, path: str | Path) -> RetentionRouteValidationResult:
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
    recovery_trajectories: tuple[PlannedTrajectory, ...]
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
            ("grasp_approach", "retention_test_lift"),
            ("retention_test_lift", "payload_lift"),
            ("payload_lift", "payload_lower"),
            ("payload_lower", "payload_replace"),
            ("payload_replace", "grasp_retreat"),
            ("grasp_retreat", "return_to_clearance"),
            ("return_to_clearance", "__handoff__"),
        )
        actual_edges = tuple((item.from_pose_id, item.to_pose_id) for item in self.trajectories)
        if actual_edges != expected_edges:
            raise ValueError("complete tabletop trajectory lifecycle is disconnected")
        if self.trajectories[0] != self.supported_escape.outbound:
            raise ValueError("complete lifecycle does not start with the supported escape")
        if self.trajectories[1:9] != self.task.trajectories:
            raise ValueError("complete lifecycle task motions differ from the task plan")
        object.__setattr__(
            self,
            "recovery_trajectories",
            tuple(
                item if isinstance(item, PlannedTrajectory) else PlannedTrajectory.from_dict(item)
                for item in self.recovery_trajectories
            ),
        )
        recovery_edges = tuple(
            (item.from_pose_id, item.to_pose_id) for item in self.recovery_trajectories
        )
        if recovery_edges != (
            ("grasp_approach", "grasp_retreat"),
            ("retention_test_lift", "payload_replace"),
        ):
            raise ValueError("tabletop recovery trajectory endpoints are invalid")
        for recovery, normal in zip(
            self.recovery_trajectories,
            (self.trajectories[7], self.trajectories[6]),
            strict=True,
        ):
            if (
                recovery.sample_time_s != normal.sample_time_s
                or recovery.command_q_rad != normal.command_q_rad
                or recovery.model_q_rad != normal.model_q_rad
            ):
                raise ValueError("tabletop recovery motion differs from its frozen reverse path")
        recovery_start_errors = (
            float(
                np.max(
                    np.abs(
                        np.asarray(self.recovery_trajectories[0].command_q_rad[0])
                        - np.asarray(self.trajectories[2].command_q_rad[-1])
                    )
                )
            ),
            float(
                np.max(
                    np.abs(
                        np.asarray(self.recovery_trajectories[1].command_q_rad[0])
                        - np.asarray(self.trajectories[3].command_q_rad[-1])
                    )
                )
            ),
        )
        if max(recovery_start_errors) > 1e-8:
            raise ValueError("tabletop recovery motion does not start at its rejection state")
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
                    np.asarray(self.trajectories[8].command_q_rad[-1])
                    - np.asarray(self.trajectories[9].command_q_rad[0])
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
            "recovery_trajectories": [item.to_dict() for item in self.recovery_trajectories],
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

    if not (loaded_request.arm == clearance_request.arm == task.arm):
        raise ValueError("loaded request, clearance request, and task select different arms")

    inbound = PlannedTrajectory(
        from_pose_id="return_to_clearance",
        to_pose_id="__handoff__",
        sample_time_s=supported_escape.inbound.sample_time_s,
        command_q_rad=supported_escape.inbound.command_q_rad,
        model_q_rad=supported_escape.inbound.model_q_rad,
        planning_time_s=supported_escape.inbound.planning_time_s,
    )
    contact_retreat = PlannedTrajectory(
        from_pose_id="grasp_approach",
        to_pose_id="grasp_retreat",
        sample_time_s=task.trajectories[6].sample_time_s,
        command_q_rad=task.trajectories[6].command_q_rad,
        model_q_rad=task.trajectories[6].model_q_rad,
        planning_time_s=0.0,
    )
    test_lift_replace = PlannedTrajectory(
        from_pose_id="retention_test_lift",
        to_pose_id="payload_replace",
        sample_time_s=task.trajectories[5].sample_time_s,
        command_q_rad=task.trajectories[5].command_q_rad,
        model_q_rad=task.trajectories[5].model_q_rad,
        planning_time_s=0.0,
    )
    return TabletopExecutionPlan(
        loaded_request_sha256=loaded_request.content_sha256,
        clearance_request_sha256=clearance_request.content_sha256,
        supported_escape=supported_escape,
        task=task,
        trajectories=(supported_escape.outbound, *task.trajectories, inbound),
        recovery_trajectories=(contact_retreat, test_lift_replace),
    )
