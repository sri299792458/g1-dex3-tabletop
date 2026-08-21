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

    @property
    def content_sha256(self) -> str:
        return _hash(self.to_dict())

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TabletopObservation:
        return cls(**data)


@dataclass(frozen=True, slots=True)
class EstimatedCameraPlanningState:
    """Proprioceptively propagated camera pose at a stationary plan boundary.

    The original visual observation remains unchanged.  This separate record
    makes it impossible to mistake a propagated pose for a second camera
    detection, and binds the estimate to the exact visual anchor and measured
    robot state used to construct it.
    """

    snapshot: RobotSnapshot
    object_T_camera: tuple[tuple[float, ...], ...]
    anchor_observation_sha256: str
    anchor_timestamp_ns: int
    timestamp_ns: int
    anchor_input_timing: dict[str, Any]
    current_input_timing: dict[str, Any]
    estimator: str = "hybrid_pelvis_position_torso_orientation"
    assumption: str = "fixed_pelvis_imu_origin_between_visual_anchor_and_boundary"

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
            "object_T_camera",
            _finite_transform(self.object_T_camera, "object_T_camera"),
        )
        if len(self.anchor_observation_sha256) != 64:
            raise ValueError("anchor observation SHA-256 must contain 64 characters")
        if self.anchor_timestamp_ns < 0 or self.timestamp_ns < self.anchor_timestamp_ns:
            raise ValueError("estimated planning-state timestamps are invalid")
        if self.estimator != "hybrid_pelvis_position_torso_orientation":
            raise ValueError("unsupported camera-state estimator")
        if self.assumption != ("fixed_pelvis_imu_origin_between_visual_anchor_and_boundary"):
            raise ValueError("unsupported camera-state translation assumption")
        for name in ("anchor_input_timing", "current_input_timing"):
            normalized = json.loads(
                json.dumps(getattr(self, name), sort_keys=True, allow_nan=False)
            )
            if not isinstance(normalized, dict):
                raise TypeError(f"{name} must be a JSON object")
            object.__setattr__(self, name, normalized)

    @property
    def camera_T_object(self) -> tuple[tuple[float, ...], ...]:
        return _finite_transform(
            np.linalg.inv(np.asarray(self.object_T_camera, dtype=np.float64)),
            "camera_T_object",
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "snapshot": self.snapshot.to_dict(),
            "object_T_camera": [list(row) for row in self.object_T_camera],
            "anchor_observation_sha256": self.anchor_observation_sha256,
            "anchor_timestamp_ns": self.anchor_timestamp_ns,
            "timestamp_ns": self.timestamp_ns,
            "anchor_input_timing": self.anchor_input_timing,
            "current_input_timing": self.current_input_timing,
            "estimator": self.estimator,
            "assumption": self.assumption,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> EstimatedCameraPlanningState:
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
class TabletopCuboid:
    """One fixed cuboid expressed in the manipulated detected-object frame."""

    object_id: str
    object_T_cuboid: tuple[tuple[float, ...], ...]
    dimensions_m: tuple[float, ...]
    role: str = "obstacle"

    def __post_init__(self) -> None:
        if not self.object_id.strip():
            raise ValueError("tabletop cuboid ID must be non-empty")
        if self.object_id in {"manipulated_object", "open_transit_table_patch"}:
            raise ValueError("tabletop cuboid ID is reserved")
        object.__setattr__(
            self,
            "object_T_cuboid",
            _finite_transform(self.object_T_cuboid, "object_T_cuboid"),
        )
        object.__setattr__(
            self,
            "dimensions_m",
            _finite_vector(self.dimensions_m, 3, "tabletop cuboid dimensions"),
        )
        if any(value <= 0.0 for value in self.dimensions_m):
            raise ValueError("tabletop cuboid dimensions must be positive")
        if self.role not in {"obstacle", "placement_support"}:
            raise ValueError(f"unsupported tabletop cuboid role: {self.role}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "object_id": self.object_id,
            "object_T_cuboid": [list(row) for row in self.object_T_cuboid],
            "dimensions_m": list(self.dimensions_m),
            "role": self.role,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TabletopCuboid:
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
    estimated_planning_state: EstimatedCameraPlanningState | None = None
    presentation_id: str = "direct"
    fixture: TabletopFixture | None = None
    environment_cuboids: tuple[TabletopCuboid, ...] = ()
    table_reference_camera_T_object: tuple[tuple[float, ...], ...] | None = None
    table_reference_object_dimensions_m: tuple[float, ...] | None = None
    object_dimensions_m: tuple[float, ...] = (0.040, 0.040, 0.040)
    open_transit_table_patch_dimensions_m: tuple[float, ...] = (0.400, 0.400, 0.020)
    minimum_hand_plane_clearance_m: float = 0.005
    supported_escape_m: float = 0.100
    retention_test_lift_m: float = 0.030
    lift_m: float = 0.100
    maximum_arm_velocity_rad_s: float = 0.200
    pregrasp_distance_m: float | None = None
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
        cuboids = tuple(
            value if isinstance(value, TabletopCuboid) else TabletopCuboid.from_dict(value)
            for value in self.environment_cuboids
        )
        ids = tuple(value.object_id for value in cuboids)
        if len(set(ids)) != len(ids):
            raise ValueError("tabletop environment cuboid IDs must be unique")
        if sum(value.role == "placement_support" for value in cuboids) > 1:
            raise ValueError("a tabletop request can have at most one placement support")
        object.__setattr__(self, "environment_cuboids", cuboids)
        table_reference = self.table_reference_camera_T_object
        table_dimensions = self.table_reference_object_dimensions_m
        if (table_reference is None) != (table_dimensions is None):
            raise ValueError("table reference pose and dimensions must be supplied together")
        if table_reference is not None:
            if self.fixture is not None:
                raise ValueError("fixture requests cannot override their table reference")
            object.__setattr__(
                self,
                "table_reference_camera_T_object",
                _finite_transform(table_reference, "table_reference_camera_T_object"),
            )
            dimensions = _finite_vector(
                table_dimensions,
                3,
                "table_reference_object_dimensions_m",
            )
            if any(value <= 0.0 for value in dimensions):
                raise ValueError("table reference object dimensions must be positive")
            object.__setattr__(self, "table_reference_object_dimensions_m", dimensions)
        object.__setattr__(self, "arm", validate_arm_side(self.arm))
        object.__setattr__(
            self,
            "observation",
            self.observation
            if isinstance(self.observation, TabletopObservation)
            else TabletopObservation.from_dict(self.observation),
        )
        if self.estimated_planning_state is not None:
            state = (
                self.estimated_planning_state
                if isinstance(self.estimated_planning_state, EstimatedCameraPlanningState)
                else EstimatedCameraPlanningState.from_dict(self.estimated_planning_state)
            )
            if state.anchor_observation_sha256 != self.observation.content_sha256:
                raise ValueError("estimated planning state belongs to a different visual anchor")
            object.__setattr__(self, "estimated_planning_state", state)
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
        if self.pregrasp_distance_m is not None:
            value = float(self.pregrasp_distance_m)
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError("pregrasp_distance_m must be positive and finite")
            object.__setattr__(self, "pregrasp_distance_m", value)
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

    @property
    def planning_snapshot(self) -> RobotSnapshot:
        if self.estimated_planning_state is None:
            return self.observation.snapshot
        return self.estimated_planning_state.snapshot

    @property
    def planning_camera_T_object(self) -> tuple[tuple[float, ...], ...]:
        if self.estimated_planning_state is None:
            return self.observation.camera_T_object
        return self.estimated_planning_state.camera_T_object

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
            "environment_cuboids": [value.to_dict() for value in self.environment_cuboids],
            "table_reference_camera_T_object": (
                None
                if self.table_reference_camera_T_object is None
                else [list(row) for row in self.table_reference_camera_T_object]
            ),
            "table_reference_object_dimensions_m": (
                None
                if self.table_reference_object_dimensions_m is None
                else list(self.table_reference_object_dimensions_m)
            ),
            "object_dimensions_m": list(self.object_dimensions_m),
            "open_transit_table_patch_dimensions_m": list(
                self.open_transit_table_patch_dimensions_m
            ),
            "minimum_hand_plane_clearance_m": self.minimum_hand_plane_clearance_m,
            "supported_escape_m": self.supported_escape_m,
            "retention_test_lift_m": self.retention_test_lift_m,
            "lift_m": self.lift_m,
            "maximum_arm_velocity_rad_s": self.maximum_arm_velocity_rad_s,
            "pregrasp_distance_m": self.pregrasp_distance_m,
            "random_seed": self.random_seed,
        }
        if self.estimated_planning_state is not None:
            result["estimated_planning_state"] = self.estimated_planning_state.to_dict()
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
class MovingGraspContinuationRequest:
    """Rebuild the payload lifecycle at the grasp actually reached by MPC.

    The frozen clearance reference and final fresh cube detection are kept
    separate.  Proprioception propagates camera motion from that reference;
    the independent cube observation supplies object motion and binds the
    continuation to the exact MPC approach that the controller executed.
    """

    tabletop_request: TabletopTaskRequest
    prior_task_plan: TabletopTaskPlan
    terminal_command_q_rad: tuple[float, ...]
    terminal_active_dex3_q_rad: tuple[float, ...]
    reference_T_camera: tuple[tuple[float, ...], ...]
    camera_T_object: tuple[tuple[float, ...], ...]
    executed_grasp_approach: PlannedTrajectory
    terminal_mpc_window_sha256: str
    target_provenance: dict[str, Any]
    schema_version: int = PLANNER_SCHEMA_VERSION
    operation: str = "plan_moving_grasp_continuation"

    def __post_init__(self) -> None:
        if self.schema_version != PLANNER_SCHEMA_VERSION:
            raise ValueError("unsupported moving-grasp continuation schema version")
        if self.operation != "plan_moving_grasp_continuation":
            raise ValueError("unsupported moving-grasp continuation operation")
        if not isinstance(self.tabletop_request, TabletopTaskRequest):
            object.__setattr__(
                self,
                "tabletop_request",
                TabletopTaskRequest.from_dict(self.tabletop_request),
            )
        if not isinstance(self.prior_task_plan, TabletopTaskPlan):
            object.__setattr__(
                self,
                "prior_task_plan",
                TabletopTaskPlan.from_dict(self.prior_task_plan),
            )
        if self.prior_task_plan.request_sha256 != self.tabletop_request.content_sha256:
            raise ValueError("moving-grasp prior task belongs to another tabletop request")
        if self.prior_task_plan.arm != self.tabletop_request.arm:
            raise ValueError("moving-grasp request and prior task select different arms")
        object.__setattr__(
            self,
            "terminal_command_q_rad",
            _finite_vector(self.terminal_command_q_rad, 7, "terminal_command_q_rad"),
        )
        object.__setattr__(
            self,
            "terminal_active_dex3_q_rad",
            _finite_vector(
                self.terminal_active_dex3_q_rad,
                7,
                "terminal_active_dex3_q_rad",
            ),
        )
        object.__setattr__(
            self,
            "reference_T_camera",
            _finite_transform(self.reference_T_camera, "reference_T_camera"),
        )
        object.__setattr__(
            self,
            "camera_T_object",
            _finite_transform(self.camera_T_object, "camera_T_object"),
        )
        if not isinstance(self.executed_grasp_approach, PlannedTrajectory):
            object.__setattr__(
                self,
                "executed_grasp_approach",
                PlannedTrajectory.from_dict(self.executed_grasp_approach),
            )
        approach = self.executed_grasp_approach
        if (approach.from_pose_id, approach.to_pose_id) != (
            "move_to_pregrasp",
            "grasp_approach",
        ):
            raise ValueError("executed MPC approach has invalid endpoints")
        expected_start = np.asarray(self.prior_task_plan.trajectories[1].command_q_rad[0])
        if float(np.max(np.abs(np.asarray(approach.command_q_rad[0]) - expected_start))) > 1.0e-8:
            raise ValueError("executed MPC approach does not start at the frozen pregrasp")
        if (
            float(
                np.max(
                    np.abs(
                        np.asarray(approach.command_q_rad[-1])
                        - np.asarray(self.terminal_command_q_rad)
                    )
                )
            )
            > 1.0e-8
        ):
            raise ValueError("executed MPC approach does not end at its terminal command")
        if len(self.terminal_mpc_window_sha256) != 64:
            raise ValueError("terminal MPC window SHA-256 must contain 64 characters")
        provenance = json.loads(
            json.dumps(self.target_provenance, sort_keys=True, allow_nan=False)
        )
        if not isinstance(provenance, dict):
            raise TypeError("moving-grasp target provenance must be a JSON object")
        object.__setattr__(self, "target_provenance", provenance)

    @property
    def content_sha256(self) -> str:
        return _hash(self.to_dict(include_hash=False))

    def to_dict(self, *, include_hash: bool = True) -> dict[str, Any]:
        result = {
            "schema_version": self.schema_version,
            "operation": self.operation,
            "tabletop_request": self.tabletop_request.to_dict(),
            "prior_task_plan": self.prior_task_plan.to_dict(),
            "terminal_command_q_rad": list(self.terminal_command_q_rad),
            "terminal_active_dex3_q_rad": list(self.terminal_active_dex3_q_rad),
            "reference_T_camera": [list(row) for row in self.reference_T_camera],
            "camera_T_object": [list(row) for row in self.camera_T_object],
            "executed_grasp_approach": self.executed_grasp_approach.to_dict(),
            "terminal_mpc_window_sha256": self.terminal_mpc_window_sha256,
            "target_provenance": self.target_provenance,
        }
        if include_hash:
            result["content_sha256"] = self.content_sha256
        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> MovingGraspContinuationRequest:
        values = dict(data)
        expected_hash = values.pop("content_sha256", None)
        request = cls(**values)
        if expected_hash is not None and expected_hash != request.content_sha256:
            raise ValueError("moving-grasp continuation request SHA-256 mismatch")
        return request

    @classmethod
    def from_json(cls, path: str | Path) -> MovingGraspContinuationRequest:
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))

    def write_json(self, path: str | Path) -> None:
        atomic_write_json(path, self.to_dict())


PICK_PLACE_PHASE_ORDER = (
    "move_to_pregrasp",
    "grasp_approach",
    "retention_test_lift",
    "payload_lift",
    "payload_transfer",
    "placement_lower",
    "placement_contact",
    "placement_retreat",
    "return_to_clearance",
)


@dataclass(frozen=True, slots=True)
class TabletopPickPlaceRequest:
    """One fixed-source cube transfer to one of several equivalent destinations.

    ``source_T_destination_objects`` uses detector-defined physical object
    frames. Each entry is a physically acceptable task goal; CuRobo chooses
    among them. They deliberately exclude the planner's private face-up
    cube-axis permutation.
    """

    source_request: TabletopTaskRequest
    source_T_destination_objects: tuple[tuple[tuple[float, ...], ...], ...]
    destination_support_object_id: str | None = None
    excluded_candidate_ids: tuple[str, ...] = ()
    schema_version: int = PLANNER_SCHEMA_VERSION
    operation: str = "plan_tabletop_pick_place"

    def __post_init__(self) -> None:
        if self.schema_version != PLANNER_SCHEMA_VERSION:
            raise ValueError("unsupported tabletop pick-place schema version")
        if self.operation != "plan_tabletop_pick_place":
            raise ValueError("unsupported tabletop pick-place operation")
        if not isinstance(self.source_request, TabletopTaskRequest):
            object.__setattr__(
                self,
                "source_request",
                TabletopTaskRequest.from_dict(self.source_request),
            )
        destinations = tuple(
            _finite_transform(value, f"source_T_destination_objects[{index}]")
            for index, value in enumerate(self.source_T_destination_objects)
        )
        if not destinations:
            raise ValueError("pick-place request requires at least one destination")
        object.__setattr__(self, "source_T_destination_objects", destinations)
        support = self.destination_support_object_id
        if support is not None:
            support = str(support).strip()
            if not support:
                raise ValueError("destination support object ID must be non-empty")
            environment_ids = {
                value.object_id for value in self.source_request.environment_cuboids
            }
            if support not in environment_ids:
                raise ValueError("destination support is absent from the source world")
            object.__setattr__(self, "destination_support_object_id", support)
        excluded = tuple(str(value).strip() for value in self.excluded_candidate_ids)
        if any(not value for value in excluded):
            raise ValueError("excluded grasp candidate IDs must be non-empty")
        if len(excluded) != len(set(excluded)):
            raise ValueError("excluded grasp candidate IDs must be unique")
        object.__setattr__(self, "excluded_candidate_ids", excluded)

    @property
    def content_sha256(self) -> str:
        return _hash(self.to_dict(include_hash=False))

    def to_dict(self, *, include_hash: bool = True) -> dict[str, Any]:
        result = {
            "schema_version": self.schema_version,
            "operation": self.operation,
            "source_request": self.source_request.to_dict(),
            "source_T_destination_objects": [
                [list(row) for row in transform]
                for transform in self.source_T_destination_objects
            ],
            "destination_support_object_id": self.destination_support_object_id,
            "excluded_candidate_ids": list(self.excluded_candidate_ids),
        }
        if include_hash:
            result["content_sha256"] = self.content_sha256
        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TabletopPickPlaceRequest:
        values = dict(data)
        expected_hash = values.pop("content_sha256", None)
        request = cls(**values)
        if expected_hash is not None and expected_hash != request.content_sha256:
            raise ValueError("tabletop pick-place request SHA-256 mismatch")
        return request

    @classmethod
    def from_json(cls, path: str | Path) -> TabletopPickPlaceRequest:
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))

    def write_json(self, path: str | Path) -> None:
        atomic_write_json(path, self.to_dict())


@dataclass(frozen=True, slots=True)
class TabletopPickPlacePlan:
    """The fixed nine-motion source-to-destination lifecycle."""

    request_sha256: str
    arm: str
    selected_candidate_id: str
    selected_destination_index: int
    source_task: TabletopTaskPlan
    destination_task: TabletopTaskPlan
    trajectories: tuple[PlannedTrajectory, ...]
    phase_order: tuple[str, ...]
    planner_provenance: dict[str, Any]
    schema_version: int = PLANNER_SCHEMA_VERSION
    kind: str = "g1_tabletop_pick_place_plan"

    def __post_init__(self) -> None:
        if len(self.request_sha256) != 64:
            raise ValueError("request SHA-256 must contain 64 characters")
        object.__setattr__(self, "arm", validate_arm_side(self.arm))
        if not isinstance(self.selected_destination_index, int):
            raise TypeError("selected destination index must be an integer")
        if self.selected_destination_index < 0:
            raise ValueError("selected destination index must be non-negative")
        for name in ("source_task", "destination_task"):
            value = getattr(self, name)
            if not isinstance(value, TabletopTaskPlan):
                value = TabletopTaskPlan.from_dict(value)
                object.__setattr__(self, name, value)
            if value.arm != self.arm:
                raise ValueError("pick-place task uses a different arm")
            if value.selected_candidate_id != self.selected_candidate_id:
                raise ValueError("pick-place task changed the selected grasp")
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
        object.__setattr__(self, "phase_order", tuple(self.phase_order))
        if self.phase_order != PICK_PLACE_PHASE_ORDER:
            raise ValueError("tabletop pick-place plan has an invalid phase order")
        if tuple(value.to_pose_id for value in self.trajectories) != self.phase_order:
            raise ValueError("pick-place trajectory endpoints differ from its phase order")
        if len(self.trajectories) != len(PICK_PLACE_PHASE_ORDER):
            raise ValueError("tabletop pick-place plan is incomplete")
        if self.trajectories[0].from_pose_id != "clearance":
            raise ValueError("tabletop pick-place plan must begin at clearance")
        for previous, current in zip(self.trajectories, self.trajectories[1:], strict=False):
            error = float(
                np.max(
                    np.abs(
                        np.asarray(previous.command_q_rad[-1])
                        - np.asarray(current.command_q_rad[0])
                    )
                )
            )
            if error > 1.0e-8:
                raise ValueError(
                    "tabletop pick-place trajectory discontinuity: "
                    f"{previous.to_pose_id}->{current.to_pose_id}={error:.9f}rad"
                )

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
            "selected_destination_index": self.selected_destination_index,
            "source_task": self.source_task.to_dict(),
            "destination_task": self.destination_task.to_dict(),
            "trajectories": [value.to_dict() for value in self.trajectories],
            "phase_order": list(self.phase_order),
            "planner_provenance": self.planner_provenance,
        }
        if include_hash:
            result["content_sha256"] = self.content_sha256
        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TabletopPickPlacePlan:
        values = dict(data)
        expected_hash = values.pop("content_sha256", None)
        plan = cls(**values)
        if expected_hash is not None and expected_hash != plan.content_sha256:
            raise ValueError("tabletop pick-place plan SHA-256 mismatch")
        return plan

    @classmethod
    def from_json(cls, path: str | Path) -> TabletopPickPlacePlan:
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))

    def write_json(self, path: str | Path) -> None:
        atomic_write_json(path, self.to_dict())


@dataclass(frozen=True, slots=True)
class PickPlaceRetentionRouteValidationRequest:
    """Recheck the complete source-to-placement route at the measured close."""

    pick_place_request: TabletopPickPlaceRequest
    pick_place_plan: TabletopPickPlacePlan
    measured_active_dex3_q_rad: tuple[float, ...]
    blocked_motor_ids: tuple[int, ...]
    schema_version: int = PLANNER_SCHEMA_VERSION
    operation: str = "validate_pick_place_retention_route"

    def __post_init__(self) -> None:
        if self.schema_version != PLANNER_SCHEMA_VERSION:
            raise ValueError("unsupported pick-place retention schema version")
        if self.operation != "validate_pick_place_retention_route":
            raise ValueError("unsupported pick-place retention operation")
        if not isinstance(self.pick_place_request, TabletopPickPlaceRequest):
            object.__setattr__(
                self,
                "pick_place_request",
                TabletopPickPlaceRequest.from_dict(self.pick_place_request),
            )
        if not isinstance(self.pick_place_plan, TabletopPickPlacePlan):
            object.__setattr__(
                self,
                "pick_place_plan",
                TabletopPickPlacePlan.from_dict(self.pick_place_plan),
            )
        if self.pick_place_plan.request_sha256 != self.pick_place_request.content_sha256:
            raise ValueError("pick-place retention plan belongs to a different request")
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
            raise ValueError("pick-place retention blocked motor IDs are invalid")
        object.__setattr__(self, "blocked_motor_ids", blocked)

    @property
    def content_sha256(self) -> str:
        return _hash(self.to_dict(include_hash=False))

    def to_dict(self, *, include_hash: bool = True) -> dict[str, Any]:
        result = {
            "schema_version": self.schema_version,
            "operation": self.operation,
            "pick_place_request": self.pick_place_request.to_dict(),
            "pick_place_plan": self.pick_place_plan.to_dict(),
            "measured_active_dex3_q_rad": list(self.measured_active_dex3_q_rad),
            "blocked_motor_ids": list(self.blocked_motor_ids),
        }
        if include_hash:
            result["content_sha256"] = self.content_sha256
        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PickPlaceRetentionRouteValidationRequest:
        values = dict(data)
        expected_hash = values.pop("content_sha256", None)
        request = cls(**values)
        if expected_hash is not None and expected_hash != request.content_sha256:
            raise ValueError("pick-place retention request SHA-256 mismatch")
        return request

    @classmethod
    def from_json(cls, path: str | Path) -> PickPlaceRetentionRouteValidationRequest:
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))

    def write_json(self, path: str | Path) -> None:
        atomic_write_json(path, self.to_dict())


@dataclass(frozen=True, slots=True)
class RetentionRouteValidationRequest:
    """Collision-check the frozen payload route at the measured close posture."""

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
    """Proof that measured close fingers do not invalidate the payload arm route."""

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
            ("move_to_pregrasp", "return_to_clearance"),
        ):
            raise ValueError("tabletop recovery trajectory endpoints are invalid")
        for recovery, normal in zip(
            self.recovery_trajectories[:2],
            (self.trajectories[7], self.trajectories[6]),
            strict=True,
        ):
            if (
                recovery.sample_time_s != normal.sample_time_s
                or recovery.command_q_rad != normal.command_q_rad
                or recovery.model_q_rad != normal.model_q_rad
            ):
                raise ValueError("tabletop recovery motion differs from its frozen reverse path")
        expected_pregrasp_return = _reversed_trajectory_with_ids(
            self.trajectories[1],
            from_pose_id="move_to_pregrasp",
            to_pose_id="return_to_clearance",
        )
        if self.recovery_trajectories[2] != expected_pregrasp_return:
            raise ValueError("pregrasp recovery is not the exact outbound reverse")
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
                        np.asarray(self.recovery_trajectories[2].command_q_rad[0])
                        - np.asarray(self.trajectories[1].command_q_rad[-1])
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


@dataclass(frozen=True, slots=True)
class TabletopPregraspPlan:
    """One validated clearance-to-pregrasp route and its exact reverse."""

    request_sha256: str
    arm: str
    selected_candidate_id: str
    object_T_grasp: tuple[tuple[float, ...], ...]
    open_active_dex3_q_rad: tuple[float, ...]
    outbound: PlannedTrajectory
    inbound: PlannedTrajectory
    planner_provenance: dict[str, Any]
    schema_version: int = PLANNER_SCHEMA_VERSION
    kind: str = "g1_tabletop_clearance_to_pregrasp_plan"

    def __post_init__(self) -> None:
        if len(self.request_sha256) != 64:
            raise ValueError("request SHA-256 must contain 64 characters")
        object.__setattr__(self, "arm", validate_arm_side(self.arm))
        if not self.selected_candidate_id:
            raise ValueError("pregrasp plan has no grasp candidate")
        object.__setattr__(
            self,
            "object_T_grasp",
            _finite_transform(self.object_T_grasp, "object_T_grasp"),
        )
        object.__setattr__(
            self,
            "open_active_dex3_q_rad",
            _finite_vector(self.open_active_dex3_q_rad, 7, "open_active_dex3_q_rad"),
        )
        expected_open, _expected_close = dex3_execution_profile(self.arm)
        if self.open_active_dex3_q_rad != expected_open:
            raise ValueError("pregrasp plan open target differs from the Dex3 descriptor")
        if not isinstance(self.outbound, PlannedTrajectory):
            object.__setattr__(self, "outbound", PlannedTrajectory.from_dict(self.outbound))
        if not isinstance(self.inbound, PlannedTrajectory):
            object.__setattr__(self, "inbound", PlannedTrajectory.from_dict(self.inbound))
        if (self.outbound.from_pose_id, self.outbound.to_pose_id) != (
            "clearance",
            "move_to_pregrasp",
        ):
            raise ValueError("pregrasp outbound route has invalid endpoints")
        if (self.inbound.from_pose_id, self.inbound.to_pose_id) != (
            "move_to_pregrasp",
            "return_to_clearance",
        ):
            raise ValueError("pregrasp inbound route has invalid endpoints")
        expected_inbound = _reversed_trajectory_with_ids(
            self.outbound,
            from_pose_id="move_to_pregrasp",
            to_pose_id="return_to_clearance",
        )
        if self.inbound != expected_inbound:
            raise ValueError("pregrasp inbound route is not the exact outbound reverse")

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
            "outbound": self.outbound.to_dict(),
            "inbound": self.inbound.to_dict(),
            "planner_provenance": self.planner_provenance,
        }
        if include_hash:
            result["content_sha256"] = self.content_sha256
        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TabletopPregraspPlan:
        values = dict(data)
        expected_hash = values.pop("content_sha256", None)
        plan = cls(**values)
        if expected_hash is not None and expected_hash != plan.content_sha256:
            raise ValueError("pregrasp plan SHA-256 mismatch")
        return plan

    @classmethod
    def from_json(cls, path: str | Path) -> TabletopPregraspPlan:
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))

    def write_json(self, path: str | Path) -> None:
        atomic_write_json(path, self.to_dict())


def _trajectory_with_ids(
    trajectory: PlannedTrajectory,
    *,
    from_pose_id: str,
    to_pose_id: str,
) -> PlannedTrajectory:
    return PlannedTrajectory(
        from_pose_id=from_pose_id,
        to_pose_id=to_pose_id,
        sample_time_s=trajectory.sample_time_s,
        command_q_rad=trajectory.command_q_rad,
        model_q_rad=trajectory.model_q_rad,
        planning_time_s=trajectory.planning_time_s,
    )


def _reversed_trajectory_with_ids(
    trajectory: PlannedTrajectory,
    *,
    from_pose_id: str,
    to_pose_id: str,
) -> PlannedTrajectory:
    duration = trajectory.sample_time_s[-1]
    return PlannedTrajectory(
        from_pose_id=from_pose_id,
        to_pose_id=to_pose_id,
        sample_time_s=tuple(duration - value for value in reversed(trajectory.sample_time_s)),
        command_q_rad=tuple(reversed(trajectory.command_q_rad)),
        model_q_rad=tuple(reversed(trajectory.model_q_rad)),
        planning_time_s=0.0,
    )


@dataclass(frozen=True, slots=True)
class PregraspRemainingPlan:
    """Hash-bound corrected lifecycle from the reached original pregrasp.

    CuRobo replans the same selected grasp from the exact active pregrasp
    command.  After replacing the cube, this plan returns through the newly
    planned pregrasp and then reverses the already validated original
    clearance-to-pregrasp route exactly.
    """

    prior_pregrasp_plan_sha256: str
    estimated_request_sha256: str
    task: TabletopTaskPlan
    trajectories: tuple[PlannedTrajectory, ...]
    recovery_trajectories: tuple[PlannedTrajectory, ...]
    schema_version: int = PLANNER_SCHEMA_VERSION
    kind: str = "g1_tabletop_pregrasp_corrected_remaining_plan"

    def __post_init__(self) -> None:
        for name in ("prior_pregrasp_plan_sha256", "estimated_request_sha256"):
            if len(getattr(self, name)) != 64:
                raise ValueError(f"{name} must contain 64 characters")
        if not isinstance(self.task, TabletopTaskPlan):
            object.__setattr__(self, "task", TabletopTaskPlan.from_dict(self.task))
        if self.task.request_sha256 != self.estimated_request_sha256:
            raise ValueError("pregrasp task belongs to a different estimated request")
        object.__setattr__(
            self,
            "trajectories",
            tuple(
                item if isinstance(item, PlannedTrajectory) else PlannedTrajectory.from_dict(item)
                for item in self.trajectories
            ),
        )
        expected_edges = (
            ("move_to_pregrasp", "estimated_pregrasp"),
            ("estimated_pregrasp", "grasp_approach"),
            ("grasp_approach", "retention_test_lift"),
            ("retention_test_lift", "payload_lift"),
            ("payload_lift", "payload_lower"),
            ("payload_lower", "payload_replace"),
            ("payload_replace", "grasp_retreat"),
            ("grasp_retreat", "return_to_pregrasp"),
            ("return_to_pregrasp", "return_to_clearance"),
        )
        if tuple((item.from_pose_id, item.to_pose_id) for item in self.trajectories) != (
            expected_edges
        ):
            raise ValueError("pregrasp-corrected remaining lifecycle is disconnected")
        object.__setattr__(
            self,
            "recovery_trajectories",
            tuple(
                item if isinstance(item, PlannedTrajectory) else PlannedTrajectory.from_dict(item)
                for item in self.recovery_trajectories
            ),
        )
        expected_recovery = (
            ("grasp_approach", "grasp_retreat"),
            ("retention_test_lift", "payload_replace"),
            ("estimated_pregrasp", "move_to_pregrasp"),
            ("move_to_pregrasp", "return_to_clearance"),
        )
        if (
            tuple((item.from_pose_id, item.to_pose_id) for item in self.recovery_trajectories)
            != expected_recovery
        ):
            raise ValueError("pregrasp-corrected recovery routes are invalid")
        joins = zip(self.trajectories, self.trajectories[1:], strict=False)
        for first, second in joins:
            error = float(
                np.max(
                    np.abs(
                        np.asarray(first.command_q_rad[-1]) - np.asarray(second.command_q_rad[0])
                    )
                )
            )
            if error > 1.0e-8:
                raise ValueError(f"pregrasp-corrected route is discontinuous by {error:.9f}rad")
        if self.task.selected_candidate_id == "":
            raise ValueError("pregrasp-corrected task has no grasp candidate")

    @property
    def selected_candidate_id(self) -> str:
        return self.task.selected_candidate_id

    @property
    def content_sha256(self) -> str:
        return _hash(self.to_dict(include_hash=False))

    def to_dict(self, *, include_hash: bool = True) -> dict[str, Any]:
        result = {
            "schema_version": self.schema_version,
            "kind": self.kind,
            "prior_pregrasp_plan_sha256": self.prior_pregrasp_plan_sha256,
            "estimated_request_sha256": self.estimated_request_sha256,
            "task": self.task.to_dict(),
            "trajectories": [item.to_dict() for item in self.trajectories],
            "recovery_trajectories": [item.to_dict() for item in self.recovery_trajectories],
        }
        if include_hash:
            result["content_sha256"] = self.content_sha256
        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PregraspRemainingPlan:
        values = dict(data)
        expected_hash = values.pop("content_sha256", None)
        plan = cls(**values)
        if expected_hash is not None and expected_hash != plan.content_sha256:
            raise ValueError("pregrasp remaining-plan SHA-256 mismatch")
        return plan

    @classmethod
    def from_json(cls, path: str | Path) -> PregraspRemainingPlan:
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))

    def write_json(self, path: str | Path) -> None:
        atomic_write_json(path, self.to_dict())


def build_pregrasp_remaining_plan(
    *,
    prior_pregrasp_plan: TabletopPregraspPlan,
    estimated_request: TabletopTaskRequest,
    task: TabletopTaskPlan,
) -> PregraspRemainingPlan:
    """Map a same-grasp replan onto the reached-boundary execution names."""

    if estimated_request.estimated_planning_state is None:
        raise ValueError("pregrasp correction requires an estimated planning state")
    if task.request_sha256 != estimated_request.content_sha256:
        raise ValueError("pregrasp task belongs to another estimated request")
    if task.selected_candidate_id != prior_pregrasp_plan.selected_candidate_id:
        raise ValueError("pregrasp correction changed the selected grasp candidate")
    old_approach = prior_pregrasp_plan.outbound
    current_command = np.asarray(old_approach.command_q_rad[-1])
    new_start = np.asarray(task.trajectories[0].command_q_rad[0])
    if float(np.max(np.abs(current_command - new_start))) > 1.0e-8:
        raise ValueError("pregrasp correction does not start at the exact active command")

    corrected = (
        _trajectory_with_ids(
            task.trajectories[0],
            from_pose_id="move_to_pregrasp",
            to_pose_id="estimated_pregrasp",
        ),
        _trajectory_with_ids(
            task.trajectories[1],
            from_pose_id="estimated_pregrasp",
            to_pose_id="grasp_approach",
        ),
        *task.trajectories[2:7],
        _trajectory_with_ids(
            task.trajectories[7],
            from_pose_id="grasp_retreat",
            to_pose_id="return_to_pregrasp",
        ),
        _reversed_trajectory_with_ids(
            old_approach,
            from_pose_id="return_to_pregrasp",
            to_pose_id="return_to_clearance",
        ),
    )
    contact_retreat = _trajectory_with_ids(
        task.trajectories[6],
        from_pose_id="grasp_approach",
        to_pose_id="grasp_retreat",
    )
    test_lift_replace = _trajectory_with_ids(
        task.trajectories[5],
        from_pose_id="retention_test_lift",
        to_pose_id="payload_replace",
    )
    estimated_pregrasp_return = _reversed_trajectory_with_ids(
        corrected[0],
        from_pose_id="estimated_pregrasp",
        to_pose_id="move_to_pregrasp",
    )
    boundary_return = pregrasp_to_clearance_return(prior_pregrasp_plan)
    return PregraspRemainingPlan(
        prior_pregrasp_plan_sha256=prior_pregrasp_plan.content_sha256,
        estimated_request_sha256=estimated_request.content_sha256,
        task=task,
        trajectories=corrected,
        recovery_trajectories=(
            contact_retreat,
            test_lift_replace,
            estimated_pregrasp_return,
            boundary_return,
        ),
    )


def pregrasp_to_clearance_return(
    pregrasp_plan: TabletopPregraspPlan,
) -> PlannedTrajectory:
    """Return from the original pregrasp by exactly reversing its frozen route."""

    return pregrasp_plan.inbound


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
    pregrasp_return = _reversed_trajectory_with_ids(
        task.trajectories[0],
        from_pose_id="move_to_pregrasp",
        to_pose_id="return_to_clearance",
    )
    return TabletopExecutionPlan(
        loaded_request_sha256=loaded_request.content_sha256,
        clearance_request_sha256=clearance_request.content_sha256,
        supported_escape=supported_escape,
        task=task,
        trajectories=(supported_escape.outbound, *task.trajectories, inbound),
        recovery_trajectories=(contact_retreat, test_lift_replace, pregrasp_return),
    )
