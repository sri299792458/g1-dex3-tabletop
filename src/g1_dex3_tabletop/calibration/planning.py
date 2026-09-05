"""Hash-bound offline planning contracts for bilateral calibration."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np

from g1_aprilcube_calibration.camera_models import RectifiedCameraInfo
from g1_aprilcube_calibration.joint_map import (
    G1_29_JOINT_NAMES,
    arm_indices,
    arm_joint_names,
    validate_full_joint_vector,
)
from g1_aprilcube_calibration.pose_schema import HANDOFF_POSE_ID
from g1_aprilcube_calibration.transforms import validate_transform
from g1_aprilcube_calibration.urdf_model import URDFModel
from g1_dex3_tabletop.calibration.design import (
    BilateralCaptureWaypoint,
    BilateralDesignCandidate,
    BilateralDesignConfig,
    BilateralDesignSelection,
    BilateralPoseDesignArtifact,
    build_valid_graph_route_schedule,
    linearize_design_candidate,
    select_bilateral_design,
)
from g1_dex3_tabletop.calibration.execution import (
    BilateralExecutionPlan,
    BilateralPlannedTransition,
)
from g1_dex3_tabletop.calibration.models import (
    BilateralCalibrationSample,
    BilateralModelSpec,
    CameraFrameArtifact,
    TargetObservation,
)
from g1_dex3_tabletop.calibration.projection import BilateralCalibrationProjection
from g1_dex3_tabletop.planning.contracts import (
    CalibrationCandidate,
    Dex3PreparationPlan,
    Dex3PreparationRequest,
    RobotSnapshot,
    atomic_write_json,
)

_SIDES = ("left", "right")
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_POSE_EPSILON_RAD = 1e-8


def _canonical_mapping(value: dict[str, Any], *, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise TypeError(f"{name} must be a mapping")
    try:
        result = json.loads(json.dumps(value, sort_keys=True, allow_nan=False))
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must contain finite JSON data") from error
    if not isinstance(result, dict):
        raise TypeError(f"{name} must be a mapping")
    return result


def _content_sha256(document: dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(
            document,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode()
    ).hexdigest()


def _finite_transform(value: Any, *, name: str) -> tuple[tuple[float, ...], ...]:
    matrix = validate_transform(value)
    return tuple(tuple(float(item) for item in row) for row in matrix)


def _side_mapping(value: dict[str, Any], *, name: str) -> dict[str, Any]:
    if set(value) != set(_SIDES):
        raise ValueError(f"{name} must contain left and right")
    return dict(value)


def _dex3_mapping(value: dict[str, Any], *, name: str) -> dict[str, tuple[float, ...]]:
    values = _side_mapping(value, name=name)
    result: dict[str, tuple[float, ...]] = {}
    for side in _SIDES:
        posture = np.asarray(values[side], dtype=np.float64).reshape(-1)
        if posture.shape != (7,) or not np.all(np.isfinite(posture)):
            raise ValueError(f"{name} {side} posture must contain seven finite values")
        result[side] = tuple(float(item) for item in posture)
    return result


@dataclass(frozen=True, slots=True)
class BilateralVisibilityConfig:
    image_margin_px: float = 40.0
    minimum_target_span_px: float = 35.0
    minimum_target_separation_px: float = 60.0
    minimum_depth_m: float = 0.20
    maximum_depth_m: float = 0.75
    image_grid_columns: int = 3
    image_grid_rows: int = 3
    depth_bins: int = 3

    def __post_init__(self) -> None:
        for name in (
            "image_margin_px",
            "minimum_target_span_px",
            "minimum_target_separation_px",
            "minimum_depth_m",
            "maximum_depth_m",
        ):
            value = float(getattr(self, name))
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be positive and finite")
        if self.minimum_depth_m >= self.maximum_depth_m:
            raise ValueError("bilateral visibility depth bounds are invalid")
        for name in ("image_grid_columns", "image_grid_rows", "depth_bins"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")

    def to_dict(self) -> dict[str, Any]:
        return {name: getattr(self, name) for name in self.__dataclass_fields__}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> BilateralVisibilityConfig:
        if set(data) != set(cls.__dataclass_fields__):
            raise ValueError("bilateral visibility fields differ from schema")
        return cls(**data)


@dataclass(frozen=True, slots=True)
class BilateralCalibrationPlanningRequest:
    """Complete immutable input to bilateral IK, selection, and route planning."""

    robot_model: str
    urdf_sha256: str
    snapshot: RobotSnapshot
    clearance_snapshot: RobotSnapshot
    dex3_preparation_request: Dex3PreparationRequest
    dex3_preparation_plan: Dex3PreparationPlan
    camera_info: dict[str, Any]
    camera_frames: CameraFrameArtifact
    design_model: BilateralModelSpec
    design_config: BilateralDesignConfig
    visibility_config: BilateralVisibilityConfig
    nominal_torso_T_camera: tuple[tuple[float, ...], ...]
    nominal_hand_T_targets: dict[str, tuple[tuple[float, ...], ...]]
    joint_position_offsets_rad: dict[str, float]
    dex3_command_positions_rad: dict[str, tuple[float, ...]]
    dex3_model_positions_rad: dict[str, tuple[float, ...]]
    candidates_by_arm: dict[str, tuple[CalibrationCandidate, ...]]
    target_artifact_sha256_by_arm: dict[str, str]
    target_corner_tag_ids_by_arm: dict[str, tuple[int, ...]]
    target_object_points_m_by_arm: dict[str, tuple[tuple[float, ...], ...]]
    ik_batch_size: int
    random_seed: int
    source_provenance: dict[str, Any]
    schema_version: int = 2

    def __post_init__(self) -> None:
        if self.schema_version != 2:
            raise ValueError("unsupported bilateral planning-request schema version")
        if not self.robot_model.strip():
            raise ValueError("bilateral planning robot model must be non-empty")
        if not _SHA256_PATTERN.fullmatch(self.urdf_sha256):
            raise ValueError("bilateral planning URDF hash must be lowercase SHA-256")
        snapshot = (
            self.snapshot
            if isinstance(self.snapshot, RobotSnapshot)
            else RobotSnapshot.from_dict(self.snapshot)
        )
        clearance_snapshot = (
            self.clearance_snapshot
            if isinstance(self.clearance_snapshot, RobotSnapshot)
            else RobotSnapshot.from_dict(self.clearance_snapshot)
        )
        preparation_request = (
            self.dex3_preparation_request
            if isinstance(self.dex3_preparation_request, Dex3PreparationRequest)
            else Dex3PreparationRequest.from_dict(self.dex3_preparation_request)
        )
        preparation_plan = (
            self.dex3_preparation_plan
            if isinstance(self.dex3_preparation_plan, Dex3PreparationPlan)
            else Dex3PreparationPlan.from_dict(self.dex3_preparation_plan)
        )
        if preparation_request.snapshot != snapshot:
            raise ValueError("Dex3 preparation did not start from the bilateral Ready snapshot")
        if preparation_plan.request_sha256 != preparation_request.content_sha256:
            raise ValueError("Dex3 preparation plan belongs to a different request")
        camera_info = RectifiedCameraInfo.from_dict(dict(self.camera_info))
        camera_frames = (
            self.camera_frames
            if isinstance(self.camera_frames, CameraFrameArtifact)
            else CameraFrameArtifact.from_dict(self.camera_frames)
        )
        if camera_info.serial_number != camera_frames.camera_serial:
            raise ValueError("bilateral request camera serial differs from its frame artifact")
        if camera_info.frame_id != camera_frames.optical_frame:
            raise ValueError("bilateral request CameraInfo frame differs from its frame artifact")
        model = (
            self.design_model
            if isinstance(self.design_model, BilateralModelSpec)
            else BilateralModelSpec.from_dict(self.design_model)
        )
        design = (
            self.design_config
            if isinstance(self.design_config, BilateralDesignConfig)
            else BilateralDesignConfig.from_dict(self.design_config)
        )
        visibility = (
            self.visibility_config
            if isinstance(self.visibility_config, BilateralVisibilityConfig)
            else BilateralVisibilityConfig.from_dict(self.visibility_config)
        )
        transforms = _side_mapping(self.nominal_hand_T_targets, name="nominal hand targets")
        nominal_targets = {
            side: _finite_transform(transforms[side], name=f"nominal {side} hand target")
            for side in _SIDES
        }
        candidates_input = _side_mapping(self.candidates_by_arm, name="candidate arms")
        candidates = {
            side: tuple(
                item
                if isinstance(item, CalibrationCandidate)
                else CalibrationCandidate.from_dict(item)
                for item in candidates_input[side]
            )
            for side in _SIDES
        }
        if any(not candidates[side] for side in _SIDES):
            raise ValueError("bilateral planning requires candidates for both arms")
        candidate_ids = [item.candidate_id for side in _SIDES for item in candidates[side]]
        if len(candidate_ids) != len(set(candidate_ids)):
            raise ValueError("bilateral planning candidate IDs must be globally unique")
        hashes = {
            side: str(value)
            for side, value in _side_mapping(
                self.target_artifact_sha256_by_arm,
                name="target artifact hashes",
            ).items()
        }
        if any(not _SHA256_PATTERN.fullmatch(value) for value in hashes.values()):
            raise ValueError("bilateral target artifact hashes must be lowercase SHA-256")
        tag_ids_input = _side_mapping(
            self.target_corner_tag_ids_by_arm,
            name="target corner tag IDs",
        )
        points_input = _side_mapping(
            self.target_object_points_m_by_arm,
            name="target object points",
        )
        tag_ids: dict[str, tuple[int, ...]] = {}
        points: dict[str, tuple[tuple[float, ...], ...]] = {}
        for side in _SIDES:
            side_points = np.asarray(points_input[side], dtype=np.float64)
            side_tag_ids = tuple(int(value) for value in tag_ids_input[side])
            if (
                side_points.ndim != 2
                or side_points.shape[1:] != (3,)
                or len(side_points) < 4
                or len(side_points) % 4
                or len(side_tag_ids) != len(side_points)
                or not np.all(np.isfinite(side_points))
            ):
                raise ValueError(f"bilateral {side} target geometry is invalid")
            tag_ids[side] = side_tag_ids
            points[side] = tuple(tuple(float(value) for value in row) for row in side_points)
        offsets = {
            str(name): float(value) for name, value in self.joint_position_offsets_rad.items()
        }
        if any(
            name not in G1_29_JOINT_NAMES or not np.isfinite(value)
            for name, value in offsets.items()
        ):
            raise ValueError("bilateral planning joint offsets are invalid")
        if self.ik_batch_size < 1:
            raise ValueError("bilateral IK batch size must be positive")
        if isinstance(self.random_seed, bool) or not isinstance(self.random_seed, int):
            raise TypeError("bilateral planning random seed must be an integer")
        command_positions = _dex3_mapping(
            self.dex3_command_positions_rad,
            name="Dex3 command",
        )
        model_positions = _dex3_mapping(
            self.dex3_model_positions_rad,
            name="Dex3 model",
        )
        if preparation_request.left_target_q_rad != command_positions["left"] or (
            preparation_request.right_target_q_rad != command_positions["right"]
        ):
            raise ValueError("Dex3 preparation targets differ from the commissioned close")
        if (
            preparation_request.left_settled_target_q_rad != model_positions["left"]
            or preparation_request.right_settled_target_q_rad != model_positions["right"]
            or preparation_request.left_return_target_q_rad is None
            or preparation_request.right_return_target_q_rad is None
            or preparation_plan.return_sweep_sample_count < 2
        ):
            raise ValueError(
                "Dex3 preparation did not certify empty-close to its declared open posture"
            )
        clearance_q = np.asarray(clearance_snapshot.measured_q29_rad, dtype=np.float64)
        clearance_q14 = clearance_q[
            np.asarray((*arm_indices("left"), *arm_indices("right")), dtype=np.int64)
        ]
        if (
            np.max(np.abs(clearance_q14 - np.asarray(preparation_plan.dual_clearance_q14_rad)))
            > _POSE_EPSILON_RAD
        ):
            raise ValueError("bilateral clearance snapshot differs from its preparation plan")
        if (
            clearance_snapshot.left_dex3_q_rad != model_positions["left"]
            or clearance_snapshot.right_dex3_q_rad != model_positions["right"]
        ):
            raise ValueError("bilateral clearance snapshot must use the closed-hand model")
        object.__setattr__(self, "snapshot", snapshot)
        object.__setattr__(self, "clearance_snapshot", clearance_snapshot)
        object.__setattr__(self, "dex3_preparation_request", preparation_request)
        object.__setattr__(self, "dex3_preparation_plan", preparation_plan)
        object.__setattr__(self, "camera_info", camera_info.to_dict())
        object.__setattr__(self, "camera_frames", camera_frames)
        object.__setattr__(self, "design_model", model)
        object.__setattr__(self, "design_config", design)
        object.__setattr__(self, "visibility_config", visibility)
        object.__setattr__(
            self,
            "nominal_torso_T_camera",
            _finite_transform(self.nominal_torso_T_camera, name="nominal torso camera"),
        )
        object.__setattr__(self, "nominal_hand_T_targets", nominal_targets)
        object.__setattr__(self, "joint_position_offsets_rad", dict(sorted(offsets.items())))
        object.__setattr__(
            self,
            "dex3_command_positions_rad",
            command_positions,
        )
        object.__setattr__(
            self,
            "dex3_model_positions_rad",
            model_positions,
        )
        object.__setattr__(self, "candidates_by_arm", candidates)
        object.__setattr__(self, "target_artifact_sha256_by_arm", hashes)
        object.__setattr__(self, "target_corner_tag_ids_by_arm", tag_ids)
        object.__setattr__(self, "target_object_points_m_by_arm", points)
        object.__setattr__(
            self,
            "source_provenance",
            _canonical_mapping(self.source_provenance, name="bilateral source provenance"),
        )

    @property
    def content_sha256(self) -> str:
        return _content_sha256(self.to_dict(include_hash=False))

    def to_dict(self, *, include_hash: bool = True) -> dict[str, Any]:
        result = {
            "schema_version": self.schema_version,
            "robot_model": self.robot_model,
            "urdf_sha256": self.urdf_sha256,
            "snapshot": self.snapshot.to_dict(),
            "clearance_snapshot": self.clearance_snapshot.to_dict(),
            "dex3_preparation_request": self.dex3_preparation_request.to_dict(),
            "dex3_preparation_plan": self.dex3_preparation_plan.to_dict(),
            "camera_info": self.camera_info,
            "camera_frames": self.camera_frames.to_dict(),
            "design_model": self.design_model.to_dict(),
            "design_config": self.design_config.to_dict(),
            "visibility_config": self.visibility_config.to_dict(),
            "nominal_torso_T_camera": [list(row) for row in self.nominal_torso_T_camera],
            "nominal_hand_T_targets": {
                side: [list(row) for row in self.nominal_hand_T_targets[side]] for side in _SIDES
            },
            "joint_position_offsets_rad": self.joint_position_offsets_rad,
            "dex3_command_positions_rad": {
                side: list(self.dex3_command_positions_rad[side]) for side in _SIDES
            },
            "dex3_model_positions_rad": {
                side: list(self.dex3_model_positions_rad[side]) for side in _SIDES
            },
            "candidates_by_arm": {
                side: [item.to_dict() for item in self.candidates_by_arm[side]] for side in _SIDES
            },
            "target_artifact_sha256_by_arm": self.target_artifact_sha256_by_arm,
            "target_corner_tag_ids_by_arm": {
                side: list(self.target_corner_tag_ids_by_arm[side]) for side in _SIDES
            },
            "target_object_points_m_by_arm": {
                side: [list(row) for row in self.target_object_points_m_by_arm[side]]
                for side in _SIDES
            },
            "ik_batch_size": self.ik_batch_size,
            "random_seed": self.random_seed,
            "source_provenance": self.source_provenance,
        }
        if include_hash:
            result["content_sha256"] = self.content_sha256
        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> BilateralCalibrationPlanningRequest:
        expected = {
            "schema_version",
            "robot_model",
            "urdf_sha256",
            "snapshot",
            "clearance_snapshot",
            "dex3_preparation_request",
            "dex3_preparation_plan",
            "camera_info",
            "camera_frames",
            "design_model",
            "design_config",
            "visibility_config",
            "nominal_torso_T_camera",
            "nominal_hand_T_targets",
            "joint_position_offsets_rad",
            "dex3_command_positions_rad",
            "dex3_model_positions_rad",
            "candidates_by_arm",
            "target_artifact_sha256_by_arm",
            "target_corner_tag_ids_by_arm",
            "target_object_points_m_by_arm",
            "ik_batch_size",
            "random_seed",
            "source_provenance",
            "content_sha256",
        }
        if set(data) != expected:
            raise ValueError("bilateral planning-request fields differ from schema version 1")
        result = cls(**{key: value for key, value in data.items() if key != "content_sha256"})
        if result.content_sha256 != data["content_sha256"]:
            raise ValueError("bilateral planning-request content SHA-256 mismatch")
        return result

    @classmethod
    def from_json(cls, path: str | Path) -> BilateralCalibrationPlanningRequest:
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))

    def write_json(self, path: str | Path) -> None:
        atomic_write_json(path, self.to_dict())


@dataclass(frozen=True, slots=True)
class BilateralFeasiblePose:
    candidate_id: str
    active_arm: Literal["left", "right"]
    active_model_q_rad: tuple[float, ...]
    active_command_q_rad: tuple[float, ...]
    full_command_q29_rad: tuple[float, ...]
    ik_position_error_m: float
    ik_rotation_error_rad: float
    candidate_metadata: dict[str, Any]

    def __post_init__(self) -> None:
        if not self.candidate_id.strip() or self.active_arm not in _SIDES:
            raise ValueError("bilateral feasible pose identity is invalid")
        model_q = np.asarray(self.active_model_q_rad, dtype=np.float64).reshape(-1)
        command_q = np.asarray(self.active_command_q_rad, dtype=np.float64).reshape(-1)
        if (
            model_q.shape != (7,)
            or command_q.shape != (7,)
            or not np.all(np.isfinite(model_q))
            or not np.all(np.isfinite(command_q))
        ):
            raise ValueError("bilateral feasible pose arm values must contain seven values")
        full_q = validate_full_joint_vector(
            self.full_command_q29_rad,
            name="bilateral feasible full command",
        )
        errors = (float(self.ik_position_error_m), float(self.ik_rotation_error_rad))
        if not np.all(np.isfinite(errors)) or any(value < 0.0 for value in errors):
            raise ValueError("bilateral feasible pose IK errors are invalid")
        indices = np.asarray(arm_indices(self.active_arm), dtype=np.int64)
        if np.max(np.abs(full_q[indices] - command_q)) > _POSE_EPSILON_RAD:
            raise ValueError("bilateral feasible full command differs from its active arm")
        object.__setattr__(self, "active_model_q_rad", tuple(float(value) for value in model_q))
        object.__setattr__(
            self,
            "active_command_q_rad",
            tuple(float(value) for value in command_q),
        )
        object.__setattr__(self, "full_command_q29_rad", tuple(float(value) for value in full_q))
        object.__setattr__(self, "ik_position_error_m", errors[0])
        object.__setattr__(self, "ik_rotation_error_rad", errors[1])
        object.__setattr__(
            self,
            "candidate_metadata",
            _canonical_mapping(self.candidate_metadata, name="candidate metadata"),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "active_arm": self.active_arm,
            "active_model_q_rad": list(self.active_model_q_rad),
            "active_command_q_rad": list(self.active_command_q_rad),
            "full_command_q29_rad": list(self.full_command_q29_rad),
            "ik_position_error_m": self.ik_position_error_m,
            "ik_rotation_error_rad": self.ik_rotation_error_rad,
            "candidate_metadata": self.candidate_metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> BilateralFeasiblePose:
        if set(data) != set(cls.__dataclass_fields__):
            raise ValueError("bilateral feasible-pose fields differ from schema")
        return cls(**data)


@dataclass(frozen=True, slots=True)
class BilateralIKResult:
    request_sha256: str
    poses: tuple[BilateralFeasiblePose, ...]
    planner_provenance: dict[str, Any]
    schema_version: int = 1

    def __post_init__(self) -> None:
        if self.schema_version != 1:
            raise ValueError("unsupported bilateral IK-result schema version")
        if not _SHA256_PATTERN.fullmatch(self.request_sha256):
            raise ValueError("bilateral IK request hash must be lowercase SHA-256")
        poses = tuple(
            item
            if isinstance(item, BilateralFeasiblePose)
            else BilateralFeasiblePose.from_dict(item)
            for item in self.poses
        )
        ids = [item.candidate_id for item in poses]
        if not poses or len(ids) != len(set(ids)):
            raise ValueError("bilateral IK result requires unique feasible poses")
        object.__setattr__(self, "poses", poses)
        object.__setattr__(
            self,
            "planner_provenance",
            _canonical_mapping(self.planner_provenance, name="IK planner provenance"),
        )

    @property
    def content_sha256(self) -> str:
        return _content_sha256(self.to_dict(include_hash=False))

    def validate_request(self, request: BilateralCalibrationPlanningRequest) -> None:
        if self.request_sha256 != request.content_sha256:
            raise ValueError("bilateral IK result belongs to a different planning request")
        expected = {
            item.candidate_id: side for side in _SIDES for item in request.candidates_by_arm[side]
        }
        if any(expected.get(item.candidate_id) != item.active_arm for item in self.poses):
            raise ValueError("bilateral IK result contains an unknown or wrong-arm candidate")

    def to_dict(self, *, include_hash: bool = True) -> dict[str, Any]:
        result = {
            "schema_version": self.schema_version,
            "request_sha256": self.request_sha256,
            "poses": [item.to_dict() for item in self.poses],
            "planner_provenance": self.planner_provenance,
        }
        if include_hash:
            result["content_sha256"] = self.content_sha256
        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> BilateralIKResult:
        expected = {
            "schema_version",
            "request_sha256",
            "poses",
            "planner_provenance",
            "content_sha256",
        }
        if set(data) != expected:
            raise ValueError("bilateral IK-result fields differ from schema version 1")
        result = cls(**{key: value for key, value in data.items() if key != "content_sha256"})
        if result.content_sha256 != data["content_sha256"]:
            raise ValueError("bilateral IK-result content SHA-256 mismatch")
        return result

    @classmethod
    def from_json(cls, path: str | Path) -> BilateralIKResult:
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))

    def write_json(self, path: str | Path) -> None:
        atomic_write_json(path, self.to_dict())


@dataclass(frozen=True, slots=True)
class BilateralRoutePlanningRequest:
    planning_request_sha256: str
    ik_result_sha256: str
    robot_model: str
    urdf_sha256: str
    snapshot: RobotSnapshot
    clearance_snapshot: RobotSnapshot
    dex3_preparation_request: Dex3PreparationRequest
    dex3_preparation_plan: Dex3PreparationPlan
    joint_position_offsets_rad: dict[str, float]
    dex3_command_positions_rad: dict[str, tuple[float, ...]]
    dex3_model_positions_rad: dict[str, tuple[float, ...]]
    anchor_candidate_ids_by_arm: dict[str, str]
    parameter_names: tuple[str, ...]
    selection: BilateralDesignSelection
    schedule: tuple[BilateralCaptureWaypoint, ...]
    waypoint_joint_positions_rad: dict[str, tuple[float, ...]]
    design_provenance: dict[str, Any]
    random_seed: int
    schema_version: int = 3

    def __post_init__(self) -> None:
        if self.schema_version != 3:
            raise ValueError("unsupported bilateral route-request schema version")
        for name in ("planning_request_sha256", "ik_result_sha256", "urdf_sha256"):
            if not _SHA256_PATTERN.fullmatch(getattr(self, name)):
                raise ValueError(f"{name} must be lowercase SHA-256")
        if not self.robot_model.strip():
            raise ValueError("bilateral route robot model must be non-empty")
        if isinstance(self.random_seed, bool) or not isinstance(self.random_seed, int):
            raise TypeError("bilateral route random seed must be an integer")
        snapshot = (
            self.snapshot
            if isinstance(self.snapshot, RobotSnapshot)
            else RobotSnapshot.from_dict(self.snapshot)
        )
        clearance_snapshot = (
            self.clearance_snapshot
            if isinstance(self.clearance_snapshot, RobotSnapshot)
            else RobotSnapshot.from_dict(self.clearance_snapshot)
        )
        preparation_request = (
            self.dex3_preparation_request
            if isinstance(self.dex3_preparation_request, Dex3PreparationRequest)
            else Dex3PreparationRequest.from_dict(self.dex3_preparation_request)
        )
        preparation_plan = (
            self.dex3_preparation_plan
            if isinstance(self.dex3_preparation_plan, Dex3PreparationPlan)
            else Dex3PreparationPlan.from_dict(self.dex3_preparation_plan)
        )
        if preparation_request.snapshot != snapshot:
            raise ValueError("route Dex3 preparation did not start from Ready")
        if preparation_plan.request_sha256 != preparation_request.content_sha256:
            raise ValueError("route Dex3 preparation plan belongs to a different request")
        selection = (
            self.selection
            if isinstance(self.selection, BilateralDesignSelection)
            else BilateralDesignSelection.from_dict(self.selection)
        )
        schedule = tuple(
            item
            if isinstance(item, BilateralCaptureWaypoint)
            else BilateralCaptureWaypoint.from_dict(item)
            for item in self.schedule
        )
        names = tuple(str(value) for value in self.parameter_names)
        if not names or len(names) != len(set(names)):
            raise ValueError("bilateral route parameter names must be unique and non-empty")
        if len(names) != selection.report.model_parameter_count:
            raise ValueError("bilateral route parameter names differ from its selection")
        poses = {
            str(name): tuple(float(value) for value in validate_full_joint_vector(position))
            for name, position in self.waypoint_joint_positions_rad.items()
        }
        if set(poses) != {item.candidate_id for item in schedule}:
            raise ValueError("bilateral route request does not define every waypoint")
        anchor_pose_ids = {item.candidate_id for item in schedule if item.capture_role == "anchor"}
        if len(anchor_pose_ids) != 1:
            raise ValueError("bilateral route request must use one repeated anchor pose")
        if (
            schedule[0].occurrence_id != HANDOFF_POSE_ID
            or schedule[-1].occurrence_id != HANDOFF_POSE_ID
            or schedule[0].capture_role != "anchor"
            or schedule[-1].capture_role != "anchor"
            or schedule[0].candidate_id != schedule[-1].candidate_id
            or any(item.hand_action is not None for item in schedule)
        ):
            raise ValueError(
                "bilateral route request must be a hand-action-free anchor-to-anchor core"
            )
        anchors = {
            str(side): str(candidate_id)
            for side, candidate_id in _side_mapping(
                self.anchor_candidate_ids_by_arm,
                name="anchor source candidates",
            ).items()
        }
        if any(not value.strip() for value in anchors.values()):
            raise ValueError("bilateral anchor source candidate IDs must be non-empty")
        if len(set(anchors.values())) != 2:
            raise ValueError("bilateral anchor must use distinct left and right candidates")
        for index in range(len(schedule) - 1):
            self._transition_arm_from(schedule, poses, index)
        offsets = {
            str(name): float(value) for name, value in self.joint_position_offsets_rad.items()
        }
        if any(
            name not in G1_29_JOINT_NAMES or not np.isfinite(value)
            for name, value in offsets.items()
        ):
            raise ValueError("bilateral route joint offsets are invalid")
        command_positions = _dex3_mapping(
            self.dex3_command_positions_rad,
            name="Dex3 command",
        )
        model_positions = _dex3_mapping(
            self.dex3_model_positions_rad,
            name="Dex3 model",
        )
        if preparation_request.left_target_q_rad != command_positions["left"] or (
            preparation_request.right_target_q_rad != command_positions["right"]
        ):
            raise ValueError("route Dex3 preparation targets differ from fixed-close")
        if (
            preparation_request.left_settled_target_q_rad != model_positions["left"]
            or preparation_request.right_settled_target_q_rad != model_positions["right"]
            or preparation_request.left_return_target_q_rad is None
            or preparation_request.right_return_target_q_rad is None
            or preparation_plan.return_sweep_sample_count < 2
        ):
            raise ValueError("route Dex3 preparation lacks the declared open-hand sweep")
        if (
            clearance_snapshot.left_dex3_q_rad != model_positions["left"]
            or clearance_snapshot.right_dex3_q_rad != model_positions["right"]
        ):
            raise ValueError("route clearance snapshot must use the closed-hand model")
        object.__setattr__(self, "snapshot", snapshot)
        object.__setattr__(self, "clearance_snapshot", clearance_snapshot)
        object.__setattr__(self, "dex3_preparation_request", preparation_request)
        object.__setattr__(self, "dex3_preparation_plan", preparation_plan)
        object.__setattr__(self, "selection", selection)
        object.__setattr__(self, "schedule", schedule)
        object.__setattr__(self, "parameter_names", names)
        object.__setattr__(self, "waypoint_joint_positions_rad", poses)
        object.__setattr__(self, "joint_position_offsets_rad", dict(sorted(offsets.items())))
        object.__setattr__(
            self,
            "dex3_command_positions_rad",
            command_positions,
        )
        object.__setattr__(
            self,
            "dex3_model_positions_rad",
            model_positions,
        )
        object.__setattr__(self, "anchor_candidate_ids_by_arm", anchors)
        object.__setattr__(
            self,
            "design_provenance",
            _canonical_mapping(self.design_provenance, name="design provenance"),
        )

    @staticmethod
    def _transition_arm_from(schedule, poses, index: int) -> str:
        start = np.asarray(poses[schedule[index].candidate_id], dtype=np.float64)
        end = np.asarray(poses[schedule[index + 1].candidate_id], dtype=np.float64)
        changed = {
            side: float(
                np.max(
                    np.abs(
                        start[np.asarray(arm_indices(side), dtype=np.int64)]
                        - end[np.asarray(arm_indices(side), dtype=np.int64)]
                    )
                )
            )
            for side in _SIDES
        }
        active = [side for side, error in changed.items() if error > _POSE_EPSILON_RAD]
        if len(active) != 1:
            edge = f"{schedule[index].occurrence_id}->{schedule[index + 1].occurrence_id}"
            raise ValueError(f"bilateral route edge must change exactly one arm: {edge}")
        full_delta = np.abs(start - end)
        active_indices = set(arm_indices(active[0]))
        if any(
            full_delta[joint_index] > _POSE_EPSILON_RAD
            for joint_index in range(len(full_delta))
            if joint_index not in active_indices
        ):
            raise ValueError("bilateral route edge changes a locked body joint")
        return active[0]

    def transition_arm(self, index: int) -> str:
        if not 0 <= index < len(self.schedule) - 1:
            raise IndexError("bilateral route edge index is out of range")
        return self._transition_arm_from(
            self.schedule,
            self.waypoint_joint_positions_rad,
            index,
        )

    @property
    def anchor_candidate_id(self) -> str:
        return next(item.candidate_id for item in self.schedule if item.capture_role == "anchor")

    @property
    def calibration_snapshot(self) -> RobotSnapshot:
        """Full selected anchor with the commissioned empty-close collision model."""

        return RobotSnapshot(
            measured_q29_rad=self.waypoint_joint_positions_rad[self.anchor_candidate_id],
            left_dex3_q_rad=self.dex3_model_positions_rad["left"],
            right_dex3_q_rad=self.dex3_model_positions_rad["right"],
        )

    @property
    def content_sha256(self) -> str:
        return _content_sha256(self.to_dict(include_hash=False))

    def to_dict(self, *, include_hash: bool = True) -> dict[str, Any]:
        result = {
            "schema_version": self.schema_version,
            "planning_request_sha256": self.planning_request_sha256,
            "ik_result_sha256": self.ik_result_sha256,
            "robot_model": self.robot_model,
            "urdf_sha256": self.urdf_sha256,
            "snapshot": self.snapshot.to_dict(),
            "clearance_snapshot": self.clearance_snapshot.to_dict(),
            "dex3_preparation_request": self.dex3_preparation_request.to_dict(),
            "dex3_preparation_plan": self.dex3_preparation_plan.to_dict(),
            "joint_position_offsets_rad": self.joint_position_offsets_rad,
            "dex3_command_positions_rad": {
                side: list(self.dex3_command_positions_rad[side]) for side in _SIDES
            },
            "dex3_model_positions_rad": {
                side: list(self.dex3_model_positions_rad[side]) for side in _SIDES
            },
            "anchor_candidate_ids_by_arm": self.anchor_candidate_ids_by_arm,
            "parameter_names": list(self.parameter_names),
            "selection": self.selection.to_dict(),
            "schedule": [item.to_dict() for item in self.schedule],
            "waypoint_joint_positions_rad": {
                name: list(position)
                for name, position in sorted(self.waypoint_joint_positions_rad.items())
            },
            "design_provenance": self.design_provenance,
            "random_seed": self.random_seed,
        }
        if include_hash:
            result["content_sha256"] = self.content_sha256
        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> BilateralRoutePlanningRequest:
        expected = {
            "schema_version",
            "planning_request_sha256",
            "ik_result_sha256",
            "robot_model",
            "urdf_sha256",
            "snapshot",
            "clearance_snapshot",
            "dex3_preparation_request",
            "dex3_preparation_plan",
            "joint_position_offsets_rad",
            "dex3_command_positions_rad",
            "dex3_model_positions_rad",
            "anchor_candidate_ids_by_arm",
            "parameter_names",
            "selection",
            "schedule",
            "waypoint_joint_positions_rad",
            "design_provenance",
            "random_seed",
            "content_sha256",
        }
        if set(data) != expected:
            raise ValueError("bilateral route-request fields differ from schema version 1")
        result = cls(**{key: value for key, value in data.items() if key != "content_sha256"})
        if result.content_sha256 != data["content_sha256"]:
            raise ValueError("bilateral route-request content SHA-256 mismatch")
        return result

    @classmethod
    def from_json(cls, path: str | Path) -> BilateralRoutePlanningRequest:
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))

    def write_json(self, path: str | Path) -> None:
        atomic_write_json(path, self.to_dict())


@dataclass(frozen=True, slots=True)
class BilateralRoutePlanningResult:
    request_sha256: str
    transitions: tuple[BilateralPlannedTransition, ...]
    disconnected_candidate_ids: tuple[str, ...]
    finger_sweep_sample_count: int
    restoration_sweep_sample_count: int
    planner_provenance: dict[str, Any]
    schema_version: int = 2

    def __post_init__(self) -> None:
        if self.schema_version != 2:
            raise ValueError("unsupported bilateral route-result schema version")
        if not _SHA256_PATTERN.fullmatch(self.request_sha256):
            raise ValueError("bilateral route request hash must be lowercase SHA-256")
        transitions = tuple(
            item
            if isinstance(item, BilateralPlannedTransition)
            else BilateralPlannedTransition.from_dict(item)
            for item in self.transitions
        )
        disconnected = tuple(str(value) for value in self.disconnected_candidate_ids)
        if len(disconnected) != len(set(disconnected)):
            raise ValueError("bilateral disconnected candidate IDs must be unique")
        if bool(transitions) == bool(disconnected):
            raise ValueError(
                "bilateral route result must contain either transitions or disconnections"
            )
        if transitions and self.finger_sweep_sample_count < 2:
            raise ValueError("connected bilateral route requires a certified closing sweep")
        if transitions and self.restoration_sweep_sample_count < 2:
            raise ValueError(
                "connected bilateral route requires a certified Ready-hand restoration sweep"
            )
        if disconnected and (
            self.finger_sweep_sample_count != 0 or self.restoration_sweep_sample_count != 0
        ):
            raise ValueError("disconnected bilateral route cannot certify finger sweeps")
        object.__setattr__(self, "transitions", transitions)
        object.__setattr__(self, "disconnected_candidate_ids", disconnected)
        object.__setattr__(
            self,
            "planner_provenance",
            _canonical_mapping(self.planner_provenance, name="route planner provenance"),
        )

    @property
    def connected(self) -> bool:
        return not self.disconnected_candidate_ids

    @property
    def content_sha256(self) -> str:
        return _content_sha256(self.to_dict(include_hash=False))

    def validate_request(self, request: BilateralRoutePlanningRequest) -> None:
        if self.request_sha256 != request.content_sha256:
            raise ValueError("bilateral route result belongs to a different request")
        rejectable_ids = {
            *(item.candidate_id for item in request.selection.candidates),
            *request.anchor_candidate_ids_by_arm.values(),
            *(
                item.candidate_id
                for item in request.schedule
                if item.capture_role == "preparation"
            ),
        }
        if not set(self.disconnected_candidate_ids).issubset(rejectable_ids):
            raise ValueError("bilateral route result rejected an unknown candidate")
        if self.connected:
            expected = tuple(
                f"{request.schedule[index].occurrence_id}->"
                f"{request.schedule[index + 1].occurrence_id}"
                for index in range(len(request.schedule) - 1)
            )
            if tuple(item.transition_id for item in self.transitions) != expected:
                raise ValueError("bilateral route result does not follow its request")
            if any(
                item.arm != request.transition_arm(index)
                for index, item in enumerate(self.transitions)
            ):
                raise ValueError("bilateral route result changes the wrong arm")

    def to_dict(self, *, include_hash: bool = True) -> dict[str, Any]:
        result = {
            "schema_version": self.schema_version,
            "request_sha256": self.request_sha256,
            "transitions": [item.to_dict() for item in self.transitions],
            "disconnected_candidate_ids": list(self.disconnected_candidate_ids),
            "finger_sweep_sample_count": self.finger_sweep_sample_count,
            "restoration_sweep_sample_count": self.restoration_sweep_sample_count,
            "planner_provenance": self.planner_provenance,
        }
        if include_hash:
            result["content_sha256"] = self.content_sha256
        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> BilateralRoutePlanningResult:
        expected = {
            "schema_version",
            "request_sha256",
            "transitions",
            "disconnected_candidate_ids",
            "finger_sweep_sample_count",
            "restoration_sweep_sample_count",
            "planner_provenance",
            "content_sha256",
        }
        if set(data) != expected:
            raise ValueError("bilateral route-result fields differ from schema version 1")
        result = cls(**{key: value for key, value in data.items() if key != "content_sha256"})
        if result.content_sha256 != data["content_sha256"]:
            raise ValueError("bilateral route-result content SHA-256 mismatch")
        return result

    @classmethod
    def from_json(cls, path: str | Path) -> BilateralRoutePlanningResult:
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))

    def write_json(self, path: str | Path) -> None:
        atomic_write_json(path, self.to_dict())


def _select_bilateral_anchor(
    request: BilateralCalibrationPlanningRequest,
    ik_result: BilateralIKResult,
    projection: BilateralCalibrationProjection,
    parameters: dict[str, float],
    *,
    excluded_candidate_ids: set[str],
) -> tuple[dict[str, str], np.ndarray, dict[str, Any]]:
    """Choose one both-visible anchor without reprojecting every arm pair.

    With the camera and body fixed, each marker projection depends only on its
    own arm.  Project every feasible arm pose once, then combine the retained
    centroids algebraically.  The previous Cartesian pair loop repeated full
    FK/projection tens of thousands of times for no additional information.
    """

    clearance_q = np.asarray(request.clearance_snapshot.measured_q29_rad, dtype=np.float64)
    by_side: dict[str, list[tuple[float, BilateralFeasiblePose, np.ndarray]]] = {
        "left": [],
        "right": [],
    }
    visibility_rejections: list[dict[str, str]] = []
    for pose in ik_result.poses:
        if pose.candidate_id in excluded_candidate_ids:
            continue
        indices = np.asarray(arm_indices(pose.active_arm), dtype=np.int64)
        distance = float(
            np.linalg.norm(np.asarray(pose.active_command_q_rad) - clearance_q[indices])
        )
        full_q = clearance_q.copy()
        full_q[indices] = np.asarray(pose.active_command_q_rad, dtype=np.float64)
        try:
            _sample, projected = _predicted_sample(
                request,
                projection,
                parameters,
                candidate_id=pose.candidate_id,
                full_q=full_q,
                capture_role="anchor",
            )
            centroid = _single_target_visibility_centroid(
                request,
                side=pose.active_arm,
                projected=projected[pose.active_arm],
            )
        except ValueError as error:
            visibility_rejections.append({"candidate_id": pose.candidate_id, "reason": str(error)})
            continue
        by_side[pose.active_arm].append((distance, pose, centroid))
    for side in _SIDES:
        by_side[side].sort(key=lambda item: (item[0], item[1].candidate_id))
        if not by_side[side]:
            raise ValueError(f"no {side} CuRobo-feasible candidate remains for the anchor")

    candidates: list[tuple[float, float, str, str, np.ndarray]] = []
    separation_rejections = 0
    for left_distance, left_pose, left_centroid in by_side["left"]:
        for right_distance, right_pose, right_centroid in by_side["right"]:
            target_separation_px = float(np.linalg.norm(left_centroid - right_centroid))
            if target_separation_px < request.visibility_config.minimum_target_separation_px:
                separation_rejections += 1
                continue
            full_q = clearance_q.copy()
            full_q[np.asarray(arm_indices("left"), dtype=np.int64)] = np.asarray(
                left_pose.active_command_q_rad, dtype=np.float64
            )
            full_q[np.asarray(arm_indices("right"), dtype=np.int64)] = np.asarray(
                right_pose.active_command_q_rad, dtype=np.float64
            )
            clearance_motion = max(left_distance, right_distance) + 0.1 * (
                left_distance + right_distance
            )
            candidates.append(
                (
                    clearance_motion,
                    -target_separation_px,
                    left_pose.candidate_id,
                    right_pose.candidate_id,
                    full_q.copy(),
                )
            )
    if not candidates:
        raise ValueError("no CuRobo-feasible left/right candidate pair keeps both targets visible")
    clearance_motion, negative_separation, left_id, right_id, anchor_q = min(
        candidates,
        key=lambda item: (item[0], item[1], item[2], item[3]),
    )
    pair_id = f"{left_id}+{right_id}"
    _sample, projected = _predicted_sample(
        request,
        projection,
        parameters,
        candidate_id=pair_id,
        full_q=anchor_q,
        capture_role="anchor",
    )
    _visibility_bins(request, projected)
    return (
        {"left": left_id, "right": right_id},
        anchor_q,
        {
            "policy": (
                "single_projection_per_arm_then_minimum_clearance_motion_among_"
                "same_frame_visible_pairs_with_separation_as_tiebreaker"
            ),
            "candidate_pool_size_by_arm": {side: len(by_side[side]) for side in _SIDES},
            "evaluated_pair_count": len(by_side["left"]) * len(by_side["right"]),
            "visible_pair_count": len(candidates),
            "selected_target_separation_px": -negative_separation,
            "selected_clearance_motion_score": clearance_motion,
            "visibility_rejections": visibility_rejections,
            "separation_rejection_count": separation_rejections,
        },
    )


@dataclass(frozen=True, slots=True)
class BilateralDesignPool:
    """Same-frame-visible design candidates around one immutable anchor."""

    parameter_names: tuple[str, ...]
    nominal_parameter_values: dict[str, float]
    anchor_candidate_ids_by_arm: dict[str, str]
    anchor_q29_rad: tuple[float, ...]
    candidates: tuple[BilateralDesignCandidate, ...]
    full_q29_rad_by_candidate_id: dict[str, tuple[float, ...]]
    visibility_rejections: tuple[dict[str, str], ...]
    anchor_search: dict[str, Any]

    def __post_init__(self) -> None:
        names = tuple(str(value) for value in self.parameter_names)
        if not names or len(names) != len(set(names)):
            raise ValueError("bilateral design-pool parameters must be unique and non-empty")
        anchors = {
            str(side): str(candidate_id)
            for side, candidate_id in self.anchor_candidate_ids_by_arm.items()
        }
        if set(anchors) != set(_SIDES) or len(set(anchors.values())) != 2:
            raise ValueError("bilateral design pool must bind distinct arm anchor candidates")
        candidates = tuple(self.candidates)
        candidate_ids = {item.candidate_id for item in candidates}
        if len(candidate_ids) != len(candidates):
            raise ValueError("bilateral design-pool candidate IDs must be unique")
        poses = {
            str(candidate_id): tuple(validate_full_joint_vector(q))
            for candidate_id, q in self.full_q29_rad_by_candidate_id.items()
        }
        required = {
            "ready",
            "right_shoulder_clearance",
            "dual_shoulder_clearance",
            "right_anchor_preparation",
            "bilateral_anchor",
            *candidate_ids,
        }
        if set(poses) != required:
            raise ValueError("bilateral design pool does not define every candidate pose")
        anchor = tuple(validate_full_joint_vector(self.anchor_q29_rad))
        if poses["bilateral_anchor"] != anchor:
            raise ValueError("bilateral design-pool anchor pose is inconsistent")
        object.__setattr__(self, "parameter_names", names)
        object.__setattr__(self, "anchor_candidate_ids_by_arm", anchors)
        object.__setattr__(self, "anchor_q29_rad", anchor)
        object.__setattr__(self, "candidates", candidates)
        object.__setattr__(self, "full_q29_rad_by_candidate_id", poses)
        object.__setattr__(
            self,
            "nominal_parameter_values",
            {str(name): float(value) for name, value in self.nominal_parameter_values.items()},
        )
        object.__setattr__(
            self,
            "visibility_rejections",
            tuple(dict(value) for value in self.visibility_rejections),
        )
        object.__setattr__(
            self,
            "anchor_search",
            _canonical_mapping(self.anchor_search, name="bilateral anchor search"),
        )


def build_bilateral_design_pool(
    request: BilateralCalibrationPlanningRequest,
    ik_result: BilateralIKResult,
    urdf_model: URDFModel,
    *,
    anchor_candidate_ids_by_arm: dict[str, tuple[str, ...]] | None = None,
) -> BilateralDesignPool:
    """Linearize every same-frame-visible pose around one fixed anchor."""

    ik_result.validate_request(request)
    if urdf_model.name != request.robot_model or urdf_model.sha256 != request.urdf_sha256:
        raise ValueError("bilateral planning request belongs to a different projection URDF")
    projection = BilateralCalibrationProjection(
        urdf_model,
        camera_frames=request.camera_frames,
        model=request.design_model,
        initial_hand_T_targets={
            side: np.asarray(request.nominal_hand_T_targets[side], dtype=np.float64)
            for side in _SIDES
        },
    )
    parameters = projection.parameters_for_nominal_model(
        torso_T_camera=np.asarray(request.nominal_torso_T_camera, dtype=np.float64),
        hand_T_targets={
            side: np.asarray(request.nominal_hand_T_targets[side], dtype=np.float64)
            for side in _SIDES
        },
        joint_position_offsets_rad=request.joint_position_offsets_rad,
    )
    excluded_anchor_ids: set[str] = set()
    if anchor_candidate_ids_by_arm is not None:
        if set(anchor_candidate_ids_by_arm) != set(_SIDES):
            raise ValueError("clearance-certified anchor candidates must define both arms")
        allowed = {
            side: {str(value) for value in anchor_candidate_ids_by_arm[side]} for side in _SIDES
        }
        if any(not allowed[side] for side in _SIDES):
            raise ValueError("no clearance-certified anchor candidate remains for one arm")
        excluded_anchor_ids = {
            pose.candidate_id
            for pose in ik_result.poses
            if pose.candidate_id not in allowed[pose.active_arm]
        }
    anchor_sources, anchor_q, anchor_search = _select_bilateral_anchor(
        request,
        ik_result,
        projection,
        parameters,
        excluded_candidate_ids=excluded_anchor_ids,
    )
    limits_by_arm = {side: urdf_model.joint_limits(arm_joint_names(side)) for side in _SIDES}
    ready_q = np.asarray(request.snapshot.measured_q29_rad, dtype=np.float64)
    right_clearance_q = ready_q.copy()
    right_clearance_q[np.asarray(arm_indices("right"), dtype=np.int64)] = np.asarray(
        request.dex3_preparation_plan.right_outbound.command_q_rad[-1],
        dtype=np.float64,
    )
    dual_clearance_q = np.asarray(request.clearance_snapshot.measured_q29_rad, dtype=np.float64)
    right_anchor_q = dual_clearance_q.copy()
    right_anchor_q[np.asarray(arm_indices("right"), dtype=np.int64)] = anchor_q[
        np.asarray(arm_indices("right"), dtype=np.int64)
    ]
    full_q_by_id: dict[str, tuple[float, ...]] = {
        "ready": tuple(ready_q),
        "right_shoulder_clearance": tuple(right_clearance_q),
        "dual_shoulder_clearance": tuple(dual_clearance_q),
        "right_anchor_preparation": tuple(right_anchor_q),
        "bilateral_anchor": tuple(anchor_q),
    }
    design_candidates: list[BilateralDesignCandidate] = []
    visibility_rejections: list[dict[str, str]] = []
    for pose in ik_result.poses:
        if pose.candidate_id in anchor_sources.values():
            continue
        paired_q = np.asarray(anchor_q, dtype=np.float64).copy()
        paired_q[np.asarray(arm_indices(pose.active_arm), dtype=np.int64)] = np.asarray(
            pose.active_command_q_rad,
            dtype=np.float64,
        )
        try:
            predicted, projected = _predicted_sample(
                request,
                projection,
                parameters,
                candidate_id=pose.candidate_id,
                full_q=paired_q,
                capture_role="excitation",
            )
            coverage = _visibility_bins(request, projected)
        except ValueError as error:
            visibility_rejections.append({"candidate_id": pose.candidate_id, "reason": str(error)})
            continue
        q = np.asarray(pose.active_command_q_rad, dtype=np.float64)
        lower = np.asarray([item.lower for item in limits_by_arm[pose.active_arm]])
        upper = np.asarray([item.upper for item in limits_by_arm[pose.active_arm]])
        normalized_q = 2.0 * (q - lower) / (upper - lower) - 1.0
        if np.any(normalized_q < -1.0 - 1e-6) or np.any(normalized_q > 1.0 + 1e-6):
            visibility_rejections.append(
                {
                    "candidate_id": pose.candidate_id,
                    "reason": "command lies outside the projection URDF joint limits",
                }
            )
            continue
        design_candidates.append(
            linearize_design_candidate(
                candidate_id=pose.candidate_id,
                active_arm=pose.active_arm,
                normalized_active_q=tuple(float(value) for value in np.clip(normalized_q, -1, 1)),
                predicted_sample=predicted,
                projection=projection,
                parameters=parameters,
                image_coverage_bins=coverage,
            )
        )
        full_q_by_id[pose.candidate_id] = tuple(paired_q)
    return BilateralDesignPool(
        parameter_names=projection.parameter_names,
        nominal_parameter_values=dict(sorted(parameters.items())),
        anchor_candidate_ids_by_arm=anchor_sources,
        anchor_q29_rad=tuple(anchor_q),
        candidates=tuple(design_candidates),
        full_q29_rad_by_candidate_id=full_q_by_id,
        visibility_rejections=tuple(visibility_rejections),
        anchor_search=anchor_search,
    )


def select_connected_bilateral_design(
    request: BilateralCalibrationPlanningRequest,
    pool: BilateralDesignPool,
    *,
    connected_candidate_ids_by_arm: dict[str, tuple[str, ...]],
) -> BilateralDesignSelection:
    """Select the statistical design from the rooted feasible component."""

    if set(connected_candidate_ids_by_arm) != set(_SIDES):
        raise ValueError("bilateral connectivity must define both arms")
    candidate_by_id = {item.candidate_id: item for item in pool.candidates}
    connected = {
        side: tuple(str(value) for value in connected_candidate_ids_by_arm[side])
        for side in _SIDES
    }
    for side in _SIDES:
        if len(connected[side]) != len(set(connected[side])):
            raise ValueError(f"bilateral {side} connected candidates must be unique")
        if any(
            candidate_id not in candidate_by_id or candidate_by_id[candidate_id].active_arm != side
            for candidate_id in connected[side]
        ):
            raise ValueError(f"bilateral {side} connectivity contains an invalid candidate")
    eligible_ids = {candidate_id for values in connected.values() for candidate_id in values}
    return select_bilateral_design(
        tuple(
            candidate for candidate in pool.candidates if candidate.candidate_id in eligible_ids
        ),
        parameter_names=pool.parameter_names,
        config=request.design_config,
    )


def build_connected_bilateral_route_request(
    request: BilateralCalibrationPlanningRequest,
    ik_result: BilateralIKResult,
    pool: BilateralDesignPool,
    *,
    connected_candidate_ids_by_arm: dict[str, tuple[str, ...]],
    valid_edges_by_arm: dict[str, tuple[tuple[float, str, str], ...]],
    connectivity_provenance: dict[str, Any],
    selection: BilateralDesignSelection | None = None,
) -> BilateralRoutePlanningRequest:
    """Select anchor-connected poses and freeze short valid-graph tours."""

    ik_result.validate_request(request)
    if set(connected_candidate_ids_by_arm) != set(_SIDES) or set(valid_edges_by_arm) != set(
        _SIDES
    ):
        raise ValueError("bilateral connectivity must define both arms")
    connected = {
        side: tuple(str(value) for value in connected_candidate_ids_by_arm[side])
        for side in _SIDES
    }
    for side in _SIDES:
        if len(connected[side]) != len(set(connected[side])):
            raise ValueError(f"bilateral {side} connected candidates must be unique")
    selected = selection or select_connected_bilateral_design(
        request,
        pool,
        connected_candidate_ids_by_arm=connected,
    )
    schedule = build_valid_graph_route_schedule(
        selected,
        anchor_candidate_id="bilateral_anchor",
        anchor_interval=request.design_config.anchor_interval,
        valid_edges_by_arm=valid_edges_by_arm,
    )
    scheduled_ids = {item.candidate_id for item in schedule}
    positions = {
        candidate_id: q
        for candidate_id, q in pool.full_q29_rad_by_candidate_id.items()
        if candidate_id in scheduled_ids
    }
    return BilateralRoutePlanningRequest(
        planning_request_sha256=request.content_sha256,
        ik_result_sha256=ik_result.content_sha256,
        robot_model=request.robot_model,
        urdf_sha256=request.urdf_sha256,
        snapshot=request.snapshot,
        clearance_snapshot=request.clearance_snapshot,
        dex3_preparation_request=request.dex3_preparation_request,
        dex3_preparation_plan=request.dex3_preparation_plan,
        joint_position_offsets_rad=request.joint_position_offsets_rad,
        dex3_command_positions_rad=request.dex3_command_positions_rad,
        dex3_model_positions_rad=request.dex3_model_positions_rad,
        anchor_candidate_ids_by_arm=pool.anchor_candidate_ids_by_arm,
        parameter_names=pool.parameter_names,
        selection=selected,
        schedule=schedule,
        waypoint_joint_positions_rad=positions,
        design_provenance={
            "ik_feasible_count": len(ik_result.poses),
            "same_frame_visible_candidate_count": len(pool.candidates),
            "same_frame_visibility_rejections": list(pool.visibility_rejections),
            "bilateral_anchor_search": pool.anchor_search,
            "bilateral_anchor_candidate_ids_by_arm": pool.anchor_candidate_ids_by_arm,
            "anchor_connected_candidate_count_by_arm": {
                side: len(connected[side]) for side in _SIDES
            },
            "selected_valid_edges_by_arm": valid_edges_by_arm,
            "nominal_parameter_values": pool.nominal_parameter_values,
            "route_order_policy": (
                "minimum_motion_cost_anchor_tours_over_selected_valid_edge_graphs"
            ),
            "connectivity": connectivity_provenance,
        },
        random_seed=request.random_seed,
    )


def assemble_bilateral_planning_artifacts(
    request: BilateralCalibrationPlanningRequest,
    ik_result: BilateralIKResult,
    route_request: BilateralRoutePlanningRequest,
    route_result: BilateralRoutePlanningResult,
) -> tuple[BilateralPoseDesignArtifact, BilateralExecutionPlan]:
    """Bind one fully connected CuRobo result into the two hardware artifacts."""

    ik_result.validate_request(request)
    if route_request.planning_request_sha256 != request.content_sha256:
        raise ValueError("bilateral route request belongs to a different planning request")
    if route_request.ik_result_sha256 != ik_result.content_sha256:
        raise ValueError("bilateral route request belongs to a different IK result")
    route_result.validate_request(route_request)
    if not route_result.connected:
        raise ValueError("cannot assemble a disconnected bilateral route")
    anchor_indices = [
        index
        for index, waypoint in enumerate(route_request.schedule)
        if waypoint.capture_role == "anchor"
    ]
    provenance = {
        **request.source_provenance,
        "planning_request_sha256": request.content_sha256,
        "ik_result_sha256": ik_result.content_sha256,
        "route_request_sha256": route_request.content_sha256,
        "route_result_sha256": route_result.content_sha256,
        "ik": ik_result.planner_provenance,
        "route": route_result.planner_provenance,
        "design": route_request.design_provenance,
        "graceful_return": {
            "policy": "latch_at_any_capture_then_stop_at_next_identical_anchor",
            "anchor_occurrence_ids": [
                route_request.schedule[index].occurrence_id for index in anchor_indices
            ],
            "terminal_boundary": HANDOFF_POSE_ID,
        },
    }
    design = BilateralPoseDesignArtifact(
        model_sha256=request.design_model.content_sha256,
        parameter_names=route_request.parameter_names,
        selection=route_request.selection,
        schedule=route_request.schedule,
        waypoint_joint_positions_rad=route_request.waypoint_joint_positions_rad,
        route_validation_sha256_by_transition={
            item.transition_id: item.content_sha256 for item in route_result.transitions
        },
        planner_provenance=provenance,
    )
    execution = BilateralExecutionPlan(
        pose_design_sha256=design.content_sha256,
        robot_model=request.robot_model,
        urdf_sha256=request.urdf_sha256,
        joint_position_offsets_rad=request.joint_position_offsets_rad,
        commanded_dex3_joint_positions_rad=request.dex3_command_positions_rad,
        modeled_dex3_joint_positions_rad=request.dex3_model_positions_rad,
        self_clearance_certificate=dict(
            route_result.planner_provenance["self_clearance_certificate"]
        ),
        transitions=route_result.transitions,
        planner_provenance=provenance,
    )
    execution.validate_design(design)
    return design, execution


def _predicted_sample(
    request: BilateralCalibrationPlanningRequest,
    projection: BilateralCalibrationProjection,
    parameters: dict[str, float],
    *,
    candidate_id: str,
    full_q,
    capture_role: Literal["anchor", "excitation"],
) -> tuple[BilateralCalibrationSample, dict[str, tuple[np.ndarray, np.ndarray]]]:
    observations = {
        side: _target_observation(
            request,
            side=side,
            image_points=np.zeros(
                (len(request.target_object_points_m_by_arm[side]), 2),
                dtype=np.float64,
            ),
        )
        for side in _SIDES
    }
    sample = BilateralCalibrationSample(
        source_session_id="offline_bilateral_design",
        capture_id=candidate_id,
        pose_group_id=candidate_id,
        day_group_id="offline",
        frame_id=candidate_id,
        capture_role=capture_role,
        raw_image_path="offline/predicted.png",
        raw_image_sha256="0" * 64,
        camera_info=request.camera_info,
        joint_positions_rad=tuple(full_q),
        joint_velocities_rad_s=(0.0,) * len(G1_29_JOINT_NAMES),
        pairing={"source": "nominal_projection"},
        left=observations["left"],
        right=observations["right"],
    )
    projected = {
        side: projection.project_side(sample, side=side, parameters=parameters) for side in _SIDES
    }
    sample = BilateralCalibrationSample(
        source_session_id=sample.source_session_id,
        capture_id=sample.capture_id,
        pose_group_id=sample.pose_group_id,
        day_group_id=sample.day_group_id,
        frame_id=sample.frame_id,
        capture_role=sample.capture_role,
        raw_image_path=sample.raw_image_path,
        raw_image_sha256=sample.raw_image_sha256,
        camera_info=sample.camera_info,
        joint_positions_rad=sample.joint_positions_rad,
        joint_velocities_rad_s=sample.joint_velocities_rad_s,
        pairing=sample.pairing,
        left=_target_observation(request, side="left", image_points=projected["left"][0]),
        right=_target_observation(request, side="right", image_points=projected["right"][0]),
    )
    return sample, projected


def _target_observation(
    request: BilateralCalibrationPlanningRequest,
    *,
    side: str,
    image_points: np.ndarray,
) -> TargetObservation:
    corner_ids = request.target_corner_tag_ids_by_arm[side]
    return TargetObservation(
        side=side,
        target_artifact_sha256=request.target_artifact_sha256_by_arm[side],
        visible_tag_ids=tuple(sorted(set(corner_ids))),
        corner_tag_ids=corner_ids,
        image_points_px=tuple(tuple(float(value) for value in row) for row in image_points),
        object_points_m=request.target_object_points_m_by_arm[side],
        correspondence_sha256="0" * 64,
    )


def _visibility_bins(
    request: BilateralCalibrationPlanningRequest,
    projected: dict[str, tuple[np.ndarray, np.ndarray]],
) -> tuple[int, ...]:
    camera = RectifiedCameraInfo.from_dict(request.camera_info)
    config = request.visibility_config
    bins: list[int] = []
    centroids: dict[str, np.ndarray] = {}
    for side in _SIDES:
        pixels, depths = projected[side]
        pixels = np.asarray(pixels, dtype=np.float64)
        depths = np.asarray(depths, dtype=np.float64)
        if np.any(depths < config.minimum_depth_m) or np.any(depths > config.maximum_depth_m):
            raise ValueError(f"{side} target depth is outside the bilateral visibility range")
        if (
            np.min(pixels[:, 0]) < config.image_margin_px
            or np.max(pixels[:, 0]) > camera.width - config.image_margin_px
            or np.min(pixels[:, 1]) < config.image_margin_px
            or np.max(pixels[:, 1]) > camera.height - config.image_margin_px
        ):
            raise ValueError(f"{side} target leaves the bilateral image margin")
        for lower in range(0, len(pixels), 4):
            corners = pixels[lower : lower + 4]
            spans = np.linalg.norm(corners - np.roll(corners, -1, axis=0), axis=1)
            if float(np.min(spans)) < config.minimum_target_span_px:
                raise ValueError(f"{side} target span is below the bilateral visibility limit")
        centroid = np.mean(pixels, axis=0)
        centroids[side] = centroid
        bins.extend(
            (
                min(
                    int(centroid[0] / camera.width * config.image_grid_columns),
                    config.image_grid_columns - 1,
                ),
                min(
                    int(centroid[1] / camera.height * config.image_grid_rows),
                    config.image_grid_rows - 1,
                ),
                min(
                    int(
                        (float(np.mean(depths)) - config.minimum_depth_m)
                        / (config.maximum_depth_m - config.minimum_depth_m)
                        * config.depth_bins
                    ),
                    config.depth_bins - 1,
                ),
            )
        )
    separation = float(np.linalg.norm(centroids["left"] - centroids["right"]))
    if separation < config.minimum_target_separation_px:
        raise ValueError(
            f"predicted hand targets are separated by only {separation:.1f}px; "
            f"limit is {config.minimum_target_separation_px:.1f}px"
        )
    return tuple(bins)


def _single_target_visibility_centroid(
    request: BilateralCalibrationPlanningRequest,
    *,
    side: str,
    projected: tuple[np.ndarray, np.ndarray],
) -> np.ndarray:
    """Validate one target and return its image centroid."""

    if side not in _SIDES:
        raise ValueError("bilateral target side is invalid")
    camera = RectifiedCameraInfo.from_dict(request.camera_info)
    config = request.visibility_config
    pixels = np.asarray(projected[0], dtype=np.float64)
    depths = np.asarray(projected[1], dtype=np.float64)
    if np.any(depths < config.minimum_depth_m) or np.any(depths > config.maximum_depth_m):
        raise ValueError(f"{side} target depth is outside the bilateral visibility range")
    if (
        np.min(pixels[:, 0]) < config.image_margin_px
        or np.max(pixels[:, 0]) > camera.width - config.image_margin_px
        or np.min(pixels[:, 1]) < config.image_margin_px
        or np.max(pixels[:, 1]) > camera.height - config.image_margin_px
    ):
        raise ValueError(f"{side} target leaves the bilateral image margin")
    for lower in range(0, len(pixels), 4):
        corners = pixels[lower : lower + 4]
        spans = np.linalg.norm(corners - np.roll(corners, -1, axis=0), axis=1)
        if float(np.min(spans)) < config.minimum_target_span_px:
            raise ValueError(f"{side} target span is below the bilateral visibility limit")
    return np.mean(pixels, axis=0)
