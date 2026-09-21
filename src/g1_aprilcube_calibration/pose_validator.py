"""Offline endpoint, interpolation, joint-limit, and collision validation."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from g1_aprilcube_calibration.collision import FCLCollisionChecker
from g1_aprilcube_calibration.joint_map import (
    G1_29_JOINT_NAMES,
    arm_indices,
    opposite_arm,
    validate_full_joint_vector,
)
from g1_aprilcube_calibration.pose_schema import HANDOFF_POSE_ID, PoseSet
from g1_aprilcube_calibration.urdf_model import URDFModel

VALIDATION_SCHEMA_VERSION = 1
VALIDATOR_VERSION = "g1-path-validator-v1"


@dataclass(frozen=True, slots=True)
class PathValidationConfig:
    maximum_joint_increment_rad: float = 0.02
    joint_limit_margin_rad: float = 0.03
    minimum_collision_clearance_m: float = 0.005
    maximum_path_length_rad: float = 5.0
    assumed_joint_velocity_rad_s: float = 0.2

    def __post_init__(self) -> None:
        for name in self.__dataclass_fields__:
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")


@dataclass(frozen=True, slots=True)
class DirectedEdgeResult:
    from_pose_id: str
    to_pose_id: str
    passed: bool
    sample_count: int
    path_length_rad: float
    estimated_duration_s: float
    minimum_clearance_m: float | None
    minimum_clearance_pair: tuple[str, str] | None
    failures: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.from_pose_id or not self.to_pose_id:
            raise ValueError("validation edge pose IDs must be non-empty")
        if self.sample_count < 2:
            raise ValueError("validation edge must contain at least two samples")
        if not isinstance(self.passed, bool):
            raise TypeError("validation edge passed field must be boolean")
        for value in (self.path_length_rad, self.estimated_duration_s):
            if not np.isfinite(value) or value < 0:
                raise ValueError("validation edge metrics must be finite and non-negative")
        if self.minimum_clearance_m is not None and not np.isfinite(self.minimum_clearance_m):
            raise ValueError("minimum clearance must be finite")
        if self.passed and self.failures:
            raise ValueError("passed edge cannot contain failures")
        if not self.passed and not self.failures:
            raise ValueError("failed edge must explain at least one failure")

    def to_dict(self) -> dict:
        return {
            "from_pose_id": self.from_pose_id,
            "to_pose_id": self.to_pose_id,
            "passed": self.passed,
            "sample_count": self.sample_count,
            "path_length_rad": self.path_length_rad,
            "estimated_duration_s": self.estimated_duration_s,
            "minimum_clearance_m": self.minimum_clearance_m,
            "minimum_clearance_pair": (
                None if self.minimum_clearance_pair is None else list(self.minimum_clearance_pair)
            ),
            "failures": list(self.failures),
        }

    @classmethod
    def from_dict(cls, data: dict) -> DirectedEdgeResult:
        pair = data["minimum_clearance_pair"]
        return cls(
            from_pose_id=data["from_pose_id"],
            to_pose_id=data["to_pose_id"],
            passed=data["passed"],
            sample_count=int(data["sample_count"]),
            path_length_rad=float(data["path_length_rad"]),
            estimated_duration_s=float(data["estimated_duration_s"]),
            minimum_clearance_m=(
                None if data["minimum_clearance_m"] is None else float(data["minimum_clearance_m"])
            ),
            minimum_clearance_pair=None if pair is None else tuple(pair),
            failures=tuple(data["failures"]),
        )


@dataclass(frozen=True, slots=True)
class ValidationReport:
    pose_set_sha256: str
    urdf_sha256: str
    collision_config_sha256: str
    reference_full_q_sha256: str
    config: dict
    edges: tuple[DirectedEdgeResult, ...]
    schema_version: int = VALIDATION_SCHEMA_VERSION
    validator_version: str = VALIDATOR_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != VALIDATION_SCHEMA_VERSION:
            raise ValueError("unsupported validation report schema")
        if self.validator_version != VALIDATOR_VERSION:
            raise ValueError("unsupported path validator version")
        for value in (
            self.pose_set_sha256,
            self.urdf_sha256,
            self.collision_config_sha256,
            self.reference_full_q_sha256,
        ):
            if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
                raise ValueError("validation report hashes must be lowercase SHA-256")
        keys = [(edge.from_pose_id, edge.to_pose_id) for edge in self.edges]
        if len(keys) != len(set(keys)):
            raise ValueError("validation report contains duplicate directed edges")
        object.__setattr__(
            self,
            "config",
            json.loads(json.dumps(self.config, allow_nan=False, sort_keys=True)),
        )
        object.__setattr__(self, "edges", tuple(self.edges))

    @property
    def passed(self) -> bool:
        return bool(self.edges) and all(edge.passed for edge in self.edges)

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

    def to_dict(self, *, include_hash: bool = True) -> dict:
        result = {
            "schema_version": self.schema_version,
            "validator_version": self.validator_version,
            "passed": self.passed,
            "pose_set_sha256": self.pose_set_sha256,
            "urdf_sha256": self.urdf_sha256,
            "collision_config_sha256": self.collision_config_sha256,
            "reference_full_q_sha256": self.reference_full_q_sha256,
            "config": self.config,
            "edges": [edge.to_dict() for edge in self.edges],
        }
        if include_hash:
            result["content_sha256"] = self.content_sha256
        return result

    def write_json(self, path: str | Path) -> None:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        data = json.dumps(self.to_dict(), indent=2, sort_keys=True).encode() + b"\n"
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{output.name}.", suffix=".tmp", dir=output.parent
        )
        temporary = Path(temporary_name)
        try:
            with os.fdopen(descriptor, "wb") as stream:
                stream.write(data)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, output)
        finally:
            temporary.unlink(missing_ok=True)

    def edge(self, from_pose_id: str, to_pose_id: str) -> DirectedEdgeResult:
        matches = [
            edge
            for edge in self.edges
            if edge.from_pose_id == from_pose_id and edge.to_pose_id == to_pose_id
        ]
        if len(matches) != 1:
            raise ValueError("validation report does not contain exactly one directed edge")
        return matches[0]

    def approval(self, from_pose_id: str, to_pose_id: str):
        from g1_aprilcube_calibration.executor_state_machine import (
            TransitionApproval,
        )

        edge = self.edge(from_pose_id, to_pose_id)
        return TransitionApproval(
            from_pose_id=from_pose_id,
            to_pose_id=to_pose_id,
            pose_set_sha256=self.pose_set_sha256,
            validation_report_sha256=self.content_sha256,
            passed=edge.passed,
        )

    @classmethod
    def from_dict(cls, data: dict) -> ValidationReport:
        expected = {
            "schema_version",
            "validator_version",
            "passed",
            "pose_set_sha256",
            "urdf_sha256",
            "collision_config_sha256",
            "reference_full_q_sha256",
            "config",
            "edges",
            "content_sha256",
        }
        if set(data) != expected:
            raise ValueError("validation report fields do not match schema version 1")
        result = cls(
            schema_version=int(data["schema_version"]),
            validator_version=data["validator_version"],
            pose_set_sha256=data["pose_set_sha256"],
            urdf_sha256=data["urdf_sha256"],
            collision_config_sha256=data["collision_config_sha256"],
            reference_full_q_sha256=data["reference_full_q_sha256"],
            config=dict(data["config"]),
            edges=tuple(DirectedEdgeResult.from_dict(item) for item in data["edges"]),
        )
        if data["passed"] is not result.passed:
            raise ValueError("validation report aggregate pass flag is inconsistent")
        if data["content_sha256"] != result.content_sha256:
            raise ValueError("validation report content SHA-256 does not match")
        return result


class PosePathValidator:
    def __init__(
        self,
        *,
        model: URDFModel,
        collision_checker: FCLCollisionChecker,
        config: PathValidationConfig | None = None,
    ) -> None:
        self.model = model
        self.collision_checker = collision_checker
        self.config = config or PathValidationConfig()

    def validate(
        self,
        pose_set: PoseSet,
        *,
        directed_edges: Sequence[tuple[str, str]],
        reference_full_q: Sequence[float] | np.ndarray,
        progress: Callable[[int, int, str, str], None] | None = None,
        cancellation_requested: Callable[[], bool] | None = None,
    ) -> ValidationReport:
        if pose_set.urdf_sha256 != self.model.sha256:
            raise ValueError("pose set was authored against a different URDF hash")
        reference = validate_full_joint_vector(reference_full_q, name="reference_full_q")
        handoff = reference[np.asarray(arm_indices(pose_set.calibration_arm))]
        hold = reference[np.asarray(arm_indices(opposite_arm(pose_set.calibration_arm)))]
        calibration_q_by_id = {
            HANDOFF_POSE_ID: handoff,
            **{pose.id: np.asarray(pose.command_calibration_q) for pose in pose_set.poses},
        }
        edge_keys = tuple(directed_edges)
        if len(edge_keys) != len(set(edge_keys)):
            raise ValueError("directed transition list contains duplicates")
        results: list[DirectedEdgeResult] = []
        edge_count = len(edge_keys)
        for edge_index, (source_id, target_id) in enumerate(edge_keys):
            if cancellation_requested is not None and cancellation_requested():
                raise KeyboardInterrupt
            if source_id not in calibration_q_by_id or target_id not in calibration_q_by_id:
                raise ValueError(
                    f"transition references an unknown pose: {source_id}->{target_id}"
                )
            results.append(
                self._validate_edge(
                    source_id,
                    target_id,
                    calibration_q_by_id[source_id],
                    calibration_q_by_id[target_id],
                    hold,
                    pose_set.calibration_arm,
                    reference,
                    cancellation_requested=cancellation_requested,
                )
            )
            if cancellation_requested is not None and cancellation_requested():
                raise KeyboardInterrupt
            if progress is not None:
                progress(edge_index + 1, edge_count, source_id, target_id)
        reference_hash = hashlib.sha256(reference.tobytes()).hexdigest()
        return ValidationReport(
            pose_set_sha256=pose_set.content_sha256,
            urdf_sha256=self.model.sha256,
            collision_config_sha256=self.collision_checker.config.content_sha256,
            reference_full_q_sha256=reference_hash,
            config={name: getattr(self.config, name) for name in self.config.__dataclass_fields__},
            edges=tuple(results),
        )

    def _validate_edge(
        self,
        source_id: str,
        target_id: str,
        source_calibration: np.ndarray,
        target_calibration: np.ndarray,
        hold: np.ndarray,
        calibration_arm: str,
        reference: np.ndarray,
        *,
        cancellation_requested: Callable[[], bool] | None = None,
    ) -> DirectedEdgeResult:
        delta = target_calibration - source_calibration
        maximum_delta = float(np.max(np.abs(delta)))
        intervals = max(int(np.ceil(maximum_delta / self.config.maximum_joint_increment_rad)), 1)
        sample_count = intervals + 1
        path_length = float(np.linalg.norm(delta))
        estimated_duration = maximum_delta / self.config.assumed_joint_velocity_rad_s
        failures: list[str] = []
        if not self.collision_checker.config.hardware_ready:
            failures.append(
                "collision configuration is not hardware-ready: "
                + ", ".join(self.collision_checker.config.blocking_reasons)
            )
        if path_length > self.config.maximum_path_length_rad:
            failures.append(
                f"path length {path_length:.3f}rad exceeds "
                f"{self.config.maximum_path_length_rad:.3f}rad"
            )
        minimum_clearance = float("inf")
        minimum_pair: tuple[str, str] | None = None
        limits = self.model.joint_limits(G1_29_JOINT_NAMES)
        for index, alpha in enumerate(np.linspace(0.0, 1.0, sample_count)):
            if cancellation_requested is not None and cancellation_requested():
                raise KeyboardInterrupt
            calibration = source_calibration + alpha * delta
            full = reference.copy()
            full[np.asarray(arm_indices(opposite_arm(calibration_arm)))] = hold
            full[np.asarray(arm_indices(calibration_arm))] = calibration
            for joint_index, (name, value, limit) in enumerate(
                zip(G1_29_JOINT_NAMES, full, limits, strict=True)
            ):
                lower = limit.lower + self.config.joint_limit_margin_rad
                upper = limit.upper - self.config.joint_limit_margin_rad
                if value < lower or value > upper:
                    failures.append(
                        f"sample {index}: {name}={value:.4f} outside [{lower:.4f}, {upper:.4f}]"
                    )
                    break
            positions = dict(zip(G1_29_JOINT_NAMES, full, strict=True))
            transforms = self.model.forward_kinematics(positions)
            collision = self.collision_checker.check(transforms)
            if collision.minimum_clearance_m < minimum_clearance:
                minimum_clearance = collision.minimum_clearance_m
                if collision.minimum_pair is not None:
                    minimum_pair = (
                        collision.minimum_pair.first,
                        collision.minimum_pair.second,
                    )
            if (
                collision.colliding_pairs
                or collision.minimum_clearance_m < self.config.minimum_collision_clearance_m
            ):
                pair = collision.minimum_pair
                pair_text = "unknown" if pair is None else f"{pair.first}/{pair.second}"
                failures.append(
                    f"sample {index}: clearance {collision.minimum_clearance_m:.4f}m "
                    f"for {pair_text} is below "
                    f"{self.config.minimum_collision_clearance_m:.4f}m"
                )
        return DirectedEdgeResult(
            from_pose_id=source_id,
            to_pose_id=target_id,
            passed=not failures,
            sample_count=sample_count,
            path_length_rad=path_length,
            estimated_duration_s=estimated_duration,
            minimum_clearance_m=(
                None if not np.isfinite(minimum_clearance) else minimum_clearance
            ),
            minimum_clearance_pair=minimum_pair,
            failures=tuple(dict.fromkeys(failures)),
        )
