"""Full-model information selection and repeated-anchor route scheduling."""

from __future__ import annotations

import hashlib
import heapq
import json
import os
import re
import tempfile
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path
from typing import Any, Literal

import numpy as np

from g1_aprilcube_calibration.joint_map import validate_full_joint_vector
from g1_aprilcube_calibration.pose_schema import HANDOFF_POSE_ID
from g1_dex3_tabletop.calibration.models import BilateralCalibrationSample
from g1_dex3_tabletop.calibration.projection import BilateralCalibrationProjection

_SIDES = ("left", "right")
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


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


@dataclass(frozen=True, slots=True)
class BilateralDesignCandidate:
    """One collision-feasible paired configuration linearized at the nominal model."""

    candidate_id: str
    active_arm: Literal["left", "right"]
    normalized_active_q: tuple[float, ...]
    normalized_jacobian: tuple[tuple[float, ...], ...]
    image_coverage_bins: tuple[int, ...]

    def __post_init__(self) -> None:
        if not self.candidate_id.strip():
            raise ValueError("bilateral design candidate ID must be non-empty")
        if self.active_arm not in _SIDES:
            raise ValueError("bilateral design active arm must be left or right")
        q = np.asarray(self.normalized_active_q, dtype=np.float64).reshape(-1)
        jacobian = np.asarray(self.normalized_jacobian, dtype=np.float64)
        if q.shape != (7,) or not np.all(np.isfinite(q)):
            raise ValueError("normalized active-arm configuration must contain seven values")
        if np.any(np.abs(q) > 1.0 + 1e-12):
            raise ValueError("normalized active-arm configuration must lie in [-1, 1]")
        if jacobian.ndim != 2 or not all(jacobian.shape) or not np.all(np.isfinite(jacobian)):
            raise ValueError("candidate Jacobian must be a non-empty finite matrix")
        bins = tuple(int(value) for value in self.image_coverage_bins)
        if not bins or any(value < 0 for value in bins):
            raise ValueError("candidate image coverage bins must be non-negative")
        q.setflags(write=False)
        jacobian.setflags(write=False)
        object.__setattr__(self, "normalized_active_q", tuple(float(value) for value in q))
        object.__setattr__(
            self,
            "normalized_jacobian",
            tuple(tuple(float(value) for value in row) for row in jacobian),
        )
        object.__setattr__(self, "image_coverage_bins", bins)

    @property
    def jacobian(self) -> np.ndarray:
        result = np.asarray(self.normalized_jacobian, dtype=np.float64)
        result.setflags(write=False)
        return result

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "active_arm": self.active_arm,
            "normalized_active_q": list(self.normalized_active_q),
            "normalized_jacobian": [list(row) for row in self.normalized_jacobian],
            "image_coverage_bins": list(self.image_coverage_bins),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> BilateralDesignCandidate:
        expected = {
            "candidate_id",
            "active_arm",
            "normalized_active_q",
            "normalized_jacobian",
            "image_coverage_bins",
        }
        if set(data) != expected:
            raise ValueError("bilateral design candidate fields differ from schema version 1")
        return cls(
            candidate_id=data["candidate_id"],
            active_arm=data["active_arm"],
            normalized_active_q=tuple(data["normalized_active_q"]),
            normalized_jacobian=tuple(tuple(row) for row in data["normalized_jacobian"]),
            image_coverage_bins=tuple(data["image_coverage_bins"]),
        )


@dataclass(frozen=True, slots=True)
class BilateralDesignConfig:
    left_excitation_count: int = 34
    right_excitation_count: int = 34
    anchor_interval: int = 7
    information_weight: float = 0.65
    joint_excitation_weight: float = 0.25
    image_coverage_weight: float = 0.10
    information_ridge: float = 1e-6
    relative_rank_threshold: float = 1e-7
    maximum_model_condition_number: float = 1e8
    maximum_joint_condition_number: float = 1e3
    require_observable_model: bool = True
    require_full_joint_excitation: bool = True

    def __post_init__(self) -> None:
        if self.left_excitation_count < 1 or self.right_excitation_count < 1:
            raise ValueError("both arms require a positive excitation count")
        if self.anchor_interval < 1:
            raise ValueError("anchor interval must be positive")
        weights = (
            self.information_weight,
            self.joint_excitation_weight,
            self.image_coverage_weight,
        )
        if not all(np.isfinite(value) and value >= 0.0 for value in weights):
            raise ValueError("bilateral design weights must be finite and non-negative")
        if sum(weights) <= 0.0:
            raise ValueError("at least one bilateral design weight must be positive")
        if not np.isfinite(self.information_ridge) or self.information_ridge <= 0.0:
            raise ValueError("information ridge must be positive and finite")
        if not 0.0 < self.relative_rank_threshold < 1.0:
            raise ValueError("relative rank threshold must lie in (0, 1)")
        if self.maximum_model_condition_number <= 1.0:
            raise ValueError("maximum model condition number must exceed one")
        if self.maximum_joint_condition_number <= 1.0:
            raise ValueError("maximum joint condition number must exceed one")
        if not isinstance(self.require_observable_model, bool):
            raise TypeError("require_observable_model must be boolean")
        if not isinstance(self.require_full_joint_excitation, bool):
            raise TypeError("require_full_joint_excitation must be boolean")

    def to_dict(self) -> dict[str, Any]:
        return {name: getattr(self, name) for name in self.__dataclass_fields__}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> BilateralDesignConfig:
        if set(data) != set(cls.__dataclass_fields__):
            raise ValueError("bilateral design configuration fields differ from schema")
        return cls(**data)


@dataclass(frozen=True, slots=True)
class BilateralDesignStep:
    selection_index: int
    candidate_id: str
    active_arm: Literal["left", "right"]
    information_gain_logdet: float
    joint_excitation_gain_logdet: float
    image_coverage_gain: float
    combined_score: float

    def __post_init__(self) -> None:
        if self.selection_index < 1:
            raise ValueError("bilateral design selection index must be positive")
        if not self.candidate_id.strip() or self.active_arm not in _SIDES:
            raise ValueError("bilateral design step has an invalid candidate")
        scores = (
            self.information_gain_logdet,
            self.joint_excitation_gain_logdet,
            self.image_coverage_gain,
            self.combined_score,
        )
        if not np.all(np.isfinite(scores)):
            raise ValueError("bilateral design step scores must be finite")

    def to_dict(self) -> dict[str, Any]:
        return {name: getattr(self, name) for name in self.__dataclass_fields__}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> BilateralDesignStep:
        if set(data) != set(cls.__dataclass_fields__):
            raise ValueError("bilateral design step fields differ from schema version 1")
        return cls(**data)


@dataclass(frozen=True, slots=True)
class BilateralDesignReport:
    model_rank: int
    model_parameter_count: int
    model_condition_number: float | None
    left_joint_rank: int
    left_joint_condition_number: float | None
    right_joint_rank: int
    right_joint_condition_number: float | None

    def __post_init__(self) -> None:
        if not 0 <= self.model_rank <= self.model_parameter_count:
            raise ValueError("bilateral model design rank is invalid")
        if self.model_parameter_count < 1:
            raise ValueError("bilateral design requires model parameters")
        for side in _SIDES:
            rank = getattr(self, f"{side}_joint_rank")
            condition = getattr(self, f"{side}_joint_condition_number")
            if not 0 <= rank <= 7:
                raise ValueError(f"bilateral {side} joint design rank is invalid")
            if condition is not None and (not np.isfinite(condition) or condition < 1.0):
                raise ValueError(f"bilateral {side} joint condition is invalid")
        if self.model_condition_number is not None and (
            not np.isfinite(self.model_condition_number) or self.model_condition_number < 1.0
        ):
            raise ValueError("bilateral model condition is invalid")

    def to_dict(self) -> dict[str, Any]:
        return {name: getattr(self, name) for name in self.__dataclass_fields__}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> BilateralDesignReport:
        if set(data) != set(cls.__dataclass_fields__):
            raise ValueError("bilateral design report fields differ from schema version 1")
        return cls(**data)


@dataclass(frozen=True, slots=True)
class BilateralDesignSelection:
    candidates: tuple[BilateralDesignCandidate, ...]
    steps: tuple[BilateralDesignStep, ...]
    report: BilateralDesignReport

    def __post_init__(self) -> None:
        candidates = tuple(self.candidates)
        steps = tuple(self.steps)
        if not candidates or len(candidates) != len(steps):
            raise ValueError("bilateral design selection requires one step per candidate")
        if len({item.candidate_id for item in candidates}) != len(candidates):
            raise ValueError("bilateral design selection candidate IDs must be unique")
        for index, (candidate, step) in enumerate(zip(candidates, steps, strict=True), start=1):
            if (
                step.selection_index != index
                or step.candidate_id != candidate.candidate_id
                or step.active_arm != candidate.active_arm
            ):
                raise ValueError("bilateral design steps do not match selected candidates")
        if any(item.jacobian.shape[1] != self.report.model_parameter_count for item in candidates):
            raise ValueError("bilateral design candidate columns differ from the report")
        object.__setattr__(self, "candidates", candidates)
        object.__setattr__(self, "steps", steps)

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidates": [candidate.to_dict() for candidate in self.candidates],
            "steps": [step.to_dict() for step in self.steps],
            "report": self.report.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> BilateralDesignSelection:
        if set(data) != {"candidates", "steps", "report"}:
            raise ValueError("bilateral design selection fields differ from schema version 1")
        return cls(
            candidates=tuple(
                BilateralDesignCandidate.from_dict(item) for item in data["candidates"]
            ),
            steps=tuple(BilateralDesignStep.from_dict(item) for item in data["steps"]),
            report=BilateralDesignReport.from_dict(data["report"]),
        )


@dataclass(frozen=True, slots=True)
class BilateralCaptureWaypoint:
    occurrence_id: str
    candidate_id: str
    capture_role: Literal["ready", "preparation", "anchor", "excitation"]
    active_arm: Literal["left", "right"] | None
    hand_action: Literal["close", "restore_ready"] | None = None

    def __post_init__(self) -> None:
        if not self.occurrence_id.strip() or not self.candidate_id.strip():
            raise ValueError("bilateral capture waypoint IDs must be non-empty")
        if self.capture_role in {"ready", "anchor"}:
            if self.active_arm is not None:
                raise ValueError(
                    f"bilateral {self.capture_role} waypoint cannot name an active arm"
                )
        elif self.capture_role in {"preparation", "excitation"}:
            if self.active_arm not in _SIDES:
                raise ValueError(f"bilateral {self.capture_role} waypoint must name an active arm")
        else:
            raise ValueError("bilateral capture waypoint role is invalid")
        if self.hand_action not in {None, "close", "restore_ready"}:
            raise ValueError("bilateral waypoint hand action is invalid")
        if self.hand_action is not None and self.capture_role != "preparation":
            raise ValueError("bilateral hand actions are only legal at preparation waypoints")

    @property
    def capturable(self) -> bool:
        return self.capture_role in {"anchor", "excitation"}

    def to_dict(self) -> dict[str, Any]:
        return {name: getattr(self, name) for name in self.__dataclass_fields__}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> BilateralCaptureWaypoint:
        if set(data) != set(cls.__dataclass_fields__):
            raise ValueError("bilateral capture waypoint fields differ from schema version 1")
        return cls(**data)


@dataclass(frozen=True, slots=True)
class BilateralPoseDesignArtifact:
    """Frozen statistical design for one closed-hand anchor-to-anchor core."""

    model_sha256: str
    parameter_names: tuple[str, ...]
    selection: BilateralDesignSelection
    schedule: tuple[BilateralCaptureWaypoint, ...]
    waypoint_joint_positions_rad: dict[str, tuple[float, ...]]
    route_validation_sha256_by_transition: dict[str, str]
    planner_provenance: dict[str, Any]
    schema_version: int = 3

    def __post_init__(self) -> None:
        if self.schema_version != 3:
            raise ValueError("unsupported bilateral pose-design schema version")
        if not _SHA256_PATTERN.fullmatch(self.model_sha256):
            raise ValueError("bilateral pose-design model hash must be lowercase SHA-256")
        names = tuple(str(name) for name in self.parameter_names)
        if not names or len(names) != len(set(names)):
            raise ValueError("bilateral pose-design parameter names must be non-empty and unique")
        if len(names) != self.selection.report.model_parameter_count:
            raise ValueError("bilateral pose-design parameters differ from the design report")
        schedule = tuple(self.schedule)
        if (
            len(schedule) < 3
            or schedule[0].capture_role != "anchor"
            or schedule[-1].capture_role != "anchor"
        ):
            raise ValueError(
                "bilateral pose-design schedule must start and finish at the visual anchor"
            )
        anchor_candidate_ids = {
            item.candidate_id for item in schedule if item.capture_role == "anchor"
        }
        if len(anchor_candidate_ids) != 1:
            raise ValueError("bilateral pose-design schedule must repeat one identical anchor")
        if any(item.hand_action is not None for item in schedule):
            raise ValueError("bilateral pose-design core cannot contain hand actions")
        occurrence_ids = [item.occurrence_id for item in schedule]
        if (
            occurrence_ids[0] != HANDOFF_POSE_ID
            or occurrence_ids[-1] != HANDOFF_POSE_ID
            or HANDOFF_POSE_ID in occurrence_ids[1:-1]
            or len(occurrence_ids[1:-1]) != len(set(occurrence_ids[1:-1]))
        ):
            raise ValueError(
                "bilateral pose-design must have unique interior occurrences and "
                "the visual-anchor executor handoff at both ends"
            )
        selected = {item.candidate_id: item.active_arm for item in self.selection.candidates}
        for item in schedule:
            if (
                item.capture_role == "excitation"
                and selected.get(item.candidate_id) != item.active_arm
            ):
                raise ValueError("bilateral pose-design schedule differs from the selection")
        pose_ids = {item.candidate_id for item in schedule}
        poses = {
            str(candidate_id): tuple(
                float(value) for value in validate_full_joint_vector(position)
            )
            for candidate_id, position in self.waypoint_joint_positions_rad.items()
        }
        if set(poses) != pose_ids:
            raise ValueError(
                "bilateral pose-design must define every scheduled joint pose exactly"
            )
        expected_transitions = {
            f"{start.occurrence_id}->{end.occurrence_id}" for start, end in pairwise(schedule)
        }
        transitions = {
            str(name): str(value)
            for name, value in self.route_validation_sha256_by_transition.items()
        }
        if set(transitions) != expected_transitions or any(
            not _SHA256_PATTERN.fullmatch(value) for value in transitions.values()
        ):
            raise ValueError(
                "bilateral pose-design requires one CuRobo validation hash per transition"
            )
        object.__setattr__(self, "parameter_names", names)
        object.__setattr__(self, "schedule", schedule)
        object.__setattr__(self, "waypoint_joint_positions_rad", poses)
        object.__setattr__(self, "route_validation_sha256_by_transition", transitions)
        object.__setattr__(
            self,
            "planner_provenance",
            _canonical_mapping(self.planner_provenance, name="planner provenance"),
        )

    @property
    def content_sha256(self) -> str:
        encoded = json.dumps(
            self.to_dict(include_hash=False),
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode()
        return hashlib.sha256(encoded).hexdigest()

    def to_dict(self, *, include_hash: bool = True) -> dict[str, Any]:
        result = {
            "schema_version": self.schema_version,
            "model_sha256": self.model_sha256,
            "parameter_names": list(self.parameter_names),
            "selection": self.selection.to_dict(),
            "schedule": [item.to_dict() for item in self.schedule],
            "waypoint_joint_positions_rad": {
                name: list(position)
                for name, position in sorted(self.waypoint_joint_positions_rad.items())
            },
            "route_validation_sha256_by_transition": dict(
                sorted(self.route_validation_sha256_by_transition.items())
            ),
            "planner_provenance": self.planner_provenance,
        }
        if include_hash:
            result["content_sha256"] = self.content_sha256
        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> BilateralPoseDesignArtifact:
        expected = {
            "schema_version",
            "model_sha256",
            "parameter_names",
            "selection",
            "schedule",
            "waypoint_joint_positions_rad",
            "route_validation_sha256_by_transition",
            "planner_provenance",
            "content_sha256",
        }
        if set(data) != expected:
            raise ValueError("bilateral pose-design fields differ from schema version 1")
        result = cls(
            schema_version=int(data["schema_version"]),
            model_sha256=data["model_sha256"],
            parameter_names=tuple(data["parameter_names"]),
            selection=BilateralDesignSelection.from_dict(data["selection"]),
            schedule=tuple(BilateralCaptureWaypoint.from_dict(item) for item in data["schedule"]),
            waypoint_joint_positions_rad={
                str(name): tuple(position)
                for name, position in data["waypoint_joint_positions_rad"].items()
            },
            route_validation_sha256_by_transition=dict(
                data["route_validation_sha256_by_transition"]
            ),
            planner_provenance=dict(data["planner_provenance"]),
        )
        if result.content_sha256 != data["content_sha256"]:
            raise ValueError("bilateral pose-design content SHA-256 mismatch")
        return result

    @classmethod
    def from_json(cls, path: str | Path) -> BilateralPoseDesignArtifact:
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))

    def write_json(self, path: str | Path) -> Path:
        destination = Path(path).resolve()
        destination.parent.mkdir(parents=True, exist_ok=True)
        data = (
            json.dumps(self.to_dict(), indent=2, sort_keys=True, allow_nan=False) + "\n"
        ).encode()
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{destination.name}.",
            suffix=".tmp",
            dir=destination.parent,
        )
        temporary = Path(temporary_name)
        try:
            with os.fdopen(descriptor, "wb") as stream:
                stream.write(data)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, destination)
        finally:
            temporary.unlink(missing_ok=True)
        return destination


def linearize_design_candidate(
    *,
    candidate_id: str,
    active_arm: Literal["left", "right"],
    normalized_active_q: tuple[float, ...],
    predicted_sample: BilateralCalibrationSample,
    projection: BilateralCalibrationProjection,
    parameters: dict[str, float],
    image_coverage_bins: tuple[int, ...],
) -> BilateralDesignCandidate:
    """Linearize one predicted same-frame observation at the declared model."""

    jacobian = projection.normalized_jacobian(parameters, (predicted_sample,))
    return BilateralDesignCandidate(
        candidate_id=candidate_id,
        active_arm=active_arm,
        normalized_active_q=normalized_active_q,
        normalized_jacobian=tuple(tuple(float(value) for value in row) for row in jacobian),
        image_coverage_bins=image_coverage_bins,
    )


def select_bilateral_design(
    candidates: tuple[BilateralDesignCandidate, ...],
    *,
    parameter_names: tuple[str, ...],
    config: BilateralDesignConfig | None = None,
) -> BilateralDesignSelection:
    """Select balanced arm excitation using the complete model Jacobian."""

    design = config or BilateralDesignConfig()
    if not parameter_names or len(parameter_names) != len(set(parameter_names)):
        raise ValueError("design parameter names must be non-empty and unique")
    if len({item.candidate_id for item in candidates}) != len(candidates):
        raise ValueError("bilateral design candidate IDs must be unique")
    column_counts = {item.jacobian.shape[1] for item in candidates}
    if column_counts != {len(parameter_names)}:
        raise ValueError("candidate Jacobian columns do not match the declared model")
    required = {
        "left": design.left_excitation_count,
        "right": design.right_excitation_count,
    }
    available = {side: sum(item.active_arm == side for item in candidates) for side in _SIDES}
    for side in _SIDES:
        if available[side] < required[side]:
            raise ValueError(
                f"only {available[side]} {side} candidates are available; "
                f"{required[side]} required"
            )

    parameter_count = len(parameter_names)
    information = np.eye(parameter_count, dtype=np.float64) * design.information_ridge
    joint_information = {
        side: np.eye(7, dtype=np.float64) * design.information_ridge for side in _SIDES
    }
    coverage_counts: dict[tuple[int, ...], int] = {}
    selected_counts = {side: 0 for side in _SIDES}
    remaining = list(candidates)
    selected: list[BilateralDesignCandidate] = []
    steps: list[BilateralDesignStep] = []
    weights = np.asarray(
        [
            design.information_weight,
            design.joint_excitation_weight,
            design.image_coverage_weight,
        ],
        dtype=np.float64,
    )
    weights /= np.sum(weights)
    while any(selected_counts[side] < required[side] for side in _SIDES):
        eligible = [
            item
            for item in remaining
            if selected_counts[item.active_arm] < required[item.active_arm]
        ]
        information_gains = np.asarray(
            [
                _logdet(information + item.jacobian.T @ item.jacobian) - _logdet(information)
                for item in eligible
            ],
            dtype=np.float64,
        )
        joint_gains = np.asarray(
            [
                _joint_gain(
                    joint_information[item.active_arm],
                    np.asarray(item.normalized_active_q),
                )
                for item in eligible
            ],
            dtype=np.float64,
        )
        coverage_gains = np.asarray(
            [1.0 / (1.0 + coverage_counts.get(item.image_coverage_bins, 0)) for item in eligible],
            dtype=np.float64,
        )
        normalized = np.column_stack(
            (
                _normalize(information_gains),
                _normalize(joint_gains),
                _normalize(coverage_gains),
            )
        )
        combined = normalized @ weights
        chosen_index = max(
            range(len(eligible)),
            key=lambda index: (
                float(combined[index]),
                float(information_gains[index]),
                eligible[index].candidate_id,
            ),
        )
        chosen = eligible[chosen_index]
        selected.append(chosen)
        remaining.remove(chosen)
        selected_counts[chosen.active_arm] += 1
        information += chosen.jacobian.T @ chosen.jacobian
        q = np.asarray(chosen.normalized_active_q)
        joint_information[chosen.active_arm] += np.outer(q, q)
        coverage_counts[chosen.image_coverage_bins] = (
            coverage_counts.get(chosen.image_coverage_bins, 0) + 1
        )
        steps.append(
            BilateralDesignStep(
                selection_index=len(selected),
                candidate_id=chosen.candidate_id,
                active_arm=chosen.active_arm,
                information_gain_logdet=float(information_gains[chosen_index]),
                joint_excitation_gain_logdet=float(joint_gains[chosen_index]),
                image_coverage_gain=float(coverage_gains[chosen_index]),
                combined_score=float(combined[chosen_index]),
            )
        )

    report = _design_report(tuple(selected), design)
    if design.require_observable_model and (
        report.model_rank != report.model_parameter_count
        or report.model_condition_number is None
        or report.model_condition_number > design.maximum_model_condition_number
    ):
        raise ValueError(
            "selected bilateral design is not observable: "
            f"rank={report.model_rank}/{report.model_parameter_count}, "
            f"condition={report.model_condition_number}"
        )
    if design.require_full_joint_excitation:
        for side in _SIDES:
            rank = getattr(report, f"{side}_joint_rank")
            condition = getattr(report, f"{side}_joint_condition_number")
            if rank != 7 or condition is None or condition > design.maximum_joint_condition_number:
                raise ValueError(
                    f"selected {side} joint excitation is inadequate: "
                    f"rank={rank}/7, condition={condition}"
                )
    return BilateralDesignSelection(tuple(selected), tuple(steps), report)


def build_repeated_anchor_schedule(
    selection: BilateralDesignSelection,
    *,
    anchor_candidate_id: str,
    anchor_interval: int,
    candidate_order_by_arm: dict[str, tuple[str, ...]] | None = None,
) -> tuple[BilateralCaptureWaypoint, ...]:
    """Build one interleaved closed-hand route rooted at a repeated anchor."""

    if not anchor_candidate_id.strip():
        raise ValueError("bilateral route anchor candidate ID must be non-empty")
    if anchor_interval < 1:
        raise ValueError("anchor interval must be positive")
    selected_by_arm = {
        side: tuple(item.candidate_id for item in selection.candidates if item.active_arm == side)
        for side in _SIDES
    }
    if candidate_order_by_arm is None:
        ordered_by_arm = selected_by_arm
    else:
        if set(candidate_order_by_arm) != set(_SIDES):
            raise ValueError("bilateral route order must contain left and right")
        ordered_by_arm = {
            side: tuple(str(value) for value in candidate_order_by_arm[side]) for side in _SIDES
        }
        for side in _SIDES:
            if len(ordered_by_arm[side]) != len(set(ordered_by_arm[side])) or set(
                ordered_by_arm[side]
            ) != set(selected_by_arm[side]):
                raise ValueError(
                    f"bilateral {side} route order differs from the statistical selection"
                )
    selected = {item.candidate_id: item for item in selection.candidates}
    waypoints: list[BilateralCaptureWaypoint] = [
        BilateralCaptureWaypoint(
            occurrence_id=HANDOFF_POSE_ID,
            candidate_id=anchor_candidate_id,
            capture_role="anchor",
            active_arm=None,
        )
    ]
    anchor_index = 0
    excitation_index = 0

    def append_anchor() -> None:
        nonlocal anchor_index
        waypoints.append(
            BilateralCaptureWaypoint(
                occurrence_id=f"anchor_{anchor_index:03d}",
                candidate_id=anchor_candidate_id,
                capture_role="anchor",
                active_arm=None,
            )
        )
        anchor_index += 1

    queues = {side: list(ordered_by_arm[side]) for side in _SIDES}
    while any(queues.values()):
        for side in _SIDES:
            block = queues[side][:anchor_interval]
            del queues[side][: len(block)]
            if not block:
                continue
            for candidate_id in block:
                candidate = selected[candidate_id]
                waypoints.append(
                    BilateralCaptureWaypoint(
                        occurrence_id=f"excitation_{excitation_index:03d}",
                        candidate_id=candidate.candidate_id,
                        capture_role="excitation",
                        active_arm=side,
                    )
                )
                excitation_index += 1
            append_anchor()
    if waypoints[-1].capture_role != "anchor":
        raise RuntimeError("bilateral route did not return to its anchor")
    waypoints[-1] = BilateralCaptureWaypoint(
        occurrence_id=HANDOFF_POSE_ID,
        candidate_id=anchor_candidate_id,
        capture_role="anchor",
        active_arm=None,
    )
    return tuple(waypoints)


def build_valid_graph_route_schedule(
    selection: BilateralDesignSelection,
    *,
    anchor_candidate_id: str,
    anchor_interval: int,
    valid_edges_by_arm: dict[str, tuple[tuple[float, str, str], ...]],
) -> tuple[BilateralCaptureWaypoint, ...]:
    """Build minimum-cost anchor tours over collision-certified edge graphs.

    Every short arm block starts and ends at the shared bilateral anchor so
    the opposite arm remains at its anchor command.  A state-space Dijkstra
    search minimizes the certified-edge motion cost while tracking which
    block poses have been visited.  Selected poses used only as intermediate
    graph nodes are explicit, non-capturing preparation waypoints; every
    excitation is captured exactly once in its assigned block.
    """

    if anchor_interval < 1:
        raise ValueError("bilateral graph-route anchor interval must be positive")
    if set(valid_edges_by_arm) != set(_SIDES):
        raise ValueError("bilateral graph route must define both arm edge graphs")
    if not anchor_candidate_id.strip():
        raise ValueError("bilateral graph-route anchor ID must be non-empty")

    selected_by_arm = {
        side: tuple(item.candidate_id for item in selection.candidates if item.active_arm == side)
        for side in _SIDES
    }
    selected_ids = {candidate_id for values in selected_by_arm.values() for candidate_id in values}
    allowed_nodes = selected_ids | {anchor_candidate_id}
    adjacency: dict[str, dict[str, tuple[tuple[float, str], ...]]] = {}
    for side in _SIDES:
        nodes = set(selected_by_arm[side]) | {anchor_candidate_id}
        neighbors: dict[str, dict[str, float]] = {node: {} for node in nodes}
        seen_pairs: set[tuple[str, str]] = set()
        for raw_cost, raw_first, raw_second in valid_edges_by_arm[side]:
            cost = float(raw_cost)
            first = str(raw_first)
            second = str(raw_second)
            pair = tuple(sorted((first, second)))
            if (
                not np.isfinite(cost)
                or cost <= 0.0
                or first == second
                or first not in nodes
                or second not in nodes
                or first not in allowed_nodes
                or second not in allowed_nodes
            ):
                raise ValueError(f"bilateral {side} graph contains an invalid edge")
            if pair in seen_pairs:
                raise ValueError(f"bilateral {side} graph contains a duplicate edge")
            seen_pairs.add(pair)
            neighbors[first][second] = cost
            neighbors[second][first] = cost
        adjacency[side] = {
            node: tuple(
                sorted(
                    ((cost, neighbor) for neighbor, cost in values.items()),
                    key=lambda item: (item[0], item[1]),
                )
            )
            for node, values in neighbors.items()
        }

    def shortest_closed_walk(side: str, captures: tuple[str, ...]) -> tuple[str, ...]:
        """Return graph nodes after the anchor, including the final anchor."""

        if not captures or len(captures) != len(set(captures)):
            raise ValueError(f"bilateral {side} graph block captures must be unique")
        capture_bit = {candidate_id: 1 << index for index, candidate_id in enumerate(captures)}
        graph = adjacency[side]
        if any(candidate_id not in graph for candidate_id in captures):
            raise ValueError(f"bilateral {side} graph block contains an unknown capture")
        full_mask = (1 << len(captures)) - 1
        start = (anchor_candidate_id, 0)
        best: dict[tuple[str, int], tuple[float, int]] = {start: (0.0, 0)}
        previous: dict[tuple[str, int], tuple[str, int]] = {}
        heap: list[tuple[float, int, str, int]] = [(0.0, 0, anchor_candidate_id, 0)]
        goal: tuple[str, int] | None = None
        while heap:
            cost, hops, node, mask = heapq.heappop(heap)
            state = (node, mask)
            if best.get(state) != (cost, hops):
                continue
            if node == anchor_candidate_id and mask == full_mask:
                goal = state
                break
            for edge_cost, neighbor in graph[node]:
                next_mask = mask | capture_bit.get(neighbor, 0)
                next_state = (neighbor, next_mask)
                next_value = (cost + edge_cost, hops + 1)
                if next_value < best.get(next_state, (float("inf"), 2**31 - 1)):
                    best[next_state] = next_value
                    previous[next_state] = state
                    heapq.heappush(
                        heap,
                        (next_value[0], next_value[1], neighbor, next_mask),
                    )
        if goal is None:
            raise ValueError(
                f"bilateral {side} graph cannot return its capture block to the anchor"
            )
        reversed_nodes: list[str] = []
        state = goal
        while state != start:
            reversed_nodes.append(state[0])
            state = previous[state]
        return tuple(reversed(reversed_nodes))

    waypoints: list[BilateralCaptureWaypoint] = [
        BilateralCaptureWaypoint(
            occurrence_id=HANDOFF_POSE_ID,
            candidate_id=anchor_candidate_id,
            capture_role="anchor",
            active_arm=None,
        ),
    ]
    occurrence_index = 0
    anchor_index = 1
    queues = {side: list(selected_by_arm[side]) for side in _SIDES}

    def append_block(side: str, captures: tuple[str, ...]) -> None:
        nonlocal occurrence_index, anchor_index
        traversal = shortest_closed_walk(side, captures)
        capture_ids = set(captures)
        captured: set[str] = set()
        for traversal_index, candidate_id in enumerate(traversal, start=1):
            is_final_anchor = candidate_id == anchor_candidate_id and traversal_index == len(
                traversal
            )
            if is_final_anchor:
                waypoints.append(
                    BilateralCaptureWaypoint(
                        occurrence_id=f"anchor_{anchor_index:03d}",
                        candidate_id=anchor_candidate_id,
                        capture_role="anchor",
                        active_arm=None,
                    )
                )
                anchor_index += 1
                continue
            capture = candidate_id in capture_ids and candidate_id not in captured
            if capture:
                captured.add(candidate_id)
            waypoints.append(
                BilateralCaptureWaypoint(
                    occurrence_id=f"graph_{occurrence_index:03d}",
                    candidate_id=candidate_id,
                    capture_role="excitation" if capture else "preparation",
                    active_arm=side,
                )
            )
            occurrence_index += 1
        if captured != set(captures):
            raise RuntimeError(f"bilateral {side} graph block did not visit every capture")

    while any(queues.values()):
        for side in _SIDES:
            block = tuple(queues[side][:anchor_interval])
            del queues[side][: len(block)]
            if block:
                append_block(side, block)

    if waypoints[-1].capture_role != "anchor":
        raise RuntimeError("bilateral graph route did not return to its anchor")
    waypoints[-1] = BilateralCaptureWaypoint(
        occurrence_id=HANDOFF_POSE_ID,
        candidate_id=anchor_candidate_id,
        capture_role="anchor",
        active_arm=None,
    )
    return tuple(waypoints)


def _joint_gain(information: np.ndarray, q: np.ndarray) -> float:
    return _logdet(information + np.outer(q, q)) - _logdet(information)


def _logdet(matrix: np.ndarray) -> float:
    sign, value = np.linalg.slogdet((matrix + matrix.T) / 2.0)
    if sign <= 0 or not np.isfinite(value):
        raise ValueError("design information matrix is not positive definite")
    return float(value)


def _normalize(values: np.ndarray) -> np.ndarray:
    lower = float(np.min(values))
    upper = float(np.max(values))
    if upper - lower <= 1e-12:
        return np.ones_like(values)
    return (values - lower) / (upper - lower)


def _rank_and_condition(
    matrix: np.ndarray,
    *,
    relative_threshold: float,
) -> tuple[int, float | None]:
    singular = np.linalg.svd(matrix, compute_uv=False)
    if not len(singular) or singular[0] <= 0.0:
        return 0, None
    threshold = relative_threshold * singular[0]
    rank = int(np.count_nonzero(singular > threshold))
    condition = float(singular[0] / singular[-1]) if singular[-1] > 0.0 else None
    return rank, condition


def _design_report(
    selected: tuple[BilateralDesignCandidate, ...],
    config: BilateralDesignConfig,
) -> BilateralDesignReport:
    jacobian = np.vstack([item.jacobian for item in selected])
    model_rank, model_condition = _rank_and_condition(
        jacobian,
        relative_threshold=config.relative_rank_threshold,
    )
    joint_results: dict[str, tuple[int, float | None]] = {}
    for side in _SIDES:
        configurations = np.asarray(
            [item.normalized_active_q for item in selected if item.active_arm == side],
            dtype=np.float64,
        )
        centered = configurations - np.mean(configurations, axis=0)
        joint_results[side] = _rank_and_condition(
            centered,
            relative_threshold=config.relative_rank_threshold,
        )
    return BilateralDesignReport(
        model_rank=model_rank,
        model_parameter_count=jacobian.shape[1],
        model_condition_number=model_condition,
        left_joint_rank=joint_results["left"][0],
        left_joint_condition_number=joint_results["left"][1],
        right_joint_rank=joint_results["right"][0],
        right_joint_condition_number=joint_results["right"][1],
    )
