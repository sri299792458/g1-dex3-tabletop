"""Frozen bilateral route execution over the commissioned arm controller."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path
from typing import Any, Literal, Protocol

import numpy as np

from g1_aprilcube_calibration.executor_state_machine import ExecutorState
from g1_aprilcube_calibration.joint_map import G1_29_JOINT_NAMES, arm_indices
from g1_aprilcube_calibration.models import utc_now_iso
from g1_aprilcube_calibration.pose_schema import (
    HANDOFF_POSE_ID,
    PoseAuditEvent,
    PoseRecord,
    PoseSet,
)
from g1_aprilcube_calibration.session_runner import (
    RecoverableCaptureError,
    finish_capture_or_raise_fault,
)
from g1_dex3_tabletop.calibration.capture import (
    BilateralFrameEvidence,
    BilateralGracefulStopRequested,
)
from g1_dex3_tabletop.calibration.design import BilateralPoseDesignArtifact
from g1_dex3_tabletop.planning.contracts import PlannedTrajectory, atomic_write_json

_SIDES = ("left", "right")
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_ENDPOINT_EPSILON_RAD = 1e-8


@dataclass(frozen=True, slots=True)
class BilateralPlannedTransition:
    """One collision-certified edge that changes exactly one arm."""

    arm: Literal["left", "right"]
    trajectory: PlannedTrajectory

    def __post_init__(self) -> None:
        if self.arm not in _SIDES:
            raise ValueError("bilateral transition arm must be left or right")
        if not isinstance(self.trajectory, PlannedTrajectory):
            object.__setattr__(
                self,
                "trajectory",
                PlannedTrajectory.from_dict(self.trajectory),
            )

    @property
    def transition_id(self) -> str:
        return f"{self.trajectory.from_pose_id}->{self.trajectory.to_pose_id}"

    @property
    def content_sha256(self) -> str:
        return hashlib.sha256(
            json.dumps(
                self.to_dict(),
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ).encode()
        ).hexdigest()

    def to_dict(self) -> dict[str, Any]:
        return {
            "arm": self.arm,
            "trajectory": self.trajectory.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> BilateralPlannedTransition:
        if set(data) != {"arm", "trajectory"}:
            raise ValueError("bilateral planned-transition fields differ from schema version 1")
        return cls(
            arm=data["arm"],
            trajectory=PlannedTrajectory.from_dict(data["trajectory"]),
        )


@dataclass(frozen=True, slots=True)
class BilateralExecutionPlan:
    """Exact closed-hand trajectories bound to one reusable bilateral core."""

    pose_design_sha256: str
    robot_model: str
    urdf_sha256: str
    joint_position_offsets_rad: dict[str, float]
    commanded_dex3_joint_positions_rad: dict[str, tuple[float, ...]]
    modeled_dex3_joint_positions_rad: dict[str, tuple[float, ...]]
    self_clearance_certificate: dict[str, Any]
    transitions: tuple[BilateralPlannedTransition, ...]
    planner_provenance: dict[str, Any]
    schema_version: int = 5

    def __post_init__(self) -> None:
        if self.schema_version != 5:
            raise ValueError("unsupported bilateral execution-plan schema version")
        for name in ("pose_design_sha256", "urdf_sha256"):
            if not _SHA256_PATTERN.fullmatch(getattr(self, name)):
                raise ValueError(f"{name} must be lowercase SHA-256")
        if not self.robot_model.strip():
            raise ValueError("bilateral execution-plan robot model must be non-empty")
        offsets = {
            str(name): float(value) for name, value in self.joint_position_offsets_rad.items()
        }
        if any(
            name not in G1_29_JOINT_NAMES or not np.isfinite(value)
            for name, value in offsets.items()
        ):
            raise ValueError("bilateral execution-plan joint offsets are invalid")
        dex3_postures: dict[str, dict[str, tuple[float, ...]]] = {}
        for field in (
            "commanded_dex3_joint_positions_rad",
            "modeled_dex3_joint_positions_rad",
        ):
            source = getattr(self, field)
            if set(source) != set(_SIDES):
                raise ValueError(f"bilateral execution plan {field} must bind both hands")
            mapped: dict[str, tuple[float, ...]] = {}
            for side in _SIDES:
                values = np.asarray(source[side], dtype=np.float64).reshape(-1)
                if values.shape != (7,) or not np.all(np.isfinite(values)):
                    raise ValueError(f"bilateral {side} Dex3 posture must contain seven values")
                mapped[side] = tuple(float(value) for value in values)
            dex3_postures[field] = mapped
        clearance = json.loads(
            json.dumps(self.self_clearance_certificate, sort_keys=True, allow_nan=False)
        )
        if (
            not isinstance(clearance, dict)
            or clearance.get("passed") is not True
            or float(clearance.get("hard_clearance_m", 0.0)) <= 0.0
            or float(clearance.get("minimum_clearance_m", -1.0)) < -1.0e-6
            or float(clearance.get("minimum_margin_to_required_clearance_m", -1.0)) < -1.0e-6
            or not clearance.get("phases")
        ):
            raise ValueError("bilateral execution plan lacks a passing self-clearance certificate")
        transitions = tuple(
            item
            if isinstance(item, BilateralPlannedTransition)
            else BilateralPlannedTransition.from_dict(item)
            for item in self.transitions
        )
        if not transitions:
            raise ValueError("bilateral execution plan requires trajectories")
        transition_ids = [item.transition_id for item in transitions]
        if len(transition_ids) != len(set(transition_ids)):
            raise ValueError("bilateral execution-plan transition IDs must be unique")
        provenance = json.loads(
            json.dumps(self.planner_provenance, sort_keys=True, allow_nan=False)
        )
        if not isinstance(provenance, dict) or not provenance:
            raise ValueError("bilateral execution-plan provenance must be non-empty")
        object.__setattr__(self, "transitions", transitions)
        object.__setattr__(self, "joint_position_offsets_rad", dict(sorted(offsets.items())))
        for field, posture in dex3_postures.items():
            object.__setattr__(self, field, posture)
        object.__setattr__(self, "self_clearance_certificate", clearance)
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

    def validate_design(self, design: BilateralPoseDesignArtifact) -> None:
        if self.pose_design_sha256 != design.content_sha256:
            raise ValueError("bilateral execution plan belongs to a different pose design")
        expected_edges = tuple(
            f"{start.occurrence_id}->{end.occurrence_id}"
            for start, end in pairwise(design.schedule)
        )
        if tuple(item.transition_id for item in self.transitions) != expected_edges:
            raise ValueError("bilateral execution trajectories do not follow the schedule")
        for (start, end), transition in zip(
            pairwise(design.schedule),
            self.transitions,
            strict=True,
        ):
            expected_hash = design.route_validation_sha256_by_transition[transition.transition_id]
            if transition.content_sha256 != expected_hash:
                raise ValueError(f"bilateral trajectory hash changed: {transition.transition_id}")
            start_q = np.asarray(
                design.waypoint_joint_positions_rad[start.candidate_id],
                dtype=np.float64,
            )
            end_q = np.asarray(
                design.waypoint_joint_positions_rad[end.candidate_id],
                dtype=np.float64,
            )
            active = np.asarray(arm_indices(transition.arm), dtype=np.int64)
            active_set = set(active)
            inactive = np.asarray(
                [index for index in range(len(start_q)) if index not in active_set],
                dtype=np.int64,
            )
            if np.max(np.abs(start_q[inactive] - end_q[inactive])) > _ENDPOINT_EPSILON_RAD:
                raise ValueError(
                    f"bilateral transition changes joints outside {transition.arm}: "
                    f"{transition.transition_id}"
                )
            command = np.asarray(transition.trajectory.command_q_rad, dtype=np.float64)
            start_error = float(np.max(np.abs(command[0] - start_q[active])))
            end_error = float(np.max(np.abs(command[-1] - end_q[active])))
            if max(start_error, end_error) > _ENDPOINT_EPSILON_RAD:
                raise ValueError(
                    f"bilateral trajectory endpoints differ from the pose design: "
                    f"{transition.transition_id}"
                )
        anchor_indices = [
            index
            for index, waypoint in enumerate(design.schedule)
            if waypoint.capture_role == "anchor"
        ]
        graceful = self.planner_provenance.get("graceful_return")
        expected_graceful = {
            "policy": "latch_at_any_capture_then_stop_at_next_identical_anchor",
            "anchor_occurrence_ids": [
                design.schedule[index].occurrence_id for index in anchor_indices
            ],
            "terminal_boundary": HANDOFF_POSE_ID,
        }
        if graceful != expected_graceful:
            raise ValueError("bilateral execution plan lacks its hash-bound graceful return")

    def to_dict(self, *, include_hash: bool = True) -> dict[str, Any]:
        result = {
            "schema_version": self.schema_version,
            "pose_design_sha256": self.pose_design_sha256,
            "robot_model": self.robot_model,
            "urdf_sha256": self.urdf_sha256,
            "joint_position_offsets_rad": self.joint_position_offsets_rad,
            "commanded_dex3_joint_positions_rad": {
                side: list(self.commanded_dex3_joint_positions_rad[side]) for side in _SIDES
            },
            "modeled_dex3_joint_positions_rad": {
                side: list(self.modeled_dex3_joint_positions_rad[side]) for side in _SIDES
            },
            "self_clearance_certificate": self.self_clearance_certificate,
            "transitions": [item.to_dict() for item in self.transitions],
            "planner_provenance": self.planner_provenance,
        }
        if include_hash:
            result["content_sha256"] = self.content_sha256
        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> BilateralExecutionPlan:
        expected = {
            "schema_version",
            "pose_design_sha256",
            "robot_model",
            "urdf_sha256",
            "joint_position_offsets_rad",
            "commanded_dex3_joint_positions_rad",
            "modeled_dex3_joint_positions_rad",
            "self_clearance_certificate",
            "transitions",
            "planner_provenance",
            "content_sha256",
        }
        if set(data) != expected:
            raise ValueError("bilateral execution-plan fields differ from schema version 5")
        result = cls(
            schema_version=int(data["schema_version"]),
            pose_design_sha256=data["pose_design_sha256"],
            robot_model=data["robot_model"],
            urdf_sha256=data["urdf_sha256"],
            joint_position_offsets_rad=dict(data["joint_position_offsets_rad"]),
            commanded_dex3_joint_positions_rad={
                side: tuple(values)
                for side, values in data["commanded_dex3_joint_positions_rad"].items()
            },
            modeled_dex3_joint_positions_rad={
                side: tuple(values)
                for side, values in data["modeled_dex3_joint_positions_rad"].items()
            },
            self_clearance_certificate=dict(data["self_clearance_certificate"]),
            transitions=tuple(
                BilateralPlannedTransition.from_dict(item) for item in data["transitions"]
            ),
            planner_provenance=dict(data["planner_provenance"]),
        )
        if result.content_sha256 != data["content_sha256"]:
            raise ValueError("bilateral execution-plan content SHA-256 mismatch")
        return result

    @classmethod
    def from_json(cls, path: str | Path) -> BilateralExecutionPlan:
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))

    def write_json(self, path: str | Path) -> None:
        atomic_write_json(path, self.to_dict())


def pose_sets_from_bilateral_plan(
    design: BilateralPoseDesignArtifact,
    plan: BilateralExecutionPlan,
) -> dict[str, PoseSet]:
    """Build occurrence-addressed endpoint metadata for each controller arm."""

    plan.validate_design(design)
    timestamp = utc_now_iso()
    result: dict[str, PoseSet] = {}
    for side in _SIDES:
        indices = np.asarray(arm_indices(side), dtype=np.int64)
        records: list[PoseRecord] = []
        audit: list[PoseAuditEvent] = []
        for order, waypoint in enumerate(design.schedule[1:-1]):
            full_q = np.asarray(
                design.waypoint_joint_positions_rad[waypoint.candidate_id],
                dtype=np.float64,
            )
            record = PoseRecord(
                id=waypoint.occurrence_id,
                group="bilateral_calibration",
                measured_calibration_q=tuple(float(value) for value in full_q[indices]),
                measured_full_q=tuple(float(value) for value in full_q),
                calibration_q_spread=(0.0,) * 7,
                recorded_at_utc=timestamp,
                recorded_monotonic_s=float(order),
                source="NVlabs/curobo_frozen_bilateral_trajectory",
                visual_quality={
                    "candidate_id": waypoint.candidate_id,
                    "capture_role": waypoint.capture_role,
                    "pose_design_sha256": design.content_sha256,
                },
            )
            records.append(record)
            audit.append(PoseAuditEvent("add", record.id, timestamp))
        result[side] = PoseSet(
            robot_model=plan.robot_model,
            mode_machine=5,
            urdf_sha256=plan.urdf_sha256,
            calibration_arm=side,
            poses=tuple(records),
            audit_log=tuple(audit),
        )
    return result


class BilateralExecutor(Protocol):
    state: ExecutorState
    fault_reason: str | None
    current_pose_id: str | None
    approved_validation_report_sha256: str
    pose_set: PoseSet

    def begin_capture(self) -> None: ...

    def finish_capture(self, *, outcome: str) -> None: ...

    def observe_state(self): ...

    def switch_validated_arm_plan(self, **kwargs) -> None: ...

    def start_trajectory(self, **kwargs) -> None: ...


class BilateralCaptureStore(Protocol):
    def append_capture(self, **kwargs): ...

    def finalize(self): ...


class BilateralBurstSource(Protocol):
    def capture_burst(
        self,
        *,
        pose_id: str,
        capture_id: str,
        remember_signatures: bool = True,
    ) -> Sequence[BilateralFrameEvidence]: ...


@dataclass(frozen=True, slots=True)
class BilateralCollectionResult:
    accepted_count: int
    rejected_count: int
    retry_count: int
    attempted_count: int
    stopped_early: bool = False
    return_anchor_occurrence_id: str | None = None
    session_finalized: bool = True


class BilateralCollectionOrchestrator:
    """Execute and capture the immutable same-frame bilateral route."""

    def __init__(
        self,
        *,
        executor: BilateralExecutor,
        design: BilateralPoseDesignArtifact,
        plan: BilateralExecutionPlan,
        pose_sets: Mapping[str, PoseSet],
        store: BilateralCaptureStore,
        frame_source: BilateralBurstSource,
        wait_until_ready: Callable[[], None],
        graceful_stop_requested: Callable[[], bool] | None = None,
        maximum_capture_attempts: int = 2,
        report_progress: Callable[[str, int, int], None] | None = None,
    ) -> None:
        plan.validate_design(design)
        if set(pose_sets) != set(_SIDES):
            raise ValueError("bilateral orchestrator requires left and right pose sets")
        if maximum_capture_attempts < 1:
            raise ValueError("bilateral capture attempts must be positive")
        if executor.approved_validation_report_sha256 != plan.content_sha256:
            raise ValueError("bilateral executor is not bound to the execution plan")
        self.executor = executor
        self.design = design
        self.plan = plan
        self.pose_sets = dict(pose_sets)
        self.store = store
        self.frame_source = frame_source
        self.wait_until_ready = wait_until_ready
        self.graceful_stop_requested = graceful_stop_requested or (lambda: False)
        self.maximum_capture_attempts = maximum_capture_attempts
        self.report_progress = report_progress or (lambda _message, _accepted, _retries: None)

    def run(self) -> BilateralCollectionResult:
        if (
            self.executor.state is not ExecutorState.READY
            or self.executor.current_pose_id != HANDOFF_POSE_ID
        ):
            raise RuntimeError("bilateral control is not ready at the handoff")
        accepted = 0
        rejected = 0
        retries = 0
        attempts = 0
        stopping = False
        return_anchor_occurrence_id: str | None = None
        for waypoint_index, waypoint in enumerate(self.design.schedule):
            stopping = stopping or self.graceful_stop_requested()
            if stopping and waypoint.capture_role == "anchor":
                return_anchor_occurrence_id = waypoint.occurrence_id
                self.report_progress(
                    f"graceful stop reached {waypoint.occurrence_id}",
                    accepted,
                    retries,
                )
                break
            if waypoint.capturable and not stopping:
                outcome, used_attempts = self._capture_waypoint(
                    waypoint_index=waypoint_index,
                )
                attempts += used_attempts
                retries += used_attempts - 1
                if outcome == "stopped":
                    stopping = True
                    self.report_progress(
                        f"graceful stop requested at {waypoint.occurrence_id}; "
                        "returning at the next anchor",
                        accepted,
                        retries,
                    )
                elif outcome == "rejected":
                    rejected += 1
                    self.report_progress(
                        f"rejected {waypoint.occurrence_id}; continuing frozen route",
                        accepted,
                        retries,
                    )
                else:
                    accepted += 1
                    self.report_progress(
                        f"accepted {waypoint.occurrence_id}",
                        accepted,
                        retries,
                    )
                stopping = stopping or self.graceful_stop_requested()
                if stopping and waypoint.capture_role == "anchor":
                    return_anchor_occurrence_id = waypoint.occurrence_id
                    self.report_progress(
                        f"graceful stop reached {waypoint.occurrence_id}",
                        accepted,
                        retries,
                    )
                    break
            if waypoint_index == len(self.design.schedule) - 1:
                break
            transition = self.plan.transitions[waypoint_index]
            self._execute_transition(transition)
        if return_anchor_occurrence_id is None:
            if self.executor.current_pose_id != HANDOFF_POSE_ID:
                raise RuntimeError("bilateral route did not terminate at the anchor handoff")
        elif self.executor.current_pose_id != return_anchor_occurrence_id:
            raise RuntimeError("bilateral graceful stop did not terminate at its repeated anchor")
        session_finalized = accepted > 0
        if session_finalized:
            self.store.finalize()
        return BilateralCollectionResult(
            accepted,
            rejected,
            retries,
            attempts,
            stopped_early=return_anchor_occurrence_id is not None,
            return_anchor_occurrence_id=return_anchor_occurrence_id,
            session_finalized=session_finalized,
        )

    def _execute_transition(
        self,
        transition: BilateralPlannedTransition,
        *,
        from_pose_id: str | None = None,
    ) -> None:
        self._activate_arm(transition.arm)
        self.executor.start_trajectory(
            from_pose_id=(
                transition.trajectory.from_pose_id if from_pose_id is None else from_pose_id
            ),
            to_pose_id=transition.trajectory.to_pose_id,
            sample_time_s=transition.trajectory.sample_time_s,
            command_q_rad=transition.trajectory.command_q_rad,
            plan_sha256=self.plan.content_sha256,
            operator_confirmed=True,
        )
        self.wait_until_ready()

    def _activate_arm(self, side: str) -> None:
        if self.executor.pose_set.calibration_arm == side:
            return
        boundary = self.executor.current_pose_id
        if boundary is None:
            raise RuntimeError("cannot switch bilateral arms at an unnamed boundary")
        reference = self.executor.observe_state()
        self.executor.switch_validated_arm_plan(
            pose_set=self.pose_sets[side],
            approved_validation_report_sha256=self.plan.content_sha256,
            validated_reference_state=reference,
            boundary_pose_id=boundary,
        )

    def _capture_waypoint(
        self,
        *,
        waypoint_index: int,
    ) -> tuple[Literal["accepted", "rejected", "stopped"], int]:
        waypoint = self.design.schedule[waypoint_index]
        for attempt_index in range(1, self.maximum_capture_attempts + 1):
            capture_id = f"capture_{waypoint_index:03d}_attempt_{attempt_index:02d}"
            self.executor.begin_capture()
            try:
                frames = self.frame_source.capture_burst(
                    pose_id=waypoint.candidate_id,
                    capture_id=capture_id,
                    remember_signatures=True,
                )
            except BilateralGracefulStopRequested as error:
                outcome, reason, frames = "aborted", str(error), ()
            except RecoverableCaptureError as error:
                outcome = "rejected" if attempt_index == self.maximum_capture_attempts else "retry"
                reason, frames = str(error), ()
            except Exception as error:
                finish_capture_or_raise_fault(
                    self.executor, outcome="bilateral burst failed", cause=error
                )
                raise
            else:
                outcome, reason = "accepted", "both targets passed the stationary burst"
            try:
                # Keep the stationary capture interlock until the durable write
                # finishes, as in the commissioned single-arm capture runner.
                self.store.append_capture(
                    capture_id=capture_id,
                    pose_group_id=waypoint.candidate_id,
                    capture_role=waypoint.capture_role,
                    outcome=outcome,
                    reason=reason,
                    frames=frames,
                    metadata={
                        "occurrence_id": waypoint.occurrence_id,
                        "attempt_index": attempt_index,
                    },
                )
            except Exception as error:
                finish_capture_or_raise_fault(
                    self.executor, outcome="bilateral capture write failed", cause=error
                )
                raise
            finish_capture_or_raise_fault(self.executor, outcome=outcome)
            if outcome == "aborted":
                return "stopped", attempt_index
            if outcome in {"accepted", "rejected"}:
                return outcome, attempt_index
        raise AssertionError("bilateral capture retry loop terminated unexpectedly")
