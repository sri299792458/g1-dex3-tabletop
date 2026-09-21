"""Capture lifecycle and approved pose-plan orchestration."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Protocol

from g1_aprilcube_calibration.authored_collection import AuthoredCollectionPlan
from g1_aprilcube_calibration.executor_state_machine import (
    ExecutorState,
    PoseExecutor,
)
from g1_aprilcube_calibration.pose_schema import HANDOFF_POSE_ID
from g1_aprilcube_calibration.pose_validator import ValidationReport
from g1_aprilcube_calibration.session_store import CaptureFrameInput


class CaptureStore(Protocol):
    def append_capture(
        self,
        *,
        capture_id: str,
        pose_id: str,
        outcome: str,
        reason: str,
        frames: Sequence[CaptureFrameInput] = (),
        recorded_at_utc: str | None = None,
    ): ...

    def finalize(self): ...


class FrameSource(Protocol):
    def capture_burst(self, *, pose_id: str, capture_id: str) -> tuple[CaptureFrameInput, ...]: ...


class RecoverableCaptureError(RuntimeError):
    """A pose-local camera failure that should not stop the replay route."""


@dataclass(frozen=True, slots=True)
class SessionExecutionPlan:
    capture_pose_ids: tuple[str, ...]
    maximum_control_steps_per_transition: int = 10_000

    def __post_init__(self) -> None:
        if not self.capture_pose_ids:
            raise ValueError("capture plan must contain at least one pose")
        if self.maximum_control_steps_per_transition <= 0:
            raise ValueError("maximum control steps must be positive")


class ScheduledFrameSource:
    """One-shot deterministic frame bursts used by offline integration runs."""

    def __init__(self, frames_by_pose: Mapping[str, Sequence[CaptureFrameInput]]) -> None:
        self._frames = {pose_id: tuple(frames) for pose_id, frames in frames_by_pose.items()}
        self._consumed: set[str] = set()

    def capture_burst(self, *, pose_id: str, capture_id: str) -> tuple[CaptureFrameInput, ...]:
        del capture_id
        if pose_id in self._consumed:
            raise RuntimeError(f"frame burst for {pose_id} was already consumed")
        if pose_id not in self._frames:
            raise ValueError(f"no scheduled frame burst for pose {pose_id}")
        self._consumed.add(pose_id)
        return self._frames[pose_id]


class CaptureSessionRunner:
    """Tie raw writes to the executor's stationary capture interlock."""

    def __init__(self, *, executor: PoseExecutor, store: CaptureStore) -> None:
        self.executor = executor
        self.store = store

    def capture(
        self,
        *,
        capture_id: str,
        pose_id: str,
        frames: Sequence[CaptureFrameInput],
        reason: str = "stationary burst passed",
    ) -> None:
        if self.executor.current_pose_id != pose_id:
            raise ValueError("capture pose does not match executor's current pose")
        self.executor.begin_capture()
        try:
            self.store.append_capture(
                capture_id=capture_id,
                pose_id=pose_id,
                outcome="accepted",
                reason=reason,
                frames=frames,
            )
        except Exception as error:
            self._finish_capture_or_raise_fault(
                outcome="raw capture failed",
                cause=error,
            )
            raise
        self._finish_capture_or_raise_fault(outcome="accepted")

    def reject(self, *, capture_id: str, pose_id: str, reason: str) -> None:
        if self.executor.current_pose_id != pose_id:
            raise ValueError("capture pose does not match executor's current pose")
        self.executor.begin_capture()
        try:
            self.store.append_capture(
                capture_id=capture_id,
                pose_id=pose_id,
                outcome="rejected",
                reason=reason,
            )
        except Exception as error:
            self._finish_capture_or_raise_fault(
                outcome="rejection write failed",
                cause=error,
            )
            raise
        self._finish_capture_or_raise_fault(outcome="rejected")

    def _finish_capture_or_raise_fault(
        self,
        *,
        outcome: str,
        cause: BaseException | None = None,
    ) -> None:
        if self.executor.state is ExecutorState.FAULT:
            reason = self.executor.fault_reason or "unknown reason"
            raise RuntimeError(f"executor faulted during capture: {reason}") from cause
        self.executor.finish_capture(outcome=outcome)


class ApprovedSessionOrchestrator:
    """Run an already validated pose plan; every physical move is confirmed."""

    def __init__(
        self,
        *,
        executor: PoseExecutor,
        validation_report: ValidationReport,
        capture_runner: CaptureSessionRunner,
        frame_source: FrameSource,
        control_step: Callable[[], None],
        confirm_move: Callable[[str, str], bool],
        plan: SessionExecutionPlan,
        report_capture_rejection: Callable[[str, str], None] | None = None,
    ) -> None:
        if validation_report.pose_set_sha256 != executor.pose_set.content_sha256:
            raise ValueError("validation report belongs to a different pose set")
        if validation_report.content_sha256 != (executor.approved_validation_report_sha256):
            raise ValueError("executor was not bound to this validation report")
        if not validation_report.passed:
            raise ValueError("cannot execute a failed validation report")
        pose_ids = {pose.id for pose in executor.pose_set.poses}
        missing = set(plan.capture_pose_ids)
        missing -= pose_ids
        if missing:
            raise ValueError(
                "execution plan references unknown poses: " + ", ".join(sorted(missing))
            )
        self.executor = executor
        self.validation_report = validation_report
        self.capture_runner = capture_runner
        self.frame_source = frame_source
        self.control_step = control_step
        self.confirm_move = confirm_move
        self.plan = plan
        self.report_capture_rejection = report_capture_rejection or (
            lambda _pose_id, _reason: None
        )

    def run(self, *, confirm_acquisition: bool, confirm_release: bool) -> None:
        try:
            self.executor.acquire(
                operator_confirmed=confirm_acquisition,
            )
            self._drive_until(ExecutorState.READY)
            for index, pose_id in enumerate(self.plan.capture_pose_ids, start=1):
                if self.executor.current_pose_id != pose_id:
                    self._move_to(pose_id)
                capture_id = f"capture_{index:03d}"
                try:
                    frames = self.frame_source.capture_burst(
                        pose_id=pose_id, capture_id=capture_id
                    )
                except RecoverableCaptureError as error:
                    reason = str(error)
                    self.capture_runner.reject(
                        capture_id=capture_id,
                        pose_id=pose_id,
                        reason=reason,
                    )
                    self.report_capture_rejection(pose_id, reason)
                    continue
                self.capture_runner.capture(
                    capture_id=capture_id,
                    pose_id=pose_id,
                    frames=frames,
                )
            if self.executor.current_pose_id != HANDOFF_POSE_ID:
                self._move_to(HANDOFF_POSE_ID)
            self.executor.begin_clean_release(
                operator_confirmed=confirm_release,
            )
            self._drive_until(ExecutorState.STOPPED)
            self.capture_runner.store.finalize()
        except Exception:
            self._emergency_release()
            raise

    def _move_to(self, pose_id: str) -> None:
        source = self.executor.current_pose_id
        if source is None:
            raise RuntimeError("executor has no current pose")
        self.executor.start_pose(
            pose_id,
            approval=self.validation_report.approval(source, pose_id),
            operator_confirmed=self.confirm_move(source, pose_id),
        )
        self._drive_until(ExecutorState.READY)

    def _drive_until(self, desired: ExecutorState) -> None:
        for _ in range(self.plan.maximum_control_steps_per_transition):
            if self.executor.state is desired:
                return
            if self.executor.state is ExecutorState.STOPPED:
                raise RuntimeError(
                    f"executor stopped before reaching {desired.value}: "
                    f"{self.executor.fault_reason or 'unknown reason'}"
                )
            self.control_step()
        raise RuntimeError(f"executor did not reach {desired.value} within step limit")

    def _emergency_release(self) -> None:
        if self.executor.state is ExecutorState.STOPPED:
            return
        self.executor.emergency_stop("session orchestration failed")
        for _ in range(self.plan.maximum_control_steps_per_transition):
            if self.executor.state is ExecutorState.STOPPED:
                return
            try:
                self.control_step()
            except (RuntimeError, TypeError, ValueError):
                return


@dataclass(frozen=True, slots=True)
class AutoCollectionExecutionResult:
    requested_accepted_count: int
    accepted_count: int
    rejected_count: int
    attempted_count: int
    route_exhausted: bool


class AuthoredCollectionOrchestrator:
    """Traverse an authored tree route and capture each target on first visit."""

    def __init__(
        self,
        *,
        executor: PoseExecutor,
        validation_report: ValidationReport,
        capture_runner: CaptureSessionRunner,
        frame_source: FrameSource,
        control_step: Callable[[], None],
        plan: AuthoredCollectionPlan,
        accepted_pose_count: int,
        report_progress: Callable[[str, int, int], None] | None = None,
        maximum_control_steps_per_transition: int = 10_000,
    ) -> None:
        if accepted_pose_count < 1:
            raise ValueError("accepted pose count must be positive")
        if accepted_pose_count > len(plan.capture_pose_ids):
            raise ValueError("accepted pose count cannot exceed authored target count")
        if maximum_control_steps_per_transition < 1:
            raise ValueError("maximum control steps must be positive")
        if executor.pose_set.content_sha256 != plan.content_sha256:
            raise ValueError("executor is not bound to the authored collection plan")
        if validation_report.pose_set_sha256 != plan.content_sha256:
            raise ValueError("validation report belongs to a different authored plan")
        if validation_report.content_sha256 != executor.approved_validation_report_sha256:
            raise ValueError("executor was not bound to this validation report")
        if not validation_report.passed:
            raise ValueError("cannot execute a failed validation report")
        for source, target in zip(plan.route_pose_ids, plan.route_pose_ids[1:]):
            if not validation_report.edge(source, target).passed:
                raise ValueError(f"authored route edge failed: {source}->{target}")
        self.executor = executor
        self.validation_report = validation_report
        self.capture_runner = capture_runner
        self.frame_source = frame_source
        self.control_step = control_step
        self.plan = plan
        self.accepted_pose_count = accepted_pose_count
        self.report_progress = report_progress or (lambda _message, _accepted, _rejected: None)
        self.maximum_control_steps_per_transition = maximum_control_steps_per_transition

    def run(
        self,
        *,
        confirm_acquisition: bool,
        confirm_release: bool,
        control_already_acquired: bool = False,
        retain_control_at_handoff: bool = False,
    ) -> AutoCollectionExecutionResult:
        accepted = 0
        rejected = 0
        attempted: set[str] = set()
        return_path = [HANDOFF_POSE_ID]
        route_exhausted = True
        try:
            if control_already_acquired:
                if (
                    self.executor.state is not ExecutorState.READY
                    or self.executor.current_pose_id != HANDOFF_POSE_ID
                ):
                    raise RuntimeError("pre-acquired collection control is not ready at handoff")
            else:
                self.executor.acquire(operator_confirmed=confirm_acquisition)
                self._drive_until(ExecutorState.READY)
            for pose_id in self.plan.route_pose_ids[1:]:
                if self.executor.current_pose_id != pose_id:
                    self._move_to(pose_id)
                    self._update_return_path(return_path, pose_id)
                if pose_id == HANDOFF_POSE_ID or pose_id in attempted:
                    continue
                attempted.add(pose_id)
                capture_id = f"capture_{len(attempted):03d}"
                self.report_progress(f"capturing {pose_id}", accepted, rejected)
                try:
                    frames = self.frame_source.capture_burst(
                        pose_id=pose_id,
                        capture_id=capture_id,
                    )
                except RecoverableCaptureError as error:
                    rejected += 1
                    self.capture_runner.reject(
                        capture_id=capture_id,
                        pose_id=pose_id,
                        reason=str(error),
                    )
                    self.report_progress(f"rejected {pose_id}: {error}", accepted, rejected)
                else:
                    self.capture_runner.capture(
                        capture_id=capture_id,
                        pose_id=pose_id,
                        frames=frames,
                    )
                    accepted += 1
                    self.report_progress(f"accepted {pose_id}", accepted, rejected)
                if accepted >= self.accepted_pose_count:
                    route_exhausted = False
                    self._return_to_handoff(
                        return_path,
                        accepted=accepted,
                        rejected=rejected,
                    )
                    break
            if self.executor.current_pose_id != HANDOFF_POSE_ID:
                self._return_to_handoff(
                    return_path,
                    accepted=accepted,
                    rejected=rejected,
                )
            if not retain_control_at_handoff:
                self.executor.begin_clean_release(operator_confirmed=confirm_release)
                self._drive_until(ExecutorState.STOPPED)
            self.capture_runner.store.finalize()
            return AutoCollectionExecutionResult(
                requested_accepted_count=self.accepted_pose_count,
                accepted_count=accepted,
                rejected_count=rejected,
                attempted_count=len(attempted),
                route_exhausted=route_exhausted,
            )
        except Exception:
            self._emergency_release()
            raise

    @staticmethod
    def _update_return_path(return_path: list[str], pose_id: str) -> None:
        """Maintain the active tree branch, cancelling completed backtracks."""

        if not return_path:
            raise RuntimeError("return path cannot be empty")
        if len(return_path) >= 2 and pose_id == return_path[-2]:
            return_path.pop()
        else:
            return_path.append(pose_id)

    def _return_to_handoff(
        self,
        return_path: list[str],
        *,
        accepted: int,
        rejected: int,
    ) -> None:
        move_count = len(return_path) - 1
        self.report_progress(
            f"accepted goal reached; returning to handoff over {move_count} validated moves",
            accepted,
            rejected,
        )
        while len(return_path) > 1:
            self._move_to(return_path[-2])
            return_path.pop()
            self.report_progress(
                f"returning to handoff; {len(return_path) - 1} validated moves remain",
                accepted,
                rejected,
            )
        self.report_progress(
            "handoff reached; releasing arm_sdk",
            accepted,
            rejected,
        )

    def _move_to(self, pose_id: str) -> None:
        source = self.executor.current_pose_id
        if source is None:
            raise RuntimeError("executor has no current pose")
        self.executor.start_pose(
            pose_id,
            approval=self.validation_report.approval(source, pose_id),
            operator_confirmed=True,
        )
        self._drive_until(ExecutorState.READY)

    def _drive_until(self, desired: ExecutorState) -> None:
        for _ in range(self.maximum_control_steps_per_transition):
            if self.executor.state is desired:
                return
            if self.executor.state is ExecutorState.STOPPED:
                raise RuntimeError(
                    f"executor stopped before reaching {desired.value}: "
                    f"{self.executor.fault_reason or 'unknown reason'}"
                )
            self.control_step()
        raise RuntimeError(f"executor did not reach {desired.value} within step limit")

    def _emergency_release(self) -> None:
        if self.executor.state is ExecutorState.STOPPED:
            return
        self.executor.emergency_stop("authored session orchestration failed")
        for _ in range(self.maximum_control_steps_per_transition):
            if self.executor.state is ExecutorState.STOPPED:
                return
            try:
                self.control_step()
            except (RuntimeError, TypeError, ValueError):
                return
