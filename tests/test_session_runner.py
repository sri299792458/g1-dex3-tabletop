from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

from g1_aprilcube_calibration.clock import ManualClock
from g1_aprilcube_calibration.executor_state_machine import ExecutorConfig, PoseExecutor
from g1_aprilcube_calibration.pose_schema import (
    HANDOFF_POSE_ID,
    PoseAuditEvent,
    PoseRecord,
    PoseSet,
)
from g1_aprilcube_calibration.pose_validator import (
    DirectedEdgeResult,
    ValidationReport,
)
from g1_aprilcube_calibration.session_runner import (
    ApprovedSessionOrchestrator,
    CaptureSessionRunner,
    RecoverableCaptureError,
    ScheduledFrameSource,
    SessionExecutionPlan,
)
from g1_aprilcube_calibration.transports.fake import FakeArmTransport

UTC = "2026-08-02T12:00:00Z"


def make_pose(pose_id: str, value: float) -> PoseRecord:
    full = np.zeros(29)
    full[15:22] = value
    return PoseRecord(
        id=pose_id,
        group="dryrun",
        measured_calibration_q=(value,) * 7,
        measured_full_q=tuple(full),
        calibration_q_spread=(0.0,) * 7,
        recorded_at_utc=UTC,
        recorded_monotonic_s=1.0,
    )


def make_pose_set() -> PoseSet:
    result = PoseSet(
        "g1_29dof_rev_1_0",
        5,
        "a" * 64,
        "left",
    )
    for pose in (make_pose("near", 0.04), make_pose("far", 0.08)):
        result = result.with_pose(pose, PoseAuditEvent("add", pose.id, UTC))
    return result


def edge(source: str, target: str) -> DirectedEdgeResult:
    return DirectedEdgeResult(
        from_pose_id=source,
        to_pose_id=target,
        passed=True,
        sample_count=3,
        path_length_rad=0.1,
        estimated_duration_s=0.5,
        minimum_clearance_m=0.1,
        minimum_clearance_pair=("hand", "torso"),
        failures=(),
    )


def report(pose_set: PoseSet) -> ValidationReport:
    return ValidationReport(
        pose_set_sha256=pose_set.content_sha256,
        urdf_sha256=pose_set.urdf_sha256,
        collision_config_sha256="b" * 64,
        reference_full_q_sha256="c" * 64,
        config={},
        edges=(
            edge(HANDOFF_POSE_ID, "near"),
            edge("near", "far"),
            edge("far", HANDOFF_POSE_ID),
        ),
    )


@dataclass
class StoredCapture:
    capture_id: str
    pose_id: str
    outcome: str
    frames: tuple


class Store:
    def __init__(self):
        self.captures = []
        self.finalized = False

    def append_capture(self, *, capture_id, pose_id, outcome, reason, frames=(), **_):
        assert reason
        self.captures.append(StoredCapture(capture_id, pose_id, outcome, tuple(frames)))

    def finalize(self):
        self.finalized = True


def make_subject(*, initial_left=0.0):
    pose_set = make_pose_set()
    validation = report(pose_set)
    clock = ManualClock(1.0)
    initial = np.zeros(29)
    initial[15:22] = initial_left
    transport = FakeArmTransport(
        clock=clock,
        initial_full_q=initial,
        tracking_velocity_rad_s=1,
    )
    executor = PoseExecutor(
        transport=transport,
        clock=clock,
        pose_set=pose_set,
        handoff_q=(0.0,) * 7,
        hold_q=(0.0,) * 7,
        approved_validation_report_sha256=validation.content_sha256,
        config=ExecutorConfig(
            maximum_joint_velocity_rad_s=0.2,
            activation_position_tolerance_rad=0.005,
            held_arm_position_tolerance_rad=0.005,
            settled_position_spread_rad=0.002,
            settle_dwell_s=0.04,
            state_freshness_timeout_s=0.1,
            nominal_tick_period_s=0.02,
            control_gap_fault_s=0.25,
            acquisition_ramp_s=0.04,
            release_ramp_s=0.04,
            motion_timeout_s=2,
        ),
    )
    store = Store()
    runner = CaptureSessionRunner(executor=executor, store=store)
    source = ScheduledFrameSource({"near": ("n",), "far": ("f",)})

    def step():
        transport.step(0.02)
        executor.tick()

    orchestrator = ApprovedSessionOrchestrator(
        executor=executor,
        validation_report=validation,
        capture_runner=runner,
        frame_source=source,
        control_step=step,
        confirm_move=lambda _source, _target: True,
        plan=SessionExecutionPlan(("near", "far")),
    )
    return transport, executor, store, orchestrator


def test_fake_end_to_end_plan_captures_returns_handoff_and_releases():
    transport, executor, store, orchestrator = make_subject()
    orchestrator.run(confirm_acquisition=True, confirm_release=True)
    assert [capture.pose_id for capture in store.captures] == ["near", "far"]
    assert [capture.frames for capture in store.captures] == [("n",), ("f",)]
    assert store.finalized
    assert transport.closed
    assert transport.commands[-1].weight == 0
    assert executor.current_pose_id == HANDOFF_POSE_ID


def test_recoverable_camera_failure_rejects_pose_and_continues_route():
    transport, executor, store, orchestrator = make_subject()
    rejections = []

    class RejectFirstPose:
        def capture_burst(self, *, pose_id, capture_id):
            del capture_id
            if pose_id == "near":
                raise RecoverableCaptureError("visual quality remained red")
            return ("f",)

    orchestrator.frame_source = RejectFirstPose()
    orchestrator.report_capture_rejection = lambda pose_id, reason: rejections.append(
        (pose_id, reason)
    )

    orchestrator.run(confirm_acquisition=True, confirm_release=True)

    assert [capture.outcome for capture in store.captures] == [
        "rejected",
        "accepted",
    ]
    assert [capture.frames for capture in store.captures] == [(), ("f",)]
    assert rejections == [("near", "visual quality remained red")]
    assert store.finalized
    assert transport.commands[-1].weight == 0
    assert not transport.commands[-1].emergency_release
    assert executor.current_pose_id == HANDOFF_POSE_ID


def test_capture_storage_does_not_mask_an_executor_fault() -> None:
    _, executor, _, orchestrator = make_subject()

    class FaultDuringWrite:
        def append_capture(self, **_):
            executor.emergency_stop("control loop gap 0.055s exceeds 0.050s")

        def finalize(self):
            raise AssertionError("faulted capture must not finalize")

    orchestrator.capture_runner.store = FaultDuringWrite()

    with pytest.raises(
        RuntimeError,
        match=(
            r"executor faulted during capture: control loop gap "
            r"0\.055s exceeds 0\.050s"
        ),
    ):
        orchestrator.run(confirm_acquisition=True, confirm_release=True)


def test_handoff_acquisition_rejects_measured_mismatch_before_publish():
    transport, _, store, orchestrator = make_subject(initial_left=0.02)
    with pytest.raises(ValueError, match="differs from handoff pose"):
        orchestrator.run(confirm_acquisition=True, confirm_release=True)
    assert transport.commands == []
    assert transport.closed
    assert not store.finalized


def test_move_confirmation_refusal_emergency_releases():
    transport, executor, store, orchestrator = make_subject()
    orchestrator.confirm_move = lambda _source, _target: False
    with pytest.raises(ValueError, match="operator confirmation"):
        orchestrator.run(confirm_acquisition=True, confirm_release=True)
    assert transport.commands[-1].weight == 0
    assert transport.commands[-1].emergency_release
    assert executor.fault_reason == "session orchestration failed"
    assert not store.finalized


def test_plan_allows_revisiting_an_anchor_pose():
    plan = SessionExecutionPlan(("near", "far", "near"))
    assert plan.capture_pose_ids.count("near") == 2
