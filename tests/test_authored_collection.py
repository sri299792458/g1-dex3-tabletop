from __future__ import annotations

import json
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import pytest
from jsonschema import Draft202012Validator

from g1_aprilcube_calibration.authored_collection import (
    AuthoredCollectionPlan,
    AuthoredPoseTarget,
    validate_exposed_camera_views,
)
from g1_aprilcube_calibration.clock import ManualClock
from g1_aprilcube_calibration.executor_state_machine import (
    ExecutorConfig,
    ExecutorState,
    PoseExecutor,
)
from g1_aprilcube_calibration.pose_schema import HANDOFF_POSE_ID
from g1_aprilcube_calibration.pose_validator import (
    DirectedEdgeResult,
    ValidationReport,
)
from g1_aprilcube_calibration.session_runner import (
    AuthoredCollectionOrchestrator,
    CaptureSessionRunner,
    RecoverableCaptureError,
    ScheduledFrameSource,
)
from g1_aprilcube_calibration.transports.fake import FakeArmTransport


def _target(target_id: str, value: float) -> AuthoredPoseTarget:
    transform = np.eye(4)
    transform[2, 3] = 0.4
    return AuthoredPoseTarget(
        id=target_id,
        authored_calibration_q=(value,) * 7,
        desired_camera_T_cube=transform,
        ik_diagnostics={"translation_error_mm": 0.1},
    )


def _plan() -> AuthoredCollectionPlan:
    return AuthoredCollectionPlan(
        robot_model="g1_29dof_rev_1_0",
        mode_machine=5,
        urdf_sha256="a" * 64,
        calibration_arm="left",
        camera_profile_sha256="b" * 64,
        reference_full_q_sha256="c" * 64,
        targets=(_target("near", 0.04), _target("far", 0.08)),
        route_pose_ids=(
            HANDOFF_POSE_ID,
            "near",
            HANDOFF_POSE_ID,
            "far",
            HANDOFF_POSE_ID,
        ),
        capture_pose_ids=("near", "far"),
        generation_config={
            "source": "camera_frustum",
            "exposed_target_normal": [0.0, 0.0, -1.0],
        },
    )


def _edge(source: str, target: str) -> DirectedEdgeResult:
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


def _report(plan: AuthoredCollectionPlan) -> ValidationReport:
    return ValidationReport(
        pose_set_sha256=plan.content_sha256,
        urdf_sha256=plan.urdf_sha256,
        collision_config_sha256="d" * 64,
        reference_full_q_sha256="e" * 64,
        config={},
        edges=(
            _edge(HANDOFF_POSE_ID, "near"),
            _edge("near", HANDOFF_POSE_ID),
            _edge(HANDOFF_POSE_ID, "far"),
            _edge("far", HANDOFF_POSE_ID),
        ),
    )


def test_authored_plan_round_trip_keeps_targets_distinct_from_measurements() -> None:
    plan = _plan()

    rebuilt = AuthoredCollectionPlan.from_dict(plan.to_dict())
    schema = json.loads(
        (
            Path(__file__).parents[1]
            / "src/g1_aprilcube_calibration/schemas/authored_collection.schema.json"
        ).read_text()
    )
    Draft202012Validator(schema).validate(plan.to_dict())

    assert rebuilt.to_dict() == plan.to_dict()
    assert rebuilt.targets[0].command_calibration_q == (0.04,) * 7
    assert "measured_calibration_q" not in plan.to_dict()["targets"][0]


def test_authored_plan_rejects_capture_order_different_from_first_visits() -> None:
    plan = _plan()

    with pytest.raises(ValueError, match="first target visits"):
        AuthoredCollectionPlan(
            robot_model=plan.robot_model,
            mode_machine=plan.mode_machine,
            urdf_sha256=plan.urdf_sha256,
            calibration_arm=plan.calibration_arm,
            camera_profile_sha256=plan.camera_profile_sha256,
            reference_full_q_sha256=plan.reference_full_q_sha256,
            targets=plan.targets,
            route_pose_ids=plan.route_pose_ids,
            capture_pose_ids=("far", "near"),
            generation_config={},
        )


def test_exposed_camera_view_validation_rejects_palm_side_target() -> None:
    plan = replace(
        _plan(),
        generation_config={
            "source": "camera_frustum",
            "exposed_target_normal": [0.0, 0.0, 1.0],
        },
    )

    with pytest.raises(ValueError, match="palm-side hemisphere"):
        validate_exposed_camera_views(
            plan,
            exposed_target_normal=np.asarray([0.0, 0.0, 1.0]),
        )


def test_exposed_camera_view_validation_accepts_recorded_mount_side() -> None:
    plan = _plan()

    validate_exposed_camera_views(
        plan,
        exposed_target_normal=np.asarray([0.0, 0.0, -1.0]),
    )


@dataclass
class _StoredCapture:
    pose_id: str
    outcome: str


class _Store:
    def __init__(self) -> None:
        self.captures: list[_StoredCapture] = []
        self.finalized = False

    def append_capture(self, *, pose_id, outcome, reason, **_):
        assert reason
        self.captures.append(_StoredCapture(pose_id, outcome))

    def finalize(self):
        self.finalized = True


def _auto_orchestrator(
    frame_source,
    *,
    accepted_pose_count=1,
    report_progress=None,
):
    plan = _plan()
    report = _report(plan)
    clock = ManualClock(1.0)
    transport = FakeArmTransport(
        clock=clock,
        initial_full_q=np.zeros(29),
        tracking_velocity_rad_s=1.0,
    )
    executor = PoseExecutor(
        transport=transport,
        clock=clock,
        pose_set=plan,
        handoff_q=np.zeros(7),
        hold_q=np.zeros(7),
        approved_validation_report_sha256=report.content_sha256,
        config=ExecutorConfig(
            maximum_joint_velocity_rad_s=1.0,
            acquisition_ramp_s=0.05,
            release_ramp_s=0.05,
            settle_dwell_s=0.02,
            state_freshness_timeout_s=1.0,
            nominal_tick_period_s=0.02,
            control_gap_fault_s=0.10,
        ),
    )
    store = _Store()
    runner = CaptureSessionRunner(executor=executor, store=store)

    def step() -> None:
        clock.advance(0.005)
        executor.tick()

    orchestrator = AuthoredCollectionOrchestrator(
        executor=executor,
        validation_report=report,
        capture_runner=runner,
        frame_source=frame_source,
        control_step=step,
        plan=plan,
        accepted_pose_count=accepted_pose_count,
        report_progress=report_progress,
    )
    return executor, store, orchestrator


def test_auto_orchestrator_stops_at_goal_and_reverses_to_handoff() -> None:
    executor, store, orchestrator = _auto_orchestrator(ScheduledFrameSource({"near": (object(),)}))

    result = orchestrator.run(confirm_acquisition=True, confirm_release=True)

    assert result.accepted_count == 1
    assert result.attempted_count == 1
    assert result.route_exhausted is False
    assert store.captures == [_StoredCapture("near", "accepted")]
    assert store.finalized is True
    assert executor.current_pose_id == HANDOFF_POSE_ID


def test_auto_orchestrator_can_retain_preacquired_control_at_handoff() -> None:
    executor, store, orchestrator = _auto_orchestrator(ScheduledFrameSource({"near": (object(),)}))
    executor.acquire(operator_confirmed=True)
    for _ in range(20):
        if executor.state is ExecutorState.READY:
            break
        orchestrator.control_step()

    result = orchestrator.run(
        confirm_acquisition=False,
        confirm_release=False,
        control_already_acquired=True,
        retain_control_at_handoff=True,
    )

    assert result.accepted_count == 1
    assert store.finalized is True
    assert executor.state is ExecutorState.READY
    assert executor.current_pose_id == HANDOFF_POSE_ID


def test_auto_orchestrator_rejects_unrecovered_image_gap_and_continues() -> None:
    class GapThenFrames:
        def capture_burst(self, *, pose_id, capture_id):
            del capture_id
            if pose_id == "near":
                raise RecoverableCaptureError(
                    "timed out with 1/7 valid frames; last rejection: "
                    "timed out before seven qualified frames"
                )
            return (object(),)

    executor, store, orchestrator = _auto_orchestrator(GapThenFrames())

    result = orchestrator.run(confirm_acquisition=True, confirm_release=True)

    assert result.accepted_count == 1
    assert result.rejected_count == 1
    assert result.attempted_count == 2
    assert store.captures == [
        _StoredCapture("near", "rejected"),
        _StoredCapture("far", "accepted"),
    ]
    assert store.finalized is True
    assert executor.current_pose_id == HANDOFF_POSE_ID


def test_auto_orchestrator_returns_only_over_active_tree_branch() -> None:
    progress: list[str] = []
    executor, store, orchestrator = _auto_orchestrator(
        ScheduledFrameSource({"near": (object(),), "far": (object(),)}),
        accepted_pose_count=2,
        report_progress=lambda message, _accepted, _rejected: progress.append(message),
    )

    result = orchestrator.run(confirm_acquisition=True, confirm_release=True)

    moves = [
        event.reason.removeprefix("approved move to ")
        for event in executor.events
        if event.reason.startswith("approved move to ")
    ]
    assert moves == ["near", HANDOFF_POSE_ID, "far", HANDOFF_POSE_ID]
    assert result.accepted_count == 2
    assert store.finalized is True
    assert "accepted goal reached; returning to handoff over 1 validated moves" in (progress)
    assert progress[-1] == "handoff reached; releasing arm_sdk"
