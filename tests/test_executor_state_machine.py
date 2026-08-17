from dataclasses import replace

import numpy as np
import pytest

from g1_aprilcube_calibration.clock import ManualClock
from g1_aprilcube_calibration.executor_state_machine import (
    ExecutorConfig,
    ExecutorState,
    PoseExecutor,
    TransitionApproval,
)
from g1_aprilcube_calibration.pose_schema import (
    HANDOFF_POSE_ID,
    PoseAuditEvent,
    PoseRecord,
    PoseSet,
)
from g1_aprilcube_calibration.transports.fake import FakeArmTransport
from g1_dex3_tabletop.mpc_command_buffer import MPCCommandWindow

UTC = "2026-08-02T12:00:00Z"
REPORT_HASH = "b" * 64


def pose(pose_id: str, calibration_q: np.ndarray) -> PoseRecord:
    full = np.zeros(29)
    full[15:22] = calibration_q
    return PoseRecord(
        id=pose_id,
        group="test",
        measured_calibration_q=tuple(calibration_q),
        measured_full_q=tuple(full),
        calibration_q_spread=(0.0,) * 7,
        recorded_at_utc=UTC,
        recorded_monotonic_s=1.0,
    )


def pose_set() -> PoseSet:
    result = PoseSet(
        robot_model="g1_29dof_rev_1_0",
        mode_machine=5,
        urdf_sha256="a" * 64,
        calibration_arm="left",
    )
    for item in (pose("pose_001", np.full(7, 0.08)),):
        result = result.with_pose(item, PoseAuditEvent("add", item.id, UTC))
    return result


def config() -> ExecutorConfig:
    return ExecutorConfig(
        maximum_joint_velocity_rad_s=0.2,
        motion_position_tolerance_rad=0.08,
        ownership_transition_position_tolerance_rad=0.005,
        activation_position_tolerance_rad=0.005,
        held_arm_position_tolerance_rad=0.005,
        settled_position_spread_rad=0.002,
        settle_dwell_s=0.06,
        state_freshness_timeout_s=0.1,
        nominal_tick_period_s=0.02,
        control_gap_fault_s=0.25,
        acquisition_ramp_s=0.06,
        release_ramp_s=0.06,
        motion_timeout_s=2.0,
    )


def subject() -> tuple[ManualClock, FakeArmTransport, PoseExecutor]:
    clock = ManualClock(1.0)
    transport = FakeArmTransport(clock=clock, initial_full_q=np.zeros(29))
    executor = PoseExecutor(
        transport=transport,
        clock=clock,
        pose_set=pose_set(),
        handoff_q=(0.0,) * 7,
        hold_q=(0.0,) * 7,
        approved_validation_report_sha256=REPORT_HASH,
        config=config(),
    )
    return clock, transport, executor


def approval(executor: PoseExecutor, source: str, target: str, **kwargs):
    values = {
        "from_pose_id": source,
        "to_pose_id": target,
        "pose_set_sha256": executor.pose_set.content_sha256,
        "validation_report_sha256": REPORT_HASH,
        "passed": True,
    }
    values.update(kwargs)
    return TransitionApproval(**values)


def test_executor_adopts_identical_full_weight_owned_command() -> None:
    _clock, transport, executor = subject()

    executor.adopt_owned_control(previous_command_q14=np.zeros(14))

    assert executor.state is ExecutorState.READY
    assert executor.current_pose_id == HANDOFF_POSE_ID
    assert transport.commands[-1].weight == pytest.approx(1.0)
    np.testing.assert_allclose(transport.commands[-1].q14, 0.0)


def test_executor_rejects_changed_owned_command() -> None:
    _clock, _transport, executor = subject()
    changed = np.zeros(14)
    changed[0] = 0.01

    with pytest.raises(ValueError, match="handoff command differs"):
        executor.adopt_owned_control(previous_command_q14=changed)


def advance_until(
    transport: FakeArmTransport,
    executor: PoseExecutor,
    desired: ExecutorState,
    *,
    limit: int = 200,
) -> None:
    for _ in range(limit):
        if executor.state is desired:
            return
        transport.step(0.02)
        executor.tick()
    raise AssertionError(f"executor did not reach {desired}; state={executor.state}")


def test_measured_handoff_changes_only_blend_weight() -> None:
    initial = np.linspace(-0.2, 0.2, 29)
    clock = ManualClock(1.0)
    transport = FakeArmTransport(clock=clock, initial_full_q=initial)
    executor = PoseExecutor(
        transport=transport,
        clock=clock,
        pose_set=pose_set(),
        handoff_q=initial[15:22],
        hold_q=initial[22:29],
        approved_validation_report_sha256=REPORT_HASH,
        config=config(),
    )
    expected_q14 = tuple(initial[15:29])

    executor.acquire(operator_confirmed=True)
    advance_until(transport, executor, ExecutorState.READY)

    assert transport.commands[0].weight == 0.0
    assert transport.commands[-1].weight == 1.0
    assert all(command.q14 == expected_q14 for command in transport.commands)
    np.testing.assert_allclose(transport.position, initial)

    executor.begin_clean_release(operator_confirmed=True)
    advance_until(transport, executor, ExecutorState.STOPPED)
    assert transport.commands[-1].weight == 0.0
    assert all(command.q14 == expected_q14 for command in transport.commands)


def test_executor_seeds_and_attaches_gravity_feedforward_to_every_command() -> None:
    class FakeGravity:
        def __init__(self) -> None:
            self.reference = None

        def seed_reference(self, full_q) -> None:
            self.reference = np.asarray(full_q).copy()

        def torque_for(self, q14):
            assert self.reference is not None
            return np.asarray(q14) + np.arange(14)

    initial = np.linspace(-0.2, 0.2, 29)
    clock = ManualClock(1.0)
    transport = FakeArmTransport(clock=clock, initial_full_q=initial)
    gravity = FakeGravity()
    executor = PoseExecutor(
        transport=transport,
        clock=clock,
        pose_set=pose_set(),
        handoff_q=initial[15:22],
        hold_q=initial[22:29],
        approved_validation_report_sha256=REPORT_HASH,
        config=config(),
        gravity_feedforward=gravity,
    )

    executor.acquire(operator_confirmed=True)
    advance_until(transport, executor, ExecutorState.READY)

    np.testing.assert_allclose(gravity.reference, initial)
    for command in transport.commands:
        np.testing.assert_allclose(
            command.tau_ff14,
            np.asarray(command.q14) + np.arange(14),
        )
    np.testing.assert_allclose(transport.position, initial)

    executor.begin_clean_release(operator_confirmed=True)
    advance_until(transport, executor, ExecutorState.STOPPED)
    assert transport.commands[-1].weight == 0.0


def test_acquisition_faults_if_either_arm_moves_beyond_transition_limit() -> None:
    clock, transport, executor = subject()
    executor.acquire(operator_confirmed=True)
    transport.position[15] += 0.006
    clock.advance(0.02)

    state = executor.tick()

    assert state is ExecutorState.FAULT
    assert "during ownership acquisition" in executor.fault_reason
    assert executor.maximum_acquisition_position_change_rad == pytest.approx(0.006)


def test_acquisition_move_capture_handoff_and_clean_release() -> None:
    _, transport, executor = subject()
    assert transport.commands == []
    executor.acquire(operator_confirmed=True)

    assert executor.state is ExecutorState.ACQUIRING
    assert transport.commands[0].weight == 0.0
    assert transport.commands[0].q14 == (0.0,) * 14
    advance_until(transport, executor, ExecutorState.READY)
    assert np.allclose(transport.position, 0.0)

    executor.start_pose(
        "pose_001",
        approval=approval(executor, HANDOFF_POSE_ID, "pose_001"),
        operator_confirmed=True,
    )
    first_motion_command = len(transport.commands)
    advance_until(transport, executor, ExecutorState.READY)

    commanded = np.asarray([command.q14 for command in transport.commands[first_motion_command:]])
    increments = np.abs(np.diff(commanded, axis=0))
    assert np.max(increments) <= 0.2 * 0.02 + 1e-12
    assert np.allclose(transport.position[15:22], 0.08, atol=0.005)
    assert np.allclose(transport.position[22:29], 0.0, atol=0.005)
    executor.begin_capture()
    with pytest.raises(RuntimeError, match="move"):
        executor.start_pose(
            HANDOFF_POSE_ID,
            approval=approval(executor, "pose_001", HANDOFF_POSE_ID),
            operator_confirmed=True,
        )
    executor.finish_capture(outcome="accepted")
    assert executor.state is ExecutorState.HOLDING
    executor.begin_capture()
    executor.finish_capture(outcome="retry accepted")

    executor.start_pose(
        HANDOFF_POSE_ID,
        approval=approval(executor, "pose_001", HANDOFF_POSE_ID),
        operator_confirmed=True,
    )
    advance_until(transport, executor, ExecutorState.READY)
    executor.begin_clean_release(operator_confirmed=True)
    advance_until(transport, executor, ExecutorState.STOPPED)

    assert transport.closed
    assert transport.commands[-1].weight == 0.0
    assert not transport.commands[-1].emergency_release
    assert [event.state for event in executor.events].count(ExecutorState.READY) == 3


def test_executor_follows_frozen_time_parameterized_trajectory() -> None:
    _clock, transport, executor = subject()
    executor.acquire(operator_confirmed=True)
    advance_until(transport, executor, ExecutorState.READY)

    executor.start_trajectory(
        from_pose_id=HANDOFF_POSE_ID,
        to_pose_id="pose_001",
        sample_time_s=(0.0, 0.1, 0.2),
        command_q_rad=((0.0,) * 7, (0.02,) * 7, (0.04,) * 7),
        plan_sha256=REPORT_HASH,
        operator_confirmed=True,
    )
    motion_start = len(transport.commands)
    advance_until(transport, executor, ExecutorState.READY)

    commanded = np.asarray([command.q14[:7] for command in transport.commands[motion_start:]])
    assert np.max(np.abs(np.diff(commanded, axis=0))) <= 0.2 * 0.02 + 1e-12
    np.testing.assert_allclose(commanded[-1], 0.04)
    assert executor.current_pose_id == "pose_001"


def test_executor_rejects_trajectory_from_a_different_plan() -> None:
    _, transport, executor = subject()
    executor.acquire(operator_confirmed=True)
    advance_until(transport, executor, ExecutorState.READY)

    with pytest.raises(ValueError, match="different approved plan"):
        executor.start_trajectory(
            from_pose_id=HANDOFF_POSE_ID,
            to_pose_id="pose_001",
            sample_time_s=(0.0, 0.2),
            command_q_rad=((0.0,) * 7, (0.04,) * 7),
            plan_sha256="c" * 64,
            operator_confirmed=True,
        )


def mpc_window(
    clock: ManualClock,
    *,
    generation: int,
    start_q: np.ndarray,
    end_q: np.ndarray,
    terminal: bool,
    feasible: bool = True,
) -> MPCCommandWindow:
    midpoint = (start_q + end_q) / 2.0
    return MPCCommandWindow(
        generation=generation,
        plan_sha256=REPORT_HASH,
        state_monotonic_s=clock.monotonic(),
        sample_time_s=(0.0, 0.1, 0.2),
        command_q_rad=(tuple(start_q), tuple(midpoint), tuple(end_q)),
        feasible=feasible,
        terminal=terminal,
        solve_time_s=0.01,
        diagnostics={"source": "executor-test"},
    )


def test_executor_streams_rebased_mpc_windows_then_settles_terminal_window() -> None:
    clock, transport, executor = subject()
    executor.acquire(operator_confirmed=True)
    advance_until(transport, executor, ExecutorState.READY)

    first = mpc_window(
        clock,
        generation=0,
        start_q=np.zeros(7),
        end_q=np.full(7, 0.01),
        terminal=False,
    )
    executor.start_streaming_trajectory(
        from_pose_id=HANDOFF_POSE_ID,
        to_pose_id="pose_001",
        window=first,
        plan_sha256=REPORT_HASH,
        operator_confirmed=True,
    )
    for _ in range(5):
        transport.step(0.02)
        executor.tick()
    active = executor.calibration_command_q
    assert np.allclose(active, 0.005)

    # The worker solved from the older zero command. Installation rebases its
    # t=0 sample atomically to the command that is now active.
    terminal = mpc_window(
        clock,
        generation=1,
        start_q=np.zeros(7),
        end_q=np.full(7, 0.02),
        terminal=True,
    )
    accepted = executor.update_streaming_trajectory(window=terminal)
    np.testing.assert_allclose(accepted.command_q_rad[0], active)
    advance_until(transport, executor, ExecutorState.READY)

    assert executor.current_pose_id == "pose_001"
    np.testing.assert_allclose(executor.calibration_command_q, 0.02)


def test_executor_faults_when_nonterminal_mpc_window_is_not_replenished() -> None:
    clock, transport, executor = subject()
    executor.acquire(operator_confirmed=True)
    advance_until(transport, executor, ExecutorState.READY)
    executor.start_streaming_trajectory(
        from_pose_id=HANDOFF_POSE_ID,
        to_pose_id="pose_001",
        window=mpc_window(
            clock,
            generation=0,
            start_q=np.zeros(7),
            end_q=np.full(7, 0.01),
            terminal=False,
        ),
        plan_sha256=REPORT_HASH,
        operator_confirmed=True,
    )

    for _ in range(24):
        transport.step(0.02)
        executor.tick()
        if executor.state is ExecutorState.FAULT:
            break

    assert executor.state is ExecutorState.FAULT
    assert "MPC command window expired" in (executor.fault_reason or "")


def test_executor_rejects_infeasible_mpc_window_before_motion() -> None:
    clock, transport, executor = subject()
    executor.acquire(operator_confirmed=True)
    advance_until(transport, executor, ExecutorState.READY)

    with pytest.raises(ValueError, match="infeasible"):
        executor.start_streaming_trajectory(
            from_pose_id=HANDOFF_POSE_ID,
            to_pose_id="pose_001",
            window=mpc_window(
                clock,
                generation=0,
                start_q=np.zeros(7),
                end_q=np.full(7, 0.01),
                terminal=False,
                feasible=False,
            ),
            plan_sha256=REPORT_HASH,
            operator_confirmed=True,
        )

    assert executor.state is ExecutorState.READY


def test_executor_never_replaces_active_stream_with_infeasible_mpc_window() -> None:
    clock, transport, executor = subject()
    executor.acquire(operator_confirmed=True)
    advance_until(transport, executor, ExecutorState.READY)
    executor.start_streaming_trajectory(
        from_pose_id=HANDOFF_POSE_ID,
        to_pose_id="pose_001",
        window=mpc_window(
            clock,
            generation=0,
            start_q=np.zeros(7),
            end_q=np.full(7, 0.01),
            terminal=False,
        ),
        plan_sha256=REPORT_HASH,
        operator_confirmed=True,
    )
    for _ in range(5):
        transport.step(0.02)
        executor.tick()
    generation_before = executor.streaming_trajectory_status()["generation"]

    with pytest.raises(ValueError, match="infeasible"):
        executor.update_streaming_trajectory(
            window=mpc_window(
                clock,
                generation=1,
                start_q=np.zeros(7),
                end_q=np.full(7, 0.02),
                terminal=False,
                feasible=False,
            )
        )

    assert executor.state is ExecutorState.MOVING
    assert executor.streaming_trajectory_status()["generation"] == generation_before


def test_loaded_handoff_can_atomically_install_a_new_validated_plan() -> None:
    clock, transport, executor = subject()

    class FakeGravity:
        def __init__(self) -> None:
            self.reference = None

        def seed_reference(self, full_q) -> None:
            self.reference = np.asarray(full_q).copy()

        def torque_for(self, q14):
            return np.zeros(14)

    gravity = FakeGravity()
    executor.gravity_feedforward = gravity
    executor.acquire(operator_confirmed=True)
    advance_until(transport, executor, ExecutorState.READY)
    loaded = transport.position.copy()
    loaded[12] = 0.004
    loaded[15:22] = 0.003
    transport.position[:] = loaded
    clock.advance(0.01)
    reference = transport.observe()
    replacement = pose_set()
    report_hash = "c" * 64

    executor.install_validated_plan(
        pose_set=replacement,
        approved_validation_report_sha256=report_hash,
        validated_reference_state=reference,
    )

    assert executor.state is ExecutorState.READY
    assert executor.current_pose_id == HANDOFF_POSE_ID
    assert executor.pose_set is replacement
    assert executor.approved_validation_report_sha256 == report_hash
    np.testing.assert_allclose(executor.handoff_q, loaded[15:22])
    np.testing.assert_allclose(transport.commands[-1].q14, loaded[15:29])
    np.testing.assert_allclose(gravity.reference, loaded)
    assert "validated plan installed" in executor.events[-1].reason


def test_loaded_plan_install_rejects_state_drift_after_validation() -> None:
    clock, transport, executor = subject()
    executor.acquire(operator_confirmed=True)
    advance_until(transport, executor, ExecutorState.READY)
    validated = transport.observe()
    transport.position[12] += executor.config.settled_position_spread_rad + 0.001
    clock.advance(0.01)

    with pytest.raises(ValueError, match="changed after path validation"):
        executor.install_validated_plan(
            pose_set=pose_set(),
            approved_validation_report_sha256="c" * 64,
            validated_reference_state=validated,
        )

    assert executor.pose_set.content_sha256 != ""
    assert executor.approved_validation_report_sha256 == REPORT_HASH


def test_reached_boundary_can_atomically_replace_only_the_remaining_plan() -> None:
    _clock, transport, executor = subject()
    executor.acquire(operator_confirmed=True)
    advance_until(transport, executor, ExecutorState.READY)
    executor.start_pose(
        "pose_001",
        approval=approval(executor, HANDOFF_POSE_ID, "pose_001"),
        operator_confirmed=True,
    )
    advance_until(transport, executor, ExecutorState.READY)
    reference = transport.observe()
    replacement = pose_set()

    executor.replace_validated_remaining_plan(
        pose_set=replacement,
        approved_validation_report_sha256="c" * 64,
        validated_reference_state=reference,
    )

    assert executor.current_pose_id == "pose_001"
    assert executor.pose_set is replacement
    assert executor.approved_validation_report_sha256 == "c" * 64
    assert "boundary-corrected remaining plan installed" in executor.events[-1].reason


def test_remaining_plan_replacement_rejects_boundary_command_discontinuity() -> None:
    _clock, _transport, executor = subject()
    executor.adopt_owned_control(previous_command_q14=np.zeros(14))
    executor.current_pose_id = "pose_001"
    reference = executor.transport.observe()
    changed = replace(
        pose_set(),
        poses=(pose("pose_001", np.full(7, 0.081)),),
    )

    with pytest.raises(ValueError, match="replacement boundary differs"):
        executor.replace_validated_remaining_plan(
            pose_set=changed,
            approved_validation_report_sha256="c" * 64,
            validated_reference_state=reference,
        )

    assert executor.approved_validation_report_sha256 == REPORT_HASH


def test_motion_uses_replay_target_without_changing_measured_pose() -> None:
    clock = ManualClock(1.0)
    transport = FakeArmTransport(clock=clock, initial_full_q=np.zeros(29))
    original = pose_set()
    adjusted_record = replace(original.poses[0], replay_calibration_q=(0.04,) * 7)
    adjusted = replace(original, poses=(adjusted_record,))
    executor = PoseExecutor(
        transport=transport,
        clock=clock,
        pose_set=adjusted,
        handoff_q=(0.0,) * 7,
        hold_q=(0.0,) * 7,
        approved_validation_report_sha256=REPORT_HASH,
        config=config(),
    )
    executor.acquire(operator_confirmed=True)
    advance_until(transport, executor, ExecutorState.READY)

    executor.start_pose(
        "pose_001",
        approval=approval(executor, HANDOFF_POSE_ID, "pose_001"),
        operator_confirmed=True,
    )
    advance_until(transport, executor, ExecutorState.READY)

    assert adjusted_record.measured_calibration_q == (0.08,) * 7
    assert np.allclose(transport.position[15:22], 0.04, atol=0.005)


def test_every_move_requires_exact_approved_pose_and_hash() -> None:
    _, transport, executor = subject()
    executor.acquire(operator_confirmed=True)
    advance_until(transport, executor, ExecutorState.READY)

    with pytest.raises(ValueError, match="operator"):
        executor.start_pose(
            "pose_001",
            approval=approval(executor, HANDOFF_POSE_ID, "pose_001"),
            operator_confirmed=False,
        )
    with pytest.raises(ValueError, match="pose-set hash"):
        executor.start_pose(
            "pose_001",
            approval=approval(
                executor,
                HANDOFF_POSE_ID,
                "pose_001",
                pose_set_sha256="c" * 64,
            ),
            operator_confirmed=True,
        )
    with pytest.raises(ValueError, match="validation-report"):
        executor.start_pose(
            "pose_001",
            approval=approval(
                executor,
                HANDOFF_POSE_ID,
                "pose_001",
                validation_report_sha256="c" * 64,
            ),
            operator_confirmed=True,
        )
    with pytest.raises(ValueError, match="did not pass"):
        executor.start_pose(
            "pose_001",
            approval=approval(executor, HANDOFF_POSE_ID, "pose_001", passed=False),
            operator_confirmed=True,
        )
    assert executor.state is ExecutorState.READY


def test_drift_of_held_right_arm_faults_during_left_arm_control() -> None:
    clock, transport, executor = subject()
    executor.acquire(operator_confirmed=True)
    advance_until(transport, executor, ExecutorState.READY)
    transport.position[22] = 0.006
    clock.advance(0.01)
    executor.tick()
    assert executor.state is ExecutorState.FAULT
    assert executor.fault_reason is not None
    assert "held right arm" in executor.fault_reason


def test_acquisition_reuses_teaching_loaded_equilibrium_hold() -> None:
    clock, transport, executor = subject()
    executor.config = replace(
        executor.config,
        ownership_transition_position_tolerance_rad=0.05,
    )
    executor.acquire(operator_confirmed=True)

    for _ in range(5):
        if executor.state is ExecutorState.READY:
            break
        transport.position[22] = 0.0228
        clock.advance(0.02)
        executor.tick()

    assert executor.state is ExecutorState.READY
    assert all(command.q14[7] == 0.0 for command in transport.commands)

    transport.position[22] = 0.0228
    clock.advance(0.01)
    executor.tick()
    assert executor.state is ExecutorState.READY

    transport.position[22] = 0.0280
    clock.advance(0.01)
    executor.tick()
    assert executor.state is ExecutorState.FAULT
    assert executor.fault_reason is not None
    assert "held right arm drifted by 0.0052rad; limit is 0.0050rad" in (executor.fault_reason)


def test_clean_release_allows_loaded_equilibrium_to_unload() -> None:
    clock, transport, executor = subject()
    executor.config = replace(
        executor.config,
        ownership_transition_position_tolerance_rad=0.05,
    )
    executor.acquire(operator_confirmed=True)
    for _ in range(5):
        if executor.state is ExecutorState.READY:
            break
        transport.position[22] = 0.0228
        clock.advance(0.02)
        executor.tick()
    assert executor.state is ExecutorState.READY

    executor.begin_clean_release(operator_confirmed=True)
    for right_position in (0.015, 0.008, 0.0, 0.0):
        if executor.state is ExecutorState.STOPPED:
            break
        transport.position[22] = right_position
        clock.advance(0.02)
        executor.tick()

    assert executor.state is ExecutorState.STOPPED
    assert transport.commands[-1].weight == 0.0


def test_pose_can_settle_at_commissioned_loaded_tracking_offset() -> None:
    clock, transport, executor = subject()
    executor.acquire(operator_confirmed=True)
    advance_until(transport, executor, ExecutorState.READY)
    executor.start_pose(
        "pose_001",
        approval=approval(executor, HANDOFF_POSE_ID, "pose_001"),
        operator_confirmed=True,
    )

    loaded_offset = 0.075
    for _ in range(100):
        clock.advance(0.01)
        if transport.commands:
            command = np.asarray(transport.commands[-1].q14[:7])
            transport.position[15:22] = command - loaded_offset
        executor.tick()
        if executor.state is ExecutorState.READY:
            break

    assert executor.state is ExecutorState.READY
    assert np.max(np.abs(transport.position[15:22] - 0.08)) == pytest.approx(loaded_offset)
    assert executor.events[-1].reason.endswith("endpoint error 0.0750rad passed")


def test_stationary_arm_that_ignores_a_large_command_faults_endpoint_gate() -> None:
    clock, transport, executor = subject()
    executor.acquire(operator_confirmed=True)
    advance_until(transport, executor, ExecutorState.READY)
    transport.tracking_velocity_rad_s = 0.0
    large_pose = pose("large_pose", np.full(7, 0.3))
    executor.pose_set = PoseSet(
        robot_model=executor.pose_set.robot_model,
        mode_machine=executor.pose_set.mode_machine,
        urdf_sha256=executor.pose_set.urdf_sha256,
        calibration_arm=executor.pose_set.calibration_arm,
        poses=(large_pose,),
        audit_log=(PoseAuditEvent("add", large_pose.id, UTC),),
    )
    executor.start_pose(
        "large_pose",
        approval=approval(executor, HANDOFF_POSE_ID, "large_pose"),
        operator_confirmed=True,
    )

    for _ in range(400):
        clock.advance(0.01)
        executor.tick()
        if executor.state is ExecutorState.FAULT:
            break

    assert executor.state is ExecutorState.FAULT
    assert executor.fault_reason is not None
    assert "motion settled outside the required endpoint tolerance" in (executor.fault_reason)
    assert "position error 0.3000rad" in executor.fault_reason


def test_attained_pose_policy_records_large_settled_endpoint_error() -> None:
    clock, transport, executor = subject()
    executor.config = replace(
        executor.config,
        require_motion_endpoint_tolerance=False,
    )
    executor.acquire(operator_confirmed=True)
    advance_until(transport, executor, ExecutorState.READY)
    transport.tracking_velocity_rad_s = 0.0
    large_pose = pose("large_pose", np.full(7, 0.3))
    executor.pose_set = PoseSet(
        robot_model=executor.pose_set.robot_model,
        mode_machine=executor.pose_set.mode_machine,
        urdf_sha256=executor.pose_set.urdf_sha256,
        calibration_arm=executor.pose_set.calibration_arm,
        poses=(large_pose,),
        audit_log=(PoseAuditEvent("add", large_pose.id, UTC),),
    )
    executor.start_pose(
        "large_pose",
        approval=approval(executor, HANDOFF_POSE_ID, "large_pose"),
        operator_confirmed=True,
    )

    for _ in range(400):
        clock.advance(0.01)
        executor.tick()
        if executor.state is ExecutorState.READY:
            break

    assert executor.state is ExecutorState.READY
    attained = executor.observe_state()
    assert np.allclose(attained.position, transport.position)
    assert attained.source_sequence is not None
    assert executor.events[-1].reason.endswith(
        "endpoint error 0.3000rad recorded; endpoint tolerance disabled"
    )


def test_reverse_move_starts_from_nominal_command_not_measured_offset() -> None:
    clock, transport, executor = subject()
    executor.config = replace(
        executor.config,
        require_motion_endpoint_tolerance=False,
    )
    executor.acquire(operator_confirmed=True)
    advance_until(transport, executor, ExecutorState.READY)

    executor.start_pose(
        "pose_001",
        approval=approval(executor, HANDOFF_POSE_ID, "pose_001"),
        operator_confirmed=True,
    )
    advance_until(transport, executor, ExecutorState.READY)
    assert np.allclose(transport.commands[-1].q14[:7], 0.08)

    # Reproduce the kind of loaded tracking offset seen during the tabletop
    # run.  The reverse command must remain continuous from the nominal 0.08
    # target instead of constructing a new segment from this measured value.
    transport.position[15:22] = 0.05
    commands_before_reverse = len(transport.commands)
    executor.start_pose(
        HANDOFF_POSE_ID,
        approval=approval(executor, "pose_001", HANDOFF_POSE_ID),
        operator_confirmed=True,
    )
    clock.advance(0.02)
    executor.tick()

    first_reverse = np.asarray(transport.commands[commands_before_reverse].q14[:7])
    np.testing.assert_allclose(first_reverse, 0.076)
    assert np.max(np.abs(first_reverse - 0.05)) > 0.02


def test_short_move_cannot_settle_before_command_completion() -> None:
    clock, transport, executor = subject()
    executor.acquire(operator_confirmed=True)
    advance_until(transport, executor, ExecutorState.READY)
    executor.start_pose(
        "pose_001",
        approval=approval(executor, HANDOFF_POSE_ID, "pose_001"),
        operator_confirmed=True,
    )

    for _ in range(60):
        clock.advance(0.01)
        transport.position[15:22] = 0.0
        executor.tick()
        if np.allclose(transport.commands[-1].q14[:7], 0.08):
            break

    assert np.allclose(transport.commands[-1].q14[:7], 0.08)
    assert executor.state is ExecutorState.SETTLING
    assert executor._last_settle_elapsed_s == 0.0


def test_measured_stationarity_completion_does_not_loosen_ownership_transition() -> None:
    clock, transport, executor = subject()
    executor.config = replace(
        executor.config,
        ownership_transition_position_tolerance_rad=0.05,
    )
    executor.acquire(operator_confirmed=True)
    transport.position[22] = 0.051
    clock.advance(0.01)
    executor.tick()

    assert executor.state is ExecutorState.FAULT
    assert executor.fault_reason is not None
    assert "held right arm moved during ownership transition by 0.0510rad" in (
        executor.fault_reason
    )


def test_measured_stationarity_completion_does_not_loosen_held_arm_safety() -> None:
    clock, transport, executor = subject()
    executor.acquire(operator_confirmed=True)
    advance_until(transport, executor, ExecutorState.READY)
    transport.position[22] = 0.006
    clock.advance(0.01)

    executor.tick()

    assert executor.state is ExecutorState.FAULT
    assert executor.fault_reason is not None
    assert "held right arm drifted by 0.0060rad; limit is 0.0050rad" in (executor.fault_reason)


@pytest.mark.parametrize("failure", ["wrong_mode", "stale_state", "loop_overrun"])
def test_active_safety_failure_emergency_ramps_to_terminal_zero(failure: str) -> None:
    _, transport, executor = subject()
    executor.acquire(operator_confirmed=True)
    advance_until(transport, executor, ExecutorState.READY)

    if failure == "wrong_mode":
        transport.mode_machine = 4
        transport.step(0.02)
        executor.tick()
    elif failure == "stale_state":
        transport.freeze_state_receipt = True
        for _ in range(6):
            transport.step(0.02)
            executor.tick()
            if executor.state is ExecutorState.FAULT:
                break
    else:
        transport.step(0.26)
        executor.tick()

    assert executor.state is ExecutorState.FAULT
    assert executor.fault_reason
    advance_until(transport, executor, ExecutorState.STOPPED)
    assert transport.commands[-1].weight == 0.0
    assert transport.commands[-1].emergency_release


def test_nonfaulting_scheduler_gap_uses_nominal_motion_step() -> None:
    _, transport, executor = subject()
    executor.acquire(operator_confirmed=True)
    advance_until(transport, executor, ExecutorState.READY)
    executor.start_pose(
        "pose_001",
        approval=approval(executor, HANDOFF_POSE_ID, "pose_001"),
        operator_confirmed=True,
    )

    transport.step(0.051)
    executor.tick()

    assert executor.state is ExecutorState.MOVING
    assert executor.fault_reason is None
    assert all("overrun" not in event.reason for event in executor.events)
    np.testing.assert_allclose(transport.commands[-1].q14[:7], 0.004)


def test_control_gap_at_hard_limit_still_faults() -> None:
    _, transport, executor = subject()
    executor.acquire(operator_confirmed=True)
    advance_until(transport, executor, ExecutorState.READY)

    transport.step(0.251)
    executor.tick()

    assert executor.state is ExecutorState.FAULT
    assert executor.fault_reason is not None
    assert "control loop gap 0.251s exceeds hard limit 0.250s" in (executor.fault_reason)


def test_acquire_fails_closed_without_mode5_or_confirmation() -> None:
    _, transport, executor = subject()
    with pytest.raises(ValueError, match="confirmation"):
        executor.acquire(operator_confirmed=False)
    assert transport.commands == []

    transport.mode_machine = 4
    with pytest.raises(ValueError, match="mode_machine=5"):
        executor.acquire(operator_confirmed=True)
    assert transport.commands == []


def test_acquisition_ignores_unreliable_raw_dq() -> None:
    _, transport, executor = subject()
    transport.velocity[15] = 10.0

    executor.acquire(operator_confirmed=True)

    assert executor.state is ExecutorState.ACQUIRING
    assert transport.commands[0].weight == 0.0


def test_stationary_right_arm_dq_noise_does_not_fault_active_hold() -> None:
    clock, transport, executor = subject()
    executor.acquire(operator_confirmed=True)
    transport.velocity[22] = 10.0
    clock.advance(0.01)

    executor.tick()

    assert executor.state is ExecutorState.ACQUIRING
    assert executor.fault_reason is None


def test_settling_requires_bounded_measured_position_spread_for_full_dwell() -> None:
    clock, transport, executor = subject()
    executor.acquire(operator_confirmed=True)
    advance_until(transport, executor, ExecutorState.READY)
    executor.start_pose(
        "pose_001",
        approval=approval(executor, HANDOFF_POSE_ID, "pose_001"),
        operator_confirmed=True,
    )
    advance_until(transport, executor, ExecutorState.SETTLING)

    for index in range(10):
        offset = 0.0015 if index % 2 else -0.0015
        transport.position[15:22] = 0.08 + offset
        clock.advance(0.01)
        executor.tick()
    assert executor.state is ExecutorState.SETTLING

    transport.position[15:22] = 0.08
    for _ in range(8):
        clock.advance(0.01)
        executor.tick()
    assert executor.state is ExecutorState.READY


def test_motion_timeout_reports_nonstationary_measured_evidence() -> None:
    _, transport, executor = subject()
    executor.acquire(operator_confirmed=True)
    advance_until(transport, executor, ExecutorState.READY)
    transport.tracking_velocity_rad_s = 0.0
    executor.start_pose(
        "pose_001",
        approval=approval(executor, HANDOFF_POSE_ID, "pose_001"),
        operator_confirmed=True,
    )

    for index in range(200):
        transport.step(0.02)
        transport.position[15:22] = 0.01 if index % 2 else 0.0
        executor.tick()
        if executor.state is ExecutorState.FAULT:
            break

    assert executor.state is ExecutorState.FAULT
    assert executor.fault_reason is not None
    assert "motion timed out after" in executor.fault_reason
    assert "while settling for pose_001" in executor.fault_reason
    assert "left_shoulder_pitch_joint" in executor.fault_reason
    assert "limit=0.0800rad" in executor.fault_reason
    assert "command remaining=0.0000rad" in executor.fault_reason
    assert "position spread=0.0100rad" in executor.fault_reason


def test_clean_release_refuses_non_handoff_pose() -> None:
    _, transport, executor = subject()
    executor.acquire(operator_confirmed=True)
    advance_until(transport, executor, ExecutorState.READY)
    executor.start_pose(
        "pose_001",
        approval=approval(executor, HANDOFF_POSE_ID, "pose_001"),
        operator_confirmed=True,
    )
    advance_until(transport, executor, ExecutorState.READY)
    with pytest.raises(ValueError, match="not at"):
        executor.begin_clean_release(operator_confirmed=True)


def test_observing_emergency_stop_closes_without_publishing() -> None:
    _, transport, executor = subject()
    executor.emergency_stop("operator quit")
    assert executor.state is ExecutorState.STOPPED
    assert transport.closed
    assert transport.commands == []


def test_confirmed_external_damping_stops_without_weight_zero() -> None:
    _, transport, executor = subject()
    executor.acquire(operator_confirmed=True)
    advance_until(transport, executor, ExecutorState.READY)
    assert transport.commands[-1].weight == 1.0

    executor.confirm_external_damping("PC2 accepted G1 Damp")

    assert executor.state is ExecutorState.STOPPED
    assert executor.fault_reason == "PC2 accepted G1 Damp"
    assert transport.closed
    assert transport.commands[-1].weight == 1.0
    assert executor.events[-1].reason.startswith("external damping confirmed")
