"""The bilateral adapter must start at the acquired command, including offsets."""

from types import SimpleNamespace

import numpy as np
import pytest

from g1_aprilcube_calibration.clock import ManualClock
from g1_aprilcube_calibration.executor_driver import SynchronizedPoseExecutor
from g1_aprilcube_calibration.executor_state_machine import (
    ExecutorConfig,
    ExecutorState,
    PoseExecutor,
)
from g1_aprilcube_calibration.transports.fake import FakeArmTransport
from g1_dex3_tabletop import hardware_bilateral_calibration as hardware
from g1_dex3_tabletop.execution_plan import pose_set_from_trajectories
from g1_dex3_tabletop.planning.contracts import (
    BilateralCalibrationAdapterRequest,
    PlannedTrajectory,
    RobotSnapshot,
)


def edge(start):
    target = np.asarray(start).copy()
    target[1] -= 0.08
    return PlannedTrajectory(
        "__handoff__",
        "right_shoulder_clearance",
        (0.0, 2.0),
        (tuple(start), tuple(target)),
        (tuple(start), tuple(target)),
        0.0,
    )


@pytest.mark.parametrize("failure", [None, "worker", "binding", "anchor", "start", "drift"])
def test_owned_adapter_reuses_held_commands_and_rejects_invalid_installation(
    monkeypatch, tmp_path, failure
):
    model = SimpleNamespace(name="g1", sha256="a" * 64)
    preflight = BilateralCalibrationAdapterRequest(
        execution_plan_sha256="b" * 64,
        robot_model=model.name,
        urdf_sha256=model.sha256,
        snapshot=RobotSnapshot((0.0,) * 29, (0.3,) * 7, (-0.3,) * 7),
        anchor_q29_rad=(0.0,) * 29,
        core_transitions=({"arm": "right", "trajectory": edge(np.zeros(7)).to_dict()},),
        joint_position_offsets_rad={},
        left_close_command_q_rad=(0.5,) * 7,
        right_close_command_q_rad=(-0.5,) * 7,
        left_close_model_q_rad=(0.49,) * 7,
        right_close_model_q_rad=(-0.49,) * 7,
    )
    # The recorded wrist-roll difference from the failed 022735 run.
    acquired = np.zeros(29)
    acquired[26] = -0.00022770464420318604
    acquired[15] = 0.0005  # Also preserve the initially held opposite arm.
    clock = ManualClock(1.0)
    transport = FakeArmTransport(clock=clock, initial_full_q=acquired)
    original = edge(np.zeros(7))
    raw = PoseExecutor(
        transport=transport,
        clock=clock,
        pose_set=pose_set_from_trajectories(
            arm="right",
            trajectories=(original,),
            reference_full_q=np.zeros(29),
            robot_model=model.name,
            urdf_sha256=model.sha256,
            source="test",
        ),
        handoff_q=np.zeros(7),
        hold_q=np.zeros(7),
        approved_validation_report_sha256="c" * 64,
        config=ExecutorConfig(),
    )
    synchronized = SynchronizedPoseExecutor(raw)
    synchronized.acquire(operator_confirmed=True)
    for _ in range(1000):
        transport.step(0.004)
        synchronized.tick()
        if synchronized.state is ExecutorState.READY:
            break
    assert synchronized.state is ExecutorState.READY
    before = synchronized.dual_arm_command_q
    with pytest.raises(ValueError, match="trajectory start differs"):
        synchronized.start_trajectory(
            from_pose_id=original.from_pose_id,
            to_pose_id=original.to_pose_id,
            sample_time_s=original.sample_time_s,
            command_q_rad=original.command_q_rad,
            plan_sha256="c" * 64,
            operator_confirmed=True,
        )
    # Loaded measurements can differ from both held commands. They must not
    # replace either arm's command in the planner input or installed command.
    transport.position[18] += 0.015
    transport.position[25] -= 0.020277314
    transport.position[12] = 0.004
    hands = SimpleNamespace(
        left=SimpleNamespace(position=(0.31,) * 7),
        right=SimpleNamespace(position=(-0.29,) * 7),
    )
    health_checks = []
    result = None

    def plan(command, *, request_path, output_path, control_check, timeout_s):
        nonlocal result
        assert command == "plan-bilateral-calibration-adapter"
        request = BilateralCalibrationAdapterRequest.from_json(request_path)
        np.testing.assert_array_equal(request.snapshot.measured_q29_rad[15:29], before)
        assert request.snapshot.measured_q29_rad[12] == 0.004
        assert request.snapshot.left_dex3_q_rad == hands.left.position
        assert request.core_transitions == preflight.core_transitions
        assert request.execution_plan_sha256 == preflight.execution_plan_sha256
        control_check()
        if failure == "worker":
            raise RuntimeError("planner rejected")
        start = before[7:].copy()
        if failure == "start":
            start[4] += 0.00022770464420318604
        result = SimpleNamespace(
            request_sha256="d" * 64 if failure == "binding" else request.content_sha256,
            content_sha256="e" * 64,
            preparation=SimpleNamespace(right_outbound=edge(start)),
            anchor_q14_rad=(0.1 if failure == "anchor" else 0.0,) * 14,
        )
        if failure == "drift":
            transport.position[12] += raw.config.settled_position_spread_rad + 0.001

    monkeypatch.setattr(hardware.BilateralCalibrationAdapterPlan, "from_json", lambda _: result)
    arguments = {
        "synchronized": synchronized,
        "driver": SimpleNamespace(check=lambda: health_checks.append(True)),
        "planner": SimpleNamespace(request=plan),
        "preflight_request": preflight,
        "held_hands": hands,
        "model": model,
        "work_directory": tmp_path,
    }
    if failure:
        with pytest.raises((RuntimeError, ValueError)):
            hardware._plan_owned_adapter(**arguments)
        assert synchronized.approved_validation_report_sha256 == "c" * 64
    else:
        request, adapter, trajectory = hardware._plan_owned_adapter(**arguments)
        assert request.content_sha256 != preflight.content_sha256
        np.testing.assert_array_equal(trajectory.command_q_rad[0], before[7:])
        synchronized.start_trajectory(
            from_pose_id=trajectory.from_pose_id,
            to_pose_id=trajectory.to_pose_id,
            sample_time_s=trajectory.sample_time_s,
            command_q_rad=trajectory.command_q_rad,
            plan_sha256=adapter.content_sha256,
            operator_confirmed=True,
        )
        assert synchronized.state is ExecutorState.MOVING
        assert "command preserved" in raw.events[-2].reason
    np.testing.assert_array_equal(synchronized.dual_arm_command_q, before)
    np.testing.assert_array_equal(transport.commands[-1].q14, before)
    assert health_checks
