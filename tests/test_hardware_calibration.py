from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from g1_dex3_tabletop.hardware_calibration import (
    _command_validated_finger_posture,
    _stage_trajectory,
    clearance_snapshot,
)


@pytest.mark.parametrize("failure", [None, "worker", "binding", "clearance", "state", "control"])
def test_loaded_finger_motion_requires_validated_current_state(monkeypatch, tmp_path, failure):
    from g1_dex3_tabletop import hardware_calibration as hardware
    from g1_dex3_tabletop.planning.contracts import Dex3PreparationRequest

    state = SimpleNamespace(position=np.linspace(0.01, 0.29, 29))
    hands = SimpleNamespace(
        left=SimpleNamespace(position=(0.3,) * 7), right=SimpleNamespace(position=(-0.4,) * 7)
    )
    calls = []
    checks = []

    def health():
        checks.append(True)
        if failure == "control" and len(checks) > 1:
            raise RuntimeError("original control fault")

    driver = SimpleNamespace(check=health)

    def command(**kwargs):
        calls.append(kwargs)
        return hands

    controller = SimpleNamespace(observer=object(), command_posture=command)
    monkeypatch.setattr(hardware, "_wait_for_hands", lambda _: hands)

    def validate(operation, *, payload, control_check, timeout_s):
        assert operation == "validate-dex3-finger-sweep"
        request = Dex3PreparationRequest.from_dict(payload)
        assert request.snapshot.measured_q29_rad == tuple(state.position)
        assert request.snapshot.left_dex3_q_rad == hands.left.position
        assert request.left_target_q_rad == (0.5,) * 7
        control_check()
        if failure == "worker":
            raise RuntimeError("finger route rejected")
        if failure == "state":
            state.position[4] += 0.1
        return {
            "payload": {
                "operation": "validate_dex3_finger_sweep",
                "passed": True,
                "request_sha256": "wrong" if failure == "binding" else request.content_sha256,
                "minimum_clearance_m": 0.004 if failure == "clearance" else 0.006,
                "required_clearance_m": 0.005,
            }
        }

    kwargs = {
        "synchronized": SimpleNamespace(observe_state=lambda: state),
        "driver": driver,
        "controller": controller,
        "planner": SimpleNamespace(request_payload=validate),
        "joint_position_offsets_rad": {},
        "left_target_q_rad": (0.5,) * 7,
        "right_target_q_rad": (-0.5,) * 7,
        "left_acceptance_q_rad": (0.49,) * 7,
        "right_acceptance_q_rad": (-0.49,) * 7,
        "body_tolerance_rad": 0.05,
        "hand_tolerance_rad": 0.08,
        "artifact_directory": tmp_path,
        "phase": "loaded_close",
        "label": "close",
    }
    if failure:
        with pytest.raises(RuntimeError):
            _command_validated_finger_posture(**kwargs)
        assert calls == []
    else:
        assert _command_validated_finger_posture(**kwargs) is hands
        assert len(calls) == 1 and calls[0]["left_acceptance_q_rad"] == (0.49,) * 7
        assert len(checks) >= 3


from g1_dex3_tabletop.planning.contracts import (
    Dex3PreparationPlan,
    PlannedTrajectory,
    RobotSnapshot,
)


def _trajectory(source: str, target: str, start: float, end: float) -> PlannedTrajectory:
    return PlannedTrajectory(
        from_pose_id=source,
        to_pose_id=target,
        sample_time_s=(0.0, 1.0),
        command_q_rad=((start,) * 7, (end,) * 7),
        model_q_rad=((start,) * 7, (end,) * 7),
        planning_time_s=0.1,
    )


def _preparation() -> Dex3PreparationPlan:
    right = _trajectory("__handoff__", "right_shoulder_clearance", 0.0, 0.2)
    left = _trajectory("right_shoulder_clearance", "dual_shoulder_clearance", 0.0, -0.3)
    return Dex3PreparationPlan(
        request_sha256="a" * 64,
        outward_offset_rad=0.08,
        right_outbound=right,
        left_outbound=left,
        left_return=_trajectory("dual_shoulder_clearance", "right_shoulder_clearance", -0.3, 0.0),
        right_return=_trajectory("right_shoulder_clearance", "__handoff__", 0.2, 0.0),
        dual_clearance_q14_rad=(-0.3,) * 7 + (0.2,) * 7,
        finger_sweep_sample_count=20,
        return_sweep_sample_count=0,
        planner_provenance={"backend": "test"},
    )


def test_clearance_snapshot_changes_only_both_arms_and_fingers() -> None:
    q29 = np.arange(29, dtype=np.float64) / 100.0
    source = RobotSnapshot(tuple(q29), (0.1,) * 7, (-0.1,) * 7)
    result = clearance_snapshot(
        source,
        _preparation(),
        left_fingers=(-1.0,) * 7,
        right_fingers=(1.0,) * 7,
    )
    expected = q29.copy()
    expected[15:22] = -0.3
    expected[22:29] = 0.2
    np.testing.assert_allclose(result.measured_q29_rad, expected)
    assert result.left_dex3_q_rad == (-1.0,) * 7
    assert result.right_dex3_q_rad == (1.0,) * 7


def test_stage_rebase_changes_ids_but_not_any_motion_sample() -> None:
    source = _preparation().left_return
    result = _stage_trajectory(source, target_id="left_shoulder_restored")
    assert result.from_pose_id == "__handoff__"
    assert result.to_pose_id == "left_shoulder_restored"
    assert result.sample_time_s == source.sample_time_s
    assert result.command_q_rad == source.command_q_rad
    assert result.model_q_rad == source.model_q_rad


@pytest.mark.parametrize("phase", ["acquire", "wait_ready"])
@pytest.mark.parametrize("error_type", [RuntimeError, KeyboardInterrupt])
def test_stage_failure_stops_driver_before_returning_to_safety_cleanup(
    monkeypatch, phase, error_type
):
    from g1_aprilcube_calibration.executor_state_machine import ExecutorConfig
    from g1_dex3_tabletop import hardware_calibration as hardware

    failure = error_type("original acquisition failure")
    calls = []

    class Driver:
        def __init__(self, *args, **kwargs):
            pass

        def start(self):
            calls.append("start")

        def close(self):
            calls.append("close")

    def acquire(**kwargs):
        calls.append("acquire")
        if phase == "acquire":
            raise failure

    def wait_ready(*args, **kwargs):
        calls.append("wait_ready")
        raise failure

    monkeypatch.setattr(hardware, "pose_set_from_trajectories", lambda **kwargs: object())
    monkeypatch.setattr(
        hardware, "PoseExecutor", lambda **kwargs: SimpleNamespace(acquire=acquire)
    )
    monkeypatch.setattr(hardware, "ExecutorControlDriver", Driver)
    monkeypatch.setattr(hardware, "_wait_ready", wait_ready)
    with pytest.raises(error_type) as caught:
        hardware._stage_executor(
            arm="right",
            trajectory=None,
            q29=np.zeros(29),
            q14=np.zeros(14),
            model=SimpleNamespace(name="test", sha256="a" * 64),
            transport=object(),
            gravity=None,
            control_config=ExecutorConfig(),
            plan_sha256="b" * 64,
            acquire=True,
            heartbeat=lambda: None,
            rate_hz=250,
        )
    assert caught.value is failure
    assert calls == (
        ["start", "acquire", "close"]
        if phase == "acquire"
        else ["start", "acquire", "wait_ready", "close"]
    )
