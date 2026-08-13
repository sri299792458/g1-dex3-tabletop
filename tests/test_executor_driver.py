import threading

import pytest

from g1_aprilcube_calibration.executor_driver import (
    ExecutorControlDriver,
    SynchronizedPoseExecutor,
)
from g1_aprilcube_calibration.executor_state_machine import ExecutorState


class Executor:
    def __init__(self):
        self.pose_set = object()
        self.approved_validation_report_sha256 = "a" * 64
        self.state = "observing"
        self.current_pose_id = None
        self.fault_reason = None
        self.events = []
        self.tick_count = 0
        self.called_thread_ids = []
        self.observed_state = object()

    def tick(self):
        self.tick_count += 1
        self.called_thread_ids.append(threading.get_ident())

    def acquire(self, **_):
        self.state = "acquiring"

    def observe_state(self):
        return self.observed_state

    def emergency_stop(self, reason):
        self.fault_reason = reason

    def install_validated_plan(self, **kwargs):
        self.installed_plan = kwargs

    def start_trajectory(self, **kwargs):
        self.trajectory = kwargs


def test_control_driver_ticks_on_dedicated_thread_and_serializes_actions():
    raw = Executor()
    synchronized = SynchronizedPoseExecutor(raw)
    driver = ExecutorControlDriver(synchronized, rate_hz=500)
    driver.start()
    synchronized.acquire(operator_confirmed=True)
    for _ in range(10_000):
        if raw.tick_count >= 2:
            break
    driver.close()
    driver.check()
    assert raw.tick_count >= 1
    assert set(raw.called_thread_ids) != {threading.get_ident()}
    assert synchronized.state == "acquiring"


def test_control_driver_heartbeat_is_emitted_by_control_thread():
    raw = Executor()
    synchronized = SynchronizedPoseExecutor(raw)
    heartbeat_thread_ids = []
    driver = ExecutorControlDriver(
        synchronized,
        rate_hz=500,
        safety_heartbeat=lambda: heartbeat_thread_ids.append(threading.get_ident()),
    )
    driver.start()
    for _ in range(10_000):
        if heartbeat_thread_ids:
            break
    driver.close()
    driver.check()
    assert heartbeat_thread_ids
    assert set(heartbeat_thread_ids) != {threading.get_ident()}


def test_synchronized_executor_exposes_transport_style_observe():
    raw = Executor()
    synchronized = SynchronizedPoseExecutor(raw)

    assert synchronized.observe() is raw.observed_state
    assert synchronized.observe_state() is raw.observed_state


def test_synchronized_executor_forwards_frozen_trajectory() -> None:
    raw = Executor()
    synchronized = SynchronizedPoseExecutor(raw)
    synchronized.start_trajectory(from_pose_id="a", to_pose_id="b")
    assert raw.trajectory == {"from_pose_id": "a", "to_pose_id": "b"}


def test_control_driver_preserves_executor_fault_reason():
    raw = Executor()
    raw.fault_reason = "held right arm drifted"
    raw.tick = lambda: ExecutorState.FAULT
    driver = ExecutorControlDriver(
        SynchronizedPoseExecutor(raw),
        rate_hz=500,
    )

    driver.start()
    driver.close()

    with pytest.raises(RuntimeError, match="held right arm drifted"):
        driver.check()


def test_synchronized_executor_serializes_validated_plan_install():
    raw = Executor()
    synchronized = SynchronizedPoseExecutor(raw)
    values = {"pose_set": object(), "approved_validation_report_sha256": "b" * 64}

    synchronized.install_validated_plan(**values)

    assert raw.installed_plan == values
