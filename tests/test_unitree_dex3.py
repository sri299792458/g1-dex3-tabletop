from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from typing import ClassVar

import numpy as np
import pytest

from g1_aprilcube_calibration.clock import ManualClock
from g1_aprilcube_calibration.transports.unitree_dex3 import (
    DEX3_CALIBRATION_POSTURE_SOURCE,
    NVIDIA_MIDDLE_CLOSE_LEFT_Q_RAD,
    NVIDIA_MIDDLE_CLOSE_RIGHT_Q_RAD,
    Dex3ControlConfig,
    Dex3SDKBindings,
    UnitreeDex3PostureController,
    UnitreeDex3StateObserver,
    dex3_motor_joint_name,
    dex3_motor_mode,
)


@dataclass
class Motor:
    mode: int = 0
    q: float = 0.0
    dq: float = 0.0
    tau: float = 0.0
    kp: float = 0.0
    kd: float = 0.0


@dataclass
class HandCommand:
    motor_cmd: list[Motor] = field(default_factory=lambda: [Motor() for _ in range(7)])


@dataclass
class HandState:
    motor_state: list[Motor]


class Subscriber:
    instances: ClassVar[list[Subscriber]] = []

    def __init__(self, topic, message_type):
        self.topic = topic
        self.message_type = message_type
        self.callback = None
        self.closed = False
        self.__class__.instances.append(self)

    def Init(self, callback, queue_length):
        self.callback = callback
        self.queue_length = queue_length

    def emit(self, message):
        self.callback(message)

    def Close(self):
        self.closed = True


class Publisher:
    instances: ClassVar[list[Publisher]] = []

    def __init__(self, topic, message_type):
        self.topic = topic
        self.message_type = message_type
        self.messages = []
        self.closed = False
        self.__class__.instances.append(self)

    def Init(self):
        self.initialized = True

    def Write(self, message):
        self.messages.append(deepcopy(message))
        return True

    def Close(self):
        self.closed = True


@pytest.fixture
def dex3_sdk():
    Subscriber.instances.clear()
    Publisher.instances.clear()
    initialized = []
    bindings = Dex3SDKBindings(
        initialize=lambda domain, interface: initialized.append((domain, interface)),
        publisher_type=Publisher,
        subscriber_type=Subscriber,
        hand_command_type=HandCommand,
        hand_state_type=HandState,
        make_hand_command=HandCommand,
    )
    return bindings, initialized


def config(**changes):
    values = {
        "network_interface": "enp3s0",
        "domain_id": 4,
        "posture_ramp_s": 0.0,
        "posture_settle_dwell_s": 0.0,
    }
    values.update(changes)
    return Dex3ControlConfig(**values)


def hand_state(q, *, dq=0.0):
    return HandState([Motor(q=float(value), dq=dq) for value in q])


def emit_pair(left, right, *, dq=0.0):
    by_topic = {item.topic: item for item in Subscriber.instances}
    by_topic["rt/dex3/left/state"].emit(hand_state(left, dq=dq))
    by_topic["rt/dex3/right/state"].emit(hand_state(right, dq=dq))


def test_observer_is_read_only_and_requires_both_fresh_states(dex3_sdk):
    bindings, initialized = dex3_sdk
    clock = ManualClock(2.0)
    observer = UnitreeDex3StateObserver(
        config(state_freshness_timeout_s=0.1), bindings=bindings, clock=clock
    )

    assert initialized == [(4, "enp3s0")]
    assert Publisher.instances == []
    emit_pair(np.zeros(7), np.zeros(7))
    pair = observer.observe()
    assert pair.maximum_abs_position_rad == 0.0
    clock.advance(0.11)
    with pytest.raises(RuntimeError, match="stale Dex3 state"):
        observer.observe()
    observer.close()
    assert all(item.closed for item in Subscriber.instances)


def test_controller_clamps_first_posture_command_to_measured_state(dex3_sdk):
    bindings, _ = dex3_sdk
    clamp_config = config(posture_position_tolerance_rad=3.0)
    observer = UnitreeDex3StateObserver(clamp_config, bindings=bindings)
    emit_pair(np.full(7, 0.8), np.full(7, -0.8))
    controller = UnitreeDex3PostureController(clamp_config, observer=observer)

    controller.maintain_posture()

    by_topic = {item.topic: item for item in Publisher.instances}
    left = by_topic["rt/dex3/left/cmd"].messages[-1]
    right = by_topic["rt/dex3/right/cmd"].messages[-1]
    np.testing.assert_allclose(
        [item.q for item in left.motor_cmd],
        np.clip(NVIDIA_MIDDLE_CLOSE_LEFT_Q_RAD, 0.55, 1.05),
    )
    np.testing.assert_allclose(
        [item.q for item in right.motor_cmd],
        np.clip(NVIDIA_MIDDLE_CLOSE_RIGHT_Q_RAD, -1.05, -0.55),
    )
    assert all(item.kp == 1.5 and item.kd == 0.2 for item in left.motor_cmd)
    controller.timeout_and_close()


def test_posture_acquisition_requires_measured_target_and_timeout_is_explicit(
    dex3_sdk,
):
    bindings, _ = dex3_sdk
    observer = UnitreeDex3StateObserver(config(), bindings=bindings)
    left = np.asarray(NVIDIA_MIDDLE_CLOSE_LEFT_Q_RAD) + 0.02
    right = np.asarray(NVIDIA_MIDDLE_CLOSE_RIGHT_Q_RAD) - 0.03
    emit_pair(left, right, dq=0.01)
    controller = UnitreeDex3PostureController(config(), observer=observer)
    heartbeats = []

    final = controller.acquire_posture(safety_heartbeat=lambda: heartbeats.append(True))
    assert final.maximum_target_error(
        NVIDIA_MIDDLE_CLOSE_LEFT_Q_RAD,
        NVIDIA_MIDDLE_CLOSE_RIGHT_Q_RAD,
    )[2] == pytest.approx(0.03)
    assert heartbeats == [True]

    with pytest.raises(RuntimeError, match="before timeout"):
        controller.close()
    controller.timeout_and_close()
    by_topic = {item.topic: item for item in Publisher.instances}
    for topic in ("rt/dex3/left/cmd", "rt/dex3/right/cmd"):
        final_message = by_topic[topic].messages[-1]
        assert [motor.mode for motor in final_message.motor_cmd] == [
            dex3_motor_mode(index, timeout=True) for index in range(7)
        ]
        assert all(
            motor.q == motor.dq == motor.tau == motor.kp == motor.kd == 0.0
            for motor in final_message.motor_cmd
        )
        assert by_topic[topic].closed


def test_posture_acquisition_finishes_dwell_after_entry_deadline(dex3_sdk):
    bindings, _ = dex3_sdk
    clock = ManualClock(0.0)
    target_left = np.asarray(NVIDIA_MIDDLE_CLOSE_LEFT_Q_RAD)
    target_right = np.asarray(NVIDIA_MIDDLE_CLOSE_RIGHT_Q_RAD)
    outside_left = target_left.copy()
    outside_left[5] += 0.2
    controller_config = config(
        posture_settle_dwell_s=0.5,
        posture_timeout_s=0.6,
        state_freshness_timeout_s=0.2,
    )
    observer = UnitreeDex3StateObserver(
        controller_config,
        bindings=bindings,
        clock=clock,
    )
    emit_pair(outside_left, target_right)

    def advance_and_emit(_duration_s):
        clock.advance(0.1)
        left = target_left if clock.monotonic() >= 0.5 else outside_left
        emit_pair(left, target_right)

    controller = UnitreeDex3PostureController(
        controller_config,
        observer=observer,
        clock=clock,
        sleep=advance_and_emit,
    )

    final = controller.acquire_posture()

    assert 1.0 <= clock.monotonic() <= 1.1
    assert final.maximum_target_error(
        NVIDIA_MIDDLE_CLOSE_LEFT_Q_RAD,
        NVIDIA_MIDDLE_CLOSE_RIGHT_Q_RAD,
    )[2] == pytest.approx(0.0)
    controller.timeout_and_close()


def test_posture_acquisition_uses_position_spread_not_raw_velocity(dex3_sdk):
    bindings, _ = dex3_sdk
    clock = ManualClock(0.0)
    target_left = np.asarray(NVIDIA_MIDDLE_CLOSE_LEFT_Q_RAD)
    target_right = np.asarray(NVIDIA_MIDDLE_CLOSE_RIGHT_Q_RAD)
    controller_config = config(
        posture_settle_dwell_s=0.2,
        posture_timeout_s=0.3,
        state_freshness_timeout_s=0.2,
    )
    observer = UnitreeDex3StateObserver(
        controller_config,
        bindings=bindings,
        clock=clock,
    )
    emit_pair(target_left, target_right, dq=0.5)

    def advance_stable_position(_duration_s):
        clock.advance(0.1)
        emit_pair(target_left, target_right, dq=0.5)

    controller = UnitreeDex3PostureController(
        controller_config,
        observer=observer,
        clock=clock,
        sleep=advance_stable_position,
    )

    final = controller.acquire_posture()

    assert final.maximum_abs_velocity_rad_s == pytest.approx(0.5)
    controller.timeout_and_close()


def test_posture_acquisition_rejects_position_spread_after_deadline(dex3_sdk):
    bindings, _ = dex3_sdk
    clock = ManualClock(0.0)
    target_left = np.asarray(NVIDIA_MIDDLE_CLOSE_LEFT_Q_RAD)
    target_right = np.asarray(NVIDIA_MIDDLE_CLOSE_RIGHT_Q_RAD)
    shifted_left = target_left.copy()
    shifted_left[5] += 0.02
    controller_config = config(
        posture_position_spread_rad=0.01,
        posture_settle_dwell_s=0.2,
        posture_timeout_s=0.3,
        state_freshness_timeout_s=0.2,
    )
    observer = UnitreeDex3StateObserver(
        controller_config,
        bindings=bindings,
        clock=clock,
    )
    emit_pair(target_left, target_right, dq=0.0)
    update_count = 0

    def advance_moving_position(_duration_s):
        nonlocal update_count
        update_count += 1
        clock.advance(0.1)
        left = shifted_left if update_count % 2 else target_left
        emit_pair(left, target_right, dq=0.0)

    controller = UnitreeDex3PostureController(
        controller_config,
        observer=observer,
        clock=clock,
        sleep=advance_moving_position,
    )

    with pytest.raises(RuntimeError, match="position spread=0.0200rad"):
        controller.acquire_posture()

    controller.timeout_and_close()


def test_controller_restores_the_measured_precommand_finger_posture(dex3_sdk):
    bindings, _ = dex3_sdk
    initial_left = np.linspace(0.10, 0.16, 7)
    initial_right = np.linspace(-0.10, -0.16, 7)
    target_left = np.asarray(NVIDIA_MIDDLE_CLOSE_LEFT_Q_RAD)
    target_right = np.asarray(NVIDIA_MIDDLE_CLOSE_RIGHT_Q_RAD)
    observer = UnitreeDex3StateObserver(config(), bindings=bindings)
    emit_pair(initial_left, initial_right)
    sleep_count = 0

    def update_measured_state(_duration_s):
        nonlocal sleep_count
        sleep_count += 1
        if sleep_count == 1:
            emit_pair(target_left, target_right)
        elif sleep_count == 2:
            emit_pair(initial_left, initial_right)

    controller = UnitreeDex3PostureController(
        config(),
        observer=observer,
        sleep=update_measured_state,
    )

    controller.acquire_posture()
    restored = controller.restore_initial_posture()

    np.testing.assert_allclose(restored.left.position, initial_left)
    np.testing.assert_allclose(restored.right.position, initial_right)
    by_topic = {item.topic: item for item in Publisher.instances}
    np.testing.assert_allclose(
        [item.q for item in by_topic["rt/dex3/left/cmd"].messages[-1].motor_cmd],
        initial_left,
    )
    np.testing.assert_allclose(
        [item.q for item in by_topic["rt/dex3/right/cmd"].messages[-1].motor_cmd],
        initial_right,
    )
    controller.timeout_and_close()


def test_measured_hold_freezes_original_posture_through_later_target_acquisition(
    dex3_sdk,
):
    bindings, _ = dex3_sdk
    clock = ManualClock(1.0)
    initial_left = np.linspace(0.10, 0.16, 7)
    initial_right = np.linspace(-0.10, -0.16, 7)
    shifted_left = initial_left + 0.01
    shifted_right = initial_right - 0.01
    target_left = np.asarray(NVIDIA_MIDDLE_CLOSE_LEFT_Q_RAD)
    target_right = np.asarray(NVIDIA_MIDDLE_CLOSE_RIGHT_Q_RAD)
    observer = UnitreeDex3StateObserver(config(), bindings=bindings, clock=clock)
    emit_pair(initial_left, initial_right)
    controller = UnitreeDex3PostureController(config(), observer=observer, clock=clock)

    acquired = controller.acquire_measured_hold()
    np.testing.assert_allclose(acquired.left.position, initial_left)
    by_topic = {item.topic: item for item in Publisher.instances}
    np.testing.assert_allclose(
        [item.q for item in by_topic["rt/dex3/left/cmd"].messages[-1].motor_cmd],
        initial_left,
    )

    clock.advance(0.02)
    emit_pair(shifted_left, shifted_right)
    controller.maintain_initial_posture()
    np.testing.assert_allclose(
        [item.q for item in by_topic["rt/dex3/left/cmd"].messages[-1].motor_cmd],
        initial_left,
    )

    emit_pair(target_left, target_right)
    controller.acquire_posture()
    emit_pair(initial_left, initial_right)
    restored = controller.restore_initial_posture()
    np.testing.assert_allclose(restored.left.position, initial_left)
    np.testing.assert_allclose(restored.right.position, initial_right)
    controller.timeout_and_close()


def test_invalid_or_partial_hand_state_never_replaces_last_good_pair(dex3_sdk):
    bindings, _ = dex3_sdk
    observer = UnitreeDex3StateObserver(config(), bindings=bindings)
    emit_pair(np.full(7, 0.1), np.full(7, 0.2))
    left_subscriber = next(
        item for item in Subscriber.instances if item.topic.endswith("left/state")
    )
    left_subscriber.emit(HandState([Motor(q=float("nan")) for _ in range(7)]))
    pair = observer.observe()
    np.testing.assert_allclose(pair.left.position, 0.1)
    observer.close()


def test_posture_hold_faults_if_fingers_leave_frozen_collision_posture(
    dex3_sdk,
):
    bindings, _ = dex3_sdk
    observer = UnitreeDex3StateObserver(config(), bindings=bindings)
    left = np.asarray(NVIDIA_MIDDLE_CLOSE_LEFT_Q_RAD).copy()
    left[3] += 0.09
    emit_pair(left, NVIDIA_MIDDLE_CLOSE_RIGHT_Q_RAD)
    controller = UnitreeDex3PostureController(config(), observer=observer)

    with pytest.raises(RuntimeError, match="departed the fixed calibration posture"):
        controller.maintain_posture()

    controller.timeout_and_close()


def test_nvidia_middle_close_targets_are_named_and_side_specific() -> None:
    target_config = config()

    assert target_config.left_target_q_rad == NVIDIA_MIDDLE_CLOSE_LEFT_Q_RAD
    assert target_config.right_target_q_rad == NVIDIA_MIDDLE_CLOSE_RIGHT_Q_RAD
    assert "GR00T-WholeBodyControl" in DEX3_CALIBRATION_POSTURE_SOURCE
    assert target_config.left_target_q_rad != target_config.right_target_q_rad


def test_motor_joint_names_follow_unitree_documented_dds_order() -> None:
    assert dex3_motor_joint_name("left", 3) == "left_hand_middle_0_joint"
    assert dex3_motor_joint_name("left", 5) == "left_hand_index_0_joint"
    assert dex3_motor_joint_name("right", 3) == "right_hand_middle_0_joint"
    assert dex3_motor_joint_name("right", 5) == "right_hand_index_0_joint"
