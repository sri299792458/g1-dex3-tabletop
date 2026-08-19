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
    Dex3GraspNotAcquiredError,
    Dex3RetentionLostError,
    Dex3SDKBindings,
    UnitreeDex3PostureController,
    UnitreeDex3StateObserver,
    classify_dex3_opposed_joint_obstruction,
    dex3_motor_joint_name,
    dex3_motor_mode,
)


@dataclass
class Motor:
    mode: int = 0
    q: float = 0.0
    dq: float = 0.0
    tau: float = 0.0
    tau_est: float = 0.0
    kp: float = 0.0
    kd: float = 0.0


@dataclass
class HandCommand:
    motor_cmd: list[Motor] = field(default_factory=lambda: [Motor() for _ in range(7)])


@dataclass
class PressureGroup:
    pressure: list[float]


@dataclass
class HandState:
    motor_state: list[Motor]
    press_sensor_state: list[PressureGroup]


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


def unloaded_pressure() -> np.ndarray:
    return np.zeros((9, 12), dtype=np.float64)


def opposed_pressure(
    *,
    thumb_group: int = 0,
    thumb_cell: int = 0,
    finger_group: int = 2,
    finger_cell: int = 0,
    delta: float = 100.0,
) -> np.ndarray:
    pressure = unloaded_pressure()
    pressure[thumb_group, thumb_cell] += delta
    pressure[finger_group, finger_cell] += delta
    return pressure


def hand_state(q, *, dq=0.0, tau_est=0.0, pressure=None):
    matrix = unloaded_pressure() if pressure is None else np.asarray(pressure)
    return HandState(
        [Motor(q=float(value), dq=dq, tau_est=tau_est) for value in q],
        [PressureGroup(list(row)) for row in matrix],
    )


def emit_pair(
    left,
    right,
    *,
    dq=0.0,
    tau_est=0.0,
    left_pressure=None,
    right_pressure=None,
):
    by_topic = {item.topic: item for item in Subscriber.instances}
    by_topic["rt/dex3/left/state"].emit(
        hand_state(left, dq=dq, tau_est=tau_est, pressure=left_pressure)
    )
    by_topic["rt/dex3/right/state"].emit(
        hand_state(right, dq=dq, tau_est=tau_est, pressure=right_pressure)
    )


def test_observer_is_read_only_and_requires_both_fresh_states(dex3_sdk):
    bindings, initialized = dex3_sdk
    clock = ManualClock(2.0)
    observer = UnitreeDex3StateObserver(
        config(state_freshness_timeout_s=0.1), bindings=bindings, clock=clock
    )

    assert initialized == [(4, "enp3s0")]
    assert Publisher.instances == []
    left_pressure = unloaded_pressure()
    left_pressure[0, 0] = 1234.0
    emit_pair(
        np.zeros(7),
        np.zeros(7),
        tau_est=2.5,
        left_pressure=left_pressure,
    )
    pair = observer.observe()
    assert pair.maximum_abs_position_rad == 0.0
    np.testing.assert_allclose(pair.left.estimated_torque, 2.5)
    assert pair.left.pressure[0, 0] == pytest.approx(1234.0)
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


def test_posture_command_can_publish_descriptor_target_against_measured_acceptance(
    dex3_sdk,
):
    bindings, _ = dex3_sdk
    clock = ManualClock(1.0)
    observer = UnitreeDex3StateObserver(config(), bindings=bindings, clock=clock)
    measured_open = np.zeros(7)
    measured_open[3] = -0.10
    empty_open_reference = np.zeros(7)
    empty_open_reference[3] = -0.03
    emit_pair(measured_open, np.zeros(7))
    controller = UnitreeDex3PostureController(config(), observer=observer, clock=clock)
    controller.acquire_measured_hold()

    result = controller.command_posture(
        left_target_q_rad=np.zeros(7),
        right_target_q_rad=np.zeros(7),
        left_acceptance_q_rad=empty_open_reference,
        right_acceptance_q_rad=np.zeros(7),
        label="measured empty-open release",
    )

    assert result.left.position[3] == pytest.approx(-0.10)
    by_topic = {item.topic: item for item in Publisher.instances}
    np.testing.assert_allclose(
        [item.q for item in by_topic["rt/dex3/left/cmd"].messages[-1].motor_cmd],
        0.0,
    )
    clock.advance(0.02)
    controller.maintain_active_posture()
    np.testing.assert_allclose(
        [item.q for item in by_topic["rt/dex3/left/cmd"].messages[-1].motor_cmd],
        0.0,
    )

    drifted = measured_open.copy()
    drifted[3] = -0.12
    clock.advance(0.02)
    emit_pair(drifted, np.zeros(7))
    with pytest.raises(RuntimeError, match="departed the active task posture"):
        controller.maintain_active_posture()
    controller.timeout_and_close()


def test_retained_replacement_release_matches_its_run_local_empty_open() -> None:
    measured_empty_open = np.asarray([-0.0320, 0.0149, 0.0149, -0.0285, -0.0289, -0.0438, -0.0191])
    measured_release = np.asarray([-0.0320, 0.0163, 0.0275, -0.1005, -0.0530, -0.0450, -0.0226])

    assert np.max(np.abs(measured_release)) > config().posture_position_tolerance_rad
    assert np.max(np.abs(measured_release - measured_empty_open)) == pytest.approx(0.072)
    assert (
        np.max(np.abs(measured_release - measured_empty_open))
        < config().posture_position_tolerance_rad
    )


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


def grasp_test_config():
    return config(
        command_rate_hz=10.0,
        posture_position_tolerance_rad=0.08,
        posture_position_spread_rad=0.01,
        posture_settle_dwell_s=0.2,
        posture_timeout_s=1.0,
        state_freshness_timeout_s=0.2,
    )


@pytest.mark.parametrize("active_side,sign", [("left", -1.0), ("right", 1.0)])
def test_grasp_close_requires_opposed_empty_close_obstruction_and_rechecks_it(
    dex3_sdk,
    active_side,
    sign,
):
    bindings, _ = dex3_sdk
    clock = ManualClock(0.0)
    active_target = sign * np.asarray((0.0, 0.5, 0.5, 0.6, 0.8, 0.7, 0.9))
    empty_close = active_target - sign * np.asarray((0.0, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02))
    active_contact = empty_close.copy()
    active_contact[1] -= sign * 0.12
    active_contact[3] -= sign * 0.15
    zero = np.zeros(7)
    controller_config = grasp_test_config()
    observer = UnitreeDex3StateObserver(controller_config, bindings=bindings, clock=clock)
    emit_pair(zero, zero)

    def advance(_duration_s):
        clock.advance(0.1)
        left_q = active_contact if active_side == "left" else zero
        right_q = active_contact if active_side == "right" else zero
        emit_pair(
            left_q,
            right_q,
            left_pressure=unloaded_pressure(),
            right_pressure=unloaded_pressure(),
        )

    controller = UnitreeDex3PostureController(
        controller_config,
        observer=observer,
        clock=clock,
        sleep=advance,
    )
    contact = controller.command_close_for_retention_test(
        active_side=active_side,
        left_target_q_rad=active_target if active_side == "left" else zero,
        right_target_q_rad=active_target if active_side == "right" else zero,
        empty_close_reference_q_rad=empty_close,
        minimum_opposed_shortfall_rad=0.05,
        label="test cube grasp",
    )

    assert contact.joint_obstruction.has_opposed_obstruction
    assert contact.blocked_motor_ids == (1, 3)
    assert contact.moved_motor_ids == (1, 2, 3, 4, 5, 6)
    assert contact.joint_obstruction.maximum_thumb_shortfall_rad == pytest.approx(0.12)
    assert contact.joint_obstruction.maximum_opposing_finger_shortfall_rad == pytest.approx(0.15)
    controller.begin_retention_test()
    retention = controller.verify_retention_at_lifted_checkpoint()
    assert retention.grasp_close == contact
    assert retention.joint_obstruction.has_opposed_obstruction
    assert retention.verified_blocked_motor_ids == (1, 3)
    assert retention.maximum_contact_shift_rad == pytest.approx(0.0)
    controller.finish_retention_test()
    controller.timeout_and_close()


def test_grasp_rejects_one_sided_obstruction_even_with_opposed_pressure(dex3_sdk):
    bindings, _ = dex3_sdk
    clock = ManualClock(0.0)
    target = np.asarray((0.0, -0.5, -0.5, -0.6, -0.8, -0.7, -0.9))
    empty_close = target + np.asarray((0.0, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02))
    contact = empty_close.copy()
    contact[3] += 0.20
    zero = np.zeros(7)
    controller_config = grasp_test_config()
    observer = UnitreeDex3StateObserver(controller_config, bindings=bindings, clock=clock)
    emit_pair(zero, zero)

    def advance(_duration_s):
        clock.advance(0.1)
        emit_pair(
            contact,
            zero,
            left_pressure=opposed_pressure(),
        )

    controller = UnitreeDex3PostureController(
        controller_config, observer=observer, clock=clock, sleep=advance
    )
    with pytest.raises(Dex3GraspNotAcquiredError, match=r"thumb=-?0\.0000rad"):
        controller.command_close_for_retention_test(
            active_side="left",
            left_target_q_rad=target,
            right_target_q_rad=zero,
            empty_close_reference_q_rad=empty_close,
            minimum_opposed_shortfall_rad=0.05,
            label="test cube grasp",
        )
    controller.timeout_and_close()


def test_retention_rejects_lost_thumb_obstruction_even_with_opposed_pressure(dex3_sdk):
    bindings, _ = dex3_sdk
    clock = ManualClock(0.0)
    target = np.asarray((0.0, -0.5, -0.5, -0.6, -0.8, -0.7, -0.9))
    empty_close = target + np.asarray((0.0, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02))
    measured = empty_close.copy()
    measured[1] += 0.12
    measured[3] += 0.15
    zero = np.zeros(7)
    lifted = False
    controller_config = grasp_test_config()
    observer = UnitreeDex3StateObserver(controller_config, bindings=bindings, clock=clock)
    emit_pair(zero, zero)

    def advance(_duration_s):
        clock.advance(0.1)
        q = measured.copy()
        if lifted:
            q[1] = empty_close[1]
        emit_pair(q, zero, left_pressure=opposed_pressure())

    controller = UnitreeDex3PostureController(
        controller_config, observer=observer, clock=clock, sleep=advance
    )
    controller.command_close_for_retention_test(
        active_side="left",
        left_target_q_rad=target,
        right_target_q_rad=zero,
        empty_close_reference_q_rad=empty_close,
        minimum_opposed_shortfall_rad=0.05,
        label="test cube grasp",
    )
    controller.begin_retention_test()
    lifted = True
    clock.advance(0.1)
    lifted_q = measured.copy()
    lifted_q[1] = empty_close[1]
    emit_pair(lifted_q, zero, left_pressure=opposed_pressure())
    with pytest.raises(Dex3RetentionLostError, match="opposed empty-close obstruction was lost"):
        controller.verify_retention_at_lifted_checkpoint()
    controller.timeout_and_close()


def test_retention_allows_obstructed_thumb_and_finger_ids_to_change(dex3_sdk):
    bindings, _ = dex3_sdk
    clock = ManualClock(0.0)
    target = np.asarray((0.0, -0.5, -0.5, -0.6, -0.8, -0.7, -0.9))
    empty_close = target + np.asarray((0.0, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02))
    measured = empty_close.copy()
    measured[1] += 0.12
    measured[3] += 0.15
    zero = np.zeros(7)
    controller_config = grasp_test_config()
    observer = UnitreeDex3StateObserver(controller_config, bindings=bindings, clock=clock)
    emit_pair(zero, zero)

    def advance(_duration_s):
        clock.advance(0.1)
        emit_pair(measured, zero)

    controller = UnitreeDex3PostureController(
        controller_config, observer=observer, clock=clock, sleep=advance
    )
    first = controller.command_close_for_retention_test(
        active_side="left",
        left_target_q_rad=target,
        right_target_q_rad=zero,
        empty_close_reference_q_rad=empty_close,
        minimum_opposed_shortfall_rad=0.05,
        label="test cube grasp",
    )
    assert first.blocked_motor_ids == (1, 3)

    controller.begin_retention_test()
    measured[1] = empty_close[1]
    measured[3] = empty_close[3]
    measured[2] += 0.11
    measured[5] += 0.14
    clock.advance(0.1)
    emit_pair(measured, zero)
    retention = controller.verify_retention_at_lifted_checkpoint()

    assert retention.verified_blocked_motor_ids == (2, 5)
    assert retention.joint_obstruction.maximum_thumb_shortfall_rad == pytest.approx(0.11)
    assert retention.joint_obstruction.maximum_opposing_finger_shortfall_rad == pytest.approx(0.14)
    controller.timeout_and_close()


@pytest.mark.parametrize(
    "label,measured_q,expected",
    [
        (
            "successful grasp 20260817T230114Z",
            (
                -0.02069699,
                0.46204895,
                0.82251316,
                -0.48448423,
                -0.82004499,
                -0.88435078,
                -0.97266388,
            ),
            True,
        ),
        (
            "successful lifted checkpoint 20260817T230114Z",
            (
                -0.02069954,
                0.46200064,
                0.82253462,
                -0.48448959,
                -0.82913196,
                -0.88441694,
                -0.97291344,
            ),
            True,
        ),
        (
            "missed grasp 20260817T231230Z",
            (
                -0.02069494,
                0.57080734,
                0.97793740,
                -0.68271422,
                -0.89992958,
                -0.88438833,
                -0.97264951,
            ),
            False,
        ),
        (
            "missed lifted checkpoint 20260817T231230Z",
            (
                -0.02069571,
                0.57123941,
                0.97809571,
                -0.81622791,
                -0.94625455,
                -0.88445187,
                -0.97287500,
            ),
            False,
        ),
        (
            "visually caged grasp 20260818T153501Z",
            (
                -0.02220099,
                0.45059583,
                0.82264823,
                -0.72241592,
                -0.97440737,
                -0.87111926,
                -0.97244912,
            ),
            True,
        ),
    ],
)
def test_commissioned_empty_close_classifies_retained_real_runs(
    label,
    measured_q,
    expected,
):
    del label
    empty_close = (
        -0.02220278,
        0.57173234,
        0.97812259,
        -0.87878031,
        -0.97525454,
        -0.88341504,
        -0.97280872,
    )
    evidence = classify_dex3_opposed_joint_obstruction(
        active_side="left",
        measured_q_rad=measured_q,
        empty_close_reference_q_rad=empty_close,
        closing_direction=(0.0, 1.0, 1.0, -1.0, -1.0, -1.0, -1.0),
        minimum_opposed_shortfall_rad=0.05,
    )

    assert evidence.has_opposed_obstruction is expected


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
    left_subscriber.emit(
        HandState(
            [Motor(q=float("nan")) for _ in range(7)],
            [PressureGroup(list(row)) for row in unloaded_pressure()],
        )
    )
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
