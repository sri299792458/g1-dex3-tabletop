from __future__ import annotations

import time
from copy import deepcopy
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import ClassVar

import numpy as np
import pytest

from g1_aprilcube_calibration.clock import ManualClock
from g1_aprilcube_calibration.transports.base import ArmCommand
from g1_aprilcube_calibration.transports.unitree_arm_sdk import (
    ARM_WEIGHT_SLOT,
    UnitreeArmSDKTransport,
    UnitreeLowStateObserver,
    UnitreeSDKBindings,
    UnitreeTorsoIMUObserver,
    UnitreeTransportConfig,
)
from g1_aprilcube_calibration.transports.unitree_debug_lowcmd import (
    UnitreeDebugLowCmdConfig,
    UnitreeDebugLowCmdTransport,
    UnitreeMotionModeManager,
    UnitreeMotionSwitcherBindings,
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
class LowCommand:
    mode_pr: int = 99
    mode_machine: int = 0
    motor_cmd: list[Motor] = field(default_factory=lambda: [Motor() for _ in range(35)])
    crc: int = 0


@dataclass
class LowState:
    mode_machine: int
    tick: int
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

    def Close(self):
        self.closed = True

    def emit(self, message):
        self.callback(message)


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
def sdk():
    Subscriber.instances.clear()
    Publisher.instances.clear()
    initialized = []
    bindings = UnitreeSDKBindings(
        initialize=lambda domain, interface: initialized.append((domain, interface)),
        publisher_type=Publisher,
        subscriber_type=Subscriber,
        low_command_type=LowCommand,
        low_state_type=LowState,
        make_low_command=LowCommand,
        calculate_crc=lambda message: 0x1234,
    )
    return bindings, initialized


def state(*, mode=5, tick=17):
    return LowState(
        mode_machine=mode,
        tick=tick,
        motor_state=[
            Motor(q=index / 10, dq=-index / 100, tau_est=index / 1000) for index in range(35)
        ],
    )


def config():
    return UnitreeTransportConfig(network_interface="enp3s0", domain_id=4)


def test_read_only_observer_never_constructs_a_publisher(sdk):
    bindings, initialized = sdk
    clock = ManualClock(2.0)
    observer = UnitreeLowStateObserver(
        config(),
        bindings=bindings,
        clock=clock,
        utc_now=lambda: "2026-08-02T12:00:00Z",
    )

    assert initialized == [(4, "enp3s0")]
    assert Publisher.instances == []
    with pytest.raises(RuntimeError, match="no valid"):
        observer.observe()
    Subscriber.instances[0].emit(state())
    sample = observer.observe()
    assert sample.source_sequence == 17
    assert sample.mode_machine == 5
    np.testing.assert_allclose(sample.position, np.arange(29) / 10)
    np.testing.assert_allclose(sample.velocity, -np.arange(29) / 100)
    np.testing.assert_allclose(sample.estimated_torque, np.arange(29) / 1000)
    observer.close()
    assert Subscriber.instances[0].closed


def test_torso_imu_observer_reuses_factory_and_normalizes_orientation(sdk):
    bindings, initialized = sdk
    bindings = UnitreeSDKBindings(
        initialize=bindings.initialize,
        publisher_type=bindings.publisher_type,
        subscriber_type=bindings.subscriber_type,
        low_command_type=bindings.low_command_type,
        low_state_type=bindings.low_state_type,
        make_low_command=bindings.make_low_command,
        calculate_crc=bindings.calculate_crc,
        imu_state_type=object,
    )
    clock = ManualClock(2.0)
    lowstate = UnitreeLowStateObserver(config(), bindings=bindings, clock=clock)
    torso = UnitreeTorsoIMUObserver(
        config(), lowstate_observer=lowstate, clock=clock
    )

    assert initialized == [(4, "enp3s0")]
    Subscriber.instances[1].emit(SimpleNamespace(quaternion=[2.0, 0.0, 0.0, 0.0]))
    sample = torso.observe()
    np.testing.assert_allclose(sample.quaternion_wxyz, [1.0, 0.0, 0.0, 0.0])
    assert sample.receipt_monotonic_s == 2.0
    torso.close()
    lowstate.close()
    assert Subscriber.instances[1].closed


def test_observer_forwards_only_valid_samples(sdk):
    bindings, _ = sdk
    received = []
    observer = UnitreeLowStateObserver(config(), bindings=bindings, on_sample=received.append)
    Subscriber.instances[0].emit(state(tick=2))
    bad = state(tick=3)
    bad.motor_state = bad.motor_state[:2]
    Subscriber.instances[0].emit(bad)
    assert [sample.source_sequence for sample in received] == [2]
    observer.close()


def test_invalid_lowstate_is_rejected_without_replacing_last_good_state(sdk):
    bindings, _ = sdk
    observer = UnitreeLowStateObserver(config(), bindings=bindings)
    Subscriber.instances[0].emit(state(tick=3))
    bad = state(tick=4)
    bad.motor_state[4].q = float("nan")
    Subscriber.instances[0].emit(bad)
    assert observer.observe().source_sequence == 3
    assert "NaN" in observer.last_error
    observer.close()


def test_lowstate_without_tau_est_is_rejected(sdk):
    bindings, _ = sdk
    observer = UnitreeLowStateObserver(config(), bindings=bindings)
    bad = state(tick=4)
    bad.motor_state[4] = SimpleNamespace(q=0.0, dq=0.0)

    Subscriber.instances[0].emit(bad)

    with pytest.raises(RuntimeError, match="tau_est"):
        observer.observe()
    observer.close()


def test_transport_maps_only_arms_and_weight_and_requires_zero_close(sdk):
    bindings, _ = sdk
    clock = ManualClock(5.0)
    transport = UnitreeArmSDKTransport(
        config(),
        bindings=bindings,
        clock=clock,
        utc_now=lambda: "2026-08-02T12:00:00Z",
    )
    assert Publisher.instances[0].messages == []
    Subscriber.instances[0].emit(state())
    kp_scale = np.ones(14)
    kd_scale = np.ones(14)
    tau_ff = np.linspace(-2.0, 2.0, 14)
    kp_scale[:7] = 0.5
    kd_scale[:7] = 0.25
    command = ArmCommand.create(
        np.arange(14) / 20,
        weight=0.6,
        issued_monotonic_s=5,
        kp_scale14=kp_scale,
        kd_scale14=kd_scale,
        tau_ff14=tau_ff,
    )
    transport.send_command(command)

    sent = Publisher.instances[0].messages[-1]
    assert sent.mode_pr == 0
    assert sent.mode_machine == 5
    assert sent.crc == 0x1234
    assert sent.motor_cmd[ARM_WEIGHT_SLOT].q == pytest.approx(0.6)
    for offset, index in enumerate(range(15, 29)):
        motor = sent.motor_cmd[index]
        assert motor.mode == 1
        assert motor.q == pytest.approx(command.q14[offset])
        assert motor.dq == 0
        assert motor.tau == pytest.approx(tau_ff[offset])
        base = (40.0, 1.5) if index in {19, 20, 21, 26, 27, 28} else (80, 3)
        expected = (base[0] * kp_scale[offset], base[1] * kd_scale[offset])
        assert (motor.kp, motor.kd) == expected
    for index in list(range(15)) + list(range(30, 35)):
        assert sent.motor_cmd[index] == Motor()

    with pytest.raises(RuntimeError, match="terminal weight-zero"):
        transport.close()
    transport.send_command(ArmCommand.create(command.q14, weight=0.0, issued_monotonic_s=5))
    transport.close()
    assert Publisher.instances[0].closed
    assert Subscriber.instances[0].closed


def test_transport_attaches_publisher_to_existing_read_only_observer(sdk):
    bindings, initialized = sdk
    observer = UnitreeLowStateObserver(config(), bindings=bindings)

    assert len(Subscriber.instances) == 1
    assert Publisher.instances == []
    transport = UnitreeArmSDKTransport(config(), observer=observer)

    assert initialized == [(4, "enp3s0")]
    assert len(Subscriber.instances) == 1
    assert len(Publisher.instances) == 1
    Subscriber.instances[0].emit(state())
    assert transport.observe().source_sequence == 17
    transport.close()
    assert Subscriber.instances[0].closed


def test_transport_rejects_mismatched_existing_observer(sdk):
    bindings, _ = sdk
    observer = UnitreeLowStateObserver(config(), bindings=bindings)
    mismatched = UnitreeTransportConfig(network_interface="other")

    with pytest.raises(ValueError, match="configuration does not match"):
        UnitreeArmSDKTransport(mismatched, observer=observer)

    assert Publisher.instances == []
    observer.close()


def test_transport_refuses_wrong_mode_before_publish(sdk):
    bindings, _ = sdk
    transport = UnitreeArmSDKTransport(config(), bindings=bindings)
    Subscriber.instances[0].emit(state(mode=4))
    with pytest.raises(RuntimeError, match="mode_machine=4"):
        transport.send_command(ArmCommand.create(np.zeros(14), weight=0, issued_monotonic_s=0))
    assert Publisher.instances[0].messages == []
    transport.close()


@pytest.mark.parametrize("name", ["kp_scale14", "kd_scale14"])
@pytest.mark.parametrize(
    "gain_scale",
    [np.ones(13), np.full(14, -0.1), np.full(14, np.nan)],
)
def test_arm_command_rejects_invalid_gain_scales(name, gain_scale):
    with pytest.raises(ValueError, match="gain scale"):
        ArmCommand.create(
            np.zeros(14),
            weight=0.0,
            issued_monotonic_s=0.0,
            **{name: gain_scale},
        )


@pytest.mark.parametrize(
    "torque",
    [np.ones(13), np.full(14, np.nan), np.full(14, np.inf)],
)
def test_arm_command_rejects_invalid_torque_feedforward(torque):
    with pytest.raises(ValueError, match="torque feedforward"):
        ArmCommand.create(
            np.zeros(14),
            weight=0.0,
            issued_monotonic_s=0.0,
            tau_ff14=torque,
        )


def test_transport_can_close_nonzero_only_after_external_damping(sdk):
    bindings, _ = sdk
    transport = UnitreeArmSDKTransport(config(), bindings=bindings)
    Subscriber.instances[0].emit(state())
    transport.send_command(ArmCommand.create(np.zeros(14), weight=1.0, issued_monotonic_s=0))
    transport.close_after_external_takeover()


class MotionSwitcher:
    def __init__(self, *, active_name="ai", release_status=0):
        self.active_name = active_name
        self.release_status = release_status
        self.timeout = None
        self.initialized = False
        self.release_calls = 0

    def SetTimeout(self, timeout):
        self.timeout = timeout

    def Init(self):
        self.initialized = True

    def CheckMode(self):
        return 0, {"name": self.active_name, "form": "0"}

    def ReleaseMode(self):
        self.release_calls += 1
        if self.release_status == 0:
            self.active_name = ""
        return self.release_status, None


class SlowMotionSwitcher(MotionSwitcher):
    def __init__(self, *, delay_s=0.08):
        super().__init__()
        self.delay_s = delay_s

    def ReleaseMode(self):
        time.sleep(self.delay_s)
        return super().ReleaseMode()


def _debug_manager(client):
    return UnitreeMotionModeManager(
        UnitreeDebugLowCmdConfig(),
        bindings=UnitreeMotionSwitcherBindings(client_factory=lambda: client),
        sleep=lambda _duration: None,
    )


def test_debug_lowcmd_seeds_complete_measured_body_and_changes_only_arms(sdk):
    bindings, _ = sdk
    observer = UnitreeLowStateObserver(config(), bindings=bindings)
    Subscriber.instances[0].emit(state())
    switcher = MotionSwitcher()
    transport = UnitreeDebugLowCmdTransport(
        config(),
        UnitreeDebugLowCmdConfig(),
        observer=observer,
        mode_manager=_debug_manager(switcher),
    )
    measured = observer.observe()
    q14 = np.concatenate((measured.left_q, measured.right_q))
    q14[:7] += np.linspace(0.0, 0.01, 7)
    tau_ff = np.linspace(-3.0, 3.0, 14)
    transport.send_command(
        ArmCommand.create(
            q14,
            weight=0.0,
            issued_monotonic_s=0.0,
            tau_ff14=tau_ff,
        )
    )

    sent = Publisher.instances[-1].messages[-1]
    assert Publisher.instances[-1].topic == "rt/lowcmd"
    assert switcher.initialized
    assert switcher.timeout == 5.0
    assert switcher.release_calls == 1
    assert transport.direct_control_active
    assert sent.mode_pr == 0
    assert sent.mode_machine == 5
    assert sent.crc == 0x1234
    for index in range(15):
        motor = sent.motor_cmd[index]
        assert motor.mode == 1
        assert motor.q == pytest.approx(measured.position[index])
        expected = (80.0, 3.0) if index in {4, 10} else (300.0, 3.0)
        assert (motor.kp, motor.kd) == expected
    for offset, index in enumerate(range(15, 29)):
        motor = sent.motor_cmd[index]
        assert motor.mode == 1
        assert motor.q == pytest.approx(q14[offset])
        assert motor.dq == 0.0
        assert motor.tau == 0.0
        expected = (40.0, 1.5) if index in {19, 20, 21, 26, 27, 28} else (80, 3)
        assert (motor.kp, motor.kd) == expected
    for index in range(29, 35):
        assert sent.motor_cmd[index] == Motor()

    transport.send_command(
        ArmCommand.create(
            q14,
            weight=0.5,
            issued_monotonic_s=0.0,
            tau_ff14=tau_ff,
        )
    )
    ramped = Publisher.instances[-1].messages[-1]
    for offset, index in enumerate(range(15, 29)):
        assert ramped.motor_cmd[index].tau == pytest.approx(0.5 * tau_ff[offset])

    with pytest.raises(RuntimeError, match="verified external controller takeover"):
        transport.close()
    transport.close_after_external_takeover()
    assert Publisher.instances[-1].closed
    assert Subscriber.instances[0].closed


def test_debug_lowcmd_refuses_nonzero_displacement_before_mode_release(sdk):
    bindings, _ = sdk
    observer = UnitreeLowStateObserver(config(), bindings=bindings)
    Subscriber.instances[0].emit(state())
    switcher = MotionSwitcher()
    transport = UnitreeDebugLowCmdTransport(
        config(),
        UnitreeDebugLowCmdConfig(),
        observer=observer,
        mode_manager=_debug_manager(switcher),
    )
    measured = observer.observe()
    q14 = np.concatenate((measured.left_q, measured.right_q))
    q14[0] += 0.021

    with pytest.raises(RuntimeError, match="non-zero-displacement"):
        transport.send_command(ArmCommand.create(q14, weight=1.0, issued_monotonic_s=0.0))

    assert switcher.release_calls == 0
    assert Publisher.instances[-1].messages == []
    transport.close()


def test_debug_lowcmd_detects_seated_body_drift(sdk):
    bindings, _ = sdk
    observer = UnitreeLowStateObserver(config(), bindings=bindings)
    Subscriber.instances[0].emit(state())
    transport = UnitreeDebugLowCmdTransport(
        config(),
        UnitreeDebugLowCmdConfig(body_hold_position_tolerance_rad=0.08),
        observer=observer,
        mode_manager=_debug_manager(MotionSwitcher()),
    )
    measured = observer.observe()
    q14 = np.concatenate((measured.left_q, measured.right_q))
    command = ArmCommand.create(q14, weight=1.0, issued_monotonic_s=0.0)
    transport.send_command(command)
    drifted = state(tick=18)
    drifted.motor_state[3].q += 0.081
    Subscriber.instances[0].emit(drifted)

    with pytest.raises(RuntimeError, match="held seated body drifted"):
        transport.send_command(command)

    transport.close_after_external_takeover()


def test_debug_lowcmd_release_failure_still_requires_external_takeover(sdk):
    bindings, _ = sdk
    observer = UnitreeLowStateObserver(config(), bindings=bindings)
    Subscriber.instances[0].emit(state())
    switcher = MotionSwitcher(release_status=7)
    transport = UnitreeDebugLowCmdTransport(
        config(),
        UnitreeDebugLowCmdConfig(),
        observer=observer,
        mode_manager=_debug_manager(switcher),
    )
    measured = observer.observe()
    command = ArmCommand.create(
        np.concatenate((measured.left_q, measured.right_q)),
        weight=0.0,
        issued_monotonic_s=0.0,
    )

    with pytest.raises(RuntimeError, match="ReleaseMode failed with status 7"):
        transport.send_command(command)

    assert transport.requires_external_takeover
    assert Publisher.instances[-1].messages == []
    with pytest.raises(RuntimeError, match="verified external controller takeover"):
        transport.close()
    transport.close_after_external_takeover()


def test_debug_takeover_pulses_keepalive_during_blocking_release():
    switcher = SlowMotionSwitcher()
    manager = UnitreeMotionModeManager(
        UnitreeDebugLowCmdConfig(motion_switch_poll_interval_s=0.01),
        bindings=UnitreeMotionSwitcherBindings(client_factory=lambda: switcher),
    )
    pulses = []

    manager.enter_debug(keepalive=lambda: pulses.append(time.monotonic()))

    assert manager.debug_mode_verified
    assert len(pulses) >= 3


def test_debug_takeover_keeps_heartbeat_alive_through_first_lowcmd_write(sdk):
    bindings, _ = sdk
    observer = UnitreeLowStateObserver(config(), bindings=bindings)
    Subscriber.instances[0].emit(state())
    pulses = []
    transport = UnitreeDebugLowCmdTransport(
        config(),
        UnitreeDebugLowCmdConfig(motion_switch_poll_interval_s=0.01),
        observer=observer,
        mode_manager=UnitreeMotionModeManager(
            UnitreeDebugLowCmdConfig(motion_switch_poll_interval_s=0.01),
            bindings=UnitreeMotionSwitcherBindings(client_factory=lambda: MotionSwitcher()),
        ),
        ownership_keepalive=lambda: pulses.append(time.monotonic()),
    )
    publisher = Publisher.instances[-1]
    immediate_write = publisher.Write

    def slow_first_write(message):
        time.sleep(0.08)
        return immediate_write(message)

    publisher.Write = slow_first_write
    measured = observer.observe()
    command = ArmCommand.create(
        np.concatenate((measured.left_q, measured.right_q)),
        weight=0.0,
        issued_monotonic_s=0.0,
    )

    transport.send_command(command)

    assert len(pulses) >= 3
    assert len(publisher.messages) == 1
    transport.close_after_external_takeover()


def test_failed_takeover_keepalive_prevents_first_lowcmd_packet(sdk):
    bindings, _ = sdk
    observer = UnitreeLowStateObserver(config(), bindings=bindings)
    Subscriber.instances[0].emit(state())
    switcher = SlowMotionSwitcher()
    pulse_count = 0

    def fail_after_initial_pulse() -> None:
        nonlocal pulse_count
        pulse_count += 1
        if pulse_count >= 2:
            raise RuntimeError("heartbeat channel closed")

    transport = UnitreeDebugLowCmdTransport(
        config(),
        UnitreeDebugLowCmdConfig(motion_switch_poll_interval_s=0.01),
        observer=observer,
        mode_manager=UnitreeMotionModeManager(
            UnitreeDebugLowCmdConfig(motion_switch_poll_interval_s=0.01),
            bindings=UnitreeMotionSwitcherBindings(client_factory=lambda: switcher),
        ),
        ownership_keepalive=fail_after_initial_pulse,
    )
    measured = observer.observe()
    command = ArmCommand.create(
        np.concatenate((measured.left_q, measured.right_q)),
        weight=0.0,
        issued_monotonic_s=0.0,
    )

    with pytest.raises(RuntimeError, match="ownership keepalive failed"):
        transport.send_command(command)

    assert switcher.release_calls == 1
    assert transport.requires_external_takeover
    assert Publisher.instances[-1].messages == []
    transport.close_after_external_takeover()


def test_motion_manager_refuses_takeover_when_already_in_debug_mode(sdk):
    del sdk
    switcher = MotionSwitcher(active_name="")
    manager = _debug_manager(switcher)

    with pytest.raises(RuntimeError, match="no active Unitree motion service"):
        manager.enter_debug()

    assert switcher.release_calls == 0
    assert Publisher.instances == []
    assert Subscriber.instances == []
