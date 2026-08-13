"""Lazy, safety-scoped Unitree HG DDS adapters for the G1 arm SDK channel.

Importing this module never imports CycloneDDS and never opens a channel.  The
read-only observer creates only ``rt/lowstate``.  The command transport creates
``rt/arm_sdk`` explicitly and leaves every non-arm command slot at its default.
"""

from __future__ import annotations

import threading
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np

from g1_aprilcube_calibration.clock import MonotonicClock, SystemClock
from g1_aprilcube_calibration.joint_map import (
    G1_DOF,
    G1_MODE_MACHINE,
    LEFT_ARM_INDICES,
    RIGHT_ARM_INDICES,
)
from g1_aprilcube_calibration.models import RobotStateSample, utc_now_iso
from g1_aprilcube_calibration.transports.base import ArmCommand

LOWSTATE_TOPIC = "rt/lowstate"
ARM_SDK_TOPIC = "rt/arm_sdk"
HG_MOTOR_COUNT = 35
ARM_WEIGHT_SLOT = 29
_ARM_INDICES = LEFT_ARM_INDICES + RIGHT_ARM_INDICES
_WRIST_INDICES = frozenset((19, 20, 21, 26, 27, 28))


@dataclass(frozen=True, slots=True)
class UnitreeTransportConfig:
    """Network and gain values matching Unitree's G1_29 arm controller."""

    network_interface: str
    domain_id: int = 0
    state_topic: str = LOWSTATE_TOPIC
    command_topic: str = ARM_SDK_TOPIC
    shoulder_elbow_kp: float = 80.0
    shoulder_elbow_kd: float = 3.0
    wrist_kp: float = 40.0
    wrist_kd: float = 1.5
    subscriber_queue_length: int = 10

    def __post_init__(self) -> None:
        if not self.network_interface.strip():
            raise ValueError("network_interface must be non-empty")
        if self.domain_id < 0:
            raise ValueError("domain_id must be non-negative")
        if not self.state_topic or not self.command_topic:
            raise ValueError("DDS topics must be non-empty")
        gains = (
            self.shoulder_elbow_kp,
            self.shoulder_elbow_kd,
            self.wrist_kp,
            self.wrist_kd,
        )
        if not all(np.isfinite(gain) and gain >= 0 for gain in gains):
            raise ValueError("motor gains must be finite and non-negative")
        if self.subscriber_queue_length <= 0:
            raise ValueError("subscriber_queue_length must be positive")


@dataclass(frozen=True, slots=True)
class UnitreeSDKBindings:
    """Injectable SDK surface; tests do not need CycloneDDS installed."""

    initialize: Callable[[int, str], None]
    publisher_type: type
    subscriber_type: type
    low_command_type: type
    low_state_type: type
    make_low_command: Callable[[], Any]
    calculate_crc: Callable[[Any], int]

    @classmethod
    def load(cls) -> UnitreeSDKBindings:
        try:
            from unitree_sdk2py.core.channel import (
                ChannelFactoryInitialize,
                ChannelPublisher,
                ChannelSubscriber,
            )
            from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_
            from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_, LowState_
            from unitree_sdk2py.utils.crc import CRC
        except (ImportError, OSError) as error:
            raise RuntimeError(
                "Unitree SDK unavailable; run tools/install_hardware_dependencies.sh "
                "and invoke hardware commands through tools/g1_calib_hardware.sh"
            ) from error
        crc = CRC()
        return cls(
            initialize=ChannelFactoryInitialize,
            publisher_type=ChannelPublisher,
            subscriber_type=ChannelSubscriber,
            low_command_type=LowCmd_,
            low_state_type=LowState_,
            make_low_command=unitree_hg_msg_dds__LowCmd_,
            calculate_crc=crc.Crc,
        )


class UnitreeLowStateObserver:
    """Subscriber-only conversion of complete G1 LowState into core records."""

    def __init__(
        self,
        config: UnitreeTransportConfig,
        *,
        bindings: UnitreeSDKBindings | None = None,
        clock: MonotonicClock | None = None,
        utc_now: Callable[[], str] = utc_now_iso,
        on_sample: Callable[[RobotStateSample], None] | None = None,
    ) -> None:
        self.config = config
        self.bindings = bindings or UnitreeSDKBindings.load()
        self.clock = clock or SystemClock()
        self._utc_now = utc_now
        self._on_sample = on_sample
        self._lock = threading.Lock()
        self._latest: RobotStateSample | None = None
        self._last_error: str | None = None
        self._closed = False

        self.bindings.initialize(config.domain_id, config.network_interface)
        self._subscriber = self.bindings.subscriber_type(
            config.state_topic, self.bindings.low_state_type
        )
        self._subscriber.Init(self._receive, config.subscriber_queue_length)

    @property
    def last_error(self) -> str | None:
        with self._lock:
            return self._last_error

    def _receive(self, message: Any) -> None:
        try:
            motors = message.motor_state
            if len(motors) < G1_DOF:
                raise ValueError(f"LowState has {len(motors)} motors; expected at least {G1_DOF}")
            position = np.asarray([motors[index].q for index in range(G1_DOF)], dtype=np.float64)
            velocity = np.asarray([motors[index].dq for index in range(G1_DOF)], dtype=np.float64)
            estimated_torque = np.asarray(
                [motors[index].tau_est for index in range(G1_DOF)],
                dtype=np.float64,
            )
            sample = RobotStateSample(
                receipt_monotonic_s=self.clock.monotonic(),
                receipt_utc=self._utc_now(),
                mode_machine=int(message.mode_machine),
                position=position,
                velocity=velocity,
                estimated_torque=estimated_torque,
                source_sequence=int(message.tick),
            )
        except (AttributeError, TypeError, ValueError) as error:
            with self._lock:
                self._last_error = f"invalid LowState: {error}"
            return
        with self._lock:
            if not self._closed:
                self._latest = sample
                self._last_error = None
            else:
                return
        if self._on_sample is not None:
            self._on_sample(sample)

    def observe(self) -> RobotStateSample:
        with self._lock:
            if self._closed:
                raise RuntimeError("Unitree LowState observer is closed")
            if self._latest is None:
                suffix = "" if self._last_error is None else f": {self._last_error}"
                raise RuntimeError(f"no valid Unitree LowState received{suffix}")
            return self._latest

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
        close = getattr(self._subscriber, "Close", None)
        if callable(close):
            close()


class UnitreeArmSDKTransport:
    """ArmTransport for ``rt/arm_sdk`` with measured-state seeding upstream."""

    def __init__(
        self,
        config: UnitreeTransportConfig,
        *,
        bindings: UnitreeSDKBindings | None = None,
        clock: MonotonicClock | None = None,
        utc_now: Callable[[], str] = utc_now_iso,
        on_sample: Callable[[RobotStateSample], None] | None = None,
        observer: UnitreeLowStateObserver | None = None,
    ) -> None:
        self.config = config
        if observer is not None:
            if observer.config != config:
                raise ValueError("existing observer configuration does not match")
            if bindings is not None or clock is not None or on_sample is not None:
                raise ValueError("bindings, clock, and on_sample cannot accompany an observer")
            self.bindings = observer.bindings
            self._observer = observer
        else:
            self.bindings = bindings or UnitreeSDKBindings.load()
            self._observer = UnitreeLowStateObserver(
                config,
                bindings=self.bindings,
                clock=clock,
                utc_now=utc_now,
                on_sample=on_sample,
            )
        self._publisher = self.bindings.publisher_type(
            config.command_topic, self.bindings.low_command_type
        )
        self._publisher.Init()
        self._message = self.bindings.make_low_command()
        if len(self._message.motor_cmd) != HG_MOTOR_COUNT:
            self._observer.close()
            self._publisher.Close()
            raise ValueError(
                f"HG LowCmd has {len(self._message.motor_cmd)} motors; expected {HG_MOTOR_COUNT}"
            )
        self._message.mode_pr = 0
        self._last_weight: float | None = None
        self._closed = False
        self.command_count = 0

    @property
    def last_state_error(self) -> str | None:
        return self._observer.last_error

    def observe(self) -> RobotStateSample:
        return self._observer.observe()

    def send_command(self, command: ArmCommand) -> None:
        if self._closed:
            raise RuntimeError("Unitree arm transport is closed")
        sample = self._observer.observe()
        if sample.mode_machine != G1_MODE_MACHINE:
            raise RuntimeError(
                f"refusing arm command in mode_machine={sample.mode_machine}; "
                f"expected {G1_MODE_MACHINE}"
            )
        self._message.mode_machine = sample.mode_machine
        for offset, motor_index in enumerate(_ARM_INDICES):
            motor = self._message.motor_cmd[motor_index]
            motor.mode = 1
            motor.q = command.q14[offset]
            motor.dq = 0.0
            motor.tau = command.tau_ff14[offset]
            if motor_index in _WRIST_INDICES:
                motor.kp = self.config.wrist_kp * command.kp_scale14[offset]
                motor.kd = self.config.wrist_kd * command.kd_scale14[offset]
            else:
                motor.kp = self.config.shoulder_elbow_kp * command.kp_scale14[offset]
                motor.kd = self.config.shoulder_elbow_kd * command.kd_scale14[offset]
        self._message.motor_cmd[ARM_WEIGHT_SLOT].q = command.weight
        self._message.crc = self.bindings.calculate_crc(self._message)
        result = self._publisher.Write(self._message)
        if result is False:
            raise RuntimeError("Unitree arm command publish failed")
        self._last_weight = command.weight
        self.command_count += 1

    def close(self) -> None:
        if self._closed:
            return
        if self.command_count and self._last_weight != 0.0:
            raise RuntimeError(
                "refusing to close after a non-zero blend weight; controller must "
                "publish a terminal weight-zero command"
            )
        self._closed = True
        self._publisher.Close()
        self._observer.close()

    def close_after_external_takeover(self) -> None:
        """Close without weight zero after a verified external takeover."""

        if self._closed:
            return
        self._closed = True
        self._publisher.Close()
        self._observer.close()
