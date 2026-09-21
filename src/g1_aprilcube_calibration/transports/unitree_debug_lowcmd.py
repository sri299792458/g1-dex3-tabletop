"""Unitree G1 debug-mode full-body transport for seated arm execution.

This is a narrow derivative of Unitree's pinned ``G1_29_ArmController`` debug
path.  It differs deliberately in three ways:

* channel creation remains lazy and explicit;
* the first complete 29-joint command is seeded from one fresh LowState sample;
* every non-commanded joint remains fixed at that takeover sample.

``rt/lowcmd`` has no arm blend-weight field.  Position/gain ownership is still
immediate, while model-based torque feedforward is multiplied by
``ArmCommand.weight`` so the first measured-state packet contains zero added
torque and the existing acquisition ramp introduces it continuously. Ownership
ends only after a separate controller takeover has been verified.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np

from g1_aprilcube_calibration.joint_map import (
    G1_MODE_MACHINE,
    LEFT_ARM_INDICES,
    RIGHT_ARM_INDICES,
)
from g1_aprilcube_calibration.transports.base import ArmCommand
from g1_aprilcube_calibration.transports.unitree_arm_sdk import (
    HG_MOTOR_COUNT,
    UnitreeLowStateObserver,
    UnitreeSDKBindings,
    UnitreeTransportConfig,
)

DEBUG_LOW_COMMAND_TOPIC = "rt/lowcmd"
_ARM_INDICES = LEFT_ARM_INDICES + RIGHT_ARM_INDICES
_BODY_INDICES = tuple(range(15))
# Matches G1_29_ArmController._Is_weak_motor at the pinned upstream revision:
# only the two ankle-pitch joints use the lower body gain. Arm gains are handled
# separately below.
_WEAK_BODY_INDICES = frozenset((4, 10))
_WRIST_INDICES = frozenset((19, 20, 21, 26, 27, 28))


@dataclass(frozen=True, slots=True)
class UnitreeDebugLowCmdConfig:
    """Pinned Unitree debug gains plus explicit commissioning limits."""

    command_topic: str = DEBUG_LOW_COMMAND_TOPIC
    body_strong_kp: float = 300.0
    body_strong_kd: float = 3.0
    body_weak_kp: float = 80.0
    body_weak_kd: float = 3.0
    activation_position_tolerance_rad: float = 0.02
    body_hold_position_tolerance_rad: float = 0.08
    motion_switch_timeout_s: float = 5.0
    motion_switch_poll_interval_s: float = 0.05

    def __post_init__(self) -> None:
        if self.command_topic != DEBUG_LOW_COMMAND_TOPIC:
            raise ValueError("debug command topic must be exactly rt/lowcmd")
        values = (
            self.body_strong_kp,
            self.body_strong_kd,
            self.body_weak_kp,
            self.body_weak_kd,
            self.activation_position_tolerance_rad,
            self.body_hold_position_tolerance_rad,
            self.motion_switch_timeout_s,
            self.motion_switch_poll_interval_s,
        )
        if not all(np.isfinite(value) and value > 0 for value in values):
            raise ValueError("debug gains, limits, and timeouts must be positive")
        if self.motion_switch_poll_interval_s >= self.motion_switch_timeout_s:
            raise ValueError("motion-switch poll interval must be below its timeout")


@dataclass(frozen=True, slots=True)
class UnitreeMotionSwitcherBindings:
    """Injectable MotionSwitcher surface for offline tests."""

    client_factory: Callable[[], Any]

    @classmethod
    def load(cls) -> UnitreeMotionSwitcherBindings:
        try:
            from unitree_sdk2py.comm.motion_switcher.motion_switcher_client import (
                MotionSwitcherClient,
            )
        except (ImportError, OSError) as error:
            raise RuntimeError(
                "Unitree MotionSwitcher unavailable; run tools/install_hardware_dependencies.sh"
            ) from error
        return cls(client_factory=MotionSwitcherClient)


class UnitreeMotionModeManager:
    """Bounded transition from an active Unitree motion service to debug mode."""

    def __init__(
        self,
        config: UnitreeDebugLowCmdConfig,
        *,
        bindings: UnitreeMotionSwitcherBindings | None = None,
        monotonic: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.config = config
        self._monotonic = monotonic
        self._sleep = sleep
        factory = (bindings or UnitreeMotionSwitcherBindings.load()).client_factory
        self._client = factory()
        self._client.SetTimeout(config.motion_switch_timeout_s)
        self._client.Init()
        self.release_attempted = False
        self.debug_mode_verified = False
        self.released_mode_name: str | None = None

    def enter_debug(self, *, keepalive: Callable[[], None] | None = None) -> None:
        """Release the active motion service under a temporary keepalive."""

        with _TakeoverKeepalive(
            keepalive,
            interval_s=self.config.motion_switch_poll_interval_s,
        ) as keepalive_guard:
            self.enter_debug_guarded(keepalive_guard)

    def enter_debug_guarded(self, keepalive_guard: _TakeoverKeepalive) -> None:
        """Release the motion service using an already-running guard.

        The transport uses this form so the same guard remains alive until the
        first complete measured-state ``rt/lowcmd`` write has returned.  There
        must be no unguarded interval between releasing seated control and that
        first packet.
        """

        if self.release_attempted:
            raise RuntimeError("debug-mode ownership transition was already attempted")
        status, result = self._checked_mode()
        del status
        keepalive_guard.check()
        active_name = str(result.get("name", "")).strip()
        if not active_name:
            raise RuntimeError(
                "refusing debug takeover because no active Unitree motion service "
                "was reported before release"
            )
        self.released_mode_name = active_name
        self.release_attempted = True
        release_status, _ = self._client.ReleaseMode()
        keepalive_guard.check()
        if release_status != 0:
            raise RuntimeError(f"Unitree ReleaseMode failed with status {release_status}")

        deadline = self._monotonic() + self.config.motion_switch_timeout_s
        while self._monotonic() < deadline:
            _, current = self._checked_mode()
            keepalive_guard.check()
            if not str(current.get("name", "")).strip():
                self.debug_mode_verified = True
                return
            self._sleep(self.config.motion_switch_poll_interval_s)
        raise RuntimeError("Unitree motion service remained active after ReleaseMode timeout")

    def _checked_mode(self) -> tuple[int, dict[str, Any]]:
        status, result = self._client.CheckMode()
        if status != 0:
            raise RuntimeError(f"Unitree CheckMode failed with status {status}")
        if not isinstance(result, dict) or "name" not in result:
            raise RuntimeError("Unitree CheckMode returned no parseable mode record")
        return status, result


class UnitreeDebugLowCmdTransport:
    """Complete G1 low-level command transport with a fixed seated body hold."""

    def __init__(
        self,
        transport_config: UnitreeTransportConfig,
        debug_config: UnitreeDebugLowCmdConfig,
        *,
        observer: UnitreeLowStateObserver,
        mode_manager: UnitreeMotionModeManager | None = None,
        sdk_bindings: UnitreeSDKBindings | None = None,
        ownership_keepalive: Callable[[], None] | None = None,
    ) -> None:
        if observer.config != transport_config:
            raise ValueError("existing observer configuration does not match")
        if sdk_bindings is not None and sdk_bindings is not observer.bindings:
            raise ValueError("SDK bindings do not match the existing observer")
        self.transport_config = transport_config
        self.debug_config = debug_config
        self.bindings = observer.bindings
        self._observer = observer
        self._mode_manager = mode_manager or UnitreeMotionModeManager(debug_config)
        self._ownership_keepalive = ownership_keepalive
        self._publisher = self.bindings.publisher_type(
            debug_config.command_topic, self.bindings.low_command_type
        )
        self._publisher.Init()
        self._message = self.bindings.make_low_command()
        if len(self._message.motor_cmd) != HG_MOTOR_COUNT:
            self._publisher.Close()
            raise ValueError(
                f"HG LowCmd has {len(self._message.motor_cmd)} motors; expected {HG_MOTOR_COUNT}"
            )
        self._message.mode_pr = 0
        self._body_hold_q: np.ndarray | None = None
        self._closed = False
        self.command_count = 0

    @property
    def last_state_error(self) -> str | None:
        return self._observer.last_error

    @property
    def ownership_transition_attempted(self) -> bool:
        return self._mode_manager.release_attempted

    @property
    def direct_control_active(self) -> bool:
        return self._mode_manager.debug_mode_verified

    @property
    def requires_external_takeover(self) -> bool:
        return self.ownership_transition_attempted or self.command_count > 0

    def observe(self):
        return self._observer.observe()

    def send_command(self, command: ArmCommand) -> None:
        if self._closed:
            raise RuntimeError("Unitree debug lowcmd transport is closed")
        sample = self._observer.observe()
        if sample.mode_machine != G1_MODE_MACHINE:
            raise RuntimeError(
                f"refusing lowcmd in mode_machine={sample.mode_machine}; "
                f"expected {G1_MODE_MACHINE}"
            )
        if self._body_hold_q is None:
            measured_q14 = np.concatenate((sample.left_q, sample.right_q))
            initial_error = float(np.max(np.abs(measured_q14 - np.asarray(command.q14))))
            if initial_error > self.debug_config.activation_position_tolerance_rad:
                raise RuntimeError(
                    "refusing non-zero-displacement debug takeover: arm target "
                    f"differs from measured state by {initial_error:.4f}rad; "
                    "limit is "
                    f"{self.debug_config.activation_position_tolerance_rad:.4f}rad"
                )
            self._seed_complete_command(sample.position)
            with _TakeoverKeepalive(
                self._ownership_keepalive,
                interval_s=self.debug_config.motion_switch_poll_interval_s,
            ) as keepalive_guard:
                self._mode_manager.enter_debug_guarded(keepalive_guard)
                self._write_command(command, sample.mode_machine)
                keepalive_guard.check()
            return
        else:
            body_error = float(
                np.max(
                    np.abs(
                        sample.position[np.asarray(_BODY_INDICES)]
                        - self._body_hold_q[np.asarray(_BODY_INDICES)]
                    )
                )
            )
            if body_error > self.debug_config.body_hold_position_tolerance_rad:
                raise RuntimeError(
                    f"held seated body drifted by {body_error:.4f}rad; limit is "
                    f"{self.debug_config.body_hold_position_tolerance_rad:.4f}rad"
                )

        self._write_command(command, sample.mode_machine)

    def _write_command(self, command: ArmCommand, mode_machine: int) -> None:
        self._message.mode_machine = mode_machine
        for offset, motor_index in enumerate(_ARM_INDICES):
            motor = self._message.motor_cmd[motor_index]
            motor.q = command.q14[offset]
            motor.dq = 0.0
            motor.tau = command.weight * command.tau_ff14[offset]
            if motor_index in _WRIST_INDICES:
                motor.kp = self.transport_config.wrist_kp * command.kp_scale14[offset]
                motor.kd = self.transport_config.wrist_kd * command.kd_scale14[offset]
            else:
                motor.kp = self.transport_config.shoulder_elbow_kp * command.kp_scale14[offset]
                motor.kd = self.transport_config.shoulder_elbow_kd * command.kd_scale14[offset]
        self._message.crc = self.bindings.calculate_crc(self._message)
        result = self._publisher.Write(self._message)
        if result is False:
            raise RuntimeError("Unitree debug lowcmd publish failed")
        self.command_count += 1

    def _seed_complete_command(self, full_q: np.ndarray) -> None:
        self._body_hold_q = np.asarray(full_q, dtype=np.float64).copy()
        self._message.mode_machine = G1_MODE_MACHINE
        for motor_index in range(29):
            motor = self._message.motor_cmd[motor_index]
            motor.mode = 1
            motor.q = float(self._body_hold_q[motor_index])
            motor.dq = 0.0
            motor.tau = 0.0
            if motor_index in _BODY_INDICES:
                weak = motor_index in _WEAK_BODY_INDICES
                motor.kp = (
                    self.debug_config.body_weak_kp if weak else self.debug_config.body_strong_kp
                )
                motor.kd = (
                    self.debug_config.body_weak_kd if weak else self.debug_config.body_strong_kd
                )
            elif motor_index in _WRIST_INDICES:
                motor.kp = self.transport_config.wrist_kp
                motor.kd = self.transport_config.wrist_kd
            else:
                motor.kp = self.transport_config.shoulder_elbow_kp
                motor.kd = self.transport_config.shoulder_elbow_kd

    def close(self) -> None:
        if self._closed:
            return
        if self.requires_external_takeover:
            raise RuntimeError(
                "refusing to close an attempted debug lowcmd session before a "
                "verified external controller takeover"
            )
        self._closed = True
        self._publisher.Close()
        self._observer.close()

    def close_after_external_takeover(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._publisher.Close()
        self._observer.close()


class _TakeoverKeepalive:
    """Pulse an independent guard while a MotionSwitcher RPC blocks.

    No lowcmd packet is published until this context has stopped and its final
    ``check`` has passed. A lost watchdog channel therefore cannot be followed
    by a late first motor command.
    """

    def __init__(
        self,
        callback: Callable[[], None] | None,
        *,
        interval_s: float,
    ) -> None:
        self._callback = callback
        self._interval_s = interval_s
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._error: Exception | None = None

    def __enter__(self) -> _TakeoverKeepalive:  # noqa: PYI034
        if self._callback is None:
            return self
        self._pulse()
        self.check()
        self._thread = threading.Thread(
            target=self._run,
            name="g1-debug-takeover-heartbeat",
            daemon=True,
        )
        self._thread.start()
        return self

    def __exit__(self, _exc_type, exc, _traceback) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=max(1.0, self._interval_s * 4.0))
            if self._thread.is_alive() and self._error is None:
                self._error = RuntimeError("ownership keepalive thread did not stop")
        if self._error is not None and exc is not None:
            raise RuntimeError(
                "motion-switch transition failed: "
                f"{type(exc).__name__}: {exc}; PC2 ownership keepalive also "
                f"failed: {type(self._error).__name__}: {self._error}"
            ) from exc
        self.check()

    def check(self) -> None:
        if self._error is not None:
            raise RuntimeError(f"PC2 ownership keepalive failed: {self._error}") from self._error

    def _pulse(self) -> None:
        assert self._callback is not None
        try:
            self._callback()
        except (OSError, RuntimeError, TypeError, ValueError) as error:
            self._error = error
            self._stop.set()

    def _run(self) -> None:
        while not self._stop.wait(self._interval_s):
            self._pulse()
            if self._error is not None:
                return
