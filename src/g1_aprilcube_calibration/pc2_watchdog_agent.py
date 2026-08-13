"""Ephemeral PC2-side heartbeat and controller-handoff watchdog.

The normal arm-SDK path uses only the Python standard library. The seated
debug path creates one persistent MotionSwitcher client from the private,
pinned raw Unitree SDK runtime before service release, then reuses that client
to restore the AI motion service. Fault recovery stops at verified zero-torque
FSM 0; clean seated completion follows the observed 0 -> 1 -> 3 transition.
The laptop copies this file to a unique path under ``/tmp`` for one
motion-producing command. The separate standing arm-SDK path can still request
verified Damp as its terminal action.
"""

from __future__ import annotations

import argparse
import fcntl
import os
import re
import selectors
import signal
import subprocess
import sys
import time
from pathlib import Path

READY_MARKER = "WATCHDOG_READY"
DAMPING_MARKER = "WATCHDOG_DAMPING"
ZERO_TORQUE_RECOVERY_MARKER = "WATCHDOG_RECOVERING_ZERO_TORQUE"
DAMPED_MARKER = "WATCHDOG_DAMPED"
ZERO_TORQUE_MARKER = "WATCHDOG_ZERO_TORQUE"
DISARMED_MARKER = "WATCHDOG_DISARMED"
SEATED_MARKER = "WATCHDOG_SEATED"
ERROR_MARKER = "WATCHDOG_ERROR"
_FSM_ID_PATTERN = re.compile(r"current fsm_id:\s*(-?\d+)")
FSM_QUERY_ATTEMPTS = 3
FSM_QUERY_RETRY_DELAY_S = 0.1


def _initialize_raw_motion_switcher(
    *, interface: str, timeout_s: float, initialize_factory: bool = True
):
    """Create the persistent raw-DDS client before any service release."""

    try:
        from unitree_sdk2py.comm.motion_switcher.motion_switcher_client import (
            MotionSwitcherClient,
        )
        from unitree_sdk2py.core.channel import ChannelFactoryInitialize
    except (ImportError, OSError) as error:
        raise RuntimeError("PC2 raw Unitree SDK MotionSwitcher runtime is unavailable") from error
    try:
        if initialize_factory:
            ChannelFactoryInitialize(0, interface)
        client = MotionSwitcherClient()
        client.SetTimeout(timeout_s)
        client.Init()
    except Exception as error:  # SDK bindings do not expose narrower exceptions.
        raise RuntimeError(
            f"PC2 raw Unitree MotionSwitcher initialization failed: {error}"
        ) from error
    return client


class _Dex3TimeoutPublisher:
    """Pre-created PC2 fallback that can only release both Dex3 hands."""

    MOTOR_COUNT = 7
    LEFT_TOPIC = "rt/dex3/left/cmd"
    RIGHT_TOPIC = "rt/dex3/right/cmd"

    def __init__(self, *, interface: str) -> None:
        try:
            from unitree_sdk2py.core.channel import (
                ChannelFactoryInitialize,
                ChannelPublisher,
            )
            from unitree_sdk2py.idl.default import unitree_hg_msg_dds__HandCmd_
            from unitree_sdk2py.idl.unitree_hg.msg.dds_ import HandCmd_
        except (ImportError, OSError) as error:
            raise RuntimeError("PC2 Unitree Dex3 timeout runtime is unavailable") from error
        try:
            ChannelFactoryInitialize(0, interface)
            self._make_command = unitree_hg_msg_dds__HandCmd_
            self._left = ChannelPublisher(self.LEFT_TOPIC, HandCmd_)
            self._right = ChannelPublisher(self.RIGHT_TOPIC, HandCmd_)
            self._left.Init()
            self._right.Init()
        except Exception as error:  # Unitree bindings expose no narrow exception.
            raise RuntimeError(
                f"PC2 Dex3 timeout publisher initialization failed: {error}"
            ) from error

    @staticmethod
    def _motor_mode(motor_id: int) -> int:
        return (motor_id & 0x0F) | (0x01 << 4) | (0x01 << 7)

    def _message(self):
        message = self._make_command()
        if len(message.motor_cmd) != self.MOTOR_COUNT:
            raise RuntimeError(f"PC2 Dex3 HandCmd has {len(message.motor_cmd)} motors; expected 7")
        for motor_id, motor in enumerate(message.motor_cmd):
            motor.mode = self._motor_mode(motor_id)
            motor.q = 0.0
            motor.dq = 0.0
            motor.tau = 0.0
            motor.kp = 0.0
            motor.kd = 0.0
        return message

    def timeout(self) -> str:
        left = self._message()
        right = self._message()
        for attempt in range(3):
            if self._left.Write(left) is False:
                raise RuntimeError("PC2 left Dex3 timeout publish failed")
            if self._right.Write(right) is False:
                raise RuntimeError("PC2 right Dex3 timeout publish failed")
            if attempt < 2:
                time.sleep(0.01)
        return "dex3_timeout=both_hands"


def _check_motion_service(client) -> dict:
    status, current = client.CheckMode()
    if status != 0:
        raise RuntimeError(f"PC2 raw Unitree MotionSwitcher CheckMode failed with status {status}")
    if not isinstance(current, dict) or "name" not in current:
        raise TypeError("PC2 raw Unitree MotionSwitcher returned no mode record")
    return current


def _request_ai_motion_service(client) -> str:
    """Request AI service restoration and return diagnostic evidence.

    On this G1 the SelectMode reply has been observed to carry a nonzero status
    even though the AI service starts. The locomotion FSM read-back is therefore
    the authoritative postcondition; this function records but does not
    interpret the SelectMode reply.
    """

    current = _check_motion_service(client)
    name = str(current.get("name", "")).strip()
    if name:
        return f"already_active={name}"
    status, _ = client.SelectMode("ai")
    return f"SelectMode_status={status}"


def _wait_for_locomotion_fsm(
    executable: Path,
    *,
    expected_fsm_id: int | None,
    attempt_timeout_s: float,
    total_timeout_s: float,
) -> tuple[int, str]:
    """Wait for the restored locomotion service and an optional exact FSM."""

    deadline = time.monotonic() + total_timeout_s
    observations: list[str] = []
    while time.monotonic() < deadline:
        remaining = max(deadline - time.monotonic(), 0.0)
        try:
            fsm_id, detail = _get_fsm_id(
                executable,
                timeout_s=min(attempt_timeout_s, remaining),
                attempts=1,
            )
            observations.append(f"fsm_id={fsm_id} | {detail}")
            if expected_fsm_id is None or fsm_id == expected_fsm_id:
                return fsm_id, detail
        except RuntimeError as error:
            observations.append(str(error))
        observations = observations[-4:]
        time.sleep(0.05)
    expectation = (
        "a responsive locomotion FSM"
        if expected_fsm_id is None
        else f"locomotion FSM {expected_fsm_id}"
    )
    detail = " || ".join(observations) or "no FSM response"
    raise RuntimeError(f"AI service did not reach {expectation}: {detail}")


def _read_active_motion_service(client) -> str:
    """Verify the raw PC2 recovery client before debug takeover."""

    current = _check_motion_service(client)
    name = str(current.get("name", "")).strip()
    if not name:
        raise RuntimeError("PC2 motion-switcher reported no active service before debug takeover")
    return name


def _run_unitree_client(
    executable: Path,
    option: str,
    success_marker: str,
    *,
    timeout_s: float,
) -> str:
    """Run Unitree's client until its one-shot action reports success."""

    process = subprocess.Popen(
        [str(executable), f"--{option}"],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        bufsize=0,
    )
    assert process.stdout is not None
    selector = selectors.DefaultSelector()
    selector.register(process.stdout, selectors.EVENT_READ)
    deadline = time.monotonic() + timeout_s
    output = bytearray()
    success_bytes = success_marker.encode()
    try:
        while success_bytes not in output and time.monotonic() < deadline:
            remaining = max(deadline - time.monotonic(), 0.0)
            events = selector.select(min(remaining, 0.1))
            for key, _ in events:
                chunk = os.read(key.fileobj.fileno(), 4096)
                if chunk:
                    output.extend(chunk)
            if process.poll() is not None:
                remainder = process.stdout.read()
                if remainder:
                    output.extend(remainder)
                break
    finally:
        selector.close()
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=1.0)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=1.0)
        remainder = process.stdout.read()
        if remainder:
            output.extend(remainder)
    lines = bytes(output).decode("utf-8", errors="replace").splitlines()
    detail = " | ".join(lines[-8:]) or "no output"
    if success_bytes not in output:
        raise RuntimeError(f"Unitree client --{option} did not report success: {detail}")
    return detail


def _get_fsm_id(
    executable: Path,
    *,
    timeout_s: float,
    attempts: int = FSM_QUERY_ATTEMPTS,
) -> tuple[int, str]:
    """Read the Unitree locomotion FSM ID from the installed client."""

    if attempts <= 0:
        raise ValueError("FSM query attempts must be positive")
    errors: list[str] = []
    for attempt in range(attempts):
        try:
            detail = _run_unitree_client(
                executable,
                "get_fsm_id",
                "current fsm_id:",
                timeout_s=timeout_s,
            )
            match = _FSM_ID_PATTERN.search(detail)
            if match is None:
                raise RuntimeError(f"Unitree client returned no parseable FSM ID: {detail}")
            return int(match.group(1)), detail
        except RuntimeError as error:
            errors.append(str(error))
            if attempt + 1 < attempts:
                time.sleep(FSM_QUERY_RETRY_DELAY_S)
    raise RuntimeError(
        f"GetFsmId failed after {attempts} read-only attempts: " + " || ".join(errors)
    )


def _restore_seated_via_damp(
    motion_switcher,
    executable: Path,
    *,
    client_timeout_s: float,
) -> str:
    """Restore AI and verify the clean FSM 0 -> 1 -> 3 transition."""

    selection = _request_ai_motion_service(motion_switcher)
    current_fsm_id, initial_detail = _wait_for_locomotion_fsm(
        executable,
        expected_fsm_id=None,
        attempt_timeout_s=client_timeout_s,
        total_timeout_s=client_timeout_s * 5.0,
    )
    if current_fsm_id == 3:
        return f"{selection} | already seated | {initial_detail}"
    if current_fsm_id == 0:
        damp_detail = _run_unitree_client(
            executable,
            "damp",
            "Damp command sent",
            timeout_s=client_timeout_s,
        )
        _, damp_fsm_detail = _wait_for_locomotion_fsm(
            executable,
            expected_fsm_id=1,
            attempt_timeout_s=client_timeout_s,
            total_timeout_s=client_timeout_s * 5.0,
        )
    elif current_fsm_id == 1:
        damp_detail = "already in Damp FSM 1"
        damp_fsm_detail = initial_detail
    else:
        raise RuntimeError(
            f"clean seated restoration requires initial FSM 0 or 1; observed FSM {current_fsm_id}"
        )
    sit_detail = _run_unitree_client(
        executable,
        "sit",
        "Sit command sent",
        timeout_s=client_timeout_s,
    )
    _, seated_fsm_detail = _wait_for_locomotion_fsm(
        executable,
        expected_fsm_id=3,
        attempt_timeout_s=client_timeout_s,
        total_timeout_s=client_timeout_s * 5.0,
    )
    return (
        f"{selection} | {initial_detail} | {damp_detail} | {damp_fsm_detail} | "
        f"{sit_detail} | {seated_fsm_detail}"
    )


def _emit(message: str) -> None:
    try:
        print(message, flush=True)
    except BrokenPipeError:
        # The safety action must still finish if the laptop link disappeared.
        pass


def run_watchdog(args: argparse.Namespace) -> int:
    if not 0.2 <= args.heartbeat_timeout_s <= 10.0:
        raise ValueError("heartbeat timeout must be within [0.2, 10.0] seconds")
    if not args.token or any(character.isspace() for character in args.token):
        raise ValueError("watchdog token must be non-empty and contain no whitespace")
    if args.restore_motion_service_before_loco and (
        re.fullmatch(r"[A-Za-z0-9_.:-]+", args.motion_switcher_interface) is None
        or not Path("/sys/class/net", args.motion_switcher_interface).exists()
    ):
        raise ValueError(
            f"invalid PC2 raw MotionSwitcher network interface: {args.motion_switcher_interface}"
        )
    if args.manage_dex3 and (
        re.fullmatch(r"[A-Za-z0-9_.:-]+", args.dex3_interface) is None
        or not Path("/sys/class/net", args.dex3_interface).exists()
    ):
        raise ValueError(f"invalid PC2 Dex3 network interface: {args.dex3_interface}")
    if (
        args.manage_dex3
        and args.restore_motion_service_before_loco
        and args.dex3_interface != args.motion_switcher_interface
    ):
        raise ValueError("PC2 Dex3 and raw MotionSwitcher must share one Unitree DDS interface")
    if not args.damp_executable.is_file() or not os.access(args.damp_executable, os.X_OK):
        raise ValueError(f"G1 locomotion client is not executable: {args.damp_executable}")

    lock_stream = args.lock_file.open("w", encoding="utf-8")
    try:
        fcntl.flock(lock_stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as error:
        lock_stream.close()
        raise RuntimeError("another PC2 damping watchdog is already armed") from error

    signal_reason: list[str] = []

    def request_damping(signum, _frame) -> None:
        signal_reason.append(f"signal_{signum}")

    signal.signal(signal.SIGHUP, request_damping)
    signal.signal(signal.SIGTERM, request_damping)
    signal.signal(signal.SIGINT, request_damping)
    motion_switcher = None
    dex3_timeout_publisher = None
    heartbeat_count = 0
    heartbeat_started_s: float | None = None
    last_heartbeat_s: float | None = None
    maximum_heartbeat_gap_s = 0.0

    def heartbeat_detail(now: float | None = None) -> str:
        current = time.monotonic() if now is None else now
        baseline = last_heartbeat_s or heartbeat_started_s or current
        return (
            f"heartbeat_count={heartbeat_count} "
            f"last_gap_s={max(current - baseline, 0.0):.6f} "
            f"max_gap_s={maximum_heartbeat_gap_s:.6f}"
        )

    def timeout_dex3() -> str:
        if dex3_timeout_publisher is None:
            return "dex3_timeout=not_managed"
        return dex3_timeout_publisher.timeout()

    def damp(reason: str) -> int:
        _emit(f"{DAMPING_MARKER} {reason} | {heartbeat_detail()}")
        errors: list[str] = []
        dex3_detail = ""
        dex3_error: str | None = None
        try:
            dex3_detail = timeout_dex3()
        except RuntimeError as error:
            dex3_error = str(error)
        for _attempt in range(args.damp_attempts):
            try:
                if args.restore_motion_service_before_loco:
                    if motion_switcher is None:
                        raise RuntimeError(
                            "raw MotionSwitcher was not initialized before takeover"
                        )
                    selection = _request_ai_motion_service(motion_switcher)
                    _, readiness = _wait_for_locomotion_fsm(
                        args.damp_executable,
                        expected_fsm_id=None,
                        attempt_timeout_s=args.client_timeout_s,
                        total_timeout_s=args.client_timeout_s * 5.0,
                    )
                    mode_name = f"ai ({selection}; {readiness})"
                else:
                    mode_name = "already-active"
                detail = _run_unitree_client(
                    args.damp_executable,
                    "damp",
                    "Damp command sent",
                    timeout_s=args.client_timeout_s,
                )
                fsm_id, fsm_detail = _get_fsm_id(
                    args.damp_executable,
                    timeout_s=args.client_timeout_s,
                )
                if fsm_id != 1:
                    raise RuntimeError(
                        f"Unitree Damp request did not reach locomotion FSM 1: {fsm_detail}"
                    )
                if dex3_error is not None:
                    _emit(
                        f"{ERROR_MARKER} Dex3 timeout failed although body reached "
                        f"Damp FSM 1: {dex3_error}"
                    )
                    return 4
                _emit(f"{DAMPED_MARKER} fsm_id=1 mode={mode_name} | {dex3_detail} | {detail}")
                return 0
            except (RuntimeError, TypeError) as error:
                errors.append(str(error))
                time.sleep(0.1)
        _emit(f"{ERROR_MARKER} damping failed: {' || '.join(errors)}")
        return 4

    def restore_zero_torque(reason: str) -> int:
        """Restore AI and verify its observed zero-torque initialization."""

        _emit(f"{ZERO_TORQUE_RECOVERY_MARKER} {reason} | {heartbeat_detail()}")
        if motion_switcher is None:
            _emit(
                f"{ERROR_MARKER} zero-torque restoration failed: "
                "raw MotionSwitcher was not initialized before takeover"
            )
            return 4
        errors: list[str] = []
        dex3_detail = ""
        dex3_error: str | None = None
        try:
            dex3_detail = timeout_dex3()
        except RuntimeError as error:
            dex3_error = str(error)
        for _attempt in range(args.damp_attempts):
            try:
                selection = _request_ai_motion_service(motion_switcher)
                fsm_id, readiness = _wait_for_locomotion_fsm(
                    args.damp_executable,
                    expected_fsm_id=None,
                    attempt_timeout_s=args.client_timeout_s,
                    total_timeout_s=args.client_timeout_s * 5.0,
                )
                zero_torque_detail = "AI initialized in zero-torque FSM 0"
                if fsm_id != 0:
                    zero_torque_detail = _run_unitree_client(
                        args.damp_executable,
                        "zero_torque",
                        "ZeroTorque command sent",
                        timeout_s=args.client_timeout_s,
                    )
                    fsm_id, readiness = _wait_for_locomotion_fsm(
                        args.damp_executable,
                        expected_fsm_id=0,
                        attempt_timeout_s=args.client_timeout_s,
                        total_timeout_s=args.client_timeout_s * 5.0,
                    )
                if dex3_error is not None:
                    _emit(
                        f"{ERROR_MARKER} Dex3 timeout failed although body reached "
                        f"zero-torque FSM {fsm_id}: {dex3_error}"
                    )
                    return 4
                _emit(
                    f"{ZERO_TORQUE_MARKER} fsm_id={fsm_id} mode=ai | "
                    f"{dex3_detail} | {selection} | {zero_torque_detail} | {readiness}"
                )
                return 0
            except (RuntimeError, TypeError) as error:
                errors.append(str(error))
                time.sleep(0.1)
        _emit(f"{ERROR_MARKER} zero-torque restoration failed: " + " || ".join(errors))
        return 4

    def recover(reason: str) -> int:
        if args.restore_motion_service_before_loco:
            return restore_zero_torque(reason)
        return damp(reason)

    def restore_seated() -> int:
        """Restore AI and follow the observed FSM 0 -> 1 -> 3 path."""

        if motion_switcher is None:
            _emit(
                f"{ERROR_MARKER} seated-controller restore failed: "
                "raw MotionSwitcher was not initialized before takeover"
            )
            return 5
        errors: list[str] = []
        dex3_detail = ""
        dex3_error: str | None = None
        try:
            dex3_detail = timeout_dex3()
        except RuntimeError as error:
            dex3_error = str(error)
        for _attempt in range(args.damp_attempts):
            try:
                detail = _restore_seated_via_damp(
                    motion_switcher,
                    args.damp_executable,
                    client_timeout_s=args.client_timeout_s,
                )
                if dex3_error is not None:
                    _emit(
                        f"{ERROR_MARKER} Dex3 timeout failed although body reached "
                        f"seated FSM 3: {dex3_error}"
                    )
                    return 5
                _emit(f"{SEATED_MARKER} fsm_id=3 mode=ai | {dex3_detail} | {detail}")
                return 0
            except (RuntimeError, TypeError) as error:
                errors.append(str(error))
                time.sleep(0.1)
        _emit(f"{ERROR_MARKER} seated-controller restore failed: " + " || ".join(errors))
        return 5

    try:
        try:
            ready_fields: list[str] = []
            if args.manage_dex3:
                dex3_timeout_publisher = _Dex3TimeoutPublisher(interface=args.dex3_interface)
                ready_fields.append("dex3_timeout=ready")
            if args.skip_initial_fsm_query:
                ready_fields.append("initial_fsm_query=skipped")
            else:
                initial_fsm_id, readiness = _get_fsm_id(
                    args.damp_executable,
                    timeout_s=args.client_timeout_s,
                )
                ready_fields.extend((f"fsm_id={initial_fsm_id}", readiness))
            if args.restore_motion_service_before_loco:
                motion_switcher = _initialize_raw_motion_switcher(
                    interface=args.motion_switcher_interface,
                    timeout_s=args.client_timeout_s,
                    initialize_factory=not args.manage_dex3,
                )
                motion_name = _read_active_motion_service(motion_switcher)
                ready_fields.append(f"motion_switcher={motion_name}")
        except (RuntimeError, TypeError) as error:
            _emit(f"{ERROR_MARKER} read-only safety preflight failed: {error}")
            return 6
        _emit(f"{READY_MARKER} " + " | ".join(ready_fields))
        heartbeat_started_s = time.monotonic()
        deadline = heartbeat_started_s + args.heartbeat_timeout_s
        input_buffer = b""
        while True:
            if signal_reason:
                return recover(signal_reason[-1])
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return recover("heartbeat_timeout")
            readable, _, _ = select_with_signals(remaining)
            if signal_reason:
                return recover(signal_reason[-1])
            if not readable:
                continue
            chunk = os.read(sys.stdin.fileno(), 4096)
            if not chunk:
                return recover("watchdog_channel_eof")
            input_buffer += chunk
            lines = input_buffer.split(b"\n")
            input_buffer = lines.pop()
            for raw_line in lines:
                line = raw_line.decode("utf-8", errors="replace")
                parts = line.strip().split(maxsplit=2)
                if len(parts) < 2 or parts[1] != args.token:
                    continue
                command = parts[0]
                if command == "PING":
                    now = time.monotonic()
                    if last_heartbeat_s is not None:
                        maximum_heartbeat_gap_s = max(
                            maximum_heartbeat_gap_s,
                            now - last_heartbeat_s,
                        )
                    heartbeat_count += 1
                    last_heartbeat_s = now
                    deadline = now + args.heartbeat_timeout_s
                elif command == "DISARM":
                    try:
                        detail = timeout_dex3()
                    except RuntimeError as error:
                        _emit(f"{ERROR_MARKER} Dex3 timeout failed: {error}")
                        return 4
                    _emit(f"{DISARMED_MARKER} {detail}")
                    return 0
                elif command == "DAMP":
                    reason = parts[2] if len(parts) == 3 else "explicit_request"
                    return damp(reason.replace("\n", " "))
                elif command == "RESTORE_ZERO_TORQUE":
                    reason = parts[2] if len(parts) == 3 else "explicit_request"
                    return restore_zero_torque(reason.replace("\n", " "))
                elif command == "RESTORE_SEATED":
                    restore_status = restore_seated()
                    if restore_status == 0:
                        return 0
                    # Remain armed so laptop cleanup can explicitly restore
                    # zero-torque FSM 0 after a failed clean transition.
                    deadline = time.monotonic() + args.heartbeat_timeout_s
    finally:
        try:
            fcntl.flock(lock_stream, fcntl.LOCK_UN)
        finally:
            lock_stream.close()
        if args.self_delete:
            try:
                Path(__file__).unlink()
            except OSError:
                pass


def select_with_signals(timeout_s: float) -> tuple[list, list, list]:
    """Keep the event loop testable while allowing signals to interrupt select."""

    import select

    return select.select([sys.stdin], [], [], min(timeout_s, 0.1))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--token", required=True)
    parser.add_argument("--heartbeat-timeout-s", type=float, required=True)
    parser.add_argument("--client-timeout-s", type=float, default=3.0)
    parser.add_argument("--damp-attempts", type=int, default=3)
    parser.add_argument("--skip-initial-fsm-query", action="store_true")
    parser.add_argument(
        "--restore-motion-service-before-loco",
        action="store_true",
        help="select and verify the AI service before Sit or Damp RPCs",
    )
    parser.add_argument(
        "--motion-switcher-interface",
        default="eth0",
        help="PC2 interface used by the persistent raw Unitree SDK client",
    )
    parser.add_argument(
        "--manage-dex3",
        action="store_true",
        help="pre-create both Dex3 timeout publishers for every terminal path",
    )
    parser.add_argument(
        "--dex3-interface",
        default="eth0",
        help="PC2 interface for the Dex3 timeout publishers",
    )
    parser.add_argument("--damp-executable", type=Path, required=True)
    parser.add_argument(
        "--lock-file",
        type=Path,
        default=Path("/tmp/g1-aprilcube-damping-watchdog.lock"),
    )
    parser.add_argument("--self-delete", action="store_true")
    return parser


def main() -> int:
    try:
        return run_watchdog(build_parser().parse_args())
    except (OSError, RuntimeError, TypeError, ValueError) as error:
        _emit(f"{ERROR_MARKER} {error}")
        return 3


if __name__ == "__main__":
    raise SystemExit(main())
