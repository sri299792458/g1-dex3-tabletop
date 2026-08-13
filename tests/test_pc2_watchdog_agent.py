from __future__ import annotations

import os
import select
import subprocess
import sys
import time
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path

import pytest

from g1_aprilcube_calibration import pc2_watchdog_agent


@dataclass
class _HandMotor:
    mode: int = 0
    q: float = 1.0
    dq: float = 1.0
    tau: float = 1.0
    kp: float = 1.0
    kd: float = 1.0


@dataclass
class _HandCommand:
    motor_cmd: list[_HandMotor] = field(default_factory=lambda: [_HandMotor() for _ in range(7)])


class _HandPublisher:
    def __init__(self):
        self.messages = []

    def Write(self, message):
        self.messages.append(deepcopy(message))
        return True


def _fake_loco_client(tmp_path: Path) -> tuple[Path, Path]:
    executable = tmp_path / "fake_g1_loco_client"
    damp_log = tmp_path / "damp.log"
    executable.write_text(
        """#!/usr/bin/env python3
import os
import sys
import time
from pathlib import Path

if "--get_fsm_id" in sys.argv:
    # Match the installed Unitree client: several log lines can arrive in one
    # OS-level read while the process remains alive.  A selector wrapped around
    # TextIOWrapper.readline() loses sight of lines buffered after the first.
    fsm_id = Path(os.environ["FAKE_FSM_STATE"]).read_text()
    os.write(
        sys.stdout.fileno(),
        (
            "Processing command: [get_fsm_id]\\n"
            f"current fsm_id: {fsm_id}\\n"
            "Done processing command: get_fsm_id\\n"
        ).encode(),
    )
elif "--damp" in sys.argv:
    Path(os.environ["FAKE_DAMP_LOG"]).write_text("damped\\n")
    if os.environ.get("FAKE_DAMP_CHANGES_FSM", "1") == "1":
        Path(os.environ["FAKE_FSM_STATE"]).write_text("1")
    print("Damp command sent", flush=True)
elif "--sit" in sys.argv:
    if os.environ.get("FAKE_SIT_CHANGES_FSM", "1") == "1":
        Path(os.environ["FAKE_FSM_STATE"]).write_text("3")
    print("Sit command sent", flush=True)
else:
    raise SystemExit(2)
time.sleep(30)
""",
        encoding="utf-8",
    )
    executable.chmod(0o755)
    return executable, damp_log


def _start_agent(
    tmp_path: Path,
    *,
    token: str,
    heartbeat_timeout_s: float,
    damp_changes_fsm: bool = True,
    skip_initial_fsm_query: bool = False,
) -> tuple[subprocess.Popen[str], Path]:
    executable, damp_log = _fake_loco_client(tmp_path)
    fsm_state = tmp_path / "fsm.state"
    fsm_state.write_text("500", encoding="utf-8")
    environment = os.environ.copy()
    environment["FAKE_DAMP_LOG"] = str(damp_log)
    environment["FAKE_FSM_STATE"] = str(fsm_state)
    environment["FAKE_DAMP_CHANGES_FSM"] = "1" if damp_changes_fsm else "0"
    command = [
        sys.executable,
        "-u",
        str(Path(pc2_watchdog_agent.__file__).resolve()),
        "--token",
        token,
        "--heartbeat-timeout-s",
        str(heartbeat_timeout_s),
        "--client-timeout-s",
        "1",
        "--damp-executable",
        str(executable),
        "--lock-file",
        str(tmp_path / "watchdog.lock"),
    ]
    if skip_initial_fsm_query:
        command.append("--skip-initial-fsm-query")
    process = subprocess.Popen(
        command,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        env=environment,
    )
    return process, damp_log


def _read_until(process: subprocess.Popen[str], marker: str, timeout_s: float) -> str:
    assert process.stdout is not None
    deadline = time.monotonic() + timeout_s
    lines: list[str] = []
    while time.monotonic() < deadline:
        readable, _, _ = select.select([process.stdout], [], [], 0.1)
        if readable:
            line = process.stdout.readline()
            if line:
                lines.append(line.rstrip())
                if line.startswith(marker):
                    return line.rstrip()
        if process.poll() is not None:
            break
    raise AssertionError(f"did not receive {marker}: {lines}")


def test_fsm_query_retries_transient_unitree_timeout(monkeypatch) -> None:
    calls = 0

    def fake_run(*_args, **_kwargs) -> str:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise RuntimeError("Unitree GetFsmId timed out")
        return "Processing command | current fsm_id: 3 | Done processing command"

    monkeypatch.setattr(pc2_watchdog_agent, "_run_unitree_client", fake_run)
    monkeypatch.setattr(pc2_watchdog_agent.time, "sleep", lambda _duration: None)

    fsm_id, detail = pc2_watchdog_agent._get_fsm_id(Path("/unused/fake-client"), timeout_s=3.0)

    assert calls == 2
    assert fsm_id == 3
    assert "current fsm_id: 3" in detail


def test_pc2_dex3_fallback_sends_only_official_timeout_packets(monkeypatch) -> None:
    subject = object.__new__(pc2_watchdog_agent._Dex3TimeoutPublisher)
    subject._make_command = _HandCommand
    subject._left = _HandPublisher()
    subject._right = _HandPublisher()
    monkeypatch.setattr(pc2_watchdog_agent.time, "sleep", lambda _duration: None)

    assert subject.timeout() == "dex3_timeout=both_hands"

    for publisher in (subject._left, subject._right):
        assert len(publisher.messages) == 3
        message = publisher.messages[-1]
        assert [motor.mode for motor in message.motor_cmd] == [
            (index & 0x0F) | (0x01 << 4) | (0x01 << 7) for index in range(7)
        ]
        assert all(
            motor.q == motor.dq == motor.tau == motor.kp == motor.kd == 0.0
            for motor in message.motor_cmd
        )


def test_fsm_query_reports_exhausted_attempts(monkeypatch) -> None:
    calls = 0

    def fake_run(*_args, **_kwargs) -> str:
        nonlocal calls
        calls += 1
        raise RuntimeError("Unitree GetFsmId timed out")

    monkeypatch.setattr(pc2_watchdog_agent, "_run_unitree_client", fake_run)
    monkeypatch.setattr(pc2_watchdog_agent.time, "sleep", lambda _duration: None)

    with pytest.raises(RuntimeError, match="failed after 3 read-only attempts"):
        pc2_watchdog_agent._get_fsm_id(Path("/unused/fake-client"), timeout_s=3.0)

    assert calls == 3


def test_agent_disarms_without_damping(tmp_path: Path) -> None:
    token = "test-token"
    process, damp_log = _start_agent(
        tmp_path,
        token=token,
        heartbeat_timeout_s=1.0,
    )
    try:
        _read_until(process, pc2_watchdog_agent.READY_MARKER, 2.0)
        assert process.stdin is not None
        process.stdin.write(f"PING {token}\nDISARM {token}\n")
        process.stdin.flush()
        _read_until(process, pc2_watchdog_agent.DISARMED_MARKER, 1.0)
        assert process.wait(timeout=1.0) == 0
        assert not damp_log.exists()
    finally:
        if process.poll() is None:
            process.kill()


def test_agent_can_skip_initial_fsm_query(tmp_path: Path) -> None:
    token = "skip-fsm-token"
    process, damp_log = _start_agent(
        tmp_path,
        token=token,
        heartbeat_timeout_s=1.0,
        skip_initial_fsm_query=True,
    )
    try:
        ready = _read_until(process, pc2_watchdog_agent.READY_MARKER, 2.0)
        assert "initial_fsm_query=skipped" in ready
        assert process.stdin is not None
        process.stdin.write(f"DISARM {token}\n")
        process.stdin.flush()
        _read_until(process, pc2_watchdog_agent.DISARMED_MARKER, 1.0)
        assert process.wait(timeout=1.0) == 0
        assert not damp_log.exists()
    finally:
        if process.poll() is None:
            process.kill()


def test_agent_heartbeat_timeout_invokes_supported_damping(tmp_path: Path) -> None:
    process, damp_log = _start_agent(
        tmp_path,
        token="timeout-token",
        heartbeat_timeout_s=0.2,
    )
    try:
        _read_until(process, pc2_watchdog_agent.READY_MARKER, 2.0)
        timeout_line = _read_until(process, pc2_watchdog_agent.DAMPING_MARKER, 2.0)
        assert "heartbeat_count=0" in timeout_line
        assert "last_gap_s=" in timeout_line
        assert "max_gap_s=0.000000" in timeout_line
        result = _read_until(process, pc2_watchdog_agent.DAMPED_MARKER, 2.0)
        assert "Damp command sent" in result
        assert "fsm_id=1" in result
        assert process.wait(timeout=1.0) == 0
        assert damp_log.read_text(encoding="utf-8") == "damped\n"
    finally:
        if process.poll() is None:
            process.kill()


def test_agent_explicit_request_invokes_supported_damping(tmp_path: Path) -> None:
    token = "damp-token"
    process, damp_log = _start_agent(
        tmp_path,
        token=token,
        heartbeat_timeout_s=1.0,
    )
    try:
        _read_until(process, pc2_watchdog_agent.READY_MARKER, 2.0)
        assert process.stdin is not None
        process.stdin.write(f"DAMP {token} operator_interrupt\n")
        process.stdin.flush()
        _read_until(process, pc2_watchdog_agent.DAMPED_MARKER, 2.0)
        assert process.wait(timeout=1.0) == 0
        assert damp_log.exists()
    finally:
        if process.poll() is None:
            process.kill()


def test_ai_motion_service_request_records_indeterminate_status() -> None:
    class MotionSwitcher:
        def __init__(self):
            self.select_calls = []

        def CheckMode(self):
            return 0, {"name": "", "form": ""}

        def SelectMode(self, name):
            self.select_calls.append(name)
            # This exact nonzero reply was observed even though AI started.
            return 7002, None

    client = MotionSwitcher()

    detail = pc2_watchdog_agent._request_ai_motion_service(client)

    assert detail == "SelectMode_status=7002"
    assert client.select_calls == ["ai"]


def test_locomotion_fsm_readback_is_authoritative_after_ai_request(
    monkeypatch,
) -> None:
    results = [
        RuntimeError("service not ready"),
        (0, "current fsm_id: 0"),
    ]

    def fake_get_fsm_id(*_args, **_kwargs):
        result = results.pop(0)
        if isinstance(result, Exception):
            raise result
        return result

    monkeypatch.setattr(pc2_watchdog_agent, "_get_fsm_id", fake_get_fsm_id)
    monkeypatch.setattr(pc2_watchdog_agent.time, "sleep", lambda _duration: None)

    fsm_id, detail = pc2_watchdog_agent._wait_for_locomotion_fsm(
        Path("/unused/fake-client"),
        expected_fsm_id=0,
        attempt_timeout_s=1.0,
        total_timeout_s=1.0,
    )

    assert fsm_id == 0
    assert detail == "current fsm_id: 0"


def test_clean_seated_restore_verifies_zero_damp_then_sit(monkeypatch) -> None:
    expected_fsm_calls = iter(
        [
            (None, (0, "fsm 0")),
            (1, (1, "fsm 1")),
            (3, (3, "fsm 3")),
        ]
    )
    loco_calls = []

    monkeypatch.setattr(
        pc2_watchdog_agent,
        "_request_ai_motion_service",
        lambda _client: "SelectMode_status=7002",
    )

    def fake_wait(_executable, *, expected_fsm_id, **_kwargs):
        expected, result = next(expected_fsm_calls)
        assert expected_fsm_id == expected
        return result

    def fake_loco(_executable, option, marker, *, timeout_s):
        loco_calls.append((option, marker, timeout_s))
        return f"{option} sent"

    monkeypatch.setattr(pc2_watchdog_agent, "_wait_for_locomotion_fsm", fake_wait)
    monkeypatch.setattr(pc2_watchdog_agent, "_run_unitree_client", fake_loco)

    detail = pc2_watchdog_agent._restore_seated_via_damp(
        object(),
        Path("/unused/fake-client"),
        client_timeout_s=3.0,
    )

    assert [call[0] for call in loco_calls] == ["damp", "sit"]
    assert "fsm 0" in detail
    assert "fsm 1" in detail
    assert "fsm 3" in detail


def test_read_only_motion_switcher_preflight_requires_active_service() -> None:
    class MotionSwitcher:
        def __init__(self, name):
            self.name = name
            self.check_calls = 0

        def CheckMode(self):
            self.check_calls += 1
            return 0, {"name": self.name, "form": "0"}

    active = MotionSwitcher("ai")
    assert pc2_watchdog_agent._read_active_motion_service(active) == "ai"
    assert active.check_calls == 1

    with pytest.raises(RuntimeError, match="no active service"):
        pc2_watchdog_agent._read_active_motion_service(MotionSwitcher(""))


def test_agent_rejects_unconfirmed_damping_transition(tmp_path: Path) -> None:
    process, _ = _start_agent(
        tmp_path,
        token="failed-damp-token",
        heartbeat_timeout_s=1.0,
        damp_changes_fsm=False,
    )
    try:
        _read_until(process, pc2_watchdog_agent.READY_MARKER, 2.0)
        assert process.stdin is not None
        process.stdin.write("DAMP failed-damp-token test_failure\n")
        process.stdin.flush()
        result = _read_until(process, pc2_watchdog_agent.ERROR_MARKER, 5.0)
        assert "did not reach locomotion FSM 1" in result
        assert process.wait(timeout=1.0) == 4
    finally:
        if process.poll() is None:
            process.kill()
