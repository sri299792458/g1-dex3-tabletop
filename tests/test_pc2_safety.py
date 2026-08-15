from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from g1_aprilcube_calibration.pc2_safety import (
    PC2DampingWatchdog,
    PC2SafetyConfig,
)


def _fake_ssh(tmp_path: Path) -> Path:
    script = tmp_path / "fake_ssh.py"
    script.write_text(
        """import sys

print("WATCHDOG_READY current fsm_id: 4", flush=True)
for line in sys.stdin:
    command = line.split(maxsplit=1)[0]
    if command == "DISARM":
        print("WATCHDOG_DISARMED", flush=True)
        raise SystemExit(0)
    if command == "DAMP":
        sys.stdout.write(
            "WATCHDOG_DAMPING explicit_request\\n"
            "WATCHDOG_DAMPED fsm_id=1 | Damp command sent\\n"
        )
        sys.stdout.flush()
        raise SystemExit(0)
    if command == "RESTORE_SEATED":
        print("WATCHDOG_SEATED fsm_id=3 mode=ai | FSM 0 -> 1 -> 3", flush=True)
        raise SystemExit(0)
    if command == "RESTORE_ZERO_TORQUE":
        print("WATCHDOG_ZERO_TORQUE fsm_id=0 mode=ai", flush=True)
        raise SystemExit(0)
""",
        encoding="utf-8",
    )
    return script


def _fake_automatic_damp_ssh(tmp_path: Path) -> Path:
    script = tmp_path / "fake_automatic_damp_ssh.py"
    script.write_text(
        """import sys

print("WATCHDOG_READY current fsm_id: 4", flush=True)
for line in sys.stdin:
    if line.startswith("PING"):
        print("WATCHDOG_DAMPED fsm_id=1 | heartbeat timeout", flush=True)
        raise SystemExit(0)
""",
        encoding="utf-8",
    )
    return script


def _subject(
    tmp_path: Path,
    *,
    required_initial_fsm_id: int | None = None,
    query_initial_fsm_id: bool = True,
) -> PC2DampingWatchdog:
    identity = tmp_path / "identity"
    identity.write_text("fake", encoding="utf-8")
    fake_ssh = _fake_ssh(tmp_path)

    def run(*_args, **_kwargs):
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    def popen(_args, **kwargs):
        return subprocess.Popen([sys.executable, "-u", str(fake_ssh)], **kwargs)

    config = PC2SafetyConfig(
        host="unitree@192.168.123.164",
        ssh_identity=identity,
        heartbeat_interval_s=0.05,
        heartbeat_timeout_s=0.2,
        connect_timeout_s=0.5,
        client_timeout_s=0.5,
        required_initial_fsm_id=required_initial_fsm_id,
        query_initial_fsm_id=query_initial_fsm_id,
    )
    return PC2DampingWatchdog(config, run=run, popen=popen)


def test_laptop_watchdog_can_disarm_cleanly(tmp_path: Path) -> None:
    watchdog = _subject(tmp_path, required_initial_fsm_id=4)
    watchdog.start()
    assert watchdog.armed
    assert watchdog.initial_fsm_id == 4
    watchdog.pulse()
    watchdog.disarm()
    assert not watchdog.armed
    assert watchdog.terminal_action == "disarmed"


def test_startup_interrupt_disarms_remote_watchdog_before_reraising(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    watchdog = _subject(tmp_path, required_initial_fsm_id=4)
    wait_for_marker = watchdog._wait_for_marker
    interrupted = False

    def interrupt_first_ready_wait(marker: str, *, timeout_s: float) -> str:
        nonlocal interrupted
        if marker == "WATCHDOG_READY" and not interrupted:
            interrupted = True
            raise KeyboardInterrupt
        return wait_for_marker(marker, timeout_s=timeout_s)

    monkeypatch.setattr(watchdog, "_wait_for_marker", interrupt_first_ready_wait)

    with pytest.raises(KeyboardInterrupt):
        watchdog.start()

    assert not watchdog.armed
    assert watchdog.terminal_action == "disarmed"
    assert watchdog._process is not None
    assert watchdog._process.poll() == 0


def test_laptop_watchdog_can_request_and_confirm_damping(tmp_path: Path) -> None:
    watchdog = _subject(tmp_path)
    watchdog.start()
    watchdog.damp("operator pressed ctrl-c")
    assert not watchdog.armed
    assert watchdog.terminal_action == "damped"
    watchdog.pulse()


def test_explicit_cleanup_accepts_already_completed_automatic_damping(
    tmp_path: Path,
) -> None:
    identity = tmp_path / "identity"
    identity.write_text("fake", encoding="utf-8")
    fake_ssh = _fake_automatic_damp_ssh(tmp_path)

    def run(*_args, **_kwargs):
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    def popen(_args, **kwargs):
        return subprocess.Popen([sys.executable, "-u", str(fake_ssh)], **kwargs)

    watchdog = PC2DampingWatchdog(
        PC2SafetyConfig(
            host="unitree@192.168.123.164",
            ssh_identity=identity,
            heartbeat_interval_s=0.05,
            heartbeat_timeout_s=0.2,
            connect_timeout_s=0.5,
            client_timeout_s=0.5,
        ),
        run=run,
        popen=popen,
    )
    watchdog.start()
    deadline = time.monotonic() + 1.0
    while watchdog._process.poll() is None and time.monotonic() < deadline:
        time.sleep(0.01)

    watchdog.damp("cleanup after local control fault")

    assert not watchdog.armed
    assert watchdog.terminal_action == "damped"


def test_laptop_watchdog_can_restore_seated_via_damp(tmp_path: Path) -> None:
    watchdog = _subject(tmp_path, required_initial_fsm_id=4)
    watchdog.start()
    watchdog.restore_seated()
    assert not watchdog.armed
    assert watchdog.terminal_action == "seated"
    watchdog.pulse()


def test_laptop_watchdog_can_restore_verified_zero_torque(tmp_path: Path) -> None:
    watchdog = _subject(tmp_path, required_initial_fsm_id=4)
    watchdog.start()
    watchdog.restore_zero_torque("debug cleanup")
    assert not watchdog.armed
    assert watchdog.terminal_action == "zero_torque"
    watchdog.pulse()


def test_laptop_watchdog_refuses_non_regular_initial_fsm(tmp_path: Path) -> None:
    watchdog = _subject(tmp_path, required_initial_fsm_id=5)

    with pytest.raises(RuntimeError, match="required commissioned fsm_id=5"):
        watchdog.start()

    assert not watchdog.armed
    assert watchdog.initial_fsm_id == 4
    assert watchdog.terminal_action == "disarmed"


def test_laptop_watchdog_can_skip_initial_fsm_query(tmp_path: Path) -> None:
    watchdog = _subject(tmp_path, query_initial_fsm_id=False)

    assert "--skip-initial-fsm-query" in watchdog._remote_command()
    watchdog.start()

    assert watchdog.armed
    assert watchdog.initial_fsm_id is None
    watchdog.disarm()


def test_debug_watchdog_remote_command_requires_motion_service_restore(
    tmp_path: Path,
) -> None:
    watchdog = _subject(tmp_path)
    object.__setattr__(
        watchdog.config,
        "restore_motion_service_before_loco",
        True,
    )

    remote_command = watchdog._remote_command()
    assert "--restore-motion-service-before-loco" in remote_command
    assert "--motion-switcher-interface eth0" in remote_command
    assert "/g1-aprilcube-watchdog/current/venv/bin/python" in remote_command
    assert "CYCLONEDDS_HOME=" in remote_command


def test_dex3_watchdog_uses_private_runtime_and_both_hand_timeout_flag(
    tmp_path: Path,
) -> None:
    watchdog = _subject(tmp_path)
    object.__setattr__(watchdog.config, "manage_dex3", True)

    remote_command = watchdog._remote_command()

    assert "--manage-dex3" in remote_command
    assert "--dex3-interface eth0" in remote_command
    assert "/g1-aprilcube-watchdog/current/venv/bin/python" in remote_command


def test_watchdog_timeout_reports_prior_output_history(tmp_path: Path) -> None:
    watchdog = _subject(tmp_path)
    watchdog.start()

    with pytest.raises(RuntimeError, match="WATCHDOG_READY"):
        watchdog._wait_for_marker("WATCHDOG_NEVER", timeout_s=0.05)

    assert watchdog._sent_ping_count == 1

    watchdog.disarm()


def test_initial_fsm_requirement_cannot_be_combined_with_skipped_query(
    tmp_path: Path,
) -> None:
    identity = tmp_path / "identity"
    identity.write_text("fake", encoding="utf-8")

    with pytest.raises(ValueError, match="when its query is skipped"):
        PC2SafetyConfig(
            host="unitree@192.168.123.164",
            ssh_identity=identity,
            required_initial_fsm_id=3,
            query_initial_fsm_id=False,
        )
