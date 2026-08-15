"""Laptop-side lifecycle for the ephemeral PC2 damping watchdog."""

from __future__ import annotations

import re
import secrets
import shlex
import subprocess
import threading
import time
from dataclasses import dataclass
from pathlib import Path

from g1_aprilcube_calibration import pc2_watchdog_agent

_HOST_PATTERN = re.compile(r"^[A-Za-z0-9_.-]+@[A-Za-z0-9_.:-]+$")


@dataclass(frozen=True, slots=True)
class PC2SafetyConfig:
    host: str
    ssh_identity: Path
    heartbeat_interval_s: float = 0.1
    heartbeat_timeout_s: float = 0.5
    connect_timeout_s: float = 5.0
    client_timeout_s: float = 3.0
    required_initial_fsm_id: int | None = None
    query_initial_fsm_id: bool = True
    restore_motion_service_before_loco: bool = False
    manage_dex3: bool = False
    remote_ros_setup: Path = Path("/opt/ros/foxy/setup.bash")
    remote_cyclonedds_setup: Path = Path("/home/unitree/cyclonedds_ws/install/setup.bash")
    remote_unitree_setup: Path = Path("/home/unitree/unitree_ros2/install/setup.bash")
    remote_cyclonedds_uri: Path = Path("/home/unitree/cyclonedds_ws/cyclonedds.xml")
    remote_watchdog_python: Path = Path(
        "/home/unitree/.local/share/g1-aprilcube-watchdog/current/venv/bin/python"
    )
    remote_cyclonedds_home: Path = Path("/home/unitree/cyclonedds_ws/install/cyclonedds")
    remote_motion_switcher_interface: str = "eth0"
    remote_dex3_interface: str = "eth0"
    remote_damp_executable: Path = Path(
        "/home/unitree/unitree_ros2/install/unitree_ros2_example/bin/g1_loco_client_example"
    )

    def __post_init__(self) -> None:
        if not _HOST_PATTERN.fullmatch(self.host):
            raise ValueError("PC2 host must have the form user@host")
        identity = self.ssh_identity.expanduser().resolve()
        object.__setattr__(self, "ssh_identity", identity)
        if not identity.is_file():
            raise ValueError(f"PC2 SSH identity does not exist: {identity}")
        if not 0 < self.heartbeat_interval_s < self.heartbeat_timeout_s:
            raise ValueError("heartbeat interval must be positive and below timeout")
        if not 0.2 <= self.heartbeat_timeout_s <= 10.0:
            raise ValueError("heartbeat timeout must be within [0.2, 10.0] seconds")
        if self.connect_timeout_s <= 0 or self.client_timeout_s <= 0:
            raise ValueError("PC2 connection/client timeouts must be positive")
        if self.required_initial_fsm_id is not None and self.required_initial_fsm_id < 0:
            raise ValueError("required initial FSM ID must be non-negative")
        if self.required_initial_fsm_id is not None and not self.query_initial_fsm_id:
            raise ValueError("cannot require an initial FSM ID when its query is skipped")
        if not re.fullmatch(r"[A-Za-z0-9_.:-]+", self.remote_motion_switcher_interface):
            raise ValueError("invalid PC2 raw MotionSwitcher network interface")
        if not re.fullmatch(r"[A-Za-z0-9_.:-]+", self.remote_dex3_interface):
            raise ValueError("invalid PC2 Dex3 network interface")
        for path in (
            self.remote_ros_setup,
            self.remote_cyclonedds_setup,
            self.remote_unitree_setup,
            self.remote_cyclonedds_uri,
            self.remote_watchdog_python,
            self.remote_cyclonedds_home,
            self.remote_damp_executable,
        ):
            if not path.is_absolute():
                raise ValueError("all PC2 safety paths must be absolute")


class PC2DampingWatchdog:
    """Maintain a fail-closed heartbeat channel to a one-session PC2 guard."""

    def __init__(
        self,
        config: PC2SafetyConfig,
        *,
        clock=time.monotonic,
        run=subprocess.run,
        popen=subprocess.Popen,
    ) -> None:
        self.config = config
        self._clock = clock
        self._run = run
        self._popen = popen
        self._token = secrets.token_hex(16)
        self._remote_agent = Path(f"/tmp/g1-aprilcube-damping-watchdog-{self._token}.py")
        self._process: subprocess.Popen[bytes] | None = None
        self._output_buffer = b""
        self._pending_output_lines: list[str] = []
        self._output_history: list[str] = []
        self._last_ping_s: float | None = None
        self._sent_ping_count = 0
        self._maximum_sent_ping_gap_s = 0.0
        self._lock = threading.RLock()
        self._armed = False
        self._terminal_action: str | None = None
        self._initial_fsm_id: int | None = None

    @property
    def armed(self) -> bool:
        with self._lock:
            return self._armed

    @property
    def terminal_action(self) -> str | None:
        with self._lock:
            return self._terminal_action

    @property
    def initial_fsm_id(self) -> int | None:
        with self._lock:
            return self._initial_fsm_id

    def start(self) -> None:
        with self._lock:
            if self._process is not None:
                raise RuntimeError("PC2 damping watchdog was already started")
            source = Path(pc2_watchdog_agent.__file__).resolve()
            ssh_options = self._ssh_options()
            upload = self._run(
                [
                    "scp",
                    *ssh_options,
                    str(source),
                    f"{self.config.host}:{self._remote_agent}",
                ],
                capture_output=True,
                text=True,
                timeout=self.config.connect_timeout_s + 5.0,
                check=False,
            )
            if upload.returncode != 0:
                detail = upload.stderr.strip() or upload.stdout.strip()
                raise RuntimeError(f"failed to stage PC2 damping watchdog: {detail}")

            remote_command = self._remote_command()
            self._process = self._popen(
                ["ssh", "-T", *ssh_options, self.config.host, remote_command],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                bufsize=0,
            )
            startup_timeout_s = (
                self.config.connect_timeout_s
                + (
                    self.config.client_timeout_s * pc2_watchdog_agent.FSM_QUERY_ATTEMPTS
                    if self.config.query_initial_fsm_id
                    else 0.0
                )
                + (
                    pc2_watchdog_agent.FSM_QUERY_RETRY_DELAY_S
                    * (pc2_watchdog_agent.FSM_QUERY_ATTEMPTS - 1)
                    if self.config.query_initial_fsm_id
                    else 0.0
                )
                + 2.0
            )
            try:
                ready = self._wait_for_marker(
                    pc2_watchdog_agent.READY_MARKER,
                    timeout_s=startup_timeout_s,
                )
                self._initial_fsm_id = (
                    self._parse_fsm_id(ready) if self.config.query_initial_fsm_id else None
                )
                required = self.config.required_initial_fsm_id
                if required is not None and self._initial_fsm_id != required:
                    self._send("DISARM")
                    self._wait_for_marker(
                        pc2_watchdog_agent.DISARMED_MARKER,
                        timeout_s=self.config.heartbeat_timeout_s + self.config.connect_timeout_s,
                    )
                    self._terminal_action = "disarmed"
                    self._finish_local_process()
                    raise RuntimeError(
                        "refusing arm ownership from locomotion "
                        f"fsm_id={self._initial_fsm_id}; required commissioned "
                        f"fsm_id={required}"
                    )
                self._armed = True
                self._send_ping(self._clock())
            except BaseException:
                if self._terminal_action is None and self._process.poll() is None:
                    try:
                        self._send("DISARM")
                        self._wait_for_marker(
                            pc2_watchdog_agent.DISARMED_MARKER,
                            timeout_s=startup_timeout_s,
                        )
                        self._armed = False
                        self._terminal_action = "disarmed"
                        self._finish_local_process()
                    except (KeyboardInterrupt, OSError, RuntimeError):
                        self._terminate_local_process()
                else:
                    self._terminate_local_process()
                raise

    @staticmethod
    def _parse_fsm_id(line: str) -> int:
        match = re.search(r"fsm_id\s*(?:=|:)\s*(-?\d+)", line)
        if match is None:
            raise RuntimeError(f"PC2 watchdog returned no parseable FSM ID: {line}")
        return int(match.group(1))

    def pulse(self) -> None:
        with self._lock:
            if not self._armed:
                if self._terminal_action is not None:
                    return
                raise RuntimeError("PC2 damping watchdog is not armed")
            self._check_running()
            now = self._clock()
            if (
                self._last_ping_s is None
                or now - self._last_ping_s >= self.config.heartbeat_interval_s
            ):
                self._send_ping(now)

    def disarm(self) -> None:
        with self._lock:
            if not self._armed:
                if self._terminal_action == "disarmed":
                    return
                raise RuntimeError("PC2 damping watchdog is not armed")
            self._send("DISARM")
            self._wait_for_marker(
                pc2_watchdog_agent.DISARMED_MARKER,
                timeout_s=self.config.heartbeat_timeout_s + self.config.connect_timeout_s,
            )
            self._armed = False
            self._terminal_action = "disarmed"
            self._finish_local_process()

    def damp(self, reason: str) -> None:
        with self._lock:
            if not self._armed:
                if self._terminal_action == "damped":
                    return
                raise RuntimeError("PC2 damping watchdog is not armed")
            if self._process is not None and self._process.poll() is not None:
                self._adopt_completed_automatic_damping()
                return
            sanitized = " ".join(reason.split()) or "unspecified_emergency"
            try:
                self._send("DAMP", sanitized)
            except RuntimeError:
                if self._process is None or self._process.poll() is None:
                    raise
                self._adopt_completed_automatic_damping()
                return
            self._wait_for_marker(
                pc2_watchdog_agent.DAMPED_MARKER,
                timeout_s=(self.config.client_timeout_s * 15)
                + self.config.connect_timeout_s
                + 2.0,
            )
            self._armed = False
            self._terminal_action = "damped"
            self._finish_local_process()

    def _adopt_completed_automatic_damping(self) -> None:
        """Accept an already-finished guard only with its explicit Damp marker."""

        self._wait_for_marker(
            pc2_watchdog_agent.DAMPED_MARKER,
            timeout_s=self.config.connect_timeout_s,
        )
        self._armed = False
        self._terminal_action = "damped"
        self._finish_local_process()

    def restore_seated(self) -> None:
        """Restore AI and verify the clean FSM 0 -> 1 -> 3 transition."""

        with self._lock:
            if not self._armed:
                if self._terminal_action == "seated":
                    return
                raise RuntimeError("PC2 damping watchdog is not armed")
            self._send("RESTORE_SEATED")
            self._wait_for_marker(
                pc2_watchdog_agent.SEATED_MARKER,
                timeout_s=(self.config.client_timeout_s * 40)
                + self.config.connect_timeout_s
                + 2.0,
            )
            self._armed = False
            self._terminal_action = "seated"
            self._finish_local_process()

    def restore_zero_torque(self, reason: str) -> None:
        """Restore the AI service and verify its zero-torque FSM 0 state."""

        with self._lock:
            if not self._armed:
                if self._terminal_action == "zero_torque":
                    return
                raise RuntimeError("PC2 damping watchdog is not armed")
            sanitized = " ".join(reason.split()) or "unspecified_debug_recovery"
            self._send("RESTORE_ZERO_TORQUE", sanitized)
            self._wait_for_marker(
                pc2_watchdog_agent.ZERO_TORQUE_MARKER,
                timeout_s=(self.config.client_timeout_s * 20)
                + self.config.connect_timeout_s
                + 2.0,
            )
            self._armed = False
            self._terminal_action = "zero_torque"
            self._finish_local_process()

    def wait_for_automatic_damping(self) -> None:
        """Stop heartbeats and wait for PC2 to report its timeout action."""

        with self._lock:
            if not self._armed:
                raise RuntimeError("PC2 damping watchdog is not armed")
            self._wait_for_marker(
                pc2_watchdog_agent.DAMPED_MARKER,
                timeout_s=self.config.heartbeat_timeout_s
                + (self.config.client_timeout_s * 15)
                + 2.0,
            )
            self._armed = False
            self._terminal_action = "damped"
            self._finish_local_process()

    def wait_for_automatic_zero_torque(self) -> None:
        """Stop heartbeats and verify automatic AI/FSM-0 restoration."""

        with self._lock:
            if not self._armed:
                raise RuntimeError("PC2 damping watchdog is not armed")
            self._wait_for_marker(
                pc2_watchdog_agent.ZERO_TORQUE_MARKER,
                timeout_s=self.config.heartbeat_timeout_s
                + (self.config.client_timeout_s * 20)
                + 2.0,
            )
            self._armed = False
            self._terminal_action = "zero_torque"
            self._finish_local_process()

    def _send(self, command: str, detail: str | None = None) -> None:
        self._check_running()
        assert self._process is not None and self._process.stdin is not None
        line = f"{command} {self._token}"
        if detail:
            line += f" {detail}"
        try:
            self._process.stdin.write((line + "\n").encode("utf-8"))
            self._process.stdin.flush()
        except (BrokenPipeError, OSError) as error:
            raise RuntimeError("lost the PC2 watchdog channel") from error

    def _send_ping(self, now: float) -> None:
        if self._last_ping_s is not None:
            self._maximum_sent_ping_gap_s = max(
                self._maximum_sent_ping_gap_s,
                now - self._last_ping_s,
            )
        self._send("PING")
        self._sent_ping_count += 1
        self._last_ping_s = now

    def _heartbeat_detail(self) -> str:
        return (
            f"laptop_ping_count={self._sent_ping_count} "
            f"laptop_max_send_gap_s={self._maximum_sent_ping_gap_s:.6f}"
        )

    def _wait_for_marker(self, marker: str, *, timeout_s: float) -> str:
        assert self._process is not None and self._process.stdout is not None
        import selectors

        selector = selectors.DefaultSelector()
        selector.register(self._process.stdout, selectors.EVENT_READ)
        deadline = self._clock() + timeout_s
        output: list[str] = []
        try:
            while self._clock() < deadline:
                while self._pending_output_lines:
                    line = self._pending_output_lines.pop(0)
                    output.append(line)
                    self._output_history.append(line)
                    del self._output_history[:-32]
                    if line.startswith(marker):
                        return line
                    if line.startswith(pc2_watchdog_agent.ERROR_MARKER):
                        raise RuntimeError(f"{line} | {self._heartbeat_detail()}")
                remaining = max(deadline - self._clock(), 0.0)
                for key, _ in selector.select(min(remaining, 0.1)):
                    chunk = key.fileobj.read(4096)
                    if chunk:
                        self._output_buffer += chunk
                        lines = self._output_buffer.split(b"\n")
                        self._output_buffer = lines.pop()
                        self._pending_output_lines.extend(
                            line.decode("utf-8", errors="replace").rstrip("\r") for line in lines
                        )
                if self._process.poll() is not None:
                    if self._pending_output_lines:
                        continue
                    if self._output_buffer:
                        self._pending_output_lines.append(
                            self._output_buffer.decode("utf-8", errors="replace").rstrip("\r")
                        )
                        self._output_buffer = b""
                        continue
                    break
        finally:
            selector.close()
        detail = " | ".join(self._output_history[-8:]) or "no output"
        raise RuntimeError(
            f"PC2 watchdog did not report {marker}: {detail} | {self._heartbeat_detail()}"
        )

    def _check_running(self) -> None:
        if self._process is None:
            raise RuntimeError("PC2 damping watchdog was not started")
        return_code = self._process.poll()
        if return_code is not None:
            raise RuntimeError(
                f"PC2 damping watchdog exited unexpectedly with status {return_code}"
            )

    def _finish_local_process(self) -> None:
        assert self._process is not None
        if self._process.stdin is not None:
            self._process.stdin.close()
        try:
            return_code = self._process.wait(timeout=2.0)
        except subprocess.TimeoutExpired:
            self._process.terminate()
            return_code = self._process.wait(timeout=2.0)
        if return_code != 0:
            raise RuntimeError(f"PC2 damping watchdog exited with status {return_code}")

    def _terminate_local_process(self) -> None:
        if self._process is None or self._process.poll() is not None:
            return
        self._process.terminate()
        try:
            self._process.wait(timeout=2.0)
        except subprocess.TimeoutExpired:
            self._process.kill()
            self._process.wait(timeout=2.0)

    def _ssh_options(self) -> list[str]:
        return _ssh_options(self.config)

    def _remote_command(self) -> str:
        config = self.config
        agent_python = (
            str(config.remote_watchdog_python)
            if config.restore_motion_service_before_loco or config.manage_dex3
            else "python3"
        )
        agent_command = shlex.join(
            [
                agent_python,
                "-u",
                str(self._remote_agent),
                "--token",
                self._token,
                "--heartbeat-timeout-s",
                str(config.heartbeat_timeout_s),
                "--client-timeout-s",
                str(config.client_timeout_s),
                "--damp-executable",
                str(config.remote_damp_executable),
                *(
                    [
                        "--restore-motion-service-before-loco",
                        "--motion-switcher-interface",
                        config.remote_motion_switcher_interface,
                    ]
                    if config.restore_motion_service_before_loco
                    else []
                ),
                *(["--skip-initial-fsm-query"] if not config.query_initial_fsm_id else []),
                *(
                    [
                        "--manage-dex3",
                        "--dex3-interface",
                        config.remote_dex3_interface,
                    ]
                    if config.manage_dex3
                    else []
                ),
                "--self-delete",
            ]
        )
        return "; ".join(
            [
                "set +u",
                f"source {shlex.quote(str(config.remote_ros_setup))}",
                f"source {shlex.quote(str(config.remote_cyclonedds_setup))}",
                f"source {shlex.quote(str(config.remote_unitree_setup))}",
                "export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp",
                f"export CYCLONEDDS_URI={shlex.quote(str(config.remote_cyclonedds_uri))}",
                f"export CYCLONEDDS_HOME={shlex.quote(str(config.remote_cyclonedds_home))}",
                "export G1_WATCHDOG_CYCLONEDDS_LIB="
                + shlex.quote(str(config.remote_cyclonedds_home / "lib")),
                "export ROS_DOMAIN_ID=0",
                'export LD_LIBRARY_PATH="${G1_WATCHDOG_CYCLONEDDS_LIB}:/usr/local/lib:${LD_LIBRARY_PATH:-}"',
                f"exec {agent_command}",
            ]
        )


def _ssh_options(config: PC2SafetyConfig) -> list[str]:
    return [
        "-i",
        str(config.ssh_identity),
        "-o",
        "BatchMode=yes",
        "-o",
        f"ConnectTimeout={config.connect_timeout_s:g}",
        "-o",
        "ServerAliveInterval=1",
        "-o",
        "ServerAliveCountMax=2",
    ]
