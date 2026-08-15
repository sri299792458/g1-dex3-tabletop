"""Parent-side client for the isolated, persistent CuRobo worker."""

from __future__ import annotations

import json
import queue
import subprocess
import threading
import time
from pathlib import Path

from typing_extensions import Self

EVENT_PREFIX = "G1_PLANNER_EVENT "


class PlannerRequestRejected(RuntimeError):
    """A planner request failed while the robot controller remained healthy."""


class PersistentTabletopPlanner:
    """Keep one Python/CUDA worker alive for a complete hardware run."""

    def __init__(self, *, executable: Path, log_path: Path) -> None:
        self.executable = executable
        self.log_path = log_path
        self._process: subprocess.Popen[str] | None = None
        self._log = None
        self._lines: queue.Queue[str] = queue.Queue()
        self._reader: threading.Thread | None = None
        self._next_request_id = 1

    @property
    def is_alive(self) -> bool:
        return self._process is not None and self._process.poll() is None

    def start(self, *, timeout_s: float = 60.0) -> None:
        if self._process is not None:
            raise RuntimeError("persistent planner was already started")
        if not self.executable.is_file():
            raise FileNotFoundError(
                "planner environment missing; run ./tools/setup_planner_env.sh"
            )
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._log = self.log_path.open("w", encoding="utf-8", buffering=1)
        self._process = subprocess.Popen(
            [str(self.executable), "serve-tabletop"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        self._reader = threading.Thread(
            target=self._read_output,
            name="persistent-curobo-output",
            daemon=True,
        )
        self._reader.start()
        event = self._wait_for_event(
            expected_type="ready",
            request_id=None,
            timeout_s=timeout_s,
            control_check=None,
        )
        if not event.get("cuda_available", False):
            raise RuntimeError("persistent planner started without a CUDA device")

    def request(
        self,
        command: str,
        *,
        request_path: Path,
        output_path: Path,
        control_check=None,
        timeout_s: float = 180.0,
    ) -> dict:
        process = self._require_alive()
        if output_path.exists():
            raise FileExistsError(f"planner output already exists: {output_path}")
        request_id = self._next_request_id
        self._next_request_id += 1
        message = {
            "id": request_id,
            "command": command,
            "request": str(request_path.resolve()),
            "output": str(output_path.resolve()),
        }
        assert process.stdin is not None
        process.stdin.write(json.dumps(message, separators=(",", ":")) + "\n")
        process.stdin.flush()
        event = self._wait_for_event(
            expected_type="result",
            request_id=request_id,
            timeout_s=timeout_s,
            control_check=control_check,
        )
        if not event.get("ok", False):
            raise PlannerRequestRejected(
                f"CuRobo {command} rejected the request: "
                f"{event.get('error_type', 'RuntimeError')}: "
                f"{event.get('error', 'no diagnostic')}; full planner log: {self.log_path}"
            )
        return event

    def close(self) -> None:
        process = self._process
        if process is None:
            return
        if process.poll() is None:
            try:
                assert process.stdin is not None
                process.stdin.write('{"command":"shutdown"}\n')
                process.stdin.flush()
                process.wait(timeout=5.0)
            except (BrokenPipeError, subprocess.TimeoutExpired):
                process.terminate()
                try:
                    process.wait(timeout=2.0)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
        if process.stdin is not None:
            process.stdin.close()
        if self._reader is not None:
            self._reader.join(timeout=2.0)
        if process.stdout is not None:
            process.stdout.close()
        if self._log is not None:
            self._log.close()
        self._process = None
        self._log = None
        self._reader = None

    def _read_output(self) -> None:
        process = self._process
        if process is None or process.stdout is None:
            return
        for line in process.stdout:
            if self._log is not None:
                self._log.write(line)
            self._lines.put(line)

    def _require_alive(self) -> subprocess.Popen[str]:
        if self._process is None or self._process.poll() is not None:
            status = None if self._process is None else self._process.returncode
            raise RuntimeError(f"persistent planner is not running (status={status})")
        return self._process

    def _wait_for_event(
        self,
        *,
        expected_type: str,
        request_id: int | None,
        timeout_s: float,
        control_check,
    ) -> dict:
        process = self._require_alive()
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            if control_check is not None:
                control_check()
            if process.poll() is not None:
                raise RuntimeError(
                    f"persistent planner exited unexpectedly with status {process.returncode}; "
                    f"full planner log: {self.log_path}"
                )
            remaining = max(deadline - time.monotonic(), 0.0)
            try:
                line = self._lines.get(timeout=min(remaining, 0.1))
            except queue.Empty:
                continue
            stripped = line.rstrip()
            if stripped.startswith(EVENT_PREFIX):
                try:
                    event = json.loads(stripped[len(EVENT_PREFIX) :])
                except json.JSONDecodeError:
                    print(stripped, flush=True)
                    continue
                if event.get("type") == "progress":
                    print(str(event.get("message", "")), flush=True)
                    continue
                if event.get("type") == expected_type and (
                    request_id is None or event.get("id") == request_id
                ):
                    return event
                continue
            print(line, end="", flush=True)
        raise RuntimeError(
            f"timed out after {timeout_s:.1f}s waiting for persistent planner "
            f"{expected_type}; full planner log: {self.log_path}"
        )

    def __enter__(self) -> Self:
        self.start()
        return self

    def __exit__(self, _error_type, _error, _traceback) -> None:
        self.close()
