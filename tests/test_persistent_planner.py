from __future__ import annotations

import json
from pathlib import Path

import pytest

from g1_dex3_tabletop.persistent_planner import (
    EVENT_PREFIX,
    PersistentTabletopPlanner,
    PlannerRequestRejected,
)


def _fake_worker(path: Path) -> None:
    path.write_text(
        """#!/usr/bin/env python3
import json
import sys

prefix = "G1_PLANNER_EVENT "
print(prefix + json.dumps({"type": "ready", "cuda_available": True, "device": "fake"}), flush=True)
for line in sys.stdin:
    message = json.loads(line)
    if message.get("command") == "shutdown":
        print(prefix + json.dumps({"type": "stopped"}), flush=True)
        break
    print(prefix + json.dumps({"type": "progress", "id": message["id"], "message": "working"}), flush=True)
    ok = message["command"] != "reject"
    result = {"type": "result", "id": message["id"], "ok": ok}
    if "payload" in message:
        result["payload"] = message["payload"]
    if not ok:
        result.update({"error_type": "RuntimeError", "error": "no route"})
    print(prefix + json.dumps(result), flush=True)
""",
        encoding="utf-8",
    )
    path.chmod(0o755)


def test_persistent_planner_reuses_one_process_and_reports_rejection(
    tmp_path: Path, capsys
) -> None:
    worker = tmp_path / "worker"
    _fake_worker(worker)
    request = tmp_path / "request.json"
    request.write_text("{}", encoding="utf-8")
    planner = PersistentTabletopPlanner(
        executable=worker,
        log_path=tmp_path / "planner.log",
    )

    planner.start()
    process = planner._process
    first = planner.request(
        "plan-tabletop-lifecycle",
        request_path=request,
        output_path=tmp_path / "plan.json",
    )
    assert first["ok"] is True
    assert planner._process is process
    payload = planner.request_payload("step", payload={"generation": 7})
    assert payload["payload"] == {"generation": 7}
    assert planner._process is process
    with pytest.raises(PlannerRequestRejected, match="no route"):
        planner.request(
            "reject",
            request_path=request,
            output_path=tmp_path / "rejected.json",
        )
    assert planner._process is process
    planner.close()

    assert "working" in capsys.readouterr().out
    log_lines = (tmp_path / "planner.log").read_text(encoding="utf-8").splitlines()
    assert all(line.startswith(EVENT_PREFIX) for line in log_lines)
    assert sum(json.loads(line[len(EVENT_PREFIX) :])["type"] == "ready" for line in log_lines) == 1


def test_persistent_planner_can_launch_before_waiting_for_cuda_ready(tmp_path: Path) -> None:
    worker = tmp_path / "worker"
    _fake_worker(worker)
    planner = PersistentTabletopPlanner(
        executable=worker,
        log_path=tmp_path / "planner.log",
    )

    planner.launch()
    assert planner.is_alive
    with pytest.raises(RuntimeError, match="not ready"):
        planner.request_payload("step", payload={"generation": 1})

    planner.wait_until_ready()
    assert planner.request_payload("step", payload={"generation": 2})["payload"] == {
        "generation": 2
    }
    planner.close()


def test_persistent_planner_request_can_run_before_parent_waits(tmp_path: Path) -> None:
    worker = tmp_path / "worker"
    _fake_worker(worker)
    planner = PersistentTabletopPlanner(
        executable=worker,
        log_path=tmp_path / "planner.log",
    )
    planner.launch()

    pending = planner.begin_payload_request("step", payload={"generation": 3})
    with pytest.raises(RuntimeError, match="still pending"):
        planner.request_payload("step", payload={"generation": 4})
    event = planner.finish_request(pending)

    assert event["payload"] == {"generation": 3}
    assert planner._ready
    planner.close()
