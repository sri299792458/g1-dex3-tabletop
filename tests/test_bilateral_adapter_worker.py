"""Keep full-core serialization on the existing worker side of the boundary."""

import os
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
from test_planning_contracts import bilateral_adapter_artifacts

from g1_dex3_tabletop.calibration.adapter import plan_owned_adapter
from g1_dex3_tabletop.persistent_planner import PersistentTabletopPlanner
from g1_dex3_tabletop.planning.bilateral_adapter import refresh_owned_bilateral_adapter
from g1_dex3_tabletop.planning.contracts import BilateralCalibrationAdapterRequest, RobotSnapshot


def setup_request(tmp_path):
    request, adapter = bilateral_adapter_artifacts()
    path = tmp_path / "preflight.json"
    request.write_json(path)
    plan_path = tmp_path / "preflight_plan.json"
    adapter.write_json(plan_path)
    snapshot = RobotSnapshot((0.001,) * 29, (0.2,) * 7, (-0.2,) * 7)
    payload = {
        "preflight_request": str(path),
        "preflight_request_sha256": request.content_sha256,
        "preflight_plan": str(plan_path),
        "preflight_plan_sha256": adapter.content_sha256,
        "snapshot": snapshot.to_dict(),
        "request_output": str(tmp_path / "owned_request.json"),
        "plan_output": str(tmp_path / "owned_plan.json"),
    }
    return request, adapter, payload


@pytest.mark.parametrize(
    "failure",
    [
        None,
        "template",
        "plan_template",
        "result_binding",
        "changed_preparation",
        "collision",
        "existing_output",
    ],
)
def test_worker_refresh_checks_bindings_and_preserves_core(tmp_path, failure):
    preflight, adapter, payload = setup_request(tmp_path)
    if failure == "template":
        payload["preflight_request_sha256"] = "c" * 64
    if failure == "plan_template":
        payload["preflight_plan_sha256"] = "c" * 64
    if failure == "existing_output":
        Path(payload["request_output"]).write_text("keep this")
    calls = []

    def plan(request, *, progress, preparation):
        calls.append(request)
        assert preparation == adapter.preparation
        assert request.core_transitions == preflight.core_transitions
        assert request.execution_plan_sha256 == preflight.execution_plan_sha256
        assert request.snapshot.to_dict() == payload["snapshot"]
        if failure == "collision":
            raise RuntimeError("live collision rejected")
        return replace(
            adapter,
            request_sha256="d" * 64 if failure == "result_binding" else request.content_sha256,
            preparation=replace(adapter.preparation, planner_provenance={"changed": True})
            if failure == "changed_preparation"
            else adapter.preparation,
        )

    if failure:
        with pytest.raises((ValueError, RuntimeError, FileExistsError)):
            refresh_owned_bilateral_adapter(payload, plan_adapter=plan)
        assert not Path(payload["plan_output"]).exists()
        if failure in {"template", "plan_template", "existing_output"}:
            assert not calls
        else:
            assert Path(payload["request_output"]).exists()
        if failure == "existing_output":
            assert Path(payload["request_output"]).read_text() == "keep this"
    else:
        receipt = refresh_owned_bilateral_adapter(payload, plan_adapter=plan)
        loaded = BilateralCalibrationAdapterRequest.from_json(payload["request_output"])
        assert receipt["request_sha256"] == loaded.content_sha256
        assert receipt["snapshot"] == payload["snapshot"]
        assert receipt["preflight_request_sha256"] == preflight.content_sha256


def test_live_client_passes_only_small_snapshot_and_paths_to_separate_worker(
    tmp_path, monkeypatch
):
    preflight, adapter, payload = setup_request(tmp_path)
    root = Path(__file__).resolve().parents[1]
    worker = tmp_path / "worker"
    worker.write_text(
        f"#!{sys.executable}\n"
        + f"sys_path = {str(root / 'tests')!r}\n"
        + """
import sys
sys.path.insert(0, sys_path)
import json
import os
from dataclasses import replace
from test_planning_contracts import bilateral_adapter_artifacts
from g1_dex3_tabletop.planning.bilateral_adapter import refresh_owned_bilateral_adapter
prefix = "G1_PLANNER_EVENT "
print(prefix + json.dumps({"type": "ready", "cuda_available": True}), flush=True)
for line in sys.stdin:
    message = json.loads(line)
    if message["command"] == "shutdown":
        print(prefix + json.dumps({"type": "stopped"}), flush=True)
        break
    assert message["command"] == "refresh-bilateral-calibration-adapter"
    assert len(line) < 4096
    def plan(request, **kwargs):
        _, adapter = bilateral_adapter_artifacts()
        return replace(adapter, request_sha256=request.content_sha256)
    receipt = refresh_owned_bilateral_adapter(message["payload"], plan_adapter=plan)
    receipt["worker_pid"] = os.getpid()
    print(prefix + json.dumps({"type": "result", "id": message["id"], "ok": True, "payload": receipt}), flush=True)
"""
    )
    worker.chmod(0o755)
    planner = PersistentTabletopPlanner(executable=worker, log_path=tmp_path / "worker.log")
    planner.start()
    snapshot = RobotSnapshot.from_dict(payload["snapshot"])
    checks = []
    control = SimpleNamespace(
        planning_boundary=lambda: ("reference", snapshot), check=lambda: checks.append(True)
    )
    expected_hash = preflight.content_sha256

    # Any accidental full request read/hash in the live parent must fail.
    def forbidden(*args, **kwargs):
        raise AssertionError("full calibration request accessed in control process")

    monkeypatch.setattr(BilateralCalibrationAdapterRequest, "from_json", forbidden)
    monkeypatch.setattr(BilateralCalibrationAdapterRequest, "content_sha256", property(forbidden))
    try:
        receipt, result, reference = plan_owned_adapter(
            control=control,
            planner=planner,
            preflight_request_path=Path(payload["preflight_request"]),
            preflight_request_sha256=expected_hash,
            preflight_plan_path=Path(payload["preflight_plan"]),
            preflight_plan_sha256=payload["preflight_plan_sha256"],
            anchor_q14=adapter.anchor_q14_rad,
            work_directory=tmp_path,
        )
        assert receipt.snapshot == snapshot
        assert result.request_sha256 == receipt.content_sha256
        assert reference == "reference"
        assert planner._process.pid != os.getpid()
        assert checks
    finally:
        planner.close()
