"""Bind the reversible live adapter to the commands held after acquisition."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from g1_dex3_tabletop.planning.contracts import (
    BilateralCalibrationAdapterPlan,
    RobotSnapshot,
)


@dataclass(frozen=True, slots=True)
class OwnedAdapterReceipt:
    """Small verified binding; the full request stays in the planner process."""

    snapshot: RobotSnapshot
    content_sha256: str


def plan_owned_adapter(
    *,
    control,
    planner,
    preflight_request_path: Path,
    preflight_request_sha256: str,
    preflight_plan_path: Path,
    preflight_plan_sha256: str,
    anchor_q14,
    work_directory: Path,
):
    """Reuse the stack's exact-command planning and installation boundary."""

    reference_state, snapshot = control.planning_boundary()
    request_path = work_directory / "owned_adapter_request.json"
    plan_path = work_directory / "owned_adapter_plan.json"
    event = planner.request_payload(
        "refresh-bilateral-calibration-adapter",
        payload={
            "preflight_request": str(preflight_request_path.resolve()),
            "preflight_request_sha256": preflight_request_sha256,
            "preflight_plan": str(preflight_plan_path.resolve()),
            "preflight_plan_sha256": preflight_plan_sha256,
            "snapshot": snapshot.to_dict(),
            "request_output": str(request_path.resolve()),
            "plan_output": str(plan_path.resolve()),
        },
        control_check=control.check,
        timeout_s=600.0,
    )
    control.check()
    receipt = event["payload"]
    if (
        receipt["preflight_request_sha256"] != preflight_request_sha256
        or receipt["preflight_plan_sha256"] != preflight_plan_sha256
        or RobotSnapshot.from_dict(receipt["snapshot"]) != snapshot
    ):
        raise ValueError("owned adapter receipt differs from the requested live boundary")
    adapter = BilateralCalibrationAdapterPlan.from_json(plan_path)
    if (
        adapter.request_sha256 != receipt["request_sha256"]
        or adapter.content_sha256 != receipt["plan_sha256"]
    ):
        raise ValueError("owned bilateral adapter plan/request binding changed")
    if not np.allclose(adapter.anchor_q14_rad, anchor_q14, atol=1e-9, rtol=0):
        raise ValueError("owned adapter endpoint differs from the reusable anchor")
    return OwnedAdapterReceipt(snapshot, receipt["request_sha256"]), adapter, reference_state
