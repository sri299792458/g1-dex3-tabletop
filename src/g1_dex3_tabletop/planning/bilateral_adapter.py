"""Refresh large calibration artifacts inside the retained planner process."""

from dataclasses import replace
from pathlib import Path

from .contracts import (
    BilateralCalibrationAdapterPlan,
    BilateralCalibrationAdapterRequest,
    RobotSnapshot,
)


def refresh_owned_bilateral_adapter(payload, *, plan_adapter, progress=None):
    """Only paths, a template hash, and the small live snapshot cross IPC."""

    request_path = Path(payload["request_output"])
    plan_path = Path(payload["plan_output"])
    for path in (request_path, plan_path):
        if path.exists():
            raise FileExistsError(f"owned adapter output already exists: {path}")
    preflight = BilateralCalibrationAdapterRequest.from_json(payload["preflight_request"])
    preflight_hash = preflight.content_sha256
    if preflight_hash != payload["preflight_request_sha256"]:
        raise ValueError("bilateral preflight template changed before owned planning")
    prepared = BilateralCalibrationAdapterPlan.from_json(payload["preflight_plan"])
    if (
        prepared.request_sha256 != preflight_hash
        or prepared.content_sha256 != payload["preflight_plan_sha256"]
    ):
        raise ValueError("bilateral preflight shoulder plan changed before loaded planning")
    request = replace(preflight, snapshot=RobotSnapshot.from_dict(payload["snapshot"]))
    request_hash = request.content_sha256
    request.write_json(request_path)
    adapter = plan_adapter(request, progress=progress, preparation=prepared.preparation)
    if adapter.request_sha256 != request_hash:
        raise ValueError("owned adapter planner returned a different request binding")
    if adapter.preparation.content_sha256 != prepared.preparation.content_sha256:
        raise ValueError("loaded adapter changed the commissioned shoulder plan")
    adapter.write_json(plan_path)
    return {
        "preflight_request_sha256": preflight_hash,
        "preflight_plan_sha256": prepared.content_sha256,
        "request_sha256": request_hash,
        "snapshot": request.snapshot.to_dict(),
        "plan_sha256": adapter.content_sha256,
    }
