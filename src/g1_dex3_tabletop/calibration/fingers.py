"""Measured-state finger validation through the existing controller and planner."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from g1_dex3_tabletop.hardware_tabletop import _command_fingers, _wait_for_hands
from g1_dex3_tabletop.planning.contracts import (
    Dex3PreparationRequest,
    RobotSnapshot,
    atomic_write_json,
)


def measured_snapshot(state, hands) -> RobotSnapshot:
    return RobotSnapshot(
        tuple(state.position),
        tuple(hands.left.position),
        tuple(hands.right.position),
    )


def command_validated_finger_posture(
    *,
    synchronized,
    driver,
    controller,
    watchdog,
    planner,
    joint_position_offsets_rad,
    left_target_q_rad,
    right_target_q_rad,
    left_acceptance_q_rad,
    right_acceptance_q_rad,
    body_tolerance_rad: float,
    hand_tolerance_rad: float,
    artifact_directory: Path,
    phase: str,
    label: str,
    recorded_restoration_start: RobotSnapshot | None = None,
):
    """Recheck measured loaded geometry while the commissioned driver holds.

    Uses the same finger validator as preflight through the existing persistent
    worker/control-health polling client. Only a successful bound result may
    reach the existing Dex3 posture controller.
    """

    driver.check()
    snapshot = measured_snapshot(
        synchronized.observe_state(), _wait_for_hands(controller.observer)
    )
    request = Dex3PreparationRequest(
        snapshot=snapshot,
        joint_position_offsets_rad=joint_position_offsets_rad,
        left_target_q_rad=tuple(left_target_q_rad),
        right_target_q_rad=tuple(right_target_q_rad),
    )
    if recorded_restoration_start is not None:
        request = Dex3PreparationRequest.for_measured_restoration(
            snapshot=snapshot,
            recorded_start=recorded_restoration_start,
            joint_position_offsets_rad=joint_position_offsets_rad,
            left_target_q_rad=left_target_q_rad,
            right_target_q_rad=right_target_q_rad,
        )
    request.write_json(artifact_directory / f"{phase}_request.json")
    atomic_write_json(
        artifact_directory / f"{phase}_motion.json",
        {
            "validation_request_sha256": request.content_sha256,
            "validation_direction": (
                "reverse_measured_restoration"
                if recorded_restoration_start is not None
                else "forward"
            ),
            "measured_start": snapshot.to_dict(),
            "left_target_q_rad": list(left_target_q_rad),
            "right_target_q_rad": list(right_target_q_rad),
        },
    )
    event = planner.request_payload(
        "validate-dex3-finger-sweep",
        payload=request.to_dict(),
        control_check=driver.check,
        timeout_s=180.0,
    )
    result = event["payload"]
    atomic_write_json(artifact_directory / f"{phase}_result.json", result)
    if (
        result.get("request_sha256") != request.content_sha256
        or result.get("passed") is not True
        or result.get("operation") != "validate_dex3_finger_sweep"
        or not np.isfinite(float(result.get("minimum_clearance_m", float("nan"))))
        or float(result["minimum_clearance_m"]) < 0.005 - 1e-6
        or float(result.get("required_clearance_m", 0.0)) != 0.005
    ):
        raise RuntimeError("loaded Dex3 sweep lacks a passing request-bound 5mm check")
    latest = measured_snapshot(synchronized.observe_state(), _wait_for_hands(controller.observer))
    body_error = float(
        np.max(np.abs(np.asarray(latest.measured_q29_rad) - np.asarray(snapshot.measured_q29_rad)))
    )
    hand_error = max(
        float(
            np.max(
                np.abs(
                    np.asarray(getattr(latest, f"{side}_dex3_q_rad"))
                    - np.asarray(getattr(snapshot, f"{side}_dex3_q_rad"))
                )
            )
        )
        for side in ("left", "right")
    )
    if body_error > body_tolerance_rad or hand_error > hand_tolerance_rad:
        raise RuntimeError(
            "loaded state changed during finger validation; no finger motion commanded "
            f"(body {body_error:.4f}rad, hand {hand_error:.4f}rad)"
        )
    driver.check()
    return _command_fingers(
        controller,
        driver,
        watchdog,
        left=left_target_q_rad,
        right=right_target_q_rad,
        left_acceptance=left_acceptance_q_rad,
        right_acceptance=right_acceptance_q_rad,
        label=label,
    )
