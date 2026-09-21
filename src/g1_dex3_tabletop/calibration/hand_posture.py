"""Operator-prepared full-close readiness; no finger motion commands."""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import numpy as np

from g1_aprilcube_calibration.transports.unitree_dex3 import DEX3_MOTOR_JOINT_SUFFIXES

FULL_CLOSE_REFERENCE_PATH = (
    Path(__file__).resolve().parents[3] / "config/calibration/dex3_full_close_reference.json"
)
HELD_FINGER_POLICY = "operator_preclosed_measured_hold"


def full_close_reference() -> dict:
    raw = FULL_CLOSE_REFERENCE_PATH.read_bytes()
    reference = json.loads(raw)
    if reference["schema_version"] != 1 or tuple(reference["motor_order"]) != tuple(
        DEX3_MOTOR_JOINT_SUFFIXES["right"]
    ):
        raise ValueError("calibration full-close reference schema or motor order changed")
    for side in ("left", "right"):
        q = np.asarray(reference["reference_q_rad"][side], dtype=np.float64)
        if q.shape != (7,) or not np.all(np.isfinite(q)):
            raise ValueError(f"invalid {side} calibration full-close reference")
    return {**reference, "file_sha256": hashlib.sha256(raw).hexdigest()}


def check_full_close(*, left_q_rad, right_q_rad, tolerance_rad: float, reference=None) -> dict:
    """Check proximity to the recorded pose, not physical end-stop contact."""

    if not np.isfinite(tolerance_rad) or tolerance_rad <= 0:
        raise ValueError("full-close readiness tolerance must be positive and finite")
    reference = full_close_reference() if reference is None else reference
    evidence = {}
    for side, values in (("left", left_q_rad), ("right", right_q_rad)):
        measured = np.asarray(values, dtype=np.float64)
        if measured.shape != (7,) or not np.all(np.isfinite(measured)):
            raise ValueError(f"{side} full-close readiness requires seven finite measurements")
        expected = np.asarray(reference["reference_q_rad"][side], dtype=np.float64)
        errors = np.abs(measured - expected)
        motor = int(np.argmax(errors))
        if errors[motor] > tolerance_rad:
            joint = f"{side}_hand_{DEX3_MOTOR_JOINT_SUFFIXES[side][motor]}_joint"
            raise ValueError(
                "Close both Dex3 hands fully before starting calibration, then rerun "
                f"preflight: {joint} differs from the full-close reference by "
                f"{errors[motor]:.4f}rad (measured={measured[motor]:.4f}, "
                f"reference={expected[motor]:.4f}, limit={tolerance_rad:.4f}). "
                "Calibration holds the measured posture; it does not close the fingers."
            )
        evidence[side] = {
            "measured_q_rad": measured.tolist(),
            "maximum_reference_error_rad": float(errors[motor]),
        }
    return {"policy": HELD_FINGER_POLICY, "reference": reference, "hands": evidence}


def wait_for_full_close(observer, config, *, clock=time.monotonic, sleep=time.sleep):
    """Require fresh full-close feedback and a continuous stationary window."""

    reference = full_close_reference()
    deadline = clock() + config.posture_timeout_s
    started = None
    low = high = None
    while clock() < deadline:
        pair = observer.observe()  # The DDS observer enforces feedback freshness.
        evidence = check_full_close(
            left_q_rad=pair.left.position,
            right_q_rad=pair.right.position,
            tolerance_rad=config.posture_position_tolerance_rad,
            reference=reference,
        )
        measured = np.r_[pair.left.position, pair.right.position]
        now = clock()
        if started is None:
            started, low, high = now, measured.copy(), measured.copy()
        else:
            low, high = np.minimum(low, measured), np.maximum(high, measured)
            if float(np.max(high - low)) > config.posture_position_spread_rad:
                started, low, high = now, measured.copy(), measured.copy()
            elif now - started >= config.posture_settle_dwell_s:
                return pair, {
                    **evidence,
                    "stationary_duration_s": now - started,
                    "maximum_position_spread_rad": float(np.max(high - low)),
                }
        sleep(min(0.01, 1.0 / config.command_rate_hz))
    raise RuntimeError(
        "Full-close fingers did not remain stationary before calibration startup; "
        "let both hands settle and rerun preflight."
    )
