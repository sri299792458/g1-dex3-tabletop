"""Per-run stationary handoff construction without command ownership."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from g1_aprilcube_calibration.joint_map import arm_indices, opposite_arm
from g1_aprilcube_calibration.models import RobotStateSample
from g1_aprilcube_calibration.pose_schema import PoseSet
from g1_aprilcube_calibration.readiness import (
    ReadinessReport,
    RecordingGateConfig,
    evaluate_recording_window,
)


@dataclass(frozen=True, slots=True)
class ActivationHandoff:
    """A stationary measured run origin derived before command ownership."""

    source_pose_set_sha256: str
    pose_set: PoseSet
    handoff_q: tuple[float, ...]
    hold_q: tuple[float, ...]
    reference_state: RobotStateSample
    readiness: ReadinessReport


def trailing_stationary_window(
    samples: Sequence[RobotStateSample],
    *,
    duration_s: float,
) -> tuple[RobotStateSample, ...]:
    """Return the latest state window, including the sample before its boundary."""

    if duration_s <= 0:
        raise ValueError("stationary-window duration must be positive")
    ordered = tuple(samples)
    if not ordered:
        return ()
    timestamps = np.asarray([sample.receipt_monotonic_s for sample in ordered], dtype=np.float64)
    if np.any(np.diff(timestamps) <= 0):
        return ordered
    cutoff = timestamps[-1] - duration_s
    before = np.flatnonzero(timestamps <= cutoff)
    if not len(before):
        return ordered
    return ordered[int(before[-1]) :]


def build_activation_handoff(
    pose_set: PoseSet,
    samples: Sequence[RobotStateSample],
    *,
    now_monotonic_s: float,
    config: RecordingGateConfig,
) -> ActivationHandoff:
    """Require a stationary trailing window and derive a median measured handoff."""

    if pose_set.calibration_arm != config.calibration_arm:
        raise ValueError("activation configuration arm does not match the pose set")
    window = trailing_stationary_window(
        samples,
        duration_s=config.stationary_duration_s,
    )
    readiness = evaluate_recording_window(
        window,
        now_monotonic_s=now_monotonic_s,
        config=config,
    )
    if not readiness.ready:
        raise ValueError(
            "activation handoff is not stationary: " + "; ".join(readiness.hard_failures)
        )
    positions = np.vstack([sample.position for sample in window])
    velocities = np.vstack([sample.velocity for sample in window])
    estimated_torques = np.vstack([sample.estimated_torque for sample in window])
    measured_position = np.median(positions, axis=0)
    measured_velocity = np.median(velocities, axis=0)
    measured_estimated_torque = np.median(estimated_torques, axis=0)
    latest = window[-1]
    reference_state = RobotStateSample(
        receipt_monotonic_s=latest.receipt_monotonic_s,
        receipt_utc=latest.receipt_utc,
        mode_machine=latest.mode_machine,
        position=measured_position,
        velocity=measured_velocity,
        estimated_torque=measured_estimated_torque,
        source_sequence=latest.source_sequence,
    )
    calibration_indices = np.asarray(arm_indices(pose_set.calibration_arm))
    hold_indices = np.asarray(arm_indices(opposite_arm(pose_set.calibration_arm)))
    return ActivationHandoff(
        source_pose_set_sha256=pose_set.content_sha256,
        pose_set=pose_set,
        handoff_q=tuple(measured_position[calibration_indices]),
        hold_q=tuple(measured_position[hold_indices]),
        reference_state=reference_state,
        readiness=readiness,
    )
