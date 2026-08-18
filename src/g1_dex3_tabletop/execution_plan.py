"""Adapters from immutable planner trajectories to the proven pose executor."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from g1_aprilcube_calibration.joint_map import arm_indices
from g1_aprilcube_calibration.models import utc_now_iso
from g1_aprilcube_calibration.pose_schema import PoseAuditEvent, PoseRecord, PoseSet
from g1_dex3_tabletop.planning.contracts import PlannedTrajectory


def pose_set_from_trajectories(
    *,
    arm: str,
    trajectories: Sequence[PlannedTrajectory],
    reference_full_q: Sequence[float],
    robot_model: str,
    urdf_sha256: str,
    source: str,
    initial_pose_id: str | None = None,
    initial_command_q_rad: Sequence[float] | None = None,
) -> PoseSet:
    """Create endpoint metadata without changing any planned command."""

    full_reference = np.asarray(reference_full_q, dtype=np.float64).reshape(-1)
    if full_reference.shape != (29,):
        raise ValueError("trajectory pose-set reference must contain 29 joints")
    indices = np.asarray(arm_indices(arm), dtype=np.int64)
    timestamp = utc_now_iso()
    records: list[PoseRecord] = []
    seen: set[str] = set()
    if (initial_pose_id is None) != (initial_command_q_rad is None):
        raise ValueError("initial pose ID and command must be provided together")
    if trajectories and trajectories[0].from_pose_id != "__handoff__":
        if initial_pose_id is None:
            raise ValueError(
                "a plan starting away from handoff must include its current boundary"
            )
        if initial_pose_id != trajectories[0].from_pose_id:
            raise ValueError(
                "initial trajectory boundary must match the first trajectory source"
            )
    if initial_pose_id is not None:
        initial = np.asarray(initial_command_q_rad, dtype=np.float64).reshape(-1)
        if initial.shape != (7,) or not np.all(np.isfinite(initial)):
            raise ValueError("initial trajectory command must contain seven finite joints")
        if initial_pose_id == "__handoff__":
            raise ValueError("initial trajectory boundary cannot be handoff")
        full_q = full_reference.copy()
        full_q[indices] = initial
        records.append(
            PoseRecord(
                id=initial_pose_id,
                group="tabletop_execution",
                measured_calibration_q=tuple(initial),
                measured_full_q=tuple(full_q),
                calibration_q_spread=(0.0,) * 7,
                recorded_at_utc=timestamp,
                recorded_monotonic_s=0.0,
                source=source,
            )
        )
        seen.add(initial_pose_id)
    for index, trajectory in enumerate(trajectories):
        if trajectory.to_pose_id == "__handoff__" or trajectory.to_pose_id in seen:
            continue
        seen.add(trajectory.to_pose_id)
        full_q = full_reference.copy()
        endpoint = tuple(trajectory.command_q_rad[-1])
        full_q[indices] = np.asarray(endpoint)
        records.append(
            PoseRecord(
                id=trajectory.to_pose_id,
                group="tabletop_execution",
                measured_calibration_q=endpoint,
                measured_full_q=tuple(full_q),
                calibration_q_spread=(0.0,) * 7,
                recorded_at_utc=timestamp,
                recorded_monotonic_s=float(index),
                source=source,
            )
        )
    audit = tuple(PoseAuditEvent("add", item.id, timestamp) for item in records)
    return PoseSet(
        robot_model=robot_model,
        mode_machine=5,
        urdf_sha256=urdf_sha256,
        calibration_arm=arm,
        poses=tuple(records),
        audit_log=audit,
    )
