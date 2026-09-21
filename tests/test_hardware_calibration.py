from __future__ import annotations

import numpy as np

from g1_dex3_tabletop.hardware_calibration import (
    _stage_trajectory,
    clearance_snapshot,
)
from g1_dex3_tabletop.planning.contracts import (
    Dex3PreparationPlan,
    PlannedTrajectory,
    RobotSnapshot,
)


def _trajectory(source: str, target: str, start: float, end: float) -> PlannedTrajectory:
    return PlannedTrajectory(
        from_pose_id=source,
        to_pose_id=target,
        sample_time_s=(0.0, 1.0),
        command_q_rad=((start,) * 7, (end,) * 7),
        model_q_rad=((start,) * 7, (end,) * 7),
        planning_time_s=0.1,
    )


def _preparation() -> Dex3PreparationPlan:
    right = _trajectory("__handoff__", "right_shoulder_clearance", 0.0, 0.2)
    left = _trajectory("right_shoulder_clearance", "dual_shoulder_clearance", 0.0, -0.3)
    return Dex3PreparationPlan(
        request_sha256="a" * 64,
        outward_offset_rad=0.08,
        right_outbound=right,
        left_outbound=left,
        left_return=_trajectory("dual_shoulder_clearance", "right_shoulder_clearance", -0.3, 0.0),
        right_return=_trajectory("right_shoulder_clearance", "__handoff__", 0.2, 0.0),
        dual_clearance_q14_rad=(-0.3,) * 7 + (0.2,) * 7,
        finger_sweep_sample_count=20,
        planner_provenance={"backend": "test"},
    )


def test_clearance_snapshot_changes_only_both_arms_and_fingers() -> None:
    q29 = np.arange(29, dtype=np.float64) / 100.0
    source = RobotSnapshot(tuple(q29), (0.1,) * 7, (-0.1,) * 7)
    result = clearance_snapshot(
        source,
        _preparation(),
        left_fingers=(-1.0,) * 7,
        right_fingers=(1.0,) * 7,
    )
    expected = q29.copy()
    expected[15:22] = -0.3
    expected[22:29] = 0.2
    np.testing.assert_allclose(result.measured_q29_rad, expected)
    assert result.left_dex3_q_rad == (-1.0,) * 7
    assert result.right_dex3_q_rad == (1.0,) * 7


def test_stage_rebase_changes_ids_but_not_any_motion_sample() -> None:
    source = _preparation().left_return
    result = _stage_trajectory(source, target_id="left_shoulder_restored")
    assert result.from_pose_id == "__handoff__"
    assert result.to_pose_id == "left_shoulder_restored"
    assert result.sample_time_s == source.sample_time_s
    assert result.command_q_rad == source.command_q_rad
    assert result.model_q_rad == source.model_q_rad
