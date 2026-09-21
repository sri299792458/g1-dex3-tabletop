from __future__ import annotations

import pytest

from g1_dex3_tabletop.execution_plan import pose_set_from_trajectories
from g1_dex3_tabletop.planning.contracts import PlannedTrajectory


def _trajectory() -> PlannedTrajectory:
    return PlannedTrajectory(
        from_pose_id="move_to_pregrasp",
        to_pose_id="estimated_pregrasp",
        sample_time_s=(0.0, 1.0),
        command_q_rad=((0.2,) * 7, (0.3,) * 7),
        model_q_rad=((0.2,) * 7, (0.3,) * 7),
        planning_time_s=0.1,
    )


def test_replacement_pose_set_contains_exact_reached_boundary() -> None:
    pose_set = pose_set_from_trajectories(
        arm="right",
        trajectories=(_trajectory(),),
        reference_full_q=(0.0,) * 29,
        robot_model="g1",
        urdf_sha256="a" * 64,
        source="test",
        initial_pose_id="move_to_pregrasp",
        initial_command_q_rad=(0.2,) * 7,
    )

    assert tuple(item.id for item in pose_set.poses) == (
        "move_to_pregrasp",
        "estimated_pregrasp",
    )
    assert pose_set.poses[0].command_calibration_q == (0.2,) * 7


def test_replacement_pose_set_cannot_omit_reached_boundary() -> None:
    with pytest.raises(ValueError, match="must include its current boundary"):
        pose_set_from_trajectories(
            arm="right",
            trajectories=(_trajectory(),),
            reference_full_q=(0.0,) * 29,
            robot_model="g1",
            urdf_sha256="a" * 64,
            source="test",
        )


def test_replacement_pose_set_boundary_must_match_first_source() -> None:
    with pytest.raises(ValueError, match="must match the first trajectory source"):
        pose_set_from_trajectories(
            arm="right",
            trajectories=(_trajectory(),),
            reference_full_q=(0.0,) * 29,
            robot_model="g1",
            urdf_sha256="a" * 64,
            source="test",
            initial_pose_id="clearance",
            initial_command_q_rad=(0.2,) * 7,
        )
