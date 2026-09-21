import json
from pathlib import Path

import pytest

from g1_aprilcube_calibration.pose_schema import HANDOFF_POSE_ID
from g1_dex3_tabletop.planning.contracts import (
    CalibrationCandidate,
    CalibrationPlanRequest,
    CalibrationPlanResult,
    Dex3PreparationPlan,
    Dex3PreparationRequest,
    PlannedCalibrationPose,
    PlannedTrajectory,
    RobotSnapshot,
)


def _identity() -> tuple[tuple[float, ...], ...]:
    return (
        (1.0, 0.0, 0.0, 0.0),
        (0.0, 1.0, 0.0, 0.0),
        (0.0, 0.0, 1.0, 0.0),
        (0.0, 0.0, 0.0, 1.0),
    )


def _snapshot() -> RobotSnapshot:
    return RobotSnapshot((0.0,) * 29, (0.0,) * 7, (0.0,) * 7)


def test_request_hash_round_trip_and_tamper_detection(tmp_path: Path) -> None:
    request = CalibrationPlanRequest(
        arm="left",
        snapshot=RobotSnapshot((0.0,) * 29, (0.0,) * 7, (0.0,) * 7),
        torso_T_camera=_identity(),
        palm_T_marker=_identity(),
        joint_position_offsets_rad={"left_shoulder_roll_joint": 0.02},
        candidates=(CalibrationCandidate("candidate_0001", _identity(), {}),),
        selection_config={"policy": "test"},
        target_count=1,
    )
    path = tmp_path / "request.json"
    request.write_json(path)

    assert CalibrationPlanRequest.from_json(path).content_sha256 == request.content_sha256
    document = json.loads(path.read_text(encoding="utf-8"))
    document["random_seed"] += 1
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        CalibrationPlanRequest.from_dict(document)


def test_plan_requires_one_frozen_edge_per_route_transition() -> None:
    pose = PlannedCalibrationPose(
        candidate_id="candidate_0001",
        camera_T_marker=_identity(),
        model_q_rad=(0.1,) * 7,
        command_q_rad=(0.1,) * 7,
        ik_position_error_m=0.0,
        ik_rotation_error_rad=0.0,
        selection_metadata={},
    )
    outbound = PlannedTrajectory(
        HANDOFF_POSE_ID,
        pose.candidate_id,
        (0.0, 1.0),
        ((0.0,) * 7, (0.1,) * 7),
        ((0.0,) * 7, (0.1,) * 7),
        0.1,
    )
    inbound = PlannedTrajectory(
        pose.candidate_id,
        HANDOFF_POSE_ID,
        (0.0, 1.0),
        ((0.1,) * 7, (0.0,) * 7),
        ((0.1,) * 7, (0.0,) * 7),
        0.0,
    )
    plan = CalibrationPlanResult(
        request_sha256="a" * 64,
        arm="right",
        active_joint_names=tuple(f"joint_{index}" for index in range(7)),
        handoff_model_q_rad=(0.0,) * 7,
        handoff_command_q_rad=(0.0,) * 7,
        poses=(pose,),
        route_pose_ids=(HANDOFF_POSE_ID, pose.candidate_id, HANDOFF_POSE_ID),
        capture_pose_ids=(pose.candidate_id,),
        trajectories=(outbound, inbound),
        selection_steps=({},),
        planner_provenance={"commit": "test"},
    )

    assert CalibrationPlanResult.from_dict(plan.to_dict()).content_sha256 == plan.content_sha256
    with pytest.raises(ValueError, match="one planned trajectory"):
        CalibrationPlanResult(
            request_sha256="a" * 64,
            arm="right",
            active_joint_names=tuple(f"joint_{index}" for index in range(7)),
            handoff_model_q_rad=(0.0,) * 7,
            handoff_command_q_rad=(0.0,) * 7,
            poses=(pose,),
            route_pose_ids=(HANDOFF_POSE_ID, pose.candidate_id, HANDOFF_POSE_ID),
            capture_pose_ids=(pose.candidate_id,),
            trajectories=(outbound,),
            selection_steps=({},),
            planner_provenance={"commit": "test"},
        )


def test_dex3_preparation_contract_round_trip(tmp_path: Path) -> None:
    request = Dex3PreparationRequest(
        snapshot=_snapshot(),
        joint_position_offsets_rad={},
        left_target_q_rad=(0.0, 0.7, 0.7, -1.0, -1.5, -1.0, -1.5),
        right_target_q_rad=(0.0, -0.7, -0.7, 1.0, 1.5, 1.0, 1.5),
    )
    right = PlannedTrajectory(
        HANDOFF_POSE_ID,
        "right_shoulder_clearance",
        (0.0, 1.0),
        ((0.0,) * 7, (0.1,) * 7),
        ((0.0,) * 7, (0.1,) * 7),
        0.2,
    )
    left = PlannedTrajectory(
        "right_shoulder_clearance",
        "dual_shoulder_clearance",
        (0.0, 1.0),
        ((0.0,) * 7, (0.1,) * 7),
        ((0.0,) * 7, (0.1,) * 7),
        0.2,
    )
    plan = Dex3PreparationPlan(
        request_sha256=request.content_sha256,
        outward_offset_rad=0.08,
        right_outbound=right,
        left_outbound=left,
        left_return=PlannedTrajectory(
            "dual_shoulder_clearance",
            "right_shoulder_clearance",
            (0.0, 1.0),
            ((0.1,) * 7, (0.0,) * 7),
            ((0.1,) * 7, (0.0,) * 7),
            0.0,
        ),
        right_return=PlannedTrajectory(
            "right_shoulder_clearance",
            HANDOFF_POSE_ID,
            (0.0, 1.0),
            ((0.1,) * 7, (0.0,) * 7),
            ((0.1,) * 7, (0.0,) * 7),
            0.0,
        ),
        dual_clearance_q14_rad=(0.1,) * 14,
        finger_sweep_sample_count=42,
        planner_provenance={"backend": "test"},
    )
    request_path = tmp_path / "request.json"
    plan_path = tmp_path / "plan.json"
    request.write_json(request_path)
    plan.write_json(plan_path)
    assert Dex3PreparationRequest.from_json(request_path) == request
    assert Dex3PreparationPlan.from_json(plan_path) == plan
    assert plan.request_sha256 == request.content_sha256
