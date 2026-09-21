from __future__ import annotations

import copy
import hashlib
import json
import xml.etree.ElementTree as ET
from dataclasses import replace
from itertools import pairwise
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from aprilcube import CorrespondenceResult, TagCorrespondence
from scipy.spatial.transform import Rotation

from g1_aprilcube_calibration.calibration_bundle import CalibrationBundle
from g1_aprilcube_calibration.camera_models import RectifiedCameraInfo
from g1_aprilcube_calibration.executor_state_machine import ExecutorState
from g1_aprilcube_calibration.joint_map import (
    G1_29_JOINT_NAMES,
    LEFT_ARM_INDICES,
    RIGHT_ARM_INDICES,
)
from g1_aprilcube_calibration.models import RobotStateSample
from g1_aprilcube_calibration.pose_schema import HANDOFF_POSE_ID
from g1_aprilcube_calibration.quality import QualityGrade, QualityReport
from g1_aprilcube_calibration.readiness import RecordingGateConfig
from g1_aprilcube_calibration.session_runner import RecoverableCaptureError
from g1_aprilcube_calibration.timestamp_pairing import (
    ImageTiming,
    PairingConfig,
    pair_state_to_image,
)
from g1_dex3_tabletop.calibration import (
    BilateralCalibrationDataset,
    BilateralCalibrationPlanningRequest,
    BilateralCalibrationProjection,
    BilateralCalibrationSample,
    BilateralCollectionOrchestrator,
    BilateralDesignCandidate,
    BilateralDesignConfig,
    BilateralExecutionPlan,
    BilateralFeasiblePose,
    BilateralFrameEvidence,
    BilateralIKResult,
    BilateralModelSpec,
    BilateralPlannedTransition,
    BilateralPoseDesignArtifact,
    BilateralRoutePlanningRequest,
    BilateralRoutePlanningResult,
    BilateralSessionStore,
    BilateralSolverResult,
    BilateralValidationConfig,
    BilateralVisibilityConfig,
    CameraFrameArtifact,
    TargetObservation,
    add_measured_color_frames,
    build_bilateral_optimizer_config,
    build_repeated_anchor_schedule,
    evaluate_bilateral_anchor_drift,
    merge_bilateral_datasets,
    parse_ferguson_output,
    pose_sets_from_bilateral_plan,
    sample_from_frame_evidence,
    sample_to_ferguson_record,
    select_bilateral_design,
    select_bilateral_medoid,
    validate_and_select_bilateral_model,
    write_bilateral_calibration_bundle,
)
from g1_dex3_tabletop.cli import build_parser
from g1_dex3_tabletop.hardware_bilateral_calibration import (
    _handoff_error_rad,
    validate_frozen_bilateral_inputs,
)
from g1_dex3_tabletop.planning.contracts import (
    CalibrationCandidate,
    PlannedTrajectory,
    RobotSnapshot,
)
from g1_dex3_tabletop.planning.worker import build_parser as build_worker_parser

LEFT_TARGET_HASH = "a" * 64
RIGHT_TARGET_HASH = "b" * 64
ROOT = Path(__file__).resolve().parents[1]


def target(
    side: str,
    *,
    image_points_px: tuple[tuple[float, float], ...] | None = None,
) -> TargetObservation:
    target_hash = LEFT_TARGET_HASH if side == "left" else RIGHT_TARGET_HASH
    x_offset = 0.0 if side == "left" else 100.0
    return TargetObservation(
        side=side,
        target_artifact_sha256=target_hash,
        visible_tag_ids=(4 if side == "left" else 5,),
        corner_tag_ids=(4 if side == "left" else 5,) * 4,
        image_points_px=(
            image_points_px
            if image_points_px is not None
            else tuple(
                (x_offset + x, y)
                for x, y in (
                    (10.0, 20.0),
                    (20.0, 20.0),
                    (20.0, 30.0),
                    (10.0, 30.0),
                )
            )
        ),
        object_points_m=(
            (0.0, 0.0, 0.0),
            (0.01, 0.0, 0.0),
            (0.01, 0.01, 0.0),
            (0.0, 0.01, 0.0),
        ),
        correspondence_sha256=("c" if side == "left" else "d") * 64,
    )


def sample() -> BilateralCalibrationSample:
    return BilateralCalibrationSample(
        source_session_id="same_frame_session",
        capture_id="capture_001",
        pose_group_id="left_excitation_001",
        day_group_id="2026-08-24",
        frame_id="frame_001",
        capture_role="excitation",
        raw_image_path="raw/frame_001.png",
        raw_image_sha256="e" * 64,
        camera_info={
            "width": 100,
            "height": 100,
            "frame_id": "camera_color_optical_frame",
            "camera_name": "test_camera",
            "serial_number": "test_serial",
            "distortion_model": "plumb_bob",
            "d": [0.0] * 5,
            "k": [100.0, 0.0, 50.0, 0.0, 100.0, 50.0, 0.0, 0.0, 1.0],
            "r": [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
            "p": [100.0, 0.0, 50.0, 0.0, 0.0, 100.0, 50.0, 0.0, 0.0, 0.0, 1.0, 0.0],
        },
        joint_positions_rad=(0.0,) * len(G1_29_JOINT_NAMES),
        joint_velocities_rad_s=(0.0,) * len(G1_29_JOINT_NAMES),
        pairing={"absolute_delta_s": 0.001},
        left=target("left"),
        right=target("right"),
    )


def model(*, optimize_hand_targets: bool = True) -> BilateralModelSpec:
    return BilateralModelSpec(
        name="selected_static_offsets",
        left_joint_offsets=("left_shoulder_roll_joint",),
        right_joint_offsets=("right_shoulder_roll_joint", "right_elbow_joint"),
        optimize_hand_targets=optimize_hand_targets,
    )


def targets() -> dict[str, np.ndarray]:
    left = np.eye(4)
    right = np.eye(4)
    left[0, 3] = 0.02
    right[1, 3] = -0.03
    return {"left": left, "right": right}


class FakeURDFModel:
    sha256 = "3" * 64

    def transform(
        self,
        parent: str,
        child: str,
        _positions: dict[str, float],
    ) -> np.ndarray:
        assert parent == "torso_link"
        result = np.eye(4)
        if child == "d435_link":
            return result
        if child == "left_rubber_hand":
            result[:3, 3] = (-0.1, 0.0, 1.0)
            return result
        if child == "right_rubber_hand":
            result[:3, 3] = (0.1, 0.0, 1.0)
            return result
        raise ValueError(f"unexpected fake link: {child}")


def identity_camera_artifact() -> CameraFrameArtifact:
    identity = tuple(tuple(float(value) for value in row) for row in np.eye(4))
    return CameraFrameArtifact(
        camera_serial="test_serial",
        urdf_parent_link="d435_link",
        parent_frame="camera_link",
        color_frame="camera_color_frame",
        optical_frame="camera_color_optical_frame",
        parent_T_color=identity,
        color_T_optical=identity,
        measured_at_utc="2026-08-24T00:00:00Z",
        acquisition={"source": "unit test"},
    )


def test_measured_camera_frame_artifact_is_hash_bound() -> None:
    path = ROOT / "config/cameras/realsense_348522074178_color_frames.json"
    artifact = CameraFrameArtifact.from_dict(json.loads(path.read_text(encoding="utf-8")))
    assert artifact.camera_serial == "348522074178"
    assert artifact.urdf_parent_link == "d435_link"
    assert artifact.content_sha256 == (
        "0057c30f6e9c3523f5215e804e6d2bfdacd4c455a795549ac48c2e81b28aa8ed"
    )
    assert artifact.parent_T_color[1][3] == pytest.approx(0.014834854751825333)
    assert artifact.parent_T_optical[:3, :3].T @ artifact.parent_T_optical[:3, :3] == (
        pytest.approx(np.eye(3), abs=1e-12)
    )


def test_measured_camera_frames_are_written_into_ferguson_urdf() -> None:
    artifact = identity_camera_artifact()
    parent_T_color = np.eye(4)
    parent_T_color[:3, :3] = Rotation.from_euler("xyz", (0.1, -0.2, 0.3)).as_matrix()
    parent_T_color[:3, 3] = (0.01, 0.02, 0.03)
    artifact = replace(
        artifact,
        parent_T_color=tuple(tuple(float(value) for value in row) for row in parent_T_color),
    )
    augmented = add_measured_color_frames(
        '<robot name="test"><link name="d435_link"/></robot>',
        camera_frames=artifact,
    )
    root = ET.fromstring(augmented)
    joints = {element.attrib["name"]: element for element in root.findall("joint")}
    color_origin = joints["camera_color_joint"].find("origin")
    assert color_origin is not None
    assert tuple(float(value) for value in color_origin.attrib["xyz"].split()) == (
        pytest.approx((0.01, 0.02, 0.03))
    )
    written_rotation = Rotation.from_euler(
        "xyz",
        [float(value) for value in color_origin.attrib["rpy"].split()],
    ).as_matrix()
    assert written_rotation == pytest.approx(parent_T_color[:3, :3], abs=1e-12)
    optical_joint = joints["camera_color_optical_joint"]
    assert optical_joint.find("parent").attrib["link"] == "camera_color_frame"
    assert optical_joint.find("child").attrib["link"] == "camera_color_optical_frame"


def test_design_linearizes_at_the_deployed_nominal_model() -> None:
    projection = BilateralCalibrationProjection(
        FakeURDFModel(),
        camera_frames=identity_camera_artifact(),
        model=model(),
        initial_hand_T_targets=targets(),
    )
    torso_T_camera = np.eye(4)
    torso_T_camera[:3, :3] = Rotation.from_euler("xyz", (0.03, -0.02, 0.01)).as_matrix()
    torso_T_camera[:3, 3] = (0.02, -0.01, 0.40)
    parameters = projection.parameters_for_nominal_model(
        torso_T_camera=torso_T_camera,
        hand_T_targets=targets(),
        joint_position_offsets_rad={
            "left_shoulder_roll_joint": 0.04,
            "right_shoulder_roll_joint": -0.03,
            "right_elbow_joint": 0.02,
        },
    )

    assert parameters["left_shoulder_roll_joint"] == pytest.approx(0.04)
    assert parameters["right_shoulder_roll_joint"] == pytest.approx(-0.03)
    for side in ("left", "right"):
        actual_camera, actual_target = projection.transforms(parameters, side=side)
        assert actual_camera == pytest.approx(torso_T_camera, abs=1e-12)
        assert actual_target == pytest.approx(targets()[side], abs=1e-12)


def test_bilateral_sample_requires_both_named_sides() -> None:
    with pytest.raises(ValueError, match="left and right"):
        BilateralCalibrationSample(
            source_session_id="same_frame_session",
            capture_id="capture",
            pose_group_id="pose",
            day_group_id="2026-08-24",
            frame_id="frame",
            capture_role="anchor",
            raw_image_path="raw/frame.png",
            raw_image_sha256="e" * 64,
            camera_info={},
            joint_positions_rad=(0.0,) * 29,
            joint_velocities_rad_s=(0.0,) * 29,
            pairing={},
            left=target("right"),
            right=target("left"),
        )


def test_bilateral_dataset_round_trip_is_hash_bound() -> None:
    dataset = BilateralCalibrationDataset(
        dataset_id="same_frame_session",
        session_manifest_sha256_by_id={"same_frame_session": "1" * 64},
        pose_design_sha256="2" * 64,
        execution_plan_sha256="a" * 64,
        urdf_sha256="3" * 64,
        rgb_optical_transform_sha256="4" * 64,
        left_target_artifact_sha256=LEFT_TARGET_HASH,
        right_target_artifact_sha256=RIGHT_TARGET_HASH,
        samples=(sample(),),
        provenance={"git_commit": "deadbeef"},
    )
    document = dataset.to_dict()
    restored = BilateralCalibrationDataset.from_dict(document)
    assert restored == dataset
    assert restored.content_sha256 == document["content_sha256"]

    tampered = copy.deepcopy(document)
    tampered["samples"][0]["joint_positions_rad"][0] = 0.1
    with pytest.raises(ValueError, match="content SHA-256"):
        BilateralCalibrationDataset.from_dict(tampered)


def test_optimizer_uses_distinct_camera_aliases_with_shared_parameters() -> None:
    config = build_bilateral_optimizer_config(
        model=model(),
        sample_count=12,
        initial_hand_T_targets=targets(),
    )
    step = config["robot_calibration"]["ros__parameters"]["bilateral_calibration"]
    assert step["models"] == ["left_arm", "right_arm", "left_camera", "right_camera"]
    assert step["left_camera"]["frame"] == step["right_camera"]["frame"]
    assert step["left_camera"]["param_name"] == "camera"
    assert step["right_camera"]["param_name"] == "camera"
    assert step["left_reprojection"]["model_2d"] == "left_camera"
    assert step["right_reprojection"]["model_2d"] == "right_camera"
    assert set(step["free_params"]) == {
        "left_shoulder_roll_joint",
        "right_shoulder_roll_joint",
        "right_elbow_joint",
    }
    assert {"left_calibration_target", "right_calibration_target"}.issubset(step["free_frames"])
    assert BilateralModelSpec.from_dict(model().to_dict()) == model()
    assert len(model().content_sha256) == 64


def test_one_same_frame_sample_becomes_four_observations() -> None:
    record = sample_to_ferguson_record(
        sample(),
        model=model(),
        initial_hand_T_targets=targets(),
    )
    assert record.joint_names == G1_29_JOINT_NAMES
    assert [item.sensor_name for item in record.observations] == [
        "left_arm",
        "left_camera",
        "right_arm",
        "right_camera",
    ]
    left_pixels = record.observations[1].points
    right_pixels = record.observations[3].points
    assert left_pixels != right_pixels
    assert record.observations[1].camera_info == record.observations[3].camera_info


def test_fixed_targets_fold_object_points_into_each_hand_frame() -> None:
    record = sample_to_ferguson_record(
        sample(),
        model=model(optimize_hand_targets=False),
        initial_hand_T_targets=targets(),
    )
    left_arm, _, right_arm, _ = record.observations
    assert left_arm.feature_frame == "left_rubber_hand"
    assert right_arm.feature_frame == "right_rubber_hand"
    assert left_arm.points[0] == pytest.approx((0.02, 0.0, 0.0))
    assert right_arm.points[0] == pytest.approx((0.0, -0.03, 0.0))


def test_bilateral_projection_and_normalized_observability() -> None:
    projected_sample = replace(
        sample(),
        left=target(
            "left",
            image_points_px=((40.0, 50.0), (41.0, 50.0), (41.0, 51.0), (40.0, 51.0)),
        ),
        right=target(
            "right",
            image_points_px=((60.0, 50.0), (61.0, 50.0), (61.0, 51.0), (60.0, 51.0)),
        ),
    )
    projection = BilateralCalibrationProjection(
        FakeURDFModel(),
        camera_frames=identity_camera_artifact(),
        model=BilateralModelSpec(
            name="camera_translation",
            camera_components=("x", "y", "z"),
            optimize_hand_targets=False,
        ),
        initial_hand_T_targets={"left": np.eye(4), "right": np.eye(4)},
    )
    parameters = projection.initial_parameters()
    assert projection.parameter_names == ("d435_joint_x", "d435_joint_y", "d435_joint_z")
    assert projection.pixel_residuals(parameters, (projected_sample,)) == pytest.approx(
        np.zeros(16),
        abs=1e-12,
    )
    jacobian = projection.normalized_jacobian(parameters, (projected_sample,))
    assert jacobian.shape == (16, 3)
    report = projection.observability(parameters, (projected_sample,))
    assert report.rank == 3
    assert report.observable
    assert report.normalized_null_vectors == ()


def design_candidate(
    candidate_id: str,
    side: str,
    q_value: float,
    jacobian: np.ndarray | None = None,
) -> BilateralDesignCandidate:
    matrix = np.eye(3) if jacobian is None else jacobian
    return BilateralDesignCandidate(
        candidate_id=candidate_id,
        active_arm=side,
        normalized_active_q=(q_value,) * 7,
        normalized_jacobian=tuple(tuple(float(value) for value in row) for row in matrix),
        image_coverage_bins=(0 if side == "left" else 1, int(q_value > 0.0)),
    )


def test_full_model_design_is_arm_balanced_and_repeats_anchor() -> None:
    candidates = (
        design_candidate("left_negative", "left", -0.8),
        design_candidate("left_positive", "left", 0.8),
        design_candidate("right_negative", "right", -0.8),
        design_candidate("right_positive", "right", 0.8),
    )
    selection = select_bilateral_design(
        candidates,
        parameter_names=("camera_x", "camera_y", "camera_z"),
        config=BilateralDesignConfig(
            left_excitation_count=1,
            right_excitation_count=1,
            anchor_interval=1,
            require_full_joint_excitation=False,
        ),
    )
    assert {item.active_arm for item in selection.candidates} == {"left", "right"}
    assert selection.report.model_rank == 3
    schedule = build_repeated_anchor_schedule(
        selection,
        anchor_candidate_id="bilateral_anchor",
        anchor_interval=1,
    )
    assert [item.capture_role for item in schedule] == [
        "anchor",
        "excitation",
        "anchor",
        "excitation",
        "anchor",
    ]
    assert {item.candidate_id for item in schedule if item.capture_role == "anchor"} == {
        "bilateral_anchor"
    }


def test_design_rejects_a_rank_deficient_declared_model() -> None:
    deficient = np.asarray(((1.0, 0.0, 0.0), (2.0, 0.0, 0.0)))
    candidates = (
        design_candidate("left", "left", -0.5, deficient),
        design_candidate("right", "right", 0.5, deficient),
    )
    with pytest.raises(ValueError, match="not observable"):
        select_bilateral_design(
            candidates,
            parameter_names=("one", "two", "three"),
            config=BilateralDesignConfig(
                left_excitation_count=1,
                right_excitation_count=1,
                require_full_joint_excitation=False,
            ),
        )


def bilateral_route_artifacts(
    *,
    urdf_sha256: str = "3" * 64,
) -> tuple[
    BilateralPoseDesignArtifact,
    BilateralExecutionPlan,
]:
    selection = select_bilateral_design(
        (
            design_candidate("left_excitation", "left", -0.8),
            design_candidate("right_excitation", "right", 0.8),
        ),
        parameter_names=("camera_x", "camera_y", "camera_z"),
        config=BilateralDesignConfig(
            left_excitation_count=1,
            right_excitation_count=1,
            anchor_interval=1,
            require_full_joint_excitation=False,
        ),
    )
    schedule = build_repeated_anchor_schedule(
        selection,
        anchor_candidate_id="bilateral_anchor",
        anchor_interval=1,
    )
    poses = {
        "bilateral_anchor": np.zeros(len(G1_29_JOINT_NAMES)),
        "left_excitation": np.zeros(len(G1_29_JOINT_NAMES)),
        "right_excitation": np.zeros(len(G1_29_JOINT_NAMES)),
    }
    poses["left_excitation"][LEFT_ARM_INDICES[0]] = 0.2
    poses["right_excitation"][RIGHT_ARM_INDICES[0]] = -0.2
    transitions = []
    left_indices = np.asarray(LEFT_ARM_INDICES)
    right_indices = np.asarray(RIGHT_ARM_INDICES)
    for start, end in pairwise(schedule):
        start_q = poses[start.candidate_id]
        end_q = poses[end.candidate_id]
        arm = "left" if np.any(start_q[left_indices] != end_q[left_indices]) else "right"
        indices = left_indices if arm == "left" else right_indices
        trajectory = PlannedTrajectory(
            from_pose_id=start.occurrence_id,
            to_pose_id=end.occurrence_id,
            sample_time_s=(0.0, 1.0),
            command_q_rad=(tuple(start_q[indices]), tuple(end_q[indices])),
            model_q_rad=(tuple(start_q[indices]), tuple(end_q[indices])),
            planning_time_s=0.1,
        )
        transitions.append(BilateralPlannedTransition(arm=arm, trajectory=trajectory))
    design = BilateralPoseDesignArtifact(
        model_sha256="2" * 64,
        parameter_names=("camera_x", "camera_y", "camera_z"),
        selection=selection,
        schedule=schedule,
        waypoint_joint_positions_rad={
            name: tuple(float(value) for value in position) for name, position in poses.items()
        },
        route_validation_sha256_by_transition={
            transition.transition_id: transition.content_sha256 for transition in transitions
        },
        planner_provenance={"backend": "unit-test-curobo"},
    )
    plan = BilateralExecutionPlan(
        pose_design_sha256=design.content_sha256,
        robot_model="g1-test",
        urdf_sha256=urdf_sha256,
        dex3_joint_positions_rad={
            "left": (0.0, 0.7, 0.7, -1.0, -1.5, -1.0, -1.5),
            "right": (0.0, -0.7, -0.7, 1.0, 1.5, 1.0, 1.5),
        },
        transitions=tuple(transitions),
        planner_provenance={"backend": "unit-test-curobo"},
    )
    return design, plan


def test_bilateral_execution_plan_is_endpoint_and_hash_bound() -> None:
    design, plan = bilateral_route_artifacts()
    assert design.schedule[0].occurrence_id == HANDOFF_POSE_ID
    assert design.schedule[-1].occurrence_id == HANDOFF_POSE_ID
    plan.validate_design(design)
    restored = BilateralExecutionPlan.from_dict(plan.to_dict())
    assert restored.content_sha256 == plan.content_sha256
    pose_sets = pose_sets_from_bilateral_plan(design, plan)
    assert set(pose_sets) == {"left", "right"}
    assert all(
        tuple(record.id for record in pose_set.poses)
        == tuple(item.occurrence_id for item in design.schedule[1:-1])
        for pose_set in pose_sets.values()
    )


def test_bilateral_offline_planning_contracts_are_hash_bound() -> None:
    left_full = np.zeros(len(G1_29_JOINT_NAMES))
    left_full[LEFT_ARM_INDICES[0]] = 0.2
    right_full = np.zeros(len(G1_29_JOINT_NAMES))
    right_full[RIGHT_ARM_INDICES[0]] = -0.2
    identity = tuple(tuple(float(value) for value in row) for row in np.eye(4))
    planning_request = BilateralCalibrationPlanningRequest(
        robot_model="g1-test",
        urdf_sha256="3" * 64,
        snapshot=RobotSnapshot(
            measured_q29_rad=(0.0,) * len(G1_29_JOINT_NAMES),
            left_dex3_q_rad=(0.0, 0.7, 0.7, -1.0, -1.5, -1.0, -1.5),
            right_dex3_q_rad=(0.0, -0.7, -0.7, 1.0, 1.5, 1.0, 1.5),
        ),
        camera_info=sample().camera_info,
        camera_frames=identity_camera_artifact(),
        design_model=model(),
        design_config=BilateralDesignConfig(
            left_excitation_count=1,
            right_excitation_count=1,
            anchor_interval=1,
            require_full_joint_excitation=False,
        ),
        visibility_config=BilateralVisibilityConfig(),
        nominal_torso_T_camera=identity,
        nominal_hand_T_targets={"left": identity, "right": identity},
        joint_position_offsets_rad={},
        candidates_by_arm={
            "left": (
                CalibrationCandidate(
                    candidate_id="left_candidate",
                    camera_T_marker=identity,
                    selection_metadata={"source": "unit test"},
                ),
            ),
            "right": (
                CalibrationCandidate(
                    candidate_id="right_candidate",
                    camera_T_marker=identity,
                    selection_metadata={"source": "unit test"},
                ),
            ),
        },
        target_artifact_sha256_by_arm={
            "left": LEFT_TARGET_HASH,
            "right": RIGHT_TARGET_HASH,
        },
        target_corner_tag_ids_by_arm={"left": (5,) * 4, "right": (4,) * 4},
        target_object_points_m_by_arm={
            side: target(side).object_points_m for side in ("left", "right")
        },
        ik_batch_size=16,
        random_seed=17,
        source_provenance={"source": "unit test"},
    )
    restored_request = BilateralCalibrationPlanningRequest.from_dict(planning_request.to_dict())
    assert restored_request.content_sha256 == planning_request.content_sha256

    ik_result = BilateralIKResult(
        request_sha256=planning_request.content_sha256,
        poses=(
            BilateralFeasiblePose(
                candidate_id="left_candidate",
                active_arm="left",
                active_model_q_rad=tuple(left_full[index] for index in LEFT_ARM_INDICES),
                active_command_q_rad=tuple(left_full[index] for index in LEFT_ARM_INDICES),
                full_command_q29_rad=tuple(left_full),
                ik_position_error_m=0.001,
                ik_rotation_error_rad=0.002,
                candidate_metadata={"source": "unit test"},
            ),
            BilateralFeasiblePose(
                candidate_id="right_candidate",
                active_arm="right",
                active_model_q_rad=tuple(right_full[index] for index in RIGHT_ARM_INDICES),
                active_command_q_rad=tuple(right_full[index] for index in RIGHT_ARM_INDICES),
                full_command_q29_rad=tuple(right_full),
                ik_position_error_m=0.001,
                ik_rotation_error_rad=0.002,
                candidate_metadata={"source": "unit test"},
            ),
        ),
        planner_provenance={"backend": "unit-test-curobo"},
    )
    ik_result.validate_request(planning_request)
    assert BilateralIKResult.from_dict(ik_result.to_dict()) == ik_result


def test_bilateral_route_contract_binds_schedule_and_transition_arms() -> None:
    design, plan = bilateral_route_artifacts()
    route_request = BilateralRoutePlanningRequest(
        planning_request_sha256="4" * 64,
        ik_result_sha256="5" * 64,
        robot_model=plan.robot_model,
        urdf_sha256=plan.urdf_sha256,
        snapshot=RobotSnapshot(
            measured_q29_rad=design.waypoint_joint_positions_rad["bilateral_anchor"],
            left_dex3_q_rad=plan.dex3_joint_positions_rad["left"],
            right_dex3_q_rad=plan.dex3_joint_positions_rad["right"],
        ),
        joint_position_offsets_rad={},
        parameter_names=design.parameter_names,
        selection=design.selection,
        schedule=design.schedule,
        waypoint_joint_positions_rad=design.waypoint_joint_positions_rad,
        design_provenance={"source": "unit test"},
        random_seed=17,
    )
    assert BilateralRoutePlanningRequest.from_dict(route_request.to_dict()) == route_request
    result = BilateralRoutePlanningResult(
        request_sha256=route_request.content_sha256,
        transitions=plan.transitions,
        disconnected_candidate_ids=(),
        planner_provenance={"backend": "unit-test-curobo"},
    )
    result.validate_request(route_request)
    assert BilateralRoutePlanningResult.from_dict(result.to_dict()) == result

    wrong = replace(result, request_sha256="6" * 64)
    with pytest.raises(ValueError, match="different request"):
        wrong.validate_request(route_request)

    moved_anchor = np.asarray(route_request.snapshot.measured_q29_rad).copy()
    moved_anchor[0] = 0.01
    with pytest.raises(ValueError, match="anchor differs"):
        replace(
            route_request,
            snapshot=RobotSnapshot(
                measured_q29_rad=tuple(moved_anchor),
                left_dex3_q_rad=route_request.snapshot.left_dex3_q_rad,
                right_dex3_q_rad=route_request.snapshot.right_dex3_q_rad,
            ),
        )


def test_bilateral_planner_commands_are_offline_worker_phases() -> None:
    operator = build_parser().parse_args(
        [
            "plan-bilateral-calibration",
            "--snapshot",
            "snapshot.json",
            "--output-directory",
            "planned",
        ]
    )
    assert operator.command == "plan-bilateral-calibration"
    assert operator.left_excitation_count == operator.right_excitation_count == 34
    for command in (
        "solve-bilateral-calibration-ik",
        "plan-bilateral-calibration-route",
    ):
        worker = build_worker_parser().parse_args(
            [command, "--request", "request.json", "--output", "result.json"]
        )
        assert worker.command == command


def test_bilateral_hardware_command_is_a_separate_frozen_route_command() -> None:
    args = build_parser().parse_args(
        [
            "collect-bilateral-calibration",
            "--network-interface",
            "enp1s0",
            "--pose-design",
            "design.json",
            "--execution-plan",
            "plan.json",
            "--confirm",
            "test-only",
        ]
    )
    assert args.command == "collect-bilateral-calibration"
    assert args.maximum_capture_attempts == 3
    assert args.hardware_config is None


def test_bilateral_hardware_preflight_binds_route_model_and_camera() -> None:
    design, plan = bilateral_route_artifacts()
    camera = identity_camera_artifact()
    model = SimpleNamespace(
        name=plan.robot_model,
        sha256=plan.urdf_sha256,
        links=(camera.urdf_parent_link,),
    )
    expected_camera = SimpleNamespace(
        serial_number=camera.camera_serial,
        frame_id=camera.optical_frame,
    )
    pose_sets = validate_frozen_bilateral_inputs(
        design=design,
        plan=plan,
        model=model,
        camera_frames=camera,
        expected_camera=expected_camera,
    )
    assert set(pose_sets) == {"left", "right"}
    anchor = design.waypoint_joint_positions_rad[design.schedule[0].candidate_id]
    assert _handoff_error_rad(design, anchor) == 0.0
    with pytest.raises(ValueError, match="RealSense serial"):
        validate_frozen_bilateral_inputs(
            design=design,
            plan=plan,
            model=model,
            camera_frames=camera,
            expected_camera=SimpleNamespace(
                serial_number="wrong",
                frame_id=camera.optical_frame,
            ),
        )


def test_bilateral_execution_plan_rejects_an_inactive_arm_change() -> None:
    design, plan = bilateral_route_artifacts()
    positions = dict(design.waypoint_joint_positions_rad)
    changed = np.asarray(positions["left_excitation"]).copy()
    changed[RIGHT_ARM_INDICES[0]] = 0.1
    positions["left_excitation"] = tuple(changed)
    invalid_design = replace(design, waypoint_joint_positions_rad=positions)
    invalid_plan = replace(plan, pose_design_sha256=invalid_design.content_sha256)
    with pytest.raises(ValueError, match="outside left"):
        invalid_plan.validate_design(invalid_design)


def test_bilateral_orchestrator_retries_and_switches_arms() -> None:
    design, plan = bilateral_route_artifacts()
    pose_sets = pose_sets_from_bilateral_plan(design, plan)

    class FakeExecutor:
        def __init__(self) -> None:
            self.state = ExecutorState.READY
            self.current_pose_id = HANDOFF_POSE_ID
            self.approved_validation_report_sha256 = plan.content_sha256
            self.pose_set = pose_sets["left"]
            self.started: list[dict] = []
            self.switches: list[str] = []
            self.capture_outcomes: list[str] = []

        def begin_capture(self) -> None:
            self.state = ExecutorState.CAPTURING

        def finish_capture(self, *, outcome: str) -> None:
            self.capture_outcomes.append(outcome)
            self.state = ExecutorState.HOLDING

        def observe_state(self):
            return object()

        def switch_validated_arm_plan(self, **kwargs) -> None:
            self.pose_set = kwargs["pose_set"]
            self.switches.append(self.pose_set.calibration_arm)

        def start_trajectory(self, **kwargs) -> None:
            self.started.append(kwargs)
            self.current_pose_id = kwargs["to_pose_id"]
            self.state = ExecutorState.READY

    class FakeSource:
        def __init__(self) -> None:
            self.calls = 0
            self.requests: list[dict] = []

        def capture_burst(self, **kwargs):
            self.calls += 1
            self.requests.append(kwargs)
            if self.calls == 1:
                raise RecoverableCaptureError("need more frames")
            return (object(),)

    class FakeStore:
        def __init__(self) -> None:
            self.captures: list[dict] = []
            self.finalized = False

        def append_capture(self, **kwargs) -> None:
            self.captures.append(kwargs)

        def finalize(self) -> None:
            self.finalized = True

    executor = FakeExecutor()
    source = FakeSource()
    store = FakeStore()
    orchestrator = BilateralCollectionOrchestrator(
        executor=executor,
        design=design,
        plan=plan,
        pose_sets=pose_sets,
        store=store,
        frame_source=source,
        wait_until_ready=lambda: None,
    )
    result = orchestrator.run()
    assert result.accepted_count == len(design.schedule)
    assert result.retry_count == 1
    assert result.attempted_count == len(design.schedule) + 1
    assert executor.switches == ["right"]
    assert len(executor.started) == len(plan.transitions)
    assert store.captures[0]["outcome"] == "retry"
    assert store.finalized
    assert all(request["remember_signatures"] for request in source.requests)


def frame_evidence(frame_id: str, corner_shift: float) -> BilateralFrameEvidence:
    timing = ImageTiming(
        receipt_monotonic_s=1.0,
        receipt_utc="2026-08-24T00:00:01Z",
        header_stamp_ns=1_000_000_000,
    )
    states = tuple(
        RobotStateSample(
            receipt_monotonic_s=value,
            receipt_utc=f"2026-08-24T00:00:0{index}Z",
            mode_machine=5,
            position=np.zeros(29),
            velocity=np.zeros(29),
            estimated_torque=np.zeros(29),
        )
        for index, value in enumerate((0.9, 1.1))
    )
    pairing = pair_state_to_image(
        timing,
        states,
        config=PairingConfig(maximum_nearest_delta_s=0.2, maximum_bracket_span_s=0.3),
    )
    camera_info = RectifiedCameraInfo.from_dict(sample().camera_info)

    def result(side: str) -> CorrespondenceResult:
        base = 20.0 if side == "left" else 60.0
        corners = (
            np.asarray(
                (
                    (base, 20.0),
                    (base + 10.0, 20.0),
                    (base + 10.0, 30.0),
                    (base, 30.0),
                )
            )
            + corner_shift
        )
        return CorrespondenceResult(
            image_size_wh=(100, 100),
            observations=(
                TagCorrespondence(
                    tag_id=4 if side == "left" else 5,
                    face_name="plate",
                    image_corners_px=corners,
                    object_corners_mm=np.zeros((4, 3)),
                    quad_quality=1.0,
                    shortest_side_px=10.0,
                    image_margin_px=10.0,
                ),
            ),
        )

    quality = QualityReport(
        grade=QualityGrade.GREEN,
        hard_failures=(),
        warnings=(),
        metrics={},
        signature=None,
        pose_diagnostic=None,
    )
    return BilateralFrameEvidence(
        frame_id=frame_id,
        image_bgr=np.zeros((100, 100, 3), dtype=np.uint8),
        image_timing=timing,
        camera_info=camera_info,
        state_window=states,
        pairing=pairing,
        left_correspondences=result("left"),
        right_correspondences=result("right"),
        left_quality=quality,
        right_quality=quality,
    )


def test_bilateral_medoid_uses_both_targets_from_the_same_frame() -> None:
    frames = (
        frame_evidence("near_left", 0.0),
        frame_evidence("medoid", 2.0),
        frame_evidence("far_right", 20.0),
    )
    assert select_bilateral_medoid(frames).frame_id == "medoid"


def test_bilateral_frame_rejects_a_missing_target() -> None:
    valid = frame_evidence("valid", 0.0)
    missing = CorrespondenceResult(image_size_wh=(100, 100), observations=())
    with pytest.raises(ValueError, match="right target"):
        replace(valid, right_correspondences=missing)


def test_verified_same_frame_evidence_becomes_one_bilateral_sample() -> None:
    result = sample_from_frame_evidence(
        frame_evidence("selected", 0.0),
        source_session_id="same_frame_session",
        capture_id="capture_007",
        pose_group_id="right_excitation_003",
        day_group_id="2026-08-24",
        capture_role="excitation",
        raw_image_path="raw/capture_007/selected.png",
        raw_image_sha256="f" * 64,
        left_target_artifact_sha256=LEFT_TARGET_HASH,
        right_target_artifact_sha256=RIGHT_TARGET_HASH,
    )
    assert result.frame_id == "selected"
    assert result.left.visible_tag_ids == (4,)
    assert result.right.visible_tag_ids == (5,)
    assert len(result.left.object_points_m) == len(result.right.object_points_m) == 4
    assert result.pairing["nearest_state_monotonic_s"] == pytest.approx(0.9)
    record = sample_to_ferguson_record(
        result,
        model=model(),
        initial_hand_T_targets=targets(),
    )
    assert record.stamp_ns == 1_000_000_000


def test_ferguson_output_parser_uses_the_final_converged_report() -> None:
    output = """
Ceres Solver Report: Iterations: 3, Initial cost: 10, Final cost: 4, Termination: CONVERGENCE
Ceres Solver Report: Iterations: 8, Initial cost: 4, Final cost: 1.25e-2, Termination: CONVERGENCE
Parameter Offsets:
left_shoulder_roll_joint: 0.052
d435_joint_x: -5.0e-4

trailing report text
"""
    offsets, iterations, cost, termination = parse_ferguson_output(output)
    assert offsets == {
        "left_shoulder_roll_joint": pytest.approx(0.052),
        "d435_joint_x": pytest.approx(-0.0005),
    }
    assert iterations == 8
    assert cost == pytest.approx(0.0125)
    assert termination == "CONVERGENCE"


def exact_projection_sample(
    index: int,
    *,
    pose_group_id: str,
    day_group_id: str,
    capture_role: str,
) -> BilateralCalibrationSample:
    return replace(
        sample(),
        capture_id=f"capture_{index:03d}",
        pose_group_id=pose_group_id,
        day_group_id=day_group_id,
        frame_id=f"frame_{index:03d}",
        capture_role=capture_role,
        raw_image_path=f"raw/frame_{index:03d}.png",
        left=target(
            "left",
            image_points_px=(
                (40.0, 50.0),
                (41.0, 50.0),
                (41.0, 51.0),
                (40.0, 51.0),
            ),
        ),
        right=target(
            "right",
            image_points_px=(
                (60.0, 50.0),
                (61.0, 50.0),
                (61.0, 51.0),
                (60.0, 51.0),
            ),
        ),
    )


def exact_projection_dataset(*, urdf_sha256: str = "3" * 64) -> BilateralCalibrationDataset:
    samples = (
        exact_projection_sample(
            0,
            pose_group_id="bilateral_anchor",
            day_group_id="2026-08-24",
            capture_role="anchor",
        ),
        exact_projection_sample(
            1,
            pose_group_id="left_001",
            day_group_id="2026-08-24",
            capture_role="excitation",
        ),
        exact_projection_sample(
            2,
            pose_group_id="bilateral_anchor",
            day_group_id="2026-08-25",
            capture_role="anchor",
        ),
        exact_projection_sample(
            3,
            pose_group_id="right_001",
            day_group_id="2026-08-25",
            capture_role="excitation",
        ),
        exact_projection_sample(
            4,
            pose_group_id="bilateral_anchor",
            day_group_id="2026-08-25",
            capture_role="anchor",
        ),
        exact_projection_sample(
            5,
            pose_group_id="left_002",
            day_group_id="2026-08-24",
            capture_role="excitation",
        ),
    )
    artifact = identity_camera_artifact()
    return BilateralCalibrationDataset(
        dataset_id="bilateral_validation_session",
        session_manifest_sha256_by_id={"same_frame_session": "1" * 64},
        pose_design_sha256="2" * 64,
        execution_plan_sha256="a" * 64,
        urdf_sha256=urdf_sha256,
        rgb_optical_transform_sha256=artifact.content_sha256,
        left_target_artifact_sha256=LEFT_TARGET_HASH,
        right_target_artifact_sha256=RIGHT_TARGET_HASH,
        samples=samples,
        provenance={"git_commit": "deadbeef"},
    )


def camera_translation_result(
    dataset: BilateralCalibrationDataset,
    selected_model: BilateralModelSpec,
) -> BilateralSolverResult:
    projection = BilateralCalibrationProjection(
        FakeURDFModel(),
        camera_frames=identity_camera_artifact(),
        model=selected_model,
        initial_hand_T_targets={"left": np.eye(4), "right": np.eye(4)},
    )
    parameters = projection.initial_parameters()
    observability = projection.observability(parameters, dataset.samples)
    torso_T_camera, left_target = projection.transforms(parameters, side="left")
    _, right_target = projection.transforms(parameters, side="right")
    return BilateralSolverResult(
        model=selected_model,
        parameters=parameters,
        torso_T_camera=torso_T_camera,
        hand_T_targets={"left": left_target, "right": right_target},
        combined_radial_rms_px=0.0,
        arm_radial_rms_px={"left": 0.0, "right": 0.0},
        observability=observability,
        iterations=1,
        final_cost=0.0,
        termination="CONVERGENCE",
    )


def test_grouped_validation_selects_model_and_writes_production_bundle(
    tmp_path: Path,
) -> None:
    base_urdf = tmp_path / "g1.urdf"
    base_urdf.write_text('<robot name="g1"/>\n', encoding="utf-8")
    urdf_sha256 = hashlib.sha256(base_urdf.read_bytes()).hexdigest()
    dataset = exact_projection_dataset(urdf_sha256=urdf_sha256)
    selected_model = BilateralModelSpec(
        name="shared_camera_fixed_targets",
        camera_components=("x", "y", "z"),
        optimize_hand_targets=False,
    )

    def fit(
        training_dataset: BilateralCalibrationDataset,
        model_spec: BilateralModelSpec,
        _output_directory: Path,
    ) -> BilateralSolverResult:
        return camera_translation_result(training_dataset, model_spec)

    report = validate_and_select_bilateral_model(
        dataset,
        FakeURDFModel(),
        camera_frames=identity_camera_artifact(),
        initial_hand_T_targets={"left": np.eye(4), "right": np.eye(4)},
        models=(selected_model,),
        fit=fit,
        output_directory=tmp_path / "validation",
        config=BilateralValidationConfig(
            pose_fold_count=2,
            require_multiple_days=True,
        ),
    )
    assert report.passed
    assert report.selected_model == selected_model
    assert report.bootstrap is not None
    assert report.bootstrap.passed
    assert report.bootstrap.successful_trials == report.config.bootstrap_trials
    assert report.bootstrap_attempts == (report.bootstrap,)
    assert report.candidates[0].pose_holdout.combined_rms_px == pytest.approx(0.0)
    assert report.candidates[0].day_holdout.combined_rms_px == pytest.approx(0.0)
    assert report.candidates[0].pose_anchor_holdout.sample_count == 3

    full_result = camera_translation_result(dataset, selected_model)
    anchor_drift = evaluate_bilateral_anchor_drift(
        dataset,
        FakeURDFModel(),
        full_result,
        config=report.config,
    )
    assert anchor_drift.passed
    destination = write_bilateral_calibration_bundle(
        result=full_result,
        dataset=dataset,
        validation_report=report,
        anchor_drift_report=anchor_drift,
        camera_frames=identity_camera_artifact(),
        initial_hand_T_targets={"left": np.eye(4), "right": np.eye(4)},
        base_urdf_path=base_urdf,
        destination=tmp_path / "candidate_bundle.json",
        bundle_id="bilateral_candidate",
        provenance={"command": "unit-test"},
    )
    loaded = CalibrationBundle.load(destination)
    assert loaded.bundle_id == "bilateral_candidate"
    assert loaded.joint_position_offsets_rad == {}
    assert loaded.provenance["dataset_sha256"] == dataset.content_sha256
    assert loaded.validation["report"]["content_sha256"] == report.content_sha256


def test_bilateral_dataset_merge_preserves_each_source_manifest() -> None:
    first = exact_projection_dataset()
    second = replace(
        first,
        dataset_id="second_session_dataset",
        session_manifest_sha256_by_id={"second_session": "9" * 64},
        samples=tuple(
            replace(
                sample,
                source_session_id="second_session",
                day_group_id="2026-08-26",
            )
            for sample in first.samples
        ),
        provenance={"git_commit": "second"},
    )
    merged = merge_bilateral_datasets(
        (first, second),
        dataset_id="two_day_commissioning",
    )
    assert merged.dataset_id == "two_day_commissioning"
    assert merged.session_manifest_sha256_by_id == {
        "same_frame_session": "1" * 64,
        "second_session": "9" * 64,
    }
    assert len(merged.samples) == 2 * len(first.samples)
    assert {sample.day_group_id for sample in merged.samples} == {
        "2026-08-24",
        "2026-08-25",
        "2026-08-26",
    }


def test_bilateral_raw_session_replays_both_correspondence_hashes(
    tmp_path: Path,
) -> None:
    artifact = identity_camera_artifact()
    robot_bytes = b'<robot name="g1"/>\n'
    robot_sha256 = hashlib.sha256(robot_bytes).hexdigest()
    design, execution_plan = bilateral_route_artifacts(urdf_sha256=robot_sha256)
    camera_bytes = json.dumps(artifact.to_dict(), indent=2, sort_keys=True).encode() + b"\n"
    source_artifacts = {
        "camera_frames.json": camera_bytes,
        "capture_quality.yaml": b"schema_version: 1\n",
        "execution_plan.json": (
            json.dumps(execution_plan.to_dict(), indent=2, sort_keys=True).encode() + b"\n"
        ),
        "left_target.json": b'{"name":"left"}\n',
        "pose_design.json": (
            json.dumps(design.to_dict(), indent=2, sort_keys=True).encode() + b"\n"
        ),
        "right_target.json": b'{"name":"right"}\n',
        "robot.urdf": robot_bytes,
    }
    frames_by_shift = {
        shift: replace(
            frame_evidence(frame_id, float(shift)),
            image_bgr=np.full((100, 100, 3), shift, dtype=np.uint8),
        )
        for shift, frame_id in ((0, "near"), (2, "medoid"), (20, "far"))
    }
    selected = frames_by_shift[2]

    class FixedDetector:
        def __init__(self, side: str) -> None:
            self.side = side

        def detect(self, _image: np.ndarray) -> CorrespondenceResult:
            shift = int(_image[0, 0, 0])
            return getattr(frames_by_shift[shift], f"{self.side}_correspondences")

    store = BilateralSessionStore(tmp_path / "bilateral_session")
    manifest = store.create(
        session_id="bilateral_session",
        created_at_utc="2026-08-24T00:00:00Z",
        day_group_id="2026-08-24",
        camera_info=selected.camera_info,
        camera_frames=artifact,
        pose_design_sha256=design.content_sha256,
        execution_plan_sha256=execution_plan.content_sha256,
        source_artifacts=source_artifacts,
        pairing_config=PairingConfig(
            maximum_nearest_delta_s=0.2,
            maximum_bracket_span_s=0.3,
        ),
        recording_gate_config=RecordingGateConfig(
            stationary_duration_s=0.1,
            maximum_state_gap_s=0.3,
            state_freshness_timeout_s=0.3,
            minimum_samples=2,
        ),
        provenance={"command": "unit-test"},
    )
    assert not manifest.finalized
    store.append_capture(
        capture_id="capture_001",
        pose_group_id="bilateral_anchor",
        capture_role="anchor",
        outcome="accepted",
        reason="both targets passed",
        frames=(frames_by_shift[0], selected, frames_by_shift[20]),
        recorded_at_utc="2026-08-24T00:00:01Z",
    )
    finalized = store.finalize()
    assert finalized.finalized
    detectors = {
        "left": FixedDetector("left"),
        "right": FixedDetector("right"),
    }
    first = store.build_dataset(
        detectors=detectors,
        output_path=tmp_path / "bilateral_dataset.json",
    )
    second = store.build_dataset(detectors=detectors)
    assert first.content_sha256 == second.content_sha256
    assert first.session_manifest_sha256_by_id == {"bilateral_session": finalized.content_sha256}
    assert first.samples[0].frame_id == "medoid"
    assert first.samples[0].day_group_id == "2026-08-24"
    assert first.samples[0].left.visible_tag_ids == (4,)
    assert first.samples[0].right.visible_tag_ids == (5,)
