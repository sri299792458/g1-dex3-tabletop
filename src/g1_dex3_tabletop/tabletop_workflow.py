"""Pure orchestration helpers for the cube-only tabletop lifecycle."""

from __future__ import annotations

import hashlib
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from g1_aprilcube_calibration.calibration_bundle import CalibrationBundle
from g1_aprilcube_calibration.joint_map import arm_indices, validate_arm_side
from g1_dex3_tabletop.camera_state_sync import SynchronizedCameraStateInput
from g1_dex3_tabletop.planning.contracts import RobotSnapshot
from g1_dex3_tabletop.state_estimation import CameraStateEstimate
from g1_dex3_tabletop.tabletop_contracts import (
    EstimatedCameraPlanningState,
    SupportedEscapePlan,
    TabletopCuboid,
    TabletopExecutionPlan,
    TabletopFixture,
    TabletopObservation,
    TabletopPickPlaceRequest,
    TabletopTaskPlan,
    TabletopTaskRequest,
    combine_tabletop_plans,
)


def file_sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_task_config(path: str | Path) -> dict[str, Any]:
    document = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(document, dict) or document.get("schema_version") != 1:
        raise ValueError("unsupported tabletop task configuration")
    if document.get("table", {}).get("collision_policy") != ("local_manipulation_geometry_plane"):
        raise ValueError("unsupported tabletop collision policy")
    maximum_velocity = float(document.get("motion", {}).get("maximum_arm_velocity_rad_s", 0.0))
    if not np.isfinite(maximum_velocity) or maximum_velocity <= 0.0:
        raise ValueError("tabletop maximum arm velocity must be positive and finite")
    minimum_hand_clearance = float(
        document.get("table", {}).get("minimum_hand_plane_clearance_m", 0.0)
    )
    if not np.isfinite(minimum_hand_clearance) or minimum_hand_clearance <= 0.0:
        raise ValueError("tabletop minimum hand-plane clearance must be positive and finite")
    return document


def build_tabletop_request(
    *,
    arm: str,
    observation: TabletopObservation,
    calibration_bundle: CalibrationBundle,
    calibration_bundle_path: str | Path,
    grasp_shortlist_path: str | Path,
    task_config_path: str | Path,
    object_dimensions_m: tuple[float, float, float],
    maximum_arm_velocity_rad_s: float | None = None,
    pregrasp_distance_m: float | None = None,
    presentation_id: str = "direct",
    fixture: TabletopFixture | None = None,
    environment_cuboids: tuple[TabletopCuboid, ...] = (),
) -> TabletopTaskRequest:
    task = load_task_config(task_config_path)
    if CalibrationBundle.load(calibration_bundle_path).content_sha256 != (
        calibration_bundle.content_sha256
    ):
        raise ValueError("calibration bundle object differs from its source file")
    shortlist = Path(grasp_shortlist_path)
    root = Path(__file__).resolve().parents[2]
    try:
        relative_shortlist = shortlist.resolve().relative_to(root)
    except ValueError as error:
        raise ValueError("grasp shortlist must be inside the repository") from error
    configured_velocity = float(task["motion"]["maximum_arm_velocity_rad_s"])
    selected_velocity = (
        configured_velocity
        if maximum_arm_velocity_rad_s is None
        else float(maximum_arm_velocity_rad_s)
    )
    shortlist_document = yaml.safe_load(shortlist.read_text(encoding="utf-8"))
    try:
        configured_pregrasp_distance = float(
            shortlist_document["execution_contract"]["approach_distance_m"]
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("grasp shortlist lacks a valid approach distance") from error
    selected_pregrasp_distance = (
        configured_pregrasp_distance
        if pregrasp_distance_m is None
        else float(pregrasp_distance_m)
    )
    return TabletopTaskRequest(
        observation=observation,
        arm=validate_arm_side(arm),
        torso_T_camera=tuple(
            tuple(float(value) for value in row) for row in calibration_bundle.torso_T_camera
        ),
        joint_position_offsets_rad=dict(calibration_bundle.joint_position_offsets_rad),
        calibration_bundle_sha256=calibration_bundle.content_sha256,
        grasp_shortlist_path=str(relative_shortlist),
        grasp_shortlist_sha256=file_sha256(shortlist),
        presentation_id=presentation_id,
        fixture=fixture,
        environment_cuboids=environment_cuboids,
        object_dimensions_m=tuple(object_dimensions_m),
        open_transit_table_patch_dimensions_m=tuple(
            task["table"]["open_transit_patch_dimensions_m"]
        ),
        minimum_hand_plane_clearance_m=float(task["table"]["minimum_hand_plane_clearance_m"]),
        supported_escape_m=float(task["motion"]["supported_escape_m"]),
        retention_test_lift_m=float(task["motion"]["retention_test_lift_m"]),
        lift_m=float(task["motion"]["payload_lift_m"]),
        maximum_arm_velocity_rad_s=selected_velocity,
        pregrasp_distance_m=selected_pregrasp_distance,
    )


def request_at_clearance(
    loaded_request: TabletopTaskRequest,
    escape: SupportedEscapePlan,
) -> TabletopTaskRequest:
    """Use the exact supported-escape endpoint as the task planning state."""

    return request_at_clearance_observation(
        loaded_request,
        escape,
        loaded_request.observation,
    )


def request_at_clearance_observation(
    loaded_request: TabletopTaskRequest,
    escape: SupportedEscapePlan,
    observation: TabletopObservation,
) -> TabletopTaskRequest:
    """Bind a fresh fixed-cube observation to the exact clearance command.

    The camera/object pose and all nonselected robot coordinates come from the
    stationary boundary observation. The selected arm remains the exact
    collision-validated escape endpoint so the replanned task has bitwise
    command continuity with the trajectory already executed.
    """

    if escape.request_sha256 != loaded_request.content_sha256:
        raise ValueError("supported escape belongs to a different loaded request")
    source = observation
    if source.camera_profile_sha256 != loaded_request.observation.camera_profile_sha256:
        raise ValueError("clearance observation uses a different camera profile")
    q29 = np.asarray(source.snapshot.measured_q29_rad).copy()
    q29[np.asarray(arm_indices(loaded_request.arm))] = np.asarray(
        escape.outbound.command_q_rad[-1]
    )
    snapshot = RobotSnapshot(
        measured_q29_rad=tuple(q29),
        left_dex3_q_rad=source.snapshot.left_dex3_q_rad,
        right_dex3_q_rad=source.snapshot.right_dex3_q_rad,
    )
    observation = TabletopObservation(
        snapshot=snapshot,
        camera_T_object=source.camera_T_object,
        camera_profile_sha256=source.camera_profile_sha256,
        source_frame_sha256=source.source_frame_sha256,
        object_translation_spread_mm=source.object_translation_spread_mm,
        object_rotation_spread_deg=source.object_rotation_spread_deg,
    )
    return TabletopTaskRequest(
        observation=observation,
        arm=loaded_request.arm,
        torso_T_camera=loaded_request.torso_T_camera,
        joint_position_offsets_rad=loaded_request.joint_position_offsets_rad,
        calibration_bundle_sha256=loaded_request.calibration_bundle_sha256,
        grasp_shortlist_path=loaded_request.grasp_shortlist_path,
        grasp_shortlist_sha256=loaded_request.grasp_shortlist_sha256,
        presentation_id=loaded_request.presentation_id,
        fixture=loaded_request.fixture,
        environment_cuboids=loaded_request.environment_cuboids,
        table_reference_camera_T_object=loaded_request.table_reference_camera_T_object,
        table_reference_object_dimensions_m=(loaded_request.table_reference_object_dimensions_m),
        object_dimensions_m=loaded_request.object_dimensions_m,
        open_transit_table_patch_dimensions_m=(
            loaded_request.open_transit_table_patch_dimensions_m
        ),
        minimum_hand_plane_clearance_m=loaded_request.minimum_hand_plane_clearance_m,
        supported_escape_m=loaded_request.supported_escape_m,
        retention_test_lift_m=loaded_request.retention_test_lift_m,
        lift_m=loaded_request.lift_m,
        maximum_arm_velocity_rad_s=loaded_request.maximum_arm_velocity_rad_s,
        random_seed=loaded_request.random_seed,
    )


def request_at_estimated_pregrasp(
    clearance_request: TabletopTaskRequest,
    *,
    snapshot: RobotSnapshot,
    estimate: CameraStateEstimate,
    anchor_input: SynchronizedCameraStateInput,
    current_input: SynchronizedCameraStateInput,
) -> TabletopTaskRequest:
    """Bind one propagated camera pose to an exact stationary pregrasp state."""

    if clearance_request.estimated_planning_state is not None:
        raise ValueError("pregrasp estimate requires the original visual clearance anchor")
    if estimate.anchor_timestamp_ns != anchor_input.sample.timestamp_ns:
        raise ValueError("camera estimate and visual-anchor state have different times")
    if estimate.timestamp_ns != current_input.sample.timestamp_ns:
        raise ValueError("camera estimate and current state have different times")
    state = EstimatedCameraPlanningState(
        snapshot=snapshot,
        object_T_camera=tuple(
            tuple(float(value) for value in row) for row in estimate.reference_T_camera
        ),
        anchor_observation_sha256=clearance_request.observation.content_sha256,
        anchor_timestamp_ns=estimate.anchor_timestamp_ns,
        timestamp_ns=estimate.timestamp_ns,
        anchor_input_timing=anchor_input.to_dict(),
        current_input_timing=current_input.to_dict(),
    )
    return replace(clearance_request, estimated_planning_state=state)


def destination_request_for_pick_place(
    request: TabletopPickPlaceRequest,
) -> TabletopTaskRequest:
    """Express the same fixed world in the destination object's frame."""

    source = request.source_request
    if source.estimated_planning_state is not None:
        raise ValueError("pick-place destination synthesis requires a visual boundary request")
    source_T_destination = np.asarray(request.source_T_destination_object, dtype=np.float64)
    destination_T_source = np.linalg.inv(source_T_destination)
    camera_T_destination = (
        np.asarray(source.observation.camera_T_object, dtype=np.float64) @ source_T_destination
    )
    destination_environment = []
    for cuboid in source.environment_cuboids:
        role = (
            "placement_support"
            if cuboid.object_id == request.destination_support_object_id
            else "obstacle"
        )
        destination_environment.append(
            TabletopCuboid(
                object_id=cuboid.object_id,
                object_T_cuboid=destination_T_source
                @ np.asarray(cuboid.object_T_cuboid, dtype=np.float64),
                dimensions_m=cuboid.dimensions_m,
                role=role,
            )
        )
    destination_observation = TabletopObservation(
        snapshot=source.observation.snapshot,
        camera_T_object=camera_T_destination,
        camera_profile_sha256=source.observation.camera_profile_sha256,
        source_frame_sha256=source.observation.source_frame_sha256,
        object_translation_spread_mm=source.observation.object_translation_spread_mm,
        object_rotation_spread_deg=source.observation.object_rotation_spread_deg,
    )
    table_reference_pose = source.table_reference_camera_T_object
    table_reference_dimensions = source.table_reference_object_dimensions_m
    if table_reference_pose is None:
        table_reference_pose = source.observation.camera_T_object
        table_reference_dimensions = source.object_dimensions_m
    return replace(
        source,
        observation=destination_observation,
        environment_cuboids=tuple(destination_environment),
        table_reference_camera_T_object=table_reference_pose,
        table_reference_object_dimensions_m=table_reference_dimensions,
    )


def assemble_execution_plan(
    *,
    loaded_request: TabletopTaskRequest,
    supported_escape: SupportedEscapePlan,
    task: TabletopTaskPlan,
) -> tuple[TabletopTaskRequest, TabletopExecutionPlan]:
    clearance_request = request_at_clearance(loaded_request, supported_escape)
    execution = combine_tabletop_plans(
        loaded_request=loaded_request,
        clearance_request=clearance_request,
        supported_escape=supported_escape,
        task=task,
    )
    return clearance_request, execution
