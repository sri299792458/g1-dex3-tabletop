"""Pure orchestration helpers for the cube-only tabletop lifecycle."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from g1_aprilcube_calibration.calibration_bundle import CalibrationBundle
from g1_aprilcube_calibration.joint_map import arm_indices, validate_arm_side
from g1_dex3_tabletop.planning.contracts import RobotSnapshot
from g1_dex3_tabletop.tabletop_contracts import (
    SupportedEscapePlan,
    TabletopExecutionPlan,
    TabletopFixture,
    TabletopObservation,
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
    return document


def build_tabletop_request(
    *,
    arm: str,
    observation: TabletopObservation,
    calibration_bundle: CalibrationBundle,
    calibration_bundle_path: str | Path,
    grasp_shortlist_path: str | Path,
    task_config_path: str | Path,
    presentation_id: str = "direct",
    fixture: TabletopFixture | None = None,
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
        object_dimensions_m=tuple(task["object"]["dimensions_m"]),
        open_transit_table_patch_dimensions_m=tuple(
            task["table"]["open_transit_patch_dimensions_m"]
        ),
        supported_escape_m=float(task["motion"]["supported_escape_m"]),
        retention_test_lift_m=float(task["motion"]["retention_test_lift_m"]),
        lift_m=float(task["motion"]["payload_lift_m"]),
        maximum_arm_velocity_rad_s=float(task["motion"]["maximum_arm_velocity_rad_s"]),
    )


def request_at_clearance(
    loaded_request: TabletopTaskRequest,
    escape: SupportedEscapePlan,
) -> TabletopTaskRequest:
    """Use the exact supported-escape endpoint as the task planning state."""

    if escape.request_sha256 != loaded_request.content_sha256:
        raise ValueError("supported escape belongs to a different loaded request")
    q29 = np.asarray(loaded_request.observation.snapshot.measured_q29_rad).copy()
    q29[np.asarray(arm_indices(loaded_request.arm))] = np.asarray(
        escape.outbound.command_q_rad[-1]
    )
    snapshot = RobotSnapshot(
        measured_q29_rad=tuple(q29),
        left_dex3_q_rad=loaded_request.observation.snapshot.left_dex3_q_rad,
        right_dex3_q_rad=loaded_request.observation.snapshot.right_dex3_q_rad,
    )
    source = loaded_request.observation
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
        object_dimensions_m=loaded_request.object_dimensions_m,
        open_transit_table_patch_dimensions_m=(
            loaded_request.open_transit_table_patch_dimensions_m
        ),
        supported_escape_m=loaded_request.supported_escape_m,
        retention_test_lift_m=loaded_request.retention_test_lift_m,
        lift_m=loaded_request.lift_m,
        maximum_arm_velocity_rad_s=loaded_request.maximum_arm_velocity_rad_s,
        random_seed=loaded_request.random_seed,
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
