"""Shared exact-command planning and installation used by hardware workflows."""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from g1_aprilcube_calibration.joint_map import arm_indices
from g1_aprilcube_calibration.urdf_model import URDFModel
from g1_dex3_tabletop.execution_plan import pose_set_from_trajectories
from g1_dex3_tabletop.planning.contracts import PlannedTrajectory, RobotSnapshot


def command_bound_snapshot(state, hands, command_q14) -> RobotSnapshot:
    """Keep measured body/fingers while binding both arms to the active command."""

    command = np.asarray(command_q14, dtype=np.float64).reshape(-1)
    if command.shape != (14,) or not np.all(np.isfinite(command)):
        raise ValueError("command-bound snapshot requires 14 finite arm commands")
    return dual_arm_command_snapshot(
        state,
        hands,
        left_command_q_rad=command[:7],
        right_command_q_rad=command[7:],
    )


def dual_arm_command_snapshot(
    state,
    hands,
    *,
    left_command_q_rad,
    right_command_q_rad,
) -> RobotSnapshot:
    """Bind planning to the exact held commands, not loaded tracking offsets."""

    left = np.asarray(left_command_q_rad, dtype=np.float64).reshape(-1)
    right = np.asarray(right_command_q_rad, dtype=np.float64).reshape(-1)
    if left.shape != (7,) or right.shape != (7,):
        raise ValueError("dual-arm command snapshot requires two seven-joint commands")
    if not np.all(np.isfinite(left)) or not np.all(np.isfinite(right)):
        raise ValueError("dual-arm command snapshot contains NaN or infinity")
    q29 = np.asarray(state.position, dtype=np.float64).copy()
    q29[np.asarray(arm_indices("left"))] = left
    q29[np.asarray(arm_indices("right"))] = right
    return RobotSnapshot(
        measured_q29_rad=tuple(q29),
        left_dex3_q_rad=tuple(hands.left.position),
        right_dex3_q_rad=tuple(hands.right.position),
    )


def install_plan_at_current_boundary(
    synchronized,
    *,
    arm: str,
    trajectories: tuple[PlannedTrajectory, ...],
    recovery_trajectories: tuple[PlannedTrajectory, ...] = (),
    plan_sha256: str,
    validated_reference_state,
    model: URDFModel,
    source: str = "NVlabs/curobo_fixed_two_cube_stack",
) -> tuple[PlannedTrajectory, ...]:
    """Install one immutable arm plan without changing the 14-joint command."""

    if not trajectories:
        raise ValueError("cannot install an empty motion plan")
    current_id = synchronized.current_pose_id
    if current_id is None:
        raise RuntimeError("plan installation requires a named settled boundary")
    command_q14 = np.asarray(synchronized.dual_arm_command_q, dtype=np.float64).reshape(-1)
    if command_q14.shape != (14,) or not np.all(np.isfinite(command_q14)):
        raise RuntimeError("executor returned an invalid dual-arm command")
    active_command = command_q14[:7] if arm == "left" else command_q14[7:]
    start_error = float(
        np.max(np.abs(np.asarray(trajectories[0].command_q_rad[0]) - active_command))
    )
    if start_error > 1.0e-9:
        raise ValueError(f"plan starts away from the active command by {start_error:.9f}rad")
    selected_is_current = synchronized.pose_set.calibration_arm == arm
    executable = trajectories
    if selected_is_current and trajectories[0].from_pose_id != current_id:
        executable = (
            replace(
                trajectories[0],
                from_pose_id=current_id,
                to_pose_id=trajectories[0].to_pose_id,
            ),
            *trajectories[1:],
        )
    boundary_id = current_id if selected_is_current else trajectories[0].from_pose_id
    all_routes = (*executable, *recovery_trajectories)
    command_reference = np.asarray(validated_reference_state.position, dtype=np.float64).copy()
    command_reference[np.asarray(arm_indices("left"))] = command_q14[:7]
    command_reference[np.asarray(arm_indices("right"))] = command_q14[7:]
    pose_set = pose_set_from_trajectories(
        arm=arm,
        trajectories=all_routes,
        reference_full_q=command_reference,
        robot_model=model.name,
        urdf_sha256=model.sha256,
        source=source,
        initial_pose_id=boundary_id if boundary_id != "__handoff__" else None,
        initial_command_q_rad=(
            executable[0].command_q_rad[0] if boundary_id != "__handoff__" else None
        ),
    )
    if selected_is_current and boundary_id == "__handoff__":
        synchronized.install_validated_plan(
            pose_set=pose_set,
            approved_validation_report_sha256=plan_sha256,
            validated_reference_state=validated_reference_state,
            preserve_current_command=True,
        )
    elif selected_is_current:
        synchronized.replace_validated_remaining_plan(
            pose_set=pose_set,
            approved_validation_report_sha256=plan_sha256,
            validated_reference_state=validated_reference_state,
        )
    else:
        synchronized.switch_validated_arm_plan(
            pose_set=pose_set,
            approved_validation_report_sha256=plan_sha256,
            validated_reference_state=validated_reference_state,
            boundary_pose_id=boundary_id,
        )
    return executable
