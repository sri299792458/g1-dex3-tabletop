"""Bounded stage execution for the independent seat-compliance probe."""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from g1_aprilcube_calibration.clock import SystemClock
from g1_aprilcube_calibration.executor_driver import (
    ExecutorControlDriver,
    SynchronizedPoseExecutor,
)
from g1_aprilcube_calibration.executor_state_machine import PoseExecutor
from g1_aprilcube_calibration.joint_map import LEFT_ARM_INDICES, RIGHT_ARM_INDICES
from g1_aprilcube_calibration.pose_schema import HANDOFF_POSE_ID
from g1_aprilcube_calibration.urdf_model import URDFModel
from g1_dex3_tabletop.execution_plan import pose_set_from_trajectories
from g1_dex3_tabletop.hardware_tabletop import _wait_ready
from g1_dex3_tabletop.planning.contracts import PlannedTrajectory


def _stage_trajectory(
    trajectory: PlannedTrajectory,
    *,
    target_id: str | None = None,
) -> PlannedTrajectory:
    """Rebase a separately planned stage onto a fresh ownership handoff."""

    return replace(
        trajectory,
        from_pose_id=HANDOFF_POSE_ID,
        to_pose_id=target_id or trajectory.to_pose_id,
    )


def _stage_executor(
    *,
    arm: str,
    trajectory: PlannedTrajectory | None,
    q29: np.ndarray,
    q14: np.ndarray,
    model: URDFModel,
    transport,
    gravity,
    control_config,
    plan_sha256: str,
    acquire: bool,
    heartbeat,
    rate_hz: float,
):
    trajectories = () if trajectory is None else (trajectory,)
    pose_set = pose_set_from_trajectories(
        arm=arm,
        trajectories=trajectories,
        reference_full_q=q29,
        robot_model=model.name,
        urdf_sha256=model.sha256,
        source="NVlabs/curobo_seat_compliance_stage",
    )
    active = np.asarray(RIGHT_ARM_INDICES if arm == "right" else LEFT_ARM_INDICES)
    opposite = np.asarray(LEFT_ARM_INDICES if arm == "right" else RIGHT_ARM_INDICES)
    raw = PoseExecutor(
        transport=transport,
        clock=SystemClock(),
        pose_set=pose_set,
        handoff_q=q29[active],
        hold_q=q29[opposite],
        approved_validation_report_sha256=plan_sha256,
        config=control_config,
        gravity_feedforward=gravity,
    )
    synchronized = SynchronizedPoseExecutor(raw)
    driver = ExecutorControlDriver(
        synchronized,
        rate_hz=rate_hz,
        safety_heartbeat=heartbeat,
    )
    try:
        if acquire:
            driver.start()
            synchronized.acquire(operator_confirmed=True)
        else:
            synchronized.adopt_owned_control(previous_command_q14=q14)
            driver.start()
        _wait_ready(
            synchronized,
            driver,
            timeout_s=control_config.acquisition_ramp_s + 5.0,
            label=f"{arm} ownership stage",
        )
    except BaseException as error:
        # The caller has not received this driver yet. Stop it before the
        # caller's existing watchdog/Damp cleanup handles the original failure.
        try:
            driver.close()
        except BaseException as cleanup_error:
            raise error from cleanup_error
        raise
    return synchronized, driver


def _execute_stage(
    synchronized,
    driver,
    trajectory: PlannedTrajectory,
    *,
    plan_sha256: str,
    timeout_s: float,
) -> None:
    synchronized.start_trajectory(
        from_pose_id=trajectory.from_pose_id,
        to_pose_id=trajectory.to_pose_id,
        sample_time_s=trajectory.sample_time_s,
        command_q_rad=trajectory.command_q_rad,
        plan_sha256=plan_sha256,
        operator_confirmed=True,
    )
    _wait_ready(
        synchronized,
        driver,
        timeout_s=max(timeout_s, trajectory.sample_time_s[-1] + 5.0),
        label=trajectory.to_pose_id,
    )


def _stop_driver(driver, guard) -> None:
    driver.close()
    driver.check()
    guard.pulse()
