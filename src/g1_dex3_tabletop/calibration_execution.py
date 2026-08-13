"""Execute frozen CuRobo calibration routes through the commissioned controller."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from g1_aprilcube_calibration.executor_state_machine import ExecutorState
from g1_aprilcube_calibration.joint_map import arm_indices
from g1_aprilcube_calibration.models import utc_now_iso
from g1_aprilcube_calibration.pose_schema import (
    HANDOFF_POSE_ID,
    PoseAuditEvent,
    PoseRecord,
    PoseSet,
)
from g1_aprilcube_calibration.session_runner import (
    CaptureSessionRunner,
    FrameSource,
    RecoverableCaptureError,
)
from g1_dex3_tabletop.planning.contracts import (
    CalibrationPlanRequest,
    CalibrationPlanResult,
)


def pose_set_from_curobo_plan(
    request: CalibrationPlanRequest,
    plan: CalibrationPlanResult,
    *,
    urdf_sha256: str,
    robot_model: str = "g1_29dof_with_dex3",
) -> PoseSet:
    """Adapt immutable CuRobo endpoints to the proven controller's pose metadata."""

    if plan.request_sha256 != request.content_sha256:
        raise ValueError("CuRobo plan belongs to a different request")
    timestamp = utc_now_iso()
    indices = np.asarray(arm_indices(plan.arm), dtype=np.int64)
    records: list[PoseRecord] = []
    audit: list[PoseAuditEvent] = []
    for index, planned in enumerate(plan.poses):
        full_q = np.asarray(request.snapshot.measured_q29_rad, dtype=np.float64).copy()
        full_q[indices] = np.asarray(planned.command_q_rad, dtype=np.float64)
        record = PoseRecord(
            id=planned.candidate_id,
            group="curobo_calibration",
            measured_calibration_q=planned.command_q_rad,
            measured_full_q=tuple(full_q),
            calibration_q_spread=(0.0,) * 7,
            recorded_at_utc=timestamp,
            recorded_monotonic_s=float(index),
            source="NVlabs/curobo_frozen_trajectory",
            visual_quality={
                "desired_camera_T_marker": [list(row) for row in planned.camera_T_marker],
                "ik_position_error_m": planned.ik_position_error_m,
                "ik_rotation_error_rad": planned.ik_rotation_error_rad,
            },
        )
        records.append(record)
        audit.append(PoseAuditEvent("add", record.id, timestamp))
    return PoseSet(
        robot_model=robot_model,
        mode_machine=5,
        urdf_sha256=urdf_sha256,
        calibration_arm=plan.arm,
        poses=tuple(records),
        audit_log=tuple(audit),
    )


@dataclass(frozen=True, slots=True)
class CuroboCollectionResult:
    accepted_count: int
    rejected_count: int
    attempted_count: int


class CuroboCalibrationOrchestrator:
    """Traverse every frozen edge once and return to its measured handoff."""

    def __init__(
        self,
        *,
        executor,
        plan: CalibrationPlanResult,
        capture_runner: CaptureSessionRunner,
        frame_source: FrameSource,
        wait_until_ready: Callable[[], None],
        report_progress: Callable[[str, int, int], None] | None = None,
    ) -> None:
        if executor.approved_validation_report_sha256 != plan.content_sha256:
            raise ValueError("executor is not bound to this CuRobo plan")
        if executor.pose_set.calibration_arm != plan.arm:
            raise ValueError("executor and CuRobo plan arms differ")
        if tuple(pose.id for pose in executor.pose_set.poses) != tuple(
            pose.candidate_id for pose in plan.poses
        ):
            raise ValueError("executor endpoint metadata differs from CuRobo plan")
        self.executor = executor
        self.plan = plan
        self.capture_runner = capture_runner
        self.frame_source = frame_source
        self.wait_until_ready = wait_until_ready
        self.report_progress = report_progress or (lambda _message, _accepted, _rejected: None)

    def run(self) -> CuroboCollectionResult:
        accepted = 0
        rejected = 0
        attempted = 0
        if (
            self.executor.state is not ExecutorState.READY
            or self.executor.current_pose_id != HANDOFF_POSE_ID
        ):
            raise RuntimeError("calibration control is not ready at the handoff")
        for trajectory in self.plan.trajectories:
            self.executor.start_trajectory(
                from_pose_id=trajectory.from_pose_id,
                to_pose_id=trajectory.to_pose_id,
                sample_time_s=trajectory.sample_time_s,
                command_q_rad=trajectory.command_q_rad,
                plan_sha256=self.plan.content_sha256,
                operator_confirmed=True,
            )
            self.wait_until_ready()
            if trajectory.to_pose_id == HANDOFF_POSE_ID:
                continue
            attempted += 1
            capture_id = f"capture_{attempted:03d}"
            self.report_progress(f"capturing {trajectory.to_pose_id}", accepted, rejected)
            try:
                frames = self.frame_source.capture_burst(
                    pose_id=trajectory.to_pose_id,
                    capture_id=capture_id,
                )
            except RecoverableCaptureError as error:
                self.capture_runner.reject(
                    capture_id=capture_id,
                    pose_id=trajectory.to_pose_id,
                    reason=str(error),
                )
                rejected += 1
                self.report_progress(
                    f"rejected {trajectory.to_pose_id}: {error}", accepted, rejected
                )
            else:
                self.capture_runner.capture(
                    capture_id=capture_id,
                    pose_id=trajectory.to_pose_id,
                    frames=frames,
                )
                accepted += 1
                self.report_progress(f"accepted {trajectory.to_pose_id}", accepted, rejected)
        if self.executor.current_pose_id != HANDOFF_POSE_ID:
            raise RuntimeError("CuRobo route did not terminate at handoff")
        return CuroboCollectionResult(accepted, rejected, attempted)
