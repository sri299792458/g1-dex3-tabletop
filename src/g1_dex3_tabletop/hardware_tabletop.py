"""Single-approval seated cube pick/lift/replace hardware workflow.

Importing this module is inert. DDS publishers are constructed only inside
``run_tabletop`` after the explicit operator approval.
"""

from __future__ import annotations

import hashlib
import json
import select
import subprocess
import sys
import termios
import time
import tty
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
from aprilcube import CorrespondenceDetector

from g1_aprilcube_calibration.activation_handoff import build_activation_handoff
from g1_aprilcube_calibration.calibration_bundle import CalibrationBundle
from g1_aprilcube_calibration.clock import SystemClock
from g1_aprilcube_calibration.config import QualityThresholds
from g1_aprilcube_calibration.executor_driver import (
    ExecutorControlDriver,
    SynchronizedPoseExecutor,
)
from g1_aprilcube_calibration.executor_state_machine import ExecutorState, PoseExecutor
from g1_aprilcube_calibration.joint_map import (
    arm_indices,
    arm_joint_names,
    validate_arm_side,
)
from g1_aprilcube_calibration.pose_schema import PoseSet
from g1_aprilcube_calibration.process_lock import CommandOwnerLock
from g1_aprilcube_calibration.readiness import StateSampleBuffer
from g1_aprilcube_calibration.ros.camera_adapter import ROSCameraSubscriber, ROSImageFrame
from g1_aprilcube_calibration.transforms import invert_transform
from g1_aprilcube_calibration.transports.unitree_arm_sdk import (
    UnitreeLowStateObserver,
    UnitreeTorsoIMUObserver,
)
from g1_aprilcube_calibration.transports.unitree_debug_lowcmd import (
    UnitreeDebugLowCmdTransport,
)
from g1_aprilcube_calibration.transports.unitree_dex3 import (
    Dex3GraspNotAcquiredError,
    Dex3RetentionLostError,
    UnitreeDex3PostureController,
    UnitreeDex3StateObserver,
)
from g1_aprilcube_calibration.urdf_model import URDFModel
from g1_dex3_tabletop.calibration_candidates import camera_info_from_hardware
from g1_dex3_tabletop.camera_state_sync import CameraStateInputBuffer
from g1_dex3_tabletop.execution_plan import pose_set_from_trajectories
from g1_dex3_tabletop.hardware_config import (
    debug_lowcmd_config,
    dex3_config,
    executor_config,
    gravity_feedforward,
    load_hardware,
    recording_configs,
    resolve_hardware_path,
    transport_config,
    watchdog,
)
from g1_dex3_tabletop.mpc_command_buffer import MPCCommandWindow, MPCHandoffBoundary
from g1_dex3_tabletop.persistent_planner import (
    PersistentTabletopPlanner,
    PlannerRequestRejected,
)
from g1_dex3_tabletop.planning.contracts import (
    PlannedTrajectory,
    RobotSnapshot,
    atomic_write_json,
)
from g1_dex3_tabletop.planning.dex3_handedness import (
    dex3_empty_close_reference,
    dex3_execution_profile,
)
from g1_dex3_tabletop.planning.tabletop_mpc import (
    MPC_HANDOFF_INTERVAL_S,
    MPC_KNOT_DT_S,
)
from g1_dex3_tabletop.raw_episode_recording import RawEpisodeRecorder, tabletop_raw_topics
from g1_dex3_tabletop.stack_workflow import snapshot_base_T_camera
from g1_dex3_tabletop.state_estimation import (
    AnchoredCameraPoseEstimators,
    AnchoredCameraStateEstimator,
    CameraPoseAnchor,
    pose_error,
)
from g1_dex3_tabletop.tabletop_contracts import (
    MovingGraspContinuationRequest,
    PregraspRemainingPlan,
    RetentionRouteValidationRequest,
    RetentionRouteValidationResult,
    SupportedEscapePlan,
    TabletopExecutionPlan,
    TabletopPregraspPlan,
    TabletopTaskPlan,
    build_pregrasp_remaining_plan,
    pregrasp_to_clearance_return,
)
from g1_dex3_tabletop.tabletop_object import load_tabletop_object_profile
from g1_dex3_tabletop.tabletop_perception import (
    camera_motion_from_fixed_cube,
    observe_live_cube_frame,
    observe_resting_cube,
)
from g1_dex3_tabletop.tabletop_presentation import (
    DIRECT_PRESENTATION_ID,
    load_tabletop_presentation,
)
from g1_dex3_tabletop.tabletop_workflow import (
    build_tabletop_request,
    load_task_config,
    request_at_clearance,
    request_at_clearance_observation,
    request_at_estimated_pregrasp,
)

ROOT = Path(__file__).resolve().parents[2]
MOTION_ACK = "I CONFIRM THE G1 IS SECURED BY THE LOAD-BEARING HARNESS AND THE WORKSPACE IS CLEAR"


class TabletopTaskRejected(RuntimeError):
    """The controller is healthy, but this pick attempt should return and stop."""


class MPCInitialWindowUnavailable(RuntimeError):
    """MPC never started, so the frozen pregrasp reverse remains valid."""


def _run_id() -> str:
    return datetime.now(timezone.utc).strftime("tabletop_%Y%m%dT%H%M%SZ")


def _snapshot(state, hands) -> RobotSnapshot:
    return RobotSnapshot(
        measured_q29_rad=tuple(state.position),
        left_dex3_q_rad=tuple(hands.left.position),
        right_dex3_q_rad=tuple(hands.right.position),
    )


def _wait_for_state(observer, timeout_s: float = 5.0):
    deadline = time.monotonic() + timeout_s
    last = "no sample"
    while time.monotonic() < deadline:
        try:
            return observer.observe()
        except RuntimeError as error:
            last = str(error)
        time.sleep(0.02)
    raise RuntimeError("timed out waiting for rt/lowstate: " + last)


def _wait_for_hands(observer, timeout_s: float = 5.0):
    deadline = time.monotonic() + timeout_s
    last = "no sample"
    while time.monotonic() < deadline:
        try:
            return observer.observe()
        except RuntimeError as error:
            last = str(error)
        time.sleep(0.02)
    raise RuntimeError("timed out waiting for Dex3 state: " + last)


def _wait_for_torso_imu(observer, timeout_s: float = 5.0):
    deadline = time.monotonic() + timeout_s
    last = "no sample"
    while time.monotonic() < deadline:
        try:
            return observer.observe()
        except RuntimeError as error:
            last = str(error)
        time.sleep(0.02)
    raise RuntimeError("timed out waiting for rt/secondary_imu: " + last)


def _wait_for_camera_state_input(
    buffer: CameraStateInputBuffer,
    *,
    target_monotonic_s: float | None,
    maximum_age_s: float,
    maximum_gap_s: float,
    control_check=None,
    timeout_s: float = 1.0,
):
    deadline = time.monotonic() + timeout_s
    last = "camera-state inputs are incomplete"
    while time.monotonic() < deadline:
        if control_check is not None:
            control_check()
        try:
            if target_monotonic_s is None:
                return buffer.latest(
                    now_monotonic_s=time.monotonic(),
                    maximum_age_s=maximum_age_s,
                    maximum_gap_s=maximum_gap_s,
                )
            return buffer.sample_at(
                target_monotonic_s,
                maximum_gap_s=maximum_gap_s,
            )
        except RuntimeError as error:
            last = str(error)
        time.sleep(0.005)
    raise RuntimeError("timed out pairing camera-state inputs: " + last)


def _mpc_camera_state_record(
    synchronized_input,
    estimate,
    state,
    *,
    maximum_time_difference_s: float,
) -> dict:
    """Bind one estimator result to the arm state used by the same MPC solve."""

    estimate_monotonic_s = estimate.timestamp_ns / 1.0e9
    arm_state_monotonic_s = float(state.receipt_monotonic_s)
    estimate_to_arm_state_s = arm_state_monotonic_s - estimate_monotonic_s
    if abs(estimate_to_arm_state_s) > maximum_time_difference_s:
        raise RuntimeError(
            "MPC camera/body estimate and arm state differ by "
            f"{estimate_to_arm_state_s:+.6f}s; limit is "
            f"{maximum_time_difference_s:.6f}s"
        )
    return {
        **estimate.to_dict(),
        "input": synchronized_input.to_dict(),
        "arm_state_monotonic_s": arm_state_monotonic_s,
        "estimate_to_arm_state_s": estimate_to_arm_state_s,
        # The window depends on both measurements, so its freshness clock is
        # deliberately the older of the two rather than only the arm sample.
        "source_monotonic_s": min(arm_state_monotonic_s, estimate_monotonic_s),
    }


def _wait_for_activation(observer, states, pose_set, recording, timeout_s: float = 5.0):
    deadline = time.monotonic() + timeout_s
    last = "no stationary window"
    while time.monotonic() < deadline:
        try:
            observer.observe()
            return build_activation_handoff(
                pose_set,
                states.snapshot(),
                now_monotonic_s=time.monotonic(),
                config=recording,
            )
        except (RuntimeError, ValueError) as error:
            last = str(error)
        time.sleep(0.01)
    raise RuntimeError("timed out waiting for stationary supported handoff: " + last)


def _collect_frames(
    rclpy,
    node,
    camera: ROSCameraSubscriber,
    *,
    count: int,
    timeout_s: float,
    control_check=None,
) -> tuple[ROSImageFrame, ...]:
    baseline = {
        (item.timing.receipt_monotonic_s, item.timing.header_stamp_ns)
        for item in camera.frames.snapshot()
    }
    selected: list[ROSImageFrame] = []
    deadline = time.monotonic() + timeout_s
    while len(selected) < count and time.monotonic() < deadline:
        if control_check is not None:
            control_check()
        rclpy.spin_once(node, timeout_sec=0.01)
        for frame in camera.frames.snapshot():
            key = (frame.timing.receipt_monotonic_s, frame.timing.header_stamp_ns)
            if key in baseline:
                continue
            baseline.add(key)
            selected.append(frame)
            if len(selected) == count:
                break
    if len(selected) != count:
        detail = f"; last camera error: {camera.last_error}" if camera.last_error else ""
        raise RuntimeError(f"received only {len(selected)}/{count} new camera frames{detail}")
    if selected[-1].timing.receipt_monotonic_s - selected[0].timing.receipt_monotonic_s > 2.0:
        raise RuntimeError("camera observation burst exceeded 2.0 seconds")
    return tuple(selected)


def _wait_for_space_with_preview(
    rclpy,
    node,
    camera,
    *,
    arm: str,
    no_window: bool,
    prompt: str | None = None,
    overlay: str = "READ-ONLY - SPACE starts complete task",
    control_check=None,
) -> None:
    if not sys.stdin.isatty():
        raise RuntimeError("operator approval requires an interactive terminal")
    message = prompt or (
        "START — G1 seated; both arms supported and still; AprilCube resting "
        f"upright and visible; complete {arm}-arm sweep clear. Press SPACE once: "
    )
    print(message, end="", flush=True)
    descriptor = sys.stdin.fileno()
    previous = termios.tcgetattr(descriptor)
    try:
        tty.setraw(descriptor)
        while True:
            if control_check is not None:
                control_check()
            rclpy.spin_once(node, timeout_sec=0.01)
            if not no_window:
                frame = camera.frames.latest
                if frame is not None:
                    rendered = frame.image_bgr.copy()
                    cv2.putText(
                        rendered,
                        overlay,
                        (24, 42),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.9,
                        (0, 220, 255),
                        2,
                    )
                    cv2.imshow("G1 tabletop", rendered)
                    cv2.waitKey(1)
            ready, _, _ = select.select([sys.stdin], [], [], 0.0)
            if ready:
                key = sys.stdin.read(1)
                if key == " ":
                    print("SPACE", flush=True)
                    return
                if key == "\x03":
                    raise KeyboardInterrupt
    finally:
        termios.tcsetattr(descriptor, termios.TCSADRAIN, previous)


def _wait_ready(executor, driver, *, timeout_s: float, label: str) -> None:
    deadline = time.monotonic() + timeout_s
    next_report = time.monotonic() + 1.0
    while time.monotonic() < deadline:
        driver.check()
        if executor.state is ExecutorState.READY:
            return
        if executor.state is ExecutorState.STOPPED:
            raise RuntimeError(f"controller stopped while waiting for {label}")
        if time.monotonic() >= next_report:
            print(executor.motion_diagnostic(prefix=f"motion status for {label}"), flush=True)
            next_report += 1.0
        time.sleep(0.01)
    raise RuntimeError(f"timed out waiting for {label}")


def _invoke_planner(command: str, request_path: Path, output_path: Path, driver=None) -> None:
    """Run a one-shot worker for the separate calibration workflow."""

    worker = ROOT / ".venv-planner/bin/g1-curobo-worker"
    if not worker.is_file():
        raise FileNotFoundError("planner environment missing; run ./tools/setup_planner_env.sh")
    log_path = output_path.with_suffix(".planner.log")
    process = subprocess.Popen(
        [str(worker), command, "--request", str(request_path), "--output", str(output_path)],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    tail: list[str] = []
    try:
        with log_path.open("w", encoding="utf-8") as log:
            assert process.stdout is not None
            for line in process.stdout:
                print(line, end="", flush=True)
                log.write(line)
                log.flush()
                if line.strip():
                    tail.append(line.strip())
                    tail = tail[-3:]
        returncode = process.wait()
    except BaseException:
        process.terminate()
        try:
            process.wait(timeout=2.0)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
        raise
    if driver is not None:
        driver.check()
    if returncode != 0:
        detail = tail[-1] if tail else "no diagnostic output"
        raise RuntimeError(
            f"CuRobo worker {command} exited with {returncode}: {detail}; "
            f"full planner log: {log_path}"
        )


def _finger_heartbeat(watchdog_value, controller):
    def maintain() -> None:
        watchdog_value.pulse()
        controller.maintain_active_posture()

    return maintain


def _command_fingers(
    controller,
    driver,
    watchdog_value,
    *,
    left,
    right,
    left_acceptance=None,
    right_acceptance=None,
    label: str,
):
    driver.safety_heartbeat = watchdog_value.pulse

    def check() -> None:
        driver.check()
        watchdog_value.pulse()

    try:
        return controller.command_posture(
            left_target_q_rad=left,
            right_target_q_rad=right,
            left_acceptance_q_rad=left_acceptance,
            right_acceptance_q_rad=right_acceptance,
            label=label,
            safety_heartbeat=check,
        )
    finally:
        driver.safety_heartbeat = _finger_heartbeat(watchdog_value, controller)


def _command_retention_test_close(
    controller,
    driver,
    watchdog_value,
    *,
    active_side: str,
    left,
    right,
    empty_close_reference_q_rad,
    minimum_opposed_shortfall_rad: float,
    label: str,
):
    driver.safety_heartbeat = watchdog_value.pulse

    def check() -> None:
        driver.check()
        watchdog_value.pulse()

    try:
        return controller.command_close_for_retention_test(
            active_side=active_side,
            left_target_q_rad=left,
            right_target_q_rad=right,
            empty_close_reference_q_rad=empty_close_reference_q_rad,
            minimum_opposed_shortfall_rad=minimum_opposed_shortfall_rad,
            label=label,
            safety_heartbeat=check,
        )
    finally:
        driver.safety_heartbeat = _finger_heartbeat(watchdog_value, controller)


def _trajectory_maps(execution: TabletopExecutionPlan):
    normal = {trajectory.to_pose_id: trajectory for trajectory in execution.trajectories}
    recovery = {
        (trajectory.from_pose_id, trajectory.to_pose_id): trajectory
        for trajectory in execution.recovery_trajectories
    }
    if len(normal) != len(execution.trajectories):
        raise ValueError("tabletop execution plan has duplicate phase names")
    return normal, recovery


def _return_to_clearance_phases(normal_routes) -> tuple[str, ...]:
    """Return the exact post-retreat phase sequence installed by the planner."""

    if "return_to_clearance" not in normal_routes:
        raise ValueError("tabletop execution plan has no return-to-clearance phase")
    if "return_to_pregrasp" in normal_routes:
        return "return_to_pregrasp", "return_to_clearance"
    return ("return_to_clearance",)


def _execute_trajectory(
    synchronized,
    driver,
    trajectory,
    *,
    plan_sha256: str,
    control_config,
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
        timeout_s=max(control_config.motion_timeout_s, trajectory.sample_time_s[-1] + 5),
        label=trajectory.to_pose_id,
    )
    print(f"completed phase: {trajectory.to_pose_id}", flush=True)


def _executed_mpc_approach(
    windows: list[dict],
    *,
    arm: str,
    joint_position_offsets_rad: dict[str, float],
) -> PlannedTrajectory:
    """Stitch exactly the portions of accepted windows exposed to control."""

    accepted = [MPCCommandWindow.from_dict(value) for value in windows]
    if not accepted or not accepted[-1].terminal:
        raise ValueError("completed moving-target approach has no terminal MPC window")
    absolute_times: list[float] = []
    commands: list[np.ndarray] = []
    predicted: list[np.ndarray] = []
    for index, window in enumerate(accepted):
        end = (
            accepted[index + 1].valid_from_monotonic_s
            if index + 1 < len(accepted)
            else window.expiration_monotonic_s
        )
        for relative_time, command, predicted_q in zip(
            window.sample_time_s,
            window.command_q_rad,
            window.predicted_q_rad,
            strict=True,
        ):
            absolute_time = window.valid_from_monotonic_s + relative_time
            if absolute_time > end + 1.0e-9:
                continue
            command_array = np.asarray(command, dtype=np.float64)
            predicted_array = np.asarray(predicted_q, dtype=np.float64)
            if absolute_times and np.isclose(
                absolute_time,
                absolute_times[-1],
                atol=1.0e-9,
                rtol=0.0,
            ):
                if float(np.max(np.abs(command_array - commands[-1]))) > 1.0e-8:
                    raise ValueError("successive MPC windows disagree at their splice")
                if float(np.max(np.abs(predicted_array - predicted[-1]))) > 1.0e-8:
                    raise ValueError("successive MPC predictions disagree at their splice")
                continue
            if absolute_times and absolute_time < absolute_times[-1]:
                raise ValueError("accepted MPC windows are not time ordered")
            absolute_times.append(absolute_time)
            commands.append(command_array)
            predicted.append(predicted_array)
    if len(absolute_times) < 2:
        raise ValueError("executed MPC approach contains fewer than two samples")
    times = np.asarray(absolute_times, dtype=np.float64) - absolute_times[0]
    command_q = np.asarray(commands, dtype=np.float64)
    predicted_q = np.asarray(predicted, dtype=np.float64)
    offsets = np.asarray(
        [joint_position_offsets_rad.get(name, 0.0) for name in arm_joint_names(arm)],
        dtype=np.float64,
    )
    return PlannedTrajectory(
        from_pose_id="move_to_pregrasp",
        to_pose_id="grasp_approach",
        sample_time_s=tuple(times),
        command_q_rad=tuple(tuple(float(value) for value in row) for row in command_q),
        model_q_rad=tuple(
            tuple(float(value) for value in row) for row in predicted_q + offsets[None, :]
        ),
        planning_time_s=float(sum(window.solve_time_s for window in accepted)),
    )


def _trajectory_with_endpoints(
    trajectory: PlannedTrajectory,
    *,
    from_pose_id: str,
    to_pose_id: str,
) -> PlannedTrajectory:
    """Rename a frozen route without changing any command or timing sample."""

    return PlannedTrajectory(
        from_pose_id=from_pose_id,
        to_pose_id=to_pose_id,
        sample_time_s=trajectory.sample_time_s,
        command_q_rad=trajectory.command_q_rad,
        model_q_rad=trajectory.model_q_rad,
        planning_time_s=trajectory.planning_time_s,
    )


def _mpc_window_rejection_reason(window: MPCCommandWindow) -> str:
    """Name the primary certified-window rejection without dumping its arrays."""

    diagnostics = window.diagnostics
    cspace = diagnostics.get("curobo_cspace_bound_diagnostics", {})
    velocity = cspace.get("velocity", {}) if isinstance(cspace, dict) else {}
    if float(velocity.get("maximum_violation", 0.0)) > 0.0:
        return (
            f"{velocity.get('joint', 'unknown joint')} velocity "
            f"{float(velocity.get('value', float('nan'))):.4f}rad/s is outside "
            f"[{float(velocity.get('active_lower_bound', float('nan'))):.4f}, "
            f"{float(velocity.get('active_upper_bound', float('nan'))):.4f}]rad/s; "
            "validated command peak="
            f"{float(diagnostics.get('peak_velocity_rad_s', float('nan'))):.4f}rad/s"
        )
    strict_failure = diagnostics.get("strict_failure")
    if strict_failure is not None:
        links = diagnostics.get("strict_failure_links")
        return f"strict {strict_failure}: links={links}"
    if not bool(diagnostics.get("curobo_feasible", True)):
        constraints = diagnostics.get("curobo_constraints")
        return f"CuRobo constraints were infeasible: {constraints}"
    return "certified MPC window was infeasible"


def _execute_mpc_phase(
    synchronized,
    driver,
    planner,
    *,
    arm: str,
    trajectory,
    plan_sha256: str,
    control_config,
    phase_record: dict | None = None,
    moving_target_provider=None,
    reference_T_camera0=None,
    prepared_mpc: dict | None = None,
) -> tuple[dict, list[dict]]:
    """Execute the local grasp approach through visual Cartesian MPC."""

    phase = trajectory.to_pose_id
    if prepared_mpc is None:
        try:
            preparation_event = planner.request_payload(
                "prepare-moving-grasp-mpc",
                payload={
                    "reference_T_camera0": np.asarray(
                        reference_T_camera0, dtype=np.float64
                    ).tolist(),
                },
                control_check=driver.check,
                timeout_s=30.0,
            )
        except (PlannerRequestRejected, RuntimeError) as error:
            driver.check()
            raise RuntimeError(f"CuRobo MPC preparation failed for {phase}: {error}") from error
        preparation = dict(preparation_event["payload"])
    else:
        preparation = dict(prepared_mpc)
        if preparation.get("phase") != phase:
            raise ValueError("prebuilt CuRobo MPC model belongs to another phase")
    if phase_record is not None:
        phase_record["preparation"] = preparation
    if preparation["reused_warm_model"]:
        preparation_detail = (
            "reconfigured the one warmed solver in "
            f"{1000.0 * preparation['reconfiguration_time_s']:.1f}ms; "
            "no solver construction or CUDA graph rebuild"
        )
    else:
        preparation_detail = (
            f"built {preparation['physical_mode']} collision model in "
            f"{preparation['preparation_time_s']:.3f}s "
            f"(construction={preparation['build_time_s']:.3f}s, "
            f"CUDA setup={preparation['setup_time_s']:.3f}s)"
        )
    print(
        f"CUROBO MPC {phase} READY — {preparation_detail}; fixed-rate robot "
        "control remained active",
        flush=True,
    )
    # The target must be observed after any cold CUDA setup. Capturing it
    # before a multi-second build would violate the unchanged source-age gate
    # on the very first moving-target window.
    pending_moving_target = None if moving_target_provider is None else moving_target_provider()

    handoff_lead_s = MPC_HANDOFF_INTERVAL_S

    def request_window() -> tuple[MPCCommandWindow, MPCHandoffBoundary]:
        nonlocal pending_moving_target
        moving_target = None
        if moving_target_provider is not None:
            if pending_moving_target is None:
                synchronized_input, estimate, visual_target = moving_target_provider()
            else:
                synchronized_input, estimate, visual_target = pending_moving_target
                pending_moving_target = None
        state = synchronized.observe_state()
        state_source_monotonic_s = float(state.receipt_monotonic_s)
        if moving_target_provider is not None:
            camera_state = _mpc_camera_state_record(
                synchronized_input,
                estimate,
                state,
                maximum_time_difference_s=(control_config.state_freshness_timeout_s),
            )
            moving_target = {
                **camera_state,
                **visual_target,
                "source_monotonic_s": min(
                    float(camera_state["source_monotonic_s"]),
                    float(visual_target["source_monotonic_s"]),
                ),
            }
            state_source_monotonic_s = moving_target["source_monotonic_s"]
        # Freeze the splice only after acquiring the observations needed for
        # this solve. The complete handoff lead is then available to the CUDA
        # worker, rather than being consumed by camera/state collection.
        boundary = synchronized.prepare_streaming_handoff(
            minimum_lead_s=handoff_lead_s,
            handoff_quantum_s=MPC_KNOT_DT_S,
        )
        # The returned trajectory is valid only at this frozen future splice.
        # Stop waiting before that instant, leaving two controller ticks for
        # IPC and installation.  The active certified horizon continues in
        # the executor throughout this wait.
        request_timeout_s = (
            boundary.valid_from_monotonic_s
            - time.monotonic()
            - 2.0 * control_config.nominal_tick_period_s
        )
        if request_timeout_s <= 0.0:
            raise RuntimeError("CuRobo MPC handoff deadline passed before the solve was submitted")
        event = planner.request_payload(
            "step-moving-grasp-mpc",
            payload={
                "phase": phase,
                "handoff_predicted_q_rad": list(boundary.predicted_q_rad),
                "handoff_predicted_dq_rad_s": list(boundary.predicted_dq_rad_s),
                "handoff_predicted_ddq_rad_s2": list(boundary.predicted_ddq_rad_s2),
                "handoff_command_q_rad": list(boundary.command_q_rad),
                "source_state_monotonic_s": state_source_monotonic_s,
                "valid_from_monotonic_s": boundary.valid_from_monotonic_s,
                "predecessor_sha256": boundary.predecessor_sha256,
                "committed_route_progress_index": (boundary.committed_route_progress_index),
                "moving_target": moving_target,
            },
            control_check=driver.check,
            timeout_s=request_timeout_s,
        )
        window = MPCCommandWindow.from_dict(event["payload"])
        if moving_target is not None:
            if window.diagnostics.get("moving_target") != moving_target:
                raise RuntimeError(
                    "CuRobo MPC window did not preserve its moving-target observation"
                )
            if phase_record is not None:
                phase_record["moving_targets"].append(moving_target)
        return window, boundary

    windows: list[dict] = []
    initial_deadline = time.monotonic() + control_config.motion_timeout_s
    initial_rejections = 0
    last_rejection = "no window was returned"
    while True:
        try:
            first, _first_boundary = request_window()
        except (PlannerRequestRejected, RuntimeError, ValueError) as error:
            driver.check()
            last_rejection = str(error)
        else:
            if first.feasible:
                break
            if phase_record is not None:
                phase_record.setdefault("rejected_windows", []).append(first.to_dict())
            last_rejection = _mpc_window_rejection_reason(first)
        initial_rejections += 1
        if time.monotonic() >= initial_deadline:
            raise MPCInitialWindowUnavailable(
                f"CuRobo MPC produced no feasible initial window for {phase} within "
                f"{control_config.motion_timeout_s:.2f}s after {initial_rejections} "
                f"rejections; last rejection: {last_rejection}; no streaming motion "
                "was started"
            )
        print(
            f"CuRobo MPC rejected initial window {initial_rejections}: "
            f"{last_rejection}; holding the stationary pregrasp and retrying",
            flush=True,
        )
    try:
        accepted = synchronized.start_streaming_trajectory(
            from_pose_id=trajectory.from_pose_id,
            to_pose_id=trajectory.to_pose_id,
            window=first,
            plan_sha256=plan_sha256,
            operator_confirmed=True,
        )
    except BaseException:
        if phase_record is not None:
            phase_record["rejected_window"] = first.to_dict()
        raise
    windows.append(accepted.to_dict())
    if phase_record is not None:
        phase_record["windows"].append(accepted.to_dict())

    def horizon_state() -> str:
        remaining_s = float(synchronized.streaming_trajectory_status()["remaining_s"])
        if remaining_s > 0.0:
            return f"the active certified horizon has {remaining_s:.3f}s remaining"
        return "the controller is holding its certified stationary endpoint"

    while synchronized.state is ExecutorState.MOVING:
        driver.check()
        status = synchronized.streaming_trajectory_status()
        if bool(status["terminal"]):
            break
        if (
            not bool(status["active"])
            or bool(status["queued"])
            or bool(status["terminal_pending"])
        ):
            time.sleep(0.01)
            continue
        try:
            window, boundary = request_window()
        except (PlannerRequestRejected, RuntimeError, ValueError) as error:
            # Confirm this was not a controller/transport failure before
            # treating it as an unavailable planner update.
            driver.check()
            if phase_record is not None:
                phase_record.setdefault("rejected_attempts", []).append(
                    {"error_type": type(error).__name__, "error": str(error)}
                )
            print(
                "CuRobo MPC update produced no certifiable replacement: "
                f"{error}; {horizon_state()}; retrying from a fresh "
                "cube/body observation",
                flush=True,
            )
            continue
        if not window.feasible:
            if phase_record is not None:
                phase_record.setdefault("rejected_windows", []).append(window.to_dict())
            print(
                "CuRobo MPC rejected replacement window "
                f"{window.generation}: {_mpc_window_rejection_reason(window)}; "
                f"{horizon_state()}; retrying from a fresh cube/body observation",
                flush=True,
            )
            continue
        try:
            accepted = synchronized.update_streaming_trajectory(
                window=window,
                handoff_boundary=boundary,
            )
        except (TypeError, ValueError, RuntimeError) as error:
            driver.check()
            if phase_record is not None:
                phase_record.setdefault("rejected_windows", []).append(window.to_dict())
                phase_record.setdefault("rejected_attempts", []).append(
                    {"error_type": type(error).__name__, "error": str(error)}
                )
            print(
                "CuRobo MPC replacement missed executor certification: "
                f"{error}; {horizon_state()}; retrying from a fresh "
                "cube/body observation",
                flush=True,
            )
            continue
        windows.append(accepted.to_dict())
        if phase_record is not None:
            phase_record["windows"].append(accepted.to_dict())
        if accepted.generation % 10 == 0:
            print(
                "CuRobo MPC progress: "
                f"phase={phase}, window={accepted.generation}, "
                f"solve={accepted.solve_time_s:.3f}s, "
                f"remaining={accepted.duration_s:.3f}s",
                flush=True,
            )
    _wait_ready(
        synchronized,
        driver,
        timeout_s=control_config.motion_timeout_s,
        label=f"{phase} MPC terminal settle",
    )
    print(
        f"completed phase: {phase} through {len(windows)} validated MPC windows",
        flush=True,
    )
    if phase_record is not None:
        phase_record["completed"] = True
    return preparation, windows


def _restore_seated_control(*, driver, dex_controller, guard, synchronized) -> None:
    """Cleanly return a healthy held controller to Unitree seated FSM 3."""

    if driver.is_alive:
        driver.close()
    driver.check()
    if not dex_controller.timed_out:
        dex_controller.timeout()
    guard.restore_seated()
    synchronized.confirm_external_takeover("PC2 verified AI FSM 0 -> 1 -> seated FSM 3")


def _teardown_ros_runtime(
    *,
    no_window: bool,
    camera,
    node,
    rclpy_module,
    transport,
    synchronized,
) -> None:
    """Tear ROS down only after direct robot ownership has ended."""

    if (
        transport is not None
        and transport.requires_external_takeover
        and (synchronized is None or synchronized.state is not ExecutorState.STOPPED)
    ):
        raise RuntimeError("refusing ROS teardown before verified external robot-control takeover")
    if not no_window:
        cv2.destroyAllWindows()
    if camera is not None:
        camera.close()
    if node is not None:
        node.destroy_node()
    if rclpy_module is not None and rclpy_module.ok():
        rclpy_module.shutdown()


def _save_frames(directory: Path, frames: tuple[ROSImageFrame, ...]) -> None:
    directory.mkdir(parents=True, exist_ok=False)
    manifest = []
    for index, frame in enumerate(frames):
        path = directory / f"frame_{index:03d}.png"
        if not cv2.imwrite(str(path), frame.image_bgr):
            raise RuntimeError(f"failed to write {path}")
        manifest.append(
            {
                "path": path.name,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "timing": {
                    "receipt_monotonic_s": frame.timing.receipt_monotonic_s,
                    "receipt_utc": frame.timing.receipt_utc,
                    "header_stamp_ns": frame.timing.header_stamp_ns,
                },
            }
        )
    atomic_write_json(directory / "manifest.json", {"frames": manifest})


def _resolve_task_velocity(
    configured_velocity_rad_s: float,
    requested_velocity_rad_s: float | None,
    controller_ceiling_rad_s: float,
) -> float:
    """Resolve one explicit run limit without weakening the hardware ceiling."""

    selected = (
        float(configured_velocity_rad_s)
        if requested_velocity_rad_s is None
        else float(requested_velocity_rad_s)
    )
    if not np.isfinite(selected) or selected <= 0.0:
        raise ValueError("tabletop arm velocity must be positive and finite")
    if selected > float(controller_ceiling_rad_s):
        raise ValueError(
            f"tabletop arm velocity {selected:.4f}rad/s exceeds the commissioned "
            f"controller ceiling {float(controller_ceiling_rad_s):.4f}rad/s"
        )
    return selected


def run_tabletop(args) -> int:
    """Run one complete seated selected-Dex3 task after a single SPACE."""

    if args.confirm != MOTION_ACK:
        raise ValueError(f"--confirm must equal exactly: {MOTION_ACK}")
    arm = validate_arm_side(args.arm)
    empty_close_reference_q_rad, minimum_opposed_shortfall_rad = dex3_empty_close_reference(arm)
    active_open_target_q_rad, _active_close_target_q_rad = dex3_execution_profile(arm)
    object_profile = load_tabletop_object_profile(args.object_profile)
    presentation = load_tabletop_presentation(
        args.presentation,
        direct_object_profile_id=(
            object_profile.profile_id if args.presentation == DIRECT_PRESENTATION_ID else None
        ),
        direct_shortlist_override=(
            object_profile.direct_grasp_shortlist_path
            if args.presentation == DIRECT_PRESENTATION_ID
            else None
        ),
    )
    presentation.require_arm(arm)
    presentation.require_object_profile(object_profile.profile_id)
    grasp_shortlist_path = presentation.grasp_shortlist_for(object_profile.profile_id)
    hardware = load_hardware(args.hardware_config)
    configured_arms = {
        str(hardware["robot"]["calibration_arm"]),
        str(hardware["control"]["calibration_arm"]),
    }
    if configured_arms != {arm}:
        raise ValueError(
            f"hardware configuration selects {sorted(configured_arms)}, not requested {arm}"
        )
    quality = QualityThresholds.from_yaml(args.quality_config)
    bundle = CalibrationBundle.load(args.calibration_bundle)
    task_config = load_task_config(args.task_config)
    patch_dimensions = tuple(task_config["table"]["open_transit_patch_dimensions_m"])
    model = URDFModel(resolve_hardware_path(args.hardware_config, hardware["robot"]["urdf"]))
    expected_camera = camera_info_from_hardware(hardware)
    task_run = (args.output_root / _run_id()).resolve()
    if task_run.exists():
        raise FileExistsError(f"tabletop run already exists: {task_run}")
    task_run.mkdir(parents=True)
    hardware_bytes = args.hardware_config.read_bytes()
    bundle_bytes = args.calibration_bundle.read_bytes()
    quality_bytes = args.quality_config.read_bytes()
    task_config_bytes = args.task_config.read_bytes()
    object_profile_bytes = object_profile.config_path.read_bytes()
    detector_config_bytes = object_profile.detector_config_path.read_bytes()
    grasp_shortlist_bytes = grasp_shortlist_path.read_bytes()
    presentation_config_bytes = (
        None if presentation.config_path is None else presentation.config_path.read_bytes()
    )
    # The D435i tabletop stream is already high-contrast.  CLAHE amplified
    # foam/print texture and corrupted marker corners in retained hardware
    # frames, while raw grayscale passed every archived tabletop burst.
    detector = CorrespondenceDetector(
        object_profile.detector_config_path,
        preprocess=False,
    )
    recording, pairing = recording_configs(args.hardware_config)
    control_config, rate_hz = executor_config(args.hardware_config)
    control_config = replace(control_config, require_motion_endpoint_tolerance=False)
    configured_task_velocity = float(task_config["motion"]["maximum_arm_velocity_rad_s"])
    task_velocity = _resolve_task_velocity(
        configured_task_velocity,
        args.maximum_arm_velocity_rad_s,
        control_config.maximum_joint_velocity_rad_s,
    )
    empty_pose_set = PoseSet(
        robot_model=model.name,
        mode_machine=5,
        urdf_sha256=model.sha256,
        calibration_arm=arm,
    )
    preflight_frames: tuple[ROSImageFrame, ...] = ()
    loaded_frames: tuple[ROSImageFrame, ...] = ()
    clearance_frames: tuple[ROSImageFrame, ...] = ()
    status: dict = {
        "status": "started",
        "commands_robot": False,
        "arm": arm,
        "object_profile_id": object_profile.profile_id,
        "presentation_id": presentation.presentation_id,
        "motion_controller": args.motion_controller,
        "maximum_arm_velocity_rad_s": task_velocity,
        "requested_pregrasp_distance_m": args.pregrasp_distance_m,
        "arm_velocity_source": (
            "task_config_default"
            if args.maximum_arm_velocity_rad_s is None
            else "command_line_override"
        ),
    }
    primary_error: BaseException | None = None
    rejection_return_completed = False
    camera = observer = torso_observer = dex_observer = transport = dex_controller = None
    node = rclpy_module = None
    raw_recorder = None
    guard = synchronized = driver = planner = None
    runtime_warmup = None
    mpc_phases: dict[str, dict] = {}
    cube_anchor_motion = None
    camera_state_inputs = CameraStateInputBuffer()
    camera_estimator = AnchoredCameraStateEstimator(
        AnchoredCameraPoseEstimators(model=model, calibration_bundle=bundle)
    )
    camera_state_anchor = None
    camera_state_estimate = None
    reference_T_camera0 = None
    command_lock = CommandOwnerLock(args.lock_file)
    command_lock.acquire()
    try:
        planner = PersistentTabletopPlanner(
            executable=ROOT / ".venv-planner/bin/g1-curobo-worker",
            log_path=task_run / "planner.log",
        )
        planner.launch()
        print(
            "PLANNER STARTED — isolated CUDA initialization is running concurrently "
            "with read-only robot and camera preflight; no command publisher exists",
            flush=True,
        )
        try:
            import rclpy
        except ImportError as error:
            raise RuntimeError("rclpy unavailable; use ./tools/g1_tabletop_hardware.sh") from error
        rclpy_module = rclpy
        rclpy.init(args=None)
        node = rclpy.create_node("g1_dex3_tabletop_task")
        try:
            camera = ROSCameraSubscriber(
                node,
                image_topic=hardware["ros"]["image_topic"],
                camera_info_topic=hardware["ros"]["camera_info_topic"],
                camera_name=hardware["camera"]["name"],
                serial_number=hardware["camera"]["serial_number"],
                reliability=hardware["ros"]["camera_reliability"],
                qos_depth=int(hardware["ros"]["camera_qos_depth"]),
                maximum_frames=30,
            )
            states = StateSampleBuffer()

            def receive_lowstate(sample) -> None:
                states.add(sample)
                camera_state_inputs.add_lowstate(sample)

            transport_cfg = transport_config(
                args.hardware_config,
                interface=args.network_interface,
                domain_id=args.domain_id,
            )
            observer = UnitreeLowStateObserver(transport_cfg, on_sample=receive_lowstate)
            _wait_for_state(observer)
            torso_observer = UnitreeTorsoIMUObserver(
                transport_cfg,
                lowstate_observer=observer,
                on_sample=camera_state_inputs.add_torso_imu,
            )
            _wait_for_torso_imu(torso_observer)
            hand_cfg = dex3_config(
                args.hardware_config,
                interface=args.network_interface,
                domain_id=args.domain_id,
            )
            dex_observer = UnitreeDex3StateObserver(hand_cfg, initialize_factory=False)
            hands = _wait_for_hands(dex_observer)
            activation = _wait_for_activation(observer, states, empty_pose_set, recording)
            preflight_frames = _collect_frames(
                rclpy, node, camera, count=args.observation_frames, timeout_s=10.0
            )
            if preflight_frames[-1].camera_info.profile_sha256 != expected_camera.profile_sha256:
                raise ValueError("live camera profile differs from hardware.yaml")
            preflight_snapshot = _snapshot(activation.reference_state, hands)
            preflight_observation = observe_resting_cube(
                [item.image_bgr for item in preflight_frames],
                camera_info=expected_camera,
                detector=detector,
                snapshot=preflight_snapshot,
                base_T_camera=snapshot_base_T_camera(
                    preflight_snapshot,
                    torso_T_camera=bundle.torso_T_camera,
                    joint_position_offsets_rad=bundle.joint_position_offsets_rad,
                    model=model,
                ),
                minimum_tag_short_side_px=quality.minimum_tag_short_side_px,
                maximum_reprojection_error_px=quality.pnp_reject_reprojection_px,
            )
            print(
                "READ-ONLY PREFLIGHT PASSED — seated stationary state, both Dex3 "
                "states, rectified camera profile, and resting AprilCube are valid; "
                f"object={object_profile.profile_id}; "
                f"presentation={presentation.presentation_id}; no command publisher exists",
                flush=True,
            )
            if presentation.fixture is not None:
                print(
                    "PRIME-TOWER CONTRACT — 60 mm tower base fixed to the table; cube "
                    "centred and yaw-aligned on its top; the exact tower mesh will be a "
                    "CuRobo obstacle",
                    flush=True,
                )
            preflight_request = build_tabletop_request(
                arm=arm,
                observation=preflight_observation,
                calibration_bundle=bundle,
                calibration_bundle_path=args.calibration_bundle,
                grasp_shortlist_path=grasp_shortlist_path,
                task_config_path=args.task_config,
                object_dimensions_m=object_profile.dimensions_m,
                maximum_arm_velocity_rad_s=task_velocity,
                pregrasp_distance_m=args.pregrasp_distance_m,
                presentation_id=presentation.presentation_id,
                fixture=presentation.fixture,
            )
            status["pregrasp_distance_m"] = preflight_request.pregrasp_distance_m
            preflight_warmup_request_path = task_run / "preflight_warmup_request.json"
            preflight_request.write_json(preflight_warmup_request_path)
            runtime_warmup = planner.begin_payload_request(
                "prewarm-tabletop-runtime",
                payload={
                    "request": str(preflight_warmup_request_path.resolve()),
                    "moving_grasp_mpc": args.motion_controller == "mpc",
                },
            )
            print(
                "RUNTIME WARMUP QUEUED — generic CUDA initialization, MotionGen"
                f"{' and moving-grasp MPC' if args.motion_controller == 'mpc' else ''} "
                "warmup will run continuously while the read-only preview remains active; "
                "SPACE still authorizes only later command-publisher creation",
                flush=True,
            )
            _wait_for_space_with_preview(
                rclpy,
                node,
                camera,
                arm=arm,
                no_window=args.no_window,
            )
            warmup_event = planner.finish_request(runtime_warmup, timeout_s=180.0)
            runtime_warmup = None
            atomic_write_json(
                task_run / "runtime_warmup.json",
                warmup_event["payload"],
            )
            print(
                "RUNTIME WARMUP READY — persistent MotionGen"
                f"{' and moving-grasp MPC' if args.motion_controller == 'mpc' else ''} "
                "models completed before robot ownership",
                flush=True,
            )
            if args.hardware_config.read_bytes() != hardware_bytes:
                raise RuntimeError("hardware configuration changed after preflight")
            if args.calibration_bundle.read_bytes() != bundle_bytes:
                raise RuntimeError("calibration bundle changed after preflight")
            if args.quality_config.read_bytes() != quality_bytes:
                raise RuntimeError("capture quality configuration changed after preflight")
            if args.task_config.read_bytes() != task_config_bytes:
                raise RuntimeError("tabletop task configuration changed after preflight")
            if object_profile.config_path.read_bytes() != object_profile_bytes:
                raise RuntimeError("tabletop object profile changed after preflight")
            if object_profile.detector_config_path.read_bytes() != detector_config_bytes:
                raise RuntimeError("object detector config changed after preflight")
            if grasp_shortlist_path.read_bytes() != grasp_shortlist_bytes:
                raise RuntimeError("object grasp shortlist changed after preflight")
            if (
                presentation.config_path is not None
                and presentation.config_path.read_bytes() != presentation_config_bytes
            ):
                raise RuntimeError("tabletop presentation configuration changed after preflight")
            raw_recorder = RawEpisodeRecorder(
                task_run / "raw_episode",
                repository=ROOT,
                topics=tabletop_raw_topics(record_camera=not args.skip_camera_recording),
            )
            raw_recorder.start()
            recording_content = (
                "state/command plus raw RGB and CameraInfo"
                if not args.skip_camera_recording
                else "state/command only; camera topics intentionally excluded"
            )
            print(
                "RAW EPISODE RECORDING — plain MCAP is active before command publisher "
                f"creation ({recording_content}); compression and conversion remain offline",
                flush=True,
            )
            activation = _wait_for_activation(observer, states, empty_pose_set, recording)
            gravity = gravity_feedforward(
                args.hardware_config, activation.reference_state.position
            )
            guard = watchdog(
                args.hardware_config,
                host=args.pc2_host,
                ssh_identity=args.pc2_ssh_identity,
                initial_fsm_id=int(hardware["control"]["required_seated_fsm_id"]),
                restore_seated=True,
            )
            transport = UnitreeDebugLowCmdTransport(
                transport_cfg,
                debug_lowcmd_config(args.hardware_config),
                observer=observer,
                ownership_keepalive=guard.pulse,
            )
            observer = None
            dex_controller = UnitreeDex3PostureController(hand_cfg, observer=dex_observer)
            dex_observer = None
            raw = PoseExecutor(
                transport=transport,
                clock=SystemClock(),
                pose_set=empty_pose_set,
                handoff_q=activation.handoff_q,
                hold_q=activation.hold_q,
                approved_validation_report_sha256="0" * 64,
                config=control_config,
                gravity_feedforward=gravity,
            )
            synchronized = SynchronizedPoseExecutor(raw)
            driver = ExecutorControlDriver(
                synchronized,
                rate_hz=rate_hz,
                safety_heartbeat=guard.pulse,
            )
            guard.start()
            held_hands = dex_controller.acquire_measured_hold(safety_heartbeat=guard.pulse)
            driver.safety_heartbeat = _finger_heartbeat(guard, dex_controller)
            driver.start()
            synchronized.acquire(operator_confirmed=True)
            _wait_ready(
                synchronized,
                driver,
                timeout_s=control_config.acquisition_ramp_s + 5.0,
                label="loaded ownership",
            )
            print(
                "CONTROL ACQUIRED — exact measured 29-joint state held with dual-Dex3 "
                "gravity feedforward; observing the fixed cube and planning the reversible "
                "supported escape",
                flush=True,
            )
            loaded_frames = _collect_frames(
                rclpy,
                node,
                camera,
                count=args.observation_frames,
                timeout_s=10.0,
                control_check=driver.check,
            )
            loaded_state = synchronized.observe_state()
            loaded_snapshot = _snapshot(loaded_state, held_hands)
            loaded_observation = observe_resting_cube(
                [item.image_bgr for item in loaded_frames],
                camera_info=expected_camera,
                detector=detector,
                snapshot=loaded_snapshot,
                base_T_camera=snapshot_base_T_camera(
                    loaded_snapshot,
                    torso_T_camera=bundle.torso_T_camera,
                    joint_position_offsets_rad=bundle.joint_position_offsets_rad,
                    model=model,
                ),
                minimum_tag_short_side_px=quality.minimum_tag_short_side_px,
                maximum_reprojection_error_px=quality.pnp_reject_reprojection_px,
            )
            loaded_request = build_tabletop_request(
                arm=arm,
                observation=loaded_observation,
                calibration_bundle=bundle,
                calibration_bundle_path=args.calibration_bundle,
                grasp_shortlist_path=grasp_shortlist_path,
                task_config_path=args.task_config,
                object_dimensions_m=object_profile.dimensions_m,
                maximum_arm_velocity_rad_s=task_velocity,
                pregrasp_distance_m=args.pregrasp_distance_m,
                presentation_id=presentation.presentation_id,
                fixture=presentation.fixture,
            )
            loaded_request_path = task_run / "loaded_request.json"
            escape_path = task_run / "supported_escape.json"
            loaded_request.write_json(loaded_request_path)
            try:
                planner.request(
                    "plan-supported-escape",
                    request_path=loaded_request_path,
                    output_path=escape_path,
                    control_check=driver.check,
                )
            except (PlannerRequestRejected, RuntimeError) as error:
                driver.check()
                raise TabletopTaskRejected(f"supported escape planning failed: {error}") from error
            escape = SupportedEscapePlan.from_json(escape_path)
            clearance_request = request_at_clearance(loaded_request, escape)
            pose_set = pose_set_from_trajectories(
                arm=arm,
                trajectories=(escape.outbound, escape.inbound),
                reference_full_q=loaded_state.position,
                robot_model=model.name,
                urdf_sha256=model.sha256,
                source="NVlabs/curobo_reversible_supported_escape",
            )
            synchronized.install_validated_plan(
                pose_set=pose_set,
                approved_validation_report_sha256=escape.content_sha256,
                validated_reference_state=loaded_state,
            )
            print(
                "PRE-MOTION PLANNING READY — the reversible supported escape is frozen; "
                "persistent MotionGen CUDA models were already warmed before ownership. "
                "The real task will still be planned from a fresh fixed-cube observation "
                "at clearance",
                flush=True,
            )
            initial_left = held_hands.left.position
            initial_right = held_hands.right.position
            grasp_close = None
            retention_evidence = None
            retention_route = None
            _execute_trajectory(
                synchronized,
                driver,
                escape.outbound,
                plan_sha256=escape.content_sha256,
                control_config=control_config,
            )
            clearance_fingers_changed = False
            try:
                clearance_fingers_changed = True
                measured_open_pair = _command_fingers(
                    dex_controller,
                    driver,
                    guard,
                    left=active_open_target_q_rad if arm == "left" else initial_left,
                    right=active_open_target_q_rad if arm == "right" else initial_right,
                    label=f"{arm}-hand empty-open acquisition at clearance",
                )
                clearance_frames = _collect_frames(
                    rclpy,
                    node,
                    camera,
                    count=args.observation_frames,
                    timeout_s=10.0,
                    control_check=driver.check,
                )
                boundary_state = synchronized.observe_state()
                boundary_hands = dex_controller.observer.observe()
                measured_empty_open_q_rad = (
                    boundary_hands.left.position.copy()
                    if arm == "left"
                    else boundary_hands.right.position.copy()
                )
                acquisition_empty_open_q_rad = (
                    measured_open_pair.left.position.copy()
                    if arm == "left"
                    else measured_open_pair.right.position.copy()
                )
                atomic_write_json(
                    task_run / "dex3_run_local_references.json",
                    {
                        "active_side": arm,
                        "open_command_target_q_rad": list(active_open_target_q_rad),
                        "measured_empty_open_at_acquisition_q_rad": (
                            acquisition_empty_open_q_rad.tolist()
                        ),
                        "measured_empty_open_at_visual_anchor_q_rad": (
                            measured_empty_open_q_rad.tolist()
                        ),
                        "commissioned_empty_close_q_rad": list(empty_close_reference_q_rad),
                        "posture_position_tolerance_rad": (
                            dex_controller.config.posture_position_tolerance_rad
                        ),
                    },
                )
                print(
                    "RUN-LOCAL EMPTY OPEN READY — descriptor zero remains the command; "
                    "planning and release use the measured clearance posture; maximum "
                    "descriptor residual="
                    f"{np.max(np.abs(measured_empty_open_q_rad)):.4f}rad, "
                    "acquisition-to-anchor change="
                    f"{np.max(np.abs(measured_empty_open_q_rad - acquisition_empty_open_q_rad)):.4f}rad",
                    flush=True,
                )
                boundary_snapshot = _snapshot(boundary_state, boundary_hands)
                boundary_observation = observe_resting_cube(
                    [item.image_bgr for item in clearance_frames],
                    camera_info=expected_camera,
                    detector=detector,
                    snapshot=boundary_snapshot,
                    base_T_camera=snapshot_base_T_camera(
                        boundary_snapshot,
                        torso_T_camera=bundle.torso_T_camera,
                        joint_position_offsets_rad=bundle.joint_position_offsets_rad,
                        model=model,
                    ),
                    minimum_tag_short_side_px=quality.minimum_tag_short_side_px,
                    maximum_reprojection_error_px=quality.pnp_reject_reprojection_px,
                )
                # Freeze an arbitrary reference frame at the cube's clearance
                # pose.  The cube must remain stationary through this sample;
                # later MPC observations are independent and may move within
                # this frame while proprioception propagates the camera pose.
                reference_T_camera0 = invert_transform(
                    np.asarray(boundary_observation.camera_T_object, dtype=np.float64)
                )
                cube_anchor_motion = camera_motion_from_fixed_cube(
                    loaded_observation.camera_T_object,
                    boundary_observation.camera_T_object,
                )
                clearance_request = request_at_clearance_observation(
                    loaded_request,
                    escape,
                    boundary_observation,
                )
                clearance_request_path = task_run / "clearance_request.json"
                clearance_request.write_json(clearance_request_path)
                # Preserve the state synchronized to the visual anchor before
                # planning. At the measured ~1 kHz input rate, the bounded
                # buffer can otherwise evict this sample during a GPU solve.
                anchor_time_s = float(
                    np.median([frame.timing.receipt_monotonic_s for frame in clearance_frames])
                )
                camera_state_anchor = _wait_for_camera_state_input(
                    camera_state_inputs,
                    target_monotonic_s=anchor_time_s,
                    maximum_age_s=control_config.state_freshness_timeout_s,
                    maximum_gap_s=pairing.maximum_bracket_span_s,
                    control_check=driver.check,
                )
                anchor_estimate = camera_estimator.reset(
                    CameraPoseAnchor(
                        reference_T_camera=reference_T_camera0,
                        sample=camera_state_anchor.sample,
                    )
                )
                atomic_write_json(
                    task_run / "camera_state_anchor.json",
                    {
                        "estimate": anchor_estimate.to_dict(),
                        "input": camera_state_anchor.to_dict(),
                        "visual_reference": {
                            "kind": "frozen_clearance_cube_frame",
                            "observation_sha256": (
                                clearance_request.observation.content_sha256
                            ),
                            "contract": (
                                "cube stationary through clearance observation; "
                                "later cube observations may move"
                            ),
                        },
                    },
                )

                def current_moving_grasp_target():
                    frame = _collect_frames(
                        rclpy,
                        node,
                        camera,
                        count=1,
                        timeout_s=1.0,
                        control_check=driver.check,
                    )[0]
                    target = observe_live_cube_frame(
                        frame.image_bgr,
                        camera_info=expected_camera,
                        detector=detector,
                        minimum_tag_short_side_px=quality.minimum_tag_short_side_px,
                        maximum_reprojection_error_px=quality.pnp_reject_reprojection_px,
                    )
                    synchronized_input = _wait_for_camera_state_input(
                        camera_state_inputs,
                        target_monotonic_s=frame.timing.receipt_monotonic_s,
                        maximum_age_s=control_config.state_freshness_timeout_s,
                        maximum_gap_s=pairing.maximum_bracket_span_s,
                        control_check=driver.check,
                    )
                    estimate = camera_estimator.estimate(synchronized_input.sample)
                    if estimate.anchor_timestamp_ns != camera_state_anchor.sample.timestamp_ns:
                        raise RuntimeError(
                            "live moving-target estimate belongs to another table anchor"
                        )
                    return (
                        synchronized_input,
                        estimate,
                        {
                            **target,
                            "source_monotonic_s": frame.timing.receipt_monotonic_s,
                            "source_utc": frame.timing.receipt_utc,
                            "source_header_stamp_ns": frame.timing.header_stamp_ns,
                            "camera_profile_sha256": frame.camera_info.profile_sha256,
                        },
                    )

                escape_return = _trajectory_with_endpoints(
                    escape.inbound,
                    from_pose_id="return_to_clearance",
                    to_pose_id="__handoff__",
                )
                mpc_clearance_preparation = None
                if args.motion_controller == "trajectory":
                    pregrasp_path = task_run / "pregrasp_plan.json"
                    planner.request(
                        "plan-tabletop-pregrasp-at-clearance",
                        request_path=clearance_request_path,
                        output_path=pregrasp_path,
                        control_check=driver.check,
                    )
                    pregrasp_plan = TabletopPregraspPlan.from_json(pregrasp_path)
                    stage_trajectories = (
                        pregrasp_plan.outbound,
                        pregrasp_plan.inbound,
                        escape_return,
                    )
                    replanned_pose_set = pose_set_from_trajectories(
                        arm=arm,
                        trajectories=stage_trajectories,
                        reference_full_q=boundary_state.position,
                        robot_model=model.name,
                        urdf_sha256=model.sha256,
                        source="NVlabs/curobo_reversible_pregrasp_boundary",
                        initial_pose_id="clearance",
                        initial_command_q_rad=escape.outbound.command_q_rad[-1],
                    )
                    stage_plan_sha256 = pregrasp_plan.content_sha256
                else:
                    replanned_execution_path = task_run / "execution_plan.json"
                    planner.request(
                        "replan-tabletop-at-clearance",
                        request_path=clearance_request_path,
                        output_path=replanned_execution_path,
                        control_check=driver.check,
                    )
                    replanned_execution = TabletopExecutionPlan.from_json(replanned_execution_path)
                    replanned_pose_set = pose_set_from_trajectories(
                        arm=arm,
                        trajectories=replanned_execution.trajectories,
                        reference_full_q=boundary_state.position,
                        robot_model=model.name,
                        urdf_sha256=model.sha256,
                        source="NVlabs/curobo_fixed_cube_clearance_replan",
                        initial_pose_id="clearance",
                        initial_command_q_rad=escape.outbound.command_q_rad[-1],
                    )
                    stage_plan_sha256 = replanned_execution.content_sha256
                    preparation_event = planner.request_payload(
                        "prepare-moving-grasp-mpc",
                        payload={
                            "reference_T_camera0": np.asarray(
                                reference_T_camera0,
                                dtype=np.float64,
                            ).tolist(),
                        },
                        control_check=driver.check,
                        timeout_s=30.0,
                    )
                    mpc_clearance_preparation = dict(preparation_event["payload"])
                synchronized.replace_validated_remaining_plan(
                    pose_set=replanned_pose_set,
                    approved_validation_report_sha256=stage_plan_sha256,
                    validated_reference_state=boundary_state,
                )
            except (PlannerRequestRejected, RuntimeError, ValueError) as error:
                driver.check()
                try:
                    if clearance_fingers_changed:
                        _command_fingers(
                            dex_controller,
                            driver,
                            guard,
                            left=initial_left,
                            right=initial_right,
                            label="initial finger posture restoration after clearance failure",
                        )
                    _execute_trajectory(
                        synchronized,
                        driver,
                        escape.inbound,
                        plan_sha256=escape.content_sha256,
                        control_config=control_config,
                    )
                except (RuntimeError, ValueError) as recovery_error:
                    raise RuntimeError(
                        f"clearance-boundary preparation failed: {error}; "
                        f"frozen reverse also failed: {recovery_error}"
                    ) from recovery_error
                rejection_return_completed = True
                raise TabletopTaskRejected(
                    f"fixed-cube clearance-boundary replan failed: {error}"
                ) from error

            if args.motion_controller == "trajectory":
                execution = pregrasp_plan
                task = None
                normal_routes = {
                    pregrasp_plan.outbound.to_pose_id: pregrasp_plan.outbound,
                    escape_return.to_pose_id: escape_return,
                }
                recovery_routes = {
                    (
                        pregrasp_plan.inbound.from_pose_id,
                        pregrasp_plan.inbound.to_pose_id,
                    ): pregrasp_plan.inbound
                }
                active_plan_sha256 = pregrasp_plan.content_sha256
                selected_candidate_id = pregrasp_plan.selected_candidate_id
                active_open = pregrasp_plan.open_active_dex3_q_rad
                print(
                    "CLEARANCE-TO-PREGRASP PLAN INSTALLED — fixed-cube anchor measured "
                    f"{cube_anchor_motion['translation_norm_mm']:.2f} mm / "
                    f"{cube_anchor_motion['rotation_deg']:.2f} deg camera motion; selected "
                    f"grasp {selected_candidate_id}; its unexecuted linear approach passed "
                    "strict validation, but only the reversible pregrasp route was installed; "
                    "the payload lifecycle will be planned once after the pregrasp state "
                    f"correction; planning={pregrasp_plan.planner_provenance['elapsed_s']:.2f}s",
                    flush=True,
                )
            else:
                execution = replanned_execution
                task = replanned_execution.task
                normal_routes, recovery_routes = _trajectory_maps(execution)
                active_plan_sha256 = execution.content_sha256
                task.write_json(task_run / "task_plan.json")
                selected_candidate_id = task.selected_candidate_id
                active_open = task.open_active_dex3_q_rad
                print(
                    "CLEARANCE-BOUNDARY REPLAN INSTALLED — fixed-cube anchor measured "
                    f"{cube_anchor_motion['translation_norm_mm']:.2f} mm / "
                    f"{cube_anchor_motion['rotation_deg']:.2f} deg camera motion; selected "
                    f"grasp {selected_candidate_id}; required hand/table execution "
                    f"margin={clearance_request.minimum_hand_plane_clearance_m * 1000.0:.1f} mm",
                    flush=True,
                )

            def execute_phase(
                name: str,
                *,
                use_mpc: bool = True,
            ) -> dict | None:
                if args.motion_controller == "mpc" and use_mpc and name == "grasp_approach":
                    phase_record = {
                        "completed": False,
                        "preparation": None,
                        "windows": [],
                        "rejected_windows": [],
                        "moving_targets": [],
                    }
                    mpc_phases[name] = phase_record
                    preparation, windows = _execute_mpc_phase(
                        synchronized,
                        driver,
                        planner,
                        arm=arm,
                        trajectory=normal_routes[name],
                        plan_sha256=active_plan_sha256,
                        control_config=control_config,
                        phase_record=phase_record,
                        moving_target_provider=current_moving_grasp_target,
                        reference_T_camera0=reference_T_camera0,
                        prepared_mpc=mpc_clearance_preparation,
                    )
                    assert phase_record["preparation"] == preparation
                    assert phase_record["windows"] == windows
                    return phase_record
                _execute_trajectory(
                    synchronized,
                    driver,
                    normal_routes[name],
                    plan_sha256=active_plan_sha256,
                    control_config=control_config,
                )
                return None

            def execute_recovery(source: str, target: str) -> None:
                _execute_trajectory(
                    synchronized,
                    driver,
                    recovery_routes[(source, target)],
                    plan_sha256=active_plan_sha256,
                    control_config=control_config,
                )

            def execute_return_to_clearance(*, use_mpc: bool) -> None:
                for phase in _return_to_clearance_phases(normal_routes):
                    execute_phase(phase, use_mpc=use_mpc)

            def open_active_hand(label: str) -> None:
                _command_fingers(
                    dex_controller,
                    driver,
                    guard,
                    left=active_open if arm == "left" else initial_left,
                    right=active_open if arm == "right" else initial_right,
                    left_acceptance=(measured_empty_open_q_rad if arm == "left" else initial_left),
                    right_acceptance=(
                        measured_empty_open_q_rad if arm == "right" else initial_right
                    ),
                    label=label,
                )

            def return_after_task_rejection(*, from_test_lift: bool) -> None:
                nonlocal rejection_return_completed
                if from_test_lift:
                    execute_recovery("retention_test_lift", "payload_replace")
                    open_active_hand("release after failed retention test")
                    execute_phase("grasp_retreat", use_mpc=False)
                else:
                    open_active_hand("open after rejected grasp attempt")
                    execute_recovery("grasp_approach", "grasp_retreat")
                execute_return_to_clearance(use_mpc=False)
                _command_fingers(
                    dex_controller,
                    driver,
                    guard,
                    left=initial_left,
                    right=initial_right,
                    label="initial finger posture restoration after task rejection",
                )
                execute_phase("__handoff__", use_mpc=False)
                rejection_return_completed = True

            execute_phase("move_to_pregrasp")
            if args.motion_controller == "trajectory":
                original_pregrasp_plan = pregrasp_plan
                original_clearance_request = clearance_request
                boundary_return = pregrasp_to_clearance_return(original_pregrasp_plan)
                try:
                    pregrasp_state = synchronized.observe_state()
                    pregrasp_hands = dex_controller.observer.observe()
                    camera_state_current = _wait_for_camera_state_input(
                        camera_state_inputs,
                        target_monotonic_s=None,
                        maximum_age_s=control_config.state_freshness_timeout_s,
                        maximum_gap_s=pairing.maximum_bracket_span_s,
                        control_check=driver.check,
                    )
                    camera_state_estimate = camera_estimator.estimate(camera_state_current.sample)
                    q29 = np.asarray(pregrasp_state.position, dtype=np.float64).copy()
                    q29[np.asarray(arm_indices(arm))] = np.asarray(
                        original_pregrasp_plan.outbound.command_q_rad[-1]
                    )
                    estimated_request = request_at_estimated_pregrasp(
                        original_clearance_request,
                        snapshot=RobotSnapshot(
                            measured_q29_rad=tuple(q29),
                            left_dex3_q_rad=tuple(pregrasp_hands.left.position),
                            right_dex3_q_rad=tuple(pregrasp_hands.right.position),
                        ),
                        estimate=camera_state_estimate,
                        anchor_input=camera_state_anchor,
                        current_input=camera_state_current,
                    )
                    estimated_request_path = task_run / "pregrasp_estimated_request.json"
                    remaining_path = task_run / "pregrasp_remaining_plan.json"
                    estimated_request.write_json(estimated_request_path)
                    planner.request(
                        "replan-tabletop-at-pregrasp",
                        request_path=estimated_request_path,
                        output_path=remaining_path,
                        control_check=driver.check,
                    )
                    remaining = PregraspRemainingPlan.from_json(remaining_path)
                    if (
                        remaining.prior_pregrasp_plan_sha256
                        != original_pregrasp_plan.content_sha256
                    ):
                        raise RuntimeError(
                            "pregrasp correction belongs to a different prior pregrasp plan"
                        )
                    if remaining.selected_candidate_id != selected_candidate_id:
                        raise RuntimeError("pregrasp correction changed the selected grasp")
                    expected_remaining = build_pregrasp_remaining_plan(
                        prior_pregrasp_plan=original_pregrasp_plan,
                        estimated_request=estimated_request,
                        task=remaining.task,
                    )
                    if remaining.content_sha256 != expected_remaining.content_sha256:
                        raise RuntimeError(
                            "pregrasp correction differs from the deterministic remapping "
                            "of its hash-bound task and prior execution"
                        )
                    installation_input = _wait_for_camera_state_input(
                        camera_state_inputs,
                        target_monotonic_s=None,
                        maximum_age_s=control_config.state_freshness_timeout_s,
                        maximum_gap_s=pairing.maximum_bracket_span_s,
                        control_check=driver.check,
                    )
                    installation_estimate = camera_estimator.estimate(installation_input.sample)
                    installation_motion = pose_error(
                        camera_state_estimate.reference_T_camera,
                        installation_estimate.reference_T_camera,
                    )
                    if installation_motion["translation_norm_mm"] > float(
                        task_config["perception"]["maximum_translation_spread_mm"]
                    ) or installation_motion["rotation_deg"] > float(
                        task_config["perception"]["maximum_rotation_spread_deg"]
                    ):
                        raise RuntimeError(
                            "camera state changed while CuRobo replanned: "
                            f"{installation_motion['translation_norm_mm']:.3f}mm / "
                            f"{installation_motion['rotation_deg']:.3f}deg"
                        )
                    atomic_write_json(
                        task_run / "pregrasp_estimator_installation_check.json",
                        {
                            "planned_estimate": camera_state_estimate.to_dict(),
                            "installation_estimate": installation_estimate.to_dict(),
                            "installation_input": installation_input.to_dict(),
                            "change": installation_motion,
                            "limits": {
                                "translation_mm": task_config["perception"][
                                    "maximum_translation_spread_mm"
                                ],
                                "rotation_deg": task_config["perception"][
                                    "maximum_rotation_spread_deg"
                                ],
                            },
                        },
                    )
                    corrected_pose_set = pose_set_from_trajectories(
                        arm=arm,
                        trajectories=remaining.trajectories,
                        reference_full_q=pregrasp_state.position,
                        robot_model=model.name,
                        urdf_sha256=model.sha256,
                        source="NVlabs/curobo_proprioceptive_pregrasp_replan",
                        initial_pose_id="move_to_pregrasp",
                        initial_command_q_rad=(original_pregrasp_plan.outbound.command_q_rad[-1]),
                    )
                    synchronized.replace_validated_remaining_plan(
                        pose_set=corrected_pose_set,
                        approved_validation_report_sha256=remaining.content_sha256,
                        validated_reference_state=pregrasp_state,
                    )
                except (PlannerRequestRejected, RuntimeError, ValueError) as error:
                    driver.check()
                    _execute_trajectory(
                        synchronized,
                        driver,
                        boundary_return,
                        plan_sha256=original_pregrasp_plan.content_sha256,
                        control_config=control_config,
                    )
                    _command_fingers(
                        dex_controller,
                        driver,
                        guard,
                        left=initial_left,
                        right=initial_right,
                        label="initial finger posture restoration after pregrasp replan rejection",
                    )
                    _execute_trajectory(
                        synchronized,
                        driver,
                        escape_return,
                        plan_sha256=original_pregrasp_plan.content_sha256,
                        control_config=control_config,
                    )
                    rejection_return_completed = True
                    raise TabletopTaskRejected(
                        f"pregrasp camera-state correction failed: {error}"
                    ) from error

                clearance_request = estimated_request
                task = remaining.task
                execution = remaining
                active_plan_sha256 = remaining.content_sha256
                normal_routes = {
                    trajectory.to_pose_id: trajectory
                    for trajectory in (
                        *remaining.trajectories,
                        escape_return,
                    )
                }
                recovery_routes = {
                    (trajectory.from_pose_id, trajectory.to_pose_id): trajectory
                    for trajectory in remaining.recovery_trajectories
                }
                task.write_json(task_run / "pregrasp_corrected_task_plan.json")
                print(
                    "PREGRASP STATE CORRECTION INSTALLED — waist/pelvis/torso state "
                    f"propagated the fixed-cube camera anchor for "
                    f"{camera_state_estimate.anchor_age_s:.3f}s; grasp "
                    f"{task.selected_candidate_id} was preserved and every remaining "
                    "motion was planned from the exact active command; "
                    f"planning={task.planner_provenance['elapsed_s']:.2f}s, "
                    "planner_pool_reuse="
                    f"strict:{task.planner_provenance['strict_checker_reused']},"
                    f"fixed_close:{task.planner_provenance['fixed_close_validator_reused']},"
                    f"open:{task.planner_provenance['open_optimizer_reused']},"
                    f"payload:{task.planner_provenance['attached_optimizer_reused']}",
                    flush=True,
                )
                execute_phase("estimated_pregrasp", use_mpc=False)
            else:
                print(
                    "LOCAL MOVING-TARGET MPC READY — MotionGen remains responsible for "
                    "clearance-to-pregrasp and every payload/return route. Only the open-hand "
                    "pregrasp-to-grasp segment uses native CuRobo Cartesian MPC; every window "
                    "combines one fresh AprilCube image with the frozen clearance-cube frame "
                    "and current pelvis/waist/torso state",
                    flush=True,
                )
            if task is None:
                raise RuntimeError("tabletop remainder planning ended without a task plan")
            retention_test_lift_mm = 1000.0 * float(
                task.planner_provenance["retention_test_lift_actual_m"]
            )
            payload_lift_mm = 1000.0 * float(clearance_request.lift_m)
            print(
                "EMPTY-CLOSE REFERENCE READY — grasp evidence will require stable "
                "thumb and opposing-finger shortfall of at least "
                f"{minimum_opposed_shortfall_rad:.4f}rad on both sides; pressure and "
                "tau_est remain recorded diagnostics; beginning grasp approach",
                flush=True,
            )
            try:
                mpc_approach_record = execute_phase("grasp_approach")
            except MPCInitialWindowUnavailable as error:
                driver.check()
                print(
                    "TASK REJECTED — moving-target MPC did not leave pregrasp; "
                    f"executing the validated pregrasp reverse: {error}",
                    flush=True,
                )
                execute_recovery("move_to_pregrasp", "return_to_clearance")
                _command_fingers(
                    dex_controller,
                    driver,
                    guard,
                    left=initial_left,
                    right=initial_right,
                    label="initial finger posture restoration after MPC rejection",
                )
                execute_phase("__handoff__", use_mpc=False)
                rejection_return_completed = True
                raise TabletopTaskRejected(
                    f"moving-target MPC could not start: {error}"
                ) from error
            if args.motion_controller == "mpc":
                if mpc_approach_record is None:
                    raise RuntimeError("moving-target MPC approach produced no execution record")
                executed_approach = _executed_mpc_approach(
                    mpc_approach_record["windows"],
                    arm=arm,
                    joint_position_offsets_rad=clearance_request.joint_position_offsets_rad,
                )
                moving_targets = mpc_approach_record["moving_targets"]
                if not moving_targets:
                    raise RuntimeError("moving-target MPC approach recorded no live cube target")
                terminal_window = MPCCommandWindow.from_dict(mpc_approach_record["windows"][-1])
                terminal_target = terminal_window.diagnostics.get("moving_target")
                if not isinstance(terminal_target, dict):
                    raise RuntimeError("terminal MPC window has no bound moving target")
                terminal_state = synchronized.observe_state()
                terminal_hands = dex_controller.observer.observe()
                terminal_active_fingers = (
                    terminal_hands.left.position
                    if arm == "left"
                    else terminal_hands.right.position
                )

                # Install the exact reverse of what was actually exposed to
                # control before asking CUDA for any new payload route. A
                # continuation-planning rejection can therefore return
                # without using the stale nominal grasp trajectory.
                # Reversing timestamps requires the same normalization used
                # by the planner's immutable trajectory helper.
                duration = executed_approach.sample_time_s[-1]
                provisional_retreat = PlannedTrajectory(
                    from_pose_id="grasp_approach",
                    to_pose_id="grasp_retreat",
                    sample_time_s=tuple(
                        duration - value for value in reversed(executed_approach.sample_time_s)
                    ),
                    command_q_rad=tuple(reversed(executed_approach.command_q_rad)),
                    model_q_rad=tuple(reversed(executed_approach.model_q_rad)),
                    planning_time_s=0.0,
                )
                provisional_return = _trajectory_with_endpoints(
                    task.trajectories[7],
                    from_pose_id="grasp_retreat",
                    to_pose_id="return_to_clearance",
                )
                provisional_document = {
                    "executed_approach": executed_approach.to_dict(),
                    "terminal_window_sha256": terminal_window.content_sha256,
                    "terminal_target": terminal_target,
                }
                provisional_sha256 = hashlib.sha256(
                    json.dumps(
                        provisional_document,
                        sort_keys=True,
                        separators=(",", ":"),
                        allow_nan=False,
                    ).encode()
                ).hexdigest()
                provisional_pose_set = pose_set_from_trajectories(
                    arm=arm,
                    trajectories=(
                        provisional_retreat,
                        provisional_return,
                        escape_return,
                    ),
                    reference_full_q=terminal_state.position,
                    robot_model=model.name,
                    urdf_sha256=model.sha256,
                    source="NVlabs/curobo_exact_accepted_mpc_reverse",
                    initial_pose_id="grasp_approach",
                    initial_command_q_rad=executed_approach.command_q_rad[-1],
                )
                synchronized.replace_validated_remaining_plan(
                    pose_set=provisional_pose_set,
                    approved_validation_report_sha256=provisional_sha256,
                    validated_reference_state=terminal_state,
                )

                # The selected grasp's finger sweep is object-relative and
                # was qualified before ownership. Close as soon as MPC reaches
                # that live grasp instead of leaving an open hand beside a
                # moving cube during the payload-planner solve. The actual
                # contact-stopped fingers are still checked against the newly
                # planned payload route before any lift.
                active_close_target = task.close_target_active_dex3_q_rad
                try:
                    grasp_close = _command_retention_test_close(
                        dex_controller,
                        driver,
                        guard,
                        active_side=arm,
                        left=active_close_target if arm == "left" else initial_left,
                        right=active_close_target if arm == "right" else initial_right,
                        empty_close_reference_q_rad=empty_close_reference_q_rad,
                        minimum_opposed_shortfall_rad=minimum_opposed_shortfall_rad,
                        label=f"descriptor-defined {arm}-hand moving-cube close",
                    )
                except Dex3GraspNotAcquiredError as error:
                    driver.check()
                    open_active_hand("open after rejected moving-cube close")
                    _execute_trajectory(
                        synchronized,
                        driver,
                        provisional_retreat,
                        plan_sha256=provisional_sha256,
                        control_config=control_config,
                    )
                    _execute_trajectory(
                        synchronized,
                        driver,
                        provisional_return,
                        plan_sha256=provisional_sha256,
                        control_config=control_config,
                    )
                    _command_fingers(
                        dex_controller,
                        driver,
                        guard,
                        left=initial_left,
                        right=initial_right,
                        label="initial finger posture restoration after rejected moving close",
                    )
                    _execute_trajectory(
                        synchronized,
                        driver,
                        escape_return,
                        plan_sha256=provisional_sha256,
                        control_config=control_config,
                    )
                    rejection_return_completed = True
                    raise TabletopTaskRejected(
                        f"no stable moving-cube grasp close: {error}"
                    ) from error
                atomic_write_json(task_run / "grasp_close.json", grasp_close.to_dict())

                continuation_request = MovingGraspContinuationRequest(
                    tabletop_request=clearance_request,
                    prior_task_plan=task,
                    terminal_command_q_rad=executed_approach.command_q_rad[-1],
                    terminal_active_dex3_q_rad=tuple(terminal_active_fingers),
                    reference_T_camera=tuple(
                        tuple(float(value) for value in row)
                        for row in terminal_target["reference_T_camera"]
                    ),
                    camera_T_object=tuple(
                        tuple(float(value) for value in row)
                        for row in terminal_target["camera_T_object"]
                    ),
                    executed_grasp_approach=executed_approach,
                    terminal_mpc_window_sha256=terminal_window.content_sha256,
                    target_provenance=terminal_target,
                )
                continuation_request_path = task_run / "moving_grasp_continuation_request.json"
                continuation_plan_path = task_run / "moving_grasp_continuation_plan.json"
                continuation_request.write_json(continuation_request_path)
                try:
                    planner.request(
                        "plan-moving-grasp-continuation",
                        request_path=continuation_request_path,
                        output_path=continuation_plan_path,
                        control_check=driver.check,
                    )
                    corrected_task = TabletopTaskPlan.from_json(continuation_plan_path)
                except (PlannerRequestRejected, RuntimeError, ValueError) as error:
                    driver.check()
                    open_active_hand("release after moving-target continuation rejection")
                    _execute_trajectory(
                        synchronized,
                        driver,
                        provisional_retreat,
                        plan_sha256=provisional_sha256,
                        control_config=control_config,
                    )
                    _execute_trajectory(
                        synchronized,
                        driver,
                        provisional_return,
                        plan_sha256=provisional_sha256,
                        control_config=control_config,
                    )
                    _command_fingers(
                        dex_controller,
                        driver,
                        guard,
                        left=initial_left,
                        right=initial_right,
                        label="initial finger posture restoration after continuation rejection",
                    )
                    _execute_trajectory(
                        synchronized,
                        driver,
                        escape_return,
                        plan_sha256=provisional_sha256,
                        control_config=control_config,
                    )
                    rejection_return_completed = True
                    raise TabletopTaskRejected(
                        f"moving-target payload continuation failed: {error}"
                    ) from error
                if corrected_task.request_sha256 != clearance_request.content_sha256:
                    raise RuntimeError("moving-grasp continuation belongs to another request")
                if corrected_task.selected_candidate_id != selected_candidate_id:
                    raise RuntimeError("moving-grasp continuation changed the selected grasp")
                continuation_pose_set = pose_set_from_trajectories(
                    arm=arm,
                    trajectories=(*corrected_task.trajectories[2:], escape_return),
                    reference_full_q=terminal_state.position,
                    robot_model=model.name,
                    urdf_sha256=model.sha256,
                    source="NVlabs/curobo_moving_grasp_payload_continuation",
                    initial_pose_id="grasp_approach",
                    initial_command_q_rad=executed_approach.command_q_rad[-1],
                )
                continuation_install_state = synchronized.observe_state()
                synchronized.replace_validated_remaining_plan(
                    pose_set=continuation_pose_set,
                    approved_validation_report_sha256=corrected_task.content_sha256,
                    validated_reference_state=continuation_install_state,
                )
                task = corrected_task
                active_plan_sha256 = corrected_task.content_sha256
                normal_routes = {
                    trajectory.to_pose_id: trajectory
                    for trajectory in (*corrected_task.trajectories, escape_return)
                }
                recovery_routes = {
                    ("grasp_approach", "grasp_retreat"): _trajectory_with_endpoints(
                        corrected_task.trajectories[6],
                        from_pose_id="grasp_approach",
                        to_pose_id="grasp_retreat",
                    ),
                    ("retention_test_lift", "payload_replace"): (
                        _trajectory_with_endpoints(
                            corrected_task.trajectories[5],
                            from_pose_id="retention_test_lift",
                            to_pose_id="payload_replace",
                        )
                    ),
                    ("move_to_pregrasp", "return_to_clearance"): (
                        _trajectory_with_endpoints(
                            corrected_task.trajectories[7],
                            from_pose_id="move_to_pregrasp",
                            to_pose_id="return_to_clearance",
                        )
                    ),
                }
                retention_test_lift_mm = 1000.0 * float(
                    task.planner_provenance["retention_test_lift_actual_m"]
                )
                print(
                    "MOVING-TARGET CONTINUATION INSTALLED — fixed close and attached "
                    "lift were rebuilt at the reached cube pose; replacement reverses "
                    "that lift and the exact accepted MPC approach",
                    flush=True,
                )
            if grasp_close is None:
                active_close_target = task.close_target_active_dex3_q_rad
                try:
                    grasp_close = _command_retention_test_close(
                        dex_controller,
                        driver,
                        guard,
                        active_side=arm,
                        left=active_close_target if arm == "left" else initial_left,
                        right=active_close_target if arm == "right" else initial_right,
                        empty_close_reference_q_rad=empty_close_reference_q_rad,
                        minimum_opposed_shortfall_rad=minimum_opposed_shortfall_rad,
                        label=f"descriptor-defined {arm}-hand cube close",
                    )
                except Dex3GraspNotAcquiredError as error:
                    driver.check()
                    print(f"TASK REJECTED — no stable grasp close: {error}", flush=True)
                    return_after_task_rejection(from_test_lift=False)
                    raise TabletopTaskRejected(f"no stable grasp close: {error}") from error
                atomic_write_json(
                    task_run / "grasp_close.json",
                    grasp_close.to_dict(),
                )
            dex_controller.begin_retention_test()
            retention_request = RetentionRouteValidationRequest(
                tabletop_request=clearance_request,
                task_plan=task,
                measured_active_dex3_q_rad=grasp_close.close_q_rad,
                blocked_motor_ids=grasp_close.blocked_motor_ids,
            )
            retention_request_path = task_run / "retention_route_request.json"
            retention_route_path = task_run / "retention_route_validation.json"
            retention_request.write_json(retention_request_path)
            try:
                planner.request(
                    "validate-retention-route",
                    request_path=retention_request_path,
                    output_path=retention_route_path,
                    control_check=driver.check,
                )
            except (PlannerRequestRejected, RuntimeError) as error:
                # If this was actually a controller fault, preserve the fail-closed
                # path. Otherwise the frozen open-hand reverse route remains valid.
                driver.check()
                print(
                    f"TASK REJECTED — measured close route is unavailable: {error}",
                    flush=True,
                )
                return_after_task_rejection(from_test_lift=False)
                raise TabletopTaskRejected(
                    f"measured close route is unavailable: {error}"
                ) from error
            retention_route = RetentionRouteValidationResult.from_json(retention_route_path)
            if retention_route.request_sha256 != retention_request.content_sha256:
                raise RuntimeError("retention-route validation belongs to another request")
            try:
                dex_controller.check_retention_test()
            except Dex3RetentionLostError as error:
                print(f"TASK REJECTED — cube contact was lost before lift: {error}")
                return_after_task_rejection(from_test_lift=False)
                raise TabletopTaskRejected(
                    f"cube contact was lost before lift: {error}"
                ) from error
            print(
                "GRASP CLOSE STABILIZED — thumb and opposing-finger obstruction relative "
                "to commissioned empty close passed the frozen payload-route collision "
                "recheck; pressure does not decide retention; beginning the "
                f"{retention_test_lift_mm:.1f} mm retention checkpoint within the payload lift",
                flush=True,
            )
            execute_phase("retention_test_lift")
            try:
                retention_evidence = dex_controller.verify_retention_at_lifted_checkpoint(
                    safety_heartbeat=lambda: (driver.check(), guard.pulse()),
                )
                dex_controller.finish_retention_test()
            except Dex3RetentionLostError as error:
                driver.check()
                print(
                    "TASK REJECTED — cube contact did not survive the lifted checkpoint; "
                    "holding the close target during the exact low-lift reverse and opening "
                    f"only after returning to support: {error}"
                )
                return_after_task_rejection(from_test_lift=True)
                raise TabletopTaskRejected(
                    f"cube contact did not survive lift checkpoint: {error}"
                ) from error
            atomic_write_json(
                task_run / "retention_evidence.json",
                retention_evidence.to_dict(),
            )
            print(
                "RETENTION TEST PASSED — stable thumb and opposing-finger obstruction "
                "relative to commissioned empty close remained after separation from "
                "the support; continuing the payload lift",
                flush=True,
            )
            execute_phase("payload_lift")
            execute_phase("payload_lower")
            execute_phase("payload_replace")
            open_active_hand("cube release after exact replacement")
            execute_phase("grasp_retreat")
            execute_return_to_clearance(use_mpc=True)
            _command_fingers(
                dex_controller,
                driver,
                guard,
                left=initial_left,
                right=initial_right,
                label="initial finger posture restoration",
            )
            # The arm starts and ends physically supported by the table. Use
            # the supported escape's exact validated reverse here, matching
            # the fixed outbound escape instead of substituting an MPC model
            # built from the later clearance body snapshot.
            execute_phase("__handoff__", use_mpc=False)
            if grasp_close is None or retention_evidence is None or retention_route is None:
                raise RuntimeError("tabletop lifecycle ended without retention evidence")
            _restore_seated_control(
                driver=driver,
                dex_controller=dex_controller,
                guard=guard,
                synchronized=synchronized,
            )
            status = {
                "status": "completed",
                "commands_robot": True,
                "arm": arm,
                "presentation_id": presentation.presentation_id,
                "selected_candidate_id": task.selected_candidate_id,
                "supported_escape_plan_sha256": escape.content_sha256,
                "active_plan_sha256": active_plan_sha256,
                "active_plan_kind": (
                    task.kind if args.motion_controller == "mpc" else execution.kind
                ),
                "pregrasp_plan_sha256": (
                    pregrasp_plan.content_sha256
                    if args.motion_controller == "trajectory"
                    else None
                ),
                "cube_anchor_motion": cube_anchor_motion,
                "pregrasp_camera_state_estimate": (
                    None if camera_state_estimate is None else camera_state_estimate.to_dict()
                ),
                "motion_controller": args.motion_controller,
                "mpc_phase_count": len(mpc_phases),
                "mpc_window_count": sum(
                    len(document["windows"]) for document in mpc_phases.values()
                ),
                "grasp_close": grasp_close.to_dict(),
                "retention_evidence": retention_evidence.to_dict(),
                "retention_route_validation_sha256": retention_route.content_sha256,
                "terminal_action": guard.terminal_action,
                "table_collision_policy": "local_manipulation_geometry_plane",
                "open_transit_table_patch_dimensions_m": list(patch_dimensions),
                "minimum_hand_plane_clearance_m": (
                    clearance_request.minimum_hand_plane_clearance_m
                ),
                "calibration_validation_claim": False,
            }
            print(
                "TABLETOP TASK PASSED — lifted opposed joint obstruction survived a "
                f"{retention_test_lift_mm:.1f} mm retention checkpoint; cube then completed the "
                f"{payload_lift_mm:.1f} mm lift, was replaced, and the arm "
                "returned to its supported start, and seated FSM 3 restored",
                flush=True,
            )
        finally:
            # Robot ownership is resolved by the outer lifecycle handlers. ROS
            # teardown must remain later because participant destruction can
            # block Python callbacks long enough to stale the control state.
            pass
    except TabletopTaskRejected as rejection:
        try:
            _restore_seated_control(
                driver=driver,
                dex_controller=dex_controller,
                guard=guard,
                synchronized=synchronized,
            )
        except BaseException as error:
            primary_error = error
            status = {
                "status": "failed",
                "commands_robot": bool(transport is not None and transport.command_count),
                "arm": arm,
                "presentation_id": presentation.presentation_id,
                "motion_controller": args.motion_controller,
                "error_type": type(error).__name__,
                "error": f"task rejection recovery failed after {rejection}: {error}",
            }
            raise
        status = {
            "status": "task_rejected",
            "commands_robot": bool(transport is not None and transport.command_count),
            "arm": arm,
            "presentation_id": presentation.presentation_id,
            "motion_controller": args.motion_controller,
            "reason": str(rejection),
            "terminal_action": guard.terminal_action,
            "frozen_reverse_return_completed": rejection_return_completed,
            "calibration_validation_claim": False,
        }
        return_description = (
            "the arm returned through the frozen reverse route"
            if rejection_return_completed
            else "the arm remained at the supported ownership pose"
        )
        print(
            f"TABLETOP TASK REJECTED — {return_description} and seated FSM 3 was "
            "restored; no emergency zero-torque transition was requested. Reason: "
            f"{rejection}",
            flush=True,
        )
    except BaseException as error:
        primary_error = error
        status = {
            "status": "failed",
            "commands_robot": bool(transport is not None and transport.command_count),
            "arm": arm,
            "presentation_id": presentation.presentation_id,
            "motion_controller": args.motion_controller,
            "error_type": type(error).__name__,
            "error": str(error),
        }
        raise
    finally:
        cleanup_errors: list[str] = []
        if driver is not None and driver.is_alive:
            try:
                driver.close()
            except BaseException as error:  # noqa: BLE001
                cleanup_errors.append(f"driver: {error}")
        if dex_controller is not None and not dex_controller.timed_out:
            try:
                dex_controller.timeout()
            except BaseException as error:  # noqa: BLE001
                cleanup_errors.append(f"Dex3 timeout: {error}")
        if (
            guard is not None
            and transport is not None
            and (guard.armed or transport.requires_external_takeover)
        ):
            try:
                if guard.armed:
                    guard.restore_zero_torque("tabletop task failed or was interrupted")
                synchronized.confirm_external_takeover(
                    "PC2 verified AI zero-torque takeover after tabletop failure"
                )
            except BaseException as error:  # noqa: BLE001
                cleanup_errors.append(f"PC2 takeover: {error}")
        if planner is not None:
            try:
                planner.close()
            except BaseException as error:  # noqa: BLE001
                cleanup_errors.append(f"planner: {error}")
        if dex_controller is not None:
            try:
                if not dex_controller.timed_out and guard is not None and guard.terminal_action:
                    dex_controller.close_after_external_timeout()
                else:
                    dex_controller.close()
            except BaseException as error:  # noqa: BLE001
                cleanup_errors.append(f"Dex3 close: {error}")
        for name, value in (
            ("torso IMU observer", torso_observer),
            ("lowstate observer", observer),
            ("Dex3 observer", dex_observer),
        ):
            if value is not None:
                try:
                    value.close()
                except BaseException as error:  # noqa: BLE001
                    cleanup_errors.append(f"{name}: {error}")
        try:
            _teardown_ros_runtime(
                no_window=args.no_window,
                camera=camera,
                node=node,
                rclpy_module=rclpy_module,
                transport=transport,
                synchronized=synchronized,
            )
            camera = None
            node = None
        except BaseException as error:  # noqa: BLE001
            cleanup_errors.append(f"ROS teardown: {error}")
        command_lock.release()
        if raw_recorder is not None and (raw_recorder.started or raw_recorder.summary is not None):
            try:
                status["recording"] = raw_recorder.stop()
            except BaseException as error:  # noqa: BLE001
                cleanup_errors.append(f"raw episode recorder: {error}")
                status["recording"] = {
                    "state": "finalization_failed",
                    "complete": False,
                    "episode_directory": str(raw_recorder.episode_directory),
                    "error": str(error),
                }
        status["cleanup_errors"] = cleanup_errors
        status.setdefault(
            "minimum_hand_plane_clearance_m",
            float(task_config["table"]["minimum_hand_plane_clearance_m"]),
        )
        if cube_anchor_motion is not None:
            status.setdefault("cube_anchor_motion", cube_anchor_motion)
        if camera_state_estimate is not None:
            status.setdefault(
                "pregrasp_camera_state_estimate",
                camera_state_estimate.to_dict(),
            )
        if args.motion_controller == "mpc":
            correction_windows = [
                window
                for phase in mpc_phases.values()
                for window in phase["windows"]
                if "camera_state_correction" in window["diagnostics"]
            ]
            status["mpc_camera_state_correction"] = {
                "enabled": True,
                "estimator": "hybrid_pelvis_position_torso_orientation",
                "visual_anchor": "fixed_cube_at_clearance",
                "corrected_window_count": len(correction_windows),
                "maximum_goal_translation_correction_m": (
                    max(
                        float(window["diagnostics"]["goal_translation_correction_m"])
                        for window in correction_windows
                    )
                    if correction_windows
                    else None
                ),
                "maximum_goal_rotation_correction_deg": (
                    max(
                        float(window["diagnostics"]["goal_rotation_correction_deg"])
                        for window in correction_windows
                    )
                    if correction_windows
                    else None
                ),
            }
        try:
            if preflight_frames:
                _save_frames(task_run / "preflight", preflight_frames)
            if loaded_frames:
                _save_frames(task_run / "loaded_observation", loaded_frames)
            if clearance_frames:
                _save_frames(task_run / "clearance_observation", clearance_frames)
            if cube_anchor_motion is not None:
                atomic_write_json(task_run / "cube_anchor_motion.json", cube_anchor_motion)
            if mpc_phases:
                phase_plan_sha256 = {
                    name: document["preparation"]["plan_sha256"]
                    for name, document in mpc_phases.items()
                    if document["preparation"] is not None
                }
                atomic_write_json(
                    task_run / "mpc_lifecycle.json",
                    {
                        "schema_version": 2,
                        "controller": "curobo_mpc",
                        "camera_state_correction": status["mpc_camera_state_correction"],
                        "run_status": status["status"],
                        "plan_sha256s": sorted(set(phase_plan_sha256.values())),
                        "phase_plan_sha256": phase_plan_sha256,
                        "phase_order": list(mpc_phases),
                        "phases": mpc_phases,
                    },
                )
            atomic_write_json(task_run / "status.json", status)
        except BaseException as error:
            if primary_error is None:
                raise
            print(f"warning: failed to write complete failure artifacts: {error}", file=sys.stderr)
    print(json.dumps({"run": str(task_run), **status}, indent=2, sort_keys=True))
    return 0
