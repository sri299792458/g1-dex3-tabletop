"""Exact-model CuRobo MPC for the validated open-hand approach route.

This module intentionally starts with the free-space ``clearance`` to
``move_to_pregrasp`` phase.  The grasp/contact and payload state transitions
remain with the already commissioned finite task state machine.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation

from g1_aprilcube_calibration.joint_map import arm_joint_names
from g1_aprilcube_calibration.transforms import invert_transform, validate_transform
from g1_dex3_tabletop.mpc_command_buffer import MPCCommandWindow
from g1_dex3_tabletop.planning.curobo_backend import COLLISION_ACTIVATION_DISTANCE_M
from g1_dex3_tabletop.planning.g1_model import (
    CUROBO_COMMIT,
    build_tabletop_robot_config,
    command_from_model_q,
    grasp_frame,
    model_source_hashes,
)
from g1_dex3_tabletop.planning.tabletop_planner import (
    _base_scene,
    _selected_open_transit_world_robot,
    _use_moving_grasp_frame_only,
)
from g1_dex3_tabletop.tabletop_contracts import TabletopExecutionPlan, TabletopTaskRequest

MPC_OPTIMIZATION_DT_S = 0.04
MPC_INTERPOLATION_STEPS = 4
MPC_DOCUMENTED_COMMAND_DT_S = MPC_OPTIMIZATION_DT_S / MPC_INTERPOLATION_STEPS
MPC_COLD_START_ITERATIONS = 200
MPC_WARM_START_ITERATIONS = 100
MPC_EXPOSED_INTERPOLATION_WINDOWS = 3
MPC_ROUTE_PHASE = "move_to_pregrasp"


def _numpy(value) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=np.float64)


def _joint_state(device_cfg, q: np.ndarray, dq: np.ndarray, names: tuple[str, ...]):
    import torch
    from curobo.types import JointState

    state = JointState.from_position(
        device_cfg.to_device(np.asarray(q, dtype=np.float64)).unsqueeze(0),
        joint_names=list(names),
    )
    state.velocity = device_cfg.to_device(np.asarray(dq, dtype=np.float64)).unsqueeze(0)
    state.acceleration = torch.zeros_like(state.position)
    return state


def _matrix_from_pose_list(value: Any) -> np.ndarray:
    pose = np.asarray(value, dtype=np.float64).reshape(-1)
    if pose.shape != (7,) or not np.all(np.isfinite(pose)):
        raise ValueError("CuRobo scene pose must contain seven finite values")
    result = np.eye(4, dtype=np.float64)
    result[:3, 3] = pose[:3]
    result[:3, :3] = Rotation.from_quat(pose[[4, 5, 6, 3]]).as_matrix()
    return validate_transform(result)


def _rigid_transform(value: Any) -> np.ndarray:
    """Project accumulated float32 transform arithmetic back onto SE(3)."""

    result = np.asarray(value, dtype=np.float64).copy()
    if result.shape != (4, 4) or not np.all(np.isfinite(result)):
        raise ValueError("rigid transform must be a finite 4x4 matrix")
    result[:3, :3] = Rotation.from_matrix(result[:3, :3]).as_matrix()
    result[3] = (0.0, 0.0, 0.0, 1.0)
    return validate_transform(result)


def _resolved_robot_with_velocity_limit(
    robot: dict[str, Any],
    *,
    device_cfg,
    maximum_velocity_rad_s: float,
):
    """Resolve the CuRobo model once, then set the actual active-joint bound.

    CuRobo's dictionary ``velocity_scale`` is applied while the URDF is loaded
    and again while its reduced kinematic parameters are constructed in the
    pinned revision.  Mutating the resolved ``RobotCfg`` avoids relying on a
    square-root scale workaround and makes the effective limit directly
    inspectable as exactly the task's commissioned velocity ceiling.
    """

    from curobo._src.types.robot import RobotCfg

    if not np.isfinite(maximum_velocity_rad_s) or maximum_velocity_rad_s <= 0.0:
        raise ValueError("MPC maximum velocity must be positive and finite")
    resolved = RobotCfg.create(robot, device_cfg)
    limits = resolved.kinematics.kinematics_config.joint_limits
    limits.velocity[0, :].fill_(-float(maximum_velocity_rad_s))
    limits.velocity[1, :].fill_(float(maximum_velocity_rad_s))
    return resolved


def _base_T_torso(robot: dict[str, Any], q: np.ndarray, *, arm: str, device_cfg) -> np.ndarray:
    """Read the locked torso pose without constructing a second optimizer."""

    import torch
    from curobo._src.types.robot import RobotCfg
    from curobo.kinematics import Kinematics
    from curobo.types import JointState

    resolved = RobotCfg.create(robot, device_cfg)
    kinematics = Kinematics(resolved.kinematics)
    state = JointState.from_position(
        device_cfg.to_device(np.asarray(q, dtype=np.float64)).unsqueeze(0),
        joint_names=list(arm_joint_names(arm)),
    )
    state.velocity = torch.zeros_like(state.position)
    state.acceleration = torch.zeros_like(state.position)
    result = (
        kinematics.compute_kinematics(state)
        .tool_poses["torso_link"]
        .get_matrix()[0]
        .detach()
        .cpu()
        .numpy()
    )
    # CuRobo evaluates FK in float32 on the GPU. Project the tiny numerical
    # drift back onto SO(3) before crossing into the strict float64 geometry
    # contracts used by the controller process.
    result[:3, :3] = Rotation.from_matrix(result[:3, :3]).as_matrix()
    return result


@dataclass(frozen=True, slots=True)
class MPCBenchmarkConfig:
    maximum_steps: int = 300
    waypoint_tolerance_rad: float = 0.005
    replan_lead_s: float = 0.1

    def __post_init__(self) -> None:
        if self.maximum_steps <= 0:
            raise ValueError("MPC benchmark step count must be positive")
        if not np.isfinite(self.waypoint_tolerance_rad) or self.waypoint_tolerance_rad <= 0.0:
            raise ValueError("MPC waypoint tolerance must be positive and finite")
        if not np.isfinite(self.replan_lead_s) or self.replan_lead_s <= 0.0:
            raise ValueError("MPC replan lead must be positive and finite")


class TabletopOpenApproachMPC:
    """Warm-start MPC instance bound to one frozen task plan and scene."""

    def __init__(self, request: TabletopTaskRequest, execution: TabletopExecutionPlan) -> None:
        import torch
        from curobo.model_predictive_control import (
            ModelPredictiveControl,
            ModelPredictiveControlCfg,
        )
        from curobo.types import DeviceCfg

        if request.content_sha256 != execution.clearance_request_sha256:
            raise ValueError("MPC request differs from the frozen clearance request")
        if request.arm != execution.task.arm:
            raise ValueError("MPC request and execution plan select different arms")
        self.request = request
        self.execution = execution
        self.arm = request.arm
        self.names = arm_joint_names(self.arm)
        self.route = execution.task.trajectories[0]
        if self.route.from_pose_id != "clearance" or self.route.to_pose_id != MPC_ROUTE_PHASE:
            raise ValueError("frozen plan has no clearance-to-pregrasp route")
        self.path_model_q = np.asarray(self.route.model_q_rad, dtype=np.float64)
        self.device_cfg = DeviceCfg(device=torch.device("cuda:0"), dtype=torch.float32)

        robot, _reference = build_tabletop_robot_config(
            arm=self.arm,
            snapshot=request.observation.snapshot,
            joint_position_offsets_rad=request.joint_position_offsets_rad,
            active_finger_q_rad=execution.task.open_active_dex3_q_rad,
        )
        base_T_torso = _base_T_torso(
            robot,
            self.path_model_q[0],
            arm=self.arm,
            device_cfg=self.device_cfg,
        )
        _use_moving_grasp_frame_only(robot, arm=self.arm)
        robot = _selected_open_transit_world_robot(robot, arm=self.arm)
        resolved_robot = _resolved_robot_with_velocity_limit(
            robot,
            device_cfg=self.device_cfg,
            maximum_velocity_rad_s=request.maximum_arm_velocity_rad_s,
        )
        scene = _base_scene(
            request,
            base_T_torso,
            include_cube=True,
            include_open_transit_table_patch=True,
        )
        self._base_T_torso0 = _rigid_transform(base_T_torso)
        self._reference_T_torso0 = _rigid_transform(
            invert_transform(np.asarray(request.observation.camera_T_object, dtype=np.float64))
            @ invert_transform(np.asarray(request.torso_T_camera, dtype=np.float64))
        )
        torso0_T_base = invert_transform(self._base_T_torso0)
        self._reference_T_obstacles: dict[str, np.ndarray] = {}
        for obstacle_group in ("cuboid", "mesh"):
            for name, obstacle in scene.get(obstacle_group, {}).items():
                base_T_obstacle = _matrix_from_pose_list(obstacle["pose"])
                self._reference_T_obstacles[name] = _rigid_transform(
                    self._reference_T_torso0 @ torso0_T_base @ base_T_obstacle
                )
        cfg = ModelPredictiveControlCfg.create(
            robot=resolved_robot,
            scene_model=scene,
            collision_cache={"cuboid": 4, "mesh": 1},
            device_cfg=self.device_cfg,
            use_cuda_graph=True,
            optimization_dt=MPC_OPTIMIZATION_DT_S,
            interpolation_steps=MPC_INTERPOLATION_STEPS,
            optimizer_collision_activation_distance=COLLISION_ACTIVATION_DISTANCE_M,
            position_tolerance=0.005,
            orientation_tolerance=0.05,
            cold_start_optimization_num_iters=MPC_COLD_START_ITERATIONS,
            warm_start_optimization_num_iters=MPC_WARM_START_ITERATIONS,
            use_deceleration_on_failure=True,
            random_seed=request.random_seed,
        )
        self.mpc = ModelPredictiveControl(cfg)
        execution_manager = self.mpc.trajectory_execution_manager
        execution_manager.command_end_idx = (
            execution_manager.command_start_idx
            + MPC_EXPOSED_INTERPOLATION_WINDOWS * MPC_INTERPOLATION_STEPS
        )
        effective = _numpy(self.mpc.kinematics.get_joint_limits().velocity[1])
        if not np.allclose(
            effective,
            request.maximum_arm_velocity_rad_s,
            atol=1.0e-7,
            rtol=0.0,
        ):
            raise RuntimeError(f"CuRobo MPC effective velocity limits are {effective.tolist()}")
        self._setup = False
        self._generation = 0
        self._route_progress_index = 0
        self._active_goal_pose: np.ndarray | None = None
        self._active_goal_is_corrected = False
        self._lookahead_rad = (
            request.maximum_arm_velocity_rad_s * self.mpc.action_horizon * MPC_OPTIMIZATION_DT_S
        )

    def setup(self, *, model_q_rad: np.ndarray, model_dq_rad_s: np.ndarray) -> float:
        state = _joint_state(self.device_cfg, model_q_rad, model_dq_rad_s, self.names)
        started = time.perf_counter()
        self.mpc.setup(state)
        import torch

        torch.cuda.synchronize()
        self._setup = True
        return time.perf_counter() - started

    def setup_at_frozen_route_start(self) -> float:
        """Build CUDA graphs before any live state freshness clock starts."""

        return self.setup(
            model_q_rad=self.path_model_q[0],
            model_dq_rad_s=np.zeros(7, dtype=np.float64),
        )

    def tool_pose(self, model_q_rad: np.ndarray) -> np.ndarray:
        state = _joint_state(
            self.device_cfg,
            model_q_rad,
            np.zeros(7, dtype=np.float64),
            self.names,
        )
        result = (
            self.mpc.compute_kinematics(state)
            .tool_poses[grasp_frame(self.arm)]
            .get_matrix()[0]
            .detach()
            .cpu()
            .numpy()
        )
        result[:3, :3] = Rotation.from_matrix(result[:3, :3]).as_matrix()
        return result

    def update_nominal_goal(self, model_q_rad: np.ndarray) -> None:
        """Track one local pose/joint waypoint from the validated route."""

        from curobo.types import GoalToolPose, Pose

        q = np.asarray(model_q_rad, dtype=np.float64).reshape(-1)
        if q.shape != (7,) or not np.all(np.isfinite(q)):
            raise ValueError("MPC goal must contain seven finite model coordinates")
        goal_state = _joint_state(self.device_cfg, q, np.zeros(7), self.names)
        goal_matrix = self.tool_pose(q)
        goal_pose = Pose.from_matrix(self.device_cfg.to_device(goal_matrix[None]))
        goals = GoalToolPose.from_poses(
            {grasp_frame(self.arm): goal_pose},
            ordered_tool_frames=[grasp_frame(self.arm)],
            num_goalset=1,
        )
        if not self.mpc.update_goal_tool_poses(goals, run_ik=False):
            raise RuntimeError("CuRobo MPC rejected the nominal tool-pose goal")
        self.mpc.update_goal_state(goal_state)
        self.mpc.enable_joint_position_tracking()
        self._active_goal_pose = goal_matrix
        self._active_goal_is_corrected = False

    def update_anchored_goal(
        self,
        model_q_rad: np.ndarray,
        *,
        reference_T_camera: np.ndarray,
    ) -> dict[str, float]:
        """Move the local goal and scene with the estimated live torso pose."""

        from curobo.types import GoalToolPose, Pose

        reference_T_camera = validate_transform(np.asarray(reference_T_camera, dtype=np.float64))
        reference_T_torso = _rigid_transform(
            reference_T_camera
            @ invert_transform(np.asarray(self.request.torso_T_camera, dtype=np.float64))
        )
        nominal_base_T_goal = self.tool_pose(model_q_rad)
        reference_T_goal = _rigid_transform(
            self._reference_T_torso0 @ invert_transform(self._base_T_torso0) @ nominal_base_T_goal
        )
        corrected_base_T_goal = _rigid_transform(
            self._base_T_torso0 @ invert_transform(reference_T_torso) @ reference_T_goal
        )
        for name, reference_T_obstacle in self._reference_T_obstacles.items():
            corrected_base_T_obstacle = _rigid_transform(
                self._base_T_torso0 @ invert_transform(reference_T_torso) @ reference_T_obstacle
            )
            self.mpc.scene_collision_checker.update_obstacle_pose(
                name,
                Pose.from_matrix(self.device_cfg.to_device(corrected_base_T_obstacle[None])),
            )
        goal_pose = Pose.from_matrix(self.device_cfg.to_device(corrected_base_T_goal[None]))
        goals = GoalToolPose.from_poses(
            {grasp_frame(self.arm): goal_pose},
            ordered_tool_frames=[grasp_frame(self.arm)],
            num_goalset=1,
        )
        if not self.mpc.update_goal_tool_poses(goals, run_ik=True):
            raise RuntimeError("CuRobo MPC could not solve the state-corrected local goal")
        self._active_goal_pose = corrected_base_T_goal
        self._active_goal_is_corrected = True
        correction = corrected_base_T_goal @ invert_transform(nominal_base_T_goal)
        return {
            "goal_translation_correction_m": float(np.linalg.norm(correction[:3, 3])),
            "goal_rotation_correction_deg": float(
                np.degrees(Rotation.from_matrix(correction[:3, :3]).magnitude())
            ),
        }

    def next_nominal_window(
        self,
        *,
        measured_command_q_rad: np.ndarray,
        measured_dq_rad_s: np.ndarray,
        active_command_q_rad: np.ndarray,
        state_monotonic_s: float,
        reference_T_camera: np.ndarray | None = None,
    ) -> MPCCommandWindow:
        """Advance along the frozen route by one collision-checked MPC horizon."""

        measured_command = np.asarray(measured_command_q_rad, dtype=np.float64).reshape(-1)
        if measured_command.shape != (7,) or not np.all(np.isfinite(measured_command)):
            raise ValueError("measured MPC arm position must contain seven finite values")
        model_q = np.asarray(
            [
                value + self.request.joint_position_offsets_rad.get(name, 0.0)
                for name, value in zip(self.names, measured_command, strict=True)
            ],
            dtype=np.float64,
        )
        remaining = self.path_model_q[self._route_progress_index :]
        self._route_progress_index += int(
            np.argmin(np.max(np.abs(remaining - model_q[None, :]), axis=1))
        )
        waypoint_index = len(self.path_model_q) - 1
        for index in range(self._route_progress_index + 1, len(self.path_model_q)):
            if float(np.max(np.abs(self.path_model_q[index] - model_q))) >= (self._lookahead_rad):
                waypoint_index = index
                break
        goal_q = self.path_model_q[waypoint_index]
        state_correction: dict[str, float] = {}
        if reference_T_camera is None:
            self.update_nominal_goal(goal_q)
        else:
            state_correction = self.update_anchored_goal(
                goal_q,
                reference_T_camera=reference_T_camera,
            )
        requested_terminal = waypoint_index == len(self.path_model_q) - 1
        window = self.solve_window(
            model_q_rad=model_q,
            model_dq_rad_s=np.asarray(measured_dq_rad_s, dtype=np.float64),
            active_command_q_rad=np.asarray(active_command_q_rad, dtype=np.float64),
            state_monotonic_s=state_monotonic_s,
            terminal=requested_terminal,
        )
        if requested_terminal:
            terminal_command = np.asarray(window.command_q_rad[-1], dtype=np.float64)
            terminal_model = np.asarray(
                [
                    value + self.request.joint_position_offsets_rad.get(name, 0.0)
                    for name, value in zip(self.names, terminal_command, strict=True)
                ]
            )
            if self._active_goal_is_corrected:
                if self._active_goal_pose is None:
                    raise RuntimeError("MPC terminal check has no active tool-pose goal")
                terminal_pose = self.tool_pose(terminal_model)
                translation_error = float(
                    np.linalg.norm(terminal_pose[:3, 3] - self._active_goal_pose[:3, 3])
                )
                rotation_error = float(
                    Rotation.from_matrix(
                        terminal_pose[:3, :3].T @ self._active_goal_pose[:3, :3]
                    ).magnitude()
                )
                terminal = translation_error <= 0.005 and rotation_error <= 0.05
            else:
                terminal = float(np.max(np.abs(terminal_model - self.path_model_q[-1]))) <= 0.005
                translation_error = None
                rotation_error = None
            if terminal != window.terminal:
                values = window.to_dict(include_hash=False)
                values["terminal"] = terminal and window.feasible
                values["diagnostics"] = {
                    **window.diagnostics,
                    "terminal_translation_error_m": translation_error,
                    "terminal_rotation_error_rad": rotation_error,
                }
                window = MPCCommandWindow.from_dict(values)
        if state_correction:
            values = window.to_dict(include_hash=False)
            values["diagnostics"] = {**window.diagnostics, **state_correction}
            window = MPCCommandWindow.from_dict(values)
        return window

    def solve_window(
        self,
        *,
        model_q_rad: np.ndarray,
        model_dq_rad_s: np.ndarray,
        active_command_q_rad: np.ndarray,
        state_monotonic_s: float,
        terminal: bool,
    ) -> MPCCommandWindow:
        """Optimize one window and reject, rather than expose, infeasible output."""

        import torch

        if not self._setup:
            raise RuntimeError("MPC must be set up before solving")
        current = _joint_state(self.device_cfg, model_q_rad, model_dq_rad_s, self.names)
        started = time.perf_counter()
        result = self.mpc.optimize_action_sequence(current)
        torch.cuda.synchronize()
        wall_s = time.perf_counter() - started
        sequence = result.action_sequence
        if sequence is None:
            model_commands = np.asarray(model_q_rad, dtype=np.float64)[None, :]
            returned_state_dt_s = MPC_OPTIMIZATION_DT_S
        else:
            model_commands = _numpy(sequence.position).reshape(-1, 7)
            returned_dt = _numpy(sequence.dt).reshape(-1)
            if len(returned_dt) == 0 or not np.all(np.isfinite(returned_dt)):
                raise RuntimeError("CuRobo MPC returned no finite JointState dt")
            if not np.allclose(returned_dt, returned_dt[0], atol=1.0e-9, rtol=0.0):
                raise RuntimeError(
                    f"CuRobo MPC returned nonuniform JointState dt {returned_dt.tolist()}"
                )
            returned_state_dt_s = float(returned_dt[0])
            if returned_state_dt_s <= 0.0:
                raise RuntimeError("CuRobo MPC returned a non-positive JointState dt")
        command_start = np.asarray(active_command_q_rad, dtype=np.float64).reshape(-1)
        if command_start.shape != (7,) or not np.all(np.isfinite(command_start)):
            raise ValueError("active MPC command must contain seven finite values")
        command_sequence = np.stack(
            [
                command_from_model_q(
                    row,
                    arm=self.arm,
                    joint_position_offsets_rad=self.request.joint_position_offsets_rad,
                )
                for row in model_commands
            ]
        )
        commands = np.concatenate((command_start[None, :], command_sequence), axis=0)
        first_command_index = self.mpc.trajectory_execution_manager.command_start_idx
        times = np.concatenate(
            (
                np.asarray([0.0], dtype=np.float64),
                (first_command_index + np.arange(len(command_sequence), dtype=np.float64))
                * returned_state_dt_s,
            )
        )
        feasible = bool(result.success is not None and bool(result.success.reshape(-1)[0].item()))
        curobo_feasible = feasible
        peak_velocity = float(np.max(np.abs(np.diff(commands, axis=0)) / np.diff(times)[:, None]))
        if peak_velocity > self.request.maximum_arm_velocity_rad_s + 1.0e-6:
            feasible = False
        diagnostics = {
            "curobo_feasible": curobo_feasible,
            "wall_time_s": wall_s,
            "curobo_reported_solve_time_s": float(result.solve_time),
            "peak_velocity_rad_s": peak_velocity,
            "reported_peak_velocity_rad_s": (
                None
                if sequence is None or sequence.velocity is None
                else float(np.max(np.abs(_numpy(sequence.velocity))))
            ),
            "reported_sequence_dt": (
                None
                if sequence is None or sequence.dt is None
                else _numpy(sequence.dt).reshape(-1).tolist()
            ),
            "position_error_m": (
                None
                if result.position_error is None
                else float(result.position_error.reshape(-1)[0].item())
            ),
            "rotation_error_rad": (
                None
                if result.rotation_error is None
                else float(result.rotation_error.reshape(-1)[0].item())
            ),
        }
        window = MPCCommandWindow(
            generation=self._generation,
            plan_sha256=self.execution.content_sha256,
            state_monotonic_s=state_monotonic_s,
            sample_time_s=tuple(times),
            command_q_rad=tuple(tuple(float(value) for value in row) for row in commands),
            feasible=feasible,
            terminal=bool(terminal and feasible),
            solve_time_s=wall_s,
            diagnostics=diagnostics,
        )
        self._generation += 1
        return window

    def close(self) -> None:
        self.mpc.destroy()


def benchmark_open_approach_mpc(
    request: TabletopTaskRequest,
    execution: TabletopExecutionPlan,
    *,
    config: MPCBenchmarkConfig | None = None,
) -> dict[str, Any]:
    """Run a deterministic, command-free MPC replay of the retained approach."""

    import torch

    config = config or MPCBenchmarkConfig()
    started = time.perf_counter()
    controller = TabletopOpenApproachMPC(request, execution)
    route_q = controller.path_model_q
    q = route_q[0].copy()
    dq = np.zeros(7, dtype=np.float64)
    command_q = command_from_model_q(
        q,
        arm=request.arm,
        joint_position_offsets_rad=request.joint_position_offsets_rad,
    )
    setup_s = controller.setup(model_q_rad=q, model_dq_rad_s=dq)
    lookahead_rad = controller._lookahead_rad
    accepted = 0
    rejected = 0
    latencies: list[float] = []
    peak_velocities: list[float] = []
    first_rejection: dict[str, Any] | None = None
    simulated_time_s = 0.0
    try:
        for _step in range(config.maximum_steps):
            window = controller.next_nominal_window(
                measured_command_q_rad=command_q,
                measured_dq_rad_s=dq,
                active_command_q_rad=command_q,
                state_monotonic_s=time.monotonic(),
            )
            latencies.append(window.solve_time_s)
            peak_velocities.append(window.peak_velocity_rad_s())
            if not window.feasible:
                rejected += 1
                if first_rejection is None:
                    first_rejection = window.to_dict()
                break
            accepted += 1
            execution_time_s = (
                window.duration_s
                if window.terminal
                else max(window.duration_s - config.replan_lead_s, 0.0)
            )
            window_times = np.asarray(window.sample_time_s, dtype=np.float64)
            window_commands = np.asarray(window.command_q_rad, dtype=np.float64)
            command_q = np.asarray(
                [
                    np.interp(execution_time_s, window_times, window_commands[:, index])
                    for index in range(7)
                ],
                dtype=np.float64,
            )
            q = np.asarray(
                [
                    value + request.joint_position_offsets_rad.get(name, 0.0)
                    for name, value in zip(controller.names, command_q, strict=True)
                ],
                dtype=np.float64,
            )
            upper = int(np.searchsorted(window_times, execution_time_s, side="right"))
            upper = min(max(upper, 1), len(window_times) - 1)
            lower = upper - 1
            dq = (window_commands[upper] - window_commands[lower]) / (
                window_times[upper] - window_times[lower]
            )
            simulated_time_s += execution_time_s
            if window.terminal:
                break
    finally:
        controller.close()
        torch.cuda.empty_cache()
    reached = float(np.max(np.abs(q - route_q[-1]))) <= config.waypoint_tolerance_rad
    latency = np.asarray(latencies, dtype=np.float64)
    return {
        "schema_version": 1,
        "kind": "g1_tabletop_open_approach_mpc_benchmark",
        "commands_robot": False,
        "request_sha256": request.content_sha256,
        "execution_plan_sha256": execution.content_sha256,
        "arm": request.arm,
        "phase": MPC_ROUTE_PHASE,
        "reached_terminal": reached,
        "accepted_windows": accepted,
        "rejected_windows": rejected,
        "first_rejection": first_rejection,
        "maximum_joint_error_to_terminal_rad": float(np.max(np.abs(q - route_q[-1]))),
        "simulated_time_s": simulated_time_s,
        "setup_time_s": setup_s,
        "benchmark_wall_time_s": time.perf_counter() - started,
        "solve_latency_s": {
            "count": len(latencies),
            "mean": float(np.mean(latency)) if len(latency) else None,
            "p95": float(np.percentile(latency, 95)) if len(latency) else None,
            "maximum": float(np.max(latency)) if len(latency) else None,
        },
        "maximum_window_velocity_rad_s": (
            float(np.max(peak_velocities)) if peak_velocities else None
        ),
        "configuration": {
            "optimization_dt_s": MPC_OPTIMIZATION_DT_S,
            "interpolation_steps": MPC_INTERPOLATION_STEPS,
            "exposed_interpolation_windows": MPC_EXPOSED_INTERPOLATION_WINDOWS,
            "documented_command_dt_s": MPC_DOCUMENTED_COMMAND_DT_S,
            "cold_start_iterations": MPC_COLD_START_ITERATIONS,
            "warm_start_iterations": MPC_WARM_START_ITERATIONS,
            "maximum_steps": config.maximum_steps,
            "waypoint_tolerance_rad": config.waypoint_tolerance_rad,
            "replan_lead_s": config.replan_lead_s,
            "maximum_arm_velocity_rad_s": request.maximum_arm_velocity_rad_s,
            "route_lookahead_rad": lookahead_rad,
            "collision_activation_distance_m": COLLISION_ACTIVATION_DISTANCE_M,
        },
        "provenance": {**model_source_hashes(), "curobo_commit": CUROBO_COMMIT},
    }


def benchmark_from_paths(
    request_path: Path,
    execution_path: Path,
    output_path: Path,
    *,
    maximum_steps: int,
) -> dict[str, Any]:
    from g1_dex3_tabletop.planning.contracts import atomic_write_json

    request = TabletopTaskRequest.from_json(request_path)
    execution = TabletopExecutionPlan.from_json(execution_path)
    result = benchmark_open_approach_mpc(
        request,
        execution,
        config=MPCBenchmarkConfig(maximum_steps=maximum_steps),
    )
    atomic_write_json(output_path, result)
    return result
