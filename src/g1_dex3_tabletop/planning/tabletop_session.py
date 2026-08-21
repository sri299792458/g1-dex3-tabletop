"""Long-lived CuRobo planning context for one tabletop hardware run."""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import replace

import numpy as np

from g1_aprilcube_calibration.joint_map import arm_indices
from g1_dex3_tabletop.mpc_command_buffer import MPCCommandWindow
from g1_dex3_tabletop.planning.tabletop_mpc import MovingGraspMPC, mpc_phase_spec
from g1_dex3_tabletop.planning.tabletop_planner import (
    PickPlaceRetentionRouteValidator,
    RetentionRouteValidator,
    TabletopPlannerPool,
    analyze_tabletop_pick_place_endpoints,
    plan_moving_grasp_continuation,
    plan_supported_escape,
    plan_tabletop_pick_place,
    plan_tabletop_pregrasp,
    plan_tabletop_task,
    prewarm_tabletop_runtime_models,
)
from g1_dex3_tabletop.tabletop_contracts import (
    MovingGraspContinuationRequest,
    PickPlaceRetentionRouteValidationRequest,
    PregraspRemainingPlan,
    RetentionRouteValidationRequest,
    RetentionRouteValidationResult,
    SupportedEscapePlan,
    TabletopExecutionPlan,
    TabletopPickPlacePlan,
    TabletopPickPlaceRequest,
    TabletopPregraspPlan,
    TabletopTaskPlan,
    TabletopTaskRequest,
    build_pregrasp_remaining_plan,
    combine_tabletop_plans,
)
from g1_dex3_tabletop.tabletop_workflow import (
    assemble_execution_plan,
    request_at_clearance,
    request_at_clearance_observation,
)


class TabletopPlanningSession:
    """Plan and validate one lifecycle without restarting Python or CUDA."""

    def __init__(self) -> None:
        self._loaded_request: TabletopTaskRequest | None = None
        self._clearance_request: TabletopTaskRequest | None = None
        self._supported_escape: SupportedEscapePlan | None = None
        self._pregrasp_plan: TabletopPregraspPlan | None = None
        self._execution: TabletopExecutionPlan | None = None
        self._active_task: TabletopTaskPlan | None = None
        self._retention_validator: RetentionRouteValidator | None = None
        self._pick_place_request: TabletopPickPlaceRequest | None = None
        self._pick_place_plan: TabletopPickPlacePlan | None = None
        self._pick_place_retention_validator: PickPlaceRetentionRouteValidator | None = None
        self._phase_mpc: MovingGraspMPC | None = None
        self._active_phase_mpc: MovingGraspMPC | None = None
        self._planner_pool = TabletopPlannerPool()

    def plan_lifecycle(
        self,
        request: TabletopTaskRequest,
        *,
        progress: Callable[[str], None] | None = None,
    ) -> TabletopExecutionPlan:
        report = progress or (lambda _message: None)
        escape = plan_supported_escape(request, progress=report)
        clearance_request = request_at_clearance(request, escape)
        task = plan_tabletop_task(
            clearance_request,
            planner_pool=self._planner_pool,
            progress=report,
        )
        _clearance, execution = assemble_execution_plan(
            loaded_request=request,
            supported_escape=escape,
            task=task,
        )
        self._loaded_request = request
        self._clearance_request = clearance_request
        self._supported_escape = escape
        self._pregrasp_plan = None
        self._execution = execution
        self._active_task = task
        report("building reusable measured-contact collision checker")
        self._retention_validator = self._planner_pool.retention_validator(
            clearance_request,
            task,
        )
        report(
            "reusable measured-contact collision checker ready; "
            f"build={self._retention_validator.cache_build_s:.3f}s"
        )
        return execution

    def plan_escape(
        self,
        request: TabletopTaskRequest,
        *,
        progress: Callable[[str], None] | None = None,
    ) -> SupportedEscapePlan:
        """Freeze only the supported lift and exact reverse needed to reach clearance."""

        report = progress or (lambda _message: None)
        escape = plan_supported_escape(request, progress=report)
        # The provisional moving-grasp MPC owns no executable route. Keep its
        # CUDA graph alive while the fresh loaded-state escape is planned.
        self._active_phase_mpc = None
        self._loaded_request = request
        self._clearance_request = request_at_clearance(request, escape)
        self._supported_escape = escape
        self._pregrasp_plan = None
        self._execution = None
        self._active_task = None
        self._retention_validator = None
        self._pick_place_request = None
        self._pick_place_plan = None
        self._pick_place_retention_validator = None
        return escape

    def prewarm_runtime(
        self,
        request: TabletopTaskRequest,
        *,
        moving_grasp_mpc: bool,
        progress: Callable[[str], None] | None = None,
    ) -> dict:
        """Warm command-free planner topology from the read-only preflight state."""

        report = progress or (lambda _message: None)
        report(
            "constructing persistent open-hand and attached-payload MotionGen models; "
            "no task solve or robot command is being used as a gate"
        )
        result = prewarm_tabletop_runtime_models(
            request,
            planner_pool=self._planner_pool,
        )
        mpc_result = None
        if moving_grasp_mpc:
            controller = self._phase_mpc
            if controller is not None:
                controller.close()
            mpc_started = time.perf_counter()
            controller = MovingGraspMPC(
                request,
                None,
                phase="grasp_approach",
            )
            try:
                mpc_setup_s = controller.setup_at_frozen_route_start(validate_strict_start=False)
            except BaseException:
                controller.close()
                raise
            self._phase_mpc = controller
            self._active_phase_mpc = None
            mpc_result = {
                "build_and_setup_time_s": time.perf_counter() - mpc_started,
                "setup_time_s": mpc_setup_s,
                "retained_for_live_binding": True,
                "strict_start_validation_deferred_to_live_route": True,
            }
        result = {
            **result,
            "moving_grasp_mpc": mpc_result,
        }
        report(
            "command-free runtime warmup complete; fresh loaded and clearance "
            "observations remain the only source of executable task feasibility"
        )
        return result

    def prewarm_stack_runtime(
        self,
        requests: tuple[TabletopTaskRequest, ...],
        *,
        progress: Callable[[str], None] | None = None,
    ) -> dict:
        """Warm both arm topologies for the fixed two-cube stack.

        One aggregate worker command is intentional: the persistent CuRobo
        process owns one CUDA context and one planner pool, while the parent is
        free to keep servicing the read-only camera preview.  The two arms are
        warmed sequentially inside that process; launching a second GPU worker
        would duplicate model memory without making either planner reusable by
        the hardware run.
        """

        by_arm = {request.arm: request for request in requests}
        if len(by_arm) != len(requests):
            raise ValueError("stack runtime warmup received a duplicate arm")
        if set(by_arm) != {"left", "right"}:
            raise ValueError("stack runtime warmup requires exactly left and right requests")
        report = progress or (lambda _message: None)
        started = time.perf_counter()
        results: dict[str, dict] = {}
        for arm in ("left", "right"):
            report(
                f"warming persistent {arm}-arm open-hand and attached-payload "
                "MotionGen models; no robot command or task solve"
            )
            results[arm] = prewarm_tabletop_runtime_models(
                by_arm[arm],
                planner_pool=self._planner_pool,
            )
        elapsed_s = time.perf_counter() - started
        report(
            "command-free dual-arm stack warmup complete; live loaded and clearance "
            "observations remain the only source of executable task feasibility"
        )
        return {
            "operation": "prewarm_stack_runtime",
            "elapsed_s": elapsed_s,
            "arms": results,
            "task_feasibility_planning_performed": False,
            "robot_command_authorized": False,
        }

    def plan_pick_place(
        self,
        request: TabletopPickPlaceRequest,
        *,
        progress: Callable[[str], None] | None = None,
    ) -> TabletopPickPlacePlan:
        """Plan one fixed source-to-destination transfer in the warm worker."""

        controller = self._phase_mpc
        if controller is not None:
            controller.close()
        self._phase_mpc = None
        self._active_phase_mpc = None
        plan = plan_tabletop_pick_place(
            request,
            planner_pool=self._planner_pool,
            progress=progress,
        )
        self._pick_place_request = request
        self._pick_place_plan = plan
        self._pick_place_retention_validator = None
        return plan

    def analyze_pick_place_endpoints(
        self,
        request: TabletopPickPlaceRequest,
        *,
        progress: Callable[[str], None] | None = None,
    ) -> dict:
        """Run only the shared source/destination endpoint feasibility pass."""

        return analyze_tabletop_pick_place_endpoints(
            request,
            planner_pool=self._planner_pool,
            progress=progress,
        )

    def validate_pick_place_retention_route(
        self,
        request: PickPlaceRetentionRouteValidationRequest,
        *,
        progress: Callable[[str], None] | None = None,
    ) -> RetentionRouteValidationResult:
        """Validate one measured close against its supplied frozen transfer.

        Stack feasibility planning can evaluate more than one complete transfer
        before either arm moves.  The most recently planned transfer is not
        necessarily the one selected for execution, so rebuild only the cheap
        collision checker when the hash-bound request names another already
        supplied plan.  Motion planning is never repeated here.
        """

        if (
            self._pick_place_request is None
            or self._pick_place_plan is None
            or self._pick_place_retention_validator is None
            or self._pick_place_request.content_sha256 != request.pick_place_request.content_sha256
            or self._pick_place_plan.content_sha256 != request.pick_place_plan.content_sha256
        ):
            self._pick_place_request = request.pick_place_request
            self._pick_place_plan = request.pick_place_plan
            self._pick_place_retention_validator = PickPlaceRetentionRouteValidator(
                request.pick_place_request,
                request.pick_place_plan,
            )
        return self._pick_place_retention_validator.validate(request, progress=progress)

    def plan_pregrasp_at_clearance(
        self,
        request: TabletopTaskRequest,
        *,
        progress: Callable[[str], None] | None = None,
    ) -> TabletopPregraspPlan:
        """Plan only a reversible route from the observed clearance boundary."""

        if self._loaded_request is None or self._supported_escape is None:
            raise RuntimeError(
                "clearance pregrasp planning requires a supported escape in this worker"
            )
        expected = request_at_clearance_observation(
            self._loaded_request,
            self._supported_escape,
            request.observation,
        )
        if request.content_sha256 != expected.content_sha256:
            raise ValueError(
                "clearance pregrasp request changed more than the boundary observation "
                "and exact supported-escape endpoint"
            )
        report = progress or (lambda _message: None)
        pregrasp = plan_tabletop_pregrasp(
            request,
            planner_pool=self._planner_pool,
            progress=report,
        )
        controller = self._phase_mpc
        if controller is not None:
            controller.close()
        self._phase_mpc = None
        self._active_phase_mpc = None
        self._clearance_request = request
        self._pregrasp_plan = pregrasp
        self._execution = None
        self._active_task = None
        self._retention_validator = None
        return pregrasp

    def validate_retention_route(
        self,
        request: RetentionRouteValidationRequest,
        *,
        progress: Callable[[str], None] | None = None,
    ) -> RetentionRouteValidationResult:
        """Validate contact geometry against this session's frozen route."""

        if (
            self._clearance_request is None
            or self._active_task is None
            or self._retention_validator is None
        ):
            raise RuntimeError("retention validation requires a lifecycle planned in this worker")
        if request.tabletop_request.content_sha256 != self._clearance_request.content_sha256:
            raise ValueError("retention request differs from the worker's clearance request")
        if request.task_plan.content_sha256 != self._active_task.content_sha256:
            raise ValueError("retention request differs from the worker's frozen task plan")
        return self._retention_validator.validate(request, progress=progress)

    def replan_at_clearance(
        self,
        request: TabletopTaskRequest,
        *,
        progress: Callable[[str], None] | None = None,
    ) -> TabletopExecutionPlan:
        """Replace only the task using a fresh fixed-cube clearance observation."""

        if self._loaded_request is None or self._supported_escape is None:
            raise RuntimeError(
                "clearance replan requires a supported escape planned in this worker"
            )
        expected = request_at_clearance_observation(
            self._loaded_request,
            self._supported_escape,
            request.observation,
        )
        if request.content_sha256 != expected.content_sha256:
            raise ValueError(
                "clearance replan changed more than the boundary observation and exact "
                "supported-escape endpoint"
            )
        report = progress or (lambda _message: None)
        task = plan_tabletop_task(
            request,
            planner_pool=self._planner_pool,
            progress=report,
        )
        execution = combine_tabletop_plans(
            loaded_request=self._loaded_request,
            clearance_request=request,
            supported_escape=self._supported_escape,
            task=task,
        )
        report("rebuilding measured-contact checker for the boundary-corrected task")
        retention_validator = self._planner_pool.retention_validator(request, task)
        self._active_phase_mpc = None
        self._clearance_request = request
        self._pregrasp_plan = None
        self._execution = execution
        self._active_task = task
        self._retention_validator = retention_validator
        report(
            "boundary-corrected lifecycle ready; measured-contact checker build="
            f"{retention_validator.cache_build_s:.3f}s"
        )
        return execution

    def replan_at_pregrasp(
        self,
        request: TabletopTaskRequest,
        *,
        progress: Callable[[str], None] | None = None,
    ) -> PregraspRemainingPlan:
        """Correct the frozen task once at the reached stationary pregrasp."""

        if self._clearance_request is None or self._pregrasp_plan is None:
            raise RuntimeError("pregrasp replan requires a pregrasp route in this worker")
        if request.estimated_planning_state is None:
            raise ValueError("pregrasp replan requires an estimated camera planning state")
        original = replace(request, estimated_planning_state=None)
        if original.content_sha256 != self._clearance_request.content_sha256:
            raise ValueError("pregrasp replan changed more than the estimated boundary state")
        expected_command = np.asarray(
            self._pregrasp_plan.outbound.command_q_rad[-1], dtype=np.float64
        )
        measured_command = np.asarray(request.planning_snapshot.measured_q29_rad)[
            np.asarray(arm_indices(request.arm))
        ]
        command_error = float(np.max(np.abs(expected_command - measured_command)))
        if command_error > 1.0e-8:
            raise ValueError(
                "pregrasp estimated request does not preserve the exact active command; "
                f"error={command_error:.9f}rad"
            )
        report = progress or (lambda _message: None)
        selected_candidate_id = self._pregrasp_plan.selected_candidate_id
        task = plan_tabletop_task(
            request,
            required_candidate_id=selected_candidate_id,
            planner_pool=self._planner_pool,
            progress=report,
        )
        remaining = build_pregrasp_remaining_plan(
            prior_pregrasp_plan=self._pregrasp_plan,
            estimated_request=request,
            task=task,
        )
        report("rebuilding measured-contact checker for the pregrasp-corrected task")
        validator = self._planner_pool.retention_validator(request, task)
        controller = self._phase_mpc
        if controller is not None:
            controller.close()
        self._phase_mpc = None
        self._active_phase_mpc = None
        self._clearance_request = request
        self._active_task = task
        self._retention_validator = validator
        report(
            "pregrasp-corrected remaining lifecycle ready; measured-contact checker build="
            f"{validator.cache_build_s:.3f}s"
        )
        return remaining

    def plan_moving_grasp_continuation(
        self,
        request: MovingGraspContinuationRequest,
        *,
        progress: Callable[[str], None] | None = None,
    ) -> TabletopTaskPlan:
        """Replace stale payload routes after MPC intercepts a moved cube."""

        if self._clearance_request is None or self._execution is None:
            raise RuntimeError("moving-grasp continuation requires a planned lifecycle")
        if request.tabletop_request.content_sha256 != self._clearance_request.content_sha256:
            raise ValueError("moving-grasp continuation uses another clearance request")
        if request.prior_task_plan.content_sha256 != self._execution.task.content_sha256:
            raise ValueError("moving-grasp continuation uses another prior task")
        task, geometry = plan_moving_grasp_continuation(
            request,
            planner_pool=self._planner_pool,
            progress=progress,
        )
        validator = self._planner_pool.retention_validator(
            request.tabletop_request,
            task,
            geometry=geometry,
        )
        controller = self._phase_mpc
        self._phase_mpc = None
        self._active_phase_mpc = None
        if controller is not None:
            controller.close()
        self._active_task = task
        self._retention_validator = validator
        return task

    def prepare_moving_grasp_mpc(
        self,
        *,
        reference_T_camera0: np.ndarray,
    ) -> dict:
        """Bind the frozen grasp approach into the prewarmed MPC model."""

        if (
            self._loaded_request is None
            or self._clearance_request is None
            or self._execution is None
        ):
            raise RuntimeError("MPC preparation requires a lifecycle planned in this worker")
        if self._active_task is not None and (
            self._active_task.content_sha256 != self._execution.task.content_sha256
        ):
            raise RuntimeError("MPC preparation is unavailable after the pregrasp boundary replan")
        phase = "grasp_approach"
        spec = mpc_phase_spec(phase)
        controller = self._phase_mpc
        if controller is not None:
            binding = controller.bind_moving_grasp_execution(
                self._clearance_request,
                self._execution,
                loaded_request=self._loaded_request,
                reference_T_camera0=np.asarray(reference_T_camera0, dtype=np.float64),
            )
            self._active_phase_mpc = controller
            return {
                "build_time_s": 0.0,
                "preparation_time_s": binding["total_time_s"],
                "setup_time_s": binding["setup_time_s"],
                "phase": phase,
                "physical_mode": spec.mode,
                "reused_warm_model": True,
                "kinematics_cache_hit": False,
                "kinematics_resolve_time_s": binding["rebind_time_s"],
                "optimizer_prewarm_time_s": binding["setup_time_s"],
                "state_correction_prewarm_time_s": getattr(
                    controller, "_last_state_correction_prewarm_s", 0.0
                ),
                "reconfiguration_time_s": binding["total_time_s"],
                "plan_sha256": self._execution.content_sha256,
            }
        build_started = time.perf_counter()
        constructor_arguments = {
            "phase": phase,
            "loaded_request": self._loaded_request,
            "measured_active_dex3_q_rad": None,
        }
        constructor_arguments["reference_T_camera0"] = np.asarray(
            reference_T_camera0, dtype=np.float64
        )
        controller = MovingGraspMPC(
            self._clearance_request,
            self._execution,
            **constructor_arguments,
        )
        build_s = time.perf_counter() - build_started
        try:
            setup_s = controller.setup_at_frozen_route_start()
        except BaseException:
            controller.close()
            raise
        self._phase_mpc = controller
        self._active_phase_mpc = controller
        return {
            "build_time_s": build_s,
            "preparation_time_s": build_s + setup_s,
            "setup_time_s": setup_s,
            "phase": phase,
            "physical_mode": spec.mode,
            "reused_warm_model": False,
            "kinematics_cache_hit": True,
            "kinematics_resolve_time_s": 0.0,
            "optimizer_prewarm_time_s": 0.0,
            "state_correction_prewarm_time_s": (
                getattr(controller, "_last_state_correction_prewarm_s", 0.0)
            ),
            "reconfiguration_time_s": 0.0,
            "plan_sha256": self._execution.content_sha256,
        }

    def step_moving_grasp_mpc(self, payload: dict) -> MPCCommandWindow:
        """Optimize one moving-grasp window from an immutable future handoff."""

        if self._active_phase_mpc is None:
            raise RuntimeError("tabletop phase MPC has not been prepared")
        moving_target = payload.get("moving_target")
        if not isinstance(moving_target, dict):
            raise TypeError("moving-grasp MPC requires one live target dictionary")
        return self._active_phase_mpc.next_moving_target_window(
            handoff_predicted_q_rad=np.asarray(
                payload["handoff_predicted_q_rad"], dtype=np.float64
            ),
            handoff_predicted_dq_rad_s=np.asarray(
                payload["handoff_predicted_dq_rad_s"], dtype=np.float64
            ),
            handoff_predicted_ddq_rad_s2=np.asarray(
                payload["handoff_predicted_ddq_rad_s2"], dtype=np.float64
            ),
            handoff_command_q_rad=np.asarray(payload["handoff_command_q_rad"], dtype=np.float64),
            source_state_monotonic_s=float(payload["source_state_monotonic_s"]),
            valid_from_monotonic_s=float(payload["valid_from_monotonic_s"]),
            predecessor_sha256=payload.get("predecessor_sha256"),
            reference_T_camera=np.asarray(
                moving_target.get("reference_T_camera"), dtype=np.float64
            ),
            camera_T_object=np.asarray(moving_target.get("camera_T_object"), dtype=np.float64),
            target_provenance=moving_target,
            committed_route_progress_index=int(payload["committed_route_progress_index"]),
        )

    def close(self) -> None:
        self._planner_pool.close()
        controller = self._phase_mpc
        self._phase_mpc = None
        self._active_phase_mpc = None
        if controller is not None:
            controller.close()
