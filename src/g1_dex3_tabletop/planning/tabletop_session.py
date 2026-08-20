"""Long-lived CuRobo planning context for one tabletop hardware run."""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import replace

import numpy as np

from g1_aprilcube_calibration.joint_map import arm_indices
from g1_dex3_tabletop.mpc_command_buffer import MPCCommandWindow
from g1_dex3_tabletop.planning.tabletop_mpc import TabletopPhaseMPC, mpc_phase_spec
from g1_dex3_tabletop.planning.tabletop_planner import (
    PickPlaceRetentionRouteValidator,
    RetentionRouteValidator,
    ReusableOpenPlanner,
    plan_supported_escape,
    plan_tabletop_pick_place,
    plan_tabletop_pregrasp,
    plan_tabletop_task,
)
from g1_dex3_tabletop.tabletop_contracts import (
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
        self._phase_mpc: TabletopPhaseMPC | None = None
        self._active_phase_mpc: TabletopPhaseMPC | None = None
        self._open_planner = ReusableOpenPlanner()

    def plan_lifecycle(
        self,
        request: TabletopTaskRequest,
        *,
        progress: Callable[[str], None] | None = None,
    ) -> TabletopExecutionPlan:
        report = progress or (lambda _message: None)
        escape = plan_supported_escape(request, progress=report)
        clearance_request = request_at_clearance(request, escape)
        task = plan_tabletop_task(clearance_request, progress=report)
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
        self._retention_validator = RetentionRouteValidator(clearance_request, task)
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
        controller = self._phase_mpc
        if controller is not None:
            controller.close()
        self._phase_mpc = None
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
            open_planner_cache=self._open_planner,
            progress=progress,
        )
        self._pick_place_request = request
        self._pick_place_plan = plan
        self._pick_place_retention_validator = None
        return plan

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
            open_planner_cache=self._open_planner,
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
        self._open_planner.close()
        task = plan_tabletop_task(request, progress=report)
        execution = combine_tabletop_plans(
            loaded_request=self._loaded_request,
            clearance_request=request,
            supported_escape=self._supported_escape,
            task=task,
        )
        report("rebuilding measured-contact checker for the boundary-corrected task")
        retention_validator = RetentionRouteValidator(request, task)
        controller = self._phase_mpc
        if controller is not None:
            controller.close()
        self._phase_mpc = None
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
        try:
            task = plan_tabletop_task(
                request,
                required_candidate_id=selected_candidate_id,
                open_planner_cache=self._open_planner,
                progress=report,
            )
        finally:
            self._open_planner.close()
        remaining = build_pregrasp_remaining_plan(
            prior_pregrasp_plan=self._pregrasp_plan,
            estimated_request=request,
            task=task,
        )
        report("rebuilding measured-contact checker for the pregrasp-corrected task")
        validator = RetentionRouteValidator(request, task)
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

    def prepare_mpc_phase(
        self,
        phase: str,
        *,
        measured_active_dex3_q_rad: np.ndarray | None = None,
    ) -> dict:
        """Create or reuse the exact physical MPC model for one motion phase."""

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
        spec = mpc_phase_spec(phase)
        measured = (
            None
            if measured_active_dex3_q_rad is None
            else np.asarray(measured_active_dex3_q_rad, dtype=np.float64)
        )
        controller = self._phase_mpc
        if controller is not None:
            if not controller.can_select_phase(
                phase,
                measured_active_dex3_q_rad=measured,
            ):
                raise ValueError(f"warmed MPC cannot represent physical phase {phase}")
            switch = controller.select_phase(
                phase,
                measured_active_dex3_q_rad=measured,
            )
            self._active_phase_mpc = controller
            return {
                "build_time_s": 0.0,
                "preparation_time_s": switch["reconfiguration_time_s"],
                "setup_time_s": 0.0,
                "phase": phase,
                "physical_mode": spec.mode,
                "reused_warm_model": True,
                "kinematics_cache_hit": switch["kinematics_cache_hit"],
                "kinematics_resolve_time_s": switch["kinematics_resolve_time_s"],
                "optimizer_prewarm_time_s": switch["optimizer_prewarm_time_s"],
                "state_correction_prewarm_time_s": switch.get(
                    "state_correction_prewarm_time_s", 0.0
                ),
                "reconfiguration_time_s": switch["reconfiguration_time_s"],
                "plan_sha256": self._execution.content_sha256,
            }
        build_started = time.perf_counter()
        controller = TabletopPhaseMPC(
            self._clearance_request,
            self._execution,
            phase=phase,
            loaded_request=self._loaded_request,
            measured_active_dex3_q_rad=measured,
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

    def step_mpc_phase(self, payload: dict) -> MPCCommandWindow:
        """Optimize one window from an immutable future handoff boundary."""

        if self._active_phase_mpc is None:
            raise RuntimeError("tabletop phase MPC has not been prepared")
        requested_phase = str(payload["phase"])
        if requested_phase != self._active_phase_mpc.spec.phase:
            raise ValueError(
                f"MPC step requests {requested_phase} while "
                f"{self._active_phase_mpc.spec.phase} is prepared"
            )
        camera_state = payload.get("camera_state_correction")
        if camera_state is not None and not isinstance(camera_state, dict):
            raise TypeError("MPC camera-state correction must be a dictionary")
        return self._active_phase_mpc.next_nominal_window(
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
            committed_route_progress_index=int(payload["committed_route_progress_index"]),
            reference_T_camera=(
                None
                if camera_state is None
                else np.asarray(camera_state.get("reference_T_camera"), dtype=np.float64)
            ),
            camera_state_provenance=camera_state,
        )

    def close(self) -> None:
        self._open_planner.close()
        controller = self._phase_mpc
        self._phase_mpc = None
        self._active_phase_mpc = None
        if controller is not None:
            controller.close()
