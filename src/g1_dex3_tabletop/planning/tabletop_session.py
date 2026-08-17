"""Long-lived CuRobo planning context for one tabletop hardware run."""

from __future__ import annotations

import time
from collections.abc import Callable

import numpy as np

from g1_dex3_tabletop.mpc_command_buffer import MPCCommandWindow
from g1_dex3_tabletop.planning.tabletop_mpc import TabletopPhaseMPC, mpc_phase_spec
from g1_dex3_tabletop.planning.tabletop_planner import (
    RetentionRouteValidator,
    plan_supported_escape,
    plan_tabletop_task,
)
from g1_dex3_tabletop.tabletop_contracts import (
    RetentionRouteValidationRequest,
    RetentionRouteValidationResult,
    SupportedEscapePlan,
    TabletopExecutionPlan,
    TabletopTaskRequest,
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
        self._execution: TabletopExecutionPlan | None = None
        self._retention_validator: RetentionRouteValidator | None = None
        self._phase_mpc: TabletopPhaseMPC | None = None
        self._active_phase_mpc: TabletopPhaseMPC | None = None

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
        self._execution = execution
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
        self._execution = None
        self._retention_validator = None
        return escape

    def validate_retention_route(
        self,
        request: RetentionRouteValidationRequest,
        *,
        progress: Callable[[str], None] | None = None,
    ) -> RetentionRouteValidationResult:
        """Validate contact geometry against this session's frozen route."""

        if (
            self._clearance_request is None
            or self._execution is None
            or self._retention_validator is None
        ):
            raise RuntimeError("retention validation requires a lifecycle planned in this worker")
        if request.tabletop_request.content_sha256 != self._clearance_request.content_sha256:
            raise ValueError("retention request differs from the worker's clearance request")
        if request.task_plan.content_sha256 != self._execution.task.content_sha256:
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
        self._execution = execution
        self._retention_validator = retention_validator
        report(
            "boundary-corrected lifecycle ready; measured-contact checker build="
            f"{retention_validator.cache_build_s:.3f}s"
        )
        return execution

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
            "reconfiguration_time_s": 0.0,
            "plan_sha256": self._execution.content_sha256,
        }

    def step_mpc_phase(self, payload: dict) -> MPCCommandWindow:
        """Optimize one window from a fresh measured state and active command."""

        if self._active_phase_mpc is None:
            raise RuntimeError("tabletop phase MPC has not been prepared")
        requested_phase = str(payload["phase"])
        if requested_phase != self._active_phase_mpc.spec.phase:
            raise ValueError(
                f"MPC step requests {requested_phase} while "
                f"{self._active_phase_mpc.spec.phase} is prepared"
            )
        return self._active_phase_mpc.next_nominal_window(
            measured_command_q_rad=np.asarray(payload["measured_command_q_rad"], dtype=np.float64),
            measured_dq_rad_s=np.asarray(payload["measured_dq_rad_s"], dtype=np.float64),
            active_command_q_rad=np.asarray(payload["active_command_q_rad"], dtype=np.float64),
            state_monotonic_s=float(payload["state_monotonic_s"]),
            reference_T_camera=(
                None
                if payload.get("reference_T_camera") is None
                else np.asarray(payload["reference_T_camera"], dtype=np.float64)
            ),
        )

    def close(self) -> None:
        controller = self._phase_mpc
        self._phase_mpc = None
        self._active_phase_mpc = None
        if controller is not None:
            controller.close()
