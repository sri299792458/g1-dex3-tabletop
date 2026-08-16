"""Long-lived CuRobo planning context for one tabletop hardware run."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from g1_dex3_tabletop.mpc_command_buffer import MPCCommandWindow
from g1_dex3_tabletop.planning.tabletop_mpc import TabletopOpenApproachMPC
from g1_dex3_tabletop.planning.tabletop_planner import (
    RetentionRouteValidator,
    plan_supported_escape,
    plan_tabletop_task,
)
from g1_dex3_tabletop.tabletop_contracts import (
    RetentionRouteValidationRequest,
    RetentionRouteValidationResult,
    TabletopExecutionPlan,
    TabletopTaskRequest,
)
from g1_dex3_tabletop.tabletop_workflow import assemble_execution_plan, request_at_clearance


class TabletopPlanningSession:
    """Plan and validate one lifecycle without restarting Python or CUDA."""

    def __init__(self) -> None:
        self._clearance_request: TabletopTaskRequest | None = None
        self._execution: TabletopExecutionPlan | None = None
        self._retention_validator: RetentionRouteValidator | None = None
        self._open_approach_mpc: TabletopOpenApproachMPC | None = None

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
        self._clearance_request = clearance_request
        self._execution = execution
        report("building reusable measured-contact collision checker")
        self._retention_validator = RetentionRouteValidator(clearance_request, task)
        report(
            "reusable measured-contact collision checker ready; "
            f"build={self._retention_validator.cache_build_s:.3f}s"
        )
        return execution

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

    def prepare_open_approach_mpc(self) -> dict:
        """Create and warm the exact-model MPC before the live approach."""

        if self._clearance_request is None or self._execution is None:
            raise RuntimeError("MPC preparation requires a lifecycle planned in this worker")
        if self._open_approach_mpc is not None:
            raise RuntimeError("open-approach MPC was already prepared")
        controller = TabletopOpenApproachMPC(
            self._clearance_request,
            self._execution,
        )
        try:
            setup_s = controller.setup_at_frozen_route_start()
        except BaseException:
            controller.close()
            raise
        self._open_approach_mpc = controller
        return {
            "setup_time_s": setup_s,
            "phase": "move_to_pregrasp",
            "plan_sha256": self._execution.content_sha256,
        }

    def step_open_approach_mpc(self, payload: dict) -> MPCCommandWindow:
        """Optimize one window from a fresh measured state and active command."""

        if self._open_approach_mpc is None:
            raise RuntimeError("open-approach MPC has not been prepared")
        return self._open_approach_mpc.next_nominal_window(
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
        if self._open_approach_mpc is not None:
            self._open_approach_mpc.close()
            self._open_approach_mpc = None
