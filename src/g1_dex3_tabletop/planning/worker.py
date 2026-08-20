"""Command-line boundary for the isolated Python 3.11 CuRobo process."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from g1_dex3_tabletop.planning.contracts import (
    CalibrationPlanRequest,
    Dex3PreparationRequest,
)
from g1_dex3_tabletop.planning.curobo_backend import (
    inspect_model,
    plan_calibration,
    plan_dex3_preparation,
)
from g1_dex3_tabletop.planning.tabletop_mpc import benchmark_from_paths
from g1_dex3_tabletop.planning.tabletop_planner import (
    PickPlaceRetentionRouteValidator,
    plan_supported_escape,
    plan_tabletop_pregrasp,
    plan_tabletop_task,
    prewarm_tabletop_model_resolution,
    validate_retention_route,
)
from g1_dex3_tabletop.planning.tabletop_session import TabletopPlanningSession
from g1_dex3_tabletop.planning.waist_yaw_analysis import analyze_waist_yaw_from_paths
from g1_dex3_tabletop.tabletop_contracts import (
    CharucoSupportedEscapeRequest,
    MovingGraspContinuationRequest,
    PickPlaceRetentionRouteValidationRequest,
    RetentionRouteValidationRequest,
    TabletopPickPlaceRequest,
    TabletopTaskRequest,
)

EVENT_PREFIX = "G1_PLANNER_EVENT "


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="g1-curobo-worker")
    subparsers = parser.add_subparsers(dest="command", required=True)
    inspect_parser = subparsers.add_parser(
        "inspect-model", help="validate the request and report the locked CuRobo model"
    )
    inspect_parser.add_argument("--request", type=Path, required=True)
    plan_parser = subparsers.add_parser(
        "plan-calibration", help="run batched IK and plan the complete frozen route"
    )
    plan_parser.add_argument("--request", type=Path, required=True)
    plan_parser.add_argument("--output", type=Path, required=True)
    preparation_parser = subparsers.add_parser(
        "plan-dex3-preparation",
        help="plan shoulder clearance and validate the complete Dex3 finger sweep",
    )
    preparation_parser.add_argument("--request", type=Path, required=True)
    preparation_parser.add_argument("--output", type=Path, required=True)
    for command, help_text in (
        (
            "plan-supported-escape",
            "plan the reversible supported-table lift before opening Dex3",
        ),
        (
            "plan-charuco-supported-escape",
            "plan the reversible supported-table lift from a fixed ChArUco board",
        ),
        (
            "plan-tabletop-pregrasp",
            "plan only the reversible clearance-to-pregrasp route",
        ),
        (
            "plan-tabletop-task",
            "plan the qualified cube pick, lift, replace, retreat, and return",
        ),
        (
            "plan-tabletop-pick-place",
            "plan one fixed source grasp, attached transfer, placement, and return",
        ),
        (
            "validate-pick-place-retention-route",
            "recheck a fixed source-to-placement route at the measured close posture",
        ),
        (
            "validate-retention-route",
            "recheck the frozen payload route at the measured stable-close hand posture",
        ),
        (
            "plan-tabletop-lifecycle",
            "plan the complete supported escape, task, and exact return in one process",
        ),
    ):
        tabletop = subparsers.add_parser(command, help=help_text)
        tabletop.add_argument("--request", type=Path, required=True)
        tabletop.add_argument("--output", type=Path, required=True)
    subparsers.add_parser(
        "serve-tabletop",
        help="serve lifecycle planning and retention validation over stdin/stdout",
    )
    benchmark = subparsers.add_parser(
        "benchmark-tabletop-mpc",
        help="offline phase-aware MPC replay of a complete retained tabletop lifecycle",
    )
    benchmark.add_argument("--loaded-request", type=Path, required=True)
    benchmark.add_argument("--clearance-request", type=Path, required=True)
    benchmark.add_argument("--plan", type=Path, required=True)
    benchmark.add_argument("--output", type=Path, required=True)
    benchmark.add_argument(
        "--grasp-close",
        type=Path,
        help="retained hardware grasp_close.json used for attached-payload replay",
    )
    benchmark.add_argument(
        "--camera-state-estimate",
        type=Path,
        help=(
            "retained CameraStateEstimate or estimator installation-check JSON used "
            "for command-free state-corrected replay"
        ),
    )
    benchmark.add_argument("--maximum-steps", type=int, default=300)
    waist = subparsers.add_parser(
        "analyze-waist-yaw",
        help=("compare locked-waist and bounded waist-yaw pregrasp IK on a retained request"),
    )
    waist.add_argument("--request", type=Path, required=True)
    waist.add_argument("--output", type=Path, required=True)
    waist.add_argument(
        "--waist-half-range-rad",
        type=float,
        action="append",
        required=True,
        help="start-relative yaw half range; repeat to compare several bounds",
    )
    waist.add_argument(
        "--candidate-id",
        action="append",
        default=[],
        help="limit the comparison to an exact retained grasp candidate; repeat if needed",
    )
    return parser


def _emit(event: dict) -> None:
    print(EVENT_PREFIX + json.dumps(event, separators=(",", ":")), flush=True)


def _serve_tabletop() -> int:
    """Keep Python, CuRobo imports, and the CUDA context alive for one run."""

    import torch

    cuda_available = bool(torch.cuda.is_available())
    model_prewarm_s = None
    if cuda_available:
        torch.cuda.init()
        model_prewarm_s = prewarm_tabletop_model_resolution()
    session = TabletopPlanningSession()
    _emit(
        {
            "type": "ready",
            "cuda_available": cuda_available,
            "device": torch.cuda.get_device_name(0) if cuda_available else None,
            "model_prewarm_s": model_prewarm_s,
        }
    )
    try:
        for line in sys.stdin:
            request_id = None
            try:
                message = json.loads(line)
                if message.get("command") == "shutdown":
                    _emit({"type": "stopped"})
                    return 0
                request_id = int(message["id"])
                command = str(message["command"])

                def progress(text: str, *, event_id: int = request_id) -> None:
                    _emit({"type": "progress", "id": event_id, "message": text})

                if command == "prepare-mpc-phase":
                    request_payload = message["payload"]
                    measured_fingers = request_payload.get("measured_active_dex3_q_rad")
                    payload = session.prepare_mpc_phase(
                        str(request_payload["phase"]),
                        measured_active_dex3_q_rad=measured_fingers,
                        reference_T_camera0=(
                            None
                            if request_payload.get("reference_T_camera0") is None
                            else np.asarray(
                                request_payload["reference_T_camera0"], dtype=np.float64
                            )
                        ),
                    )
                    event = {
                        "type": "result",
                        "id": request_id,
                        "ok": True,
                        "operation": command,
                        "payload": payload,
                    }
                elif command == "prewarm-tabletop-at-clearance":
                    request = TabletopTaskRequest.from_json(Path(message["payload"]["request"]))
                    payload = session.prewarm_task_at_clearance(
                        request,
                        progress=progress,
                    )
                    event = {
                        "type": "result",
                        "id": request_id,
                        "ok": True,
                        "operation": command,
                        "payload": payload,
                    }
                elif command == "step-mpc-phase":
                    window = session.step_mpc_phase(message["payload"])
                    event = {
                        "type": "result",
                        "id": request_id,
                        "ok": True,
                        "operation": command,
                        "payload": window.to_dict(),
                    }
                else:
                    request_path = Path(message["request"])
                    output_path = Path(message["output"])
                    if output_path.exists():
                        raise FileExistsError(f"planner output already exists: {output_path}")
                    if command == "plan-tabletop-lifecycle":
                        request = TabletopTaskRequest.from_json(request_path)
                        result = session.plan_lifecycle(request, progress=progress)
                    elif command == "plan-supported-escape":
                        request = TabletopTaskRequest.from_json(request_path)
                        result = session.plan_escape(request, progress=progress)
                    elif command == "plan-tabletop-pick-place":
                        request = TabletopPickPlaceRequest.from_json(request_path)
                        result = session.plan_pick_place(request, progress=progress)
                    elif command == "replan-tabletop-at-clearance":
                        request = TabletopTaskRequest.from_json(request_path)
                        result = session.replan_at_clearance(request, progress=progress)
                    elif command == "plan-tabletop-pregrasp-at-clearance":
                        request = TabletopTaskRequest.from_json(request_path)
                        result = session.plan_pregrasp_at_clearance(
                            request,
                            progress=progress,
                        )
                    elif command == "replan-tabletop-at-pregrasp":
                        request = TabletopTaskRequest.from_json(request_path)
                        result = session.replan_at_pregrasp(request, progress=progress)
                    elif command == "plan-moving-grasp-continuation":
                        request = MovingGraspContinuationRequest.from_json(request_path)
                        result = session.plan_moving_grasp_continuation(
                            request,
                            progress=progress,
                        )
                    elif command == "plan-charuco-supported-escape":
                        request = CharucoSupportedEscapeRequest.from_json(request_path)
                        result = plan_supported_escape(request, progress=progress)
                    elif command == "validate-retention-route":
                        request = RetentionRouteValidationRequest.from_json(request_path)
                        result = session.validate_retention_route(request, progress=progress)
                    elif command == "validate-pick-place-retention-route":
                        request = PickPlaceRetentionRouteValidationRequest.from_json(request_path)
                        result = session.validate_pick_place_retention_route(
                            request,
                            progress=progress,
                        )
                    else:
                        raise ValueError(f"unsupported persistent planner command: {command}")
                    result.write_json(output_path)
                    event = {
                        "type": "result",
                        "id": request_id,
                        "ok": True,
                        "operation": command,
                        "output": str(output_path.resolve()),
                        "plan_sha256": result.content_sha256,
                    }
                _emit(event)
            except (
                FileNotFoundError,
                FileExistsError,
                KeyError,
                RuntimeError,
                TypeError,
                ValueError,
            ) as error:
                _emit(
                    {
                        "type": "result",
                        "id": locals().get("request_id"),
                        "ok": False,
                        "error_type": type(error).__name__,
                        "error": str(error),
                    }
                )
        return 0
    finally:
        session.close()


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "serve-tabletop":
            return _serve_tabletop()
        if args.command == "benchmark-tabletop-mpc":
            if args.output.exists():
                raise FileExistsError(f"planner output already exists: {args.output}")
            result = benchmark_from_paths(
                args.loaded_request,
                args.clearance_request,
                args.plan,
                args.output,
                maximum_steps=args.maximum_steps,
                grasp_close_path=args.grasp_close,
                camera_state_estimate_path=args.camera_state_estimate,
            )
            print(json.dumps(result, indent=2, sort_keys=True))
            return 0
        if args.command == "analyze-waist-yaw":
            if args.output.exists():
                raise FileExistsError(f"planner output already exists: {args.output}")
            result = analyze_waist_yaw_from_paths(
                args.request,
                args.output,
                waist_half_ranges_rad=tuple(args.waist_half_range_rad),
                candidate_ids=tuple(args.candidate_id),
                progress=lambda message: print(message, file=sys.stderr, flush=True),
            )
            print(json.dumps(result, indent=2, sort_keys=True))
            return 0
        if args.command == "inspect-model":
            request = CalibrationPlanRequest.from_json(args.request)
            print(json.dumps(inspect_model(request), indent=2, sort_keys=True))
            return 0
        if args.output.exists():
            raise FileExistsError(f"planner output already exists: {args.output}")
        if args.command == "validate-retention-route":
            request = RetentionRouteValidationRequest.from_json(args.request)
            result = validate_retention_route(
                request,
                progress=lambda message: print(message, file=sys.stderr, flush=True),
            )
            summary = {
                "commands_robot": False,
                "output": str(args.output.resolve()),
                "plan_sha256": result.content_sha256,
                "operation": args.command,
            }
        elif args.command == "validate-pick-place-retention-route":
            request = PickPlaceRetentionRouteValidationRequest.from_json(args.request)
            validator = PickPlaceRetentionRouteValidator(
                request.pick_place_request,
                request.pick_place_plan,
            )
            result = validator.validate(
                request,
                progress=lambda message: print(message, file=sys.stderr, flush=True),
            )
            summary = {
                "commands_robot": False,
                "output": str(args.output.resolve()),
                "plan_sha256": result.content_sha256,
                "operation": args.command,
            }
        elif args.command == "plan-charuco-supported-escape":
            request = CharucoSupportedEscapeRequest.from_json(args.request)
            result = plan_supported_escape(
                request,
                progress=lambda message: print(message, file=sys.stderr, flush=True),
            )
            summary = {
                "commands_robot": False,
                "output": str(args.output.resolve()),
                "plan_sha256": result.content_sha256,
                "operation": args.command,
            }
        elif args.command in {
            "plan-supported-escape",
            "plan-tabletop-pregrasp",
            "plan-tabletop-task",
            "plan-tabletop-lifecycle",
        }:
            request = TabletopTaskRequest.from_json(args.request)
            if args.command == "plan-supported-escape":
                result = plan_supported_escape(
                    request,
                    progress=lambda message: print(message, file=sys.stderr, flush=True),
                )
            elif args.command == "plan-tabletop-task":
                result = plan_tabletop_task(
                    request,
                    progress=lambda message: print(message, file=sys.stderr, flush=True),
                )
            elif args.command == "plan-tabletop-pregrasp":
                result = plan_tabletop_pregrasp(
                    request,
                    progress=lambda message: print(message, file=sys.stderr, flush=True),
                )
            else:
                result = TabletopPlanningSession().plan_lifecycle(
                    request,
                    progress=lambda message: print(message, file=sys.stderr, flush=True),
                )
            summary = {
                "commands_robot": False,
                "output": str(args.output.resolve()),
                "plan_sha256": result.content_sha256,
                "operation": args.command,
            }
        elif args.command == "plan-tabletop-pick-place":
            request = TabletopPickPlaceRequest.from_json(args.request)
            session = TabletopPlanningSession()
            try:
                result = session.plan_pick_place(
                    request,
                    progress=lambda message: print(message, file=sys.stderr, flush=True),
                )
            finally:
                session.close()
            summary = {
                "commands_robot": False,
                "output": str(args.output.resolve()),
                "plan_sha256": result.content_sha256,
                "operation": args.command,
            }
        elif args.command == "plan-dex3-preparation":
            request = Dex3PreparationRequest.from_json(args.request)
            result = plan_dex3_preparation(
                request,
                progress=lambda message: print(message, file=sys.stderr, flush=True),
            )
            summary = {
                "commands_robot": False,
                "output": str(args.output.resolve()),
                "plan_sha256": result.content_sha256,
                "outward_offset_rad": result.outward_offset_rad,
                "finger_sweep_sample_count": result.finger_sweep_sample_count,
            }
        else:
            request = CalibrationPlanRequest.from_json(args.request)
            result = plan_calibration(
                request,
                progress=lambda message: print(message, file=sys.stderr, flush=True),
            )
            summary = {
                "commands_robot": False,
                "output": str(args.output.resolve()),
                "plan_sha256": result.content_sha256,
                "selected_count": len(result.capture_pose_ids),
                "trajectory_count": len(result.trajectories),
            }
        result.write_json(args.output)
        print(json.dumps(summary, indent=2, sort_keys=True))
        return 0
    except (FileNotFoundError, FileExistsError, RuntimeError, TypeError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
