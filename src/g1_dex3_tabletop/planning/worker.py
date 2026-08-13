"""Command-line boundary for the isolated Python 3.11 CuRobo process."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from g1_dex3_tabletop.planning.contracts import (
    CalibrationPlanRequest,
    Dex3PreparationRequest,
)
from g1_dex3_tabletop.planning.curobo_backend import (
    inspect_model,
    plan_calibration,
    plan_dex3_preparation,
)
from g1_dex3_tabletop.planning.tabletop_planner import (
    plan_supported_escape,
    plan_tabletop_task,
)
from g1_dex3_tabletop.tabletop_contracts import TabletopTaskRequest


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
            "plan-tabletop-task",
            "plan the qualified cube pick, lift, replace, retreat, and return",
        ),
    ):
        tabletop = subparsers.add_parser(command, help=help_text)
        tabletop.add_argument("--request", type=Path, required=True)
        tabletop.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "inspect-model":
            request = CalibrationPlanRequest.from_json(args.request)
            print(json.dumps(inspect_model(request), indent=2, sort_keys=True))
            return 0
        if args.output.exists():
            raise FileExistsError(f"planner output already exists: {args.output}")
        if args.command in {"plan-supported-escape", "plan-tabletop-task"}:
            request = TabletopTaskRequest.from_json(args.request)
            result = (
                plan_supported_escape(
                    request,
                    progress=lambda message: print(message, file=sys.stderr, flush=True),
                )
                if args.command == "plan-supported-escape"
                else plan_tabletop_task(
                    request,
                    progress=lambda message: print(message, file=sys.stderr, flush=True),
                )
            )
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
