"""Focused operator and read-only planning commands."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import yaml

from g1_aprilcube_calibration.calibration_bundle import CalibrationBundle
from g1_aprilcube_calibration.dataset_builder import CalibrationDataset, DatasetBuilder
from g1_dex3_tabletop.calibration_candidates import (
    CandidateDesignConfig,
    camera_info_from_hardware,
    generate_calibration_candidates,
)
from g1_dex3_tabletop.calibration_workflow import (
    solve_fixed_marker_calibration,
    write_calibration_bundle,
)
from g1_dex3_tabletop.planning.contracts import (
    CalibrationPlanRequest,
    CalibrationPlanResult,
    RobotSnapshot,
)
from g1_dex3_tabletop.planning.g1_model import CUROBO_COMMIT
from g1_dex3_tabletop.tabletop_presentation import (
    DIRECT_PRESENTATION_ID,
    PRESENTATION_CONFIGS,
)

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BUNDLE = ROOT / "config/calibrations/dex3_shared_20260812_selected_free.json"
PROVENANCE = ROOT / "config/provenance.json"
CALIBRATION_URDF = ROOT / "config/urdf/g1_29dof_rev_1_0_g1pilot_collision.urdf"
ROBOT_CALIBRATION = ROOT / "third_party/robot_calibration"
ROBOT_CALIBRATION_RUNNER = ROOT / "tools/g1_robot_calibration.sh"
DEFAULT_TASK_CONFIG = ROOT / "config/tabletop/task.yaml"
DEFAULT_CUBE_CONFIG = ROOT / "third_party/aprilcube/models/dex3_safe_cube/config.json"
DEFAULT_QUALITY = ROOT / "config/capture_quality_dex3_aruco.yaml"


def _arm_paths(arm: str) -> tuple[Path, Path]:
    if arm == "right":
        return (
            ROOT / "config/hardware_dex3_aruco.yaml",
            ROOT / "config/dex3_dorsal_aruco_target.json",
        )
    if arm == "left":
        return (
            ROOT / "config/hardware_dex3_left_aruco_id5.yaml",
            ROOT / "config/dex3_left_dorsal_aruco_id5_target.json",
        )
    raise ValueError("arm must be left or right")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="g1-tabletop")
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("inspect", help="read-only repository and pin validation")
    design = subparsers.add_parser(
        "design-calibration",
        help="generate visible marker candidates and one immutable CuRobo request",
    )
    design.add_argument("--arm", choices=("left", "right"), required=True)
    design.add_argument("--snapshot", type=Path, required=True)
    design.add_argument("--output", type=Path, required=True)
    design.add_argument("--calibration-bundle", type=Path, default=DEFAULT_BUNDLE)
    design.add_argument("--target-count", type=int, default=80)
    design.add_argument("--candidate-count", type=int, default=1600)
    design.add_argument("--ik-batch-size", type=int, default=128)
    design.add_argument("--seed", type=int, default=17)
    plan = subparsers.add_parser(
        "plan-calibration",
        help="invoke the isolated CuRobo worker; this never commands the robot",
    )
    plan.add_argument("--request", type=Path, required=True)
    plan.add_argument("--output", type=Path, required=True)
    inspect_plan = subparsers.add_parser(
        "inspect-plan", help="validate a frozen CuRobo plan and print its summary"
    )
    inspect_plan.add_argument("--plan", type=Path, required=True)
    dataset = subparsers.add_parser(
        "build-calibration-dataset",
        help="verify a finalized raw calibration session and build solver input",
    )
    dataset.add_argument("--session", type=Path, required=True)
    dataset.add_argument("--output", type=Path, required=True)
    solve = subparsers.add_parser(
        "solve-calibration",
        help="run fixed-marker Ferguson calibration and emit a removable bundle",
    )
    solve.add_argument("--dataset", type=Path, required=True)
    solve.add_argument("--output-directory", type=Path, required=True)
    solve.add_argument("--bootstrap-trials", type=int, default=50)
    solve.add_argument("--bootstrap-seed", type=int, default=17)
    solve.add_argument("--timeout-s", type=float, default=300.0)
    collect = subparsers.add_parser(
        "collect-calibration",
        help=(
            "single-approval standing CuRobo calibration collection with "
            "bilateral Dex3 preparation"
        ),
    )
    collect.add_argument("--arm", choices=("left", "right"), required=True)
    collect.add_argument("--network-interface", required=True)
    collect.add_argument("--domain-id", type=int, default=0)
    collect.add_argument("--hardware-config", type=Path)
    collect.add_argument("--target-config", type=Path)
    collect.add_argument("--quality-config", type=Path, default=DEFAULT_QUALITY)
    collect.add_argument("--calibration-bundle", type=Path, default=DEFAULT_BUNDLE)
    collect.add_argument("--session-directory", type=Path)
    collect.add_argument("--target-count", type=int, default=80)
    collect.add_argument("--candidate-count", type=int, default=1600)
    collect.add_argument("--ik-batch-size", type=int, default=128)
    collect.add_argument("--seed", type=int, default=17)
    collect.add_argument("--camera-timeout-s", type=float, default=10.0)
    collect.add_argument("--burst-timeout-s", type=float, default=3.0)
    collect.add_argument(
        "--pc2-host",
        default=os.environ.get("G1_PC2_HOST", "unitree@192.168.123.164"),
    )
    collect.add_argument(
        "--pc2-ssh-identity",
        type=Path,
        default=Path(
            os.environ.get(
                "G1_PC2_SSH_IDENTITY",
                str(Path.home() / ".ssh/g1_pc2_ed25519"),
            )
        ),
    )
    collect.add_argument("--confirm", required=True)
    collect.add_argument("--no-window", action="store_true")
    collect.add_argument(
        "--lock-file",
        type=Path,
        default=Path("/tmp/g1-dex3-tabletop-command.lock"),
    )
    tabletop = subparsers.add_parser(
        "run-tabletop",
        help="single-approval seated AprilCube pick, 100mm lift, and exact replacement",
    )
    tabletop.add_argument("--arm", choices=("left", "right"), required=True)
    tabletop.add_argument("--network-interface", required=True)
    tabletop.add_argument("--domain-id", type=int, default=0)
    tabletop.add_argument("--hardware-config", type=Path)
    tabletop.add_argument("--calibration-bundle", type=Path, default=DEFAULT_BUNDLE)
    tabletop.add_argument("--task-config", type=Path, default=DEFAULT_TASK_CONFIG)
    tabletop.add_argument("--cube-config", type=Path, default=DEFAULT_CUBE_CONFIG)
    tabletop.add_argument("--quality-config", type=Path, default=DEFAULT_QUALITY)
    tabletop.add_argument(
        "--presentation",
        choices=(DIRECT_PRESENTATION_ID, *PRESENTATION_CONFIGS),
        default=DIRECT_PRESENTATION_ID,
        help="object presentation; direct keeps the existing tabletop behavior",
    )
    tabletop.add_argument(
        "--grasp-shortlist",
        type=Path,
        help="optional direct-table shortlist override; unavailable for fixture modes",
    )
    tabletop.add_argument("--output-root", type=Path, default=ROOT / "runs")
    tabletop.add_argument("--observation-frames", type=int, default=5)
    tabletop.add_argument(
        "--motion-controller",
        choices=("trajectory", "mpc"),
        default="trajectory",
        help=(
            "arm-motion controller for every normal tabletop phase; mpc uses "
            "phase-aware continuously replenished CuRobo windows"
        ),
    )
    tabletop.add_argument(
        "--pc2-host",
        default=os.environ.get("G1_PC2_HOST", "unitree@192.168.123.164"),
    )
    tabletop.add_argument(
        "--pc2-ssh-identity",
        type=Path,
        default=Path(
            os.environ.get(
                "G1_PC2_SSH_IDENTITY",
                str(Path.home() / ".ssh/g1_pc2_ed25519"),
            )
        ),
    )
    tabletop.add_argument(
        "--confirm",
        required=True,
        help=("must equal the load-bearing-harness and clear-workspace acknowledgement"),
    )
    tabletop.add_argument("--no-window", action="store_true")
    tabletop.add_argument(
        "--skip-camera-recording",
        action="store_true",
        help=(
            "exclude raw RGB, native depth, and their CameraInfo from the MCAP; "
            "camera perception and low-bandwidth RealSense motion recording remain active"
        ),
    )
    tabletop.add_argument(
        "--lock-file",
        type=Path,
        default=Path("/tmp/g1-dex3-tabletop-command.lock"),
    )
    compliance = subparsers.add_parser(
        "measure-seat-compliance",
        help=("seated fixed-ChArUco A/B diagnostic: lift and exactly return both arms"),
    )
    compliance.add_argument("--network-interface", required=True)
    compliance.add_argument("--domain-id", type=int, default=0)
    compliance.add_argument("--hardware-config", type=Path)
    compliance.add_argument("--calibration-bundle", type=Path, default=DEFAULT_BUNDLE)
    compliance.add_argument("--task-config", type=Path, default=DEFAULT_TASK_CONFIG)
    compliance.add_argument("--chair-condition", choices=("cushion", "rigid"), required=True)
    compliance.add_argument("--repetitions", type=int, default=5)
    compliance.add_argument("--observation-frames", type=int, default=5)
    compliance.add_argument("--seed", type=int, default=17)
    compliance.add_argument("--output-root", type=Path, default=ROOT / "runs")
    compliance.add_argument(
        "--pc2-host",
        default=os.environ.get("G1_PC2_HOST", "unitree@192.168.123.164"),
    )
    compliance.add_argument(
        "--pc2-ssh-identity",
        type=Path,
        default=Path(
            os.environ.get(
                "G1_PC2_SSH_IDENTITY",
                str(Path.home() / ".ssh/g1_pc2_ed25519"),
            )
        ),
    )
    compliance.add_argument("--confirm", required=True)
    compliance.add_argument("--no-window", action="store_true")
    compliance.add_argument(
        "--skip-camera-recording",
        action="store_true",
        help=(
            "exclude raw RGB, native depth, and their CameraInfo from the MCAP; "
            "live ChArUco perception and low-bandwidth IMU recording remain active"
        ),
    )
    compliance.add_argument(
        "--lock-file",
        type=Path,
        default=Path("/tmp/g1-dex3-tabletop-command.lock"),
    )
    return parser


def run_inspect(_args: argparse.Namespace) -> int:
    provenance = json.loads(PROVENANCE.read_text(encoding="utf-8"))
    actual = {name: _git_head(ROOT / "third_party" / name) for name in provenance["upstream"]}
    expected = {name: relative["commit"] for name, relative in provenance["upstream"].items()}
    if actual != expected:
        raise ValueError(f"third-party pins differ: expected={expected}, actual={actual}")
    bundle = CalibrationBundle.load(DEFAULT_BUNDLE)
    result = {
        "commands_robot": False,
        "repository": str(ROOT),
        "third_party_commits": actual,
        "curobo_model_commit": CUROBO_COMMIT,
        "calibration_bundle_id": bundle.bundle_id,
        "calibration_bundle_sha256": bundle.content_sha256,
        "running_notes": str(ROOT / "running_notes.md"),
        "data_recording_documentation": str(ROOT / "docs/data-recording.md"),
        "planner_environment": str(ROOT / ".venv-planner"),
        "control_environment": str(ROOT / ".venv"),
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


def run_design_calibration(args: argparse.Namespace) -> int:
    if args.output.exists():
        raise FileExistsError(f"planner request already exists: {args.output}")
    hardware_path, target_path = _arm_paths(args.arm)
    hardware = yaml.safe_load(hardware_path.read_text(encoding="utf-8"))
    target = json.loads(target_path.read_text(encoding="utf-8"))
    snapshot = RobotSnapshot.from_dict(json.loads(args.snapshot.read_text(encoding="utf-8")))
    bundle = CalibrationBundle.load(args.calibration_bundle)
    palm_T_marker = hardware["robot"]["calibration_target_modeled_hand_T_target"]
    design = CandidateDesignConfig(
        target_count=args.target_count,
        candidate_count=args.candidate_count,
        seed=args.seed,
    )
    candidates = generate_calibration_candidates(
        camera_info=camera_info_from_hardware(hardware),
        target_config=target,
        torso_T_camera=bundle.torso_T_camera,
        palm_T_marker=palm_T_marker,
        config=design,
    )
    request = CalibrationPlanRequest(
        arm=args.arm,
        snapshot=snapshot,
        torso_T_camera=tuple(tuple(float(v) for v in row) for row in bundle.torso_T_camera),
        palm_T_marker=tuple(tuple(float(v) for v in row) for row in palm_T_marker),
        joint_position_offsets_rad=dict(bundle.joint_position_offsets_rad),
        candidates=candidates,
        selection_config=design.to_dict(),
        target_count=args.target_count,
        ik_batch_size=args.ik_batch_size,
        random_seed=args.seed,
    )
    request.write_json(args.output)
    print(
        json.dumps(
            {
                "commands_robot": False,
                "output": str(args.output.resolve()),
                "request_sha256": request.content_sha256,
                "arm": args.arm,
                "geometrically_visible_candidates": len(candidates),
                "requested_capture_count": args.target_count,
                "marker_transform_policy": "fixed_CAD_palm_T_marker",
                "next": (
                    f"./tools/g1_tabletop.sh plan-calibration --request {args.output} "
                    f"--output {args.output.with_name('calibration_plan.json')}"
                ),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def run_plan_calibration(args: argparse.Namespace) -> int:
    CalibrationPlanRequest.from_json(args.request)
    worker = ROOT / ".venv-planner/bin/g1-curobo-worker"
    if not worker.is_file():
        raise FileNotFoundError(
            f"planner environment is unavailable: {worker}; run ./tools/setup_planner_env.sh"
        )
    completed = subprocess.run(
        [
            str(worker),
            "plan-calibration",
            "--request",
            str(args.request),
            "--output",
            str(args.output),
        ],
        check=False,
    )
    return int(completed.returncode)


def run_inspect_plan(args: argparse.Namespace) -> int:
    plan = CalibrationPlanResult.from_json(args.plan)
    durations = [trajectory.sample_time_s[-1] for trajectory in plan.trajectories]
    print(
        json.dumps(
            {
                "commands_robot": False,
                "plan_sha256": plan.content_sha256,
                "request_sha256": plan.request_sha256,
                "arm": plan.arm,
                "capture_count": len(plan.capture_pose_ids),
                "trajectory_count": len(plan.trajectories),
                "route_duration_s": sum(durations),
                "returns_to_handoff": plan.route_pose_ids[-1] == "__handoff__",
                "planner_provenance": plan.planner_provenance,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def run_build_calibration_dataset(args: argparse.Namespace) -> int:
    dataset = DatasetBuilder(args.session).build(output_path=args.output)
    print(
        json.dumps(
            {
                "commands_robot": False,
                "session_id": dataset.session_id,
                "arm": dataset.calibration_arm,
                "sample_count": len(dataset.samples),
                "content_sha256": dataset.content_sha256,
                "output": str(args.output.resolve()),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def run_solve_calibration(args: argparse.Namespace) -> int:
    dataset = CalibrationDataset.from_json(args.dataset)
    hardware_path, target_path = _arm_paths(dataset.calibration_arm)
    result, provenance = solve_fixed_marker_calibration(
        dataset,
        hardware_path=hardware_path,
        target_path=target_path,
        urdf_path=CALIBRATION_URDF,
        output_directory=args.output_directory,
        robot_calibration_directory=ROBOT_CALIBRATION,
        runner_path=ROBOT_CALIBRATION_RUNNER,
        bootstrap_trials=args.bootstrap_trials,
        bootstrap_seed=args.bootstrap_seed,
        timeout_s=args.timeout_s,
    )
    left_hardware, left_target = _arm_paths("left")
    right_hardware, right_target = _arm_paths("right")
    bundle_path = write_calibration_bundle(
        result=result,
        dataset=dataset,
        output_directory=args.output_directory,
        urdf_path=CALIBRATION_URDF,
        left_hardware_path=left_hardware,
        left_target_path=left_target,
        right_hardware_path=right_hardware,
        right_target_path=right_target,
        provenance=provenance,
    )
    print(
        json.dumps(
            {
                "commands_robot": False,
                "arm": dataset.calibration_arm,
                "sample_count": len(dataset.samples),
                "training_sample_count": len(result.training_samples),
                "holdout_sample_count": len(result.holdout_samples),
                "training_radial_rms_px": result.residuals.training.rms_px,
                "holdout_radial_rms_px": result.residuals.holdout.rms_px,
                "observable": result.solution.observability.observable,
                "free_parameters": list(result.solution.parameter_names),
                "target_transform_policy": "fixed_CAD_palm_T_marker",
                "result": str((args.output_directory / "result.json").resolve()),
                "calibration_bundle": str(bundle_path.resolve()),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def _git_head(path: Path) -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=path,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    handlers = {
        "inspect": run_inspect,
        "design-calibration": run_design_calibration,
        "plan-calibration": run_plan_calibration,
        "inspect-plan": run_inspect_plan,
        "build-calibration-dataset": run_build_calibration_dataset,
        "solve-calibration": run_solve_calibration,
    }
    if args.command == "run-tabletop":
        from g1_dex3_tabletop.hardware_tabletop import run_tabletop

        hardware_path, _target_path = _arm_paths(args.arm)
        if args.hardware_config is None:
            args.hardware_config = hardware_path
        handlers["run-tabletop"] = run_tabletop
    if args.command == "collect-calibration":
        from g1_dex3_tabletop.hardware_calibration import run_collect_calibration

        hardware_path, target_path = _arm_paths(args.arm)
        if args.hardware_config is None:
            args.hardware_config = hardware_path
        if args.target_config is None:
            args.target_config = target_path
        handlers["collect-calibration"] = run_collect_calibration
    if args.command == "measure-seat-compliance":
        from g1_dex3_tabletop.hardware_seat_compliance import (
            run_measure_seat_compliance,
        )

        hardware_path, _target_path = _arm_paths("right")
        if args.hardware_config is None:
            args.hardware_config = hardware_path
        handlers["measure-seat-compliance"] = run_measure_seat_compliance
    try:
        return handlers[args.command](args)
    except KeyboardInterrupt:
        print("interrupted by operator", file=sys.stderr)
        return 130
    except (
        FileNotFoundError,
        FileExistsError,
        KeyError,
        RuntimeError,
        subprocess.CalledProcessError,
        TypeError,
        ValueError,
    ) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
