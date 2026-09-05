"""Focused operator and read-only planning commands."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import yaml
from aprilcube import CorrespondenceDetector

from g1_aprilcube_calibration.calibration_bundle import CalibrationBundle
from g1_aprilcube_calibration.dataset_builder import CalibrationDataset, DatasetBuilder
from g1_aprilcube_calibration.joint_map import arm_indices
from g1_aprilcube_calibration.transforms import validate_transform
from g1_aprilcube_calibration.urdf_model import URDFModel
from g1_dex3_tabletop.calibration import (
    BilateralCalibrationDataset,
    BilateralCalibrationPlanningRequest,
    BilateralDesignConfig,
    BilateralIKResult,
    BilateralModelSpec,
    BilateralRoutePlanningRequest,
    BilateralRoutePlanningResult,
    BilateralSessionStore,
    BilateralValidationConfig,
    BilateralVisibilityConfig,
    CameraFrameArtifact,
    assemble_bilateral_planning_artifacts,
    evaluate_bilateral_anchor_drift,
    merge_bilateral_datasets,
    solve_bilateral_dataset,
    validate_and_select_bilateral_model,
    write_bilateral_calibration_bundle,
)
from g1_dex3_tabletop.calibration_candidates import (
    CandidateDesignConfig,
    camera_info_from_hardware,
    generate_calibration_candidates,
    target_corner_tag_ids,
    target_object_points_m,
)
from g1_dex3_tabletop.calibration_workflow import (
    solve_fixed_marker_calibration,
    write_calibration_bundle,
)
from g1_dex3_tabletop.planning.contracts import (
    CalibrationCandidate,
    CalibrationPlanRequest,
    CalibrationPlanResult,
    Dex3PreparationPlan,
    Dex3PreparationRequest,
    RobotSnapshot,
)
from g1_dex3_tabletop.planning.dex3_handedness import (
    dex3_empty_close_reference,
    dex3_execution_profile,
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
DEFAULT_QUALITY = ROOT / "config/capture_quality_dex3_aruco.yaml"
DEFAULT_BILATERAL_CAMERA_FRAMES = ROOT / "config/cameras/realsense_348522074178_color_frames.json"
DEFAULT_BILATERAL_MODELS = ROOT / "config/calibration_models/bilateral_static_offsets_v1.json"
DEFAULT_LEFT_TARGET = ROOT / "config/dex3_left_dorsal_aruco_id5_target.json"
DEFAULT_RIGHT_TARGET = ROOT / "config/dex3_dorsal_aruco_target.json"


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
    bilateral_plan = subparsers.add_parser(
        "plan-bilateral-calibration",
        help=(
            "generate same-frame bilateral poses and certify the complete route "
            "offline with CuRobo"
        ),
    )
    bilateral_plan.add_argument("--snapshot", type=Path, required=True)
    bilateral_plan.add_argument("--output-directory", type=Path, required=True)
    bilateral_plan.add_argument("--calibration-bundle", type=Path, default=DEFAULT_BUNDLE)
    bilateral_plan.add_argument(
        "--camera-frames",
        type=Path,
        default=DEFAULT_BILATERAL_CAMERA_FRAMES,
    )
    bilateral_plan.add_argument(
        "--models",
        type=Path,
        default=DEFAULT_BILATERAL_MODELS,
    )
    bilateral_plan.add_argument(
        "--design-model",
        default="shared_camera_two_targets_prior_selected_offsets",
    )
    bilateral_plan.add_argument(
        "--left-target-config",
        type=Path,
        default=DEFAULT_LEFT_TARGET,
    )
    bilateral_plan.add_argument(
        "--right-target-config",
        type=Path,
        default=DEFAULT_RIGHT_TARGET,
    )
    bilateral_plan.add_argument(
        "--left-hardware-config",
        type=Path,
        default=ROOT / "config/hardware_dex3_left_aruco_id5.yaml",
    )
    bilateral_plan.add_argument(
        "--right-hardware-config",
        type=Path,
        default=ROOT / "config/hardware_dex3_aruco.yaml",
    )
    bilateral_plan.add_argument("--left-excitation-count", type=int, default=34)
    bilateral_plan.add_argument("--right-excitation-count", type=int, default=34)
    bilateral_plan.add_argument("--anchor-interval", type=int, default=7)
    bilateral_plan.add_argument("--candidate-count", type=int, default=1600)
    bilateral_plan.add_argument("--ik-batch-size", type=int, default=128)
    bilateral_plan.add_argument("--seed", type=int, default=17)
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
    bilateral_dataset = subparsers.add_parser(
        "build-bilateral-calibration-dataset",
        help="re-detect both hand targets in a finalized same-frame raw session",
    )
    bilateral_dataset.add_argument("--session", type=Path, required=True)
    bilateral_dataset.add_argument("--output", type=Path, required=True)
    bilateral_merge = subparsers.add_parser(
        "merge-bilateral-calibration-datasets",
        help="strictly merge compatible bilateral sessions for later-day validation",
    )
    bilateral_merge.add_argument("--datasets", type=Path, nargs="+", required=True)
    bilateral_merge.add_argument("--output", type=Path, required=True)
    bilateral_merge.add_argument("--dataset-id")
    solve = subparsers.add_parser(
        "solve-calibration",
        help="run fixed-marker Ferguson calibration and emit a removable bundle",
    )
    solve.add_argument("--dataset", type=Path, required=True)
    solve.add_argument("--output-directory", type=Path, required=True)
    solve.add_argument("--bootstrap-trials", type=int, default=50)
    solve.add_argument("--bootstrap-seed", type=int, default=17)
    solve.add_argument("--timeout-s", type=float, default=300.0)
    bilateral_solve = subparsers.add_parser(
        "solve-bilateral-calibration",
        help="cross-validate bilateral models and emit a candidate production bundle",
    )
    bilateral_solve.add_argument("--dataset", type=Path, required=True)
    bilateral_solve.add_argument("--output-directory", type=Path, required=True)
    bilateral_solve.add_argument("--bundle-id")
    bilateral_solve.add_argument(
        "--camera-frames",
        type=Path,
        default=DEFAULT_BILATERAL_CAMERA_FRAMES,
    )
    bilateral_solve.add_argument(
        "--models",
        type=Path,
        default=DEFAULT_BILATERAL_MODELS,
    )
    bilateral_solve.add_argument("--pose-fold-count", type=int, default=5)
    bilateral_solve.add_argument("--bootstrap-trials", type=int, default=20)
    bilateral_solve.add_argument("--require-multiple-days", action="store_true")
    bilateral_solve.add_argument("--timeout-s", type=float, default=300.0)
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
    bilateral_collect = subparsers.add_parser(
        "collect-bilateral-calibration",
        help=(
            "execute one frozen CuRobo route and require both hand targets in every saved frame"
        ),
    )
    bilateral_collect.add_argument("--network-interface", required=True)
    bilateral_collect.add_argument("--domain-id", type=int, default=0)
    bilateral_collect.add_argument("--hardware-config", type=Path)
    bilateral_collect.add_argument(
        "--left-target-config",
        type=Path,
        default=DEFAULT_LEFT_TARGET,
    )
    bilateral_collect.add_argument(
        "--right-target-config",
        type=Path,
        default=DEFAULT_RIGHT_TARGET,
    )
    bilateral_collect.add_argument("--quality-config", type=Path, default=DEFAULT_QUALITY)
    bilateral_collect.add_argument(
        "--camera-frames",
        type=Path,
        default=DEFAULT_BILATERAL_CAMERA_FRAMES,
    )
    bilateral_collect.add_argument("--pose-design", type=Path, required=True)
    bilateral_collect.add_argument("--execution-plan", type=Path, required=True)
    bilateral_collect.add_argument("--session-directory", type=Path)
    bilateral_collect.add_argument("--day-group-id")
    bilateral_collect.add_argument("--camera-timeout-s", type=float, default=10.0)
    bilateral_collect.add_argument("--burst-timeout-s", type=float, default=3.0)
    bilateral_collect.add_argument("--maximum-capture-attempts", type=int, default=2)
    bilateral_collect.add_argument("--seed", type=int, default=17)
    bilateral_collect.add_argument(
        "--pc2-host",
        default=os.environ.get("G1_PC2_HOST", "unitree@192.168.123.164"),
    )
    bilateral_collect.add_argument(
        "--pc2-ssh-identity",
        type=Path,
        default=Path(
            os.environ.get(
                "G1_PC2_SSH_IDENTITY",
                str(Path.home() / ".ssh/g1_pc2_ed25519"),
            )
        ),
    )
    bilateral_collect.add_argument("--confirm", required=True)
    bilateral_collect.add_argument("--no-window", action="store_true")
    bilateral_collect.add_argument(
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
    tabletop.add_argument(
        "--object-profile",
        required=True,
        help="bundled tabletop object ID, or a repository-local profile YAML",
    )
    tabletop.add_argument("--quality-config", type=Path, default=DEFAULT_QUALITY)
    tabletop.add_argument(
        "--presentation",
        choices=(DIRECT_PRESENTATION_ID, *PRESENTATION_CONFIGS),
        default=DIRECT_PRESENTATION_ID,
        help="object presentation; direct keeps the existing tabletop behavior",
    )
    tabletop.add_argument("--output-root", type=Path, default=ROOT / "runs")
    tabletop.add_argument("--observation-frames", type=int, default=5)
    tabletop.add_argument(
        "--motion-controller",
        choices=("trajectory", "mpc"),
        default="trajectory",
        help=(
            "controller between the frozen supported escape and exact return; "
            "mpc experimentally tracks a visually updated cube only from "
            "pregrasp to grasp"
        ),
    )
    tabletop.add_argument(
        "--maximum-arm-velocity-rad-s",
        type=float,
        default=None,
        help=(
            "explicit per-run arm trajectory limit; omitted uses the conservative "
            "task-config default"
        ),
    )
    tabletop.add_argument(
        "--pregrasp-distance-m",
        type=float,
        default=None,
        help="object-frame approach distance; omitted uses the object-profile default",
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
    stack = subparsers.add_parser(
        "run-stack",
        help="pick either 60 mm cube and place it directly on the other",
    )
    stack.add_argument("--network-interface", required=True)
    stack.add_argument("--domain-id", type=int, default=0)
    stack.add_argument("--hardware-config", type=Path)
    stack.add_argument("--calibration-bundle", type=Path, default=DEFAULT_BUNDLE)
    stack.add_argument("--task-config", type=Path, default=DEFAULT_TASK_CONFIG)
    stack.add_argument("--quality-config", type=Path, default=DEFAULT_QUALITY)
    stack.add_argument("--output-root", type=Path, default=ROOT / "runs")
    stack.add_argument("--observation-frames", type=int, default=5)
    stack.add_argument(
        "--grasp-retries",
        type=int,
        default=1,
        help=(
            "reobserve and replan this many times after a physically rejected grasp; default: 1"
        ),
    )
    stack.add_argument(
        "--maximum-arm-velocity-rad-s",
        type=float,
        default=None,
        help="explicit per-run limit; omitted uses the task-config default",
    )
    stack.add_argument(
        "--pregrasp-distance-m",
        type=float,
        default=None,
        help="object-frame approach distance; omitted uses the shared 60 mm profile default",
    )
    stack.add_argument(
        "--pc2-host",
        default=os.environ.get("G1_PC2_HOST", "unitree@192.168.123.164"),
    )
    stack.add_argument(
        "--pc2-ssh-identity",
        type=Path,
        default=Path(
            os.environ.get(
                "G1_PC2_SSH_IDENTITY",
                str(Path.home() / ".ssh/g1_pc2_ed25519"),
            )
        ),
    )
    stack.add_argument("--confirm", required=True)
    stack.add_argument("--no-window", action="store_true")
    stack.add_argument(
        "--skip-camera-recording",
        action="store_true",
        help="exclude raw RGB, depth, and CameraInfo from the MCAP",
    )
    stack.add_argument(
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
    empty_close = subparsers.add_parser(
        "measure-dex3-empty-close",
        help="command one empty Dex3 hand open then closed and print measured joints",
    )
    empty_close.add_argument("--arm", choices=("left", "right"), required=True)
    empty_close.add_argument("--network-interface", required=True)
    empty_close.add_argument("--domain-id", type=int, default=0)
    empty_close.add_argument("--hardware-config", type=Path)
    empty_close.add_argument("--confirm", required=True)
    empty_close.add_argument(
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
    return _invoke_planner_worker("plan-calibration", args.request, args.output)


def _invoke_planner_worker(command: str, request: Path, output: Path) -> int:
    worker = ROOT / ".venv-planner/bin/g1-curobo-worker"
    if not worker.is_file():
        raise FileNotFoundError(
            f"planner environment is unavailable: {worker}; run ./tools/setup_planner_env.sh"
        )
    completed = subprocess.run(
        [
            str(worker),
            command,
            "--request",
            str(request),
            "--output",
            str(output),
        ],
        check=False,
    )
    return int(completed.returncode)


def _run_required_planner_worker(command: str, request: Path, output: Path) -> None:
    return_code = _invoke_planner_worker(command, request, output)
    if return_code:
        raise RuntimeError(f"isolated CuRobo worker failed for {command} ({return_code})")


def _run_required_bilateral_design_planner(
    *,
    planning_request: Path,
    ik_result: Path,
    urdf: Path,
    route_request_output: Path,
    route_result_output: Path,
) -> None:
    worker = ROOT / ".venv-planner/bin/g1-curobo-worker"
    if not worker.is_file():
        raise FileNotFoundError(
            f"planner environment is unavailable: {worker}; run ./tools/setup_planner_env.sh"
        )
    completed = subprocess.run(
        [
            str(worker),
            "plan-bilateral-calibration-design",
            "--request",
            str(planning_request),
            "--ik-result",
            str(ik_result),
            "--urdf",
            str(urdf),
            "--route-request-output",
            str(route_request_output),
            "--output",
            str(route_result_output),
        ],
        check=False,
    )
    if completed.returncode:
        raise RuntimeError(
            f"isolated CuRobo worker failed for bilateral design planning ({completed.returncode})"
        )


def _prefix_bilateral_candidates(
    side: str,
    candidates: tuple[CalibrationCandidate, ...],
) -> tuple[CalibrationCandidate, ...]:
    return tuple(
        CalibrationCandidate(
            candidate_id=f"{side}_{candidate.candidate_id}",
            camera_T_marker=candidate.camera_T_marker,
            selection_metadata={
                **candidate.selection_metadata,
                "active_arm": side,
                "source_candidate_id": candidate.candidate_id,
            },
        )
        for candidate in candidates
    )


def run_plan_bilateral_calibration(args: argparse.Namespace) -> int:
    """Build both immutable collection artifacts without commanding the robot."""

    output = args.output_directory.resolve()
    if output.exists():
        raise FileExistsError(f"bilateral planner output already exists: {output}")
    snapshot = RobotSnapshot.from_dict(json.loads(args.snapshot.read_text(encoding="utf-8")))
    bundle = CalibrationBundle.load(args.calibration_bundle)
    urdf_model = URDFModel(CALIBRATION_URDF)
    if bundle.base_urdf_sha256 != urdf_model.sha256:
        raise ValueError("calibration bundle belongs to a different projection URDF")
    camera_frames = CameraFrameArtifact.from_json(args.camera_frames)
    matching_models = tuple(
        model for model in _load_bilateral_models(args.models) if model.name == args.design_model
    )
    if len(matching_models) != 1:
        raise ValueError(
            f"bilateral design model is not present exactly once: {args.design_model}"
        )

    hardware_by_arm = {
        "left": yaml.safe_load(args.left_hardware_config.read_text(encoding="utf-8")),
        "right": yaml.safe_load(args.right_hardware_config.read_text(encoding="utf-8")),
    }
    target_path_by_arm = {
        "left": args.left_target_config,
        "right": args.right_target_config,
    }
    target_by_arm = {
        side: json.loads(path.read_text(encoding="utf-8"))
        for side, path in target_path_by_arm.items()
    }
    camera_info_by_arm = {
        side: camera_info_from_hardware(hardware) for side, hardware in hardware_by_arm.items()
    }
    if camera_info_by_arm["left"] != camera_info_by_arm["right"]:
        raise ValueError("bilateral hardware profiles specify different rectified cameras")
    camera_info = camera_info_by_arm["left"]
    if camera_info.serial_number != camera_frames.camera_serial:
        raise ValueError("camera-frame artifact belongs to a different camera serial")
    if camera_info.frame_id != camera_frames.optical_frame:
        raise ValueError("camera-frame artifact optical frame differs from CameraInfo")
    if camera_frames.urdf_parent_link not in urdf_model.links:
        raise ValueError("camera-frame artifact parent link is absent from the projection URDF")

    target_hashes = {
        side: hashlib.sha256(path.read_bytes()).hexdigest()
        for side, path in target_path_by_arm.items()
    }
    for side in ("left", "right"):
        if bundle.targets[side].target_artifact_sha256 != target_hashes[side]:
            raise ValueError(f"{side} target file differs from the calibration bundle")
    target_tag_ids = {
        side: target_corner_tag_ids(target_by_arm[side]) for side in ("left", "right")
    }
    if set(target_tag_ids["left"]) & set(target_tag_ids["right"]):
        raise ValueError("bilateral target files must use disjoint marker IDs")

    dex3_command_positions = {side: dex3_execution_profile(side)[1] for side in ("left", "right")}
    dex3_model_positions = {
        side: dex3_empty_close_reference(side)[0] for side in ("left", "right")
    }

    output.parent.mkdir(parents=True, exist_ok=True)
    preparation_request = Dex3PreparationRequest(
        snapshot=snapshot,
        joint_position_offsets_rad=dict(bundle.joint_position_offsets_rad),
        left_target_q_rad=dex3_command_positions["left"],
        right_target_q_rad=dex3_command_positions["right"],
        left_settled_target_q_rad=dex3_model_positions["left"],
        right_settled_target_q_rad=dex3_model_positions["right"],
        left_return_target_q_rad=snapshot.left_dex3_q_rad,
        right_return_target_q_rad=snapshot.right_dex3_q_rad,
        random_seed=args.seed,
    )
    with tempfile.TemporaryDirectory(
        prefix=f".{output.name}.preparation.",
        dir=output.parent,
    ) as preparation_directory:
        preparation_root = Path(preparation_directory)
        preparation_request_path = preparation_root / "dex3_preparation_request.json"
        preparation_plan_path = preparation_root / "dex3_preparation_plan.json"
        preparation_request.write_json(preparation_request_path)
        _run_required_planner_worker(
            "plan-dex3-preparation",
            preparation_request_path,
            preparation_plan_path,
        )
        preparation_plan = Dex3PreparationPlan.from_json(preparation_plan_path)
    clearance_q29 = np.asarray(snapshot.measured_q29_rad, dtype=np.float64).copy()
    clearance_q29[np.asarray((*arm_indices("left"), *arm_indices("right")), dtype=np.int64)] = (
        np.asarray(preparation_plan.dual_clearance_q14_rad, dtype=np.float64)
    )
    clearance_snapshot = RobotSnapshot(
        measured_q29_rad=tuple(clearance_q29),
        left_dex3_q_rad=dex3_model_positions["left"],
        right_dex3_q_rad=dex3_model_positions["right"],
    )

    design_config = BilateralDesignConfig(
        left_excitation_count=args.left_excitation_count,
        right_excitation_count=args.right_excitation_count,
        anchor_interval=args.anchor_interval,
    )
    candidates_by_arm: dict[str, tuple[CalibrationCandidate, ...]] = {}
    for side_index, side in enumerate(("left", "right")):
        candidate_config = CandidateDesignConfig(
            target_count=getattr(design_config, f"{side}_excitation_count"),
            candidate_count=args.candidate_count,
            seed=args.seed + side_index,
        )
        candidates_by_arm[side] = _prefix_bilateral_candidates(
            side,
            generate_calibration_candidates(
                camera_info=camera_info,
                target_config=target_by_arm[side],
                torso_T_camera=bundle.torso_T_camera,
                palm_T_marker=bundle.targets[side].hand_T_target,
                exposed_marker_normal=np.asarray(
                    hardware_by_arm[side]["robot"][
                        "calibration_target_exposed_face_normal_target"
                    ],
                    dtype=np.float64,
                ),
                config=candidate_config,
            ),
        )

    request = BilateralCalibrationPlanningRequest(
        robot_model=urdf_model.name,
        urdf_sha256=urdf_model.sha256,
        snapshot=snapshot,
        clearance_snapshot=clearance_snapshot,
        dex3_preparation_request=preparation_request,
        dex3_preparation_plan=preparation_plan,
        camera_info=camera_info.to_dict(),
        camera_frames=camera_frames,
        design_model=matching_models[0],
        design_config=design_config,
        visibility_config=BilateralVisibilityConfig(),
        nominal_torso_T_camera=tuple(
            tuple(float(value) for value in row) for row in bundle.torso_T_camera
        ),
        nominal_hand_T_targets={
            side: tuple(
                tuple(float(value) for value in row) for row in bundle.targets[side].hand_T_target
            )
            for side in ("left", "right")
        },
        joint_position_offsets_rad=dict(bundle.joint_position_offsets_rad),
        dex3_command_positions_rad=dex3_command_positions,
        dex3_model_positions_rad=dex3_model_positions,
        candidates_by_arm=candidates_by_arm,
        target_artifact_sha256_by_arm=target_hashes,
        target_corner_tag_ids_by_arm=target_tag_ids,
        target_object_points_m_by_arm={
            side: tuple(
                tuple(float(value) for value in row)
                for row in target_object_points_m(target_by_arm[side])
            )
            for side in ("left", "right")
        },
        ik_batch_size=args.ik_batch_size,
        random_seed=args.seed,
        source_provenance={
            "command": "g1-tabletop plan-bilateral-calibration",
            "calibration_bundle_path": str(args.calibration_bundle.resolve()),
            "calibration_bundle_sha256": bundle.content_sha256,
            "camera_frames_path": str(args.camera_frames.resolve()),
            "camera_frames_sha256": camera_frames.content_sha256,
            "models_path": str(args.models.resolve()),
            "models_sha256": hashlib.sha256(args.models.read_bytes()).hexdigest(),
            "target_paths": {
                side: str(path.resolve()) for side, path in target_path_by_arm.items()
            },
            "hardware_config_sha256_by_arm": {
                "left": hashlib.sha256(args.left_hardware_config.read_bytes()).hexdigest(),
                "right": hashlib.sha256(args.right_hardware_config.read_bytes()).hexdigest(),
            },
            "snapshot_path": str(args.snapshot.resolve()),
            "snapshot_sha256": hashlib.sha256(args.snapshot.read_bytes()).hexdigest(),
            "ready_snapshot_policy": "normal_measured_ready_not_manual_anchor",
            "dex3_policy": (
                "reference_clearance_then_fixed_close; restore_measured_starting_fingers; "
                "hardware_uses_a_live_reversible_adapter"
            ),
            "dex3_preparation_request_sha256": preparation_request.content_sha256,
            "dex3_preparation_plan_sha256": preparation_plan.content_sha256,
        },
    )

    stage = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent))
    try:
        preparation_request.write_json(stage / "dex3_preparation_request.json")
        preparation_plan.write_json(stage / "dex3_preparation_plan.json")
        request_path = stage / "planning_request.json"
        ik_path = stage / "ik_result.json"
        request.write_json(request_path)
        _run_required_planner_worker("solve-bilateral-calibration-ik", request_path, ik_path)
        ik_result = BilateralIKResult.from_json(ik_path)
        ik_result.validate_request(request)

        route_request_path = stage / "route_request.json"
        route_result_path = stage / "route_result.json"
        _run_required_bilateral_design_planner(
            planning_request=request_path,
            ik_result=ik_path,
            urdf=CALIBRATION_URDF,
            route_request_output=route_request_path,
            route_result_output=route_result_path,
        )
        final_route_request = BilateralRoutePlanningRequest.from_json(route_request_path)
        final_route_result = BilateralRoutePlanningResult.from_json(route_result_path)
        final_route_result.validate_request(final_route_request)
        pose_design, execution_plan = assemble_bilateral_planning_artifacts(
            request,
            ik_result,
            final_route_request,
            final_route_result,
        )
        pose_design.write_json(stage / "pose_design.json")
        execution_plan.write_json(stage / "execution_plan.json")
        os.replace(stage, output)
    except BaseException:
        print(f"bilateral planning failed; diagnostics retained at {stage}", file=sys.stderr)
        raise

    print(
        json.dumps(
            {
                "commands_robot": False,
                "output_directory": str(output),
                "pose_design": str(output / "pose_design.json"),
                "pose_design_sha256": pose_design.content_sha256,
                "execution_plan": str(output / "execution_plan.json"),
                "execution_plan_sha256": execution_plan.content_sha256,
                "selected_excitation_count": len(final_route_request.selection.candidates),
                "capture_count": sum(item.capturable for item in final_route_request.schedule),
                "trajectory_count": len(final_route_result.transitions),
                "connected_candidate_count_by_arm": final_route_request.design_provenance[
                    "anchor_connected_candidate_count_by_arm"
                ],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


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


def run_build_bilateral_calibration_dataset(args: argparse.Namespace) -> int:
    store = BilateralSessionStore(args.session)
    dataset = store.build_dataset(
        detectors={
            "left": CorrespondenceDetector(store.directory / "left_target.json"),
            "right": CorrespondenceDetector(store.directory / "right_target.json"),
        },
        output_path=args.output,
    )
    print(
        json.dumps(
            {
                "commands_robot": False,
                "dataset_id": dataset.dataset_id,
                "source_session_ids": sorted(dataset.session_manifest_sha256_by_id),
                "sample_count": len(dataset.samples),
                "every_sample_is_same_frame_bilateral": True,
                "content_sha256": dataset.content_sha256,
                "output": str(args.output.resolve()),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def run_merge_bilateral_calibration_datasets(args: argparse.Namespace) -> int:
    datasets = tuple(BilateralCalibrationDataset.from_json(path) for path in args.datasets)
    merged = merge_bilateral_datasets(datasets, dataset_id=args.dataset_id)
    merged.write_json(args.output)
    print(
        json.dumps(
            {
                "commands_robot": False,
                "dataset_id": merged.dataset_id,
                "source_session_ids": sorted(merged.session_manifest_sha256_by_id),
                "day_group_ids": sorted({sample.day_group_id for sample in merged.samples}),
                "sample_count": len(merged.samples),
                "content_sha256": merged.content_sha256,
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


def _load_bilateral_models(path: Path) -> tuple[BilateralModelSpec, ...]:
    document = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(document, dict) or set(document) != {
        "schema_version",
        "description",
        "models",
    }:
        raise ValueError("bilateral model configuration fields differ")
    if document["schema_version"] != 1:
        raise ValueError("unsupported bilateral model configuration version")
    return tuple(BilateralModelSpec.from_dict(item) for item in document["models"])


def _initial_bilateral_hand_targets() -> dict[str, np.ndarray]:
    transforms = {}
    for side in ("left", "right"):
        hardware_path, _target_path = _arm_paths(side)
        hardware = yaml.safe_load(hardware_path.read_text(encoding="utf-8"))
        transforms[side] = validate_transform(
            hardware["robot"]["calibration_target_modeled_hand_T_target"]
        )
    return transforms


def run_solve_bilateral_calibration(args: argparse.Namespace) -> int:
    dataset = BilateralCalibrationDataset.from_json(args.dataset)
    camera_frames = CameraFrameArtifact.from_json(args.camera_frames)
    models = _load_bilateral_models(args.models)
    initial_targets = _initial_bilateral_hand_targets()
    urdf_model = URDFModel(CALIBRATION_URDF)
    output = args.output_directory.resolve()

    def fit(
        training_dataset: BilateralCalibrationDataset,
        model: BilateralModelSpec,
        fit_directory: Path,
    ):
        return solve_bilateral_dataset(
            training_dataset,
            urdf_model,
            fit_directory,
            camera_frames=camera_frames,
            model=model,
            initial_hand_T_targets=initial_targets,
            robot_calibration_directory=ROBOT_CALIBRATION,
            runner_path=ROBOT_CALIBRATION_RUNNER,
            timeout_s=args.timeout_s,
        )

    report = validate_and_select_bilateral_model(
        dataset,
        urdf_model,
        camera_frames=camera_frames,
        initial_hand_T_targets=initial_targets,
        models=models,
        fit=fit,
        output_directory=output / "validation",
        config=BilateralValidationConfig(
            pose_fold_count=args.pose_fold_count,
            bootstrap_trials=args.bootstrap_trials,
            require_multiple_days=args.require_multiple_days,
        ),
    )
    if not report.passed:
        raise RuntimeError(
            "no bilateral model passed; inspect "
            f"{output / 'validation' / 'validation_report.json'}"
        )
    result = fit(dataset, report.selected_model, output / "full_dataset_fit")
    anchor_drift = evaluate_bilateral_anchor_drift(
        dataset,
        urdf_model,
        result,
        config=report.config,
    )
    bundle_id = args.bundle_id or output.name
    bundle_path = write_bilateral_calibration_bundle(
        result=result,
        dataset=dataset,
        validation_report=report,
        anchor_drift_report=anchor_drift,
        camera_frames=camera_frames,
        initial_hand_T_targets=initial_targets,
        base_urdf_path=CALIBRATION_URDF,
        destination=output / "calibration_bundle.json",
        bundle_id=bundle_id,
        provenance={
            "command": "g1-tabletop solve-bilateral-calibration",
            "models_config_sha256": hashlib.sha256(args.models.read_bytes()).hexdigest(),
            "models_config_path": str(args.models.resolve()),
        },
    )
    selected = next(
        candidate
        for candidate in report.candidates
        if candidate.model.content_sha256 == report.selected_model_sha256
    )
    print(
        json.dumps(
            {
                "commands_robot": False,
                "sample_count": len(dataset.samples),
                "selected_model": report.selected_model_name,
                "selected_model_sha256": report.selected_model_sha256,
                "pose_holdout": selected.pose_holdout.to_dict(),
                "day_holdout": (
                    None if selected.day_holdout is None else selected.day_holdout.to_dict()
                ),
                "anchor_drift": anchor_drift.to_dict(),
                "validation_report": str(
                    (output / "validation" / "validation_report.json").resolve()
                ),
                "calibration_bundle": str(bundle_path),
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
        "plan-bilateral-calibration": run_plan_bilateral_calibration,
        "inspect-plan": run_inspect_plan,
        "build-calibration-dataset": run_build_calibration_dataset,
        "build-bilateral-calibration-dataset": run_build_bilateral_calibration_dataset,
        "merge-bilateral-calibration-datasets": run_merge_bilateral_calibration_datasets,
        "solve-calibration": run_solve_calibration,
        "solve-bilateral-calibration": run_solve_bilateral_calibration,
    }
    if args.command == "run-tabletop":
        from g1_dex3_tabletop.hardware_tabletop import run_tabletop

        hardware_path, _target_path = _arm_paths(args.arm)
        if args.hardware_config is None:
            args.hardware_config = hardware_path
        handlers["run-tabletop"] = run_tabletop
    if args.command == "run-stack":
        from g1_dex3_tabletop.hardware_stack import run_stack

        hardware_path, _target_path = _arm_paths("left")
        if args.hardware_config is None:
            args.hardware_config = hardware_path
        handlers["run-stack"] = run_stack
    if args.command == "collect-calibration":
        from g1_dex3_tabletop.hardware_calibration import run_collect_calibration

        hardware_path, target_path = _arm_paths(args.arm)
        if args.hardware_config is None:
            args.hardware_config = hardware_path
        if args.target_config is None:
            args.target_config = target_path
        handlers["collect-calibration"] = run_collect_calibration
    if args.command == "collect-bilateral-calibration":
        from g1_dex3_tabletop.hardware_bilateral_calibration import (
            run_collect_bilateral_calibration,
        )

        hardware_path, _target_path = _arm_paths("left")
        if args.hardware_config is None:
            args.hardware_config = hardware_path
        handlers["collect-bilateral-calibration"] = run_collect_bilateral_calibration
    if args.command == "measure-seat-compliance":
        from g1_dex3_tabletop.hardware_seat_compliance import (
            run_measure_seat_compliance,
        )

        hardware_path, _target_path = _arm_paths("right")
        if args.hardware_config is None:
            args.hardware_config = hardware_path
        handlers["measure-seat-compliance"] = run_measure_seat_compliance
    if args.command == "measure-dex3-empty-close":
        from g1_dex3_tabletop.hardware_dex3_empty_close import (
            run_measure_dex3_empty_close,
        )

        hardware_path, _target_path = _arm_paths(args.arm)
        if args.hardware_config is None:
            args.hardware_config = hardware_path
        handlers["measure-dex3-empty-close"] = run_measure_dex3_empty_close
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
