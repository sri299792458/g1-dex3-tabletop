#!/usr/bin/env python3
"""Prepare/certify the September 7 diagnostic with the existing collection stack.

CPU preparation:
  .venv/bin/python tools/plan_bilateral_diagnostic.py --prepare-only
CuRobo certification, required before collection:
  .venv-planner/bin/python tools/plan_bilateral_diagnostic.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import replace
from pathlib import Path

import numpy as np

from g1_aprilcube_calibration.joint_map import arm_hand_link, arm_indices, arm_joint_names
from g1_aprilcube_calibration.transforms import invert_transform
from g1_aprilcube_calibration.urdf_model import URDFModel
from g1_dex3_tabletop.calibration.design import (
    BilateralDesignConfig,
    BilateralPoseDesignArtifact,
    linearize_design_candidate,
    select_bilateral_design,
)
from g1_dex3_tabletop.calibration.diagnostic import (
    build_diagnostic_schedule,
    matched_hand_configurations,
    shortest_diagnostic_path,
)
from g1_dex3_tabletop.calibration.models import BilateralModelSpec
from g1_dex3_tabletop.calibration.planning import (
    BilateralCalibrationPlanningRequest,
    BilateralFeasiblePose,
    BilateralIKResult,
    BilateralRoutePlanningRequest,
    _predicted_sample,
    _visibility_bins,
    assemble_bilateral_planning_artifacts,
)
from g1_dex3_tabletop.calibration.projection import BilateralCalibrationProjection
from g1_dex3_tabletop.calibration.visibility import BilateralSightlineChecker
from g1_dex3_tabletop.planning.contracts import CalibrationCandidate, atomic_write_json

ROOT = Path(__file__).resolve().parents[1]
ANALYSIS = ROOT / "work/calibration_analysis_231145"
BASE_POSES = (
    "left_candidate_3089",
    "right_candidate_2376",
    "left_candidate_6003",
    "right_candidate_1336",
)


def prepare(output, source_path):
    core_path = ROOT / "work/bilateral_full_20260904_reuse/pose_design.json"
    route_path = ROOT / "work/bilateral_full_20260904_reuse/route_request.json"
    solution_path = ANALYSIS / "standard_solve/full_dataset_fit/solution.json"
    source = BilateralCalibrationPlanningRequest.from_json(source_path)
    from g1_dex3_tabletop.calibration.hand_posture import HELD_FINGER_POLICY, check_full_close

    if not source.dex3_preparation_request.holds_measured_fingers or (
        source.dex3_preparation_plan.planner_provenance.get("finger_motion_policy")
        != HELD_FINGER_POLICY
    ):
        raise ValueError("Regenerate --planning-request with operator-preclosed measured hold")
    check_full_close(
        left_q_rad=source.snapshot.left_dex3_q_rad,
        right_q_rad=source.snapshot.right_dex3_q_rad,
        tolerance_rad=0.08,
    )
    core = BilateralPoseDesignArtifact.from_json(core_path)
    reference_route = BilateralRoutePlanningRequest.from_json(route_path)
    reference_route = replace(
        reference_route,
        snapshot=source.snapshot,
        clearance_snapshot=source.clearance_snapshot,
        dex3_preparation_request=source.dex3_preparation_request,
        dex3_preparation_plan=source.dex3_preparation_plan,
        dex3_command_positions_rad=source.dex3_command_positions_rad,
        dex3_model_positions_rad=source.dex3_model_positions_rad,
    )
    solution = json.loads(solution_path.read_text())
    offsets = {
        name: value for name, value in solution["parameters"].items() if name.endswith("_joint")
    }
    # Use the declared diagnostic model for both FK equivalence and clearance.
    # This is scoped to this plan and does not deploy a calibration bundle.
    source = replace(
        source,
        design_model=BilateralModelSpec.from_dict(solution["model"]),
        nominal_torso_T_camera=solution["torso_T_camera"],
        nominal_hand_T_targets=solution["hand_T_targets"],
        joint_position_offsets_rad=offsets,
    )
    urdf = URDFModel(ROOT / "sessions/dex3_bilateral_curobo_20260905T231145Z/robot.urdf")
    if urdf.sha256 != source.urdf_sha256:
        raise ValueError("diagnostic URDF differs from the reference plan")
    sightlines = BilateralSightlineChecker(source, urdf)
    projection = BilateralCalibrationProjection(
        urdf,
        camera_frames=source.camera_frames,
        model=source.design_model,
        initial_hand_T_targets=source.nominal_hand_T_targets,
    )
    parameters = projection.parameters_for_nominal_model(
        torso_T_camera=np.asarray(source.nominal_torso_T_camera),
        hand_T_targets={
            side: np.asarray(value) for side, value in source.nominal_hand_T_targets.items()
        },
        joint_position_offsets_rad=offsets,
    )
    positions = {name: np.asarray(q) for name, q in core.waypoint_joint_positions_rad.items()}
    excluded_transit_poses = []
    for name, q in tuple(positions.items()):
        invalid = False
        for side in ("left", "right"):
            names = arm_joint_names(side)
            corrected = q[list(arm_indices(side))] + np.array(
                [offsets.get(joint, 0.0) for joint in names]
            )
            limits = urdf.joint_limits(names)
            invalid = invalid or any(
                value < limit.lower or value > limit.upper
                for value, limit in zip(corrected, limits, strict=True)
            )
        if invalid:
            if name in (*BASE_POSES, "bilateral_anchor"):
                raise ValueError(f"{name} exceeds arm joint limits under the diagnostic model")
            excluded_transit_poses.append(name)
            del positions[name]
    anchor = positions["bilateral_anchor"]
    _, predicted = _predicted_sample(
        source,
        projection,
        parameters,
        candidate_id="bilateral_anchor",
        full_q=anchor,
        capture_role="anchor",
    )
    _visibility_bins(source, predicted)
    sightlines.check(anchor, required_sides=("left", "right"))

    def visible(name, side, q):
        _, predicted = _predicted_sample(
            source,
            projection,
            parameters,
            candidate_id=name,
            full_q=q,
            capture_role="excitation",
            active_arm=side,
        )
        _visibility_bins(source, predicted, active_arm=side)
        sightlines.check(q, required_sides=(side,))

    families = []
    for name in BASE_POSES:
        side = name.split("_")[0]
        indices = list(arm_indices(side))
        q = positions[name]
        visible(name, side, q)
        names = arm_joint_names(side)
        limits = urdf.joint_limits(names)
        candidates = matched_hand_configurations(
            urdf, arm=side, command_q=q[indices], joint_offsets=offsets
        )
        alternatives, rejected = [], []
        for index, candidate in enumerate(candidates):
            if (
                candidate["elbow_displacement_m"] < 0.02
                or candidate["maximum_joint_change_rad"] > 0.85
            ):
                continue
            candidate_id = f"{name}_matched_{index:02d}"
            paired = q.copy()
            paired[indices] = candidate["command_q_rad"]
            try:
                visible(candidate_id, side, paired)
            except ValueError as error:
                rejected.append({"candidate_id": candidate_id, "reason": str(error)})
                continue
            positions[candidate_id] = paired
            alternatives.append({"candidate_id": candidate_id, **candidate})
        approaches = []
        for joint_index in (0, 2, 3, 4):
            pair = []
            for direction, label in ((-1, "minus"), (1, "plus")):
                value = q[indices[joint_index]] + direction * np.deg2rad(5)
                corrected = value + offsets.get(names[joint_index], 0.0)
                if (
                    not limits[joint_index].lower + 1e-4
                    < corrected
                    < limits[joint_index].upper - 1e-4
                ):
                    break
                helper_id = f"{name}_approach_{joint_index}_{label}"
                helper = q.copy()
                helper[indices[joint_index]] = value
                pair.append((helper_id, helper))
            if len(pair) == 2:
                positions.update(pair)
                approaches.append(
                    {"joint": names[joint_index], "candidate_ids": [item[0] for item in pair]}
                )
        families.append(
            {
                "arm": side,
                "base_id": name,
                "alternatives": alternatives,
                "approaches": approaches,
                "visibility_rejections": rejected,
                "fk_proposal_count": len(candidates),
            }
        )
        print(
            f"{name}: {len(alternatives)} visible matched-hand proposals; {len(approaches)} opposed approach pairs",
            flush=True,
        )
    protocol = {
        "commands_robot": False,
        "status": "prepared_requires_curobo_certification",
        "purpose": "return repeatability and same predicted hand pose with different arm configuration",
        "blocks": 3,
        "base_returns": 24,
        "base_pose_ids": list(BASE_POSES),
        "families": families,
        "minimum_alternative_elbow_displacement_m": 0.02,
        "excluded_transit_poses_outside_joint_limits": excluded_transit_poses,
        "approach_joint_offset_deg": 5,
        "held_state_analysis": {
            "projection": "frozen source solution evaluated at measured joints, using all four corners",
            "maximum_predicted_hand_translation_difference_m": 0.0005,
            "maximum_predicted_hand_rotation_difference_deg": 0.15,
            "maximum_predicted_corner_radial_difference_px": 0.5,
            "mismatch_policy": "report unmatched comparisons as inconclusive; retain raw and compensated differences",
            "noise_reference": "within-hold variation and independent return variation, not frames treated as independent visits",
            "split_policy": "keep every revisit and matched-hand family in the same statistical group",
            "accuracy_claim": "diagnostic only; no full-model calibration or absolute 3D accuracy claim",
        },
        "recording": "existing full continuous RGB/depth/state recorder plus raw stationary bursts; one attempt per pose",
        "sightline_check": sightlines.provenance,
        "sources": {
            str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in (core_path, source_path, route_path, solution_path, urdf.path)
        },
    }
    atomic_write_json(output / "diagnostic_protocol.json", protocol)
    atomic_write_json(
        output / "proposed_joint_poses.json", {name: q.tolist() for name, q in positions.items()}
    )
    return source, reference_route, urdf, projection, parameters, positions, families, protocol


def certify(output, prepared):
    import torch
    from curobo.types import DeviceCfg

    from g1_dex3_tabletop.planning import curobo_backend as backend

    # Force actual context creation: NVML/device_count alone can succeed with a broken UVM device.
    torch.zeros(1, device="cuda")
    source, reference_route, urdf, projection, parameters, positions, families, protocol = prepared
    cfg = DeviceCfg(device=torch.device("cuda:0"), dtype=torch.float32)
    robot, _ = backend.build_robot_config_for_active_joints(
        active_joint_names=(*arm_joint_names("left"), *arm_joint_names("right")),
        snapshot=reference_route.calibration_snapshot,
        joint_position_offsets_rad=source.joint_position_offsets_rad,
        ignore_internal_hand_collisions=True,
        ignore_adjacent_shoulder_collisions=True,
        ignore_static_body_collisions=True,
    )
    checker = backend.CuroboKinematicCollisionChecker(robot=robot, device_cfg=cfg)
    joint_names = tuple(checker.kinematics.joint_names)
    model_q = {
        name: backend._full_arm_model_reference(
            q,
            joint_position_offsets_rad=source.joint_position_offsets_rad,
            joint_names=joint_names,
        )
        for name, q in positions.items()
    }
    reference, _ = checker.self_collision_link_pair_clearances(
        model_q["bilateral_anchor"][None], joint_names=joint_names
    )
    required = torch.full_like(reference[0], backend.CALIBRATION_SELF_CLEARANCE_M)
    if float(reference.min().cpu()) < backend.CALIBRATION_SELF_CLEARANCE_M - 1e-6:
        raise ValueError(
            "reference anchor fails strict 10 mm clearance under the diagnostic model"
        )
    edges_by_arm = {}
    for side in ("left", "right"):
        ids = sorted(
            name for name in positions if name == "bilateral_anchor" or name.startswith(f"{side}_")
        )
        pairs = set()
        for name in ids:
            nearest = sorted(
                (float(np.linalg.norm(model_q[name] - model_q[other])), other)
                for other in ids
                if other != name
            )
            for _, other in nearest[:8]:
                pairs.add(tuple(sorted((name, other))))
            if name != "bilateral_anchor":
                pairs.add(tuple(sorted((name, "bilateral_anchor"))))
        for family in families:
            if family["arm"] != side:
                continue
            neighbors = [item["candidate_id"] for item in family["alternatives"]]
            neighbors += [item for pair in family["approaches"] for item in pair["candidate_ids"]]
            pairs.update(tuple(sorted((family["base_id"], other))) for other in neighbors)
        edges = [(float(np.linalg.norm(model_q[a] - model_q[b])), a, b) for a, b in sorted(pairs)]
        edges_by_arm[side] = backend._edge_clearance_passes(
            checker=checker,
            joint_names=joint_names,
            required_clearance=required,
            edges=edges,
            q_by_id=model_q,
        )
        print(
            f"{side}: {len(edges_by_arm[side])}/{len(edges)} edges pass strict 10 mm clearance",
            flush=True,
        )
    selected_families, capture_ids = [], []
    for family in families:
        side, base = family["arm"], family["base_id"]
        edges = edges_by_arm[side]
        shortest_diagnostic_path(edges, "bilateral_anchor", base)
        pairs = {frozenset((a, b)) for _, a, b in edges}
        approaches = next(
            (
                item
                for item in family["approaches"]
                if all(frozenset((base, name)) in pairs for name in item["candidate_ids"])
            ),
            None,
        )
        if approaches is None:
            raise ValueError(f"{base} has no certified opposed approach pair")
        selected = {
            "arm": side,
            "base_id": base,
            "approach_ids": approaches["candidate_ids"],
            "approach_joint": approaches["joint"],
        }
        capture_ids.append((base, side))
        options = sorted(
            family["alternatives"],
            key=lambda item: (
                -min(item["elbow_displacement_m"], 0.06),
                item["maximum_joint_change_rad"],
            ),
        )
        for alternate in options:
            try:
                shortest_diagnostic_path(edges, base, alternate["candidate_id"])
            except ValueError:
                continue
            selected.update(
                alternate_id=alternate["candidate_id"], matched_hand_evidence=alternate
            )
            capture_ids.append((alternate["candidate_id"], side))
            break
        selected_families.append(selected)
    if {item["arm"] for item in selected_families if "alternate_id" in item} != {"left", "right"}:
        raise ValueError(
            "no useful certified matched-hand pair for both arms; inspect proposals before changing the experiment"
        )
    schedule, visits = build_diagnostic_schedule(tuple(selected_families), edges_by_arm)
    candidates, ik_poses, requested = [], [], {"left": [], "right": []}
    for name, side in capture_ids:
        q = positions[name]
        sample, predicted = _predicted_sample(
            source,
            projection,
            parameters,
            candidate_id=name,
            full_q=q,
            capture_role="excitation",
            active_arm=side,
        )
        active = q[list(arm_indices(side))]
        names = arm_joint_names(side)
        offsets = np.array([source.joint_position_offsets_rad.get(joint, 0.0) for joint in names])
        limits = urdf.joint_limits(names)
        lower, upper = (
            np.array([limit.lower for limit in limits]),
            np.array([limit.upper for limit in limits]),
        )
        candidates.append(
            linearize_design_candidate(
                candidate_id=name,
                active_arm=side,
                normalized_active_q=tuple(2 * (active + offsets - lower) / (upper - lower) - 1),
                predicted_sample=sample,
                projection=projection,
                parameters=parameters,
                image_coverage_bins=_visibility_bins(source, predicted, active_arm=side),
            )
        )
        ik_poses.append(
            BilateralFeasiblePose(
                name,
                side,
                tuple(active + offsets),
                tuple(active),
                tuple(q),
                0.0,
                0.0,
                {"source": "diagnostic FK proposals independently certified with CuRobo"},
            )
        )
        torso_T_hand = urdf.transform(
            "torso_link", arm_hand_link(side), dict(zip(names, active + offsets, strict=True))
        )
        camera_T_marker = (
            invert_transform(np.asarray(source.nominal_torso_T_camera))
            @ torso_T_hand
            @ np.asarray(source.nominal_hand_T_targets[side])
        )
        requested[side].append(
            CalibrationCandidate(
                name, tuple(map(tuple, camera_T_marker)), {"purpose": "diagnostic"}
            )
        )
    design_config = BilateralDesignConfig(
        left_excitation_count=len(requested["left"]),
        right_excitation_count=len(requested["right"]),
        require_observable_model=False,
        require_full_joint_excitation=False,
    )
    selection = select_bilateral_design(
        tuple(candidates), parameter_names=projection.parameter_names, config=design_config
    )
    protocol.update(
        status="certified",
        selected_families=selected_families,
        visits_by_occurrence=visits,
        capture_count=sum(item.capturable for item in schedule),
        anchor_count=sum(item.capture_role == "anchor" for item in schedule),
    )
    source = replace(
        source,
        candidates_by_arm={side: tuple(items) for side, items in requested.items()},
        design_config=design_config,
        source_provenance={**source.source_provenance, "diagnostic": protocol},
    )
    ik = BilateralIKResult(
        source.content_sha256,
        tuple(ik_poses),
        {"method": "constrained FK matching with independent CuRobo edge certification"},
    )
    route = replace(
        reference_route,
        planning_request_sha256=source.content_sha256,
        ik_result_sha256=ik.content_sha256,
        joint_position_offsets_rad=source.joint_position_offsets_rad,
        parameter_names=projection.parameter_names,
        selection=selection,
        schedule=schedule,
        waypoint_joint_positions_rad={
            item.candidate_id: tuple(positions[item.candidate_id]) for item in schedule
        },
        design_provenance={
            "diagnostic": protocol,
            "nominal_parameter_values": parameters,
            "sightline_check": protocol["sightline_check"],
        },
    )
    result = backend.plan_bilateral_calibration_route(route, progress=print)
    if not result.connected:
        atomic_write_json(output / "route_rejection.json", result.to_dict())
        raise ValueError("independent route certification failed; see route_rejection.json")
    design, execution = assemble_bilateral_planning_artifacts(source, ik, route, result)
    source.write_json(output / "planning_request.json")
    ik.write_json(output / "ik_result.json")
    route.write_json(output / "route_request.json")
    result.write_json(output / "route_result.json")
    design.write_json(output / "pose_design.json")
    execution.write_json(output / "execution_plan.json")
    protocol["route_duration_s"] = sum(
        item.trajectory.sample_time_s[-1] for item in execution.transitions
    )
    protocol["minimum_clearance_m"] = execution.self_clearance_certificate["minimum_clearance_m"]
    atomic_write_json(output / "diagnostic_protocol.json", protocol)
    print(
        json.dumps(
            {
                "commands_robot": False,
                "capture_count": protocol["capture_count"],
                "route_duration_s": protocol["route_duration_s"],
                "minimum_clearance_m": protocol["minimum_clearance_m"],
                "output_directory": str(output),
            },
            indent=2,
        )
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-directory", type=Path, default=ROOT / "work/bilateral_diagnostic_20260907"
    )
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument(
        "--planning-request",
        type=Path,
        required=True,
        help="New operator-preclosed planning_request.json",
    )
    args = parser.parse_args()
    output = args.output_directory.resolve()
    if (output / "execution_plan.json").exists() or (output / "pose_design.json").exists():
        raise FileExistsError(
            "certified diagnostic output already exists; use a new output directory"
        )
    output.mkdir(parents=True, exist_ok=True)
    prepared = prepare(output, args.planning_request.resolve())
    if args.prepare_only:
        print(
            f"Prepared diagnostic proposals at {output}; CuRobo certification is still required."
        )
        return
    certify(output, prepared)


if __name__ == "__main__":
    main()
