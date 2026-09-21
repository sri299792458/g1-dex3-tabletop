"""Matched hand poses and repeated visits for the calibration investigation."""

from __future__ import annotations

import heapq

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from g1_aprilcube_calibration.joint_map import arm_hand_link, arm_joint_names
from g1_aprilcube_calibration.pose_schema import HANDOFF_POSE_ID
from g1_dex3_tabletop.calibration.design import BilateralCaptureWaypoint


def matched_hand_configurations(urdf, *, arm, command_q, joint_offsets):
    """Solve six pose constraints while stepping one joint; return exact FK matches.

    These are geometric proposals. The caller must check sightlines, collisions,
    and route connectivity before using them in an executable artifact.
    """

    names = arm_joint_names(arm)
    initial = np.asarray(command_q, dtype=float)
    offsets = np.array([joint_offsets.get(name, 0.0) for name in names])
    limits = urdf.joint_limits(names)
    lower = np.array([limit.lower for limit in limits]) - offsets + 1e-5
    upper = np.array([limit.upper for limit in limits]) - offsets - 1e-5

    def fk(q, link=None):
        return urdf.transform(
            "torso_link", link or arm_hand_link(arm), dict(zip(names, q + offsets, strict=True))
        )

    target = fk(initial)
    elbow = fk(initial, f"{arm}_elbow_link")[:3, 3]
    proposals = []
    for fixed in (0, 1, 2, 4, 5):
        free = [index for index in range(7) if index != fixed]
        for step in (-0.5, -0.35, -0.2, -0.1, 0.1, 0.2, 0.35, 0.5):
            fixed_value = initial[fixed] + step
            if not lower[fixed] < fixed_value < upper[fixed]:
                continue

            def unpack(values, free=free, fixed=fixed, fixed_value=fixed_value):
                result = initial.copy()
                result[free] = values
                result[fixed] = fixed_value
                return result

            def residual(values, unpack=unpack):
                actual = fk(unpack(values))
                return np.r_[
                    (actual[:3, 3] - target[:3, 3]) / 0.3,
                    Rotation.from_matrix(target[:3, :3].T @ actual[:3, :3]).as_rotvec(),
                ]

            fit = least_squares(
                residual,
                np.clip(initial[free], lower[free], upper[free]),
                bounds=(lower[free], upper[free]),
                max_nfev=150,
                ftol=1e-11,
                xtol=1e-11,
                gtol=1e-11,
            )
            q = unpack(fit.x)
            pose_error = residual(fit.x)
            translation_error = float(np.linalg.norm(pose_error[:3]) * 0.3)
            rotation_error = float(np.linalg.norm(pose_error[3:]))
            if translation_error > 1e-6 or rotation_error > 1e-5:
                continue
            if any(np.linalg.norm(q - item["command_q_rad"]) < 0.02 for item in proposals):
                continue
            proposals.append(
                {
                    "command_q_rad": q.tolist(),
                    "fixed_joint": names[fixed],
                    "fixed_joint_change_rad": step,
                    "translation_error_m": translation_error,
                    "rotation_error_rad": rotation_error,
                    "elbow_displacement_m": float(
                        np.linalg.norm(fk(q, f"{arm}_elbow_link")[:3, 3] - elbow)
                    ),
                    "maximum_joint_change_rad": float(np.max(np.abs(q - initial))),
                }
            )
    return sorted(
        proposals,
        key=lambda item: (-item["elbow_displacement_m"], item["maximum_joint_change_rad"]),
    )


def shortest_diagnostic_path(edges, source, target):
    graph = {}
    for distance, first, second in edges:
        graph.setdefault(first, []).append((float(distance), second))
        graph.setdefault(second, []).append((float(distance), first))
    queue = [(0.0, source, (source,))]
    visited = set()
    while queue:
        cost, node, path = heapq.heappop(queue)
        if node in visited:
            continue
        if node == target:
            return path
        visited.add(node)
        for distance, neighbor in graph.get(node, ()):
            if neighbor not in visited:
                heapq.heappush(queue, (cost + distance, neighbor, (*path, neighbor)))
    raise ValueError(f"no certified diagnostic path from {source} to {target}")


def build_diagnostic_schedule(families, edges_by_arm, *, blocks=3):
    """Two opposed final approaches per base pose, with interleaved FK matches.

    Repeated base/alternative configurations keep the same candidate IDs. Each
    independent visit gets its own occurrence and retains its block/approach.
    """

    if blocks < 3 or len(families) < 2:
        raise ValueError("diagnostic needs at least three blocks and two pose families")
    anchor = "bilateral_anchor"
    schedule = [BilateralCaptureWaypoint(HANDOFF_POSE_ID, anchor, "anchor", None)]
    visits = {}

    def move(target, side, *, capture=False, metadata=None, direct=False):
        source = schedule[-1].candidate_id
        if direct:
            if not any({source, target} == {a, b} for _, a, b in edges_by_arm[side]):
                raise ValueError("the final diagnostic approach must be a certified direct edge")
            path = (source, target)
        else:
            path = shortest_diagnostic_path(edges_by_arm[side], source, target)
        if len(path) < 2:
            raise ValueError("diagnostic revisits must move away before returning")
        for index, node in enumerate(path[1:], start=1):
            final = index == len(path) - 1
            role = "excitation" if final and capture else "preparation"
            if final and target == anchor:
                role = "anchor"
            occurrence = f"diagnostic_{len(schedule):03d}"
            schedule.append(
                BilateralCaptureWaypoint(
                    occurrence, node, role, None if role == "anchor" else side
                )
            )
            if final and capture:
                visits[occurrence] = dict(metadata)

    for block in range(blocks):
        ordered = families if block % 2 == 0 else tuple(reversed(families))
        for family in ordered:
            side, base = family["arm"], family["base_id"]
            alternate = family.get("alternate_id")
            approaches = (
                family["approach_ids"]
                if block % 2 == 0
                else tuple(reversed(family["approach_ids"]))
            )

            def visit_alternate(alternate=alternate, side=side, block=block, base=base):
                move(
                    alternate,
                    side,
                    capture=True,
                    metadata={"block": block, "family": base, "kind": "matched_hand_alternative"},
                )

            # The middle block adds B-A-B before its second A: reverse-order
            # evidence without sacrificing either of the two base approaches.
            if alternate and block % 2:
                visit_alternate()
            for approach_index, approach in enumerate(approaches):
                move(approach, side)
                move(
                    base,
                    side,
                    capture=True,
                    direct=True,
                    metadata={
                        "block": block,
                        "family": base,
                        "kind": "return_to_base",
                        "approach_id": approach,
                    },
                )
                if alternate and approach_index == 0:
                    visit_alternate()
            move(anchor, side)
    schedule[-1] = BilateralCaptureWaypoint(HANDOFF_POSE_ID, anchor, "anchor", None)
    return tuple(schedule), visits
