from collections import Counter
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from g1_aprilcube_calibration.joint_map import arm_hand_link, arm_joint_names
from g1_aprilcube_calibration.pose_schema import HANDOFF_POSE_ID
from g1_aprilcube_calibration.urdf_model import URDFModel
from g1_dex3_tabletop.calibration.diagnostic import (
    build_diagnostic_schedule,
    matched_hand_configurations,
    shortest_diagnostic_path,
)


def test_matched_hand_proposals_preserve_full_pose_with_joint_offsets():
    model = URDFModel(
        Path(__file__).resolve().parents[1] / "config/urdf/g1_29dof_rev_1_0_g1pilot_collision.urdf"
    )
    q = np.array([0.25, 0.8, 0.25, 1.0, -0.2, 0.2, 0.1])
    offsets = {"left_shoulder_roll_joint": 0.03, "left_elbow_joint": -0.02}
    names = arm_joint_names("left")

    def hand_pose(command):
        positions = {
            name: value + offsets.get(name, 0.0)
            for name, value in zip(names, command, strict=True)
        }
        return model.transform("torso_link", arm_hand_link("left"), positions)

    target = hand_pose(q)
    proposals = matched_hand_configurations(model, arm="left", command_q=q, joint_offsets=offsets)
    assert any(item["elbow_displacement_m"] > 0.02 for item in proposals)
    for item in proposals:
        actual = hand_pose(item["command_q_rad"])
        assert np.linalg.norm(actual[:3, 3] - target[:3, 3]) <= 1e-6
        assert Rotation.from_matrix(target[:3, :3].T @ actual[:3, :3]).magnitude() <= 1e-5
        for name, value, limit in zip(
            names, item["command_q_rad"], model.joint_limits(names), strict=True
        ):
            assert limit.lower <= value + offsets.get(name, 0.0) <= limit.upper


def schedule_fixture():
    families = []
    edges = {"left": [], "right": []}
    for side, region in (("left", "high"), ("right", "high"), ("left", "low"), ("right", "low")):
        base = f"{side}_{region}"
        approaches = (base + "_minus", base + "_plus")
        family = {"arm": side, "base_id": base, "approach_ids": approaches}
        edges[side].extend((1.0, base, helper) for helper in approaches)
        edges[side].append((2.0, "bilateral_anchor", base))
        if base != "left_high":
            family["alternate_id"] = base + "_matched"
            edges[side].append((1.0, base, family["alternate_id"]))
        families.append(family)
    return tuple(families), edges


def test_diagnostic_repeats_preserve_groups_and_opposed_final_approaches():
    families, edges = schedule_fixture()
    schedule, visits = build_diagnostic_schedule(families, edges)
    assert schedule[0].occurrence_id == schedule[-1].occurrence_id == HANDOFF_POSE_ID
    assert len({item.occurrence_id for item in schedule[1:-1]}) == len(schedule) - 2
    assert sum(item.capture_role == "anchor" for item in schedule) == 13
    assert sum(item.capture_role == "excitation" for item in schedule) == 36
    assert all(item.hand_action is None for item in schedule)
    assert Counter(v["family"] for v in visits.values() if v["kind"] == "return_to_base") == {
        f["base_id"]: 6 for f in families
    }
    for index, item in enumerate(schedule):
        if item.occurrence_id not in visits:
            continue
        visit = visits[item.occurrence_id]
        if visit["kind"] == "return_to_base":
            assert item.candidate_id == visit["family"]
            assert schedule[index - 1].candidate_id == visit["approach_id"]
        assert item.active_arm in ("left", "right")
    for family in families:
        base_visits = [
            v
            for v in visits.values()
            if v["family"] == family["base_id"] and v["kind"] == "return_to_base"
        ]
        assert Counter(v["approach_id"] for v in base_visits) == {
            key: 3 for key in family["approach_ids"]
        }
        if "alternate_id" in family:
            middle = [
                v["kind"]
                for v in visits.values()
                if v["family"] == family["base_id"] and v["block"] == 1
            ]
            assert middle == [
                "matched_hand_alternative",
                "return_to_base",
                "matched_hand_alternative",
                "return_to_base",
            ]


def test_diagnostic_rejects_uncertified_final_approach_even_with_indirect_path():
    families, edges = schedule_fixture()
    edges["left"].remove((1.0, "left_high", "left_high_minus"))
    edges["left"].append((1.0, "bilateral_anchor", "left_high_minus"))
    with pytest.raises(ValueError, match="final diagnostic approach"):
        build_diagnostic_schedule(families, edges)


def test_diagnostic_disconnected_configuration_cannot_be_scheduled():
    with pytest.raises(ValueError, match="no certified diagnostic path"):
        shortest_diagnostic_path([(1.0, "anchor", "other")], "anchor", "target")
