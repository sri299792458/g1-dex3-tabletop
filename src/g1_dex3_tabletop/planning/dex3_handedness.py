"""Exact mapping from GraspGenX's canonical Dex3 grasp to either physical hand."""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from g1_aprilcube_calibration.joint_map import validate_arm_side
from g1_aprilcube_calibration.transports.unitree_dex3 import (
    DEX3_MOTOR_JOINT_SUFFIXES,
)

# GraspGenX generated and PhysX-qualified the retained atlas with its right
# Dex3 descriptor. Both descriptors express an isometric hand in the same G
# frame, but the mirrored left descriptor reverses every joint coordinate and
# exchanges the canonical index/middle chains. Keep that conversion in one
# place; object_T_G itself is identical for both sides.
CANONICAL_DEX3_JOINT_SUFFIXES = DEX3_MOTOR_JOINT_SUFFIXES["right"]
CANONICAL_DEX3_PROFILE = (
    Path(__file__).resolve().parents[3] / "config/tabletop/dex3_rev1_canonical_profile.json"
)
_LEFT_SUFFIX_FROM_CANONICAL = {
    "thumb_0": "thumb_0",
    "thumb_1": "thumb_1",
    "thumb_2": "thumb_2",
    "middle_0": "index_0",
    "middle_1": "index_1",
    "index_0": "middle_0",
    "index_1": "middle_1",
}


def dex3_q_from_canonical(values: Sequence[float], *, arm: str) -> tuple[float, ...]:
    """Return one canonical GraspGenX posture in Unitree motor-ID order."""

    selected = validate_arm_side(arm)
    source = np.asarray(values, dtype=np.float64).reshape(-1)
    if source.shape != (7,) or not np.all(np.isfinite(source)):
        raise ValueError("canonical Dex3 posture must contain seven finite values")
    canonical = dict(zip(CANONICAL_DEX3_JOINT_SUFFIXES, source, strict=True))
    if selected == "right":
        return tuple(float(canonical[suffix]) for suffix in DEX3_MOTOR_JOINT_SUFFIXES[selected])
    physical = {
        physical_suffix: -float(canonical[canonical_suffix])
        for canonical_suffix, physical_suffix in _LEFT_SUFFIX_FROM_CANONICAL.items()
    }
    return tuple(float(physical[suffix]) for suffix in DEX3_MOTOR_JOINT_SUFFIXES[selected])


def dex3_execution_profile(arm: str) -> tuple[tuple[float, ...], tuple[float, ...]]:
    """Return the descriptor-defined open and fixed close targets for one hand."""

    document = json.loads(CANONICAL_DEX3_PROFILE.read_text(encoding="utf-8"))
    if tuple(document.get("canonical_joint_order", ())) != CANONICAL_DEX3_JOINT_SUFFIXES:
        raise ValueError("canonical Dex3 profile joint order is invalid")
    if document.get("close_command_policy") != (
        "fixed_descriptor_target_with_contact_limited_physical_motion"
    ):
        raise ValueError("canonical Dex3 close-command policy is invalid")
    open_q = dex3_q_from_canonical(document["open_q_rad"], arm=arm)
    close_q = dex3_q_from_canonical(document["close_q_rad"], arm=arm)
    if open_q == close_q:
        raise ValueError("canonical Dex3 open and close targets must differ")
    return open_q, close_q


def dex3_empty_close_reference(arm: str) -> tuple[tuple[float, ...], float]:
    """Return the commissioned empty close and opposed-obstruction margin."""

    selected = validate_arm_side(arm)
    document = json.loads(CANONICAL_DEX3_PROFILE.read_text(encoding="utf-8"))
    commissioning = document.get("empty_close_commissioning")
    if not isinstance(commissioning, dict):
        raise RuntimeError(  # noqa: TRY004 - this is missing commissioned data
            "Dex3 empty-close commissioning is missing"
        )
    entry = commissioning.get(selected)
    if not isinstance(entry, dict):
        raise RuntimeError(  # noqa: TRY004 - this is missing commissioned data
            f"Dex3 empty-close reference is not commissioned for {selected}; "
            f"run measure-dex3-empty-close --arm {selected} and commission its "
            "empty_close.measured_q_rad first"
        )
    measured = np.asarray(entry.get("measured_q_rad"), dtype=np.float64).reshape(-1)
    if measured.shape != (7,) or not np.all(np.isfinite(measured)):
        raise ValueError(f"Dex3 {selected} empty-close reference is invalid")
    threshold = float(commissioning.get("minimum_opposed_shortfall_rad", 0.0))
    if not np.isfinite(threshold) or threshold <= 0.0:
        raise ValueError("Dex3 opposed-obstruction threshold must be positive")
    return tuple(float(value) for value in measured), threshold
