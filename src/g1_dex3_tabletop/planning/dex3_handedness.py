"""Exact mapping from GraspGenX's canonical Dex3 grasp to either physical hand."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

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


def dex3_q_from_qualified_right_mapping(
    values: Mapping[str, float],
    *,
    arm: str,
) -> tuple[float, ...]:
    """Adapt one retained right-descriptor PhysX posture to either hand."""

    canonical = []
    for suffix in CANONICAL_DEX3_JOINT_SUFFIXES:
        name = f"right_hand_{suffix}_joint"
        if name not in values:
            raise ValueError(f"qualified grasp posture is missing {name}")
        canonical.append(float(values[name]))
    return dex3_q_from_canonical(canonical, arm=arm)
