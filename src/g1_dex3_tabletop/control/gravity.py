"""Load the exact commissioned Unitree dual-Dex3 gravity model."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import yaml

from g1_aprilcube_calibration.gravity_compensation import (
    G1PinocchioGravityFeedforward,
)

ROOT = Path(__file__).resolve().parents[3]
COMMISSIONED_GRAVITY_URDF_SHA256 = (
    "97da67732d067c3147fc5fb7b7bafc8982718f4e7f8c92ff82266a4d9c07200d"
)


def prepare_gravity_feedforward(
    hardware_config: str | Path,
    reference_full_q: np.ndarray,
) -> G1PinocchioGravityFeedforward:
    path = Path(hardware_config)
    with path.open(encoding="utf-8") as stream:
        hardware = yaml.safe_load(stream)
    control = hardware["control"]
    relative_urdf = Path(control["gravity_model_urdf"])
    urdf = relative_urdf if relative_urdf.is_absolute() else ROOT / relative_urdf
    locked = control.get("gravity_locked_joint_positions_rad")
    if not isinstance(locked, dict) or len(locked) != 14:
        raise ValueError("gravity model requires all fourteen explicit Dex3 joints")
    gravity = G1PinocchioGravityFeedforward(
        urdf,
        locked_joint_positions_rad=locked,
    )
    if gravity.urdf_sha256 != COMMISSIONED_GRAVITY_URDF_SHA256:
        raise ValueError(
            "gravity URDF differs from the physically commissioned dual-Dex3 model: "
            f"expected={COMMISSIONED_GRAVITY_URDF_SHA256}, "
            f"actual={gravity.urdf_sha256}"
        )
    gravity.seed_reference(reference_full_q)
    return gravity
