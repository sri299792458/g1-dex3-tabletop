"""Diagnostics comparing operator-supported and controller-held observations."""

from __future__ import annotations

import numpy as np

from g1_aprilcube_calibration.session_store import CaptureFrameInput


def supported_vs_held_metrics(
    supported: CaptureFrameInput,
    held: CaptureFrameInput,
    *,
    calibration_arm: str,
) -> dict:
    """Return finite JSON-ready measurements for one before/after observation pair."""

    if supported.camera_info.profile_sha256 != held.camera_info.profile_sha256:
        raise ValueError("paired observations use different camera profiles")

    supported_by_tag = {
        observation.tag_id: observation for observation in supported.correspondences.observations
    }
    held_by_tag = {
        observation.tag_id: observation for observation in held.correspondences.observations
    }
    common_tag_ids = tuple(sorted(set(supported_by_tag) & set(held_by_tag)))
    corner_displacements = np.asarray(
        [
            np.linalg.norm(
                held_by_tag[tag_id].image_corners_px[corner_index]
                - supported_by_tag[tag_id].image_corners_px[corner_index]
            )
            for tag_id in common_tag_ids
            for corner_index in range(4)
        ],
        dtype=np.float64,
    )

    supported_state = supported.pairing.nearest
    held_state = held.pairing.nearest
    position_delta = held_state.arm_q(calibration_arm) - supported_state.arm_q(calibration_arm)
    torque_delta = held_state.arm_tau_est(calibration_arm) - supported_state.arm_tau_est(
        calibration_arm
    )

    corner_metrics = None
    if corner_displacements.size:
        corner_metrics = {
            "median": float(np.median(corner_displacements)),
            "rms": float(np.sqrt(np.mean(np.square(corner_displacements)))),
            "maximum": float(np.max(corner_displacements)),
        }

    return {
        "supported_frame_id": supported.frame_id,
        "held_frame_id": held.frame_id,
        "observation_interval_s": float(
            held.image_timing.receipt_monotonic_s - supported.image_timing.receipt_monotonic_s
        ),
        "common_tag_ids": list(common_tag_ids),
        "common_corner_count": int(corner_displacements.size),
        "corner_displacement_px": corner_metrics,
        "arm_position_delta_rad": position_delta.tolist(),
        "arm_position_delta_l2_rad": float(np.linalg.norm(position_delta)),
        "arm_position_delta_max_abs_rad": float(np.max(np.abs(position_delta))),
        "estimated_torque_delta": torque_delta.tolist(),
        "estimated_torque_delta_l2": float(np.linalg.norm(torque_delta)),
    }
