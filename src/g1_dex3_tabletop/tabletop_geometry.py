"""Shared rigid geometry for face-resting tabletop cubes."""

from __future__ import annotations

from itertools import permutations, product

import numpy as np


def canonical_resting_cube_pose(frame_T_detected_object: np.ndarray) -> np.ndarray:
    """Map the uppermost physical cube face to canonical object +Z."""

    detected = np.asarray(frame_T_detected_object, dtype=np.float64)
    if detected.shape != (4, 4) or not np.all(np.isfinite(detected)):
        raise ValueError("detected cube pose must be one finite 4x4 transform")
    rotations = []
    for permutation in permutations(range(3)):
        for signs in product((-1.0, 1.0), repeat=3):
            symmetry = np.zeros((3, 3), dtype=np.float64)
            symmetry[list(permutation), range(3)] = signs
            if np.linalg.det(symmetry) > 0.0:
                rotations.append(symmetry)
    candidates = [
        symmetry
        for symmetry in rotations
        if float((detected[:3, :3] @ symmetry)[2, 2]) >= np.cos(np.deg2rad(20.0))
    ]
    if not candidates:
        raise RuntimeError(
            "AprilCube is not resting on a face: no face normal points upward within 20 degrees"
        )
    symmetry = max(candidates, key=lambda value: (float(np.trace(value)), *value.ravel()))
    canonical = detected.copy()
    canonical[:3, :3] = detected[:3, :3] @ symmetry
    return canonical
