"""Rigid-transform helpers with one documented six-parameter convention."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from scipy.spatial.transform import Rotation


def pose_vector_to_transform(values: Sequence[float] | np.ndarray) -> np.ndarray:
    """Convert ``[tx, ty, tz, rx, ry, rz]`` to a parent-from-child transform."""
    vector = np.asarray(values, dtype=np.float64).reshape(-1)
    if vector.shape != (6,) or not np.all(np.isfinite(vector)):
        raise ValueError("pose vector must contain six finite values")
    result = np.eye(4)
    result[:3, :3] = Rotation.from_rotvec(vector[3:]).as_matrix()
    result[:3, 3] = vector[:3]
    return result


def transform_to_pose_vector(transform: np.ndarray) -> np.ndarray:
    matrix = validate_transform(transform)
    result = np.concatenate(
        (matrix[:3, 3], Rotation.from_matrix(matrix[:3, :3].copy()).as_rotvec())
    )
    result.setflags(write=False)
    return result


def validate_transform(transform: np.ndarray) -> np.ndarray:
    matrix = np.asarray(transform, dtype=np.float64).copy()
    if matrix.shape != (4, 4) or not np.all(np.isfinite(matrix)):
        raise ValueError("rigid transform must be a finite 4x4 matrix")
    if not np.allclose(matrix[3], [0, 0, 0, 1], atol=1e-12):
        raise ValueError("rigid transform has an invalid homogeneous row")
    if not np.allclose(matrix[:3, :3].T @ matrix[:3, :3], np.eye(3), atol=1e-8):
        raise ValueError("rigid transform rotation is not orthonormal")
    if np.linalg.det(matrix[:3, :3]) < 0.999999:
        raise ValueError("rigid transform rotation must be right-handed")
    matrix.setflags(write=False)
    return matrix


def invert_transform(transform: np.ndarray) -> np.ndarray:
    matrix = validate_transform(transform)
    result = np.eye(4)
    result[:3, :3] = matrix[:3, :3].T
    result[:3, 3] = -result[:3, :3] @ matrix[:3, 3]
    return result


def transform_points(transform: np.ndarray, points: np.ndarray) -> np.ndarray:
    matrix = validate_transform(transform)
    array = np.asarray(points, dtype=np.float64)
    if array.ndim != 2 or array.shape[1] != 3 or not np.all(np.isfinite(array)):
        raise ValueError("points must have finite shape (N, 3)")
    return (matrix[:3, :3] @ array.T).T + matrix[:3, 3]
