"""Camera-frustum sampling and information-balanced calibration selection.

The geometry and D-optimal selection are a focused port of
``auto_collection_designer.py`` from commissioned source commit
97a78a5c9c48701400820922f0966cc3d3a9b7bc. CuRobo, in the planner process,
is solely responsible for IK, self-collision, and trajectory feasibility.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
from scipy.stats import qmc

from g1_aprilcube_calibration.camera_models import RectifiedCameraInfo
from g1_aprilcube_calibration.transforms import (
    invert_transform,
    pose_vector_to_transform,
    transform_points,
    transform_to_pose_vector,
    validate_transform,
)
from g1_dex3_tabletop.planning.contracts import CalibrationCandidate


@dataclass(frozen=True, slots=True)
class CandidateDesignConfig:
    target_count: int = 80
    candidate_count: int = 1600
    minimum_depth_m: float = 0.25
    maximum_depth_m: float = 0.60
    image_margin_px: float = 60.0
    minimum_projected_target_span_px: float = 45.0
    maximum_view_obliquity_deg: float = 55.0
    maximum_view_roll_deg: float = 35.0
    image_grid_columns: int = 3
    image_grid_rows: int = 3
    depth_bins: int = 3
    obliquity_bins: int = 3
    azimuth_bins: int = 6
    in_plane_rotation_bins: int = 6
    coverage_score_weight: float = 0.35
    information_translation_scale_m: float = 0.01
    information_rotation_scale_deg: float = 5.0
    information_ridge: float = 1e-6
    seed: int = 17

    def __post_init__(self) -> None:
        if self.target_count < 1 or self.candidate_count < self.target_count:
            raise ValueError("candidate count must be at least the positive target count")
        if not 0 < self.minimum_depth_m < self.maximum_depth_m:
            raise ValueError("camera depth bounds are invalid")
        positive_floats = (
            "image_margin_px",
            "minimum_projected_target_span_px",
            "maximum_view_obliquity_deg",
            "maximum_view_roll_deg",
            "information_translation_scale_m",
            "information_rotation_scale_deg",
            "information_ridge",
        )
        for name in positive_floats:
            value = float(getattr(self, name))
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be positive and finite")
        if self.maximum_view_obliquity_deg >= 90:
            raise ValueError("maximum target obliquity must be below 90 degrees")
        for name in (
            "image_grid_columns",
            "image_grid_rows",
            "depth_bins",
            "obliquity_bins",
            "azimuth_bins",
            "in_plane_rotation_bins",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if not 0 <= self.coverage_score_weight <= 1:
            raise ValueError("coverage_score_weight must lie in [0, 1]")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CandidateDesignConfig:
        expected = set(cls.__dataclass_fields__)
        if set(data) != expected:
            raise ValueError("candidate design configuration fields do not match")
        return cls(**data)


def camera_info_from_hardware(hardware: dict[str, Any]) -> RectifiedCameraInfo:
    camera = hardware["camera"]
    profile = camera["color_profile"]
    return RectifiedCameraInfo(
        width=int(profile["width"]),
        height=int(profile["height"]),
        frame_id=str(camera["calibration_frame"]),
        camera_name=str(camera["name"]),
        serial_number=str(camera["serial_number"]),
        distortion_model=str(profile["distortion_model"]),
        d=tuple(profile["d"]),
        k=tuple(profile["k"]),
        r=tuple(profile["r"]),
        p=tuple(profile["p"]),
    )


def generate_calibration_candidates(
    *,
    camera_info: RectifiedCameraInfo,
    target_config: dict[str, Any],
    torso_T_camera: np.ndarray,
    palm_T_marker: np.ndarray,
    exposed_marker_normal: np.ndarray | None = None,
    config: CandidateDesignConfig | None = None,
) -> tuple[CalibrationCandidate, ...]:
    """Generate deterministic visible marker poses without performing IK."""

    design = config or CandidateDesignConfig()
    if exposed_marker_normal is None:
        exposed_marker_normal = np.asarray([0.0, 0.0, 1.0])
    torso_T_camera = validate_transform(torso_T_camera)
    palm_T_marker = validate_transform(palm_T_marker)
    dimensions = _target_dimensions_m(target_config)
    points = _target_object_points_m(target_config)
    samples = qmc.Halton(d=6, scramble=True, seed=design.seed).random(design.candidate_count)
    candidates: list[CalibrationCandidate] = []
    for sample_index, sample in enumerate(samples):
        camera_T_marker = _sample_camera_T_target(
            sample,
            camera_info=camera_info,
            target_dimensions_m=dimensions,
            exposed_target_normal=exposed_marker_normal,
            config=design,
        )
        if camera_T_marker is None:
            continue
        torso_T_palm = validate_transform(
            torso_T_camera @ camera_T_marker @ invert_transform(palm_T_marker)
        )
        information = _fixed_marker_information_matrix(
            torso_T_palm=torso_T_palm,
            camera_info=camera_info,
            object_points_m=points,
            torso_T_camera=torso_T_camera,
            palm_T_marker=palm_T_marker,
            translation_scale_m=design.information_translation_scale_m,
            rotation_scale_rad=np.deg2rad(design.information_rotation_scale_deg),
        )
        coverage = _coverage_signature(
            camera_T_marker,
            camera_info=camera_info,
            exposed_target_normal=exposed_marker_normal,
            config=design,
        )
        candidates.append(
            CalibrationCandidate(
                candidate_id=f"candidate_{sample_index + 1:04d}",
                camera_T_marker=tuple(tuple(float(v) for v in row) for row in camera_T_marker),
                selection_metadata={
                    "sampling_index": sample_index,
                    "coverage": coverage,
                    "fixed_marker_camera_information": information.tolist(),
                },
            )
        )
    if len(candidates) < design.target_count:
        raise ValueError(
            f"camera sampling retained only {len(candidates)} candidates; "
            f"requested {design.target_count}"
        )
    return tuple(candidates)


def select_information_candidates(
    candidates: list[CalibrationCandidate],
    *,
    count: int,
    config: CandidateDesignConfig,
) -> tuple[list[CalibrationCandidate], list[dict[str, Any]]]:
    """Apply the commissioned image-balanced D-optimal selection to feasible IK poses."""

    if count < 1 or count > len(candidates):
        raise ValueError(f"cannot select {count} poses from {len(candidates)}")
    information = np.eye(6, dtype=np.float64) * config.information_ridge
    capacities: dict[tuple[int, int], int] = {}
    for candidate in candidates:
        cell = _coverage_cell(candidate)
        capacities[cell] = capacities.get(cell, 0) + 1
    quotas = _balanced_image_quotas(capacities, count)
    coverage_counts = {
        "depth": [0] * config.depth_bins,
        "obliquity": [0] * config.obliquity_bins,
        "azimuth": [0] * config.azimuth_bins,
        "in_plane_rotation": [0] * config.in_plane_rotation_bins,
    }
    remaining = list(candidates)
    selected: list[CalibrationCandidate] = []
    steps: list[dict[str, Any]] = []
    for selection_index in range(count):
        eligible = [item for item in remaining if quotas[_coverage_cell(item)] > 0]
        if not eligible:
            raise RuntimeError("image-cell quotas became infeasible")
        base_logdet = _information_logdet(information)
        information_gains = np.asarray(
            [
                _information_logdet(information + _candidate_information(item)) - base_logdet
                for item in eligible
            ],
            dtype=np.float64,
        )
        coverage_gains = np.asarray(
            [_marginal_coverage_gain(item, coverage_counts) for item in eligible],
            dtype=np.float64,
        )
        combined = (1.0 - config.coverage_score_weight) * _normalize_scores(
            information_gains
        ) + config.coverage_score_weight * _normalize_scores(coverage_gains)
        chosen_index = max(
            range(len(eligible)),
            key=lambda index: (
                float(combined[index]),
                float(information_gains[index]),
                eligible[index].candidate_id,
            ),
        )
        chosen = eligible[chosen_index]
        selected.append(chosen)
        remaining.remove(chosen)
        quotas[_coverage_cell(chosen)] -= 1
        _increment_coverage_counts(chosen, coverage_counts)
        information += _candidate_information(chosen)
        steps.append(
            {
                "selection_index": selection_index + 1,
                "candidate_id": chosen.candidate_id,
                "information_gain_logdet": float(information_gains[chosen_index]),
                "coverage_gain": float(coverage_gains[chosen_index]),
                "combined_score": float(combined[chosen_index]),
                "coverage": chosen.selection_metadata["coverage"],
            }
        )
    return selected, steps


def _candidate_information(candidate: CalibrationCandidate) -> np.ndarray:
    matrix = np.asarray(
        candidate.selection_metadata["fixed_marker_camera_information"], dtype=np.float64
    )
    if matrix.shape != (6, 6) or not np.all(np.isfinite(matrix)):
        raise ValueError(f"invalid information matrix for {candidate.candidate_id}")
    return matrix


def _coverage_cell(candidate: CalibrationCandidate) -> tuple[int, int]:
    value = candidate.selection_metadata["coverage"]["image_cell"]
    return int(value[0]), int(value[1])


def _marginal_coverage_gain(
    candidate: CalibrationCandidate, counts: dict[str, list[int]]
) -> float:
    coverage = candidate.selection_metadata["coverage"]
    indices = {
        "depth": int(coverage["depth_bin"]),
        "obliquity": int(coverage["obliquity_bin"]),
        "azimuth": int(coverage["azimuth_bin"]),
        "in_plane_rotation": int(coverage["in_plane_rotation_bin"]),
    }
    return float(sum(1.0 / (1.0 + counts[name][index]) for name, index in indices.items()))


def _increment_coverage_counts(
    candidate: CalibrationCandidate, counts: dict[str, list[int]]
) -> None:
    coverage = candidate.selection_metadata["coverage"]
    counts["depth"][int(coverage["depth_bin"])] += 1
    counts["obliquity"][int(coverage["obliquity_bin"])] += 1
    counts["azimuth"][int(coverage["azimuth_bin"])] += 1
    counts["in_plane_rotation"][int(coverage["in_plane_rotation_bin"])] += 1


def _balanced_image_quotas(
    capacities: dict[tuple[int, int], int], count: int
) -> dict[tuple[int, int], int]:
    if count < 0 or count > sum(capacities.values()):
        raise ValueError("image-cell quota count exceeds candidate capacity")
    quotas = {cell: 0 for cell in sorted(capacities)}
    for _ in range(count):
        available = [cell for cell in sorted(capacities) if quotas[cell] < capacities[cell]]
        if not available:
            raise RuntimeError("image-cell capacities were exhausted")
        quotas[min(available, key=lambda item: (quotas[item], item))] += 1
    return quotas


def _normalize_scores(values: np.ndarray) -> np.ndarray:
    scores = np.asarray(values, dtype=np.float64)
    if not np.all(np.isfinite(scores)):
        raise ValueError("selection scores must be finite")
    lower, upper = float(np.min(scores)), float(np.max(scores))
    if upper - lower <= 1e-12:
        return np.ones_like(scores)
    return (scores - lower) / (upper - lower)


def _information_logdet(information: np.ndarray) -> float:
    sign, value = np.linalg.slogdet((information + information.T) / 2.0)
    if sign <= 0 or not np.isfinite(value):
        raise ValueError("information matrix is not positive definite")
    return float(value)


def _fixed_marker_information_matrix(
    *,
    torso_T_palm: np.ndarray,
    camera_info: RectifiedCameraInfo,
    object_points_m: np.ndarray,
    torso_T_camera: np.ndarray,
    palm_T_marker: np.ndarray,
    translation_scale_m: float,
    rotation_scale_rad: float,
) -> np.ndarray:
    parameters = transform_to_pose_vector(torso_T_camera)
    scales = np.asarray([translation_scale_m] * 3 + [rotation_scale_rad] * 3)
    normalized_step = 1e-4
    jacobian = np.empty((2 * len(object_points_m), 6), dtype=np.float64)
    for index, scale in enumerate(scales):
        delta = np.zeros(6, dtype=np.float64)
        delta[index] = normalized_step * scale
        plus = _project_fixed_marker(
            parameters + delta,
            torso_T_palm=torso_T_palm,
            palm_T_marker=palm_T_marker,
            object_points_m=object_points_m,
            camera_info=camera_info,
        )
        minus = _project_fixed_marker(
            parameters - delta,
            torso_T_palm=torso_T_palm,
            palm_T_marker=palm_T_marker,
            object_points_m=object_points_m,
            camera_info=camera_info,
        )
        jacobian[:, index] = ((plus - minus) / (2.0 * normalized_step)).reshape(-1)
    result = jacobian.T @ jacobian
    return (result + result.T) / 2.0


def _project_fixed_marker(
    camera_parameters: np.ndarray,
    *,
    torso_T_palm: np.ndarray,
    palm_T_marker: np.ndarray,
    object_points_m: np.ndarray,
    camera_info: RectifiedCameraInfo,
) -> np.ndarray:
    torso_T_camera = pose_vector_to_transform(camera_parameters)
    camera_points = transform_points(
        invert_transform(torso_T_camera) @ torso_T_palm @ palm_T_marker,
        object_points_m,
    )
    if np.any(camera_points[:, 2] <= 0):
        raise ValueError("predicted calibration marker lies behind the camera")
    projected = (camera_info.rectified_camera_matrix @ camera_points.T).T
    return projected[:, :2] / projected[:, 2, None]


def _sample_camera_T_target(
    sample: np.ndarray,
    *,
    camera_info: RectifiedCameraInfo,
    target_dimensions_m: np.ndarray,
    exposed_target_normal: np.ndarray,
    config: CandidateDesignConfig,
) -> np.ndarray | None:
    sample = np.asarray(sample, dtype=np.float64).reshape(-1)
    if sample.shape != (6,) or not np.all(np.isfinite(sample)):
        raise ValueError("camera target sample must contain six finite values")
    dimensions = np.asarray(target_dimensions_m, dtype=np.float64).reshape(-1)
    margin = config.image_margin_px
    depth = config.minimum_depth_m + sample[2] * (config.maximum_depth_m - config.minimum_depth_m)
    matrix = camera_info.rectified_camera_matrix
    fx, fy, cx, cy = matrix[0, 0], matrix[1, 1], matrix[0, 2], matrix[1, 2]
    bounding_radius = float(np.linalg.norm(dimensions) / 2.0)
    nearest_depth = depth - bounding_radius
    if nearest_depth <= 0:
        return None
    half_width_px = fx * bounding_radius / nearest_depth
    half_height_px = fy * bounding_radius / nearest_depth
    lower_u, upper_u = margin + half_width_px, camera_info.width - margin - half_width_px
    lower_v, upper_v = margin + half_height_px, camera_info.height - margin - half_height_px
    if lower_u >= upper_u or lower_v >= upper_v:
        return None
    center_u = lower_u + sample[0] * (upper_u - lower_u)
    center_v = lower_v + sample[1] * (upper_v - lower_v)
    center = np.asarray([(center_u - cx) * depth / fx, (center_v - cy) * depth / fy, depth])
    exposed_normal = _unit(exposed_target_normal)
    reference = np.asarray([1.0, 0.0, 0.0])
    if abs(float(reference @ exposed_normal)) > 0.9:
        reference = np.asarray([0.0, 1.0, 0.0])
    tangent_x = _unit(np.cross(exposed_normal, reference))
    tangent_y = np.cross(exposed_normal, tangent_x)
    azimuth = 2.0 * np.pi * sample[3]
    minimum_cosine = np.cos(np.deg2rad(config.maximum_view_obliquity_deg))
    cosine = 1.0 - sample[4] * (1.0 - minimum_cosine)
    sine = np.sqrt(max(0.0, 1.0 - cosine * cosine))
    target_to_camera = cosine * exposed_normal + sine * (
        np.cos(azimuth) * tangent_x + np.sin(azimuth) * tangent_y
    )
    roll = np.deg2rad((2.0 * sample[5] - 1.0) * config.maximum_view_roll_deg)
    rotation = _view_rotation(
        target_to_camera_in_target=target_to_camera,
        target_to_camera_in_camera=-center / np.linalg.norm(center),
        roll_rad=roll,
    )
    transform = np.eye(4)
    transform[:3, :3] = rotation
    transform[:3, 3] = center
    if not _target_projects_inside(
        transform,
        camera_info=camera_info,
        target_dimensions_m=dimensions,
        margin_px=margin,
        minimum_span_px=config.minimum_projected_target_span_px,
    ):
        return None
    return validate_transform(transform)


def _view_rotation(
    *,
    target_to_camera_in_target: np.ndarray,
    target_to_camera_in_camera: np.ndarray,
    roll_rad: float,
) -> np.ndarray:
    source_z = _unit(target_to_camera_in_target)
    source_reference = np.asarray([0.0, 0.0, 1.0])
    if abs(float(source_reference @ source_z)) > 0.9:
        source_reference = np.asarray([0.0, 1.0, 0.0])
    source_x = _unit(np.cross(source_reference, source_z))
    source_y = np.cross(source_z, source_x)
    source_basis = np.column_stack((source_x, source_y, source_z))
    target_z = _unit(target_to_camera_in_camera)
    image_up = np.asarray([0.0, -1.0, 0.0])
    if abs(float(image_up @ target_z)) > 0.9:
        image_up = np.asarray([1.0, 0.0, 0.0])
    target_x = _unit(np.cross(image_up, target_z))
    target_y = np.cross(target_z, target_x)
    cosine, sine = np.cos(roll_rad), np.sin(roll_rad)
    target_basis = np.column_stack(
        (cosine * target_x + sine * target_y, -sine * target_x + cosine * target_y, target_z)
    )
    return target_basis @ source_basis.T


def _target_projects_inside(
    camera_T_target: np.ndarray,
    *,
    camera_info: RectifiedCameraInfo,
    target_dimensions_m: np.ndarray,
    margin_px: float,
    minimum_span_px: float,
) -> bool:
    half = np.asarray(target_dimensions_m, dtype=np.float64) / 2.0
    corners = np.asarray(
        [
            [x, y, z]
            for x in (-half[0], half[0])
            for y in (-half[1], half[1])
            for z in (-half[2], half[2])
        ]
    )
    camera_points = (camera_T_target[:3, :3] @ corners.T).T + camera_T_target[:3, 3]
    if np.any(camera_points[:, 2] <= 0):
        return False
    projected = (camera_info.rectified_camera_matrix @ camera_points.T).T
    pixels = projected[:, :2] / projected[:, 2, None]
    if (
        np.min(pixels[:, 0]) < margin_px
        or np.max(pixels[:, 0]) > camera_info.width - margin_px
        or np.min(pixels[:, 1]) < margin_px
        or np.max(pixels[:, 1]) > camera_info.height - margin_px
    ):
        return False
    face_corners = np.asarray(
        [
            [-half[0], half[1], 0.0],
            [half[0], half[1], 0.0],
            [half[0], -half[1], 0.0],
            [-half[0], -half[1], 0.0],
        ]
    )
    camera_face = (camera_T_target[:3, :3] @ face_corners.T).T + camera_T_target[:3, 3]
    projected_face = (camera_info.rectified_camera_matrix @ camera_face.T).T
    face_pixels = projected_face[:, :2] / projected_face[:, 2, None]
    edge_lengths = np.linalg.norm(face_pixels - np.roll(face_pixels, -1, axis=0), axis=1)
    return float(np.min(edge_lengths)) >= minimum_span_px


def _coverage_signature(
    camera_T_target: np.ndarray,
    *,
    camera_info: RectifiedCameraInfo,
    exposed_target_normal: np.ndarray,
    config: CandidateDesignConfig,
) -> dict[str, Any]:
    transform = validate_transform(camera_T_target)
    center = transform[:3, 3]
    matrix = camera_info.rectified_camera_matrix
    center_pixel = matrix @ center
    center_pixel = center_pixel[:2] / center_pixel[2]
    normalized = (
        float(np.clip(center_pixel[0] / camera_info.width, 0.0, 1.0)),
        float(np.clip(center_pixel[1] / camera_info.height, 0.0, 1.0)),
    )
    image_cell = (
        min(int(normalized[0] * config.image_grid_columns), config.image_grid_columns - 1),
        min(int(normalized[1] * config.image_grid_rows), config.image_grid_rows - 1),
    )
    normal = _unit(exposed_target_normal)
    target_to_camera = _unit(transform[:3, :3].T @ -center)
    cosine = float(np.clip(target_to_camera @ normal, -1.0, 1.0))
    obliquity_deg = float(np.degrees(np.arccos(cosine)))
    reference = np.asarray([1.0, 0.0, 0.0])
    if abs(float(reference @ normal)) > 0.9:
        reference = np.asarray([0.0, 1.0, 0.0])
    tangent_x = _unit(np.cross(normal, reference))
    tangent_y = np.cross(normal, tangent_x)
    tangent = target_to_camera - cosine * normal
    azimuth_deg = (
        0.0
        if np.linalg.norm(tangent) <= 1e-12
        else float(
            np.degrees(
                np.arctan2(float(_unit(tangent) @ tangent_y), float(_unit(tangent) @ tangent_x))
            )
            % 360.0
        )
    )
    endpoint = center + 0.01 * transform[:3, 0]
    endpoint_pixel = matrix @ endpoint
    endpoint_pixel = endpoint_pixel[:2] / endpoint_pixel[2]
    image_axis = endpoint_pixel - center_pixel
    in_plane_deg = float(np.degrees(np.arctan2(image_axis[1], image_axis[0])) % 360.0)
    return {
        "image_cell": list(image_cell),
        "depth_bin": _bounded_bin(
            float(center[2]),
            design_range=(config.minimum_depth_m, config.maximum_depth_m),
            count=config.depth_bins,
        ),
        "obliquity_bin": _bounded_bin(
            obliquity_deg,
            design_range=(0.0, config.maximum_view_obliquity_deg),
            count=config.obliquity_bins,
        ),
        "azimuth_bin": _cyclic_bin(azimuth_deg, config.azimuth_bins),
        "in_plane_rotation_bin": _cyclic_bin(in_plane_deg, config.in_plane_rotation_bins),
        "normalized_centroid": list(normalized),
        "depth_m": float(center[2]),
        "obliquity_deg": obliquity_deg,
        "azimuth_deg": azimuth_deg,
        "in_plane_rotation_deg": in_plane_deg,
    }


def _bounded_bin(value: float, *, design_range: tuple[float, float], count: int) -> int:
    lower, upper = design_range
    fraction = float(np.clip((value - lower) / (upper - lower), 0.0, 1.0))
    return min(int(fraction * count), count - 1)


def _cyclic_bin(angle_deg: float, count: int) -> int:
    return min(int((angle_deg % 360.0) / 360.0 * count), count - 1)


def _target_dimensions_m(target: dict[str, Any]) -> np.ndarray:
    dimensions = np.asarray(target["box_dims"], dtype=np.float64).reshape(-1)
    if dimensions.shape != (3,) or np.any(dimensions <= 0):
        raise ValueError("target box_dims must contain three positive millimetre values")
    return dimensions / 1000.0


def _target_object_points_m(target: dict[str, Any]) -> np.ndarray:
    markers = target.get("markers")
    if not isinstance(markers, list) or not markers:
        raise ValueError("target markers must be a non-empty list")
    points = [np.asarray(marker["corners_mm"], dtype=np.float64) for marker in markers]
    if any(item.shape != (4, 3) or not np.all(np.isfinite(item)) for item in points):
        raise ValueError("every marker must contain four finite 3D corners")
    return np.vstack(points) / 1000.0


def _unit(vector: np.ndarray) -> np.ndarray:
    array = np.asarray(vector, dtype=np.float64)
    norm = float(np.linalg.norm(array))
    if norm <= 0 or not np.isfinite(norm):
        raise ValueError("cannot normalize a zero or non-finite vector")
    return array / norm
