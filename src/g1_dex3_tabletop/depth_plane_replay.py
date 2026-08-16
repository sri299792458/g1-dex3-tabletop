"""Compare recorded D435i depth planes with the fixed ChArUco board.

This is an offline measurement tool.  It consumes an existing continuous RGB
replay report and the matching MCAP; it never creates a ROS publisher.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from g1_aprilcube_calibration.ros.camera_adapter import camera_info_from_ros
from g1_aprilcube_calibration.transforms import invert_transform, validate_transform
from g1_dex3_tabletop.state_estimation_replay import (
    _atomic_write_json,
    _clock_model,
    _message_stamp_ns,
    _sha256,
)

BOARD_SIZE_X_M = 0.180
BOARD_SIZE_Y_M = 0.270


@dataclass(frozen=True, slots=True)
class DepthPlaneMeasurement:
    sequence: int
    header_stamp_ns: int
    record_time_ns: int
    nearest_rgb_header_gap_ms: float
    interpolation_bracket_ms: float
    selected_point_count: int
    inlier_point_count: int
    normal_angle_error_deg: float
    signed_plane_offset_mm: float
    residual_rms_mm: float

    def to_dict(self) -> dict[str, int | float]:
        return {
            "sequence": self.sequence,
            "header_stamp_ns": self.header_stamp_ns,
            "record_time_ns": self.record_time_ns,
            "nearest_rgb_header_gap_ms": self.nearest_rgb_header_gap_ms,
            "interpolation_bracket_ms": self.interpolation_bracket_ms,
            "selected_point_count": self.selected_point_count,
            "inlier_point_count": self.inlier_point_count,
            "normal_angle_error_deg": self.normal_angle_error_deg,
            "signed_plane_offset_mm": self.signed_plane_offset_mm,
            "residual_rms_mm": self.residual_rms_mm,
        }


def _transform_from_ros(transform: Any) -> np.ndarray:
    from scipy.spatial.transform import Rotation

    translation = transform.translation
    quaternion = transform.rotation
    result = np.eye(4, dtype=np.float64)
    result[:3, :3] = Rotation.from_quat(
        [quaternion.x, quaternion.y, quaternion.z, quaternion.w]
    ).as_matrix()
    result[:3, 3] = [translation.x, translation.y, translation.z]
    return validate_transform(result)


def _lookup_transform(transforms: list[Any], source_frame: str, target_frame: str) -> np.ndarray:
    """Return source_T_target from a connected ROS static-TF tree."""

    graph: dict[str, list[tuple[str, np.ndarray]]] = {}
    for stamped in transforms:
        parent = str(stamped.header.frame_id)
        child = str(stamped.child_frame_id)
        parent_T_child = _transform_from_ros(stamped.transform)
        graph.setdefault(parent, []).append((child, parent_T_child))
        graph.setdefault(child, []).append((parent, invert_transform(parent_T_child)))
    queue: deque[tuple[str, np.ndarray]] = deque([(source_frame, np.eye(4))])
    visited = {source_frame}
    while queue:
        frame, source_T_frame = queue.popleft()
        if frame == target_frame:
            return validate_transform(source_T_frame)
        for neighbor, frame_T_neighbor in graph.get(frame, []):
            if neighbor in visited:
                continue
            visited.add(neighbor)
            queue.append((neighbor, source_T_frame @ frame_T_neighbor))
    raise ValueError(f"no static-TF path from {source_frame} to {target_frame}")


def _depth_m_from_ros(message: Any) -> np.ndarray:
    width = int(message.width)
    height = int(message.height)
    step = int(message.step)
    if str(message.encoding).upper() != "16UC1":
        raise ValueError(f"unsupported depth encoding {message.encoding!r}; expected 16UC1")
    if width <= 0 or height <= 0 or step < 2 * width or step % 2:
        raise ValueError("invalid 16UC1 depth image dimensions or row stride")
    dtype = np.dtype("<u2" if int(message.is_bigendian) == 0 else ">u2")
    raw = np.frombuffer(message.data, dtype=dtype)
    row_elements = step // 2
    required = height * row_elements
    if raw.size < required:
        raise ValueError("ROS depth image data is truncated")
    return raw[:required].reshape(height, row_elements)[:, :width].astype(np.float64) * 0.001


def _fit_board_plane(
    depth_m: np.ndarray,
    camera_matrix: np.ndarray,
    board_T_depth: np.ndarray,
    *,
    pixel_stride: int = 2,
) -> dict[str, float | int]:
    """Fit depth points selected by the visually observed board footprint."""

    image = np.asarray(depth_m, dtype=np.float64)
    matrix = np.asarray(camera_matrix, dtype=np.float64)
    if image.ndim != 2 or matrix.shape != (3, 3):
        raise ValueError("depth plane fit requires a 2D depth image and 3x3 intrinsics")
    if pixel_stride < 1:
        raise ValueError("depth pixel stride must be positive")
    rows = np.arange(0, image.shape[0], pixel_stride)
    columns = np.arange(0, image.shape[1], pixel_stride)
    vv, uu = np.meshgrid(rows, columns, indexing="ij")
    z = image[vv, uu]
    valid = np.isfinite(z) & (z >= 0.15) & (z <= 2.0)
    z = z[valid]
    u = uu[valid]
    v = vv[valid]
    fx, fy = matrix[0, 0], matrix[1, 1]
    cx, cy = matrix[0, 2], matrix[1, 2]
    depth_points = np.column_stack(((u - cx) * z / fx, (v - cy) * z / fy, z))
    board_points = (board_T_depth[:3, :3] @ depth_points.T + board_T_depth[:3, 3, None]).T
    footprint = (
        (board_points[:, 0] >= 0.0)
        & (board_points[:, 0] <= BOARD_SIZE_X_M)
        & (board_points[:, 1] >= 0.0)
        & (board_points[:, 1] <= BOARD_SIZE_Y_M)
        & (np.abs(board_points[:, 2]) <= 0.030)
    )
    selected = board_points[footprint]
    if len(selected) < 100:
        raise ValueError(f"only {len(selected)} depth points lie on the board footprint")
    median_z = float(np.median(selected[:, 2]))
    absolute_deviation = np.abs(selected[:, 2] - median_z)
    mad = float(np.median(absolute_deviation))
    half_band_m = max(0.003, 4.0 * 1.4826 * mad)
    inliers = selected[absolute_deviation <= half_band_m]
    if len(inliers) < 100:
        raise ValueError(f"only {len(inliers)} robust depth-plane inliers remain")
    centroid = np.mean(inliers, axis=0)
    _u, _singular, vh = np.linalg.svd(inliers - centroid, full_matrices=False)
    normal = vh[-1]
    if normal[2] < 0.0:
        normal = -normal
    normal /= np.linalg.norm(normal)
    residuals = (inliers - centroid) @ normal
    board_center = np.asarray([0.5 * BOARD_SIZE_X_M, 0.5 * BOARD_SIZE_Y_M, 0.0])
    cosine = float(np.clip(normal[2], -1.0, 1.0))
    return {
        "selected_point_count": len(selected),
        "inlier_point_count": len(inliers),
        "normal_angle_error_deg": float(np.degrees(np.arccos(cosine))),
        "signed_plane_offset_mm": 1000.0 * float(np.dot(normal, centroid - board_center)),
        "residual_rms_mm": 1000.0 * float(np.sqrt(np.mean(np.square(residuals)))),
    }


def _summary(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "standard_deviation": float(np.std(array)),
        "p95_absolute": float(np.percentile(np.abs(array), 95)),
        "maximum_absolute": float(np.max(np.abs(array))),
    }


def _continuous_trajectory(path: Path) -> tuple[np.ndarray, list[np.ndarray]]:
    document = json.loads(path.read_text())
    if document.get("kind") != "g1_continuous_camera_state_estimation_research_report":
        raise ValueError("input is not a continuous state-estimation report")
    trajectory = document["evaluation"]["trajectory"]
    if len(trajectory) < 2:
        raise ValueError("continuous report contains fewer than two board poses")
    times = np.asarray([int(item["header_stamp_ns"]) for item in trajectory], dtype=np.int64)
    poses = [validate_transform(np.asarray(item["board_T_camera"])) for item in trajectory]
    return times, poses


def _interpolate_transform(
    sorted_times_ns: np.ndarray,
    transforms: list[np.ndarray],
    timestamp_ns: int,
) -> tuple[np.ndarray, float, float]:
    from scipy.spatial.transform import Rotation, Slerp

    if timestamp_ns < int(sorted_times_ns[0]) or timestamp_ns > int(sorted_times_ns[-1]):
        raise ValueError("depth header timestamp is outside the RGB trajectory")
    upper = int(np.searchsorted(sorted_times_ns, timestamp_ns, side="left"))
    if upper == 0 or int(sorted_times_ns[upper]) == timestamp_ns:
        return transforms[upper].copy(), 0.0, 0.0
    lower = upper - 1
    first_time = int(sorted_times_ns[lower])
    second_time = int(sorted_times_ns[upper])
    alpha = (timestamp_ns - first_time) / (second_time - first_time)
    result = np.eye(4)
    result[:3, 3] = (1.0 - alpha) * transforms[lower][:3, 3] + alpha * transforms[upper][:3, 3]
    rotations = Rotation.from_matrix(
        np.stack([transforms[lower][:3, :3], transforms[upper][:3, :3]])
    )
    result[:3, :3] = Slerp([0.0, 1.0], rotations)([alpha]).as_matrix()[0]
    nearest_gap_ms = min(timestamp_ns - first_time, second_time - timestamp_ns) / 1.0e6
    bracket_ms = (second_time - first_time) / 1.0e6
    return validate_transform(result), nearest_gap_ms, bracket_ms


def analyze_depth_plane(
    run_directory: Path,
    *,
    continuous_report_path: Path,
    output_path: Path,
    frame_stride: int = 10,
    maximum_pairing_gap_ms: float = 100.0,
) -> dict[str, Any]:
    if frame_stride < 1 or maximum_pairing_gap_ms <= 0:
        raise ValueError("frame stride and maximum pairing gap must be positive")
    try:
        import rosbag2_py
        from rclpy.serialization import deserialize_message
        from sensor_msgs.msg import CameraInfo, Image
        from tf2_msgs.msg import TFMessage
    except ImportError as error:
        raise RuntimeError("ROS 2 depth MCAP readers are unavailable") from error
    run_directory = run_directory.resolve()
    bag_directory = run_directory / "raw_episode/bag"
    if not bag_directory.is_dir():
        raise FileNotFoundError(f"raw MCAP is unavailable: {bag_directory}")
    rgb_header_times, board_T_color_values = _continuous_trajectory(continuous_report_path)

    metadata_reader = rosbag2_py.SequentialReader()
    metadata_reader.open(
        rosbag2_py.StorageOptions(uri=str(bag_directory), storage_id="mcap"),
        rosbag2_py.ConverterOptions("", ""),
    )
    metadata_reader.set_filter(
        rosbag2_py.StorageFilter(topics=["/camera/depth/camera_info", "/tf_static"])
    )
    depth_info = None
    static_transforms: list[Any] = []
    while metadata_reader.has_next():
        topic, raw, _record_time_ns = metadata_reader.read_next()
        if topic == "/camera/depth/camera_info" and depth_info is None:
            depth_info = deserialize_message(raw, CameraInfo)
        elif topic == "/tf_static":
            static_transforms.extend(deserialize_message(raw, TFMessage).transforms)
    if depth_info is None or not static_transforms:
        raise ValueError("depth CameraInfo or recorded static TF is unavailable")
    profile = camera_info_from_ros(
        depth_info, camera_name="recorded_g1_depth", serial_number="recorded"
    )
    depth_T_color = _lookup_transform(
        static_transforms,
        profile.frame_id,
        "camera_color_optical_frame",
    )
    color_T_depth = invert_transform(depth_T_color)

    reader = rosbag2_py.SequentialReader()
    reader.open(
        rosbag2_py.StorageOptions(uri=str(bag_directory), storage_id="mcap"),
        rosbag2_py.ConverterOptions("", ""),
    )
    reader.set_filter(rosbag2_py.StorageFilter(topics=["/camera/depth/image_rect_raw"]))
    seen = 0
    sampled = 0
    clock_pairs: list[tuple[int, int]] = []
    measurements: list[DepthPlaneMeasurement] = []
    rejections: Counter[str] = Counter()
    while reader.has_next():
        _topic, raw, record_time_ns = reader.read_next()
        sequence = seen
        seen += 1
        if sequence % frame_stride:
            continue
        sampled += 1
        message = deserialize_message(raw, Image)
        header_stamp_ns = _message_stamp_ns(message)
        clock_pairs.append((header_stamp_ns, record_time_ns))
        try:
            board_T_color, pairing_gap_ms, interpolation_bracket_ms = _interpolate_transform(
                rgb_header_times,
                board_T_color_values,
                header_stamp_ns,
            )
        except ValueError as error:
            rejections[str(error)] += 1
            continue
        if pairing_gap_ms > maximum_pairing_gap_ms:
            rejections["rgb_pairing_gap"] += 1
            continue
        board_T_depth = board_T_color @ color_T_depth
        try:
            fit = _fit_board_plane(
                _depth_m_from_ros(message),
                profile.rectified_camera_matrix,
                board_T_depth,
            )
        except ValueError as error:
            rejections[str(error)] += 1
            continue
        measurements.append(
            DepthPlaneMeasurement(
                sequence=sequence,
                header_stamp_ns=header_stamp_ns,
                record_time_ns=record_time_ns,
                nearest_rgb_header_gap_ms=pairing_gap_ms,
                interpolation_bracket_ms=interpolation_bracket_ms,
                selected_point_count=int(fit["selected_point_count"]),
                inlier_point_count=int(fit["inlier_point_count"]),
                normal_angle_error_deg=float(fit["normal_angle_error_deg"]),
                signed_plane_offset_mm=float(fit["signed_plane_offset_mm"]),
                residual_rms_mm=float(fit["residual_rms_mm"]),
            )
        )
    if len(measurements) < 2:
        raise ValueError("depth replay produced fewer than two board-plane fits")
    report = {
        "schema_version": 1,
        "kind": "g1_depth_plane_research_report",
        "commands_robot": False,
        "run_directory": str(run_directory),
        "continuous_report": str(continuous_report_path.resolve()),
        "continuous_report_sha256": _sha256(continuous_report_path),
        "depth_profile": profile.to_dict(),
        "depth_T_color_optical": depth_T_color.tolist(),
        "frame_stride": frame_stride,
        "depth_message_count": seen,
        "sampled_frame_count": sampled,
        "accepted_frame_count": len(measurements),
        "rejection_count_by_reason": dict(sorted(rejections.items())),
        "depth_header_to_mcap_clock": _clock_model(clock_pairs),
        "summary": {
            "nearest_rgb_header_gap_ms": _summary(
                [measurement.nearest_rgb_header_gap_ms for measurement in measurements]
            ),
            "rgb_interpolation_bracket_ms": _summary(
                [measurement.interpolation_bracket_ms for measurement in measurements]
            ),
            "signed_plane_offset_mm": _summary(
                [measurement.signed_plane_offset_mm for measurement in measurements]
            ),
            "normal_angle_error_deg": _summary(
                [measurement.normal_angle_error_deg for measurement in measurements]
            ),
            "plane_fit_residual_rms_mm": _summary(
                [measurement.residual_rms_mm for measurement in measurements]
            ),
        },
        "measurements": [measurement.to_dict() for measurement in measurements],
        "interpretation_limits": [
            "The ChArUco board supplies the expected plane and the depth ROI.",
            "RGB and depth are interpolated directly in their shared D435i hardware-header clock.",
            "Depth constrains plane normal and translation normal to the plane, not in-plane X/Y or yaw.",
            "RealSense depth bias and RGB-depth extrinsic error are included in this disagreement.",
        ],
    }
    _atomic_write_json(output_path, report)
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="g1-depth-plane-research")
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--continuous-report", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--frame-stride", default=10, type=int)
    parser.add_argument("--maximum-pairing-gap-ms", default=100.0, type=float)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        report = analyze_depth_plane(
            args.run,
            continuous_report_path=args.continuous_report,
            output_path=args.output,
            frame_stride=args.frame_stride,
            maximum_pairing_gap_ms=args.maximum_pairing_gap_ms,
        )
    except (FileNotFoundError, RuntimeError, TypeError, ValueError) as error:
        print(f"error: {error}")
        return 1
    print(
        json.dumps(
            {
                "commands_robot": False,
                "output": str(args.output.resolve()),
                "sampled_frames": report["sampled_frame_count"],
                "accepted_frames": report["accepted_frame_count"],
                "summary": report["summary"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
