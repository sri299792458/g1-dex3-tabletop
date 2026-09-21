from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from g1_dex3_tabletop.depth_plane_replay import (
    _depth_m_from_ros,
    _fit_board_plane,
    _interpolate_transform,
    _lookup_transform,
)


def test_depth_decoder_honors_padded_row_stride() -> None:
    rows = np.asarray([[500, 600, 999], [700, 800, 999]], dtype="<u2")
    message = SimpleNamespace(
        width=2,
        height=2,
        step=6,
        encoding="16UC1",
        is_bigendian=0,
        data=rows.tobytes(),
    )

    result = _depth_m_from_ros(message)

    np.testing.assert_allclose(result, [[0.5, 0.6], [0.7, 0.8]])


def test_depth_plane_fit_recovers_exact_frontoparallel_board() -> None:
    depth = np.full((48, 64), 0.6)
    intrinsics = np.asarray([[100.0, 0.0, 31.5], [0.0, 100.0, 23.5], [0.0, 0.0, 1.0]])
    board_T_depth = np.eye(4)
    board_T_depth[:3, 3] = [0.09, 0.135, -0.6]

    result = _fit_board_plane(depth, intrinsics, board_T_depth)

    assert result["selected_point_count"] >= 100
    assert result["normal_angle_error_deg"] == pytest.approx(0.0)
    assert result["signed_plane_offset_mm"] == pytest.approx(0.0, abs=1.0e-9)
    assert result["residual_rms_mm"] == pytest.approx(0.0, abs=1.0e-9)


def _tf(parent: str, child: str, xyz: tuple[float, float, float]):
    return SimpleNamespace(
        header=SimpleNamespace(frame_id=parent),
        child_frame_id=child,
        transform=SimpleNamespace(
            translation=SimpleNamespace(x=xyz[0], y=xyz[1], z=xyz[2]),
            rotation=SimpleNamespace(x=0.0, y=0.0, z=0.0, w=1.0),
        ),
    )


def test_static_tf_lookup_composes_and_inverts_edges() -> None:
    transforms = [_tf("root", "middle", (1.0, 0.0, 0.0)), _tf("middle", "tip", (0.0, 2.0, 0.0))]

    root_T_tip = _lookup_transform(transforms, "root", "tip")
    tip_T_root = _lookup_transform(transforms, "tip", "root")

    assert root_T_tip[:3, 3] == pytest.approx([1.0, 2.0, 0.0])
    assert tip_T_root @ root_T_tip == pytest.approx(np.eye(4))


def test_transform_interpolation_uses_header_clock_and_slerp() -> None:
    from scipy.spatial.transform import Rotation

    first = np.eye(4)
    second = np.eye(4)
    second[:3, 3] = [0.2, 0.0, 0.0]
    second[:3, :3] = Rotation.from_euler("z", 90.0, degrees=True).as_matrix()

    result, nearest_gap_ms, bracket_ms = _interpolate_transform(
        np.asarray([1_000_000, 5_000_000]), [first, second], 2_000_000
    )

    assert result[:3, 3] == pytest.approx([0.05, 0.0, 0.0])
    assert Rotation.from_matrix(result[:3, :3].copy()).magnitude() == pytest.approx(np.pi / 8)
    assert nearest_gap_ms == pytest.approx(1.0)
    assert bracket_ms == pytest.approx(4.0)
