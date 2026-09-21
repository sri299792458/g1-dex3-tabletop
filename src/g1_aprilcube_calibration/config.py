"""Validated configuration models for the prototype."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True, slots=True)
class QualityThresholds:
    minimum_decoded_tags: int
    minimum_tag_short_side_px: float
    minimum_corner_image_margin_px: float
    reject_duplicate_tag_ids: bool
    preferred_visible_faces: int
    preferred_tag_short_side_px: float
    preferred_corner_image_margin_px: float
    pnp_warning_reprojection_px: float
    pnp_reject_reprojection_px: float
    image_grid_columns: int
    image_grid_rows: int
    depth_bins: int
    novelty_translation_m: float
    novelty_rotation_deg: float
    stationary_burst_frames: int
    stationary_burst_maximum_duration_s: float

    def __post_init__(self) -> None:
        positive_values = {
            "minimum_decoded_tags": self.minimum_decoded_tags,
            "minimum_tag_short_side_px": self.minimum_tag_short_side_px,
            "preferred_tag_short_side_px": self.preferred_tag_short_side_px,
            "preferred_visible_faces": self.preferred_visible_faces,
            "image_grid_columns": self.image_grid_columns,
            "image_grid_rows": self.image_grid_rows,
            "depth_bins": self.depth_bins,
            "novelty_translation_m": self.novelty_translation_m,
            "novelty_rotation_deg": self.novelty_rotation_deg,
            "stationary_burst_frames": self.stationary_burst_frames,
            "stationary_burst_maximum_duration_s": (self.stationary_burst_maximum_duration_s),
        }
        for name, value in positive_values.items():
            if value <= 0:
                raise ValueError(f"{name} must be positive, got {value}")
        if self.minimum_corner_image_margin_px < 0:
            raise ValueError("minimum_corner_image_margin_px must be non-negative")
        if self.preferred_corner_image_margin_px < self.minimum_corner_image_margin_px:
            raise ValueError("preferred image margin must be at least the hard minimum")
        if self.preferred_tag_short_side_px < self.minimum_tag_short_side_px:
            raise ValueError("preferred tag size must be at least the hard minimum")
        if self.pnp_warning_reprojection_px <= 0:
            raise ValueError("PnP warning threshold must be positive")
        if self.pnp_reject_reprojection_px < self.pnp_warning_reprojection_px:
            raise ValueError("PnP reject threshold must be at least the warning threshold")

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> QualityThresholds:
        hard = data["hard_gates"]
        preferred = data["preferred_quality"]
        coverage = data["coverage"]
        burst = data["stationary_burst"]
        return cls(
            minimum_decoded_tags=int(hard["minimum_decoded_tags"]),
            minimum_tag_short_side_px=float(hard["minimum_tag_short_side_px"]),
            minimum_corner_image_margin_px=float(hard["minimum_corner_image_margin_px"]),
            reject_duplicate_tag_ids=bool(hard["reject_duplicate_tag_ids"]),
            preferred_visible_faces=int(preferred["minimum_visible_faces"]),
            preferred_tag_short_side_px=float(preferred["minimum_tag_short_side_px"]),
            preferred_corner_image_margin_px=float(preferred["minimum_corner_image_margin_px"]),
            pnp_warning_reprojection_px=float(preferred["multiface_pnp_warning_reprojection_px"]),
            pnp_reject_reprojection_px=float(preferred["multiface_pnp_reject_reprojection_px"]),
            image_grid_columns=int(coverage["image_grid_columns"]),
            image_grid_rows=int(coverage["image_grid_rows"]),
            depth_bins=int(coverage["depth_bins"]),
            novelty_translation_m=float(coverage["novelty_translation_m"]),
            novelty_rotation_deg=float(coverage["novelty_rotation_deg"]),
            stationary_burst_frames=int(burst["frame_count"]),
            stationary_burst_maximum_duration_s=float(burst["maximum_duration_s"]),
        )

    @classmethod
    def from_yaml(cls, path: str | Path) -> QualityThresholds:
        with Path(path).open(encoding="utf-8") as stream:
            data = yaml.safe_load(stream)
        if not isinstance(data, Mapping):
            raise TypeError(f"quality config must contain a mapping: {path}")
        return cls.from_mapping(data)
