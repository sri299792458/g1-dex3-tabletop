"""Receipt-time pairing between unsynchronized camera and LowState streams."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from g1_aprilcube_calibration.models import RobotStateSample, validate_utc_iso


@dataclass(frozen=True, slots=True)
class ImageTiming:
    receipt_monotonic_s: float
    receipt_utc: str
    header_stamp_ns: int | None = None

    def __post_init__(self) -> None:
        if not np.isfinite(self.receipt_monotonic_s) or self.receipt_monotonic_s < 0:
            raise ValueError("image receipt time must be finite and non-negative")
        validate_utc_iso(self.receipt_utc)
        if self.header_stamp_ns is not None and self.header_stamp_ns < 0:
            raise ValueError("image header stamp must be non-negative")


@dataclass(frozen=True, slots=True)
class PairingConfig:
    maximum_nearest_delta_s: float = 0.05
    maximum_bracket_span_s: float = 0.1

    def __post_init__(self) -> None:
        if self.maximum_nearest_delta_s <= 0:
            raise ValueError("maximum_nearest_delta_s must be positive")
        if self.maximum_bracket_span_s <= 0:
            raise ValueError("maximum_bracket_span_s must be positive")


@dataclass(frozen=True, slots=True)
class PairingResult:
    image: ImageTiming
    before: RobotStateSample
    after: RobotStateSample
    nearest: RobotStateSample
    nearest_delta_s: float
    bracket_span_s: float

    def to_dict(self) -> dict:
        return {
            "image_receipt_monotonic_s": self.image.receipt_monotonic_s,
            "image_receipt_utc": self.image.receipt_utc,
            "image_header_stamp_ns": self.image.header_stamp_ns,
            "before_state_monotonic_s": self.before.receipt_monotonic_s,
            "after_state_monotonic_s": self.after.receipt_monotonic_s,
            "nearest_state_monotonic_s": self.nearest.receipt_monotonic_s,
            "nearest_delta_s": self.nearest_delta_s,
            "bracket_span_s": self.bracket_span_s,
        }


def pair_state_to_image(
    image: ImageTiming,
    samples: Sequence[RobotStateSample],
    *,
    config: PairingConfig,
) -> PairingResult:
    if not samples:
        raise ValueError("cannot pair image without robot-state samples")
    ordered = tuple(samples)
    timestamps = np.asarray([sample.receipt_monotonic_s for sample in ordered], dtype=np.float64)
    if np.any(np.diff(timestamps) <= 0):
        raise ValueError("robot-state receipt times must be strictly increasing")

    before_indices = np.flatnonzero(timestamps <= image.receipt_monotonic_s)
    after_indices = np.flatnonzero(timestamps >= image.receipt_monotonic_s)
    if before_indices.size == 0 or after_indices.size == 0:
        raise ValueError("image is not bracketed by robot-state samples")
    before = ordered[int(before_indices[-1])]
    after = ordered[int(after_indices[0])]
    bracket_span = after.receipt_monotonic_s - before.receipt_monotonic_s
    if bracket_span > config.maximum_bracket_span_s:
        raise ValueError(
            f"state bracket spans {bracket_span:.3f}s; "
            f"limit is {config.maximum_bracket_span_s:.3f}s"
        )

    nearest = min(
        (before, after),
        key=lambda sample: (
            abs(sample.receipt_monotonic_s - image.receipt_monotonic_s),
            sample.receipt_monotonic_s,
        ),
    )
    nearest_delta = abs(nearest.receipt_monotonic_s - image.receipt_monotonic_s)
    if nearest_delta > config.maximum_nearest_delta_s:
        raise ValueError(
            f"nearest robot state is {nearest_delta:.3f}s from image; "
            f"limit is {config.maximum_nearest_delta_s:.3f}s"
        )
    return PairingResult(
        image=image,
        before=before,
        after=after,
        nearest=nearest,
        nearest_delta_s=nearest_delta,
        bracket_span_s=bracket_span,
    )
