"""Measured-state freshness and stationary-window gates."""

from __future__ import annotations

import threading
from collections import deque
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import numpy as np

from g1_aprilcube_calibration.joint_map import opposite_arm, validate_arm_side
from g1_aprilcube_calibration.models import RobotStateSample


@dataclass(frozen=True, slots=True)
class RecordingGateConfig:
    calibration_arm: str = "left"
    state_freshness_timeout_s: float = 0.1
    stationary_duration_s: float = 0.5
    maximum_state_gap_s: float = 0.1
    maximum_calibration_position_spread_rad: float = 0.01
    maximum_hold_position_spread_rad: float = 0.01
    minimum_samples: int = 5

    def __post_init__(self) -> None:
        object.__setattr__(self, "calibration_arm", validate_arm_side(self.calibration_arm))
        for name in (
            "state_freshness_timeout_s",
            "stationary_duration_s",
            "maximum_state_gap_s",
            "maximum_calibration_position_spread_rad",
            "maximum_hold_position_spread_rad",
        ):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.minimum_samples < 2:
            raise ValueError("minimum_samples must be at least two")


@dataclass(frozen=True, slots=True)
class ReadinessReport:
    ready: bool
    hard_failures: tuple[str, ...]
    sample_count: int
    duration_s: float
    maximum_state_gap_s: float | None
    calibration_arm: str
    maximum_calibration_velocity_rad_s: float | None
    maximum_calibration_position_spread_rad: float | None
    maximum_hold_velocity_rad_s: float | None
    maximum_hold_position_spread_rad: float | None

    def to_dict(self) -> dict:
        return {
            "ready": self.ready,
            "hard_failures": list(self.hard_failures),
            "sample_count": self.sample_count,
            "duration_s": self.duration_s,
            "maximum_state_gap_s": self.maximum_state_gap_s,
            "calibration_arm": self.calibration_arm,
            "maximum_calibration_velocity_rad_s": (self.maximum_calibration_velocity_rad_s),
            "maximum_calibration_position_spread_rad": (
                self.maximum_calibration_position_spread_rad
            ),
            "maximum_hold_velocity_rad_s": self.maximum_hold_velocity_rad_s,
            "maximum_hold_position_spread_rad": self.maximum_hold_position_spread_rad,
        }


class StateSampleBuffer:
    """A bounded monotonic receipt-time buffer for complete robot states."""

    def __init__(self, *, maximum_samples: int = 4096) -> None:
        if maximum_samples < 2:
            raise ValueError("maximum_samples must be at least two")
        self._samples: deque[RobotStateSample] = deque(maxlen=maximum_samples)
        self._lock = threading.Lock()

    def add(self, sample: RobotStateSample) -> None:
        with self._lock:
            if (
                self._samples
                and sample.receipt_monotonic_s <= self._samples[-1].receipt_monotonic_s
            ):
                raise ValueError("state receipt times must be strictly increasing")
            self._samples.append(sample)

    def extend(self, samples: Iterable[RobotStateSample]) -> None:
        for sample in samples:
            self.add(sample)

    def snapshot(self) -> tuple[RobotStateSample, ...]:
        with self._lock:
            return tuple(self._samples)

    @property
    def latest(self) -> RobotStateSample | None:
        with self._lock:
            return None if not self._samples else self._samples[-1]

    def centered_window(
        self,
        *,
        center_monotonic_s: float,
        duration_s: float,
    ) -> tuple[RobotStateSample, ...]:
        """Return samples spanning a centered interval, including edge brackets."""
        if duration_s <= 0:
            raise ValueError("duration_s must be positive")
        half_duration = duration_s / 2.0
        start = center_monotonic_s - half_duration
        end = center_monotonic_s + half_duration
        samples = self.snapshot()
        before = [sample for sample in samples if sample.receipt_monotonic_s <= start]
        after = [sample for sample in samples if sample.receipt_monotonic_s >= end]
        if not before or not after:
            return ()
        first = before[-1]
        last = after[0]
        return tuple(
            sample
            for sample in samples
            if first.receipt_monotonic_s <= sample.receipt_monotonic_s <= last.receipt_monotonic_s
        )


def evaluate_recording_window(
    samples: Sequence[RobotStateSample],
    *,
    now_monotonic_s: float,
    config: RecordingGateConfig,
) -> ReadinessReport:
    failures: list[str] = []
    if not np.isfinite(now_monotonic_s) or now_monotonic_s < 0:
        failures.append("decision time must be finite and non-negative")
    ordered = tuple(samples)
    if not ordered:
        failures.append("no robot-state samples in recording window")
        return ReadinessReport(
            False,
            tuple(failures),
            0,
            0.0,
            None,
            config.calibration_arm,
            None,
            None,
            None,
            None,
        )

    timestamps = np.asarray([sample.receipt_monotonic_s for sample in ordered], dtype=np.float64)
    if np.any(np.diff(timestamps) <= 0):
        failures.append("robot-state receipt times are not strictly increasing")
    if now_monotonic_s < timestamps[-1]:
        failures.append("decision time precedes the latest robot state")
    elif now_monotonic_s - timestamps[-1] > config.state_freshness_timeout_s:
        failures.append("latest robot state is stale")
    if any(not sample.is_mode5 for sample in ordered):
        failures.append("recording window contains a state outside mode_machine=5")
    if len(ordered) < config.minimum_samples:
        failures.append(
            f"need at least {config.minimum_samples} robot-state samples; received {len(ordered)}"
        )

    duration = float(timestamps[-1] - timestamps[0])
    if duration + 1e-12 < config.stationary_duration_s:
        failures.append(
            f"stationary window is {duration:.3f}s; need {config.stationary_duration_s:.3f}s"
        )
    gaps = np.diff(timestamps)
    maximum_gap = None if gaps.size == 0 else float(np.max(gaps))
    if maximum_gap is not None and maximum_gap > config.maximum_state_gap_s + 1e-12:
        failures.append(
            f"robot-state gap is {maximum_gap:.3f}s; limit is {config.maximum_state_gap_s:.3f}s"
        )

    calibration_velocity = np.vstack([sample.arm_dq(config.calibration_arm) for sample in ordered])
    maximum_velocity = float(np.max(np.abs(calibration_velocity)))

    calibration_position = np.vstack([sample.arm_q(config.calibration_arm) for sample in ordered])
    spread = np.ptp(calibration_position, axis=0)
    maximum_spread = float(np.max(spread))
    if maximum_spread > config.maximum_calibration_position_spread_rad:
        failures.append(
            f"{config.calibration_arm}-arm position spread is "
            f"{maximum_spread:.4f}rad; limit is "
            f"{config.maximum_calibration_position_spread_rad:.4f}rad"
        )
    hold_arm = opposite_arm(config.calibration_arm)
    hold_velocity = np.vstack([sample.arm_dq(hold_arm) for sample in ordered])
    maximum_hold_velocity = float(np.max(np.abs(hold_velocity)))
    hold_position = np.vstack([sample.arm_q(hold_arm) for sample in ordered])
    maximum_hold_spread = float(np.max(np.ptp(hold_position, axis=0)))
    if maximum_hold_spread > config.maximum_hold_position_spread_rad:
        failures.append(
            f"held {hold_arm}-arm position spread is {maximum_hold_spread:.4f}rad; "
            f"limit is {config.maximum_hold_position_spread_rad:.4f}rad"
        )
    return ReadinessReport(
        ready=not failures,
        hard_failures=tuple(failures),
        sample_count=len(ordered),
        duration_s=duration,
        maximum_state_gap_s=maximum_gap,
        calibration_arm=config.calibration_arm,
        maximum_calibration_velocity_rad_s=maximum_velocity,
        maximum_calibration_position_spread_rad=maximum_spread,
        maximum_hold_velocity_rad_s=maximum_hold_velocity,
        maximum_hold_position_spread_rad=maximum_hold_spread,
    )
