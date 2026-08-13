"""Immutable core records shared by adapters, recorder, and executor."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

import numpy as np

from g1_aprilcube_calibration.joint_map import (
    G1_MODE_MACHINE,
    extract_arm,
    extract_left_arm,
    extract_right_arm,
    validate_full_joint_vector,
)


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def validate_utc_iso(value: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError("UTC timestamp must be a non-empty string")
    # Python 3.10 does not accept the ISO 8601 ``Z`` UTC designator here;
    # normalize it to the equivalent explicit offset before parsing.
    parsed = datetime.fromisoformat(f"{value[:-1]}+00:00" if value.endswith("Z") else value)
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError("UTC timestamp must include a timezone")
    if parsed.utcoffset() != timedelta(0):
        raise ValueError("UTC timestamp must have a zero UTC offset")
    return value


@dataclass(frozen=True, slots=True)
class RobotStateSample:
    """One complete receipt-stamped mode-5 LowState observation."""

    receipt_monotonic_s: float
    receipt_utc: str
    mode_machine: int
    position: np.ndarray
    velocity: np.ndarray
    estimated_torque: np.ndarray
    source_sequence: int | None = None

    def __post_init__(self) -> None:
        if not np.isfinite(self.receipt_monotonic_s) or self.receipt_monotonic_s < 0:
            raise ValueError("receipt_monotonic_s must be finite and non-negative")
        validate_utc_iso(self.receipt_utc)
        if isinstance(self.mode_machine, bool) or not isinstance(self.mode_machine, int):
            raise TypeError("mode_machine must be an integer")
        position = validate_full_joint_vector(self.position, name="joint position")
        velocity = validate_full_joint_vector(self.velocity, name="joint velocity")
        estimated_torque = validate_full_joint_vector(
            self.estimated_torque, name="estimated joint torque"
        )
        object.__setattr__(self, "position", position)
        object.__setattr__(self, "velocity", velocity)
        object.__setattr__(self, "estimated_torque", estimated_torque)
        if self.source_sequence is not None and self.source_sequence < 0:
            raise ValueError("source_sequence must be non-negative")

    @property
    def is_mode5(self) -> bool:
        return self.mode_machine == G1_MODE_MACHINE

    @property
    def left_q(self) -> np.ndarray:
        return extract_left_arm(self.position)

    @property
    def right_q(self) -> np.ndarray:
        return extract_right_arm(self.position)

    @property
    def left_dq(self) -> np.ndarray:
        return extract_left_arm(self.velocity)

    @property
    def right_dq(self) -> np.ndarray:
        return extract_right_arm(self.velocity)

    @property
    def left_tau_est(self) -> np.ndarray:
        return extract_left_arm(self.estimated_torque)

    @property
    def right_tau_est(self) -> np.ndarray:
        return extract_right_arm(self.estimated_torque)

    def arm_q(self, side: str) -> np.ndarray:
        return extract_arm(self.position, side=side)

    def arm_dq(self, side: str) -> np.ndarray:
        return extract_arm(self.velocity, side=side)

    def arm_tau_est(self, side: str) -> np.ndarray:
        return extract_arm(self.estimated_torque, side=side)

    def age_s(self, now_monotonic_s: float) -> float:
        if now_monotonic_s < self.receipt_monotonic_s:
            raise ValueError("now_monotonic_s precedes the state receipt time")
        return now_monotonic_s - self.receipt_monotonic_s

    def to_dict(self) -> dict:
        return {
            "receipt_monotonic_s": self.receipt_monotonic_s,
            "receipt_utc": self.receipt_utc,
            "mode_machine": self.mode_machine,
            "position": self.position.tolist(),
            "velocity": self.velocity.tolist(),
            "estimated_torque": self.estimated_torque.tolist(),
            "source_sequence": self.source_sequence,
        }

    @classmethod
    def from_dict(cls, data: dict) -> RobotStateSample:
        return cls(
            receipt_monotonic_s=float(data["receipt_monotonic_s"]),
            receipt_utc=str(data["receipt_utc"]),
            mode_machine=int(data["mode_machine"]),
            position=np.asarray(data["position"], dtype=np.float64),
            velocity=np.asarray(data["velocity"], dtype=np.float64),
            estimated_torque=np.asarray(data["estimated_torque"], dtype=np.float64),
            source_sequence=(
                None if data.get("source_sequence") is None else int(data["source_sequence"])
            ),
        )
