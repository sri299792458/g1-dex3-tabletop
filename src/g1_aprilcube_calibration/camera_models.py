"""Immutable rectified-camera metadata used by raw capture and the solver."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

import numpy as np


def _finite_tuple(values, *, size: int, name: str) -> tuple[float, ...]:
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    if array.shape != (size,) or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain exactly {size} finite values")
    return tuple(float(value) for value in array)


@dataclass(frozen=True, slots=True)
class RectifiedCameraInfo:
    width: int
    height: int
    frame_id: str
    camera_name: str
    serial_number: str
    distortion_model: str
    d: tuple[float, ...]
    k: tuple[float, ...]
    r: tuple[float, ...]
    p: tuple[float, ...]

    def __post_init__(self) -> None:
        if self.width <= 0 or self.height <= 0:
            raise ValueError("camera dimensions must be positive")
        for name in ("frame_id", "camera_name", "serial_number", "distortion_model"):
            if not getattr(self, name):
                raise ValueError(f"{name} must be non-empty")
        d = np.asarray(self.d, dtype=np.float64).reshape(-1)
        if not np.all(np.isfinite(d)):
            raise ValueError("camera distortion coefficients must be finite")
        object.__setattr__(self, "d", tuple(float(value) for value in d))
        object.__setattr__(self, "k", _finite_tuple(self.k, size=9, name="camera K"))
        object.__setattr__(self, "r", _finite_tuple(self.r, size=9, name="camera R"))
        object.__setattr__(self, "p", _finite_tuple(self.p, size=12, name="camera P"))
        if self.k[0] <= 0 or self.k[4] <= 0:
            raise ValueError("camera focal lengths must be positive")
        if self.p[0] <= 0 or self.p[5] <= 0:
            raise ValueError("rectified projection focal lengths must be positive")
        if any(abs(value) > 1e-12 for value in self.d):
            raise ValueError("rectified camera info must have zero distortion coefficients")

    @property
    def rectified_camera_matrix(self) -> np.ndarray:
        """Return the 3x3 matrix that projects the rectified image pixels."""
        matrix = np.asarray(self.p, dtype=np.float64).reshape(3, 4)[:, :3].copy()
        matrix.setflags(write=False)
        return matrix

    @property
    def profile_sha256(self) -> str:
        payload = json.dumps(
            self.to_dict(), sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
        return hashlib.sha256(payload).hexdigest()

    def to_dict(self) -> dict:
        return {
            "width": self.width,
            "height": self.height,
            "frame_id": self.frame_id,
            "camera_name": self.camera_name,
            "serial_number": self.serial_number,
            "distortion_model": self.distortion_model,
            "d": list(self.d),
            "k": list(self.k),
            "r": list(self.r),
            "p": list(self.p),
        }

    @classmethod
    def from_dict(cls, data: dict) -> RectifiedCameraInfo:
        return cls(
            width=int(data["width"]),
            height=int(data["height"]),
            frame_id=str(data["frame_id"]),
            camera_name=str(data["camera_name"]),
            serial_number=str(data["serial_number"]),
            distortion_model=str(data["distortion_model"]),
            d=tuple(data["d"]),
            k=tuple(data["k"]),
            r=tuple(data["r"]),
            p=tuple(data["p"]),
        )
