"""Rectified ROS Image/CameraInfo conversion and bounded live buffering."""

from __future__ import annotations

import threading
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np

from g1_aprilcube_calibration.camera_models import RectifiedCameraInfo
from g1_aprilcube_calibration.clock import MonotonicClock, SystemClock
from g1_aprilcube_calibration.models import utc_now_iso
from g1_aprilcube_calibration.timestamp_pairing import ImageTiming


def ros_stamp_to_ns(stamp: Any) -> int:
    seconds = int(stamp.sec)
    nanoseconds = int(stamp.nanosec)
    if seconds < 0 or not 0 <= nanoseconds < 1_000_000_000:
        raise ValueError("ROS stamp fields are out of range")
    return seconds * 1_000_000_000 + nanoseconds


def camera_info_from_ros(
    message: Any,
    *,
    camera_name: str,
    serial_number: str,
) -> RectifiedCameraInfo:
    """Convert CameraInfo and fail if it does not describe a rectified stream."""
    return RectifiedCameraInfo(
        width=int(message.width),
        height=int(message.height),
        frame_id=str(message.header.frame_id),
        camera_name=camera_name,
        serial_number=serial_number,
        distortion_model=str(message.distortion_model),
        d=tuple(message.d),
        k=tuple(message.k),
        r=tuple(message.r),
        p=tuple(message.p),
    )


def image_bgr_from_ros(message: Any) -> np.ndarray:
    """Decode supported 8-bit ROS Image encodings while honoring row stride."""
    width = int(message.width)
    height = int(message.height)
    step = int(message.step)
    encoding = str(message.encoding).lower()
    if width <= 0 or height <= 0 or step <= 0:
        raise ValueError("ROS image dimensions and step must be positive")
    channels = {"bgr8": 3, "rgb8": 3, "mono8": 1}.get(encoding)
    if channels is None:
        raise ValueError(
            f"unsupported ROS image encoding {message.encoding!r}; expected bgr8, rgb8, or mono8"
        )
    minimum_step = width * channels
    if step < minimum_step:
        raise ValueError("ROS image step is smaller than its encoded row")
    raw = np.frombuffer(message.data, dtype=np.uint8)
    required = height * step
    if raw.size < required:
        raise ValueError("ROS image data is truncated")
    rows = raw[:required].reshape(height, step)[:, :minimum_step]
    image = rows.reshape(height, width, channels)
    if encoding == "bgr8":
        return image.copy()
    if encoding == "rgb8":
        return image[:, :, ::-1].copy()
    return np.repeat(image, 3, axis=2)


@dataclass(frozen=True, slots=True)
class ROSImageFrame:
    image_bgr: np.ndarray
    timing: ImageTiming
    camera_info: RectifiedCameraInfo

    def __post_init__(self) -> None:
        image = np.asarray(self.image_bgr, dtype=np.uint8).copy()
        expected = (self.camera_info.height, self.camera_info.width, 3)
        if image.shape != expected:
            raise ValueError(f"image shape {image.shape} does not match CameraInfo {expected}")
        image.setflags(write=False)
        object.__setattr__(self, "image_bgr", image)


class ROSFrameBuffer:
    def __init__(self, *, maximum_frames: int = 30) -> None:
        if maximum_frames <= 0:
            raise ValueError("maximum_frames must be positive")
        self._frames: deque[ROSImageFrame] = deque(maxlen=maximum_frames)
        self._lock = threading.Lock()

    def add(self, frame: ROSImageFrame) -> None:
        with self._lock:
            if (
                self._frames
                and frame.timing.receipt_monotonic_s <= self._frames[-1].timing.receipt_monotonic_s
            ):
                raise ValueError("image receipt times must be strictly increasing")
            self._frames.append(frame)

    @property
    def latest(self) -> ROSImageFrame | None:
        with self._lock:
            return None if not self._frames else self._frames[-1]

    def snapshot(self) -> tuple[ROSImageFrame, ...]:
        with self._lock:
            return tuple(self._frames)


class ROSCameraSubscriber:
    """Subscriptions owned by an existing rclpy node; caller controls spinning."""

    def __init__(
        self,
        node: Any,
        *,
        image_topic: str,
        camera_info_topic: str,
        camera_name: str,
        serial_number: str,
        reliability: str = "reliable",
        qos_depth: int = 2,
        clock: MonotonicClock | None = None,
        utc_now: Callable[[], str] = utc_now_iso,
        maximum_frames: int = 30,
    ) -> None:
        if not image_topic or not camera_info_topic:
            raise ValueError("ROS camera topics must be non-empty")
        if reliability not in {"reliable", "best-effort"}:
            raise ValueError("ROS camera reliability must be 'reliable' or 'best-effort'")
        if qos_depth <= 0:
            raise ValueError("ROS camera QoS depth must be positive")
        try:
            from rclpy.qos import (
                DurabilityPolicy,
                HistoryPolicy,
                QoSProfile,
                ReliabilityPolicy,
            )
            from sensor_msgs.msg import CameraInfo, Image
        except ImportError as error:
            raise RuntimeError(
                "ROS 2 Python camera messages unavailable; source the ROS "
                "environment before creating ROSCameraSubscriber"
            ) from error
        reliability_policy = (
            ReliabilityPolicy.RELIABLE
            if reliability == "reliable"
            else ReliabilityPolicy.BEST_EFFORT
        )
        subscription_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=qos_depth,
            reliability=reliability_policy,
            durability=DurabilityPolicy.VOLATILE,
        )
        self.clock = clock or SystemClock()
        self._utc_now = utc_now
        self.camera_name = camera_name
        self.serial_number = serial_number
        self.reliability = reliability
        self.qos_depth = qos_depth
        self.frames = ROSFrameBuffer(maximum_frames=maximum_frames)
        self.last_error: str | None = None
        self._camera_info: RectifiedCameraInfo | None = None
        self._node = node
        self._info_subscription = node.create_subscription(
            CameraInfo,
            camera_info_topic,
            self._on_camera_info,
            subscription_qos,
        )
        self._image_subscription = node.create_subscription(
            Image, image_topic, self._on_image, subscription_qos
        )

    def _on_camera_info(self, message: Any) -> None:
        try:
            self._camera_info = camera_info_from_ros(
                message,
                camera_name=self.camera_name,
                serial_number=self.serial_number,
            )
            self.last_error = None
        except (AttributeError, TypeError, ValueError) as error:
            self.last_error = f"invalid rectified CameraInfo: {error}"

    def _on_image(self, message: Any) -> None:
        receipt = self.clock.monotonic()
        try:
            info = self._camera_info
            if info is None:
                raise ValueError("no valid rectified CameraInfo received")
            if str(message.header.frame_id) != info.frame_id:
                raise ValueError("Image and CameraInfo frame_id differ")
            frame = ROSImageFrame(
                image_bgr=image_bgr_from_ros(message),
                timing=ImageTiming(
                    receipt_monotonic_s=receipt,
                    receipt_utc=self._utc_now(),
                    header_stamp_ns=ros_stamp_to_ns(message.header.stamp),
                ),
                camera_info=info,
            )
            self.frames.add(frame)
            self.last_error = None
        except (AttributeError, TypeError, ValueError) as error:
            self.last_error = f"invalid rectified Image: {error}"

    def close(self) -> None:
        self._node.destroy_subscription(self._image_subscription)
        self._node.destroy_subscription(self._info_subscription)
