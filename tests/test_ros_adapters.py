from __future__ import annotations

import sys
from dataclasses import dataclass, field
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

from g1_aprilcube_calibration.joint_map import G1_29_JOINT_NAMES
from g1_aprilcube_calibration.ros.camera_adapter import (
    ROSCameraSubscriber,
    ROSFrameBuffer,
    ROSImageFrame,
    camera_info_from_ros,
    image_bgr_from_ros,
    ros_stamp_to_ns,
)
from g1_aprilcube_calibration.ros.joint_state_adapter import (
    robot_state_from_joint_state,
)
from g1_aprilcube_calibration.timestamp_pairing import ImageTiming


@dataclass
class Stamp:
    sec: int = 2
    nanosec: int = 3


@dataclass
class Header:
    frame_id: str = "camera_color_optical_frame"
    stamp: Stamp = field(default_factory=Stamp)


@dataclass
class CameraInfoMessage:
    width: int = 2
    height: int = 2
    header: Header = field(default_factory=Header)
    distortion_model: str = "plumb_bob"
    d: list[float] = field(default_factory=lambda: [0.0] * 5)
    k: list[float] = field(
        default_factory=lambda: [
            600.0,
            0.0,
            0.5,
            0.0,
            600.0,
            0.5,
            0.0,
            0.0,
            1.0,
        ]
    )
    r: list[float] = field(default_factory=lambda: [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0])
    p: list[float] = field(
        default_factory=lambda: [
            600.0,
            0.0,
            0.5,
            0.0,
            0.0,
            600.0,
            0.5,
            0.0,
            0.0,
            0.0,
            1.0,
            0.0,
        ]
    )


@dataclass
class ImageMessage:
    width: int
    height: int
    step: int
    encoding: str
    data: bytes
    header: Header = field(default_factory=Header)


def camera_info():
    return camera_info_from_ros(
        CameraInfoMessage(), camera_name="head_color", serial_number="D435-TEST"
    )


def test_rectified_camera_info_and_ros_stamp_conversion():
    info = camera_info()
    assert info.width == 2
    assert info.profile_sha256
    assert ros_stamp_to_ns(Stamp()) == 2_000_000_003
    message = CameraInfoMessage()
    message.d = [0.1, 0, 0, 0, 0]
    with pytest.raises(ValueError, match="zero distortion"):
        camera_info_from_ros(message, camera_name="head", serial_number="serial")


def test_image_decoder_supports_padding_and_channel_conversion():
    rgb_with_padding = bytes([255, 0, 0, 0, 255, 0, 9, 9])
    image = image_bgr_from_ros(ImageMessage(2, 1, 8, "rgb8", rgb_with_padding))
    assert image.tolist() == [[[0, 0, 255], [0, 255, 0]]]
    mono = image_bgr_from_ros(ImageMessage(2, 1, 2, "mono8", bytes([4, 7])))
    assert mono.tolist() == [[[4, 4, 4], [7, 7, 7]]]
    with pytest.raises(ValueError, match="unsupported"):
        image_bgr_from_ros(ImageMessage(1, 1, 4, "rgba8", bytes(4)))


def test_frame_buffer_is_bounded_monotonic_and_immutable():
    buffer = ROSFrameBuffer(maximum_frames=2)
    for time_s in (1.0, 2.0, 3.0):
        source = np.zeros((2, 2, 3), dtype=np.uint8)
        frame = ROSImageFrame(
            source,
            ImageTiming(time_s, "2026-08-02T12:00:00Z"),
            camera_info(),
        )
        source[:] = 255
        buffer.add(frame)
    assert [item.timing.receipt_monotonic_s for item in buffer.snapshot()] == [2, 3]
    assert not buffer.latest.image_bgr.flags.writeable
    assert int(buffer.latest.image_bgr.max()) == 0
    with pytest.raises(ValueError, match="strictly increasing"):
        buffer.add(buffer.latest)


class FakeNode:
    def __init__(self):
        self.subscriptions = []
        self.destroyed = []

    def create_subscription(self, message_type, topic, callback, qos):
        subscription = SimpleNamespace(
            message_type=message_type,
            topic=topic,
            callback=callback,
            qos=qos,
        )
        self.subscriptions.append(subscription)
        return subscription

    def destroy_subscription(self, subscription):
        self.destroyed.append(subscription)


def _install_fake_ros_messages(monkeypatch):
    rclpy = ModuleType("rclpy")
    qos = ModuleType("rclpy.qos")
    sensor_msgs = ModuleType("sensor_msgs")
    sensor_msgs_msg = ModuleType("sensor_msgs.msg")

    class QoSProfile:
        def __init__(self, **values):
            self.__dict__.update(values)

    qos.QoSProfile = QoSProfile
    qos.ReliabilityPolicy = SimpleNamespace(RELIABLE="reliable", BEST_EFFORT="best-effort")
    qos.HistoryPolicy = SimpleNamespace(KEEP_LAST="keep-last")
    qos.DurabilityPolicy = SimpleNamespace(VOLATILE="volatile")
    sensor_msgs_msg.CameraInfo = type("CameraInfo", (), {})
    sensor_msgs_msg.Image = type("Image", (), {})
    rclpy.qos = qos
    sensor_msgs.msg = sensor_msgs_msg
    monkeypatch.setitem(sys.modules, "rclpy", rclpy)
    monkeypatch.setitem(sys.modules, "rclpy.qos", qos)
    monkeypatch.setitem(sys.modules, "sensor_msgs", sensor_msgs)
    monkeypatch.setitem(sys.modules, "sensor_msgs.msg", sensor_msgs_msg)


@pytest.mark.parametrize("reliability", ["reliable", "best-effort"])
def test_camera_subscriber_uses_bounded_requested_qos(monkeypatch, reliability):
    _install_fake_ros_messages(monkeypatch)
    node = FakeNode()

    camera = ROSCameraSubscriber(
        node,
        image_topic="/camera/color/image_raw",
        camera_info_topic="/camera/color/camera_info",
        camera_name="head_color",
        serial_number="D435-TEST",
        reliability=reliability,
    )

    assert camera.reliability == reliability
    assert camera.qos_depth == 2
    assert len(node.subscriptions) == 2
    for subscription in node.subscriptions:
        assert subscription.qos.reliability == reliability
        assert subscription.qos.history == "keep-last"
        assert subscription.qos.depth == 2
        assert subscription.qos.durability == "volatile"
    camera.close()
    assert node.destroyed == list(reversed(node.subscriptions))


def test_camera_subscriber_rejects_invalid_qos_before_ros_import():
    with pytest.raises(ValueError, match="reliability"):
        ROSCameraSubscriber(
            SimpleNamespace(),
            image_topic="image",
            camera_info_topic="info",
            camera_name="head_color",
            serial_number="D435-TEST",
            reliability="sometimes",
        )


def test_named_joint_state_requires_complete_measured_velocity():
    names = list(reversed(G1_29_JOINT_NAMES))
    by_name = {name: index for index, name in enumerate(G1_29_JOINT_NAMES)}

    joint_state = SimpleNamespace(
        name=names,
        position=[by_name[name] / 10 for name in names],
        velocity=[-by_name[name] / 100 for name in names],
        effort=[by_name[name] / 1000 for name in names],
    )

    result = robot_state_from_joint_state(
        joint_state,
        receipt_monotonic_s=1.0,
        receipt_utc="2026-08-02T12:00:00Z",
        mode_machine=5,
    )
    np.testing.assert_allclose(result.position, np.arange(29) / 10)
    np.testing.assert_allclose(result.velocity, -np.arange(29) / 100)
    np.testing.assert_allclose(result.estimated_torque, np.arange(29) / 1000)
    joint_state.velocity = []
    with pytest.raises(ValueError, match="equal length"):
        robot_state_from_joint_state(
            joint_state,
            receipt_monotonic_s=1,
            receipt_utc="2026-08-02T12:00:00Z",
            mode_machine=5,
        )
    joint_state.velocity = [-by_name[name] / 100 for name in names]
    joint_state.effort = []
    with pytest.raises(ValueError, match="equal length"):
        robot_state_from_joint_state(
            joint_state,
            receipt_monotonic_s=1,
            receipt_utc="2026-08-02T12:00:00Z",
            mode_machine=5,
        )
