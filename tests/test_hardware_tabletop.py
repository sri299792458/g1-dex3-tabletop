from types import SimpleNamespace

import pytest

from g1_aprilcube_calibration.executor_state_machine import ExecutorState
from g1_dex3_tabletop import hardware_tabletop


class _Resource:
    def __init__(self, events: list[str], event: str) -> None:
        self._events = events
        self._event = event

    def close(self) -> None:
        self._events.append(self._event)

    def destroy_node(self) -> None:
        self._events.append(self._event)


class _Rclpy:
    def __init__(self, events: list[str]) -> None:
        self._events = events

    @staticmethod
    def ok() -> bool:
        return True

    def shutdown(self) -> None:
        self._events.append("rclpy.shutdown")


def test_ros_teardown_refuses_active_direct_robot_ownership(monkeypatch) -> None:
    events: list[str] = []
    monkeypatch.setattr(
        hardware_tabletop.cv2,
        "destroyAllWindows",
        lambda: events.append("cv2.destroyAllWindows"),
    )

    with pytest.raises(RuntimeError, match="verified external robot-control takeover"):
        hardware_tabletop._teardown_ros_runtime(
            no_window=False,
            camera=_Resource(events, "camera.close"),
            node=_Resource(events, "node.destroy_node"),
            rclpy_module=_Rclpy(events),
            transport=SimpleNamespace(requires_external_takeover=True),
            synchronized=SimpleNamespace(state=ExecutorState.READY),
        )

    assert events == []


def test_ros_teardown_runs_after_verified_external_takeover(monkeypatch) -> None:
    events: list[str] = []
    monkeypatch.setattr(
        hardware_tabletop.cv2,
        "destroyAllWindows",
        lambda: events.append("cv2.destroyAllWindows"),
    )

    hardware_tabletop._teardown_ros_runtime(
        no_window=False,
        camera=_Resource(events, "camera.close"),
        node=_Resource(events, "node.destroy_node"),
        rclpy_module=_Rclpy(events),
        transport=SimpleNamespace(requires_external_takeover=True),
        synchronized=SimpleNamespace(state=ExecutorState.STOPPED),
    )

    assert events == [
        "cv2.destroyAllWindows",
        "camera.close",
        "node.destroy_node",
        "rclpy.shutdown",
    ]
