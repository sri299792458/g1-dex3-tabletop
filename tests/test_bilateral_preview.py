"""Q must retire the preview without interrupting the certified arm motion."""

from types import SimpleNamespace

import pytest

from g1_aprilcube_calibration.executor_state_machine import ExecutorState
from g1_dex3_tabletop import hardware_bilateral_calibration as calibration
from g1_dex3_tabletop import hardware_tabletop as tabletop


@pytest.mark.parametrize("key", ["q", "Q"])
def test_q_during_motion_closes_windows_once_and_waits_for_ready(monkeypatch, capsys, key):
    stop = {"requested": False}
    progress = {}
    events = []
    now = [0.0]
    keys = iter([-1, ord(key)])
    executor = SimpleNamespace(state=ExecutorState.MOVING)
    driver = SimpleNamespace(check=lambda: events.append("control_check"))
    monkeypatch.setattr(tabletop.time, "monotonic", lambda: now[0])
    monkeypatch.setattr(calibration.cv2, "waitKey", lambda _: next(keys))
    monkeypatch.setattr(calibration.cv2, "destroyAllWindows", lambda: events.append("closed"))

    def wait_once(duration):
        calibration._poll_graceful_stop_key(stop, progress, no_window=False)
        now[0] += duration
        if now[0] >= 0.03:
            executor.state = ExecutorState.READY

    tabletop._wait_ready(
        executor, driver, timeout_s=1.0, label="recorded return edge", wait_once=wait_once
    )

    assert stop["requested"]
    assert executor.state is ExecutorState.READY
    assert now[0] == pytest.approx(0.03)
    assert events.count("closed") == 1
    assert events[-1] == "control_check"
    assert "CERTIFIED RETURN PENDING" in progress["message"]
    assert capsys.readouterr().out.count("Q RECEIVED") == 1


def test_headless_stop_poll_does_not_touch_highgui(monkeypatch):
    monkeypatch.setattr(calibration.cv2, "waitKey", lambda _: pytest.fail("headless window poll"))
    monkeypatch.setattr(
        calibration.cv2, "destroyAllWindows", lambda: pytest.fail("headless window close")
    )
    stop = {"requested": False}
    calibration._poll_graceful_stop_key(stop, {}, no_window=True)
    assert not stop["requested"]
