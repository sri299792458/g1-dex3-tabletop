"""Operator full-close readiness and immutable measured-hold contracts."""

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from g1_aprilcube_calibration.clock import ManualClock
from g1_aprilcube_calibration.transports.unitree_dex3 import Dex3ControlConfig
from g1_dex3_tabletop.calibration.hand_posture import (
    check_full_close,
    full_close_reference,
    wait_for_full_close,
)
from g1_dex3_tabletop.planning.contracts import Dex3PreparationRequest, RobotSnapshot
from g1_dex3_tabletop.planning.dex3_handedness import dex3_execution_profile


def pair():
    return SimpleNamespace(
        **{
            side: SimpleNamespace(position=tuple(q))
            for side, q in full_close_reference()["reference_q_rad"].items()
        }
    )


def check(hands):
    return check_full_close(
        left_q_rad=hands.left.position, right_q_rad=hands.right.position, tolerance_rad=0.08
    )


@pytest.mark.parametrize("side", ["left", "right"])
@pytest.mark.parametrize("posture", ["open", "grasp", "nan", "infinity", "short", "outside"])
def test_rejects_unready_hand(side, posture):
    hands = pair()
    values = list(getattr(hands, side).position)
    if posture == "open":
        values = [0.0] * 7
    elif posture == "grasp":
        values = dex3_execution_profile(side)[1]
    elif posture == "short":
        values = values[:6]
    else:
        values[1] = {
            "nan": float("nan"),
            "infinity": float("inf"),
            "outside": values[1] + 0.080001,
        }[posture]
    getattr(hands, side).position = values
    with pytest.raises(ValueError, match=side):
        check(hands)


def test_accepts_reference_and_both_sides_of_readiness_tolerance():
    hands = pair()
    assert check(hands)["hands"]["left"]["maximum_reference_error_rad"] == 0
    hands.left.position = tuple(q + 0.079999 for q in hands.left.position)
    hands.right.position = tuple(q - 0.079999 for q in hands.right.position)
    check(hands)


@pytest.mark.parametrize("behavior", ["stationary", "moves_once", "never_settles", "stale"])
def test_requires_continuous_fresh_stationary_window(behavior):
    clock = ManualClock(0)
    config = replace(
        Dex3ControlConfig(network_interface="offline"), posture_ramp_s=0, posture_timeout_s=1.5
    )
    calls = []

    def observe():
        now = clock.monotonic()
        calls.append(now)
        if behavior == "stale" and now >= 0.1:
            raise RuntimeError("stale Dex3 feedback")
        hands = pair()
        delta = 0
        if behavior == "moves_once" and now >= 0.25:
            delta = 0.02
        elif behavior == "never_settles":
            delta = 0.02 * (int(now * 10) % 2)
        hands.left.position = tuple(q + delta for q in hands.left.position)
        return hands

    kwargs = {
        "observer": SimpleNamespace(observe=observe),
        "config": config,
        "clock": clock.monotonic,
        "sleep": clock.advance,
    }
    if behavior in ("never_settles", "stale"):
        with pytest.raises(RuntimeError, match="stationary|stale"):
            wait_for_full_close(**kwargs)
    else:
        measured, evidence = wait_for_full_close(**kwargs)
        assert evidence["stationary_duration_s"] >= 0.5
        assert evidence["maximum_position_spread_rad"] <= 0.01
        assert clock.monotonic() >= (0.75 if behavior == "moves_once" else 0.5)
        assert measured.left.position == observe().left.position


def test_measured_hold_serializes_actual_readings_and_forbids_any_retarget():
    hands = pair()
    snapshot = RobotSnapshot((0.0,) * 29, hands.left.position, hands.right.position)
    request = Dex3PreparationRequest.for_measured_hold(
        snapshot=snapshot, joint_position_offsets_rad={}
    )
    assert Dex3PreparationRequest.from_dict(request.to_dict()) == request
    assert request.holds_measured_fingers
    for side in ("left", "right"):
        for name in ("target", "settled_target", "return_target"):
            field = f"{side}_{name}_q_rad"
            assert getattr(request, field) == getattr(hands, side).position
            with pytest.raises(ValueError, match="measured"):
                replace(request, **{field: tuple(np.asarray(getattr(request, field)) + 0.001)})
