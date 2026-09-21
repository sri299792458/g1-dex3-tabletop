from __future__ import annotations

from argparse import Namespace

import numpy as np
import pytest

from g1_aprilcube_calibration.transports.unitree_dex3 import (
    Dex3HandState,
    Dex3StatePair,
)
from g1_dex3_tabletop.cli import build_parser
from g1_dex3_tabletop.hardware_dex3_empty_close import (
    EMPTY_CLOSE_ACK,
    _hand_result,
)


def _hand(q: list[float]) -> Dex3HandState:
    return Dex3HandState(
        receipt_monotonic_s=1.0,
        position=np.asarray(q),
        velocity=np.zeros(7),
        estimated_torque=np.arange(7),
        pressure=np.zeros((9, 12)),
    )


def test_empty_close_command_is_a_separate_hardware_command() -> None:
    args = build_parser().parse_args(
        [
            "measure-dex3-empty-close",
            "--arm",
            "left",
            "--network-interface",
            "enp1s0",
            "--confirm",
            EMPTY_CLOSE_ACK,
        ]
    )

    assert isinstance(args, Namespace)
    assert args.arm == "left"
    assert args.hardware_config is None


def test_hand_result_reports_the_measured_descriptor_residual() -> None:
    pair = Dex3StatePair(_hand([0.0, 0.4, 0.8, -0.7, -0.9, -0.8, -0.9]), _hand([0.0] * 7))
    target = (0.0, 0.6, 1.0, -0.9, -1.0, -0.9, -1.0)

    result = _hand_result(pair, arm="left", target_q_rad=target)

    assert result["maximum_abs_target_error_rad"] == pytest.approx(0.2)
    assert result["worst_joint"] == "left_hand_middle_0_joint"
    assert result["pressure"] == np.zeros((9, 12)).tolist()
