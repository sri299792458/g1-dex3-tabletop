"""Standalone empty-hand Dex3 open/close measurement.

This command never constructs an arm, camera, planner, or PC2 watchdog object.
"""

from __future__ import annotations

import json
import time

import numpy as np

from g1_aprilcube_calibration.process_lock import CommandOwnerLock
from g1_aprilcube_calibration.transports.unitree_dex3 import (
    Dex3StatePair,
    UnitreeDex3PostureController,
    UnitreeDex3StateObserver,
    dex3_motor_joint_name,
)
from g1_dex3_tabletop.hardware_config import dex3_config
from g1_dex3_tabletop.planning.dex3_handedness import dex3_execution_profile

EMPTY_CLOSE_ACK = "I CONFIRM THE SELECTED DEX3 FINGER SWEEP IS CLEAR"


def _wait_for_state(
    observer: UnitreeDex3StateObserver,
    *,
    timeout_s: float = 5.0,
) -> Dex3StatePair:
    deadline = time.monotonic() + timeout_s
    last_error: RuntimeError | None = None
    while time.monotonic() < deadline:
        try:
            return observer.observe()
        except RuntimeError as error:
            last_error = error
            time.sleep(0.02)
    detail = "no valid state received" if last_error is None else str(last_error)
    raise RuntimeError(f"timed out waiting for Dex3 state: {detail}")


def _hand_result(
    pair: Dex3StatePair,
    *,
    arm: str,
    target_q_rad: tuple[float, ...],
) -> dict:
    hand = pair.left if arm == "left" else pair.right
    target = np.asarray(target_q_rad, dtype=np.float64)
    residual = target - hand.position
    worst = int(np.argmax(np.abs(residual)))
    return {
        "target_q_rad": target.tolist(),
        "measured_q_rad": hand.position.tolist(),
        "target_minus_measured_q_rad": residual.tolist(),
        "maximum_abs_target_error_rad": float(np.max(np.abs(residual))),
        "worst_motor_id": worst,
        "worst_joint": dex3_motor_joint_name(arm, worst),
        "measured_dq_rad_s": hand.velocity.tolist(),
        "measured_tau_est": hand.estimated_torque.tolist(),
        "pressure": hand.pressure.tolist(),
    }


def run_measure_dex3_empty_close(args) -> int:
    """Open and close one empty hand while holding the other at its measured state."""

    if args.confirm != EMPTY_CLOSE_ACK:
        raise ValueError(f"--confirm must equal: {EMPTY_CLOSE_ACK}")
    config = dex3_config(
        args.hardware_config,
        interface=args.network_interface,
        domain_id=args.domain_id,
    )
    open_q, close_q = dex3_execution_profile(args.arm)
    observer: UnitreeDex3StateObserver | None = None
    controller: UnitreeDex3PostureController | None = None
    with CommandOwnerLock(args.lock_file):
        try:
            observer = UnitreeDex3StateObserver(config)
            _wait_for_state(observer)
            controller = UnitreeDex3PostureController(config, observer=observer)
            observer = None
            held = controller.acquire_measured_hold()
            inactive_q = held.right.position if args.arm == "left" else held.left.position
            if args.arm == "left":
                open_pair = controller.command_posture(
                    left_target_q_rad=open_q,
                    right_target_q_rad=inactive_q,
                    label="left empty-hand open measurement",
                )
                close_pair = controller.command_posture(
                    left_target_q_rad=close_q,
                    right_target_q_rad=inactive_q,
                    label="left empty-hand descriptor close measurement",
                )
            else:
                open_pair = controller.command_posture(
                    left_target_q_rad=inactive_q,
                    right_target_q_rad=open_q,
                    label="right empty-hand open measurement",
                )
                close_pair = controller.command_posture(
                    left_target_q_rad=inactive_q,
                    right_target_q_rad=close_q,
                    label="right empty-hand descriptor close measurement",
                )
            result = {
                "commands_robot": True,
                "scope": "Dex3 fingers only; no arm, camera, planner, or PC2 command",
                "arm": args.arm,
                "initial_measured_q_rad": (
                    held.left.position.tolist()
                    if args.arm == "left"
                    else held.right.position.tolist()
                ),
                "open": _hand_result(open_pair, arm=args.arm, target_q_rad=open_q),
                "empty_close": _hand_result(
                    close_pair,
                    arm=args.arm,
                    target_q_rad=close_q,
                ),
                "terminal_action": "Unitree Dex3 timeout",
            }
            controller.timeout_and_close()
            controller = None
            print(json.dumps(result, indent=2, sort_keys=True))
            return 0
        finally:
            if controller is not None:
                controller.timeout_and_close()
            elif observer is not None:
                observer.close()
