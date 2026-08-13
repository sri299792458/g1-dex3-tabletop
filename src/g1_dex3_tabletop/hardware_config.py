"""Narrow construction of the already commissioned G1 runtime components."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import yaml

from g1_aprilcube_calibration.executor_state_machine import ExecutorConfig
from g1_aprilcube_calibration.gravity_compensation import G1PinocchioGravityFeedforward
from g1_aprilcube_calibration.pc2_safety import PC2DampingWatchdog, PC2SafetyConfig
from g1_aprilcube_calibration.readiness import RecordingGateConfig
from g1_aprilcube_calibration.timestamp_pairing import PairingConfig
from g1_aprilcube_calibration.transports.unitree_arm_sdk import UnitreeTransportConfig
from g1_aprilcube_calibration.transports.unitree_debug_lowcmd import UnitreeDebugLowCmdConfig
from g1_aprilcube_calibration.transports.unitree_dex3 import Dex3ControlConfig


def load_hardware(path: str | Path) -> dict[str, Any]:
    value = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError("hardware configuration must contain a mapping")
    return value


def resolve_hardware_path(path: str | Path, value: str | Path) -> Path:
    source = Path(path).resolve()
    configured = Path(value)
    return configured if configured.is_absolute() else source.parent.parent / configured


def transport_config(
    path: str | Path, *, interface: str, domain_id: int
) -> UnitreeTransportConfig:
    control = load_hardware(path)["control"]
    return UnitreeTransportConfig(
        network_interface=interface,
        domain_id=domain_id,
        shoulder_elbow_kp=float(control["hold_shoulder_elbow_kp"]),
        shoulder_elbow_kd=float(control["hold_shoulder_elbow_kd"]),
        wrist_kp=float(control["hold_wrist_kp"]),
        wrist_kd=float(control["hold_wrist_kd"]),
    )


def dex3_config(path: str | Path, *, interface: str, domain_id: int) -> Dex3ControlConfig:
    hardware = load_hardware(path)
    value = hardware["dex3_control"]
    if value.get("enabled") is not True or value.get("posture_control_commissioned") is not True:
        raise RuntimeError("Dex3 posture control is not commissioned in hardware.yaml")
    return Dex3ControlConfig(
        network_interface=interface,
        domain_id=domain_id,
        left_command_topic=str(value["left_command_topic"]),
        right_command_topic=str(value["right_command_topic"]),
        left_state_topic=str(value["left_state_topic"]),
        right_state_topic=str(value["right_state_topic"]),
        command_rate_hz=float(value["command_rate_hz"]),
        kp=float(value["kp"]),
        kd=float(value["kd"]),
        left_target_q_rad=tuple(value["left_target_q_rad"]),
        right_target_q_rad=tuple(value["right_target_q_rad"]),
        state_freshness_timeout_s=float(value["state_freshness_timeout_s"]),
        maximum_measured_command_delta_rad=float(value["maximum_measured_command_delta_rad"]),
        posture_ramp_s=float(value["posture_ramp_s"]),
        posture_position_tolerance_rad=float(value["posture_position_tolerance_rad"]),
        posture_position_spread_rad=float(control_value(path, "settled_position_spread_rad")),
        posture_settle_dwell_s=float(value["posture_settle_dwell_s"]),
        posture_timeout_s=float(value["posture_timeout_s"]),
        timeout_repetitions=int(value["timeout_repetitions"]),
    )


def control_value(path: str | Path, name: str):
    return load_hardware(path)["control"][name]


def executor_config(path: str | Path) -> tuple[ExecutorConfig, float]:
    hardware = load_hardware(path)
    control = hardware["control"]
    rate_hz = float(control["command_rate_hz"])
    gap = float(control["control_gap_fault_s"])
    heartbeat_timeout = float(hardware["safety"]["heartbeat_timeout_s"])
    if gap > heartbeat_timeout / 2.0:
        raise ValueError("control gap limit exceeds half the PC2 watchdog timeout")
    return (
        ExecutorConfig(
            maximum_joint_velocity_rad_s=float(control["max_joint_velocity_rad_s"]),
            motion_position_tolerance_rad=float(control["motion_position_tolerance_rad"]),
            ownership_transition_position_tolerance_rad=float(
                control["ownership_transition_position_tolerance_rad"]
            ),
            activation_position_tolerance_rad=float(control["activation_position_tolerance_rad"]),
            held_arm_position_tolerance_rad=float(control["held_arm_position_tolerance_rad"]),
            settled_position_spread_rad=float(control["settled_position_spread_rad"]),
            settle_dwell_s=float(control["settle_dwell_s"]),
            state_freshness_timeout_s=float(control["lowstate_timeout_s"]),
            nominal_tick_period_s=1.0 / rate_hz,
            control_gap_fault_s=gap,
            acquisition_ramp_s=float(control["acquisition_weight_ramp_s"]),
            release_ramp_s=float(control["release_weight_ramp_s"]),
            motion_timeout_s=float(control["motion_timeout_s"]),
        ),
        rate_hz,
    )


def recording_configs(path: str | Path) -> tuple[RecordingGateConfig, PairingConfig]:
    hardware = load_hardware(path)
    control = hardware["control"]
    value = hardware["pose_recording"]
    return (
        RecordingGateConfig(
            calibration_arm=str(control["calibration_arm"]),
            state_freshness_timeout_s=float(control["lowstate_timeout_s"]),
            stationary_duration_s=float(value["stationary_duration_s"]),
            maximum_state_gap_s=float(value["maximum_state_gap_s"]),
            maximum_calibration_position_spread_rad=float(
                value["maximum_calibration_position_spread_rad"]
            ),
            maximum_hold_position_spread_rad=float(value["maximum_hold_position_spread_rad"]),
            minimum_samples=int(value["minimum_state_samples"]),
        ),
        PairingConfig(
            maximum_nearest_delta_s=float(value["maximum_image_state_delta_s"]),
            maximum_bracket_span_s=float(value["maximum_image_state_bracket_s"]),
        ),
    )


def debug_lowcmd_config(path: str | Path) -> UnitreeDebugLowCmdConfig:
    hardware = load_hardware(path)
    control = hardware["control"]
    if control.get("seated_debug_lowcmd_commissioned") is not True:
        raise RuntimeError("seated debug lowcmd ownership is not commissioned")
    if control.get("seated_gravity_feedforward_commissioned") is not True:
        raise RuntimeError("seated Dex3 gravity feedforward is not commissioned")
    return UnitreeDebugLowCmdConfig(
        command_topic=str(control["seated_debug_command_topic"]),
        body_strong_kp=float(control["debug_body_strong_kp"]),
        body_strong_kd=float(control["debug_body_strong_kd"]),
        body_weak_kp=float(control["debug_body_weak_kp"]),
        body_weak_kd=float(control["debug_body_weak_kd"]),
        activation_position_tolerance_rad=float(control["activation_position_tolerance_rad"]),
        body_hold_position_tolerance_rad=float(control["debug_body_hold_position_tolerance_rad"]),
        motion_switch_timeout_s=float(control["debug_motion_switch_timeout_s"]),
        motion_switch_poll_interval_s=float(control["debug_motion_switch_poll_interval_s"]),
    )


def gravity_feedforward(
    path: str | Path, reference_q29: np.ndarray
) -> G1PinocchioGravityFeedforward:
    hardware = load_hardware(path)
    control = hardware["control"]
    if control.get("gravity_feedforward") != "pinocchio_rnea_at_commanded_q":
        raise ValueError("gravity feedforward must use Pinocchio RNEA")
    urdf = resolve_hardware_path(path, control["gravity_model_urdf"])
    result = G1PinocchioGravityFeedforward(
        urdf,
        locked_joint_positions_rad=control["gravity_locked_joint_positions_rad"],
    )
    result.seed_reference(np.asarray(reference_q29, dtype=np.float64))
    return result


def watchdog(
    path: str | Path,
    *,
    host: str,
    ssh_identity: Path,
    initial_fsm_id: int,
    restore_seated: bool,
) -> PC2DampingWatchdog:
    hardware = load_hardware(path)
    safety = hardware["safety"]
    if safety.get("physical_support") != "load_bearing_harness":
        raise ValueError("hardware safety precondition must be the load-bearing harness")
    config = PC2SafetyConfig(
        host=host,
        ssh_identity=ssh_identity,
        heartbeat_interval_s=float(safety["heartbeat_interval_s"]),
        heartbeat_timeout_s=float(safety["heartbeat_timeout_s"]),
        connect_timeout_s=float(safety["connect_timeout_s"]),
        client_timeout_s=float(safety["loco_client_timeout_s"]),
        required_initial_fsm_id=initial_fsm_id,
        restore_motion_service_before_loco=restore_seated,
        manage_dex3=bool(safety.get("manage_dex3", False)),
        remote_ros_setup=Path(safety["pc2_ros_setup"]),
        remote_cyclonedds_setup=Path(safety["pc2_cyclonedds_setup"]),
        remote_unitree_setup=Path(safety["pc2_unitree_ros2_setup"]),
        remote_cyclonedds_uri=Path(safety["pc2_cyclonedds_uri"]),
        remote_watchdog_python=Path(safety["pc2_watchdog_python"]),
        remote_cyclonedds_home=Path(safety["pc2_cyclonedds_home"]),
        remote_motion_switcher_interface=str(safety["pc2_motion_switcher_interface"]),
        remote_dex3_interface=str(safety["pc2_dex3_interface"]),
        remote_damp_executable=Path(safety["pc2_g1_loco_client"]),
    )
    return PC2DampingWatchdog(config)
