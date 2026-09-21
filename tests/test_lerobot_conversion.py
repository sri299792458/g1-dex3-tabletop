from types import SimpleNamespace

import pytest

from g1_dex3_tabletop.lerobot_conversion import (
    ALL_JOINT_NAMES,
    IMU_NAMES,
    PRESSURE_NAMES,
    _validate_depth_encoder_bounds,
    feature_schema,
    handcmd_is_active,
    lowcmd_is_active,
    parse_hand_state,
    parse_handcmd,
    parse_lowcmd,
    parse_lowstate,
)


def _motor(index: int, *, timeout: bool = False):
    return SimpleNamespace(
        mode=(0x80 if timeout else 0x10) | (index & 0x0F),
        q=float(index),
        dq=float(index) + 0.1,
        tau=float(index) + 0.2,
        tau_est=float(index) + 0.3,
        kp=1.0,
        kd=0.1,
    )


def _imu():
    return SimpleNamespace(
        quaternion=[0.0, 0.0, 0.0, 1.0],
        gyroscope=[1.0, 2.0, 3.0],
        accelerometer=[4.0, 5.0, 6.0],
        rpy=[7.0, 8.0, 9.0],
    )


def test_unitree_parsers_have_frozen_dimensions():
    lowstate = SimpleNamespace(motor_state=[_motor(index) for index in range(35)], imu_state=_imu())
    lowcmd = SimpleNamespace(motor_cmd=[_motor(index) for index in range(35)])
    handstate = SimpleNamespace(
        motor_state=[_motor(index) for index in range(7)],
        press_sensor_state=[SimpleNamespace(pressure=list(range(12))) for _ in range(9)],
        imu_state=_imu(),
    )
    handcmd = SimpleNamespace(motor_cmd=[_motor(index) for index in range(7)])

    assert parse_lowstate(lowstate).shape == (100,)
    assert parse_lowcmd(lowcmd).shape == (145,)
    assert parse_hand_state(handstate, side="left").shape == (142,)
    assert parse_handcmd(handcmd, side="left").shape == (35,)
    assert lowcmd_is_active(lowcmd)
    assert handcmd_is_active(handcmd)


def test_terminal_timeout_commands_are_not_training_actions():
    handcmd = SimpleNamespace(motor_cmd=[_motor(index, timeout=True) for index in range(7)])
    assert not handcmd_is_active(handcmd)

    lowcmd = SimpleNamespace(motor_cmd=[_motor(index) for index in range(35)])
    for motor in lowcmd.motor_cmd:
        motor.kp = 0.0
        motor.kd = 0.0
    assert not lowcmd_is_active(lowcmd)


def test_lerobot_schema_names_match_vectors():
    schema = feature_schema((720, 1280, 3), (480, 640, 1))
    assert len(ALL_JOINT_NAMES) == 43
    assert len(PRESSURE_NAMES) == 216
    assert len(IMU_NAMES) == 52
    assert schema["observation.state"]["shape"] == (43,)
    assert schema["action"]["shape"] == (43,)
    assert schema["observation.depth.head"]["info"]["is_depth_map"] is True


def test_resumed_dataset_must_keep_identical_depth_bounds():
    features = {
        "observation.depth.head": {
            "info": {"video.depth_min": 0.15, "video.depth_max": 2.0}
        }
    }
    _validate_depth_encoder_bounds(
        features,
        ["observation.depth.head"],
        depth_min=0.15,
        depth_max=2.0,
    )
    with pytest.raises(RuntimeError, match="not \\[0.15, 3.0\\]"):
        _validate_depth_encoder_bounds(
            features,
            ["observation.depth.head"],
            depth_min=0.15,
            depth_max=3.0,
        )
