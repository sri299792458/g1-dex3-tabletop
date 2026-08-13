import numpy as np
import pytest

from g1_aprilcube_calibration.motion_profile import velocity_limited_step


def test_velocity_limited_step_clamps_each_joint_without_overshoot() -> None:
    result = velocity_limited_step(
        np.zeros(14),
        np.array([1.0, -1.0, 0.001, *([0.0] * 11)]),
        maximum_velocity_rad_s=0.2,
        duration_s=0.01,
    )
    assert np.allclose(result[:3], [0.002, -0.002, 0.001])
    assert not result.flags.writeable


@pytest.mark.parametrize(
    "kwargs",
    [
        {"current": np.zeros(13), "goal": np.zeros(14)},
        {"current": np.zeros(14), "goal": np.full(14, np.nan)},
        {"current": np.zeros(14), "goal": np.zeros(14), "duration_s": -1.0},
    ],
)
def test_velocity_limited_step_rejects_bad_input(kwargs) -> None:
    defaults = {
        "current": np.zeros(14),
        "goal": np.zeros(14),
        "maximum_velocity_rad_s": 0.2,
        "duration_s": 0.01,
    }
    defaults.update(kwargs)
    with pytest.raises(ValueError):
        velocity_limited_step(**defaults)
