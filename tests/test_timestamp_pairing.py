import numpy as np
import pytest

from g1_aprilcube_calibration.models import RobotStateSample
from g1_aprilcube_calibration.timestamp_pairing import (
    ImageTiming,
    PairingConfig,
    pair_state_to_image,
)


def state(time_s: float) -> RobotStateSample:
    return RobotStateSample(
        time_s,
        "2026-08-02T12:00:00Z",
        5,
        np.zeros(29),
        np.zeros(29),
        np.full(29, -time_s),
    )


def test_image_is_paired_to_nearest_bracketed_state() -> None:
    image = ImageTiming(1.04, "2026-08-02T12:00:00Z", header_stamp_ns=123)
    result = pair_state_to_image(
        image,
        [state(1.0), state(1.05), state(1.1)],
        config=PairingConfig(0.05, 0.1),
    )
    assert result.before.receipt_monotonic_s == 1.0
    assert result.after.receipt_monotonic_s == 1.05
    assert result.nearest.receipt_monotonic_s == 1.05
    assert result.nearest_delta_s == pytest.approx(0.01)


def test_pairing_rejects_unbracketed_and_distant_states() -> None:
    config = PairingConfig(0.02, 0.1)
    with pytest.raises(ValueError, match="not bracketed"):
        pair_state_to_image(
            ImageTiming(0.9, "2026-08-02T12:00:00Z"),
            [state(1.0), state(1.05)],
            config=config,
        )
    with pytest.raises(ValueError, match="nearest"):
        pair_state_to_image(
            ImageTiming(1.025, "2026-08-02T12:00:00Z"),
            [state(1.0), state(1.05)],
            config=config,
        )


def test_pairing_rejects_wide_and_nonmonotonic_brackets() -> None:
    image = ImageTiming(1.1, "2026-08-02T12:00:00Z")
    with pytest.raises(ValueError, match="spans"):
        pair_state_to_image(
            image,
            [state(1.0), state(1.2)],
            config=PairingConfig(0.11, 0.1),
        )
    with pytest.raises(ValueError, match="strictly increasing"):
        pair_state_to_image(
            image,
            [state(1.2), state(1.0)],
            config=PairingConfig(0.2, 0.3),
        )
