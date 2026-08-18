from __future__ import annotations

import numpy as np
import pytest

from g1_dex3_tabletop.planning.tabletop_planner import (
    _validate_supported_escape_endpoint,
)


def transform(z: float) -> np.ndarray:
    value = np.eye(4, dtype=np.float64)
    value[2, 3] = z
    return value


def test_supported_escape_endpoint_uses_requested_displacement() -> None:
    achieved_m, error_m = _validate_supported_escape_endpoint(
        start=transform(-0.060),
        target=transform(0.040),
        endpoint=transform(0.040),
        down=np.array((0.0, 0.0, -1.0)),
        position_tolerance_m=0.005,
    )

    assert achieved_m == pytest.approx(0.100)
    assert error_m == pytest.approx(0.0)


def test_supported_escape_endpoint_rejects_half_completed_lift() -> None:
    with pytest.raises(RuntimeError, match="missed its requested Cartesian endpoint"):
        _validate_supported_escape_endpoint(
            start=transform(0.0),
            target=transform(0.100),
            endpoint=transform(0.051),
            down=np.array((0.0, 0.0, -1.0)),
            position_tolerance_m=0.005,
        )
