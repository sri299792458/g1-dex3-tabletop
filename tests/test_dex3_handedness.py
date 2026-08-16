from __future__ import annotations

from g1_dex3_tabletop.planning.dex3_handedness import (
    dex3_execution_profile,
    dex3_q_from_canonical,
)


def test_canonical_posture_is_identity_for_right_and_exact_mirror_for_left() -> None:
    canonical = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7)

    assert dex3_q_from_canonical(canonical, arm="right") == canonical
    assert dex3_q_from_canonical(canonical, arm="left") == (
        -0.1,
        -0.2,
        -0.3,
        -0.6,
        -0.7,
        -0.4,
        -0.5,
    )


def test_execution_profile_has_one_fixed_close_target_mirrored_to_each_hand() -> None:
    right_open, right_close = dex3_execution_profile("right")
    left_open, left_close = dex3_execution_profile("left")

    assert right_open == (0.0,) * 7
    assert right_close == (
        0.0,
        -0.5984,
        -0.99731429,
        0.8976,
        0.99731429,
        0.8976,
        0.99731429,
    )
    assert left_open == (0.0,) * 7
    assert left_close == (
        -0.0,
        0.5984,
        0.99731429,
        -0.8976,
        -0.99731429,
        -0.8976,
        -0.99731429,
    )
