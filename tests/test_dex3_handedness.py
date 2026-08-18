from __future__ import annotations

import pytest

from g1_dex3_tabletop.planning.dex3_handedness import (
    dex3_empty_close_reference,
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


def test_left_empty_close_reference_is_commissioned_but_right_is_not() -> None:
    left, threshold = dex3_empty_close_reference("left")

    assert left == (
        -0.02220277674496174,
        0.5717323422431946,
        0.9781225919723511,
        -0.8787803053855896,
        -0.9752545356750488,
        -0.8834150433540344,
        -0.9728087186813354,
    )
    assert threshold == 0.05

    with pytest.raises(RuntimeError, match="not commissioned for right"):
        dex3_empty_close_reference("right")
