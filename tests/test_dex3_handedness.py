from __future__ import annotations

from g1_dex3_tabletop.planning.dex3_handedness import (
    dex3_q_from_canonical,
    dex3_q_from_qualified_right_mapping,
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


def test_qualified_right_posture_adapts_to_left() -> None:
    source = {
        "right_hand_thumb_0_joint": 0.1,
        "right_hand_thumb_1_joint": 0.2,
        "right_hand_thumb_2_joint": 0.3,
        "right_hand_middle_0_joint": 0.4,
        "right_hand_middle_1_joint": 0.5,
        "right_hand_index_0_joint": 0.6,
        "right_hand_index_1_joint": 0.7,
    }

    assert dex3_q_from_qualified_right_mapping(source, arm="left") == (
        -0.1,
        -0.2,
        -0.3,
        -0.6,
        -0.7,
        -0.4,
        -0.5,
    )
