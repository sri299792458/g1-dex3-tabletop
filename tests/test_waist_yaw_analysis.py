import pytest

from g1_dex3_tabletop.planning.waist_yaw_analysis import _validated_ranges
from g1_dex3_tabletop.planning.worker import build_parser


def test_waist_yaw_ranges_must_be_positive_unique_and_increasing() -> None:
    assert _validated_ranges((0.1, 0.2, 0.3)) == (0.1, 0.2, 0.3)
    for invalid in ((), (0.0,), (-0.1,), (0.2, 0.1), (0.1, 0.1)):
        with pytest.raises(ValueError):
            _validated_ranges(invalid)


def test_worker_requires_explicit_waist_yaw_study_bounds() -> None:
    args = build_parser().parse_args(
        [
            "analyze-waist-yaw",
            "--request",
            "request.json",
            "--output",
            "analysis.json",
            "--waist-half-range-rad",
            "0.1",
            "--waist-half-range-rad",
            "0.3",
            "--candidate-id",
            "grasp_17",
        ]
    )

    assert args.waist_half_range_rad == [0.1, 0.3]
    assert args.candidate_id == ["grasp_17"]
