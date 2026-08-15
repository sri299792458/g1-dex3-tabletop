import pytest

from g1_dex3_tabletop.recording_benchmark import aggregate_results, summarize_tick_times


def test_tick_summary_reports_tail_gaps_and_conservative_missed_periods() -> None:
    timestamps = [0, 4_000_000, 8_000_000, 20_000_000, 24_000_000]

    summary = summarize_tick_times(timestamps, rate_hz=250.0)

    assert summary["maximum_gap_ms"] == pytest.approx(12.0)
    assert summary["gaps_over_10ms"] == 1
    assert summary["gaps_over_20ms"] == 0
    assert summary["conservative_missed_periods"] == 2


def test_aggregate_results_retains_worst_tail_and_sums_gap_counts() -> None:
    def result(scenario: str, maximum: float, over_10: int, camera_rate: float, write: float):
        return {
            "scenario": scenario,
            "timing": {
                "p99_gap_ms": maximum - 2,
                "p99_9_gap_ms": maximum - 1,
                "maximum_gap_ms": maximum,
                "gaps_over_10ms": over_10,
                "gaps_over_20ms": 0,
                "gaps_over_50ms": 0,
            },
            "camera_receive_rate_hz": camera_rate,
            "bag": {"write_mib_s": write},
        }

    aggregate = aggregate_results(
        [
            result("baseline", 8.0, 0, 15.0, 0.0),
            result("baseline", 12.0, 1, 14.8, 0.0),
        ]
    )["baseline"]

    assert aggregate["trials"] == 2
    assert aggregate["median_p99_gap_ms"] == pytest.approx(8.0)
    assert aggregate["worst_maximum_gap_ms"] == pytest.approx(12.0)
    assert aggregate["total_gaps_over_10ms"] == 1
    assert aggregate["minimum_camera_receive_rate_hz"] == pytest.approx(14.8)
