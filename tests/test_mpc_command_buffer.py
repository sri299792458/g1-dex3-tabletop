import numpy as np
import pytest

from g1_dex3_tabletop.mpc_command_buffer import (
    MPCCommandWindow,
    RollingMPCCommandBuffer,
    command_sequence_from_measured_plan,
)

PLAN_HASH = "a" * 64


def _window(*, generation: int = 0, feasible: bool = True, terminal: bool = False):
    return MPCCommandWindow(
        generation=generation,
        plan_sha256=PLAN_HASH,
        state_monotonic_s=10.0,
        sample_time_s=(0.0, 0.01, 0.02),
        command_q_rad=(
            (0.0,) * 7,
            (0.001,) * 7,
            (0.002,) * 7,
        ),
        feasible=feasible,
        terminal=terminal,
        solve_time_s=0.004,
        diagnostics={"source": "test"},
    )


def _buffer() -> RollingMPCCommandBuffer:
    return RollingMPCCommandBuffer(
        plan_sha256=PLAN_HASH,
        maximum_velocity_rad_s=0.1,
        maximum_state_age_s=0.1,
        maximum_window_gap_s=0.05,
    )


def test_mpc_window_hash_round_trip() -> None:
    source = _window()
    restored = MPCCommandWindow.from_dict(source.to_dict())
    assert restored == source
    assert restored.peak_velocity_rad_s() == pytest.approx(0.1)


def test_window_start_can_be_rebased_then_revalidated() -> None:
    source = _window()
    rebased = source.rebase_start(np.full(7, 0.00025))

    assert rebased.command_q_rad[0] == pytest.approx((0.00025,) * 7)
    assert rebased.command_q_rad[1:] == source.command_q_rad[1:]
    assert rebased.diagnostics["start_rebase_rad"] == pytest.approx(0.00025)
    assert rebased.diagnostics["worker_window_sha256"] == source.content_sha256
    _buffer().accept(rebased, now_s=10.01, active_command_q_rad=np.full(7, 0.00025))


def test_measured_state_plan_preserves_tracking_offset_for_complete_window() -> None:
    # Exact first-window values retained from physical run
    # tabletop_20260819T002307Z.  Joining CuRobo's measured-state future
    # directly to the active command produced 0.1238 rad/s at the elbow even
    # though the planned motion itself was slow.
    active = np.asarray(
        [
            -0.3004261851,
            0.4897119889,
            0.1692741811,
            -0.5552965999,
            -0.5192253584,
            0.6265174150,
            0.4055374563,
        ]
    )
    measured = np.asarray(
        [
            -0.2878479,
            0.4716216,
            0.1592374,
            -0.5352041,
            -0.5164650,
            0.6197702,
            0.4031002,
        ]
    )
    planned_first = np.asarray(
        [
            -0.2926384807,
            0.4718958030,
            0.1593182832,
            -0.5354849100,
            -0.5128390485,
            0.6201477647,
            0.4044316709,
        ]
    )
    planned_future = np.stack((planned_first, planned_first))
    future_times = np.asarray([0.16, 0.60])
    translated, offset = command_sequence_from_measured_plan(
        planned_future,
        measured_q_rad=measured,
        active_command_q_rad=active,
        future_sample_time_s=future_times,
    )

    np.testing.assert_allclose(offset, active - measured)
    np.testing.assert_allclose(translated, planned_future + offset[None, :])
    uncompensated_velocity = float(np.max(np.abs(planned_future[0] - active)) / 0.16)
    command_samples = np.concatenate((active[None, :], translated), axis=0)
    command_times = np.concatenate(([0.0], future_times))
    translated_velocity = float(
        np.max(
            np.abs(np.diff(command_samples, axis=0))
            / np.diff(command_times)[:, None]
        )
    )
    assert uncompensated_velocity == pytest.approx(0.1238230619)
    assert translated_velocity == pytest.approx(0.0299411294)
    assert translated_velocity < 0.1

    window = MPCCommandWindow(
        generation=0,
        plan_sha256=PLAN_HASH,
        state_monotonic_s=10.0,
        sample_time_s=tuple(command_times),
        command_q_rad=tuple(tuple(row) for row in command_samples),
        feasible=True,
        terminal=False,
        solve_time_s=0.044,
        diagnostics={"command_tracking_offset_rad": offset.tolist()},
    )
    _buffer().accept(window, now_s=10.01, active_command_q_rad=active)


def test_measured_state_plan_translation_validates_inputs() -> None:
    with pytest.raises(ValueError, match="N x 7"):
        command_sequence_from_measured_plan(
            np.zeros((1, 6)),
            measured_q_rad=np.zeros(7),
            active_command_q_rad=np.zeros(7),
            future_sample_time_s=[0.1],
        )
    with pytest.raises(ValueError, match="finite"):
        command_sequence_from_measured_plan(
            np.full((1, 7), np.nan),
            measured_q_rad=np.zeros(7),
            active_command_q_rad=np.zeros(7),
            future_sample_time_s=[0.1],
        )
    with pytest.raises(ValueError, match="sample times"):
        command_sequence_from_measured_plan(
            np.zeros((2, 7)),
            measured_q_rad=np.zeros(7),
            active_command_q_rad=np.zeros(7),
            future_sample_time_s=[0.1, 0.1],
        )


def test_buffer_interpolates_and_holds_one_accepted_window() -> None:
    buffer = _buffer()
    buffer.accept(_window(), now_s=10.01, active_command_q_rad=np.zeros(7))

    assert np.allclose(buffer.command(now_s=10.015), 0.0005)
    assert np.allclose(buffer.command(now_s=10.04), 0.002)
    assert buffer.remaining_s(now_s=10.015) == pytest.approx(0.015)
    with pytest.raises(RuntimeError, match="expired"):
        buffer.command(now_s=10.081)


def test_buffer_rejects_infeasible_stale_discontinuous_and_fast_windows() -> None:
    buffer = _buffer()
    with pytest.raises(ValueError, match="infeasible"):
        buffer.accept(_window(feasible=False), now_s=10.01, active_command_q_rad=np.zeros(7))
    with pytest.raises(ValueError, match="source state age"):
        buffer.accept(_window(), now_s=10.101, active_command_q_rad=np.zeros(7))
    with pytest.raises(ValueError, match="active command"):
        buffer.accept(_window(), now_s=10.01, active_command_q_rad=np.full(7, 0.01))

    fast = MPCCommandWindow(
        generation=0,
        plan_sha256=PLAN_HASH,
        state_monotonic_s=10.0,
        sample_time_s=(0.0, 0.01),
        command_q_rad=((0.0,) * 7, (0.002,) * 7),
        feasible=True,
        terminal=False,
        solve_time_s=0.004,
        diagnostics={},
    )
    with pytest.raises(ValueError, match="velocity"):
        buffer.accept(fast, now_s=10.01, active_command_q_rad=np.zeros(7))


def test_buffer_rejects_replayed_generation() -> None:
    buffer = _buffer()
    buffer.accept(_window(), now_s=10.01, active_command_q_rad=np.zeros(7))
    with pytest.raises(ValueError, match="generation"):
        buffer.accept(_window(), now_s=10.02, active_command_q_rad=np.zeros(7))
