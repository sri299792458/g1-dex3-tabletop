import numpy as np
import pytest

from g1_dex3_tabletop.mpc_command_buffer import MPCCommandWindow, RollingMPCCommandBuffer

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
