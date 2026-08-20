import numpy as np
import pytest

from g1_dex3_tabletop.mpc_command_buffer import (
    MPCCommandWindow,
    RollingMPCCommandBuffer,
    command_sequence_from_measured_plan,
)

PLAN_HASH = "a" * 64


def _window(
    *,
    generation: int = 0,
    source_s: float = 10.0,
    valid_from_s: float = 10.12,
    start_q: float = 0.0,
    end_q: float = 0.02,
    predecessor_sha256: str | None = None,
    feasible: bool = True,
    terminal: bool = False,
    route_progress: int = 0,
    predicted_dq_rad_s: float | None = None,
) -> MPCCommandWindow:
    times = np.linspace(0.0, 0.2, 51)
    values = np.linspace(start_q, end_q, len(times))[:, None] * np.ones((1, 7))
    velocity = (end_q - start_q) / times[-1] if predicted_dq_rad_s is None else predicted_dq_rad_s
    velocities = np.full_like(values, velocity)
    accelerations = np.zeros_like(values)
    return MPCCommandWindow(
        generation=generation,
        plan_sha256=PLAN_HASH,
        source_state_monotonic_s=source_s,
        valid_from_monotonic_s=valid_from_s,
        sample_time_s=tuple(times),
        command_q_rad=tuple(tuple(row) for row in values),
        predicted_q_rad=tuple(tuple(row) for row in values),
        predicted_dq_rad_s=tuple(tuple(row) for row in velocities),
        predicted_ddq_rad_s2=tuple(tuple(row) for row in accelerations),
        predecessor_sha256=predecessor_sha256,
        feasible=feasible,
        terminal=terminal,
        solve_time_s=0.004,
        diagnostics={"source": "test", "proposed_route_progress_index": route_progress},
    )


def _buffer() -> RollingMPCCommandBuffer:
    return RollingMPCCommandBuffer(
        plan_sha256=PLAN_HASH,
        maximum_velocity_rad_s=0.2,
        maximum_handoff_position_error_rad=0.08,
        maximum_handoff_velocity_error_rad_s=0.2,
        activation_lateness_s=0.01,
    )


def test_mpc_window_hash_round_trip_and_detects_mutation() -> None:
    source = _window()
    document = source.to_dict()
    restored = MPCCommandWindow.from_dict(document)

    assert restored == source
    assert restored.peak_velocity_rad_s() == pytest.approx(0.1)

    document["command_q_rad"][1][0] += 0.001
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        MPCCommandWindow.from_dict(document)


def test_measured_state_plan_preserves_tracking_offset_for_complete_window() -> None:
    # Exact first-window values retained from physical run
    # tabletop_20260819T002307Z. Joining CuRobo's measured-state future
    # directly to the active command creates a false high-speed first edge.
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
        [-0.2878479, 0.4716216, 0.1592374, -0.5352041, -0.5164650, 0.6197702, 0.4031002]
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
        np.max(np.abs(np.diff(command_samples, axis=0)) / np.diff(command_times)[:, None])
    )
    assert uncompensated_velocity == pytest.approx(0.1238230619)
    assert translated_velocity == pytest.approx(0.0299411294)
    assert translated_velocity < 0.1


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


def test_buffer_uses_absolute_time_and_holds_the_certified_endpoint() -> None:
    buffer = _buffer()
    window = _window()
    buffer.install(window, now_s=10.05, active_command_q_rad=np.zeros(7))

    assert np.allclose(
        buffer.command(now_s=10.10, measured_q_rad=np.zeros(7), measured_dq_rad_s=np.zeros(7)),
        0.0,
    )
    assert np.allclose(
        buffer.command(now_s=10.12, measured_q_rad=np.zeros(7), measured_dq_rad_s=np.full(7, 0.1)),
        0.0,
    )
    assert np.allclose(
        buffer.command(
            now_s=10.17, measured_q_rad=np.full(7, 0.005), measured_dq_rad_s=np.full(7, 0.1)
        ),
        0.005,
    )
    assert buffer.remaining_s(now_s=10.17) == pytest.approx(0.15)
    assert np.allclose(
        buffer.command(
            now_s=10.371, measured_q_rad=np.full(7, 0.02), measured_dq_rad_s=np.zeros(7)
        ),
        0.02,
    )


def test_next_window_can_restart_from_a_held_certified_endpoint() -> None:
    buffer = _buffer()
    first = _window(route_progress=6, predicted_dq_rad_s=0.0)
    buffer.install(first, now_s=10.05, active_command_q_rad=np.zeros(7))
    buffer.command(now_s=10.12, measured_q_rad=np.zeros(7), measured_dq_rad_s=np.zeros(7))
    buffer.command(now_s=10.40, measured_q_rad=np.full(7, 0.02), measured_dq_rad_s=np.zeros(7))

    boundary = buffer.handoff_boundary(
        now_s=10.40,
        minimum_lead_s=0.12,
        handoff_quantum_s=0.04,
    )
    assert boundary.valid_from_monotonic_s == pytest.approx(10.52)
    assert boundary.command_q_rad == pytest.approx((0.02,) * 7)
    assert boundary.predicted_q_rad == pytest.approx((0.02,) * 7)
    assert boundary.predicted_dq_rad_s == pytest.approx((0.0,) * 7)
    assert boundary.predecessor_sha256 == first.content_sha256
    assert boundary.committed_route_progress_index == 6

    second = _window(
        generation=1,
        source_s=10.41,
        valid_from_s=boundary.valid_from_monotonic_s,
        start_q=0.02,
        end_q=0.024,
        predecessor_sha256=boundary.predecessor_sha256,
        terminal=True,
        route_progress=10,
        predicted_dq_rad_s=0.0,
    )
    buffer.install(second, now_s=10.45, active_command_q_rad=np.full(7, 0.02))
    assert np.allclose(
        buffer.command(
            now_s=10.50,
            measured_q_rad=np.full(7, 0.02),
            measured_dq_rad_s=np.zeros(7),
        ),
        0.02,
    )
    assert np.allclose(
        buffer.command(
            now_s=10.52,
            measured_q_rad=np.full(7, 0.02),
            measured_dq_rad_s=np.zeros(7),
        ),
        0.02,
    )
    assert buffer.terminal


def test_next_window_starts_at_exact_certified_future_boundary() -> None:
    buffer = _buffer()
    first = _window(route_progress=6)
    buffer.install(first, now_s=10.05, active_command_q_rad=np.zeros(7))
    buffer.command(now_s=10.12, measured_q_rad=np.zeros(7), measured_dq_rad_s=np.full(7, 0.1))

    boundary = buffer.handoff_boundary(now_s=10.13, minimum_lead_s=0.12, handoff_quantum_s=0.04)
    assert boundary.valid_from_monotonic_s == pytest.approx(10.28)
    assert boundary.command_q_rad == pytest.approx((0.016,) * 7)
    assert boundary.predicted_q_rad == pytest.approx((0.016,) * 7)
    assert boundary.predicted_ddq_rad_s2 == pytest.approx((0.0,) * 7)
    assert boundary.predecessor_sha256 == first.content_sha256
    assert boundary.committed_route_progress_index == 6

    second = _window(
        generation=1,
        source_s=10.14,
        valid_from_s=boundary.valid_from_monotonic_s,
        start_q=boundary.command_q_rad[0],
        end_q=0.024,
        predecessor_sha256=boundary.predecessor_sha256,
        terminal=True,
        route_progress=10,
        predicted_dq_rad_s=boundary.predicted_dq_rad_s[0],
    )
    buffer.install(second, now_s=10.20, active_command_q_rad=np.zeros(7))
    assert np.allclose(
        buffer.command(
            now_s=10.28, measured_q_rad=np.full(7, 0.016), measured_dq_rad_s=np.full(7, 0.04)
        ),
        0.016,
    )
    assert buffer.terminal
    assert buffer.committed_route_progress_index == 10


def test_buffer_rejects_late_infeasible_discontinuous_and_fast_windows() -> None:
    with pytest.raises(ValueError, match="missed its absolute handoff"):
        _buffer().install(_window(), now_s=10.12, active_command_q_rad=np.zeros(7))
    with pytest.raises(ValueError, match="infeasible"):
        _buffer().install(_window(feasible=False), now_s=10.05, active_command_q_rad=np.zeros(7))
    with pytest.raises(ValueError, match="discontinuous"):
        _buffer().install(_window(start_q=0.01), now_s=10.05, active_command_q_rad=np.zeros(7))
    with pytest.raises(ValueError, match="velocity"):
        _buffer().install(_window(end_q=0.08), now_s=10.05, active_command_q_rad=np.zeros(7))


def test_buffer_rejects_wrong_predecessor_and_live_handoff_state() -> None:
    buffer = _buffer()
    first = _window()
    buffer.install(first, now_s=10.05, active_command_q_rad=np.zeros(7))
    with pytest.raises(RuntimeError, match="position error"):
        buffer.command(
            now_s=10.12, measured_q_rad=np.full(7, 0.09), measured_dq_rad_s=np.full(7, 0.1)
        )

    buffer = _buffer()
    buffer.install(first, now_s=10.05, active_command_q_rad=np.zeros(7))
    buffer.command(now_s=10.12, measured_q_rad=np.zeros(7), measured_dq_rad_s=np.full(7, 0.1))
    boundary = buffer.handoff_boundary(now_s=10.13, minimum_lead_s=0.12, handoff_quantum_s=0.04)
    wrong = _window(
        generation=1,
        source_s=10.14,
        valid_from_s=boundary.valid_from_monotonic_s,
        start_q=boundary.command_q_rad[0],
        end_q=0.024,
        predecessor_sha256="b" * 64,
    )
    with pytest.raises(ValueError, match="predecessor"):
        buffer.install(wrong, now_s=10.20, active_command_q_rad=np.zeros(7))


def test_buffer_can_finish_the_unchanged_active_horizon_after_planning_failure() -> None:
    buffer = _buffer()
    first = _window(route_progress=6)
    buffer.install(first, now_s=10.05, active_command_q_rad=np.zeros(7))
    buffer.command(
        now_s=10.12,
        measured_q_rad=np.zeros(7),
        measured_dq_rad_s=np.full(7, 0.1),
    )

    buffer.finish_active_horizon()

    assert buffer.stop_requested
    assert buffer.terminal
    assert not buffer.has_queued
    assert np.allclose(
        buffer.command(
            now_s=10.32,
            measured_q_rad=np.full(7, 0.02),
            measured_dq_rad_s=np.zeros(7),
        ),
        0.02,
    )
    with pytest.raises(RuntimeError, match="finishing its active safe horizon"):
        buffer.handoff_boundary(
            now_s=10.13,
            minimum_lead_s=0.04,
            handoff_quantum_s=0.04,
        )
