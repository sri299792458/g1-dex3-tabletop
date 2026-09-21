"""Exercise one complete standing lifecycle with the real arm executor."""

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from test_bilateral_calibration import bilateral_route_artifacts

from g1_aprilcube_calibration.clock import ManualClock
from g1_aprilcube_calibration.executor_state_machine import ExecutorConfig, ExecutorState
from g1_aprilcube_calibration.transports.fake import FakeArmTransport
from g1_dex3_tabletop.calibration import (
    BilateralCollectionOrchestrator,
    PreparedBilateralCollection,
)
from g1_dex3_tabletop.calibration.control import StandingCalibrationControl
from g1_dex3_tabletop.calibration.hand_posture import full_close_reference

CLOSED = {side: tuple(q) for side, q in full_close_reference()["reference_q_rad"].items()}

from g1_dex3_tabletop.calibration.route import BilateralCalibrationRoute
from g1_dex3_tabletop.planning.contracts import (
    PlannedTrajectory,
    RobotSnapshot,
)


class Harness:
    def __init__(self, monkeypatch, *, initial_q=None, model=None, config=None, gravity=None):
        self.clock = ManualClock(1.0)
        self.events = []
        self.drivers = []
        self.failure = None
        self.initial_q = np.zeros(29) if initial_q is None else np.array(initial_q).copy()
        if initial_q is None:
            self.initial_q[26] = -0.00022770464420318604
            self.initial_q[15] = 0.0005
        outer = self

        class Guard:
            armed = False
            terminal_action = None
            last_pulse = None

            def start(self):
                self.armed = True
                self.last_pulse = outer.clock.monotonic()
                outer.events.append("guard.start")

            def pulse(self):
                assert self.armed
                now = outer.clock.monotonic()
                assert now - self.last_pulse < 0.5
                self.last_pulse = now

            def disarm(self):
                self.armed = False
                self.terminal_action = "disarmed"
                outer.events.append("guard.disarm")

            def damp(self, reason):
                self.armed = False
                self.terminal_action = "damped"
                outer.events.append("guard.damp")

        self.guard = Guard()

        class Transport(FakeArmTransport):
            @property
            def command_count(self):
                return len(self.commands)

            def close(self):
                if not self.closed:
                    assert not self.commands or self.commands[-1].weight == 0
                    outer.events.append("transport.close")
                super().close()

            def close_after_external_takeover(self):
                assert outer.guard.terminal_action == "damped"
                outer.events.append("transport.close_after_recovery")
                super().close_after_external_takeover()

        self.transport = Transport(clock=self.clock, initial_full_q=self.initial_q)

        class Hands:
            command_count = 0
            timed_out = False
            closed = False

            def __init__(self):
                self.pair = SimpleNamespace(
                    left=SimpleNamespace(position=CLOSED["left"]),
                    right=SimpleNamespace(position=CLOSED["right"]),
                )
                self.observer = SimpleNamespace(observe=lambda: self.pair, close=lambda: None)
                self.postures = []

            def acquire_measured_hold(self, *, safety_heartbeat):
                safety_heartbeat()
                self.command_count += 1
                outer.events.append("hands.acquire")
                return self.pair

            def maintain_active_posture(self):
                assert not self.timed_out
                self.command_count += 1

            def command_posture(self, **kwargs):
                kwargs["safety_heartbeat"]()
                self.postures.append(kwargs)
                self.pair = SimpleNamespace(
                    **{
                        side: SimpleNamespace(position=kwargs[f"{side}_acceptance_q_rad"])
                        for side in ("left", "right")
                    }
                )
                return self.pair

            def timeout(self):
                assert not any(driver.alive for driver in outer.drivers)
                outer.events.append("hands.timeout")
                self.timed_out = True
                if outer.failure == "timeout":
                    self.timed_out = False
                    raise RuntimeError("injected hand timeout failure")

            def close(self):
                assert not self.command_count or self.timed_out
                self.closed = True

            def close_after_external_timeout(self):
                assert outer.guard.terminal_action == "damped"
                self.timed_out = True
                self.close()

        self.hands = Hands()

        def transport_factory(config, *, observer):
            assert not self.guard.armed
            self.clock.advance(0.2)
            self.events.append("arm.publisher")
            return self.transport

        def hand_factory(config, *, observer):
            assert not self.guard.armed
            self.clock.advance(0.4)
            self.events.append("hand.publishers")
            return self.hands

        class Driver:
            def __init__(self, executor, *, rate_hz, safety_heartbeat):
                self.executor = executor
                self.safety_heartbeat = safety_heartbeat
                self.alive = False
                outer.drivers.append(self)

            def start(self):
                outer.events.append("driver.start")
                self.alive = True

            def check(self):
                if self.executor.state is ExecutorState.FAULT:
                    raise RuntimeError(self.executor.fault_reason)

            def close(self):
                if self.alive:
                    outer.events.append("driver.close")
                self.alive = False

            def step(self, duration):
                assert self.alive
                outer.transport.step(duration)
                if (
                    outer.failure == "acquisition"
                    and self.executor.state is ExecutorState.ACQUIRING
                ):
                    outer.transport.position[25] += 0.0555
                self.executor.tick()
                self.check()
                self.safety_heartbeat()

        def sleep(duration):
            count = max(1, round(duration / 0.004))
            for _ in range(count):
                next(driver for driver in self.drivers if driver.alive).step(duration / count)

        def wait_ready(executor, driver, *, label, timeout_s):
            deadline = self.clock.monotonic() + timeout_s
            while executor.state is not ExecutorState.READY:
                assert self.clock.monotonic() < deadline, (label, executor.motion_diagnostic())
                sleep(0.004)

        self.control = StandingCalibrationControl(
            model=model or SimpleNamespace(name="g1-test", sha256="3" * 64),
            transport_config=None,
            hand_config=SimpleNamespace(
                posture_position_tolerance_rad=0.08, posture_position_spread_rad=0.01
            ),
            executor_config=config or ExecutorConfig(),
            rate_hz=250,
            observer=None,
            hand_observer=self.hands.observer,
            guard=self.guard,
            gravity=gravity,
            transport_factory=transport_factory,
            hand_factory=hand_factory,
            driver_factory=Driver,
            clock=self.clock,
            wait_ready=wait_ready,
            sleep=sleep,
        )

    def acquire(self, *, preflight_snapshot=None):
        preparation = make_adapter(self.initial_q, self.initial_q).preparation
        self.control.acquire(
            preparation=preparation,
            preflight_snapshot=preflight_snapshot
            or RobotSnapshot(tuple(self.initial_q), CLOSED["left"], CLOSED["right"]),
        )

    def request_payload(self, operation, *, payload, control_check, timeout_s):
        control_check()
        if operation == "validate-bilateral-shoulder-return":
            return {
                "payload": {
                    "snapshot": payload["snapshot"],
                    "preparation_sha256": payload["preparation"]["content_sha256"],
                    "certificate": {
                        "passed": True,
                        "hard_clearance_m": 0.005,
                        "minimum_clearance_m": 0.005,
                        "minimum_margin_to_required_clearance_m": 0.0,
                    },
                }
            }
        raise AssertionError(f"unexpected planner operation: {operation}")


def make_adapter(initial, anchor):
    ready = np.array(initial)[15:29]
    anchor = np.asarray(anchor)[15:29]
    right_clearance = ready.copy()
    right_clearance[8] -= 0.08
    dual_clearance = right_clearance.copy()
    dual_clearance[1] += 0.08
    right_anchor = dual_clearance.copy()
    right_anchor[7:] = anchor[7:]

    def edge(arm, source, target, start, end):
        active = slice(0, 7) if arm == "left" else slice(7, 14)
        values = (tuple(start[active]), tuple(end[active]))
        return PlannedTrajectory(source, target, (0.0, 2.0), values, values, 0.0)

    return SimpleNamespace(
        content_sha256="b" * 64,
        anchor_q14_rad=tuple(anchor),
        preparation=SimpleNamespace(
            dual_clearance_q14_rad=tuple(dual_clearance),
            content_sha256="c" * 64,
            to_dict=lambda: {"content_sha256": "c" * 64},
            right_outbound=edge(
                "right", "__handoff__", "right_shoulder_clearance", ready, right_clearance
            ),
            left_outbound=edge(
                "left",
                "right_shoulder_clearance",
                "dual_shoulder_clearance",
                right_clearance,
                dual_clearance,
            ),
            left_return=edge(
                "left",
                "dual_shoulder_clearance",
                "right_shoulder_clearance",
                dual_clearance,
                right_clearance,
            ),
            right_return=edge(
                "right", "right_shoulder_clearance", "__handoff__", right_clearance, ready
            ),
        ),
        right_anchor_outbound=edge(
            "right",
            "dual_shoulder_clearance",
            "right_anchor_preparation",
            dual_clearance,
            right_anchor,
        ),
        left_anchor_outbound=edge(
            "left", "right_anchor_preparation", "__handoff__", right_anchor, anchor
        ),
        left_anchor_return=edge(
            "left", "__handoff__", "right_anchor_preparation", anchor, right_anchor
        ),
        right_anchor_return=edge(
            "right",
            "right_anchor_preparation",
            "dual_shoulder_clearance",
            right_anchor,
            dual_clearance,
        ),
    )


@pytest.mark.parametrize("stop_after", [None, 1, 2, 3])
@pytest.mark.parametrize("recorded_tip", [CLOSED["left"][-1], -1.7745474576950073])
def test_complete_collection_and_graceful_return_reuse_commissioned_shoulders(
    monkeypatch, tmp_path, stop_after, recorded_tip
):
    design, plan = bilateral_route_artifacts()
    prepared = PreparedBilateralCollection.from_artifacts(design, plan)

    def unexpected_hash(_self):
        raise AssertionError("large calibration artifact hashed during robot ownership")

    monkeypatch.setattr(type(plan), "content_sha256", property(unexpected_hash))
    monkeypatch.setattr(type(design), "content_sha256", property(unexpected_hash))
    harness = Harness(monkeypatch)
    harness.hands.pair.left.position = (*harness.hands.pair.left.position[:6], recorded_tip)
    initial_hands = harness.hands.pair
    harness.acquire(
        preflight_snapshot=RobotSnapshot(
            tuple(harness.initial_q), initial_hands.left.position, initial_hands.right.position
        )
    )
    control = harness.control
    control.move_to_shoulder_clearance()
    anchor = design.waypoint_joint_positions_rad[design.schedule[0].candidate_id]
    adapter = make_adapter(harness.initial_q, anchor)
    route = BilateralCalibrationRoute(
        control=control,
        adapter=adapter,
        plan=plan,
        planner=harness,
        artifact_directory=tmp_path,
    )
    control.adopt_collection_control(plan_sha256="b" * 64)
    executor = control.executor
    route.move_to_anchor(validated_reference_state=executor.observe_state())
    pose_sets = prepared.pose_sets
    control.install_collection(
        pose_set=pose_sets[plan.transitions[0].arm],
        anchor_q14=adapter.anchor_q14_rad,
        plan_sha256=prepared.plan_sha256,
    )
    captures = []
    result = BilateralCollectionOrchestrator(
        executor=executor,
        prepared=prepared,
        store=SimpleNamespace(
            append_capture=lambda **kw: captures.append(kw), finalize=lambda: None
        ),
        frame_source=SimpleNamespace(capture_burst=lambda **kw: (object(),)),
        wait_until_ready=lambda: control.wait_ready(label="capture route"),
        graceful_stop_requested=lambda: stop_after is not None and len(captures) >= stop_after,
    ).run()
    assert result.stopped_early == (stop_after is not None)
    route.return_to_ready()
    control.release()
    assert control.close() == []
    assert control.executor is control.clearance_executor
    assert len(harness.drivers) == 3
    assert harness.events[:3] == ["arm.publisher", "hand.publishers", "guard.start"]
    assert harness.events[-3:] == ["driver.close", "hands.timeout", "guard.disarm"]
    assert harness.transport.commands[-1].weight == 0
    np.testing.assert_array_equal(harness.transport.commands[-1].q14, harness.initial_q[15:29])
    assert harness.hands.postures == []
    assert all(
        command.weight == 1
        for command in harness.transport.commands
        if 2.7 < command.issued_monotonic_s < harness.clock.monotonic() - 1.1
    )
    count = harness.transport.command_count
    control.close()
    assert harness.transport.command_count == count
    with pytest.raises(RuntimeError, match="control is closed"):
        route.return_to_ready()
    assert harness.transport.command_count == count
    with pytest.raises(RuntimeError, match="only be acquired once"):
        harness.acquire()


def test_planning_reads_fresh_fingers_and_rejects_hold_drift(monkeypatch):
    harness = Harness(monkeypatch)
    harness.acquire()
    harness.control.move_to_shoulder_clearance()
    harness.control.adopt_collection_control(plan_sha256="a" * 64)
    original = harness.hands.pair.left.position
    try:
        harness.hands.pair = SimpleNamespace(
            left=SimpleNamespace(position=tuple(q + 0.005 for q in original)),
            right=harness.hands.pair.right,
        )
        _, snapshot = harness.control.planning_boundary()
        assert snapshot.left_dex3_q_rad == harness.hands.pair.left.position
        assert snapshot.left_dex3_q_rad != original
        harness.hands.pair.left.position = tuple(q + 0.02 for q in original)
        with pytest.raises(ValueError, match="certified measured hold"):
            harness.control.planning_boundary()
        assert harness.hands.postures == []
    finally:
        harness.control.close()


@pytest.mark.parametrize("failure", ["acquisition", "hold_drift", "timeout"])
def test_failure_stops_driver_and_recovers_before_closing_transport(
    monkeypatch, tmp_path, failure
):
    harness = Harness(monkeypatch)
    harness.failure = failure
    control = harness.control
    try:
        if failure == "acquisition":
            with pytest.raises(RuntimeError, match="arm position changed"):
                harness.acquire()
        else:
            harness.acquire()
            if failure == "hold_drift":
                harness.hands.pair = SimpleNamespace(
                    left=SimpleNamespace(position=tuple(q + 0.02 for q in CLOSED["left"])),
                    right=harness.hands.pair.right,
                )
                with pytest.raises(ValueError, match="certified measured hold"):
                    control.check()
                assert harness.hands.postures == []
    finally:
        errors = control.close()
    assert bool(errors) == (failure == "timeout")
    assert harness.events[-4:] == [
        "driver.close",
        "hands.timeout",
        "guard.damp",
        "transport.close_after_recovery",
    ]
    assert not harness.drivers[0].alive
    assert harness.transport.closed and harness.hands.closed
    assert control.close() == errors
    if failure == "acquisition":
        assert "arm position changed" in control.describe()["fault_reason"]


def test_failed_finger_timeout_requires_recovery_even_before_first_arm_command(monkeypatch):
    harness = Harness(monkeypatch)
    harness.failure = "timeout"

    def interrupted(**kwargs):
        raise KeyboardInterrupt("interrupted after finger hold")

    monkeypatch.setattr(
        "g1_dex3_tabletop.calibration.control.DualArmClearanceExecutor", interrupted
    )
    try:
        with pytest.raises(KeyboardInterrupt, match="after finger hold"):
            harness.acquire()
    finally:
        errors = harness.control.close()
    assert harness.transport.command_count == 0
    assert len(errors) == 1 and "Dex3 timeout" in errors[0]
    assert harness.guard.terminal_action == "damped"
    assert harness.hands.closed and harness.transport.closed
    assert "guard.disarm" not in harness.events


def test_return_rejects_an_incorrect_opposite_arm_before_any_motion(monkeypatch, tmp_path):
    harness = Harness(monkeypatch)
    harness.acquire()
    harness.control.move_to_shoulder_clearance()
    harness.control.adopt_collection_control(plan_sha256="a" * 64)
    anchor = harness.initial_q.copy()
    anchor[22] += 0.001  # Left start matches, right hold does not.
    route = BilateralCalibrationRoute(
        control=harness.control,
        adapter=make_adapter(harness.initial_q, anchor),
        plan=None,
        planner=None,
        artifact_directory=tmp_path,
    )
    count = harness.transport.command_count
    try:
        with pytest.raises(ValueError, match="both arms at the fixed anchor"):
            route.return_to_ready()
        assert harness.transport.command_count == count
    finally:
        assert harness.control.close() == []


def test_recorded_body_transient_finishes_before_loaded_planning(monkeypatch):
    """Recorded body feedback with ideal arm tracking; not a dynamics simulation."""
    data = json.loads(
        (Path(__file__).parent / "fixtures/standing_takeover_210712.json").read_text()
    )
    times = np.array([s["time_s"] for s in data["samples"]])
    body = np.array([s["body_q15"] for s in data["samples"]])
    harness = Harness(monkeypatch, initial_q=data["initial_q29"])
    step = harness.transport.step

    def recorded_step(duration):
        step(duration)
        if harness.transport.commands:
            elapsed = harness.clock.monotonic() - harness.transport.commands[0].issued_monotonic_s
            harness.transport.position[:15] = [
                np.interp(elapsed, times, body[:, i]) for i in range(15)
            ]

    monkeypatch.setattr(harness.transport, "step", recorded_step)
    try:
        harness.acquire()
        early = harness.transport.observe()
        ramp_end = harness.clock.monotonic()
        harness.control.move_to_shoulder_clearance()
        harness.control.adopt_collection_control(plan_sha256="a" * 64)
        loaded, snapshot = harness.control.planning_boundary()
        assert abs(loaded.position[14] - early.position[14]) > 0.13
        assert loaded.position[14] == pytest.approx(body[-1, 14], abs=0.001)
        assert snapshot.measured_q29_rad[14] == loaded.position[14]
        q14 = harness.control.clearance_plan.target("dual_shoulder_clearance")
        np.testing.assert_array_equal(snapshot.measured_q29_rad[15:29], q14)
        commands = harness.transport.commands
        changed = next(c for c in commands if abs(c.q14[8] - commands[0].q14[8]) > 1e-6)
        assert changed.issued_monotonic_s - ramp_end <= 0.008
        assert all(c.weight == 1 for c in commands if c.issued_monotonic_s > ramp_end)
        assert [e["phase"] for e in harness.control.events][-3:] == [
            "right_shoulder_clearance",
            "dual_shoulder_clearance",
            "loaded_clearance_hold",
        ]
    finally:
        assert harness.control.close() == []


def test_commissioned_preparation_accepts_small_measured_start_change(monkeypatch):
    harness = Harness(monkeypatch)
    harness.transport.position[26] += 0.00022770464420318604
    try:
        harness.acquire()
        seed = harness.transport.commands[0].q14
        assert seed[11] == harness.transport.position[26]
        harness.control.move_to_shoulder_clearance()
        harness.control.adopt_collection_control(plan_sha256="a" * 64)
        np.testing.assert_array_equal(
            harness.control.executor.dual_arm_command_q,
            harness.control.clearance_plan.target("dual_shoulder_clearance"),
        )
    finally:
        assert harness.control.close() == []


@pytest.mark.parametrize(
    "failure", ["snapshot", "hash", "negative_clearance", "nan_margin", "body_drift"]
)
def test_shoulder_return_requires_valid_geometry_and_unchanged_body(
    monkeypatch, tmp_path, failure
):
    harness = Harness(monkeypatch)
    control = harness.control
    try:
        harness.acquire()
        control.move_to_shoulder_clearance()
        control.adopt_collection_control(plan_sha256="a" * 64)

        def plan(operation, **kwargs):
            event = harness.request_payload(operation, **kwargs)
            result = event["payload"]
            if failure == "snapshot":
                result["snapshot"] = {}
            elif failure == "hash":
                result["preparation_sha256"] = "d" * 64
            elif failure == "negative_clearance":
                result["certificate"]["minimum_clearance_m"] = -0.00367
            elif failure == "nan_margin":
                result["certificate"]["minimum_margin_to_required_clearance_m"] = float("nan")
            else:
                harness.transport.position[14] += 0.02
            return event

        count = harness.transport.command_count
        with pytest.raises(ValueError):
            reference = control.validate_shoulder_return(
                planner=SimpleNamespace(request_payload=plan),
                joint_position_offsets_rad={},
                artifact_directory=tmp_path,
            )
            control.return_shoulders(validated_reference_state=reference)
        assert harness.transport.command_count == count
        assert len(harness.drivers) == 2
        assert not harness.drivers[0].alive and harness.drivers[1].alive
    finally:
        assert control.close() == []


def test_nonstraight_preparation_is_rejected_before_publishers(monkeypatch):
    from dataclasses import replace

    harness = Harness(monkeypatch)
    prep = make_adapter(harness.initial_q, harness.initial_q).preparation
    edge = prep.right_outbound
    start, end = np.asarray(edge.command_q_rad)
    middle = (start + end) / 2
    middle[4] += 0.01
    samples = tuple(tuple(q) for q in (start, middle, end))
    prep.right_outbound = replace(
        edge, sample_time_s=(0, 1, 2), command_q_rad=samples, model_q_rad=samples
    )
    with pytest.raises(ValueError, match="only shoulder roll"):
        harness.control.acquire(
            preparation=prep,
            preflight_snapshot=RobotSnapshot(
                tuple(harness.initial_q), CLOSED["left"], CLOSED["right"]
            ),
        )
    assert harness.events == [] and harness.transport.command_count == 0


def test_hand_change_after_preflight_is_rejected_before_arm_commands(monkeypatch):
    harness = Harness(monkeypatch)
    harness.hands.pair.left.position = tuple(q + 0.02 for q in CLOSED["left"])
    try:
        with pytest.raises(ValueError, match="Dex3 state changed"):
            harness.acquire()
        assert harness.transport.command_count == 0 and not harness.drivers
    finally:
        assert harness.control.close() == []


@pytest.mark.parametrize("arm", ["left", "right"])
def test_oversized_saved_shoulder_route_is_rejected_before_publishers(monkeypatch, arm):
    from dataclasses import replace

    harness = Harness(monkeypatch)
    prep = make_adapter(harness.initial_q, harness.initial_q).preparation
    edge = getattr(prep, f"{arm}_outbound")
    samples = np.asarray(edge.command_q_rad).copy()
    samples[-1, 1] = samples[0, 1] + (0.16 if arm == "left" else -0.16)
    samples = tuple(tuple(q) for q in samples)
    setattr(prep, f"{arm}_outbound", replace(edge, command_q_rad=samples, model_q_rad=samples))
    with pytest.raises(ValueError, match="exceeds 0.14 rad"):
        harness.control.acquire(
            preparation=prep,
            preflight_snapshot=RobotSnapshot(
                tuple(harness.initial_q), CLOSED["left"], CLOSED["right"]
            ),
        )
    assert harness.events == [] and harness.transport.command_count == 0


@pytest.mark.parametrize("side", ["left", "right"])
def test_open_hand_rejects_before_any_publisher(monkeypatch, side):
    harness = Harness(monkeypatch)
    getattr(harness.hands.pair, side).position = (0.0,) * 7
    try:
        with pytest.raises(ValueError, match="Close both Dex3 hands fully"):
            harness.acquire()
        assert harness.events == []
        assert harness.transport.command_count == 0
        assert harness.hands.command_count == 0
    finally:
        assert harness.control.close() == []
