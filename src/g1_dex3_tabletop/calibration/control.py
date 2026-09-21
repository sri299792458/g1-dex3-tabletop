"""One standing ownership lifecycle for the bilateral calibration collection.

The commissioned shoulder and pose executors share one transport and one
ownership ramp, with the same full-weight handoffs as the earlier collector.
"""

from __future__ import annotations

import time

import numpy as np

from g1_aprilcube_calibration.clock import SystemClock
from g1_aprilcube_calibration.executor_driver import (
    ExecutorControlDriver,
    SynchronizedPoseExecutor,
)
from g1_aprilcube_calibration.executor_state_machine import ExecutorState, PoseExecutor
from g1_aprilcube_calibration.pose_schema import HANDOFF_POSE_ID, PoseSet
from g1_aprilcube_calibration.transports.unitree_arm_sdk import UnitreeArmSDKTransport
from g1_aprilcube_calibration.transports.unitree_dex3 import UnitreeDex3PostureController
from g1_dex3_tabletop.control_boundary import (
    command_bound_snapshot,
    install_plan_at_current_boundary,
)
from g1_dex3_tabletop.hardware_tabletop import _wait_ready
from g1_dex3_tabletop.planning.contracts import PlannedTrajectory, atomic_write_json

from .hand_posture import check_full_close, full_close_reference
from .shoulder_clearance import (
    DUAL_CLEARANCE_POSE_ID,
    RIGHT_CLEARANCE_POSE_ID,
    DualArmClearanceExecutor,
    ShoulderClearancePlan,
)


class StandingCalibrationControl:
    """Own publishers, watchdog, fingers, and exclusive active arm control."""

    def __init__(
        self,
        *,
        model,
        transport_config,
        hand_config,
        executor_config,
        rate_hz,
        observer,
        hand_observer,
        guard,
        gravity,
        transport_factory=UnitreeArmSDKTransport,
        hand_factory=UnitreeDex3PostureController,
        driver_factory=ExecutorControlDriver,
        clock=None,
        wait_ready=None,
        sleep=time.sleep,
    ):
        # Construction is inert. The caller transfers its read-only observers
        # here; acquire() is the only publisher-creation boundary after SPACE.
        self.model = model
        self.transport_config = transport_config
        self.hand_config = hand_config
        self.config = executor_config
        self.rate_hz = rate_hz
        self.observer = observer
        self.hand_observer = hand_observer
        self.guard = guard
        self.gravity = gravity
        self.clock = clock or SystemClock()
        self._wait_ready = wait_ready or _wait_ready
        self._sleep = sleep
        self._transport_factory = transport_factory
        self._hand_factory = hand_factory
        self._driver_factory = driver_factory
        self.transport = None
        self.hands = None
        self.executor = None
        self.driver = None
        self.held_hands = None
        self._geometry_hand_snapshot = None
        self._full_close_reference = full_close_reference()
        self.initial_command_q14 = None
        self.clearance_executor = None
        self.clearance_plan = None
        self.phase = "observing"
        self.events = []
        self.released = False
        self._closed = False
        self._cleanup_errors = []

    def _phase(self, name: str) -> None:
        self.phase = name
        self.events.append({"phase": name, "monotonic_s": self.clock.monotonic()})

    @property
    def commands_robot(self) -> bool:
        return bool(
            (self.transport is not None and self.transport.command_count)
            or (self.hands is not None and self.hands.command_count)
        )

    def acquire(self, *, preparation, preflight_snapshot) -> None:
        if self.phase != "observing" or self._closed:
            raise RuntimeError("standing calibration ownership can only be acquired once")
        self.clearance_plan = ShoulderClearancePlan(preparation)
        if not np.allclose(
            self.clearance_plan.source_q14,
            preflight_snapshot.measured_q29_rad[15:29],
            atol=1e-9,
            rtol=0,
        ):
            raise ValueError("shoulder preparation source differs from its preflight snapshot")
        self.verify_finger_hold(self.hand_observer.observe(), reference=preflight_snapshot)
        self._geometry_hand_snapshot = preflight_snapshot
        self._phase("initializing_publishers")
        self.transport = self._transport_factory(self.transport_config, observer=self.observer)
        self.observer = None
        self.hands = self._hand_factory(self.hand_config, observer=self.hand_observer)
        self.hand_observer = None
        # Three SDK publisher initializations each sleep 0.2 s. Arm the 0.5 s
        # watchdog only when they have finished, before the first motor command.
        self._phase("arming_watchdog")
        self.guard.start()
        self._phase("acquiring_finger_hold")
        self.held_hands = self.hands.acquire_measured_hold(safety_heartbeat=self.guard.pulse)
        self.verify_finger_hold(self.held_hands, reference=preflight_snapshot)
        self.executor = SynchronizedPoseExecutor(
            DualArmClearanceExecutor(
                transport=self.transport,
                clock=self.clock,
                plan=self.clearance_plan,
                config=self.config,
                gravity_feedforward=self.gravity,
            )
        )
        self.clearance_executor = self.executor
        self.driver = self._driver_factory(
            self.executor,
            rate_hz=self.rate_hz,
            safety_heartbeat=self._heartbeat,
        )
        self._phase("acquiring_arms")
        self.driver.start()
        self.executor.acquire(operator_confirmed=True)
        self.wait_ready(label="standing ownership", timeout_s=self.config.acquisition_ramp_s + 5.0)
        self.initial_command_q14 = np.asarray(self.clearance_plan.source_q14)
        self._phase("owned_hold")

    def move_to_shoulder_clearance(self) -> None:
        """The old collector moves out immediately after the ownership ramp."""
        if self.executor is not self.clearance_executor or self.phase != "owned_hold":
            raise RuntimeError("shoulder preparation requires the acquired clearance controller")
        for pose_id in (RIGHT_CLEARANCE_POSE_ID, DUAL_CLEARANCE_POSE_ID):
            self._phase(pose_id)
            self.executor.start_pose(pose_id, operator_confirmed=True)
            self.wait_ready(label=pose_id)

    def adopt_collection_control(self, *, plan_sha256: str) -> None:
        """Reuse the old full-weight handoff after shoulder/finger preparation."""
        self.check()
        if (
            self.executor is not self.clearance_executor
            or self.executor.current_pose_id != DUAL_CLEARANCE_POSE_ID
        ):
            raise RuntimeError("collection handoff requires completed shoulder preparation")
        command = self.clearance_plan.target(DUAL_CLEARANCE_POSE_ID)
        next_executor = SynchronizedPoseExecutor(
            PoseExecutor(
                transport=self.transport,
                clock=self.clock,
                pose_set=PoseSet(
                    robot_model=self.model.name,
                    mode_machine=5,
                    urdf_sha256=self.model.sha256,
                    calibration_arm="right",
                ),
                handoff_q=command[7:],
                hold_q=command[:7],
                approved_validation_report_sha256=plan_sha256,
                config=self.config,
                gravity_feedforward=self.gravity,
            )
        )
        self.driver.close()
        self.driver.check()
        self.guard.pulse()
        self.executor = next_executor
        self.executor.adopt_owned_control(previous_command_q14=command)
        self.driver = self._driver_factory(
            self.executor,
            rate_hz=self.rate_hz,
            safety_heartbeat=self._heartbeat,
        )
        self.driver.start()
        self._phase("loaded_clearance_hold")

    def validate_shoulder_return(self, *, planner, joint_position_offsets_rad, artifact_directory):
        reference, snapshot = self.planning_boundary()
        preparation = self.clearance_plan.preparation
        payload = {
            "snapshot": snapshot.to_dict(),
            "preparation": preparation.to_dict(),
            "joint_position_offsets_rad": joint_position_offsets_rad,
        }
        atomic_write_json(artifact_directory / "shoulder_return_request.json", payload)
        event = planner.request_payload(
            "validate-bilateral-shoulder-return",
            payload=payload,
            control_check=self.check,
            timeout_s=180.0,
        )
        result = event["payload"]
        atomic_write_json(artifact_directory / "shoulder_return_result.json", result)
        certificate = result.get("certificate", {})
        if (
            result.get("snapshot") != snapshot.to_dict()
            or result.get("preparation_sha256") != preparation.content_sha256
            or certificate.get("passed") is not True
            or certificate.get("hard_clearance_m") != 0.005
            or not float(certificate.get("minimum_clearance_m", -1.0)) >= -1e-6
            or not float(certificate.get("minimum_margin_to_required_clearance_m", -1.0)) >= -1e-6
        ):
            raise ValueError("shoulder return lacks a passing snapshot-bound certificate")
        self._return_hand_snapshot = snapshot
        return reference

    def return_shoulders(self, *, validated_reference_state) -> None:
        """Resume the old shoulder controller at its identical held command."""
        self.check()
        drift = float(
            np.max(
                np.abs(self.executor.observe_state().position - validated_reference_state.position)
            )
        )
        if drift > self.config.settled_position_spread_rad:
            raise ValueError(f"body changed after shoulder-return validation by {drift:.4f}rad")
        self.verify_finger_hold(
            self.hands.observer.observe(), reference=self._return_hand_snapshot
        )
        command = self.clearance_plan.target(DUAL_CLEARANCE_POSE_ID)
        if not np.allclose(self.executor.dual_arm_command_q, command, atol=1e-9, rtol=0):
            raise ValueError("shoulder return requires both exact clearance commands")
        self.driver.close()
        self.driver.check()
        self.guard.pulse()
        self.executor = self.clearance_executor
        self.executor.resume_owned_control()
        self.driver = self._driver_factory(
            self.executor,
            rate_hz=self.rate_hz,
            safety_heartbeat=self._heartbeat,
        )
        self.driver.start()
        for pose_id in (RIGHT_CLEARANCE_POSE_ID, HANDOFF_POSE_ID):
            self._phase("return_" + pose_id)
            self.executor.start_pose(pose_id, operator_confirmed=True)
            self.wait_ready(label=pose_id)

    def _heartbeat(self):
        self.guard.pulse()
        self.verify_finger_hold(self.hands.observer.observe())
        self.hands.maintain_active_posture()

    def check(self) -> None:
        if self._closed or self.released:
            raise RuntimeError("standing calibration control is closed")
        if self.driver is None:
            raise RuntimeError("standing calibration controller has not started")
        self.driver.check()
        self.verify_finger_hold(self.hands.observer.observe())

    def wait_ready(self, *, label: str, timeout_s: float | None = None) -> None:
        self._wait_ready(
            self.executor,
            self.driver,
            label=label,
            timeout_s=self.config.motion_timeout_s if timeout_s is None else timeout_s,
        )

    def planning_boundary(self):
        self.check()
        state, command = self.executor.observe_dual_arm_control_input()
        hands = self.hands.observer.observe()
        self.verify_finger_hold(hands)
        snapshot = command_bound_snapshot(state, hands, command)
        self._geometry_hand_snapshot = snapshot
        return state, snapshot

    def execute(
        self,
        *,
        arm: str,
        trajectory: PlannedTrajectory,
        plan_sha256: str,
        validated_reference_state=None,
    ) -> None:
        self.check()
        self._phase(trajectory.to_pose_id)
        reference = (
            self.executor.observe_state()
            if validated_reference_state is None
            else validated_reference_state
        )
        executable = install_plan_at_current_boundary(
            self.executor,
            arm=arm,
            trajectories=(trajectory,),
            plan_sha256=plan_sha256,
            validated_reference_state=reference,
            model=self.model,
            source="NVlabs/curobo_bilateral_calibration",
        )[0]
        self.executor.start_trajectory(
            from_pose_id=executable.from_pose_id,
            to_pose_id=executable.to_pose_id,
            sample_time_s=executable.sample_time_s,
            command_q_rad=executable.command_q_rad,
            plan_sha256=plan_sha256,
            operator_confirmed=True,
        )
        self.wait_ready(
            label=executable.to_pose_id,
            timeout_s=max(self.config.motion_timeout_s, executable.sample_time_s[-1] + 5.0),
        )

    def install_collection(self, *, pose_set, anchor_q14, plan_sha256: str) -> None:
        self.check()
        current = self.executor.dual_arm_command_q
        if not np.allclose(current, anchor_q14, atol=1e-9, rtol=0):
            raise ValueError("collection anchor differs from the held dual-arm command")
        if self.executor.current_pose_id != HANDOFF_POSE_ID:
            raise ValueError("collection must begin at the validated anchor handoff")
        reference = self.executor.observe_state()
        arguments = {
            "pose_set": pose_set,
            "approved_validation_report_sha256": plan_sha256,
            "validated_reference_state": reference,
        }
        if pose_set.calibration_arm == self.executor.pose_set.calibration_arm:
            self.executor.install_validated_plan(**arguments, preserve_current_command=True)
        else:
            self.executor.switch_validated_arm_plan(**arguments, boundary_pose_id=HANDOFF_POSE_ID)
        self._phase("collecting")

    def verify_finger_hold(self, hands, *, reference=None):
        check_full_close(
            left_q_rad=hands.left.position,
            right_q_rad=hands.right.position,
            tolerance_rad=self.hand_config.posture_position_tolerance_rad,
            reference=self._full_close_reference,
        )
        if reference is None:
            reference = self._geometry_hand_snapshot
        if reference is None:
            expected = {
                side: getattr(self.held_hands, side).position for side in ("left", "right")
            }
        else:
            expected = {
                side: getattr(reference, f"{side}_dex3_q_rad") for side in ("left", "right")
            }
        error = max(
            float(np.max(np.abs(np.asarray(getattr(hands, side).position) - expected[side])))
            for side in ("left", "right")
        )
        if error > self.hand_config.posture_position_spread_rad:
            raise ValueError(
                f"Dex3 state changed from the certified measured hold by {error:.4f}rad"
            )

    def release(self) -> None:
        self.check()
        if self.executor is not self.clearance_executor:
            raise ValueError("clean release requires the commissioned shoulder return")
        self._phase("releasing")
        self.executor.begin_clean_release(operator_confirmed=True)
        deadline = self.clock.monotonic() + self.config.release_ramp_s + 5.0
        while self.executor.state is not ExecutorState.STOPPED:
            self.check()
            if self.clock.monotonic() >= deadline:
                raise RuntimeError("clean bilateral arm_sdk release timed out")
            self._sleep(0.01)
        self.driver.close()
        self.driver.check()
        self.guard.pulse()
        self.hands.timeout()
        self.guard.disarm()
        self.released = True
        self._phase("released")

    def close(self) -> list[str]:
        """Finish recovery before closing publishers; retain the original fault."""
        if self._closed:
            return list(self._cleanup_errors)
        self._closed = True

        def attempt(label, action):
            try:
                action()
            except BaseException as error:  # noqa: BLE001
                self._cleanup_errors.append(f"{label}: {error}")

        if self.driver is not None:
            attempt("driver", self.driver.close)
        if self.hands is not None and self.hands.command_count and not self.hands.timed_out:
            attempt("Dex3 timeout", self.hands.timeout)
        if self.guard.armed:
            arm_commanded = self.transport is not None and self.transport.command_count > 0
            fingers_active = (
                self.hands is not None and self.hands.command_count and not self.hands.timed_out
            )
            if not arm_commanded and not fingers_active:
                attempt("PC2 disarm", self.guard.disarm)
            else:
                attempt(
                    "PC2 Damp",
                    lambda: self.guard.damp("bilateral calibration stopped before clean release"),
                )
        if self.transport is not None:
            if self.released or not self.transport.command_count:
                attempt("arm transport", self.transport.close)
            elif self.guard.terminal_action == "damped":
                attempt(
                    "arm transport after recovery", self.transport.close_after_external_takeover
                )
        if self.hands is not None:
            if not self.hands.timed_out and self.guard.terminal_action == "damped":
                attempt("Dex3 close after recovery", self.hands.close_after_external_timeout)
            else:
                attempt("Dex3 close", self.hands.close)
        for name, observer in (
            ("LowState observer", self.observer),
            ("Dex3 observer", self.hand_observer),
        ):
            if observer is not None:
                attempt(name, observer.close)
        return list(self._cleanup_errors)

    def describe(self) -> dict:
        return {
            "phase": self.phase,
            "events": list(self.events),
            "released": self.released,
            "commands_robot": self.commands_robot,
            "terminal_action": self.guard.terminal_action,
            "fault_reason": None if self.executor is None else self.executor.fault_reason,
            "cleanup_errors": list(self._cleanup_errors),
        }
