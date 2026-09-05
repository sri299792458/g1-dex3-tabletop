"""Regression checks for live preparation failures before motion is enabled."""

import sys
from types import SimpleNamespace

import numpy as np
import pytest

from g1_aprilcube_calibration.joint_map import LEFT_ARM_INDICES, RIGHT_ARM_INDICES, arm_joint_names
from g1_aprilcube_calibration.pose_schema import HANDOFF_POSE_ID
from g1_dex3_tabletop.planning import curobo_backend as backend
from g1_dex3_tabletop.planning.contracts import (
    BilateralCalibrationAdapterRequest,
    Dex3PreparationPlan,
    Dex3PreparationRequest,
    PlannedTrajectory,
    RobotSnapshot,
)
from g1_dex3_tabletop.planning.g1_model import Dex3FingerTargetLimitError


@pytest.fixture
def fake_device(monkeypatch):
    # Synthetic joint values here exercise orchestration, not model bounds.
    monkeypatch.setattr(backend, "validate_dex3_finger_targets", lambda **kwargs: None)
    monkeypatch.setitem(
        sys.modules,
        "torch",
        SimpleNamespace(
            device=lambda value: value,
            float32="float32",
            cuda=SimpleNamespace(is_available=lambda: True, get_device_name=lambda _: "test"),
        ),
    )
    monkeypatch.setitem(sys.modules, "curobo.types", SimpleNamespace(DeviceCfg=SimpleNamespace))


def _edge(source, target, start, end):
    return PlannedTrajectory(
        source, target, (0.0, 1.0), (tuple(start), tuple(end)), (tuple(start), tuple(end)), 0.0
    )


@pytest.mark.parametrize("phase", ["close", "return"])
def test_fixed_finger_limit_failure_is_named_and_precedes_offset_search(monkeypatch, phase):
    left = (0.0, 0.6, 1.0, -0.9, -1.0, -0.9, -1.0)
    right = (0.0, -0.6, -1.0, 0.9, 1.0, 0.9, 1.0)
    outside = (*left[:5], -1.5784151554107666, left[6])
    request = Dex3PreparationRequest(
        snapshot=RobotSnapshot((0.0,) * 29, outside, right),
        joint_position_offsets_rad={},
        left_target_q_rad=outside if phase == "close" else left,
        right_target_q_rad=right,
        left_settled_target_q_rad=left,
        right_settled_target_q_rad=right,
        left_return_target_q_rad=outside if phase == "return" else left,
        right_return_target_q_rad=right,
    )
    monkeypatch.setattr(
        backend, "_plan_clearance_arm", lambda **kwargs: pytest.fail("must not plan shoulders")
    )
    progress = []
    with pytest.raises(Dex3FingerTargetLimitError, match="left_hand_index_0_joint") as error:
        backend.plan_dex3_preparation(request, progress=progress.append)
    assert ("restoration target" if phase == "return" else "close target") in str(error.value)
    assert "-1.578415155" in str(error.value)
    assert "-1.570796320" in str(error.value)
    assert progress == []


def test_loaded_sweep_rejects_out_of_limit_target_before_cuda():
    request = Dex3PreparationRequest(
        snapshot=RobotSnapshot((0.0,) * 29, (0.0,) * 7, (0.0,) * 7),
        joint_position_offsets_rad={},
        left_target_q_rad=(0.0, 0.6, 1.0, -0.9, -1.0, -1.578415155, -1.0),
        right_target_q_rad=(0.0, -0.6, -1.0, 0.9, 1.0, 0.9, 1.0),
    )
    with pytest.raises(Dex3FingerTargetLimitError, match="left_hand_index_0_joint"):
        backend.validate_dex3_finger_sweep(request)


def test_preparation_rejects_invalid_return_hand_geometry_before_search(monkeypatch, fake_device):
    snapshot = RobotSnapshot((0.0,) * 29, (0.3,) * 7, (-0.3,) * 7)
    request = Dex3PreparationRequest(
        snapshot=snapshot,
        joint_position_offsets_rad={},
        left_target_q_rad=(0.5,) * 7,
        right_target_q_rad=(-0.5,) * 7,
        left_settled_target_q_rad=(0.49,) * 7,
        right_settled_target_q_rad=(-0.49,) * 7,
        left_return_target_q_rad=(0.0,) * 7,
        right_return_target_q_rad=(0.0,) * 7,
    )

    def build(**kwargs):
        assert kwargs["snapshot"].left_dex3_q_rad == (0.0,) * 7
        assert kwargs["snapshot"].right_dex3_q_rad == (0.0,) * 7
        assert kwargs["ignore_internal_hand_collisions"]
        return {}, (0.0,) * 14

    monkeypatch.setattr(backend, "build_robot_config_for_active_joints", build)
    monkeypatch.setattr(
        backend,
        "CuroboKinematicCollisionChecker",
        lambda **kwargs: SimpleNamespace(
            kinematics=SimpleNamespace(
                joint_names=(*arm_joint_names("left"), *arm_joint_names("right"))
            ),
            self_collision_pair_penetrations=lambda *args, **kwargs: [
                {("left_hand_thumb_2_link", "left_hip_pitch_link"): 0.02669}
            ],
        ),
    )
    progress = []
    with pytest.raises(ValueError, match="cannot fix this fixed return endpoint"):
        backend.plan_dex3_preparation(request, progress=progress.append)
    assert progress == []


@pytest.mark.parametrize("clearance_passes", [True, False])
def test_adapter_uses_live_body_and_requires_core_replay(
    monkeypatch, fake_device, clearance_passes
):
    live = np.zeros(29)
    live[4] = 0.050132334  # Captured ankle mismatch that the old gate rejected.
    live[14] = 0.012
    live[list(LEFT_ARM_INDICES)] = 0.02
    live[list(RIGHT_ARM_INDICES)] = -0.02
    snapshot = RobotSnapshot(tuple(live), (0.3,) * 7, (-0.3,) * 7)
    anchor = np.zeros(29)
    anchor[list(LEFT_ARM_INDICES)] = 0.2
    anchor[list(RIGHT_ARM_INDICES)] = -0.2
    core_edge = _edge(HANDOFF_POSE_ID, "excitation", (-0.2,) * 7, (-0.3,) * 7)
    request = BilateralCalibrationAdapterRequest(
        execution_plan_sha256="a" * 64,
        robot_model="test",
        urdf_sha256="b" * 64,
        snapshot=snapshot,
        anchor_q29_rad=tuple(anchor),
        core_transitions=({"arm": "right", "trajectory": core_edge.to_dict()},),
        joint_position_offsets_rad={},
        left_close_command_q_rad=(0.5,) * 7,
        right_close_command_q_rad=(-0.5,) * 7,
        left_close_model_q_rad=(0.49,) * 7,
        right_close_model_q_rad=(-0.49,) * 7,
    )
    assert request.schema_version == 2
    assert "maximum_locked_joint_error_rad" not in request.to_dict()
    clearance = live.copy()
    clearance[16] += 0.08
    clearance[23] -= 0.08
    right = _edge(
        HANDOFF_POSE_ID,
        "right_shoulder_clearance",
        live[list(RIGHT_ARM_INDICES)],
        clearance[list(RIGHT_ARM_INDICES)],
    )
    left = _edge(
        "right_shoulder_clearance",
        "dual_shoulder_clearance",
        live[list(LEFT_ARM_INDICES)],
        clearance[list(LEFT_ARM_INDICES)],
    )

    def preparation(prep_request, **kwargs):
        assert prep_request.snapshot == snapshot
        assert prep_request.left_return_target_q_rad == snapshot.left_dex3_q_rad
        assert prep_request.right_return_target_q_rad == snapshot.right_dex3_q_rad
        return Dex3PreparationPlan(
            request_sha256=prep_request.content_sha256,
            outward_offset_rad=0.08,
            right_outbound=right,
            left_outbound=left,
            left_return=backend._reverse_trajectory(left),
            right_return=backend._reverse_trajectory(right),
            dual_clearance_q14_rad=tuple(clearance[[*LEFT_ARM_INDICES, *RIGHT_ARM_INDICES]]),
            finger_sweep_sample_count=8,
            return_sweep_sample_count=8,
            planner_provenance={"backend": "test"},
        )

    edge_calls = []

    def anchor_edge(**kwargs):
        state = np.array(kwargs["snapshot"].measured_q29_rad)
        np.testing.assert_array_equal(state[:15], live[:15])
        assert kwargs["snapshot"].left_dex3_q_rad == request.left_close_model_q_rad
        edge_calls.append(kwargs["side"])
        indices = LEFT_ARM_INDICES if kwargs["side"] == "left" else RIGHT_ARM_INDICES
        return _edge(
            kwargs["source_id"], kwargs["target_id"], state[list(indices)], kwargs["target_q"]
        ), "passed"

    certifications = []

    def certify(**kwargs):
        expected = anchor.copy()
        expected[:15] = live[:15]
        np.testing.assert_array_equal(kwargs["snapshot"].measured_q29_rad, expected)
        np.testing.assert_array_equal(kwargs["reference_q29"], expected)
        np.testing.assert_array_equal(kwargs["segments"][0][3], expected)
        assert kwargs["segments"][0][2] == core_edge
        certifications.append(kwargs["phase"])
        return {"passed": clearance_passes}

    monkeypatch.setattr(backend, "plan_dex3_preparation", preparation)
    monkeypatch.setattr(backend, "_plan_bilateral_preparation_edge", anchor_edge)
    monkeypatch.setattr(backend, "_certify_bilateral_arm_segments", certify)
    monkeypatch.setattr(
        backend,
        "_combine_self_clearance_certificates",
        lambda values: {
            "passed": values[0]["passed"],
            "hard_clearance_m": 0.005,
            "minimum_clearance_m": 0.01,
            "minimum_margin_to_required_clearance_m": 0.005 if clearance_passes else -0.001,
            "phases": [{"phase": "live_locked_body_closed_core", "passed": clearance_passes}],
        },
    )
    monkeypatch.setattr(backend, "model_source_hashes", lambda: {"test": "test"})
    if clearance_passes:
        plan = backend.plan_bilateral_calibration_adapter(request)
        assert plan.maximum_locked_joint_error_rad_observed == pytest.approx(live[4])
        assert plan.live_core_self_clearance_certificate["passed"]
    else:
        with pytest.raises(ValueError, match="failed live-body self-clearance replay"):
            backend.plan_bilateral_calibration_adapter(request)
    assert edge_calls == ["right", "left"]
    assert certifications == ["live_locked_body_closed_core"]
