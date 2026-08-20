from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from g1_aprilcube_calibration.calibration_bundle import CalibrationBundle
from g1_aprilcube_calibration.timestamp_pairing import ImageTiming
from g1_dex3_tabletop.hardware_tabletop import (
    _executed_mpc_approach,
    _return_to_clearance_phases,
    _save_frames,
)
from g1_dex3_tabletop.mpc_command_buffer import MPCCommandWindow
from g1_dex3_tabletop.planning import tabletop_planner
from g1_dex3_tabletop.planning.contracts import PlannedTrajectory, RobotSnapshot
from g1_dex3_tabletop.planning.curobo_backend import sample_linear_joint_sweep
from g1_dex3_tabletop.planning.tabletop_planner import (
    _anchor_trajectory_start,
    _base_scene,
    _batched_ik_failure_diagnostic,
    _BranchRejected,
    _canonical_resting_cube_pose,
    _cuboid_cover_spheres,
    _enumerate_pregrasp_branches,
    _FixedCloseSweepValidator,
    _fresh_branch_start_state,
    _load_shortlist,
    _local_plane_clearance,
    _local_table_plane_links,
    _planned_trajectory,
    _pregrasp_endpoint_self_collision_reasons,
    _PregraspBranch,
    _selected_open_transit_world_robot,
    _split_lift_trajectory,
    _table_from_resting_object,
    _try_branch_pool,
    _validate_open_route_segments,
    _validate_start_relative_retention_clearance,
    _validate_strict_supported_escape_self_collision,
)
from g1_dex3_tabletop.tabletop_contracts import (
    SupportedEscapePlan,
    TabletopCuboid,
    TabletopObservation,
    TabletopPickPlaceRequest,
    TabletopTaskRequest,
)
from g1_dex3_tabletop.tabletop_object import load_tabletop_object_profile
from g1_dex3_tabletop.tabletop_perception import (
    camera_motion_from_fixed_cube,
    observe_live_cube_frame,
    observe_resting_cube,
)
from g1_dex3_tabletop.tabletop_workflow import (
    build_tabletop_request,
    destination_request_for_pick_place,
    request_at_clearance,
    request_at_clearance_observation,
)


def _identity():
    return np.eye(4)


def _observation() -> TabletopObservation:
    return TabletopObservation(
        RobotSnapshot((0.0,) * 29, (0.0,) * 7, (0.0,) * 7),
        tuple(tuple(row) for row in _identity()),
        "a" * 64,
        ("b" * 64, "c" * 64, "d" * 64),
        0.1,
        0.1,
    )


def test_live_cube_target_is_one_exact_frame_not_a_lagging_burst(monkeypatch) -> None:
    image = np.zeros((12, 16, 3), dtype=np.uint8)
    transform = np.eye(4)
    transform[0, 3] = 0.42
    estimate = SimpleNamespace(
        camera_T_target=transform,
        to_dict=lambda: {
            "camera_T_target": transform.tolist(),
            "reprojection_error_px": 0.2,
            "point_count": 8,
            "marker_ids": [1],
            "visible_faces": ["top"],
            "second_solution_error_px": None,
        },
    )
    received = {}

    def detect(value, camera_info, detector, **kwargs):
        received.update(kwargs)
        assert value is image
        assert camera_info == "camera"
        assert detector == "detector"
        return estimate

    monkeypatch.setattr(
        "g1_dex3_tabletop.tabletop_perception.detect_hand_target_pose",
        detect,
    )
    observation = observe_live_cube_frame(
        image,
        camera_info="camera",
        detector="detector",
        minimum_tag_short_side_px=26.0,
        maximum_reprojection_error_px=2.5,
    )

    np.testing.assert_allclose(observation["camera_T_object"], transform)
    assert len(observation["source_frame_sha256"]) == 64
    assert observation["pose_evidence"]["reprojection_error_px"] == pytest.approx(0.2)
    assert received["single_best_face"]
    assert received["minimum_tag_short_side_px"] == pytest.approx(26.0)


def test_executed_mpc_approach_stitches_only_committed_window_segments() -> None:
    def window(
        generation: int,
        valid_from: float,
        rows: tuple[tuple[float, ...], ...],
        *,
        terminal: bool,
        predecessor: str | None,
        predicted_rows: tuple[tuple[float, ...], ...] | None = None,
    ) -> MPCCommandWindow:
        return MPCCommandWindow(
            generation=generation,
            plan_sha256="a" * 64,
            source_state_monotonic_s=valid_from - 0.1,
            valid_from_monotonic_s=valid_from,
            sample_time_s=(0.0, 0.1, 0.2),
            command_q_rad=rows,
            predicted_q_rad=rows if predicted_rows is None else predicted_rows,
            predicted_dq_rad_s=((0.0,) * 7,) * 3,
            predicted_ddq_rad_s2=((0.0,) * 7,) * 3,
            predecessor_sha256=predecessor,
            feasible=True,
            terminal=terminal,
            solve_time_s=0.02,
            diagnostics={"moving_target": {}},
        )

    q0 = (0.0,) * 7
    q1 = (0.01,) * 7
    q2 = (0.02,) * 7
    q3 = (0.03,) * 7
    first = window(0, 10.0, (q0, q1, q2), terminal=False, predecessor=None)
    second = window(
        1,
        10.2,
        (q2, q3, (0.04,) * 7),
        terminal=True,
        predecessor=first.content_sha256,
        predicted_rows=(q2, (0.025,) * 7, (0.035,) * 7),
    )

    approach = _executed_mpc_approach(
        [first.to_dict(), second.to_dict()],
        arm="left",
        joint_position_offsets_rad={"left_shoulder_pitch_joint": 0.1},
    )

    assert approach.sample_time_s == pytest.approx((0.0, 0.1, 0.2, 0.3, 0.4))
    assert approach.command_q_rad == (q0, q1, q2, q3, (0.04,) * 7)
    assert approach.model_q_rad[0][0] == pytest.approx(0.1)
    assert approach.model_q_rad[-1][0] == pytest.approx(0.135)
    assert approach.planning_time_s == pytest.approx(0.04)


def test_supported_escape_accepts_only_strict_collision_free_samples() -> None:
    assert _validate_strict_supported_escape_self_collision([{}, {}, {}]) is None


def test_cube_contact_links_are_disabled_only_during_final_grasp_approach(
    monkeypatch,
) -> None:
    calls = []

    def validate(**kwargs):
        calls.append(kwargs)
        if len(calls) == 1:
            return 0.030, "transit_link", 1
        return 0.020, "grasp_link", 2

    monkeypatch.setattr(tabletop_planner, "_validate_open_route", validate)
    result = _validate_open_route_segments(
        planner=object(),
        device_cfg=object(),
        transit_q=np.zeros((3, 7)),
        grasp_q=np.zeros((4, 7)),
        open_robot={},
        strict_checker=object(),
        request=object(),
        base_T_torso=np.eye(4),
        plane_point=np.zeros(3),
        down=np.asarray((0.0, 0.0, -1.0)),
        arm="left",
    )

    assert calls[0]["disabled_cube_links"] == set()
    assert calls[1]["disabled_cube_links"] == {
        "left_hand_thumb_0_link",
        "left_hand_thumb_1_link",
        "left_hand_thumb_2_link",
        "left_hand_middle_0_link",
        "left_hand_middle_1_link",
        "left_hand_index_0_link",
        "left_hand_index_1_link",
    }
    assert result == (0.020, "grasp_link", 4)


def test_pick_place_destination_reexpresses_world_and_marks_only_support() -> None:
    source = TabletopTaskRequest(
        observation=_observation(),
        arm="left",
        torso_T_camera=tuple(tuple(row) for row in np.eye(4)),
        joint_position_offsets_rad={},
        calibration_bundle_sha256="a" * 64,
        grasp_shortlist_path="config/tabletop/cube_dex3_executable_v1/shortlist.yaml",
        grasp_shortlist_sha256="b" * 64,
        object_dimensions_m=(0.04, 0.04, 0.04),
        environment_cuboids=(
            TabletopCuboid(
                object_id="cube60",
                object_T_cuboid=(
                    (1.0, 0.0, 0.0, 0.2),
                    (0.0, 1.0, 0.0, 0.0),
                    (0.0, 0.0, 1.0, 0.01),
                    (0.0, 0.0, 0.0, 1.0),
                ),
                dimensions_m=(0.06, 0.06, 0.06),
            ),
        ),
    )
    source_T_destination = np.eye(4)
    source_T_destination[0, 3] = 0.2
    source_T_destination[2, 3] = 0.05
    destination = destination_request_for_pick_place(
        TabletopPickPlaceRequest(
            source_request=source,
            source_T_destination_object=source_T_destination,
            destination_support_object_id="cube60",
        )
    )

    assert destination.environment_cuboids[0].role == "placement_support"
    expected_destination_T_support = np.eye(4)
    expected_destination_T_support[2, 3] = -0.04
    np.testing.assert_allclose(
        destination.environment_cuboids[0].object_T_cuboid,
        expected_destination_T_support,
    )
    assert destination.observation.camera_T_object[0][3] == pytest.approx(0.2)
    assert destination.observation.camera_T_object[2][3] == pytest.approx(0.05)
    np.testing.assert_allclose(
        destination.table_reference_camera_T_object,
        source.observation.camera_T_object,
    )
    point, _canonical, _down = _table_from_resting_object(destination, np.eye(4))
    np.testing.assert_allclose(point, (0.0, 0.0, -0.02))


def test_environment_cuboid_uses_detector_frame_not_face_up_permutation() -> None:
    request = TabletopTaskRequest(
        observation=_observation(),
        arm="left",
        torso_T_camera=tuple(tuple(row) for row in np.eye(4)),
        joint_position_offsets_rad={},
        calibration_bundle_sha256="a" * 64,
        grasp_shortlist_path="config/tabletop/cube_dex3_executable_v1/shortlist.yaml",
        grasp_shortlist_sha256="b" * 64,
        environment_cuboids=(
            TabletopCuboid(
                object_id="other",
                object_T_cuboid=(
                    (1.0, 0.0, 0.0, 0.2),
                    (0.0, 1.0, 0.0, 0.0),
                    (0.0, 0.0, 1.0, 0.0),
                    (0.0, 0.0, 0.0, 1.0),
                ),
                dimensions_m=(0.06, 0.06, 0.06),
            ),
        ),
    )
    detector_pose = np.eye(4)
    detector_pose[:3, :3] = Rotation.from_euler("y", 90, degrees=True).as_matrix()
    document = request.to_dict(include_hash=False)
    document["observation"]["camera_T_object"] = detector_pose.tolist()
    document["observation"].pop("content_sha256", None)
    request = TabletopTaskRequest.from_dict(document)

    scene = _base_scene(request, np.eye(4), include_cube=False)

    expected = detector_pose.copy()
    expected[:3, 3] += detector_pose[:3, :3] @ np.asarray((0.2, 0.0, 0.0))
    pose = scene["cuboid"]["other"]["pose"]
    actual = np.eye(4)
    actual[:3, 3] = pose[:3]
    actual[:3, :3] = Rotation.from_quat((*pose[4:], pose[3])).as_matrix()
    np.testing.assert_allclose(actual, expected, atol=1.0e-9)


def test_corrected_return_executes_both_post_retreat_legs() -> None:
    corrected = {
        "return_to_pregrasp": object(),
        "return_to_clearance": object(),
    }
    legacy = {"return_to_clearance": object()}

    assert _return_to_clearance_phases(corrected) == (
        "return_to_pregrasp",
        "return_to_clearance",
    )
    assert _return_to_clearance_phases(legacy) == ("return_to_clearance",)


def test_float32_planner_start_is_anchored_to_exact_serialized_state() -> None:
    expected = np.asarray((0.1,) * 6 + (0.4922822415828705,))
    rounded = tuple(np.asarray(expected, dtype=np.float32).astype(np.float64))
    planned = PlannedTrajectory(
        "clearance",
        "move_to_pregrasp",
        (0.0, 1.0),
        (rounded, (0.2,) * 7),
        (rounded, (0.2,) * 7),
        0.1,
    )

    anchored = _anchor_trajectory_start(
        planned,
        command_q_rad=expected,
        model_q_rad=expected,
    )

    assert anchored.command_q_rad[0] == tuple(expected)
    assert anchored.model_q_rad[0] == tuple(expected)


def test_planner_start_anchor_rejects_a_real_discontinuity() -> None:
    planned = PlannedTrajectory(
        "clearance",
        "move_to_pregrasp",
        (0.0, 1.0),
        ((0.0,) * 7, (0.2,) * 7),
        ((0.0,) * 7, (0.2,) * 7),
        0.1,
    )
    with pytest.raises(RuntimeError, match="does not begin"):
        _anchor_trajectory_start(
            planned,
            command_q_rad=np.asarray((0.01,) * 7),
            model_q_rad=np.asarray((0.01,) * 7),
        )


def test_linear_finger_sweep_has_exact_endpoints_and_bounded_steps() -> None:
    start = np.zeros(7)
    target = np.asarray((0.0, -0.5984, -0.99731429, 0.8976, 0.99731429, 0.2, 0.3))

    sweep = sample_linear_joint_sweep(start, target)

    np.testing.assert_array_equal(sweep[0], start)
    np.testing.assert_array_equal(sweep[-1], target)
    assert np.max(np.abs(np.diff(sweep, axis=0))) <= 0.02


def test_measured_close_route_can_escape_from_a_positive_submargin_boundary() -> None:
    clearances = np.asarray(
        [
            0.0038077514,
            0.0038077511,
            0.0041641479,
            0.0049591359,
            0.0062544693,
            0.0102200422,
            0.0062544693,
            0.0049591359,
            0.0041641479,
            0.0038077514,
        ]
    )
    links = ("left_hand_middle_1_link",) * len(clearances)

    result = _validate_start_relative_retention_clearance(
        clearances,
        links,
        required_m=0.005,
    )

    assert result.minimum_m == pytest.approx(0.0038077511)
    assert result.boundary_m == pytest.approx(0.0038077514)
    assert result.first_full_margin_sample == 4
    assert result.last_full_margin_sample == 6


def test_measured_close_escape_cannot_move_below_its_starting_clearance() -> None:
    clearances = np.asarray([0.0038, 0.0030, 0.0060, 0.0038])
    links = ("left_hand_middle_1_link",) * len(clearances)

    with pytest.raises(RuntimeError, match="moves closer.*already-achieved grasp boundary"):
        _validate_start_relative_retention_clearance(
            clearances,
            links,
            required_m=0.005,
        )


def test_measured_close_route_cannot_dip_below_margin_in_free_space() -> None:
    clearances = np.asarray([0.0038, 0.0060, 0.0040, 0.0060, 0.0038])
    links = ("left_hand_middle_1_link",) * len(clearances)

    with pytest.raises(RuntimeError, match="drops below.*free-space"):
        _validate_start_relative_retention_clearance(
            clearances,
            links,
            required_m=0.005,
        )


def test_measured_close_route_must_reach_full_free_space_margin() -> None:
    clearances = np.asarray([0.0038, 0.0045, 0.0038])
    links = ("left_hand_middle_1_link",) * len(clearances)

    with pytest.raises(RuntimeError, match="never reaches.*free-space"):
        _validate_start_relative_retention_clearance(
            clearances,
            links,
            required_m=0.005,
        )


def test_measured_close_start_relative_rule_never_allows_table_penetration() -> None:
    clearances = np.asarray([-0.0001, 0.0060, -0.0001])
    links = ("left_hand_middle_1_link",) * len(clearances)

    with pytest.raises(RuntimeError, match="not above the observed table plane"):
        _validate_start_relative_retention_clearance(
            clearances,
            links,
            required_m=0.005,
        )


def test_fixed_close_sweep_rejects_table_crossing_before_pregrasp(monkeypatch) -> None:
    class FakeSphereTensor:
        def detach(self):
            return self

        def cpu(self):
            return self

        def numpy(self):
            return np.zeros((3, 1, 4), dtype=np.float64)

    class FakeChecker:
        config = SimpleNamespace(kinematics_config=object())

        def robot_spheres(self, values, *, joint_names):
            assert values.shape == (3, 14)
            assert len(joint_names) == 14
            return FakeSphereTensor()

        @staticmethod
        def self_collision_pair_penetrations_from_spheres(_spheres):
            return [{}, {}, {}]

    monkeypatch.setattr(
        "g1_dex3_tabletop.planning.tabletop_planner._local_plane_clearance_from_spheres",
        lambda *_args, **_kwargs: (-0.0038, "left_hand_index_1_link", 2),
    )
    validator = _FixedCloseSweepValidator.__new__(_FixedCloseSweepValidator)
    validator.request = SimpleNamespace(
        minimum_hand_plane_clearance_m=0.005,
        fixture=object(),
    )
    validator.arm = "left"
    validator.plane_point = np.zeros(3)
    validator.down = np.asarray((0.0, 0.0, -1.0))
    validator.finger_sweep = np.zeros((3, 7))
    validator.active_joint_names = tuple(f"joint_{index}" for index in range(14))
    validator.checker = FakeChecker()
    validator.fixture_checker = None

    with pytest.raises(_BranchRejected) as caught:
        validator.validate(np.zeros(7), {})

    assert caught.value.stage == "fixed_close_sweep_table_plane"
    assert "left_hand_index_1_link" in caught.value.reason
    assert "-0.0038m" in caught.value.reason
    assert "required=0.0050m" in caught.value.reason


def test_direct_fixed_close_uses_exact_qualified_hand_clearance(monkeypatch) -> None:
    class FakeSphereTensor:
        def detach(self):
            return self

        def cpu(self):
            return self

        def numpy(self):
            return np.zeros((3, 1, 4), dtype=np.float64)

    class FakeChecker:
        config = SimpleNamespace(kinematics_config=object())

        def robot_spheres(self, values, *, joint_names):
            assert values.shape == (3, 14)
            assert len(joint_names) == 14
            return FakeSphereTensor()

        @staticmethod
        def self_collision_pair_penetrations_from_spheres(_spheres):
            return [{}, {}, {}]

    monkeypatch.setattr(
        "g1_dex3_tabletop.planning.tabletop_planner._local_plane_clearance_from_spheres",
        lambda *_args, **_kwargs: (0.012, "left_wrist_yaw_link", 1),
    )
    validator = _FixedCloseSweepValidator.__new__(_FixedCloseSweepValidator)
    validator.request = SimpleNamespace(
        minimum_hand_plane_clearance_m=0.005,
        fixture=None,
    )
    validator.arm = "left"
    validator.plane_point = np.zeros(3)
    validator.down = np.asarray((0.0, 0.0, -1.0))
    validator.finger_sweep = np.zeros((3, 7))
    validator.active_joint_names = tuple(f"joint_{index}" for index in range(14))
    validator.checker = FakeChecker()
    validator.fixture_checker = None
    candidate = {
        "execution_evidence": {
            "fixed_close_sweep_table_clearance_m": 0.0061,
            "fixed_close_sweep_minimum_link": "right_hand_index_1_link",
            "fixed_close_sweep_minimum_sample": 2,
        }
    }

    result = validator.validate(np.zeros(7), candidate)

    assert result.minimum_plane_clearance_m == pytest.approx(0.0061)
    assert result.minimum_plane_link == "left_hand_index_1_link"
    assert result.minimum_plane_sample == 2


def test_fixed_close_sweep_rejects_cuda_fixture_collision(monkeypatch) -> None:
    class FakeSphereTensor:
        def detach(self):
            return self

        def cpu(self):
            return self

        def numpy(self):
            return np.zeros((3, 1, 4), dtype=np.float64)

    kinematics_config = object()

    class FakeChecker:
        config = SimpleNamespace(kinematics_config=kinematics_config)

        def robot_spheres(self, values, *, joint_names):
            assert values.shape == (3, 14)
            assert len(joint_names) == 14
            return FakeSphereTensor()

        @staticmethod
        def self_collision_pair_penetrations_from_spheres(_spheres):
            return [{}, {}, {}]

    class FakeFixtureChecker:
        def first_collision(self, spheres, *, kinematics_config):
            assert isinstance(spheres, FakeSphereTensor)
            assert kinematics_config is FakeChecker.config.kinematics_config
            return 0.0044, "left_hand_middle_1_link", 2

    monkeypatch.setattr(
        "g1_dex3_tabletop.planning.tabletop_planner._local_plane_clearance_from_spheres",
        lambda *_args, **_kwargs: (0.012, "left_wrist_yaw_link", 1),
    )
    validator = _FixedCloseSweepValidator.__new__(_FixedCloseSweepValidator)
    validator.request = SimpleNamespace(
        minimum_hand_plane_clearance_m=0.005,
        fixture=object(),
    )
    validator.arm = "left"
    validator.plane_point = np.zeros(3)
    validator.down = np.asarray((0.0, 0.0, -1.0))
    validator.finger_sweep = np.zeros((3, 7))
    validator.active_joint_names = tuple(f"joint_{index}" for index in range(14))
    validator.checker = FakeChecker()
    validator.fixture_checker = FakeFixtureChecker()

    with pytest.raises(_BranchRejected) as caught:
        validator.validate(np.zeros(7), {})

    assert caught.value.stage == "fixed_close_sweep_fixture"
    assert "left_hand_middle_1_link=4.400mm" in caught.value.reason
    assert "sample 2/2" in caught.value.reason


def test_supported_escape_reports_live_start_collision_in_millimetres() -> None:
    pair = ("right_shoulder_yaw_link", "torso_link")
    with pytest.raises(
        RuntimeError,
        match=(
            r"live supported-start state.*"
            r"right_shoulder_yaw_link/torso_link=1\.203mm.*"
            r"Reposition the robot and rerun"
        ),
    ):
        _validate_strict_supported_escape_self_collision([{pair: 0.001203}, {}])


def test_supported_escape_reports_route_collision_sample_and_penetration() -> None:
    pair = ("right_elbow_link", "torso_link")
    with pytest.raises(
        RuntimeError,
        match=r"sample 1/2: right_elbow_link/torso_link=0\.400mm",
    ):
        _validate_strict_supported_escape_self_collision([{}, {pair: 0.0004}, {}])


def test_failure_frames_preserve_explicit_image_timing(tmp_path: Path) -> None:
    frame = SimpleNamespace(
        image_bgr=np.zeros((8, 8, 3), dtype=np.uint8),
        timing=ImageTiming(1.25, "2026-08-13T12:00:00Z", 123),
    )
    destination = tmp_path / "preflight"

    _save_frames(destination, (frame,))

    manifest = json.loads((destination / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["frames"][0]["timing"] == {
        "receipt_monotonic_s": 1.25,
        "receipt_utc": "2026-08-13T12:00:00Z",
        "header_stamp_ns": 123,
    }


def test_tabletop_detection_names_the_cube_not_the_hand(monkeypatch) -> None:
    def reject(_image, _camera_info, _detector, **kwargs):
        assert kwargs["target_label"] == "tabletop AprilCube"
        assert kwargs["minimum_tag_short_side_px"] == 25.0
        assert kwargs["maximum_reprojection_error_px"] == 3.0
        assert kwargs["single_best_face"] is True
        raise ValueError("tabletop AprilCube was not detected")

    monkeypatch.setattr("g1_dex3_tabletop.tabletop_perception.detect_hand_target_pose", reject)

    with pytest.raises(ValueError, match="tabletop AprilCube was not detected"):
        observe_resting_cube(
            [np.full((8, 8, 3), index, dtype=np.uint8) for index in range(5)],
            camera_info=object(),
            detector=object(),
            snapshot=_observation().snapshot,
        )


def test_task_config_uses_only_a_local_open_transit_table_patch() -> None:
    bundle_path = (
        Path(__file__).resolve().parents[1]
        / "config/calibrations/dex3_shared_20260812_selected_free.json"
    )
    shortlist = Path(__file__).resolve().parents[1] / (
        "config/tabletop/cube_dex3_executable_v1/shortlist.yaml"
    )
    task = Path(__file__).resolve().parents[1] / "config/tabletop/task.yaml"
    bundle = CalibrationBundle.load(bundle_path)
    request = build_tabletop_request(
        arm="right",
        observation=_observation(),
        calibration_bundle=bundle,
        calibration_bundle_path=bundle_path,
        grasp_shortlist_path=shortlist,
        task_config_path=task,
        object_dimensions_m=(0.040, 0.040, 0.040),
    )
    assert "physical_table_dimensions_m" not in request.to_dict()
    assert "physical_table_thickness_m" not in request.to_dict()
    assert request.open_transit_table_patch_dimensions_m == (0.400, 0.400, 0.020)
    assert request.maximum_arm_velocity_rad_s == 0.100
    assert request.retention_test_lift_m == 0.030
    assert request.minimum_hand_plane_clearance_m == 0.005

    faster_request = build_tabletop_request(
        arm="right",
        observation=_observation(),
        calibration_bundle=bundle,
        calibration_bundle_path=bundle_path,
        grasp_shortlist_path=shortlist,
        task_config_path=task,
        object_dimensions_m=(0.040, 0.040, 0.040),
        maximum_arm_velocity_rad_s=0.200,
    )
    assert faster_request.maximum_arm_velocity_rad_s == 0.200


def test_tabletop_trajectory_is_retimed_to_task_velocity() -> None:
    model_q = np.zeros((2, 7), dtype=np.float64)
    model_q[1, 0] = 0.2
    trajectory = _planned_trajectory(
        from_id="source",
        to_id="target",
        model_q=model_q,
        native_dt=0.1,
        arm="right",
        offsets={},
        planning_time_s=0.5,
        maximum_velocity_rad_s=0.1,
    )

    assert trajectory.sample_time_s == pytest.approx((0.0, 2.0))
    command = np.asarray(trajectory.command_q_rad)
    velocity = np.max(np.abs(np.diff(command, axis=0))) / trajectory.sample_time_s[-1]
    assert velocity == pytest.approx(0.1)


def test_validated_lift_split_preserves_every_sample_and_exact_join() -> None:
    trajectory = PlannedTrajectory(
        "grasp_approach",
        "payload_lift",
        (0.0, 1.0, 2.0, 3.0),
        tuple((value,) * 7 for value in (0.0, 0.1, 0.2, 0.3)),
        tuple((value,) * 7 for value in (0.0, 0.1, 0.2, 0.3)),
        0.5,
    )

    test_lift, payload_lift = _split_lift_trajectory(trajectory, split_index=1)

    assert test_lift.from_pose_id == "grasp_approach"
    assert test_lift.to_pose_id == "retention_test_lift"
    assert payload_lift.from_pose_id == "retention_test_lift"
    assert payload_lift.to_pose_id == "payload_lift"
    assert test_lift.command_q_rad[-1] == payload_lift.command_q_rad[0]
    assert test_lift.command_q_rad + payload_lift.command_q_rad[1:] == (trajectory.command_q_rad)
    assert payload_lift.sample_time_s == pytest.approx((0.0, 1.0, 2.0))


@pytest.mark.parametrize(
    ("profile_id", "dimensions_m", "candidate_count", "top_marker_id"),
    (
        ("cube40-r3", (0.040, 0.040, 0.040), 5, 4),
        ("cube60-r3", (0.060, 0.060, 0.060), 57, 14),
    ),
)
def test_object_profile_binds_detector_mesh_dimensions_and_shortlist(
    profile_id,
    dimensions_m,
    candidate_count,
    top_marker_id,
) -> None:
    root = Path(__file__).resolve().parents[1]
    bundle_path = root / "config/calibrations/dex3_shared_20260812_selected_free.json"
    task_path = root / "config/tabletop/task.yaml"
    profile = load_tabletop_object_profile(profile_id)
    bundle = CalibrationBundle.load(bundle_path)
    request = build_tabletop_request(
        arm="right",
        observation=_observation(),
        calibration_bundle=bundle,
        calibration_bundle_path=bundle_path,
        grasp_shortlist_path=profile.direct_grasp_shortlist_path,
        task_config_path=task_path,
        object_dimensions_m=profile.dimensions_m,
    )

    _shortlist, candidates = _load_shortlist(request)
    detector = json.loads(profile.detector_config_path.read_text(encoding="utf-8"))

    assert request.object_dimensions_m == dimensions_m
    assert len(candidates) == candidate_count
    assert all(
        item["execution_evidence"]["qualification_model"]
        == "stationary_cube_fixed_descriptor_close"
        for item in candidates
    )
    assert (
        min(
            item["execution_evidence"]["fixed_close_sweep_table_clearance_m"]
            for item in candidates
        )
        >= request.minimum_hand_plane_clearance_m
    )
    assert detector["dict"] == "4x4_100"
    assert detector["box_dims"] == [1000.0 * value for value in dimensions_m]
    assert detector["faces"]["+Z"] == [top_marker_id]


def test_cube_observation_defines_plane_and_configured_open_transit_patch() -> None:
    from g1_dex3_tabletop.tabletop_contracts import TabletopTaskRequest

    loaded = TabletopTaskRequest(
        _observation(),
        "right",
        tuple(tuple(row) for row in _identity()),
        {},
        "e" * 64,
        "config/tabletop/x.yaml",
        "f" * 64,
    )
    point, object_pose, down = _table_from_resting_object(loaded, _identity())
    np.testing.assert_allclose(object_pose, np.eye(4))
    np.testing.assert_allclose(point, [0.0, 0.0, -0.0200])
    np.testing.assert_allclose(down, [0.0, 0.0, -1.0])
    assert _base_scene(loaded, _identity(), include_cube=False) == {"cuboid": {}}
    assert set(_base_scene(loaded, _identity(), include_cube=True)["cuboid"]) == {"cube"}
    scene = _base_scene(
        loaded,
        _identity(),
        include_cube=True,
        include_open_transit_table_patch=True,
    )
    assert set(scene["cuboid"]) == {"cube", "open_transit_table_patch"}
    assert scene["cuboid"]["open_transit_table_patch"]["dims"] == [0.4, 0.4, 0.02]
    np.testing.assert_allclose(
        scene["cuboid"]["open_transit_table_patch"]["pose"][:3],
        [0.0, 0.0, -0.030],
    )


@pytest.mark.parametrize(
    "rotation",
    (
        np.eye(3),
        np.diag([1.0, -1.0, -1.0]),
        np.asarray(((0.0, 0.0, -1.0), (0.0, 1.0, 0.0), (1.0, 0.0, 0.0))),
        np.asarray(((0.0, 0.0, 1.0), (0.0, 1.0, 0.0), (-1.0, 0.0, 0.0))),
        np.asarray(((1.0, 0.0, 0.0), (0.0, 0.0, -1.0), (0.0, 1.0, 0.0))),
        np.asarray(((1.0, 0.0, 0.0), (0.0, 0.0, 1.0), (0.0, -1.0, 0.0))),
    ),
)
def test_cube_observation_accepts_every_physical_support_face(rotation) -> None:
    from g1_dex3_tabletop.tabletop_contracts import TabletopTaskRequest

    camera_T_object = np.eye(4)
    camera_T_object[:3, :3] = rotation
    observation = TabletopObservation(
        _observation().snapshot,
        tuple(tuple(row) for row in camera_T_object),
        "a" * 64,
        ("b" * 64, "c" * 64, "d" * 64),
        0.1,
        0.1,
    )
    loaded = TabletopTaskRequest(
        observation,
        "right",
        tuple(tuple(row) for row in _identity()),
        {},
        "e" * 64,
        "config/tabletop/x.yaml",
        "f" * 64,
    )

    point, canonical, down = _table_from_resting_object(loaded, _identity())
    np.testing.assert_allclose(canonical[:3, 3], camera_T_object[:3, 3])
    np.testing.assert_allclose(canonical[:3, 2], [0.0, 0.0, 1.0])
    np.testing.assert_allclose(point, [0.0, 0.0, -0.0200])
    np.testing.assert_allclose(down, [0.0, 0.0, -1.0])


def test_cube_observation_still_rejects_a_cube_not_resting_on_a_face() -> None:
    tilted = np.eye(4)
    tilted[:3, :3] = Rotation.from_euler("x", 30.0, degrees=True).as_matrix()
    with pytest.raises(RuntimeError, match="not resting on a face"):
        _canonical_resting_cube_pose(tilted)


def test_pregrasp_enumeration_gives_each_candidate_independent_seeds(monkeypatch) -> None:
    class Result:
        success = np.asarray(
            [
                [True, True, False],
                [True, True, True],
            ]
        )
        solution = np.asarray(
            [
                [[0.2] * 7, [0.1] * 7, [0.3] * 7],
                [[0.4] * 7, [0.400000001] * 7, [0.05] * 7],
            ],
            dtype=np.float64,
        )
        position_error = np.asarray([[0.002, 0.001, 0.003], [0.004, 0.004, 0.0005]])
        rotation_error = np.asarray([[0.02, 0.01, 0.03], [0.04, 0.04, 0.005]])

    class Solver:
        def solve_pose(self, goals, *, return_seeds, current_state):
            assert goals == "goals"
            assert return_seeds > 0
            assert current_state == "batched-start"
            return Result()

    state = SimpleNamespace(position=np.zeros((1, 7)))
    monkeypatch.setattr(
        "g1_dex3_tabletop.planning.tabletop_planner._repeat_joint_state",
        lambda current, count: "batched-start" if current is state and count == 2 else None,
    )
    branches, result = _enumerate_pregrasp_branches(
        Solver(),
        "goals",
        state,
        candidate_count=2,
    )

    assert isinstance(result, Result)
    assert [branch.candidate_local_index for branch in branches] == [1, 0, 0, 1]
    assert [branch.solver_seed_index for branch in branches] == [2, 1, 0, 0]
    np.testing.assert_allclose(branches[0].model_q_rad, [0.05] * 7)
    np.testing.assert_allclose(branches[1].model_q_rad, [0.1] * 7)
    np.testing.assert_allclose(branches[2].model_q_rad, [0.2] * 7)
    np.testing.assert_allclose(branches[3].model_q_rad, [0.4] * 7)


def test_complete_branch_search_does_not_discard_candidate_after_first_failure() -> None:
    branches = [
        _PregraspBranch(0, 2, np.asarray([0.1] * 7), 0.001, 0.01),
        _PregraspBranch(0, 7, np.asarray([0.2] * 7), 0.001, 0.01),
    ]
    attempted: list[int] = []

    def attempt(branch):
        attempted.append(branch.solver_seed_index)
        if len(attempted) == 1:
            raise _BranchRejected("attached_payload_lift", "first branch cannot lift")
        return "open-plan", "lift-plan"

    reports: list[str] = []
    selected, result, rejected = _try_branch_pool(
        branches,
        candidate_ids=["candidate_0"],
        attempt=attempt,
        report=reports.append,
    )

    assert attempted == [2, 7]
    assert selected is branches[1]
    assert result == ("open-plan", "lift-plan")
    assert rejected == [
        {
            "candidate_id": "candidate_0",
            "pool_branch_index": 1,
            "candidate_branch_index": 1,
            "candidate_branch_count": 2,
            "solver_seed_index": 2,
            "stage": "attached_payload_lift",
            "reason": "first branch cannot lift",
        }
    ]
    assert any(message.startswith("rejected candidate_0 IK branch 1/2") for message in reports)


def test_pregrasp_endpoint_collisions_are_batched_and_named() -> None:
    branches = [
        _PregraspBranch(0, 2, np.asarray([0.1] * 7), 0.001, 0.01),
        _PregraspBranch(1, 7, np.asarray([0.2] * 7), 0.001, 0.01),
    ]

    class Checker:
        @staticmethod
        def self_collision_pair_penetrations(samples):
            np.testing.assert_allclose(samples, [[0.1] * 7, [0.2] * 7])
            return [
                {("left_elbow_link", "torso_link"): 0.004612},
                {},
            ]

    reasons = _pregrasp_endpoint_self_collision_reasons(
        branches,
        checker=Checker(),
    )

    assert reasons == ["left_elbow_link/torso_link=4.612mm", None]


def test_rejected_branch_cannot_mutate_the_next_branch_start(monkeypatch) -> None:
    reference = np.asarray([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7])

    def fake_joint_state(_device_cfg, values, names):
        assert names == (
            "left_shoulder_pitch_joint",
            "left_shoulder_roll_joint",
            "left_shoulder_yaw_joint",
            "left_elbow_joint",
            "left_wrist_roll_joint",
            "left_wrist_pitch_joint",
            "left_wrist_yaw_joint",
        )
        return SimpleNamespace(position=np.asarray(values).copy())

    monkeypatch.setattr(
        "g1_dex3_tabletop.planning.tabletop_planner._joint_state",
        fake_joint_state,
    )
    branches = [
        _PregraspBranch(0, 0, np.zeros(7), 0.0, 0.0),
        _PregraspBranch(0, 1, np.ones(7), 0.0, 0.0),
    ]
    starts = []

    def attempt(branch):
        state = _fresh_branch_start_state(None, reference, arm="left")
        starts.append(state)
        if branch.solver_seed_index == 0:
            state.position[:] = 9.0
            raise _BranchRejected("test", "simulated CuRobo mutation")
        np.testing.assert_allclose(state.position, reference)
        return "open-plan", "lift-plan"

    selected, result, rejected = _try_branch_pool(
        branches,
        candidate_ids=["candidate_0"],
        attempt=attempt,
        report=lambda _message: None,
    )

    assert selected is branches[1]
    assert result == ("open-plan", "lift-plan")
    assert len(rejected) == 1
    assert starts[0] is not starts[1]
    np.testing.assert_allclose(starts[0].position, 9.0)
    np.testing.assert_allclose(starts[1].position, reference)


def test_batched_ik_failure_diagnostic_names_candidate_and_collision_pairs(monkeypatch) -> None:
    class Result:
        position_error = np.asarray([[0.02, 0.02], [0.0, 0.02]])
        rotation_error = np.asarray([[0.03, 0.03], [0.0, 0.03]])
        solution = np.asarray(
            [
                [[0.3] * 7, [0.4] * 7],
                [[0.1] * 7, [0.2] * 7],
            ]
        )
        success = np.asarray([[False, False], [False, False]])

    monkeypatch.setattr(
        "g1_dex3_tabletop.planning.tabletop_planner._self_collision_pair_penetrations",
        lambda **_kwargs: [
            {
                ("right_shoulder_yaw_link", "torso_link"): 0.012919,
                ("right_hand_thumb_2_link", "cube"): 0.1,
            }
        ],
    )
    monkeypatch.setattr(
        "g1_dex3_tabletop.planning.tabletop_planner._world_cuboid_clearances",
        lambda **_kwargs: [{}],
    )

    message = _batched_ik_failure_diagnostic(
        result=Result(),
        robot={},
        scene={},
        candidates=[{"candidate_id": "candidate_0"}, {"candidate_id": "candidate_1"}],
        device_cfg=None,
        disabled_collision_links={"right_hand_thumb_2_link"},
    )

    assert "candidate_1" in message
    assert "right_shoulder_yaw_link/torso_link=12.919mm" in message
    assert "right_hand_thumb_2_link" not in message


def test_local_plane_guard_excludes_unlocated_table_geometry() -> None:
    for arm in ("left", "right"):
        links = _local_table_plane_links(arm)
        assert all(name.startswith(f"{arm}_") for name in links)
        assert not any("elbow" in name for name in links)
        assert not any("torso" in name or "hip" in name for name in links)


def test_open_transit_world_scope_keeps_only_selected_local_geometry() -> None:
    strict = {
        "kinematics": {
            "collision_sphere_buffer": 0.0,
            "collision_spheres": {
                "left_wrist_pitch_link": [{"radius": 0.020}],
                "left_hand_palm_link": [{"radius": 0.015}],
                "left_elbow_link": [{"radius": 0.030}],
                "right_hand_palm_link": [{"radius": 0.025}],
                "torso_link": [{"radius": 0.100}],
            },
        }
    }

    scoped = _selected_open_transit_world_robot(strict, arm="left")
    buffers = scoped["kinematics"]["collision_sphere_buffer"]

    assert buffers["left_wrist_pitch_link"] == pytest.approx(0.0)
    assert buffers["left_hand_palm_link"] == pytest.approx(0.0)
    assert buffers["left_elbow_link"] < -0.030
    assert buffers["right_hand_palm_link"] < -0.025
    assert buffers["torso_link"] < -0.100
    assert strict["kinematics"]["collision_sphere_buffer"] == 0.0


def test_attached_cube_spheres_conservatively_cover_every_subcell_corner() -> None:
    dimensions = np.asarray([0.040, 0.040, 0.040])
    spheres = _cuboid_cover_spheres(tuple(dimensions))

    assert spheres.shape == (27, 4)
    # A sphere circumscribes each of the 27 grid cells, so all eight corners
    # of every subcell—and therefore the complete cuboid—are covered.
    boundaries = [np.linspace(-dimension / 2.0, dimension / 2.0, 4) for dimension in dimensions]
    for x in boundaries[0]:
        for y in boundaries[1]:
            for z in boundaries[2]:
                distances = np.linalg.norm(spheres[:, :3] - [x, y, z], axis=1)
                assert np.min(distances - spheres[:, 3]) <= 1e-12


def test_local_plane_clearance_uses_sphere_surface_not_center() -> None:
    torch = __import__("pytest").importorskip("torch")
    __import__("pytest").importorskip("curobo")
    from curobo.types import DeviceCfg

    class KinematicsConfig:
        @staticmethod
        def get_sphere_index_from_link_name(_name):
            return torch.tensor([0])

    class Planner:
        device_cfg = DeviceCfg(device=torch.device("cpu"), dtype=torch.float32)
        kinematics = SimpleNamespace(config=SimpleNamespace(kinematics_config=KinematicsConfig()))

        @staticmethod
        def compute_kinematics(state):
            count = state.position.shape[0]
            spheres = torch.zeros((count, 1, 4), dtype=torch.float32)
            spheres[:, 0, 2] = torch.tensor([0.10, 0.02])[:count]
            spheres[:, 0, 3] = 0.01
            return SimpleNamespace(robot_spheres=spheres)

    clearance, link, sample = _local_plane_clearance(
        Planner(),
        np.zeros((2, 7)),
        arm="left",
        plane_point=np.zeros(3),
        down=np.array([0.0, 0.0, -1.0]),
        include_payload=False,
    )
    assert np.isclose(clearance, 0.01)
    assert link in _local_table_plane_links("left")
    assert sample == 1


@pytest.mark.parametrize(
    ("arm", "joint_slice"), (("left", slice(15, 22)), ("right", slice(22, 29)))
)
def test_clearance_request_uses_exact_escape_endpoint(arm, joint_slice) -> None:
    source = _observation()
    from g1_dex3_tabletop.tabletop_contracts import TabletopTaskRequest

    loaded = TabletopTaskRequest(
        source,
        arm,
        tuple(tuple(row) for row in _identity()),
        {},
        "e" * 64,
        "config/tabletop/x.yaml",
        "f" * 64,
    )
    outbound = PlannedTrajectory(
        "__handoff__",
        "clearance",
        (0.0, 1.0),
        ((0.0,) * 7, (0.2,) * 7),
        ((0.0,) * 7, (0.2,) * 7),
        1.0,
    )
    inbound = PlannedTrajectory(
        "clearance",
        "__handoff__",
        (0.0, 1.0),
        ((0.2,) * 7, (0.0,) * 7),
        ((0.2,) * 7, (0.0,) * 7),
        0.0,
    )
    escape = SupportedEscapePlan(loaded.content_sha256, outbound, inbound, 0.1, {})
    result = request_at_clearance(loaded, escape)
    assert result.observation.snapshot.measured_q29_rad[joint_slice] == (0.2,) * 7


def test_clearance_replan_uses_fresh_cube_and_body_but_exact_arm_endpoint() -> None:
    source = _observation()
    from g1_dex3_tabletop.tabletop_contracts import TabletopTaskRequest

    loaded = TabletopTaskRequest(
        source,
        "left",
        tuple(tuple(row) for row in _identity()),
        {},
        "e" * 64,
        "config/tabletop/x.yaml",
        "f" * 64,
    )
    outbound = PlannedTrajectory(
        "__handoff__",
        "clearance",
        (0.0, 1.0),
        ((0.0,) * 7, (0.2,) * 7),
        ((0.0,) * 7, (0.2,) * 7),
        1.0,
    )
    inbound = PlannedTrajectory(
        "clearance",
        "__handoff__",
        (0.0, 1.0),
        ((0.2,) * 7, (0.0,) * 7),
        ((0.2,) * 7, (0.0,) * 7),
        0.0,
    )
    escape = SupportedEscapePlan(loaded.content_sha256, outbound, inbound, 0.1, {})
    fresh_q = np.linspace(-0.2, 0.2, 29)
    fresh_pose = np.eye(4)
    fresh_pose[0, 3] = 0.01
    fresh = TabletopObservation(
        RobotSnapshot(tuple(fresh_q), (0.3,) * 7, (0.4,) * 7),
        tuple(tuple(row) for row in fresh_pose),
        source.camera_profile_sha256,
        ("1" * 64, "2" * 64, "3" * 64),
        0.2,
        0.1,
    )

    result = request_at_clearance_observation(loaded, escape, fresh)

    assert result.observation.snapshot.measured_q29_rad[15:22] == (0.2,) * 7
    assert result.observation.snapshot.measured_q29_rad[:15] == tuple(fresh_q[:15])
    assert result.observation.snapshot.right_dex3_q_rad == (0.4,) * 7
    assert result.observation.camera_T_object == fresh.camera_T_object
    assert result.observation.source_frame_sha256 == fresh.source_frame_sha256


def test_fixed_cube_anchor_reports_camera_motion_in_cube_frame() -> None:
    reference = np.eye(4)
    current = np.eye(4)
    # Moving the camera +10 mm along cube X makes the cube appear at -10 mm.
    current[0, 3] = -0.010

    result = camera_motion_from_fixed_cube(reference, current)

    assert result["anchor"] == "fixed_tabletop_aprilcube"
    assert result["translation_norm_mm"] == pytest.approx(10.0)
    assert result["translation_object_xyz_mm"] == pytest.approx([10.0, 0.0, 0.0])
