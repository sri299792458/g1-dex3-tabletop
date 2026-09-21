import json
import os
from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np
import pytest
import yaml
from aprilcube import CorrespondenceDetector
from aprilcube.generate import DICT_MAP

from g1_aprilcube_calibration.authored_collection import (
    AuthoredCollectionPlan,
    AuthoredPoseTarget,
)
from g1_aprilcube_calibration.camera_models import RectifiedCameraInfo
from g1_aprilcube_calibration.capture_diagnostics import supported_vs_held_metrics
from g1_aprilcube_calibration.collision import CollisionConfig
from g1_aprilcube_calibration.dataset_builder import (
    CalibrationDataset,
    CalibrationSample,
    DatasetBuilder,
    deterministic_holdout_split,
)
from g1_aprilcube_calibration.models import RobotStateSample
from g1_aprilcube_calibration.pose_schema import (
    HANDOFF_POSE_ID,
    PoseAuditEvent,
    PoseRecord,
    PoseSet,
)
from g1_aprilcube_calibration.pose_validator import (
    DirectedEdgeResult,
    ValidationReport,
)
from g1_aprilcube_calibration.quality import QualityGrade, QualityReport
from g1_aprilcube_calibration.readiness import RecordingGateConfig
from g1_aprilcube_calibration.session_store import (
    CaptureFrameInput,
    IsolatedSessionStore,
    SessionStore,
)
from g1_aprilcube_calibration.timestamp_pairing import (
    ImageTiming,
    PairingConfig,
    pair_state_to_image,
)

ROOT = Path(__file__).parents[1]
TARGET = ROOT / "third_party" / "aprilcube" / "models" / "dex3_safe_cube" / "config.json"
COLLISIONS = ROOT / "tests" / "fixtures" / "collision_pairs.yaml"
QUALITY = ROOT / "config" / "capture_quality_dex3_aruco.yaml"
UTC = "2026-08-02T12:00:00Z"


def camera_info() -> RectifiedCameraInfo:
    return RectifiedCameraInfo(
        width=640,
        height=480,
        frame_id="camera_color_optical_frame",
        camera_name="head_color",
        serial_number="TEST123",
        distortion_model="plumb_bob",
        d=(0.0,) * 5,
        k=(600.0, 0.0, 319.5, 0.0, 600.0, 239.5, 0.0, 0.0, 1.0),
        r=(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0),
        p=(600.0, 0.0, 319.5, 0.0, 0.0, 600.0, 239.5, 0.0, 0.0, 0.0, 1.0, 0.0),
    )


def marker_image() -> np.ndarray:
    image = np.full((480, 640, 3), 220, dtype=np.uint8)
    dictionary = cv2.aruco.getPredefinedDictionary(DICT_MAP["4x4_100"])
    marker = cv2.aruco.generateImageMarker(dictionary, 0, 180)
    image[140:320, 230:410] = cv2.cvtColor(marker, cv2.COLOR_GRAY2BGR)
    return image


def quality() -> QualityReport:
    return QualityReport(
        grade=QualityGrade.GREEN,
        hard_failures=(),
        warnings=(),
        metrics={"tag_count": 1},
        signature=None,
        pose_diagnostic=None,
    )


def frame(frame_id: str, center_s: float) -> CaptureFrameInput:
    timing = ImageTiming(center_s + 0.04, UTC, int(center_s * 1e9))
    states = tuple(
        RobotStateSample(
            center_s + offset,
            UTC,
            5,
            np.arange(29, dtype=float) / 100.0,
            np.zeros(29),
            np.arange(29, dtype=float) / 1000.0,
        )
        for offset in (0.0, 0.05, 0.1)
    )
    pairing_config = PairingConfig(0.05, 0.11)
    image = marker_image()
    return CaptureFrameInput(
        frame_id=frame_id,
        image_bgr=image,
        image_timing=timing,
        camera_info=camera_info(),
        state_window=states,
        pairing=pair_state_to_image(timing, states, config=pairing_config),
        correspondences=CorrespondenceDetector(TARGET).detect(image),
        quality=quality(),
    )


def create_store(tmp_path) -> SessionStore:
    store = SessionStore(tmp_path / "session_001")
    pose_set = PoseSet(
        robot_model="g1_29dof_rev_1_0",
        mode_machine=5,
        urdf_sha256="a" * 64,
        calibration_arm="left",
    )
    collision = CollisionConfig.from_yaml(COLLISIONS)
    validation = ValidationReport(
        pose_set_sha256=pose_set.content_sha256,
        urdf_sha256=pose_set.urdf_sha256,
        collision_config_sha256=collision.content_sha256,
        reference_full_q_sha256="c" * 64,
        config={},
        edges=(
            DirectedEdgeResult(
                from_pose_id="__handoff__",
                to_pose_id="pose_001",
                passed=True,
                sample_count=2,
                path_length_rad=0.1,
                estimated_duration_s=0.5,
                minimum_clearance_m=0.1,
                minimum_clearance_pair=("hand", "torso"),
                failures=(),
            ),
        ),
    )
    store.create(
        session_id="session_001",
        created_at_utc=UTC,
        camera_info=camera_info(),
        pose_set_content_sha256=pose_set.content_sha256,
        artifacts={
            "pose_set.yaml": yaml.safe_dump(pose_set.to_dict(), sort_keys=False).encode(),
            "hardware.yaml": b"robot: test\n",
            "target.json": TARGET.read_bytes(),
            "collision_pairs.yaml": COLLISIONS.read_bytes(),
            "capture_quality.yaml": QUALITY.read_bytes(),
            "validation_report.json": json.dumps(validation.to_dict()).encode(),
        },
        pairing_config=PairingConfig(0.05, 0.11),
        recording_gate_config=RecordingGateConfig(
            calibration_arm="left",
            state_freshness_timeout_s=0.1,
            stationary_duration_s=0.1,
            maximum_state_gap_s=0.06,
            maximum_calibration_position_spread_rad=0.01,
            minimum_samples=3,
        ),
        provenance={"git_commit": "test", "head_witness_ack": True},
    )
    return store


def test_authored_automatic_session_builds_dataset_from_measured_frames_only(
    tmp_path,
) -> None:
    transform = np.eye(4)
    transform[2, 3] = 0.4
    target = AuthoredPoseTarget(
        id="auto_001",
        authored_calibration_q=(0.1,) * 7,
        desired_camera_T_cube=transform,
    )
    plan = AuthoredCollectionPlan(
        robot_model="g1_29dof_rev_1_0",
        mode_machine=5,
        urdf_sha256="a" * 64,
        calibration_arm="left",
        camera_profile_sha256=camera_info().profile_sha256,
        reference_full_q_sha256="b" * 64,
        targets=(target,),
        route_pose_ids=(HANDOFF_POSE_ID, target.id, HANDOFF_POSE_ID),
        capture_pose_ids=(target.id,),
        generation_config={"source": "camera_frustum"},
    )
    collision = CollisionConfig.from_yaml(COLLISIONS)
    validation = ValidationReport(
        pose_set_sha256=plan.content_sha256,
        urdf_sha256=plan.urdf_sha256,
        collision_config_sha256=collision.content_sha256,
        reference_full_q_sha256="c" * 64,
        config={},
        edges=(
            DirectedEdgeResult(
                from_pose_id=HANDOFF_POSE_ID,
                to_pose_id=target.id,
                passed=True,
                sample_count=2,
                path_length_rad=0.1,
                estimated_duration_s=0.5,
                minimum_clearance_m=0.1,
                minimum_clearance_pair=("hand", "torso"),
                failures=(),
            ),
            DirectedEdgeResult(
                from_pose_id=target.id,
                to_pose_id=HANDOFF_POSE_ID,
                passed=True,
                sample_count=2,
                path_length_rad=0.1,
                estimated_duration_s=0.5,
                minimum_clearance_m=0.1,
                minimum_clearance_pair=("hand", "torso"),
                failures=(),
            ),
        ),
    )
    store = SessionStore(tmp_path / "auto_session")
    store.create(
        session_id="auto_session",
        created_at_utc=UTC,
        camera_info=camera_info(),
        pose_set_content_sha256=plan.content_sha256,
        artifacts={
            "authored_plan.json": json.dumps(plan.to_dict()).encode(),
            "hardware.yaml": b"robot: test\n",
            "target.json": TARGET.read_bytes(),
            "collision_pairs.yaml": COLLISIONS.read_bytes(),
            "capture_quality.yaml": QUALITY.read_bytes(),
            "validation_report.json": json.dumps(validation.to_dict()).encode(),
        },
        pairing_config=PairingConfig(0.05, 0.11),
        recording_gate_config=RecordingGateConfig(
            calibration_arm="left",
            state_freshness_timeout_s=0.1,
            stationary_duration_s=0.1,
            maximum_state_gap_s=0.06,
            maximum_calibration_position_spread_rad=0.01,
            minimum_samples=3,
        ),
        provenance={"command": "g1-calib collect-auto"},
        collection_method="authored_automatic",
    )
    store.append_capture(
        capture_id="capture_001",
        pose_id=target.id,
        outcome="accepted",
        reason="stationary burst passed",
        frames=(frame("capture_001_000", 1.0),),
    )
    store.finalize()

    dataset = DatasetBuilder(store.directory).build()

    assert dataset.pose_set_sha256 == plan.content_sha256
    assert dataset.samples[0].pose_id == target.id
    assert dataset.samples[0].measured_state["position"] != list(target.authored_calibration_q)


def manual_pose(pose_id: str = "pose_001") -> PoseRecord:
    full_q = tuple(float(value) / 100.0 for value in range(29))
    return PoseRecord(
        id=pose_id,
        group="calibration",
        measured_calibration_q=full_q[15:22],
        measured_full_q=full_q,
        calibration_q_spread=(0.001,) * 7,
        recorded_at_utc=UTC,
        recorded_monotonic_s=1.0,
    )


def create_manual_store(tmp_path) -> tuple[SessionStore, PoseSet]:
    store = SessionStore(tmp_path / "manual_session")
    pose_set = PoseSet(
        robot_model="g1_29dof_rev_1_0",
        mode_machine=5,
        urdf_sha256="a" * 64,
        calibration_arm="left",
    )
    store.create(
        session_id="manual_session",
        created_at_utc=UTC,
        camera_info=camera_info(),
        pose_set_content_sha256=pose_set.content_sha256,
        artifacts={
            "pose_set.yaml": yaml.safe_dump(pose_set.to_dict(), sort_keys=False).encode(),
            "hardware.yaml": b"robot: test\n",
            "target.json": TARGET.read_bytes(),
            "collision_pairs.yaml": COLLISIONS.read_bytes(),
            "capture_quality.yaml": QUALITY.read_bytes(),
        },
        pairing_config=PairingConfig(0.05, 0.11),
        recording_gate_config=RecordingGateConfig(
            calibration_arm="left",
            state_freshness_timeout_s=0.1,
            stationary_duration_s=0.1,
            maximum_state_gap_s=0.06,
            maximum_calibration_position_spread_rad=0.01,
            minimum_samples=3,
        ),
        provenance={"git_commit": "test"},
        collection_method="manual_teaching",
    )
    return store, pose_set


def test_raw_first_manifest_second_and_offline_rebuild(tmp_path) -> None:
    store = create_store(tmp_path)
    frames = tuple(frame(f"capture_001_{index}", float(index + 1)) for index in range(3))
    manifest = store.append_capture(
        capture_id="capture_001",
        pose_id="pose_001",
        outcome="accepted",
        reason="stationary burst passed",
        frames=frames,
        recorded_at_utc=UTC,
    )

    capture = manifest.captures[0]
    assert capture.selected_frame_id == "capture_001_0"
    assert len(capture.frames) == 3
    assert store.find_orphans() == ()
    assert all((store.directory / item.image_path).exists() for item in capture.frames)
    store.finalize()

    output = tmp_path / "dataset.json"
    dataset = DatasetBuilder(store.directory).build(output_path=output)
    sample = dataset.samples[0]
    assert dataset.calibration_arm == "left"
    assert dataset.schema_version == 3
    assert dataset.observation_phase == "held"
    assert sample.capture_id == "capture_001"
    assert sample.measured_state["estimated_torque"] == list(np.arange(29, dtype=float) / 1000.0)
    assert sample.visible_tag_ids == (0,)
    assert sample.corner_tag_ids == (0, 0, 0, 0)
    assert len(sample.image_points_px) == len(sample.object_points_m) == 4
    assert max(abs(value) for point in sample.object_points_m for value in point) < 0.1
    assert json.loads(output.read_text())["content_sha256"] == dataset.content_sha256


def test_isolated_writer_persists_and_finalizes_in_spawned_process(tmp_path) -> None:
    store = create_store(tmp_path)
    isolated = IsolatedSessionStore(store.directory, poll_interval_s=0.001)
    health_checks = []
    try:
        isolated.start()
        assert isolated.worker_pid is not None
        assert isolated.worker_pid != os.getpid()
        isolated.set_health_check(lambda: health_checks.append(True))
        isolated.append_capture(
            capture_id="capture_001",
            pose_id="pose_001",
            outcome="accepted",
            reason="stationary burst passed",
            frames=(frame("capture_001_000", 1.0),),
            recorded_at_utc=UTC,
        )
        isolated.finalize()
    finally:
        isolated.close()

    manifest = store.load()
    assert manifest.finalized
    assert manifest.captures[0].pose_id == "pose_001"
    assert health_checks


def test_manual_session_updates_undoes_resumes_and_builds(tmp_path) -> None:
    store, empty = create_manual_store(tmp_path)
    first_pose = manual_pose()
    with_first = empty.with_pose(
        first_pose,
        PoseAuditEvent("add", first_pose.id, UTC, {"source": "test"}),
    )
    store.update_manual_pose_set(with_first)
    with pytest.raises(ValueError, match="not aligned"):
        store.finalize()
    store.append_capture(
        capture_id="capture_001",
        pose_id=first_pose.id,
        outcome="accepted",
        reason="manual stationary burst passed",
        frames=(frame("capture_001_000", 1.0),),
        supported_frames=(frame("capture_001_supported_000", 0.5),),
        recorded_at_utc=UTC,
    )
    store.validate_manual_alignment(with_first)

    without_first = with_first.without_last_pose(
        PoseAuditEvent("undo", first_pose.id, UTC, {"reason": "operator undo"})
    )
    store.undo_manual_capture(
        capture_id="capture_001",
        updated_pose_set=without_first,
        reason="operator undo",
    )
    undone = store.load().captures[0]
    assert undone.selected_frame_id is None
    assert undone.selected_supported_frame_id is None
    store.validate_manual_alignment(without_first)

    replacement_pose = replace(first_pose, recorded_monotonic_s=2.0)
    final_poses = without_first.with_pose(
        replacement_pose,
        PoseAuditEvent("add", replacement_pose.id, UTC, {"source": "test"}),
    )
    store.update_manual_pose_set(final_poses)
    supported = frame("capture_002_supported_000", 1.5)
    held = frame("capture_002_000", 2.0)
    comparison = supported_vs_held_metrics(supported, held, calibration_arm="left")
    store.append_capture(
        capture_id="capture_002",
        pose_id=replacement_pose.id,
        outcome="accepted",
        reason="manual stationary burst passed",
        frames=(held,),
        supported_frames=(supported,),
        metadata={
            "capture_phase": "continuous_guide_to_weight_1_hold",
            "supported_vs_held": comparison,
        },
        recorded_at_utc=UTC,
    )

    resumed = SessionStore(store.directory)
    resumed.validate_manual_alignment(final_poses)
    manifest = resumed.finalize()
    dataset = DatasetBuilder(resumed.directory).build(observation_phase="held")
    supported_dataset = DatasetBuilder(resumed.directory).build(observation_phase="supported")
    assert manifest.provenance["collection_method"] == "manual_teaching"
    assert [capture.outcome for capture in manifest.captures] == [
        "rejected",
        "accepted",
    ]
    assert [sample.capture_id for sample in dataset.samples] == ["capture_002"]
    assert manifest.schema_version == 3
    assert manifest.captures[-1].metadata["capture_phase"] == ("continuous_guide_to_weight_1_hold")
    assert manifest.captures[-1].supported_frames
    assert manifest.captures[-1].supported_frames[0].frame_id == "capture_002_supported_000"
    assert dataset.samples[0].frame_id == "capture_002_000"
    assert supported_dataset.samples[0].frame_id == "capture_002_supported_000"
    assert supported_dataset.observation_phase == "supported"
    assert comparison["common_corner_count"] == 4
    assert comparison["corner_displacement_px"]["maximum"] == pytest.approx(0.0)
    assert store.find_orphans() == ()


def test_finalized_session_rejects_append(tmp_path) -> None:
    store = create_store(tmp_path)
    store.finalize()
    with pytest.raises(RuntimeError, match="finalized"):
        store.append_capture(
            capture_id="capture_001",
            pose_id="pose_001",
            outcome="skipped",
            reason="operator skipped",
        )


def test_manual_capture_requires_equal_supported_and_held_bursts(tmp_path) -> None:
    store, _ = create_manual_store(tmp_path)

    with pytest.raises(ValueError, match="equal supported and held bursts"):
        store.append_capture(
            capture_id="capture_001",
            pose_id="pose_001",
            outcome="accepted",
            reason="invalid unequal pair",
            frames=(frame("capture_001_000", 1.0),),
        )


def test_manual_capture_requires_supported_burst_before_held_burst(tmp_path) -> None:
    store, _ = create_manual_store(tmp_path)

    with pytest.raises(ValueError, match="supported burst to precede"):
        store.append_capture(
            capture_id="capture_001",
            pose_id="pose_001",
            outcome="accepted",
            reason="invalid phase order",
            frames=(frame("capture_001_000", 1.0),),
            supported_frames=(frame("capture_001_supported_000", 1.5),),
        )


def test_accepted_capture_rejects_red_or_invalid_frames(tmp_path) -> None:
    store = create_store(tmp_path)
    valid = frame("frame_001", 1.0)
    red = CaptureFrameInput(
        frame_id="frame_red",
        image_bgr=valid.image_bgr,
        image_timing=valid.image_timing,
        camera_info=valid.camera_info,
        state_window=valid.state_window,
        pairing=valid.pairing,
        correspondences=valid.correspondences,
        quality=QualityReport(QualityGrade.RED, ("bad",), (), {}, None, None),
    )
    with pytest.raises(ValueError, match="invalid/red"):
        store.append_capture(
            capture_id="capture_001",
            pose_id="pose_001",
            outcome="accepted",
            reason="should fail",
            frames=(red,),
        )
    assert store.load().captures == ()
    assert store.find_orphans() == ()


def test_builder_detects_orphans_and_corruption(tmp_path) -> None:
    store = create_store(tmp_path)
    item = frame("frame_001", 1.0)
    manifest = store.append_capture(
        capture_id="capture_001",
        pose_id="pose_001",
        outcome="accepted",
        reason="passed",
        frames=(item,),
        recorded_at_utc=UTC,
    )
    orphan = store.directory / "raw" / "images" / "orphan.png"
    orphan.write_bytes(b"orphan")
    store.finalize()
    with pytest.raises(ValueError, match="orphan"):
        DatasetBuilder(store.directory).build()
    orphan.unlink()

    image_path = store.directory / manifest.captures[0].frames[0].image_path
    image_path.chmod(0o644)
    image_path.write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="image hash"):
        DatasetBuilder(store.directory).build()


def test_builder_verifies_nonselected_raw_frames_too(tmp_path) -> None:
    store = create_store(tmp_path)
    manifest = store.append_capture(
        capture_id="capture_001",
        pose_id="pose_001",
        outcome="accepted",
        reason="passed",
        frames=(frame("frame_001", 1.0), frame("frame_002", 2.0)),
        recorded_at_utc=UTC,
    )
    store.finalize()
    nonselected = next(
        item
        for item in manifest.captures[0].frames
        if item.frame_id != manifest.captures[0].selected_frame_id
    )
    states_path = store.directory / nonselected.states_path
    states_path.chmod(0o644)
    states_path.write_bytes(b"[]")
    with pytest.raises(ValueError, match="states hash"):
        DatasetBuilder(store.directory).build()


def test_invalid_capture_metadata_writes_no_raw_files(tmp_path) -> None:
    store = create_store(tmp_path)
    with pytest.raises(ValueError, match="unsupported"):
        store.append_capture(
            capture_id="capture_001",
            pose_id="pose_001",
            outcome="mystery",
            reason="invalid",
            frames=(frame("frame_001", 1.0),),
            recorded_at_utc=UTC,
        )
    assert store.find_orphans() == ()


def test_ten_pose_session_resumes_and_rebuilds_deterministically(tmp_path) -> None:
    store = create_store(tmp_path)
    for index in range(5):
        store.append_capture(
            capture_id=f"capture_{index:03d}",
            pose_id=f"pose_{index:03d}",
            outcome="accepted",
            reason="synthetic pass",
            frames=(frame(f"frame_{index:03d}", float(index + 1)),),
            recorded_at_utc=UTC,
        )

    resumed = SessionStore(store.directory)
    assert len(resumed.load().captures) == 5
    for index in range(5, 10):
        resumed.append_capture(
            capture_id=f"capture_{index:03d}",
            pose_id=f"pose_{index:03d}",
            outcome="accepted",
            reason="synthetic pass after resume",
            frames=(frame(f"frame_{index:03d}", float(index + 1)),),
            recorded_at_utc=UTC,
        )
    resumed.finalize()

    first = DatasetBuilder(resumed.directory).build()
    second = DatasetBuilder(resumed.directory).build()
    assert len(first.samples) == 10
    assert first.content_sha256 == second.content_sha256


def test_session_manifest_hash_detects_manual_edit(tmp_path) -> None:
    store = create_store(tmp_path)
    data = json.loads(store.manifest_path.read_text())
    data["provenance"]["git_commit"] = "tampered"
    store.manifest_path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="content SHA-256"):
        store.load()


def test_rectified_camera_info_rejects_nonzero_distortion() -> None:
    data = camera_info().to_dict()
    data["d"][0] = 0.01
    with pytest.raises(ValueError, match="zero distortion"):
        RectifiedCameraInfo.from_dict(data)


def test_deterministic_holdout_split_preserves_source_order() -> None:
    samples = tuple(
        CalibrationSample(
            capture_id=f"capture_{index}",
            pose_id=f"pose_{index}",
            frame_id=f"frame_{index}",
            raw_image_path=f"raw/{index}.png",
            raw_image_sha256="a" * 64,
            camera_info={},
            measured_state={},
            pairing={},
            visible_tag_ids=(0,),
            corner_tag_ids=(0,),
            image_points_px=((0.0, 0.0),),
            object_points_m=((0.0, 0.0, 0.0),),
            correspondence_sha256="b" * 64,
        )
        for index in range(10)
    )
    dataset = CalibrationDataset(
        "session",
        "c" * 64,
        "d" * 64,
        "e" * 64,
        "f" * 64,
        "left",
        "held",
        samples,
    )
    training_a, holdout_a = deterministic_holdout_split(dataset)
    training_b, holdout_b = deterministic_holdout_split(dataset)
    assert training_a == training_b
    assert holdout_a == holdout_b
    assert len(training_a) == 8
    assert len(holdout_a) == 2
    assert [sample.capture_id for sample in training_a] == sorted(
        sample.capture_id for sample in training_a
    )
