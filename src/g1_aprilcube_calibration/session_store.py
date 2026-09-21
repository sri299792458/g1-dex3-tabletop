"""Crash-safe raw session writer with immutable finalized artifacts."""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeoutError
from dataclasses import dataclass, replace
from multiprocessing import get_context
from pathlib import Path

import cv2
import numpy as np
import yaml
from aprilcube import CorrespondenceResult

from g1_aprilcube_calibration.authored_collection import AuthoredCollectionPlan
from g1_aprilcube_calibration.camera_models import RectifiedCameraInfo
from g1_aprilcube_calibration.collision import CollisionConfig
from g1_aprilcube_calibration.correspondence import correspondence_sha256
from g1_aprilcube_calibration.models import (
    RobotStateSample,
    utc_now_iso,
    validate_utc_iso,
)
from g1_aprilcube_calibration.pose_schema import PoseSet
from g1_aprilcube_calibration.pose_validator import ValidationReport
from g1_aprilcube_calibration.quality import QualityGrade, QualityReport
from g1_aprilcube_calibration.readiness import (
    RecordingGateConfig,
    evaluate_recording_window,
)
from g1_aprilcube_calibration.session_models import (
    CAPTURE_OUTCOMES,
    CaptureRecord,
    RawFrameRecord,
    SessionManifest,
)
from g1_aprilcube_calibration.timestamp_pairing import (
    ImageTiming,
    PairingConfig,
    PairingResult,
    pair_state_to_image,
)

COMMON_SESSION_ARTIFACTS = frozenset(
    {
        "hardware.yaml",
        "target.json",
        "collision_pairs.yaml",
        "capture_quality.yaml",
    }
)
BASE_SESSION_ARTIFACTS = COMMON_SESSION_ARTIFACTS | {"pose_set.yaml"}
REPLAY_SESSION_ARTIFACTS = BASE_SESSION_ARTIFACTS | {"validation_report.json"}
AUTHORED_SESSION_ARTIFACTS = COMMON_SESSION_ARTIFACTS | {
    "authored_plan.json",
    "validation_report.json",
}
CUROBO_SESSION_ARTIFACTS = frozenset(
    {
        "hardware.yaml",
        "target.json",
        "capture_quality.yaml",
        "calibration_request.json",
        "calibration_plan.json",
    }
)
COLLECTION_METHODS = frozenset(
    {"replay", "manual_teaching", "authored_automatic", "curobo_automatic"}
)
_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,95}$")


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _canonical_json(data) -> bytes:
    return json.dumps(data, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


@dataclass(frozen=True, slots=True)
class CaptureFrameInput:
    frame_id: str
    image_bgr: np.ndarray
    image_timing: ImageTiming
    camera_info: RectifiedCameraInfo
    state_window: tuple[RobotStateSample, ...]
    pairing: PairingResult
    correspondences: CorrespondenceResult
    quality: QualityReport

    def __post_init__(self) -> None:
        if not _ID_PATTERN.fullmatch(self.frame_id):
            raise ValueError(f"invalid capture frame ID: {self.frame_id!r}")
        image = np.asarray(self.image_bgr)
        if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
            raise ValueError("raw rectified image must be uint8 BGR")
        if image.shape[:2] != (self.camera_info.height, self.camera_info.width):
            raise ValueError("raw image dimensions do not match CameraInfo")
        if self.correspondences.image_size_wh != (
            self.camera_info.width,
            self.camera_info.height,
        ):
            raise ValueError("correspondence dimensions do not match CameraInfo")
        if self.pairing.image != self.image_timing:
            raise ValueError("pairing result belongs to a different image")
        if not self.state_window:
            raise ValueError("capture frame requires its complete state window")
        object.__setattr__(self, "image_bgr", image.copy())
        self.image_bgr.setflags(write=False)
        object.__setattr__(self, "state_window", tuple(self.state_window))


class SessionStore:
    def __init__(self, session_directory: str | Path) -> None:
        self.directory = Path(session_directory)
        self.manifest_path = self.directory / "manifest.json"

    def create(
        self,
        *,
        session_id: str,
        created_at_utc: str,
        camera_info: RectifiedCameraInfo,
        pose_set_content_sha256: str,
        artifacts: Mapping[str, bytes],
        pairing_config: PairingConfig,
        recording_gate_config: RecordingGateConfig,
        provenance: dict,
        collection_method: str = "replay",
    ) -> SessionManifest:
        if self.directory.exists():
            raise FileExistsError(f"session directory already exists: {self.directory}")
        if collection_method not in COLLECTION_METHODS:
            raise ValueError(f"unsupported collection method: {collection_method}")
        if collection_method == "replay":
            expected_artifacts = REPLAY_SESSION_ARTIFACTS
        elif collection_method == "authored_automatic":
            expected_artifacts = AUTHORED_SESSION_ARTIFACTS
        elif collection_method == "curobo_automatic":
            expected_artifacts = CUROBO_SESSION_ARTIFACTS
        else:
            expected_artifacts = BASE_SESSION_ARTIFACTS
        if set(artifacts) != expected_artifacts:
            raise ValueError(
                "session artifacts must be exactly: " + ", ".join(sorted(expected_artifacts))
            )
        self._validate_source_artifacts(
            artifacts,
            expected_pose_set_content_sha256=pose_set_content_sha256,
            collection_method=collection_method,
        )
        resolved_provenance = dict(provenance)
        existing_method = resolved_provenance.get("collection_method")
        if existing_method is not None and existing_method != collection_method:
            raise ValueError("provenance collection method is inconsistent")
        resolved_provenance["collection_method"] = collection_method
        self.directory.mkdir(parents=True)
        (self.directory / "raw" / "images").mkdir(parents=True)
        (self.directory / "raw" / "states").mkdir(parents=True)
        (self.directory / "preview").mkdir(parents=True)
        artifact_hashes: dict[str, str] = {}
        for name, content in artifacts.items():
            if not isinstance(content, bytes):
                raise TypeError(f"session artifact {name} must be bytes")
            self._write_new_file(self.directory / name, content)
            artifact_hashes[name] = _sha256(content)
        manifest = SessionManifest(
            session_id=session_id,
            created_at_utc=created_at_utc,
            camera_profile_sha256=camera_info.profile_sha256,
            pose_set_content_sha256=pose_set_content_sha256,
            artifact_sha256=artifact_hashes,
            pairing_config={
                "maximum_nearest_delta_s": pairing_config.maximum_nearest_delta_s,
                "maximum_bracket_span_s": pairing_config.maximum_bracket_span_s,
            },
            recording_gate_config={
                name: getattr(recording_gate_config, name)
                for name in recording_gate_config.__dataclass_fields__
            },
            provenance=resolved_provenance,
        )
        self._write_manifest(manifest, first=True)
        return manifest

    def load(self) -> SessionManifest:
        with self.manifest_path.open(encoding="utf-8") as stream:
            data = json.load(stream)
        return SessionManifest.from_dict(data)

    def append_capture(
        self,
        *,
        capture_id: str,
        pose_id: str,
        outcome: str,
        reason: str,
        frames: Sequence[CaptureFrameInput] = (),
        supported_frames: Sequence[CaptureFrameInput] = (),
        metadata: dict | None = None,
        recorded_at_utc: str | None = None,
    ) -> SessionManifest:
        manifest = self.load()
        if manifest.finalized:
            raise RuntimeError("cannot append to a finalized session")
        if any(item.capture_id == capture_id for item in manifest.captures):
            raise ValueError(f"duplicate capture ID: {capture_id}")
        if not _ID_PATTERN.fullmatch(capture_id) or not _ID_PATTERN.fullmatch(pose_id):
            raise ValueError("capture and pose IDs must use safe identifier characters")
        if outcome not in CAPTURE_OUTCOMES:
            raise ValueError(f"unsupported capture outcome: {outcome}")
        if not reason:
            raise ValueError("capture reason must be non-empty")
        capture_utc = recorded_at_utc or utc_now_iso()
        validate_utc_iso(capture_utc)
        frame_inputs = tuple(frames)
        supported_inputs = tuple(supported_frames)
        all_frame_inputs = (*frame_inputs, *supported_inputs)
        if len({frame.frame_id for frame in all_frame_inputs}) != len(all_frame_inputs):
            raise ValueError("capture input contains duplicate frame IDs")
        if any(
            frame.camera_info.profile_sha256 != manifest.camera_profile_sha256
            for frame in all_frame_inputs
        ):
            raise ValueError("camera profile changed during the session")
        pairing_config = PairingConfig(**manifest.pairing_config)
        gate_config = RecordingGateConfig(**manifest.recording_gate_config)
        for frame in all_frame_inputs:
            rebuilt_pairing = pair_state_to_image(
                frame.image_timing,
                frame.state_window,
                config=pairing_config,
            )
            if rebuilt_pairing.to_dict() != frame.pairing.to_dict():
                raise ValueError("provided image/state pairing is not reproducible")
            readiness = evaluate_recording_window(
                frame.state_window,
                now_monotonic_s=frame.state_window[-1].receipt_monotonic_s,
                config=gate_config,
            )
            if not readiness.ready:
                raise ValueError(
                    "capture frame state window is not stationary: "
                    + "; ".join(readiness.hard_failures)
                )
        selected_frame_id = None
        selected_supported_frame_id = None
        if outcome == "accepted":
            if not frame_inputs:
                raise ValueError("accepted capture requires raw frames")
            if manifest.provenance.get("collection_method") == "manual_teaching" and len(
                supported_inputs
            ) != len(frame_inputs):
                raise ValueError(
                    "accepted manual capture requires equal supported and held bursts"
                )
            if (
                manifest.provenance.get("collection_method") == "manual_teaching"
                and supported_inputs
                and max(frame.image_timing.receipt_monotonic_s for frame in supported_inputs)
                >= min(frame.image_timing.receipt_monotonic_s for frame in frame_inputs)
            ):
                raise ValueError(
                    "accepted manual capture requires the complete supported burst "
                    "to precede the held burst"
                )
            if any(
                frame.quality.grade is QualityGrade.RED or not frame.correspondences.valid
                for frame in frame_inputs
            ):
                raise ValueError("accepted capture contains an invalid/red raw frame")
            selected_frame_id = self.select_medoid_frame(frame_inputs).frame_id
            if supported_inputs:
                if any(
                    frame.quality.grade is QualityGrade.RED or not frame.correspondences.valid
                    for frame in supported_inputs
                ):
                    raise ValueError("accepted capture contains an invalid/red supported frame")
                selected_supported_frame_id = self.select_medoid_frame(supported_inputs).frame_id

        existing_frame_ids = {
            frame.frame_id for capture in manifest.captures for frame in capture.raw_frames
        }
        if any(frame.frame_id in existing_frame_ids for frame in all_frame_inputs):
            raise ValueError("session already contains one of the raw frame IDs")

        records: list[RawFrameRecord] = []
        for frame in frame_inputs:
            records.append(self._write_raw_frame(frame))
        supported_records = tuple(self._write_raw_frame(frame) for frame in supported_inputs)
        capture = CaptureRecord(
            capture_id=capture_id,
            pose_id=pose_id,
            outcome=outcome,
            reason=reason,
            recorded_at_utc=capture_utc,
            metadata={} if metadata is None else metadata,
            frames=tuple(records),
            supported_frames=supported_records,
            selected_supported_frame_id=selected_supported_frame_id,
            selected_frame_id=selected_frame_id,
        )
        updated = replace(manifest, captures=(*manifest.captures, capture))
        self._write_manifest(updated)
        return updated

    def finalize(self) -> SessionManifest:
        manifest = self.load()
        if manifest.finalized:
            return manifest
        if manifest.provenance.get("collection_method") == "manual_teaching":
            pose_set = PoseSet.from_dict(
                yaml.safe_load((self.directory / "pose_set.yaml").read_bytes())
            )
            self._validate_manual_alignment(manifest, pose_set)
        updated = replace(manifest, finalized=True)
        self._write_manifest(updated)
        for path in self.directory.rglob("*"):
            if path.is_file():
                path.chmod(0o444)
        return updated

    def update_manual_pose_set(self, pose_set: PoseSet) -> SessionManifest:
        """Replace the evolving pose snapshot of an unfinalized manual session."""

        manifest = self.load()
        self._require_manual_session(manifest)
        self._validate_pose_set_identity(pose_set)
        accepted_pose_ids = {
            capture.pose_id for capture in manifest.captures if capture.outcome == "accepted"
        }
        recorded_pose_ids = {pose.id for pose in pose_set.poses}
        missing = sorted(accepted_pose_ids - recorded_pose_ids)
        if missing:
            raise ValueError(
                "manual pose update would orphan accepted captures: " + ", ".join(missing)
            )
        return self._replace_manual_pose_set(manifest, pose_set)

    def validate_manual_alignment(self, pose_set: PoseSet) -> None:
        """Require a one-to-one active-pose/accepted-capture relationship."""

        manifest = self.load()
        self._require_manual_session(manifest)
        self._validate_pose_set_identity(pose_set)
        self._validate_manual_alignment(manifest, pose_set)

    def undo_manual_capture(
        self,
        *,
        capture_id: str,
        updated_pose_set: PoseSet,
        reason: str,
    ) -> SessionManifest:
        """Reject one accepted manual capture while replacing its pose snapshot."""

        if not reason.strip():
            raise ValueError("manual capture undo reason must be non-empty")
        manifest = self.load()
        self._require_manual_session(manifest)
        self._validate_pose_set_identity(updated_pose_set)
        matches = [
            index
            for index, capture in enumerate(manifest.captures)
            if capture.capture_id == capture_id
        ]
        if len(matches) != 1:
            raise ValueError("manual capture ID is not present exactly once")
        index = matches[0]
        capture = manifest.captures[index]
        if capture.outcome != "accepted":
            raise ValueError("only an accepted manual capture can be undone")
        replacement = replace(
            capture,
            outcome="rejected",
            reason=reason.strip(),
            selected_supported_frame_id=None,
            selected_frame_id=None,
        )
        captures = list(manifest.captures)
        captures[index] = replacement
        accepted_pose_ids = {item.pose_id for item in captures if item.outcome == "accepted"}
        recorded_pose_ids = {pose.id for pose in updated_pose_set.poses}
        missing = sorted(accepted_pose_ids - recorded_pose_ids)
        if missing:
            raise ValueError(
                "manual undo would orphan other accepted captures: " + ", ".join(missing)
            )
        return self._replace_manual_pose_set(
            replace(manifest, captures=tuple(captures)), updated_pose_set
        )

    def find_orphans(self) -> tuple[str, ...]:
        manifest = self.load()
        referenced = {
            frame.image_path for capture in manifest.captures for frame in capture.raw_frames
        } | {frame.states_path for capture in manifest.captures for frame in capture.raw_frames}
        actual = {
            path.relative_to(self.directory).as_posix()
            for path in (self.directory / "raw").rglob("*")
            if path.is_file()
        }
        return tuple(sorted(actual - referenced))

    def verify_artifacts(self) -> None:
        manifest = self.load()
        for name, expected in manifest.artifact_sha256.items():
            content = (self.directory / name).read_bytes()
            if _sha256(content) != expected:
                raise ValueError(f"frozen session artifact hash mismatch: {name}")

    def _write_raw_frame(self, frame: CaptureFrameInput) -> RawFrameRecord:
        success, encoded = cv2.imencode(".png", frame.image_bgr)
        if not success:
            raise RuntimeError("OpenCV failed to encode raw PNG")
        image_bytes = encoded.tobytes()
        states_bytes = _canonical_json([sample.to_dict() for sample in frame.state_window])
        image_path = f"raw/images/{frame.frame_id}.png"
        states_path = f"raw/states/{frame.frame_id}.json"
        self._write_new_file(self.directory / image_path, image_bytes)
        # If the second write fails, the image intentionally remains as a
        # detectable orphan; recovery never silently overwrites it.
        self._write_new_file(self.directory / states_path, states_bytes)
        return RawFrameRecord(
            frame_id=frame.frame_id,
            image_path=image_path,
            image_sha256=_sha256(image_bytes),
            states_path=states_path,
            states_sha256=_sha256(states_bytes),
            image_timing={
                "receipt_monotonic_s": frame.image_timing.receipt_monotonic_s,
                "receipt_utc": frame.image_timing.receipt_utc,
                "header_stamp_ns": frame.image_timing.header_stamp_ns,
            },
            camera_info=frame.camera_info.to_dict(),
            pairing=frame.pairing.to_dict(),
            quality=frame.quality.to_dict(),
            live_correspondence_sha256=correspondence_sha256(frame.correspondences),
        )

    @staticmethod
    def _validate_source_artifacts(
        artifacts: Mapping[str, bytes],
        *,
        expected_pose_set_content_sha256: str,
        collection_method: str,
    ) -> None:
        pose_set = None
        authored_plan = None
        if collection_method == "curobo_automatic":
            try:
                from g1_dex3_tabletop.planning.contracts import (
                    CalibrationPlanRequest,
                    CalibrationPlanResult,
                )

                request = CalibrationPlanRequest.from_dict(
                    json.loads(artifacts["calibration_request.json"])
                )
                plan = CalibrationPlanResult.from_dict(
                    json.loads(artifacts["calibration_plan.json"])
                )
            except Exception as error:
                raise ValueError(
                    "CuRobo calibration request/plan artifacts are invalid"
                ) from error
            if plan.request_sha256 != request.content_sha256:
                raise ValueError("CuRobo plan belongs to a different request")
            if plan.content_sha256 != expected_pose_set_content_sha256:
                raise ValueError("motion-definition hash does not match CuRobo plan")
            try:
                hardware = yaml.safe_load(artifacts["hardware.yaml"])
                target = json.loads(artifacts["target.json"])
                quality = yaml.safe_load(artifacts["capture_quality.yaml"])
            except Exception as error:
                raise ValueError(
                    "one or more frozen CuRobo session artifacts cannot be parsed"
                ) from error
            if not all(isinstance(value, dict) for value in (hardware, target, quality)):
                raise TypeError("hardware, target, and capture-quality artifacts must be mappings")
            return
        if collection_method == "authored_automatic":
            try:
                authored_plan = AuthoredCollectionPlan.from_dict(
                    json.loads(artifacts["authored_plan.json"])
                )
            except Exception as error:
                raise ValueError(
                    "authored_plan.json is not a valid content-hashed plan"
                ) from error
            if authored_plan.content_sha256 != expected_pose_set_content_sha256:
                raise ValueError("motion-definition content hash does not match authored plan")
        else:
            try:
                pose_document = yaml.safe_load(artifacts["pose_set.yaml"])
                pose_set = PoseSet.from_dict(pose_document)
            except Exception as error:
                raise ValueError("pose_set.yaml is not a valid content-hashed pose set") from error
            if pose_set.content_sha256 != expected_pose_set_content_sha256:
                raise ValueError("pose-set content hash does not match pose_set.yaml")
        try:
            hardware = yaml.safe_load(artifacts["hardware.yaml"])
            target = json.loads(artifacts["target.json"])
            collision_document = yaml.safe_load(artifacts["collision_pairs.yaml"])
            quality_document = yaml.safe_load(artifacts["capture_quality.yaml"])
        except Exception as error:
            raise ValueError("one or more frozen session artifacts cannot be parsed") from error
        if (
            not isinstance(hardware, dict)
            or not isinstance(target, dict)
            or not isinstance(quality_document, dict)
        ):
            raise TypeError(
                "hardware, target, and capture-quality artifacts must contain mappings"
            )
        collision = CollisionConfig.from_mapping(collision_document)
        if not collision.hardware_ready:
            raise ValueError("session cannot freeze an unready collision config")
        if collection_method in {"replay", "authored_automatic"}:
            try:
                validation_document = json.loads(artifacts["validation_report.json"])
            except Exception as error:
                raise ValueError("validation report artifact cannot be parsed") from error
            if not isinstance(validation_document, dict):
                raise TypeError("validation report artifact must contain a mapping")
            validation = ValidationReport.from_dict(validation_document)
            motion_definition = authored_plan if authored_plan is not None else pose_set
            if validation.pose_set_sha256 != motion_definition.content_sha256:
                raise ValueError("validation report belongs to a different motion definition")
            if validation.urdf_sha256 != motion_definition.urdf_sha256:
                raise ValueError("validation report belongs to a different URDF")
            if not validation.passed:
                raise ValueError("session cannot freeze a failed transition report")
            if validation.collision_config_sha256 != collision.content_sha256:
                raise ValueError(
                    "validation report belongs to a different collision configuration"
                )

    def _require_manual_session(self, manifest: SessionManifest) -> None:
        if manifest.finalized:
            raise RuntimeError("cannot modify a finalized session")
        if manifest.provenance.get("collection_method") != "manual_teaching":
            raise RuntimeError("pose-set updates require a manual-teaching session")
        if set(manifest.artifact_sha256) != BASE_SESSION_ARTIFACTS:
            raise RuntimeError("manual session artifact set is inconsistent")

    def _validate_pose_set_identity(self, pose_set: PoseSet) -> None:
        existing = PoseSet.from_dict(
            yaml.safe_load((self.directory / "pose_set.yaml").read_bytes())
        )
        identity_fields = (
            "robot_model",
            "mode_machine",
            "urdf_sha256",
            "calibration_arm",
        )
        if any(getattr(existing, field) != getattr(pose_set, field) for field in identity_fields):
            raise ValueError("manual pose update changed the robot/session identity")

    @staticmethod
    def _validate_manual_alignment(manifest: SessionManifest, pose_set: PoseSet) -> None:
        accepted_pose_ids = [
            capture.pose_id for capture in manifest.captures if capture.outcome == "accepted"
        ]
        active_pose_ids = [pose.id for pose in pose_set.poses]
        if accepted_pose_ids != active_pose_ids:
            raise ValueError(
                "manual session poses and accepted captures are not aligned: "
                f"poses={active_pose_ids}, captures={accepted_pose_ids}"
            )

    def _replace_manual_pose_set(
        self, manifest: SessionManifest, pose_set: PoseSet
    ) -> SessionManifest:
        pose_bytes = yaml.safe_dump(
            pose_set.to_dict(), sort_keys=False, allow_unicode=True
        ).encode()
        pose_hash = _sha256(pose_bytes)
        artifact_hashes = dict(manifest.artifact_sha256)
        artifact_hashes["pose_set.yaml"] = pose_hash
        updated = replace(
            manifest,
            pose_set_content_sha256=pose_set.content_sha256,
            artifact_sha256=artifact_hashes,
        )
        self._replace_file(self.directory / "pose_set.yaml", pose_bytes)
        self._write_manifest(updated)
        return updated

    @staticmethod
    def select_medoid_frame(
        frames: tuple[CaptureFrameInput, ...],
    ) -> CaptureFrameInput:
        signatures: dict[tuple[int, ...], list[CaptureFrameInput]] = {}
        for frame in frames:
            signatures.setdefault(frame.correspondences.tag_ids, []).append(frame)
        candidates = max(
            signatures.values(),
            key=lambda group: (len(group), len(group[0].correspondences.tag_ids)),
        )
        vectors = np.asarray(
            [
                np.vstack(
                    [
                        observation.image_corners_px
                        for observation in frame.correspondences.observations
                    ]
                ).reshape(-1)
                for frame in candidates
            ]
        )
        median = np.median(vectors, axis=0)
        distances = np.linalg.norm(vectors - median, axis=1)
        index = min(range(len(candidates)), key=lambda item: (distances[item], item))
        return candidates[index]

    def _write_manifest(self, manifest: SessionManifest, *, first: bool = False) -> None:
        data = json.dumps(manifest.to_dict(), indent=2, sort_keys=True).encode() + b"\n"
        if first:
            self._write_new_file(self.manifest_path, data)
            return
        self._replace_file(self.manifest_path, data)

    @staticmethod
    def _write_new_file(path: Path, data: bytes) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
        )
        temporary = Path(temporary_name)
        try:
            with os.fdopen(descriptor, "wb") as stream:
                stream.write(data)
                stream.flush()
                os.fsync(stream.fileno())
            try:
                os.link(temporary, path)
            except FileExistsError:
                raise FileExistsError(f"immutable artifact already exists: {path}") from None
            SessionStore._fsync_directory(path.parent)
        finally:
            temporary.unlink(missing_ok=True)

    @staticmethod
    def _replace_file(path: Path, data: bytes) -> None:
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
        )
        temporary = Path(temporary_name)
        try:
            with os.fdopen(descriptor, "wb") as stream:
                stream.write(data)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, path)
            SessionStore._fsync_directory(path.parent)
        finally:
            temporary.unlink(missing_ok=True)

    @staticmethod
    def _fsync_directory(path: Path) -> None:
        descriptor = os.open(path, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)


def _session_writer_ready(store_factory) -> int:
    """Return the worker PID after the spawned writer has imported this module."""

    # Passing the factory also imports its module in the spawned worker before
    # control ownership. The bilateral store uses the same process lifecycle.
    del store_factory
    return os.getpid()


def _append_capture_isolated(store_factory, directory: str, arguments: dict) -> str:
    manifest = store_factory(directory).append_capture(**arguments)
    return manifest.content_sha256


def _create_session_isolated(store_factory, directory: str, arguments: dict) -> str:
    manifest = store_factory(directory).create(**arguments)
    return manifest.content_sha256


def _finalize_session_isolated(store_factory, directory: str) -> str:
    manifest = store_factory(directory).finalize()
    return manifest.content_sha256


class IsolatedSessionStore:
    """Persist captures in one prewarmed process outside control scheduling.

    The main process only submits immutable capture inputs and polls control
    health. OpenCV encoding, state-window validation/JSON serialization,
    fsync, and manifest replacement all execute in the writer process.
    """

    def __init__(
        self,
        directory: str | Path,
        *,
        poll_interval_s: float = 0.004,
        store_factory=SessionStore,
    ) -> None:
        if poll_interval_s <= 0:
            raise ValueError("writer poll interval must be positive")
        self.directory = Path(directory)
        self._store_factory = store_factory
        self.poll_interval_s = poll_interval_s
        self._health_check: Callable[[], None] | None = None
        self._pool = ProcessPoolExecutor(
            max_workers=1,
            mp_context=get_context("spawn"),
        )
        self._started = False
        self._closed = False
        self._worker_pid: int | None = None

    @property
    def worker_pid(self) -> int | None:
        return self._worker_pid

    def start(self) -> None:
        if self._closed:
            raise RuntimeError("isolated session writer is closed")
        if self._started:
            return
        self._worker_pid = int(
            self._wait(self._pool.submit(_session_writer_ready, self._store_factory))
        )
        if self._worker_pid == os.getpid():
            raise RuntimeError("session writer did not start in a separate process")
        self._started = True

    def set_health_check(self, health_check: Callable[[], None]) -> None:
        if not callable(health_check):
            raise TypeError("writer health check must be callable")
        self._health_check = health_check

    def append_capture(self, **arguments) -> str:
        """Delegate the selected store's capture schema to the isolated writer."""

        self._require_ready()
        for name in ("frames", "supported_frames"):
            if name in arguments:
                arguments[name] = tuple(arguments[name])
        return str(
            self._wait(
                self._pool.submit(
                    _append_capture_isolated,
                    self._store_factory,
                    str(self.directory),
                    arguments,
                )
            )
        )

    def create(self, **arguments) -> str:
        """Freeze loaded-plan artifacts outside the active control process."""

        self._require_ready()
        return str(
            self._wait(
                self._pool.submit(
                    _create_session_isolated,
                    self._store_factory,
                    str(self.directory),
                    arguments,
                )
            )
        )

    def finalize(self) -> str:
        self._require_ready()
        return str(
            self._wait(
                self._pool.submit(
                    _finalize_session_isolated,
                    self._store_factory,
                    str(self.directory),
                )
            )
        )

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._pool.shutdown(wait=True, cancel_futures=False)

    def _require_ready(self) -> None:
        if self._closed:
            raise RuntimeError("isolated session writer is closed")
        if not self._started:
            raise RuntimeError("isolated session writer has not been started")

    def _wait(self, future):
        while True:
            try:
                return future.result(timeout=self.poll_interval_s)
            except FutureTimeoutError:
                # A worker may itself raise TimeoutError; do not confuse that
                # completed exception with this polling timeout.
                if future.done():
                    return future.result()
                if self._health_check is not None:
                    self._health_check()
