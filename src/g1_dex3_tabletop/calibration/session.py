"""Crash-safe raw storage and deterministic replay for bilateral captures."""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Literal

import cv2
import numpy as np
from aprilcube import CorrespondenceDetector

from g1_aprilcube_calibration.camera_models import RectifiedCameraInfo
from g1_aprilcube_calibration.correspondence import correspondence_sha256
from g1_aprilcube_calibration.models import RobotStateSample, utc_now_iso, validate_utc_iso
from g1_aprilcube_calibration.quality import QualityGrade
from g1_aprilcube_calibration.readiness import (
    RecordingGateConfig,
    evaluate_recording_window,
)
from g1_aprilcube_calibration.timestamp_pairing import (
    ImageTiming,
    PairingConfig,
    pair_state_to_image,
)
from g1_dex3_tabletop.calibration.capture import (
    BilateralFrameEvidence,
    select_bilateral_medoid,
)
from g1_dex3_tabletop.calibration.dataset import (
    target_observation_from_correspondences,
)
from g1_dex3_tabletop.calibration.design import BilateralPoseDesignArtifact
from g1_dex3_tabletop.calibration.execution import BilateralExecutionPlan
from g1_dex3_tabletop.calibration.models import (
    BilateralCalibrationDataset,
    BilateralCalibrationSample,
    CameraFrameArtifact,
)

BILATERAL_SESSION_SCHEMA_VERSION = 1
BILATERAL_SOURCE_ARTIFACTS = frozenset(
    {
        "adapter_plan.json",
        "adapter_request.json",
        "camera_frames.json",
        "capture_quality.yaml",
        "execution_plan.json",
        "left_target.json",
        "pose_design.json",
        "right_target.json",
        "robot.urdf",
    }
)
_CAPTURE_OUTCOMES = frozenset({"accepted", "rejected", "retry", "skipped", "aborted"})
_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,95}$")
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_SIDES = ("left", "right")


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _canonical_json(data: Any) -> bytes:
    return json.dumps(
        data,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode()


def _json_mapping(value: Mapping[str, Any], *, name: str) -> dict[str, Any]:
    try:
        result = json.loads(json.dumps(dict(value), sort_keys=True, allow_nan=False))
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must contain finite JSON data") from error
    if not isinstance(result, dict):
        raise TypeError(f"{name} must be a mapping")
    return result


@dataclass(frozen=True, slots=True)
class BilateralRawFrameRecord:
    frame_id: str
    image_path: str
    image_sha256: str
    states_path: str
    states_sha256: str
    image_timing: dict[str, Any]
    camera_info: dict[str, Any]
    pairing: dict[str, Any]
    quality_by_arm: dict[str, Any]
    correspondence_sha256_by_arm: dict[str, str]

    def __post_init__(self) -> None:
        if not _ID_PATTERN.fullmatch(self.frame_id):
            raise ValueError(f"invalid bilateral raw frame ID: {self.frame_id!r}")
        for path in (self.image_path, self.states_path):
            if Path(path).is_absolute() or ".." in Path(path).parts:
                raise ValueError("bilateral raw paths must be session-relative")
        for value in (self.image_sha256, self.states_sha256):
            if not _SHA256_PATTERN.fullmatch(value):
                raise ValueError("bilateral raw artifact hash must be lowercase SHA-256")
        correspondences = {
            str(side): str(value) for side, value in self.correspondence_sha256_by_arm.items()
        }
        if set(correspondences) != set(_SIDES) or any(
            not _SHA256_PATTERN.fullmatch(value) for value in correspondences.values()
        ):
            raise ValueError("bilateral frame requires left/right correspondence hashes")
        quality = _json_mapping(self.quality_by_arm, name="quality_by_arm")
        if set(quality) != set(_SIDES):
            raise ValueError("bilateral frame requires left/right quality evidence")
        object.__setattr__(
            self, "image_timing", _json_mapping(self.image_timing, name="image_timing")
        )
        object.__setattr__(
            self, "camera_info", _json_mapping(self.camera_info, name="camera_info")
        )
        object.__setattr__(self, "pairing", _json_mapping(self.pairing, name="pairing"))
        object.__setattr__(self, "quality_by_arm", quality)
        object.__setattr__(self, "correspondence_sha256_by_arm", correspondences)

    def to_dict(self) -> dict[str, Any]:
        return {name: getattr(self, name) for name in self.__dataclass_fields__}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> BilateralRawFrameRecord:
        if set(data) != set(cls.__dataclass_fields__):
            raise ValueError("bilateral raw frame fields differ from schema version 1")
        return cls(**data)


@dataclass(frozen=True, slots=True)
class BilateralRawCaptureRecord:
    capture_id: str
    pose_group_id: str
    capture_role: Literal["anchor", "excitation"]
    outcome: str
    reason: str
    recorded_at_utc: str
    frames: tuple[BilateralRawFrameRecord, ...]
    selected_frame_id: str | None
    metadata: dict[str, Any]

    def __post_init__(self) -> None:
        if not _ID_PATTERN.fullmatch(self.capture_id):
            raise ValueError(f"invalid bilateral capture ID: {self.capture_id!r}")
        if not _ID_PATTERN.fullmatch(self.pose_group_id):
            raise ValueError(f"invalid bilateral pose group ID: {self.pose_group_id!r}")
        if self.capture_role not in {"anchor", "excitation"}:
            raise ValueError("bilateral capture role must be anchor or excitation")
        if self.outcome not in _CAPTURE_OUTCOMES:
            raise ValueError(f"unsupported bilateral capture outcome: {self.outcome}")
        if not self.reason.strip():
            raise ValueError("bilateral capture reason must be non-empty")
        validate_utc_iso(self.recorded_at_utc)
        frames = tuple(
            frame
            if isinstance(frame, BilateralRawFrameRecord)
            else BilateralRawFrameRecord.from_dict(frame)
            for frame in self.frames
        )
        frame_ids = [frame.frame_id for frame in frames]
        if len(frame_ids) != len(set(frame_ids)):
            raise ValueError("bilateral capture contains duplicate frame IDs")
        if self.outcome == "accepted":
            if not frames or self.selected_frame_id not in frame_ids:
                raise ValueError("accepted bilateral capture requires a selected raw frame")
        elif self.selected_frame_id is not None:
            raise ValueError("non-accepted bilateral capture cannot select a frame")
        object.__setattr__(self, "frames", frames)
        object.__setattr__(self, "metadata", _json_mapping(self.metadata, name="metadata"))

    def to_dict(self) -> dict[str, Any]:
        return {
            "capture_id": self.capture_id,
            "pose_group_id": self.pose_group_id,
            "capture_role": self.capture_role,
            "outcome": self.outcome,
            "reason": self.reason,
            "recorded_at_utc": self.recorded_at_utc,
            "frames": [frame.to_dict() for frame in self.frames],
            "selected_frame_id": self.selected_frame_id,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> BilateralRawCaptureRecord:
        expected = {
            "capture_id",
            "pose_group_id",
            "capture_role",
            "outcome",
            "reason",
            "recorded_at_utc",
            "frames",
            "selected_frame_id",
            "metadata",
        }
        if set(data) != expected:
            raise ValueError("bilateral capture fields differ from schema version 1")
        return cls(
            **{
                **data,
                "frames": tuple(
                    BilateralRawFrameRecord.from_dict(item) for item in data["frames"]
                ),
            }
        )


@dataclass(frozen=True, slots=True)
class BilateralSessionManifest:
    session_id: str
    created_at_utc: str
    day_group_id: str
    camera_profile_sha256: str
    pose_design_sha256: str
    execution_plan_sha256: str
    urdf_sha256: str
    camera_frame_content_sha256: str
    target_artifact_sha256_by_arm: dict[str, str]
    source_artifact_sha256: dict[str, str]
    pairing_config: dict[str, Any]
    recording_gate_config: dict[str, Any]
    provenance: dict[str, Any]
    captures: tuple[BilateralRawCaptureRecord, ...] = ()
    finalized: bool = False
    schema_version: int = BILATERAL_SESSION_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != BILATERAL_SESSION_SCHEMA_VERSION:
            raise ValueError("unsupported bilateral session schema version")
        if not _ID_PATTERN.fullmatch(self.session_id):
            raise ValueError(f"invalid bilateral session ID: {self.session_id!r}")
        if not self.day_group_id.strip():
            raise ValueError("bilateral session day group must be non-empty")
        validate_utc_iso(self.created_at_utc)
        for name in (
            "camera_profile_sha256",
            "pose_design_sha256",
            "execution_plan_sha256",
            "urdf_sha256",
            "camera_frame_content_sha256",
        ):
            if not _SHA256_PATTERN.fullmatch(getattr(self, name)):
                raise ValueError(f"{name} must be lowercase SHA-256")
        targets = dict(self.target_artifact_sha256_by_arm)
        if set(targets) != set(_SIDES) or any(
            not _SHA256_PATTERN.fullmatch(value) for value in targets.values()
        ):
            raise ValueError("bilateral session requires left/right target hashes")
        source_hashes = dict(self.source_artifact_sha256)
        if set(source_hashes) != BILATERAL_SOURCE_ARTIFACTS or any(
            not _SHA256_PATTERN.fullmatch(value) for value in source_hashes.values()
        ):
            raise ValueError("bilateral session source artifacts differ from the contract")
        captures = tuple(
            capture
            if isinstance(capture, BilateralRawCaptureRecord)
            else BilateralRawCaptureRecord.from_dict(capture)
            for capture in self.captures
        )
        capture_ids = [capture.capture_id for capture in captures]
        frame_ids = [frame.frame_id for capture in captures for frame in capture.frames]
        if len(capture_ids) != len(set(capture_ids)):
            raise ValueError("bilateral session contains duplicate capture IDs")
        if len(frame_ids) != len(set(frame_ids)):
            raise ValueError("bilateral session contains duplicate frame IDs")
        if not isinstance(self.finalized, bool):
            raise TypeError("bilateral session finalized must be boolean")
        object.__setattr__(self, "target_artifact_sha256_by_arm", targets)
        object.__setattr__(self, "source_artifact_sha256", source_hashes)
        object.__setattr__(
            self, "pairing_config", _json_mapping(self.pairing_config, name="pairing_config")
        )
        object.__setattr__(
            self,
            "recording_gate_config",
            _json_mapping(self.recording_gate_config, name="recording_gate_config"),
        )
        object.__setattr__(self, "provenance", _json_mapping(self.provenance, name="provenance"))
        object.__setattr__(self, "captures", captures)

    @property
    def content_sha256(self) -> str:
        return _sha256(_canonical_json(self.to_dict(include_hash=False)))

    def to_dict(self, *, include_hash: bool = True) -> dict[str, Any]:
        result = {
            "schema_version": self.schema_version,
            "session_id": self.session_id,
            "created_at_utc": self.created_at_utc,
            "day_group_id": self.day_group_id,
            "camera_profile_sha256": self.camera_profile_sha256,
            "pose_design_sha256": self.pose_design_sha256,
            "execution_plan_sha256": self.execution_plan_sha256,
            "urdf_sha256": self.urdf_sha256,
            "camera_frame_content_sha256": self.camera_frame_content_sha256,
            "target_artifact_sha256_by_arm": self.target_artifact_sha256_by_arm,
            "source_artifact_sha256": self.source_artifact_sha256,
            "pairing_config": self.pairing_config,
            "recording_gate_config": self.recording_gate_config,
            "provenance": self.provenance,
            "captures": [capture.to_dict() for capture in self.captures],
            "finalized": self.finalized,
        }
        if include_hash:
            result["content_sha256"] = self.content_sha256
        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> BilateralSessionManifest:
        expected = {
            "schema_version",
            "session_id",
            "created_at_utc",
            "day_group_id",
            "camera_profile_sha256",
            "pose_design_sha256",
            "execution_plan_sha256",
            "urdf_sha256",
            "camera_frame_content_sha256",
            "target_artifact_sha256_by_arm",
            "source_artifact_sha256",
            "pairing_config",
            "recording_gate_config",
            "provenance",
            "captures",
            "finalized",
            "content_sha256",
        }
        if set(data) != expected:
            raise ValueError("bilateral session fields differ from schema version 1")
        content_hash = data["content_sha256"]
        result = cls(
            **{
                **{name: data[name] for name in expected - {"content_sha256", "captures"}},
                "captures": tuple(
                    BilateralRawCaptureRecord.from_dict(item) for item in data["captures"]
                ),
            }
        )
        if result.content_sha256 != content_hash:
            raise ValueError("bilateral session content SHA-256 mismatch")
        return result


class BilateralSessionStore:
    """Persist both targets, the shared image, and the shared state window."""

    def __init__(self, directory: str | Path) -> None:
        self.directory = Path(directory).resolve()
        self.manifest_path = self.directory / "manifest.json"

    def create(
        self,
        *,
        session_id: str,
        created_at_utc: str,
        day_group_id: str,
        camera_info: RectifiedCameraInfo,
        camera_frames: CameraFrameArtifact,
        pose_design_sha256: str,
        execution_plan_sha256: str,
        source_artifacts: Mapping[str, bytes],
        pairing_config: PairingConfig,
        recording_gate_config: RecordingGateConfig,
        provenance: Mapping[str, Any],
    ) -> BilateralSessionManifest:
        if self.directory.exists():
            raise FileExistsError(f"bilateral session directory exists: {self.directory}")
        if set(source_artifacts) != BILATERAL_SOURCE_ARTIFACTS:
            raise ValueError(
                "bilateral source artifacts must be exactly: "
                + ", ".join(sorted(BILATERAL_SOURCE_ARTIFACTS))
            )
        if not _SHA256_PATTERN.fullmatch(pose_design_sha256):
            raise ValueError("pose design hash must be lowercase SHA-256")
        if not _SHA256_PATTERN.fullmatch(execution_plan_sha256):
            raise ValueError("execution plan hash must be lowercase SHA-256")
        artifacts = {name: bytes(value) for name, value in source_artifacts.items()}
        frozen_camera_frames = CameraFrameArtifact.from_dict(
            json.loads(artifacts["camera_frames.json"])
        )
        if frozen_camera_frames.content_sha256 != camera_frames.content_sha256:
            raise ValueError("source camera-frame artifact differs from the selected object")
        frozen_pose_design = BilateralPoseDesignArtifact.from_dict(
            json.loads(artifacts["pose_design.json"])
        )
        if frozen_pose_design.content_sha256 != pose_design_sha256:
            raise ValueError("source pose-design artifact differs from the selected hash")
        frozen_execution_plan = BilateralExecutionPlan.from_dict(
            json.loads(artifacts["execution_plan.json"])
        )
        if frozen_execution_plan.content_sha256 != execution_plan_sha256:
            raise ValueError("source execution-plan artifact differs from the selected hash")
        frozen_execution_plan.validate_design(frozen_pose_design)
        source_hashes = {name: _sha256(content) for name, content in artifacts.items()}
        if frozen_execution_plan.urdf_sha256 != source_hashes["robot.urdf"]:
            raise ValueError("execution plan belongs to a different robot URDF")
        manifest = BilateralSessionManifest(
            session_id=session_id,
            created_at_utc=created_at_utc,
            day_group_id=day_group_id,
            camera_profile_sha256=camera_info.profile_sha256,
            pose_design_sha256=pose_design_sha256,
            execution_plan_sha256=execution_plan_sha256,
            urdf_sha256=source_hashes["robot.urdf"],
            camera_frame_content_sha256=camera_frames.content_sha256,
            target_artifact_sha256_by_arm={
                "left": source_hashes["left_target.json"],
                "right": source_hashes["right_target.json"],
            },
            source_artifact_sha256=source_hashes,
            pairing_config={
                name: getattr(pairing_config, name) for name in pairing_config.__dataclass_fields__
            },
            recording_gate_config={
                name: getattr(recording_gate_config, name)
                for name in recording_gate_config.__dataclass_fields__
            },
            provenance=provenance,
        )
        # Complete all semantic validation before creating the destination.
        # A bad request must not leave a directory that looks resumable.
        self.directory.mkdir(parents=True)
        (self.directory / "raw" / "images").mkdir(parents=True)
        (self.directory / "raw" / "states").mkdir(parents=True)
        for name, content in artifacts.items():
            self._write_new_file(self.directory / name, content)
        self._write_manifest(manifest, first=True)
        return manifest

    def load(self) -> BilateralSessionManifest:
        data = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise TypeError("bilateral manifest must contain a JSON object")
        return BilateralSessionManifest.from_dict(data)

    def append_capture(
        self,
        *,
        capture_id: str,
        pose_group_id: str,
        capture_role: Literal["anchor", "excitation"],
        outcome: str,
        reason: str,
        frames: Sequence[BilateralFrameEvidence] = (),
        metadata: Mapping[str, Any] | None = None,
        recorded_at_utc: str | None = None,
    ) -> BilateralSessionManifest:
        manifest = self.load()
        if manifest.finalized:
            raise RuntimeError("cannot append to a finalized bilateral session")
        if not _ID_PATTERN.fullmatch(capture_id):
            raise ValueError(f"invalid bilateral capture ID: {capture_id!r}")
        if not _ID_PATTERN.fullmatch(pose_group_id):
            raise ValueError(f"invalid bilateral pose group ID: {pose_group_id!r}")
        if capture_role not in {"anchor", "excitation"}:
            raise ValueError("bilateral capture role must be anchor or excitation")
        if outcome not in _CAPTURE_OUTCOMES:
            raise ValueError(f"unsupported bilateral capture outcome: {outcome}")
        if not reason.strip():
            raise ValueError("bilateral capture reason must be non-empty")
        if any(capture.capture_id == capture_id for capture in manifest.captures):
            raise ValueError(f"duplicate bilateral capture ID: {capture_id}")
        values = tuple(frames)
        if len({frame.frame_id for frame in values}) != len(values):
            raise ValueError("bilateral capture input contains duplicate frame IDs")
        existing_frame_ids = {
            frame.frame_id for capture in manifest.captures for frame in capture.frames
        }
        if any(frame.frame_id in existing_frame_ids for frame in values):
            raise ValueError("bilateral session already contains one of the frame IDs")
        pairing_config = PairingConfig(**manifest.pairing_config)
        gate_config = RecordingGateConfig(**manifest.recording_gate_config)
        for frame in values:
            if frame.camera_info.profile_sha256 != manifest.camera_profile_sha256:
                raise ValueError("camera profile changed during bilateral session")
            rebuilt = pair_state_to_image(
                frame.image_timing,
                frame.state_window,
                config=pairing_config,
            )
            if rebuilt.to_dict() != frame.pairing.to_dict():
                raise ValueError("provided bilateral pairing is not reproducible")
            readiness = evaluate_recording_window(
                frame.state_window,
                now_monotonic_s=frame.state_window[-1].receipt_monotonic_s,
                config=gate_config,
            )
            if not readiness.ready:
                raise ValueError(
                    "bilateral frame is not stationary: " + "; ".join(readiness.hard_failures)
                )
            for side in _SIDES:
                result = getattr(frame, f"{side}_correspondences")
                quality = getattr(frame, f"{side}_quality")
                if not result.valid or quality.grade is QualityGrade.RED:
                    raise ValueError(f"bilateral raw frame has invalid {side} evidence")
        if outcome == "accepted" and not values:
            raise ValueError("accepted bilateral capture requires raw frames")
        selected_frame_id = (
            select_bilateral_medoid(values).frame_id if outcome == "accepted" else None
        )
        records = tuple(self._write_raw_frame(frame) for frame in values)
        capture = BilateralRawCaptureRecord(
            capture_id=capture_id,
            pose_group_id=pose_group_id,
            capture_role=capture_role,
            outcome=outcome,
            reason=reason,
            recorded_at_utc=recorded_at_utc or utc_now_iso(),
            frames=records,
            selected_frame_id=selected_frame_id,
            metadata={} if metadata is None else dict(metadata),
        )
        updated = replace(manifest, captures=(*manifest.captures, capture))
        self._write_manifest(updated)
        return updated

    def finalize(self) -> BilateralSessionManifest:
        manifest = self.load()
        if manifest.finalized:
            return manifest
        if not any(capture.outcome == "accepted" for capture in manifest.captures):
            raise ValueError("cannot finalize a bilateral session without accepted captures")
        updated = replace(manifest, finalized=True)
        self._write_manifest(updated)
        return updated

    def verify_artifacts(self) -> None:
        manifest = self.load()
        for name, expected in manifest.source_artifact_sha256.items():
            if _sha256((self.directory / name).read_bytes()) != expected:
                raise ValueError(f"bilateral source artifact hash mismatch: {name}")
        for capture in manifest.captures:
            for frame in capture.frames:
                if _sha256((self.directory / frame.image_path).read_bytes()) != frame.image_sha256:
                    raise ValueError(f"bilateral raw image hash mismatch: {frame.frame_id}")
                if (
                    _sha256((self.directory / frame.states_path).read_bytes())
                    != frame.states_sha256
                ):
                    raise ValueError(f"bilateral raw states hash mismatch: {frame.frame_id}")

    def find_orphans(self) -> tuple[str, ...]:
        manifest = self.load()
        referenced = {
            path
            for capture in manifest.captures
            for frame in capture.frames
            for path in (frame.image_path, frame.states_path)
        }
        actual = {
            path.relative_to(self.directory).as_posix()
            for path in (self.directory / "raw").rglob("*")
            if path.is_file()
        }
        return tuple(sorted(actual - referenced))

    def build_dataset(
        self,
        *,
        detectors: Mapping[str, CorrespondenceDetector],
        output_path: str | Path | None = None,
    ) -> BilateralCalibrationDataset:
        manifest = self.load()
        if not manifest.finalized:
            raise RuntimeError("bilateral session must be finalized before dataset replay")
        if set(detectors) != set(_SIDES):
            raise ValueError("bilateral replay requires left and right detectors")
        self.verify_artifacts()
        orphans = self.find_orphans()
        if orphans:
            raise ValueError(f"bilateral raw session contains orphan files: {orphans}")
        pairing_config = PairingConfig(**manifest.pairing_config)
        samples: list[BilateralCalibrationSample] = []
        for capture in manifest.captures:
            verified = {
                frame.frame_id: self._verify_frame(
                    frame,
                    detectors=detectors,
                    pairing_config=pairing_config,
                    camera_profile_sha256=manifest.camera_profile_sha256,
                )
                for frame in capture.frames
            }
            if capture.outcome != "accepted":
                continue
            assert capture.selected_frame_id is not None
            frame = next(
                item for item in capture.frames if item.frame_id == capture.selected_frame_id
            )
            image, states, timing, pairing, camera_info, results = verified[frame.frame_id]
            del image, states, timing
            samples.append(
                BilateralCalibrationSample(
                    source_session_id=manifest.session_id,
                    capture_id=capture.capture_id,
                    pose_group_id=capture.pose_group_id,
                    day_group_id=manifest.day_group_id,
                    frame_id=frame.frame_id,
                    capture_role=capture.capture_role,
                    raw_image_path=frame.image_path,
                    raw_image_sha256=frame.image_sha256,
                    camera_info=camera_info.to_dict(),
                    joint_positions_rad=tuple(float(value) for value in pairing.nearest.position),
                    joint_velocities_rad_s=tuple(
                        float(value) for value in pairing.nearest.velocity
                    ),
                    pairing=pairing.to_dict(),
                    left=target_observation_from_correspondences(
                        results["left"],
                        side="left",
                        target_artifact_sha256=(manifest.target_artifact_sha256_by_arm["left"]),
                    ),
                    right=target_observation_from_correspondences(
                        results["right"],
                        side="right",
                        target_artifact_sha256=(manifest.target_artifact_sha256_by_arm["right"]),
                    ),
                )
            )
        dataset = BilateralCalibrationDataset(
            dataset_id=manifest.session_id,
            session_manifest_sha256_by_id={manifest.session_id: manifest.content_sha256},
            pose_design_sha256=manifest.pose_design_sha256,
            execution_plan_sha256=manifest.execution_plan_sha256,
            urdf_sha256=manifest.urdf_sha256,
            rgb_optical_transform_sha256=manifest.camera_frame_content_sha256,
            left_target_artifact_sha256=(manifest.target_artifact_sha256_by_arm["left"]),
            right_target_artifact_sha256=(manifest.target_artifact_sha256_by_arm["right"]),
            samples=tuple(samples),
            provenance={
                **manifest.provenance,
                "source_artifact_sha256": manifest.source_artifact_sha256,
                "replay_policy": "offline_redetect_both_targets_and_repair_state",
            },
        )
        if output_path is not None:
            self._atomic_replace(
                Path(output_path),
                json.dumps(
                    dataset.to_dict(),
                    indent=2,
                    sort_keys=True,
                    allow_nan=False,
                ).encode()
                + b"\n",
            )
        return dataset

    def _write_raw_frame(self, frame: BilateralFrameEvidence) -> BilateralRawFrameRecord:
        success, encoded = cv2.imencode(".png", frame.image_bgr)
        if not success:
            raise RuntimeError("OpenCV failed to encode bilateral raw PNG")
        image_bytes = encoded.tobytes()
        states_bytes = _canonical_json([sample.to_dict() for sample in frame.state_window])
        image_path = f"raw/images/{frame.frame_id}.png"
        states_path = f"raw/states/{frame.frame_id}.json"
        self._write_new_file(self.directory / image_path, image_bytes)
        self._write_new_file(self.directory / states_path, states_bytes)
        return BilateralRawFrameRecord(
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
            quality_by_arm={side: getattr(frame, f"{side}_quality").to_dict() for side in _SIDES},
            correspondence_sha256_by_arm={
                side: correspondence_sha256(getattr(frame, f"{side}_correspondences"))
                for side in _SIDES
            },
        )

    def _verify_frame(
        self,
        frame: BilateralRawFrameRecord,
        *,
        detectors: Mapping[str, CorrespondenceDetector],
        pairing_config: PairingConfig,
        camera_profile_sha256: str,
    ) -> tuple[
        np.ndarray,
        tuple[RobotStateSample, ...],
        ImageTiming,
        Any,
        RectifiedCameraInfo,
        dict[str, Any],
    ]:
        image_bytes = (self.directory / frame.image_path).read_bytes()
        if _sha256(image_bytes) != frame.image_sha256:
            raise ValueError(f"bilateral raw image hash mismatch: {frame.frame_id}")
        image = cv2.imdecode(np.frombuffer(image_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"bilateral raw image cannot be decoded: {frame.frame_id}")
        results = {side: detectors[side].detect(image) for side in _SIDES}
        for side in _SIDES:
            if correspondence_sha256(results[side]) != frame.correspondence_sha256_by_arm[side]:
                raise ValueError(f"offline {side} correspondence hash changed: {frame.frame_id}")
        states_bytes = (self.directory / frame.states_path).read_bytes()
        if _sha256(states_bytes) != frame.states_sha256:
            raise ValueError(f"bilateral raw states hash mismatch: {frame.frame_id}")
        states = tuple(RobotStateSample.from_dict(item) for item in json.loads(states_bytes))
        timing = ImageTiming(**frame.image_timing)
        pairing = pair_state_to_image(timing, states, config=pairing_config)
        if pairing.to_dict() != frame.pairing:
            raise ValueError(f"bilateral image/state pairing changed: {frame.frame_id}")
        camera_info = RectifiedCameraInfo.from_dict(frame.camera_info)
        if camera_info.profile_sha256 != camera_profile_sha256:
            raise ValueError(f"bilateral camera profile changed: {frame.frame_id}")
        return image, states, timing, pairing, camera_info, results

    def _write_manifest(
        self,
        manifest: BilateralSessionManifest,
        *,
        first: bool = False,
    ) -> None:
        data = (
            json.dumps(
                manifest.to_dict(),
                indent=2,
                sort_keys=True,
                allow_nan=False,
            ).encode()
            + b"\n"
        )
        if first:
            self._write_new_file(self.manifest_path, data)
        else:
            self._atomic_replace(self.manifest_path, data)

    @staticmethod
    def _write_new_file(path: Path, data: bytes) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{path.name}.",
            suffix=".tmp",
            dir=path.parent,
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
            BilateralSessionStore._fsync_directory(path.parent)
        finally:
            temporary.unlink(missing_ok=True)

    @staticmethod
    def _atomic_replace(path: Path, data: bytes) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{path.name}.",
            suffix=".tmp",
            dir=path.parent,
        )
        temporary = Path(temporary_name)
        try:
            with os.fdopen(descriptor, "wb") as stream:
                stream.write(data)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, path)
            BilateralSessionStore._fsync_directory(path.parent)
        finally:
            temporary.unlink(missing_ok=True)

    @staticmethod
    def _fsync_directory(path: Path) -> None:
        descriptor = os.open(path, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
