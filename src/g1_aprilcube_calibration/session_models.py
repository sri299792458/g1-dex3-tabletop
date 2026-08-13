"""Versioned immutable-session manifest records."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from importlib.resources import files
from typing import Any

from jsonschema import Draft202012Validator

from g1_aprilcube_calibration.models import validate_utc_iso

SESSION_SCHEMA_VERSION = 3
_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,95}$")
_SHA_PATTERN = re.compile(r"^[0-9a-f]{64}$")
CAPTURE_OUTCOMES = frozenset({"accepted", "rejected", "retry", "skipped", "aborted"})


def _json_mapping(value: dict[str, Any], *, name: str) -> dict[str, Any]:
    try:
        return json.loads(json.dumps(value, sort_keys=True, allow_nan=False))
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be finite JSON-compatible data") from error


@dataclass(frozen=True, slots=True)
class RawFrameRecord:
    frame_id: str
    image_path: str
    image_sha256: str
    states_path: str
    states_sha256: str
    image_timing: dict[str, Any]
    camera_info: dict[str, Any]
    pairing: dict[str, Any]
    quality: dict[str, Any]
    live_correspondence_sha256: str

    def __post_init__(self) -> None:
        if not _ID_PATTERN.fullmatch(self.frame_id):
            raise ValueError(f"invalid raw frame ID: {self.frame_id!r}")
        for path in (self.image_path, self.states_path):
            if path.startswith("/") or ".." in path.split("/"):
                raise ValueError("raw artifact paths must be session-relative")
        for value in (
            self.image_sha256,
            self.states_sha256,
            self.live_correspondence_sha256,
        ):
            if not _SHA_PATTERN.fullmatch(value):
                raise ValueError("raw frame hashes must be lowercase SHA-256")
        for name in ("image_timing", "camera_info", "pairing", "quality"):
            object.__setattr__(self, name, _json_mapping(getattr(self, name), name=name))

    def to_dict(self) -> dict:
        return {
            "frame_id": self.frame_id,
            "image_path": self.image_path,
            "image_sha256": self.image_sha256,
            "states_path": self.states_path,
            "states_sha256": self.states_sha256,
            "image_timing": self.image_timing,
            "camera_info": self.camera_info,
            "pairing": self.pairing,
            "quality": self.quality,
            "live_correspondence_sha256": self.live_correspondence_sha256,
        }

    @classmethod
    def from_dict(cls, data: dict) -> RawFrameRecord:
        return cls(**data)


@dataclass(frozen=True, slots=True)
class CaptureRecord:
    capture_id: str
    pose_id: str
    outcome: str
    reason: str
    recorded_at_utc: str
    metadata: dict[str, Any]
    frames: tuple[RawFrameRecord, ...] = ()
    supported_frames: tuple[RawFrameRecord, ...] = ()
    selected_supported_frame_id: str | None = None
    selected_frame_id: str | None = None

    def __post_init__(self) -> None:
        if not _ID_PATTERN.fullmatch(self.capture_id):
            raise ValueError(f"invalid capture ID: {self.capture_id!r}")
        if not _ID_PATTERN.fullmatch(self.pose_id):
            raise ValueError(f"invalid capture pose ID: {self.pose_id!r}")
        if self.outcome not in CAPTURE_OUTCOMES:
            raise ValueError(f"unsupported capture outcome: {self.outcome}")
        if not self.reason:
            raise ValueError("capture reason must be non-empty")
        validate_utc_iso(self.recorded_at_utc)
        object.__setattr__(self, "metadata", _json_mapping(self.metadata, name="capture metadata"))
        held_frame_ids = [frame.frame_id for frame in self.frames]
        supported_frame_ids = [frame.frame_id for frame in self.supported_frames]
        frame_ids = [*held_frame_ids, *supported_frame_ids]
        if len(frame_ids) != len(set(frame_ids)):
            raise ValueError("capture contains duplicate frame IDs")
        if self.outcome == "accepted":
            if not self.frames or self.selected_frame_id not in held_frame_ids:
                raise ValueError("accepted capture requires a selected held frame")
            if self.supported_frames and (
                self.selected_supported_frame_id not in supported_frame_ids
            ):
                raise ValueError("accepted paired capture requires a selected supported frame")
            if not self.supported_frames and self.selected_supported_frame_id is not None:
                raise ValueError("capture cannot select a missing supported frame")
        elif self.selected_frame_id is not None or self.selected_supported_frame_id is not None:
            raise ValueError("non-accepted capture cannot select a raw frame")
        object.__setattr__(self, "frames", tuple(self.frames))
        object.__setattr__(self, "supported_frames", tuple(self.supported_frames))

    @property
    def raw_frames(self) -> tuple[RawFrameRecord, ...]:
        """All persisted held and operator-supported observations."""

        return (*self.frames, *self.supported_frames)

    def to_dict(self) -> dict:
        return {
            "capture_id": self.capture_id,
            "pose_id": self.pose_id,
            "outcome": self.outcome,
            "reason": self.reason,
            "recorded_at_utc": self.recorded_at_utc,
            "metadata": self.metadata,
            "frames": [frame.to_dict() for frame in self.frames],
            "supported_frames": [frame.to_dict() for frame in self.supported_frames],
            "selected_supported_frame_id": self.selected_supported_frame_id,
            "selected_frame_id": self.selected_frame_id,
        }

    @classmethod
    def from_dict(cls, data: dict) -> CaptureRecord:
        return cls(
            capture_id=data["capture_id"],
            pose_id=data["pose_id"],
            outcome=data["outcome"],
            reason=data["reason"],
            recorded_at_utc=data["recorded_at_utc"],
            metadata=dict(data["metadata"]),
            frames=tuple(RawFrameRecord.from_dict(item) for item in data["frames"]),
            supported_frames=tuple(
                RawFrameRecord.from_dict(item) for item in data["supported_frames"]
            ),
            selected_supported_frame_id=data["selected_supported_frame_id"],
            selected_frame_id=data["selected_frame_id"],
        )


@dataclass(frozen=True, slots=True)
class SessionManifest:
    session_id: str
    created_at_utc: str
    camera_profile_sha256: str
    pose_set_content_sha256: str
    artifact_sha256: dict[str, str]
    pairing_config: dict[str, Any]
    recording_gate_config: dict[str, Any]
    provenance: dict[str, Any]
    captures: tuple[CaptureRecord, ...] = ()
    finalized: bool = False
    schema_version: int = SESSION_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != SESSION_SCHEMA_VERSION:
            raise ValueError("unsupported session schema version")
        if not _ID_PATTERN.fullmatch(self.session_id):
            raise ValueError(f"invalid session ID: {self.session_id!r}")
        validate_utc_iso(self.created_at_utc)
        for value in (self.camera_profile_sha256, self.pose_set_content_sha256):
            if not _SHA_PATTERN.fullmatch(value):
                raise ValueError("session hashes must be lowercase SHA-256")
        if not self.artifact_sha256:
            raise ValueError("session must freeze at least one source artifact")
        if any(
            not name or not _SHA_PATTERN.fullmatch(value)
            for name, value in self.artifact_sha256.items()
        ):
            raise ValueError("artifact hashes must be named lowercase SHA-256 values")
        capture_ids = [capture.capture_id for capture in self.captures]
        frame_ids = [frame.frame_id for capture in self.captures for frame in capture.raw_frames]
        if len(capture_ids) != len(set(capture_ids)):
            raise ValueError("session contains duplicate capture IDs")
        if len(frame_ids) != len(set(frame_ids)):
            raise ValueError("session contains duplicate frame IDs")
        object.__setattr__(self, "artifact_sha256", dict(sorted(self.artifact_sha256.items())))
        object.__setattr__(
            self,
            "pairing_config",
            _json_mapping(self.pairing_config, name="pairing_config"),
        )
        object.__setattr__(
            self,
            "recording_gate_config",
            _json_mapping(self.recording_gate_config, name="recording_gate_config"),
        )
        object.__setattr__(self, "provenance", _json_mapping(self.provenance, name="provenance"))
        object.__setattr__(self, "captures", tuple(self.captures))

    @property
    def content_sha256(self) -> str:
        encoded = json.dumps(
            self.to_dict(include_hash=False),
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode()
        return hashlib.sha256(encoded).hexdigest()

    def to_dict(self, *, include_hash: bool = True) -> dict:
        result = {
            "schema_version": self.schema_version,
            "session_id": self.session_id,
            "created_at_utc": self.created_at_utc,
            "finalized": self.finalized,
            "camera_profile_sha256": self.camera_profile_sha256,
            "pose_set_content_sha256": self.pose_set_content_sha256,
            "artifact_sha256": self.artifact_sha256,
            "pairing_config": self.pairing_config,
            "recording_gate_config": self.recording_gate_config,
            "provenance": self.provenance,
            "captures": [capture.to_dict() for capture in self.captures],
        }
        if include_hash:
            result["content_sha256"] = self.content_sha256
        return result

    @classmethod
    def from_dict(cls, data: dict) -> SessionManifest:
        schema_path = files("g1_aprilcube_calibration.schemas").joinpath(
            "session_manifest.schema.json"
        )
        schema = json.loads(schema_path.read_text(encoding="utf-8"))
        Draft202012Validator(schema).validate(data)
        allowed = {
            "schema_version",
            "session_id",
            "created_at_utc",
            "finalized",
            "camera_profile_sha256",
            "pose_set_content_sha256",
            "artifact_sha256",
            "pairing_config",
            "recording_gate_config",
            "provenance",
            "captures",
            "content_sha256",
        }
        if set(data) != allowed:
            raise ValueError("session manifest fields do not match schema version 3")
        if not isinstance(data["finalized"], bool):
            raise TypeError("session finalized field must be boolean")
        result = cls(
            schema_version=int(data["schema_version"]),
            session_id=data["session_id"],
            created_at_utc=data["created_at_utc"],
            finalized=data["finalized"],
            camera_profile_sha256=data["camera_profile_sha256"],
            pose_set_content_sha256=data["pose_set_content_sha256"],
            artifact_sha256=dict(data["artifact_sha256"]),
            pairing_config=dict(data["pairing_config"]),
            recording_gate_config=dict(data["recording_gate_config"]),
            provenance=dict(data["provenance"]),
            captures=tuple(CaptureRecord.from_dict(item) for item in data["captures"]),
        )
        if data["content_sha256"] != result.content_sha256:
            raise ValueError("session manifest content SHA-256 does not match")
        return result
