"""Deterministic raw-session verification and calibration-dataset derivation."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import yaml
from aprilcube import CorrespondenceDetector

from g1_aprilcube_calibration.authored_collection import AuthoredCollectionPlan
from g1_aprilcube_calibration.camera_models import RectifiedCameraInfo
from g1_aprilcube_calibration.correspondence import correspondence_sha256
from g1_aprilcube_calibration.joint_map import validate_arm_side
from g1_aprilcube_calibration.models import RobotStateSample
from g1_aprilcube_calibration.pose_schema import PoseSet
from g1_aprilcube_calibration.session_store import SessionStore
from g1_aprilcube_calibration.timestamp_pairing import (
    ImageTiming,
    PairingConfig,
    pair_state_to_image,
)

DATASET_SCHEMA_VERSION = 3
_SHA256_LENGTH = 64
OBSERVATION_PHASES = frozenset({"held", "supported"})


@dataclass(frozen=True, slots=True)
class CalibrationSample:
    capture_id: str
    pose_id: str
    frame_id: str
    raw_image_path: str
    raw_image_sha256: str
    camera_info: dict
    measured_state: dict
    pairing: dict
    visible_tag_ids: tuple[int, ...]
    corner_tag_ids: tuple[int, ...]
    image_points_px: tuple[tuple[float, float], ...]
    object_points_m: tuple[tuple[float, float, float], ...]
    correspondence_sha256: str

    def __post_init__(self) -> None:
        if not self.capture_id or not self.pose_id or not self.frame_id:
            raise ValueError("dataset sample IDs must be non-empty")
        for value in (self.raw_image_sha256, self.correspondence_sha256):
            if len(value) != _SHA256_LENGTH or any(
                character not in "0123456789abcdef" for character in value
            ):
                raise ValueError("dataset sample hashes must be lowercase SHA-256")
        image_points = np.asarray(self.image_points_px, dtype=np.float64)
        object_points = np.asarray(self.object_points_m, dtype=np.float64)
        if image_points.ndim != 2 or image_points.shape[1:] != (2,):
            raise ValueError("image points must have shape (N, 2)")
        if object_points.shape != (len(image_points), 3):
            raise ValueError("object points must have shape (N, 3) and match image points")
        if not len(image_points) or not np.all(np.isfinite(image_points)):
            raise ValueError("image points must be non-empty and finite")
        if not np.all(np.isfinite(object_points)):
            raise ValueError("object points must be finite")
        if len(self.corner_tag_ids) != len(image_points):
            raise ValueError("corner tag IDs must match the point count")
        if tuple(sorted(set(self.corner_tag_ids))) != tuple(self.visible_tag_ids):
            raise ValueError("visible tag IDs do not match corner tag IDs")
        for name in ("camera_info", "measured_state", "pairing"):
            value = json.loads(json.dumps(getattr(self, name), allow_nan=False))
            object.__setattr__(self, name, value)
        object.__setattr__(
            self,
            "image_points_px",
            tuple(tuple(float(value) for value in point) for point in image_points),
        )
        object.__setattr__(
            self,
            "object_points_m",
            tuple(tuple(float(value) for value in point) for point in object_points),
        )
        object.__setattr__(self, "visible_tag_ids", tuple(self.visible_tag_ids))
        object.__setattr__(self, "corner_tag_ids", tuple(self.corner_tag_ids))

    def to_dict(self) -> dict:
        return {
            "capture_id": self.capture_id,
            "pose_id": self.pose_id,
            "frame_id": self.frame_id,
            "raw_image_path": self.raw_image_path,
            "raw_image_sha256": self.raw_image_sha256,
            "camera_info": self.camera_info,
            "measured_state": self.measured_state,
            "pairing": self.pairing,
            "visible_tag_ids": list(self.visible_tag_ids),
            "corner_tag_ids": list(self.corner_tag_ids),
            "image_points_px": [list(point) for point in self.image_points_px],
            "object_points_m": [list(point) for point in self.object_points_m],
            "correspondence_sha256": self.correspondence_sha256,
        }

    @classmethod
    def from_dict(cls, data: dict) -> CalibrationSample:
        return cls(
            capture_id=data["capture_id"],
            pose_id=data["pose_id"],
            frame_id=data["frame_id"],
            raw_image_path=data["raw_image_path"],
            raw_image_sha256=data["raw_image_sha256"],
            camera_info=dict(data["camera_info"]),
            measured_state=dict(data["measured_state"]),
            pairing=dict(data["pairing"]),
            visible_tag_ids=tuple(data["visible_tag_ids"]),
            corner_tag_ids=tuple(data["corner_tag_ids"]),
            image_points_px=tuple(tuple(point) for point in data["image_points_px"]),
            object_points_m=tuple(tuple(point) for point in data["object_points_m"]),
            correspondence_sha256=data["correspondence_sha256"],
        )


@dataclass(frozen=True, slots=True)
class CalibrationDataset:
    session_id: str
    session_manifest_sha256: str
    target_artifact_sha256: str
    pose_set_sha256: str
    urdf_sha256: str
    calibration_arm: str
    observation_phase: str
    samples: tuple[CalibrationSample, ...]
    schema_version: int = DATASET_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != DATASET_SCHEMA_VERSION:
            raise ValueError("unsupported dataset schema version")
        if not self.session_id:
            raise ValueError("dataset session ID must be non-empty")
        object.__setattr__(self, "calibration_arm", validate_arm_side(self.calibration_arm))
        if self.observation_phase not in OBSERVATION_PHASES:
            raise ValueError(f"unsupported dataset observation phase: {self.observation_phase}")
        for value in (
            self.session_manifest_sha256,
            self.target_artifact_sha256,
            self.pose_set_sha256,
            self.urdf_sha256,
        ):
            if len(value) != _SHA256_LENGTH or any(
                character not in "0123456789abcdef" for character in value
            ):
                raise ValueError("dataset source hashes must be lowercase SHA-256")
        capture_ids = [sample.capture_id for sample in self.samples]
        frame_ids = [sample.frame_id for sample in self.samples]
        if len(capture_ids) != len(set(capture_ids)):
            raise ValueError("dataset contains duplicate accepted capture IDs")
        if len(frame_ids) != len(set(frame_ids)):
            raise ValueError("dataset contains duplicate selected frame IDs")
        object.__setattr__(self, "samples", tuple(self.samples))

    @property
    def content_sha256(self) -> str:
        return hashlib.sha256(
            json.dumps(
                self.to_dict(include_hash=False),
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ).encode()
        ).hexdigest()

    def to_dict(self, *, include_hash: bool = True) -> dict:
        result = {
            "schema_version": self.schema_version,
            "session_id": self.session_id,
            "session_manifest_sha256": self.session_manifest_sha256,
            "target_artifact_sha256": self.target_artifact_sha256,
            "pose_set_sha256": self.pose_set_sha256,
            "urdf_sha256": self.urdf_sha256,
            "calibration_arm": self.calibration_arm,
            "observation_phase": self.observation_phase,
            "samples": [sample.to_dict() for sample in self.samples],
        }
        if include_hash:
            result["content_sha256"] = self.content_sha256
        return result

    @classmethod
    def from_dict(cls, data: dict) -> CalibrationDataset:
        expected = {
            "schema_version",
            "session_id",
            "session_manifest_sha256",
            "target_artifact_sha256",
            "pose_set_sha256",
            "urdf_sha256",
            "calibration_arm",
            "observation_phase",
            "samples",
            "content_sha256",
        }
        if set(data) != expected:
            raise ValueError("dataset fields do not match schema version 3")
        result = cls(
            schema_version=int(data["schema_version"]),
            session_id=data["session_id"],
            session_manifest_sha256=data["session_manifest_sha256"],
            target_artifact_sha256=data["target_artifact_sha256"],
            pose_set_sha256=data["pose_set_sha256"],
            urdf_sha256=data["urdf_sha256"],
            calibration_arm=data["calibration_arm"],
            observation_phase=data["observation_phase"],
            samples=tuple(CalibrationSample.from_dict(item) for item in data["samples"]),
        )
        if data["content_sha256"] != result.content_sha256:
            raise ValueError("dataset content SHA-256 does not match")
        return result

    @classmethod
    def from_json(cls, path: str | Path) -> CalibrationDataset:
        with Path(path).open(encoding="utf-8") as stream:
            return cls.from_dict(json.load(stream))


class DatasetBuilder:
    def __init__(self, session_directory: str | Path) -> None:
        self.store = SessionStore(session_directory)
        self.directory = self.store.directory

    def build(
        self,
        *,
        output_path: str | Path | None = None,
        require_finalized: bool = True,
        observation_phase: str = "held",
    ) -> CalibrationDataset:
        if observation_phase not in OBSERVATION_PHASES:
            raise ValueError(f"unsupported dataset observation phase: {observation_phase}")
        manifest = self.store.load()
        if require_finalized and not manifest.finalized:
            raise RuntimeError("session must be finalized before dataset construction")
        self.store.verify_artifacts()
        orphans = self.store.find_orphans()
        if orphans:
            raise ValueError(f"raw session contains unreferenced orphan files: {orphans}")
        detector = CorrespondenceDetector(self.directory / "target.json")
        collection_method = manifest.provenance.get("collection_method")
        if collection_method == "curobo_automatic":
            from g1_dex3_tabletop.planning.contracts import CalibrationPlanResult

            motion_definition = CalibrationPlanResult.from_dict(
                json.loads((self.directory / "calibration_plan.json").read_bytes())
            )
        elif collection_method == "authored_automatic":
            motion_definition = AuthoredCollectionPlan.from_dict(
                json.loads((self.directory / "authored_plan.json").read_bytes())
            )
        else:
            motion_definition = PoseSet.from_dict(
                yaml.safe_load((self.directory / "pose_set.yaml").read_bytes())
            )
        if motion_definition.content_sha256 != manifest.pose_set_content_sha256:
            raise ValueError("session motion-definition content hash changed")
        pairing_config = PairingConfig(**manifest.pairing_config)
        samples: list[CalibrationSample] = []
        for capture in manifest.captures:
            verified = {
                frame.frame_id: self._verify_frame(
                    frame,
                    detector=detector,
                    pairing_config=pairing_config,
                    camera_profile_sha256=manifest.camera_profile_sha256,
                )
                for frame in capture.raw_frames
            }
            if capture.outcome != "accepted":
                continue
            phase_frames = (
                capture.frames if observation_phase == "held" else capture.supported_frames
            )
            selected_frame_id = (
                capture.selected_frame_id
                if observation_phase == "held"
                else capture.selected_supported_frame_id
            )
            if not phase_frames or selected_frame_id is None:
                raise ValueError(
                    f"accepted capture {capture.capture_id} has no {observation_phase} observation"
                )
            selected = next(frame for frame in phase_frames if frame.frame_id == selected_frame_id)
            result, pairing, camera_info = verified[selected.frame_id]
            result_hash = correspondence_sha256(result)

            image_points: list[tuple[float, float]] = []
            object_points: list[tuple[float, float, float]] = []
            corner_tag_ids: list[int] = []
            for observation in result.observations:
                for image_point, object_point_mm in zip(
                    observation.image_corners_px,
                    observation.object_corners_mm,
                    strict=True,
                ):
                    corner_tag_ids.append(observation.tag_id)
                    image_points.append(tuple(float(value) for value in image_point))
                    object_points.append(tuple(float(value) / 1000.0 for value in object_point_mm))
            if not image_points or len(image_points) != len(object_points):
                raise ValueError(f"invalid correspondence count: {selected.frame_id}")
            samples.append(
                CalibrationSample(
                    capture_id=capture.capture_id,
                    pose_id=capture.pose_id,
                    frame_id=selected.frame_id,
                    raw_image_path=selected.image_path,
                    raw_image_sha256=selected.image_sha256,
                    camera_info=camera_info.to_dict(),
                    measured_state=pairing.nearest.to_dict(),
                    pairing=pairing.to_dict(),
                    visible_tag_ids=result.tag_ids,
                    corner_tag_ids=tuple(corner_tag_ids),
                    image_points_px=tuple(image_points),
                    object_points_m=tuple(object_points),
                    correspondence_sha256=result_hash,
                )
            )
        dataset = CalibrationDataset(
            session_id=manifest.session_id,
            session_manifest_sha256=manifest.content_sha256,
            target_artifact_sha256=manifest.artifact_sha256["target.json"],
            pose_set_sha256=motion_definition.content_sha256,
            urdf_sha256=motion_definition.urdf_sha256,
            calibration_arm=motion_definition.calibration_arm,
            observation_phase=observation_phase,
            samples=tuple(samples),
        )
        if output_path is not None:
            self._write_dataset(Path(output_path), dataset)
        return dataset

    def _verify_frame(
        self,
        frame,
        *,
        detector: CorrespondenceDetector,
        pairing_config: PairingConfig,
        camera_profile_sha256: str,
    ):
        image_bytes = (self.directory / frame.image_path).read_bytes()
        if hashlib.sha256(image_bytes).hexdigest() != frame.image_sha256:
            raise ValueError(f"raw image hash mismatch: {frame.frame_id}")
        image = cv2.imdecode(np.frombuffer(image_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"raw image cannot be decoded: {frame.frame_id}")
        result = detector.detect(image)
        if correspondence_sha256(result) != frame.live_correspondence_sha256:
            raise ValueError(f"offline correspondence hash changed: {frame.frame_id}")
        states_bytes = (self.directory / frame.states_path).read_bytes()
        if hashlib.sha256(states_bytes).hexdigest() != frame.states_sha256:
            raise ValueError(f"raw states hash mismatch: {frame.frame_id}")
        states = tuple(RobotStateSample.from_dict(item) for item in json.loads(states_bytes))
        timing = ImageTiming(**frame.image_timing)
        pairing = pair_state_to_image(timing, states, config=pairing_config)
        if pairing.to_dict() != frame.pairing:
            raise ValueError(f"image/state pairing changed: {frame.frame_id}")
        camera_info = RectifiedCameraInfo.from_dict(frame.camera_info)
        if camera_info.profile_sha256 != camera_profile_sha256:
            raise ValueError(f"camera profile mismatch: {frame.frame_id}")
        return result, pairing, camera_info

    @staticmethod
    def _write_dataset(path: Path, dataset: CalibrationDataset) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        data = json.dumps(dataset.to_dict(), indent=2, sort_keys=True).encode() + b"\n"
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
        finally:
            temporary.unlink(missing_ok=True)


def deterministic_holdout_split(
    dataset: CalibrationDataset,
    *,
    holdout_fraction: float = 0.2,
) -> tuple[tuple[CalibrationSample, ...], tuple[CalibrationSample, ...]]:
    if not 0 < holdout_fraction < 1:
        raise ValueError("holdout_fraction must lie strictly between zero and one")
    pose_ids = sorted({sample.pose_id for sample in dataset.samples})
    if len(pose_ids) < 2:
        raise ValueError("at least two distinct poses are required for a holdout split")
    ranked_pose_ids = sorted(
        pose_ids,
        key=lambda pose_id: hashlib.sha256(
            f"{dataset.content_sha256}:{pose_id}".encode()
        ).digest(),
    )
    count = min(
        max(round(len(ranked_pose_ids) * holdout_fraction), 1),
        len(ranked_pose_ids) - 1,
    )
    holdout_pose_ids = set(ranked_pose_ids[:count])
    training = tuple(
        sample for sample in dataset.samples if sample.pose_id not in holdout_pose_ids
    )
    holdout = tuple(sample for sample in dataset.samples if sample.pose_id in holdout_pose_ids)
    return training, holdout
