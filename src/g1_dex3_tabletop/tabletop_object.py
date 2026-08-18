"""Hash-bound object selection for the shared tabletop runtime."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[2]
OBJECT_PROFILE_DIRECTORY = ROOT / "config/tabletop/objects"
DEFAULT_OBJECT_PROFILE_ID = "cube40-r3"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _repository_file(value: str | Path, *, label: str) -> Path:
    path = (ROOT / value).resolve()
    if not path.is_relative_to(ROOT):
        raise ValueError(f"{label} must be inside the repository")
    if not path.is_file():
        raise FileNotFoundError(f"{label} does not exist: {path}")
    return path


@dataclass(frozen=True, slots=True)
class TabletopObjectProfile:
    profile_id: str
    object_type: str
    dimensions_m: tuple[float, float, float]
    detector_config_path: Path
    direct_grasp_shortlist_path: Path
    config_path: Path


def _profile_path(value: str | Path) -> Path:
    text = str(value).strip()
    if not text:
        raise ValueError("tabletop object profile cannot be empty")
    requested = Path(text)
    if requested.is_absolute() or requested.parent != Path(".") or requested.suffix:
        return _repository_file(requested, label="tabletop object profile")
    return _repository_file(
        OBJECT_PROFILE_DIRECTORY / f"{text}.yaml",
        label="tabletop object profile",
    )


def load_tabletop_object_profile(value: str | Path) -> TabletopObjectProfile:
    """Load one detector/geometry/grasp bundle selected by a short profile ID."""

    config_path = _profile_path(value)
    document = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(document, dict) or document.get("schema_version") != 1:
        raise ValueError("unsupported tabletop object profile")
    profile_id = str(document.get("profile_id", "")).strip()
    if config_path.parent == OBJECT_PROFILE_DIRECTORY and profile_id != config_path.stem:
        raise ValueError("tabletop object profile ID differs from its filename")
    if str(document.get("object_type")) != "aprilcube":
        raise ValueError("the current tabletop perception requires an AprilCube object")
    dimensions = tuple(float(item) for item in document.get("dimensions_m", ()))
    if (
        len(dimensions) != 3
        or not np.all(np.isfinite(dimensions))
        or not np.all(np.asarray(dimensions) > 0.0)
    ):
        raise ValueError("tabletop object dimensions must contain three positive values")

    detector_path = _repository_file(
        document["detector_config"], label="object detector config"
    )
    shortlist_path = _repository_file(
        document["direct_grasp_shortlist"], label="direct-table grasp shortlist"
    )
    expected_hashes = document.get("sha256", {})
    if _sha256(detector_path) != expected_hashes.get("detector_config"):
        raise ValueError("object detector config differs from its profile hash")
    if _sha256(shortlist_path) != expected_hashes.get("direct_grasp_shortlist"):
        raise ValueError("object grasp shortlist differs from its profile hash")

    detector = json.loads(detector_path.read_text(encoding="utf-8"))
    if detector.get("target", {}).get("type") != "cuboid":
        raise ValueError("tabletop AprilCube detector must describe one cuboid")
    detector_dimensions = np.asarray(detector.get("box_dims", ()), dtype=np.float64)
    if detector_dimensions.shape != (3,) or not np.allclose(
        detector_dimensions / 1000.0,
        dimensions,
        atol=1.0e-9,
        rtol=0.0,
    ):
        raise ValueError("object detector dimensions differ from the object profile")
    expected_faces = {"+X", "-X", "+Y", "-Y", "+Z", "-Z"}
    if set(detector.get("faces", {})) != expected_faces:
        raise ValueError("tabletop AprilCube detector must define all six faces")

    shortlist = yaml.safe_load(shortlist_path.read_text(encoding="utf-8"))
    if (
        shortlist.get("format") != "g1_aprilcube_executable_grasp_shortlist"
        or shortlist.get("object_id") != "cube_head"
        or set(shortlist.get("applicable_hand_sides", ())) != {"left", "right"}
    ):
        raise ValueError("object profile does not select a bilateral cube shortlist")

    return TabletopObjectProfile(
        profile_id=profile_id,
        object_type="aprilcube",
        dimensions_m=dimensions,
        detector_config_path=detector_path,
        direct_grasp_shortlist_path=shortlist_path,
        config_path=config_path,
    )
