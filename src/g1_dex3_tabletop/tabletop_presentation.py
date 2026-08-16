"""Small opt-in object-presentation layer for the shared tabletop runtime."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

from g1_aprilcube_calibration.joint_map import validate_arm_side
from g1_dex3_tabletop.tabletop_contracts import TabletopFixture
from g1_dex3_tabletop.tabletop_workflow import file_sha256

ROOT = Path(__file__).resolve().parents[2]
DIRECT_PRESENTATION_ID = "direct"
TRIPOD_H50_PRESENTATION_ID = "tripod-h50"
DEFAULT_GRASP_SHORTLIST = ROOT / "config/tabletop/cube_dex3_executable_v1/shortlist.yaml"
PRESENTATION_CONFIGS = {
    TRIPOD_H50_PRESENTATION_ID: (ROOT / "config/tabletop/presentations/tripod_h50.yaml"),
}


@dataclass(frozen=True, slots=True)
class TabletopPresentation:
    presentation_id: str
    applicable_hand_sides: tuple[str, ...]
    grasp_shortlist_path: Path
    fixture: TabletopFixture | None
    config_path: Path | None = None

    def require_arm(self, arm: str) -> None:
        selected = validate_arm_side(arm)
        if selected not in self.applicable_hand_sides:
            raise ValueError(
                f"presentation {self.presentation_id!r} is not qualified for {selected}"
            )


def _repository_path(value: str | Path, *, label: str) -> Path:
    path = (ROOT / value).resolve()
    if not path.is_relative_to(ROOT):
        raise ValueError(f"{label} must be inside the repository")
    if not path.is_file():
        raise FileNotFoundError(f"{label} does not exist: {path}")
    return path


def load_tabletop_presentation(
    presentation_id: str,
    *,
    direct_shortlist_override: str | Path | None = None,
) -> TabletopPresentation:
    """Resolve one presentation without changing direct-table defaults."""

    selected = str(presentation_id).strip()
    if selected == DIRECT_PRESENTATION_ID:
        shortlist = _repository_path(
            direct_shortlist_override or DEFAULT_GRASP_SHORTLIST,
            label="direct-table grasp shortlist",
        )
        return TabletopPresentation(
            presentation_id=selected,
            applicable_hand_sides=("left", "right"),
            grasp_shortlist_path=shortlist,
            fixture=None,
        )
    if direct_shortlist_override is not None:
        raise ValueError("--grasp-shortlist is available only with --presentation direct")
    if selected not in PRESENTATION_CONFIGS:
        raise ValueError(f"unsupported tabletop presentation: {selected!r}")
    config_path = PRESENTATION_CONFIGS[selected]
    document = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(document, dict) or document.get("schema_version") != 1:
        raise ValueError("unsupported tabletop presentation configuration")
    if document.get("presentation_id") != selected:
        raise ValueError("tabletop presentation configuration ID mismatch")
    sides = tuple(validate_arm_side(value) for value in document["applicable_hand_sides"])
    if len(set(sides)) != len(sides):
        raise ValueError("tabletop presentation repeats an applicable hand side")
    shortlist = _repository_path(document["grasp_shortlist"], label="grasp shortlist")
    fixture_data = dict(document["fixture"])
    mesh = _repository_path(fixture_data["mesh_path"], label="fixture mesh")
    if file_sha256(mesh) != fixture_data["mesh_sha256"]:
        raise ValueError("fixture mesh SHA-256 differs from the presentation config")
    fixture = TabletopFixture.from_dict(fixture_data)
    return TabletopPresentation(
        presentation_id=selected,
        applicable_hand_sides=sides,
        grasp_shortlist_path=shortlist,
        fixture=fixture,
        config_path=config_path,
    )
