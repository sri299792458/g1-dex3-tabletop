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
PRIME_TOWER_PRESENTATION_ID = "prime-tower"
PRESENTATION_CONFIGS = {
    PRIME_TOWER_PRESENTATION_ID: (
        ROOT / "config/tabletop/presentations/prime_tower.yaml"
    ),
}


@dataclass(frozen=True, slots=True)
class TabletopPresentation:
    presentation_id: str
    applicable_hand_sides: tuple[str, ...]
    grasp_shortlist_paths: tuple[tuple[str, Path], ...]
    fixture: TabletopFixture | None
    object_profile_ids: tuple[str, ...] | None = None
    config_path: Path | None = None

    def require_arm(self, arm: str) -> None:
        selected = validate_arm_side(arm)
        if selected not in self.applicable_hand_sides:
            raise ValueError(
                f"presentation {self.presentation_id!r} is not qualified for {selected}"
            )

    def require_object_profile(self, profile_id: str) -> None:
        if self.object_profile_ids is not None and profile_id not in self.object_profile_ids:
            raise ValueError(
                f"presentation {self.presentation_id!r} is not qualified for object "
                f"profile {profile_id!r}"
            )

    def grasp_shortlist_for(self, profile_id: str) -> Path:
        """Return the shortlist qualified for this presentation/profile pair."""

        self.require_object_profile(profile_id)
        paths = dict(self.grasp_shortlist_paths)
        try:
            return paths[profile_id]
        except KeyError as error:
            raise ValueError(
                f"presentation {self.presentation_id!r} has no grasp shortlist for "
                f"object profile {profile_id!r}"
            ) from error


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
    direct_object_profile_id: str | None = None,
    direct_shortlist_override: str | Path | None = None,
) -> TabletopPresentation:
    """Resolve one presentation without changing direct-table defaults."""

    selected = str(presentation_id).strip()
    if selected == DIRECT_PRESENTATION_ID:
        profile_id = str(direct_object_profile_id or "").strip()
        if not profile_id or direct_shortlist_override is None:
            raise ValueError(
                "direct presentation requires its selected object profile and shortlist"
            )
        shortlist = _repository_path(
            direct_shortlist_override,
            label="direct-table grasp shortlist",
        )
        return TabletopPresentation(
            presentation_id=selected,
            applicable_hand_sides=("left", "right"),
            grasp_shortlist_paths=((profile_id, shortlist),),
            fixture=None,
            object_profile_ids=(profile_id,),
        )
    if direct_object_profile_id is not None or direct_shortlist_override is not None:
        raise ValueError("direct object selection is valid only with --presentation direct")
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
    object_profile_ids = tuple(str(value) for value in document["object_profile_ids"])
    if not object_profile_ids or len(set(object_profile_ids)) != len(object_profile_ids):
        raise ValueError("tabletop presentation object-profile contract is invalid")
    shortlist_values = document.get("grasp_shortlists")
    if not isinstance(shortlist_values, dict) or set(shortlist_values) != set(
        object_profile_ids
    ):
        raise ValueError(
            "tabletop presentation must provide exactly one shortlist per object profile"
        )
    shortlists = tuple(
        (
            profile_id,
            _repository_path(
                shortlist_values[profile_id],
                label=f"{profile_id} grasp shortlist",
            ),
        )
        for profile_id in object_profile_ids
    )
    fixture_data = dict(document["fixture"])
    mesh = _repository_path(fixture_data["mesh_path"], label="fixture mesh")
    if file_sha256(mesh) != fixture_data["mesh_sha256"]:
        raise ValueError("fixture mesh SHA-256 differs from the presentation config")
    fixture = TabletopFixture.from_dict(fixture_data)
    return TabletopPresentation(
        presentation_id=selected,
        applicable_hand_sides=sides,
        grasp_shortlist_paths=shortlists,
        fixture=fixture,
        object_profile_ids=object_profile_ids,
        config_path=config_path,
    )
