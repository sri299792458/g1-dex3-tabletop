import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from g1_aprilcube_calibration.calibration_bundle import CalibrationBundle
from g1_aprilcube_calibration.camera_initialization import (
    nominal_torso_T_color_optical,
)
from g1_aprilcube_calibration.joint_map import G1_29_JOINT_NAMES
from g1_aprilcube_calibration.urdf_model import URDFModel
from g1_dex3_tabletop import cli
from g1_dex3_tabletop.cli import main

ROOT = Path(__file__).parents[1]
BASE_URDF = ROOT / "config/urdf/g1_29dof_rev_1_0_g1pilot_collision.urdf"
BUNDLE = ROOT / "config/calibrations/dex3_shared_20260812_selected_free.json"


def test_checked_bundle_materializes_equivalent_calibrated_kinematics(
    tmp_path: Path,
) -> None:
    original_hash = hashlib.sha256(BASE_URDF.read_bytes()).hexdigest()
    bundle = CalibrationBundle.load(BUNDLE)
    output = tmp_path / "calibrated.urdf"

    bundle.materialize_urdf(BASE_URDF, output)

    assert hashlib.sha256(BASE_URDF.read_bytes()).hexdigest() == original_hash
    calibrated = URDFModel(output)
    np.testing.assert_allclose(
        nominal_torso_T_color_optical(calibrated),
        bundle.torso_T_camera,
        atol=1e-12,
    )
    nominal = URDFModel(BASE_URDF)
    measured = {name: 0.01 * np.sin(index) for index, name in enumerate(G1_29_JOINT_NAMES)}
    corrected = {
        name: measured[name] + bundle.joint_position_offsets_rad.get(name, 0.0)
        for name in G1_29_JOINT_NAMES
    }
    for hand in ("left_rubber_hand", "right_rubber_hand"):
        np.testing.assert_allclose(
            calibrated.transform("torso_link", hand, measured),
            nominal.transform("torso_link", hand, corrected),
            atol=1e-12,
        )


def test_bundle_rejects_content_tampering(tmp_path: Path) -> None:
    document = json.loads(BUNDLE.read_text())
    document["joint_position_offsets_rad"]["left_wrist_roll_joint"] += 0.01
    tampered = tmp_path / "tampered.json"
    tampered.write_text(json.dumps(document))

    with pytest.raises(ValueError, match="content SHA-256 mismatch"):
        CalibrationBundle.load(tampered)


def test_bundle_refuses_to_replace_base_urdf() -> None:
    with pytest.raises(ValueError, match="must not replace"):
        CalibrationBundle.load(BUNDLE).materialize_urdf(BASE_URDF, BASE_URDF)


def test_focused_inspect_cli_is_read_only_by_default(capsys) -> None:
    assert main(["inspect"]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["calibration_bundle_id"] == "dex3_shared_20260812_selected_free"
    assert result["calibration_bundle_sha256"] == CalibrationBundle.load(BUNDLE).content_sha256
    assert result["commands_robot"] is False


def test_cli_reports_operator_interrupt_without_traceback(monkeypatch, capsys) -> None:
    def interrupt(_args) -> int:
        raise KeyboardInterrupt

    monkeypatch.setattr(cli, "run_inspect", interrupt)

    assert main(["inspect"]) == 130
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == "interrupted by operator\n"
