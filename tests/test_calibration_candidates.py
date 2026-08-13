import json
from pathlib import Path

import numpy as np
import yaml

from g1_aprilcube_calibration.calibration_bundle import CalibrationBundle
from g1_dex3_tabletop.calibration_candidates import (
    CandidateDesignConfig,
    camera_info_from_hardware,
    generate_calibration_candidates,
    select_information_candidates,
)

ROOT = Path(__file__).resolve().parents[1]


def test_fixed_marker_candidates_are_deterministic_and_information_ranked() -> None:
    hardware = yaml.safe_load(
        (ROOT / "config/hardware_dex3_aruco.yaml").read_text(encoding="utf-8")
    )
    target = json.loads(
        (ROOT / "config/dex3_dorsal_aruco_target.json").read_text(encoding="utf-8")
    )
    bundle = CalibrationBundle.load(
        ROOT / "config/calibrations/dex3_shared_20260812_selected_free.json"
    )
    config = CandidateDesignConfig(target_count=8, candidate_count=100)
    kwargs = {
        "camera_info": camera_info_from_hardware(hardware),
        "target_config": target,
        "torso_T_camera": bundle.torso_T_camera,
        "palm_T_marker": hardware["robot"]["calibration_target_modeled_hand_T_target"],
        "config": config,
    }

    first = generate_calibration_candidates(**kwargs)
    second = generate_calibration_candidates(**kwargs)

    assert [item.to_dict() for item in first] == [item.to_dict() for item in second]
    assert len(first) == 97
    assert all(
        np.asarray(item.selection_metadata["fixed_marker_camera_information"]).shape == (6, 6)
        for item in first
    )
    selected, steps = select_information_candidates(
        list(first), count=config.target_count, config=config
    )
    assert len(selected) == config.target_count
    assert len({item.candidate_id for item in selected}) == config.target_count
    assert [step["selection_index"] for step in steps] == list(range(1, config.target_count + 1))


def test_candidate_design_config_rejects_incomplete_serialized_policy() -> None:
    document = CandidateDesignConfig().to_dict()
    document.pop("image_margin_px")

    try:
        CandidateDesignConfig.from_dict(document)
    except ValueError as error:
        assert "fields do not match" in str(error)
    else:
        raise AssertionError("incomplete design policy was accepted")
