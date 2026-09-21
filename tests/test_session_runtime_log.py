import json

from g1_aprilcube_calibration.session_runtime_log import SessionRuntimeLog


def test_session_runtime_log_appends_compact_json_lines(tmp_path) -> None:
    path = tmp_path / "runtime.jsonl"
    log = SessionRuntimeLog(path)

    log.append("run_started", session_id="run_001", accepted_goal=50)
    log.append(
        "executor_transition",
        sequence=1,
        previous_state="observing",
        state="acquiring",
        reason="operator confirmed acquisition",
    )

    records = [json.loads(line) for line in path.read_text().splitlines()]
    assert [record["event"] for record in records] == [
        "run_started",
        "executor_transition",
    ]
    assert records[0]["session_id"] == "run_001"
    assert records[0]["accepted_goal"] == 50
    assert records[1]["state"] == "acquiring"
    assert all(record["utc"].endswith("Z") for record in records)
