import json
from pathlib import Path

import pytest
import yaml

from g1_dex3_tabletop import raw_episode_recording as recording
from g1_dex3_tabletop.cli import build_parser


def test_tabletop_topic_contract_uses_official_general_streams() -> None:
    topics = {topic.name: topic.message_type for topic in recording.TABLETOP_RAW_TOPICS}

    assert topics == {
        "/lowstate": "unitree_hg/msg/LowState",
        "/lowcmd": "unitree_hg/msg/LowCmd",
        "/dex3/left/state": "unitree_hg/msg/HandState",
        "/dex3/right/state": "unitree_hg/msg/HandState",
        "/dex3/left/cmd": "unitree_hg/msg/HandCmd",
        "/dex3/right/cmd": "unitree_hg/msg/HandCmd",
        "/camera/color/image_raw": "sensor_msgs/msg/Image",
        "/camera/color/camera_info": "sensor_msgs/msg/CameraInfo",
    }
    assert not any("cube" in name or "plan" in name or "grasp" in name for name in topics)


def test_camera_recording_is_on_by_default_and_can_be_excluded_as_one_pair() -> None:
    default_topics = {topic.name for topic in recording.tabletop_raw_topics()}
    state_only_topics = {
        topic.name for topic in recording.tabletop_raw_topics(record_camera=False)
    }

    assert default_topics == {topic.name for topic in recording.TABLETOP_RAW_TOPICS}
    assert default_topics - state_only_topics == {
        "/camera/color/image_raw",
        "/camera/color/camera_info",
    }
    assert state_only_topics == {topic.name for topic in recording.TABLETOP_STATE_COMMAND_TOPICS}


def test_tabletop_cli_records_camera_by_default_and_exposes_explicit_skip() -> None:
    command = [
        "run-tabletop",
        "--arm",
        "right",
        "--network-interface",
        "test0",
        "--confirm",
        "test acknowledgement",
    ]

    assert build_parser().parse_args(command).skip_camera_recording is False
    assert (
        build_parser().parse_args([*command, "--skip-camera-recording"]).skip_camera_recording
        is True
    )


def test_recorder_command_is_plain_uncompressed_mcap(tmp_path: Path) -> None:
    command = recording.build_recorder_command(tmp_path / "bag", recording.TABLETOP_RAW_TOPICS)

    assert command[:7] == [
        "ros2",
        "bag",
        "record",
        "--output",
        str(tmp_path / "bag"),
        "--storage",
        "mcap",
    ]
    assert "--compression-mode" not in command
    assert "--compression-format" not in command
    assert "--storage-preset-profile" not in command
    assert "--include-unpublished-topics" in command


def test_read_bag_metadata_preserves_topic_types_and_counts(tmp_path: Path) -> None:
    bag = tmp_path / "bag"
    bag.mkdir()
    (bag / "bag_0.mcap").write_bytes(b"mcap")
    (bag / "metadata.yaml").write_text(
        yaml.safe_dump(
            {
                "rosbag2_bagfile_information": {
                    "storage_identifier": "mcap",
                    "duration": {"nanoseconds": 2_000_000_000},
                    "starting_time": {"nanoseconds_since_epoch": 123},
                    "message_count": 10,
                    "topics_with_message_count": [
                        {
                            "topic_metadata": {
                                "name": "/lowstate",
                                "type": "unitree_hg/msg/LowState",
                            },
                            "message_count": 10,
                        }
                    ],
                }
            }
        ),
        encoding="utf-8",
    )

    result = recording.read_bag_metadata(bag)

    assert result["storage_identifier"] == "mcap"
    assert result["duration_ns"] == 2_000_000_000
    assert result["message_count"] == 10
    assert result["topics"]["/lowstate"] == {
        "message_type": "unitree_hg/msg/LowState",
        "message_count": 10,
    }
    assert result["size_bytes"] > 4


def test_recorder_writes_spark_artifacts_and_audits_completion(
    tmp_path: Path, monkeypatch
) -> None:
    popen_call: dict = {}

    class FakeProcess:
        pid = 4321

        def __init__(self, command, **kwargs) -> None:
            self.returncode = None
            popen_call.update({"command": command, **kwargs})
            kwargs["stdout"].write("[INFO] [rosbag2_recorder]: Recording...\n")
            kwargs["stdout"].write("[INFO] [rosbag2_recorder]: Subscribed to topic '/lowstate'\n")
            kwargs["stdout"].flush()

        def poll(self):
            return self.returncode

        def wait(self, timeout=None):
            self.returncode = 0
            return 0

    signals: list[tuple[int, int]] = []
    monkeypatch.setattr(
        recording,
        "_git_provenance",
        lambda _repository: {"git_commit": "abc", "git_worktree_dirty": True},
    )
    monkeypatch.setattr(recording.subprocess, "Popen", FakeProcess)
    monkeypatch.setattr(
        recording.os,
        "killpg",
        lambda process_group, signal_number: signals.append((process_group, signal_number)),
    )
    topic = recording.TopicSpec(
        "/lowstate",
        "unitree_hg/msg/LowState",
        "measured state",
        "bag receipt time",
    )
    episode = tmp_path / "run_001" / "raw_episode"
    recorder = recording.RawEpisodeRecorder(
        episode,
        repository=Path(__file__).resolve().parents[1],
        topics=(topic,),
    )

    recorder.start()
    recorder.bag_directory.mkdir()
    (recorder.bag_directory / "bag_0.mcap").write_bytes(b"mcap")
    (recorder.bag_directory / "metadata.yaml").write_text(
        yaml.safe_dump(
            {
                "rosbag2_bagfile_information": {
                    "storage_identifier": "mcap",
                    "duration": {"nanoseconds": 1_000_000},
                    "starting_time": {"nanoseconds_since_epoch": 123},
                    "message_count": 2,
                    "topics_with_message_count": [
                        {
                            "topic_metadata": {
                                "name": topic.name,
                                "type": topic.message_type,
                            },
                            "message_count": 2,
                        }
                    ],
                }
            }
        ),
        encoding="utf-8",
    )

    summary = recorder.stop()
    manifest = json.loads(recorder.manifest_path.read_text(encoding="utf-8"))

    assert popen_call["command"] == recording.build_recorder_command(
        recorder.bag_directory, (topic,)
    )
    assert popen_call["stdin"] is recording.subprocess.DEVNULL
    assert popen_call["start_new_session"] is True
    assert signals == [(4321, recording.signal.SIGINT)]
    assert summary["complete"] is True
    assert manifest["capture"]["state"] == "completed"
    assert manifest["capture"]["complete"] is True
    assert manifest["episode"]["episode_id"] == "run_001"
    assert manifest["profile"]["camera_recording_enabled"] is False
    assert manifest["provenance"]["design_source"]["commit"] == recording.SPARK_COMMIT
    assert recorder.notes_path.is_file()


def test_recorder_refuses_start_without_every_requested_subscription(
    tmp_path: Path, monkeypatch
) -> None:
    class FakeProcess:
        pid = 4321

        def __init__(self, _command, **kwargs) -> None:
            self.returncode = None
            kwargs["stdout"].write("[INFO] [rosbag2_recorder]: Recording...\n")
            kwargs["stdout"].write("[INFO] [rosbag2_recorder]: Subscribed to topic '/lowstate'\n")
            kwargs["stdout"].flush()

        def poll(self):
            return self.returncode

        def wait(self, timeout=None):
            self.returncode = 0
            return 0

    monkeypatch.setattr(
        recording,
        "_git_provenance",
        lambda _repository: {"git_commit": "abc", "git_worktree_dirty": True},
    )
    monkeypatch.setattr(recording.subprocess, "Popen", FakeProcess)
    monkeypatch.setattr(recording.os, "killpg", lambda _process_group, _signal: None)
    topics = (
        recording.TopicSpec("/lowstate", "unitree_hg/msg/LowState", "state", "receipt"),
        recording.TopicSpec("/lowcmd", "unitree_hg/msg/LowCmd", "command", "receipt"),
    )
    recorder = recording.RawEpisodeRecorder(
        tmp_path / "run_missing" / "raw_episode",
        repository=Path(__file__).resolve().parents[1],
        topics=topics,
        startup_timeout_s=0.01,
    )

    with pytest.raises(RuntimeError, match="raw MCAP recorder did not start"):
        recorder.start()

    manifest = json.loads(recorder.manifest_path.read_text(encoding="utf-8"))
    assert manifest["capture"]["state"] == "failed_to_start"
    assert manifest["capture"]["problems"] == [
        "ros2 bag record did not subscribe to all requested topics before timeout: /lowcmd"
    ]


def test_required_empty_topic_marks_episode_incomplete(tmp_path: Path) -> None:
    topic = recording.TopicSpec("/required", "example/msg/State", "state", "receipt time")
    recorder = recording.RawEpisodeRecorder(
        tmp_path / "run_002" / "raw_episode",
        repository=Path(__file__).resolve().parents[1],
        topics=(topic,),
    )
    recorder.episode_directory.mkdir(parents=True)
    recorder._start_time_ns = 1
    recorder._provenance = {"git_commit": "abc", "git_worktree_dirty": False}

    class ExitedProcess:
        pid = 1
        returncode = 0

        def poll(self):
            return 0

    recorder._process = ExitedProcess()
    recorder.bag_directory.mkdir()
    (recorder.bag_directory / "metadata.yaml").write_text(
        yaml.safe_dump(
            {
                "rosbag2_bagfile_information": {
                    "storage_identifier": "mcap",
                    "duration": {"nanoseconds": 0},
                    "message_count": 0,
                    "topics_with_message_count": [],
                }
            }
        ),
        encoding="utf-8",
    )

    summary = recorder.stop()

    assert summary["complete"] is False
    assert summary["problems"] == ["required topic has no messages: /required"]
