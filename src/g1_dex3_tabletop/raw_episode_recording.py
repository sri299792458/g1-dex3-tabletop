"""Low-overhead raw ROS episode recording for a tabletop hardware run.

The recorder follows SPARK's live-capture boundary: delegate to a separate
``ros2 bag record`` process, write plain MCAP, and keep compression and dataset
conversion out of the robot-control process.
"""

from __future__ import annotations

import os
import signal
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, TextIO

import yaml

from g1_dex3_tabletop.planning.contracts import atomic_write_json

SPARK_REPOSITORY = "RPM-lab-UMN/spark-data-collection"
SPARK_COMMIT = "be284c2f8138f383d260526f68613c7a28d364d4"
SPARK_RECORDER_PATH = "data_pipeline/record_episode.py"
PROFILE_NAME = "g1_seated_tabletop_raw_v2"
STANDING_CALIBRATION_PROFILE_NAME = "g1_standing_calibration_raw_v1"


@dataclass(frozen=True)
class TopicSpec:
    """One stable raw-topic contract entry."""

    name: str
    message_type: str
    role: str
    timestamp_meaning: str
    required: bool = True


TABLETOP_STATE_COMMAND_TOPICS = (
    TopicSpec(
        "/lowstate",
        "unitree_hg/msg/LowState",
        "measured complete G1 state, including the pelvis IMU",
        "Unitree message tick/device fields retained; MCAP record time is laptop ROS receive time",
    ),
    TopicSpec(
        "/secondary_imu",
        "unitree_hg/msg/IMUState",
        "measured G1 torso IMU state",
        "MCAP record time at laptop DDS receipt; Unitree IMUState has no message header",
    ),
    TopicSpec(
        "/camera/gyro/sample",
        "sensor_msgs/msg/Imu",
        "raw D435i angular velocity",
        "message header stamp from the RealSense ROS producer; MCAP record time also retained",
    ),
    TopicSpec(
        "/camera/accel/sample",
        "sensor_msgs/msg/Imu",
        "raw D435i linear acceleration",
        "message header stamp from the RealSense ROS producer; MCAP record time also retained",
    ),
    TopicSpec(
        "/tf_static",
        "tf2_msgs/msg/TFMessage",
        "static D435i depth, color, gyroscope, and accelerometer frame transforms",
        "transform header stamps from the RealSense ROS producer; MCAP record time also retained",
    ),
    TopicSpec(
        "/lowcmd",
        "unitree_hg/msg/LowCmd",
        "complete 29-joint debug-lowcmd command stream",
        "MCAP record time at laptop DDS receipt",
    ),
    TopicSpec(
        "/dex3/left/state",
        "unitree_hg/msg/HandState",
        "measured left Dex3 motor state",
        "MCAP record time at laptop DDS receipt",
    ),
    TopicSpec(
        "/dex3/right/state",
        "unitree_hg/msg/HandState",
        "measured right Dex3 motor state",
        "MCAP record time at laptop DDS receipt",
    ),
    TopicSpec(
        "/dex3/left/cmd",
        "unitree_hg/msg/HandCmd",
        "left Dex3 motor command stream",
        "MCAP record time at laptop DDS receipt",
    ),
    TopicSpec(
        "/dex3/right/cmd",
        "unitree_hg/msg/HandCmd",
        "right Dex3 motor command stream",
        "MCAP record time at laptop DDS receipt",
    ),
)

TABLETOP_CAMERA_TOPICS = (
    TopicSpec(
        "/camera/color/image_raw",
        "sensor_msgs/msg/Image",
        "raw head-camera RGB observation",
        "message header stamp from the RealSense ROS producer; MCAP record time also retained",
    ),
    TopicSpec(
        "/camera/color/camera_info",
        "sensor_msgs/msg/CameraInfo",
        "head-camera intrinsics and rectified image profile",
        "message header stamp from the RealSense ROS producer; MCAP record time also retained",
    ),
    TopicSpec(
        "/camera/depth/image_rect_raw",
        "sensor_msgs/msg/Image",
        "native unaligned D435i Z16 depth observation",
        "message header stamp from the RealSense ROS producer; MCAP record time also retained",
    ),
    TopicSpec(
        "/camera/depth/camera_info",
        "sensor_msgs/msg/CameraInfo",
        "native D435i depth intrinsics and rectified image profile",
        "message header stamp from the RealSense ROS producer; MCAP record time also retained",
    ),
)

TABLETOP_RAW_TOPICS = TABLETOP_STATE_COMMAND_TOPICS + TABLETOP_CAMERA_TOPICS

# Reuse the official state/hand contracts, with standing arm-SDK commands.
# Calibration already retains selected camera frames in its session store.
STANDING_CALIBRATION_TOPICS = tuple(
    topic
    for topic in TABLETOP_STATE_COMMAND_TOPICS
    if topic.name == "/lowstate" or topic.name.startswith("/dex3/")
) + (
    TopicSpec(
        "/arm_sdk",
        "unitree_hg/msg/LowCmd",
        "standing arm commands including joint gains, feedforward torque, and slot-29 ownership weight",
        "MCAP record time at laptop DDS receipt; not robot receipt acknowledgement",
    ),
)


def tabletop_raw_topics(*, record_camera: bool = True) -> tuple[TopicSpec, ...]:
    """Select the stable tabletop profile; camera capture is enabled by default."""

    if record_camera:
        return TABLETOP_RAW_TOPICS
    return TABLETOP_STATE_COMMAND_TOPICS


def build_recorder_command(bag_directory: Path, topics: tuple[TopicSpec, ...]) -> list[str]:
    """Build the deliberately plain SPARK-style rosbag command."""

    return [
        "ros2",
        "bag",
        "record",
        "--output",
        str(bag_directory),
        "--storage",
        "mcap",
        "--include-unpublished-topics",
        *(topic.name for topic in topics),
    ]


def read_bag_metadata(bag_directory: Path) -> dict[str, Any]:
    """Read the small, recorder-written metadata summary without opening MCAP data."""

    metadata_path = bag_directory / "metadata.yaml"
    if not metadata_path.is_file():
        return {
            "metadata_present": False,
            "storage_identifier": None,
            "duration_ns": 0,
            "starting_time_ns": None,
            "message_count": 0,
            "size_bytes": 0,
            "topics": {},
        }
    document = yaml.safe_load(metadata_path.read_text(encoding="utf-8")) or {}
    information = document.get("rosbag2_bagfile_information", {})
    topic_results: dict[str, dict[str, Any]] = {}
    for entry in information.get("topics_with_message_count", []):
        topic_metadata = entry.get("topic_metadata", {})
        name = str(topic_metadata.get("name", "")).strip()
        if name:
            topic_results[name] = {
                "message_type": str(topic_metadata.get("type", "")),
                "message_count": int(entry.get("message_count", 0)),
            }
    size_bytes = sum(path.stat().st_size for path in bag_directory.rglob("*") if path.is_file())
    return {
        "metadata_present": True,
        "storage_identifier": information.get("storage_identifier"),
        "duration_ns": int(information.get("duration", {}).get("nanoseconds", 0)),
        "starting_time_ns": information.get("starting_time", {}).get("nanoseconds_since_epoch"),
        "message_count": int(information.get("message_count", 0)),
        "size_bytes": size_bytes,
        "topics": topic_results,
    }


def _git_provenance(repository: Path) -> dict[str, Any]:
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    dirty = bool(
        subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=repository,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    )
    return {"git_commit": revision, "git_worktree_dirty": dirty}


def _notes_text(
    episode_id: str, *, camera_recording_enabled: bool, profile_name: str = PROFILE_NAME
) -> str:
    return (
        "# Raw episode notes\n\n"
        f"- Episode: `{episode_id}`\n"
        f"- Profile: `{profile_name}`\n"
        f"- Camera recording enabled: `{str(camera_recording_enabled).lower()}`\n"
        "- Artifact role: source-of-truth asynchronous ROS capture\n"
        "- Storage: plain, untrimmed MCAP with no live compression\n\n"
        "## Operator notes\n\n"
        "No inline operator note was supplied by the hardware workflow. "
        "Add observations here after the run without modifying the bag.\n"
    )


class RawEpisodeRecorder:
    """Own one external ``ros2 bag record`` process and its durable manifest."""

    def __init__(
        self,
        episode_directory: Path,
        *,
        repository: Path,
        topics: tuple[TopicSpec, ...] = TABLETOP_RAW_TOPICS,
        startup_timeout_s: float = 8.0,
        profile_name: str = PROFILE_NAME,
    ) -> None:
        self.episode_directory = Path(episode_directory)
        self.repository = Path(repository)
        self.topics = topics
        self.profile_name = profile_name
        self.startup_timeout_s = float(startup_timeout_s)
        self.bag_directory = self.episode_directory / "bag"
        self.manifest_path = self.episode_directory / "episode_manifest.json"
        self.notes_path = self.episode_directory / "notes.md"
        self.log_path = self.episode_directory / "recorder.log"
        self.command = build_recorder_command(self.bag_directory, self.topics)
        self.camera_recording_enabled = bool(
            {topic.name for topic in TABLETOP_CAMERA_TOPICS}
            & {topic.name for topic in self.topics}
        )
        self._process: subprocess.Popen | None = None
        self._log: TextIO | None = None
        self._start_time_ns: int | None = None
        self._provenance: dict[str, Any] | None = None
        self._summary: dict[str, Any] | None = None

    @property
    def started(self) -> bool:
        return self._process is not None

    @property
    def summary(self) -> dict[str, Any] | None:
        return self._summary

    def check(self) -> None:
        """Verify the recorder survived the operator wait before commanding."""

        if self._process is None:
            raise RuntimeError("raw episode recorder was not started")
        return_code = self._process.poll()
        if return_code is not None:
            raise RuntimeError(f"raw MCAP recorder exited with {return_code}; see {self.log_path}")

    def _manifest(
        self,
        *,
        state: str,
        end_time_ns: int | None = None,
        recorder_exit_code: int | None = None,
        bag: dict[str, Any] | None = None,
        complete: bool = False,
        problems: list[str] | None = None,
    ) -> dict[str, Any]:
        assert self._start_time_ns is not None
        assert self._provenance is not None
        return {
            "schema_version": 1,
            "episode": {
                "episode_id": self.episode_directory.parent.name,
                "parent_run_directory": str(self.episode_directory.parent),
            },
            "profile": {
                "name": self.profile_name,
                "camera_recording_enabled": self.camera_recording_enabled,
            },
            "capture": {
                "state": state,
                "complete": complete,
                "start_time_ns": self._start_time_ns,
                "end_time_ns": end_time_ns,
                "storage": {
                    "bag_storage_id": "mcap",
                    "live_compression": "none",
                    "bag_directory": str(self.bag_directory),
                },
                "recorder_command": self.command,
                "recorder_exit_code": recorder_exit_code,
                "problems": problems or [],
                "bag": bag,
            },
            "recorded_topics": [asdict(topic) for topic in self.topics],
            "provenance": {
                **self._provenance,
                "design_source": {
                    "repository": SPARK_REPOSITORY,
                    "commit": SPARK_COMMIT,
                    "recorder_path": SPARK_RECORDER_PATH,
                },
            },
        }

    def start(self) -> None:
        """Start recording and require every requested rosbag subscription."""

        if self._process is not None:
            raise RuntimeError("raw episode recorder is already started")
        self.episode_directory.mkdir(parents=True, exist_ok=False)
        self._start_time_ns = time.time_ns()
        self._provenance = _git_provenance(self.repository)
        self.notes_path.write_text(
            _notes_text(
                self.episode_directory.parent.name,
                camera_recording_enabled=self.camera_recording_enabled,
                profile_name=self.profile_name,
            ),
            encoding="utf-8",
        )
        atomic_write_json(self.manifest_path, self._manifest(state="starting"))
        self._log = self.log_path.open("w", encoding="utf-8")
        try:
            self._process = subprocess.Popen(
                self.command,
                stdin=subprocess.DEVNULL,
                stdout=self._log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        except OSError as error:
            self._close_log()
            self._record_start_failure(f"could not launch ros2 bag record: {error}", None)
            raise RuntimeError(
                f"raw MCAP recorder could not launch; see {self.log_path}"
            ) from error
        deadline = time.monotonic() + self.startup_timeout_s
        missing_subscriptions = [topic.name for topic in self.topics]
        while time.monotonic() < deadline:
            return_code = self._process.poll()
            log_text = self.log_path.read_text(encoding="utf-8", errors="replace")
            if return_code is not None:
                self._close_log()
                self._record_start_failure(
                    f"ros2 bag record exited before startup with code {return_code}",
                    return_code,
                )
                raise RuntimeError(
                    f"raw MCAP recorder exited with {return_code}; see {self.log_path}"
                )
            missing_subscriptions = [
                topic.name
                for topic in self.topics
                if f"Subscribed to topic '{topic.name}'" not in log_text
            ]
            if "Recording..." in log_text and not missing_subscriptions:
                atomic_write_json(self.manifest_path, self._manifest(state="recording"))
                return
            time.sleep(0.05)
        self._terminate(signal.SIGINT, timeout_s=5.0)
        self._close_log()
        problem = (
            "ros2 bag record did not subscribe to all requested topics before timeout: "
            + ", ".join(missing_subscriptions)
            if missing_subscriptions
            else "ros2 bag record did not report Recording state before timeout"
        )
        self._record_start_failure(problem, self._process.returncode)
        raise RuntimeError(f"raw MCAP recorder did not start; see {self.log_path}")

    def _record_start_failure(self, problem: str, return_code: int | None) -> None:
        bag = read_bag_metadata(self.bag_directory)
        atomic_write_json(
            self.manifest_path,
            self._manifest(
                state="failed_to_start",
                end_time_ns=time.time_ns(),
                recorder_exit_code=return_code,
                bag=bag,
                problems=[problem],
            ),
        )
        self._summary = {
            "state": "failed_to_start",
            "complete": False,
            "episode_directory": str(self.episode_directory),
            "bag_directory": str(self.bag_directory),
            "manifest": str(self.manifest_path),
            "notes": str(self.notes_path),
            "recorder_log": str(self.log_path),
            "message_count": bag["message_count"],
            "size_bytes": bag["size_bytes"],
            "duration_ns": bag["duration_ns"],
            "problems": [problem],
        }

    def _terminate(self, signal_number: int, *, timeout_s: float) -> int | None:
        assert self._process is not None
        if self._process.poll() is None:
            os.killpg(self._process.pid, signal_number)
            try:
                self._process.wait(timeout=timeout_s)
            except subprocess.TimeoutExpired:
                os.killpg(self._process.pid, signal.SIGKILL)
                self._process.wait(timeout=5.0)
        return self._process.returncode

    def _close_log(self) -> None:
        if self._log is not None and not self._log.closed:
            self._log.close()

    def stop(self) -> dict[str, Any]:
        """Stop and audit the bag; the returned summary is safe for status.json."""

        if self._summary is not None:
            return self._summary
        if self._process is None:
            raise RuntimeError("raw episode recorder was not started")
        return_code = self._terminate(signal.SIGINT, timeout_s=20.0)
        self._close_log()
        bag = read_bag_metadata(self.bag_directory)
        problems: list[str] = []
        if return_code != 0:
            problems.append(f"ros2 bag record exit code was {return_code}")
        if not bag["metadata_present"]:
            problems.append("bag metadata.yaml is missing")
        if bag["storage_identifier"] != "mcap":
            problems.append(f"bag storage identifier is {bag['storage_identifier']!r}, not 'mcap'")
        for topic in self.topics:
            observed = bag["topics"].get(topic.name)
            if observed is None or int(observed["message_count"]) == 0:
                if topic.required:
                    problems.append(f"required topic has no messages: {topic.name}")
                continue
            if observed["message_type"] != topic.message_type:
                problems.append(
                    f"topic type mismatch for {topic.name}: "
                    f"{observed['message_type']!r} != {topic.message_type!r}"
                )
        complete = not problems
        end_time_ns = time.time_ns()
        state = "completed" if complete else "incomplete"
        manifest = self._manifest(
            state=state,
            end_time_ns=end_time_ns,
            recorder_exit_code=return_code,
            bag=bag,
            complete=complete,
            problems=problems,
        )
        atomic_write_json(self.manifest_path, manifest)
        self._summary = {
            "state": state,
            "complete": complete,
            "episode_directory": str(self.episode_directory),
            "bag_directory": str(self.bag_directory),
            "manifest": str(self.manifest_path),
            "notes": str(self.notes_path),
            "recorder_log": str(self.log_path),
            "message_count": bag["message_count"],
            "size_bytes": bag["size_bytes"],
            "duration_ns": bag["duration_ns"],
            "problems": problems,
        }
        return self._summary
