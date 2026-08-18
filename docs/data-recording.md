# Raw Episode Recording

## Purpose

This document defines the raw ROS recording contract for a G1 Dex3 tabletop
run. It is authoritative for:

- which topics are captured;
- what their timestamps mean;
- where the artifacts live;
- when recording starts and stops;
- how incomplete recording is reported;
- which work must remain offline.

The design follows SPARK's separation between raw capture, offline archive
generation, and published learning datasets. The implementation is adapted from
`RPM-lab-UMN/spark-data-collection` commit
`be284c2f8138f383d260526f68613c7a28d364d4`, principally
`data_pipeline/record_episode.py` and its artifact documentation.

## Ownership Boundary

The tabletop runtime owns robot safety, perception, CuRobo planning, and command
execution. A separate `ros2 bag record` process owns raw data persistence.

The recorder does not:

- create Unitree command publishers;
- participate in the 250 Hz controller loop;
- evaluate safety or trigger Damp/zero torque;
- compress images;
- encode CuRobo plans or task decisions into new ROS messages;
- convert data into LeRobot format.

This keeps recording failure from becoming a new robot-control decision.

## One Run, One Raw Episode

Every approved `run-tabletop` or `measure-seat-compliance` invocation creates
the same raw-episode subtree inside its normal run directory. Task-specific JSON
files remain beside it.

```text
runs/tabletop_<UTC>/
├── status.json
├── loaded_request.json
├── supported_escape.json
├── clearance_request.json
├── pregrasp_plan.json
├── pregrasp_estimated_request.json
├── pregrasp_remaining_plan.json
├── pregrasp_corrected_task_plan.json
├── grasp_close.json
├── retention_route_validation.json
├── retention_evidence.json
├── planner.log
└── raw_episode/
    ├── bag/
    │   ├── bag_0.mcap
    │   └── metadata.yaml
    ├── episode_manifest.json
    ├── notes.md
    └── recorder.log
```

Those five boundary files are produced by the default `trajectory` controller.
The experimental `mpc` controller instead retains `execution_plan.json` and
`task_plan.json` because it still plans one frozen lifecycle at clearance.
Rejected runs contain only the artifacts that were completed before rejection.
`grasp_close.json` records the stable supported close relative to the
commissioned empty-close posture. `retention_evidence.json` records the fresh
post-lift repetition of the same thumb-plus-opposing-finger obstruction test.
Raw pressure and `tau_est` remain in the official Dex3 state messages inside
the MCAP as diagnostics; neither signal decides the live grasp result.

The MCAP is the source of truth for asynchronous sensor, measured-state, and
command streams. Existing run JSON files are the source of truth for task
interpretation, selected grasps, planning, and failure diagnostics. Neither
artifact impersonates the other.

## Topic Contract

The `g1_seated_tabletop_raw_v2` profile records the official Unitree transport
streams through their ROS graph names. CycloneDDS exposes Unitree's SDK channel
`rt/lowstate` as ROS topic `/lowstate`: the SDK's `rt` partition is not part of
the ROS topic name. The same mapping applies to LowCmd and Dex3.

| Topic | Message type | Meaning | Required |
|---|---|---|---:|
| `/lowstate` | `unitree_hg/msg/LowState` | Complete measured G1 state, including IMU, motor state, mode, tick, and remote fields | yes |
| `/secondary_imu` | `unitree_hg/msg/IMUState` | Independent torso IMU | yes |
| `/camera/gyro/sample` | `sensor_msgs/msg/Imu` | Raw D435i gyroscope | yes |
| `/camera/accel/sample` | `sensor_msgs/msg/Imu` | Raw D435i accelerometer | yes |
| `/tf_static` | `tf2_msgs/msg/TFMessage` | Factory transforms among D435i depth, color, gyro, and accel frames | yes |
| `/lowcmd` | `unitree_hg/msg/LowCmd` | Complete 29-joint debug-lowcmd command stream used by the seated task | yes |
| `/dex3/left/state` | `unitree_hg/msg/HandState` | Measured left Dex3 motor state | yes |
| `/dex3/right/state` | `unitree_hg/msg/HandState` | Measured right Dex3 motor state | yes |
| `/dex3/left/cmd` | `unitree_hg/msg/HandCmd` | Left Dex3 motor commands | yes |
| `/dex3/right/cmd` | `unitree_hg/msg/HandCmd` | Right Dex3 motor commands | yes |
| `/camera/color/image_raw` | `sensor_msgs/msg/Image` | Raw head-camera RGB image | yes |
| `/camera/color/camera_info` | `sensor_msgs/msg/CameraInfo` | Intrinsics and rectified camera profile | yes |
| `/camera/depth/image_rect_raw` | `sensor_msgs/msg/Image` | Native unaligned 640x480 Z16 depth | yes |
| `/camera/depth/camera_info` | `sensor_msgs/msg/CameraInfo` | Native depth intrinsics and profile | yes |

Camera recording is enabled by default. Passing `--skip-camera-recording`
removes RGB, native depth, and both CameraInfo topics from the selected profile
and completeness audit. It retains the RealSense gyro, accelerometer, and
`/tf_static`; it also does not stop the RealSense or remove live perception.
The manifest records `profile.camera_recording_enabled` so a state-only episode
cannot be mistaken for a dropped-camera episode.

The Unitree `rt/arm_sdk` channel is intentionally absent: seated tabletop
execution uses `rt/lowcmd`. A future standing-calibration recording profile may
select `rt/arm_sdk`, but the tabletop profile must not record an idle command
surface and imply that it controlled this run.

The bag deliberately excludes:

- AprilCube detections and fitted poses;
- table planes;
- grasp candidates and selected candidate IDs;
- CuRobo requests, trajectories, and collision diagnostics;
- calibration-result interpretations;
- derived rewards or task-success labels.

Those are task-specific or derived artifacts already represented by the run
JSON files, or work for an offline conversion stage.

## Timestamp Contract

Rosbag preserves a receive/record timestamp for every message. Interpret the
streams as follows:

- `/lowstate`, `/lowcmd`, and Dex3 topics use laptop ROS/DDS receipt time
  for cross-topic bag alignment. Unitree `tick` and other device fields remain
  in the original message and must not be discarded.
- RGB, depth, both CameraInfo streams, and D435i IMUs retain the RealSense ROS
  producer's message header stamp as well as the bag record timestamp.
- `/secondary_imu` has no ROS header; use laptop DDS receipt time. `/tf_static`
  retains each transform's producer header stamp.
- Host time synchronization remains an operational prerequisite. The bag does
  not repair a clock that was reset or unsynchronized during capture.

No derived message is stamped later and presented as if it were captured at the
sensor.

## Runtime Lifecycle

The sequence is fixed:

1. The normal read-only seated, hand, camera, and cube preflight passes.
2. The operator presses SPACE.
3. Frozen configuration files are rechecked.
4. The external plain-MCAP recorder starts and must report `Recording...` plus
   an active subscription for every selected topic. `--include-unpublished-topics`
   lets the command subscriptions exist before the controller publishers are
   constructed.
5. Only then may activation be reacquired and Unitree command publishers be
   constructed.
6. The recorder remains active through perception, CuRobo planning, execution,
   controller shutdown, Dex3 timeout, and PC2 handback.
7. After control cleanup, the parent sends SIGINT to the recorder's process
   group and audits `metadata.yaml`.

The recorder's stdin is disconnected because the parent owns the single
operator keyboard interaction. This is the only embedding-specific difference
from SPARK's standalone recorder loop.

## Failure Semantics

Before robot command creation, failure to start MCAP aborts the run. No motion
publisher has been constructed at that point.

After command creation, recording is observational:

- it is never polled from the 250 Hz loop;
- a recorder-finalization problem does not request a new robot mode;
- robot cleanup and PC2 handback happen before recorder shutdown;
- `status.json` and `episode_manifest.json` report recording separately from
  task success.

An episode is `complete` only when:

- `ros2 bag record` exits cleanly after SIGINT;
- every required topic has at least one message;
- every observed message type matches this contract.

Otherwise the retained episode is marked `incomplete` with explicit problems.
It can still be useful for diagnosis, but must not silently enter a training
dataset as a complete take.

## Capture, Archive, And Published Data

The live capture is intentionally:

- plain MCAP;
- untrimmed;
- uncompressed;
- never rewritten in place.

Compression, head/tail trimming, image transcoding, temporal resampling, and
LeRobot export belong in later offline tools. The original episode remains
available if any derived job fails.

The no-robot benchmark at
`work/recording_benchmark/20260814T160128Z/report.json` measured approximately
`39.86 MiB/s` for state/command topics plus raw 1280x720 RGB8 at 15 Hz. Budget
an additional measured `8.9 MiB/s` for native depth plus camera motion streams,
or roughly `2.9 GiB/min` for the current combined profile, and check free space
before a physical run. A nominal 24 GiB free allocation is only about eight
minutes at that conservative combined rate.

## One-Time Account-Local Setup

No sudo installation is required:

```bash
cd /home/kanth042/g1-dex3-tabletop
./tools/setup_recording_benchmark.sh
```

The hardware launcher then:

- exposes the account-local `rosbag2_storage_mcap` plugin under `deps/`;
- sources `/home/kanth042/g1pilot_ws/install/setup.bash` for the official
  `unitree_hg` ROS message type support;
- verifies MCAP and all four required Unitree message interfaces before it
  starts the RealSense or the tabletop program.

Sourcing that workspace provides message introspection only. It does not launch
G1Pilot or create another controller. Set `G1_TABLETOP_UNITREE_ROS_SETUP` only
if the same official message installation is intentionally located elsewhere.

## Inspection

After a run:

```bash
cd /home/kanth042/g1-dex3-tabletop
RUN=/home/kanth042/g1-dex3-tabletop/runs/tabletop_<UTC>
source /opt/ros/humble/setup.bash
export AMENT_PREFIX_PATH="$PWD/deps/rosbag2_mcap_prefix/opt/ros/humble:${AMENT_PREFIX_PATH:-}"
export LD_LIBRARY_PATH="$PWD/deps/rosbag2_mcap_prefix/opt/ros/humble/lib:${LD_LIBRARY_PATH:-}"
ros2 bag info "$RUN/raw_episode/bag"
python -m json.tool "$RUN/raw_episode/episode_manifest.json"
```

Review `status.json` and `raw_episode/episode_manifest.json` together. Task
completion does not imply recording completeness, and recording completeness
does not imply task success.

## Local Space Cleanup

Delete one retained MCAP payload with an explicit path:

```bash
cd /home/kanth042/g1-dex3-tabletop
./tools/delete_mcap.sh runs/tabletop_<UTC>/raw_episode/bag/<file>.mcap
```

The command shows the exact file and size and requires typing `DELETE`. It
refuses paths outside `runs/tabletop_*/raw_episode/bag/` and deletes neither the
run diagnostics nor the small recording manifest. After deletion, the retained
manifest describes the original capture, but the raw episode is no longer
available for playback or dataset conversion.

## Commissioning Boundary

The synthetic benchmark showed no scheduling gap above 10 ms in three 15-second
raw-RGB trials on this laptop. It did not connect to Unitree topics and cannot
replace one stationary, harnessed robot-connected A/B run before treating the
recording profile as physically commissioned.
