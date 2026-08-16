# G1 Dex3 Tabletop

Focused Unitree G1 software for two connected operations:

1. automatically collect dorsal-Dex3 calibration observations and solve the
   fixed-marker camera extrinsic with Mike Ferguson's Ceres optimizer;
2. use a removable calibration bundle and official NVLabs CuRobo to plan and
   execute a selected-Dex3 40 mm AprilCube pick, 100 mm lift, exact replacement,
   retreat, and controller handback.

The repository contains no historical manual-teaching pipeline or custom IK.
CuRobo is the only IK, collision, grasp-goal, attachment, and trajectory
planner. `running_notes.md` records the physical commissioning evidence,
rejected approaches, and remaining physical limits.

## Safety boundary

- Inspection, candidate generation, Ferguson solving, and CuRobo planning do
  not create robot command publishers.
- CUDA planning runs in one persistent separate Python 3.11 process per task.
  It is started and warmed before SPACE, never imports Unitree transport code,
  and remains alive for complete lifecycle planning plus measured-contact
  validation. ROS/control runs in Python 3.10 and keeps publishing through the
  commissioned fixed-rate Unitree controller while planning is in progress.
- Hardware commands require the exact harness/workspace acknowledgement and a
  second interactive SPACE after a read-only live preflight.
- Standing calibration uses `rt/arm_sdk`, full gravity feedforward, measured
  opposite-arm hold, and the independent PC2 Damp watchdog.
- Seated tabletop execution uses complete 29-joint `rt/lowcmd` ownership and
  restores Unitree control through the commissioned FSM `0 -> 1 -> 3` path.
- No contact, contact-geometry rejection, and failed 10 mm retention are task
  rejections: the controller opens or lowers as appropriate, follows exact
  frozen reverse trajectories to the supported start, and restores seated FSM
  3. State/transport/controller faults and Ctrl+C retain the independent PC2
  zero-torque safety path. Planning failure before publication cannot command
  the robot.

These are commissioned control mechanisms, not permission to skip the harness,
clear-sweep inspection, live preview, or deliberately slow first physical run.

## Pinned upstream code

- `third_party/curobo`: official NVLabs CuRobo.
- `third_party/robot_calibration`: Mike Ferguson's Ceres optimizer.
- `third_party/aprilcube`: the stateless marker correspondence detector.
- `third_party/unitree_ros`: the official G1/Dex3 model used by gravity
  feedforward.

Exact revisions and source provenance are machine-readable in
`config/provenance.json`.

## One-time setup

```bash
cd /home/kanth042/g1-dex3-tabletop
git submodule update --init --recursive
./tools/setup_control_env.sh
./tools/setup_planner_env.sh
./tools/install_robot_calibration_local.sh
./tools/setup_recording_benchmark.sh
./tools/g1_tabletop.sh inspect
```

The runtimes intentionally remain separate:

- `.venv`: ROS/control, capture, calibration analysis, and tests (Python 3.10).
- `.venv-planner`: Torch/Warp/CuRobo planning (Python 3.11).

Verify CUDA before any long plan:

```bash
.venv-planner/bin/python -c "import torch; assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0))"
```

If `nvidia-smi` works but that assertion fails, inspect `/dev/nvidia-uvm`.
The August 13 diagnosis on this laptop found `open('/dev/nvidia-uvm') = EIO`
after repeated suspend/resume cycles; that is a host NVIDIA/UVM failure, not a
CuRobo error. A reboot is the first recovery step.

The hardware commands also require the already commissioned account-local
CycloneDDS runtime, PC2 watchdog installation and SSH key, and the RealSense ROS
color node publishing the profile frozen in the hardware YAML.

The recording setup command installs the MCAP storage plugin under ignored
`deps/` without sudo. The raw topic contract, lifecycle, artifact layout,
failure semantics, and storage budget are documented in
[`docs/data-recording.md`](docs/data-recording.md).

## Automatic Dex3 calibration

Physical starting state: G1 standing in Ready/FSM 4 under the load-bearing
harness, both arms stationary, both Dex3 marker plates mounted, and the camera
running. The single command performs read-only bilateral Dex3 clearance and
calibration planning first, then asks for SPACE before it can create publishers:

```bash
cd /home/kanth042/g1-dex3-tabletop
./tools/g1_calibrate_hardware.sh right \
  --network-interface enp134s0 \
  --confirm 'I CONFIRM THE G1 IS SECURED BY THE LOAD-BEARING HARNESS AND THE WORKSPACE IS CLEAR'
```

Use `left` instead of `right` for the ID-5 left marker. The default is a finite
80-target route. Every pose is attempted once; a red/no-marker burst is
recorded as rejected and the route continues, while yellow is accepted with its
warning. The route returns to its measured handoff, restores the measured
finger posture and bilateral shoulder-clearance route, publishes terminal
arm-SDK weight zero, and finalizes the immutable session.

Build and solve one completed session without touching hardware:

```bash
SESSION=/home/kanth042/g1-dex3-tabletop/sessions/<session-name>
RUN=/home/kanth042/g1-dex3-tabletop/runs/calibrations/<session-name>
./tools/g1_tabletop.sh build-calibration-dataset \
  --session "$SESSION" --output "$RUN/dataset.json"
./tools/g1_tabletop.sh solve-calibration \
  --dataset "$RUN/dataset.json" --output-directory "$RUN/solve"
```

The sole production solve is Ferguson/Ceres with the CAD palm-to-marker
transform fixed. It uses a deterministic 64/16 split for an 80-observation
dataset, computes holdout residuals and bootstrap observability, and writes
`$RUN/solve/calibration_bundle.json`. The bundle is a hash-checked overlay:
select it with `--calibration-bundle`, or remove that argument to return cleanly
to the repository default. It never rewrites the base URDF.

## Cushion-versus-rigid seat diagnostic

This is deliberately separate from grasping and calibration validation. Tape
the frozen `DICT_5X5_50` 6x9 ChArUco board (30 mm squares, 22 mm markers) to the
table where it stays visible throughout both arm lifts. Use the same chair
frame, robot/table/head/feet/harness arrangement, and board placement for both
conditions; change only the cushion versus rigid non-slip seat support.

Each invocation takes one approval, acquires seated full-body lowcmd control,
plans both arms from the loaded measured state, then performs five pairs of:

1. left arm 100 mm table-normal lift and exact reverse;
2. right arm 100 mm table-normal lift and exact reverse.

The hands retain their measured posture. There is no cube, grasp, finger close,
or payload motion. The fixed-board pose is measured at loaded baseline and,
within every arm cycle, immediately before the lift, at the lifted endpoint, and
at the returned endpoint. `status.json` reports only those explicit same-cycle
comparisons; it does not use the earlier pre-planning loaded observation as a
lift baseline. Plain MCAP simultaneously records pelvis and torso IMUs,
waist/arm state and command, both Dex3 streams, raw D435i IMUs, native unaligned
depth, RGB, CameraInfo, and the factory camera transform tree.

```bash
cd /home/kanth042/g1-dex3-tabletop
./tools/g1_tabletop_hardware.sh measure-seat-compliance \
  --network-interface enp134s0 \
  --chair-condition cushion \
  --repetitions 5 \
  --confirm 'I CONFIRM THE G1 IS SECURED BY THE LOAD-BEARING HARNESS AND THE WORKSPACE IS CLEAR'
```

Repeat with only `--chair-condition rigid` changed after replacing the cushion
with the rigid support. Results are written under
`runs/seat_compliance_<condition>_<UTC>/`. A single condition can show camera
motion relative to the board; only the matched A/B comparison can assign an
excess to seat compliance.

The read-only endpoint, continuous-trajectory, and native-depth replay tools,
their measured results, observability limits, and proposed task integration are
documented in
[`docs/state-estimation-research.md`](docs/state-estimation-research.md).

## Tabletop cube task

Physical starting state: G1 seated in FSM 3, both arms supported and stationary
on the table, the printed 40 mm `dex3_safe_cube` resting flat on any face and
visible, the complete selected-arm sweep clear, and the RealSense node running.
Tabletop yaw is free;
the detected face identity is only a coordinate convention and does not limit
which face may be on top.

```bash
cd /home/kanth042/g1-dex3-tabletop
./tools/g1_tabletop_hardware.sh run-tabletop \
  --arm right \
  --network-interface enp134s0 \
  --calibration-bundle "$RUN/solve/calibration_bundle.json" \
  --confirm 'I CONFIRM THE G1 IS SECURED BY THE LOAD-BEARING HARNESS AND THE WORKSPACE IS CLEAR'
```

The direct-table behavior above remains the default. For the separate 50 mm
tripod presenter, tape its base to the table and place the cube centred and
yaw-aligned on the three pads, then add exactly one argument:

```bash
./tools/g1_tabletop_hardware.sh run-tabletop \
  --presentation tripod-h50 \
  --arm right \
  --network-interface enp134s0 \
  --calibration-bundle "$RUN/solve/calibration_bundle.json" \
  --confirm 'I CONFIRM THE G1 IS SECURED BY THE LOAD-BEARING HARNESS AND THE WORKSPACE IS CLEAR'
```

`tripod-h50` changes only the object presentation. It uses the exact printed
STL as a CuRobo obstacle, derives its base pose from the freshly observed cube,
and places the table plane 50 mm below the cube bottom. Camera observation,
calibration, persistent planning, arm/Dex3 control, contact-stall detection,
10 mm retention test, reverse recovery, seated restoration, and MCAP recording
are the same shared implementation. Omitting `--presentation tripod-h50`
removes the fixture completely from the request and scene.

The packaged tripod shortlist contains all 372 independently h50-qualified
candidates. Their achieved PhysX finger joints remain qualification evidence;
they are not robot commands. Every physical grasp uses the one fixed close
target from the Dex3 descriptor, while contact limits each finger's measured
travel. Every candidate is qualified for the common 70 mm approach used at
runtime; 365 and 353 also pass the longer 100 and 150 mm approaches respectively.
CuRobo receives all 372 in one goal set and chooses using live reachability and
complete-scene collision.

Before SPACE, this verifies the seated stationary state, both Dex3 states,
camera profile, and cube observation without creating publishers. After SPACE,
it acquires exact measured 29-joint lowcmd control, applies dual-Dex3 gravity
feedforward, observes the cube again in the loaded state, and sends one complete
lifecycle request to the already-warm isolated CuRobo worker for:

1. a straight supported-hand escape along the observed support-plane normal;
2. bounded branch-aware complete-path selection from the 15 shared
   GraspGen-X/Isaac-qualified Dex3 grasps, with the exact side adapter applied
   and every rejected IK branch and failure stage recorded;
3. approach and smooth finger closure toward the fixed descriptor close target;
4. a collision-aware 27-sphere conservative payload lift, exact reverse
   replacement, release, retreat, clearance return, and exact reverse
   supported return.

Physical closure does not require the fingers to reproduce the exact final
PhysX joint vector. The live controller first observes finger motion in the
commanded closing direction, then accepts a contact posture only when at least
one finger remains more than the commissioned `0.08 rad` endpoint tolerance
from the empty-hand target and the posture stays within the existing `0.01 rad`
stability band for `0.5 s`. No other finger is required to reach the target.
The already-recorded Dex3 pressure fields remain available in the MCAP for
offline analysis, but pressure is not a live pass/fail signal.

Because the physical contact posture can differ from the simulated posture,
the same worker rechecks the frozen payload route using the measured finger
angles; it does not replan or alter the arm samples. Its arm-plus-finger FK and
self-collision model is built once with the lifecycle and retained in memory.
In tripod mode this same recheck also measures every robot collision sphere
against the exact presenter mesh; direct mode incurs no fixture check.
The planned lift is split at the first sample at least `10 mm` above contact
without changing any sample. The fixed close target remains commanded during
that small lift. At its endpoint the controller collects a fresh `0.5 s` stable
window and requires at least one closing joint still to remain more than
`0.08 rad` short of the empty-hand target. The fingers may settle farther and
the blocked-joint set may change; reaching the complete empty-hand target is a
failed retention test. A failed test follows the exact frozen 10 mm reverse,
opens on the table, retreats, returns to the supported start, and restores
seated control.

Only the moving selected wrist, articulated hand, and payload are checked against
the locally observed support plane because one cube cannot reveal the table's
finite edges. Full G1 self-collision—including the fixed opposite arm, torso,
legs, both hands, and marker plates—remains enabled. The opposite arm receives
no changing target and is held at the measured takeover state with gravity
feedforward. The exact live start must also be collision-free in CuRobo's sphere
model: an overlap aborts before changing motion and prints the link pair and
penetration in millimetres so the operator can reposition the robot and rerun.
There is no start-state collision exception. Use `--arm left` to run the same
shared grasp set and lifecycle with the left Dex3.

For initial physical commissioning, `config/tabletop/task.yaml` limits every
selected-arm trajectory to `0.100 rad/s`. The limit is hash-bound into each
planning request and stored in planner provenance; the executor independently
rejects a plan above the hardware configuration's `0.200 rad/s` ceiling. Dex3
posture changes retain their separately commissioned two-second smooth ramp.

The persistent planner's complete output is streamed to the terminal and
retained as `planner.log` in that run directory. Terminal IK failures name the candidate
and colliding links/scene object with penetration or signed clearance when a
Cartesian-converged branch exists; otherwise they report the best pose residual.
The top-level failure status includes that final diagnostic and log path.

The task is a visual-localization and motion-execution test, not an independent
ground-truth calibration measurement. The selected calibration bundle itself
records its holdout residuals and validation status. The CUDA/UVM issue was
cleared by reboot and the complete planning path has been exercised offline on
the laptop GPU; physical execution still requires a deliberately slow first
commissioning run.

After SPACE, the same command also records a general raw episode under
`runs/tabletop_<UTC>/raw_episode/`: official complete G1 state (including the
pelvis IMU), the independently published torso IMU, lowcmd, both Dex3 states and
commands, raw D435i gyroscope and accelerometer streams, raw RGB, native
unaligned Z16 depth, both CameraInfo streams, and the RealSense static frame
transforms required for offline depth-to-color alignment. It is a separate
plain MCAP process with no live compression; task-specific cube and CuRobo
artifacts remain the adjacent JSON files. Recording starts before command
publishers and ends after terminal controller handback.

Raw camera recording is enabled by default. For a temporary lower-throughput
run, add `--skip-camera-recording`; this removes RGB, native depth, and their
CameraInfo streams from the MCAP completeness contract but leaves the
low-bandwidth D435i gyroscope and accelerometer streams, live camera, and all
perception behavior unchanged. Depth remains native and unaligned at 640x480x15;
PC2 does not generate a point cloud or perform live depth-to-color alignment.
The driver publishes the two raw motion streams separately; it does not
synthesize an orientation estimate.

## Verification

CPU-safe checks (no robot commands):

```bash
cd /home/kanth042/g1-dex3-tabletop
.venv/bin/pytest -q
.venv/bin/ruff check src tests
.venv/bin/ruff format --check src tests
./tools/g1_tabletop.sh inspect
```

CuRobo tests require the planner environment and working CUDA. The persistent
worker protocol still exchanges file/hash-bound immutable requests and results:
control code never imports CuRobo and the planner process never imports or
constructs Unitree transports.

### No-robot recording interference benchmark

Before adding continuous rosbag capture to a hardware run, exercise the exact
laptop-side load without exposing any Unitree topic:

```bash
cd /home/kanth042/g1-dex3-tabletop
./tools/g1_recording_benchmark.sh
```

The launcher forces `ROS_LOCALHOST_ONLY=1` on an isolated ROS domain and creates
only synthetic `/g1_recording_benchmark/...` topics. It compares the existing
250 Hz Python control-driver timing under three conditions: no recorder,
state/command-only plain MCAP, and state/command plus 1280x720 RGB8 at 15 Hz.
The account-local Humble MCAP plugin is downloaded into ignored `deps/`; no sudo
or system installation is used. Reports and bags are retained under ignored
`work/recording_benchmark/`.

This benchmark detects laptop scheduling, DDS, memory-copy, and disk contention.
It cannot commission physical recording safety or replace a stationary
robot-connected A/B test using the official Unitree topics.
