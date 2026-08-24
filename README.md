# G1 Dex3 Tabletop

Focused Unitree G1 software for two connected operations:

1. automatically collect dorsal-Dex3 calibration observations and solve the
   fixed-marker camera extrinsic with Mike Ferguson's Ceres optimizer;
2. use a removable calibration bundle and official NVLabs CuRobo to plan and
   execute a selected-Dex3 hash-bound AprilCube pick, 100 mm lift, exact replacement,
   retreat, and controller handback.

The repository contains no historical manual-teaching pipeline or custom IK.
CuRobo is the only IK, collision, grasp-goal, attachment, and trajectory
planner. `running_notes.md` records the physical commissioning evidence,
rejected approaches, and remaining physical limits.

## Safety boundary

- Inspection, candidate generation, Ferguson solving, and CuRobo planning do
  not create robot command publishers.
- CUDA planning runs in one persistent separate Python 3.11 process per task.
  It is launched as soon as the hardware command starts, never imports Unitree
  transport code,
  and remains alive for the reversible supported escape, clearance-boundary
  pregrasp selection, corrected remaining-task plan, MPC, and measured-contact
  validation. The default trajectory path retains a small lazy per-arm pool of
  fixed-shape open-hand and attached-payload optimizers plus their strict and
  fixed-close checkers. Compatible locked-joint values, scenes, and attachments
  are updated in place; left/right or open/payload topologies never share a
  slot. As soon as the read-only preflight supplies the selected arm and object
  topology, the worker constructs and exercises the selected arm's open and
  payload optimizer graphs while the preview and SPACE prompt remain active.
  MPC mode also constructs its one retained moving-grasp solver there. This is
  command-free topology warmup: it does not create a nominal task or replace
  fresh boundary feasibility. After SPACE, the retained objects are rebound to
  the fresh loaded and clearance states without rebuilding their CUDA graphs.
  ROS/control runs in Python 3.10 and keeps publishing through the
  commissioned fixed-rate Unitree controller while planning is in progress.
- Hardware commands require the exact harness/workspace acknowledgement and a
  second interactive SPACE after a read-only live preflight.
- Standing calibration uses `rt/arm_sdk`, full gravity feedforward, measured
  opposite-arm hold, and the independent PC2 Damp watchdog.
- Seated tabletop execution uses complete 29-joint `rt/lowcmd` ownership and
  restores Unitree control through the commissioned FSM `0 -> 1 -> 3` path.
- No contact, contact-geometry rejection, and failed 30 mm retention are task
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
The selected hybrid observer is an independent numerical library component,
not a separate process. It uses only a visual anchor, three waist joints, and
the pelvis/torso orientations. The default trajectory workflow now anchors it
to the fixed cube at clearance and consumes one propagated estimate at the
stationary pregrasp boundary. CuRobo then replans the same selected grasp and
the complete remaining lifecycle from the exact active command. At clearance,
only the reversible route to pregrasp is exposed to the controller; candidate
selection also checks the unexecuted linear grasp approach so an
already-invalid grasp is not knowingly approached. Depth remains an independent
recorded/replay measurement rather than an unvalidated estimator input.

## Tabletop cube task

Physical starting state: G1 seated in FSM 3, both arms supported and stationary
on the table, the selected AprilCube resting flat on any face and visible, the
complete selected-arm sweep clear, and the RealSense node running. Object
selection is always explicit: use `cube40-r3` for the printed 40 mm
`dex3_safe_cube` or `cube60-r3` for the 60 mm R3 cube.
Tabletop yaw is free;
the detected face identity is only a coordinate convention and does not limit
which face may be on top.

```bash
cd /home/kanth042/g1-dex3-tabletop
./tools/g1_tabletop_hardware.sh run-tabletop \
  --arm right \
  --network-interface enp134s0 \
  --object-profile cube40-r3 \
  --calibration-bundle "$RUN/solve/calibration_bundle.json" \
  --confirm 'I CONFIRM THE G1 IS SECURED BY THE LOAD-BEARING HARNESS AND THE WORKSPACE IS CLEAR'
```

For the 60 mm R3 print with 45 mm `DICT_4X4_100` markers 10--15, select it
instead:

```bash
  --object-profile cube60-r3
```

An object profile hash-binds the detector geometry, exact qualified collision
mesh, dimensions, and direct-table grasp shortlist. `cube40-r3` contains the
existing five candidates; `cube60-r3` contains all 57 unchanged candidates
that passed intrinsic retention and the stationary-cube fixed-close 5 mm table
contract. No controller or safety setting changes with the profile.

The default replays the complete frozen CuRobo MotionGen lifecycle. The
experimental `--motion-controller mpc` path keeps MotionGen for global,
payload, placement, and return motion and uses MPC only for the visually
updated pregrasp-to-grasp segment. The stationary cube observation at clearance
defines its frozen reference frame; no separate board is required. The
AprilCube must remain detectable during the approach. The exact lifecycle,
window checks, retained offline evidence, and current commissioning status are
documented in
[`docs/tabletop-mpc.md`](docs/tabletop-mpc.md).

With the default `trajectory` controller, the program performs one additional
stationary correction after reaching pregrasp. The original clearance image
remains the visual anchor; no hand marker or second cube image is required at
pregrasp. The clearance transaction serializes only the validated
clearance-to-pregrasp route and its exact reverse; it does not compute a
provisional payload lifecycle that will be discarded. The warmed per-arm
CuRobo pool reuses the compatible open-hand, strict, fixed-close, and payload
objects across planning boundaries and source/destination queries. If the
synchronized waist/IMU inputs, same-grasp replan, post-plan
state check, or atomic plan installation fails, the arm exactly reverses the
already validated pregrasp route and then the supported escape.
Before that pregrasp route is accepted, the planner holds its exact grasp
contact arm pose and checks the complete descriptor open-to-close finger sweep
against strict self collision. For a direct cube, the hand/table result is the
object profile's hash-bound exact Dex3 collision-mesh sweep, while the live
wrist retains the 5 mm plane check. Fixture modes retain their live table and
fixture checks. In fixture mode, the target-fixed hand portion of every
51-sample close sweep is first checked in CUDA batches; invalid candidates are
removed before arm IK or trajectory optimization. The surviving arm IK
endpoints are then strict-collision-checked in one GPU pass before route
planning. After physical closure, the measured stable-close angles still
revalidate the frozen payload route before beginning the payload lift. If the
actual contact-stalled fingers are already positively above the table but less
than 5 mm away, the 30 mm test lift may escape from that exact boundary and its
return must be the exact reverse: neither side may move closer than the measured
boundary, and every intervening free-space sample must retain 5 mm.
Each retained grasp receives one independent CuRobo IK problem with 16 seeds.
The resulting finite joint-solution pool is then tested by the unchanged
single-route planner and strict validators; candidates do not compete for one
shared 16-seed goal set.
With `--motion-controller mpc`, the cube must remain stationary through the
clearance observation. That pose is frozen as the reference frame. Each local
grasp-approach window then uses a fresh cube image plus pelvis, waist, and torso
state to update the Cartesian goal relative to it. After the hand closes, the
attached-payload lifecycle is rebuilt at the reached object pose and executed
as frozen MotionGen trajectories. Heavy CuRobo structures are paid at the
stationary clearance boundary; the retained composed replay reduced the
post-grasp rebuild to about 2.0 seconds. This path still requires staged
hardware commissioning. See [`docs/tabletop-mpc.md`](docs/tabletop-mpc.md).

The direct-table presentation above remains the default. To use the prime
tower, fix its base to the table and place the selected cube centred and
yaw-aligned on its top. Pass the object and presentation independently:

```bash
./tools/g1_tabletop_hardware.sh run-tabletop \
  --arm right \
  --network-interface enp134s0 \
  --object-profile cube40-r3 \
  --presentation prime-tower \
  --calibration-bundle "$RUN/solve/calibration_bundle.json" \
  --confirm 'I CONFIRM THE G1 IS SECURED BY THE LOAD-BEARING HARNESS AND THE WORKSPACE IS CLEAR'
```

`prime-tower` changes only the object presentation. It uses the supplied
35.42 x 34.75 x 60 mm STL as a CuRobo obstacle, derives its base pose from the
freshly observed cube, and places the table plane 60 mm below the cube bottom.
It supports both `cube40-r3` and `cube60-r3`; replace the object-profile value
with `cube60-r3` for the larger cube. Camera observation,
calibration, persistent planning, arm/Dex3 control, commissioned empty-close
obstruction detection, 30 mm lifted retention checkpoint, reverse recovery,
seated restoration, and MCAP recording are the same shared implementation. Omitting
`--presentation prime-tower`
removes the fixture completely from the request and scene.

The packaged prime-tower shortlists contain 113 unchanged 40 mm poses and 312
unchanged 60 mm poses, selected from their complete 3,178- and 3,279-pose
intrinsic-retention pools. Qualification uses the real descriptor close—not a
candidate-specific PhysX endpoint—and checks the exact 70 mm open approach and
all 51 fixed-close samples against the exact tower and table. CuRobo repeats
the fixed-close pruning against the live placed scene, gives survivors
independent batched arm IK problems, and reserves expensive trajectory
optimization for ranked strict-endpoint-valid branches.

Before SPACE, this verifies the seated stationary state, both Dex3 states,
camera profile, and cube observation without creating publishers. After SPACE,
it acquires exact measured 29-joint lowcmd control, applies dual-Dex3 gravity
feedforward, and observes the fixed cube in the loaded state. The worker first
freezes only the supported escape and its exact reverse. After that lift reaches
clearance, the controller holds the exact arm command, sends the already-required
descriptor open command to the empty selected hand, and records the achieved
finger posture. It then observes the unchanged cube again and plans the only
grasp lifecycle eligible for execution from that fresh camera/body and measured
empty-open state:

1. a straight supported-hand escape along the observed support-plane normal;
2. bounded branch-aware complete-path selection from every stationary-cube,
   fixed-close-qualified grasp in the selected presentation/profile pair
   (direct: five for 40 mm or 57 for 60 mm; prime tower: 113 or 312), with the
   exact side adapter applied
   and every rejected IK branch and failure stage recorded;
3. approach and smooth finger closure toward the fixed descriptor close target;
4. a collision-aware 27-sphere conservative payload lift, exact reverse
   replacement, release, retreat, clearance return, and exact reverse
   supported return.

The cube begins that lift in intentional contact with the prime tower. During
the direct separation and its exact reverse, only the attached-cube/tower pair
is excluded from CuRobo's optimizer world. The closed hand and every other robot
sphere are independently checked against the exact tower mesh over the
generated route; the cube-support exemption does not permit hand contact.

The cube is the task-local table anchor during this boundary update; it must not
move between the loaded and clearance observations. The run records both image
bursts, both hash-bound requests, and the inferred camera motion in the cube
frame. A failed boundary observation or task replan follows the already-frozen
clearance-to-handoff reverse and stops without opening the hand.

Physical closure does not require the fingers to reproduce the exact final
PhysX joint vector. The selected hand has a separately commissioned measured
empty-close posture for the exact descriptor close command. During closure,
starting the low retention lift requires observed commanded-direction finger
motion, posture spread within `0.01 rad` for `0.5 s`, and at least `0.05 rad`
shortfall from that empty close on both sides of the grasp: one thumb closing
joint and one middle/index closing joint. A residual on only one side is an
empty or pushed-cube result and is rejected. Raw pressure, `tau_est`, and
velocity remain recorded diagnostics and never decide the live result.

The run-start finger posture is arbitrary and is retained only for restoration
before the exact supported return. It is not an open-hand reference. Open-hand
collision geometry comes from the selected hand posture measured after the
descriptor open command at clearance. The same descriptor target remains
published after exact cube replacement, but completion is checked against that
run-local measured empty-open posture using the commissioned tracking tolerance;
the open command remains active throughout retreat. The two targets and both
measured references are recorded in `dex3_run_local_references.json`. The task
does not add an empty close/open cycle: the persistent empty-close commission is
unchanged, and the empty-open measurement comes from the opening motion already
required before pregrasp.

Because the physical contact posture can differ from the simulated posture,
the same worker rechecks the frozen payload route using the measured finger
angles; it does not replan or alter the arm samples. Its arm-plus-finger FK and
self-collision model is built once with the lifecycle and retained in memory.
In presentation mode this same recheck also measures every robot collision sphere
against the exact presenter mesh; direct mode incurs no fixture check.
The one planned payload lift is split at its first sample at least `30 mm`
above contact without changing any sample or invoking the planner again. The
fixed close target remains commanded during this first segment. Only after
this separation does the controller collect a fresh `0.5 s` stable window and
repeat the same commissioned empty-close test. The obstructed thumb and
opposing-finger motor IDs may change as the cube settles, but both sides must
still exceed the `0.05 rad` shortfall. Losing either side fails the checkpoint.
Failure keeps the hand closed through the exact frozen 30 mm reverse, opens
only after returning to support, retreats, returns to the supported start, and
restores seated control.

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

For every post-escape wrist/hand route, merely positive table clearance is not
accepted. `config/tabletop/task.yaml` requires 5 mm, matching the commissioned
minimum collision clearance used by the calibration route validator. Frozen
open/closed planning and every MPC window retain that hash-bound floor. The
measured-contact payload route has one narrower boundary rule: a positive
contact-stalled grasp below 5 mm may only escape without decreasing its starting
clearance, must attain and retain 5 mm in free space, and must return over the
exact reverse without going below the same boundary. The rule does not apply to
the resting cube or its conservative payload proxy at initial support contact.

By default, `config/tabletop/task.yaml` limits every selected-arm trajectory to
`0.200 rad/s`, following successful physical trials with both the 40 mm and
60 mm tabletop cubes. A run can request a slower value with
`--maximum-arm-velocity-rad-s`; the selected value is hash-bound into each
planning request and stored in planner provenance. The executor independently
rejects anything above the hardware configuration's `0.200 rad/s` ceiling.
Dex3 posture changes retain their separately commissioned two-second smooth
ramp.

The direct-table object profile supplies the default pregrasp distance. A run
may override it with `--pregrasp-distance-m`; the resolved positive distance is
hash-bound into every planning request, and CuRobo rechecks the complete IK,
collision, table-clearance, approach, and reverse-route contracts at that
distance. Both direct-table 60 mm profiles currently default to `0.050 m`.

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

### Direct two-cube stack

`run-stack` is one explicit workflow, not a generic task language. Both 60 mm
R3 cubes start on the bare table and must be jointly visible. The original cube
uses `DICT_4X4_100` IDs 10–15 and the second print uses IDs 20–25. Either cube
may be picked; the other stays fixed and becomes the placement support. The
robot starts seated in FSM 3 with both arms supported and stationary on the
table.

```bash
cd /home/kanth042/g1-dex3-tabletop
./tools/g1_tabletop_hardware.sh run-stack \
  --network-interface enp134s0 \
  --calibration-bundle config/calibrations/dex3_shared_20260812_selected_free.json \
  --confirm 'I CONFIRM THE G1 IS SECURED BY THE LOAD-BEARING HARNESS AND THE WORKSPACE IS CLEAR'
```

The read-only preflight orders both arms by their nearest cube but does not
discard any assignment. After ownership, the controller lifts only the first
arm and tests both directed tasks: primary onto secondary and secondary onto
primary. If neither has a complete plan, that hand and arm return to their exact
supported starts before the other arm is lifted and both directions are tested
again. Execution stops at the first complete plan. At clearance, the planner
first checks all source/destination grasp endpoints, then performs expensive
route planning only for a grasp that is viable at both endpoints. Common
candidates are tried in ascending order of their best source-plus-destination
IK joint distance from the measured clearance state. This reuses the existing
IK solutions; it does not add a wrist-specific weight or a new acceptance
threshold.

CuRobo launches as soon as `run-stack` starts. After the read-only two-cube
observation, the persistent CUDA worker warms both fixed-waist arms' open-hand
and attached-60-mm-cube MotionGen models while the camera preview remains
active. Pressing Space still creates no publisher until that command-free
warmup has completed. Live supported-escape and task solves remain after
ownership because their joint and object states do not exist beforehand.

The selected assignment picks one cube and places it directly on the other. The
stationary cube remains a finite collision obstacle during pickup and becomes
the finite placement support at the destination. No midpoint, preliminary cube
relocation, table boundary, separate board, or second pick is involved. The
selected arm finally reverses its exact supported escape; the unused arm never
leaves its supported start. Only one arm is ever away from its supported start
at a time.

After a completed or safely rejected episode, `run-stack` keeps the warmed
planner, camera, 250 Hz controller, PC2 watchdog, and Dex3 transport alive at
the supported start. Reposition both cubes and press Space to create another
ordinary `runs/stack_<timestamp>/` run with its own status, images, plans,
planner log, and MCAP. Ctrl+C while waiting between episodes performs the one
clean seated handback and exits. Ctrl+C during observation, planning, or motion
remains a fault and follows the existing Zero Torque cleanup path.

During the attached transfer, CuRobo retains full-robot self-collision and the
stationary support cube as an explicit obstacle. It does not add a whole-robot
table cuboid because the unused arm is deliberately supported on that table.
The already-established local table-plane check independently validates every
sample of the moving wrist/hand and attached cube.

The direct-stack request supplies the four upright quarter-turn wrist targets
to CuRobo as one goal set because redundant arm IK and hand clearance depend on
wrist orientation. CuRobo returns the selected goal-set index; the coordinator
does not run four yaw-specific planning pipelines. This remains only a nominal
path choice. The software does not assume that the physical cube retains an
exact yaw inside Dex3. Placement targets the moving cube's nominal center over
the observed support-cube center, so unavoidable in-hand rotation does not
create a false exact-yaw success claim.

Destination planning carries the original on-table cube observation as explicit
evidence for the unchanged real table plane. It does not infer a new plane under
the elevated destination and therefore does not turn the top of the finite
support cube into a large fictitious raised table.

Planning and measured-close validation reuse the single-cube CuRobo, Dex3,
gravity-feedforward, controller, watchdog, RealSense, and MCAP implementations.
The fixed stack executor currently retains its complete clearance-boundary
pick/place plans; the single-cube trajectory mode's additional pregrasp
state-correction replan is deliberately not part of this first stack
commissioning path.
Expected grasp rejection opens the active hand and returns it through the frozen
grasp-retreat and clearance routes. The default `--grasp-retries 1` then takes a
fresh joint/camera snapshot, detects both cubes again, excludes the physically
failed grasp candidate, and replans the same moving-cube/arm choice. A second
grasp rejection ends the task normally and returns the selected arm. Planner,
controller, transport, and watchdog faults are never retried and retain the
existing fail-closed behavior. `--maximum-arm-velocity-rad-s` and
`--pregrasp-distance-m` are hash-bound per-run overrides;
`--skip-camera-recording` has the same meaning as in `run-tabletop`.

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
