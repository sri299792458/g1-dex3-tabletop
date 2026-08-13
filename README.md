# G1 Dex3 Tabletop

Focused Unitree G1 software for two connected operations:

1. automatically collect dorsal-Dex3 calibration observations and solve the
   fixed-marker camera extrinsic with Mike Ferguson's Ceres optimizer;
2. use a removable calibration bundle and official NVLabs CuRobo to plan and
   execute a right-Dex3 45 mm AprilCube pick, 100 mm lift, exact replacement,
   retreat, and controller handback.

The repository contains no historical manual-teaching pipeline or custom IK.
CuRobo is the only IK, collision, grasp-goal, attachment, and trajectory
planner. `running_notes.md` records the physical commissioning evidence,
rejected approaches, and remaining physical limits.

## Safety boundary

- Inspection, candidate generation, Ferguson solving, and CuRobo planning do
  not create robot command publishers.
- CUDA planning runs in a separate Python 3.11 process. ROS/control runs in
  Python 3.10 and keeps publishing through the commissioned fixed-rate Unitree
  controller while planning is in progress.
- Hardware commands require the exact harness/workspace acknowledgement and a
  second interactive SPACE after a read-only live preflight.
- Standing calibration uses `rt/arm_sdk`, full gravity feedforward, measured
  opposite-arm hold, and the independent PC2 Damp watchdog.
- Seated tabletop execution uses complete 29-joint `rt/lowcmd` ownership and
  restores Unitree control through the commissioned FSM `0 -> 1 -> 3` path.
- Failure after command publication stops the local command path and delegates
  terminal takeover to the independent PC2 watchdog. Planning failure before
  publication cannot command the robot.

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
./tools/generate_tabletop_cube.sh
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

## Tabletop cube task

Physical starting state: G1 seated in FSM 3, both arms supported and stationary
on the table, the 45 mm AprilCube resting in the shortlist's canonical
orientation (tag 132 / object +Z face upward) and visible, the complete
right-arm sweep clear, and the RealSense node running. Tabletop yaw is free;
placing any other cube face downward is rejected during read-only preflight.

```bash
cd /home/kanth042/g1-dex3-tabletop
./tools/g1_tabletop_hardware.sh run-tabletop \
  --network-interface enp134s0 \
  --calibration-bundle "$RUN/solve/calibration_bundle.json" \
  --confirm 'I CONFIRM THE G1 IS SECURED BY THE LOAD-BEARING HARNESS AND THE WORKSPACE IS CLEAR'
```

Before SPACE, this verifies the seated stationary state, both Dex3 states,
camera profile, and cube observation without creating publishers. After SPACE,
it acquires exact measured 29-joint lowcmd control, applies dual-Dex3 gravity
feedforward, observes the cube again in the loaded state, and asks isolated
CuRobo workers for:

1. a straight supported-hand escape along the observed support-plane normal;
2. complete-path selection from the 15 committed GraspGen-X/Isaac-qualified
   Dex3 grasps, with every rejected candidate and failure stage recorded;
3. approach, grasp, a collision-aware 27-sphere conservative payload lift,
   exact reverse replacement, release, retreat, clearance return, and exact
   reverse supported return.

Only the moving right wrist, articulated hand, and payload are checked against
the locally observed support plane because one cube cannot reveal the table's
finite edges. Full G1 self-collision—including the fixed left arm, torso, legs,
both hands, and marker plates—remains enabled. The left arm receives no changing
target and is held at the measured takeover state with gravity feedforward.

The task is a visual-localization and motion-execution test, not an independent
ground-truth calibration measurement. The selected calibration bundle itself
records its holdout residuals and validation status. The CUDA/UVM issue was
cleared by reboot and the complete planning path has been exercised offline on
the laptop GPU; physical execution still requires a deliberately slow first
commissioning run.

## Verification

CPU-safe checks (no robot commands):

```bash
cd /home/kanth042/g1-dex3-tabletop
.venv/bin/pytest -q
.venv/bin/ruff check src tests
.venv/bin/ruff format --check src tests
./tools/g1_tabletop.sh inspect
```

CuRobo tests require the planner environment and working CUDA. The worker
contract is file/hash based: control code never imports CuRobo and the planner
process never imports or constructs Unitree transports.
