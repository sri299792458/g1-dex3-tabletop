# Running Notes

## 2026-08-13 — Focused CuRobo calibration and tabletop repository started

### Scope

- This repository contains the production Dex3 dorsal-marker calibration
  workflow and the tabletop cube pick/lift/replace application that consumes
  its removable calibration bundle.
- Official NVLabs CuRobo is the sole backend for calibration candidate IK,
  robot/self/world collision checking, route generation, time-parameterized
  trajectories, grasp selection, constrained approach/lift, and attached-object
  collision geometry.
- The application retains only functionality CuRobo does not provide: camera
  observation, calibration-information target selection, synchronized evidence
  capture, the pinned Mike Ferguson Ceres calibration backend, Unitree ownership
  and transport safety, gravity feedforward, Dex3 actuation, and operator-visible
  task sequencing.
- Historical manual teaching, dummy-hand/AprilCube paths, custom SciPy IK,
  selected-pair FCL routing, raw experiment sessions, and discarded hypotheses
  remain in `robot-calibration-aprilcube-prototype`; they are not production
  dependencies of this repository.

### Reuse and provenance policy

- Commissioned hardware, capture, and calibration modules are imported exactly
  from `robot-calibration-aprilcube-prototype` commit
  `97a78a5c9c48701400820922f0966cc3d3a9b7bc` before any focused adaptation.
  The source commit is recorded here and in machine-readable provenance.
- CuRobo is an official submodule pinned to NVLabs commit
  `8e734f3ced1df898990bcd92de40abce475907db`. The project calls its native APIs;
  it does not maintain an independent IK, collision, or trajectory-planning
  facade.
- Mike Ferguson's `robot_calibration` is an official submodule pinned to commit
  `db991b040d1dc28af09d8865fc72f09720e12b73`.
- Unitree's official `unitree_ros` is pinned to commit
  `f3772ce54c56ef2d34c6aee8100bc768896c7d19`. Its dual-Dex3 URDF has SHA-256
  `97da67732d067c3147fc5fb7b7bafc8982718f4e7f8c92ff82266a4d9c07200d`,
  exactly matching the physically commissioned gravity-feedforward model.
- Planner and ROS/control runtimes remain separate processes. CUDA/Torch/Warp
  planning work cannot stall the commissioned fixed-rate Unitree controller.
- Planning and inspection are read-only. A separate explicit execution command
  is required before any Unitree command publisher can be created.

### Production calibration flow

1. Sample informative dorsal-marker poses in the calibrated camera frustum.
2. Solve and collision-filter them with CuRobo batched IK using the complete G1
   and both-Dex3 collision model.
3. Select the requested image/depth/orientation coverage set.
4. Use CuRobo to plan every directed transition, including the final return to
   the measured handoff, before creating an arm command publisher.
5. Follow CuRobo's frozen interpolated trajectories inside the proven Unitree
   watchdog/gravity-control process and capture stationary synchronized bursts.
6. Solve with the pinned Ferguson/Ceres backend and emit a removable calibration
   bundle.

### CuRobo integration evidence collected on 2026-08-13

- NVIDIA's G1 model is a planning adaptation of Unitree's model, not an
  independently identified physical model. For every common link, inertial
  mass, center, inertia tensor, collision mesh, and collision transform match
  Unitree's pinned URDF; all common actuated-joint origins and axes match as
  well. NVIDIA adds a fixed `base_link`, changes ten planning limits (waist and
  ankle effort/velocity plus several Dex3 thumb ranges/velocities), changes the
  fixed Mid-360 LiDAR mount transform, and inlines visual materials. A
  Pinocchio RNEA comparison at a nonzero 29-joint configuration with both hands
  locked in the commissioned middle-close posture produced exactly `0.0 Nm`
  maximum arm-torque difference. Therefore CuRobo uses NVIDIA's URDF and sphere
  configuration for planning, while physical gravity feedforward uses the
  exact commissioned Unitree dual-Dex3 URDF hash recorded above.
- NVIDIA's pinned `unitree_g1.yml` was reduced by locking every joint except the
  selected seven-joint arm. The resulting solver has 7 active and 42 locked
  joints while retaining the full robot and both articulated Dex3 hands.
- Both physical dorsal carriers are added to the native collision-sphere model:
  30 conservative spheres per palm, generated from each committed CAD
  manifest's nominal palm-to-marker transform. The resulting complete model has
  734 spheres. This is neither an omitted plate nor the discarded monolithic
  hand box.
- An all-at-once 1,544-candidate, 16/32-seed solve is not valid merely because
  the laptop has 24 GB VRAM: CuRobo had already allocated about 20 GB and needed
  a further 13.5 GB gradient-sphere buffer. Production uses fixed 128-candidate
  GPU batches. A measured 128-candidate batch used about 4.2 GB and took about
  two seconds without weakening any IK or collision criterion.
- CuRobo's current goal-buffer implementation did not accept a smaller final
  batch after a CUDA-graph-sized first batch. The worker therefore pads only the
  final batch with duplicate goals and discards the padded results. Candidate
  identity, success, and selection remain unchanged.
- CuRobo's interpolated `JointState` contains the complete 49-coordinate model
  even when the planner input has seven active joints. The worker explicitly
  reorders/extracts the named active arm before creating the immutable Unitree
  command trajectory; it never slices by an assumed position.
- Collision-free IK does not imply a configuration is connected to the live
  handoff. Every selected pose must also have a native CuRobo trajectory from
  handoff. Unconnected candidates are rejected offline and replaced by the next
  information-ranked candidate. Direct pose-to-pose edges are planned when
  possible; if one is unavailable, the final artifact freezes the two already
  validated paths through handoff. There are no runtime backup candidates or
  unplanned transitions.
- A two-target, three-edge dry run completed end to end on the laptop GPU:
  batched collision-aware IK, fixed-marker six-parameter information ranking,
  handoff-connectivity filtering, native trajectory planning, 0.2 rad/s time
  scaling, and a planned return to handoff. No robot interface was opened.

### Production manipulation flow

1. Observe the upright cube before approval, then observe it again after loaded
   29-joint seated control has settled. The latter observation and the active
   calibration bundle locate the fixed cube in the current G1 base frame.
2. Use CuRobo's native grasp-goal, collision, trajectory, and attachment APIs
   to plan the complete motion from the loaded state before changing any target.
3. Execute the supported escape, open, qualified approach, close, attached-cube
   lift, exact reverse replacement, open, retreat, exact supported return, and
   commissioned seated-controller restoration.
4. This first production path does not claim a same-image hand-marker visual
   correction at pregrasp. The dorsal marker is used to produce the removable
   camera/FK bundle; the tabletop run measures task success rather than pure
   extrinsic error. Any later closed-loop correction must be an explicit,
   separately observed and replanned contract rather than an implied feature.

### Hardware facts carried forward

- On this G1, Ready is locomotion FSM 4, seated is FSM 3, Damp is FSM 1, and
  selecting AI motion service first reaches zero-torque FSM 0. Seated debug
  lowcmd restoration must follow the physically verified `0 -> 1 -> 3` chain.
- A zero-motion seated lowcmd takeover succeeded only when the PC2 watchdog kept
  an independent heartbeat while releasing the motion service. The first lowcmd
  packet must target the exact measured complete 29-joint state.
- The laptop controller publishes complete joint commands. The nonmoving arm and
  body joints remain held; they are not left implicit.
- Gravity feedforward follows Unitree XR teleoperation's Pinocchio
  `rnea(q, 0, 0)` implementation. It reduced measured arm takeover drift to
  0.008317 rad in the commissioned seated probe. It is an execution feature,
  not a substitute for CuRobo dynamics/collision planning.
- The independent PC2 watchdog heartbeat timeout is 0.5 s. A single 50 ms laptop
  scheduling-gap fault was removed after producing false emergency stops; state
  freshness, transport failures, tracking evidence, and the independent PC2
  watchdog remain authoritative.
- Dex3 and arm DDS control are separate. The physically commissioned NVIDIA
  middle-close posture can settle with residual motor error/velocity; arbitrary
  exact-target gates must not be reintroduced. Shoulder clearance is planned
  from the fresh measured Ready state, beginning at the commissioned 0.08 rad
  outward offset and increasing only when collision validation requires it.
- The existing selected-free bundle is an analysis candidate, not independent
  physical tabletop validation. The focused fixed-marker workflow reports
  deterministic holdout and bootstrap evidence in each new bundle. The current
  tabletop task uses that bundle for visual localization but does not claim
  same-image hand-marker cancellation or calibration ground truth.

## 2026-08-04 — Continuous GUIDE/HOLD teaching replaced per-pose acquisition

### Decisions

- Removed the per-pose Damp/manual/acquire/release teaching pipeline. The G1
  enters `teach-poses` already in Regular/Ready, and the program requires exact
  startup locomotion FSM ID 4 in addition to `mode_machine=5`. The live
  LocoClient check on this G1 identified Ready as FSM 4; the earlier assumed
  value 500 was rejected before an arm command transport was created.
- A dedicated teaching controller acquires `rt/arm_sdk` once from the measured
  arms-down state. It has no target-pose or autonomous-motion API. GUIDE follows
  the latest measured calibration-arm position at 250 Hz; HOLD freezes that
  measured position while keeping blend weight one. The opposite arm remains at
  its activation position for the entire run.
- GUIDE is lower-gain hand-guiding, not validated gravity compensation. The
  operator-facing UI exposes only `SPACE` for its currently displayed action
  and `Q` for verified Damp. Per pose, `SPACE` records the supported burst,
  confirms hands-off capture only after a green `HOLD ACTIVE — REMOVE YOUR HAND
  NOW` banner, and resumes motion only after a `POSE SAVED — SUPPORT ARM`
  banner. Internal GUIDE/HOLD states are not operator terminology.
- GUIDE now scales both Kp and Kd on only the calibration arm to 50% of the
  pinned Unitree HOLD gains. This 0.5 scale is an operator-selected initial
  commissioning heuristic, not an upstream freedrive recommendation. The
  frozen measured target remains fixed while gains ramp linearly to HOLD over
  one second; after support is confirmed with `SPACE`, they ramp back down before
  measured-state following resumes. The opposite arm stays at full HOLD gains.
- `Q` ends the physical session through verified PC2 whole-body Damp and leaves
  all completed captures immediately persisted and resumable. Manual return to
  an invisible startup pose, terminal weight blending, and UI-level dataset
  finalization were removed from teaching.

### Implemented

- Added isolated teaching controller/driver modules with acquisition, GUIDE,
  HOLD, CAPTURE, release, fault, actual-URDF-limit enforcement, frozen-arm drift,
  opposite-arm drift and state-freshness checks. A 50 ms single-tick deadline
  was removed after a live 78 ms laptop scheduling pause falsely triggered
  Damp during collision validation; a true controller stall remains covered by
  the independent 500 ms PC2 heartbeat watchdog. The initial
  0.03 rad GUIDE cutoff was removed after live commissioning showed that it
  trapped an operator-driven joint still inside its real URDF limit; the margin
  is now display-only.
- `teach-poses` now takes equal seven-frame supported and held bursts, stores
  both losslessly, selects a medoid from each, and records pixel, measured-joint,
  and `tau_est` differences. Camera staleness during GUIDE creates a protective
  HOLD rather than leaving the arm in measured-state following without a view.
- Removed per-pose FCL evaluation after manual freeze. It ran only after the
  operator had already placed the arm, added no protection against commanded
  motion, complicated the UI, and contributed to control-thread scheduling
  jitter. Collision/path validation remains isolated to autonomous replay.
- Session and dataset schemas are version 3 with no compatibility path. Finalize
  writes a held production dataset and a paired supported diagnostic dataset;
  offline rebuilding selects the phase explicitly.
- The PC2 watchdog exposes its initial FSM ID and disarms without publishing arm
  commands when it does not match the configured Regular FSM.

### Verification

- Pure fake-transport tests cover every teaching transition, zero-displacement
  freeze/resume, verified external damping, display-only limit warning,
  hard-limit and drift faults, thread ownership, paired storage, phase-specific
  dataset rebuilding, and Regular-FSM refusal.
- Hardware was not contacted by the rewrite. Physical commissioning still must
  start with the harness, clear arm/table sweep, naturally down arms, and a
  deliberately slow supported/held capture cycle.

## 2026-08-03 — Left-palm AprilCube configuration completed

### Decisions

- The existing 40 mm AprilCube is attached directly to the rigid left rubber-
  hand palm with the purchased double-sided mounting tape, approximately as
  shown in the reference photograph. There is no printed carrier, skirt,
  clamp, or fastener assembly and no physical-measurement gate.
- The cube-to-hand transform remains a free calibration variable. Its nominal
  placement is used only for initialization, visualization, and conservative
  collision checking; the visible tag-face orientation need not be exact.
- The collision model uses a 60 x 58 x 60 mm axis-aligned box centered at
  `[40, -34, 0]` mm in `left_rubber_hand`. This contains the 40 mm cube and
  numerical allowance for the tape footprint. The tape is not separately
  rendered.

### Implemented

- Made the calibration carrier an explicit `calibration_arm` setting throughout
  pose recording, schema validation, readiness gating, replay, collision
  validation, dataset construction, solving, residual reporting, synthetic
  checks, and both CLIs. The selected configuration moves left joints 15--21
  while continuously monitoring and holding right joints 22--28.
- Added continuous velocity and position-drift interlocks for both the moving
  calibration arm and the opposite held arm. Collision preflight now requires
  the AprilCube attachment to be present in an enabled collision pair before a
  command publisher can be constructed.
- Freeze `collision_pairs.yaml` alongside each immutable raw session and bind it
  to the transition-validation report by content hash.
- Use the rectified projection matrix `CameraInfo.P[:3, :3]` for detection,
  PnP, optimization, anchor checks, and synthetic generation. The RealSense
  intrinsics are still used; this selects the intrinsics belonging to the
  rectified color stream instead of the raw/distorted matrix.
- Keep repeated captures from leaking between train and holdout sets by
  splitting and bootstrapping whole pose groups. Added a linear warm start for
  the robust solve to avoid poor local minima.
- Added a reproducible left-palm geometry render and JSON record using the
  official Unitree hand STL and released AprilCube 3MF. Only the hand, cube,
  and conservative collision box are drawn.

### Verification

- Root prototype test suite: 102 passed.
- Ruff lint and format checks pass across source, tests, and the render tool;
  source and wheel builds also pass.
- Artifact inspection reports no blockers and binds the ready collision model
  to SHA-256 `bb8a9352b00dfa5f155f29a819887f4ebde8a3edc75eaa897ea63a2efb17c97f`.
- The real FCL model reports no collision at the configured zero pose; its
  minimum checked clearance is about 36.5 mm.
- A noisy 20-pose left-arm synthetic run recovered the 12-parameter solution
  at full Jacobian rank, with 0.279 px holdout radial RMS, 0.110 mm camera
  translation error, and 0.0123 degree camera rotation error.

## 2026-08-02 — Operator teaching, commissioning, and collection completed

### Added

- Added the read-only `teach-poses` application. It combines subscriber-only
  Unitree `LowState` with rectified ROS image/`CameraInfo`, displays the same
  red/yellow/green detector overlay, freezes a selected image, waits for the
  required post-image state bracket, and atomically records the measured pose.
  It creates no Unitree command publisher.
- Pose recording now verifies that the left arm is stationary and remains near
  the pose-set's frozen left hold, in addition to all existing right-arm gates.
- Added explicit weight-zero, measured-pose hold, and single-pose round-trip
  commissioning commands. The increasing-risk stages require exact typed
  acknowledgements before the publisher is constructed and share an exclusive
  command-owner lock.
- Added the production `collect-session` command: strict configuration/hash
  preflight, exact-home acquisition, explicit passed edges, optional per-move
  terminal confirmation, rectified stationary bursts, return home, terminal
  weight zero, and immutable session finalization.
- Added a synchronized 250 Hz executor driver. Detection, preview rendering,
  ROS spinning, PNG encoding, and manifest fsync cannot starve the command tick.
- Session plans may intentionally revisit an anchor pose. Added a post-solve,
  joint-compensated repeated-anchor report that reconstructs a hand-to-target
  transform for each anchor capture and fails configurable pairwise translation
  or rotation stability thresholds.
- Added example directed-edge and repeated-anchor session-plan YAML files and a
  complete command-by-command hardware runbook in the README.

### Verification

- Root prototype test suite: 94 passed.
- Ruff lint and format checks pass across 64 files.
- Source and wheel builds pass. The wheel contains all offline, ROS boundary,
  Unitree transport, operator, solver, report, and bundled JSON Schema modules.
- Hardware was deliberately not contacted. `config/collision_pairs.yaml` still
  blocks motion approval until the installed cube envelope pose is measured.

### Pinned SDK installation check

- A direct editable install correctly exposed that Unitree's pinned
  `cyclonedds==0.10.2` build cannot discover ROS Jazzy's CycloneDDS prefix by
  itself. Added a local prefix shim using the installed ROS headers, binaries,
  and libraries, matching the mechanism in the pinned G1Pilot setup.
- Added an idempotent hardware dependency installer and a hardware command
  wrapper that exports the ROS/CycloneDDS runtime paths before Python starts.
- Verified the real pinned SDK creates the expected 35-slot HG `LowCmd`, leaves
  blend slot 29 at zero by default, loads the correct HG LowState/LowCmd types,
  and computes a CRC successfully. No DDS domain/channel was initialized during
  this message/CRC check.

## 2026-08-02 — Live burst construction and offline workflow CLI implemented

### Added

- Made the complete state buffer thread-safe for concurrent DDS callback writes
  and operator-loop reads. The Unitree observer can now forward every valid
  receipt-stamped state into that buffer; malformed samples are never forwarded.
- Added a live burst source that consumes only new rectified frames, re-runs the
  stateless detector, applies red/yellow/green visual gates, reconstructs a
  centered stationary state window, verifies image/state pairing, and emits the
  exact raw `CaptureFrameInput` consumed by the immutable session store.
- Added CLI commands for offline artifact inspection, subscriber-only hardware
  inspection, pose-set initialization/summary/undo, explicit directed transition
  validation, raw-session verification, dataset construction, real dataset
  solving, and known-truth synthetic recovery.
- The solve command initializes the camera from the official URDF optical frame
  and the target from one rectified PnP frame, then still fits all twelve
  parameters to all training corners and exports held-out residual diagnostics.

### Verification

- Root prototype test suite: 88 passed after this slice.
- A CLI synthetic run with 20 poses and 0.2 px injected noise recovered both
  transforms with rank 12, about 0.30 px holdout radial RMS, 0.10 mm camera
  translation error, and 0.011 degree camera rotation error, and wrote all five
  expected report artifacts.
- `inspect-artifacts` correctly returns non-zero and names the intentionally
  unmeasured AprilCube collision-envelope pose as the hardware blocker.

## 2026-08-02 — Full fake capture/replay lifecycle implemented

### Added

- Acquisition can now be bound to a named, validated home pose. Before the first
  publish it verifies measured left-arm hold and right-arm home errors against
  the configured settled tolerance. A known-home acquisition becomes capture-
  ready after its blend ramp; an unbound commissioning acquisition retains the
  previous anonymous hold behavior.
- Added a capture interlock adapter around the immutable session store. Raw
  writes occur only while the executor is in `CAPTURING`, and every success,
  rejection, or write failure returns the executor to a non-capture hold state.
- Added a one-shot scheduled frame source and a complete approved-session
  orchestrator. It acquires at home, captures the requested pose sequence,
  requires confirmation and a hash-bound passed edge for every move, returns
  home, sends the terminal blend-weight-zero release, then finalizes the store.
- Any orchestration error initiates the executor's emergency release path. A
  failed run is not finalized and therefore remains inspectable/recoverable.

### Verification

- Root prototype test suite: 84 passed.
- The fake end-to-end run captures home and two arm poses, returns home, and
  closes on a non-emergency zero-weight command. Tests also prove that a measured
  home mismatch publishes nothing and that refused move confirmation triggers a
  terminal emergency zero-weight command without finalizing the session.

## 2026-08-02 — Rectified ROS camera boundary implemented

### Added

- Added strict ROS `Image` conversion for `bgr8`, `rgb8`, and `mono8`, including
  row-stride handling, truncation checks, and owned immutable BGR arrays.
- Added ROS `CameraInfo` conversion into the frozen camera profile. Non-zero
  distortion is rejected, so an accidentally selected raw RealSense topic
  cannot silently enter a solver that assumes rectified pixels.
- Preserved both ROS header time and local monotonic receipt time. The latter is
  the clock used to pair images with the independently received Unitree state.
- Added a bounded, thread-safe rectified frame buffer and optional subscriptions
  owned by a caller-provided ROS 2 node. ROS packages are imported only when the
  subscriber is constructed.
- Added a complete named `JointState` adapter for integration tests or bridges;
  it refuses missing measured velocities. Native `rt/lowstate` remains the
  production source because it also provides `mode_machine` directly.

### Verification

- Root prototype test suite: 81 passed.
- Tests cover zero-distortion enforcement, ROS time conversion, padded rows,
  RGB/BGR and mono conversion, immutable bounded buffering, monotonic receipts,
  authoritative named-joint ordering, and missing velocity rejection.

## 2026-08-02 — Hardware integration references pinned

- Cloned the current official `unitreerobotics/xr_teleoperate` reference at
  revision `64ed45b4177e6297936940866df623b72621643a`.
- Cloned the exact `lnotspotl/unitree_sdk2_python` fork and revision used by the
  pinned G1Pilot checkout:
  `7c661d27f4ae064ffd0dd633fd9d5b518ef0b508`.
- Recorded both pins in `config/upstream_pins.yaml` and added an idempotent
  bootstrap script. Both remain independent, ignored Git worktrees.
- Confirmed from the official controller that the G1 uses `rt/lowstate`,
  `rt/arm_sdk`, 35-slot HG messages, 29 physical joints, the dual-arm indices
  15–28, blend weight in slot 29, measured `mode_machine`, and a CRC on every
  command. The prototype adapter deliberately does not inherit the official
  example's eager publishing or zero-target startup behavior.

## 2026-08-02 — Raiden added

- Cloned `TRI-ML/raiden` into `raiden/`.
  - Branch: `main`
  - Revision: `2353b1040c8ffb67158fc7059a8ed61b4c4e672e`
- Materialized the repository's four Git LFS stereo-model weight files.
- Initialized the `third_party/i2rt` submodule at revision
  `7b6d5016f05ca63f9ef0185b7143e63f2c7a5708`.
- Kept `raiden/` as an independent Git worktree, consistent with the other
  upstream repositories.

## 2026-08-02 — G1 pose data boundary implemented

### Added

- Encoded the complete authoritative mode-5 joint order and explicit left-arm
  indices 15–21 and right-arm indices 22–28.
- Added immutable, receipt-stamped 29-DoF robot-state records. Partial arrays,
  non-finite values, non-UTC timestamps, and invalid sequences fail closed.
- Added versioned pose-set and pose-record models. Every stored pose includes the
  measured full state, derived right-arm state, per-joint recording spread,
  visual-quality evidence, head witness acknowledgement, and preview path.
- Added a strict bundled JSON Schema and a canonical SHA-256 over every pose-set
  document. Loading detects schema drift or manual content corruption.
- Added an append/undo audit trail and validates that replaying the audit actions
  exactly reproduces the active pose order.
- Added atomic YAML persistence using a same-directory temporary file, file and
  directory `fsync`, `os.replace`, and a previous-version `.bak` file.

### Verification

- Root prototype test suite: 25 passed.
- Ruff lint and format checks pass.
- Tests cover exact joint ordering, named-state reordering, immutability,
  malformed state, pose/full-state disagreement, duplicate IDs, append/backup,
  undo, schema rejection, audit inconsistency, and content-hash tampering.

## 2026-08-02 — Read-only pose recording core implemented

### Added

- Added a strictly monotonic, bounded complete-state buffer and a centered
  stationary-window selector with bracketing samples at both edges.
- Added fail-closed recording readiness checks for mode 5, state freshness,
  sample count, continuous duration, state gaps, measured right-arm velocity,
  and measured right-arm position spread.
- Added local receipt-time camera/state pairing. An image must be bracketed by
  state observations, satisfy a maximum bracket span, and have a sufficiently
  close nearest measured state. The original camera header timestamp is retained
  separately for provenance but is not compared to unsynchronized LowState time.
- Added a read-only `PoseRecorder`. Green visual quality is accepted, yellow
  requires a non-empty recorded override reason, and red cannot be saved. The
  head-pitch witness mark must be acknowledged.
- Successful recording stores median full/right joint positions, per-right-joint
  peak-to-peak spread, visual metrics, state-readiness metrics, timestamp-pairing
  evidence, and the preview reference through the atomic pose store.
- Added the initial manual-recording thresholds to `config/hardware.yaml`.

### Verification

- Root prototype test suite: 40 passed.
- Tests exercise wrong mode, motion, drift, stale state, state gaps,
  non-monotonic timestamps, unbracketed/distant images, red/yellow/head gates,
  median pose extraction, evidence persistence, and prevention of writes after a
  failed assessment.

## 2026-08-02 — Deterministic safety executor implemented

### Added

- Added a narrow arm-transport protocol whose only mutation is a validated
  fourteen-joint command plus blend weight. Commands require finite exact-length
  arrays, a monotonic timestamp, and a weight in `[0, 1]`.
- Added a manual clock and deterministic fake mode-5 G1 transport. It simulates
  measured tracking, state loss, mode changes, command history, and shutdown
  without any DDS or hardware dependency.
- Added the explicit observing/acquiring/holding/moving/settling/ready/capturing/
  releasing/fault/stopped state machine.
- Acquisition seeds both arm targets from fresh measured state and publishes
  weight zero before ramping. The measured left arm is then held for the entire
  executor lifetime.
- Moves can only reference a pose in the immutable pose set. Every move requires
  operator confirmation plus an exact passed directed-edge approval whose pose-
  set and validation-report hashes match the artifacts loaded by the executor.
- Added per-tick joint-velocity limiting, measured coarse arrival, continuous
  position/velocity dwell, motion timeout, capture/motion interlocks, and
  control-loop overrun detection.
- Wrong mode, stale state, explicit emergency stop, or loop overrun enters a
  logged fault and ramps the blend weight to a terminal emergency weight-zero
  command. Clean release is only possible at an approved, measured-settled home
  pose and also ends with a terminal weight-zero command.
- Added an advisory process lock to prevent two copies of this calibration
  executor from owning the laptop command path.

### Verification

- Root prototype test suite: 53 passed.
- The fake end-to-end executor test performs acquisition, a bounded move,
  measured settling, capture, return home, and clean release. Additional tests
  cover incorrect/stale approvals, missing operator confirmation, capture/move
  interlocks, wrong robot mode, frozen state, loop overrun, emergency release,
  no-publish observation exit, motion-profile bounds, and exclusive ownership.

## 2026-08-02 — Immutable raw sessions and dataset builder implemented

### Added

- Added immutable rectified-camera metadata with exact stream dimensions,
  optical frame, camera name/serial, full `K/R/P`, zero-distortion enforcement,
  and a canonical camera-profile hash.
- Added canonical current-frame AprilCube correspondence serialization and
  hashing, including decoded corners, generated 3D geometry, and explicit
  duplicate/ignored/rejected metadata.
- Added a strict version-1 session manifest schema and content hash. A session
  freezes the exact pose set, hardware YAML, target JSON, and transition
  validation report and verifies the pose-set content hash before creation.
- Added crash-safe raw capture. Lossless PNGs and complete state-window JSON are
  written with exclusive atomic creation before the manifest advances. A
  failure between those operations is visible as an orphan and is never
  silently overwritten.
- Every frame rechecks reproducible image/state bracketing, stationary measured
  state, unchanged camera profile, current-frame detector validity, and visual
  quality. Accepted bursts deterministically select the real frame nearest the
  median corner vector among the most common visible-tag signature.
- Finalization marks all session files read-only. Accepted, rejected, retry,
  skipped, and aborted outcomes remain explicit in the manifest.
- Added a deterministic offline dataset builder. It verifies every raw frame,
  including non-selected burst frames; re-runs the stateless detector; compares
  correspondence hashes; reconstructs receipt-time pairing; converts target
  millimetres to metres exactly once; and emits one content-hashed sample per
  accepted pose.
- Added a stable content-derived train/holdout split and dataset load/validation
  contract.

### Verification

- Root prototype test suite: 63 passed.
- A synthetic-camera integration test writes five accepted captures, recreates
  the session writer to simulate interruption, appends five more, finalizes,
  and rebuilds the same ten-sample dataset twice with an identical content hash.
- Additional tests cover camera/profile changes, red frames, finalize/append
  exclusion, invalid metadata with no raw writes, source/manifest tampering,
  orphan files, selected and non-selected raw corruption, metre conversion, and
  deterministic holdout membership.

## 2026-08-02 — Exact URDF FK and offline transition validation implemented

### Added

- Added a strict lightweight URDF parser for the exact official
  `g1_29dof_rev_1_0.urdf`. It resolves the tree, fixed/revolute/prismatic joint
  transforms, finite joint limits, relative meshes, primitive collision shapes,
  full-tree FK, and arbitrary ancestor-to-child chains.
- Added a numerical regression for the eight-joint
  `torso_link -> right_rubber_hand` chain, the nominal D435 transform, all seven
  right-arm limit records, and the exact official URDF SHA-256.
- Added `python-fcl` and selected-pair Trimesh/FCL clearance checking. The
  official collision meshes are used except for the rigid dummy hand, whose
  official visual mesh is an explicit fallback because that URDF link has no
  collision element.
- Added configurable attached-box collision objects so the printed AprilCube
  envelope can be placed on `right_rubber_hand` once its installed pose is
  measured.
- Added directed transition validation. Every edge checks both endpoints and
  interpolation samples spaced by maximum joint increment, all 29 joint limits
  with margin, selected arm/body/leg collision distances, path length, and
  estimated duration.
- Added a content-hashed validation report tied to the pose-set, official URDF,
  collision configuration, reference full-body state, validator version, and
  every directed edge. It produces the exact `TransitionApproval` consumed by
  the executor; stale or failed hashes remain unusable.
- Session creation now parses the frozen validation report and refuses a failed
  report or one belonging to another pose set/URDF.

### Deliberate pre-hardware block

`config/collision_pairs.yaml` is intentionally `hardware_ready: false`. The
installed cube-to-hand pose cannot be obtained from software or the photograph.
After the cube is taped in its final position, measure a conservative cube
envelope pose on `right_rubber_hand`, add its body collision pairs, inspect the
result, and only then set the flag true. Until that is done, every production
path report fails and cannot authorize the executor.

### Verification

- Root prototype test suite: 69 passed.
- Real FCL reports over 30 mm minimum selected-pair clearance at the nominal G1
  zero-arm configuration and reports negative signed distance for a deliberately
  overlapping box/sphere URDF.
- Tests cover FK/limits, interpolation count, passed approval generation,
  report round-trip hashing, joint-limit margin rejection, stale URDF rejection,
  and the unmeasured-AprilCube hardware block.

## 2026-08-02 — Twelve-parameter calibration and residual reporting implemented

### Added

- Added one explicit transform convention: parent-from-child homogeneous
  matrices and six-vectors ordered as translation XYZ in metres followed by a
  rotation vector in radians.
- Added the extrinsics-only nonlinear solver for the intended twelve unknowns:
  `torso_T_color_camera` and `right_rubber_hand_T_aprilcube`. Each residual uses
  the sample's complete measured mode-5 state, exact URDF FK, rectified
  `CameraInfo`, target points in metres, and raw ordered image corners.
- Added configurable robust SciPy least squares, positive-depth penalties,
  optimization status, numerical Jacobian singular values/rank/condition,
  approximate parameter standard deviations, and fail-closed degeneracy
  rejection.
- Added nominal `d435_link -> color optical` initialization and a single-frame
  rectified PnP initializer for the unknown hand-to-target transform. PnP is only
  an initial value; the batch solve still minimizes every raw corner.
- Added stable content-derived training/holdout orchestration and pose-level
  bootstrap resampling. Dataset/solver URDF hashes must match.
- Added per-corner signed U/V and radial residuals, depth, capture/pose/frame,
  tag/corner identity, per-capture RMS, per-tag RMS, and correlations between
  pose-mean residual structure and all seven measured right-arm joints.
- Added versioned, schema-validated, content-hashed result export containing
  `result.json`, `calibrated_extrinsics.yaml`, `corner_residuals.csv`,
  `provenance.json`, and a concise Markdown report.
- The report explicitly requires examining train/holdout residual structure and
  joint correlations before proposing any joint offset. Correlation is a reason
  to investigate physical/model error, not permission to add parameters.

### Verification

- Root prototype test suite: 73 passed.
- A noiseless 40-pose synthetic G1 dataset recovers both six-DoF transforms to
  numerical precision from perturbed initial values with Jacobian rank 12.
- Repeating one pose is rejected at rank 6/12.
- A noisy 30-pose synthetic run performs deterministic holdout evaluation and
  three successful bootstrap solves, recovers each translation within 2 mm and
  each rotation within 0.3 degrees, stays below 0.8 px holdout radial RMS, emits
  all run artifacts, and detects manual result tampering.

## 2026-08-01 — Project setup

### Goal

Create a lightweight prototype workspace for quickly experimenting with robot
calibration and AprilCube-based perception.

### Added

- Cloned `mikeferguson/robot_calibration` into `robot_calibration/`.
  - Branch: `ros2`
  - Revision: `db991b040d1dc28af09d8865fc72f09720e12b73`
- Cloned `sri299792458/aprilcube` into `aprilcube/`.
  - Branch: `main`
  - Revision: `fc18d50c8bbaadc9646dfd0aa5fcd2404a9868c5`
- Downloaded arXiv paper `2602.16705` into `papers/2602.16705.pdf`.
  - Title: *HERO: Learning Humanoid End-Effector Control for Visual Whole-Body
    Open-Vocabulary Object Grasping*
  - SHA-256: `4cf2da3f6f1697edc747465beff918082f7ac6a6ae4ef1089a0767489f5b8d99`

### Workspace conventions

- Keep the upstream repositories as independent Git worktrees so experiments can
  be committed to the appropriate codebase without mixing histories.
- Track workspace-level research material and decisions in this parent repository.
- Favor small, reversible experiments and document commands, observations, and
  decisions here as the prototype evolves.

### Current state

- Initial assets are present and verified.
- No source changes have been made to either upstream repository.

## 2026-08-01 — Camera calibration problem formulation

### Added

- Cloned the official `unitreerobotics/unitree_ros` repository into
  `unitree_ros/`.
  - Branch: `master`
  - Revision: `f3772ce54c56ef2d34c6aee8100bc768896c7d19`
- Reviewed HERO sections III-B, V-C, and appendix B.1-B.2, focusing on G1
  analytical-FK error, base odometry, and MOCAP-assisted camera calibration.
- Inspected the official G1 URDF variants and nominal D435/D455 camera chains.
- Inspected `robot_calibration`'s 2D camera/3D chain reprojection optimizer and
  AprilCube's raw tag-corner geometry/detections.
- Wrote `docs/problem_formulation.md` with the measurement equations,
  observability constraints, staged AprilTag approach, and validation criteria.

### Key decision

Use `torso_link` as the provisional calibration reference because it is rigidly
connected to the standard G1 camera mount. Treat the URDF camera transform only
as an optimizer initial value. Confirm this after identifying the physical G1
variant and camera/head hardware.

### Main risk

A hand-mounted AprilCube plus nominal FK can produce a low reprojection error by
absorbing G1 kinematic error into the camera extrinsic. The prototype will jointly
estimate a limited set of parameters and use an independently registered target
fixture for validation when possible.

### Information needed from the physical robot

- Hand configuration.
- RealSense model and serial number.
- ROS 2 image, `CameraInfo`, and joint-state topic names.

## 2026-08-01 — Physical G1 variant confirmed

### Confirmed

- `mode_machine = 5`, selecting the 29-DOF revision-1.0 G1 family.
- Camera pitch is manually adjustable and unsensed.
- Base URDF: `unitree_ros/robots/g1_description/g1_29dof_rev_1_0.urdf`.

### Calibration consequence

Treat the camera as a fixed six-DoF child of `torso_link` only after physically
locking it at the intended operating angle. Any manual pitch change invalidates
the complete extrinsic—not only its pitch component—and requires a fresh
calibration.

### Hardware decision

Physically fix the head pitch at one operating angle and apply a witness mark.
Check the mark before each calibration/capture run. If it shifts or the mount is
loosened, discard the saved extrinsic and recalibrate.

## 2026-08-01 — Calibration target carrier

### Confirmed

- Both Dex3 and dummy rubber hands are available.
- Use the rubber-hand kinematic configuration for the first calibration fixture.
- The physical dummy hand is rigid, despite the URDF link name
  `right_rubber_hand`; the earlier concern about rubber deformation does not
  apply.
- The mode-5 URDF attaches `right_rubber_hand` to `right_wrist_yaw_link` with a
  fixed `right_hand_palm_joint` at `[0.0415, -0.003, 0]` meters.

### Mount recommendation

Prefer a keyed two-piece clamshell registered directly against the rigid dummy
hand. Use hard locating pads, an end stop, an asymmetric anti-rotation feature,
and two screw clamps without compliant liners. A replacement wrist flange is the
fallback if clamshell removal/reinstallation is insufficiently repeatable. Do not
use the Dex3 grasp, tape, or hook-and-loop material as the metrology interface.

### Next dependency

Obtain photographs and caliper measurements of two candidate clamp sections on
the physical dummy hand. Use the official STL for the nominal cradle surface and
print a small fit coupon before the complete mount. See
`docs/target_mount_design.md`.

## 2026-08-01 — Fixture fabrication constraint

### Confirmed

- Custom parts can only be made by FDM 3D printing; machining is unavailable.
- Standard metric screws, washers, and nuts can be purchased.

### Design response

Use three printed custom parts: a primary conformal cradle with integrated target
stalk, a clamp cap, and the AprilCube. Join them with the M4/M3 fastener set below,
captured by printed counterbores and nut traps. Validate hand-fit clearance with a
short printed cradle slice before printing the full fixture.

### Fastener selection

Revised the preliminary two-bolt idea to a fully specified six-bolt assembly:

- four M4 x 20 mm ISO 4762 socket-head screws, four M4 DIN 985 nyloc nuts, and
  four M4 flat washers for two clamping flanges per side; and
- two M3 x 16 mm ISO 4762 socket-head screws, two M3 DIN 985 nyloc nuts, and two
  M3 flat washers for the keyed AprilCube-to-stalk connection.

All nuts sit in printed captive pockets; no heat-set inserts or machining are
required. Print a tolerance coupon before the fixture.

## 2026-08-01 — Mount concept render

### Added

- Created a project-local `uv` environment with locked dependencies for Trimesh,
  Manifold, NumPy, SciPy, Matplotlib, and Pillow.
- Added `tools/render_mount_concept.py`.
- Rendered assembled and exploded views to
  `renders/dummy_hand_aprilcube_mount_concept.png` using the official Unitree
  dummy-hand STL at URDF scale.

### Interpretation

The render communicates the intended assembly: an orange printed cradle and
integrated short target stalk, a blue printed clamp cap, four M4 side-flange
fasteners, and a two-M3 keyed AprilCube attachment. The current oval cradle and
target offset are provisional and must not be sliced as production CAD.

## 2026-08-01 — Physical dummy-hand fit check

### Observation

The provided physical photo shows that the dummy palm is rigid, broad, and flat,
and that the existing AprilCube has a stable full-face seating area near the
wrist. This removes the need for a printed fixture in the first experiment.

### Revised MVP

Mount the cube directly to the palm using thin, high-tack double-sided film tape.
Avoid Velcro and foam tape because their compliant layers permit pose changes
under gravity and wrist rotation. The covered bottom tag is not needed.

The hand-to-cube transform is optimized as a free parameter, so its exact value
and reinstall repeatability are unnecessary. Rigidity during one dataset is the
requirement. Run a wrist-orientation return test against stationary detector noise
before capture. Retain the printed clamshell only as a fallback if tape moves.

## 2026-08-01 — Walmart tape verification

The inexpensive Scotch permanent office tape is advertised for paper and crafts
and has no relevant mounting-load specification, so it is not an acceptable
robot target-mount recommendation. The approximately $3.56 Scotch Indoor
Mounting Tape 110H is load-rated and easily supports a 40 g cube, but 3M lists
its carrier as flexible polyethylene foam. It is therefore suitable for fall
prevention but not preferred as the calibration interface because its elastic
pose change is unspecified.

The cheapest Walmart candidate found with a thin, non-foam carrier and
manufacturer technical data is T-Rex Double-Sided Super Glue Tape, 0.5 in x
7.5 yd, approximately $5.43 at the time of checking. Its technical data lists a
0.19 mm polyester-film carrier, acrylic adhesive, plastic as an intended surface,
and representative peel adhesion to steel of 180 oz/in of width. Use three
adjacent 40 mm strips to cover 38.1 x 40 mm of the cube base. Store price and
availability are location-dependent.

A 40 g target produces 0.392 N at rest and 1.96 N under a conservative 5 g total
load. Across the 1,524 mm^2 taped area, the corresponding average loads are only
0.257 kPa and 1.29 kPa. These figures establish a comfortable bulk-strength
margin, but do not establish adhesion to the unknown dummy-hand polymer or bound
peel under the cube's lever arm. The wrist-orientation return test remains
mandatory; passing means no pose change beyond stationary detector noise.

## 2026-08-02 — Three-repository review and implementation plan

### Reviewed revisions

- `robot_calibration` at `db991b040d1dc28af09d8865fc72f09720e12b73`.
- `aprilcube` at `fc18d50c8bbaadc9646dfd0aa5fcd2404a9868c5`.
- `raiden` at `2353b1040c8ffb67158fc7059a8ed61b4c4e672e`,
  including the initialized `i2rt` submodule.
- Used `unitree_ros` at `f3772ce54c56ef2d34c6aee8100bc768896c7d19`
  only to validate the mode-5 G1 frame and joint chain.

### Main technical conclusions

- `robot_calibration` already has the correct raw-corner batch residual:
  `Chain3dToCamera2d`. Its free-frame mechanism can estimate both the
  `d435_joint` correction and the unknown `right_rubber_hand -> AprilCube`
  transform.
- Its 2D camera model ignores distortion, uses plain squared loss, and does not
  export final per-corner residuals or observability metrics. Use rectified color
  pixels for the MVP and add residual/Jacobian reporting before considering
  kinematic offsets.
- The built-in capture path is not sufficiently synchronized for this use case;
  create a small G1-specific recorder that pairs each stationary image with the
  nearest complete `JointState` by timestamp.
- AprilCube's default printed target is a 40 mm cube with six 30 mm OpenCV ArUco
  `4x4_100` markers. Its 3D geometry is in millimetres and its detector corners
  are ordered TL/TR/BR/BL. The ROS bridge must convert geometry to metres once.
- AprilCube's tracking stack can use temporal filtering, prediction, rejected
  quad recovery, and optical flow. Calibration observations must instead come
  from a new stateless current-frame correspondence API.
- Raiden's solver is tied to YAM kinematics and a fixed ChArUco board, so it is
  not the G1 solution. Its RealSense intrinsics/bag code, clock cautions, Rerun
  pattern, and versioned JSON schema are useful references. Keep it out of the
  first ROS runtime dependency graph.
- A hand-mounted target without an independent reference yields a
  kinematics-conditioned camera extrinsic. Held-out residuals establish internal
  consistency but cannot prove that nominal G1 FK bias was not absorbed into the
  camera transform.

### Physical protocol corrections

The earlier tape recommendation and wrist-orientation test are superseded:

- use the Scotch foam mounting tape already purchased; do not buy replacement
  tape before testing the installed target;
- bulk holding strength is ample for the approximately 40 g cube;
- foam matters only if the cube-to-hand transform changes, so qualify it with
  stationary measurements and repeated anchor arm poses;
- keep wrist roll, pitch, and yaw fixed for the full dataset; and
- use slow shoulder pitch/roll/yaw and elbow motion to create pose diversity.

### Verification completed

- AprilCube generator/web tests: 14 passed.
- Default-cube synthetic detector: 48/48 viewpoints detected and passed with
  temporal filtering disabled.
- Python syntax compilation passed for AprilCube and Raiden sources.
- `robot_calibration_msgs` builds under ROS 2 Jazzy when CMake is directed to
  `/usr/bin/python3`. The main package configure is currently blocked by missing
  system dependencies: `camera_calibration_parsers`, Ceres, gflags, and protobuf
  development packages. This is an environment dependency gap, not a source
  failure.

### Plan

Added `docs/detailed_implementation_plan.md`, covering the repository audit,
data contract, software boundaries, pose design, synthetic recovery tests,
physical qualification, extrinsics-only solve, residual diagnostics,
observability gates, optional joint-offset criteria, independent validation,
deployment provenance, and risk register.

The next implementation milestone is a public stateless AprilCube
correspondence API, followed by a synthetic 12-parameter recovery test before
connecting to the physical G1.

## 2026-08-02 — G1Pilot added

- Cloned `sri299792458/g1pilot` into `g1pilot/`.
  - Branch: `dev`
  - Revision: `6b5af59b109e2ee687920fdf66ded6182725e945`
- The checkout has no configured Git submodules or Git LFS objects.
- Kept `g1pilot/` as an independent Git worktree, consistent with the other
  upstream repositories.

## 2026-08-02 — SPARK data-collection pose workflow reviewed

- Cloned `RPM-lab-UMN/spark-data-collection` into
  `spark-data-collection/`.
  - Branch: `main`
  - Revision: `be284c2f8138f383d260526f68613c7a28d364d4`
- The checkout has no configured Git submodules or Git LFS objects and is kept
  as an independent Git worktree.
- Its calibration workflow is a useful behavioral template rather than a
  reusable robot driver:
  - `record_calibration_poses.py` enables UR freedrive and records measured
    joint positions and TCP pose when the operator presses `r`;
  - `calibrate_rig.py` later replays each saved joint vector with UR `moveJ`,
    waits for stabilization, and captures camera observations; and
  - the UR implementation depends on RTDE APIs that the G1 does not provide.
- The equivalent G1 workflow is feasible by recording motor positions from
  `rt/lowstate` while the robot is in a verified manual-teaching/compliant mode,
  then replaying discrete joint vectors through one exclusive `rt/arm_sdk`
  owner. The official Unitree control layer identified below should own DDS;
  G1Pilot remains useful for offline limit/collision checks, not as the motor
  transport.
- For this calibration, save and replay the complete measured seven-joint
  right-arm vector. Replay only one operator-confirmed pose at a time,
  interpolate slowly, verify measured settling, and then trigger capture.
- Unitree's current official SDK example confirms that 29-DoF G1 arm positions
  are commanded over `rt/arm_sdk` while states arrive over `rt/lowstate`. The
  exact availability and behavior of the Explore app's manual Teaching mode is
  firmware-dependent and must be verified on the lab G1 before relying on it.

## 2026-08-02 — Full joint-space pose recording selected

- The earlier fixed-wrist collection rule is superseded. Wrist roll, pitch, and
  yaw may vary naturally between manually recorded configurations because their
  measured positions are saved and included in G1 FK for every capture.
- Replaying a recorded joint vector does not use IK. Each endpoint is reachable
  by construction, so there is no IK convergence or branch-selection failure.
- This does not make arbitrary transitions safe. A straight interpolation in
  joint space can collide even when both endpoints are safe. At minimum, sample
  and check each interpolated path against joint limits and the relevant
  arm-versus-body collision geometry, then execute it slowly under operator
  confirmation. Full MuJoCo simulation is optional rather than a prerequisite.
- Always pair an image with the measured joint state at image time rather than
  the recorded target. This captures finite tracking error and settling.
- Varying wrist orientation changes the gravity load on the taped cube. Qualify
  mechanical rigidity using repeated returns to the exact same complete
  seven-joint anchor vector; replace the mount only if measured drift exceeds
  stationary detector scatter.

## 2026-08-02 — Existing G1 trajectory implementations audited

The earlier conclusion that the G1 lacked a reusable `moveJ`-like layer was too
strong. The public ecosystem contains three distinct levels of implementation:

- Unitree's official `unitree_ros2` G1 arm example contains a private `MoveTo`
  helper. It publishes `/arm_sdk`, subscribes `/lowstate`, interpolates at
  50 Hz, applies a 0.5 rad/s per-joint clamp, and ramps out the arm blend
  weight. It is a hard-coded demonstration, not a callable service/action; it
  does not wait for arbitrary goals to settle from measured feedback.
- Unitree's official `xr_teleoperate` repository provides the reusable
  `G1_29_ArmController`. In motion mode it owns `rt/arm_sdk`, reads
  `rt/lowstate`, streams at 250 Hz, clips requested motion using measured arm
  positions, and accepts arbitrary 14-joint dual-arm targets through
  `ctrl_dual_arm`. This is the preferred transport/control base for the
  calibration prototype.
- Unitree's official `unitree_lerobot` repository already replays recorded G1
  datasets by sending each recorded arm frame through that controller at the
  dataset frequency. This proves official record-and-replay support, but its
  dense LeRobot episode loop is not the desired discrete
  move-settle-capture protocol.

No official Unitree repository currently exposes arbitrary G1 arm goals through
ROS 2 `control_msgs/action/FollowJointTrajectory`. The standard ROS 2
`joint_trajectory_controller` supplies interpolation, goal tolerances, feedback,
and a blocking action result, but it requires a G1 `ros2_control` hardware
interface that Unitree does not publish.

Community alternatives exist but are not the prototype baseline:

- MyBotShop's G1 integration documents 7-DoF left, right, and dual-arm
  `FollowJointTrajectory` actions, but it is a vendor integration rather than
  an official/open Unitree driver.
- `fiveages-sim/unitree-ros2-control` and `Adyansh04/grove-g1` bridge G1 to
  `ros2_control`; both are very new. The former contains questionable arm-index
  handling in its current `g1_arm_sdk` path. The latter is thoughtfully tested
  in simulation and uses only `/arm_sdk`, but was created in July 2026, has no
  repository license, and explicitly leaves real-hardware validation pending.
- A recent whole-body community workspace exposes the standard trajectory
  controller through raw `/lowcmd`; that would replace the onboard balance
  controller and is inappropriate for this standing calibration workflow.

### Revised implementation choice

Vendor `G1_29_ArmController` (or a minimal pinned copy of it) and add only the
calibration-specific wrapper that Unitree does not provide:

1. preserve the left-arm target at its measured value while moving the right
   arm to a saved seven-joint vector;
2. use conservative configurable velocity/timeout limits;
3. reject non-finite and out-of-URDF-limit targets and prevalidated unsafe
   transitions;
4. wait on measured right-arm position and velocity tolerances from
   `rt/lowstate`, then require a dwell interval; and
5. trigger image/joint capture only after settling.

This reproduces the useful blocking behavior of UR `moveJ` without reimplementing
Unitree DDS ownership or introducing a full ROS 2 control stack for the MVP.

## 2026-08-02 — Lightweight calibration package selected over G1Pilot runtime

Build the recorder/replayer as a small project-local package rather than adding
the G1Pilot manipulation stack to the calibration runtime.

The checked-out G1Pilot `dev` arm path is an OpenSoT Cartesian controller: its
public goals are right/left `PoseStamped` hand poses, it solves IK continuously,
and optional collision avoidance pulls in XBot, OpenSoT, FCL, and Python binding
builds. It does not expose a discrete measured joint-vector replay interface.
Using it as the executor would therefore add an unnecessary IK layer and a large
dependency surface to a joint-space record/replay problem.

Keep G1Pilot as a development-time validation tool only:

- reuse its mode-5 collision geometry and selected arm/body collision pairs;
- use lightweight kinematic path sampling for transition checks, with MuJoCo
  available only when dynamic simulation is useful;
- do not import G1Pilot modules into the hardware recorder/replayer; and
- never allow G1Pilot and the calibration executor to publish `rt/arm_sdk` at
  the same time.

The project-local package should have four narrow boundaries: an official
Unitree-controller adapter; versioned pose/session storage; a blocking
move-and-settle executor; and capture orchestration. Detection, calibration,
and collision-validation logic remain separate processes/packages. This keeps
the robot-facing code small enough to review and test while preserving a future
path to replace only the adapter with a standard ROS trajectory action.

## 2026-08-02 — Full implementation plan revised

Reworked `docs/detailed_implementation_plan.md` into the executable plan for the
lightweight package. It now defines the package layout, command-line surface,
transport protocol, motion/capture state machine, safety invariants, pose/session
and result schemas, immutable raw-versus-derived data split, transition
validation contract, hardware commissioning sequence, verification matrix,
risk register, and milestones M0 through M12.

One additional controller-safety finding changes how the official code is
reused. `xr_teleoperate` is Apache-2.0, but its `G1_29_ArmController` must not be
instantiated unchanged for calibration: motion mode starts its publisher thread
with a zero dual-arm target and weight one, and it has no explicit joined-thread
shutdown. Implement a small attributed G1-only derivative that waits for fresh
mode-5 state, seeds both arms from measured q, starts at weight zero, ramps under
explicit acquisition, checks state freshness, and provides clean/emergency
release.

Updated `config/hardware.yaml` with the initial commissioning configuration:
250 Hz commands, 0.2 rad/s joint speed, 0.02 rad position tolerance, 0.03 rad/s
velocity tolerance, 0.75 s dwell, and 0.1 s LowState timeout. These are starting
values requiring supervised hardware validation, not final performance claims.
The configuration and problem formulation now consistently record and replay
all seven measured right-arm joints rather than fixing the wrist.

## 2026-08-02 — Full versus lock-waist revision-1.0 URDF comparison

Compared the official `g1_29dof_rev_1_0.urdf` and
`g1_29dof_lock_waist_rev_1_0.urdf` directly and as parsed XML. They have the
same 40 links, 39 joints, link/joint name sets, origins, axes, limits, inertials,
meshes, collision geometry, right-arm chain, dummy-hand transform, and D435
transform. The only changes are:

- robot name;
- `waist_roll_joint`: revolute -> fixed;
- `waist_pitch_joint`: revolute -> fixed; and
- removal of the non-standard `dont_collapse="true"` attribute from the two
  fixed hand-palm joints.

Despite its name, the lock-waist file leaves `waist_yaw_joint` revolute and has
27 movable joints instead of 29. With `torso_link` as the calibration root, the
camera-to-right-hand FK is identical in both files. Retain the full 29-DoF file
for the project because it matches `LowState`, preserves measured waist
roll/pitch for pelvis/world TF and whole-body collision checks, and retains the
hand-frame preservation hint for converters that honor it.

## 2026-08-02 — AprilCube mounting-face convention

The face taped to the dummy hand does not need to have a particular ID or a
pre-measured orientation. The calibration model estimates the complete rigid
six-DoF transform `right_rubber_hand_T_aprilcube` together with
`torso_T_color_camera`, so choosing another cube face or rotating the whole
cube on the palm changes the first unknown transform rather than invalidating
the method. The attachment must remain rigid and unchanged throughout a
dataset; reattaching the cube creates a new hand-to-cube transform and requires
a new calibration run.

For `models/dex3_safe_cube/config.json`, the cube origin is at its center and the
40 mm cube faces are assigned as follows: tag 0 is `+X`, tag 1 is `-X`, tag 2
is `+Y`, tag 3 is `-Y`, tag 4 is `+Z`, and tag 5 is `-Z`. These are cube-local
directions, not robot, hand, or camera directions. The corresponding face
planes are at `X/Y/Z = +/-20 mm`, and each 30 mm tag's detected corners are
mapped in the generated TL/TR/BR/BL order. The ID-to-face assignment and the
printed rotation of each tag must therefore stay consistent with this exact
config file even though the whole cube may be mounted in any orientation.

Running the AprilCube detector on the supplied mounting photo decoded tag 5
(`-Z`) on top and tag 0 (`+X`) on the front-facing side. Therefore the hidden
face against the hand is tag 4 (`+Z`). This is a good usable mounting and does
not need to be changed. A hidden tag simply supplies no observations. Keep the
other faces unobstructed and collect many views in which two adjacent faces are
visible when practical; this improves corner geometry and rejection of bad
detections, but seeing all six tags is neither required nor possible in one
image.

## 2026-08-02 — Physical AprilCube provenance confirmed from demo workspace

Checked `/home/srinivas/Desktop/demo/third_party/aprilcube` after the user
identified it as the print workspace. Its session log records that the first
sharp `models/1x1x1_30_cube` candidate was superseded and that the physical
print completed on 2026-07-14 from `models/dex3_safe_cube/cube.3mf`. The actual
target is the compact 40 x 40 x 40 mm dual-color PLA release with a 3 mm tangent
edge/corner radius, six 30 mm `4x4_100` markers, and IDs 0 through 5.

Both Desktop checkouts are at AprilCube commit
`fc18d50c8bbaadc9646dfd0aa5fcd2404a9868c5`. The demo and calibration copies
of the rounded config and 3MF are byte-identical. The sharp and rounded configs
produce identical per-ID 3D tag-corner maps; rounding affects only the outer
perimeter, not the planar marker coordinates. Re-running detection on the
mounting photo with `models/dex3_safe_cube/config.json` again decoded IDs 5 and
0 on faces `-Z` and `+X`, confirming that hidden ID 4 / `+Z` and all prior
mounting conclusions remain correct. Use the rounded config in all calibration
session manifests for exact physical-artifact provenance.

## 2026-08-02 — Live laptop preview and pose acceptance workflow

Pose teaching and replay capture will both show the rectified RealSense color
stream on the laptop with current-frame AprilCube IDs, corner overlays, and
quality metrics. The calibration detector must be stateless: no optical flow,
prediction, rejected-quad recovery, or temporal filter may make a missing tag
appear valid for capture.

The operator display separates three judgments. Capture readiness checks fresh
camera/state data and the measured settle window. Visual quality checks tag
size, boundary margin, duplicate IDs, visible faces, and multi-face PnP only as
a diagnostic. Dataset value checks whether the view adds image-region, depth,
orientation, or FK diversity relative to saved poses. Red candidates cannot be
saved, yellow candidates require an explicit reason/confirmation, and green
candidates are recommended. Green calibration quality is not a motion-safety
approval; every recorded joint configuration and directed replay transition
still requires the offline collision/path report and operator clearance.

During manual teaching, Space stores the stationary median of measured joints
plus the preview and quality metadata. During replay, the same gates are rerun
after measured settling, a seven-frame stationary burst is retained, and only
one best/median-corner calibration observation is emitted per pose. Aim for
60–80 accepted diverse configurations rather than 60–80 consecutive frames;
record additional candidates as needed because weak, redundant, or unsafe poses
will be rejected. Initial tunable thresholds are now recorded in
`config/capture_quality.yaml`.

## 2026-08-02 — First runnable implementation: stateless visual preview

Implemented the first hardware-independent vertical slice. The AprilCube fork
now exposes `CorrespondenceDetector`, immutable per-tag 2D/3D observations,
explicit duplicate/ignored/rejected metadata, and a stateless PnP diagnostic.
It uses only markers decoded in the current image and deliberately has no
optical flow, temporal prediction, rejected-quad recovery, or prior-pose input.
This work is committed independently in the AprilCube checkout as
`80ed7c72ed00aef6dc70f77d8169a199e9a612cd`.

The root project is now an installable `uv`/Hatch package with a `g1-calib`
entry point. `g1-calib preview` accepts an image, video, or local camera, loads
the exact `models/dex3_safe_cube/config.json`, applies the configured visual
hard/preferred gates, tracks in-memory coverage novelty, and renders tag IDs,
corner order, cube axes, metrics, reasons, and a red/yellow/green side panel.
The panel explicitly says `VISUAL QUALITY ONLY` and reports robot
readiness/collision status as not connected so it cannot be mistaken for
authorization to save or replay a G1 joint configuration.

Verified the headless preview on the supplied physical mounting photograph with
temporary bench intrinsics (`fx=fy=1000 px`, centered principal point). It
decoded tags 0 and 5 on faces `+X` and `-Z`, found a 101.89 px minimum tag side,
221.56 px minimum image margin, and 1.11 px multi-face PnP reprojection error,
and graded the visual view green. The PnP values validate the preview path only;
they are not calibration measurements because the photograph is not a
rectified RealSense frame with its exact `CameraInfo`.

Verification completed:

- root package: 8/8 pytest tests passed;
- AprilCube correspondence plus existing generator/web suite: 18/18 passed;
- Ruff lint and format checks passed for all new root and AprilCube files;
- both packages compiled successfully; and
- the installed `g1-calib` and `g1-calib preview` help surfaces execute.

The next slice remains hardware-facing: discover and verify the rectified
RealSense image/`CameraInfo` topics and add receipt-stamped mode-5 `LowState`
readiness/settling to the same report before pose storage is enabled.

## 2026-08-02 — Python environment aligned with ROS Jazzy ABI

The first `uv sync` selected the active Conda Python 3.13 and NumPy 2.x. A
read-only environment probe then demonstrated that this is incompatible with
the installed ROS Jazzy extensions: `rclpy` is built for Python 3.12 and
`cv_bridge` is built against NumPy 1.x. Continuing with that environment would
make the standalone preview work but break the later ROS camera/state adapter.

Constrained the project to Python `>=3.12,<3.13`, NumPy `>=1.26,<2`, and a
compatible SciPy range, then recreated `.venv` explicitly from
`/usr/bin/python3.12`. The resolved environment uses Python 3.12.3, NumPy
1.26.4, OpenCV contrib 4.11.0, and SciPy 1.14.1. After sourcing
`/opt/ros/jazzy/setup.bash`, the same `uv` environment successfully imports
`rclpy`, `cv_bridge`, and `sensor_msgs`; all eight root tests and the physical
photo preview still pass.

The laptop currently has no Unitree ROS message package or
`unitree_sdk2py` checkout installed in this project environment, and no
RealSense device was returned by `rs-enumerate-devices`. Those are expected
hardware-integration prerequisites for the next slice, not reasons to weaken
the tested Python/ROS boundary.

## 2026-08-04 — `manual_run_001` root-cause analysis completed

Performed a fresh, layered analysis of all 39 selected observations and all 273
raw frame/state windows using the locally built pinned Mike Ferguson optimizer.
An independent projector now reproduces the native `8.204677 px` extrinsics
radial RMS and `4.875132 px` shoulder-roll radial RMS exactly; the calibration
YAML frame rotations are axis-angle vectors, not RPY.

The image-only per-frame PnP floor is `1.541 px` pooled radial RMS, with no
meaningful correlation to the post-shoulder robot residual. Reasonable free
intrinsics, distortion, and cube-dimension fits improve this floor by only
about `0.01--0.10 px` and leave the shoulder estimate essentially unchanged.
The RealSense rectified intrinsics remain the baseline.

Profiled each of the seven left-arm joints with the native optimizer and the
same five-degree aggregate prior. Shoulder roll is uniquely strong:
`+4.346 deg`, `4.875 px`; wrist roll is second at `+3.586 deg`, `5.893 px`;
all other single joints remain at `8.17--8.20 px`. All seven offsets reach only
`4.363 px` on the full data and do not justify a seven-joint calibration.

The shoulder result is insensitive to prior sigma from 1 degree through
unregularized, Huber loss scale, high-PnP-error exclusion, within-burst-motion
exclusion, free intrinsics, or free cube half-spacing. Fifty pose bootstraps
give a 5th-to-95th percentile shoulder interval of `3.221--4.906 deg` and
standard deviation `0.551 deg`. Per-pose shoulder profiles have median
`4.442 deg` and standard deviation `0.656 deg`. A URDF-inertial gravity-torque
proxy spanning `-1.51--+1.81 Nm` has no correlation with the inferred
correction (`r=0.067`, `p=0.685`), weakening a simple gravity-sag explanation.

Capture 028 has a user-confirmed image-stream lag, not evidence that should be
assigned to mechanical compliance. Its inconsistent raw frames were not used
by the optimizer: frame 003 was selected, with `0.891 px` image-only PnP error
and `5.422 px` post-shoulder residual. Removing the entire capture changes the
shoulder estimate only from `+4.346` to `+4.523 deg`. A buffered old image
paired by receipt time to a fresh state produces the observed large
image-versus-q discrepancy, so the earlier inference of proven downstream
motion was retracted. The complete lagged burst is `3.499 s` long with a
`2.833 s` final gap and would be rejected by the current duration/gap gate.
Future collection should also purge pre-capture images and check source/header
freshness; cross-frame corner motion remains a conservative quality diagnostic,
not proof of compliance.

When receipt time tracks exposure, timing is negligible at the recorded
quasi-static velocities: propagating dq over the stored receipt-pairing deltas
produces `0.007 px` RMS and `0.028 px` maximum corner motion. Capture 028 shows
why receipt time alone is not a freshness proof during buffering. Dataset
schema/home-pose history, ROS distribution, arm-side mapping,
measured-versus-commanded q, optical-frame convention, AprilCube mapping,
units, LFS integrity, and the native build were also ruled out.

The working conclusion is a real approximately constant
shoulder-roll-correlated model discrepancy plus unresolved pose-dependent
residual. The old data do not prove that manual support caused the latter. The
exact physical cause of the constant term remains unproven because it is
correlated with camera extrinsics and could be an encoder zero, proximal
joint/link origin error, or force-dependent output deflection.

Documented the complete evidence, hypothesis ranking, and decisive next-data
protocol in `docs/manual_run_001_analysis.md`. The new session must use the
implemented weight-1 measured-pose hold with the operator's hand removed,
retain `tau_est`, reject stale/buffered frames using source freshness plus the
existing gap/duration gate, repeat anchors at the beginning/middle/end, run an
unsupported/touched/unsupported A/B/A pose test, and validate any shoulder
correction on a later independent session before deployment.

## 2026-08-05 — GraspGen-X G1 deployment and calibration audit

Audited arXiv 2606.00998, its full TeX source and supplement, the official
GraspGen-X checkout and history, the released G1 hand descriptor, the project
videos, and related public code. The paper's G1 demo uses a chest stereo
camera, a full object point cloud, an IK target, and joint-space linear
interpolation, and reports 5/5 mustard-bottle grasps while the robot is held by
an overhead support. The footage also shows a high-contrast fiducial on the
dorsal Dex3 housing and a fiducial cube on the table, but does not establish how
either was used.

The exact G1 calibration method is not public. The released code contains only
the G1 gripper inference descriptor, not a whole-G1 controller, IK integration,
camera path, calibration optimizer/data, or saved G1 extrinsic. Its real-world
loader consumes a precomputed `camera_pose` and transforms the back-projected
camera cloud into world coordinates, confirming that calibration is an
external prerequisite rather than part of GraspGen-X.

Two architectures are consistent with the publication. A conventional
base-frame IK pipeline requires the fixed-camera constraint
`B_T_C * C_T_M(i) = FK_B_P(q_i) * P_T_M`, which is the same simultaneous
camera-to-base and target-to-hand problem implemented in this prototype. But
if their chest camera observed both the hand fiducial and object at planning
time, they could instead form the desired hand displacement entirely in the
camera frame, compose that relative displacement with current FK, and run IK
without explicitly knowing `B_T_C`. Ordinary encoder-closed-loop IK alone does
not provide this cancellation; it specifically requires camera-relative visual
hand/object registration. The paper and released code do not distinguish the
two. Our general calibration goal still requires `B_T_C` because the temporary
hand target will not be an online dependency. Detailed findings and the
complete evidence/inference boundary are recorded in
`docs/graspgenx_g1_calibration_analysis.md`.

## 2026-08-08 — Official Quest 3 / Dex3 teleoperation audit

Updated the ignored official `unitreerobotics/xr_teleoperate` checkout from
`64ed45b4177e6297936940866df623b72621643a` to current `main` revision
`845b25a32f7febedf220e830952a7134897adb9d` and initialized its pinned
DexPilot, teleimager, and televuer submodules. Cloned the compatible official
`unitreerobotics/unitree_sdk2_python` revision
`65691c8a8bc53b98d3976dba4dbf9d5d20b2e7f5` separately; the older
`lnotspotl` fork remains pinned solely for G1Pilot provenance and does not meet
the current XR repository's documented minimum revision.

Confirmed the complete path: Quest Browser WebXR emits two 25-joint hand
skeletons; televuer converts wrist and landmark coordinates; dual-arm IK sends
14 arm targets through `rt/arm_sdk` in motion mode; DexPilot maps six human-hand
relative vectors into seven joints per Dex3; and the hand controller publishes
at 100 Hz on `rt/dex3/{left,right}/cmd` while reading the matching state topics.
Dex3 supports hand tracking only: the current CLI rejects controller input.

The upstream program's dual-Dex3 design matches the planned configuration of
two installed Dex3 hands; the earlier rubber dummy hand and AprilCube setup was
specific to calibration and must not be inferred as the teleoperation hardware
configuration. The program is still not ready to run unchanged on this robot:
it starts arm-SDK publishing with zero arm targets and weight one, starts both
Dex3 targets at zero before XR data, uses a 30 rad/s arm limit, lacks an XR
freshness watchdog, and moves both arms to zero on exit. The implementation
must add a receive-only Quest probe, measured-state seeding, conservative
slew/bounds, tracking-loss hold/release, and single-owner shutdown. An
unsupported standing G1 must remain in ready/regular mode and use `--motion`;
the default debug `rt/lowcmd` path is not appropriate. Detailed findings are in
`docs/quest3_dex3_teleoperation_audit.md`.

Arm-only XR operation without Dex3 is kinematically supported, but omitting the
`--ee` argument does not change `G1_29_ArmIK`'s dynamics model. It always loads
`g1_body29_hand14.urdf` and uses RNEA gravity feed-forward with approximately
`0.696546 kg` of modeled Dex3 mass per side. The stock G1 rubber-hand URDF uses
`0.170 kg` per side, a difference of approximately `0.526546 kg` at each wrist.
Two installed Dex3 hands match the current model. Rubber-hand or bare-wrist
operation requires a matching payload mass/COM/inertia model; IK itself does
not change with end-effector mass.

Nominal G1 FK being inaccurate does not prevent manual XR teleoperation. The XR
stack has no external measurement of the physical robot hand; nominal IK maps
Quest wrist targets to joints, while the human closes the outer loop by looking
at the robot or raw head-camera stream and correcting the remaining offset. In
HERO (`2602.16705`), analytical EE FK error is `1.30 cm / 6.03 deg`, but the
automated SONIC baseline has `13.38 cm` real task-space error; HERO needs learned
FK and replanning because its target is autonomous and metric. “Usable manual
teleoperation” therefore does not imply metrically accurate FK. Calibration is
less tolerant because pose-dependent FK errors directly contaminate the
multi-pose transform fit and cannot be corrected by the operator after capture.

HERO's neural residual FK also depends on MOCAP, but only during offline
training. A tracking policy swept one arm and the waist through the workspace
while the authors recorded the 10 joint coordinates, nominal analytical FK,
and Optitrack EE/base poses. The target correction is the SE(3) residual from
nominal FK to the MOCAP-relative EE pose. A three-layer MLP with separate
translation and 6D-rotation heads was trained with MSE using two hours of a
three-hour capture and validated on the final hour. At deployment it needs only
the joint state and nominal FK; MOCAP is replaced by the predicted residual.
This is offline MOCAP-supervised system identification, not MOCAP-free ground
truth or online visual correction.

## 2026-08-08 — SPARK ChArUco board identified

Inspected current `RPM-lab-UMN/spark-data-collection` `main` at
`be284c2f8138f383d260526f68613c7a28d364d4`. Its camera calibration uses an
OpenCV ChArUco board, not a plain ArUco grid: `DICT_4X4_50`, 6 by 9 squares,
30 mm square length, and 22 mm marker length. The active board area is therefore
180 by 270 mm and contains marker IDs 0 through 26 plus 40 interpolated ChArUco
corners. The repository does not include the printable target itself.

The existing local file
`/home/srinivas/Downloads/calib.io_charuco_180x270_9x6_30_22_DICT_4X4.pdf`
matches this specification. Rendering it and running OpenCV 4.11 with the
repository's exact board constructor detected all 27 markers and all 40
ChArUco corners. Print it at actual size with scaling disabled and keep it flat
on a rigid backing.

## 2026-08-08 — Torso-mounted ChArUco fixture feasibility

Audited the complete mode-5 waist/torso URDF chain, all relevant meshes, the
official G1 V1.4 user manual, the official waist-fastener service manual, and
Unitree's newer public USD model. The suspected rear waist slot is the insertion
path for Unitree's two-piece waist fastener: it crosses the articulated waist,
uses two M5 screws, and requires waist motor locking. It is not an accessory
datum that remains rigid to `torso_link` in normal three-DOF waist operation.
A printed insert there could load the waist, break, or pinch wiring.

The correct interface is the four documented front torso M6 installation
holes. The user manual publishes front spacings of 62.4 mm upper, 64.5 mm
lower, and 194.2 mm vertical. The first outline-only drawing/mesh alignment put
the midpoint at `torso_link z=0.150 m`; the shoulder-datum reconstruction on
2026-08-09 proved that estimate wrong and replaced it with `z=0.167996 m`.
Neither the 51,410-face STL nor Unitree's newer USD contains the threaded
inserts; the USD has exactly the same 154,230 triangle vertices. Insert recess,
tolerance, and safe engagement are not available publicly.

Defined a provisional 180 x 270 mm ChArUco target pose centered at
`[0.285, 0, 0.155] m` in `torso_link`, tilted +18 degrees about torso y toward
the head camera. Using the saved 1280 x 720 rectified-color projection matrix
and nominal URDF camera pose, all active corners have at least 56 pixels of
image margin, optical depths are 0.290--0.423 m, and view incidence is 32.5
degrees. A vertical closer board fits but has a poorer 52.5-degree incidence.
The proposed tilted pose improves PnP geometry while remaining a preliminary
live-preview target rather than a print-release dimension.

The nominal FOV result assumes the URDF's nominal camera pitch and is sensitive
to it: a rotation-only sweep keeps the complete active target visible for only
about -3.46 to +3.11 degrees of pitch offset, and keeps the desired 50 px
margin for about -0.66 to +0.35 degrees. This assumption affects fixture
visibility only. The calibration equation recovers the actual locked camera
pose from PnP and does not require the real head pitch to equal the URDF. Before
finalizing the carrier, either aim and lock/mark the adjustable head at the
mechanically fixed board, or hold the board in live rectified color at the
already-fixed head pitch and then encode that observed placement in one rigid,
keyed carrier. Any later head movement invalidates the extrinsic.

Implemented reusable frame algebra, target projection, incidence and fixed-
board camera recovery helpers, plus a command-line nominal FOV audit and unit
tests. The fixed-target method directly computes
`torso_T_camera = torso_T_board @ inverse(camera_T_board)` and eliminates arm
FK, arm compliance, joint synchronization, and multi-pose replay. Detailed
mechanical architecture, accuracy limits, capture procedure, and the required
M6-hole fit coupon/engagement checks are documented in
`docs/torso_charuco_fixture_analysis.md`.

## 2026-08-09 — Front M6 hardware inspected and nominal datum recovered

The user opened the front torso shell and inspected the four intended front M6
mounting holes. The actual upper/lower spacings match the manual, all four
thread axes are parallel to one another, and approximately eight threads are
available for engagement. In the supplied front-interior photograph, the four
mount points are the upper brass-colored pair and the small lower pair; the two
large dark openings in the middle are unrelated.

The vector drawing contains a usable absolute datum even though it does not
print a `torso_link_T_mount` matrix. The visible shoulder-roll circles register
to the shoulder-roll centers computed from the URDF. URDF shoulder separation
is 281.1109 mm; its scale in the drawing differs from the independent scale
given by the published 194.2 mm row spacing by only 0.059%. The upper row is
8.68885 drawing units above the shoulder centerline. This recovers upper
`z=0.2650962 m`, lower `z=0.0708962 m`, and pattern-center `z=0.1679962 m`.
The independent V1.1 diagram agrees within about 0.4 mm, and the registered
official STL outline matches the manual while the prior `z=0.150 m` estimate
does not.

Intended centerline symmetry and published row widths give upper
`y=+/-0.03120 m` and lower `y=+/-0.03225 m`. The front orthographic drawing
shows circular M6 features; combined with the hardware-confirmed parallel
threads, their nominal axes are torso `+x`. Ray-casting these axes into the
outermost front surface of the official torso STL gives:

```text
upper-left   [0.0617071, +0.0312000, 0.2650962] m
upper-right  [0.0615783, -0.0312000, 0.2650962] m
lower-left   [0.0695163, +0.0322500, 0.0708962] m
lower-right  [0.0694891, -0.0322500, 0.0708962] m
```

The four-point average defines the nominal layout frame at
`[0.065572722, 0, 0.167996167] m` with identity rotation. Its x coordinate is
only a convenient average through a curved shell, not an insert-face plane.
Local shell normals differ from torso +x by about 26.5 degrees at the upper
row and 7 degrees at the lower row; fixture contact pads must accommodate the
shell curvature while screw passages follow +x.

This datum is an observability requirement: a fixed camera observing a fixed
torso board measures only `camera_T_board`; without independently known
`torso_T_board`, it cannot separate `torso_T_camera` from the target mount.
The recovered geometry is nominal CAD, not a manufacturing-tolerance report.
An optional physical waist-axis measurement can independently check the
70.90/265.10 mm row heights. Insert recess and thread depth still must be
checked to select screw length, and long lightly engaged screws can verify the
nominal +x direction, but these no longer block the CAD transform.

Resolved the blocker by separating clamp geometry from datum geometry. The
fixture will be modeled around `torso_link_rev_1_0.STL` at identity and will
seat on separated hard pads plus side/contour stops on the rigid external torso
surface. The four M6 passages will be oversized or slotted and will provide
clamp force only. Consequently `torso_T_fixture` and `torso_T_board` come
directly from the CAD assembly and do not require an authoritative
`torso_T_hole`. The black internal foam is not a datum, and the external shell
must be in its repeatable installed state; screw bottoming is not a datum. A
narrow saddle coupon must still demonstrate unique, rock-free seating and free
M6 passage before the full board carrier is printed. Code, tests, the analysis
tool, and the detailed fixture document were corrected to the recovered
coordinates.

## 2026-08-09 — Fixture search and four-leg ChArUco prototype

Searched GitHub, Thingiverse, Printables, MakerWorld, Thangs/STL indexes,
robot-calibration projects, NVIDIA Isaac ROS documentation, and calib.io's
commercial mounting hardware for reusable target fixtures. No downloadable
design combines the G1's front M6 pattern with the exact 180 by 270 mm ChArUco
target. The useful existing work falls into three groups:

- NVIDIA and calib.io use the sound calibration architecture: a rigid optical
  plate, a compact rear adapter, and a robot-specific fastened interface.
- `jywilson2/charuco` provides a parametric split printable target, but it is
  GPL-3.0, square-only, hard-coded to `DICT_4X4_250`, and makes the optical
  target itself an FDM surface rather than supporting the existing exact PDF.
- A recovered Thingiverse frame is a 190 by 190 mm unsupported-paper tension
  frame under CC BY-NC-SA 3.0. The MakerWorld PhotonVision model is also a
  target rather than a robot fixture and explicitly needs full-area bonding to
  a flat wood or plexiglass backer.

Selected NVIDIA/calib.io's general rear-adapter architecture, but implemented
the geometry independently without importing third-party meshes or code. The
new nominal G1 prototype uses four unique printed arms, one per M6 hole. Each
arm doglegs outward beside the target before extending forward, so the camera
sees the complete active pattern. The arms bolt into two rear side rails on a
210 by 300 mm split carrier. The paper target is bonded over its full area with
a thin uniform adhesive; isolated foam-tape pads are prohibited under the
optical plane.

Added `tools/generate_torso_charuco_fixture.py`, reusable geometry in
`src/g1_aprilcube_calibration/torso_fixture_cad.py`, unit tests, eight generated
STLs, a JSON design manifest, and a two-view assembly render. All exported
meshes are watertight. The generator rotates them for a nominal 220 by 220 mm
printer: panels are 210 by 150 mm, the largest arm footprint is about 216 by
216 mm, and the 270 mm seam-bridging rails occupy about 199 by 199 mm on the
diagonal. The earlier full torso-contour saddle concept is superseded by four
small local M6 root seats; a root coupon still has to prove rock-free seating
and clearance on the manufactured robot.

The design remains `nominal_prototype_v0`, not a blind print release. Remaining
physical checks are safe non-bottoming M6 screw length, root-seat fit, the full
target in the actual locked-head rectified preview, mounted pattern dimensions,
flatness, and remove/reinstall pose repeatability.

## 2026-08-09 — All-PLA screwless structural redesign

The user clarified that every structural connection must avoid both fasteners
and adhesive; adhesive is allowed only beneath the ChArUco paper. Superseded
the M3/M4 version with `screwless_prototype_v1`. The only purchased structural
hardware is now four M6 socket-head screws and four washers at the robot.

The 210 by 300 mm carrier remains split for printing. Three skin-depth fingers
interleave across its center seam. Each panel half has one integral captured
tenon for each of the two rear rails, so every rail bridges both panels. Four
tapered PLA keys preload those rail joints. Each arm now ends in a rectangular
10 by 10 mm through-tenon that crosses the rear rail and blank carrier margin;
four more identical front keys pull the arm shoulders against the rails. The
rectangular tenons provide positive anti-rotation geometry, while the keys
provide preload. No snap tabs, M3/M4 hardware, nuts, or structural glue remain.

All eight joints use one universal wedge geometry: 6.5 by 28 mm and tapered
from 1.5 to 3.3 mm. The wedge slot leaves printed side webs connecting each
tenon cap to its body. Nominal mortise clearance is 0.30 mm per side and must be
tuned with a coupon for the actual printer rather than by scaling an STL.

The generator now exports twelve unique STLs. Nine define the assembly: two
panels, two rails, four arms, and one wedge STL printed eight times, for sixteen
PLA assembly pieces. Three more files provide the mating tenon and mortise
joinery coupon plus one contact-face-down M6-root coupon printed twice. Printing
all coupons and assembly parts therefore produces twenty PLA pieces. The
generator also produces a close-up joinery render. Automated generation rejects
a disconnected or non-watertight part, any arm/carrier interference above 0.01
cubic millimetres, or a part exceeding a nominal 220 mm square bed. The current
result has one connected watertight body per STL, zero modeled arm/carrier
interference, and a maximum 219.35 mm bed footprint. The upper arms therefore
need the slicer's full usable bed without an external brim, or a larger printer.

The design is still not a blind print release. The root-seat coupon, printed
joint coupon, M6 engagement and tool access, actual locked-head preview, target
flatness, and repeated remove/install pose spread remain required physical
checks.

## 2026-08-09 — Simplified bolted fixture v2 and root-access correction

The user correctly noticed that the original bolted v0 render placed the first
arm segment directly in front of each torso M6 screw. A mesh audit confirmed
that the holes themselves existed but the assembly path did not: the v0 arm
intersected a modeled 5 mm Allen-key approach by about 18 cubic millimetres,
an M6 socket-head insertion envelope by about 216 cubic millimetres, and a
13 mm washer path by about 405 cubic millimetres per root. The old v0 arm STLs
must not be printed.

Replaced both v0 and the later all-PLA wedge experiment with
`bolted_prototype_v2`. Each M6 root now has a 12 mm deep front hub with an
exposed washer/head and an outboard side bridge. The generator explicitly
places a 14.4 mm diameter, 70 mm long virtual approach cylinder in front of
every M6 head and rejects any intersection. The structural beam similarly ends
outboard of the board-side M4 axis, leaving a 5.5 mm radius rear socket/nut
approach free. Automated results are zero modeled M6 and M4 tool-path
interference.

The structural assembly is eight PLA/PLA+ prints: two carrier halves, two rear
rails, and four arms. Three interleaved skin fingers align the carrier seam.
Four square panel bosses locate the rails and are clamped by four M3 x 16
screws total. Four square arm keys locate the arm flanges and are clamped by
four M4 x 30 screws total. Four M6 screws and maximum-13-mm-OD washers attach
the roots to the robot. Adhesive remains limited to a thin full-area layer
beneath the paper target. The obsolete wedge and wedge-coupon outputs are
removed automatically by the generator.

The revised fixed nominal board pose is center `[0.277, 0.000, 0.161] m` in
`torso_link` with +18 degrees about torso y. This shortened the upper arms and
kept the URDF-only nominal image check just above the requested 50 pixel top
and bottom margin. The largest printable footprint is 212.79 mm on a nominal
220 mm square bed. Modeled structural volume is 368.73 cubic centimetres,
25.19% below v0's 492.88 cubic centimetres before slicer walls and infill.

The generator exports nine unique STLs: the eight structural parts plus one
M6 root/access coupon to print twice. Every output is one connected watertight
body. Maximum arm/carrier intersection is only a sub-0.001 cubic-millimetre
boolean boundary artifact, below the 0.01 threshold. The assembly and detailed
fastener renders were regenerated. Targeted lint passes and the complete test
suite passes with 172 tests.

Physical release gates remain: seat one coupon at an upper and a lower hole,
prove both seats are rock-free, select an M6 length that cannot bottom while
providing about 6--7 mm engagement, verify the full target in live rectified
color at one exactly locked and marked head pitch (nominally 65 degrees within
the proposed 60--70 degree range), check mounted target scale/flatness, and
measure remove/reinstall pose spread. CAD defines the nominal board transform;
these physical checks bound the difference from the assembled transform.

## 2026-08-09 — H2D one-piece fixture v3

The Bambu Lab H2D removes the 220 mm-bed constraint: its official single-nozzle
build area is 325 by 320 mm. Replaced the split carrier, rear rails, M3 hardware,
panel seam, and printed locating keys with one continuous 210 by 300 mm carrier.
The carrier has a 2.4 mm optical skin, two integral full-width load crossbars,
three narrow integral anti-bow ribs, and four local integral arm pads. These are
features of one print, not separate pieces.

The final structural architecture is five PLA/PLA+ prints: one carrier and four
independent arms. Four M6 x 20 socket-head screws with maximum-13-mm-OD washers
attach the arms to the torso. Each arm uses two M4 x 30 clearance screws, 12 mm
apart, with a flat washer under every head and nyloc nut to attach to the
carrier without concentrating clamp load in the PLA. The paired screws prevent
arm rotation without a tolerance-sensitive printed key. There
are no adjustable joints, snap fits, wedges, rails, M3 fasteners, or structural
adhesive; adhesive is used only as a thin full-area layer under the paper.

The generated structural volume is 338.63 cubic centimetres: 8.16% below split
v2 and 31.30% below v0. The carrier print footprint is exactly 210 by 300 mm;
the largest arm is about 213.68 by 213.49 mm. Every exported STL is one connected
watertight body and fits the H2D single-nozzle area. Modeled M6 and M4 tool-path
interference is zero; maximum arm/carrier intersection is 0.00042 cubic
millimetres, a boolean boundary artifact below the 0.01 threshold.

The physical release gate is now concrete: print the M6 root coupon twice, seat
it at one upper and one lower torso hole, and verify an M6 x 20 screw clamps
before bottoming. Its nominal under-head stack is 12 mm printed root plus about
1.6 mm washer plus about 6.4 mm insert engagement. Also verify the finished
target is flat and dimensionally correct, visible at the locked head pitch, and
repeatable after several remove/reinstall cycles.

## 2026-08-09 — From-scratch fixture review and multicolor v4

The user confirmed that the normal silver exterior torso shell will be fully
installed during calibration. That fixes the intended axial datum: each arm
seats on a small exterior-shell contact around its M6 passage. The hidden foam
and recessed insert faces are not fixture datums. Recovered local triangle
normals are now stored and reproducibly emitted by
`tools/recover_g1_front_m6_pattern.py`. They differ from torso +x by about 26.5
degrees on the upper row and 7 degrees on the lower row, so the previous flat
root face was not adequate. A tangent disk would still contact a curved shell
primarily at one point, so the final 20 mm-diameter counterfaces directly sample
the local official shell surface in polar rings. Their planar front washer
faces remain 12 mm forward along the M6 axes. Separate upper and lower coupons
test these two geometries before printing the long arms.

Reconsidered the entire fixture topology. A one-piece 210 by 300 mm carrier
plus four independent arms remains the simplest support-free architecture for
the H2D. A two-arm frame would introduce long redundant crossbars and span
manufacturing error between torso holes. The carrier now has one M4 x 30 clamp
per arm rather than two. One flange by itself would be free to rotate if
treated as an isolated pin joint, but that is not this assembly: four broad,
clamped flange faces at four widely separated carrier locations jointly
constrain the single rigid plate. The complete hardware is four M6 x 20 screws,
four maximum-13-mm-OD M6 washers, four M4 x 30 screws, eight M4 washers, and
four M4 nyloc nuts.

The four connector centers are frozen in carrier coordinates. Board center or
pitch changes now alter only the four arms, so the expensive carrier remains
reusable. The active target is no longer paper. The generator creates the
legacy OpenCV 6 by 9 ChArUco layout with `DICT_5X5_50` as a 0.6 mm black PLA
inlay in the white structural body. Both coplanar bodies share one print
transform in
`carrier_multicolor_h2d.3mf`; the intended H2D setup is matte black and matte
white PLA through the two 0.4 mm nozzles, optical face down on smooth PEI. The
carrier's 210 by 300 mm footprint fits the official 300 by 320 mm dual-nozzle
area.

Automated generation detects all 27 ArUco markers and all 40 ChArUco corners,
finds a flush 0.6 mm optical face, zero volumetric white/black overlap, zero M6
and M4 tool-path interference, and only 0.00029 cubic millimetres of
arm/carrier boundary overlap. Every structural body is watertight; the
intentionally disconnected black pattern contains 62
watertight shells. The standard 3MF archive loads as a two-geometry scene with
210 by 300 by 9.4 mm bounds. The five structural prints model 325.02 cubic
centimetres, 34.06% below v0 and 11.85% below split v2.

This supersedes v3 and its paper target, eight M4 screws, generic root coupon,
and `carrier_one_piece.stl`. Remaining hardware gates are deliberately short:
test the two row-specific root coupons against the installed silver shell,
confirm M6 x 20 clamps without bottoming, verify the full board in the actual
locked-head camera view, measure the printed active boundary and diagonals,
check optical-face flatness, run the detector, and quantify remove/reinstall
pose repeatability.

## 2026-08-09 — Torso target dictionary separated from AprilCube

Changed only the fixed torso ChArUco target from `DICT_4X4_50` to
`DICT_5X5_50` so detections cannot be confused with the hand-mounted AprilCube,
whose saved target configuration uses `DICT_4X4_100`. The physical layout
remains 6 by 9 squares with 30 mm squares, 22 mm markers, a 180 by 270 mm
active area, marker IDs 0 through 26, and 40
ChArUco corners. The old calib.io `DICT_4X4_50` PDF is no longer a valid target
for this fixture. The generated two-color 3MF and its design manifest are the
source of truth, and the eventual torso-board detector must construct the same
`DICT_5X5_50` board.

## 2026-08-11 — Direct Dex3 dorsal ArUco mount

Reconstructed the flat wrist target visible in GraspGen-X. Multiple video
frames decode it as OpenCV `DICT_6X6_50`, ID 4. The resulting target is one
50 x 62 x 4 mm printed carrier: a 40 mm active marker with a 5 mm quiet zone
and a separate 12 mm finger-side screw tab. White and black PLA are exported as
one two-material H2D 3MF. No cube, adhesive, secondary carrier, or moving-finger
attachment is involved.

The physical hand established two dorsal M3 holes. Initial image analysis and
caliper checks suggested roughly 15–15.5 mm spacing, but the official 20-page
Dex3-1 user manual was then recovered from Unitree's image-based documentation.
Its dimensioned mounting-hole view specifies 2 x M3, 3 mm deep, 15.00 mm center
spacing, and a hole-pair centerline 66.70 mm from the wrist-side datum. The URDF
places the palm origin at that wrist datum and finger bases at x=77.7 mm, so the
hole midpoint is palm `[x,z]=[66.70,0] mm`; the individual holes are at
`z=+/-7.50 mm`.

Ray intersections with Unitree's official palm STL at the documented holes
give dorsal surface `y=-21.33 mm`. The screw pads are 1.5 mm high and the wrist
pad is 2.0 mm high, forming a three-point seat on the shallowly curved shell.
The nominal marker-center translation is therefore
`[0.03720,-0.02683,0] m` in `palm_link`. Two M3 x 8 ISO 10642 screws pass
through a 5.5 mm printed stack, engage 2.5 mm of each official 3 mm blind hole,
and retain 0.5 mm bottom clearance.

## 2026-08-11 — Dex3 dorsal mount physical-fit correction

The first physical two-hole carrier print showed that the nominal 2.0 mm third
pad did not contact the shell, leaving the marker plate able to flex about the
M3 screw line. The three-point locating principle was sound; the installed-pose
calculation was not. It had treated the marker plate as parallel to the palm
link's `x-z` plane instead of following the local tangent at the M3 bosses.

A least-squares tangent fit over the official palm-mesh boss footprints gives
`dy/dx=0.019947`, or 1.143 degrees. Across the original 44.5 mm support span,
that omitted pitch accounts for approximately 0.89 mm of the observed missing
pad height. Revision 2 keeps the 4.0 mm plate and M3 x 8 fastener stack, narrows
the unused full-width tab to a centered 30 x 10 mm tab, and moves the third
datum onto the flatter central shell. The new support is 35 mm wristward of the
screw line, 6 mm in diameter, 2.57 mm high at its center, and has a
1.994-degree shell-matching face. The marker center remains inside the support
triangle. The two screw bosses are reduced to 7 mm diameter to make the hard
datums more local. The resulting overall carrier envelope is 50 x 60 x 4 mm;
the 40 mm active `DICT_6X6_50`, ID 4 target and 5 mm quiet zone are unchanged.

The selected standalone `unitree_ros` Dex3 URDF has the same palm-mesh hash and
all seven finger joint origins/axes as the GraspGen-X cuRobo model. Unitree's
public `xr_teleoperate`, historical palm STLs, `unitree_cad`, and current
Hugging Face `unitree_model` archive were also audited. None publishes the
detailed Dex3 manufacturing solid; the dimensioned user-manual drawing plus
the official URDF/STL provide the required datums. GraspGen-X publishes no
marker-fixture CAD.

The generated carrier, marker inlay, and 15 mm fit coupon are watertight. The
neutral-pose collision test clears every finger link, the rendered assembly
matches the intended dorsal placement, and the project test suite passes. The
remaining physical gate is deliberately cheap: print the coupon first and
confirm that both M3 screws enter simultaneously without bending it, then print
the full two-color plate and check the marker with the detector.

## 2026-08-12 — Bilateral Dex3 posture commissioning passed

- The live loaded articulated measured-to-middle-close sweep passed with
  `22.5 mm` minimum clearance at
  `left_hand_index_0_link/left_hip_yaw_link` across 16 samples.
- Both Dex3 hands reached and position-settled at NVIDIA GR00T's middle-close
  target. Worst final error was `0.077351 rad` at
  `left_hand_index_0_joint`; raw maximum `dq=0.114504 rad/s` is diagnostic only.
- The original finger posture was restored within `0.016643 rad`, the validated
  shoulder route reversed, both hands received terminal timeout, and arm-SDK
  weight returned to zero.
- The bilateral physical test commissions the shared Dex3 control path for both
  matched right-ID-4 and left-ID-5 hardware profiles.
- Automatic collection reuses that complete lifecycle instead of closing Dex3
  directly in Ready: measured finger hold, right-then-left shoulder clearance,
  loaded finger-sweep recheck, middle-close, unchanged full-weight handoff to
  the calibration executor, calibration return to the outward handoff,
  initial-finger restoration, unchanged handoff back to the clearance executor,
  reverse left-then-right shoulder route, hand timeout, and arm-SDK weight zero.

## 2026-08-12 — G1Pilot collision-model correction for Dex3 auto collection

- The first 100-target Dex3 plan omitted G1Pilot's selected
  `shoulder_yaw_link/torso_link` pair on both arms. The physically failed
  `candidate_0627` is `62.8 mm` inside G1Pilot's right-shoulder-yaw/torso
  primitive proxy. Of the old plan's 100 endpoints, only 35 meet G1Pilot's
  `10 mm` threshold and 52 are in collision, so that plan must not be reused.
- Collision dimensions are not reimplemented in project YAML. The generator
  `tools/generate_g1pilot_collision_urdf.py` preserves Unitree's exact
  revision-1.0 links, joints, limits, inertials, visuals, and camera frames and
  mechanically copies primitive `<collision>` elements from pinned G1Pilot
  commit `72acc803edefe583c24f53e76a21d8d4ed10ed14` (source URDF SHA-256
  `59a3f308bc0b9ef14c3aa4009009ced2c931b1eb703a5468485978a7f0dbde7d`).
- The Dex3 pair policy adds the two exact upstream shoulder-yaw/torso pairs.
  The only project-specific geometry remains the segmented Dex3 hands and
  dorsal plates because G1Pilot models rubber hands instead.
- Replacement plan `work/dex3_auto_collection_g1pilot_001` contains 80
  primary targets and 20 backups selected from 139 feasible candidates. All
  200 directed tree edges pass the frozen `10 mm` clearance policy, the route
  begins and ends at the measured handoff, and `candidate_0627` is absent.

## 2026-08-12 — Automatic-capture writer isolation

- Session `sessions/dex3_auto_20260812T162341Z` retained nine complete
  seven-frame captures. During the ninth manifest commit, the 250 Hz control
  thread observed a `55 ms` scheduling gap against its unchanged `50 ms`
  local limit. The separate PC2 heartbeat timeout remains `500 ms`.
- The former exception path then called `finish_capture()` after the executor
  was already in `FAULT`, masking the real controller fault with `capture can
  only finish while capturing`. Capture cleanup now preserves and reports the
  original executor fault.
- The dummy-hand and Dex3 workflows used the same synchronous persistence
  path. A representative Dex3 capture is only about 2.6% larger than a
  dummy-hand capture; the successful dummy run simply did not encounter the
  scheduler-tail event.
- Automatic collection now prewarms one spawned writer process before any
  command publisher is created. One completed capture at a time is validated,
  PNG/JSON encoded, fsynced, and added to the manifest there. The controller
  holds the current pose and its parent process continues polling control
  health until that durable commit finishes; only then may the route continue.
- A realistic retained seven-frame capture took `0.479 s` to commit in the
  isolated writer while the parent ticker's maximum gap was `17.35 ms`. This
  fixes the interference without weakening either safety timeout and preserves
  capture-by-capture recovery after cancellation, process failure, or power
  loss.

## 2026-08-12 — Dex3 automatic-route backup phase removed

- Session `sessions/dex3_auto_20260812T170649Z` attempted all 80 primary
  targets, accepted 62, and rejected 18. The old authored plan then entered a
  finite 20-target backup phase implemented as a separate
  `handoff -> backup -> handoff` sortie for every backup. It accepted two
  backups before the run stopped at 64 accepted poses. This was not an infinite
  loop, but the repeated return/restart behavior was operationally unacceptable.
- Fifteen of the eighteen primary rejections were caused only by the former
  `0.5 s` consecutive-image-gap gate. That gate measured RealSense transport
  continuity rather than calibration validity. It is removed as a hard gate;
  every image still requires its own measured-state pairing, the continuous
  low-state history across the burst must remain stationary, and the complete
  seven-frame burst remains bounded to `2.0 s`.
- Backup targets and the accepted-goal mechanism are removed rather than
  rerouted. Automatic plans now contain exactly 80 information/coverage-selected
  targets. Each target is attempted once, rejected views are retained as such,
  and the session finalizes the valid captures after the finite route is
  exhausted instead of treating fewer than 80 accepted captures as a runtime
  failure.
- Replacement plan `work/dex3_auto_collection_g1pilot_002` selects 80 targets
  from the same 139 feasible candidates. Its finite route has 160 directed
  steps; all 160 pass the frozen full-state IK/FCL policy. The default
  `tools/g1_dex3_calibration.sh` launcher now selects this plan.

## 2026-08-12 — Right Dex3 62-primary fixed/free target comparison

- The unfinalized but hash-verifiable session
  `sessions/dex3_auto_20260812T170649Z` contains 64 accepted captures. The final
  two are the old backup-phase `candidate_0728` and `candidate_0808`; analysis
  uses exactly the preceding 62 accepted primary observations.
- A true six-parameter solver mode now holds the configured CAD
  palm-to-marker transform fixed. The existing twelve-parameter mode remains an
  explicit `--target-transform-mode optimize` comparison. Fixed is the Dex3 CLI
  default; the result records which mode was used.
- With the same deterministic 50-train/12-holdout split, fixed CAD gives
  `22.724 px` training and `23.045 px` holdout radial RMS. Free target gives
  `10.534 px` training and `9.737 px` holdout radial RMS, an `82.15%` reduction
  in holdout squared error. Both Jacobians are full rank and all 100 pose-level
  bootstrap trials succeed.
- The free solution's effective marker correction from CAD is `13.638 mm` and
  `4.621 deg`; its bootstrap component spreads are `0.65-0.90 mm` and
  `0.54-0.79 deg`. This is a stable effective correction, not proof of a
  physical mount error, because a free hand target absorbs arm-FK/joint-zero
  bias.
- Vision/timing evidence is much cleaner than either kinematic residual:
  selected-frame marker PnP reprojection is `0.215 px` median, `0.455 px` p95,
  and `0.539 px` maximum; image/state nearest pairing is `0.832 ms` median and
  `2.359 ms` maximum. After freeing the target, residual correlations still
  reach `|r|=0.518` with right wrist roll and `|r|=0.489` with elbow, so the
  remaining error is pose/joint dependent rather than image noise alone.
- The validated left ID-5 plan
  `work/dex3_left_auto_collection_g1pilot_001` is bound to the commissioned
  bilateral outward-clearance handoff, contains 80 unique targets, and passes
  all 160 directed edges. `tools/g1_dex3_calibration.sh left` selects the left
  profile; no robot command was issued while authoring or validating it.

## 2026-08-12 — Control-gap policy corrected after isolated-writer run

- After writer isolation, session `dex3_auto_20260812T164451Z` reached 32
  accepted captures and three recoverable visual rejects, then the controller
  observed one `51 ms` interval while holding `candidate_1504`. No capture
  commit was active. This proves the earlier write stall and this event are
  separate; an occasional controller/DDS/OS scheduling overrun remains possible
  on general-purpose Linux.
- Git history shows the `50 ms` immediate-fault rule originated in the first
  local prototype commit `5011fe9`, whose hardware configuration explicitly
  called its controller values initial commissioning values. It is not from
  Unitree, NVIDIA GR00T, G1Pilot, or HERO. Unitree's official G1 arm-SDK example
  publishes at 50 Hz, and the pinned XR controller uses a Python thread at
  250 Hz without an equivalent single-overrun Damp rule.
- The local `50 ms` threshold has been removed completely. It produces neither
  a warning nor a control action. Motion increments use the fixed nominal
  `4 ms` period of the configured 250 Hz controller, matching Unitree's fixed-
  period target limiter, so a late cycle cannot catch up by issuing a larger
  joint step. Every resumed tick still requires fresh mode-5 LowState and
  passes the unchanged held-arm drift checks before a normal command continues.
- A `250 ms` gap remains a local hard fault. Configuration validation requires
  it to be no more than half of the commissioned `500 ms` PC2 heartbeat
  timeout, retaining an independent `250 ms` fail-closed margin. A process
  crash, controller hang, or network loss still stops the heartbeat and causes
  PC2 to request Damp.

## 2026-08-12 — Left Dex3 dorsal-frame correction

- The first left ID-5 plan was physically tested and the marker was not visible
  at its commanded views. The failure was not the left joint indexing or the
  bilateral shoulder-clearance route. The left hardware profile had copied the
  right palm-to-marker transform unchanged.
- Unitree's official left and right Dex3 palm meshes are reflections across
  palm-link Y. Both markers are physically dorsal, but dorsal is `-palm-Y` on
  the right and `+palm-Y` on the left. The copied transform therefore modeled
  the left marker normal through the palm, approximately opposite its real
  exposed direction.
- The geometry source is now side-specific. `palm_T_plate_mm()` and
  `palm_T_marker_face()` derive proper right-handed transforms for `left` and
  `right`; the ID-5 manifest/render uses the official left palm. The left
  marker/palm collision box was also regenerated around the mirrored physical
  plate rather than the former right-side placement.
- Automatic collection had also reused the route's `10 mm` minimum-clearance
  policy for the preliminary live Dex3 shoulder/finger search, while the
  physically passed Dex3 commissioning routine used `5 mm`. Both preliminary
  paths now share the commissioned `5 mm` policy; the authored calibration
  route remains independently validated at `10 mm`. The outward search still
  starts at `0.08 rad` and increases only when the live geometry requires it,
  with every rejected offset and pair now printed.
- The invalid plan `work/dex3_left_auto_collection_g1pilot_001` is superseded
  and no longer selected. A new hardware/plan binding check rejects it before
  command creation because its recorded hand-to-target transform differs from
  the current left hardware profile.
- Replacement plan `work/dex3_left_auto_collection_g1pilot_002` was generated
  from the same measured bilateral `0.08 rad` clearance reference. It contains
  80 unique left-arm targets and a finite 160-edge handoff-rooted round trip;
  every edge passes, with `10.549 mm` minimum modeled clearance. All desired
  target views are front-facing (`target-to-camera dot = 0.581..0.993`). The
  left launcher now selects this replacement plan.

## 2026-08-13 — Table extent is not inferred from the resting cube

- The physical table is 1400 x 700 mm, but its dimensions do not locate its
  robot-facing edge, center, or yaw in the robot frame. Centering that rectangle
  under the observed cube can make the modeled table pass through the seated
  torso even though the physical table does not.
- Table length, width, and thickness were therefore removed from the task
  request and collision scene. The resting 45 mm AprilCube supplies only the
  support-plane point and normal from its gravity-aligned bottom face.
- CuRobo retains the complete locked robot geometry, so the active right arm is
  still checked against the fixed torso, legs, left arm, both Dex3 hands, and
  marker plates. The left arm receives no changing target; low-level execution
  holds its measured acquisition pose with gravity feedforward.
- A separate local plane guard checks the right wrist, articulated Dex3, and
  attached payload. It deliberately excludes the elbow, torso, legs, and left
  arm: geometry below the table height but outside an unseen finite footprint
  is not a table collision. Conversely, exact elbow/forearm-versus-table-edge
  certification remains unavailable until a table boundary pose is observed or
  registered. The software records this limitation and does not claim otherwise.

## 2026-08-13 — Focused end-to-end command surface and CUDA/UVM diagnosis

- `collect-calibration` now owns the complete finite standing lifecycle in one
  operator command: live read-only Ready/Dex3/camera snapshot, isolated CuRobo
  bilateral clearance planning, isolated 80-view calibration planning, one
  SPACE, exact measured arm/Dex3 acquisition, NVIDIA middle-close posture,
  frozen route capture, per-view visual rejection without aborting the route,
  return to the live handoff, measured finger restoration, reverse bilateral
  shoulder route, and clean arm-SDK weight-zero release. No robot transport is
  constructed before SPACE.
- `solve-calibration` is fixed to the pinned Mike Ferguson Ceres backend with
  only `d435_joint` free. The CAD palm-to-marker transform is baked into the
  observation model, the deterministic holdout and bootstrap reports are
  emitted, and the result becomes one hash-checked removable bundle rather than
  an edited source URDF.
- `run-tabletop` owns the complete seated right-Dex3 lifecycle in one operator
  command. It takes over all 29 joints at the measured state, keeps the left arm
  fixed with gravity feedforward, plans in a separate CUDA process, executes
  eight connected CuRobo trajectories with explicit Dex3 open/close/restore
  transitions, and restores Unitree control through the commissioned
  `FSM 0 -> 1 -> 3` path.
- The table collision policy was narrowed to what the single resting cube can
  actually establish: a support-plane point and normal. Full G1 self-collision
  remains active; a local plane guard covers the moving right wrist/hand and
  attached cube without inventing unseen table edges or placing a fabricated
  table box through the seated robot.
- On the laptop, `nvidia-smi` sees the RTX 5090 Laptop GPU and the loaded kernel
  module plus `libcuda.so.1` both report `595.71.05`. PyTorch 2.8.0+cu128 still
  reports CUDA unavailable because direct `cuInit(0)` returns error 999 and an
  `strace` proves `open('/dev/nvidia-uvm', O_RDWR|O_CLOEXEC) = -1 EIO`.
  Forcing runtime power from D3cold to D0 did not change the failure. The UVM
  module has no clients, and the machine has remained booted since August 4
  through repeated suspend/resume cycles. Reboot is the selected recovery;
  CuRobo/PyTorch reinstall is not justified by this evidence.

## 2026-08-13 — Post-reboot real-state CuRobo verification

- Reboot restored the existing environment without reinstalling it. Direct
  `cuInit(0)` returns success, Torch 2.8.0+cu128 sees the RTX 5090 Laptop, and
  Warp/CuRobo initialize on CUDA normally.
- Three integration defects were found with real retained G1/Dex3 state and
  fixed at their source boundaries: collision-only models now preserve
  NVIDIA's required tool frames; NVIDIA's explicit
  `extra_collision_spheres: null` is normalized before reserving attachment
  slots; and Unitree Dex3 DDS finger values are mapped by joint name into
  CuRobo's different kinematic-tree order.
- The actual calibration workflow was checked after applying the planned
  shoulder-clearance and middle-close preparation state, not from the raw
  Ready snapshot. From 1544 visible candidates, CuRobo found 212 feasible
  endpoints and selected all 80 requested information/coverage poses. The
  frozen route contains 81 trajectories, every capture has a validated path
  back to the measured handoff, and the one unavailable direct inter-pose edge
  correctly falls back through that handoff. No robot command was issued.
- The tabletop shortlist is valid only with object +Z upward, which is tag 132
  on the generated 45 mm cube; arbitrary tabletop yaw remains allowed. This is
  now an explicit preflight condition rather than an implicit assumption from
  the Isaac support-plane filter.
- CuRobo's grasp helper selects one reachable final grasp before checking its
  approach and does not automatically try a different goal-set member after a
  later failure. The wrapper now backtracks through the remaining qualified
  goal set and accepts a grasp only after the complete approach, grasp,
  closed-hand attached-payload lift, table-plane guards, and exact reverse
  lifecycle pass. Each rejected candidate records its stage and numerical
  reason.
- A requested 16-sphere MorphIt payload fit silently returned only two spheres.
  It was replaced by a deterministic 3 x 3 x 3 circumscribed-cell cuboid cover:
  all 27 spheres are installed through CuRobo's AttachmentManager and their
  union conservatively contains the complete 45 mm cube.
- The complete real-state offline tabletop plan selected
  `cube_head__seed_0000000089__sample_163`. Before that, it rejected sample 217
  at attached-lift IK (`0/16` collision-constrained successes, best position
  miss `18.224 mm`), sample 194 at approach planning, and sample 160 for a
  `15.0 mm` right-middle-finger table-plane crossing. The accepted route has
  `46.1 mm` open-hand, `45.7 mm` closed-hand, and `27.6 mm` payload minimum
  plane clearance, and includes all six outbound/reverse phases. Its ignored
  verification artifact is
  `work/post_reboot_gpu_verification/tabletop_task_plan_upright_conservative.json`
  with plan SHA-256
  `4a8f01bfef1d91bd8d2afe1c3b3731da0a49efdaa62cc4962c034a8c577ae3d5`.
- Final local verification is `233 passed, 3 skipped` in the Python 3.10
  control environment and `229 passed, 1 skipped` in the Python 3.11 CuRobo
  environment; Ruff is clean for project source, tests, and tools.

## 2026-08-13 — Focused-repository hardware runtime made self-contained

- The first `run-tabletop` launcher attempt stopped before ROS initialization
  because the new focused repository lacked its local
  `deps/cyclonedds_python_prefix`; the equivalent commissioned shim still
  existed only under the prototype repository. The same inspection also found
  that the focused Python 3.10 environment lacked `cyclonedds` and
  `unitree_sdk2py`. No DDS channel or robot command was created.
- The exact commissioned `lnotspotl/unitree_sdk2_python` fork is now a pinned
  submodule at `7c661d27f4ae064ffd0dd633fd9d5b518ef0b508`, rather than an undeclared
  dependency on another working tree. It is isolated in the `hardware` project
  extra so the Python 3.11 CUDA planner does not install robot transports.
- `tools/setup_control_env.sh` now constructs the account-local ROS
  CycloneDDS include/bin/lib shim before dependency resolution, installs the
  hardware extra, and verifies the real 35-slot HG LowCmd, CRC, and raw
  MotionSwitcher bindings without initializing DDS. ROS Humble `rclpy`,
  CycloneDDS 0.10.2, and the pinned Unitree SDK all import through the hardware
  launcher environment.

## 2026-08-13 — Restored the commissioned PC2 RealSense lifecycle

- The first camera-enabled `run-tabletop` attempt reached the read-only frame
  preflight but received `0/5` frames. Read-only PC2 status showed the cause:
  Unitree's `video_hub_pc4` factory service was running and the temporary ROS
  RealSense node was not. No robot publisher or arm ownership had been created.
- The focused repository now carries the older prototype's commissioned
  `tools/g1_realsense_pc2.sh` byte-for-byte. Its `start` action exclusively
  locks camera lifecycle changes, releases `video_hub_pc4`, starts the pinned
  D435i serial at RGB8 1280x720x15, and verifies serial, USB 3.2, stream
  profile, launcher, and node process. A failed start rolls back to the factory
  service; `stop` terminates only the tracked node and restores that service.
- `g1_tabletop_hardware.sh` invokes this idempotent startup before the two
  camera-dependent commands, `collect-calibration` and `run-tabletop`.
  `inspect-hardware` remains read-only and does not change camera ownership.
- The first started driver could be discovered over DDS but its PC2 log showed
  continuous `Frames didn't arrived within 5 seconds`; a direct laptop probe
  consequently received neither Image nor CameraInfo. USB enumeration itself
  was healthy at 5 Gb/s. The older viewer's proven clean `stop` then `start`
  lifecycle recovered the stream immediately, after which a reliable laptop
  subscriber received 5/5 RGB8 1280x720 images and 5/5 matching CameraInfo
  messages. Camera-dependent focused commands now perform that same clean
  restart instead of reusing a tracked but potentially wedged driver process.
- The next read-only tabletop preflight recovered a clear image but correctly
  rejected the physical object before SPACE. The object on the table was the
  existing 40 mm `4x4_100` ArUco cube (visible IDs 3 and 5), whereas the
  imported tabletop grasp contract, detector, collision payload, and retained
  physics-qualified shortlist all describe the 45 mm AprilTag-36h11 cube with
  IDs 128–133. Substituting only the detector would leave grasp/contact and
  payload geometry inconsistent, so no such shortcut was made.
- The same failure exposed two diagnostic defects. Tabletop observations now
  label the required object as `tabletop AprilCube`, rather than inheriting the
  calibration helper's `hand target` wording, and failure-frame manifests
  serialize `ImageTiming` explicitly. Consequently future preflight failures
  retain all captured PNG evidence and a valid timing manifest.

## 2026-08-13 — Qualified 40 mm cube grasps replace the mismatched object contract

- Pulled `sri299792458/g1-aprilcube-demo` commit
  `724ee07079c25765b3e630f5508bca3b7731eef5`, which regenerates the grasp
  pipeline specifically for the printed 40 mm `dex3_safe_cube`; the poses were
  inferred for this mesh rather than rescaled from the earlier 45 mm object.
- The source retained 3178 of 4096 newly inferred grasps through Isaac VIRAL,
  then admitted 15 right-Dex3 candidates through the complete open-hand,
  support-plane, straight-pregrasp, closed-hand, moved-object, closure-pose,
  and required-contact gates. The committed shortlist records every gate and
  its non-exclusive rejection counts.
- The focused runtime now uses one consistent object contract: the 40 mm
  `dex3_safe_cube` mesh with SHA-256
  `27c8460e40a85475e87c3cc0d6090c3c9500de4fa7f5728a2462fc099ef3d927`,
  its `4x4_100` detector configuration, 40 mm task/payload geometry, and the
  corresponding 15-grasp shortlist. Runtime loading verifies both the mesh
  hash and its OBJ extents against the frozen task request.
- The shortlist uses object +Z as its canonical support normal, but the
  physical cube is symmetric. Runtime planning maps whichever detected face is
  uppermost to canonical +Z before applying the same 15 grasps; no tag ID is a
  physical setup restriction, and tabletop yaw remains free.
- The earlier 45 mm/AprilTag-36h11 planning artifacts and numerical clearance
  evidence remain historical evidence only. They are superseded for physical
  execution and are not selected by the focused runtime defaults.
- Offline reprocessing of the retained preflight frame with the correct cube
  model detects faces `-Y` and `-Z` at `2.224 px` reprojection RMS. Tabletop
  perception had incorrectly used the commissioned `1.5 px` preferred-quality
  warning boundary as a hard gate; it now reuses the existing `3.0 px` reject
  boundary. This frame therefore passes geometry quality; its tag-5-up pose is
  canonicalized exactly like any other face-up resting pose.
- A real-state, no-command CuRobo check reused the retained upright object pose.
  The 100 mm supported escape passed with `185.8 mm` G-frame plane clearance.
  The complete task did not pass at that old object placement: four candidates
  reached grasp selection but failed respectively at two local hand-plane
  guards, attached-lift planning, and approach planning; the remaining eleven
  had no reachable final grasp. This does not validate a robot run and does not
  justify changing any grasp or collision gate. The next live run must plan
  from the newly observed, tag-4-up cube pose and may require relocating the
  cube if CuRobo reports the same reachability result.
- A subsequent live preflight decoded the tabletop cube in every frame, but a
  tabletop-only `30 px` minimum rejected four markers measuring `26.0..28.3
  px`. That duplicated and contradicted the commissioned capture policy, whose
  hard minimum is `25 px` and preferred minimum is `40 px`. The tabletop
  command now loads the same validated quality configuration and passes its
  `25 px` hard tag-size and `3.0 px` hard reprojection limits to both preflight
  and post-ownership observation; those values are no longer independently
  chosen in the hardware workflow.
- The next burst passed those hard per-frame gates but reported `3.859 deg`
  orientation spread. The five saved frames span only `0.14 s`; offline corner
  inspection showed the large top marker stable within `0.2 px`, while the
  strongly foreshortened side marker jumped by as much as `8.9 px`, producing
  about `9 mm` of false depth motion when both faces were forced into one PnP
  solve. This is estimator noise, not evidence that the stationary cube rotated.
- Tabletop pose estimation now deliberately uses the decoded face with the
  largest shortest side in each current frame. This retains a stateless raw
  image measurement and every existing burst-spread gate, while excluding a
  lower-quality oblique face from corrupting a good face. Its selected
  correspondences are passed directly to AprilCube's public stateless
  `estimate_pose_diagnostic`, which delegates to AprilCube's SQPnP/RANSAC plus
  LM implementation; the local planar-PnP implementation was deleted.
  Reprocessing the exact saved burst remains below the unchanged `2.0 deg` and
  `5.0 mm` burst gates.

## 2026-08-13 — CuRobo supported escape preserves the G1Pilot pair policy

- The first owned tabletop run stopped before any changing target because all
  supported-escape IK seeds were marked infeasible. Detailed CuRobo pair
  diagnostics found constant sub-millimetre collision-sphere overlaps inside
  both locked Dex3 hands. G1Pilot's selected-pair policy likewise never checks
  collisions internal to a fixed hand. Arm planning now omits only pairs
  within each locked hand; hand-to-arm, hand-to-body, hand-to-world, payload,
  and opposite-hand checks remain active.
- Removing that masked rejection exposed the measured seated start geometry.
  The older G1Pilot policy intentionally omits the adjacent
  `torso_link/right_shoulder_roll_link` pair but explicitly retains
  `torso_link/right_shoulder_yaw_link`. G1Pilot's exact primitive geometry and
  NVIDIA's CuRobo spheres independently measure the latter start overlap as
  `1.194 mm` and `1.203 mm`, respectively.
- The focused CuRobo model therefore carries the same adjacent-pair omission
  and a fixed `1.5 mm` proxy-fit allowance on the checked right shoulder-yaw
  link. This is not derived afresh at runtime. Replanning the exact retained
  hardware request passes the 100 mm supported escape. Across its 41 samples,
  the exact G1Pilot shoulder-yaw/torso clearance is least at the starting
  `-1.194 mm` state and improves to `+24.77 mm`; the route never deepens the
  pre-existing overlap. The grasp frame finishes `144.6 mm` above the inferred
  table plane. No robot command was sent during this diagnosis.
- The same retained observation now advances to complete-task planning. Three
  reachable open-hand approaches are rejected by the table-plane guard at
  `-3.4`, `-2.1`, and `-3.2 mm`; a fourth fails approach planning and the
  remaining eleven have no reachable final grasp. For the leading candidate,
  NVIDIA's exact Dex3/wrist collision meshes independently give `-3.55 mm` at
  the same terminal wrist-pitch sample as CuRobo's `-3.44 mm` sphere result.
  This is not a sphere-fit false positive, so the table gate remains unchanged
  and that retained observation is not a runnable complete task plan.
