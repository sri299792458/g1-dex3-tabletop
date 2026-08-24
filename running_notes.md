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

- This physical G1 publishes two distinct IMU streams. `/lowstate.imu_state`
  measures the pelvis, while `/secondary_imu` (`unitree_hg/msg/IMUState`)
  measures the torso. A read-only live check on 2026-08-16 observed different
  orientations and raw accelerations from the two streams. Raw episode profile
  `g1_seated_tabletop_raw_v2` therefore requires both; the torso stream directly
  observes camera-body rotation but does not provide translation.
- The same raw profile requires the D435i's separate
  `/camera/gyro/sample` and `/camera/accel/sample` streams. PC2 enables the two
  physical motion streams with the RealSense driver's `unite_imu_method=0`, so
  the bag retains raw angular velocity and linear acceleration rather than a
  driver-interpolated orientation. `--skip-camera-recording` continues to omit
  the high-bandwidth image streams while these low-bandwidth motion topics
  remain recorded.
- Native unaligned D435i Z16 depth is enabled at 640x480x15 and recorded with
  its CameraInfo by default. This is the RealSense measurement relevant to
  table-normal translation: an offline plane fit can recover camera-to-table
  distance and plane normal. It cannot observe translation parallel to an
  otherwise featureless plane. Live alignment, point-cloud generation, and
  depth processing remain disabled on PC2. A camera-only live check on
  2026-08-16 confirmed 15.00 Hz depth, 199.5 Hz gyro, and 100.1 Hz accel on the
  laptop, after which the temporary node was stopped and `video_hub_pc4` was
  restored. No robot command publisher was created.
- `/tf_static` is also a required low-bandwidth recording topic. The same live
  check verified that the serial-specific driver publishes the complete
  `camera_link` transforms for depth, color, gyro, and accel frames, including
  the nonzero factory depth-to-color baseline. Consequently pixel alignment can
  be performed offline from the raw images, both CameraInfo messages, and the
  recorded static transforms; no live alignment is needed on PC2.
- A 3.87-second plain-MCAP integration check started after the driver and still
  received the transient-local transform tree: 1 `/tf_static` message, 58 depth
  images, 58 depth CameraInfo messages, 773 gyro messages, and 387 accel
  messages. The depth-plus-motion bag wrote 34.5 MiB (about 8.9 MiB/s). The
  temporary verification bag was deleted and the factory camera service was
  restored; no robot command publisher was created.
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

## 2026-08-16 — Matched cushion/rigid fixed-board diagnostic

- Added `measure-seat-compliance` as a diagnostic separate from the cube grasp
  lifecycle. It requires the already frozen `DICT_5X5_50`, 6x9, 30/22 mm
  ChArUco board fixed to the table. It never fabricates a cube, table extent,
  grasp, finger motion, or payload.
- After one SPACE approval, the existing seated debug-lowcmd takeover, complete
  dual-Dex3 gravity feedforward, PC2 watchdog, fixed-rate controller, measured
  Dex3 hold, and verified FSM `0 -> 1 -> 3` restoration remain unchanged.
- Both arm plans are generated from one post-takeover loaded state before the
  first changing target. The existing CuRobo supported-escape implementation
  now accepts the board plane as an alternative plane source: strict full-robot
  self-collision, selected wrist/hand plane guard, 100 mm normal lift, and the
  exact frozen reverse are unchanged. No fake scene object or table box is
  added.
- The controller switches active arms through the same full-weight
  `adopt_owned_control` mechanism already commissioned by automatic
  calibration. Only one arm moves at a time; the other arm and complete body
  remain held. The measured finger postures are never changed.
- The default run performs five left/right pairs. It measures the fixed-board
  pose at loaded baseline, each lifted endpoint, and each exact return. Board
  burst spread is recorded rather than promoted into a new arbitrary rejection
  threshold; every accepted frame still passes the frozen ChArUco geometry,
  corner-count, positive-depth, IPPE-ambiguity, and reprojection checks.
- Raw profile `g1_seated_tabletop_raw_v2` records the synchronized evidence
  needed for offline attribution: pelvis and torso IMUs, waist/arm measured and
  commanded states, both Dex3 streams, RealSense gyro/accel, native depth/RGB,
  both CameraInfo streams, and `/tf_static`. One condition alone establishes
  camera motion relative to the board. Only a matched run on the same chair
  frame with the cushion replaced by rigid non-slip support can attribute an
  excess to seat compliance.
- This implementation was verified offline only; no robot command was issued.

### Physical runs, result, and operating decision

- The physical runs are retained at
  `runs/seat_compliance_cushion_20260816T120404Z` and
  `runs/seat_compliance_rigid_20260816T121420Z`. Both completed all five
  left/right lift-return pairs, restored seated control, reported no cleanup or
  recorder problems, and retained complete 14-topic MCAPs. The cushion run has
  1,212,481 messages over 266.80 s; the rigid run has 1,128,698 messages over
  248.71 s.
- Do not interpret the `status.json` motion summary as the isolated lift
  response. It compares every endpoint with the early loaded observation made
  before both CuRobo plans and therefore includes planning-time startup
  settling. The analysis below uses repetitions 2--5 and brackets every lift
  with the immediately preceding and following returned observations.
- That warning applies to the two retained physical runs above. The diagnostic
  code was subsequently corrected: every new cycle records an explicit
  `pre_lift` board/state burst while the fixed-rate controller holds the arm,
  and new `status.json` summaries report `pre_lift -> lifted`,
  `pre_lift -> returned`, and `lifted -> returned`. The early loaded observation
  remains planning provenance only and is no longer labeled as lift motion.
- Relative to the preceding return, the cushion run measured
  `10.825 +/- 0.132 mm, 1.399 +/- 0.008 deg` for the left arm and
  `10.386 +/- 0.346 mm, 1.659 +/- 0.056 deg` for the right arm. The rigid run
  measured `5.355 +/- 0.221 mm, 0.707 +/- 0.029 deg` and
  `5.260 +/- 0.046 mm, 0.790 +/- 0.009 deg`, respectively.
- Comparing each lift with its following exact return gives the same
  conclusion: cushion `9.112 +/- 0.103 mm, 1.212 +/- 0.015 deg` left and
  `10.719 +/- 0.347 mm, 1.660 +/- 0.043 deg` right; rigid
  `4.260 +/- 0.068 mm, 0.522 +/- 0.007 deg` left and
  `3.783 +/- 0.140 mm, 0.529 +/- 0.018 deg` right. The rigid support reduced
  the observed translation by approximately 50--65 percent and rotation by
  approximately 50--69 percent, but did not eliminate body-relative table
  motion.
- This is physical motion rather than ChArUco estimation noise. Rigid-run
  endpoint bursts had median spread `0.068 mm / 0.020 deg` and maximum spread
  `0.210 mm / 0.047 deg`. Native D435i depth independently measured the
  board-normal component as approximately `3.5--4.3 mm` on the cushion and
  `0.8--1.3 mm` on rigid support, closely matching the ChArUco differential.
- The endpoint IMUs independently confirm the rotation. In steady repetitions,
  the torso IMU changed by approximately `1.396 deg` left and `1.589 deg`
  right on the cushion, versus `0.752 deg` and `0.790 deg` on rigid support.
  The pelvis IMU moved less, showing that both seat/pelvis motion and measured
  waist deflection contribute.
- The complete-body lowcmd waist targets did not change during a lift. Measured
  waist-pitch error at lifted endpoints was approximately `2.3 deg` on the
  cushion and `1.2 deg` on rigid support. Combining the fixed-board observation
  with measured waist FK attributes the rigid-run response approximately to a
  `3.72 mm / 0.514 deg` waist-state effect plus a `1.96 mm / 0.221 deg`
  pelvis/support effect on the left, and `2.61 mm / 0.408 deg` plus
  `3.30 mm / 0.430 deg` on the right. These are 3D transform effects and their
  scalar norms are not additive.
- **Causal limitation:** the physical runs were not posture-matched. Loaded
  waist pitch was `7.118 deg` for the cushion run and `0.101 deg` for the rigid
  run; the observed board normal differed by `14.984 deg` in the camera frame,
  and the loaded arm configurations and resulting CuRobo trajectories also
  differed. Moving the board does not invalidate within-run relative motion,
  but the posture change prevents assigning every millimetre of the difference
  uniquely to cushion compression. Preserve this limitation in any report.
- **Operating decision:** continue tabletop development on the rigid chair.
  This is an engineering decision supported by the consistent reduction across
  both arms, ChArUco, depth, pelvis IMU, torso IMU, measured joints, and unchanged
  commands; it is not a claim that the present experiment perfectly isolated
  cushion compliance. Do not change the camera calibration bundle from this
  diagnostic.
- The residual rigid-chair motion remains task-significant for a 40 mm cube.
  The tabletop pipeline should eventually observe the cube again after the
  supported arm has lifted and the loaded body has settled, then plan from the
  fresh measured 29-joint state. This corrects the proven observation-to-motion
  state change without requiring the wrist marker. Gain or integral tuning is
  deferred; it cannot remove support translation and should not be substituted
  for this measurement-timing correction.
- A future causal follow-up should use a rigid spacer with the cushion's seat
  height, keep the board and head pitch fixed, match the loaded waist and arm
  posture, and record an explicit board burst immediately before every lift.
  Three repetitions per condition should be adequate given the repeatability
  seen here. Until that test, retain the raw runs and describe the cushion
  contribution as strong but posture-confounded evidence.

### Offline camera state-estimation benchmark

- Added a read-only research stack in `state_estimation.py` and
  `state_estimation_replay.py`. It filters the retained MCAP to state and timing
  topics, creates no Unitree publisher, hash-binds the source observations,
  URDF, and removable calibration bundle, and evaluates explicit estimator
  hypotheses against the fixed-board camera pose. The design and limitations
  are documented in `docs/state-estimation-research.md`.
- The rigid legacy run uses repetitions 2--5 and anchors each lift to its
  immediately preceding returned observation. Its uncorrected camera motion is
  `5.355 mm / 0.707 deg` left and `5.260 mm / 0.790 deg` right. A hybrid using
  pelvis-IMU orientation plus measured waist FK for position and the torso IMU
  for orientation leaves `1.052 mm / 0.123 deg` left and
  `1.220 mm / 0.104 deg` right after pairing through the fitted image-header to
  MCAP clock mapping.
- These numbers depend on a short-horizon fixed-pelvis-IMU-origin contact
  hypothesis. They do not prove globally observable odometry. Absolute
  position and yaw still require a visual/depth landmark, VIO/SLAM, or another
  external update. Do not copy the hybrid into the robot controller until it
  passes continuous-trajectory and changed-posture replay tests.
- Replaying the same estimator on the cushion run reduced
  `10.825 mm / 1.399 deg` to `1.920 mm / 0.139 deg` left and
  `10.386 mm / 1.659 deg` to `1.963 mm / 0.135 deg` right. The remaining
  translation is about twice the rigid-chair residual, consistent with the
  fixed-pelvis-origin hypothesis degrading on compliant support. The
  cross-condition posture confound still applies.
- MCAP timing shows roughly 1,000 changed LowState ticks/s; about four percent
  of recorded messages are consecutive duplicate ticks and must be
  deduplicated. RealSense producer clocks have a stable large offset from MCAP
  receipt time, and color versus D435i IMU headers differ systematically by
  about `68.6 ms`; fit each stream to MCAP time independently.
- After fitting the RealSense color header clock to MCAP time, the lowcmd joint
  target was already static for `0.82--0.85 s` before every lifted image burst.
  Waiting longer is not the primary remedy. The application image callback
  receipt was about `102 ms` late at the median endpoint and reached `190 ms`,
  so it remains diagnostic metadata rather than a fusion timestamp. Endpoint
  accelerometer noise is approximately `0.03--0.06 m/s^2` per axis,
  far too large for unanchored double integration to recover millimetre
  translation over these multi-second motions.
- Added a continuous fixed-board replay using the unchanged strict ChArUco
  detector. It evaluates only loaded-baseline through final-return time; the
  earlier draft incorrectly included post-task seated-controller restoration,
  which produced unrelated cushion outliers. Every tenth RGB frame was enough
  for the architectural comparison. A single initial hybrid anchor accumulated
  `4.661 mm` mean error on rigid support. With fresh 6D visual anchors at a
  realized `1.335 s` cadence, propagation was `0.371 mm` mean and `0.873 mm`
  p95; at `2.002 s`, it was `0.427 mm` mean and `0.993 mm` p95. Cushion results
  were `0.435/1.189 mm` and `0.497/1.389 mm` at the same cadences.
- Added an independent native-depth plane replay using the recorded D435i
  color/depth static TF. RGB and depth are phase-shifted by `66.67 ms`; poses
  are interpolated in their common hardware-header clock. On rigid support,
  338 sampled frames gave `-0.341 mm` mean depth-minus-ChArUco plane offset,
  `0.513 mm` p95 absolute offset, `0.292 deg` mean normal disagreement, and
  `0.827 mm` plane-fit RMS. Cushion results were `-1.869 mm`, `2.035 mm`,
  `0.639 deg`, and `0.746 mm`. Within-run offset standard deviations were only
  `0.104 mm` and `0.094 mm`; depth is useful for relative tilt/table-normal
  correction but not for table X/Y or yaw, and its absolute bias is not shared
  across these changed views.
- The first integration contract is therefore visual 6D anchoring after loaded
  ownership, high-rate hybrid IMU/waist-FK propagation, depth-plane tilt/height
  updates, and fresh visual resets at 1--2 second stationary task boundaries.
  Missing visual evidence grows uncertainty; it does not authorize an invented
  in-plane correction. CuRobo replanning consumes the boundary estimate later;
  continuous trajectory deformation is not part of this research stack.

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

## 2026-08-14 — Shoulder-yaw overlap is a start-only CuRobo recovery

- This section supersedes the fixed `1.5 mm` shoulder-yaw buffer described
  above. A negative per-link buffer was too broad because it relaxed every
  later CuRobo plan involving that link. The general tabletop robot model is
  strict again: `right_shoulder_yaw_link` has its upstream zero buffer and the
  torso pair remains enabled.
- An offline diagnostic at the retained measured start found `+5.706 mm`
  clearance between the official Unitree torso and right-shoulder-yaw URDF
  collision meshes. The existing NVIDIA spheres report `-1.203 mm` for the
  same configuration because one `78.04 mm` torso sphere fills the physical
  shoulder recess. This mesh comparison established the cause; it is not a
  second runtime collision stack, and the upstream sphere geometry is not
  modified.
- Only a deep-copied model used to generate the supported escape excludes the
  known `torso_link/right_shoulder_yaw_link` pair. The resulting trajectory is
  immediately evaluated again with the strict CuRobo sphere model. The exact
  measured-start overlap is the baseline: no other pair may exist or appear,
  the overlap may never increase from one sample to the next, it may not
  reappear after clearing, and it must be absent at the escape endpoint. The
  task planner then starts from that clear endpoint with the strict model. The
  supported return is the exact reversed escape.
- CuRobo's stored per-pair kernel value is a squared overlap score, despite the
  old diagnostic helper treating it as metres. The helper now calculates true
  linear sphere penetration from the transformed sphere centers, effective
  radii, and configured padding. Replanning retained request
  `runs/tabletop_20260814T001909Z/loaded_request.json` passes all `41` recovery
  samples with `1.203 mm` initial penetration and `144.6 mm` terminal G-frame
  table-plane clearance. This was offline-only; no robot command was sent.

## 2026-08-14 — Local table obstacle steers the open-hand transit

- The existing table policy was only a post-planning guard. It correctly
  checks the complete right wrist, Dex3, and attached-payload route against the
  infinite plane inferred from the resting cube while ignoring unrelated
  torso, leg, left-arm, shoulder, and elbow geometry. CuRobo therefore had no
  table obstacle during optimization and could generate a route that the
  independent guard subsequently rejected.
- On retained live request `runs/tabletop_20260814T001909Z/loaded_request.json`,
  all three reachable grasp endpoints and their straight approach segments
  were above the plane. Their unconstrained clearance-to-pregrasp trajectories
  nevertheless dipped `3.441`, `2.131`, and `3.174 mm` below it at intermediate
  samples. The failure was therefore the open transit, not the GraspGenX grasp
  endpoints or the final straight approach.
- `config/tabletop/task.yaml` now explicitly supplies a `400 x 400 x 20 mm`
  open-transit planning patch centred under the detected cube, with its top
  face coincident with the inferred plane. The patch is used only while the
  open hand moves from clearance through grasp approach. Supported escape and
  attached-payload planning do not include it. The patch may extend beyond the
  actual table edge: that only invents extra obstacle space and cannot weaken
  the independent infinite-plane validation, so no cube-to-edge setup distance
  is required or claimed.
- The patch is a planner steering aid, not the safety authority. The unchanged
  infinite-plane guard still validates every right-wrist/hand/payload sample,
  including motion outside the finite patch. Full-robot self-collision also
  remains active. No table collision tolerance or robot collision sphere was
  relaxed.
- The production CuRobo worker, without monkey patches, replanned the retained
  observation and selected
  `cube_head__seed_0000000109__sample_213` for a complete
  pick/lift/replace/return lifecycle. Its open-route minimum plane clearance is
  `+4.775 mm` and its closed-hand minimum is `+2.898 mm`. The conservative
  attached-cube sphere cover begins at `-1.813 mm` relative to the inferred
  plane and never deepens that initial support-contact proxy overlap, preserving
  the existing payload policy. The first two candidates remain correctly
  rejected for closed-finger clearances of `-3.8` and `-5.0 mm`. The verified
  task-plan SHA-256 is
  `890e08c3e89e1825259bbcaef9f74d4602ecee58c25b98e8ea9a4d112f94fe44`.
- The isolated CuRobo task worker initially reproduced the supported-escape
  endpoint with one `2.98e-8 rad` float32 rounding difference. This is
  physically meaningless but exceeded the immutable execution contract's
  `1e-8 rad` exact-join threshold. The task planner now accepts only a start
  state within a scale-aware float32 numerical bound and then preserves the
  exact serialized request state as sample zero; a real discontinuity still
  fails. The downstream controller adapter now assembles all eight trajectories
  (`41, 62, 22, 41, 41, 22, 62, 41` samples) with exact clearance, return, and
  handoff joins. The complete execution-plan SHA-256 is
  `3edd0c46b22f25e5b1638b8b75f6fdaeb40309b32edec040b1d43a3e93fd3838`.
  This verification was offline-only; no robot command was sent.

## 2026-08-14 — One canonical Dex3 grasp atlas now serves either arm

- The tabletop command now requires `--arm left|right`. The selected hardware
  configuration, seven active arm joints, grasp/attachment frames, collision
  links, table-plane guard, trajectory records, and Dex3 commands all follow
  that one value. The opposite arm and hand remain locked at their measured
  takeover state, and the opposite fingers are never commanded by the task.
- A second GraspGenX/PhysX qualification run is not needed for the left hand.
  Direct inspection of GraspGenX commit `e45a6f6` showed that the left and right
  canonical open meshes agree within about `2 um`; applying the descriptor's
  exact mirror mapping makes articulated FK agree to numerical precision
  (`1.16e-15 m`). The retained `object_T_G` poses are therefore side-independent.
- The former right-only shortlist is stored once as
  `config/tabletop/cube_dex3_executable_v1/shortlist.yaml`. Its original
  right-descriptor joint values and source paths remain immutable qualification
  evidence. Runtime adapts only the Dex3 posture and fixed `G_T_palm` frame:
  every left joint changes sign, and the canonical index/middle chains exchange
  physical motor names. This adapter is centralized in `dex3_handedness.py` and
  is hash-bound through the canonical profile.
- A fixed opposite shoulder-yaw/torso sphere overlap was exposed when the exact
  retained right-arm request was mirrored into a left-arm offline test. Those
  two links are both locked during left planning, so their relative pose cannot
  change. The model now excludes only that invariant fixed-arm pair, while the
  selected shoulder-yaw/torso pair remains strict. Both proximal shoulder-roll
  adjacency omissions remain the existing G1Pilot policy, independent of the
  selected arm.
- Complete production-worker replays passed without robot commands for both
  sides using the same retained live cube observation. The right arm selected
  `cube_head__seed_0000000109__sample_213`. Under the final scoped-patch policy,
  the left arm strictly rejected three generated wrist/shoulder-to-torso routes
  and selected `cube_head__seed_0000000149__sample_203`. The left supported
  escape ended with `145.5 mm` G-frame plane clearance, and its final complete
  task plan SHA-256 is
  `14b3f7d4fd5182ab30e4e7ff6d5a43daab8cfdc8b749880cb97a0bac3f7e9e49`.

## 2026-08-14 — Exact diagnosis of the corrected-tilt right-arm IK failure

- Frozen run `runs/tabletop_20260814T130028Z` placed the cube about `43.4 mm`
  farther robot-left than the earlier solvable observation. The final-grasp
  goal set returned no valid right-arm IK, but the original status did not name
  which constraint rejected the solutions.
- Replaying the exact production goal-set call and its subsequent diagnostic
  pass identified the first blocker as a scene collision, not the right hand:
  the fixed, table-resting `left_wrist_pitch_link` sphere penetrated the local
  `400 x 400 mm` open-transit table patch by `0.213 mm`. The left middle-finger
  and palm spheres were also within the configured `10 mm` collision activation
  zone. The local patch was intended to steer the selected arm, but CuRobo's
  world checker applied it to the complete locked robot model.
- Removing that scene blocker for diagnosis exposed secondary Cartesian-exact
  right-arm branches for `cube_head__seed_0000000119__sample_205` that strict
  self-collision correctly rejects. One branch penetrated the sphere model at
  `right_shoulder_yaw_link/torso_link` by `80.214 mm` and
  `right_elbow_link/torso_link` by `36.379 mm`; its secondary
  `right_hand_palm_link/left_hand_thumb_1_link` overlap was `2.249 mm`.
  The alternate exact branch still penetrated
  `right_shoulder_yaw_link/torso_link` by `12.919 mm` (plus a `6.498 mm`
  logo/shoulder-yaw overlap). These are genuine cross-torso alternatives, but
  they must not be confused with the fixed-left-wrist/table-patch blocker that
  terminated the unmodified production goal set.
- No collision exception was added for either finding. Moving the selected arm
  through the torso or opposite hand remains invalid. The finite patch policy
  also must not silently reject an unrelated supported arm; the new selected-arm
  path is the clean immediate test when the object lies on that side.

## 2026-08-14 — Tabletop failure logging policy

- A terminal failure must identify its pipeline stage and the lowest useful
  physical cause already available: IK pose error for unreachable targets,
  named self-collision links and penetration for converged-but-colliding IK,
  named link/sample/clearance for the independent table-plane guards, and the
  existing measured joint/error diagnostics for execution faults. Successful
  per-frame internals and optimizer tensors are not dumped.
- A final-grasp goal-set failure now performs one read-only diagnostic IK pass.
  Cartesian-converged branches are mapped back to the exact shortlist candidate;
  enabled CuRobo self-collision pairs report penetration in millimetres, while
  named robot-link/scene-object pairs report signed clearance. If no branch
  converges, the error instead reports the best translation and rotation
  residual and does not claim a collision.
- Each isolated CuRobo worker is now streamed unchanged to the terminal and to
  a stage-specific `*.planner.log` beside its request/output artifacts. On
  failure the retained top-level `status.json` error includes the worker's final
  diagnostic line and the complete log path, rather than only an exit code.

## 2026-08-14 — Local patch steering is scoped without weakening self-collision

- The corrected-tilt failure proved that a local planning patch cannot be
  applied as an ordinary full-robot world obstacle: an unrelated supported arm
  may legitimately occupy the same inferred table region. The patch now acts
  only on the selected wrist and Dex3 links used by the independent table-plane
  guard.
- This does not delete the rest of the robot from CuRobo. The steering model
  uses CuRobo's per-link `collision_sphere_buffer` to make unrelated radii
  negative for world collisions; CuRobo compensates those same offsets in its
  self-collision padding, preserving the original full-robot self geometry.
  After trajectory generation, every sample is independently rechecked against
  the strict full-robot self model and the strict full-robot cube scene (with
  only the commissioned fingertip contact links exempt at contact), followed by
  the existing selected wrist/hand infinite-plane check.
- Replanning frozen corrected-tilt right request
  `runs/tabletop_20260814T130028Z/clearance_request.json` now proceeds past the
  unrelated left wrist. The strict recheck rejects two generated alternatives
  for `right thumb/right hip` penetration (`0.470 mm` and `2.008 mm`), then two
  closed grasps for below-plane finger clearance (`-5.0 mm` and `-3.8 mm`), and
  selects `cube_head__seed_0000000109__sample_213` for the complete lifecycle.
  Its offline task-plan SHA-256 is
  `8cdb9a371378c4e46a32f81f608f38523688fbbb8a94f1f8427ef35c0a4ed1db`.
- The same frozen placement does not produce a complete left-arm task: two
  alternatives fail strict wrist/torso self-collision, one attached lift is
  outside pose tolerance, one approach fails, and the remaining goal set has
  no Cartesian-converged branch (`22.328 mm`, `1.999 deg` best residuals).
  This is a clean reachability result, not a handedness-adapter or table-patch
  failure. All checks were offline; no robot command was sent.

## 2026-08-14 — Raw MCAP control-contention benchmark

- Cloned `RPM-lab-UMN/spark-data-collection` at commit `be284c2` as a separate
  read-only reference checkout. Its `generate_dummy_episode.py` validates
  synthetic bag-to-LeRobot data, while its live recorder delegates to a separate
  `ros2 bag record` process. It contains no concurrent controller-jitter test;
  the documented claim that plain MCAP is low overhead was therefore not treated
  as evidence for this G1's 250 Hz command loop.
- Added a no-robot benchmark that reuses the production
  `ExecutorControlDriver` scheduling loop and production ROS image conversion.
  Separate synthetic publishers provide 29-joint measured/command streams at
  250 Hz and RGB8 1280x720 at 15 Hz. The benchmark rotates baseline,
  state-only-MCAP, and state-plus-camera-MCAP conditions and reports p99/p99.9/max
  tick gaps, threshold counts, camera receipt rate, recorded topic rates, disk
  throughput, and child-process CPU time.
- The launcher enforces `ROS_LOCALHOST_ONLY=1`, ROS domain 221 by default, and
  creates only `/g1_recording_benchmark/...` topics. It cannot discover the G1
  and contains no Unitree transport or robot command publisher. The Humble MCAP
  plugin and vendor library are extracted account-locally into ignored `deps/`;
  no sudo or system package mutation is required.
- Three 15-second trials per condition completed on this laptop. State-only
  capture wrote `0.54 MiB/s`. Raw RGB capture sustained a mean `39.86 MiB/s`
  and recorded `14.81-15.04 Hz`. Worst observed control gaps were `9.272 ms`
  with no recorder, `7.387 ms` with state-only recording, and `7.603 ms` with
  state plus raw RGB. None of the nine trials produced a gap above `10 ms`, and
  none approached the old `50 ms` failure.
- This evidence does not show the bursty failure mode seen when PNG/manifest
  persistence ran alongside the old calibration controller. Plain MCAP uses a
  buffered sequential external writer instead. It also does not prove physical
  safety: the synthetic publishers consume laptop CPU unlike PC2 producers, and
  the run does not include a live Unitree transport. The next adoption boundary
  remains a stationary robot-connected A/B timing run before enabling recording
  during motion.

## 2026-08-14 — SPARK-style raw episode recording integrated

- `run-tabletop` now owns one adjacent `raw_episode/` artifact containing a
  plain untrimmed MCAP, `episode_manifest.json`, `notes.md`, and
  `recorder.log`. The implementation reuses SPARK commit `be284c2`'s central
  pattern—a separate `ros2 bag record` process and per-episode manifest—rather
  than adding serialization or image encoding to the controller.
- The stable tabletop profile records only general source streams: official
  complete G1 LowState (including IMU), complete debug LowCmd, both Dex3 state
  and command pairs, raw head RGB, and CameraInfo. Cube detections, table
  estimates, grasp decisions, CuRobo plans, and derived success remain the
  existing run JSON artifacts and are not republished as invented bag topics.
- Recording starts after SPACE and immutable-config rechecks but before
  activation reacquisition or any command publisher. It remains active through
  control, Dex3, and PC2 cleanup, then receives SIGINT. A start failure blocks
  command creation. A finalization failure is recorded separately and cannot
  issue a robot-mode request.
- Completion is audited from rosbag's `metadata.yaml`: clean recorder exit,
  MCAP storage, non-empty required topics, and exact message types. An
  incomplete bag is retained and named as such; it does not silently become a
  valid learning episode. Full behavior and the capture/archive/published-data
  boundary are documented in `docs/data-recording.md`.
- The hardware launcher reuses the account-local MCAP plugin and the existing
  official `unitree_hg` install from `g1pilot_ws` strictly for ROS type support.
  It verifies both before RealSense startup. No G1Pilot node or controller is
  launched.
- A final localhost-only smoke test exercised the production recorder class,
  actual Humble `ros2 bag record`, and the account-local MCAP plugin against two
  synthetic 250 Hz JointState streams. Graceful SIGINT produced a complete
  1.96-second bag with 981 messages and an audited manifest. No Unitree topic or
  robot transport was present.
- Camera capture remains the default profile. `--skip-camera-recording` removes
  raw RGB and CameraInfo together from the recorder and its completeness audit,
  while the live RealSense perception path remains unchanged. The selected
  mode is explicit in the manifest rather than inferred from missing messages.

## 2026-08-14 — Supported escape now requires a strictly clear live start

- Removed the selected shoulder-yaw/torso start-recovery exception. The earlier
  `1.203 mm` NVIDIA sphere-proxy overlap belonged to one retained measured state;
  it is not a fixed property of every Ready or seated posture. Two later retained
  hardware starts both measured zero penetration for that pair.
- Every supported-escape run now checks the exact live state with the strict
  CuRobo self-collision model before planning. Any enabled overlap stops before
  changing motion and reports every physical link pair and penetration in
  millimetres. The operator can reposition the arm and rerun.
- The complete generated escape is independently checked with the same strict
  model. There is no allowed baseline penetration, monotonic-recovery rule, or
  optimizer-only pair omission. Successful plan provenance records that the
  start was clear and the number of strictly checked trajectory samples.

## 2026-08-14 — Ctrl+C during watchdog startup exits cleanly

- A Ctrl+C received while the laptop was waiting for PC2's initial
  `WATCHDOG_READY` marker previously terminated only the local SSH process. The
  remote watchdog could remain alive briefly with its exclusive lock, causing
  the next run to report that another watchdog was already armed.
- Startup exception handling now sends the watchdog's existing `DISARM` command,
  waits for `WATCHDOG_DISARMED` and remote process exit, and then reraises the
  original interruption. No new recovery mode or protocol was added.
- The top-level CLI converts `KeyboardInterrupt` into one
  `interrupted by operator` message and exit status 130 after normal cleanup,
  instead of printing a Python traceback.

## 2026-08-14 — Raw Unitree rosbag topic mapping corrected

- Run `tabletop_20260814T170502Z` retained a 6.88 GB MCAP containing only RGB
  and CameraInfo. Its manifest correctly marked all six requested Unitree
  streams empty. The recorder had been given SDK channel strings such as
  `/rt/lowstate`, but the live ROS graph exposes that DDS channel as
  `/lowstate`; `rt` is the Unitree DDS partition, not part of the ROS name.
- The recording profile now uses `/lowstate`, `/lowcmd`, and
  `/dex3/{left,right}/{state,cmd}`. A read-only recorder probe against the live
  graph subscribed to all six exact names without publishing any robot command.
- Startup no longer treats the generic `Recording...` line as sufficient. The
  recorder uses rosbag's `--include-unpublished-topics` support and requires a
  `Subscribed to topic` confirmation for every selected stream. A missing
  subscription therefore blocks before activation reacquisition, watchdog
  arming, or command publisher creation instead of yielding another camera-only
  episode.
- `tools/delete_mcap.sh` provides explicit local space reclamation without a
  recursive-delete surface. It accepts exactly one resolved `.mcap` under a
  tabletop run's `raw_episode/bag/`, reports its size, and requires `DELETE`.

## 2026-08-14 — Tabletop planning now preserves arm IK branches

- Frozen run `tabletop_20260814T170502Z` did not prove that its retained
  GraspGenX candidates were unreachable. The old selection layer let CuRobo
  choose one final arm configuration for a grasp, then removed the complete
  grasp candidate when that configuration failed the closed-hand attached-cube
  lift. Other IK configurations for the same Cartesian grasp were never tested.
- The task planner now asks CuRobo for its finite collision-valid pregrasp IK
  branch pool. Each branch is evaluated through the existing joint-space route,
  straight Cartesian grasp approach, strict full-robot self/cube rechecks,
  infinite-plane hand guard, closed-Dex3 model, attached-cube lift, and payload
  guard. A candidate is removed only after every returned branch for it has
  failed. Collision geometry, table policies, lift distance, approach distance,
  tolerances, and the exact-reverse return remain unchanged.
- Branch failures are retained with candidate ID, pool/solver branch indices,
  exact failure stage, and concise physical reason. The selected solver branch,
  number of branches tested, and search round are stored in plan provenance.
- The unmodified production worker replayed the exact frozen request and found
  a complete route on the first preserved branch for
  `cube_head__seed_0000000119__sample_205`. The six trajectory sample counts are
  `61, 41, 41, 41, 41, 61`, and every serialized join is exact. Existing guards
  measured `+1.918 mm` open-route hand clearance and `+0.206 mm` closed-lift
  hand clearance. The conservative payload cover began at `-4.880 mm` relative
  to the inferred support plane and never deepened that initial contact proxy
  overlap. Plan SHA-256:
  `ad6c75c6d88584022eb978f12c9a744fd480bb58fcc2332f14aa58cc8869363f`.
  This was offline-only; no robot command was sent.

## 2026-08-14 — Conservative tabletop commissioning speed

- The successful frozen branch-search plan showed that every arm phase reached
  the former `0.200 rad/s` planner ceiling. Tabletop motion now carries a
  task-specific, hash-bound `maximum_arm_velocity_rad_s` value; the initial
  commissioning configuration sets it to `0.100 rad/s`.
- CuRobo path geometry is unchanged. Only the serialized timestamps are scaled,
  so the supported escape, pregrasp, grasp approach, payload lift, exact reverse
  replacement, retreat, and final supported return all obey the same limit.
  The executor still independently rejects any trajectory above the hardware
  controller's `0.200 rad/s` ceiling.
- Dex3 opening, closure, release, and restoration remain on the physically
  commissioned two-second smooth posture ramp. They are separate from the
  seven-joint arm trajectory and were not silently retuned.
- Offline replanning of the frozen `170502` geometry preserved candidate
  `cube_head__seed_0000000119__sample_205` and the complete route. The six task
  phases now last `16.178, 14.146, 13.526, 13.526, 14.146, 16.178 s`; each
  measured exactly `0.100000 rad/s` maximum serialized joint velocity. The
  supported escape and exact reverse each last `10.968 s` and also measure
  exactly `0.100000 rad/s`. Task-plan SHA-256:
  `b401ea3d5072513121ca1de9a5f398d84774688eabb4585471826e0ed90917ad`;
  supported-escape SHA-256:
  `fd0f32ea04bd2a4be28821b63f0e7d4633a20f2cb8190a51fd7704d431eb79db`.
  No robot command was sent.

## 2026-08-14 — MCAP launcher false negative removed

- Run `tabletop_20260814T180638Z` correctly refused ownership from zero-torque
  FSM 0 instead of the commissioned seated FSM 3. It sent no robot command and
  still finalized a complete 290 MB MCAP with all eight required topics, which
  independently proves that the account-local MCAP plugin is installed and
  functional.
- The next launch nevertheless printed `BrokenPipeError` followed by
  `account-local MCAP storage plugin was not discovered`. The precheck piped
  Python's `ros2 bag list storage` into `grep -q` while Bash `pipefail` was
  active. Once `grep` found `mcap`, it exited early; Python then wrote to the
  closed pipe, and `pipefail` converted the successful match into failure.
- The launcher now captures the complete plugin listing before applying the
  exact-line check. A read-only replay discovered `mcap`, `sqlite3`, and both
  ROS test plugins, and all four official Unitree message-type checks passed.
  No camera process, watchdog, publisher, or robot command was started.

## 2026-08-14 — CuRobo branch start-state isolation

- Run `tabletop_20260814T181113Z` passed perception and supported-escape
  planning, then rejected its first pregrasp IK branch for a real
  `1.096 mm` left-shoulder-yaw/torso sphere overlap. The second branch aborted
  at the exact serialized-start guard with a `0.969315350 rad` discontinuity.
- Offline instrumentation proved that CuRobo had mutated the `JointState`
  supplied to the rejected branch: the next attempt received the preceding
  branch's pregrasp configuration instead of the frozen clearance state. The
  warmed planner itself did not need to be rebuilt.
- Every branch attempt now receives a newly constructed seven-joint start
  state from the immutable serialized clearance reference. The frozen failed
  request proceeded through normal physical rejection of branches 1–4 and
  selected branch 5 of `cube_head__seed_0000000119__sample_205`, producing a
  complete pick/lift/replace/return plan with SHA-256
  `c438787ed85fef7a2cb4e97eabf86b982e4f36e28f5cab5af49424af83235f23`.
  This verification was offline-only; no robot command was sent.

## 2026-08-14 — Strict self-collision audit restored to the GPU

- The strict post-plan audit had been calculating linear penetration by moving
  all robot spheres and all `216,578` enabled sphere pairs to NumPy, then
  iterating every pair for every route sample in Python. The exact millimetre
  value is diagnostic only; route acceptance requires only whether any enabled
  pair overlaps.
- CuRobo's native CUDA self-collision kernel now identifies the sparse set of
  overlapping sphere pairs. Only actual hits are copied to Python for physical
  link names. Linear penetration is calculated for those hits alone, so the
  existing useful rejection text remains without participating in the
  collision-free critical path. Collision geometry and rejection policy are
  unchanged.
- Offline replay of retained real request `tabletop_20260814T183817Z`
  reproduced the same six rejected branches, link pairs, penetration values,
  selected candidate, and bit-identical joint trajectories. Complete task
  planning fell from the recorded `295.81 s` to `39.05 s`. The full account
  environment suite passes (`270 passed, 4 skipped`). No robot command was
  sent.

## 2026-08-14 — Supported escape begins at the exact acquired command

- Physical run `tabletop_20260814T183817Z` completed all planning but stopped
  before changing motion because CuRobo's float32-supported-escape sample zero
  differed from the loaded Unitree command by `1.8852615e-8 rad`. The executor's
  deliberately exact `1e-9 rad` trajectory-join contract correctly rejected
  it; its old six-decimal diagnostic misleadingly displayed `0.000000 rad`.
- The existing float32-boundary anchoring helper is now also applied to the
  supported escape, not only to later task phases. It first rejects a real
  discontinuity and only then replaces sample zero with the exact serialized
  loaded model and command coordinates. The executor tolerance was not relaxed.
- Offline replay of the retained loaded request produced a supported escape
  whose first command exactly equals the measured active-arm handoff (`0.0 rad`
  error). Future executor boundary failures are printed with nine decimals. No
  robot command was sent.

## 2026-08-15 — Retained MCAP isolates the failed physical closure

- Physical run `tabletop_20260814T190430Z` completed the supported escape,
  pregrasp route, and straight grasp approach, then stopped during left-Dex3
  closure. The complete 5.68 GB MCAP contains synchronized raw RGB, LowState,
  LowCmd, and both Dex3 state/command pairs. This analysis was offline-only; no
  robot connection or command was used.
- Five open-hand endpoint frames immediately before closure put the measured
  `object_T_G` only `3.61 mm / 2.99 deg` from selected GraspGenX candidate
  `cube_head__seed_0000000119__sample_205`. The measured seven arm joints were
  all within `0.0291 rad` of the frozen grasp-approach command. The camera/FK
  placement therefore did not miss the grasp by anything comparable to the
  reported `0.5003 rad` finger error.
- Six valid frames at the stalled closed-hand endpoint put measured
  `object_T_G` `4.80 mm / 4.07 deg` from the candidate's initial grasp frame.
  From the pre-closure burst to that endpoint, the physical cube moved only
  `1.48 mm / 0.91 deg` in the camera observation. The retained images visibly
  show the cube between the thumb and opposing finger at the timeout.
- The selected PhysX evidence tells a materially different closure story. Its
  cube moved `14.87 mm / 14.52 deg` during closure, within the shortlist's very
  permissive `20 mm / 45 deg` retention gates. Its final
  `isaac_closed_object_T_G` is `31.87 mm / 14.52 deg` from the candidate's
  initial `object_T_G`, and `33.23 mm / 10.49 deg` from the physical stalled
  endpoint. The production controller then required the exact
  `isaac_closed_q` from that displaced simulated cube state while the real
  table-supported cube had barely shifted.
- The stalled joint is exactly the planned opposing contact chain after the
  commissioned right-to-left mirror: left `middle_0` measured `-0.0618 rad`
  against the mirrored simulated target `-0.5620 rad`; thumb and the other
  finger largely reached their targets. There were no Dex3 hardware error bits.
  This is evidence of real object contact followed by an inapplicable exact
  simulated-posture endpoint, not a missing hand command or a gross arm/camera
  placement miss.
- The other retained grasps do not repair this automatically. All 15 qualified
  candidates rely on `7.50-19.80 mm` and `5.7-42.8 deg` of simulated cube
  movement during closure. The least-moving candidate changes
  `object_T_G` by `12.80 mm / 5.7 deg`. For this physical cube placement,
  CuRobo found collision-valid pregrasp IK branches for only the selected
  candidate and one alternative; the alternative has still worse simulated
  closure motion (`19.54 mm / 36.5 deg`, `66.57 mm` relative-frame shift).
- Conclusion: accepting the `0.5003 rad` residual as an endpoint tolerance
  would hide a mismatch in the grasp qualification/execution contract. The
  next design must either qualify grasps with near-static tabletop closure or
  terminate physical closure on validated contact/retention evidence instead
  of requiring every finger to reproduce an exact post-PhysX joint vector.

## 2026-08-15 — Dex3 pressure evidence corrected and applied to the failed grasp

- The local `/home/kanth042/dex3_pressure_tools` repository contains physical
  right-hand evidence from Dex3-1 hand `214-R-T`; the pressure fields are not
  arbitrary or generally unusable. Its untouched audit identified exactly 33
  active taxels and 75 slots fixed at the `30000` invalid sentinel. Active
  baseline noise had a 99th-percentile absolute drift of only `16-24` raw
  counts. Deliberate free touches produced repeatable multi-sample responses
  up to `13,800` counts, with 22 of 33 active taxels exceeding the repo's
  conservative `500`-count definite-touch threshold.
- Applying that repository's unchanged validity and baseline rules to the
  retained left-hand MCAP from `tabletop_20260814T190430Z` recovered the same
  33 active slots, the same 75 sentinel slots, and the same `16-24`-count idle
  band. This independently confirms that the recorded left pressure matrix was
  decoded correctly; the message's cumulative `lost` field is not a reason to
  discard these samples.
- The physical closure itself produced no sustained tactile response. During
  both the closing and fully stalled windows, no active taxel remained even
  `50` counts above its local baseline. One approximately one-sample outlier at
  `t=106.242038 s` appeared at group 0, cell 9 (`+297,016` counts), then
  immediately disappeared. A dwell or median contact test must reject such an
  isolated spike.
- Therefore the earlier broad statement that Dex3 pressure is unreliable was
  wrong. The narrower result is that this particular cube grasp stalled a
  finger without loading a mapped tactile surface. It may have contacted a
  nonsensing edge or shell surface; the retained data cannot distinguish that
  from a left-hand spatial-map mismatch because the local pressure study
  physically mapped only the right hand.
- For this grasp family, finger position stall is the available primary
  closure evidence and pressure can only corroborate it when a sustained
  taxel response is present. Making pressure a primary retention test first
  requires a short left-hand taxel mapping/validation and grasp contacts that
  deliberately land on the validated finger pads. Camera visibility is then
  optional during a small retention lift rather than a prerequisite.

## 2026-08-15 — Contact-stall and small-lift retention contract

- The production close command now ramps toward the selected qualified Dex3
  posture but no longer requires the exact post-PhysX endpoint. It accepts only
  after at least one commanded finger has visibly moved in the closing
  direction by the existing `0.01 rad` stability band and at least one finger
  remains more than the existing `0.08 rad` endpoint tolerance short of its
  empty-hand target while the complete hand settles within `0.01 rad` for the
  commissioned `0.5 s` dwell.
- No finger is required to reach the empty-hand target. That earlier proposed
  gate was removed because object contact can validly block every closing
  finger; observed commanded-direction motion is the direct evidence that the
  hand command executed. A test covers this all-fingers-stalled case.
- Raw Dex3 velocity, effort, and pressure do not decide the live result. Both
  official Dex3 state topics, including their pressure matrices, were already
  part of the plain MCAP contract and remain available for offline analysis;
  no duplicate pressure recorder was added.
- CuRobo still produces one complete payload lift and exact reverse. The plan
  is split at its first sample at least `10 mm` above contact, preserving every
  original joint sample and duplicating only the exact shared boundary. After
  physical closure, an isolated read-only worker checks that same frozen route
  once with the measured stalled finger angles. It does not replan the arm or
  introduce a new geometric tolerance.
- After the small test lift, the blocked joints must remain within the existing
  `0.01 rad` stability band of the contact posture and remain short of the
  empty-hand target. Only then does the unchanged controller continue the full
  configured lift. CPU tests cover stable contact, empty-hand closure, contact
  loss, exact trajectory splitting, and the new hash-bound planner contracts.
- Offline CuRobo verification reused retained physical run
  `tabletop_20260814T190430Z`; it generated all eight task phases, split the
  requested `10 mm` boundary at the unchanged trajectory's `10.158 mm` sample,
  and checked all 81 payload-route samples with the final measured left-Dex3
  posture from the MCAP (`middle_0=-0.0580 rad`). The measured-posture route
  passed strict self-collision and table-plane checks; its minimum hand-plane
  clearance was `1.552 mm` at `left_hand_middle_1_link`. No robot connection or
  command was used for this verification.
- Repository verification after the change: `276 passed, 4 skipped`; Ruff
  formatting and lint checks both pass.

## 2026-08-15 — One planner session and explicit task-rejection return

- The tabletop runtime previously started a fresh Python/CUDA process for the
  supported escape, complete task, and post-contact route check. It also chose
  finger actions by trajectory list index. Both were orchestration defects,
  not CuRobo requirements.
- A single isolated planner now starts and initializes CUDA before SPACE. After
  loaded settling it receives one request that freezes the supported escape,
  eight task phases, exact supported return, and two exact-reverse rejection
  edges. The controller addresses every action by phase name; no `index == 2`
  or equivalent task semantics remain.
- The planner retains a 14-coordinate selected-arm-plus-Dex3 collision checker.
  The seven measured contact angles are inserted into the unchanged frozen arm
  payload route after closure. No IK, trajectory optimization, process startup,
  or robot interface participates in that check.
- Offline replay of retained real run `tabletop_20260814T190430Z` produced a
  complete lifecycle in `30.96 s` on the laptop GPU, down from `43.96 s` before
  reusing strict route kinematics. The retained physical contact posture checked
  all 81 payload samples in `0.010 s` wall time after a one-time `1.81 s` cache
  build, with the same `1.552 mm` minimum hand-plane clearance as the earlier
  standalone validation. No robot connection or command was used.
- No stable grasp, measured-contact route rejection, and contact loss during
  the 10 mm test are now explicit task outcomes. The first two open at contact
  and reverse the frozen grasp/approach routes. Test-lift loss first follows the
  exact frozen 10 mm reverse, then opens and returns. Both restore seated FSM 3.
  State freshness, transport, fixed-rate controller, or unexpected Dex3 hold
  faults—and Ctrl+C—still retain the PC2 zero-torque path.
- Repository verification after the refactor: `278 passed, 4 skipped`; Ruff
  formatting and lint checks both pass.

## 2026-08-15 — Opt-in h50 tripod presentation

- Pulled `sri299792458/g1-aprilcube-demo` through merge commit `25748ec`; its
  source change `e9b4c1f` adds exact 40/50/60 mm printed presenter meshes and
  fixture-conditioned right-Dex3 proposal sets. The h50 STL SHA-256 is
  `2da7c59130b78a777fdb85b1aef5d3291adccb990bf7a75ca32bb30066eaade2`.
- The fixture is an opt-in presentation, not a second task controller. The
  existing `run-tabletop` command gains `--presentation tripod-h50`; the
  omitted/default value is `direct`, whose request and CuRobo scene contain no
  fixture. All observation, safety, planning-session, execution, retention,
  recovery, recording, and seated-restoration code remains shared.
- The h50 request hash-binds the fixture ID, repository-relative mesh path,
  mesh hash, millimetre-to-metre scale, 50 mm support height, and the explicit
  `centred_and_yaw_aligned` cube contract. The observed cube fixes the fixture
  pose. Only tripod mode moves the inferred table plane from the cube bottom
  to the actual table 50 mm below it.
- CuRobo receives the exact mesh for supported escape, open approach, and
  attached-payload planning. After physical contact, the retained 14-joint
  validator inserts the measured finger posture into the unchanged payload
  arm route and additionally checks all resulting robot spheres against the
  exact transformed presenter mesh. Direct mode skips mesh loading and that
  recheck.
- Remote commit `4325bc9` adds the previously omitted source pool with each
  candidate's exact `closed_before_tug` PhysX joint state. The runtime
  shortlist joins all 372 h50-qualified candidates to that evidence by
  candidate ID, source index, content hash, and pose. These achieved contact
  states remain qualification evidence only; the physical close command comes
  from the single fixed Dex3 descriptor profile. All 372 are source-qualified
  for the common 70 mm runtime approach; the source also records 365 at 100 mm
  and 353 at 150 mm. CuRobo receives the complete 372-candidate set as one goal
  set.
- Validation was local only; no robot command was issued. The full repository
  suite passes (`282 passed, 4 skipped`), Ruff formatting/lint and diff checks
  pass, the exact-mesh CPU query reports the expected `-70..-20 mm` vertical
  bounds, and an offline CUDA/CuRobo construction accepted the combined cube,
  table patch, and exact tripod mesh scene from a retained robot snapshot with
  goal-set capacity 372.

## 2026-08-15 — Single-face planar PnP branch correction

- Retained run `tabletop_20260815T160830Z` failed its unchanged 5 mm cube-pose
  spread check even though its five loaded-observation images show a stationary
  object. The selected observation was tag 1 on face `-X` in every frame, with
  a 61.7--62.0 px short side.
- Exact offline replay isolated one bad PnP branch in `frame_000`: the existing
  cold-start SQPnP result was about 19.8 mm and 61 degrees from the other four
  frames and had 1.242 px RMS reprojection error. Generic planar IPPE returned
  both mathematical solutions; its other branch had 0.107 px RMS error and
  joined the stationary cluster. This was neither object motion nor a reason
  to relax the tabletop quality gate.
- The correction is centralized in the vendored AprilCube pose estimator. Only
  a cold-start observation with exactly four coplanar object points uses
  generic `SOLVEPNP_IPPE`; all returned solutions are required to keep every
  object point at positive camera depth, and the lower-reprojection solution is
  selected before the existing LM refinement. Prior-seeded, multi-point, and
  nonplanar solves keep their existing paths. `SOLVEPNP_IPPE_SQUARE` is not
  used because the cube-face points are expressed in the cube frame rather than
  OpenCV's required centered square convention.
- A numerical AprilCube regression test stores only the four detected image
  corners, four known object corners, and camera matrix from the failed frame;
  it does not depend on retaining the large run. Replaying all five saved images
  through the unchanged tabletop observation code now measures 1.585 mm
  translation spread and 0.472 degrees rotation spread, below the existing
  5 mm and 2 degree limits. No robot command was issued for this diagnosis or
  fix.

## 2026-08-15 — Grasp pose and finger-close contracts separated

- Reading the GraspGen-X paper and NVIDIA's released end-to-end code confirmed
  the intended interface: inference returns an SE(3) grasp pose and score for a
  gripper with one predefined open-to-close motion. A candidate-specific final
  finger posture is not a model output.
- Our shortlist's `isaac_closed_q` is the measured result after simulated
  contact. It is useful qualification evidence, but the initial tabletop code
  incorrectly promoted it into the hardware close command. The fixed descriptor
  close target was loaded in the first implementation and then left unused.
- The canonical Dex3 profile now carries the exact fixed close target used by
  the qualified descriptor. Both the provisional attached-payload model and
  physical controller use that target for every grasp. Finite gains allow
  contact to limit each physical joint independently; the resulting measured
  posture still undergoes the existing strict route recheck before any lift.
- Executable plans now name the field `close_target_active_dex3_q_rad`. Plan
  construction and deserialization reject open or close values that differ from
  the hash-bound descriptor profile. No runtime Python source reads
  `isaac_closed_q`; the retained shortlist data remain unchanged for offline
  qualification analysis.
- This is a reusable interface lesson: generated intent, simulated outcome, and
  hardware command must have separate types and names even when all three are
  stored in one artifact pipeline.
- Offline verification after the change: 282 tests passed and 4
  hardware/environment tests were skipped; Ruff formatting, Ruff linting, JSON
  parsing, and Git whitespace checks all passed. No robot command was sent.

## 2026-08-15 — Robot ownership now precedes ROS teardown

- Retained run `tabletop_20260815T162444Z` reached the exact final handoff
  command and held it for 1.120 seconds. The laptop command stream then paused
  for 209.2 ms and the controller rejected its 206 ms-old cached LowState
  against the unchanged 100 ms freshness limit. The independent MCAP recorder
  saw only an 18.5 ms maximum LowState gap, proving the robot/network stream
  itself remained healthy.
- The failure path returned from the complete frozen rejection route and then
  destroyed the camera node and called `rclpy.shutdown()` before its outer
  handler restored seated control. That ROS participant teardown temporarily
  starved the Python control thread and its Unitree callback while direct
  lowcmd ownership was still active. Earlier failed runs contain the same
  approximately 199--210 ms terminal command gap, but their primary errors
  hid this secondary lifecycle fault.
- ROS teardown now occurs only in the outer resource cleanup, after either the
  normal verified seated takeover or the failure-path verified PC2 zero-torque
  takeover. Planner shutdown was moved behind the same ownership boundary.
  A runtime invariant refuses camera/node/ROS teardown whenever a direct-control
  transport still requires takeover and the executor has not reached `STOPPED`.
- The 100 ms state-freshness, 250 ms controller-gap, and 500 ms PC2-watchdog
  limits remain unchanged. Regression tests prove ROS resources are untouched
  before takeover and close in order after takeover. Offline verification:
  284 tests passed and 4 hardware/environment tests were skipped; Ruff and Git
  whitespace checks passed. No robot command was sent.

## 2026-08-15 — Post-lift stall is remeasured instead of frozen at first contact

- Source review corrected the naming: the fixed descriptor close is NVIDIA
  GR00T-VisualSim2Real's explicit Dex3 `close` profile. It is distinct from
  GR00T-WholeBodyControl's `middle_close`, which its VLA runner uses as a
  generic closed-hand posture. The tabletop runtime retains the
  VisualSim2Real profile used by the GraspGenX/PhysX qualification; no new
  close target was introduced.
- The prior retention check incorrectly treated the first stable contact angle
  as immutable. It rejected any initially blocked motor that advanced more
  than the `0.01 rad` settling-band value, even though the unchanged fixed
  close target remains commanded and a loaded grasp can settle farther during
  the test lift.
- The controller now continues publishing the exact same fixed close target
  throughout the test lift and makes no contact decision from a single moving
  sample. At the lifted endpoint it collects a new stable window using the
  unchanged `0.08 rad` endpoint tolerance, `0.01 rad` peak-to-peak stability
  band, `0.5 s` dwell, and existing timeout. At least one closing joint must
  remain short of the empty-hand target. The final blocked-motor set may differ
  from the initial set; stable complete empty-hand closure is a failed grasp.
- Retention evidence now records the post-lift joint vector, post-lift blocked
  IDs and names, residuals to the fixed close target, peak-to-peak settling
  spread, and maximum shift from the initial contact posture. That shift is
  diagnostic rather than a pass/fail threshold. Raw velocity, effort, and
  pressure remain diagnostic only.
- No additional finger-sweep or post-lift collision pass was added. The
  existing provisional fixed-close CuRobo planning and measured first-contact
  route validation remain unchanged. Offline verification passes with
  `285 passed, 4 skipped`; Ruff and Git whitespace checks pass. No robot
  command was sent.

## 2026-08-16 — Bounded waist-yaw planning study

- Commit `96f11d2` is the local checkpoint immediately before this study. It
  adds the rolling CuRobo MPC command buffer, persistent worker seam, 250 Hz
  executor interpolation, and pelvis/torso IMU observation plumbing. The
  checkpoint passed `315` tests with `4` skipped; the user-owned dirty
  `third_party/aprilcube` submodule was deliberately excluded.
- The existing tabletop planner and hardware executor are seven-arm-joint
  contracts. The complete 29-joint lowcmd transport holds waist yaw at its
  exact takeover value. Waist yaw was therefore added first as a read-only
  planning study, not hidden inside a seven-joint executable trajectory.
- `g1-curobo-worker analyze-waist-yaw` constructs NVIDIA's complete G1/Dex3
  model with `waist_yaw_joint + selected seven arm joints` active. It compares
  the locked baseline with explicit start-relative yaw bounds supplied on the
  command line. Each result uses CuRobo's collision-constrained pregrasp IK and
  then independently rechecks strict full-robot self collision, cube
  collision, and the selected wrist/hand table-plane guard. Artifacts state
  `commands_robot: false` and cannot be consumed by the hardware executor.
- Simply exposing the URDF range is invalid for seated manipulation. On retained
  run `tabletop_20260815T224443Z`, unconstrained CuRobo solutions changed waist
  yaw by as much as `2.599 rad` (`148.9 deg`). This was useful for detecting the
  missing policy but is not retained as an executable option.
- The exact grasp selected by each of four retained runs was compared at locked,
  `+/-0.1`, `+/-0.2`, `+/-0.3`, and `+/-0.5 rad` yaw ranges. The two older
  requests were normalized into ignored `work/` artifacts because their stored
  hashes predate the current request schema; production hash checks were not
  weakened.

  | Retained run | Locked best peak arm move | Best bounded result | Peak reduction | Arm L2 change |
  | --- | ---: | ---: | ---: | ---: |
  | `20260815T224443Z` | `0.718 rad` | `0.724 rad` at `+/-0.1 rad` | none | `0.4%` lower |
  | `20260815T162444Z` | `0.835 rad` | `0.753 rad` at `+/-0.1 rad` | `9.8%` | `10.5%` higher |
  | `20260814T190430Z` | `0.898 rad` | `0.766 rad` at `+/-0.1 rad` | `14.7%` | `1.4%` lower |
  | `20260814T183817Z` | `1.017 rad` | `0.918 rad` at `+/-0.2 rad` | `9.7%` | `4.0%` lower |

- The effect is useful but not monotonic. Three of four grasps admit a smaller
  peak arm excursion, but the newest grasp gains nothing, one case trades peak
  reduction for greater total arm motion, and `+/-0.3`/`+/-0.5 rad` frequently
  add strict cube-collision branches. Waist yaw should therefore be treated as
  lightly used redundancy with an explicit cost, not as free reach.
- This study proves only pregrasp IK feasibility. Before waist yaw can command
  hardware, the serialized trajectory must explicitly name eight coordinates,
  the complete-body lowcmd target must interpolate waist yaw in the existing
  fixed-rate process, full routes must be revalidated, and a deliberately small
  seated waist-motion commissioning test must pass. No robot command was sent.
- Verification after the study: planner-environment focused tests `10 passed`;
  control-environment repository tests `317 passed, 6 skipped`; Ruff and Git
  whitespace checks pass.

## 2026-08-16 — Complete phase-aware CuRobo MPC lifecycle

- The initial MPC checkpoint replaced only `clearance -> move_to_pregrasp`.
  That was not a complete tabletop controller because the cube changes from a
  world obstacle, to intended finger contact, to an attached payload, and back
  to a world obstacle after release.
- The persistent worker distinguishes supported, open-free, open-contact, and
  attached-payload modes across all ten normal lifecycle routes. It retains
  one warmed controller per physical mode: the reverse open-contact,
  open-free, and supported legs reuse the exact models built outbound instead
  of destroying and reconstructing them. The attached model is still created
  only after the measured physical finger stall is known, and its four
  connected payload motions share that one instance.
- No finger or task logic moved into MPC. The existing descriptor-defined open
  and close commands, measured contact stall, retained-route validation,
  post-lift retention evidence, release, initial-finger restoration, and
  seated takeover remain unchanged. Task-rejection recovery deliberately uses
  the frozen reverse trajectories.
- Attached phases build the same deterministic conservative 3 x 3 x 3 cube
  sphere cover used by the frozen planner, expressed in the selected grasp
  frame. The actual first-contact Dex3 posture is locked into the runtime MPC
  model; it is not replaced by a simulated candidate posture.
- Every returned window is independently rechecked before the controller can
  accept it: strict full-robot self collision; the existing non-contact-link
  cube activation-distance policy; selected wrist/hand table plane; payload
  plane relative to the existing start allowance; and optional exact fixture
  mesh. Contact-link exceptions exactly follow the frozen grasp approach.
- The first implementation repeated FK three times per window. Complete
  latency was about 72--75 ms on average with rare 102--108 ms windows, too
  close to the unchanged 100 ms state-age contract. The checker now computes
  FK once and reuses those spheres for all checks; cuboid distances were
  vectorized without changing the collision threshold or geometry.
- Exact command-free replay used retained run `tabletop_20260815T224443Z`.
  After physical-mode caching and removal of duplicate FK-only model builds,
  all ten phases completed with 239 accepted windows, zero rejected windows,
  and a maximum endpoint error of `0.004800 rad`. Complete computation fell
  from the original `56.81 s` to `36.80 s`. Rolling solves consumed `11.73 s`
  and overlap physical motion; the four cold models required `22.84 s` of
  construction and CUDA setup. The largest individual window took `87.2 ms`.
  All three reverse modes were cache hits, and the four-model set fit on the
  24 GB GPU.
- Initial lifecycle planning also used full MotionPlanner instances for two FK
  reads before immediately destroying them. Those reads now use the strict
  kinematics/collision checkers already required by route validation. A cold
  like-for-like replay retained the same grasp candidate and reduced planning
  from `28.88 s` to `25.04 s`; IK, trajectory optimization, collision policy,
  and returned paths are unchanged.
- `run-tabletop --motion-controller mpc` selects this full normal-motion path.
  Default trajectory execution remains available. MPC runs store one
  `mpc_lifecycle.json` with preparation and every validated window grouped by
  phase. The artifact is flushed after robot ownership is resolved even for a
  failed run, so live control performs no file writes and the rejected window
  is still retained.
- This is nominal route-tracking MPC, not yet visual/table-state correction.
  The experimental camera anchoring path and hardware waist-yaw command remain
  disabled. No robot command was sent while implementing or benchmarking this
  change.
- Final offline verification: control environment `325 passed, 6 skipped`;
  planner-environment focused suite `44 passed`; all changed Python files pass
  Ruff lint/format checks and `git diff --check` passes. The user-owned dirty
  `third_party/aprilcube` submodule was not modified.

## 2026-08-17 — One warmed CuRobo solver for the complete lifecycle

- Profiling showed the dominant avoidable MPC cost was not Dex3 collision
  checking or per-window validation. It was constructing and CUDA-warming four
  solver instances for supported, open-free, open-contact, and attached modes:
  `22.84 s` before the first complete lifecycle was ready.
- The worker now constructs one fixed-shape solver. It preallocates the union
  scene and reserved payload spheres, then switches physical phases by copying
  resolved kinematic/collision tensor values in place and toggling existing
  obstacles. CUDA graph addresses, active seven-joint shape, and collision
  cache capacity remain fixed.
- Initial and open Dex3 postures are resolved before setup. The actual
  contact-stalled posture is necessarily unknown until grasp closure; its
  folded fixed transforms are resolved once (`1.82 s` in the retained replay)
  and reused through all attached-payload phases.
- One action seed is retained per physical mode. This matters on the reverse
  path: the first implementation discarded the prior open-contact seed and a
  different optimized retreat put a thumb proxy `0.291 mm` below the strict
  table plane. The strict checker rejected it. Restoring the prior mode seed
  recovered a valid return without restoring multiple solver/CUDA instances.
- An independent GPU-tensor audit rebuilt the former mode-specific models and
  compared each against the switched solver. Fixed transforms, joint maps,
  locked joints, collision-pair tensors, self-collision padding, and attached
  payload matched exactly. Open-hand collision-sphere radii matched within
  `3.73e-9 m` float32 roundoff. This audit also corrected misleading prose:
  CuRobo uses one optimizer sphere set for world and self collision; the
  independent strict full-robot post-check is what enforces complete geometry
  before a window can enter the controller.
- Command-free replay of retained run `tabletop_20260815T224443Z` completed all
  ten phases with `237` accepted windows, zero rejected windows, maximum
  endpoint error `0.004950 rad`, and maximum velocity below `0.1 rad/s`.
  Final preparation fell from `22.84 s` to `14.19 s` (`37.9%`); complete
  benchmark wall time fell from `36.80 s` to `27.75 s` (`24.6%`). Rolling solve
  work was essentially unchanged at `11.55 s`.
- An intermediate single-solver build produced cold first windows as high as
  `94.6 ms`, leaving too little transport margin under the unchanged `100 ms`
  source-state age gate. Cold solves now run during phase preparation while the
  existing controller holds the robot and before any live state timestamp is
  sampled. The final lifecycle's largest live window was `64.1 ms`; no timeout
  or collision threshold was relaxed.
- No robot command was sent. The benchmark and tensor audit are offline uses of
  retained request and execution-plan artifacts.
- Final verification: control environment `326 passed, 6 skipped`; planner
  environment `325 passed, 1 skipped`; changed Python files pass Ruff lint and
  format checks; `git diff --check` passes. The user-owned dirty
  `third_party/aprilcube` submodule remains untouched.

## 2026-08-17 — Initial lifecycle planning profile and resolved-model reuse

- A cold, command-free profile used retained run
  `tabletop_20260815T224443Z`. The prior artifact spent `9.419 s` planning the
  supported escape and `13.507 s` planning the task: `22.926 s` before the
  measured-contact checker. The assumption that all of this was irreducible
  live planning was incorrect.
- Resolving NVIDIA's complete G1/Dex3 kinematic tree and 766 collision spheres
  took roughly `1.8 s` per warm CUDA model (and longer for the process's first
  CUDA model). Constructing a MotionPlanner around an already-resolved
  `RobotCfg` took only about `0.04--0.06 s`. The strict checker and optimizer
  were parsing identical kinematic trees independently.
- The strict model is now resolved once per physical finger posture. The
  optimizer receives an independent clone of its tensors, with its one moving
  grasp frame and exact open-transit sphere policy applied in place. Collision
  pairs, padding, tool maps, joint influence maps, and all 766 sphere values
  were compared with a separately parsed former optimizer model and matched
  exactly. The strict and optimizer tensors have distinct storage, so temporary
  contact-link changes cannot weaken the strict checker.
- The retained IK pool's first two branches ended in strict torso collisions:
  `11.834 mm` at the left wrist and `39.312 mm` at the left elbow. Previously
  CuRobo spent several seconds planning each permissive route before the strict
  post-check rejected it. All IK endpoints are now strict-checked in one
  batched FK call (`~1.1 ms`). An invalid terminal configuration is rejected
  before trajectory optimization; valid branches remain in their original
  order and still receive every existing route check.
- The same grasp candidate and rejection ordering were retained. The resolved
  model clone alone reproduced every old trajectory sample exactly. With the
  endpoint check enabled, CuRobo generates a different but fully validated
  redundant-arm path to the same selected grasp because the two doomed solver
  calls no longer advance its optimizer state.
- Final cold replays varied with first-CUDA initialization but consistently
  reduced the task stage to about `9.0 s`; one complete profiled lifecycle was
  `19.96 s` including the `1.80 s` measured-contact checker, versus about
  `24.78 s` for the retained baseline plus that checker. No tolerance,
  collision policy, candidate pool, or planning attempt limit was relaxed.
- The process's first model resolution included about `3.7 s` of one-time
  CuRobo/CUDA initialization. The persistent worker now performs one
  command-free generic G1/Dex3 model warmup before it reports `ready`, while no
  command publisher exists and before the operator presses Space. A retained
  replay then resolved the actual live-posture model in `1.75 s` instead of
  `5.40 s`; post-approval lifecycle plus contact-checker time was `16.37 s`.
  The live model is still freshly resolved from the measured joints and Dex3
  state—the dummy warmup supplies no kinematic result to planning.
- The new plan artifacts include compact per-stage timings. After these fixes,
  the genuine remaining work is approximately `8.1 s` of pose/c-space planning
  plus goal-set IK and approximately `7.3 s` of live-posture model resolution
  in the profiled initial stages. The largest individual solves are the
  supported escape and attached-payload lift; Python collision validation is
  only milliseconds and is not worth further optimization.
- No robot command was sent. Retained-run planning and all tensor comparisons
  were performed offline.

## 2026-08-17 — Fixed-cube clearance-boundary replan and physical table margin

- The rigid-chair measurements established that the camera/table relationship
  changes by about `5.3 mm` between the supported loaded state and lifted arm
  state. The production task now uses the existing visible AprilCube as a
  task-local fixed anchor: it is observed after loaded ownership and again
  while the exact supported-escape endpoint is held. The second observation
  replaces the camera/object transform and all measured locked-body/finger
  coordinates. The selected arm coordinate remains the exact serialized escape
  endpoint, preserving command continuity.
- Before the first changing target, CuRobo now freezes only the 100 mm supported
  escape and its exact reverse. This is the complete recovery contract needed
  to reach the stationary observation boundary. After the boundary burst, the
  same persistent worker plans the only grasp task eligible for execution and
  atomically replaces the executor's remaining pose set and plan hash. If the
  observation, replan, or boundary install fails while the controller remains
  healthy, the arm follows the already-frozen exact reverse to handoff and the
  task stops. No task plan from the known-stale loaded scene can execute.
- A merely positive wrist/hand-to-table distance is no longer accepted.
  `task.yaml` hash-binds a `5 mm` minimum, matching the existing commissioned
  calibration route validator's general collision-clearance policy. Frozen
  open routes, provisional closed-hand payload routes, measured-contact route
  validation, and non-supported MPC windows enforce that same floor. The
  supported escape retains its start-relative policy, and the resting/just-
  attached cube remains allowed to contact its supporting plane.
- A command-free replay of retained physical request
  `tabletop_20260815T224443Z` proved why both changes are needed. Its only
  otherwise reachable grasp branches had `0.9 mm` minimum clearance at
  `left_hand_thumb_2_link`; all were rejected against the new `5.0 mm` floor.
  The other IK branches retained their prior genuine torso collisions. This is
  the same sub-millimetre route class that preceded the physical table strike;
  it is no longer labeled execution-ready. Requiring that stale grasp plan
  before the supported lift would prevent reaching the measurement boundary,
  so the reversible escape and boundary task are deliberately separate
  planning transactions.
- Run artifacts now retain `loaded_observation/`, `clearance_observation/`,
  `loaded_request.json`, `initial_clearance_request.json`, the final
  `clearance_request.json`, `supported_escape.json`, `execution_plan.json`, and
  `cube_anchor_motion.json`. The latter reports inferred camera translation and
  rotation in the fixed cube frame. Moving the cube between observations
  violates this explicit contract; the code does not relabel object motion as
  robot state.
- The production worker generated the escape-only artifact from that retained
  physical request without ROS or robot commands: terminal G-frame clearance
  `149.0 mm`, plan SHA-256
  `0255cc7f001322c580a9bbe79bbfc3294e561e0d4e8edb5d296933b3d9098422`.
  Final verification is `334 passed, 6 skipped` in the control environment and
  `333 passed, 1 skipped` in the CUDA planner environment; Ruff and Git
  whitespace checks pass. No robot command was sent.

## 2026-08-17 — Independent minimal camera-state observer

- Promoted the retained-data winner into `AnchoredCameraStateEstimator`, a
  pure numerical component with no ROS, CuRobo, MPC, or robot-command
  dependency. It stores one synchronized full-pose visual anchor and exposes
  the hybrid pelvis-position/torso-orientation prediction as a timestamped
  `reference_T_camera` estimate.
- Narrowed the estimator input from the complete 29-joint vector to exactly
  `waist_yaw_joint`, `waist_roll_joint`, and `waist_pitch_joint`, plus the
  pelvis and torso orientation matrices. The pelvis-to-camera URDF chain does
  not contain an arm, hand, or leg joint. MCAP replay now extracts only those
  three measured joints for estimation.
- Removed the prior unused latest-sample pairing helper. It assigned the later
  receipt timestamp to two unsynchronized messages; a future live adapter must
  perform explicit interpolation or bounded pairing instead.
- Did not add depth, raw D435i IMU, commands, arms, hands, or legs to the
  estimator. Depth independently validates table-normal motion, but no retained
  replay yet proves that adding it improves the hybrid estimate. It remains a
  separate recorded research measurement.
- Did not fabricate covariance. The estimate reports its visual-anchor age and
  fixed-pelvis-IMU-origin assumption. Covariance and freshness policy belong to
  a later, separately validated integration contract.
- The strict continuous rigid-chair replay reproduced every prior numerical
  camera error exactly: a single global anchor gives `4.661 mm` mean error;
  realized `1.335 s` visual resets give `0.371 mm` mean / `0.873 mm` p95, and
  realized `2.002 s` resets give `0.427 mm` mean / `0.993 mm` p95. No robot
  command was sent.

## 2026-08-17 — Stationary pregrasp state correction

- Connected the independent hybrid observer to the default tabletop trajectory
  workflow at exactly one stationary boundary. The clearance cube observation
  is the visual anchor. At reached pregrasp, synchronized waist yaw/roll/pitch,
  pelvis IMU, and torso IMU measurements propagate the fixed-object camera
  pose; no depth, arm state, hand marker, or second pregrasp image was added to
  the estimator.
- The propagated pose is not written into `TabletopObservation`. A separate
  hash-bound `EstimatedCameraPlanningState` records the original observation
  hash, anchor/current timestamps and pairing diagnostics, exact robot snapshot,
  estimator name, fixed-pelvis-IMU-origin assumption, and `object_T_camera`.
- The persistent CuRobo worker preserves the already selected grasp candidate
  and replans every remaining motion from the exact active pregrasp command.
  It may search alternate IK branches only for that same grasp. The remapped
  lifecycle returns through the corrected pregrasp, then exactly reverses the
  original clearance-to-pregrasp path and the original supported escape.
- Before atomic installation, a fresh propagated estimate must remain within
  the already configured `5 mm / 2 deg` task perception limits of the pose used
  for planning. Executor joint-state stability and exact command-continuity
  checks remain independent. Any estimator, planning, or installation rejection
  returns over the frozen old route and stops the task.
- The optional MPC path remains unchanged and explicitly reports that it does
  not consume this correction. No continuous trajectory deformation or new
  controller was introduced.
- Both complete test environments pass: `342 passed, 6 skipped` in `.venv` and
  `341 passed, 1 skipped` in `.venv-planner`; Ruff and whitespace checks pass.
  A command-free retained CUDA exercise preserved the selected grasp and built
  all nine remapped remaining trajectories. That retained run used its historic
  sub-5-mm margin and therefore is not evidence for the current 5 mm physical
  policy. Re-running the same artifact under the current margin correctly
  rejected it at `0.9 mm` thumb/table clearance. No robot command was sent.

## 2026-08-17 — Removed duplicate trajectory lifecycle planning

- The default trajectory workflow previously planned a complete
  grasp/lift/replace/return lifecycle at clearance, executed only its first
  `clearance -> pregrasp` edge, and then planned the same complete lifecycle
  again after the stationary camera-state correction. That duplicate payload
  planning was an architectural error, not an inherent CuRobo cost.
- Clearance planning now returns a dedicated hash-bound `TabletopPregraspPlan`.
  It serializes exactly one executable route to pregrasp and its exact reverse.
  Candidate selection still plans and strictly validates the unexecuted linear
  grasp approach, including full-robot self collision, cube/fixture geometry,
  and the 5 mm table margin; an obviously invalid grasp is therefore not
  selected merely to make the first stage faster.
- Only after the arm reaches stationary pregrasp does CuRobo plan the corrected
  grasp, retention lift, payload lift, replacement, retreat, return through the
  old pregrasp, and exact pregrasp-to-clearance reverse. A failed estimator,
  plan, or installation still has the pre-existing pregrasp reverse plus the
  supported-escape reverse. The MPC path retains its existing full-lifecycle
  transaction and was not silently changed.
- The persistent worker now owns one `ReusableOpenPlanner`. The first boundary
  allocates its fixed-shape MotionPlanner and CUDA graphs with the full grasp
  goal-set capacity. At pregrasp, a freshly resolved strict robot remains the
  source of truth; the worker requires identical active/locked joint names,
  tool frames, tensor shapes, and self-collision pair topology before copying
  new kinematic values into the retained optimizer and calling CuRobo's public
  world-update API. A topology or seed/capacity change rebuilds instead.
- A command-free CUDA test changed both the waist-pitch witness and object pose
  between requests. Reusing the fixed-shape optimizer reduced open setup from
  about `0.18 s` to `0.002 s`, goal-set IK from about `1.3 s` to `0.003 s`, and
  the already-warmed open route stage from about `2.4 s` to `1.2 s` in the
  corrected complete plan. With the retained run's historical 0.5 mm table
  policy solely to permit like-for-like timing, the split stages took
  `10.41 s` cold plus `6.72 s`; the first stage's `5.36 s` first-process model
  cost is moved before SPACE by the existing worker warmup and is about `1.8 s`
  there. The current 5 mm policy correctly rejects that historical scene and
  was not relaxed in production.
- No robot command was sent. These are retained-data CUDA planning tests only.
- Final regression results are `343 passed, 6 skipped` in the control
  environment and `342 passed, 1 skipped` in the CUDA planner environment;
  changed Python files pass Ruff and `git diff --check` passes. The pre-existing
  dirty `third_party/aprilcube` submodule was not modified.
## 2026-08-17 — Supported escape validates the requested displacement

- Removed the unrelated requirement that the terminal grasp frame be at least
  `50 mm` above the observed table plane. The escape now checks its FK endpoint
  against the actual requested Cartesian target using the same `5 mm` position
  tolerance that CuRobo uses to declare the trajectory successful.
- The plan provenance and terminal output record requested lift, achieved lift,
  endpoint error, and CuRobo tolerance. The independent sampled wrist/hand
  table-plane guard and strict self-collision validation are unchanged. No
  robot command was sent for this change.

## 2026-08-17 — Clearance plan replacement includes its reached start

- Physical run `tabletop_20260817T210626Z` completed the exact 100 mm supported
  escape and found a valid left-arm pregrasp, then rejected the replacement
  before pregrasp motion because its pose set omitted the already-reached
  `clearance` boundary. The frozen reverse completed and seated control was
  restored.
- Both trajectory and MPC clearance replacements now include the exact escape
  endpoint as `clearance`. The later pregrasp correction already included its
  reached `move_to_pregrasp` boundary. The common builder now rejects every
  non-handoff plan that omits or misnames its current boundary, preventing the
  same error at future replacement sites. No robot command was sent.

## 2026-08-17 — Visual-anchor state is saved before planning

- Run `tabletop_20260817T211231Z` completed the exact supported lift and found
  a valid pregrasp. The clearance image's synchronized LowState/torso-IMU
  sample was then unavailable because the approximately 4.8-second solve
  produced 5,000 samples while the live buffer retained only 4,096.
- The workflow now saves the synchronized camera-state anchor immediately
  after the clearance image, before invoking CuRobo. All file writes and
  planning finish before the controller switches to the new path, so a failure
  still uses the already-approved escape reverse. If that reverse itself ever
  fails, the status retains both the original and recovery errors instead of
  hiding the first one. No robot command was sent for this fix.

## 2026-08-17 — Fixed-close table sweep is checked before pregrasp

- Physical run `tabletop_20260817T211927Z` reached the selected left-arm
  pregrasp and then rejected the corrected remainder because every IK branch
  put `left_hand_index_1_link` `3.6-3.8 mm` below the inferred table during
  the provisional fixed-close lift, against the unchanged `5 mm` margin. The
  frozen pregrasp and supported-escape reverses completed and seated control
  was restored; no close or lift was attempted.
- The simulated per-candidate `isaac_closed_q` remains qualification evidence
  only. Prior physical data established that those angles came from simulations
  where the cube moved substantially during closure and do not represent the
  table-supported hardware contact posture. The one descriptor close target
  and the measured-contact route validation are unchanged.
- Candidate selection now evaluates the complete open-to-fixed-close finger
  sweep at the exact planned grasp contact arm pose before it serializes or
  executes a route to pregrasp. The existing 20 mrad maximum joint sampling
  step is shared with Dex3 preparation. Every sample receives the strict
  full-robot self-collision check, the selected wrist/hand `5 mm` table-plane
  check, and the exact fixture-mesh check when a presenter is active.
- Command-free CUDA replay of the retained request rejected the previously
  selected grasp at `-3.8 mm` during this new early check. The other returned
  IK branches retained their genuine wrist/elbow-to-torso collisions, so that
  saved scene has no executable alternative; production will now stop at
  clearance and reverse instead of first moving to a doomed pregrasp.
- After real closure, the existing validator still inserts the measured
  contact-stalled finger angles into every frozen payload-route sample before
  even the `10 mm` retention lift. No table margin was relaxed and no robot
  command was sent for this change.

## 2026-08-17 — Requalified the existing 40 mm cube for the real close command

- The prior 15-grasp direct-table shortlist used the cube pose and finger
  endpoint after free-object Isaac closure. That evidence did not answer the
  hardware question, where the tabletop cube is stationary and every grasp
  receives the same fixed Dex3 descriptor close command.
- Rechecked all 3,178 retained GraspGenX poses without changing any
  `object_T_G`. Five pass the stationary-cube contract: open hand clear of the
  cube, thumb and an opposing finger reach the fixed cube during the commanded
  close, and the exact Dex3 collision meshes remain at least `5 mm` above the
  table at open, pregrasp, and all 51 close-sweep samples. The tightest passing
  clearance is `5.161 mm`; the first candidate has `6.053 mm`.
- Direct-table planning uses that hash-bound exact-mesh hand/table evidence.
  CuRobo remains responsible for IK, arm routes, strict self-collision, cube
  collision, and wrist clearance. After physical closure, the pre-existing
  measured-contact route check still runs before the `10 mm` retention lift.
- A command-free replay of retained scene `tabletop_20260817T215708Z` selected
  unchanged candidate `cube_head__seed_0000000079__sample_213` on its second IK
  branch and planned the complete pick/lift/replace/return lifecycle in about
  16 seconds. Open-route clearance was `9.313 mm`; exact fixed-close clearance
  was `6.053 mm`; the configured requirement stayed `5 mm`. No robot command
  was sent.

## 2026-08-17 — Each retained grasp now receives an independent IK search

- Replaced the shared five-pose CuRobo goal set with one IK-only GPU batch:
  five independent rows, one goal per row, and 16 seeds per grasp. A shared
  goal set had only 16 seeds total and could concentrate all returned solutions
  on one candidate; it was not a complete five-candidate search.
- The batch does not change trajectory planning. Its IK solver is destroyed
  after copying out the finite joint-solution pool. The existing single-route
  MotionPlanner then tests each solution with the existing strict endpoint,
  route, fixed-close, payload, and return checks. This supersedes the earlier
  note that the reusable MotionPlanner itself has full goal-set capacity; that
  planner is deliberately single-goal again.
- Command-free replay of retained physical scene
  `tabletop_20260817T215708Z` produced 15 unique collision-valid IK solutions
  for candidate `cube_head__seed_0000000079__sample_213` and zero for each of
  the other four candidates. The strict full-robot endpoint check rejected the
  first three solutions for named torso collisions. Solution 4 completed the
  full pick/lift/replace/return lifecycle. Total planner time was `16.26 s`;
  batched-IK construction was `0.062 s` and its solve was `1.384 s`.
- The resulting route retained `9.313 mm` minimum open-route table clearance
  and `6.053 mm` minimum exact fixed-close clearance against the unchanged
  `5 mm` requirement. A second command-free test exercised the actual split
  boundary workflow: pregrasp selection took `12.29 s`, the later task kept
  candidate 213, reused the single-route planner, and produced all eight phases
  in `8.89 s`. Final regression results are `350 passed, 6 skipped` in the
  control environment and `349 passed, 1 skipped` in the CUDA planner
  environment; Ruff passes. No robot command was sent.

## 2026-08-17 — First complete physical lift exposed a missing return leg

- Physical run `tabletop_20260817T230114Z` successfully acquired the left-hand
  grasp, passed the measured stalled-finger route check with `6.525 mm` minimum
  hand/table clearance, completed the retention test, lifted the 40 mm cube,
  lowered it, and replaced it on the table.
- The failure after replacement was a control-sequencing bug, not an IK,
  collision, grasp, retention, or payload-planning failure. The pregrasp-
  corrected plan contains `grasp_retreat -> return_to_pregrasp ->
  return_to_clearance`. Hardware execution performed `grasp_retreat` and then
  tried to start the second return edge directly. The executor correctly
  rejected the source-pose mismatch, after which fail-closed cleanup produced
  the observed zero-torque state.
- Normal completion and both task-rejection returns now execute the optional
  `return_to_pregrasp` edge before `return_to_clearance`. Unsplit plans retain
  their original direct return. Source-pose errors now print both the required
  and current logical pose IDs. No threshold, planned trajectory, collision
  policy, or robot-control gain changed.

## 2026-08-18 — Grasp evidence rebuilt around opposed Dex3 tactile contact

- The stall-only contract was removed. Retained run analysis had already shown
  that an empty Dex3 hand can stop with the same fixed-close joint residual as
  the failed cube attempt, so joint residual alone cannot classify a grasp.
  `tau_est` and raw `dq` remain recorded diagnostics; neither has a physically
  commissioned object-contact model in this repository.
- The implementation copies the validated signal contract from the local
  `dex3_pressure_tools` study: the `30000 +/- 1000` invalid-slot rule, exactly
  33 active taxels, per-run median unloaded baseline, p99 idle-noise audit, and
  a fixed minimum contact rise of 50 raw counts. The baseline is collected for
  `0.5 s` from at least 20 fresh active-hand samples while the hand is open at
  pregrasp. A noisy or differently mapped hand is rejected; the runtime never
  raises the contact threshold until noise appears valid.
- Live closure now requires one stable `0.5 s` window containing all of:
  commanded-direction finger motion, at least one `>0.08 rad` residual to the
  fixed empty-close target, complete-hand spread `<=0.01 rad`, and simultaneous
  above-baseline loading on a thumb group and a middle/index group. Thumb-only
  or palm-only pressure, an isolated spike, pressure without closure residual,
  and residual without opposed pressure all fail.
- The fixed close command remains active during the existing planned retention
  lift. At its lifted endpoint, a fresh `0.5 s` window must again contain both
  opposed pressure and a close residual. Fingers and loaded taxels may migrate
  as the cube settles; the post-lift blocked-motor set, taxels, pressure peaks,
  and joint shift are recorded instead of being required to match first
  contact exactly.
- Official left/right Dex3 states now expose the complete `9 x 12` pressure
  matrix and `tau_est` in diagnostic snapshots. The MCAP contract is unchanged
  because it already records both official state topics. New artifacts are
  `tactile_baseline.json`, `grasp_contact.json`, and the updated
  `retention_evidence.json`; inactive pressure slots serialize as JSON `null`,
  never non-standard NaN.
- A baseline rejection occurs at pregrasp, not at grasp contact. Both complete
  and pregrasp-corrected execution contracts therefore include hash-bound exact
  reverse routes from their actual pregrasp states. This prevents the rejection
  handler from executing a recovery whose source pose the robot never reached.
- Command-free verification: `357 passed, 6 skipped` in the control environment;
  the affected CUDA planner/contract set passes `56` tests; Ruff and Git diff
  checks pass. No robot command was sent.

## 2026-08-18 — Retention is a 30 mm checkpoint inside the normal lift

- Opposed tactile contact at table height proves loading between the thumb and
  an opposing finger, but not that the cube is supported independently of the
  table. A lifted retention check therefore remains necessary.
- There is no separate test-lift plan or extra planner call. CuRobo still
  produces one 100 mm payload lift. The runtime now splits that unchanged
  trajectory at its first sample at least `30 mm` above contact, pauses for the
  fresh tactile/residual window, then either continues the suffix or exactly
  reverses the prefix.
- The prior 10 mm split was too close to the table relative to physical
  tracking and table-pose uncertainty. The 30 mm boundary gives the 40 mm cube
  materially clearer table separation while keeping a rejected-grasp drop
  much lower than the full lift. Collision checks, speed, controller gains,
  tactile thresholds, and the complete 100 mm payload target are unchanged.

## 2026-08-18 — The 60 mm R3 cube is a hash-bound runtime object profile

- A controlled size study in `sri299792458/g1-aprilcube-demo` retained the
  released Dex3 descriptor, fixed close command, 3 mm edge radius, 5 mm table
  requirement, and all GraspGenX/Isaac settings. The 40, 50, and 60 mm cubes
  produced 5, 5, and 57 stationary-cube fixed-close-qualified grasps,
  respectively. The 60 mm pool began with 3,279 intrinsic-retention passes;
  none of the 57 admitted `object_T_G` poses was changed.
- The audit was rerun with exact minimum-link and close-sweep-sample
  provenance. Its 57 clearances span `5.008--15.988 mm`; runtime keeps the
  complete shortlist and lets the existing independent batched IK plus strict
  route validators choose against the live robot and cube pose.
- `--object-profile cube60-r3` now selects one hash-bound bundle: exact 60 mm
  R3 qualification mesh, 45 mm `DICT_4X4_100` detector with IDs 10--15, and
  the 57-candidate direct-table shortlist. The existing `cube40-r3` profile
  remains the default. Object selection changes no controller gain, speed,
  safety rule, calibration transform, state estimator, or recording path.
- The printable production mesh and the qualification mesh have identical
  60 mm bounds and signed volume; their bidirectional vertex-to-surface
  difference is below `4e-15 mm`. Their triangulation differs only because the
  production target preserves the released cube's 75 percent marker-to-face
  ratio. The runtime imports only the compact shortlist and exact qualified
  mesh, not the roughly 183 MB experiment directory.
- The 50 mm tripod presentation remains explicitly qualified only for
  `cube40-r3`; selecting it with the 60 mm profile fails before robot ownership.
  No robot command was sent while adding or validating the 60 mm profile.

## 2026-08-18 — Tripod search is GPU-pruned before route optimization

- Retained physical run `tabletop_20260818T125007Z` timed out after 180 seconds
  despite producing 1,095 collision-valid pregrasp IK branches. The timeout did
  not prove the scene was infeasible. An exact Trimesh signed-distance query,
  originally written for one final measured-contact validation, had been moved
  into the per-branch fixed-close loop. Some rejected branches consequently
  spent about 20 seconds in CPU mesh queries after CuRobo had already loaded
  the same exact tripod mesh on CUDA.
- The CPU fixture query was removed from candidate search, selected-route
  validation, measured-contact validation, and MPC checks. One reusable CuRobo
  world checker now performs exact hash-bound mesh queries on CUDA. A deliberate
  sphere/mesh penetration probe verified that the query detects contact; after
  warmup, retained route queries take sub-millisecond time.
- Before arm IK, the planner now expresses the complete 51-sample fixed Dex3
  close sweep in the grasp frame, transforms it across every candidate, and
  checks hand/table and hand/tripod geometry in bounded CUDA batches. It then
  checks all arm IK endpoints for strict full-robot self collision in one GPU
  call and ranks only the survivors by distance from the live arm state.
  Trajectory optimization is never invoked for either rejected set.
- The replay also exposed a separate physical modeling error. After attachment,
  the cube begins in deliberate contact with the tripod that supports it, so
  treating the attached cube/tripod pair as a collision makes every upward lift
  start invalid. The attached-lift optimizer now excludes only that one support
  pair. The resulting closed-hand route is independently checked against the
  exact tripod mesh without payload spheres, preserving all robot/tripod
  collision rules. Replacement is the exact reverse of the validated lift.
- Command-free replay of the exact retained request pruned 220 of 372 grasp
  candidates before arm IK and pruned 234 IK endpoints in one strict GPU pass.
  Five trajectory branches were attempted. Candidate
  `cube_head__seed_0000000079__sample_230` produced the complete
  pick/lift/replace/return lifecycle in `18.88 s`; fixed-close candidate pruning
  itself took `0.014 s`. The selected-candidate-only complete plan took
  `16.53 s`. The former 180-second timeout is gone without changing the 5 mm
  table margin, collision geometry, controller gains, or motion speed.
- Final command-free verification is `359 passed, 6 skipped` in the control
  environment and `61 passed` for the affected CUDA planner set. Ruff and Git
  diff checks pass. No robot command was sent.

## 2026-08-18 — Supported pressure is provisional; retention is decided after separation

- Physical run `tabletop_20260818T145822Z` completed supported escape, grasp
  selection, corrected pregrasp planning, approach, and the descriptor close.
  The close held its final target for about `0.615 s`. Four fingers retained
  `0.173--0.250 rad` residuals and the cube remained mechanically trapped.
- During the stable close window, opposing-finger pressure was `184--208` raw
  counts above baseline while the thumb peak remained `24--48`, just below the
  unchanged `50`-count contact threshold. The old table-height gate therefore
  rejected the close and immediately opened the hand. MCAP frames show the cube
  still held before that command, beginning to tilt during opening, and falling
  only after the fingers separated. The frozen reverse and seated restoration
  otherwise completed without cleanup errors.
- This run does not prove that the cube was supported against gravity; it may
  have been wedged against the tripod. That is precisely the question answered
  by the existing low `30 mm` separation checkpoint. Simultaneous opposed
  pressure is therefore no longer a pre-lift requirement. A stable commanded
  close with at least one `>0.08 rad` residual is recorded as
  `grasp_close.json`, collision-checked at its measured finger angles, and
  admitted to the unchanged low lift.
- The fixed close target remains commanded throughout the low lift. The hard
  decision remains at the lifted checkpoint, which still requires a fresh
  stable residual and simultaneous thumb/opposing-finger pressure. Failure
  follows the exact low-lift reverse while the hand remains closed and opens
  only after returning to support. No tactile threshold, arm trajectory,
  collision margin, control gain, or motion speed changed.
- Command-free verification after this change: control environment `359 passed,
  6 skipped`; planner/CUDA environment `358 passed, 1 skipped`; full Ruff lint,
  touched-file format checks, repository diff checks, and the read-only local
  hardware inspection all passed. No robot command was sent.

## 2026-08-18 — Empty-close commissioning replaces pressure as grasp evidence

- The preceding pressure-based policies are superseded. A standalone left-hand
  empty-close commission measured the exact descriptor command at
  `[-0.022203, 0.571732, 0.978123, -0.878780, -0.975255, -0.883415,
  -0.972809] rad`; its worst descriptor-target tracking residual was only
  `0.026668 rad`. The normal task does not repeat this open/close cycle.
- A grasp now requires a stable shortfall of at least `0.05 rad` from that
  commissioned empty close on both mechanical sides: at least one thumb
  closing joint and at least one middle/index closing joint. Closing direction
  is derived from the live open-to-descriptor command per motor; it is not
  assumed to have one sign across the hand. The same two-sided test is repeated
  after the existing 30 mm retention lift.
- Retained physical evidence separates the cases cleanly. Successful lift run
  `tabletop_20260817T230114Z` had maximum thumb/opposing shortfalls
  `0.155609/0.394296 rad`; missed-cube run `tabletop_20260817T231230Z` had
  `0.000925/0.196066 rad`; visually caged run
  `tabletop_20260818T153501Z` had `0.155474/0.156364 rad`. The former and latter
  pass; the missed cube fails because its thumb reached the empty posture.
- Raw pressure, `tau_est`, and velocity continue to be captured in the official
  Dex3 state topics in MCAP, but no live branch reads them. The pressure gate
  was disproved by the latest caged grasp: no mapped taxel crossed the rule even
  though the object moved with the hand. `grasp_close.json` and
  `retention_evidence.json` now contain the exact commissioned reference,
  closing directions, per-joint shortfalls, threshold, and selected motor IDs.
- Only the left empty-close reference is commissioned. Selecting the right arm
  fails during read-only setup and instructs the operator to run and record the
  standalone right-hand measurement first; left data are never mirrored or
  invented.
- Command-free verification is `360 passed, 6 skipped` in the control
  environment and `359 passed, 1 skipped` in the planner environment; full
  Ruff lint, format, and repository diff checks pass. No robot command was
  sent.

## 2026-08-18 — Physical tripod runs exposed a transit-only thumb exception bug

- Operator-observed run `tabletop_20260818T162801Z` lifted the cube and
  completed every commanded reverse/release/return phase. The commissioned
  empty-close test passed strongly at support and after the 30 mm checkpoint:
  maximum thumb/opposing shortfalls were `0.23454/0.24631 rad`. The cube did
  not balance exactly back on the three small tripod contacts. The current
  `completed` result proves the commanded motion lifecycle and seated handback;
  it does not contain a post-release visual placement claim.
- Run `tabletop_20260818T163107Z` rejected before task planning and exactly
  reversed the supported escape because the five-frame camera-to-cube pose
  burst spanned `21.477 mm` against the unchanged `5 mm` stationary-boundary
  limit. Offline replay isolated this as a CLAHE-induced visual outlier, not
  object motion: one accepted frame misplaced a marker corner by about 13 px
  and differed from the two good accepted frames by `30.649-33.817 mm`; two
  other frames were rejected at `3.411/3.434 px` reprojection error. The torso
  changed by at most about `0.011 deg` during the burst. Raw-grayscale replay
  detected the larger face in all five frames with `0.145 mm / 0.221 deg`
  spread.
- During run `tabletop_20260818T163252Z`, the operator saw the left thumb strike
  the cube on clearance-to-pregrasp. The subsequent close correctly failed the
  new evidence test at `-0.0009/0.0113 rad` thumb/opposing shortfall and the
  frozen reverse completed. Read-only replay found the exact nominal event:
  `left_hand_thumb_2_link` was only `9.784 mm` from the cube at pregrasp-route
  sample 14. Measured arm tracking reduced the modeled clearance to
  `8.357 mm`; the worst joint error at that sample was `0.02070 rad`. The
  anchored estimator measured another `0.910 mm / 0.281 deg` of body/camera
  change over the complete clearance-to-pregrasp leg.
- Root cause was not a missing cube obstacle. The strict post-planning checker
  used the intentional thumb/middle/index contact-tip exemption over the whole
  combined route. That exemption belongs only to the final straight grasp
  approach. Clearance-to-pregrasp now checks every hand link against the cube;
  the contact-tip exemption begins only after pregrasp.
- Replaying the third request now rejects candidate 204 and ten other
  thumb-skimming IK branches at the existing `<10 mm` activation band, then
  selects candidate 157 with `25.958 mm` nominal minimum transit clearance.
  Replaying the first request retains its physically successful candidate 213.
  No collision threshold, sphere radius, grasp pose, controller gain, or robot
  speed changed. Verification is `361 passed, 6 skipped` in the control
  environment and `360 passed, 1 skipped` in the planner environment; Ruff and
  diff checks pass. No robot command was sent during diagnosis or correction.
- The tabletop D435i object detector now explicitly disables the AprilCube
  library's general-purpose CLAHE default. Across all retained tabletop data,
  raw grayscale produced `326/326` valid frame poses and passed all 65 complete
  bursts; CLAHE produced `324/326` valid frame poses and the one false burst
  rejection above. The AprilCube library default, pose gates, and thresholds
  remain unchanged.

## 2026-08-18 — Current tripod MPC lifecycle replay

- The trajectory pipeline was checkpointed first as commit `4bb74cc`; the
  pre-existing dirty `third_party/aprilcube` submodule remained excluded.
- A command-free replay rebuilt the nominal full lifecycle from successful
  left-arm tripod run `tabletop_20260818T173706Z`. It selected the same
  `cube_head__seed_0000000129__sample_218` grasp and consumed the retained
  physical `grasp_close.json` for attached-payload kinematics.
- The first replay exposed two MPC/frozen-planner mismatches. Supported escape
  and return were being modeled with the later clearance snapshot even though
  these contact motions already have an exact validated frozen route. They are
  now both kept outside MPC. Attached mode also checked the cube's intentional
  tripod support contact as a collision. It now mirrors the frozen payload
  planner: the fixture is absent from attached optimization, while the strict
  exact-mesh check retains every robot sphere and excludes only the attached
  object proxy. No table, collision, or speed threshold changed.
- The corrected replay completed all eight MPC-controlled phases through `179`
  accepted windows with zero rejections. Maximum endpoint error was
  `0.004289 rad`; maximum velocity was `0.098770 rad/s` under the unchanged
  `0.1 rad/s` limit.
- The remaining timing failure was isolated to CuRobo optimization, not strict
  validation: exact checks cost about `2.4 ms`, while 100 warm iterations made
  open-contact windows reach `102 ms` before IPC. CuRobo requires multiples of
  25 iterations. A 50-iteration replay generated a `0.105691 rad/s` window and
  was rejected. Three complete 75-iteration replays had zero rejections and a
  worst window of `87.277 ms`; 75 is therefore the lowest tested valid value,
  with the `100 ms` source-state age limit unchanged.
- The official benchmark now accepts a retained `--grasp-close` artifact and
  hash-records it. The final artifact is ignored at
  `work/mpc_replay_20260818T173706Z/mpc_benchmark_final.json`. No robot command
  was sent.

## 2026-08-18 — Continuous MPC body correction and measured latency fix

- The clearance cube observation is now the fixed six-dimensional task
  anchor for the MPC path. Every rolling request receipt-pairs a fresh arm
  state with the existing pelvis/waist/torso estimator, uses the older source
  timestamp for the unchanged 100 ms freshness rule, and hash-binds the exact
  `reference_T_camera` estimate into the returned command window.
- The worker moves both the local Cartesian goal and the cube/table/fixture
  scene from the frozen body frame into the estimated live body frame. Strict
  robot spheres are transformed back into the nominal scene frame before the
  existing full self/world checks. The final clearance endpoint remains the
  exact body-relative joint junction with the frozen supported return.
- Pose-only continuous CuRobo updates were tested and rejected. The redundant
  seven-joint arm lost its validated branch, used 245 windows for the first
  leg, and failed the cube check at grasp retreat. A nominal-joint posture
  regularizer was faster but missed the cube activation band by 0.132 mm.
  Production therefore retains CuRobo local retargeting IK seeded at the
  matching frozen-route waypoint and requires the existing 0.005 rad corrected
  endpoint agreement in addition to the existing Cartesian tolerances.
- A temporary experiment reduced only that auxiliary IK iteration count; it
  was reverted because iteration tuning was not the root cause. The 75 MPC
  iterations, 0.1 rad/s speed, 100 ms freshness, 10 mm activation distance,
  and every collision geometry remain unchanged.
- Component profiling showed about 8--9 ms for body/goal correction and about
  3 ms for the independent strict checks. Attached phases optimized in
  31--34 ms. The slow open phases spent up to 98 ms inside optimization because
  the tripod triangle mesh was queried at every one of 75 iterations. On the
  identical pregrasp route, removing only the fixture from the iterative
  optimizer reduced optimizer mean/max from 44.2/69.6 ms to 31.4/35.2 ms;
  removing the table made no measurable difference and removing the cube only
  a small difference.
- The rolling optimizer now follows the already fixture-validated frozen route
  with cube/table costs, while the independent CUDA checker remains the exact
  fixture authority for every returned window. Open-hand phases enforce the
  same 10 mm activation margin in that exact check. Attached phases retain the
  pre-existing no-penetration policy because the cube deliberately rests on
  the tripod. Exact-clearance replay measured every open phase at least 10 mm
  away; attached retention/replacement reached 4.455/3.270 mm without
  penetration.
- Profiling also exposed two cold/correctness issues. The live-scene sphere
  transform kernel had not been exercised during setup, causing one 80 ms
  first-use spike; setup now warms the same path. When fixture-excluded contact
  or payload links were present, strict fixture checking accidentally cloned
  untransformed spheres; it now consistently uses the corrected nominal-scene
  spheres before disabling only the named links.
- Three complete final-code corrected eight-phase replays passed with 181
  accepted and zero rejected windows each. Worst complete windows were 57.72,
  51.72, and 53.16 ms, leaving at least 42 ms before the unchanged 100 ms
  source-age limit. Every open phase retained at least 10 mm exact fixture
  clearance; attached pickup/replacement reached 4.455/3.270 mm without
  penetration. These are command-free retained-run results, not physical MPC
  commissioning. No robot command was sent.

## 2026-08-18 — MPC warm iterations raised to 100

- The configured rolling CuRobo solve was changed from 75 to 100 warm
  iterations at operator request. Speed remains 0.1 rad/s, state freshness
  remains 100 ms, and exact fixture checking still validates every returned
  window outside the iterative optimizer. A complete command-free retained-run
  replay passed all eight phases in 161 accepted windows with zero rejections;
  its worst complete window was 50.82 ms, maximum velocity was 0.088084 rad/s,
  and every open phase retained at least 10 mm exact fixture clearance. The
  preceding 75-iteration timing measurements remain historical evidence rather
  than the current runtime bound. No robot command was sent.

## 2026-08-18 — Prime tower replaces the tripod presentation

- The `tripod-h50` presentation and all three packaged tripod artifacts were
  removed. There is no compatibility alias. The new opt-in interface is
  `--presentation prime-tower`, and direct-table behavior remains the default.
- The operator-supplied `prime_tower_collision.stl` is one watertight,
  axis-aligned 35.42 x 34.75 x 60.00 mm mesh, centered in X/Y with its base at
  Z=0. Its SHA-256 is
  `2e1409589b2a4cb394b8ac89756f8e9739e2c8ebbe2324815258f8cc7eb2ad56`.
  The shared placement contract fixes its base to the table and centers either
  cube, yaw-aligned, on the 60 mm top.
- Profile-specific grasp lists sit behind the one presentation argument. The
  existing full intrinsic-retention pools were filtered without changing any
  `object_T_G`: 113/3,178 candidates for `cube40-r3` and 312/3,279 for
  `cube60-r3` passed. Qualification reused the existing GraspGenX exact Dex3
  geometry/FCL code, the actual fixed descriptor close, a 70 mm open approach,
  51 close samples, the exact prime-tower mesh, and the table mesh. It did not
  reuse candidate-specific Isaac closing joints or the old tripod shortlist.
- Presentation loading now resolves and hash-binds the shortlist selected by
  the object profile. The selected shortlist is also frozen across the
  read-only preflight; previously the preflight byte check watched the direct
  object shortlist even when a fixture-specific shortlist was selected.
- `run-tabletop` now requires an explicit `--object-profile` on every run.
  `--object-profile cube40-r3 --presentation prime-tower` and
  `--object-profile cube60-r3 --presentation prime-tower` select the two
  independent contracts without inferring cube geometry from presentation.
- Full control tests passed `370 passed, 6 skipped`; the full planner
  environment passed `369 passed, 1 skipped`. No robot command was sent.

## 2026-08-18 — Wrist-heavy pregrasp selection retained for later study

- Successful physical left-arm direct-table run `tabletop_20260818T234114Z`
  selected `cube_head__seed_0000000169__sample_217`. Its clearance-to-pregrasp
  route changed left wrist roll by `+128.86 deg`, wrist pitch by `-35.31 deg`,
  and wrist yaw by `+65.08 deg`. FK confirms this was not a left/right Dex3
  adapter error: the selected grasp required a `158.28 deg` physical palm
  reorientation, and the planned endpoint matched that target within
  `0.42 deg`.
- The custom wrapper currently orders all strict-endpoint-valid IK branches by
  unweighted Euclidean joint displacement from the measured start and accepts
  the first branch whose route passes. This rule was introduced locally in
  commit `4bb74cc`; it is not a CuRobo or GraspGenX selection policy.
- Command-free replay of the exact hash-matching request tested two simple
  alternatives without changing production code. Minimizing maximum
  normalized joint travel selected a first-attempt route with `97.78 deg`
  wrist roll but shifted `119.32 deg` into shoulder yaw. Minimizing maximum
  absolute joint travel selected a second-attempt route with `106.77 deg`
  wrist roll and `109.71 deg` maximum travel. Neither removes the large
  reorientation, so the working selection policy remains unchanged pending a
  broader comparison across arms, object sizes, and presentations. No robot
  command was sent during this analysis.

## 2026-08-19 — MPC cleanup and measured/command continuity

- Physical MPC run `tabletop_20260819T002307Z` failed before motion because
  its first future sample was planned from the measured arm while sample zero
  was the different active position command. The largest retained tracking
  offset was `0.02009 rad`; replacing only sample zero still produced a
  `0.12382 rad/s` command edge against the unchanged `0.1 rad/s` limit.
- The interim command boundary carried that measured-to-command offset into the
  returned window but removed it linearly by the endpoint. Both the predicted
  measured path and translated command path retained independent hard-limit and
  collision validation. This first-window repair was later superseded: it did
  not preserve desired-state continuity when gravity/load tracking error
  persisted across every receding-horizon update. See the mature-controller
  audit below.
- The subsequent one-window, continuous route-projection, and direct-endpoint
  experiments were removed. Production is restored to the previously proven
  monotonic frozen-route lookup, one-action-horizon joint lookahead, and three
  continuously replenished interpolation windows. Warm optimization remains
  100 iterations.
- A command-free replay using the exact retained active command, measured arm
  state, and live camera correction made the first window feasible: the
  uncorrected edge was `0.12558 rad/s`, the translated command peak was
  `0.05219 rad/s`, and the strict checker passed. Its later windows used the
  benchmark's ideal `measured == command` plant, however, so its full
  eight-phase completion did **not** validate persistent tracking offset. The
  temporary result is `/tmp/mpc_clean_route_full.json`; no robot command was
  sent.
- The controller continues to accept a generic `reference_T_camera` estimate
  rather than depending on an AprilCube detector. The current producer uses
  the fixed cube as its anchor. Future moving-cube work will replace that
  producer with a fixed table marker and supply the cube separately as the
  changing task target; it must not alter the command buffer or safety checks.
- Full tests passed `373 passed, 6 skipped` in the control environment and
  `372 passed, 1 skipped` in the planner environment. Ruff passed.

## 2026-08-19 — Open-transit object margin changed to 5 mm

- The operator requested a 5 mm hard hand-to-object margin for open transit.
  CuRobo's separate 10 mm collision-cost activation distance remains
  unchanged, as do self-collision, table, fixture, contact-phase, and payload
  policies. MPC records both values independently.
- The exact route diagnostic now reports the closest point over the complete
  route rather than the first sample entering the rejection band. This exposed
  that the earlier `8.996-9.870 mm` messages from run
  `tabletop_20260819T115257Z` were first-entry values, not route minima.
- The first command-free GPU replay under the new policy still found no
  acceptable route: the optimizer had planned against the bare cube, and its
  12 endpoint-valid branches reached true nominal minima of
  `3.278-4.391 mm`, all at `left_hand_palm_link/cube`. The exact checker was
  correct to reject them.
- This experiment padded only the cube by 5 mm on every face while the
  independent checker retained the physical cube and exact 5 mm distance
  test. It was removed later the same day; see "Removed cube-specific optimizer
  enlargement" below. Replaying the same retained request had selected candidate
  `cube_head__seed_0000000129__sample_206`, solver branch 3, and the complete
  pick/lift/replace/return planner passed. CuRobo's 10 mm cost activation,
  every non-object collision policy, and all physical geometry remain
  unchanged. No robot command was sent.
- Full verification passed `375 passed, 6 skipped` in the control environment
  and `374 passed, 1 skipped` in the planner environment. Ruff and repository
  diff checks passed.

## 2026-08-19 — Mature rolling-controller audit and persistent desired-state continuity

### Why the previous fix was wrong

- Physical run `tabletop_20260819T121046Z` accepted all 54/54 feasible MPC
  windows but never reached the first phase endpoint. Route progress advanced
  to index 6 and then remained there for 94 simulated/diagnostic updates. The
  controller was repeatedly fading the live `active command - measured state`
  offset to zero inside every short window. With a persistent gravity/load
  offset, each replan therefore removed the same low-level position-loop
  effort again. Cadence, iteration count, endpoint tolerance, and collision
  margins do not repair that controller-boundary error.
- The earlier full command-free replay was insufficient because only its first
  window used the retained physical offset; its ideal plant then assigned the
  command directly to measured state. An isolated policy replay with the
  retained offset held persistently reproduced the stall: the linear-fade
  policy remained at route index 6 for 94 of 100 feasible windows. Holding the
  offset across each short window reached that isolated phase terminal in 35
  windows, with a `0.08145 rad/s` maximum command speed. That comparison
  established the continuity bug; it was not a complete production lifecycle
  replay.

### Source-backed controller patterns checked

- CuRobo's maintainer describes MPC output as a trajectory to be interpolated
  and tracked by a separate low-level controller, rather than as a new measured
  position target that should erase the controller's existing tracking lead:
  <https://github.com/NVlabs/curobo/discussions/681>.
- ROS 2 `joint_trajectory_controller` has an explicit
  `interpolate_from_desired_state` policy for successive trajectory messages.
  Its documentation calls out MPC-like applications and preserving continuity
  from the currently desired state instead of restarting each update from the
  lagging measured state:
  <https://github.com/ros-controls/ros2_controllers/blob/master/joint_trajectory_controller/src/joint_trajectory_controller_parameters.yaml>.
- MoveIt Servo similarly continues from the buffered future command while it
  is valid rather than discarding that command state at every update:
  <https://github.com/moveit/moveit2/blob/main/moveit_ros/moveit_servo/src/servo_node.cpp>.
- NVIDIA STORM separates the desired `q/dq/ddq` trajectory from the measured
  state and uses a model-plus-PD tracking controller on the robot:
  <https://github.com/NVlabs/storm> and
  <https://github.com/mohakbhardwaj/franka_motion_control>. That validates the
  separation of planner and tracking state, but transplanting STORM's torque
  controller is **not** appropriate here because the commissioned Unitree
  lowcmd PD and gravity-feedforward path already fills that role.
- Unitree XR teleoperation likewise sends a desired arm position plus gravity
  torque and bounds target changes relative to measured joints; it does not
  deliberately collapse the desired/measured separation at the end of every
  receding horizon:
  <https://github.com/unitreerobotics/xr_teleoperate/wiki/Motion>.

### Decision and scope

- Preserve the live desired-state offset for the complete short CuRobo window:
  `bias = active_command - measured`, then
  `q_command(t) = q_curobo_measured(t) + bias`. Remeasure that bias from the
  live command and state on every MPC update. This is a continuity translation,
  not integral control and not accumulated error.
- Keep the existing CuRobo route/IK branch, phase collision models, dual strict
  validation of predicted measured motion and translated command motion,
  Unitree lowcmd gains, gravity feedforward, hard limits, velocity limit,
  source-age checks, command buffer, watchdog, and recovery unchanged.
- Do not return to the rejected detours: direct phase-endpoint MPC, continuous
  nearest-route projection, one-window open-loop execution, iteration/cadence
  tuning as a controller fix, relaxed collision/velocity thresholds, or a new
  torque controller. Those address different problems or duplicate mature
  components already present in the stack.

### Implementation and command-free production regression

- Production now applies the complete live offset to every future sample in a
  short CuRobo window and records
  `command_tracking_offset_policy=hold_complete_window_remeasure_each_update`.
  The offset is remeasured rather than accumulated. The former linear endpoint
  fade is gone.
- The offline lifecycle benchmark can now simulate a persistent seven-joint
  tracking offset rather than silently using an ideal `measured == command`
  plant. This is a benchmark-only option; it cannot publish a robot command.
- Replaying the median offset from physical failure run
  `tabletop_20260819T121046Z` advanced `move_to_pregrasp` for 30 accepted
  windows to route index 36, instead of stalling at index 6/7. It then failed
  closed because the translated command target produced a strict
  `left_wrist_pitch_link/torso_link` sphere overlap of `0.060 mm`. Maximum
  command speed was `0.07321 rad/s`. The artifact is
  `/tmp/mpc_persistent_offset_full.json`.
- A temporary reference-governor experiment was tested and removed. It moved
  farther, but then CuRobo's predicted physical path itself crossed the same
  strict pair by `0.045 mm`. No threshold, collision exclusion, sphere change,
  or start-overlap exception was retained. Therefore the original continuity
  stall is fixed, but this retained near-zero-clearance route does **not** prove
  the complete physical lifecycle ready. The next independent issue is route
  robustness, not another MPC cadence/iteration change.
- Focused verification passes: `27 passed` across command-buffer and phase-MPC
  tests. Full verification passes `376 passed, 6 skipped` in the control
  environment and `375 passed, 1 skipped` in the planner environment. Ruff
  passes for the changed Python files; repository-wide Ruff still reports
  pre-existing findings inside vendored `third_party` trees. No robot command
  was sent.

## 2026-08-19 — Mature self-clearance implementation audit

### This is related to, but not identical to, the cube margin

- The open-transit cube fix enlarged only the optimizer's cube by the explicit
  5 mm physical policy and then checked the returned route against the exact
  physical cube and the same 5 mm distance requirement. That is the general
  mature pattern: give the optimizer room to converge, then retain an
  independent hard validation boundary.
- CuRobo deliberately separates world geometry padding from self-collision
  padding. `collision_sphere_buffer` changes robot radii used against the
  world. During model loading, that radius change is subtracted back out of
  the self-collision padding. Positive self-clearance must instead be expressed
  through the per-link `self_collision_buffer` robot-model field.
- The pinned CuRobo G1 configuration sets every `self_collision_buffer` entry
  to zero. Its shipped Franka configuration uses nonzero per-link values (for
  example 20 mm at the hand, 10 mm at the fingers, 50/100 mm at two base
  links), and its UR10e configuration uses 70 mm at the shoulder. These are
  commissioned, link-specific robot-model values, not task-time scalar sweeps.
- CuRobo's `optimizer_collision_activation_distance` updates scene-collision
  activation only. It does not give self-collision a 10 mm avoidance band.
  The shipped MPC task configuration places self-collision in its constraint
  manager, but with a zero G1 `self_collision_buffer` the boundary is still
  zero penetration. Existing local provenance that calls the common 10 mm
  constant a `self_collision_activation_distance` is therefore misleading and
  must not be used to justify a self-clearance claim.

### Comparison with other mature stacks

- MoveIt's ordinary planning-scene self-collision check deliberately uses the
  unpadded robot. World collision uses the padded robot. MoveIt Servo then adds
  a separate online proximity policy: its collision monitor scales velocity
  down as self distance enters a configured threshold and stops at collision;
  the official UR Servo configuration uses a 10 mm self threshold.
- Tesseract/TrajOpt supports collision as both cost and constraint. Its
  maintainer's recommended setup uses a larger contact distance for the cost
  than for the hard constraint and a positive contact-margin buffer to aid
  convergence. This is the same separation between steering margin and hard
  feasibility, expressed in the optimizer rather than by changing physical
  geometry.

Primary references:

- <https://curobo.org/tutorials/1_robot_configuration.html>
- <https://curobo.org/_api/curobo.cuda_robot_model.types.html>
- <https://moveit.picknik.ai/main/doc/examples/planning_scene/planning_scene_tutorial.html>
- <https://github.com/moveit/moveit2/blob/main/moveit_ros/moveit_servo/src/collision_monitor.cpp>
- <https://github.com/UniversalRobots/Universal_Robots_ROS2_Driver/blob/main/ur_moveit_config/config/ur_servo.yaml>
- <https://github.com/tesseract-robotics/tesseract_planning/discussions/190>

### Decision before implementation

- Do not add a generic self-collision padding sweep and choose the smallest
  value that makes one retained replay pass. First define and measure the G1
  tracking-clearance contract across retained runs and both arms, then encode
  any selected clearance through CuRobo's upstream-supported per-link
  `self_collision_buffer` in every boundary-planner and MPC optimizer model.
- Keep the exact physical zero-penetration checker independent and unchanged.
  It must continue to validate both predicted measured motion and translated
  command motion. A planner buffer can steer away from the boundary; it cannot
  redefine collision or excuse a failing route.
- The current retained failure is specifically a 0.060 mm
  `left_wrist_pitch_link/torso_link` overlap in the translated command target;
  CuRobo's predicted measured-state path remains strict-clear in the retained
  production replay. This is the dataset against which a source-supported
  self-clearance policy should be evaluated before another hardware run.

## 2026-08-19 — Removed cube-specific optimizer enlargement

- The optimizer-only 5 mm enlargement of the cube was removed from both the
  boundary planner and MPC. Both now use the exact configured object dimensions.
- The independent 5 mm open-transit cube-clearance requirement is unchanged.
  Every generated frozen route and every MPC predicted/command window still
  fails closed below that distance. Contact-link exceptions remain limited to
  their existing contact phases.
- No replacement robot padding, obstacle padding, relaxed threshold, or new
  collision exception was introduced. A route that CuRobo generates inside the
  hard 5 mm band is rejected and the boundary planner may try another existing
  grasp/IK branch.

## 2026-08-19 — Measured-close start-relative retention escape

- Trajectory run `tabletop_20260819T133703Z` passed the opposed-finger close
  test but was rejected before its 30 mm retention lift. The planned fixed-close
  hand had 5.838 mm table clearance; the contact-stalled physical posture put
  `left_hand_middle_1_link` at 3.808 mm. Earlier successful measured closes had
  remained above the 5 mm free-space floor, so the existing August 17 gate had
  not exposed its boundary-policy error.
- Command-free replay of all 81 measured-close route samples proved that the
  retained path did not move farther toward the table. Samples 0--2 remained at
  3.808 mm within float precision, sample 5 reached 4.959 mm, sample 6 reached
  6.254 mm, and clearance then continued increasing. Samples 75--80 were the
  exact symmetric return. There was no modeled table penetration.
- The measured-close frozen payload validator now treats only this already-
  achieved positive grasp boundary start-relatively. It requires the outbound
  escape and exact return never to go below that boundary, requires the route to
  reach the unchanged 5 mm floor, and enforces 5 mm throughout the intervening
  free-space samples. Self-collision, fixture checks, open/fixed-close planning,
  and the global configured clearance are unchanged. MPC retains its existing
  stricter per-window table rule pending its separate physical commissioning.

## 2026-08-19 — Run-local measured empty-open geometry and release evidence

- Trajectory run `tabletop_20260819T135229Z` physically grasped, lifted, and
  exactly replaced the cube. Software then rejected the descriptor open command:
  `left_hand_middle_0_joint` reached `-0.1005 rad` against the ideal `0 rad`
  target and the shared `0.08 rad` tolerance. The exception occurred before
  grasp retreat, clearance return, initial-finger restoration, and seated
  handback, so the existing failure cleanup deliberately selected zero torque.
  Cleanup and the 6.42 GB MCAP recording both completed without error.
- MCAP replay proved this was not a missing command. The hand received the zero
  target continuously for the complete eight-second interval. Six joints opened
  normally; the middle proximal joint moved monotonically from about `-0.297`
  to `-0.1005 rad`. Earlier in the same run, while empty at clearance, it had
  physically settled near `-0.0285 rad`. The post-replacement difference from
  that physical empty-open state was `0.0720 rad`, inside the existing tracking
  tolerance. Prior completed runs settled the same joint near `-0.027 rad`.
- The arbitrary finger posture measured at lowcmd takeover is not an open-hand
  reference; it can contain any operator/boot posture and exists only so it can
  be restored before returning control. The persistent measured empty-close is
  also not an open reference: it remains the no-object baseline used to detect
  opposed grasp obstruction and is not repeated during every task.
- The required descriptor open now occurs immediately after the supported arm
  reaches stationary clearance, before the clearance image and grasp planning.
  Its achieved active-hand posture is recorded at command acquisition and again
  at the visual anchor in `dex3_run_local_references.json`. The latter posture
  supplies the open-hand CuRobo geometry. The fixed descriptor zero target still
  supplies every command.
- After exact replacement, the controller continues publishing the descriptor
  zero target but verifies settling against the run-local measured empty-open
  posture. The same acceptance posture remains active during grasp retreat. If
  it is not recovered, the hand does not retreat and the existing fail-closed
  cleanup remains unchanged. A clearance-stage failure first restores the
  arbitrary initial fingers before executing the supported escape's exact
  reverse.
- This reorders an existing finger motion rather than adding a close/open cycle.
  Full task planning begins after the measured open and fresh clearance image
  because both bind the collision model; the persistent CUDA worker is already
  warm. Nominal motion time is therefore unchanged. No robot command was sent
  while implementing this change.

## 2026-08-20 — Fixed two-cube stack coordinator

- The stacking task is deliberately one explicit coordinator, not a generic
  task graph or home-grown task language: move the 60 mm R3 cube on the bare
  table, observe its actual result, then place the 40 mm R3 cube on it.
- A separate table marker is not required for this fixed task. Before stage one,
  the stationary 40 mm cube is the finite world obstacle. Before stage two, the
  reobserved 60 mm cube is the finite placement support. Both uniquely tagged
  cubes must be detected in the same image subset.
- Destination requests retain the original on-table cube observation solely as
  explicit plane evidence. The planner therefore keeps the real table at its
  observed height while checking the 60 mm cube as a finite cuboid; it does not
  infer a fictitious full table plane beneath the elevated 40 mm destination.
- Both arms first traverse independently planned supported escapes. Only one
  arm subsequently moves at a time. Proximity to the two live hands orders the
  two possible arm assignments but cannot approve one: complete source,
  destination, attached-transfer, fixed-close, measured-close, self-collision,
  table-plane, and finite-world checks remain authoritative.
- Five placement proposals span only the non-overlapping interior of the table
  segment evidenced by the two observed cube centers. The proposal nearest the
  original 60 mm position is tried first; no preferred midpoint or synthetic
  table boundary is introduced. Full CuRobo feasibility makes the final choice.
- Stage-one feasibility includes a nominal stage-two proof so the 60 mm cube is
  not deliberately moved into a known dead end. After physical stage one, the
  final stage-two request is rebuilt from the observed 60 mm pose; failure to
  plan then is an ordinary task rejection and returns both arms through their
  frozen supported routes.
- The same grasp candidate must pass source pickup, destination placement, and
  the attached transfer. A source-only winner is no longer accepted and then
  failed at the destination; the planner records each rejected grasp and tries
  the next existing shortlist candidate. Measured-contact collision checkers
  are built lazily only for a plan selected for execution, and validation never
  reruns motion planning.
- Existing single-cube implementations are reused for RealSense launch,
  calibration, gravity feedforward, fixed-rate control, Dex3 open/close and
  retention evidence, PC2 watchdog/handback, raw MCAP recording, and frozen
  rejection routes. No new robot command was sent while implementing or
  testing this coordinator.

## 2026-08-20 — MPC diagnosis, branch isolation, and immutable handoffs

- The planner-lifecycle performance work is isolated from correctness work on
  branch `perf/persistent-curobo-planner-pool`, commit `ee8c54f`. The original
  `feature/curobo-tabletop` branch remains at `d0d3701`. The timing and handoff
  repair is being developed separately on
  `fix/immutable-curobo-mpc-handoffs`.
- The supplied external diagnosis was substantially correct: the old wrapper
  overloaded CuRobo timing semantics, rewrote sample zero after strict
  validation, restarted a relative clock when an asynchronous result arrived,
  advanced route state before the window was executed, and treated a predicted
  terminal result too directly. A proposed six-class rewrite was not required;
  the existing worker, command buffer, and executor boundaries can enforce the
  invariant directly.
- One diagnosis detail was incorrect for the pinned B-spline backend. With a
  10 ms returned state period and four interpolation samples per knot, the
  sliced `action_sequence` begins at the internal sample-four offset, 40 ms
  after the boundary, not 10 ms. The repaired adapter uses the complete
  `robot_state_sequence` instead. Its first sample exactly preserves supplied
  position, velocity, and acceleration at the future handoff.
- `MPCCommandWindow` now carries separate source and absolute activation times,
  predicted q/dq/ddq, the exact desired command path, predecessor hash, and
  proposed route progress. No `rebase_start()` operation remains. The worker
  resamples and strictly checks both predicted and desired paths at no more than
  4 ms spacing before hashing the window. The executor schedules it unchanged.
- Rolling handoffs are frozen before the GPU solve. A replacement must match
  the active trajectory's command and predicted q/dq/ddq at that exact future
  time, arrive before that time, name the active window hash, and pass live q/dq
  activation gates. Route progress becomes committed only after activation.
- CuRobo can install an infeasible optimizer result into its internal execution
  manager. The wrapper now retains the last controller-accepted action seed and
  restores it after an infeasible solve. The controller keeps executing the
  unchanged prior trajectory and retries; it never installs or executes the
  rejected result.
- A direct command-free GPU probe found that the complete CuRobo rollout has 81
  states over 0.8 s. The previously selected 0.64 s prefix still ended at
  0.03705 rad/s. CuRobo's existing terminal support reduced velocity to zero by
  sample 76 and held it through sample 80. The complete rollout is now the
  certified fallback tail. No hand-authored braking spline and no CuRobo fork
  were introduced.
- If no new trajectory can be installed, the executor finishes that unchanged
  full rollout, holds and verifies measured endpoint settling, clears the
  uncompleted phase identity, and only then surfaces the planner failure. A
  nominal terminal window still records its phase endpoint. These two outcomes
  have separate tests.
- The retained RTX 5090 replay contained one 180 ms solve, so the former 120 ms
  future handoff was not robust. The handoff is now six 40 ms knots (240 ms),
  leaving at least 232 ms after the two-tick installation guard and more than
  half of the 0.8 s certified rollout after activation. In a later 334-window
  replay the worst complete window was 217.61 ms and no install deadline was
  missed.
- The offline benchmark was changed to use the same future-boundary,
  predecessor, retry, and fallback semantics as hardware. It no longer rebases
  a solve from the current state or stops at the first infeasible optimizer
  result.
- The honest replay exposed a separate unresolved problem. The wrapper chooses
  a monotonic local endpoint from the frozen route, but CuRobo MPC receives only
  that point goal—not the intervening frozen route segment as a reference. The
  current retained lifecycle completed `move_to_pregrasp` in 36 accepted
  windows, then remained around grasp-approach route index 26 despite continuing
  to return constraint-feasible windows. The older tripod regression similarly
  remained around move-to-pregrasp route index 33. A direct grasp-phase replay
  from its exact frozen start reproduced the same index-26 stall. Raising the
  solve count is not a fix.
- Therefore the correctness branch is offline-only and must not be presented as
  hardware-ready. Exact timing and failure containment are improved, but the
  point-goal MPC wrapper still lacks a reliable way to follow collision-critical
  bends in a frozen route. That question should be answered at the CuRobo goal/
  reference interface, not with a collision exception, a larger iteration cap,
  or another task-specific state machine.
- No robot command was sent during this diagnosis or any replay above.

## 2026-08-20 — Moving-target MPC scope and object-relative progress

- The former eight-phase MPC lifecycle was the wrong abstraction for the
  intended moving-cube task. MotionGen remains responsible for the supported
  escape, global clearance-to-pregrasp route, payload lift, placement, and
  return. MPC is now limited to the open-hand pregrasp-to-grasp segment where a
  changing visual goal is useful. Dex3 transitions and all commissioned
  ownership/safety behavior remain outside MPC.
- Moving-target perception now separates a fixed table reference from the
  object. The existing 6 x 9 `DICT_5X5_50` ChArUco board is observed once at
  clearance. Every MPC window receives one fresh AprilCube frame plus a
  synchronized pelvis/waist/torso estimate. Camera motion updates the board-to-
  camera transform; cube motion independently updates the goal and cube
  obstacle.
- The first moving-target replay revealed the actual progress bug. Route
  progress was inferred by comparing a corrected joint state against the old
  nominal joint path. A moved cube necessarily changes the required joints, so
  that comparison froze around route index 14 even while CuRobo returned valid
  windows. Progress is now measured by the tool pose relative to the live cube
  and advanced monotonically along the nominal object-relative approach.
- Cube contact and fixture contact now have separate link policies. The cube
  may contact all seven movable Dex3 finger links during final approach;
  otherwise valid grasps were being rejected at proximal `middle_0` links.
  The presentation fixture still permits only the existing three distal
  contact links. Palm, wrist, table, torso, and self collision remain enabled.
- In the retained CUDA replay with a 5 mm cube translation, the corrected
  approach reached terminal in 85 accepted windows and 20.96 s of simulated
  motion with zero rejected windows. Terminal error was 3.6329 mm and 0.9164
  degrees, within the existing 5 mm / 0.05 rad contract. The old implementation
  had not completed after 200 windows.
- The exact command and predicted paths actually accepted by the controller are
  stitched into a hash-bound trajectory. Before any new CUDA work, its exact
  reverse plus the original pregrasp return are installed as provisional
  recovery. The hand then closes immediately at the reached grasp. A
  continuation request rebuilds the fixed-close check and attached payload
  lift at the terminal live cube pose; successful replacement and retreat are
  exact reverses of the new lift and accepted MPC approach.
- A retained static continuation probe passed the fixed-close and attached-lift
  checks, preserved the selected grasp, and produced an exact reverse. It took
  about 14.29 s after grasp; attached-lift planning accounted for about 4.50 s
  and repeated model/checker construction for most of the remainder. This is
  the remaining operational blocker. It should be removed by preparing or
  reusing compatible structures before motion, not by weakening collision
  checks or changing the MPC formulation again.
- Verification after the implementation: Ruff passed; the complete test suite
  passed with `406 passed, 6 skipped`. All work was command-free; no robot
  command was sent. This branch is not cleared for hardware MPC commissioning
  until the post-grasp hold latency is removed and the composed lifecycle is
  replayed.

## 2026-08-20 — Moving-target continuation reuses the planner pool

- The isolated `perf/persistent-curobo-planner-pool` work was integrated rather
  than implementing a second continuation cache. The hardware path now builds
  the strict checker, fixed-close validator, open planner, and attached planner
  at the stationary clearance boundary. Compatible live object, arm, finger,
  and world values are updated in place; topology mismatches still rebuild and
  fail through the existing planning boundary.
- The moving-target continuation uses that same pool for terminal FK, the live
  fixed-close sweep, attached-payload lift planning, and measured-close route
  validation. It preserves the separate live object/table geometry and exact
  accepted MPC reverse introduced above.
- A first retained static probe paid `16.77 s` before motion and reduced the
  post-grasp continuation from `14.29 s` to `1.91 s`. A complete retained 60 mm
  direct-table replay then paid `13.68 s` of pre-motion construction and `2.50
  s` of MPC setup, moved the cube by `5 mm`, reached the grasp in `81` accepted
  windows over `20.00 s` of simulated motion, and rebuilt the post-grasp
  lifecycle in `2.01 s`. Terminal error was `3.704 mm / 0.954 deg`; the same
  grasp was retained and the approach return was its exact reverse.
- The composed replay reported reuse of the strict checker, fixed-close
  validator, and attached optimizer, with no topology rebuild. Payload planning
  itself took `0.322 s`; the remainder of the `2.01 s` continuation was live
  kinematic/collision reconfiguration and validation.
- MPC construction was still located at pregrasp and the first camera target
  was incorrectly captured before that multi-second setup. MPC is now prepared
  immediately after the nominal lifecycle is planned at clearance, before the
  clearance-to-pregrasp route. The first moving target is captured afterward,
  preserving the unchanged source-age contract.
- These are command-free retained-run results. They clear the architectural and
  latency blockers for a staged physical test; they do not constitute physical
  moving-cube commissioning.

## 2026-08-20 — Pruned invariant one-arm self-collision pairs

- The normal tabletop model exposes only the selected arm's seven joints. The
  head, trunk, legs, opposite arm, and opposite hand are all locked to the live
  measured snapshot, so collision between any two of those links cannot change
  during IK or trajectory optimization. Those locked-link/locked-link pairs are
  now added to CuRobo's existing self-collision ignore topology. Every pair
  involving the selected arm or hand remains governed by the prior policy.
- CUDA resolution of the exact production models reduced the left-arm topology
  from 216,578 to 113,926 sphere pairs and the right-arm topology to 117,206.
  Both now use one map-reduce block per state, and an explicit audit found zero
  remaining locked-link/locked-link pairs. The optional offline waist-yaw model
  does not use this pruning because waist motion changes upper-body-to-leg
  relationships.
- A command-free alternating A/B replay used the same retained 60 mm left-arm
  task request and selected the same grasp in every run. After excluding each
  ordering's first-process CUDA/model cold-start outlier, two unpruned runs
  averaged 14.58 s and two pruned runs averaged 12.72 s. The measured saving is
  1.86 s, or 12.8%, for `plan_tabletop_task`; it is not a measurement of camera,
  controller, supported-escape, or complete two-cube stacking wall time.
- The two physical dorsal marker carriers remain modeled. Their 60 spheres add
  23,520 pairs to the pruned left-arm topology: 17,370 from the selected plate,
  5,250 from the opposite plate, and 900 plate-to-plate pairs. Removing them
  would make the collision model physically incomplete while reducing a
  measured 256-state strict validation by only about 0.18 ms, so they were not
  changed.
- No robot command was sent while implementing or benchmarking this change.

## 2026-08-20 — Persistent per-arm CuRobo planner pool

- Replaced the session's single open-hand cache with one small lazy pool keyed
  by physical topology and arm: left/right open motion planners, left/right
  closed-hand attached-payload planners, strict open/payload collision
  checkers, and fixed-close sweep validators. Switching arms or moving from an
  open phase to an attached phase can no longer evict an incompatible model.
- Compatible CuRobo motion planners retain their CUDA graphs and update folded
  kinematic tensors, collision-sphere padding, random seeds, and world scenes
  in place. Strict checkers likewise retain their CUDA collision buffers when
  locked-joint values change without changing topology. A topology mismatch
  remains fail-safe: that one slot is rebuilt rather than copying incompatible
  tensors.
- Payload installation still updates both the IK and trajectory-optimization
  `AttachmentManager` instances in the pinned CuRobo revision. The pool does
  not weaken or remove any branch search, strict endpoint check, fixed-close
  sweep, exact route validation, table/fixture check, measured-close check, or
  phase-specific attachment rule.
- The fixed-close cache key now describes its actual 14-DOF checker model. The
  active arm/finger query posture is updated separately, so changing only the
  reached pregrasp state does not falsely require a new checker. Compatible
  changes to locked waist, legs, opposite arm, opposite hand, or joint offsets
  now refold values into the retained checker; only a topology mismatch rebuilds
  it.
- A command-free retained 60 mm left-arm task selected
  `cube_head__seed_0000000039__sample_205` with identical rejection decisions
  and trajectory arrays between standalone construction and the pool's cold
  path. Repeating the complete task in the same pool reduced wall time from
  `15.29 s` cold to `1.85 s` warm. The warm solve preserved the same candidate,
  phase order, trajectory shapes, and rejection decisions; GPU numerical
  variation was at most `0.000514 rad` in joint samples and `0.000730 s` in
  resampled timestamps, with every strict validation rerun.
- A more representative command-free fixed pick/place replay moved the same
  retained 60 mm scene by 100 mm. Standalone source, destination, and transfer
  planning took `36.12 s`; the pooled lifecycle took `22.23 s`, saving
  `13.90 s` (`38.5%`) while selecting the same grasp and passing the same full
  source/destination/attached-transfer lifecycle. The actual clearance-to-
  pregrasp then pregrasp-correction sequence also completed with the grasp
  preserved; its corrected task took `7.98 s` because open and strict objects
  were reused while the first attached planner and changed locked-state
  fixed-close model still had to be created/resolved.
- Real CUDA probes additionally changed the locked waist-pitch witness by
  `0.01 rad`: both the motion planner and strict checker retained object
  identity and updated compatible tensors without rebuilding topology. Normal
  tests passed `402 passed, 7 skipped`; the complete CUDA planner-environment
  suite passed `402 passed, 1 skipped` at this checkpoint.
- No robot command was sent. All timing and equivalence checks used retained
  request artifacts and command-free CUDA planning.

## 2026-08-20 — Paid task-model startup before the first arm trajectory

- The trajectory workflow now constructs the selected arm's strict,
  fixed-close, open-hand, and attached-payload CuRobo objects after the
  supported escape is solved but before that escape is executed. The two
  motion optimizers each receive one disposable 5 mm upward query to exercise
  their CUDA graphs. No result from those probes is installed for execution.
- This is deliberately not a nominal task plan. A retained initial-clearance
  request could not complete a grasp lifecycle even though its later fresh
  clearance observation selected and executed
  `cube_head__seed_0000000039__sample_205`. Treating nominal feasibility as a
  pre-motion gate would therefore have been a regression. The warmup only
  constructs compatible models; fresh visual/proprioceptive boundaries remain
  authoritative.
- The fixed-close validator now updates compatible locked-joint kinematics in
  place. Its already-resolved 14-DOF checker is also reused by the subsequent
  measured-contact retention validator instead of constructing the identical
  model again. A topology change remains fail-safe and rebuilds the checker.
- In a command-free replay of the retained successful 60 mm left-arm sequence,
  construction-only warmup cost `13.35 s` before motion. The fresh-clearance
  pregrasp plan then took `11.23 s`; `7.32 s` of that was genuine route search
  over the observed scene. The post-pregrasp corrected task fell from the prior
  `7.65 s` replay to `6.02 s`, and the post-plan retention-checker preparation
  fell to `0.024 s`.
- The remaining corrected-task time was measured rather than labeled startup:
  `1.38 s` strict locked-body model resolution, `1.51 s` fixed-close model
  resolution, `0.68 s` batched IK, `0.64 s` open-route planning/validation, and
  `1.60 s` attached-lift planning/validation. Eliminating the two model-folding
  costs would require a different collision-model representation, not another
  cache wrapper.
- The selected candidate remained identical, every normal and CUDA validation
  still ran, and the final suites pass `405 passed, 7 skipped` in the control
  environment and `405 passed, 1 skipped` in the CUDA planner environment. No
  robot command was sent while implementing or benchmarking this change.

## 2026-08-20 — Commissioned 0.2 rad/s tabletop default

- Successful physical 40 mm and 60 mm tabletop trials established
  `0.200 rad/s` as the normal selected-arm trajectory limit. It is now the task
  configuration default instead of requiring a per-run override.
- `--maximum-arm-velocity-rad-s` remains available for deliberately slower
  runs. The independent hardware ceiling remains `0.200 rad/s`; Dex3 posture
  timing is unchanged.

## 2026-08-20 — Planner lifecycle begins with the hardware command

- The isolated CUDA worker is now spawned immediately after the hardware run
  lock is acquired, before ROS initialization, camera startup, or the read-only
  preview. No Unitree command publisher is created by this worker.
- Once read-only preflight supplies the selected arm, object profile, and
  presentation topology, command-free warmup runs asynchronously while the
  preview and SPACE prompt remain active. It retains the open-hand and
  attached-payload MotionGen models. MPC mode additionally retains one
  provisional moving-grasp solver; trajectory mode does not pay that cost.
- SPACE still gates every robot command. If it is pressed before warmup has
  completed, the command path waits for the worker and surfaces any warmup
  failure before creating the recorder or command publisher. Preflight
  observations are never accepted as executable feasibility evidence.
- The retained moving-grasp solver is rebound after the fresh loaded and
  clearance observations to the exact selected route, live scene, locked
  joints, object pose, and state-estimator reference. Scene topology must match
  exactly. A mismatch fails instead of silently rebuilding or using stale
  geometry.
- A command-free retained 60 mm left-arm replay measured `17.842 s` of
  MotionGen plus MPC construction/warmup before motion. Exact live planning
  then measured `3.547 s` for supported escape and `10.633 s` for the complete
  clearance task. Binding the retained MPC to
  `cube_head__seed_0000000039__sample_205` took `1.499 s`
  (`1.377 s` live kinematic/scene rebind and `0.114 s` route setup), versus the
  previous approximately `6--11 s` disposable MPC construction path. All ten
  execution edges and the selected grasp were preserved.
- The fixed-close validator is intentionally not constructed from the
  preflight state. Measurements showed its locked-state model still had to be
  resolved again at clearance, so doing both added work without reducing the
  live-boundary latency.
- Production MPC is now limited in code as well as policy to the visually
  updated pregrasp-to-grasp segment. The obsolete multi-phase lifecycle
  benchmark, phase switching API, and nominal route-tracking window path were
  removed. MotionGen continues to own global, payload, placement, and return
  motion.
- Final verification passed Ruff, `414 passed, 7 skipped` in the control
  environment, and `414 passed, 1 skipped` in the CUDA planner environment. No
  robot command was sent during implementation or replay.

## 2026-08-20 — Removed the one-time ChArUco dependency from moving-target MPC

- Physical run `tabletop_20260820T225844Z` reached clearance and passed the
  supported escape, but the moving-target path rejected all five frames because
  no ChArUco board was present. The AprilCube itself was visible.
- Inspection showed that MPC observed the board only once at clearance. It did
  not reobserve the board during the approach, so the board supplied no ongoing
  external correction and could not measure unmodelled seat translation.
- The stationary cube observation at clearance now defines the same kind of
  frozen arbitrary reference frame. The cube must remain stationary through
  that observation. Every later MPC window independently detects the cube and
  combines it with the unchanged pelvis/waist/torso camera-state propagation,
  so the cube may move after the moving-target controller is ready.
- The runtime no longer imports or invokes ChArUco detection, writes a
  `table_board_anchor.json`, or blocks MPC when the board is absent. The
  separate chair-compliance and state-estimation research tools retain their
  ChArUco board because they use it as external measurement/evaluation evidence.
- This simplification does not claim to observe camera translation missing from
  the proprioceptive estimator. Adding that capability later would require a
  continuously visible fixed workspace landmark, not the removed one-time
  sample.
- Verification passed Ruff, `414 passed, 7 skipped` in the control environment,
  and `414 passed, 1 skipped` in the CUDA planner environment. No robot command
  was sent.

## 2026-08-20 — Initial MPC rejection now holds and retries instead of faulting

- Physical run `tabletop_20260820T231034Z` reached the selected left-arm
  pregrasp and then entered zero torque. The cube-frame change was not the
  cause. The first MPC result passed both strict collision checks, including
  `5.026 mm` hand/table clearance for the required `5.000 mm`, but its command
  path peaked at `0.2570 rad/s` against the unchanged `0.2000 rad/s` physical
  limit. CuRobo's full state reached `0.2635 rad/s` against its reserved
  `0.1900 rad/s` optimizer limit, so rejecting that window was correct.
- The wrapper error was the response: no MPC command had been installed, yet a
  generic `RuntimeError` bypassed the existing pregrasp recovery and entered
  the fail-closed zero-torque cleanup.
- A command-free replay of the exact retained pregrasp boundary showed the
  receding-horizon behavior is not deterministic on one solve. Its first solve
  was infeasible; the immediate next solve from the same stationary state was
  feasible with a `0.0923 rad/s` command peak and passed strict validation.
- Production now holds the unchanged pregrasp and retries rejected initial
  windows with fresh cube/body observations under the existing motion timeout.
  No streaming command is created until a feasible window exists. Exhausting
  that timeout runs the already validated pregrasp-to-clearance recovery,
  restores the initial finger posture, reverses the supported escape, and then
  restores seated FSM 3. Actual controller, transport, and watchdog faults
  still retain the zero-torque path.
- Verification passed Ruff, `416 passed, 7 skipped` in the control environment,
  and `416 passed, 1 skipped` in the CUDA planner environment. No robot command
  was sent while diagnosing or implementing the change.

## 2026-08-20 — Replacement-window rejection no longer expires robot control

- Physical run `tabletop_20260820T232437Z` accepted its second initial MPC
  solve, then rejected generations 2--4 because the translated command path put
  `left_hand_middle_1_link` only `3.736--3.885 mm` above the table against the
  unchanged `5.000 mm` requirement. Those collision rejections were correct.
  The cube moved only `0.42--0.48 mm`, the camera/body correction was only
  `1.15--1.37 mm`, and each complete CUDA solve plus validation took
  `88--91 ms`.
- The actual failure was a wrapper lifecycle error. Its certified generation-1
  window ended after `0.800 s`; a subsequent one-frame camera wait was allowed
  to outlive the remaining horizon, and the command buffer treated being
  `0.2521 s` past that horizon as a controller fault. This conflated planner
  availability with the health of the independent 250 Hz robot controller.
- Production feasible MPC windows already require a decelerated endpoint; the
  retained accepted window ended at exactly zero predicted velocity and
  acceleration. The fixed-rate controller now continues publishing that exact
  endpoint for as long as a replacement is unavailable. A later feasible
  window can splice from the held endpoint on the same hash-bound predecessor
  and warm-start chain, even when its handoff is later than the prior horizon.
- The obsolete `maximum_window_gap_s` path was removed. It had incorrectly
  reused the executor's real control-loop scheduling-gap threshold as a
  high-level planning deadline. Control-loop timing, source freshness, live
  handoff error, collision, velocity, DDS/transport, and PC2 heartbeat checks
  are unchanged and still fail closed.
- Rejection logging now includes the generation, concrete collision or bound
  reason, and whether the prior certified horizon is still running or its
  endpoint is being held. This is a narrow correction to the existing rolling
  controller, not a second restart state machine.
- Reference correction: NVIDIA STORM has public MPPI MPC and Franka controller
  code, but it is not NVIDIA's newer Grasp-MPC system. Grasp-MPC uses an
  ordinary planner to reach pregrasp and closed-loop vision MPC for the final
  grasp; its project page still lists code as forthcoming. Therefore no public
  Grasp-MPC hardware wrapper was available to copy for this failure policy.
- Verification passed Ruff and `418 passed, 7 skipped` in the control
  environment; the CUDA planner environment passed `418 passed, 1 skipped`.
  No robot command was sent.

## 2026-08-20 — MPC handoffs now perform the documented live remeasurement

- Review of physical run `tabletop_20260820T235231Z` found a concrete mismatch
  between the August 19 controller decision and production. The run installed
  seven windows and rejected 197; after early transients, its recorded
  `command_tracking_offset_rad` remained byte-for-byte constant even though a
  fresh LowState sample was collected for every request.
- The live state was used for camera/body synchronization, but
  `prepare_streaming_handoff()` obtained both the next command and the next
  predicted position from the previously active window. The planner therefore
  calculated `command - predicted` from two old-window samples. It did not
  remeasure `active command - actual measured position` as the notes claimed.
- The executor now atomically pairs one fresh arm state with the exact active
  arm command on every replacement request. The already-certified future
  command splice remains unchanged. Its predicted physical position is
  reanchored with the newly measured offset before CuRobo plans the replacement
  window.
- Replacement installation now receives the exact frozen handoff object and
  verifies its predecessor hash, absolute time, command, reanchored predicted
  position, velocity, and acceleration. This permits the predicted physical
  state to be corrected while retaining a mathematically continuous command
  stream. It does not add a torque controller, change PD/gravity feedforward,
  alter any collision margin, or return to one-step MPC.
- A focused regression proves that a newly observed `+0.010 rad` tracking
  offset keeps an immutable `0.016 rad` future command while reanchoring the
  predicted physical handoff from `0.016` to `0.006 rad`; a stale worker result
  is rejected, and the correctly reanchored replacement installs at the exact
  command boundary.
- Verification passed Ruff and `419 passed, 7 skipped` in the control
  environment; the CUDA planner environment passed `419 passed, 1 skipped`.
  The 60 mm print artifact was generated separately with `DICT_4X4_100` IDs
  20--25 and geometry identical to the existing commissioned 60 mm R3 cube.
  No robot command was sent.

## 2026-08-21 — Offset-aware table tightening was tested and rejected

- Physical run `tabletop_20260821T100244Z` used the live-remeasured handoff
  implementation. That fix worked: 191 distinct tracking offsets appeared
  across 194 results, and the final measured arm error was only `0.0012 rad`.
  The run nevertheless accepted only four windows; all 190 rejections were the
  desired-command hand path falling below the unchanged `5.000 mm` table gate.
- A bounded command-free experiment raised only CuRobo's internal table proxy
  by the geometric clearance loss caused by the live command-minus-measured
  offset. The real plane and both strict post-checks stayed unchanged. A replay
  of all 194 retained targets and offset samples then made all 194 generated
  windows strict-valid and advanced the frozen route from index 14 to 38.
- The experiment also proved why it cannot finish this grasp. At the final live
  Cartesian goal, the predicted physical configuration has `7.074 mm` table
  clearance, but adding the measured command offset puts the desired-command
  configuration at `-1.588 mm`. Meeting the `5.000 mm` desired-command rule
  therefore needs `6.588 mm` of displacement away from the exact goal. The
  offset-aware replay converged `6.270 mm` from the goal, outside the unchanged
  `5.000 mm` terminal tolerance, and never became terminal.
- This is a geometric incompatibility at the endpoint, not another route-search
  or scheduling failure. The table-tightening experiment was removed rather
  than committed as a production workaround. The independent live-handoff fix
  remains valid. Do not retry the same MPC/table-padding strategy physically;
  use the commissioned boundary-replanning workflow, or separately change the
  grasp/control assumptions so the goal itself has a safe desired-command
  configuration.
- Command-free evidence is retained at
  `work/mpc_replay_20260821T100244Z_robust_sequence.json`. No robot command was
  sent during the experiment.

## 2026-08-21 — Two-60-mm stacking and bounded physical grasp retry

- The fixed stack coordinator now uses the two physically distinct 60 mm R3
  cubes. The lower-cube profile keeps `DICT_4X4_100` IDs 10--15. A second
  hash-bound profile uses the newly printed cube's IDs 20--25 and shares the
  same commissioned bilateral 60 mm grasp shortlist because its geometry is
  identical.
- This remains one explicit operation, not a task graph: move the original
  60 mm cube to a feasible point on the observed table segment, reobserve both
  cubes, then place the second 60 mm cube on the first. Both centers are 30 mm
  above the table, the stacked upper center is 60 mm above the lower center,
  and equal 60 mm footprint radii are used when proposing non-overlapping
  lower-cube destinations.
- A physical grasp rejection is now a typed, recoverable result only after the
  existing code has opened the active Dex3 hand and followed the frozen retreat
  and return-to-clearance trajectories. With the default `--grasp-retries 1`,
  the coordinator then captures fresh synchronized images and robot state,
  redetects both cubes, excludes the failed grasp candidate, and runs the same
  complete CuRobo pick/place planner again. It never repeats a stale route.
- Stage-one retry recomputes both the lower placement and the nominal upper
  placement proof from the new scene. Stage-two retry uses the reobserved upper
  and lower cube poses directly. If the fresh scene cannot be detected or
  planned, or the one retry is also physically rejected, the task returns both
  arms through their supported routes and restores seated control.
- Planner, controller, transport, DDS, camera, and watchdog faults are not
  retries. They retain the established fail-closed handling. Retry count,
  excluded candidate IDs, each reason, recovery completion, requests, plans,
  images, and MCAP remain recorded in the run directory.
- Verification passed Ruff and `421 passed, 7 skipped` in the control
  environment; the CUDA planner environment passed `421 passed, 1 skipped`.
  No robot command was sent while implementing or testing this change.

## 2026-08-21 — Stack planner startup moved entirely before ownership

- Review found that the single-cube path launched its isolated CUDA worker at
  command start and warmed reusable MotionGen models during the read-only
  preview, but the new stack path synchronously waited for only basic CUDA
  readiness after camera preflight. This was an implementation-parity gap, not
  an unavoidable two-cube planning cost.
- `run-stack` now launches the same persistent worker before ROS/camera
  preflight. Once both cubes are detected, one aggregate command warms the
  left and right open-hand and attached-60-mm-cube planner topologies inside
  the same CUDA context while the operator preview remains active. Pressing
  Space cannot create a robot publisher until that warmup has completed.
- The aggregate request is intentionally sequential inside one GPU worker.
  Running two planner processes would duplicate the G1/Dex3 models and GPU
  memory; the useful concurrency is between CUDA work and the independent
  camera/operator preflight.
- No executable route is accepted from preflight data. Supported escapes,
  cube observations at clearance, and complete task feasibility remain bound
  to the live post-ownership state. The additional single-cube trajectory
  mode pregrasp state-correction replan was deliberately not added to the
  stack path in this change.
- The real RTX 5090 command-free probe warmed all four retained planner roles
  in `18.28 s`; the exact asynchronous worker-protocol repetition took
  `13.98 s`. Both completed within the 24 GB GPU and reported
  `robot_command_authorized=false`. Verification passed Ruff,
  `422 passed, 7 skipped` in the control environment, and
  `422 passed, 1 skipped` in the planner environment. No robot command was
  sent while implementing or testing this lifecycle change.

## 2026-08-21 — First stack run exposed a measured/command boundary regression

- Physical run `stack_20260821T112912Z` completed the left supported escape,
  then stopped before any right-arm motion with `trajectory start differs from
  the current command by 0.020277314rad`. Cleanup completed without a reported
  error and the MCAP was retained.
- The new dual-arm coordinator had planned the right supported escape from the
  measured right-arm LowState after the left arm moved. The controller was
  correctly still holding the exact right-arm command installed at the loaded
  boundary. The right elbow measurement was `-0.020277314rad` from that held
  command; CuRobo's first right-arm sample matched the measurement exactly and
  therefore violated the executor's command-continuity invariant.
- This was specific to the newly added left-to-right arm switch. The existing
  single-cube trajectory path never changes selected arms and already replaces
  measured active-arm joints with the exact command at every planning
  boundary.
- The stack now constructs every dual-arm planning snapshot from both exact
  held commands while retaining the synchronized measured body and Dex3
  state. The corrected retained right-arm request passed real CuRobo supported
  escape planning with `0.000000000rad` start-to-held-command error. A focused
  regression fixes the observed `0.020277314rad` right-elbow offset in the
  measured state and proves it cannot enter the command-bound planning
  snapshot.
- No robot command was sent while diagnosing, correcting, or replaying this
  failure.

## 2026-08-21 — Stack feasibility search no longer performs blind nested route planning

- Physical run `stack_20260821T120104Z` reached both supported clearances and
  opened both hands, then spent about 8 minutes 40 seconds in planning before
  the operator interrupted it. The worker tried four lower-cube placement
  samples, planned a complete lower-cube transfer for each, and then attempted
  hundreds of upper-cube source/destination IK branches. The retained log has
  208 `trying next source/destination grasp` messages and 327 explicit branch
  attempts. This was wrapper-level exhaustive search, not one slow CuRobo
  solve.
- The dominant structural error was that source and destination batched IK
  results were available internally but route planning began before comparing
  them. A placement could therefore spend minutes proving individual source
  and destination routes even when no identical object-to-hand grasp was valid
  at both endpoints.
- The existing fixed-close GPU pruning, batched CuRobo IK, and strict full-robot
  endpoint collision check are now exposed as a route-free feasibility pass.
  Pick/place planning intersects source and destination candidate IDs first. If
  that set is empty, no MotionGen route or attached-payload transfer is tried.
  If it is nonempty, only those common IDs enter complete lifecycle planning,
  with the restrictive destination checked before the source and transfer.
- The stack coordinator now checks the upper-on-lower stage before planning the
  lower-cube move. An infeasible stack target can no longer cause a redundant
  complete lower-cube solve for every placement sample.
- Command-free replay of the exact retained `segment_05_of_05` upper-stage
  request changed failure time from `91.64 s` to `11.30 s` in a fresh worker.
  It reported the actual result directly: source and destination each had
  endpoint-valid grasps, but their common set was `0/57`. The other retained
  placement samples failed the same endpoint intersection in `8.37--12.21 s`.
  Synthetic opposite-arm replays also produced `0/57` common endpoint-valid
  grasps for all five samples. These are one-shot process timings; the hardware
  path retains one warmed worker and planner pool. Replaying through that exact
  persistent lifecycle took `15.14 s` for the pre-Space dual-arm warmup, then
  `5.41 s` for the first placement and `2.12--2.25 s` for each subsequent
  placement. Thus the four samples that consumed roughly 8 minutes in the
  physical run now produce the same rejection in about 12 seconds after
  ownership.
- A command-free experiment preserving the upper cube's yaw produced a small
  common endpoint set at two samples, but every resulting route still failed
  the unchanged table/approach checks. That unproven geometry change was not
  added to production.
- The interrupted run's raw MCAP flushed cleanly and remains at approximately
  30.6 GB under the run directory. It was not deleted. No robot command was
  sent during diagnosis, implementation, or replay.
- Verification passed Ruff and `424 passed, 7 skipped` in the control
  environment.

## 2026-08-21 — Stack task reduced to one direct pick/place

- The original two-stage operation was removed. It needlessly moved one cube
  to a new table location before picking the other cube, doubling physical
  execution and multiplying placement and arm-assignment searches.
- `run-stack` now observes the two uniquely tagged 60 mm cubes, considers each
  cube with its nearest arm, and selects one complete transfer that places the
  moving cube directly on the stationary cube. There is no preliminary cube
  relocation, midpoint, generic task graph, or second pick.
- The stationary cube is a finite obstacle during source pickup and the same
  finite object becomes the destination placement support. The real table
  plane remains bound to the original on-table observation.
- Four upright quarter-turns may be evaluated as nominal wrist-path choices.
  This does not assert that Dex3 preserves the cube's yaw. Some in-hand cube
  rotation is physically unavoidable; the task objective is therefore the
  moving cube's nominal center over the observed support-cube center, not an
  exact final cube orientation. No post-grasp visual observation is fabricated
  while the cube is occluded.
- Every option receives the route-free source/destination endpoint
  intersection first. Only endpoint-compatible options can enter the existing
  complete single-cube pick/place planner and the existing execution/retry
  path. A physical grasp retry keeps the same moving cube and arm, reobserves
  both cubes, and excludes the failed grasp candidate.
- Obsolete two-stage geometry and tests were deleted. Verification passed Ruff,
  `422 passed, 7 skipped` in the control environment, and `422 passed,
  1 skipped` in the CUDA planner environment. No robot command was sent.

## 2026-08-21 — Direct stack now uses the commissioned one-arm lifecycle

- Review of retained run `stack_20260821T130046Z` found that the simplified
  one-pick task still inherited the old coordinator's two-arm preparation: it
  lifted both supported arms and opened both hands before choosing the mover.
  That placed the unused hand at transfer height and created the dominant
  hand-to-hand planning collisions. This behavior was absent from the proven
  single-cube workflow.
- `run-stack` now chooses the globally nearest cube/arm pair during read-only
  preflight, warms only that arm, executes only that supported escape, opens
  only that hand, and returns only that arm. The unused arm and Dex3 remain at
  their exact supported initial commands throughout the task and retry path.
- The first attempted run after this refactor stopped before ownership because
  the selected-arm pose set was paired with the original left-arm activation
  gate. The post-preflight ownership gate is now rebound to the selected arm,
  preserving the required pose-set/activation-arm equality. No command was
  published during the rejected attempt.
- The next run, `stack_20260821T132754Z`, selected the primary cube with the
  right arm, completed the right supported escape, opened the right hand, and
  found a complete nine-phase pick/place plan. Execution then stopped before
  the pregrasp because the physically measured empty-close reference used by
  grasp validation had only been commissioned for the left Dex3. Cleanup and
  the 5.36 GB MCAP completed without a reported error. This is a real missing
  right-hand commissioning prerequisite, not a CuRobo failure.
- Empty-close availability is now checked during read-only preflight, before
  SPACE, ownership, or arm motion. The stack release/recovery helper was also
  corrected to open only the selected hand and retain the unused hand at its
  measured supported posture.
- The right empty-hand close was then physically commissioned with the
  standalone finger-only command. A read-only stationary state snapshot
  recorded `[-0.04185495, -0.58644527, -0.97625828, 0.87804961,
  0.98250055, 0.87173128, 0.98171157] rad`; maximum velocity was zero and the
  maximum descriptor-target residual was `0.041855 rad`. Both hands now have
  explicit measured empty-close references; no mirrored or nominal value was
  substituted.
- The shared direct-table 60 mm grasp shortlist now uses a `50 mm` pregrasp
  approach instead of `100 mm`. This applies identically to both uniquely
  tagged 60 mm profiles and therefore to standalone 60 mm pickup and direct
  stacking; the 40 mm profile remains at `50 mm`. Runtime CuRobo still plans
  and validates the complete approach and exact reverse at the selected
  distance.
- An active-only replay removed the retained hand-to-hand failures but exposed
  a second mismatch: the attached-transfer optimizer's whole-robot table
  cuboid collided with the deliberately table-supported unused hand. The
  transfer now retains full-robot self-collision and the finite support cube,
  while the existing independent plane check enforces clearance for every
  moving-hand and attached-payload sample.
- The exact retained right-arm/primary-cube yaw-0 request now produces a full
  nine-phase CuRobo plan. The transfer kept `support_cube` as its only world
  cuboid, planned in `0.355 s`, and passed with `104.49 mm` moving-hand and
  `90.33 mm` payload table-plane clearance. Ruff and the control suite passed
  with `425 passed, 7 skipped`. No robot command was sent.

## 2026-08-21 — Pregrasp distance is an explicit run contract

- Both uniquely tagged direct-table 60 mm profiles share one grasp shortlist
  whose default pregrasp distance is now `0.050 m`.
- `run-tabletop` and `run-stack` both expose `--pregrasp-distance-m`. Omitting
  it resolves the object shortlist's default; supplying it records that exact
  positive finite value in every hash-bound `TabletopTaskRequest`.
- CuRobo uses the request value to construct every pregrasp target. The
  existing full IK, collision, table-clearance, approach, and exact-reverse
  validation remains unchanged. Fixture shortlists still reject distances not
  included in their qualified approach set.
- The reverted same-frame Dex3 dorsal-marker pregrasp dependency remains
  absent. Verification passed Ruff, `427 passed, 7 skipped` in the control
  environment, and `427 passed, 1 skipped` in the planner environment. No
  robot command was sent.

## 2026-08-21 — Clearance perception rejection returns through the supported route

- Retained run `stack_20260821T164205Z` reached the validated left-arm
  clearance normally. Its open index finger then occluded every decodable
  primary-cube marker: offline replay detects primary IDs `10`, `13`, and `15`
  in all preflight frames and zero primary IDs in all five clearance frames;
  the secondary cube remained detected.
- The perception rejection was a normal `ValueError`, but the stack workflow
  allowed it to enter generic emergency cleanup. That cleanup correctly
  requested AI zero torque, yet it was the wrong policy for a healthy
  controller holding a known clearance endpoint with a frozen reverse route.
- Post-clearance camera-capture and cube-detection failures now become
  `TabletopTaskRejected` only after `driver.check()` confirms control remains
  healthy. The existing rejection path restores the initial finger postures,
  reverses the validated supported escape, and restores seated FSM 3. A real
  controller, transport, watchdog, state-observer, or unexpected software
  fault still enters the zero-torque cleanup path.
- Pair-observation errors now identify the uniquely tagged cube explicitly:
  secondary IDs `20-25` or primary IDs `10-15`.
- Verification passed Ruff over maintained `src` and `tests`, `429 passed,
  7 skipped` in the control environment, and `429 passed, 1 skipped` in the
  CUDA planner environment. The retained failed images were replayed through
  the updated detector path; no robot command was sent.

## 2026-08-21 — Direct-stack yaw is one CuRobo goal set

- The four upright cube symmetries were introduced as equivalent wrist goals,
  but the coordinator had implemented them as four separate endpoint analyses
  and up to four complete pick/place searches. That made a task-level symmetry
  a sequential Python planning policy.
- A pick/place request now carries a non-empty tuple of destination transforms.
  Ordinary transfers carry one transform. Direct stacking carries the upright
  quarter turns `(0, 1, 3, 2)` in one request.
- Destination endpoint IK is one CuRobo batch with one four-entry goal set per
  grasp candidate. CuRobo's returned `goalset_index` stays attached to each IK
  branch, the lowest-joint-travel branches enter the unchanged strict route
  checks first, and the selected destination index is hash-bound into the final
  nine-phase plan and feasibility artifacts. The coordinator submits one
  endpoint request and one complete planning request; it no longer orders four
  yaw-specific planners.
- Command-free replay merged the four retained requests from successful run
  `stack_20260821T174330Z`. The joint source/destination endpoint pass retained
  `17/57` grasp candidates. Full planning produced a valid nine-phase plan in
  `35.126 s`, selected candidate
  `cube_head__seed_0000000059__sample_162`, and CuRobo selected goal-set index
  `0` (nominal yaw `0`). No robot command was created or published.
- Verification passed Ruff, `430 passed, 7 skipped` in the control environment,
  and `430 passed, 1 skipped` in the CUDA planner environment.

## 2026-08-21 — Direct-stack grasp order retains endpoint joint distance

- The batched endpoint pass already produced up to 16 strict IK branches per
  grasp and ordered branches by Euclidean joint distance from the measured
  clearance state. The stack wrapper retained only branch counts, reconstructed
  the original GraspGenX shortlist order, and therefore did not compare that
  existing motion cost across common source/destination grasp candidates.
- Endpoint feasibility now retains each candidate's minimum joint distance.
  Common candidates are ordered by the sum of their best source and destination
  distances before unchanged full destination, source, transfer, collision,
  and table-plane validation. Ties preserve shortlist order. No wrist-specific
  weight, hard rotation gate, or additional route solve was introduced.
- Command-free replay of retained physical run `stack_20260821T181848Z`
  changed the first candidate from the physically rejected
  `cube_head__seed_0000000089__sample_168` to the later physically successful
  `cube_head__seed_0000000149__sample_155`. The rejected plan's
  clearance-to-pregrasp wrist changes were `+128.2/-14.3/+71.1 deg`; the newly
  selected plan uses approximately `+21.4/-17.8/+13.9 deg`. Its route-free
  source/destination joint distances were `0.8409/1.9171 rad`, versus
  `3.0871/2.9254 rad` for the rejected candidate.
- The replay planned the complete nine-phase route in `20.582 s` and selected
  destination goal-set index `2`. It created no robot command. Ruff passed;
  the control environment passed `431` tests with `7` skipped and the CUDA
  planner environment passed `431` tests with `1` skipped.

## 2026-08-24 — Rebuild from the physically successful Friday baseline

- `feature/friday-stack-rebuild` starts at `0a0fa0c`, retaining the complete
  57-candidate shortlist. Later experiments are preserved separately at
  `b6266d4` on `archive/post-friday-stack-experiments-20260824`; the invalid
  34-candidate release filter was not carried forward.
- LowState freshness now timestamps after the asynchronous observation. This
  removes the race that produced `now_monotonic_s precedes the state receipt
  time` without changing the 100 ms freshness limit or fault behavior.
- Every stack planning observation now combines measured body and Dex3 state
  with the exact 14-joint arm command being published. Initial plan
  installation preserves that command and retains the exact `1e-9 rad`
  trajectory-start invariant. This directly covers the retained
  `0.021020331 rad` measured-versus-command right-elbow failure.
- No waist motion, gain experiment, compact-hand experiment, or simulated
  release filter was included. The 57-candidate Friday shortlist remains
  unchanged.

## 2026-08-22 — MCAP-to-LeRobot conversion avoids high-rate over-decoding

- The commissioned converter dynamically decoded every 250-1000 Hz state
  message and read camera payloads in separate passes even though the LeRobot
  timeline is only the 15 Hz color stream. A 14 GiB episode spent `325.31 s`
  in indexing and `164.14 s` in its staged writer.
- The indexer now scans raw MCAP records in log-time order, retains the latest
  state sample, and decodes only the state/action values that can be selected
  by a color frame. On the same-size retained run, indexing took `11.67 s`.
- Exact comparison against the old implementation on retained run
  `stack_20260821T164205Z` produced the same 196 color/depth references,
  identical state/action arrays, and identical timeline bounds.
- RGB and depth are now fed through LeRobot's built-in
  `StreamingVideoEncoder` in one image pass. Retained full-run output hashes
  were identical between the preload and single-pass implementations. The
  12.95 GiB `seat_compliance_rigid_20260816T121420Z` conversion completed and
  reload-verified in `74.61 s` for 3,447 frames.
- Source raw deletion remains unchanged: a bag is deleted only after that
  episode is reloaded and verified and the replacement receipt is written.
- Archival conversion now omits individual color frames that cannot satisfy
  the unchanged state/action-age or RGB/depth-skew bounds, records every
  omitted timestamp and reason in `diagnostics.json`, and retains the original
  per-frame source time. This avoids discarding an otherwise valid episode for
  one or two bad frames without forward-filling stale commands or depth. A
  retained two-gap run reload-verified with 252 qualified frames and both
  rejected timestamps recorded exactly.

## 2026-08-24 — Restored strict observation and retained-control stack episodes

- Five-frame cube observation now accepts the largest pose-consistent subset
  containing at least three frames without relaxing the existing `5 mm / 2 deg`
  limits.
- Every episode searches both cube directions for each fixed-waist arm, in
  nearest-source order. Only one arm is lifted at a time; an unsuccessful arm
  returns through its frozen supported escape before the other arm is tried.
- A completed or safely rejected episode returns the selected arm to the
  supported start and keeps the camera, CuRobo worker, 250 Hz controller, PC2
  watchdog, and Dex3 transport alive. Each Space creates a normal independent
  `runs/stack_<timestamp>/` directory with its own MCAP and artifacts.
- Ctrl+C is a clean seated handback only while waiting between episodes.
  Interruptions or controller, DDS, watchdog, and unknown-state failures during
  an episode retain the existing Zero Torque cleanup.
- No waist motion, gain experiment, compact-hand orchestration, simulated
  release filter, or alternate shortlist was restored. The Friday 57-candidate
  shortlist remains unchanged.

## 2026-08-24 — Resting-cube all-hypothesis disambiguation

- Retained run `stack_20260824T163511Z` did not show physical cube motion. The
  secondary cube's largest marker was tag 24 on face `+Z` in all five loaded
  frames. Its single-face solve was internally consistent but falsely tilted
  `29.35--29.89 deg` from every possible resting face, with only
  `2.834--2.957 px` reprojection error.
- The already-supported multi-face solve on those same immutable images was
  `3.51--4.77 deg` from face-up. Smaller visible faces also supplied the
  physically correct planar branch. This isolates a planar ambiguity that
  additional identical-view frames cannot resolve if only one branch is kept:
  the wrong largest-face answer is systematic, not a temporal outlier.
- AprilCube now exposes a stateless all-hypothesis API. Every visible face
  contributes both positive-depth OpenCV IPPE branches, and a joint non-planar
  solve is added when multiple faces are visible. The library reports these
  candidates and their supporting tag IDs without embedding gravity, table,
  pose-history, or application-specific selection policy.
- Resting-cube observation decodes each frame once, removes candidates above
  the unchanged `3 px` reprojection or 20-degree resting-face limits, and then
  selects one candidate per frame in the largest subset passing the unchanged
  `5 mm / 2 deg`, minimum-three-frame burst consensus. Thus a lower-error wrong
  planar branch cannot defeat a slightly higher-error branch that agrees with
  gravity and the rest of the burst.
- The resting-face check uses the calibrated base-from-camera orientation at
  the exact command-bound snapshot. No table edge, torso-to-edge distance,
  planner scene, motion limit, or controller behavior changed.
- Command-free replay of the failed run now accepts all five loaded and all
  five post-clearance frames for both cubes. Loaded secondary/primary results
  are `5.35/4.28 deg` from face-up with `2.38 mm / 0.47 deg` and
  `1.01 mm / 0.80 deg` spread. Post-clearance secondary/primary results are
  `7.22/6.04 deg` from face-up with `1.93 mm / 0.47 deg` and
  `0.25 mm / 0.27 deg` spread.
