# Dex3 calibration redesign

Status: implementation specification. The existing single-arm workflow remains
available only until this replacement passes the replay and hardware gates at
the end of this document.

## Decision

Replace the focused repository's current calibration domain and workflow with a
same-frame bilateral calibration system. Preserve the generic evidence,
optimizer, robot-control, and motion-planning foundations. Do not modify the
pose executor, its 0.2 rad/s command limit, settling gates, gravity
feedforward, CuRobo collision checking, or the deployed calibration bundle as
part of the rewrite.

This is a clean implementation, not a port of the old CLI. The prototype at
`/home/kanth042/robot-calibration-aprilcube-prototype` is the evidence and
algorithm reference; it is not a package dependency.

## What the audit found

The prototype contains a coherent, lossless single-arm pipeline:

1. A capture is accepted only after image/state pairing, stationarity, and
   target-quality checks.
2. Raw PNGs, the full 29-joint state, timestamps, configuration hashes, and
   correspondence hashes are retained.
3. Dataset construction re-runs the detector and verifies the correspondence
   hash rather than trusting derived observations.
4. Mike Ferguson's `robot_calibration` at commit
   `db991b040d1dc28af09d8865fc72f09720e12b73` is the sole optimizer.
5. Reprojection, held-out error, bootstrap variation, and numerical
   observability are evaluated independently of the optimizer.
6. Results are deployed as a removable, hash-bound overlay rather than by
   editing the base URDF.

The prototype also contains a later bilateral model-comparison tool. That tool
fits a shared camera, a separate hand-target transform for each arm, and
selected static joint zero offsets. It merges independently captured left and
right sessions; it does not use simultaneous left/right observations. Its
model selection and bundle export were never promoted into the production
workflow.

The focused repository retained the strong evidence and execution machinery
and added CuRobo feasibility. Its current production calibration path,
however, regressed to a one-arm model:

- `design-calibration` generates one marker pose in camera space and scores a
  six-parameter camera information matrix.
- `CalibrationPlanRequest` and `CalibrationPlanResult` contain one active arm
  and one marker transform.
- `collect-calibration` moves one arm while holding the other and runs only the
  selected arm's detector.
- `CalibrationDataset` represents one target observation per frame.
- `solve-calibration` fixes the hand-target transform and supplies no free
  joint offsets. It therefore estimates only the six camera-mount parameters.
- Its bundle writer cannot reproduce the installed bundle, which contains
  selected offsets for both arms.

The executor state machine is not involved in that model regression.

## Evidence from the retained runs

The retained right-arm raw session has 448 accepted frames with marker 4 and
no marker 5 detections. The retained left-arm raw session has 287 accepted
frames, 276 with marker 5 and no marker 4 detections. No accepted frame in
either session contains both hand markers. The solver datasets contain 62
right samples and 41 left samples.

The old joint poses cover useful ranges but contain strong correlations. For
example, right shoulder pitch and yaw have approximately -0.90 correlation.
The standardized joint-design condition numbers are approximately 7.71 on the
right and 8.76 on the left. A newer CuRobo-generated right-arm dry plan has a
better condition number, approximately 4.44, but that is incidental because
the selector does not score joint excitation.

The prototype's model study selected a shared-camera model with optimized
left/right target transforms and selected joint offsets. Its held-out radial
RMS was about 5.41 px overall (5.16 px left and 5.57 px right). Joint state
predicted substantial right-arm residual energy on held-out data, while
learning curves largely plateaued after roughly 30--40 samples. Therefore,
merely collecting more frames from the old pose distribution is not the main
fix.

Freeing every arm joint is not supported by the data. With a planar hand
target, shoulder-pitch/camera-pitch and wrist-yaw/target-orientation directions
contain structural gauges. Parameter sets must be declared and tested for
rank; the solver must reject an unobservable model rather than regularize it
until it looks plausible.

The serial-specific RealSense frame chain was measured live on 2026-08-24 with
the repository's guarded PC2 camera handoff. For serial `348522074178`,
`camera_link -> camera_color_frame` contains translation
`[-0.000503254, 0.014834855, -0.000375290]` metres and approximately
`[0.65383, 0.25072, -0.29239]` degrees RPY. The following optical-frame
rotation has zero translation. This chain must be a frozen input artifact. Its
earlier omission did not invalidate the fully free direct torso-to-optical fit;
it made the fitted external camera correction physically ambiguous.

## Preservation boundary

Keep these foundations:

- camera intrinsics and transform utilities;
- `RobotStateSample`, the G1 joint map, clocks, and ROS state/camera buffers;
- image/state pairing, readiness, stationarity, and quality gates;
- immutable raw-image and state-window storage concepts;
- the pinned Ferguson backend and independent projection diagnostics;
- the removable `CalibrationBundle` consumer and materializer;
- CuRobo, the pose executor, control watchdogs, gravity compensation, and Dex3
  posture preparation;
- all retained raw sessions, derived datasets, reports, and the currently
  deployed bundle.

Replace these calibration-specific interfaces:

- `calibration_candidates.py`;
- `calibration_workflow.py`;
- the calibration request/result types embedded in `planning/contracts.py`;
- calibration-specific CuRobo entry points in `planning/curobo_backend.py`;
- the calibration orchestration in `hardware_calibration.py`;
- the single-target dataset contract and camera-only solve/export path;
- the corresponding calibration CLI commands and tests.

The old files are deleted only after the new offline pipeline can rebuild a
bundle from immutable evidence. This avoids losing the only executable path
while the replacement is incomplete; Git still makes the deletion recoverable.

## New observation contract

One accepted sample represents exactly one camera frame paired with exactly one
full robot state. It contains two target observations:

- `left`: target configuration hash, detected IDs, ordered object points,
  ordered image points, and correspondence hash;
- `right`: the same fields for the right target.

Both observations are mandatory. They share:

- capture and frame IDs;
- raw-image SHA-256;
- camera-info artifact/hash;
- full 29-joint position and velocity state;
- image and state timestamps plus pairing diagnostics;
- pose/anchor group identity;
- URDF, target, hardware, detector, and code provenance.

The raw burst remains the source of truth. Dataset construction re-detects both
targets from the selected raw frame and verifies each correspondence hash.

The medoid is selected over the concatenated, image-normalized left and right
corner coordinates. A frame cannot be accepted because one target is good
while the other is missing or unstable.

## New experiment design

CuRobo remains the geometry authority, but it does not choose statistically
informative samples. Candidate selection operates on feasible paired arm
configurations and the declared calibration model.

The lifecycle has a reusable frozen core and a short live adapter:

1. Read the current stationary normal Ready posture and plan only a reversible
   right-then-left shoulder-clearance adapter for that run.
2. At bilateral shoulder clearance, command both hands to the descriptor-defined
   fixed full close, require the commissioned empty-close measurements, and
   certify the complete closing sweep.
3. Follow the run-specific adapter to the fixed collision-free visual anchor.
4. Execute the offline anchor-to-anchor core: alternate short left- and
   right-arm excitation blocks, returning to the
   same bilateral anchor between blocks.
5. Reverse the adapter to bilateral clearance, command and verify the
   commissioned open posture, then execute the exact reverse run-specific
   shoulder trajectories to the measured Ready state before releasing ownership.

Every capture sees both targets. Each arm must receive independent variation in
all intended-to-be-estimated joint directions. Candidate scoring combines:

- the incremental log determinant of the full declared-model residual
  Jacobian;
- smallest singular-value/rank protection;
- normalized joint range coverage;
- a correlation penalty for redundant joint motion;
- image coverage, depth, incidence, and target-separation terms;
- repeated-anchor allocation for temporal drift estimation.

IK and collision failures filter candidates; they do not redefine the
statistical objective. A route is accepted only if its post-CuRobo selected
set still passes the numerical rank and condition gates.

## Solver model

The initial production model is:

- one shared six-degree torso-to-camera mount correction;
- fixed factory intrinsics;
- a separate optimized hand-to-target transform for each arm;
- an explicitly selected subset of static joint zero offsets for each arm;
- Gaussian-equivalent priors only on selected joint offsets;
- no learned pose-dependent correction.

Each same-frame sample is one Ferguson `CalibrationData` message with four
observations:

- `left_arm`;
- `left_camera`;
- `right_arm`;
- `right_camera`.

The two camera model aliases use the same optical frame and the same camera
parameter name. This lets the stock Ferguson optimizer create two reprojection
blocks in one sample while sharing camera extrinsics and intrinsics. No
upstream C++ fork is required. Reusing one camera sensor name would be wrong:
the backend selects the first matching observation, but the left and right
pixel feature arrays differ.

The solver runs a declared set of nested candidate models. Selection requires:

- full numerical rank under a stated tolerance;
- a bounded condition number;
- improved grouped held-out error across both arms;
- stable bootstrap signs/magnitudes;
- plausible physical values;
- no material regression on either arm or repeated anchors.

The selected model is exported directly to the versioned bundle. Manual JSON
curation is not a production step.

## Anchors and pose-dependent error

Seeing both hand markers in one frame is valuable because it couples both arm
residuals to the same camera image, robot state, and time. It removes the
between-session camera/drift ambiguity present in the old bilateral study. It
does not establish absolute torso-to-camera truth or remove every
joint/target-frame gauge.

External table or torso markers are outside this implementation. Anchors mean
repeated bilateral robot configurations observed through the two hand targets.
They measure temporal repeatability but are not claimed as external absolute
ground truth.

Static zero offsets should be commissioned before any pose-dependent FK model.
The new dataset deliberately records enough joint variation to characterize
held-out residual versus pose and load. A pose-dependent correction may be
considered later only with new independent physical evidence and a later-day
holdout; fitting image residual alone can simply transfer error among camera,
target, and kinematic parameters.

## Migration stages

1. Introduce a central `g1_dex3_tabletop.calibration` package with the bilateral
   observation, dataset, experiment, solver-model, and report contracts.
2. Add a pure offline converter that can preserve the two legacy datasets as
   unpaired diagnostic inputs, while refusing to label them simultaneous.
3. Implement the four-observation Ferguson export and independent bilateral
   projection/observability evaluation.
4. Implement full-model candidate scoring, then pass candidates through
   CuRobo and re-check the selected design.
5. Implement two-target same-frame capture and replay verification.
6. Collect a pilot route with repeated anchors; compare declared models on
   grouped holdout folds.
7. Export a candidate bundle and replay manipulation validation without
   replacing the deployed default.
8. After all gates pass, switch the CLI to the new package and delete the old
   single-arm calibration workflow and its calibration-only contracts.

Stages 1, 3, 4, and 5 now have executable implementations. The offline stage-4
entry point is:

```bash
./tools/g1_tabletop.sh plan-bilateral-calibration \
  --snapshot /absolute/path/to/ready_snapshot.json \
  --output-directory /absolute/path/to/bilateral_plan
```

The snapshot is a stationary reference used only to generate the reusable core;
hardware runs are not required to reproduce its arm commands. The command
creates no ROS node or robot publisher. It runs collision-aware batched IK with the
commissioned empty-close hand model. It chooses a both-visible bilateral anchor,
filters all remaining same-frame-visible poses through one batched CuRobo
joint-space graph, and scores only the anchor-rooted component using the
complete declared bilateral model. It then certifies the complete selected-pose
edge graph, freezes minimum-motion-cost anchor tours for interleaved blocks,
and certifies every edge of a closed-hand anchor-to-anchor core.
There is no select-fail-reselect loop and no motion-optimizer construction for
each capture edge.

The output directory retains the hash-bound planning request, IK result, the
connected route request/result, `pose_design.json`,
and `execution_plan.json`. The last artifact binds the fixed full-close command,
commissioned empty-close collision model, joint offsets, and core clearance
certificate. Hardware collection accepts only the mutually bound pose design
and execution plan, then asks the isolated CuRobo worker for a small live
Ready-to-anchor adapter before creating any command publisher.

During hardware collection, `q` requests a graceful early finish. It never
interrupts an active arm trajectory: the request is latched at the stationary
capture boundary, remaining captures are skipped until the next repeated
anchor. The run-specific adapter then returns to bilateral shoulder clearance,
opens both hands, returns both arms to the same measured Ready state, and
releases ownership cleanly. `Ctrl+C`, watchdog trips, tracking
faults, and failed return motion still invoke Damp.

## Required verification gates

- Every accepted new sample contains both targets from one raw image and one
  paired full state.
- Rebuilding a dataset from raw evidence produces the same content hash.
- Ferguson receives four uniquely named observations per sample and two
  reprojection blocks sharing one camera parameterization.
- The selected free-parameter Jacobian is full rank and below the commissioned
  condition limit on training and bootstrap resamples.
- Pose-group and day-group holdouts are reported separately for left, right,
  and combined residuals.
- Repeated anchors bound temporal drift; an A/B/A excursion that exceeds the
  commissioned translation or rotation threshold invalidates the run.
- The generated bundle exactly lists the fitted parameters, their units,
  source hashes, uncertainty diagnostics, and the measured RGB optical-frame
  artifact.
- Existing executor, controller, CuRobo, and tabletop manipulation tests remain
  unchanged and passing.
- The deployed bundle remains available until the new bundle passes offline
  replay, collision validation, a low-speed dry run, and later-day validation.
