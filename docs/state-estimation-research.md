# G1 Camera State-Estimation Research Stack

## Scope

This stack answers one narrow question before changing robot control: given a
visually anchored camera pose, how much subsequent head-camera motion can the
G1 proprioceptive sensors explain?

The selected numerical observer is now a standalone pure-Python component. It
has no ROS, CuRobo, MPC, or robot-command dependency. It accepts exactly one
full 6D visual anchor, the three measured waist joints, pelvis orientation, and
torso orientation. The hardware timing adapter and every control consumer are
deliberately still absent.

The validation tools remain offline and read-only. They open an existing MCAP,
never create a Unitree publisher, and write reports under the ignored `work/`
directory. The first benchmark is the fixed-board rigid-chair run:

```bash
./tools/g1_state_estimation_research.sh \
  --run runs/seat_compliance_rigid_20260816T121420Z \
  --output work/state_estimation/rigid_20260816_report.json
```

The report hash-binds the board observations, URDF, and removable calibration
bundle. It uses MCAP record time as the common clock and fits a separate affine
mapping from each RealSense producer header clock to that clock. Raw camera
header seconds are not compared directly with Unitree receipt times.

### Implemented estimator boundary

`AnchoredCameraStateEstimator` retains only the empirically best
`hybrid_pelvis_position_torso_orientation` hypothesis. A visual measurement
resets all six pose dimensions. Between anchors, pelvis orientation plus the
three waist joints propagate camera position and the torso IMU propagates
camera orientation. The estimate reports anchor age and the explicit
fixed-pelvis-IMU-origin assumption.

Arm, hand, leg, command, RGB, raw D435i IMU, and depth inputs are not part of
this estimator. The pelvis-to-camera URDF chain contains only the three waist
joints. Depth remains a separately validated table-plane measurement because
the retained recordings have not yet demonstrated that fusing it improves the
hybrid estimate. Likewise, no numerical covariance is published: the retained
replays measured errors, but no covariance model has yet been calibrated and
validated. A future consumer must not invent either.

## What the rigid-chair recording proves

Repetitions 2 through 5 are evaluated because the retained physical run
predates explicit same-cycle `pre_lift` observations. Each lift is anchored to
the immediately preceding returned observation. Future diagnostic runs use the
explicit same-cycle pre-lift anchor automatically.

| Estimator | Left translation / rotation error | Right translation / rotation error |
|---|---:|---:|
| Camera assumed fixed | 5.355 mm / 0.707 deg | 5.260 mm / 0.790 deg |
| Measured waist FK, pelvis fixed | 1.963 mm / 0.221 deg | 3.314 mm / 0.429 deg |
| Torso IMU orientation, torso-IMU origin fixed | 2.154 mm / 0.123 deg | 2.663 mm / 0.104 deg |
| Pelvis IMU orientation + measured waist FK, pelvis-IMU origin fixed | 1.052 mm / 0.166 deg | 1.220 mm / 0.295 deg |
| Hybrid pelvis-based position + torso-IMU orientation | **1.052 mm / 0.123 deg** | **1.220 mm / 0.104 deg** |

The hybrid removes roughly 77--81 percent of the observed translation and
83--87 percent of the observed rotation at these endpoints. This is a
retrospective rigid-chair result, not yet a guarantee for other postures or
supports.

As an out-of-condition check, the same code was replayed on the cushion run.
It reduced `10.825 mm / 1.399 deg` to `1.920 mm / 0.139 deg` on the left and
`10.386 mm / 1.659 deg` to `1.963 mm / 0.135 deg` on the right. The roughly
two-times-larger remaining translation than on the rigid chair is consistent
with the fixed-pelvis-IMU-origin assumption becoming less accurate on a
compliant support. The posture mismatch between physical runs still prevents
using this as a clean material-identification experiment.

After aligning image headers to MCAP time, the command target had already been
static for `0.82--0.85 s` before each lifted observation. The residual is
therefore a loaded equilibrium, not a transient which can be removed by waiting
a little longer.

Endpoint accelerometer standard deviations are approximately 0.03--0.06
`m/s^2` per axis across the pelvis, torso, and D435i units. A 1 mm displacement
spread over several seconds corresponds to orders of magnitude less
acceleration. Double-integrating these accelerometers is not a credible source
of millimetre translation; they remain useful for attitude, bias estimation,
and short propagation between external observations.

## Continuous task-window replay

The retained RGB stream was also replayed through the same strict ChArUco
detector every tenth frame (about 1.5 Hz). The evaluation begins at the loaded
baseline and ends at the final recorded return. Frames after that boundary,
where the high-level seated controller is restored, are deliberately excluded.

The table below reports mean / 95th-percentile translation error. Requested
visual-update periods are rounded up to the next sampled image: the nominal
1-second period was 1.335 seconds in this coarse replay, the 2-second period was
2.002 seconds, and the 5-second period was 5.34 seconds.

| Chair | One loaded-baseline anchor for complete task | 1.335 s visual updates | 2.002 s visual updates | 5.34 s visual updates |
|---|---:|---:|---:|---:|
| Rigid | 4.661 / 5.589 mm | **0.371 / 0.873 mm** | **0.427 / 0.993 mm** | 0.581 / 1.261 mm |
| Cushion | 4.001 / 5.687 mm | **0.435 / 1.189 mm** | **0.497 / 1.389 mm** | 0.868 / 2.268 mm |

For the rigid run, the corresponding mean / 95th-percentile orientation errors
were 0.042 / 0.095 degrees at 1.335 seconds and 0.051 / 0.117 degrees at 2.002
seconds. For the cushion run they were 0.056 / 0.164 degrees and 0.061 / 0.166
degrees. These are offline propagation errors with a fixed board supplying each
fresh 6D anchor; they are not evidence that IMU and joints alone provide global
pose.

The continuous report can be regenerated without a robot connection:

```bash
./tools/g1_state_estimation_research.sh continuous \
  --run runs/seat_compliance_rigid_20260816T121420Z \
  --output work/state_estimation/rigid_continuous.json \
  --frame-stride 10
```

## Recorded depth-plane result

The native 640x480 `16UC1` depth stream and its recorded color/depth static TF
were evaluated independently. Depth points are selected only inside the
visually observed 180x270 mm board footprint and robustly fit to a plane. RGB
and depth are phase-shifted by one 15 Hz frame, so board poses are interpolated
in their shared D435i hardware-header clock instead of pairing callback receipt
times.

| Chair | Plane normal disagreement | Signed depth-minus-ChArUco offset | Absolute offset p95 | Plane-fit RMS |
|---|---:|---:|---:|---:|
| Rigid | 0.292 deg mean | -0.341 mm mean | **0.513 mm** | 0.827 mm |
| Cushion | 0.639 deg mean | -1.869 mm mean | **2.035 mm** | 0.746 mm |

The within-run offset standard deviation is only 0.104 mm on rigid support and
0.094 mm on the cushion. Depth is therefore a useful relative table-normal and
tilt measurement. The different absolute bias between runs also shows why it
must not be treated as a complete or globally unbiased camera pose: workspace
location, incidence angle, RGB-depth extrinsics, and RealSense depth bias are
all included. A plane still provides no in-plane X/Y position or yaw.

```bash
./tools/g1_state_estimation_research.sh depth-plane \
  --run runs/seat_compliance_rigid_20260816T121420Z \
  --continuous-report work/state_estimation/rigid_continuous.json \
  --output work/state_estimation/rigid_depth_plane.json \
  --frame-stride 10
```

## Estimator hypotheses

The pure estimator implementation is in
`src/g1_dex3_tabletop/state_estimation.py`. Every hypothesis begins from the
same fixed-board camera pose and synchronized body sample:

1. `fixed_camera` is the null model.
2. `fixed_pelvis_fk` keeps the pelvis frame fixed and uses measured waist
   joints plus the calibrated torso-camera transform.
3. `fixed_torso_imu_origin` rotates the torso from `/secondary_imu` and keeps
   the torso-IMU origin fixed. The known IMU-camera lever arm turns orientation
   change into a camera translation prediction.
4. `fixed_pelvis_imu_origin_fk` keeps the pelvis-IMU origin fixed, rotates the
   pelvis from `/lowstate`, and then applies measured waist FK.
5. `hybrid_pelvis_position_torso_orientation` uses hypothesis 4 for position
   and the direct torso IMU for orientation.

The fixed-origin statements are explicit contact hypotheses. They are
reasonable short-horizon models for the commissioned rigid seated setup but
are not universal constraints. They must be weakened or reset if the chair,
feet, pelvis, or support contact changes.

## Observability

| Quantity | Proprioceptive source | Is it observable without an external anchor? |
|---|---|---:|
| Torso roll/pitch change | Torso IMU gravity/orientation | yes, locally |
| Pelvis roll/pitch change | Pelvis IMU gravity/orientation | yes, locally |
| Waist articulation | Joint encoders | yes |
| Camera motion from those rotations and known lever arms | IMUs + URDF + calibration | yes, relative to an anchor |
| Camera height/normal distance from a fixed table | Depth plane | one translation component |
| Absolute X/Y position and yaw in the table/world | IMUs + joints alone | no |
| Complete drift-bounded 6D table/world camera pose | Visual landmarks, VIO/SLAM, or another external position/yaw source | yes, subject to measurement quality |

This is not specific to our implementation. Contact-aided invariant-EKF
observability analysis shows that absolute position and rotation about gravity
remain unobservable from IMU and contact measurements alone. A contact model
can make relative motion useful over a short interval, but it does not create a
global reference.

### Why Unitree odometry is not the present answer

Unitree's high-level `SportModeState` interface contains position and velocity
fields. The official low-level `LowState` used by this task contains the pelvis
IMU and motor states but no base position. More importantly, seated debug
`lowcmd` execution releases the high-level AI motion service, so a
`SportModeState` pose cannot be assumed to remain published, fresh, or valid in
the mode where this camera correction is needed. It was not recorded in either
chair MCAP. A future read-only probe can test it, but the research estimator
must not depend on it without that evidence.

`robot_localization` is also not the core estimator here. It can combine
already-defined odometry and IMU measurements, but it does not by itself model
an articulated waist, two body IMUs, chair/foot/hand contact modes, or their
observability. It may consume a finished odometry source later; it should not
replace the kinematic/contact model.

## Recommended research architecture

The estimator should ultimately publish a timestamped `reference_T_camera`
with covariance and provenance. It should not silently rewrite the static
camera calibration. The layers are:

1. **Clock and replay layer.** Deduplicate `/lowstate` by Unitree `tick`, use
   MCAP receipt time for Unitree streams, and map each RealSense header clock to
   receipt time separately. The rigid recording contains about 1,000 changed
   LowState ticks/s and a stable RealSense clock rate, but color and D435i IMU
   producer stamps have a systematic offset of about 68.6 ms from each other.
   Application callback receipt was about `102 ms` late at the median endpoint
   and reached `190 ms`; it is diagnostic metadata, not a fusion clock.
2. **High-rate proprioceptive prediction.** Use torso orientation directly,
   pelvis orientation, measured waist joints, the URDF IMU lever arms, and the
   calibrated torso-camera transform. This is the hybrid hypothesis already
   benchmarked offline.
3. **Soft contact updates.** In the rigid seated mode, treat pelvis position and
   zero velocity as low-variance hypotheses only while the contact state is
   known. Feet, chair, and table-hand contacts should have independent mode
   flags and covariances; none should be hard-coded as permanently fixed.
4. **Optional depth-plane update.** Fit the table plane in native depth only if
   a later replay demonstrates incremental value over the hybrid observer. Its
   normal constrains tilt and its distance constrains translation along the
   normal. It cannot determine in-plane X/Y or yaw on an untextured plane.
5. **External 6D update.** During research, the fixed ChArUco board provides
   direct ground truth and an update. For a general workspace, use visual
   landmarks or tested VIO/SLAM. For a task with a visible object, a fresh
   object observation is the strongest task-local update.
6. **Stage-boundary integration.** Initially update and replan only at safe
   stationary boundaries: after control acquisition, after a supported lift,
   and before descent. Do not continuously bend a frozen CuRobo trajectory
   until estimator latency, covariance, and closed-loop stability have been
   measured.

For the tabletop cube task, this becomes a concrete contract:

1. Obtain one fresh 6D task anchor from the cube or another fixed visual target
   after loaded arm ownership and settling.
2. Propagate camera pose at state rate with the hybrid pelvis-IMU-origin,
   measured-waist-FK, and torso-orientation model. Grow covariance with time
   and contact uncertainty; do not silently claim fixed-base kinematics.
3. Accept a fresh 6D visual update whenever the cube or fixed landmark is
   visible. The current evidence supports an initial target of one update every
   1--2 seconds on rigid support. If it is occluded, propagate and increase
   uncertainty rather than inventing an in-plane correction.
4. First consume the updated pose only at stationary task boundaries and replan
   the remaining CuRobo stages. Continuous trajectory deformation remains a
   later control experiment.

Native depth remains available as an independent table-normal check. It should
be added to this contract only after a fused replay shows lower held-out error,
not merely because the sensor is recorded.

The first production integration implements two deliberately discrete
boundaries. At the supported-lift boundary,
the fixed AprilCube is observed after loaded ownership and again at clearance.
Only the reversible escape exists before that motion. The second observation
replaces the camera/object and measured locked-body state, while the selected
arm coordinate is pinned to the exact escape command endpoint. CuRobo then
selects a grasp by validating its pregrasp route and unexecuted linear grasp
approach, but serializes only the reversible route to pregrasp. The payload
lifecycle is deliberately deferred. This corrects the measured loaded-to-lift
camera/body change without claiming continuous state estimation or permitting
the cube itself to move.

For the default trajectory controller, the clearance observation also becomes
the full visual anchor for `AnchoredCameraStateEstimator`. The live adapter
receipt-pairs and interpolates only the three waist joints, pelvis orientation,
and torso orientation at one monotonic timestamp. After the originally planned
clearance-to-pregrasp motion settles, the observer propagates
`object_T_camera` to that stationary boundary. The planner request retains the
original visual observation unchanged and adds a separate hash-bound estimated
planning-state record, including both synchronized-input timing records and the
exact current robot snapshot.

The persistent CuRobo worker must preserve the previously selected grasp. Its
fixed-shape open-hand optimizer and CUDA graphs survive the boundary; only
topology-compatible kinematic values and the world are updated. It may choose
another IK branch for that same physical grasp, and plans the
corrected-pregrasp, approach, retention-test, payload, replacement, and retreat
motions exactly once. The selected arm starts at the exact active command
rather than the tracking measurement. After replacement, the new plan returns
to that exact old pregrasp state and reverses the original
clearance-to-pregrasp samples to clearance. The original supported-escape
reverse then remains the final return. Thus a state-estimation or planning
rejection has a complete pre-existing return route and cannot silently switch
to a different grasp.

Planning may take long enough for the body state to change again. The default
trajectory path therefore obtains another synchronized estimate before
installation and compares it with the estimate that was planned. It reuses the
task's existing 5 mm / 2 degree perception-spread limits rather than adding
another uncommissioned threshold. The executor then independently checks
complete joint-state stability and exact command continuity during atomic
replacement.

The optional MPC path consumes the same estimator only during the local
pregrasp-to-grasp segment. The stationary cube pose at clearance defines an
arbitrary frozen reference; no separate board is required. Every rolling
window pairs one fresh cube image with a synchronized arm/body snapshot,
propagates the reference-to-camera transform, and updates the cube and
Cartesian grasp goal independently. The older input time remains the window
source time, so the unchanged 100 ms acceptance rule covers both inputs.
Global, payload, placement, and return motions remain frozen MotionGen
trajectories.

## Candidate upstream implementations

- Unitree's own G1 low-level example subscribes to both `rt/lowstate` and
  `rt/secondary_imu`, confirming that the second stream is the official torso
  IMU rather than a locally invented topic:
  <https://github.com/unitreerobotics/unitree_sdk2/blob/main/example/g1/low_level/g1_ankle_swing_example.cpp>
- Hartley et al.'s contact-aided InEKF supplies the right process/contact model
  and, importantly, the relevant observability result:
  <https://arxiv.org/abs/1805.10410>
- OpenVINS is a suitable visual-inertial research reference. Its documentation
  emphasizes camera-IMU extrinsic and time calibration, and its estimator can
  estimate camera time offset:
  <https://docs.openvins.com/gs-calibration.html>
- NVIDIA Isaac ROS Visual SLAM has an official D435i stereo-inertial launch
  using both infrared cameras and the combined IMU stream. It is GPU-accelerated
  and worth an offline comparison, not automatic adoption:
  <https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_visual_slam/blob/main/isaac_ros_visual_slam/launch/isaac_ros_visual_slam_realsense.launch.py>
- RTAB-Map supplies ROS 2 RGB-D odometry and accepts an IMU topic for VIO inputs
  and gravity constraints:
  <https://github.com/introlab/rtabmap_ros>

Generic VIO benchmark claims do not establish millimetre accuracy for our slow,
small, seated motion. Each candidate must replay the same rigid-chair data and
be scored against the ChArUco trajectory before it can replace or augment the
hybrid observer.

## Next experiments

1. Run the corrected diagnostic once on the rigid chair so every cycle has an
   explicit pre-lift anchor.
2. Repeat the continuous replay at full 15 Hz only if a later design decision
   needs denser statistics; the coarse pass already resolves the architecture.
3. Calibrate the D435i camera-IMU transform and timing before benchmarking VIO.
4. Compare the hybrid observer, RGB-D odometry, and stereo-inertial VIO on the
   identical replay. Promote no estimator without cross-run validation on
   changed arm poses and both arms.
