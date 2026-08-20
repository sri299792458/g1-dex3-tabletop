# Moving-target grasp MPC

The optional MPC path has one job: update the open-hand motion from pregrasp
to grasp while the cube pose changes. It does not replace CuRobo MotionGen for
global motion, payload motion, placement, or return.

This division is deliberate:

- MotionGen searches globally for collision-free routes.
- CuRobo MPC solves short local Cartesian corrections from the trajectory that
  is already executing.
- The existing task state machine owns Dex3 opening, closing, grasp evidence,
  attachment, release, recovery, and seated-control restoration.

The default `trajectory` controller is unchanged. The moving-target path is
selected explicitly with `--motion-controller mpc`.

## Perception contract

Moving-target operation uses the AprilCube for two observations with different
roles:

1. The stationary cube observation at clearance defines an arbitrary reference
   frame that is then frozen in software.
2. One new rectified RGB frame supplies the changing cube pose for each MPC
   update.

Pelvis orientation, the three waist joints, and torso IMU orientation propagate
the reference-to-camera pose after the clearance observation. Each new cube
detection is transformed into that frozen frame, separating the estimated
camera/body motion from the observed object motion.

The cube must remain stationary until its clearance reference has been
recorded. It may move after the program reports that moving-target MPC is
ready, and it must remain detectable during the open-hand MPC approach. Losing
the visual target is a planning failure; the controller does not guess that an
unobserved cube stayed still. No separate ChArUco board is required. This does
not add external measurement of unmodelled seat translation; doing that would
require a continuously observed fixed workspace landmark rather than the old
one-time board sample.

## Physical lifecycle

The complete single-cube task remains:

1. Plan and execute the reversible supported escape.
2. Open the selected Dex3 hand at clearance.
3. Observe the stationary cube, freeze its reference frame, then plan the
   complete nominal lifecycle.
4. Execute the MotionGen clearance-to-pregrasp route.
5. Run moving-target MPC only from pregrasp to grasp.
6. Close the Dex3 hand immediately at the reached grasp.
7. Rebuild the fixed-close and attached-payload lift at the reached cube pose.
8. Validate the measured physical close against that exact payload route.
9. Execute the retention lift, payload lift, reverse placement, release, and
   return with frozen MotionGen trajectories.
10. Reverse the supported escape and restore seated control.

The return after a successful moving grasp is not the stale nominal return. It
is the exact reverse of the newly planned payload lift, followed by the exact
reverse of the MPC command path that the controller accepted, followed by the
reverse of the original clearance-to-pregrasp route.

Before post-grasp CUDA planning begins, the exact accepted MPC reverse is
installed as the provisional recovery route. A continuation-planning rejection
can therefore open the hand and return without replaying the old nominal grasp
approach.

## MPC update

Each update performs the following transaction:

1. Collect one new cube image and its receipt time.
2. Pair it with a bounded-age body-state sample.
3. Predict the arm command and measured state at a future immutable handoff.
4. Express the current tool and the nominal approach in the live cube frame.
5. Advance monotonically along that object-relative approach.
6. Retarget the matching seven-joint branch to a short Cartesian lookahead.
7. Ask CuRobo MPC for one complete decelerating horizon.
8. Validate both the predicted path and the actual translated command path.
9. Install the window only if its predecessor, future boundary, measurements,
   and hash still match.

The object-relative progress calculation is essential. Comparing the corrected
joint state with the old nominal joint route caused progress to freeze whenever
the cube moved, even though individual MPC solves succeeded.

The complete CuRobo `robot_state_sequence` is retained. Its first state is the
supplied future boundary and its tail decelerates to zero. No sample is changed
after collision validation and no arrival-time clock reset is performed.

## Collision semantics

The open-hand approach still checks full-robot self collision, the table, the
presentation fixture when present, and the cube.

During final object contact, the cube alone may overlap any of the seven
movable Dex3 finger links. This is required for a physical grasp: proximal
finger links can legitimately contact a cube. It does not disable palm, wrist,
table, torso, or self collision. Presentation-fixture contact remains limited
to the existing three distal contact links; the broader object-contact rule is
not reused for the fixture.

Every accepted MPC window is checked on the executor grid before it can be
published. Finger motion is not controlled by MPC.

## Failure behavior

An infeasible or late MPC result never replaces the active certified horizon.
The executor finishes the unchanged horizon and holds its endpoint. A rejected
physical close or post-grasp continuation opens the hand and uses the already
installed exact reverse. Controller, transport, or watchdog failures retain the
commissioned fail-closed ownership path.

## Current evidence and limitation

The composed retained GPU replay moved the cube by 5 mm after the original task
was planned. The repaired controller reached the live grasp in 81 accepted
windows with no rejected window or collision. Simulated approach duration was
20.00 s; terminal error was 3.70 mm and 0.95 degrees, inside the existing 5 mm
and 0.05 rad contracts. It then preserved the same grasp, rebuilt the payload
continuation in 2.01 s, and produced the exact accepted-approach reverse.

The worker now launches when the hardware command starts. As soon as read-only
preflight has supplied the selected arm and object topology, it constructs the
persistent MotionGen models and a provisional moving-grasp MPC while the live
preview and SPACE prompt remain active. No robot publisher exists during that
work. A retained 60 mm replay measured 17.84 s for this command-free startup.
After the live clearance plan was known, rebinding that same MPC to the exact
route and current scene took 1.50 s instead of constructing another roughly
6--11 s MPC instance. The exact live supported escape and global pregrasp route
still require planning after ownership; their retained replay times were 3.55 s
and 10.63 s respectively. Those are state-dependent searches, not startup
construction, and have not been hidden or replaced with stale preflight plans.

The production worker exposes only this moving-grasp MPC. The obsolete
multi-phase route-tracking benchmark was removed: payload, replacement,
retreat, and supported return remain frozen MotionGen trajectories.

This is an offline retained-run result, not a physical moving-cube
commissioning result. The remaining approximately 2 s after closing is genuine
live-pose payload planning and validation while the controller holds the
grasp. It is no longer repeated model construction, and removing it would need
safe concurrency with Dex3 closure rather than another cache or looser check.

## Invocation after commissioning

For a staged physical commissioning run after explicit review, the interface
is:

```bash
./tools/g1_tabletop_hardware.sh run-tabletop \
  --arm left \
  --object-profile cube40-r3 \
  --presentation direct \
  --motion-controller mpc \
  --network-interface enp134s0 \
  --calibration-bundle config/calibrations/dex3_shared_20260812_selected_free.json \
  --confirm 'I CONFIRM THE G1 IS SECURED BY THE LOAD-BEARING HARNESS AND THE WORKSPACE IS CLEAR'
```

Do not use that command on hardware merely because the interface exists. The
clearance-anchor and cube-visibility requirements and the current commissioning
status above are part of the contract.
