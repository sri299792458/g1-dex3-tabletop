# Phase-aware tabletop MPC

After the reversible supported escape and fixed-cube boundary replan, the
tabletop workflow can execute every remaining normal arm motion with CuRobo MPC
instead of replaying each frozen trajectory verbatim. The boundary-corrected
frozen lifecycle supplies the selected grasp, a complete collision-checked
route, exact phase boundaries, and reverse recovery trajectories. MPC uses that
route as its local reference and replans short arm-command windows from fresh
measured joint states. The outbound supported escape and its final exact reverse
remain frozen trajectories because both touch the physically supported handoff.

The isolated CUDA worker never publishes robot commands. Each returned window
is hash-bound to the frozen execution plan and then installed atomically in the
existing 250 Hz controller. That controller alone interpolates and publishes
the complete low-level command.

## Physical phase model

One collision model cannot correctly describe the complete task because the
cube changes physical role:

| Motion endpoint | Finger/cube state | MPC collision model |
| --- | --- | --- |
| `clearance` | initial fingers; cube on table | frozen supported escape, not MPC |
| `move_to_pregrasp` | open hand | cube and local table patch are world obstacles |
| `grasp_approach` | open hand entering contact | designated fingertip links may contact the cube; table and all self-collisions remain checked |
| `retention_test_lift` | measured stable-close fingers | cube is a 27-sphere payload attached to the grasp frame |
| `payload_lift` | same measured close | reuses the warm attached-payload model |
| `payload_lower` | same measured close | reuses the warm attached-payload model |
| `payload_replace` | same measured close | reuses the warm attached-payload model |
| `grasp_retreat` | cube released; hand open | contact retreat with cube fixed in the world again |
| `return_to_clearance` | open hand | cube and local table patch are world obstacles |
| `__handoff__` | initial fingers restored | frozen exact supported return, not MPC |

The worker retains one warmed controller for all eight MPC-controlled phases.
The active seven arm joints, tensor sizes, collision-scene capacity, and CUDA
graph addresses never change. At each phase boundary it copies pre-resolved
fixed-size kinematics and collision values into that controller, toggles the
already allocated cube/table/fixture obstacles, and installs or removes the
payload in reserved sphere slots. Returning to open-contact or open-free mode
restores that mode's last action seed, preserving the useful reverse-path warm
start without constructing another solver or CUDA graph.

Initial and deterministic open-finger kinematics are resolved before setup.
The measured close posture cannot be known earlier; it is resolved once after
the physical close stabilizes and then reused for all four connected payload
motions. No geometry is approximated to make the switch cheap.

For a presentation fixture, the rolling optimizer does not repeatedly query
the triangle mesh. Every route entering MPC was already planned against that
mesh, and every returned window is checked against it independently on CUDA.
Open-hand windows must retain the existing 10 mm activation margin. During
attached pickup/replacement, where the cube intentionally contacts the
fixture, every robot sphere must remain non-penetrating while only the
attached-object proxy is excluded. This preserves the frozen planner's
physical policy without evaluating the same mesh 100 times per rolling solve.

Finger opening, stable-close acquisition, retention validation, release, and
seated-control restoration remain discrete operations in the existing task
state machine. MPC does not infer or change those transitions.

Every candidate MPC window receives one full FK evaluation. The resulting
spheres are reused for strict full-robot self-collision, cube clearance,
selected wrist/hand table-plane clearance, attached-payload clearance, and the
optional exact presenter-mesh check. Fixture diagnostics record the closest
link and clearance. A window marked infeasible never enters
the command buffer. A planner or controller failure follows the existing
fail-closed PC2 takeover path. Normal grasp rejection uses the already-frozen
reverse recovery trajectories rather than MPC.

CuRobo retains its 10 mm collision-cost activation distance. The independent
hard clearance-to-object check is 5 mm for open-hand transit; entering the
outer optimizer cost band is therefore not mislabeled as a physical collision.
The optimizer and strict checker both use the cube's exact physical dimensions.
The 5 mm requirement exists only in the independent route/window validation; a
route below it is rejected rather than changing the object's optimizer geometry.

Non-supported phases require the request's 5 mm selected-hand/table margin;
`> 0 mm` is not considered execution-ready. Supported motion retains its
separate start-relative policy because the hand begins and ends physically
supported, while the resting/just-attached cube is intentionally allowed to
touch its support plane.

## Hardware selection

The default remains deterministic frozen-trajectory execution. Select MPC
explicitly with one additional argument:

```bash
./tools/g1_tabletop_hardware.sh run-tabletop \
  --object-profile cube60-r3 \
  --presentation direct \
  --arm left \
  --motion-controller mpc \
  --network-interface enp134s0 \
  --calibration-bundle config/calibrations/dex3_shared_20260812_selected_free.json \
  --confirm 'I CONFIRM THE G1 IS SECURED BY THE LOAD-BEARING HARNESS AND THE WORKSPACE IS CLEAR'
```

The run stores `mpc_lifecycle.json` beside the existing plan, perception,
retention, status, planner log, and raw MCAP artifacts. It records preparation
time and every validated window for each phase. It is written only after robot
ownership is resolved, including on a failed or interrupted run; this preserves
the last rejected window without adding file I/O to the live control period.

## Frozen-scene baseline evidence

Before continuous body correction was added, retained tripod run
`tabletop_20260818T173706Z` was replayed without ROS or robot commands. The full
clearance-boundary planner selected the same grasp as the successful hardware
run, and the benchmark consumed that run's actual measured `grasp_close.json`
rather than substituting the unreachable descriptor close target. All eight
MPC-controlled phases reached their frozen endpoints through 179 accepted
windows with zero rejected windows. The largest terminal joint error was
0.00429 rad and maximum velocity was 0.09877 rad/s against the unchanged
0.1 rad/s contract. The supported escape and its exact reverse are the two
deliberately frozen trajectories outside this count.

The current pinned optimizer advances in 25-iteration inner blocks. Fifty
iterations was fast but produced a 0.10569 rad/s window that the velocity
validator correctly rejected. Seventy-five iterations was the lowest tested
valid setting, but the configured controller now deliberately uses 100 warm
iterations for additional optimization effort. The timing measurements in the
continuous-correction section include the later fixture-query correction; they
supersede the older 102 ms result obtained while the exact tripod mesh still
participated inside every optimizer iteration.

That frozen-scene replay took 20.31 s of command-free compute. Initial solver
and CUDA setup took 10.94 s, measured-close kinematics and attached-mode
prewarm took 1.81 s, and cached phase switches took less than 10 ms after their
first use. Rolling solves overlap physical motion. Its largest window was
84.81 ms. The continuous-correction measurements below supersede that timing
result for the current controller.

An independent audit rebuilt each former physical-mode robot from its source
configuration and compared it with the in-place model. Fixed transforms, joint
maps, collision topology, self-collision padding, locked joints, and payload
spheres matched exactly; open-hand sphere radii differed only by float32 roundoff
(maximum 3.73e-9 m). The strict full-robot checker remains separate and is
updated from the same resolved finger posture before every window check.

## Continuous body/camera-state correction

The clearance cube observation supplies one fixed six-dimensional task anchor.
Before every rolling window, the hardware process synchronizes the measured
arm state with fresh pelvis orientation, three waist joints, and torso IMU
orientation. `AnchoredCameraStateEstimator` propagates the calibrated camera
pose relative to that stationary anchor. The worker then expresses both the
local route goal and the fixed cube/table/fixture scene in the live body frame.
The correction record and its source timestamps are hash-bound into the
returned window.

The seven-joint G1 arm is redundant, so Cartesian target updates alone are not
enough: retained replay lost the validated joint branch, took 245 windows for
one leg, and eventually failed the cube check on retreat. Each corrected local
goal therefore uses CuRobo retargeting IK seeded by its matching frozen-route
waypoint before the rolling solve. The final endpoint must satisfy the existing
5 mm / 0.05 rad Cartesian tolerances and the existing 0.005 rad corrected-joint
contract.

Profiling separated the complete corrected window into goal/body update,
CuRobo optimization, and strict checking. The estimator and goal update cost
about 8--9 ms; the independent exact checks cost about 3 ms. Repeating the
tripod triangle-mesh query inside every optimizer iteration was the dominant
avoidable cost. Keeping that mesh in the exact post-check reduced three
75-iteration eight-phase replays to worst windows of 57.72, 51.72, and
53.16 ms. The subsequently requested 100-iteration configuration completed a
full replay in 161 windows with zero rejections; its worst complete window was
50.82 ms and maximum velocity was 0.08808 rad/s. The 0.1 rad/s speed, 100 ms
source-age, 10 mm activation-distance, and collision-geometry contracts remain
unchanged. Every open-hand window remained at least 10 mm from the fixture;
attached pickup/replacement reached 4.46/3.27 mm without penetration, matching
their pre-existing support-contact policy.

This is continuous proprioceptive body/camera correction, not continuous image
tracking. The cube and fixture must remain stationary after the clearance
anchor. It is also still an offline retained-run result, not a hardware MPC
commissioning result. Commanded waist yaw remains disabled.

## Clean route controller and future table anchor

The controller retains one simple separation of responsibilities:

- the boundary planner supplies a collision-validated joint route and its IK
  branch;
- MPC follows a monotonic one-horizon lookahead on that route from each fresh
  measured arm state;
- the command boundary receives the simultaneously active arm command and
  preserves that desired-to-measured tracking offset for the complete returned
  window, then remeasures it on the next rolling update; and
- the independent strict checker validates both CuRobo's predicted measured
  path and the translated command path before either can be installed.

The temporary one-window, continuous route-projection, and direct-phase-endpoint
experiments were removed. They were introduced after the first physical MPC
window failed, but that failure was only a command-boundary discontinuity; it
did not invalidate the already replayed frozen route.

Retained physical run `tabletop_20260819T002307Z` provides the exact regression.
Its active command and measured elbow differed by `0.02009 rad`. Directly
joining the measured-state plan would require `0.12558 rad/s`; the complete
window translation removes that edge without erasing the low-level controller's
existing holding effort. The resulting window is CuRobo-feasible and passes
every strict collision check. No robot command was sent for this replay.

The former clean-route benchmark completed the retained eight-phase lifecycle,
but its ideal plant set `measured == command` after the first window. It
therefore validated geometry, phase switching, and timing, not persistent
gravity/load tracking error. A persistent-offset replay is a separate required
regression.

This boundary follows the same desired-state-continuity principle used by
rolling trajectory controllers. CuRobo emits a trajectory for a downstream
controller to interpolate and track
(<https://github.com/NVlabs/curobo/discussions/681>). ROS 2's joint trajectory
controller explicitly offers `interpolate_from_desired_state` for MPC-style
successive goals
(<https://github.com/ros-controls/ros2_controllers/blob/master/joint_trajectory_controller/src/joint_trajectory_controller_parameters.yaml>),
and MoveIt Servo advances from its buffered future command while that command
remains valid
(<https://github.com/moveit/moveit2/blob/main/moveit_ros/moveit_servo/src/servo_node.cpp>).
The local implementation does not import those controllers: it adapts their
continuity invariant to the already commissioned Unitree 250 Hz lowcmd PD and
gravity-feedforward loop.

The benchmark now supports a fixed simulated tracking offset across every
window. With the median offset recorded by physical run
`tabletop_20260819T121046Z`, the continuity policy advanced the first phase for
30 accepted windows to frozen-route index 36 instead of repeating index 6/7.
It then rejected a `0.060 mm` translated-command sphere overlap between
`left_wrist_pitch_link` and `torso_link`. A temporary command-governor test was
not retained: it merely shifted the failure until CuRobo's predicted physical
path crossed the same strict pair by `0.045 mm`. This distinction is important:
desired-state continuity fixes the observed no-progress loop, but does not turn
a near-zero-clearance frozen route into a robust physical route. The strict
collision boundary remains unchanged and fails closed.

The camera-state input is deliberately a generic `reference_T_camera`
transform; the MPC controller does not know which visual target produced it.
Today that transform is anchored by the cube, so the cube must stay fixed. For
moving-object work, a fixed table marker will become the source of
`reference_T_camera`, while a separate live cube observation updates the task
goal. That later perception/goal update must not change the command buffer,
continuity bridge, route-branch policy, or strict validation boundary described
above.
