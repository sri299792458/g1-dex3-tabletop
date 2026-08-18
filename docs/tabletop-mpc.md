# Phase-aware tabletop MPC

After the reversible supported escape and fixed-cube boundary replan, the
tabletop workflow can execute every remaining normal arm motion with CuRobo MPC
instead of replaying each frozen trajectory verbatim. The boundary-corrected
frozen lifecycle supplies the selected grasp, a complete collision-checked
route, exact phase boundaries, and reverse recovery trajectories. MPC uses that
route as its local reference and replans short arm-command windows from fresh
measured joint states. The outbound supported escape itself remains the exact
trajectory that established the observation boundary.

The isolated CUDA worker never publishes robot commands. Each returned window
is hash-bound to the frozen execution plan and then installed atomically in the
existing 250 Hz controller. That controller alone interpolates and publishes
the complete low-level command.

## Physical phase model

One collision model cannot correctly describe the complete task because the
cube changes physical role:

| Motion endpoint | Finger/cube state | MPC collision model |
| --- | --- | --- |
| `clearance` | initial fingers; cube on table | supported start, cube is a world obstacle |
| `move_to_pregrasp` | open hand | cube and local table patch are world obstacles |
| `grasp_approach` | open hand entering contact | designated fingertip links may contact the cube; table and all self-collisions remain checked |
| `retention_test_lift` | measured stable-close fingers | cube is a 27-sphere payload attached to the grasp frame |
| `payload_lift` | same measured close | reuses the warm attached-payload model |
| `payload_lower` | same measured close | reuses the warm attached-payload model |
| `payload_replace` | same measured close | reuses the warm attached-payload model |
| `grasp_retreat` | cube released; hand open | contact retreat with cube fixed in the world again |
| `return_to_clearance` | open hand | cube and local table patch are world obstacles |
| `__handoff__` | initial fingers restored | supported return, cube is a world obstacle |

The worker now retains one warmed controller for the complete lifecycle. The
active seven arm joints, tensor sizes, collision-scene capacity, and CUDA graph
addresses never change. At each phase boundary it copies pre-resolved
fixed-size kinematics and collision values into that controller, toggles the
already allocated cube/table obstacles, and installs or removes the payload in
reserved sphere slots. Returning to open-contact, open-free, or supported mode
restores that mode's last action seed, preserving the useful reverse-path warm
start without constructing another solver or CUDA graph.

Initial and deterministic open-finger kinematics are resolved before setup.
The measured close posture cannot be known earlier; it is resolved once after
the physical close stabilizes and then reused for all four connected payload
motions. No geometry is approximated to make the switch cheap.

Finger opening, stable-close acquisition, retention validation, release, and
seated-control restoration remain discrete operations in the existing task
state machine. MPC does not infer or change those transitions.

Every candidate MPC window receives one full FK evaluation. The resulting
spheres are reused for strict full-robot self-collision, cube clearance,
selected wrist/hand table-plane clearance, attached-payload clearance, and the
optional exact presenter-mesh check. A window marked infeasible never enters
the command buffer. A planner or controller failure follows the existing
fail-closed PC2 takeover path. Normal grasp rejection uses the already-frozen
reverse recovery trajectories rather than MPC.

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

## Offline replay evidence

The retained left-arm run `tabletop_20260815T224443Z` was replayed through all
ten physical phases without ROS or robot commands. The single-solver replay
completed 237 accepted windows with zero rejected windows. Every phase reached
its frozen endpoint; the largest terminal joint error was 0.00495 rad and the
maximum commanded velocity remained below the 0.1 rad/s contract.

Final preparation fell from 22.84 s with four warmed solvers to 14.19 s with
one: a 37.9% reduction. Complete benchmark wall time fell from 36.80 s to 27.75
s, a 24.6% reduction. Rolling optimization remained essentially unchanged at
11.55 s and overlaps physical motion.

Every first-use cold solve is deliberately completed during phase preparation,
before a live LowState freshness timestamp exists. Consequently the largest
live window fell from an intermediate 94.6 ms to 64.1 ms against the unchanged
100 ms age limit. Remaining preparation is primarily the first
solver/CUDA-graph setup and prewarm (11.98 s), the one unavoidable
measured-contact finger resolution (1.82 s), and about 0.28 s of first-use
phase prewarming. Cached phase switches take roughly 1--10 ms.

An independent audit rebuilt each former physical-mode robot from its source
configuration and compared it with the in-place model. Fixed transforms, joint
maps, collision topology, self-collision padding, locked joints, and payload
spheres matched exactly; open-hand sphere radii differed only by float32 roundoff
(maximum 3.73e-9 m). The strict full-robot checker remains separate and is
updated from the same resolved finger posture before every window check.

This proves the nominal phase-aware lifecycle against one exact retained plan.
It is not a hardware commissioning result and does not claim closed-loop visual
correction. The current MPC follows the frozen route from measured arm states;
camera/table state correction and commanded waist yaw remain disabled until
their independent contracts are commissioned.
