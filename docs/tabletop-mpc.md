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

For a presentation fixture, the attached cube/fixture pair is the one
intentional support contact. The attached optimizer omits the fixture, and the
independent exact-mesh check tests every robot sphere while excluding only the
attached-object proxy. This matches the frozen payload planner without hiding
any hand/fixture collision.

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

The current retained tripod run `tabletop_20260818T173706Z` was replayed without
ROS or robot commands. The full clearance-boundary planner selected the same
grasp as the successful hardware run, and the benchmark consumed that run's
actual measured `grasp_close.json` rather than substituting the unreachable
descriptor close target. All eight MPC-controlled phases reached their frozen
endpoints through 179 accepted windows with zero rejected windows. The largest
terminal joint error was 0.00429 rad and maximum velocity was 0.09877 rad/s
against the unchanged 0.1 rad/s contract. The supported escape and its exact
reverse are the two deliberately frozen trajectories outside this count.

The current pinned optimizer advances in 25-iteration inner blocks. With 100
warm-start iterations, the exact tripod replay produced 102 ms open-contact
windows before IPC, violating the unchanged 100 ms source-state age contract.
Fifty iterations was fast but produced a 0.10569 rad/s window that the velocity
validator correctly rejected. Seventy-five iterations is therefore the lowest
tested valid setting. Three complete repeatability runs each reached all eight
endpoints with zero rejections; their worst window was 87.28 ms, leaving the
state-age check intact instead of enlarging it.

The production replay took 20.31 s of command-free compute. Initial solver and
CUDA setup took 10.94 s, measured-close kinematics and attached-mode prewarm
took 1.81 s, and cached phase switches took less than 10 ms after their first
use. Rolling solves overlap physical motion. The largest production replay
window was 84.81 ms; the three-run repeatability maximum above is the retained
timing bound.

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
