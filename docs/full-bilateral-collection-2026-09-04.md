Latest status (`20260905T022735Z`): ownership reached full weight, then the
first shoulder trajectory was rejected because its preflight start differed
from the acquired right-wrist-roll command by 0.000227704644 rad. No arm target
changed. This repeats the measured/command boundary issue documented for the
August 21 stack run.

The bilateral collector now directly reuses the stack's
`_command_bound_snapshot` and `_install_plan_at_current_boundary`: acquire the
measured hold, refresh the reversible adapter from both held arm commands and
the measured body, then install while preserving commands. The fixed core is
replayed against that body without regeneration. The starting finger posture
is the measured hold-acquisition posture. Preflight artifacts remain in work/;
the existing isolated writer freezes the actual owned adapter into the session
before motion. All continuity, state-drift, and collision checks remain active.

Validation: 519 tests passed, 9 skipped. A controlled real-CuRobo comparison
using recorded arm commands with the earlier preflight body passed: start error
8.67e-19 rad, 0.08 rad shoulder preparation, and the 81-transition core passed.
This comparison does not certify the recorded full-ownership body state.
That state independently fails the existing collision model: left index/hip
roll clearance is -1.41 mm with measured arms and -2.15 mm with held commands,
versus +2.60 mm at preflight. The operator did not confirm physical contact;
these are sphere-model results. The command-boundary fix is complete, but an
unchanged physical setup is not declared ready on this evidence. No collision
exclusion or safety tolerance was changed to pass the replay.

Evidence: [owned-boundary diagnostics](../work/bilateral_plan_review_20260904/takeover_022735).
The full collection command below remains the launch command once the loaded
clearance issue is resolved; it is not a recommendation for another blind retry.

Earlier diagnosis: the bag from `20260905T020319Z` identifies a startup-order
regression. The collector armed the 0.5 s PC2 watchdog before three SDK publisher
initializations that each sleep 0.2 s. The watchdog timed out before the next
heartbeat. Its Dex3 fallback packets appear 109 ms before the first arm command;
whole-body torque then drops around 0.46 s into arm acquisition, before the
right elbow exceeds the guard at 0.534 s. Arm commands were constant and arrived
every 3.98 ms on median, with a maximum interval of 5.77 ms.

Both standing collectors now restore the working collector's startup order:
construct all three publishers after SPACE, then start the watchdog, then
acquire measured hand/arm holds. The constructors publish no commands; the
watchdog is armed before the first control command. The 0.5 s watchdog deadline,
one-second arm ramp, gains, and 0.05 rad acquisition allowance are unchanged.
An offline execution of the actual startup statements with the actual local
watchdog agent reproduced the previous timeout (`heartbeat_count=1`) and passed
both corrected startup paths. The full suite passed (513 tests, 9 skipped).
The next recorded run reached full ownership with this corrected order, as
described above. Rosbag recording remains enabled.

See [the takeover timeline](../work/bilateral_plan_review_20260904/takeover_020319/takeover_timeline.png)
and [watchdog reproduction](../work/bilateral_plan_review_20260904/takeover_020319/watchdog_startup_order.json).

The collector starts plain MCAP recording before SPACE and requires subscriptions
to `/arm_sdk`, `/lowstate`, and both Dex3 state/command topics before allowing
takeover. It checks that the recorder is still alive before creating command
publishers and stops it after control cleanup, including failure or Ctrl+C.
Use the same command below; no pilot or separate recording command is needed.
The bag is saved under `work/<session>_collection/raw_episode/bag`, with runtime
hardware settings alongside it and the recording summary in `status.json`.
Existing calibration image capture remains in the session store.

Verified with real Unitree DDS types and the MCAP recorder on loopback/domain
221: all six subscriptions were ready before any command publisher existed;
all 134 synthetic arm commands and 134 state messages were retained, including
the first weight-zero command and simulated elbow excursion. Full suite:
513 passed, 9 skipped; lint and shell syntax checks passed. This test did not
connect to the robot. MCAP timestamps describe laptop DDS receipt, not robot
receipt acknowledgement or internal controller-loop execution times.

The runtime reuse fixes and the full standing calibration artifacts are ready
for the collector's live preflight. Use the recertified files in
[work/bilateral_full_20260904_reuse](../work/bilateral_full_20260904_reuse).
The older `work/bilateral_full_20260904` execution plan is superseded because
it did not enforce the commissioned 10 mm core clearance requirement.

| Prepared run | Result |
| --- | ---: |
| Distinct excitation poses | 34 left + 34 right |
| Repeated-anchor captures | 11 |
| Total same-frame captures | 79 |
| Frozen core transitions | 81 |
| Connected candidate pool | 756 left + 626 right |
| Selected model rank / condition | 25 / 25; 59.41 |
| Core arm-motion duration | 570.9 s, about 9.5 minutes |
| Minimum modeled core clearance | 10.588 mm |
| Required core clearance | 10 mm; no close-reference exception |

Settling, capture, retries, and the live adapter add time beyond the arm-motion
duration. This is the full requested collection, with no preceding pilot.
The candidate IK from the earlier full preparation was reused. Connectivity,
selection, route generation, and independent final certification were rerun
using the restored collision-pair and clearance policies. Input bindings,
counts, camera identity, and commissioned close postures passed the collector's
offline checks; see
[collection_readiness.json](../work/bilateral_full_20260904_reuse/collection_readiness.json).

From `/home/kanth042/g1-dex3-tabletop`, use the existing standing Ready setup:

```bash
./tools/g1_tabletop_hardware.sh collect-bilateral-calibration \
  --network-interface enp134s0 \
  --pose-design work/bilateral_full_20260904_reuse/pose_design.json \
  --execution-plan work/bilateral_full_20260904_reuse/execution_plan.json \
  --confirm 'I CONFIRM THE G1 IS SECURED BY THE LOAD-BEARING HARNESS AND THE WORKSPACE IS CLEAR'
```

The launcher starts the commissioned camera service. Before SPACE or any motion
publisher, the collector reads the actual Ready and hand state, plans the
reversible adapter, replays every core transition against that live body
geometry, and prewarms the existing isolated capture writer. The old historical
body-posture tolerance is gone. Changes after preflight are still checked.

After SPACE, the arms move to shoulder clearance while holding the measured
starting fingers. The persistent worker then checks a fresh measured finger
sweep while the control driver holds position. Both hands close to the
commissioned calibration posture and settle before reaching the visual anchor.
Every capture is durably written by the existing isolated process before the
capture interlock is released. The shared cleanup helper preserves control
faults instead of masking them with a second capture-state error.

Return reverses the anchor adapter to shoulder clearance, rechecks the measured
finger sweep, restores this run's starting finger posture, and reverses both
shoulder paths to Ready before clean arm_sdk release. There is no canonical-open
hand command. Lowercase `q` requests graceful return through the next repeated
anchor. `Ctrl+C` remains the Damp path.

The [lifecycle audit](calibration-lifecycle-audit-2026-09-04.md) records the
reused implementations and collision policies. Same-hand contacts and adjacent
shoulder-roll/torso are excluded as in the working pipeline; shoulder-yaw/torso,
external hand, and cross-arm checks remain. Core clearance is a strict 10 mm.
Ready shoulder preparation has its separate 5 mm/reference-bounded policy;
external-hand finger sweeps require 5 mm. These are model checks using the
current CuRobo spheres, not physical clearance measurements.

Execution schema: 5. Pose-design schema: 3.
Execution hash: `4f7f7b941c604d5b868793104c28075e579f57a7f83c455312f058ea274dfb15`.
Pose-design hash: `7ce1384d6fafbd23c64c107a480d5d066238c8872d1182f25c0e658de8f81d87`.

The [planning log](../work/bilateral_plan_review_20260904/runtime_reuse_design.log),
[artifact assembler](../work/bilateral_plan_review_20260904/assemble_reused_core.py),
[input checker](../work/bilateral_plan_review_20260904/verify_full_collection_inputs.py),
and [reused input manifest](../work/bilateral_full_20260904_reuse/reused_input_sources.json)
are retained locally. Work artifacts are ignored by Git. No robot motion or
physical collection was performed while preparing these changes.

Final verification: **502 tests passed, 9 skipped** in the main environment;
**13 planner model tests passed** in the planner environment, including the
GPU-dependent checks and strict-clearance regression. Lint, formatting, and
whitespace checks passed. Both original live startup snapshots pass the first
**0.08 rad** preparation candidate, complete return certification, and all
**81 core transitions** with their captured body geometry. Adapter replay took
32.1 s and 31.1 s. Minimum core clearance in both replays is **10.588 mm**.
See [the replay summary](../work/bilateral_plan_review_20260904/strict10_live_replay/summary.json)
and [replay script](../work/bilateral_plan_review_20260904/replay_strict_core.py).

The persistent worker's close/restore checks also passed at the nominal planned
clearance postures, with minimum external-hand gaps of 20.959–22.439 mm.
These offline sweep inputs use planned arm positions, not measured loaded
positions. The collector obtains fresh loaded measurements for those checks
on the actual run. No physical execution has been validated by these replays.

If preflight reports a starting-posture restoration target outside a finger's
hard limits, stop the collector, ease the named finger away from its tightly
closed endpoint, and restart the same command to capture a fresh state. The
collector now reports the exact joint, reading, and limits before starting
CuRobo; it does not increase shoulder offsets for this fixed target error.
The `20260905T005649Z` startup had left index-base `−1.578415155 rad` versus a
lower limit of `−1.570796320 rad`, a difference of 0.44 degrees. Its close target
was valid; exact restoration caused the rejection. Return targets are not
clipped, and hard limits are unchanged. The correction has 505 passing main
suite tests (9 skipped) and replays that rejected request before any candidate
attempt. The full core and launch command above are unchanged.

The diagnostic retry `20260905T011456Z` reproduced the acquisition fault at the
right elbow: 0.979770 → 1.031638 rad, 0.5317 s into takeover, last arm-SDK weight
0.5277, state age 0.0007 s. No shoulder trajectory had started. Routine seated
tabletop manipulation uses a different transport: direct complete-body
`rt/lowcmd`, full arm PD gains from the first packet, and locally ramped gravity
feedforward after releasing the motion service. Standing calibration retains
the standing controller and uses `rt/arm_sdk` with an ownership-weight ramp.
The older standing calibration used this same arm-SDK path. That transport
comparison did not explain the standing failure. The subsequent recorded run
identified the premature watchdog timeout and publisher startup-order regression
described at the top of this document; that order is now corrected.
