# September calibration development

This branch preserves the September 4 checkpoint and subsequent local runtime
changes. It descends from the clean August 25 demo `main`. Runtime files,
configuration and tests are copied without behavioral edits. Publication does
not establish additional hardware validation.

## Source map

| Area | Source | Role |
|---|---|---|
| Collection | [`hardware_bilateral_calibration.py`](../src/g1_dex3_tabletop/hardware_bilateral_calibration.py) | Frozen route inputs, preflight, capture and cleanup |
| Standing control | [`calibration/control.py`](../src/g1_dex3_tabletop/calibration/control.py) | Arm-SDK ownership, watchdog and lifecycle |
| Route and adapter | [`calibration/route.py`](../src/g1_dex3_tabletop/calibration/route.py), [`calibration/adapter.py`](../src/g1_dex3_tabletop/calibration/adapter.py) | Planned core and live-state connections |
| Hand posture | [`calibration/hand_posture.py`](../src/g1_dex3_tabletop/calibration/hand_posture.py), [`calibration/fingers.py`](../src/g1_dex3_tabletop/calibration/fingers.py) | Operator-preclosed readiness and measured hold |
| Capture | [`calibration/capture.py`](../src/g1_dex3_tabletop/calibration/capture.py), [`calibration/session.py`](../src/g1_dex3_tabletop/calibration/session.py) | Image/state evidence and session artifacts |
| Visibility and diagnostics | [`calibration/visibility.py`](../src/g1_dex3_tabletop/calibration/visibility.py), [`calibration/diagnostic.py`](../src/g1_dex3_tabletop/calibration/diagnostic.py) | Marker sightlines and repeated-configuration routes |
| Solving | [`calibration/solver.py`](../src/g1_dex3_tabletop/calibration/solver.py), [`calibration/validation.py`](../src/g1_dex3_tabletop/calibration/validation.py) | Shared-camera fits and grouped validation |
| Command boundary | [`control_boundary.py`](../src/g1_dex3_tabletop/control_boundary.py) | Bind planning snapshots and installations to active commands |
| Waist hold | [`unitree_arm_sdk.py`](../src/g1_aprilcube_calibration/transports/unitree_arm_sdk.py) | Measured waist reference during standing arm-SDK control |
| Recording | [`raw_episode_recording.py`](../src/g1_dex3_tabletop/raw_episode_recording.py) | Continuous standing camera/state recording through cleanup |

Shared changes in planning, tabletop execution, stack execution and regression
fixtures are retained too. The calibration directory alone is not a complete transplant.

## Runtime boundaries

Standing collection uses `rt/arm_sdk`; seated stacking uses complete 29-joint
`rt/lowcmd` ownership. Their watchdog and handback paths differ. The CLI no longer
exposes the older single-arm calibration workflow.

The final standing lifecycle expects both hands to be fully closed by the
operator before planning and acquisition. The saved reference is a readiness
check, not a motion target. Planning and return use measured finger geometry.
Routes certified using the earlier grasp posture cannot be reused as certificates
for this lifecycle. The final operator-preclosed lifecycle and measured-waist
behavior still require hardware validation.

Excitation capture requires the active marker; an available inactive marker is
retained without blocking capture. Anchors require both. Standing raw recording
includes RGB, depth, camera metadata and robot state/commands; see the
[recording contract](data-recording.md).

## Findings and unresolved limits

The September 5 collection has 47 distinct active configurations. September 7
adds repeated visits and opposite approaches. The geometric residual remained
repeatable and configuration dependent. Tested timing uncertainty, ordinary
corner variation and raw approach motion did not explain its dominant magnitude.
Broad coverage, depth optimization, static offsets and axis/link models have
already been investigated.

Camera adjustment between runs was intentional; each run requires its own
constant registration. Calibration uses measured joints. Exact agreement with
commanded endpoints is not a prerequisite for interpreting those observations.
Marker attachment transforms are constant within a run and can be estimated
separately between runs.

The operator's wrist-spacing measurement supports approximately 51 mm rather
than the model's 46 mm. Refitting the original August calibration after this
geometry change reduced grouped pixel RMS from 5.409 to 5.031 px. A larger
geometry fit achieved lower held-out errors while adjusting many other parameters;
it does not uniquely diagnose a physical defect. The physical explanation
remains unresolved.

No September candidate replaced the historical stacking bundle. Wrist patches
remain unapplied, the vendor issue draft remains unposted, and the proposed
broader sweep is paused. A forearm marker was rejected because there is no
mounting space. These are recorded boundaries, not instructions to resume work.

## Evidence and checks

The [G1 research guide](https://sri299792458.github.io/g1-research-docs/) provides
the curated account. Raw notes, internal proposals, investigation work directories
and recordings remain separate; their absence from this clone does not mean
those experiments never happened.

Offline regressions include
[`test_standing_calibration_control.py`](../tests/test_standing_calibration_control.py),
[`test_calibration_hand_posture.py`](../tests/test_calibration_hand_posture.py),
[`test_calibration_visibility.py`](../tests/test_calibration_visibility.py),
[`test_bilateral_adapter_worker.py`](../tests/test_bilateral_adapter_worker.py)
and [`test_bilateral_calibration.py`](../tests/test_bilateral_calibration.py).
Passing fixture/fake-transport tests is not physical commissioning. Publication
does not regenerate GPU plans, collect data, apply a bundle or operate the robot.
