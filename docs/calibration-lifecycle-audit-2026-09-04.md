This audit records the reuse fixes identified during the earlier review. Its
original claim that the runtime omissions were all fixed was too broad: the
subsequent recorded takeover failure exposed a duplicated startup sequence that
armed the watchdog before blocking publisher initialization. That failure had
already been documented and fixed during August 10 tabletop commissioning.
Both standing collectors now initialize publishers before arming the watchdog,
as the working standing and tabletop code did. Offline watchdog reproduction
passes; physical validation of this correction remains pending. See the
[current failure evidence and fix](full-bilateral-collection-2026-09-04.md).

The bilateral collector reuses the commissioned writer, capture cleanup,
control driver, hand controller, and persistent planner. Sharing these
components did not preserve the complete lifecycle automatically. Both current
standing collectors still separately assemble publisher/watchdog startup,
arm-stage handoffs, clean release, and failure cleanup. These are the remaining
duplication sites; their presence alone does not establish another runtime
fault. Any consolidation should extract the corrected existing standing
sequence and exercise it through both entry points, retaining bilateral route
and same-frame capture behavior in the bilateral collector. This audit concerns
execution policy; it does not reopen the camera timing or flex investigations.

| Earlier working behavior | Correction in the bilateral collector |
| --- | --- |
| Start from this run's measured Ready state. | Remove the historical body-posture gate. Use live body geometry for the adapter and mandatory full core replay; retain the mismatch as diagnostic evidence. |
| Exclude internal contacts within each hand. | Share the existing same-hand exclusion for arm planning and articulated finger checks. Preserve external hand pairs and collision spheres. |
| Restore starting fingers before reversing shoulders. | Restore the hash-bound startup measurements at clearance. Certify the return endpoint and both reversed shoulder paths with those fingers. |
| Isolate capture persistence before acquiring control. | Generalize the existing `IsolatedSessionStore` with a store factory and use `BilateralSessionStore` in its prewarmed process. Poll control health until durable commit. |
| Recheck fingers at the measured loaded state. | Reuse the finger-sweep validator through the existing persistent worker at both close and restore. Bind results to fresh measurements and exact targets, enforce 5 mm external-hand clearance, and reject state drift before commanding fingers. |
| Preserve the original control fault during capture cleanup. | Share `finish_capture_or_raise_fault` with the earlier capture runner. Keep the stationary capture interlock through the durable write. |
| Exclude adjacent shoulder-roll/torso; retain shoulder-yaw/torso. | Extract and share the existing tabletop exclusion. Both shoulder-yaw/torso pairs remain checked. |
| Keep invariant body/body pairs out of arm-motion clearance requirements. | Reuse the existing static-pair helper while retaining both arm/hand subtrees. Leg/waist joints must be locked. All arm/body and cross-arm pairs remain checked except the commissioned adjacent-assembly exclusions. |
| Require 10 mm for the authored core and 5 mm for preparation. | Reconnect, reselect, and independently certify the full core at a strict 10 mm. No close-reference exception is allowed in the core. Ready shoulder preparation retains its separate 5 mm/reference-bounded policy; finger sweeps require 5 mm externally. |

Primary historical evidence is the prototype's
[August 12 notes](/home/kanth042/robot-calibration-aprilcube-prototype/running_notes.md:1683),
[selected collision pairs](/home/kanth042/robot-calibration-aprilcube-prototype/config/collision_pairs_dex3_aruco.yaml),
and [articulated pair expansion](/home/kanth042/robot-calibration-aprilcube-prototype/src/g1_aprilcube_calibration/dual_arm_clearance.py:363).
Production imports stay within this repo. Shared implementations are in
[session_store.py](../src/g1_aprilcube_calibration/session_store.py),
[session_runner.py](../src/g1_aprilcube_calibration/session_runner.py),
[g1_model.py](../src/g1_dex3_tabletop/planning/g1_model.py), and
[hardware_calibration.py](../src/g1_dex3_tabletop/hardware_calibration.py).

The first live failure was a 0.050132334 rad ankle mismatch against an archived
snapshot. The second repeatedly increased shoulder offset because tightly
closed fingers produced invariant same-hand contacts in the generic model.
Independent replay also found **29.571 mm modeled left-thumb/hip overlap** when
canonical-open fingers were returned along shoulder paths checked with the
starting closed fingers. The original requests and
[open-return diagnostic](../work/bilateral_plan_review_20260904/live_adapter_replay/232718_open_return_certificate.json)
remain unchanged. Those are the reasons for the fixes, not evidence that more
shoulder displacement or opening the hands was needed.

The earlier corrected-pair replay still exposed a real clearance-policy drift:
the old core came as close as 5.865 mm. It has been superseded by
[the recertified full collection](full-bilateral-collection-2026-09-04.md).
Candidate IK was reused; connectivity, selection, trajectories, and final
certification were rerun. An obsolete 5 mm core now fails the collector's
read-only input checks before planner startup. Archived IK-source preparation
records remain historical evidence; hardware always constructs a new live
adapter that restores this run's starting fingers.

The CuRobo implementation still uses NVIDIA collision spheres with the mounted
marker geometry. The earlier authored route used the G1Pilot collision overlay.
Restoring pair selection and clearance requirements does not make those two
geometry representations identical. The reported gaps are modeled clearances,
not physical measurements. Ready shoulder preparation can preserve an already
closer pair with at most 0.25 mm degradation; that exception does not apply to
the 10 mm core or the external-hand 5 mm sweep requirement.

The existing 250 Hz control driver, 0.25 s local gap fault, PC2 watchdog,
measured hold, gravity compensation, state pairing, finite capture retries,
and graceful return through the next identical anchor remain in use. No
performance rewrite, robot command, or pilot collection was issued. Runtime
checks are covered by regression tests; actual loaded-state checks still run
on the robot during collection. Offline results and the current launch command
are recorded in the full collection instructions linked above.

Final evidence: 502 main-environment tests passed (9 skipped), and all 13 planner
model tests passed. The tests cover durable bilateral image replay through a
separate process, health polling, faults during capture and persistence,
request/result mismatches, unsafe finger clearance, state drift, old-core
rejection, retained collision pairs, and the absence of a core reference
exception. Both saved live startup states pass the new full 81-transition core
and reversible adapter at the first 0.08 rad candidate; see the
[retained replay summary](../work/bilateral_plan_review_20260904/strict10_live_replay/summary.json).
Those tests covered the listed fixes, but did not establish startup equivalence
with the working collector. In particular, they did not expose the SDK's
blocking publisher initialization under an already armed watchdog. The later
watchdog reproduction covers that failure. Fresh live preflight and
loaded-state checks remain part of execution.
