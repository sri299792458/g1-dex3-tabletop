Latest runtime follow-up: the [lifecycle audit](calibration-lifecycle-audit-2026-09-04.md) records the completed reuse fixes, and the [full collection instructions](full-bilateral-collection-2026-09-04.md) point to the replacement strict-10-mm core. The performance measurements below describe the earlier review.

Review dated 2026-09-04, limited to the calibration pipeline and the shared helpers it actually calls. No production code was changed. Further performance investigation was stopped at the user's request because the measured CPU saving is small in absolute terms.

Subsequent authorized fix: both CLI defects below have now been corrected. Planning no longer reads the removed argument, and its final summary reports the current connected-candidate counts. A command-level regression test covers parser-to-artifact publication and the success summary with simulated worker results; the focused calibration, planning-contract, and hardware-calibration suites pass (35 tests). No performance refactor was made. The findings below describe the reviewed state before these fixes.

Full-run follow-up: a clearance-reference mismatch discovered during final replay was also corrected, and failed planning now retains diagnostics. The final focused suite has 36 passing tests. The [full collection instructions and certification results](full-bilateral-collection-2026-09-04.md) describe the completed 34-per-arm artifact.

**The current evidence does not justify a broad performance refactor. Two CLI defects deserve priority over optimization.**

1. **Planning fails before reading its inputs.** [cli.py](../src/g1_dex3_tabletop/cli.py), line 691, reads `args.maximum_route_reselections`, but the parser no longer defines it. A dry parser/entry-point probe reproduced the `AttributeError`. This is a stale reference to the removed reselection workflow; remove the unused check.
2. **After that is fixed, successful planning would fail while printing its result.** The same file, lines 928–930, indexes `design_provenance["route_disconnected_candidate_ids_excluded"]`. The current [route-request builder](../src/g1_dex3_tabletop/calibration/planning.py), lines 1409–1431, never produces that key. This was confirmed by inspecting the producer and all repository references, without rerunning full planning. The access occurs after `os.replace(stage, output)`, so the output would exist despite a failure status, and a retry at the same path would refuse to start. Report the current connectivity fields instead of the removed reselection field. Verify both fixes with one command-level test that substitutes the expensive planner worker while exercising the real parser and result summary.

**Measured CPU costs are seconds, not a reason for a lengthy basic diagnostic.** A single warm-process measurement used the production helpers, 136 visible candidate poses, the retained fixed anchor, and the same 9+9 selection:

| Operation | Wall time |
| --- | ---: |
| Load planning request | 0.111 s |
| Build candidate pool, including finite-difference Jacobians | 4.437 s |
| Select the 18 poses from the retained connected pool | 0.030 s |
| Recompute the 18-observation-pair Jacobian | 0.556 s |
| Evaluate its singular values | 0.0002 s |
| Build the same pool with temporary reuse of existing transform/FK results | 2.223 s |

The temporary cache experiment reproduced every Jacobian exactly and the entire selection result exactly. It avoided 12,792 of 13,883 FK calls and 13,780 of 13,882 camera/target-transform calls. Profiling a three-pose Jacobian showed repeated transform validation consuming about 60% of cumulative time. This is useful evidence for a future narrow optimization; it is not a measured end-to-end speedup, a hardware benchmark, or a reason to remove input validation. No cache was added to production.

**Reuse already exists.** Bilateral collection imports the commissioned transport, state pairing, quality evaluation, executor, and preparation helpers. Projection uses the existing URDF, camera and transform implementations. The August 27 notes also document that the expensive Cartesian anchor projection loop was already removed. Recommending that fix again would be stale advice.

There is additional duplication in native solver launching, output parsing, revision verification and ROS bag serialization between [bilateral solver.py](../src/g1_dex3_tabletop/calibration/solver.py) and [robot_calibration_bridge.py](../src/g1_aprilcube_calibration/robot_calibration_bridge.py). These are reasonable future extraction candidates, but their performance impact was not measured. The same-frame dataset and grouped-validation rules have different responsibilities from the earlier single-arm code and should remain explicit.

The [design worker](../src/g1_dex3_tabletop/planning/worker.py) also constructs separate collision checkers for anchor filtering, pool connectivity and selected connectivity. These phases use the same fixed model and could share a checker and matching edge results. Each individual phase already batches checks and reuses its checker. Savings across phases are unmeasured; retain the independent final trajectory replay.

The longer investigation included raw-image re-detection, 36 fresh native fits, a graph replay, an anchor sweep and an IK rerun. It was much broader than inspecting an existing report. The lack of a resumable design-diagnostic interface made that work less convenient, but the measured results do not attribute the overall wait to a generally slow calibration codebase. Immediate priority: correct the two CLI defects and the candidate-clearance/IK-branch issues documented in the [standing design assessment](bilateral-standing-design-assessment-2026-09-04.md), then proceed with calibration work.

Local reproducibility: [profile script](../work/bilateral_plan_review_20260904/profile_pipeline.py), [timings and equivalence checks](../work/bilateral_plan_review_20260904/pipeline_profile.json), [cProfile output](../work/bilateral_plan_review_20260904/pipeline_jacobian_profile.txt). These files are in the ignored work directory; timings exclude GPU work and native fitting.
