# G1 Dex3 Tabletop — September calibration development

`experimental/september-calibration` preserves the later bilateral-calibration
runtime, diagnostics, recording and shared control changes. It includes the
September 4 checkpoint and the subsequent local implementation. Calibration
accuracy remains unresolved, and the final runtime revisions were checked
offline without completing hardware validation.

Use [`main`](https://github.com/sri299792458/g1-dex3-tabletop/tree/main) for the
committed source baseline associated with the August 25 cube-stacking demo.
This branch has not replaced that demonstrated baseline or its calibration bundle.

Read [September calibration development](docs/september-calibration.md) for the
source map, findings and limits. The
[G1 research guide](https://sri299792458.github.io/g1-research-docs/) explains the
broader project.

## Development setup

Initialize the pinned submodules before using the existing setup scripts.
They retain separate Python 3.10 control/analysis and Python 3.11 CUDA-planning
environments. See the baseline README for platform and dependency prerequisites.

```bash
git submodule update --init --recursive
./tools/setup_control_env.sh
./tools/setup_planner_env.sh
./tools/install_robot_calibration_local.sh
./tools/g1_tabletop.sh --help
```

The calibration commands are `plan-bilateral-calibration`,
`collect-bilateral-calibration`, `build-bilateral-calibration-dataset` and
`solve-bilateral-calibration`. Read the branch's runtime limits before adapting
an earlier command or route. Hardware operation requires the commissioned
watchdog, harness, current geometry, preflight and interactive acknowledgement;
repository publication does not commission this version.

Additional technical references:

- [Recording contract](docs/data-recording.md)
- [Dex3 dorsal marker mounts](docs/dex3_dorsal_aruco_mount.md)
- [Experimental tabletop MPC](docs/tabletop-mpc.md)

Personal running notes and internal proposals remain outside the public source.
