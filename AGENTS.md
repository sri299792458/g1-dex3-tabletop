# Maintaining the published G1 baseline

- This branch starts from the committed August 25, 2026 stacking-demo baseline,
  `d1b010351189259633ccaa536e3618aa5892dd88`. Its initial publication preserves
  runtime code, tests, configuration, CAD and submodule pins from that revision.
- Read the README and relevant technical documentation before changing a
  hardware workflow. Distinguish offline checks from physical validation.
- Personal `running_notes.md` files are private. Keep new working notes in
  ignored local storage or an explicitly private repository; do not force-add
  or publish them. Internal design proposals and research/implementation plans
  are private as well; labelling them historical does not authorize publication.
  Publish curated documentation of implemented behavior, setup and limitations.
- The G1 research guide also describes later experimental code. Match the
  documented source revision before transferring a change into this baseline.
- Reviewing documentation or code does not authorize operating the robot.
