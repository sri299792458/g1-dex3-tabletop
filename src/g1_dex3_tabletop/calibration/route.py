"""Execute the reversible adapter around the frozen bilateral capture core."""

from __future__ import annotations

import numpy as np


class BilateralCalibrationRoute:
    def __init__(self, *, control, adapter, plan, planner, artifact_directory):
        self.control = control
        self.adapter = adapter
        self.plan = plan
        self.planner = planner
        self.artifact_directory = artifact_directory

    def _move(self, arm, trajectory, **kwargs):
        self.control.execute(
            arm=arm,
            trajectory=trajectory,
            plan_sha256=self.adapter.content_sha256,
            **kwargs,
        )

    def move_to_anchor(self, *, validated_reference_state):
        self._move(
            "right",
            self.adapter.right_anchor_outbound,
            validated_reference_state=validated_reference_state,
        )
        self._move("left", self.adapter.left_anchor_outbound)

    def return_to_ready(self):
        self.control.check()
        if not np.allclose(
            self.control.executor.dual_arm_command_q,
            self.adapter.anchor_q14_rad,
            atol=1e-9,
            rtol=0,
        ):
            raise ValueError("adapter return requires both arms at the fixed anchor command")
        # Also works at an identical repeated anchor after Q: installation
        # retains its current name and verifies the exact held command.
        self._move("left", self.adapter.left_anchor_return)
        self._move("right", self.adapter.right_anchor_return)
        return_reference = self.control.validate_shoulder_return(
            planner=self.planner,
            joint_position_offsets_rad=self.plan.joint_position_offsets_rad,
            artifact_directory=self.artifact_directory,
        )
        self.control.return_shoulders(validated_reference_state=return_reference)
