"""Unitree-style arm gravity feedforward from the exact configured G1 URDF.

Unitree's pinned XR controller evaluates Pinocchio RNEA at the commanded arm
configuration with zero velocity and acceleration, then writes the resulting
generalized gravity torque to each arm motor's ``tau`` field.  This module
implements that same control term without importing the XR IK stack.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from g1_aprilcube_calibration.joint_map import (
    G1_29_JOINT_NAMES,
    LEFT_ARM_INDICES,
    RIGHT_ARM_INDICES,
    dual_arm_vector,
    validate_full_joint_vector,
)

UNITREE_XR_GRAVITY_REFERENCE = (
    "unitreerobotics/xr_teleoperate G1_29_ArmIK.solve_ik -> pinocchio.rnea(q, 0, 0)"
)
_ARM_INDICES = LEFT_ARM_INDICES + RIGHT_ARM_INDICES


class ArmGravityFeedforward(Protocol):
    """Stateful gravity provider used by the deterministic pose executor."""

    def seed_reference(self, full_q: Sequence[float] | np.ndarray) -> None: ...

    def torque_for(self, q14: Sequence[float] | np.ndarray) -> np.ndarray: ...


class G1PinocchioGravityFeedforward:
    """Evaluate ``rnea(q_command, 0, 0)`` for both G1 arms.

    The configured URDF is a fixed-base 29-DoF model.  Legs and waist are held
    at the measured ownership-takeover configuration; both commanded arms are
    replaced on every evaluation.  The returned vector follows Unitree's
    left-seven then right-seven DDS motor ordering.
    """

    def __init__(
        self,
        urdf_path: str | Path,
        *,
        locked_joint_positions_rad: Mapping[str, float] | None = None,
        pinocchio_module: Any | None = None,
    ) -> None:
        self.urdf_path = Path(urdf_path).resolve()
        if not self.urdf_path.is_file():
            raise FileNotFoundError(f"gravity model URDF does not exist: {self.urdf_path}")
        self.urdf_sha256 = hashlib.sha256(self.urdf_path.read_bytes()).hexdigest()
        self._pin = pinocchio_module or self._load_pinocchio()
        self.backend_version = str(getattr(self._pin, "__version__", "unknown"))
        full_model = self._pin.buildModelFromUrdf(str(self.urdf_path))
        configured_locked = {
            str(name): float(value) for name, value in (locked_joint_positions_rad or {}).items()
        }
        if not all(np.isfinite(value) for value in configured_locked.values()):
            raise ValueError("locked gravity-model joint positions must be finite")
        g1_names = set(G1_29_JOINT_NAMES)
        full_joint_names = {
            str(full_model.names[joint_id]) for joint_id in range(1, int(full_model.njoints))
        }
        unknown_locked = sorted(set(configured_locked) - full_joint_names)
        if unknown_locked:
            raise ValueError(
                "locked gravity-model joints are absent from the URDF: "
                + ", ".join(unknown_locked)
            )
        extra_joint_ids = [
            joint_id
            for joint_id in range(1, int(full_model.njoints))
            if str(full_model.names[joint_id]) not in g1_names
        ]
        extra_joint_names = tuple(str(full_model.names[joint_id]) for joint_id in extra_joint_ids)
        missing_locked = sorted(set(extra_joint_names) - set(configured_locked))
        if missing_locked and configured_locked:
            raise ValueError(
                "gravity-model posture must specify every non-G1 joint: "
                + ", ".join(missing_locked)
            )
        reference = np.asarray(self._pin.neutral(full_model), dtype=np.float64).copy()
        resolved_locked: dict[str, float] = {}
        for joint_id, name in zip(extra_joint_ids, extra_joint_names, strict=True):
            joint = full_model.joints[joint_id]
            if int(joint.nq) != 1 or int(joint.nv) != 1:
                raise ValueError(f"non-G1 gravity-model joint {name} is not one DoF")
            value = configured_locked.get(name, 0.0)
            q_index = int(joint.idx_q)
            lower = float(full_model.lowerPositionLimit[q_index])
            upper = float(full_model.upperPositionLimit[q_index])
            if value < lower or value > upper:
                raise ValueError(
                    f"locked gravity-model joint {name}={value:.6f}rad is "
                    f"outside [{lower:.6f}, {upper:.6f}]"
                )
            reference[q_index] = value
            resolved_locked[name] = value
        self.full_model_nq = int(full_model.nq)
        self.locked_joint_positions_rad = resolved_locked
        self._model = (
            full_model
            if not extra_joint_ids
            else self._pin.buildReducedModel(full_model, extra_joint_ids, reference)
        )
        self.reduced_model_nq = int(self._model.nq)
        self._data = self._model.createData()
        self._q_indices: tuple[int, ...]
        self._v_indices: tuple[int, ...]
        q_indices: list[int] = []
        v_indices: list[int] = []
        for name in G1_29_JOINT_NAMES:
            joint_id = int(self._model.getJointId(name))
            if joint_id <= 0:
                raise ValueError(f"gravity model is missing G1 joint {name}")
            joint = self._model.joints[joint_id]
            if int(joint.nq) != 1 or int(joint.nv) != 1:
                raise ValueError(f"gravity model joint {name} is not one DoF")
            q_indices.append(int(joint.idx_q))
            v_indices.append(int(joint.idx_v))
        if len(set(q_indices)) != len(G1_29_JOINT_NAMES):
            raise ValueError("gravity model has duplicate G1 position coordinates")
        if len(set(v_indices)) != len(G1_29_JOINT_NAMES):
            raise ValueError("gravity model has duplicate G1 velocity coordinates")
        if int(self._model.nq) != len(G1_29_JOINT_NAMES):
            raise ValueError(
                "gravity model must contain exactly the 29 actuated G1 joints; "
                f"model nq is {self._model.nq}"
            )
        if int(self._model.nv) != len(G1_29_JOINT_NAMES):
            raise ValueError(
                "gravity model must contain exactly the 29 actuated G1 joints; "
                f"model nv is {self._model.nv}"
            )
        self._q_indices = tuple(q_indices)
        self._v_indices = tuple(v_indices)
        self._arm_q_indices = tuple(q_indices[index] for index in _ARM_INDICES)
        self._arm_v_indices = tuple(v_indices[index] for index in _ARM_INDICES)
        effort_limits = np.asarray(self._model.effortLimit, dtype=np.float64)
        self._arm_effort_limits = effort_limits[
            np.asarray(self._arm_v_indices, dtype=np.int64)
        ].copy()
        if (
            self._arm_effort_limits.shape != (14,)
            or not np.all(np.isfinite(self._arm_effort_limits))
            or np.any(self._arm_effort_limits <= 0)
        ):
            raise ValueError("gravity model has invalid arm effort limits")
        self._reference_model_q: np.ndarray | None = None
        self._reference_g1_q: np.ndarray | None = None
        self._cached_q14: np.ndarray | None = None
        self._cached_tau14: np.ndarray | None = None

    @staticmethod
    def _load_pinocchio() -> Any:
        try:
            import pinocchio as pin
        except (ImportError, OSError) as error:
            raise RuntimeError(
                "Pinocchio is unavailable; run uv sync --python /usr/bin/python3 "
                "--group dev before gravity-compensated hardware execution"
            ) from error
        return pin

    def seed_reference(self, full_q: Sequence[float] | np.ndarray) -> None:
        """Freeze the non-arm coordinates at one measured takeover sample."""

        measured = validate_full_joint_vector(
            full_q,
            name="gravity feedforward reference joint position",
        )
        model_q = np.zeros(int(self._model.nq), dtype=np.float64)
        model_q[np.asarray(self._q_indices, dtype=np.int64)] = measured
        self._reference_model_q = model_q
        self._reference_g1_q = measured.copy()
        self._cached_q14 = None
        self._cached_tau14 = None

    @property
    def reference_full_q(self) -> np.ndarray:
        """Return the exact measured state used for the current torque model."""

        if self._reference_g1_q is None:
            raise RuntimeError("gravity feedforward has no measured reference state")
        return self._read_only_copy(self._reference_g1_q)

    def torque_for(self, q14: Sequence[float] | np.ndarray) -> np.ndarray:
        """Return finite, effort-bounded gravity torque in dual-arm order."""

        if self._reference_model_q is None:
            raise RuntimeError("gravity feedforward has no measured reference state")
        command = np.asarray(q14, dtype=np.float64).reshape(-1).copy()
        if command.shape != (14,):
            raise ValueError("gravity feedforward command must contain 14 joints")
        if not np.all(np.isfinite(command)):
            raise ValueError("gravity feedforward command contains NaN or infinity")
        if self._cached_q14 is not None and np.array_equal(command, self._cached_q14):
            assert self._cached_tau14 is not None
            return self._read_only_copy(self._cached_tau14)

        model_q = self._reference_model_q.copy()
        model_q[np.asarray(self._arm_q_indices, dtype=np.int64)] = command
        zero_velocity = np.zeros(int(self._model.nv), dtype=np.float64)
        generalized_torque = np.asarray(
            self._pin.rnea(
                self._model,
                self._data,
                model_q,
                zero_velocity,
                zero_velocity,
            ),
            dtype=np.float64,
        ).reshape(-1)
        if generalized_torque.shape != (int(self._model.nv),):
            raise RuntimeError("Pinocchio RNEA returned an unexpected torque shape")
        arm_torque = generalized_torque[np.asarray(self._arm_v_indices, dtype=np.int64)].copy()
        if not np.all(np.isfinite(arm_torque)):
            raise RuntimeError("Pinocchio RNEA returned NaN or infinite arm torque")
        over_limit = np.flatnonzero(np.abs(arm_torque) > self._arm_effort_limits + 1e-9)
        if len(over_limit):
            index = int(over_limit[0])
            raise RuntimeError(
                "gravity feedforward exceeds the URDF motor effort limit at "
                f"{G1_29_JOINT_NAMES[_ARM_INDICES[index]]}: "
                f"{arm_torque[index]:.4f}Nm vs "
                f"{self._arm_effort_limits[index]:.4f}Nm"
            )
        self._cached_q14 = command
        self._cached_tau14 = arm_torque
        return self._read_only_copy(arm_torque)

    def torque_for_full_state(
        self,
        full_q: Sequence[float] | np.ndarray,
    ) -> np.ndarray:
        """Convenience evaluator for offline reports and pre-command diagnostics."""

        measured = validate_full_joint_vector(full_q)
        self.seed_reference(measured)
        return self.torque_for(dual_arm_vector(measured[15:22], measured[22:29]))

    @staticmethod
    def _read_only_copy(values: np.ndarray) -> np.ndarray:
        result = np.asarray(values, dtype=np.float64).copy()
        result.setflags(write=False)
        return result
