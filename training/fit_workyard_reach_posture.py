"""Fit and validate a collision-aware Workyard reach posture map.

This tool is intentionally offline.  It produces reviewable coefficients for
the batched CUDA reach teacher; it never grants runtime control authority.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

import mujoco
import numpy as np
from scipy.optimize import minimize

import workyard_mjlab_env as base
import workyard_residual_mjlab_env as residual


ARM_JOINT_NAMES = (
  base.BODY_JOINT_NAMES[15:22],
  base.BODY_JOINT_NAMES[22:29],
)
WRIST_BODY_NAMES = ("left_wrist_yaw_link", "right_wrist_yaw_link")
ROD_CENTER_WORLD = np.asarray(base.ROD_START_POSITION, dtype=np.float64)
NEUTRAL = np.asarray(base.BODY_DEFAULT_POSITIONS[15:], dtype=np.float64).reshape(2, 7)
AUTHORITY_RAD = 0.7


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser()
  parser.add_argument("--feature-scale-m", type=float, default=0.16)
  parser.add_argument("--collision-clearance-m", type=float, default=0.005)
  parser.add_argument("--maximum-iterations", type=int, default=180)
  parser.add_argument(
    "--output",
    type=Path,
    default=Path("artifacts/training/workyard-residual-teacher/collision-aware-fit-v1.json"),
  )
  args = parser.parse_args()
  if not 0.05 <= args.feature_scale_m <= 0.5:
    parser.error("feature scale must be inside [0.05, 0.5]")
  return args


def name_id(model: mujoco.MjModel, kind: mujoco.mjtObj, name: str) -> int:
  identifier = mujoco.mj_name2id(model, kind, name)
  if identifier < 0:
    raise ValueError(f"MuJoCo model is missing {name}")
  return identifier


def initialize_model() -> tuple[mujoco.MjModel, mujoco.MjData]:
  previous_directory = Path.cwd()
  try:
    os.chdir(base.G1_MODEL_PATH.parent)
    spec = residual._load_hybrid_actuated_g1_spec()
    spec.compiler.meshdir = str(base.G1_MODEL_PATH.parent / "meshes")
    model = spec.compile()
  finally:
    os.chdir(previous_directory)
  data = mujoco.MjData(model)
  mujoco.mj_resetData(model, data)

  free_joints = np.flatnonzero(model.jnt_type == mujoco.mjtJoint.mjJNT_FREE)
  if free_joints.size != 1:
    raise ValueError(f"Expected one G1 free joint, found {free_joints.size}")
  root_adr = int(model.jnt_qposadr[int(free_joints[0])])
  data.qpos[root_adr : root_adr + 3] = residual.RESIDUAL_ENTRY_ROOT_POSITION
  data.qpos[root_adr + 3 : root_adr + 7] = (1.0, 0.0, 0.0, 0.0)

  for joint_name, value in zip(
    (*base.BODY_JOINT_NAMES, *base.HAND_JOINT_NAMES),
    (*base.BODY_DEFAULT_POSITIONS, *((0.0,) * len(base.HAND_JOINT_NAMES))),
    strict=True,
  ):
    joint_id = name_id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
    data.qpos[int(model.jnt_qposadr[joint_id])] = value
  data.qvel[:] = 0.0
  mujoco.mj_forward(model, data)
  return model, data


def arm_joint_ids(model: mujoco.MjModel, arm: int) -> np.ndarray:
  return np.asarray([
    name_id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
    for name in ARM_JOINT_NAMES[arm]
  ], dtype=np.int32)


def arm_qpos_addresses(model: mujoco.MjModel, arm: int) -> np.ndarray:
  return model.jnt_qposadr[arm_joint_ids(model, arm)].astype(np.int32)


def arm_bounds(model: mujoco.MjModel, arm: int) -> list[tuple[float, float]]:
  joint_ids = arm_joint_ids(model, arm)
  neutral = NEUTRAL[arm]
  lower = np.maximum(neutral - AUTHORITY_RAD, model.jnt_range[joint_ids, 0] + 0.02)
  upper = np.minimum(neutral + AUTHORITY_RAD, model.jnt_range[joint_ids, 1] - 0.02)
  return list(zip(lower.tolist(), upper.tolist(), strict=True))


def collision_geom_pairs(model: mujoco.MjModel, arm: int) -> tuple[tuple[int, int], ...]:
  side = "left" if arm == 0 else "right"
  other = "right" if arm == 0 else "left"
  moving_bodies = {
    body_id
    for body_id in range(model.nbody)
    if (
      (name := mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id))
      and rearmatch(name, side)
    )
  }
  obstacle_bodies = {
    name_id(model, mujoco.mjtObj.mjOBJ_BODY, name)
    for name in ("torso_link",)
  }
  obstacle_bodies.update(
    body_id
    for body_id in range(model.nbody)
    if (
      (name := mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id))
      and rearmatch(name, other)
    )
  )
  moving_geoms = [
    geom_id for geom_id in range(model.ngeom)
    if int(model.geom_bodyid[geom_id]) in moving_bodies
    and (model.geom_contype[geom_id] != 0 or model.geom_conaffinity[geom_id] != 0)
  ]
  obstacle_geoms = [
    geom_id for geom_id in range(model.ngeom)
    if int(model.geom_bodyid[geom_id]) in obstacle_bodies
    and (model.geom_contype[geom_id] != 0 or model.geom_conaffinity[geom_id] != 0)
  ]
  return tuple((moving, obstacle) for moving in moving_geoms for obstacle in obstacle_geoms)


def rearmatch(name: str, side: str) -> bool:
  return name.startswith((
    f"{side}_shoulder_yaw",
    f"{side}_elbow",
    f"{side}_wrist",
    f"{side}_hand",
  )) and name.endswith("_link")


def geometry_top_sample(
  model: mujoco.MjModel,
  data: mujoco.MjData,
  arm: int,
  forward_offset_m: float,
  lateral_magnitude_m: float,
  yaw_magnitude_rad: float,
) -> tuple[np.ndarray, np.ndarray]:
  """Set one production-correlated base placement and return world/pelvis target."""

  side = -1.0 if arm == 0 else 1.0
  yaw = side * yaw_magnitude_rad
  root = np.asarray((
    ROD_CENTER_WORLD[0] - forward_offset_m,
    ROD_CENTER_WORLD[1] + side * lateral_magnitude_m,
    residual.RESIDUAL_ENTRY_ROOT_POSITION[2],
  ), dtype=np.float64)
  free_joints = np.flatnonzero(model.jnt_type == mujoco.mjtJoint.mjJNT_FREE)
  root_adr = int(model.jnt_qposadr[int(free_joints[0])])
  data.qpos[root_adr : root_adr + 3] = root
  data.qpos[root_adr + 3 : root_adr + 7] = (
    math.cos(yaw / 2.0), 0.0, 0.0, math.sin(yaw / 2.0)
  )
  target = ROD_CENTER_WORLD + np.asarray(
    residual.RUNTIME_GEOMETRY_TOP_WRIST_OFFSET_M[arm], dtype=np.float64
  )
  delta = target - root
  cosine = math.cos(yaw)
  sine = math.sin(yaw)
  target_pelvis = np.asarray((
    cosine * delta[0] + sine * delta[1],
    -sine * delta[0] + cosine * delta[1],
    delta[2],
  ), dtype=np.float64)
  return target, target_pelvis


class ArmProblem:
  def __init__(
    self,
    model: mujoco.MjModel,
    data: mujoco.MjData,
    arm: int,
    collision_clearance_m: float,
  ):
    self.model = model
    self.data = data
    self.arm = arm
    self.qpos_addresses = arm_qpos_addresses(model, arm)
    self.bounds = arm_bounds(model, arm)
    self.wrist_body_id = name_id(
      model, mujoco.mjtObj.mjOBJ_BODY, WRIST_BODY_NAMES[arm]
    )
    self.collision_pairs = collision_geom_pairs(model, arm)
    self.collision_clearance_m = collision_clearance_m
    self.fromto = np.zeros(6, dtype=np.float64)

  def set_q(self, q: np.ndarray) -> None:
    self.data.qpos[self.qpos_addresses] = q
    self.data.qvel[:] = 0.0
    mujoco.mj_forward(self.model, self.data)

  def wrist(self) -> np.ndarray:
    return self.data.xpos[self.wrist_body_id].copy()

  def minimum_clearance(self) -> float:
    if not self.collision_pairs:
      return math.inf
    return min(
      mujoco.mj_geomDistance(
        self.model, self.data, first, second, 0.10, self.fromto
      )
      for first, second in self.collision_pairs
    )

  def objective(self, q: np.ndarray, target: np.ndarray) -> float:
    self.set_q(q)
    wrist_error = self.wrist() - target
    collision_penalty = 0.0
    for first, second in self.collision_pairs:
      distance = mujoco.mj_geomDistance(
        self.model, self.data, first, second, 0.10, self.fromto
      )
      violation = max(self.collision_clearance_m - distance, 0.0)
      collision_penalty += violation * violation
    regularization = np.square((q - NEUTRAL[self.arm]) / AUTHORITY_RAD).sum()
    return (
      2_000.0 * float(wrist_error @ wrist_error)
      + 200_000.0 * collision_penalty
      + 0.01 * float(regularization)
    )

  def solve(self, target: np.ndarray, seeds: tuple[np.ndarray, ...], iterations: int):
    candidates = []
    for seed in seeds:
      result = minimize(
        self.objective,
        np.asarray(seed, dtype=np.float64),
        args=(target,),
        method="Powell",
        bounds=self.bounds,
        options={"maxiter": iterations, "ftol": 1e-10, "xtol": 1e-6},
      )
      self.set_q(result.x)
      wrist_error = float(np.linalg.norm(self.wrist() - target))
      clearance = self.minimum_clearance()
      score = (
        clearance < self.collision_clearance_m - 1e-4,
        wrist_error > 0.06,
        wrist_error,
        -clearance,
      )
      candidates.append((score, result.x.copy(), result))
    candidates.sort(key=lambda candidate: candidate[0])
    _, q, result = candidates[0]
    self.set_q(q)
    return {
      "q": q,
      "wrist_error_m": float(np.linalg.norm(self.wrist() - target)),
      "minimum_clearance_m": self.minimum_clearance(),
      "optimizer_success": bool(result.success),
      "optimizer_message": str(result.message),
    }


def features(targets_xy: np.ndarray, center: np.ndarray, scale: float) -> np.ndarray:
  normalized = (targets_xy - center) / scale
  x = normalized[:, 0]
  y = normalized[:, 1]
  return np.stack((np.ones_like(x), x, y, x * x, x * y, y * y), axis=-1)


def fit(args: argparse.Namespace) -> dict[str, object]:
  model, data = initialize_model()
  placements = np.asarray(
    residual.REACH_ENTRY_OBJECT_RELATIVE_PLACEMENTS, dtype=np.float64
  )
  output: dict[str, object] = {
    "protocol": "hear-collision-aware-reach-posture-fit-v2",
    "target_protocol": residual.RUNTIME_GEOMETRY_TOP_TARGET_PROTOCOL,
    "entry_correlation_protocol": residual.REACH_ENTRY_CORRELATION_PROTOCOL,
    "object_relative_placement_catalog": placements.tolist(),
    "collision_clearance_m": args.collision_clearance_m,
    "active_hand_allocation": "hand_signed_root_placement",
    "authority_rad": AUTHORITY_RAD,
    "feature_scale_m": args.feature_scale_m,
    "feature_order": ["bias", "x", "y", "x2", "xy", "y2"],
    "arms": [],
  }

  for arm in range(2):
    for reset_arm in range(2):
      data.qpos[arm_qpos_addresses(model, reset_arm)] = NEUTRAL[reset_arm]
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)
    problem = ArmProblem(model, data, arm, args.collision_clearance_m)
    training_targets = []
    training_solutions = []
    fit_samples = []
    previous = NEUTRAL[arm].copy()
    for forward, lateral, yaw in placements:
      target, target_pelvis = geometry_top_sample(
        model, data, arm, float(forward), float(lateral), float(yaw)
      )
      result = problem.solve(
        target,
        (previous.copy(), NEUTRAL[arm].copy()),
        args.maximum_iterations,
      )
      previous = result["q"].copy()
      training_targets.append(target_pelvis)
      training_solutions.append(result["q"])
      fit_samples.append({
        "object_forward_offset_m": float(forward),
        "object_lateral_magnitude_m": float(lateral),
        "yaw_magnitude_rad": float(yaw),
        "target_pelvis": target_pelvis.tolist(),
        "normalized_action": (
          (result["q"] - NEUTRAL[arm]) / AUTHORITY_RAD
        ).tolist(),
        "wrist_error_m": result["wrist_error_m"],
        "minimum_clearance_m": result["minimum_clearance_m"],
        "optimizer_success": result["optimizer_success"],
        "optimizer_message": result["optimizer_message"],
      })

    training_targets_array = np.asarray(training_targets)
    training_actions = (
      np.asarray(training_solutions) - NEUTRAL[arm]
    ) / AUTHORITY_RAD
    center = training_targets_array[:, :2].mean(axis=0)
    design = features(
      training_targets_array[:, :2], center, args.feature_scale_m
    )
    coefficients = np.linalg.lstsq(design, training_actions, rcond=None)[0].T

    validation = []
    for forward, lateral, yaw in placements:
      target, target_pelvis = geometry_top_sample(
        model, data, arm, float(forward), float(lateral), float(yaw)
      )
      predicted_action = coefficients @ features(
        target_pelvis[None, :2], center, args.feature_scale_m
      )[0]
      predicted_action = np.clip(predicted_action, -1.0, 1.0)
      q = NEUTRAL[arm] + predicted_action * AUTHORITY_RAD
      problem.set_q(q)
      validation.append({
        "wrist_error_m": float(np.linalg.norm(problem.wrist() - target)),
        "minimum_clearance_m": problem.minimum_clearance(),
        "action_clamped": bool(
          np.any(np.abs(predicted_action) >= 1.0 - 1e-9)
        ),
      })

    errors = np.asarray([sample["wrist_error_m"] for sample in validation])
    clearances = np.asarray([sample["minimum_clearance_m"] for sample in validation])
    accepted = (errors <= 0.06) & (clearances >= args.collision_clearance_m - 1e-4)
    output["arms"].append({
      "side": "left" if arm == 0 else "right",
      "feature_center_xy_m": center.tolist(),
      "normalized_action_coefficients": coefficients.tolist(),
      "fit_samples": fit_samples,
      "validation": {
        "sample_count": len(validation),
        "accepted_count": int(accepted.sum()),
        "accepted_rate": float(accepted.mean()),
        "within_tolerance_rate": float((errors <= 0.06).mean()),
        "collision_clear_rate": float(
          (clearances >= args.collision_clearance_m - 1e-4).mean()
        ),
        "mean_error_m": float(errors.mean()),
        "p90_error_m": float(np.quantile(errors, 0.9)),
        "maximum_error_m": float(errors.max()),
        "minimum_clearance_m": float(clearances.min()),
        "action_clamp_rate": float(np.mean([
          sample["action_clamped"] for sample in validation
        ])),
      },
    })
  return output


def main() -> None:
  args = parse_args()
  report = fit(args)
  args.output.parent.mkdir(parents=True, exist_ok=True)
  args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
  summary = {
    arm["side"]: arm["validation"]
    for arm in report["arms"]
  }
  print(json.dumps({"output": str(args.output), "validation": summary}, indent=2))


if __name__ == "__main__":
  main()
