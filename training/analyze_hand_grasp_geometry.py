"""Scan the real G1 hand meshes for opposing cylindrical grasp pockets.

This is an offline geometry diagnostic, not a policy or simulator substitute.
It uses MuJoCo's compiled collision model to find wrist-frame rod positions
where a configured thumb and at least one configured finger contact opposite
sides of the same 3 cm-radius vertical cylinder.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import mujoco
import numpy as np


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = REPOSITORY_ROOT / "assets" / "humanoid" / "g1" / "g1_with_hands.xml"
HAND_ENDPOINTS = {
  "left": (0.0, 1.0472, 1.74533, -1.5708, -1.74533, -1.5708, -1.74533),
  "right": (0.0, -1.0472, -1.74533, 1.5708, 1.74533, 1.5708, 1.74533),
}
HAND_JOINT_SUFFIXES = (
  "thumb_0_joint",
  "thumb_1_joint",
  "thumb_2_joint",
  "middle_0_joint",
  "middle_1_joint",
  "index_0_joint",
  "index_1_joint",
)


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser()
  parser.add_argument("--side", choices=("left", "right", "both"), default="both")
  parser.add_argument("--thumb", type=float, default=0.95)
  parser.add_argument("--finger", type=float, default=0.88)
  parser.add_argument("--x-min", type=float, default=0.10)
  parser.add_argument("--x-max", type=float, default=0.21)
  parser.add_argument("--lateral-min", type=float, default=0.0)
  parser.add_argument("--lateral-max", type=float, default=0.10)
  parser.add_argument("--step", type=float, default=0.002)
  parser.add_argument("--limit", type=int, default=30)
  parser.add_argument("--probe-x", type=float, default=0.12)
  parser.add_argument("--probe-lateral", type=float, default=0.04)
  args = parser.parse_args()
  if (
    not 0.0 <= args.thumb <= 1.0
    or not 0.0 <= args.finger <= 1.0
    or args.x_min >= args.x_max
    or args.lateral_min < 0.0
    or args.lateral_min >= args.lateral_max
    or args.step <= 0.0
    or args.limit <= 0
  ):
    parser.error("Invalid coordination, scan bounds, step, or limit")
  return args


def build_model() -> tuple[mujoco.MjModel, int, int]:
  if not MODEL_PATH.is_file():
    raise FileNotFoundError(f"G1 model is missing: {MODEL_PATH}")
  spec = mujoco.MjSpec.from_file(str(MODEL_PATH))
  spec.compiler.meshdir = str(MODEL_PATH.parent / "meshes")
  rod = spec.worldbody.add_body(name="grasp_scan_rod")
  rod.add_freejoint(name="grasp_scan_rod_joint")
  rod.add_geom(
    name="grasp_scan_rod_geom",
    type=mujoco.mjtGeom.mjGEOM_CYLINDER,
    size=(0.03, 0.11, 0.0),
    mass=0.35,
    friction=(0.8, 0.012, 0.002),
    condim=4,
  )
  model = spec.compile()
  rod_joint_id = mujoco.mj_name2id(
    model, mujoco.mjtObj.mjOBJ_JOINT, "grasp_scan_rod_joint"
  )
  rod_geom_id = mujoco.mj_name2id(
    model, mujoco.mjtObj.mjOBJ_GEOM, "grasp_scan_rod_geom"
  )
  if rod_joint_id < 0 or rod_geom_id < 0:
    raise RuntimeError("Compiled grasp scanner lost the rod joint or geom")
  return model, rod_joint_id, rod_geom_id


def set_hand_coordination(
  model: mujoco.MjModel,
  data: mujoco.MjData,
  side: str,
  thumb: float,
  finger: float,
) -> None:
  coordination = (0.0, thumb, thumb, finger, finger, finger, finger)
  for suffix, endpoint, value in zip(
    HAND_JOINT_SUFFIXES, HAND_ENDPOINTS[side], coordination, strict=True
  ):
    joint_id = mujoco.mj_name2id(
      model, mujoco.mjtObj.mjOBJ_JOINT, f"{side}_hand_{suffix}"
    )
    if joint_id < 0:
      raise RuntimeError(f"Missing hand joint: {side}_hand_{suffix}")
    data.qpos[model.jnt_qposadr[joint_id]] = endpoint * value


def hand_contact_group(model: mujoco.MjModel, geom_id: int, side: str) -> str | None:
  body_id = int(model.geom_bodyid[geom_id])
  body_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id) or ""
  if not body_name.startswith(f"{side}_hand_"):
    return None
  return "thumb" if "_thumb_" in body_name else "finger"


def scan_side(
  model: mujoco.MjModel,
  rod_joint_id: int,
  rod_geom_id: int,
  side: str,
  args: argparse.Namespace,
) -> dict[str, object]:
  data = mujoco.MjData(model)
  set_hand_coordination(model, data, side, args.thumb, args.finger)
  mujoco.mj_forward(model, data)
  wrist_id = mujoco.mj_name2id(
    model, mujoco.mjtObj.mjOBJ_BODY, f"{side}_wrist_yaw_link"
  )
  if wrist_id < 0:
    raise RuntimeError(f"Missing {side} wrist body")
  wrist_position = data.xpos[wrist_id].copy()
  wrist_rotation = data.xmat[wrist_id].reshape(3, 3).copy()
  wrist_quaternion = np.empty(4, dtype=np.float64)
  mujoco.mju_mat2Quat(wrist_quaternion, wrist_rotation.reshape(-1))
  rod_qpos_address = int(model.jnt_qposadr[rod_joint_id])
  lateral_sign = -1.0 if side == "left" else 1.0

  def inspect_position(local_position: np.ndarray) -> dict[str, object]:
    data.qpos[rod_qpos_address:rod_qpos_address + 3] = (
      wrist_position + wrist_rotation @ local_position
    )
    data.qpos[rod_qpos_address + 3:rod_qpos_address + 7] = wrist_quaternion
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)
    grouped_normals: dict[str, list[np.ndarray]] = {"thumb": [], "finger": []}
    surfaces: dict[str, set[str]] = {"thumb": set(), "finger": set()}
    minimum_distance = 0.0
    for contact_index in range(data.ncon):
      contact = data.contact[contact_index]
      if rod_geom_id not in (contact.geom1, contact.geom2):
        continue
      hand_geom_id = contact.geom2 if contact.geom1 == rod_geom_id else contact.geom1
      group = hand_contact_group(model, hand_geom_id, side)
      if group is None:
        continue
      normal = np.asarray(contact.frame[:3], dtype=np.float64)
      if contact.geom2 == rod_geom_id:
        normal = -normal
      normal_norm = np.linalg.norm(normal)
      if normal_norm <= 1e-9:
        continue
      grouped_normals[group].append(normal / normal_norm)
      body_id = int(model.geom_bodyid[hand_geom_id])
      surfaces[group].add(
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id) or "unknown"
      )
      minimum_distance = min(minimum_distance, float(contact.dist))
    minimum_dot = None
    if grouped_normals["thumb"] and grouped_normals["finger"]:
      minimum_dot = min(
        float(np.dot(thumb_normal, finger_normal))
        for thumb_normal in grouped_normals["thumb"]
        for finger_normal in grouped_normals["finger"]
      )
    return {
      "rod_position_wrist_frame_m": local_position.tolist(),
      "maximum_penetration_m": -minimum_distance,
      "minimum_thumb_finger_normal_dot": minimum_dot,
      "thumb_surfaces": sorted(surfaces["thumb"]),
      "finger_surfaces": sorted(surfaces["finger"]),
    }

  probe = inspect_position(np.array((
    args.probe_x,
    lateral_sign * args.probe_lateral,
    0.0,
  )))
  candidates: list[dict[str, object]] = []
  scanned = 0
  x_values = np.arange(args.x_min, args.x_max + args.step * 0.5, args.step)
  lateral_values = np.arange(
    args.lateral_min, args.lateral_max + args.step * 0.5, args.step
  )
  for x in x_values:
    for lateral in lateral_values:
      scanned += 1
      local_position = np.array((x, lateral_sign * lateral, 0.0))
      data.qpos[rod_qpos_address:rod_qpos_address + 3] = (
        wrist_position + wrist_rotation @ local_position
      )
      data.qpos[rod_qpos_address + 3:rod_qpos_address + 7] = wrist_quaternion
      data.qvel[:] = 0.0
      mujoco.mj_forward(model, data)
      grouped_normals: dict[str, list[np.ndarray]] = {"thumb": [], "finger": []}
      surfaces: dict[str, set[str]] = {"thumb": set(), "finger": set()}
      minimum_distance = 0.0
      for contact_index in range(data.ncon):
        contact = data.contact[contact_index]
        if rod_geom_id not in (contact.geom1, contact.geom2):
          continue
        hand_geom_id = contact.geom2 if contact.geom1 == rod_geom_id else contact.geom1
        group = hand_contact_group(model, hand_geom_id, side)
        if group is None:
          continue
        normal = np.asarray(contact.frame[:3], dtype=np.float64)
        if contact.geom2 == rod_geom_id:
          normal = -normal
        normal_norm = np.linalg.norm(normal)
        if normal_norm <= 1e-9:
          continue
        grouped_normals[group].append(normal / normal_norm)
        body_id = int(model.geom_bodyid[hand_geom_id])
        surfaces[group].add(
          mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id) or "unknown"
        )
        minimum_distance = min(minimum_distance, float(contact.dist))
      if not grouped_normals["thumb"] or not grouped_normals["finger"]:
        continue
      minimum_dot = min(
        float(np.dot(thumb_normal, finger_normal))
        for thumb_normal in grouped_normals["thumb"]
        for finger_normal in grouped_normals["finger"]
      )
      candidates.append({
        "rod_position_wrist_frame_m": local_position.tolist(),
        "minimum_thumb_finger_normal_dot": minimum_dot,
        "maximum_penetration_m": -minimum_distance,
        "thumb_surfaces": sorted(surfaces["thumb"]),
        "finger_surfaces": sorted(surfaces["finger"]),
      })
  candidates.sort(key=lambda item: (
    item["minimum_thumb_finger_normal_dot"],
    item["maximum_penetration_m"],
  ))
  opposing = [
    candidate
    for candidate in candidates
    if candidate["minimum_thumb_finger_normal_dot"] < -0.20
  ]
  shallow_opposing = sorted(
    opposing,
    key=lambda item: (
      item["maximum_penetration_m"],
      item["minimum_thumb_finger_normal_dot"],
    ),
  )
  return {
    "side": side,
    "scanned_position_count": scanned,
    "dual_surface_position_count": len(candidates),
    "opposing_position_count": len(opposing),
    "probe": probe,
    "best_candidates": candidates[:args.limit],
    "shallow_opposing_candidates": shallow_opposing[:args.limit],
  }


def main() -> None:
  args = parse_args()
  model, rod_joint_id, rod_geom_id = build_model()
  sides = ("left", "right") if args.side == "both" else (args.side,)
  payload = {
    "protocol": "hear-g1-hand-cylinder-grasp-geometry-v1",
    "model": str(MODEL_PATH.relative_to(REPOSITORY_ROOT)).replace("\\", "/"),
    "rod_radius_m": 0.03,
    "thumb_coordination": args.thumb,
    "finger_coordination": args.finger,
    "opposition_coordination": 0.0,
    "scan": [
      scan_side(model, rod_joint_id, rod_geom_id, side, args) for side in sides
    ],
  }
  print(json.dumps(payload, indent=2))


if __name__ == "__main__":
  main()
