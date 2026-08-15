"""Proprioceptive whole-body get-up expert for HEAR's Unitree G1.

The actor is deliberately below the Agent Harness boundary.  It receives only
measured proprioception, previous action, root height, and binary foot contact;
reference identity, motion phase, scripted trajectories, goals, and Agent state
are absent.  Training follows a two-stage recovery curriculum: first discover
feasible stand-up behavior from progressively harder ground states with a
decaying vertical assist, then retain success while adding smoothness, effort,
impact, symmetry, and domain-randomization pressure.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Final

import mujoco
import torch

from mjlab.actuator.xml_actuator import XmlActuatorCfg
from mjlab.entity import Entity, EntityArticulationInfoCfg, EntityCfg
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp
from mjlab.envs.mdp import dr
from mjlab.managers.action_manager import ActionTerm, ActionTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.rl import RslRlModelCfg, RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg
from mjlab.scene import SceneCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.tasks.registry import list_tasks, register_mjlab_task
from mjlab.terrains import TerrainEntityCfg
from mjlab.utils.lab_api.math import quat_from_euler_xyz
from mjlab.viewer import ViewerConfig

import workyard_mjlab_env as base

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


TASK_ID: Final = "Hear-G1-Getup-v1"
OBSERVATION_PROTOCOL: Final = "hear-g1-getup-proprioception-v1"
ACTION_PROTOCOL: Final = "hear-g1-getup-joint-target-v1"
OBSERVATION_SIZE: Final = 99
ACTION_SIZE: Final = 29
CONTROL_STEP_SECONDS: Final = 0.02
PHYSICS_STEP_SECONDS: Final = 0.005
STABLE_CONTROL_STEPS: Final = 20
CURRICULUM_CONTROL_STEPS: Final = 40_000
EPISODE_SECONDS: Final = 12.0

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
BODY_REPORT_PATH = (
  REPOSITORY_ROOT
  / "assets" / "humanoid" / "controllers" / "mjlab-g1-velocity"
  / "training-report.json"
)

BODY_JOINT_NAMES: Final = base.BODY_JOINT_NAMES
HAND_JOINT_NAMES: Final = base.HAND_JOINT_NAMES
DEFAULT_JOINT_POSITIONS: Final = base.BODY_DEFAULT_POSITIONS

# Soft deployment bounds preserve 4% headroom at the physical limits while
# retaining deep knee/hip flexion required by prone and supine recovery.
PHYSICAL_JOINT_LIMITS: Final = (
  (-2.5307, 2.8798), (-0.5236, 2.9671), (-2.7576, 2.7576),
  (-0.087267, 2.8798), (-0.87267, 0.5236), (-0.2618, 0.2618),
  (-2.5307, 2.8798), (-2.9671, 0.5236), (-2.7576, 2.7576),
  (-0.087267, 2.8798), (-0.87267, 0.5236), (-0.2618, 0.2618),
  (-2.618, 2.618), (-0.52, 0.52), (-0.52, 0.52),
  (-3.0892, 2.6704), (-1.5882, 2.2515), (-2.618, 2.618),
  (-1.0472, 2.0944), (-1.97222, 1.97222), (-1.61443, 1.61443),
  (-1.61443, 1.61443), (-3.0892, 2.6704), (-2.2515, 1.5882),
  (-2.618, 2.618), (-1.0472, 2.0944), (-1.97222, 1.97222),
  (-1.61443, 1.61443), (-1.61443, 1.61443),
)


def _soft_limit(limit: tuple[float, float], neutral: float) -> tuple[float, float]:
  lower, upper = limit
  return (
    neutral + 0.96 * (lower - neutral),
    neutral + 0.96 * (upper - neutral),
  )


SOFT_JOINT_LIMITS: Final = tuple(
  _soft_limit(limit, neutral)
  for limit, neutral in zip(
    PHYSICAL_JOINT_LIMITS, DEFAULT_JOINT_POSITIONS, strict=True
  )
)
JOINT_LOWER_LIMITS: Final = tuple(limit[0] for limit in SOFT_JOINT_LIMITS)
JOINT_UPPER_LIMITS: Final = tuple(limit[1] for limit in SOFT_JOINT_LIMITS)
RESET_POSE_NAMES: Final = (
  "standing", "crouched", "prone", "supine", "left_side", "right_side",
  "intermediate",
)


def _metadata_vector(
  source: object,
  label: str,
  size: int,
  predicate=lambda _: True,
) -> tuple[float, ...]:
  if not isinstance(source, str):
    raise ValueError(f"Body policy {label} metadata is missing")
  values = tuple(float(value) for value in source.split(","))
  if len(values) != size or any(
    not math.isfinite(value) or not predicate(value) for value in values
  ):
    raise ValueError(f"Body policy {label} metadata is invalid")
  return values


def _body_policy_actuation() -> tuple[tuple[float, ...], tuple[float, ...]]:
  report = json.loads(BODY_REPORT_PATH.read_text(encoding="utf-8"))
  metadata = report.get("onnx", {}).get("metadata", {})
  names = metadata.get("joint_names")
  if not isinstance(names, str) or tuple(names.split(",")) != BODY_JOINT_NAMES:
    raise ValueError("Body policy joint order differs from the get-up morphology")
  stiffness = _metadata_vector(
    metadata.get("joint_stiffness"), "stiffness", ACTION_SIZE,
    lambda value: value > 0.0,
  )
  damping = _metadata_vector(
    metadata.get("joint_damping"), "damping", ACTION_SIZE,
    lambda value: value > 0.0,
  )
  reported_default = _metadata_vector(
    metadata.get("default_joint_pos"), "neutral pose", ACTION_SIZE,
  )
  if any(
    not math.isclose(left, right, abs_tol=1e-5)
    for left, right in zip(
      reported_default, DEFAULT_JOINT_POSITIONS, strict=True
    )
  ):
    raise ValueError("Body and get-up experts disagree on the neutral pose")
  return stiffness, damping


JOINT_STIFFNESS, JOINT_DAMPING = _body_policy_actuation()


def _load_getup_g1_spec() -> mujoco.MjSpec:
  """Match the deployed G1 joint plant instead of the generic XML kp=500."""
  from mjlab.asset_zoo.robots.unitree_g1.g1_constants import G1_ARTICULATION

  spec = base._load_g1_spec()
  joints = {joint.name: joint for joint in spec.joints}
  actuators = {actuator.target: actuator for actuator in spec.actuators}
  for index, joint_name in enumerate(BODY_JOINT_NAMES):
    matches = [
      actuator
      for actuator in G1_ARTICULATION.actuators
      if any(
        re.fullmatch(pattern, joint_name)
        for pattern in actuator.target_names_expr
      )
    ]
    if len(matches) != 1:
      raise ValueError(f"Source G1 actuation is ambiguous for {joint_name}")
    source = matches[0]
    joint = joints[joint_name]
    actuator = actuators[joint_name]
    if source.armature is None or source.effort_limit is None:
      raise ValueError(f"Source G1 actuation is incomplete for {joint_name}")
    joint.armature = float(source.armature)
    joint.damping[:] = 0.0
    joint.frictionloss = 0.0
    actuator.set_to_position(
      kp=JOINT_STIFFNESS[index],
      kv=JOINT_DAMPING[index],
      inheritrange=True,
    )
  return spec


def _robot_cfg() -> EntityCfg:
  joint_pos = dict(zip(
    BODY_JOINT_NAMES, DEFAULT_JOINT_POSITIONS, strict=True
  ))
  joint_pos.update({name: 0.0 for name in HAND_JOINT_NAMES})
  return EntityCfg(
    spec_fn=_load_getup_g1_spec,
    init_state=EntityCfg.InitialStateCfg(
      pos=(0.0, 0.0, 0.793),
      joint_pos=joint_pos,
      joint_vel={".*": 0.0},
    ),
    articulation=EntityArticulationInfoCfg(
      actuators=(XmlActuatorCfg(
        target_names_expr=BODY_JOINT_NAMES + HAND_JOINT_NAMES,
        command_field="position",
      ),),
      soft_joint_pos_limit_factor=0.96,
    ),
  )


def _contact_sensors() -> tuple[ContactSensorCfg, ...]:
  feet = ContactSensorCfg(
    name="feet_ground_contact",
    primary=ContactMatch(
      mode="subtree",
      pattern=("left_ankle_roll_link", "right_ankle_roll_link"),
      entity="robot",
    ),
    secondary=ContactMatch(mode="body", pattern="terrain"),
    fields=("found", "force"),
    reduce="netforce",
    num_slots=1,
    track_air_time=True,
    history_length=4,
  )
  body = ContactSensorCfg(
    name="body_ground_contact",
    primary=ContactMatch(mode="body", pattern=r"^(?:pelvis|.*_link)$", entity="robot"),
    secondary=ContactMatch(mode="body", pattern="terrain"),
    fields=("found", "force"),
    reduce="maxforce",
    num_slots=1,
    history_length=4,
  )
  return feet, body


def curriculum_progress(env: ManagerBasedRlEnv) -> float:
  return min(1.0, float(env.common_step_counter) / CURRICULUM_CONTROL_STEPS)


@dataclass(kw_only=True)
class G1GetupActionCfg(ActionTermCfg):
  assistance_force_n: float = 120.0

  def build(self, env: ManagerBasedRlEnv) -> "G1GetupAction":
    return G1GetupAction(self, env)


class G1GetupAction(ActionTerm):
  cfg: G1GetupActionCfg

  def __init__(self, cfg: G1GetupActionCfg, env: ManagerBasedRlEnv):
    super().__init__(cfg, env)
    body_ids, names = self._entity.find_joints(
      BODY_JOINT_NAMES, preserve_order=True
    )
    hand_ids, hand_names = self._entity.find_joints(
      HAND_JOINT_NAMES, preserve_order=True
    )
    pelvis_ids, pelvis_names = self._entity.find_bodies(
      "pelvis", preserve_order=True
    )
    if tuple(names) != BODY_JOINT_NAMES or tuple(hand_names) != HAND_JOINT_NAMES:
      raise ValueError("Get-up joint order differs from the deployment contract")
    if len(pelvis_ids) != 1 or pelvis_names != ["pelvis"]:
      raise ValueError("Get-up assistance requires the G1 pelvis")
    self._body_ids = torch.tensor(body_ids, dtype=torch.long, device=self.device)
    self._hand_ids = torch.tensor(hand_ids, dtype=torch.long, device=self.device)
    self._pelvis_ids = pelvis_ids
    self._neutral = torch.tensor(
      DEFAULT_JOINT_POSITIONS, dtype=torch.float32, device=self.device
    ).unsqueeze(0)
    self._lower = torch.tensor(
      JOINT_LOWER_LIMITS, dtype=torch.float32, device=self.device
    ).unsqueeze(0)
    self._upper = torch.tensor(
      JOINT_UPPER_LIMITS, dtype=torch.float32, device=self.device
    ).unsqueeze(0)
    self._raw_action = torch.zeros(
      (self.num_envs, ACTION_SIZE), dtype=torch.float32, device=self.device
    )
    self._targets = self._neutral.repeat(self.num_envs, 1)
    self._open_hands = torch.zeros(
      (self.num_envs, len(HAND_JOINT_NAMES)),
      dtype=torch.float32,
      device=self.device,
    )
    self._assist_force = torch.zeros(
      (self.num_envs, 1, 3), dtype=torch.float32, device=self.device
    )
    self._assist_torque = torch.zeros_like(self._assist_force)

  @property
  def action_dim(self) -> int:
    return ACTION_SIZE

  @property
  def raw_action(self) -> torch.Tensor:
    return self._raw_action

  @property
  def targets(self) -> torch.Tensor:
    return self._targets

  def process_actions(self, actions: torch.Tensor) -> None:
    if actions.shape != (self.num_envs, ACTION_SIZE):
      raise ValueError(f"Expected get-up action [B, {ACTION_SIZE}], got {actions.shape}")
    if not torch.isfinite(actions).all():
      raise ValueError("Get-up actor emitted a non-finite action")
    self._raw_action[:] = actions.clamp(-1.0, 1.0)
    positive = self._neutral + self._raw_action * (self._upper - self._neutral)
    negative = self._neutral + self._raw_action * (self._neutral - self._lower)
    self._targets[:] = torch.where(self._raw_action >= 0.0, positive, negative)

  def apply_actions(self) -> None:
    self._entity.set_joint_position_target(self._targets, joint_ids=self._body_ids)
    self._entity.set_joint_position_target(self._open_hands, joint_ids=self._hand_ids)
    self._assist_force.zero_()
    if self.cfg.assistance_force_n > 0.0:
      progress = curriculum_progress(self._env)
      fallen = (self._entity.data.root_link_pos_w[:, 2] < 0.67).float()
      self._assist_force[:, 0, 2] = (
        self.cfg.assistance_force_n * (1.0 - progress) * fallen
      )
    self._entity.write_external_wrench_to_sim(
      self._assist_force,
      self._assist_torque,
      body_ids=self._pelvis_ids,
    )

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    if env_ids is None:
      env_ids = slice(None)
    self._raw_action[env_ids] = 0.0
    self._targets[env_ids] = self._neutral


def _getup_action(env: ManagerBasedRlEnv) -> G1GetupAction:
  action = env.action_manager.get_term("getup")
  if not isinstance(action, G1GetupAction):
    raise TypeError("G1 get-up action term is unavailable")
  return action


class G1GetupObservation:
  def __init__(self, cfg: ObservationTermCfg, env: ManagerBasedRlEnv):
    del cfg
    self.robot: Entity = env.scene["robot"]
    body_ids, names = self.robot.find_joints(
      BODY_JOINT_NAMES, preserve_order=True
    )
    if tuple(names) != BODY_JOINT_NAMES:
      raise ValueError("Get-up observation joint order is invalid")
    self.body_ids = torch.tensor(body_ids, dtype=torch.long, device=env.device)
    self.neutral = torch.tensor(
      DEFAULT_JOINT_POSITIONS, dtype=torch.float32, device=env.device
    ).unsqueeze(0)

  def __call__(self, env: ManagerBasedRlEnv) -> torch.Tensor:
    feet, _, _ = base._foot_contact_summary(env)
    action = _getup_action(env)
    observation = torch.cat((
      self.robot.data.projected_gravity_b,
      self.robot.data.root_link_ang_vel_b,
      self.robot.data.root_link_lin_vel_b,
      self.robot.data.root_link_pos_w[:, 2:3],
      self.robot.data.joint_pos[:, self.body_ids] - self.neutral,
      self.robot.data.joint_vel[:, self.body_ids],
      action.raw_action,
      feet.float(),
    ), dim=-1)
    if observation.shape != (env.num_envs, OBSERVATION_SIZE):
      raise RuntimeError(f"G1 get-up observation contract drifted: {observation.shape}")
    return observation


def _reset_pose_catalog(device: str) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
  neutral = torch.tensor(DEFAULT_JOINT_POSITIONS, device=device)
  poses = neutral.repeat(len(RESET_POSE_NAMES), 1)
  # Crouched and intermediate states teach the final support transition before
  # the reset distribution expands to full prone/supine/side recovery.
  for row in (1, 6):
    poses[row, 0] = -0.95
    poses[row, 3] = 1.75
    poses[row, 4] = -0.70
    poses[row, 6] = -0.95
    poses[row, 9] = 1.75
    poses[row, 10] = -0.70
  # Ground states begin with collision-safe arm clearance, not a deployment
  # trajectory.  Noise and PPO remain responsible for the control sequence.
  for row in (2, 3, 4, 5):
    poses[row, 0] = -0.45
    poses[row, 3] = 1.05
    poses[row, 6] = -0.45
    poses[row, 9] = 1.05
    poses[row, 15] = 0.75
    poses[row, 18] = 1.20
    poses[row, 22] = 0.75
    poses[row, 25] = 1.20
  root_height = torch.tensor(
    (0.793, 0.53, 0.245, 0.245, 0.27, 0.27, 0.42),
    dtype=torch.float32,
    device=device,
  )
  euler = torch.tensor((
    (0.0, 0.0, 0.0),
    (0.0, 0.0, 0.0),
    (0.0, math.pi / 2, 0.0),
    (0.0, -math.pi / 2, 0.0),
    (math.pi / 2, 0.0, 0.0),
    (-math.pi / 2, 0.0, 0.0),
    (0.65, 0.75, 0.0),
  ), dtype=torch.float32, device=device)
  return poses, root_height, euler


def reset_getup_state(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor | None,
  curriculum: bool,
) -> None:
  robot: Entity = env.scene["robot"]
  if env_ids is None:
    env_ids = torch.arange(env.num_envs, dtype=torch.long, device=env.device)
  else:
    env_ids = env_ids.to(device=env.device, dtype=torch.long)
  count = len(env_ids)
  progress = curriculum_progress(env) if curriculum else 1.0
  sample = torch.rand(count, device=env.device)
  if progress < 0.25:
    # Early feasibility discovery: crouch/intermediate with a small standing
    # retention slice.
    category = torch.where(
      sample < 0.15,
      torch.zeros_like(sample, dtype=torch.long),
      torch.where(
        sample < 0.55,
        torch.ones_like(sample, dtype=torch.long),
        torch.full_like(sample, 6, dtype=torch.long),
      ),
    )
  elif progress < 0.60:
    middle_catalog = torch.tensor((1, 2, 3, 4, 5, 6), device=env.device)
    category = middle_catalog[torch.randint(0, len(middle_catalog), (count,), device=env.device)]
  else:
    probabilities = torch.tensor(
      (0.05, 0.10, 0.20, 0.20, 0.17, 0.17, 0.11),
      dtype=torch.float32,
      device=env.device,
    )
    category = torch.multinomial(probabilities, count, replacement=True)

  if not hasattr(env, "getup_reset_category"):
    env.getup_reset_category = torch.zeros(
      env.num_envs, dtype=torch.long, device=env.device
    )
  env.getup_reset_category[env_ids] = category

  poses, root_heights, eulers = _reset_pose_catalog(env.device)
  joint_position = poses[category].clone()
  noise_scale = torch.where(
    (category == 0).unsqueeze(-1),
    torch.full((count, 1), 0.04, device=env.device),
    torch.full((count, 1), 0.14, device=env.device),
  )
  joint_position += (2.0 * torch.rand_like(joint_position) - 1.0) * noise_scale
  lower = torch.tensor(JOINT_LOWER_LIMITS, device=env.device).unsqueeze(0)
  upper = torch.tensor(JOINT_UPPER_LIMITS, device=env.device).unsqueeze(0)
  joint_position.clamp_(lower, upper)
  joint_velocity = torch.empty_like(joint_position).uniform_(-0.15, 0.15)

  euler_values = eulers[category].clone()
  euler_values[:, 0:2] += torch.empty(
    (count, 2), device=env.device
  ).uniform_(-0.12, 0.12)
  euler_values[:, 2] = torch.empty(count, device=env.device).uniform_(
    -math.pi, math.pi
  )
  root_quaternion = quat_from_euler_xyz(
    euler_values[:, 0], euler_values[:, 1], euler_values[:, 2]
  )
  root_position = env.scene.env_origins[env_ids].clone()
  root_position[:, :2] += torch.empty(
    (count, 2), device=env.device
  ).uniform_(-0.08, 0.08)
  root_position[:, 2] += root_heights[category]
  root_velocity = torch.empty((count, 6), device=env.device).uniform_(-0.18, 0.18)

  body_ids, _ = robot.find_joints(BODY_JOINT_NAMES, preserve_order=True)
  hand_ids, _ = robot.find_joints(HAND_JOINT_NAMES, preserve_order=True)
  robot.data.write_root_pose(
    torch.cat((root_position, root_quaternion), dim=-1), env_ids
  )
  robot.data.write_root_velocity(root_velocity, env_ids)
  robot.data.write_joint_state(
    joint_position,
    joint_velocity,
    torch.tensor(body_ids, dtype=torch.long, device=env.device),
    env_ids,
  )
  open_hands = torch.zeros(
    (count, len(HAND_JOINT_NAMES)), device=env.device
  )
  robot.data.write_joint_state(
    open_hands,
    open_hands,
    torch.tensor(hand_ids, dtype=torch.long, device=env.device),
    env_ids,
  )


class HeightProgress:
  def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
    self.body_name: str = cfg.params["body_name"]
    self.robot: Entity = env.scene["robot"]
    self.previous = torch.zeros(env.num_envs, device=env.device)
    self.valid = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    if self.body_name == "root":
      self.body_id = None
    else:
      ids, names = self.robot.find_bodies(self.body_name, preserve_order=True)
      if len(ids) != 1 or names != [self.body_name]:
        raise ValueError(f"Get-up reward body is missing: {self.body_name}")
      self.body_id = ids[0]

  def reset(self, env_ids: torch.Tensor | slice | None) -> None:
    if env_ids is None:
      env_ids = slice(None)
    self.valid[env_ids] = False

  def __call__(self, env: ManagerBasedRlEnv, body_name: str) -> torch.Tensor:
    del env, body_name
    current = (
      self.robot.data.root_link_pos_w[:, 2]
      if self.body_id is None
      else self.robot.data.body_link_pos_w[:, self.body_id, 2]
    )
    progress = torch.where(
      self.valid,
      (current - self.previous).clamp(-0.08, 0.08),
      torch.zeros_like(current),
    )
    self.previous[:] = current
    self.valid[:] = True
    return progress


def root_height_reward(env: ManagerBasedRlEnv) -> torch.Tensor:
  robot: Entity = env.scene["robot"]
  return ((robot.data.root_link_pos_w[:, 2] - 0.20) / 0.59).clamp(0.0, 1.0)


def torso_height_reward(env: ManagerBasedRlEnv) -> torch.Tensor:
  robot: Entity = env.scene["robot"]
  ids, _ = robot.find_bodies("torso_link", preserve_order=True)
  return ((robot.data.body_link_pos_w[:, ids[0], 2] - 0.20) / 0.95).clamp(0.0, 1.0)


def upright_reward(env: ManagerBasedRlEnv) -> torch.Tensor:
  robot: Entity = env.scene["robot"]
  return ((-robot.data.projected_gravity_b[:, 2] + 1.0) * 0.5).clamp(0.0, 1.0)


def foot_support_reward(env: ManagerBasedRlEnv) -> torch.Tensor:
  found, _, _ = base._foot_contact_summary(env)
  return 0.35 * found.any(dim=-1).float() + 0.65 * found.all(dim=-1).float()


def standing_pose_reward(env: ManagerBasedRlEnv) -> torch.Tensor:
  action = _getup_action(env)
  position = action._entity.data.joint_pos[:, action._body_ids]
  error = position - action._neutral
  upright = upright_reward(env)
  return torch.exp(-error.square().mean(dim=-1) / (0.30 ** 2)) * upright


def stable_standing(env: ManagerBasedRlEnv) -> torch.Tensor:
  robot: Entity = env.scene["robot"]
  found, _, _ = base._foot_contact_summary(env)
  root_speed = torch.linalg.vector_norm(robot.data.root_link_lin_vel_b, dim=-1)
  angular_speed = torch.linalg.vector_norm(robot.data.root_link_ang_vel_b, dim=-1)
  body_ids = _getup_action(env)._body_ids
  joint_speed = robot.data.joint_vel[:, body_ids].abs().amax(dim=-1)
  return (
    (robot.data.root_link_pos_w[:, 2] >= 0.70)
    & (-robot.data.projected_gravity_b[:, 2] >= 0.90)
    & found.all(dim=-1)
    & (root_speed <= 0.35)
    & (angular_speed <= 0.50)
    & (joint_speed <= 1.50)
  )


class StableStandingTermination:
  def __init__(self, cfg: TerminationTermCfg, env: ManagerBasedRlEnv):
    del cfg
    self.counter = torch.zeros(
      env.num_envs, dtype=torch.long, device=env.device
    )

  def reset(self, env_ids: torch.Tensor | slice | None) -> None:
    if env_ids is None:
      env_ids = slice(None)
    self.counter[env_ids] = 0

  def __call__(self, env: ManagerBasedRlEnv) -> torch.Tensor:
    stable = stable_standing(env)
    self.counter[:] = torch.where(stable, self.counter + 1, 0)
    return self.counter >= STABLE_CONTROL_STEPS


def stable_bonus(env: ManagerBasedRlEnv) -> torch.Tensor:
  return stable_standing(env).float()


def regularization_progress(env: ManagerBasedRlEnv) -> torch.Tensor:
  progress = curriculum_progress(env)
  value = max(0.0, min(1.0, (progress - 0.35) / 0.65))
  return torch.full((env.num_envs,), value, device=env.device)


def joint_velocity_cost(env: ManagerBasedRlEnv) -> torch.Tensor:
  action = _getup_action(env)
  value = action._entity.data.joint_vel[:, action._body_ids].square().mean(dim=-1)
  return value * regularization_progress(env)


def actuator_effort_cost(env: ManagerBasedRlEnv) -> torch.Tensor:
  action = _getup_action(env)
  effort = action._entity.data.qfrc_actuator[:, action._body_ids]
  return effort.square().mean(dim=-1) * regularization_progress(env) / (80.0 ** 2)


def impact_cost(env: ManagerBasedRlEnv) -> torch.Tensor:
  sensor = base._contact_sensor(env, "body_ground_contact")
  force = sensor.data.force
  if force is None:
    return torch.zeros(env.num_envs, device=env.device)
  maximum = torch.linalg.vector_norm(force, dim=-1).flatten(1).amax(dim=-1)
  return torch.relu(maximum - 350.0) / 350.0 * regularization_progress(env)


def symmetry_cost(env: ManagerBasedRlEnv) -> torch.Tensor:
  action = _getup_action(env).raw_action
  lower_sign = torch.tensor(
    (1.0, -1.0, -1.0, 1.0, 1.0, -1.0), device=env.device
  )
  upper_sign = torch.tensor(
    (1.0, -1.0, -1.0, 1.0, -1.0, 1.0, -1.0), device=env.device
  )
  lower = (action[:, :6] - action[:, 6:12] * lower_sign).square().mean(dim=-1)
  upper = (action[:, 15:22] - action[:, 22:29] * upper_sign).square().mean(dim=-1)
  return (0.5 * lower + 0.5 * upper) * regularization_progress(env)


def numerical_instability(env: ManagerBasedRlEnv) -> torch.Tensor:
  robot: Entity = env.scene["robot"]
  action = _getup_action(env)
  finite = (
    torch.isfinite(robot.data.root_link_pose_w).all(dim=-1)
    & torch.isfinite(robot.data.root_link_vel_w).all(dim=-1)
    & torch.isfinite(robot.data.joint_pos[:, action._body_ids]).all(dim=-1)
    & torch.isfinite(robot.data.joint_vel[:, action._body_ids]).all(dim=-1)
  )
  height = robot.data.root_link_pos_w[:, 2]
  return (~finite) | (height < 0.08) | (height > 1.50)


def make_g1_getup_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  observations = {
    "actor": ObservationGroupCfg(
      terms={"getup": ObservationTermCfg(
        func=G1GetupObservation,
        delay_min_lag=0,
        delay_max_lag=0 if play else 1,
        clip=(-20.0, 20.0),
      )},
      concatenate_terms=True,
      enable_corruption=False,
    ),
    "critic": ObservationGroupCfg(
      terms={"getup": ObservationTermCfg(func=G1GetupObservation)},
      concatenate_terms=True,
      enable_corruption=False,
    ),
  }
  events = {
    "reset_scene": EventTermCfg(func=mdp.reset_scene_to_default, mode="reset"),
    "reset_getup_state": EventTermCfg(
      func=reset_getup_state,
      mode="reset",
      params={"curriculum": not play},
    ),
  }
  if not play:
    events.update({
      "actuator_strength": EventTermCfg(
        func=base.randomize_joint_actuator_strength,
        mode="reset",
        params={"scale_range": (0.85, 1.15)},
      ),
      "robot_friction": EventTermCfg(
        func=dr.geom_friction,
        mode="reset",
        params={
          "asset_cfg": SceneEntityCfg("robot", geom_names=(".*",)),
          "operation": "scale",
          "distribution": "uniform",
          "axes": [0, 1, 2],
          "ranges": (0.75, 1.25),
        },
      ),
      "push_robot": EventTermCfg(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(2.0, 4.0),
        params={"velocity_range": {
          "x": (-0.25, 0.25), "y": (-0.25, 0.25), "z": (-0.15, 0.15),
          "roll": (-0.25, 0.25), "pitch": (-0.25, 0.25), "yaw": (-0.35, 0.35),
        }},
      ),
    })
  return ManagerBasedRlEnvCfg(
    scene=SceneCfg(
      terrain=TerrainEntityCfg(terrain_type="plane"),
      entities={"robot": _robot_cfg()},
      sensors=_contact_sensors(),
      num_envs=256 if play else 4096,
      env_spacing=2.5,
    ),
    observations=observations,
    actions={"getup": G1GetupActionCfg(
      entity_name="robot",
      assistance_force_n=0.0 if play else 120.0,
    )},
    commands={},
    events=events,
    rewards={
      "root_height": RewardTermCfg(func=root_height_reward, weight=2.0),
      "torso_height": RewardTermCfg(func=torso_height_reward, weight=2.0),
      "root_height_progress": RewardTermCfg(
        func=HeightProgress, weight=18.0, params={"body_name": "root"}
      ),
      "torso_height_progress": RewardTermCfg(
        func=HeightProgress, weight=12.0, params={"body_name": "torso_link"}
      ),
      "upright": RewardTermCfg(func=upright_reward, weight=3.0),
      "foot_support": RewardTermCfg(func=foot_support_reward, weight=2.0),
      "standing_pose": RewardTermCfg(func=standing_pose_reward, weight=2.0),
      "stable_standing": RewardTermCfg(func=stable_bonus, weight=12.0),
      "action_rate": RewardTermCfg(func=mdp.action_rate_l2, weight=-0.04),
      "joint_velocity": RewardTermCfg(func=joint_velocity_cost, weight=-0.015),
      "actuator_effort": RewardTermCfg(func=actuator_effort_cost, weight=-0.10),
      "impact": RewardTermCfg(func=impact_cost, weight=-0.20),
      "symmetry": RewardTermCfg(func=symmetry_cost, weight=-0.015),
      "joint_limits": RewardTermCfg(
        func=mdp.joint_pos_limits,
        weight=-0.25,
        params={"asset_cfg": SceneEntityCfg(
          "robot", joint_names=BODY_JOINT_NAMES
        )},
      ),
    },
    terminations={
      "time_out": TerminationTermCfg(func=mdp.time_out, time_out=True),
      "stable_standing": TerminationTermCfg(func=StableStandingTermination),
      "numerical_instability": TerminationTermCfg(func=numerical_instability),
    },
    curriculum={},
    viewer=ViewerConfig(
      origin_type=ViewerConfig.OriginType.ASSET_BODY,
      entity_name="robot",
      body_name="torso_link",
      distance=3.2,
      elevation=-12.0,
      azimuth=135.0,
    ),
    sim=SimulationCfg(
      nconmax=512,
      njmax=4000,
      contact_sensor_maxmatch=1024,
      mujoco=MujocoCfg(
        timestep=PHYSICS_STEP_SECONDS,
        iterations=10,
        ls_iterations=20,
        impratio=10,
        cone="elliptic",
      ),
    ),
    decimation=round(CONTROL_STEP_SECONDS / PHYSICS_STEP_SECONDS),
    episode_length_s=EPISODE_SECONDS,
    is_finite_horizon=True,
  )


def g1_getup_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
  return RslRlOnPolicyRunnerCfg(
    actor=RslRlModelCfg(
      hidden_dims=(512, 512, 256),
      activation="elu",
      obs_normalization=True,
      distribution_cfg={
        "class_name": "BetaDistribution",
        "action_range": (-1.0, 1.0),
      },
    ),
    critic=RslRlModelCfg(
      hidden_dims=(512, 512, 256), activation="elu", obs_normalization=True
    ),
    algorithm=RslRlPpoAlgorithmCfg(
      value_loss_coef=1.0,
      use_clipped_value_loss=True,
      clip_param=0.2,
      entropy_coef=0.002,
      num_learning_epochs=5,
      num_mini_batches=8,
      learning_rate=3.0e-4,
      schedule="adaptive",
      gamma=0.995,
      lam=0.95,
      desired_kl=0.01,
      max_grad_norm=1.0,
    ),
    experiment_name="hear_g1_getup",
    save_interval=100,
    num_steps_per_env=32,
    max_iterations=5_000,
  )


def register_g1_getup_task() -> None:
  if TASK_ID in list_tasks():
    return
  register_mjlab_task(
    task_id=TASK_ID,
    env_cfg=make_g1_getup_env_cfg(play=False),
    play_env_cfg=make_g1_getup_env_cfg(play=True),
    rl_cfg=g1_getup_ppo_runner_cfg(),
  )


register_g1_getup_task()
