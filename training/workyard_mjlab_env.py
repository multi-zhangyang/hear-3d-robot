"""mjlab environment for HEAR's contact-rich Workyard policy.

The policy trained here is deliberately below the Agent Harness boundary.  It
receives an already-authorized task-space command and produces only whole-body
joint residuals plus hand-coordination deltas.  Goal, Skill, object, hand,
strategy, recovery, execution authorization, and acceptance remain HEAR Harness
responsibilities.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Final, Literal

import mujoco
import torch
import torch.nn.functional as torch_f

from mjlab.actuator.xml_actuator import XmlActuatorCfg
from mjlab.entity import Entity, EntityArticulationInfoCfg, EntityCfg
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp
from mjlab.envs.mdp import dr
from mjlab.managers.action_manager import ActionTerm, ActionTermCfg
from mjlab.managers.command_manager import CommandTerm, CommandTermCfg
from mjlab.managers.curriculum_manager import CurriculumTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.event_manager import requires_model_fields
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.rl import RslRlModelCfg, RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg
from mjlab.scene import SceneCfg
from mjlab.sensor import ContactMatch, ContactSensor, ContactSensorCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.tasks.registry import list_tasks, register_mjlab_task
from mjlab.terrains import TerrainEntityCfg
from mjlab.utils.lab_api.math import (
  quat_apply,
  quat_apply_inverse,
  quat_conjugate,
  quat_from_euler_xyz,
  quat_mul,
)
from mjlab.viewer import ViewerConfig

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


TASK_ID: Final = "Hear-Workyard-Skill-Conditioned-G1-v2"
OBSERVATION_SIZE: Final = 221
ACTION_SIZE: Final = 37
BODY_ACTION_SIZE: Final = 29
HAND_ACTION_SIZE: Final = 8

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
G1_MODEL_PATH = REPOSITORY_ROOT / "assets" / "humanoid" / "g1" / "g1_with_hands.xml"

BODY_JOINT_NAMES: Final = (
  "left_hip_pitch_joint",
  "left_hip_roll_joint",
  "left_hip_yaw_joint",
  "left_knee_joint",
  "left_ankle_pitch_joint",
  "left_ankle_roll_joint",
  "right_hip_pitch_joint",
  "right_hip_roll_joint",
  "right_hip_yaw_joint",
  "right_knee_joint",
  "right_ankle_pitch_joint",
  "right_ankle_roll_joint",
  "waist_yaw_joint",
  "waist_roll_joint",
  "waist_pitch_joint",
  "left_shoulder_pitch_joint",
  "left_shoulder_roll_joint",
  "left_shoulder_yaw_joint",
  "left_elbow_joint",
  "left_wrist_roll_joint",
  "left_wrist_pitch_joint",
  "left_wrist_yaw_joint",
  "right_shoulder_pitch_joint",
  "right_shoulder_roll_joint",
  "right_shoulder_yaw_joint",
  "right_elbow_joint",
  "right_wrist_roll_joint",
  "right_wrist_pitch_joint",
  "right_wrist_yaw_joint",
)

HAND_JOINT_NAMES: Final = (
  "left_hand_thumb_0_joint",
  "left_hand_thumb_1_joint",
  "left_hand_thumb_2_joint",
  "left_hand_middle_0_joint",
  "left_hand_middle_1_joint",
  "left_hand_index_0_joint",
  "left_hand_index_1_joint",
  "right_hand_thumb_0_joint",
  "right_hand_thumb_1_joint",
  "right_hand_thumb_2_joint",
  "right_hand_middle_0_joint",
  "right_hand_middle_1_joint",
  "right_hand_index_0_joint",
  "right_hand_index_1_joint",
)

HAND_JOINT_LIMITS: Final = (
  (-1.0472, 1.0472),
  (-0.724312, 1.0472),
  (0.0, 1.74533),
  (-1.5708, 0.0),
  (-1.74533, 0.0),
  (-1.5708, 0.0),
  (-1.74533, 0.0),
  (-1.0472, 1.0472),
  (-1.0472, 0.724312),
  (-1.74533, 0.0),
  (0.0, 1.5708),
  (0.0, 1.74533),
  (0.0, 1.5708),
  (0.0, 1.74533),
)

# This is the same 29-joint neutral used by HEAR's production reference
# controller.  The learned action is a residual around this reference.
BODY_DEFAULT_POSITIONS: Final = (
  -0.312, 0.0, 0.0, 0.669, -0.363, 0.0,
  -0.312, 0.0, 0.0, 0.669, -0.363, 0.0,
  0.0, 0.0, 0.0,
  0.2, 0.2, 0.0, 0.6, 0.0, 0.0, 0.0,
  0.2, -0.2, 0.0, 0.6, 0.0, 0.0, 0.0,
)

TEACHER_STAGES: Final = ("reach", "contact", "grasp", "lift", "carry", "place")
POLICY_CAPABILITIES: Final = (
  "balance",
  "locomotion",
  "joint_reference_tracking",
  "contact_rich_manipulation",
  "bimanual_manipulation",
)
STAGE_MINIMUM_EPISODES: Final = (20_000, 20_000, 30_000, 30_000, 40_000, 50_000)
STAGE_PROMOTION_RATES: Final = (0.90, 0.85, 0.80, 0.80, 0.75, 0.70)
STAGE_REPLAY_FRACTIONS: Final = (0.0, 0.20, 0.25, 0.25, 0.30, 0.35)

SOURCE_POSITION: Final = (0.80, 0.0, 0.555)
ROD_START_POSITION: Final = (0.80, 0.0, 0.675)
TARGET_POSITION: Final = (2.40, 0.0, 0.015)
TARGET_HALF_EXTENT: Final = 0.70
PREGRASP_SHELL_RADIUS_M: Final = 0.10
PREGRASP_LATERAL_CLEARANCE_M: Final = 0.10
PREGRASP_TARGET_PROTOCOL: Final = "shoulder-ray-side-clearance-pregrasp-v1"
ACTIVE_HAND_ALLOCATION_PROTOCOL: Final = "nearest_lateral_hand_centerline_balanced-v1"


def _load_g1_spec() -> mujoco.MjSpec:
  if not G1_MODEL_PATH.is_file():
    raise FileNotFoundError(f"HEAR G1 model is missing: {G1_MODEL_PATH}")
  spec = mujoco.MjSpec.from_file(str(G1_MODEL_PATH))
  # MjSpec.attach() recompiles the child model as part of the scene.  MuJoCo
  # otherwise keeps the MJCF's relative compiler directory ("assets" in the
  # upstream Menagerie file) and resolves it from the process working
  # directory, even though HEAR vendors the meshes in the adjacent `meshes`
  # directory.  Pin the compiler directory to the loaded model so every
  # caller, including derived contact plants, remains self-contained.
  spec.compiler.meshdir = str(G1_MODEL_PATH.parent / "meshes")
  return spec


def _make_box_spec(
  name: str,
  half_size: tuple[float, float, float],
  rgba: tuple[float, float, float, float],
  collision: bool = True,
) -> mujoco.MjSpec:
  spec = mujoco.MjSpec()
  body = spec.worldbody.add_body(name=name)
  body.add_geom(
    name=f"{name}_geom",
    type=mujoco.mjtGeom.mjGEOM_BOX,
    size=half_size,
    rgba=rgba,
    contype=1 if collision else 0,
    conaffinity=1 if collision else 0,
    friction=(0.8, 0.01, 0.001),
  )
  return spec


def _make_rod_spec() -> mujoco.MjSpec:
  spec = mujoco.MjSpec()
  body = spec.worldbody.add_body(name="assembly_rod")
  body.add_freejoint(name="assembly_rod_free_joint")
  body.add_geom(
    name="assembly_rod_geom",
    type=mujoco.mjtGeom.mjGEOM_CYLINDER,
    size=(0.03, 0.11, 0.0),
    mass=0.35,
    rgba=(0.82, 0.54, 0.27, 1.0),
    friction=(0.8, 0.012, 0.002),
    condim=4,
  )
  return spec


def _robot_cfg() -> EntityCfg:
  joint_pos = dict(zip(BODY_JOINT_NAMES, BODY_DEFAULT_POSITIONS, strict=True))
  joint_pos.update({name: 0.0 for name in HAND_JOINT_NAMES})
  return EntityCfg(
    spec_fn=_load_g1_spec,
    init_state=EntityCfg.InitialStateCfg(
      pos=(0.0, 0.0, 0.79),
      joint_pos=joint_pos,
      joint_vel={".*": 0.0},
    ),
    articulation=EntityArticulationInfoCfg(
      actuators=(
        XmlActuatorCfg(
          target_names_expr=BODY_JOINT_NAMES + HAND_JOINT_NAMES,
          command_field="position",
        ),
      ),
      soft_joint_pos_limit_factor=0.92,
    ),
  )


def _source_cfg() -> EntityCfg:
  return EntityCfg(
    spec_fn=lambda: _make_box_spec(
      "pickup_stand", (0.06, 0.06, 0.005), (0.46, 0.53, 0.49, 1.0)
    ),
    init_state=EntityCfg.InitialStateCfg(pos=SOURCE_POSITION),
  )


def _rod_cfg() -> EntityCfg:
  return EntityCfg(
    spec_fn=_make_rod_spec,
    init_state=EntityCfg.InitialStateCfg(pos=ROD_START_POSITION),
  )


def _target_cfg() -> EntityCfg:
  return EntityCfg(
    spec_fn=lambda: _make_box_spec(
      "assembly_bay", (TARGET_HALF_EXTENT, TARGET_HALF_EXTENT, 0.005),
      (0.32, 0.73, 0.58, 0.28), collision=False,
    ),
    init_state=EntityCfg.InitialStateCfg(pos=TARGET_POSITION),
  )


@dataclass(kw_only=True)
class WorkyardActionCfg(ActionTermCfg):
  body_scale: float = 0.35
  hand_scale: float = 0.08

  def build(self, env: ManagerBasedRlEnv) -> "WorkyardAction":
    return WorkyardAction(self, env)


class WorkyardAction(ActionTerm):
  """37D HEAR action: 29 body residuals and eight hand-synergy deltas."""

  cfg: WorkyardActionCfg

  def __init__(self, cfg: WorkyardActionCfg, env: ManagerBasedRlEnv):
    super().__init__(cfg, env)
    body_ids, body_names = self._entity.find_joints(BODY_JOINT_NAMES, preserve_order=True)
    hand_ids, hand_names = self._entity.find_joints(HAND_JOINT_NAMES, preserve_order=True)
    if tuple(body_names) != BODY_JOINT_NAMES or tuple(hand_names) != HAND_JOINT_NAMES:
      raise ValueError("HEAR Workyard joint order does not match the training contract")
    self._body_ids = torch.tensor(body_ids, dtype=torch.long, device=self.device)
    self._hand_ids = torch.tensor(hand_ids, dtype=torch.long, device=self.device)
    self._raw_action = torch.zeros((self.num_envs, ACTION_SIZE), device=self.device)
    self.coordination = torch.zeros((self.num_envs, HAND_ACTION_SIZE), device=self.device)
    self._body_targets = torch.zeros((self.num_envs, BODY_ACTION_SIZE), device=self.device)
    self._hand_targets = torch.zeros((self.num_envs, len(HAND_JOINT_NAMES)), device=self.device)
    self._body_default = torch.tensor(
      BODY_DEFAULT_POSITIONS, dtype=torch.float32, device=self.device
    ).unsqueeze(0)
    self._hand_endpoint = torch.tensor(
      (
        HAND_JOINT_LIMITS[0][0],
        HAND_JOINT_LIMITS[1][1],
        HAND_JOINT_LIMITS[2][1],
        HAND_JOINT_LIMITS[3][0],
        HAND_JOINT_LIMITS[4][0],
        HAND_JOINT_LIMITS[5][0],
        HAND_JOINT_LIMITS[6][0],
        HAND_JOINT_LIMITS[7][0],
        HAND_JOINT_LIMITS[8][0],
        HAND_JOINT_LIMITS[9][0],
        HAND_JOINT_LIMITS[10][1],
        HAND_JOINT_LIMITS[11][1],
        HAND_JOINT_LIMITS[12][1],
        HAND_JOINT_LIMITS[13][1],
      ),
      dtype=torch.float32,
      device=self.device,
    ).unsqueeze(0)
    # Joint -> [left opposition, left thumb, left index, left middle,
    #           right opposition, right thumb, right index, right middle].
    self._hand_synergy_indexes = torch.tensor(
      (0, 1, 1, 3, 3, 2, 2, 4, 5, 5, 7, 7, 6, 6),
      dtype=torch.long,
      device=self.device,
    )

  @property
  def action_dim(self) -> int:
    return ACTION_SIZE

  @property
  def raw_action(self) -> torch.Tensor:
    return self._raw_action

  @property
  def body_targets(self) -> torch.Tensor:
    return self._body_targets

  @property
  def hand_targets(self) -> torch.Tensor:
    return self._hand_targets

  def process_actions(self, actions: torch.Tensor) -> None:
    if actions.shape != (self.num_envs, ACTION_SIZE):
      raise ValueError(f"Expected Workyard action [B, {ACTION_SIZE}], got {actions.shape}")
    self._raw_action[:] = actions.clamp(-1.0, 1.0)
    self._body_targets[:] = (
      self._body_default + self._raw_action[:, :BODY_ACTION_SIZE] * self.cfg.body_scale
    )
    self.coordination[:] = torch.clamp(
      self.coordination + self._raw_action[:, BODY_ACTION_SIZE:] * self.cfg.hand_scale,
      0.0,
      1.0,
    )
    self._hand_targets[:] = (
      self.coordination[:, self._hand_synergy_indexes] * self._hand_endpoint
    )

  def apply_actions(self) -> None:
    self._entity.set_joint_position_target(self._body_targets, joint_ids=self._body_ids)
    self._entity.set_joint_position_target(self._hand_targets, joint_ids=self._hand_ids)

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    if env_ids is None:
      env_ids = slice(None)
    self._raw_action[env_ids] = 0.0
    self.coordination[env_ids] = 0.0
    self._body_targets[env_ids] = self._body_default
    self._hand_targets[env_ids] = 0.0


@dataclass(kw_only=True)
class WorkyardCommandCfg(CommandTermCfg):
  object_position_jitter_m: float = 0.08
  target_position_jitter_m: float = 0.15
  pregrasp_shell_radius_m: float = PREGRASP_SHELL_RADIUS_M
  pregrasp_lateral_clearance_m: float = PREGRASP_LATERAL_CLEARANCE_M
  wrist_frame_safe_pregrasp: bool = False
  wrist_frame_pregrasp_forward_m: float = 0.18
  wrist_frame_pregrasp_lateral_m: float = 0.20
  contact_pocket_forward_m: float = 0.12
  contact_pocket_lateral_m: float = 0.04
  contact_pocket_vertical_m: float = 0.0
  contact_alignment_radius_m: float = 0.0
  contact_alignment_tolerance_m: float = 0.035
  contact_retreat_tolerance_m: float = 0.010
  contact_alignment_bearing_tolerance_rad: float = math.pi
  contact_alignment_axis_tolerance_rad: float = math.pi
  contact_alignment_max_force_n: float = 1.0e9
  contact_preshape_ready_coordination: float = 0.70
  contact_closure_tolerance_m: float = 0.015
  contact_closure_vertical_tolerance_m: float = 0.040

  def build(self, env: ManagerBasedRlEnv) -> "WorkyardCommand":
    return WorkyardCommand(self, env)


class WorkyardCommand(CommandTerm):
  """Teacher command generator for the deployed Skill-Call-conditioned student.

  ``teacher_stage`` and ``teacher_target_stage`` exist only to generate diverse
  commands, rewards, and labels.  They are deliberately absent from ``command``
  and from the actor observation.  The deployed student receives the same
  capability/window/task-space fields supplied by HumanoidEmbodiedSkillCall v2.
  """

  cfg: WorkyardCommandCfg

  def __init__(self, cfg: WorkyardCommandCfg, env: ManagerBasedRlEnv):
    super().__init__(cfg, env)
    self.robot: Entity = env.scene["robot"]
    self.rod: Entity = env.scene["assembly_rod"]
    self.target: Entity = env.scene["assembly_bay"]
    self.teacher_stage = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
    self.teacher_target_stage = torch.zeros_like(self.teacher_stage)
    self.active_hand = torch.zeros_like(self.teacher_stage)
    self.completed_target = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
    self.contact_pocket_active = torch.zeros_like(self.completed_target)
    self.contact_alignment_active = torch.zeros_like(self.completed_target)
    self.contact_alignment_completed = torch.zeros_like(self.completed_target)
    self.contact_retreat_active = torch.zeros_like(self.completed_target)
    self.contact_retreat_completed = torch.zeros_like(self.completed_target)
    self.max_stage = 0
    self.episode_counts = torch.zeros(
      len(TEACHER_STAGES), dtype=torch.long, device=self.device
    )
    self.success_counts = torch.zeros_like(self.episode_counts)
    self.desired_base_twist = torch.zeros((self.num_envs, 3), device=self.device)
    self.requested_capabilities = torch.zeros(
      (self.num_envs, len(POLICY_CAPABILITIES)), device=self.device
    )
    self.skill_window_progress = torch.zeros((self.num_envs, 1), device=self.device)
    self.wrist_targets_pelvis = torch.zeros((self.num_envs, 14), device=self.device)
    self.wrist_tolerances = torch.full((self.num_envs, 2), 0.06, device=self.device)
    self.grasp_requirements = torch.zeros((self.num_envs, 4), device=self.device)
    self.target_position = torch.zeros((self.num_envs, 3), device=self.device)
    self.initial_rod_position = torch.zeros(
      (self.num_envs, 3), device=self.device
    )
    self.metrics["target_stage_success"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["current_stage"] = torch.zeros(self.num_envs, device=self.device)

    body_ids, names = self.robot.find_bodies(
      ("left_wrist_yaw_link", "right_wrist_yaw_link"), preserve_order=True
    )
    if tuple(names) != ("left_wrist_yaw_link", "right_wrist_yaw_link"):
      raise ValueError("HEAR Workyard wrist bodies are missing from the G1 model")
    self._wrist_body_ids = torch.tensor(body_ids, dtype=torch.long, device=self.device)
    shoulder_ids, shoulder_names = self.robot.find_bodies(
      ("left_shoulder_pitch_link", "right_shoulder_pitch_link"),
      preserve_order=True,
    )
    if tuple(shoulder_names) != (
      "left_shoulder_pitch_link", "right_shoulder_pitch_link"
    ):
      raise ValueError("HEAR Workyard shoulder anchors are missing from the G1 model")
    self._shoulder_body_ids = torch.tensor(
      shoulder_ids, dtype=torch.long, device=self.device
    )

  @property
  def command(self) -> torch.Tensor:
    return torch.cat(
      (
        self.requested_capabilities,
        self.skill_window_progress,
        torch_f.one_hot(self.active_hand, num_classes=2).float(),
        self.desired_base_twist,
        self.wrist_targets_pelvis,
        self.wrist_tolerances,
        self.grasp_requirements,
      ),
      dim=-1,
    )

  def _update_metrics(self) -> None:
    self.metrics["target_stage_success"][:] = self.completed_target.float()
    self.metrics["current_stage"][:] = self.teacher_stage.float()

  def _resample_command(self, env_ids: torch.Tensor) -> None:
    count = len(env_ids)
    self.teacher_stage[env_ids] = 0
    self.completed_target[env_ids] = False
    self.contact_pocket_active[env_ids] = False
    self.contact_alignment_active[env_ids] = False
    self.contact_alignment_completed[env_ids] = False
    self.contact_retreat_active[env_ids] = False
    self.contact_retreat_completed[env_ids] = False

    if self.max_stage == 0:
      self.teacher_target_stage[env_ids] = 0
    else:
      replay_fraction = STAGE_REPLAY_FRACTIONS[self.max_stage]
      replay = torch.rand(count, device=self.device) < replay_fraction
      replay_stage = torch.randint(0, self.max_stage, (count,), device=self.device)
      self.teacher_target_stage[env_ids] = torch.where(
        replay, replay_stage, torch.full_like(replay_stage, self.max_stage)
      )

    origins = self._env.scene.env_origins[env_ids]
    object_jitter = torch.zeros((count, 3), device=self.device)
    object_jitter[:, :2].uniform_(
      -self.cfg.object_position_jitter_m, self.cfg.object_position_jitter_m
    )
    # Phase-one reach owns neither the waist nor cross-body re-positioning.
    # Route each object to the laterally nearest hand instead of asking an arm
    # to intersect the torso.  Continuous jitter balances both hands; an exact
    # centerline sample is assigned randomly.
    centerline = object_jitter[:, 1].abs() <= 1e-6
    nearest_hand = (object_jitter[:, 1] < 0.0).long()
    centerline_hand = torch.randint(0, 2, (count,), device=self.device)
    self.active_hand[env_ids] = torch.where(
      centerline, centerline_hand, nearest_hand
    )
    rod_pos = torch.tensor(ROD_START_POSITION, device=self.device).expand(count, 3)
    rod_pos = rod_pos + origins + object_jitter
    # Entity data is synchronized only after the simulator advances.  Keep the
    # commanded reset pose directly so downstream object-loss checks never
    # compare against stale pre-reset buffers.
    self.initial_rod_position[env_ids] = rod_pos
    yaw = torch.empty(count, device=self.device).uniform_(-0.25, 0.25)
    rod_quat = quat_from_euler_xyz(
      torch.zeros_like(yaw), torch.zeros_like(yaw), yaw
    )
    self.rod.data.write_root_pose(torch.cat((rod_pos, rod_quat), dim=-1), env_ids)
    self.rod.data.write_root_velocity(torch.zeros((count, 6), device=self.device), env_ids)

    target_jitter = torch.zeros((count, 3), device=self.device)
    target_jitter[:, :2].uniform_(
      -self.cfg.target_position_jitter_m, self.cfg.target_position_jitter_m
    )
    target_pos = torch.tensor(TARGET_POSITION, device=self.device).expand(count, 3)
    target_pos = target_pos + origins + target_jitter
    self.target_position[env_ids] = target_pos
    target_pose = torch.cat(
      (
        target_pos,
        torch.tensor((1.0, 0.0, 0.0, 0.0), device=self.device).expand(count, 4),
      ),
      dim=-1,
    )
    self.target.data.write_mocap_pose(target_pose, env_ids)
    self.desired_base_twist[env_ids] = 0.0
    self.requested_capabilities[env_ids] = 0.0
    self.skill_window_progress[env_ids] = 0.0
    self.grasp_requirements[env_ids] = torch.tensor(
      (4.0, 2.0, 0.15, 0.0), device=self.device
    )

  def _update_command(self) -> None:
    self._update_task_space_targets()
    self._advance_teacher_curriculum()

  def _advance_teacher_curriculum(self) -> None:
    wrist_pos = self.robot.data.body_link_pos_w[:, self._wrist_body_ids]
    rod_pos = self.rod.data.root_link_pos_w
    active_wrist = wrist_pos[
      torch.arange(self.num_envs, device=self.device), self.active_hand
    ]
    root_pos = self.robot.data.root_link_pos_w
    root_quat = self.robot.data.root_link_quat_w
    wrist_targets_p = self.wrist_targets_pelvis.reshape(self.num_envs, 2, 7)[..., :3]
    wrist_targets_w = root_pos.unsqueeze(1) + quat_apply(
      root_quat.unsqueeze(1).expand(-1, 2, -1), wrist_targets_p
    )
    rows = torch.arange(self.num_envs, device=self.device)
    active_target = wrist_targets_w[rows, self.active_hand]
    active_tolerance = self.wrist_tolerances[
      rows, self.active_hand
    ].clamp_min(0.025)
    reach = torch.linalg.vector_norm(
      active_wrist - active_target, dim=-1
    ) <= active_tolerance
    hand_found, hand_force, hand_surfaces, opposing = _hand_contact_summary(self._env)
    active_found = hand_found[rows, self.active_hand]
    active_force = hand_force[rows, self.active_hand]
    active_surfaces = hand_surfaces[rows, self.active_hand]
    active_opposing = opposing[rows, self.active_hand]
    lifted = rod_pos[:, 2] > SOURCE_POSITION[2] + 0.16
    horizontal_error = torch.linalg.vector_norm(
      rod_pos[:, :2] - self.target_position[:, :2], dim=-1
    )
    carried = horizontal_error < 0.22
    inside = _object_inside_zone(self._env)
    settled = torch.linalg.vector_norm(self.rod.data.root_link_vel_w, dim=-1) < 0.10
    action = _workyard_action(self._env)
    active_coord = action.coordination[rows, self.active_hand * 4 + 1]
    released = (~active_found) & (active_coord < 0.20)

    transitions = (
      reach,
      self.contact_pocket_active & reach & active_found & (active_force >= 2.0),
      active_opposing & (active_surfaces >= 2.0) & (active_force >= 4.0),
      lifted,
      carried,
      inside & settled & released,
    )
    # A typed stage transition must become observable for at least one control
    # update before another transition can occur.  Using the mutating stage in
    # this loop allowed reach -> contact -> grasp cascades against conditions
    # computed from the old target pose.
    stage_at_update = self.teacher_stage.clone()
    for stage_index, condition in enumerate(transitions):
      at_stage = stage_at_update == stage_index
      if stage_index < len(TEACHER_STAGES) - 1:
        self.teacher_stage[at_stage & condition] = stage_index + 1
      else:
        self.completed_target |= at_stage & condition
    self.completed_target |= self.teacher_stage > self.teacher_target_stage

  def _update_task_space_targets(self) -> None:
    root_pos = self.robot.data.root_link_pos_w
    root_quat = self.robot.data.root_link_quat_w
    rod_pos = self.rod.data.root_link_pos_w
    rod_quat = self.rod.data.root_link_quat_w
    wrist_pose = self.robot.data.body_link_pose_w[:, self._wrist_body_ids]

    relative_target = quat_apply_inverse(root_quat, self.target_position - root_pos)
    horizontal = relative_target[:, :2]
    distance = torch.linalg.vector_norm(horizontal, dim=-1, keepdim=True).clamp_min(1e-6)
    direction = horizontal / distance
    travel_stage = self.teacher_stage >= TEACHER_STAGES.index("carry")
    self.desired_base_twist[:] = 0.0
    self.desired_base_twist[:, :2] = torch.where(
      travel_stage.unsqueeze(-1), direction * torch.clamp(distance, max=0.65), 0.0
    )
    self.desired_base_twist[:, 2] = torch.where(
      travel_stage, torch.atan2(horizontal[:, 1], horizontal[:, 0]).clamp(-0.7, 0.7), 0.0
    )

    # Reach is a hand-locked pre-grasp phase, not contact.  Retain the safe
    # shoulder-ray x/z offset, but keep each wrist at least 10 cm on its own side
    # of the rod.  A pure shoulder ray drove shoulder-yaw links through the
    # torso; collision-aware offline fitting and the nearest-hand router define
    # this phase's physically admissible command distribution.  Exact side
    # targets activate only after contact authority exists.
    shoulder_pos = self.robot.data.body_link_pos_w[:, self._shoulder_body_ids]
    shoulder_ray = shoulder_pos - rod_pos.unsqueeze(1)
    shoulder_ray /= torch.linalg.vector_norm(
      shoulder_ray, dim=-1, keepdim=True
    ).clamp_min(1e-6)
    pregrasp_offset = self.cfg.pregrasp_shell_radius_m * shoulder_ray
    lateral_side = torch.tensor((1.0, -1.0), device=self.device)
    pregrasp_offset[:, :, 1] = lateral_side.unsqueeze(0) * torch.maximum(
      pregrasp_offset[:, :, 1].abs(),
      torch.full_like(
        pregrasp_offset[:, :, 1], self.cfg.pregrasp_lateral_clearance_m
      ),
    )
    pregrasp = rod_pos.unsqueeze(1) + pregrasp_offset
    if self.cfg.wrist_frame_safe_pregrasp:
      wrist_frame_pregrasp = torch.tensor(
        (
          (
            self.cfg.wrist_frame_pregrasp_forward_m,
            -self.cfg.wrist_frame_pregrasp_lateral_m,
            0.0,
          ),
          (
            self.cfg.wrist_frame_pregrasp_forward_m,
            self.cfg.wrist_frame_pregrasp_lateral_m,
            0.0,
          ),
        ),
        device=self.device,
      ).unsqueeze(0).expand(self.num_envs, -1, -1)
      pregrasp = rod_pos.unsqueeze(1) - quat_apply(
        wrist_pose[..., 3:7], wrist_frame_pregrasp
      )
    # The accepted reach teacher controls wrist position (3D), not
    # orientation.  Define the contact pocket in each hand's *measured* wrist
    # frame so the rod stays between thumb and fingers even when the frozen
    # actor retains its natural wrist posture.
    wrist_to_rod_local = torch.tensor(
      (
        (
          self.cfg.contact_pocket_forward_m,
          -self.cfg.contact_pocket_lateral_m,
          self.cfg.contact_pocket_vertical_m,
        ),
        (
          self.cfg.contact_pocket_forward_m,
          self.cfg.contact_pocket_lateral_m,
          self.cfg.contact_pocket_vertical_m,
        ),
      ),
      device=self.device,
    ).unsqueeze(0).expand(self.num_envs, -1, -1)
    wrist_to_rod_world = quat_apply(wrist_pose[..., 3:7], wrist_to_rod_local)
    contact_approach = rod_pos.unsqueeze(1) - wrist_to_rod_world
    contact_stage = self.teacher_stage >= TEACHER_STAGES.index("contact")
    contact_preshape = pregrasp.clone()
    approach = torch.where(
      contact_stage[:, None, None], contact_preshape, pregrasp
    )
    contact_authorized = contact_stage.clone()
    rows = torch.arange(self.num_envs, device=self.device)
    # Derived tasks may require a hand-state gate before the deterministic
    # terminal approach.  A zero threshold means geometry-safe open-hand
    # insertion; positive values retain the legacy preshape behavior.
    if bool(contact_authorized.any().item()):
      action = _workyard_action(self._env)
      coordination = action.coordination.reshape(self.num_envs, 2, 4)
      preshape_ready = (
        coordination[rows, self.active_hand, 2:].amin(dim=-1)
        >= self.cfg.contact_preshape_ready_coordination
      )
      contact_authorized &= preshape_ready
    if self.cfg.contact_alignment_radius_m > 0.0:
      final_planar = wrist_to_rod_local[..., :2]
      final_planar_direction = final_planar / torch.linalg.vector_norm(
        final_planar, dim=-1, keepdim=True
      ).clamp_min(1e-6)
      alignment_local = torch.cat((
        final_planar_direction * self.cfg.contact_alignment_radius_m,
        wrist_to_rod_local[..., 2:3],
      ), dim=-1)
      measured_rod_local = quat_apply_inverse(
        wrist_pose[..., 3:7], rod_pos.unsqueeze(1) - wrist_pose[..., :3]
      )
      measured_planar = measured_rod_local[..., :2]
      measured_planar_radius = torch.linalg.vector_norm(
        measured_planar, dim=-1
      )
      retreat_completed_before = self.contact_retreat_completed.clone()
      self.contact_retreat_completed |= (
        contact_authorized
        & (
          measured_planar_radius[rows, self.active_hand]
          >= self.cfg.contact_alignment_radius_m
            - self.cfg.contact_retreat_tolerance_m
        )
      )
      self.contact_retreat_active[:] = (
        contact_authorized & ~retreat_completed_before
      )
      retreat_planar_direction = measured_planar / measured_planar_radius[
        ..., None
      ].clamp_min(1e-6)
      retreat_local = torch.cat((
        retreat_planar_direction * self.cfg.contact_alignment_radius_m,
        measured_rod_local[..., 2:3],
      ), dim=-1)
      retreat_approach = rod_pos.unsqueeze(1) - quat_apply(
        wrist_pose[..., 3:7], retreat_local
      )
      approach = torch.where(
        self.contact_retreat_active[:, None, None],
        retreat_approach,
        approach,
      )
      measured_bearing = torch.atan2(
        measured_rod_local[..., 1], measured_rod_local[..., 0]
      )
      desired_bearing = torch.atan2(
        wrist_to_rod_local[..., 1], wrist_to_rod_local[..., 0]
      )
      bearing_error = torch.remainder(
        measured_bearing - desired_bearing + math.pi,
        2.0 * math.pi,
      ) - math.pi
      local_z = torch.zeros_like(rod_pos)
      local_z[:, 2] = 1.0
      rod_axis_w = quat_apply(rod_quat, local_z)
      wrist_axis_w = quat_apply(
        wrist_pose[..., 3:7], local_z.unsqueeze(1).expand(-1, 2, -1)
      )
      axis_alignment_error = torch.acos(
        torch.sum(
          wrist_axis_w * rod_axis_w.unsqueeze(1), dim=-1
        ).abs().clamp(0.0, 1.0)
      )
      active_alignment_error = torch.linalg.vector_norm(
        measured_rod_local[rows, self.active_hand]
          - alignment_local[rows, self.active_hand],
        dim=-1,
      )
      alignment_completed_before = self.contact_alignment_completed.clone()
      alignment_authorized = contact_authorized & retreat_completed_before
      hand_found, hand_force, _, _ = _hand_contact_summary(self._env)
      active_alignment_force = hand_force[rows, self.active_hand]
      self.contact_alignment_completed |= (
        alignment_authorized
        & (active_alignment_error <= self.cfg.contact_alignment_tolerance_m)
        & (
          bearing_error[rows, self.active_hand].abs()
          <= self.cfg.contact_alignment_bearing_tolerance_rad
        )
        & (
          axis_alignment_error[rows, self.active_hand]
          <= self.cfg.contact_alignment_axis_tolerance_rad
        )
        & (active_alignment_force < self.cfg.contact_alignment_max_force_n)
      )
      self.contact_alignment_active[:] = (
        alignment_authorized & ~alignment_completed_before
      )
      alignment_approach = rod_pos.unsqueeze(1) - quat_apply(
        wrist_pose[..., 3:7], alignment_local
      )
      approach = torch.where(
        self.contact_alignment_active[:, None, None],
        alignment_approach,
        approach,
      )
      contact_authorized = alignment_authorized & alignment_completed_before
    else:
      self.contact_alignment_active[:] = False
      self.contact_retreat_active[:] = False
    self.contact_pocket_active[:] = contact_authorized
    approach = torch.where(
      contact_authorized[:, None, None], contact_approach, approach
    )
    lift_offset = torch.tensor((0.0, 0.0, 0.08), device=self.device)
    approach = torch.where(
      (self.teacher_stage >= TEACHER_STAGES.index("lift"))[:, None, None],
      approach + lift_offset,
      approach,
    )
    active_mask = torch_f.one_hot(self.active_hand, num_classes=2).bool().unsqueeze(-1)
    target_pos = torch.where(active_mask, approach, wrist_pose[..., :3])
    target_quat = torch.where(
      active_mask.expand(-1, -1, 4),
      rod_quat.unsqueeze(1).expand(-1, 2, -1),
      wrist_pose[..., 3:7],
    )
    pos_pelvis = quat_apply_inverse(
      root_quat.unsqueeze(1).expand(-1, 2, -1), target_pos - root_pos.unsqueeze(1)
    )
    quat_pelvis = quat_mul(
      quat_conjugate(root_quat).unsqueeze(1).expand(-1, 2, -1), target_quat
    )
    self.wrist_targets_pelvis[:] = torch.cat(
      (pos_pelvis, quat_pelvis), dim=-1
    ).reshape(self.num_envs, 14)
    self.grasp_requirements[:, 3] = (
      self.teacher_stage == TEACHER_STAGES.index("place")
    ).float()

    self.requested_capabilities[:] = 0.0
    self.requested_capabilities[:, 0] = 1.0  # balance
    self.requested_capabilities[:, 1] = travel_stage.float()  # locomotion
    self.requested_capabilities[:, 2] = 1.0  # joint/task-space tracking
    self.requested_capabilities[:, 3] = (
      self.teacher_stage >= TEACHER_STAGES.index("contact")
    ).float()
    self.skill_window_progress[:, 0] = (
      self._env.episode_length_buf.float()
      / max(1, self._env.max_episode_length)
    ).clamp(0.0, 1.0)


class WorkyardObservation:
  def __init__(self, cfg: ObservationTermCfg, env: ManagerBasedRlEnv):
    del cfg
    self.robot: Entity = env.scene["robot"]
    self.rod: Entity = env.scene["assembly_rod"]
    body_ids, _ = self.robot.find_joints(BODY_JOINT_NAMES, preserve_order=True)
    hand_ids, _ = self.robot.find_joints(HAND_JOINT_NAMES, preserve_order=True)
    ee_ids, _ = self.robot.find_bodies(
      (
        "left_wrist_yaw_link",
        "right_wrist_yaw_link",
        "left_ankle_roll_link",
        "right_ankle_roll_link",
      ),
      preserve_order=True,
    )
    self.body_ids = torch.tensor(body_ids, dtype=torch.long, device=env.device)
    self.hand_ids = torch.tensor(hand_ids, dtype=torch.long, device=env.device)
    self.ee_ids = torch.tensor(ee_ids, dtype=torch.long, device=env.device)
    self.body_default = torch.tensor(BODY_DEFAULT_POSITIONS, device=env.device)
    self.hand_min = torch.tensor([limit[0] for limit in HAND_JOINT_LIMITS], device=env.device)
    self.hand_span = torch.tensor(
      [limit[1] - limit[0] for limit in HAND_JOINT_LIMITS], device=env.device
    )

  def __call__(self, env: ManagerBasedRlEnv) -> torch.Tensor:
    action = _workyard_action(env)
    command = _workyard_command(env)
    root_pos = self.robot.data.root_link_pos_w
    root_quat = self.robot.data.root_link_quat_w
    body_pos = self.robot.data.joint_pos[:, self.body_ids]
    body_vel = self.robot.data.joint_vel[:, self.body_ids]
    hand_pos = self.robot.data.joint_pos[:, self.hand_ids]
    hand_vel = self.robot.data.joint_vel[:, self.hand_ids]

    ee_pose = self.robot.data.body_link_pose_w[:, self.ee_ids]
    ee_pos_p = quat_apply_inverse(
      root_quat.unsqueeze(1).expand(-1, 4, -1), ee_pose[..., :3] - root_pos.unsqueeze(1)
    )
    ee_quat_p = quat_mul(
      quat_conjugate(root_quat).unsqueeze(1).expand(-1, 4, -1), ee_pose[..., 3:7]
    )
    ee_pose_p = torch.cat((ee_pos_p, ee_quat_p), dim=-1).reshape(env.num_envs, 28)

    foot_found, foot_force, foot_slip = _foot_contact_summary(env)
    support = torch.cat((foot_found, foot_force, foot_slip), dim=-1)
    hand_found, hand_force, hand_surfaces, _ = _hand_contact_summary(env)
    hand_contact = torch.cat((hand_found.float(), hand_force, hand_surfaces), dim=-1)

    rod_pose = self.rod.data.root_link_pose_w
    rod_pos_p = quat_apply_inverse(root_quat, rod_pose[:, :3] - root_pos)
    rod_quat_p = quat_mul(quat_conjugate(root_quat), rod_pose[:, 3:7])
    rod_vel = self.rod.data.root_link_vel_w
    rod_twist_p = torch.cat(
      (
        quat_apply_inverse(root_quat, rod_vel[:, :3]),
        quat_apply_inverse(root_quat, rod_vel[:, 3:6]),
      ),
      dim=-1,
    )
    zone_delta = quat_apply_inverse(root_quat, command.target_position - rod_pose[:, :3])
    zone_distance = torch.linalg.vector_norm(zone_delta[:, :2], dim=-1, keepdim=True)
    zone_inside = _object_inside_zone(env).float().unsqueeze(-1)

    observation = torch.cat(
      (
        body_pos - self.body_default,
        body_vel,
        action.raw_action[:, :BODY_ACTION_SIZE],
        ((hand_pos - self.hand_min) / self.hand_span).clamp(0.0, 1.0),
        hand_vel,
        action.raw_action[:, BODY_ACTION_SIZE:],
        self.robot.data.root_link_lin_vel_b,
        self.robot.data.root_link_ang_vel_b,
        self.robot.data.projected_gravity_b,
        ee_pose_p,
        support,
        hand_contact,
        rod_pos_p,
        rod_quat_p,
        rod_twist_p,
        zone_delta,
        zone_distance,
        zone_inside,
        command.requested_capabilities,
        command.skill_window_progress,
        torch_f.one_hot(command.active_hand, num_classes=2).float(),
        command.desired_base_twist,
        command.wrist_targets_pelvis,
        command.wrist_tolerances,
        command.grasp_requirements,
      ),
      dim=-1,
    )
    if observation.shape != (env.num_envs, OBSERVATION_SIZE):
      raise RuntimeError(
        f"HEAR Workyard observation contract drifted: {observation.shape}"
      )
    return observation


class ProgressReward:
  """Stage-gated progress for reach, lift, and carry."""

  def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
    self._env = env
    self.metric: Literal["reach", "lift", "carry"] = cfg.params["metric"]
    self.stage_index: int = cfg.params["stage_index"]
    self.previous = torch.zeros(env.num_envs, device=env.device)
    self.valid = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

  def _value(self, env: ManagerBasedRlEnv) -> torch.Tensor:
    command = _workyard_command(env)
    if self.metric == "reach":
      wrist_ids = command._wrist_body_ids
      wrists = command.robot.data.body_link_pos_w[:, wrist_ids]
      rows = torch.arange(env.num_envs, device=env.device)
      wrist = wrists[rows, command.active_hand]
      return torch.linalg.vector_norm(wrist - command.rod.data.root_link_pos_w, dim=-1)
    if self.metric == "lift":
      return -command.rod.data.root_link_pos_w[:, 2]
    return torch.linalg.vector_norm(
      command.rod.data.root_link_pos_w[:, :2] - command.target_position[:, :2], dim=-1
    )

  def reset(self, env_ids: torch.Tensor | slice | None) -> None:
    if env_ids is None:
      env_ids = slice(None)
    self.valid[env_ids] = False

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    metric: str,
    stage_index: int,
  ) -> torch.Tensor:
    del metric, stage_index
    current = self._value(env)
    progress = torch.where(
      self.valid,
      (self.previous - current).clamp(-0.1, 0.1),
      torch.zeros_like(current),
    )
    self.previous[:] = current
    self.valid[:] = True
    return progress * (
      _workyard_command(env).teacher_stage == self.stage_index
    )


def _teacher_stage_mask(env: ManagerBasedRlEnv, stage: str) -> torch.Tensor:
  return (
    _workyard_command(env).teacher_stage == TEACHER_STAGES.index(stage)
  ).float()


def wrist_orientation_alignment(env: ManagerBasedRlEnv) -> torch.Tensor:
  command = _workyard_command(env)
  wrists = command.robot.data.body_link_quat_w[:, command._wrist_body_ids]
  rows = torch.arange(env.num_envs, device=env.device)
  wrist = wrists[rows, command.active_hand]
  rod = command.rod.data.root_link_quat_w
  alignment = torch.abs(torch.sum(wrist * rod, dim=-1)).square()
  return alignment * _teacher_stage_mask(env, "reach")


def qualified_hand_contact(env: ManagerBasedRlEnv) -> torch.Tensor:
  command = _workyard_command(env)
  found, force, surfaces, _ = _hand_contact_summary(env)
  rows = torch.arange(env.num_envs, device=env.device)
  active = command.active_hand
  qualified = found[rows, active] & (force[rows, active] >= 2.0) & (surfaces[rows, active] >= 1)
  return qualified.float() * _teacher_stage_mask(env, "contact")


def opposing_contact(env: ManagerBasedRlEnv) -> torch.Tensor:
  command = _workyard_command(env)
  _, _, _, opposed = _hand_contact_summary(env)
  rows = torch.arange(env.num_envs, device=env.device)
  return opposed[rows, command.active_hand].float() * _teacher_stage_mask(env, "contact")


def grasp_verified(env: ManagerBasedRlEnv) -> torch.Tensor:
  command = _workyard_command(env)
  _, force, surfaces, opposed = _hand_contact_summary(env)
  rows = torch.arange(env.num_envs, device=env.device)
  active = command.active_hand
  verified = opposed[rows, active] & (surfaces[rows, active] >= 2) & (force[rows, active] >= 4)
  return verified.float() * _teacher_stage_mask(env, "grasp")


def object_inside_zone(env: ManagerBasedRlEnv) -> torch.Tensor:
  return _object_inside_zone(env).float() * _teacher_stage_mask(env, "place")


def released_and_settled(env: ManagerBasedRlEnv) -> torch.Tensor:
  command = _workyard_command(env)
  action = _workyard_action(env)
  found, _, _, _ = _hand_contact_summary(env)
  rows = torch.arange(env.num_envs, device=env.device)
  released = (~found[rows, command.active_hand]) & (
    action.coordination[rows, command.active_hand * 4 + 1] < 0.20
  )
  settled = torch.linalg.vector_norm(command.rod.data.root_link_vel_w, dim=-1) < 0.10
  return (
    _object_inside_zone(env) & released & settled
  ).float() * _teacher_stage_mask(env, "place")


def upright_support(env: ManagerBasedRlEnv) -> torch.Tensor:
  robot: Entity = env.scene["robot"]
  upright = (-robot.data.projected_gravity_b[:, 2]).clamp(0.0, 1.0)
  feet, _, _ = _foot_contact_summary(env)
  return upright * feet.any(dim=-1).float()


def fell_over(env: ManagerBasedRlEnv) -> torch.Tensor:
  robot: Entity = env.scene["robot"]
  return (robot.data.root_link_pos_w[:, 2] < 0.50) | (
    robot.data.projected_gravity_b[:, 2] > -0.45
  )


def fall_cost(env: ManagerBasedRlEnv) -> torch.Tensor:
  return fell_over(env).float()


def non_foot_collision_cost(env: ManagerBasedRlEnv) -> torch.Tensor:
  sensor = _contact_sensor(env, "non_foot_ground_contact")
  force = sensor.data.force
  if force is None:
    return torch.zeros(env.num_envs, device=env.device)
  return (torch.linalg.vector_norm(force, dim=-1) > 10.0).any(dim=-1).float()


def foot_slip_cost(env: ManagerBasedRlEnv) -> torch.Tensor:
  _, _, slip = _foot_contact_summary(env)
  return slip.square().sum(dim=-1)


class JointActuatorSaturationCost:
  """Penalize use of the joint-level actuator authority declared by the G1.

  The G1 XML limits the *sum* of all actuator forces transmitted to each joint via
  ``joint.actuatorfrcrange``.  Its position actuators intentionally have no separate
  ``actuator.forcerange``.  Comparing ``actuator_force`` against that unset range
  therefore invents a near-zero denominator and is not a physical saturation
  metric.  Cache only the joint mapping here: per-environment limits remain dynamic
  because training randomizes them on reset.
  """

  def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
    del cfg
    self.robot: Entity = env.scene["robot"]
    self.joint_ids = self.robot.indexing.joint_ids
    limited = env.sim.model.jnt_actfrclimited[self.joint_ids]
    default_ranges = env.sim.get_default_field("jnt_actfrcrange")[self.joint_ids]
    valid_ranges = (
      torch.isfinite(default_ranges).all(dim=-1)
      & (default_ranges[:, 0] < 0.0)
      & (default_ranges[:, 1] > 0.0)
    )
    if not bool(limited.all().item()) or not bool(valid_ranges.all().item()):
      raise ValueError(
        "Every controlled G1 joint must declare a finite, enabled actuatorfrcrange"
      )

  def force_ratio(self, env: ManagerBasedRlEnv) -> torch.Tensor:
    effort = self.robot.data.qfrc_actuator
    ranges = env.sim.model.jnt_actfrcrange[:, self.joint_ids]
    directional_limit = torch.where(effort >= 0.0, ranges[..., 1], -ranges[..., 0])
    return effort.abs() / directional_limit

  def __call__(self, env: ManagerBasedRlEnv) -> torch.Tensor:
    return torch.relu(self.force_ratio(env) - 0.90).mean(dim=-1)


@requires_model_fields("jnt_actfrcrange")
def randomize_joint_actuator_strength(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor | None,
  scale_range: tuple[float, float],
) -> None:
  """Scale the G1's joint-level total actuator-force limits per environment."""

  robot: Entity = env.scene["robot"]
  if env_ids is None:
    env_ids = torch.arange(env.num_envs, dtype=torch.long, device=env.device)
  else:
    env_ids = env_ids.to(device=env.device, dtype=torch.long)
  joint_ids = robot.indexing.joint_ids
  default_ranges = env.sim.get_default_field("jnt_actfrcrange")[joint_ids]
  scales = torch.empty(
    (len(env_ids), len(joint_ids)), dtype=default_ranges.dtype, device=env.device
  ).uniform_(*scale_range)
  env.sim.model.jnt_actfrcrange[env_ids[:, None], joint_ids, 0] = (
    default_ranges[:, 0] * scales
  )
  env.sim.model.jnt_actfrcrange[env_ids[:, None], joint_ids, 1] = (
    default_ranges[:, 1] * scales
  )


def stage_completed(env: ManagerBasedRlEnv) -> torch.Tensor:
  return _workyard_command(env).completed_target


def dropped_after_grasp(env: ManagerBasedRlEnv) -> torch.Tensor:
  command = _workyard_command(env)
  was_grasped = command.teacher_stage >= TEACHER_STAGES.index("lift")
  dropped = command.rod.data.root_link_pos_w[:, 2] < SOURCE_POSITION[2] + 0.03
  return was_grasped & dropped & ~_object_inside_zone(env)


def illegal_ground_contact(env: ManagerBasedRlEnv) -> torch.Tensor:
  return non_foot_collision_cost(env) > 0.0


def workyard_curriculum(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor | slice,
) -> dict[str, float]:
  command = _workyard_command(env)
  if isinstance(env_ids, slice):
    ids = torch.arange(env.num_envs, device=env.device)[env_ids]
  else:
    ids = env_ids
  completed_episode = env.episode_length_buf[ids] > 0
  if torch.any(completed_episode):
    finished_ids = ids[completed_episode]
    targets = command.teacher_target_stage[finished_ids]
    successes = command.completed_target[finished_ids]
    for stage_index in range(len(TEACHER_STAGES)):
      mask = targets == stage_index
      command.episode_counts[stage_index] += mask.sum()
      command.success_counts[stage_index] += (mask & successes).sum()

  current = command.max_stage
  count = int(command.episode_counts[current].item())
  success = int(command.success_counts[current].item())
  rate = success / count if count else 0.0
  if (
    current < len(TEACHER_STAGES) - 1
    and count >= STAGE_MINIMUM_EPISODES[current]
    and rate >= STAGE_PROMOTION_RATES[current]
  ):
    command.max_stage += 1
  return {
    "max_stage": float(command.max_stage),
    "episodes": float(count),
    "success_rate": rate,
  }


def _workyard_action(env: ManagerBasedRlEnv) -> WorkyardAction:
  term = env.action_manager.get_term("workyard")
  if not isinstance(term, WorkyardAction):
    raise TypeError("HEAR Workyard action term is not active")
  return term


def _workyard_command(env: ManagerBasedRlEnv) -> WorkyardCommand:
  term = env.command_manager.get_term("workyard")
  if not isinstance(term, WorkyardCommand):
    raise TypeError("HEAR Workyard command term is not active")
  return term


def _contact_sensor(env: ManagerBasedRlEnv, name: str) -> ContactSensor:
  sensor = env.scene[name]
  if not isinstance(sensor, ContactSensor):
    raise TypeError(f"HEAR Workyard contact sensor is missing: {name}")
  return sensor


def _hand_contact_summary(
  env: ManagerBasedRlEnv,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
  found_outputs = []
  force_outputs = []
  surface_outputs = []
  opposing_outputs = []
  for name in ("left_hand_object_contact", "right_hand_object_contact"):
    sensor = _contact_sensor(env, name)
    data = sensor.data
    if data.found is None or data.force is None or data.normal is None:
      raise RuntimeError(f"Incomplete HEAR hand contact sensor: {name}")
    contacted = data.found > 0
    force_norm = torch.linalg.vector_norm(data.force, dim=-1)
    total_force = torch.where(contacted, force_norm, 0.0).sum(dim=-1)
    surfaces = contacted.sum(dim=-1).float()
    normals = torch_f.normalize(data.normal, dim=-1, eps=1e-6)
    dot = torch.einsum("bik,bjk->bij", normals, normals)
    valid_pairs = contacted.unsqueeze(2) & contacted.unsqueeze(1)
    eye = torch.eye(dot.shape[-1], dtype=torch.bool, device=env.device).unsqueeze(0)
    opposed = ((dot < -0.20) & valid_pairs & ~eye).any(dim=(1, 2))
    found_outputs.append(contacted.any(dim=-1))
    force_outputs.append(total_force)
    surface_outputs.append(surfaces)
    opposing_outputs.append(opposed)
  return (
    torch.stack(found_outputs, dim=-1),
    torch.stack(force_outputs, dim=-1),
    torch.stack(surface_outputs, dim=-1),
    torch.stack(opposing_outputs, dim=-1),
  )


def _foot_contact_summary(
  env: ManagerBasedRlEnv,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
  sensor = _contact_sensor(env, "feet_ground_contact")
  data = sensor.data
  if data.found is None or data.force is None:
    raise RuntimeError("Incomplete HEAR foot contact sensor")
  found = data.found > 0
  force = torch.linalg.vector_norm(data.force, dim=-1)
  robot: Entity = env.scene["robot"]
  body_ids, _ = robot.find_bodies(
    ("left_ankle_roll_link", "right_ankle_roll_link"), preserve_order=True
  )
  velocity = robot.data.body_link_lin_vel_w[:, body_ids, :2]
  slip = torch.linalg.vector_norm(velocity, dim=-1) * found.float()
  return found.float(), force, slip


def _object_inside_zone(env: ManagerBasedRlEnv) -> torch.Tensor:
  command = _workyard_command(env)
  rod = command.rod.data.root_link_pos_w
  delta = rod - command.target_position
  horizontal_inside = (delta[:, :2].abs() <= TARGET_HALF_EXTENT).all(dim=-1)
  supported_height = rod[:, 2] <= 0.16
  return horizontal_inside & supported_height


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
  non_foot = ContactSensorCfg(
    name="non_foot_ground_contact",
    primary=ContactMatch(
      mode="body",
      pattern=(
        r"^pelvis$",
        r"^(?!left_ankle_roll_link$|right_ankle_roll_link$).*_link$",
      ),
      entity="robot",
    ),
    secondary=ContactMatch(mode="body", pattern="terrain"),
    fields=("found", "force"),
    reduce="maxforce",
    num_slots=1,
    history_length=4,
  )

  def hand(side: str) -> ContactSensorCfg:
    return ContactSensorCfg(
      name=f"{side}_hand_object_contact",
      primary=ContactMatch(
        mode="body",
        pattern=rf"^{side}_hand_.*_link$",
        entity="robot",
      ),
      secondary=ContactMatch(mode="body", pattern="assembly_rod", entity="assembly_rod"),
      fields=("found", "force", "normal"),
      reduce="maxforce",
      num_slots=1,
      history_length=4,
    )

  return (feet, hand("left"), hand("right"), non_foot)


def make_workyard_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  observations = {
    "actor": ObservationGroupCfg(
      terms={
        "workyard": ObservationTermCfg(
          func=WorkyardObservation,
          delay_min_lag=0 if play else 0,
          delay_max_lag=0 if play else 2,
          clip=(-20.0, 20.0),
        )
      },
      concatenate_terms=True,
      enable_corruption=False,
    ),
    "critic": ObservationGroupCfg(
      terms={"workyard": ObservationTermCfg(func=WorkyardObservation)},
      concatenate_terms=True,
      enable_corruption=False,
    ),
  }

  events = {
    "reset_scene": EventTermCfg(func=mdp.reset_scene_to_default, mode="reset"),
    "object_mass": EventTermCfg(
      func=dr.pseudo_inertia,
      mode="reset",
      params={
        "asset_cfg": SceneEntityCfg("assembly_rod", body_names=("assembly_rod",)),
        "distribution": "uniform",
        # pseudo_inertia applies mass scale exp(2 * alpha).
        "alpha_range": (0.5 * math.log(0.7), 0.5 * math.log(1.3)),
      },
    ),
    "object_friction": EventTermCfg(
      func=dr.geom_friction,
      mode="reset",
      params={
        "asset_cfg": SceneEntityCfg("assembly_rod", geom_names=("assembly_rod_geom",)),
        "operation": "scale",
        "distribution": "uniform",
        "axes": [0, 1, 2],
        "ranges": (0.7, 1.3),
      },
    ),
    "actuator_strength": EventTermCfg(
      func=randomize_joint_actuator_strength,
      mode="reset",
      params={
        "scale_range": (0.9, 1.1),
      },
    ),
  }
  if play:
    events = {"reset_scene": events["reset_scene"]}

  cfg = ManagerBasedRlEnvCfg(
    scene=SceneCfg(
      terrain=TerrainEntityCfg(terrain_type="plane"),
      entities={
        "robot": _robot_cfg(),
        "pickup_stand": _source_cfg(),
        "assembly_rod": _rod_cfg(),
        "assembly_bay": _target_cfg(),
      },
      sensors=_contact_sensors(),
      num_envs=1 if play else 2048,
      env_spacing=5.0,
    ),
    observations=observations,
    actions={
      "workyard": WorkyardActionCfg(
        entity_name="robot", body_scale=0.35, hand_scale=0.08
      )
    },
    commands={
      "workyard": WorkyardCommandCfg(
        resampling_time_range=(20.0, 20.0),
        debug_vis=False,
      )
    },
    events=events,
    rewards={
      "reach_distance_progress": RewardTermCfg(
        func=ProgressReward, weight=2.0,
        params={"metric": "reach", "stage_index": TEACHER_STAGES.index("reach")},
      ),
      "wrist_orientation_alignment": RewardTermCfg(
        func=wrist_orientation_alignment, weight=0.25
      ),
      "qualified_hand_contact": RewardTermCfg(func=qualified_hand_contact, weight=0.5),
      "opposing_contact": RewardTermCfg(func=opposing_contact, weight=1.5),
      "grasp_verified": RewardTermCfg(func=grasp_verified, weight=8.0),
      "lift_height_progress": RewardTermCfg(
        func=ProgressReward, weight=3.0,
        params={"metric": "lift", "stage_index": TEACHER_STAGES.index("lift")},
      ),
      "object_to_zone_progress": RewardTermCfg(
        func=ProgressReward, weight=3.0,
        params={"metric": "carry", "stage_index": TEACHER_STAGES.index("carry")},
      ),
      "object_inside_zone": RewardTermCfg(func=object_inside_zone, weight=4.0),
      "object_released_and_settled": RewardTermCfg(
        func=released_and_settled, weight=15.0
      ),
      "upright_support": RewardTermCfg(func=upright_support, weight=0.5),
      "fall": RewardTermCfg(func=fall_cost, weight=-25.0),
      "non_foot_collision": RewardTermCfg(func=non_foot_collision_cost, weight=-2.0),
      "joint_limit_proximity": RewardTermCfg(
        func=mdp.joint_pos_limits, weight=-0.5,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=(".*",))},
      ),
      "foot_slip": RewardTermCfg(func=foot_slip_cost, weight=-0.2),
      "actuator_saturation": RewardTermCfg(
        func=JointActuatorSaturationCost, weight=-0.5
      ),
      "action_rate": RewardTermCfg(func=mdp.action_rate_l2, weight=-0.02),
    },
    terminations={
      "time_out": TerminationTermCfg(func=mdp.time_out, time_out=True),
      "fall": TerminationTermCfg(func=fell_over),
      "non_foot_ground": TerminationTermCfg(func=illegal_ground_contact),
      "stage_success": TerminationTermCfg(func=stage_completed),
      "dropped_after_grasp": TerminationTermCfg(func=dropped_after_grasp),
    },
    curriculum={
      "workyard_stage": CurriculumTermCfg(func=workyard_curriculum),
    },
    viewer=ViewerConfig(
      origin_type=ViewerConfig.OriginType.ASSET_BODY,
      entity_name="robot",
      body_name="torso_link",
      distance=3.2,
      elevation=-12.0,
      azimuth=135.0,
    ),
    sim=SimulationCfg(
      nconmax=256,
      njmax=2000,
      contact_sensor_maxmatch=512,
      mujoco=MujocoCfg(
        timestep=0.005,
        iterations=10,
        ls_iterations=20,
        impratio=10,
        cone="elliptic",
      ),
    ),
    decimation=4,
    episode_length_s=20.0,
    is_finite_horizon=True,
  )
  if play:
    cfg.curriculum = {}
    cfg.commands["workyard"].resampling_time_range = (20.0, 20.0)
  return cfg


def workyard_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
  return RslRlOnPolicyRunnerCfg(
    actor=RslRlModelCfg(
      hidden_dims=(512, 512, 256),
      activation="elu",
      obs_normalization=True,
      distribution_cfg={
        "class_name": "GaussianDistribution",
        "init_std": 0.6,
        "std_type": "scalar",
      },
    ),
    critic=RslRlModelCfg(
      hidden_dims=(512, 512, 256),
      activation="elu",
      obs_normalization=True,
    ),
    algorithm=RslRlPpoAlgorithmCfg(
      value_loss_coef=1.0,
      use_clipped_value_loss=True,
      clip_param=0.2,
      entropy_coef=0.004,
      num_learning_epochs=5,
      num_mini_batches=8,
      learning_rate=5.0e-4,
      schedule="adaptive",
      gamma=0.99,
      lam=0.95,
      desired_kl=0.01,
      max_grad_norm=1.0,
    ),
    experiment_name="hear_workyard_g1",
    save_interval=100,
    num_steps_per_env=32,
    max_iterations=10_000,
  )


def register_workyard_task() -> None:
  if TASK_ID in list_tasks():
    return
  register_mjlab_task(
    task_id=TASK_ID,
    env_cfg=make_workyard_env_cfg(play=False),
    play_env_cfg=make_workyard_env_cfg(play=True),
    rl_cfg=workyard_ppo_runner_cfg(),
  )


register_workyard_task()
