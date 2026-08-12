"""HEAR phase-two contact/grasp environment.

The accepted 14D reach policy is a frozen Harness child skill.  A separate 8D
hand-synergy actor receives gradient authority only after the typed contact
capability becomes active.  The resulting physical command is a 22D logical
composition while the learner itself cannot alter reach, locomotion, waist, or
the inactive hand.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Final

import mujoco
import torch
import torch.nn.functional as torch_f

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers.action_manager import ActionTermCfg
from mjlab.managers.command_manager import CommandTermCfg
from mjlab.managers.observation_manager import (
  ObservationGroupCfg,
  ObservationTermCfg,
)
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.tasks.registry import list_tasks, register_mjlab_task

import workyard_mjlab_env as base
import workyard_residual_mjlab_env as reach

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


TASK_ID: Final = "Hear-Workyard-Frozen-Reach-Hand-Synergy-G1-v1"
REACH_OBSERVATION_SIZE: Final = reach.OBSERVATION_SIZE
HAND_OBSERVATION_SIZE: Final = REACH_OBSERVATION_SIZE + 16
HAND_ACTION_SIZE: Final = 8
COMPOSED_ACTION_SIZE: Final = reach.ACTION_SIZE + HAND_ACTION_SIZE
MAXIMUM_NUMERICALLY_STABLE_QVEL: Final = 250.0
MAXIMUM_NUMERICALLY_STABLE_QACC: Final = 1.0e6
CONTACT_STAGE_INDEX: Final = base.TEACHER_STAGES.index("contact")
GRASP_STAGE_INDEX: Final = base.TEACHER_STAGES.index("grasp")
HAND_SYNERGY_NAMES: Final = (
  "left_opposition",
  "left_thumb",
  "left_index",
  "left_middle",
  "right_opposition",
  "right_thumb",
  "right_index",
  "right_middle",
)
REACH_POLICY_PROTOCOL: Final = "hear-frozen-reach-policy-export-v1"
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REACH_POLICY_ROOT = (
  REPOSITORY_ROOT / "artifacts" / "training" / "workyard-reach-frozen-v15"
)
DEFAULT_REACH_POLICY_JIT = DEFAULT_REACH_POLICY_ROOT / "workyard_reach_selected.jit.pt"
DEFAULT_REACH_POLICY_REPORT = DEFAULT_REACH_POLICY_ROOT / "reach-policy-report.json"
HAND_ENDPOINTS: Final = (
  base.HAND_JOINT_LIMITS[0][0],
  base.HAND_JOINT_LIMITS[1][1],
  base.HAND_JOINT_LIMITS[2][1],
  base.HAND_JOINT_LIMITS[3][0],
  base.HAND_JOINT_LIMITS[4][0],
  base.HAND_JOINT_LIMITS[5][0],
  base.HAND_JOINT_LIMITS[6][0],
  base.HAND_JOINT_LIMITS[7][0],
  base.HAND_JOINT_LIMITS[8][0],
  base.HAND_JOINT_LIMITS[9][0],
  base.HAND_JOINT_LIMITS[10][1],
  base.HAND_JOINT_LIMITS[11][1],
  base.HAND_JOINT_LIMITS[12][1],
  base.HAND_JOINT_LIMITS[13][1],
)
HAND_SYNERGY_INDEX_BY_JOINT: Final = (
  0, 1, 1, 3, 3, 2, 2,
  4, 5, 5, 7, 7, 6, 6,
)


def _seeded_unit_interval(
  episode_seeds: torch.Tensor, salt: int
) -> torch.Tensor:
  """Portable integer PRNG used only for held-out per-environment resets."""
  modulus = 2_147_483_647
  state = torch.remainder(
    episode_seeds * 48_271 + (salt + 1) * 69_621,
    modulus,
  )
  state = torch.remainder(state * 48_271, modulus)
  return state.to(dtype=torch.float32) / float(modulus)


class FrozenReachPolicy:
  """Accepted reach actor with no gradient or mutable checkpoint authority."""

  def __init__(
    self,
    jit_path: Path,
    report_path: Path,
    device: torch.device | str,
    validation_batch_size: int,
  ):
    if not jit_path.is_file() or not report_path.is_file():
      raise FileNotFoundError("Accepted frozen reach policy artifacts are missing")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    source = report.get("source")
    policy = report.get("policy")
    if (
      report.get("protocol") != REACH_POLICY_PROTOCOL
      or not isinstance(source, dict)
      or source.get("phase_one_accepted") is not True
      or source.get("hand_checkpoint_expansion_authorized") is not True
      or source.get("waist_checkpoint_expansion_authorized") is not False
      or source.get("held_out_environment_count", 0) < 500
      or source.get("held_out_success_rate", 0.0) < 0.85
      or not isinstance(policy, dict)
      or policy.get("file") != jit_path.name
      or policy.get("bytes") != jit_path.stat().st_size
      or policy.get("sha256") != reach._sha256(jit_path)
      or policy.get("runtime") != "torchscript_cuda"
      or policy.get("input_size") != REACH_OBSERVATION_SIZE
      or policy.get("output_size") != reach.ACTION_SIZE
      or policy.get("batch_dynamic") is not True
      or policy.get("gradient_parameter_count") != 0
    ):
      raise ValueError("Frozen reach policy identity does not match its accepted report")

    loaded = torch.jit.load(str(jit_path), map_location=device).eval()
    for parameter in loaded.parameters():
      parameter.requires_grad_(False)
    self.module = torch.jit.freeze(loaded)
    self.parameter_count = sum(
      parameter.numel() for parameter in self.module.parameters()
    )
    self.gradient_parameter_count = sum(
      parameter.numel()
      for parameter in self.module.parameters()
      if parameter.requires_grad
    )
    if self.gradient_parameter_count != 0:
      raise RuntimeError("Frozen reach policy retained gradient authority")
    self.identity = {
      "protocol": "hear-frozen-qualified-reach-runtime-v1",
      "jit_sha256": policy["sha256"],
      "report_sha256": reach._sha256(report_path),
      "source_checkpoint_sha256": source["checkpoint_sha256"],
      "source_selected_source": source["selected_source"],
      "parameter_count": self.parameter_count,
      "gradient_parameter_count": self.gradient_parameter_count,
      "held_out_environment_count": source["held_out_environment_count"],
      "held_out_success_rate": source["held_out_success_rate"],
      "execution_authority": "frozen_14d_upper_body_only",
    }
    with torch.inference_mode():
      probe = torch.zeros(
        (validation_batch_size, REACH_OBSERVATION_SIZE),
        dtype=torch.float32,
        device=device,
      )
      output = self.module(probe)
    if (
      tuple(output.shape) != (validation_batch_size, reach.ACTION_SIZE)
      or not torch.isfinite(output).all()
      or float(output.abs().max().item()) > 1.0 + 1.0e-6
    ):
      raise RuntimeError("Frozen reach policy failed dynamic-batch validation")

  def infer(self, observation: torch.Tensor) -> torch.Tensor:
    if observation.ndim != 2 or observation.shape[1] != REACH_OBSERVATION_SIZE:
      raise ValueError(
        f"Expected frozen reach observation [B, {REACH_OBSERVATION_SIZE}], "
        f"got {observation.shape}"
      )
    with torch.inference_mode():
      action = self.module(observation)
    if (
      action.shape != (observation.shape[0], reach.ACTION_SIZE)
      or not torch.isfinite(action).all()
      or float(action.abs().max().item()) > 1.0 + 1.0e-6
    ):
      raise RuntimeError("Frozen reach policy emitted an invalid action")
    return action


def _load_contact_actuated_g1_spec() -> mujoco.MjSpec:
  """Keep the accepted body plant and use compliant position control for hands."""
  spec = reach._load_hybrid_actuated_g1_spec()
  joints = {joint.name: joint for joint in spec.joints}
  actuators = {actuator.target: actuator for actuator in spec.actuators}
  for joint_name in base.HAND_JOINT_NAMES:
    joint = joints.get(joint_name)
    actuator = actuators.get(joint_name)
    if joint is None or actuator is None:
      raise ValueError(f"Contact G1 is missing hand actuation for {joint_name}")
    joint.damping[:] = 0.0
    # Free-object opposing contact can amplify a one-control-step target
    # update into a short force spike.  This compliant hand still clears the
    # 4 N verified-grasp floor while leaving margin below the 30 N hard gate.
    actuator.set_to_position(kp=2.5, kv=0.3, inheritrange=True)
  return spec


def _make_contact_source_spec() -> mujoco.MjSpec:
  """Pickup plate with a shallow, vertically open rod guide."""
  spec = mujoco.MjSpec()
  body = spec.worldbody.add_body(name="pickup_stand")
  common = {
    "type": mujoco.mjtGeom.mjGEOM_BOX,
    "contype": 1,
    "conaffinity": 1,
    "friction": (0.8, 0.01, 0.001),
  }
  body.add_geom(
    name="pickup_stand_geom",
    size=(0.06, 0.06, 0.005),
    rgba=(0.46, 0.53, 0.49, 1.0),
    **common,
  )
  # The 3 mm radial clearance does not clamp the object.  Four 4 cm-high
  # guides only keep the base of the vertical rod from toppling during grasp
  # acquisition; the rod remains free to lift out along +z.
  wall_rgba = (0.38, 0.46, 0.42, 1.0)
  for axis, sign in ((0, -1.0), (0, 1.0), (1, -1.0), (1, 1.0)):
    position = [0.0, 0.0, 0.025]
    position[axis] = sign * 0.037
    size = [0.045, 0.045, 0.020]
    size[axis] = 0.004
    body.add_geom(
      name=f"pickup_socket_{'xy'[axis]}_{'negative' if sign < 0 else 'positive'}",
      pos=tuple(position),
      size=tuple(size),
      rgba=wall_rgba,
      **common,
    )
  return spec


@dataclass(kw_only=True)
class WorkyardContactCommandCfg(base.WorkyardCommandCfg):
  contact_base_assist_enabled: bool = False
  contact_base_assist_gain_s_inv: float = 1.5
  contact_base_assist_min_speed_m_s: float = 0.10
  contact_base_assist_max_speed_m_s: float = 0.12
  evaluation_episode_seed_base: int | None = None

  def build(self, env: ManagerBasedRlEnv) -> "WorkyardContactCommand":
    return WorkyardContactCommand(self, env)


class WorkyardContactCommand(base.WorkyardCommand):
  """Fixed reach -> contact -> verified-grasp Harness training window."""

  def __init__(self, cfg: WorkyardContactCommandCfg, env: ManagerBasedRlEnv):
    super().__init__(cfg, env)
    pocket_radius = (
      cfg.contact_pocket_forward_m ** 2
      + cfg.contact_pocket_lateral_m ** 2
    ) ** 0.5
    if (
      cfg.contact_alignment_radius_m <= pocket_radius
      or cfg.contact_alignment_tolerance_m <= 0.0
      or cfg.contact_retreat_tolerance_m <= 0.0
      or cfg.contact_alignment_max_force_n <= 0.0
      or cfg.contact_closure_tolerance_m <= 0.0
      or cfg.contact_closure_vertical_tolerance_m <= 0.0
      or cfg.contact_base_assist_gain_s_inv <= 0.0
      or cfg.contact_base_assist_min_speed_m_s <= 0.0
      or cfg.contact_base_assist_max_speed_m_s <= 0.0
      or cfg.contact_base_assist_min_speed_m_s
      > cfg.contact_base_assist_max_speed_m_s
      or (
        cfg.evaluation_episode_seed_base is not None
        and cfg.evaluation_episode_seed_base < 0
      )
    ):
      raise ValueError("Contact alignment shell must enclose the grasp pocket")
    self.max_stage = GRASP_STAGE_INDEX
    self.contact_base_assist_active = torch.zeros(
      self.num_envs, dtype=torch.bool, device=self.device
    )
    self.contact_base_assist_direction_latched = torch.zeros_like(
      self.contact_base_assist_active
    )
    self.contact_base_assist_direction_w = torch.zeros(
      (self.num_envs, 2), device=self.device
    )

  def _resample_command(self, env_ids: torch.Tensor) -> None:
    super()._resample_command(env_ids)
    if self.cfg.evaluation_episode_seed_base is not None:
      # One vectorized environment maps to one explicit held-out episode seed.
      # This replaces only contact-task reset randomness; formal evaluation
      # uses the play configuration, so no training randomization is active.
      episode_seeds = (
        env_ids.to(dtype=torch.long)
        + self.cfg.evaluation_episode_seed_base
      )
      origins = self._env.scene.env_origins[env_ids]
      object_jitter = torch.zeros(
        (len(env_ids), 3), dtype=torch.float32, device=self.device
      )
      object_jitter[:, 0] = (
        2.0 * _seeded_unit_interval(episode_seeds, 0) - 1.0
      ) * self.cfg.object_position_jitter_m
      object_jitter[:, 1] = (
        2.0 * _seeded_unit_interval(episode_seeds, 1) - 1.0
      ) * self.cfg.object_position_jitter_m
      self.active_hand[env_ids] = (object_jitter[:, 1] < 0.0).long()
      rod_pos = torch.tensor(
        base.ROD_START_POSITION, dtype=torch.float32, device=self.device
      ).expand(len(env_ids), 3)
      rod_pos = rod_pos + origins + object_jitter
      self.initial_rod_position[env_ids] = rod_pos
      yaw = (
        2.0 * _seeded_unit_interval(episode_seeds, 2) - 1.0
      ) * 0.25
      rod_quat = base.quat_from_euler_xyz(
        torch.zeros_like(yaw), torch.zeros_like(yaw), yaw
      )
      self.rod.data.write_root_pose(
        torch.cat((rod_pos, rod_quat), dim=-1), env_ids
      )
      self.rod.data.write_root_velocity(
        torch.zeros((len(env_ids), 6), device=self.device), env_ids
      )
    self.teacher_stage[env_ids] = 0
    self.teacher_target_stage[env_ids] = GRASP_STAGE_INDEX
    self.completed_target[env_ids] = False
    self.contact_base_assist_active[env_ids] = False
    self.contact_base_assist_direction_latched[env_ids] = False
    self.contact_base_assist_direction_w[env_ids] = 0.0

  def _update_task_space_targets(self) -> None:
    super()._update_task_space_targets()
    self.contact_base_assist_active[:] = False
    if not bool(self.contact_pocket_active.any().item()):
      return
    action = base._workyard_action(self._env)
    coordination = action.coordination.reshape(self.num_envs, 2, 4)
    rows = torch.arange(self.num_envs, device=self.device)
    active_coordination = coordination[rows, self.active_hand]
    found, _, _, _ = base._hand_contact_summary(self._env)
    active_found = found[rows, self.active_hand]
    wrist_delta = reach.active_wrist_position_delta(self._env)
    target_delta_b = -wrist_delta[:, :2]
    target_delta_w = base.quat_apply(
      self.robot.data.root_link_quat_w,
      torch.cat((target_delta_b, torch.zeros_like(target_delta_b[:, :1])), dim=-1),
    )[:, :2]
    target_distance_w = torch.linalg.vector_norm(
      target_delta_w, dim=-1, keepdim=True
    ).clamp_min(1e-6)
    newly_entered_pocket = (
      self.contact_pocket_active
      & ~self.contact_base_assist_direction_latched
    )
    self.contact_base_assist_direction_w[:] = torch.where(
      newly_entered_pocket.unsqueeze(-1),
      target_delta_w / target_distance_w,
      self.contact_base_assist_direction_w,
    )
    self.contact_base_assist_direction_latched |= self.contact_pocket_active
    direction_b = base.quat_apply_inverse(
      self.robot.data.root_link_quat_w,
      torch.cat((
        self.contact_base_assist_direction_w,
        torch.zeros_like(self.contact_base_assist_direction_w[:, :1]),
      ), dim=-1),
    )[:, :2]
    direction_b = direction_b / torch.linalg.vector_norm(
      direction_b, dim=-1, keepdim=True
    ).clamp_min(1e-6)
    planar_wrist_error = torch.linalg.vector_norm(wrist_delta[:, :2], dim=-1)
    remaining_distance = torch.sum(
      target_delta_b * direction_b, dim=-1
    ).clamp_min(0.0)
    self.contact_base_assist_active[:] = (
      self.cfg.contact_base_assist_enabled
      & self.contact_pocket_active
      & (planar_wrist_error > self.cfg.contact_closure_tolerance_m)
      & (remaining_distance > 0.0)
      & (active_coordination.abs().amax(dim=-1) <= 0.02)
      & ~active_found
    )
    requested_speed = torch.clamp(
      remaining_distance * self.cfg.contact_base_assist_gain_s_inv,
      min=self.cfg.contact_base_assist_min_speed_m_s,
      max=self.cfg.contact_base_assist_max_speed_m_s,
    )
    requested_velocity = direction_b * requested_speed.unsqueeze(-1)
    self.desired_base_twist[:, :2] = torch.where(
      self.contact_base_assist_active.unsqueeze(-1),
      requested_velocity,
      torch.zeros_like(requested_velocity),
    )
    self.desired_base_twist[:, 2] = 0.0

  def _update_command(self) -> None:
    stage_before = self.teacher_stage.clone()
    super()._update_command()
    # Base command generation computes task-space targets before advancing the
    # typed stage.  Refresh immediately on a transition so contact authority
    # sees the new wrist-frame grasp pocket rather than one stale pre-grasp
    # command.
    if bool((self.teacher_stage != stage_before).any().item()):
      self._update_task_space_targets()


def hand_closure_ready(env: ManagerBasedRlEnv) -> torch.Tensor:
  command = base._workyard_command(env)
  wrist_delta = reach.active_wrist_position_delta(env)
  return (
    command.contact_pocket_active
    & (
      torch.linalg.vector_norm(wrist_delta[:, :2], dim=-1)
      <= command.cfg.contact_closure_tolerance_m
    )
    & (
      wrist_delta[:, 2].abs()
      <= command.cfg.contact_closure_vertical_tolerance_m
    )
  )


@dataclass(kw_only=True)
class WorkyardContactActionCfg(reach.WorkyardResidualActionCfg):
  hand_synergy_step: float = 0.02
  # The policy's coordination state is a desired grasp, not permission to
  # accumulate arbitrary position-servo error behind a blocked fingertip.
  # Bound only the closing direction against measured joint position; opening
  # and force-release commands remain immediate.
  hand_max_closing_joint_lead_rad: float = 0.25
  contact_force_stop_n: float = 6.0
  contact_force_emergency_release_n: float = 12.0
  preclosure_pose_hold_force_n: float = 2.0
  preclosure_pose_hold_planar_tolerance_m: float = 0.045
  preclosure_pose_hold_right_planar_tolerance_m: float = 0.060
  preclosure_pose_hold_vertical_tolerance_m: float = 0.045
  finger_preshape_coordination: float = 0.88
  precontact_thumb_safety_margin_m: float = 0.105
  preshape_thumb_safety_margin_m: float = 0.145
  contact_approach_task_space_feedback_gain: float = 1.0
  contact_approach_posture_attractor_gain: float = 0.1
  contact_approach_wrist_bearing_feedback_gain: float = 1.0
  contact_approach_wrist_axis_alignment_feedback_gain: float = 1.0
  contact_approach_wrist_bearing_task_weight_m_per_rad: float = 0.12
  contact_approach_max_wrist_bearing_step_rad: float = 0.35
  contact_approach_max_joint_correction_rad: float = 0.12
  contact_approach_max_solver_target_slew_rad: float = 0.015
  # Contact is sampled once per 20 ms control update.  Slow only the final
  # open-hand pocket insertion so reach/retreat/alignment retain their proven
  # convergence rate while first-contact impulse remains bounded.
  contact_pocket_max_solver_target_slew_rad: float = 0.0020
  contact_approach_max_command_lead_rad: float = 0.10
  # Bound stored position-servo energy in the final pocket independently of
  # the outer reach/alignment lead.  A slow target can still build a large
  # spring error before first contact unless this lag is capped explicitly.
  contact_pocket_max_command_lead_rad: float = 0.04
  contact_approach_hold_enter_error_m: float = 0.010
  contact_approach_hold_release_error_m: float = 0.020
  reach_policy_jit_path: str = str(DEFAULT_REACH_POLICY_JIT)
  reach_policy_report_path: str = str(DEFAULT_REACH_POLICY_REPORT)

  def build(self, env: ManagerBasedRlEnv) -> "WorkyardContactAction":
    return WorkyardContactAction(self, env)


class WorkyardContactAction(reach.WorkyardResidualAction):
  """8D learned hand action composed with an immutable 14D reach action."""

  cfg: WorkyardContactActionCfg

  def __init__(self, cfg: WorkyardContactActionCfg, env: ManagerBasedRlEnv):
    super().__init__(cfg, env)
    if (
      cfg.contact_approach_task_space_feedback_gain <= 0.0
      or not 0.0 < cfg.finger_preshape_coordination < 1.0
      or cfg.hand_max_closing_joint_lead_rad <= 0.0
      or cfg.contact_force_stop_n <= 0.0
      or cfg.contact_force_emergency_release_n <= cfg.contact_force_stop_n
      or cfg.preclosure_pose_hold_force_n <= 0.0
      or cfg.preclosure_pose_hold_force_n >= cfg.contact_force_stop_n
      or cfg.preclosure_pose_hold_planar_tolerance_m <= 0.0
      or cfg.preclosure_pose_hold_right_planar_tolerance_m <= 0.0
      or cfg.preclosure_pose_hold_vertical_tolerance_m <= 0.0
      or cfg.contact_approach_posture_attractor_gain < 0.0
      or cfg.contact_approach_wrist_bearing_feedback_gain <= 0.0
      or cfg.contact_approach_wrist_axis_alignment_feedback_gain <= 0.0
      or cfg.contact_approach_wrist_bearing_task_weight_m_per_rad <= 0.0
      or cfg.contact_approach_max_wrist_bearing_step_rad <= 0.0
      or cfg.contact_approach_max_joint_correction_rad <= 0.0
      or cfg.contact_approach_max_solver_target_slew_rad <= 0.0
      or cfg.contact_pocket_max_solver_target_slew_rad <= 0.0
      or cfg.contact_pocket_max_solver_target_slew_rad
        > cfg.contact_approach_max_solver_target_slew_rad
      or cfg.contact_approach_max_command_lead_rad <= 0.0
      or cfg.contact_pocket_max_command_lead_rad <= 0.0
      or cfg.contact_pocket_max_command_lead_rad
        > cfg.contact_approach_max_command_lead_rad
      or cfg.contact_approach_hold_enter_error_m <= 0.0
      or cfg.contact_approach_hold_release_error_m
        <= cfg.contact_approach_hold_enter_error_m
    ):
      raise ValueError("Contact approach DLS gains and hold band are invalid")
    # Phase one's label generator deliberately disables Cartesian feedback.
    # This contact-only instance becomes a deterministic terminal approach
    # executor; it is never exposed as an action dimension or trainable module.
    self.reach_teacher.task_space_feedback_gain = (
      cfg.contact_approach_task_space_feedback_gain
    )
    self.reach_teacher.posture_attractor_gain = (
      cfg.contact_approach_posture_attractor_gain
    )
    self.reach_teacher.wrist_bearing_feedback_gain = (
      cfg.contact_approach_wrist_bearing_feedback_gain
    )
    self.reach_teacher.wrist_axis_alignment_feedback_gain = (
      cfg.contact_approach_wrist_axis_alignment_feedback_gain
    )
    self.reach_teacher.wrist_bearing_task_weight_m_per_rad = (
      cfg.contact_approach_wrist_bearing_task_weight_m_per_rad
    )
    self.reach_teacher.max_wrist_bearing_step_rad = (
      cfg.contact_approach_max_wrist_bearing_step_rad
    )
    self.reach_teacher.max_joint_correction_rad = (
      cfg.contact_approach_max_joint_correction_rad
    )
    self.reach_teacher.max_solver_target_slew_rad = (
      cfg.contact_approach_max_solver_target_slew_rad
    )
    self.reach_teacher.pocket_max_solver_target_slew_rad = (
      cfg.contact_pocket_max_solver_target_slew_rad
    )
    self.reach_teacher.max_command_lead_rad = (
      cfg.contact_approach_max_command_lead_rad
    )
    self.reach_teacher.pocket_max_command_lead_rad = (
      cfg.contact_pocket_max_command_lead_rad
    )
    self.reach_teacher.hold_enter_error_m = (
      cfg.contact_approach_hold_enter_error_m
    )
    self.reach_teacher.hold_release_error_m = (
      cfg.contact_approach_hold_release_error_m
    )
    self._hand_action = torch.zeros(
      (self.num_envs, HAND_ACTION_SIZE), device=self.device
    )
    self._requested_hand_action = torch.zeros_like(self._hand_action)
    self._authority_mask = torch.zeros_like(self._hand_action, dtype=torch.bool)
    self._frozen_reach_action = torch.zeros(
      (self.num_envs, reach.ACTION_SIZE), device=self.device
    )
    self._contact_approach_correction_active = torch.zeros(
      self.num_envs, dtype=torch.bool, device=self.device
    )
    self._contact_approach_correction_delta = torch.zeros_like(
      self._frozen_reach_action
    )
    self._teacher_thumb_contact_latched = torch.zeros(
      (self.num_envs, 2), dtype=torch.bool, device=self.device
    )
    self._teacher_opposing_contact_latched = torch.zeros_like(
      self._teacher_thumb_contact_latched
    )
    self._contact_pose_hold_active = torch.zeros(
      self.num_envs, dtype=torch.bool, device=self.device
    )
    self._closure_pose_hold_active = torch.zeros_like(
      self._contact_pose_hold_active
    )
    self._hand_endpoint = torch.tensor(
      HAND_ENDPOINTS, dtype=torch.float32, device=self.device
    ).unsqueeze(0)
    self._hand_synergy_indexes = torch.tensor(
      HAND_SYNERGY_INDEX_BY_JOINT, dtype=torch.long, device=self.device
    )
    thumb_body_ids, thumb_body_names = self._entity.find_bodies(
      ("left_hand_thumb_2_link", "right_hand_thumb_2_link"),
      preserve_order=True,
    )
    if tuple(thumb_body_names) != (
      "left_hand_thumb_2_link", "right_hand_thumb_2_link"
    ):
      raise ValueError("Contact safety shield cannot resolve both thumb tips")
    self._thumb_body_ids = torch.tensor(
      thumb_body_ids, dtype=torch.long, device=self.device
    )
    self._safe_precontact_upper_targets = self._body_targets[
      :, reach.UPPER_BODY_SLICE
    ].clone()
    self._last_contact_free_upper_targets = (
      self._safe_precontact_upper_targets.clone()
    )
    self._guarded_contact_release_upper_targets = (
      self._safe_precontact_upper_targets.clone()
    )
    self._guarded_contact_active = torch.zeros(
      self.num_envs, dtype=torch.bool, device=self.device
    )
    self._contact_pose_hold_upper_targets = self._safe_precontact_upper_targets.clone()
    self._closure_pose_hold_upper_targets = self._safe_precontact_upper_targets.clone()
    self._approach_safety_intervention = torch.zeros(
      self.num_envs, dtype=torch.bool, device=self.device
    )
    self._hand_contact_sensors = (
      base._contact_sensor(env, "left_hand_object_contact"),
      base._contact_sensor(env, "right_hand_object_contact"),
    )
    self._thumb_contact_indexes: list[torch.Tensor] = []
    self._opposing_contact_indexes: list[torch.Tensor] = []
    for sensor in self._hand_contact_sensors:
      expanded_names = tuple(
        name
        for name in sensor.primary_names
        for _ in range(sensor.cfg.num_slots)
      )
      thumb_indexes = [
        index for index, name in enumerate(expanded_names) if "_thumb_" in name
      ]
      opposing_indexes = [
        index for index, name in enumerate(expanded_names) if "_thumb_" not in name
      ]
      if not thumb_indexes or not opposing_indexes:
        raise ValueError("Contact hand sensor did not resolve thumb/finger surfaces")
      self._thumb_contact_indexes.append(torch.tensor(
        thumb_indexes, dtype=torch.long, device=self.device
      ))
      self._opposing_contact_indexes.append(torch.tensor(
        opposing_indexes, dtype=torch.long, device=self.device
      ))
    self.frozen_reach = FrozenReachPolicy(
      Path(cfg.reach_policy_jit_path),
      Path(cfg.reach_policy_report_path),
      self.device,
      self.num_envs,
    )
    self._reach_observation_builder = reach.WorkyardResidualObservation(
      ObservationTermCfg(func=reach.WorkyardResidualObservation), env
    )

  @property
  def action_dim(self) -> int:
    return HAND_ACTION_SIZE

  @property
  def hand_action(self) -> torch.Tensor:
    return self._hand_action

  @property
  def requested_hand_action(self) -> torch.Tensor:
    return self._requested_hand_action

  @property
  def authority_mask(self) -> torch.Tensor:
    return self._authority_mask

  @property
  def approach_safety_intervention(self) -> torch.Tensor:
    return self._approach_safety_intervention

  @property
  def frozen_reach_action(self) -> torch.Tensor:
    return self._frozen_reach_action

  @property
  def contact_approach_correction_active(self) -> torch.Tensor:
    return self._contact_approach_correction_active

  @property
  def contact_approach_correction_delta(self) -> torch.Tensor:
    return self._contact_approach_correction_delta

  @property
  def teacher_thumb_contact_latched(self) -> torch.Tensor:
    return self._teacher_thumb_contact_latched

  @property
  def teacher_opposing_contact_latched(self) -> torch.Tensor:
    return self._teacher_opposing_contact_latched

  @property
  def contact_pose_hold_active(self) -> torch.Tensor:
    return self._contact_pose_hold_active

  @property
  def closure_pose_hold_active(self) -> torch.Tensor:
    return self._closure_pose_hold_active

  @property
  def composed_action(self) -> torch.Tensor:
    return torch.cat((self._raw_action, self._hand_action), dim=-1)

  def hand_teacher_authority_mask(self) -> torch.Tensor:
    """Return the exact 8D Harness authority mask for the current state."""
    command = base._workyard_command(self._env)
    active_side = torch_f.one_hot(
      command.active_hand, num_classes=2
    ).bool()
    active_synergy = active_side.unsqueeze(-1).expand(-1, -1, 4).reshape(
      self.num_envs, HAND_ACTION_SIZE
    )
    contact_capability = command.teacher_stage >= CONTACT_STAGE_INDEX
    contact_authorized = (
      hand_closure_ready(self._env) | self._closure_pose_hold_active
    )
    return (
      active_synergy
      & contact_capability.unsqueeze(-1)
      & contact_authorized.unsqueeze(-1)
    )

  def compute_hand_teacher_action(self) -> torch.Tensor:
    command = base._workyard_command(self._env)
    _, force, _, opposed = base._hand_contact_summary(self._env)
    rows = torch.arange(self.num_envs, device=self.device)
    active = command.active_hand
    active_force = force[rows, active]
    active_opposed = opposed[rows, active]
    thumb_contact = torch.zeros(
      (self.num_envs, 2), dtype=torch.bool, device=self.device
    )
    opposing_contact = torch.zeros_like(thumb_contact)
    for side, sensor in enumerate(self._hand_contact_sensors):
      sensor_found = sensor.data.found
      if sensor_found is None:
        raise RuntimeError("Contact hand sensor has no found field")
      contacted = sensor_found > 0
      thumb_contact[:, side] = contacted[
        :, self._thumb_contact_indexes[side]
      ].any(dim=-1)
      opposing_contact[:, side] = contacted[
        :, self._opposing_contact_indexes[side]
      ].any(dim=-1)
    active_thumb = thumb_contact[rows, active]
    active_opposing = opposing_contact[rows, active]
    active_thumb &= command.contact_pocket_active
    active_opposing &= command.contact_pocket_active
    qualified_digit_contact = (
      command.contact_pocket_active & (active_force >= 2.0)
    )
    self._teacher_thumb_contact_latched[rows, active] |= (
      active_thumb & qualified_digit_contact
    )
    self._teacher_opposing_contact_latched[rows, active] |= (
      active_opposing & qualified_digit_contact
    )
    # A free-object contact can disappear for a control update while the next
    # digit group is closing.  Preserve which side has been physically
    # qualified so the analytic teacher does not oscillate back to preshape.
    # These episode-local booleans are teacher state, never learner input.
    active_thumb |= self._teacher_thumb_contact_latched[rows, active]
    active_opposing |= self._teacher_opposing_contact_latched[rows, active]
    current_active = self.coordination.reshape(self.num_envs, 2, 4)[rows, active]

    # Preshape the two fingers first.  Once one side of the grasp touches the
    # rod, hold that group and close only the missing side; this avoids a
    # single high-force fingertip pushing the free object off its support.
    finger_closure_target = torch.tensor(
      (
        0.0,
        0.0,
        self.cfg.finger_preshape_coordination,
        self.cfg.finger_preshape_coordination,
      ),
      device=self.device,
    ).expand(self.num_envs, 4).clone()
    target_active = torch.where(
      command.contact_pocket_active.unsqueeze(-1),
      finger_closure_target,
      torch.zeros_like(finger_closure_target),
    )
    thumb_only = active_thumb & ~active_opposing
    target_active[thumb_only, :2] = current_active[thumb_only, :2]
    # The expanded 64-seed qualification exposed shallow left-hand contacts
    # where the thumb side was physically qualified but 0.94 curl left the
    # opposing fingers a few millimetres short at only 6--7 N.  Full travel is
    # authorized only for this missing-side state; the 6 N release reflex still
    # backs off immediately once contact appears.
    target_active[thumb_only, 2:] = 1.0
    opposing_only = active_opposing & ~active_thumb
    target_active[opposing_only, :2] = torch.tensor(
      (0.0, 1.0), device=self.device
    )
    # An opposing link can touch the rod while its coordination is still zero.
    # Freezing that incidental contact at zero leaves no stable surface for the
    # thumb to oppose and deadlocks the grasp.  Establish a shallow support
    # curl while the missing thumb closes; the directional 6 N reflex below
    # still unloads these already-contacting fingers first.
    target_active[opposing_only, 2:] = torch.maximum(
      current_active[opposing_only, 2:],
      torch.tensor(0.40, device=self.device),
    )
    both_sides = active_thumb & active_opposing
    target_active[both_sides] = torch.tensor(
      (0.0, 0.95, 0.92, 0.92), device=self.device
    )
    grasp_profile = torch.tensor(
      (0.0, 0.96, 0.96, 0.96), device=self.device
    ).expand(self.num_envs, 4)
    target_active = torch.where(
      active_opposed.unsqueeze(-1), grasp_profile, target_active
    )
    # Do not keep increasing closure once an opposing grasp has reached the
    # qualified force band.  A one-sided force spike must never freeze the
    # still-open opposing digits.
    target_active = torch.where(
      (
        active_opposed & (active_force >= self.cfg.contact_force_stop_n)
      ).unsqueeze(-1),
      current_active,
      target_active,
    )
    # Unload the side that is already carrying force without taking motion
    # away from the still-missing side.  Releasing all curls on a thumb-only
    # contact created a deadlock: the opposing fingers could never reach the
    # rod, even though total force remained low and the arm was safely held.
    force_release_target = target_active.clone()
    release_all = ~(thumb_only | opposing_only)
    force_release_target[release_all, 0] = current_active[release_all, 0]
    force_release_target[release_all, 1:] = torch.clamp(
      current_active[release_all, 1:] - self.cfg.hand_synergy_step,
      min=0.0,
    )
    force_release_target[thumb_only, 0] = current_active[thumb_only, 0]
    force_release_target[thumb_only, 1] = torch.clamp(
      current_active[thumb_only, 1] - self.cfg.hand_synergy_step,
      min=0.0,
    )
    force_release_target[opposing_only, 2:] = torch.clamp(
      current_active[opposing_only, 2:] - self.cfg.hand_synergy_step,
      min=0.0,
    )
    target_active = torch.where(
      (active_force >= self.cfg.contact_force_stop_n).unsqueeze(-1),
      force_release_target,
      target_active,
    )
    emergency_release_target = current_active.clone()
    emergency_release_target[:, 1:] = torch.clamp(
      emergency_release_target[:, 1:] - self.cfg.hand_synergy_step,
      min=0.0,
    )
    target_active = torch.where(
      (active_force >= self.cfg.contact_force_emergency_release_n).unsqueeze(-1),
      emergency_release_target,
      target_active,
    )
    target = torch.zeros_like(self.coordination)
    target.reshape(self.num_envs, 2, 4)[rows, active] = target_active
    authorized = (
      hand_closure_ready(self._env) | self._closure_pose_hold_active
    )
    target = torch.where(authorized.unsqueeze(-1), target, torch.zeros_like(target))
    return ((target - self.coordination) / self.cfg.hand_synergy_step).clamp(-1.0, 1.0)

  def process_actions(self, actions: torch.Tensor) -> None:
    if actions.shape != (self.num_envs, HAND_ACTION_SIZE):
      raise ValueError(
        f"Expected contact hand action [B, {HAND_ACTION_SIZE}], got {actions.shape}"
      )
    command = base._workyard_command(self._env)
    self._reach_teacher_action[:] = self.compute_reach_teacher_action().detach()
    reach_observation = self._reach_observation_builder(self._env)
    self._frozen_reach_action[:] = self.frozen_reach.infer(
      reach_observation
    ).detach()
    rows = torch.arange(self.num_envs, device=self.device)
    active_arm = torch_f.one_hot(command.active_hand, num_classes=2).bool()
    active_arm_action = active_arm.unsqueeze(-1).expand(-1, -1, 7).reshape(
      self.num_envs, reach.ACTION_SIZE
    )
    # The accepted v15 actor is immutable and remains the reach command source
    # everywhere except the typed contact pocket.  Its training distribution
    # stops at a collision-free pre-grasp shell, so a deterministic DLS Harness
    # executor owns the final target-space approach.  This mask cannot be
    # influenced by the 8D hand action and grants no learner, waist, locomotion,
    # inactive-arm, or checkpoint authority.
    self._contact_approach_correction_active[:] = (
      command.contact_retreat_active
      | command.contact_alignment_active
      | command.contact_pocket_active
    )
    correction_mask = (
      self._contact_approach_correction_active.unsqueeze(-1)
      & active_arm_action
    )
    self._raw_action[:] = torch.where(
      correction_mask,
      self._reach_teacher_action,
      self._frozen_reach_action,
    )
    self._contact_approach_correction_delta[:] = torch.where(
      correction_mask,
      self._raw_action - self._frozen_reach_action,
      0.0,
    )
    self._teacher_action[:] = self.teacher.infer(self._teacher_observation())
    self._teacher_body_targets[:] = (
      self.teacher.default_joint_positions
      + self._teacher_action * self.teacher.action_scale
    )
    self._upper_body_residual[:] = self._raw_action * self.cfg.upper_body_scale
    self._body_targets[:] = self.teacher.default_joint_positions
    self._body_targets[:, :15] = self._teacher_body_targets[:, :15]
    proposed_upper_targets = (
      self._body_targets[:, reach.UPPER_BODY_SLICE]
      + self._upper_body_residual
    )

    active_thumb_position = self._entity.data.body_link_pos_w[
      rows, self._thumb_body_ids[command.active_hand]
    ]
    thumb_to_rod = torch.linalg.vector_norm(
      active_thumb_position - command.rod.data.root_link_pos_w, dim=-1
    )
    protected_approach = ~(
      command.contact_alignment_active | command.contact_pocket_active
      | command.contact_retreat_active
    )
    thumb_safety_margin = torch.where(
      command.teacher_stage < CONTACT_STAGE_INDEX,
      torch.full_like(thumb_to_rod, self.cfg.precontact_thumb_safety_margin_m),
      torch.full_like(thumb_to_rod, self.cfg.preshape_thumb_safety_margin_m),
    )
    self._approach_safety_intervention[:] = (
      protected_approach
      & (thumb_to_rod < thumb_safety_margin)
    )
    self._body_targets[:, reach.UPPER_BODY_SLICE] = torch.where(
      self._approach_safety_intervention.unsqueeze(-1),
      self._safe_precontact_upper_targets,
      proposed_upper_targets,
    )
    safe_update = protected_approach & ~self._approach_safety_intervention
    self._safe_precontact_upper_targets[:] = torch.where(
      safe_update.unsqueeze(-1),
      proposed_upper_targets,
      self._safe_precontact_upper_targets,
    )
    found, force, _, _ = base._hand_contact_summary(self._env)
    active_found = found[rows, command.active_hand]
    active_force = force[rows, command.active_hand]
    wrist_delta = reach.active_wrist_position_delta(self._env)
    contact_supported_planar_tolerance = torch.where(
      command.active_hand == 0,
      torch.full_like(
        active_force, self.cfg.preclosure_pose_hold_planar_tolerance_m
      ),
      torch.full_like(
        active_force, self.cfg.preclosure_pose_hold_right_planar_tolerance_m
      ),
    )
    contact_supported_geometry = (
      torch.linalg.vector_norm(wrist_delta[:, :2], dim=-1)
      <= contact_supported_planar_tolerance
    ) & (
      wrist_delta[:, 2].abs()
      <= self.cfg.preclosure_pose_hold_vertical_tolerance_m
    )
    contact_supported_closure = (
      command.contact_pocket_active
      & active_found
      & (active_force >= self.cfg.preclosure_pose_hold_force_n)
      & contact_supported_geometry
    )
    # Guarded terminal motion: if the open hand meets the authorized object
    # before closure geometry is valid, command the most recent contact-free
    # arm target instead of integrating farther into the obstacle.  Once the
    # contact clears, the slow pose controller can retry with updated bearing
    # and axis feedback.  The buffer is one control update old, so this is a
    # bounded release rather than a jump back to the alignment shell.
    contact_free_update = self._contact_approach_correction_active & ~active_found
    current_upper_targets = self._body_targets[:, reach.UPPER_BODY_SLICE]
    self._last_contact_free_upper_targets[:] = torch.where(
      contact_free_update.unsqueeze(-1),
      current_upper_targets,
      self._last_contact_free_upper_targets,
    )
    guarded_early_contact = (
      command.contact_pocket_active
      & active_found
      & (active_force >= self.cfg.preclosure_pose_hold_force_n)
      & ~contact_supported_geometry
    )
    new_guarded_contact = guarded_early_contact & ~self._guarded_contact_active
    guarded_release_target = (
      self._last_contact_free_upper_targets
      - 4.0 * (
        current_upper_targets - self._last_contact_free_upper_targets
      )
    )
    self._guarded_contact_release_upper_targets[:] = torch.where(
      new_guarded_contact.unsqueeze(-1),
      guarded_release_target,
      self._guarded_contact_release_upper_targets,
    )
    self._guarded_contact_active[:] = (
      (self._guarded_contact_active | guarded_early_contact)
      & command.contact_pocket_active
      & active_found
      & ~contact_supported_geometry
    )
    self._body_targets[:, reach.UPPER_BODY_SLICE] = torch.where(
      self._guarded_contact_active.unsqueeze(-1),
      self._guarded_contact_release_upper_targets,
      current_upper_targets,
    )
    self.reach_teacher.rewind_active_arm_target(
      self._guarded_contact_active,
      self._guarded_contact_release_upper_targets,
    )
    # The alignment latch has already validated bearing and rod-axis geometry.
    # Once the open hand meets the rod inside the typed pocket, contact is a
    # better terminal constraint than an exact Cartesian point.  Capture the
    # measured arm immediately so the position servo cannot turn a gentle
    # first touch into a one-frame force spike, then let the 8D hand policy
    # establish opposing contact around that physically grounded pose.
    closure_ready = hand_closure_ready(self._env) | contact_supported_closure
    new_closure_pose_hold = (
      closure_ready & ~self._closure_pose_hold_active
    )
    measured_upper_positions = self._entity.data.joint_pos[
      :, self._body_ids[reach.UPPER_BODY_SLICE]
    ]
    self._closure_pose_hold_upper_targets[:] = torch.where(
      new_closure_pose_hold.unsqueeze(-1),
      measured_upper_positions,
      self._closure_pose_hold_upper_targets,
    )
    self._closure_pose_hold_active |= closure_ready
    active_coordination = self.coordination.reshape(
      self.num_envs, 2, 4
    )[rows, command.active_hand].abs().amax(dim=-1)
    new_contact_pose_hold = (
      command.contact_pocket_active
      & active_found
      & (active_force >= 2.0)
      & (
        self._closure_pose_hold_active
        | (active_coordination > self.cfg.hand_synergy_step)
      )
    )
    self._contact_pose_hold_upper_targets[:] = torch.where(
      new_contact_pose_hold.unsqueeze(-1),
      measured_upper_positions,
      self._contact_pose_hold_upper_targets,
    )
    self._contact_pose_hold_active |= new_contact_pose_hold
    selected_upper_targets = self._body_targets[:, reach.UPPER_BODY_SLICE].clone()
    selected_upper_targets = torch.where(
      self._closure_pose_hold_active.unsqueeze(-1),
      self._closure_pose_hold_upper_targets,
      selected_upper_targets,
    )
    self._body_targets[:, reach.UPPER_BODY_SLICE] = torch.where(
      self._contact_pose_hold_active.unsqueeze(-1),
      self._contact_pose_hold_upper_targets,
      selected_upper_targets,
    )
    contact_pose_update = (
      self._contact_approach_correction_active
      & ~self._contact_pose_hold_active
    )
    self._contact_pose_hold_upper_targets[:] = torch.where(
      contact_pose_update.unsqueeze(-1),
      selected_upper_targets,
      self._contact_pose_hold_upper_targets,
    )

    active_side = torch_f.one_hot(command.active_hand, num_classes=2).bool()
    active_synergy = active_side.unsqueeze(-1).expand(-1, -1, 4).reshape(
      self.num_envs, HAND_ACTION_SIZE
    )
    contact_capability = command.teacher_stage >= CONTACT_STAGE_INDEX
    capability_mask = active_synergy & contact_capability.unsqueeze(-1)
    self._authority_mask[:] = self.hand_teacher_authority_mask()
    self._requested_hand_action[:] = actions.clamp(-1.0, 1.0)
    self._hand_action[:] = torch.where(
      self._authority_mask,
      self._requested_hand_action,
      torch.zeros_like(self._requested_hand_action),
    )
    next_coordination = torch.clamp(
      self.coordination + self._hand_action * self.cfg.hand_synergy_step,
      0.0,
      1.0,
    )
    retained_or_updated = torch.where(
      self._authority_mask, next_coordination, self.coordination
    )
    self.coordination[:] = torch.where(
      capability_mask, retained_or_updated, torch.zeros_like(next_coordination)
    )
    requested_hand_targets = (
      self.coordination[:, self._hand_synergy_indexes] * self._hand_endpoint
    )
    measured_hand_positions = self._entity.data.joint_pos[:, self._hand_ids]
    endpoint_sign = torch.sign(self._hand_endpoint)
    endpoint_magnitude = self._hand_endpoint.abs()
    measured_closing_progress = measured_hand_positions * endpoint_sign
    requested_closing_progress = requested_hand_targets * endpoint_sign
    guarded_closing_progress = torch.minimum(
      requested_closing_progress,
      measured_closing_progress + self.cfg.hand_max_closing_joint_lead_rad,
    ).clamp(min=0.0)
    guarded_closing_progress = torch.minimum(
      guarded_closing_progress, endpoint_magnitude
    )
    self._hand_targets[:] = guarded_closing_progress * endpoint_sign

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    super().reset(env_ids)
    if env_ids is None:
      env_ids = slice(None)
    self._hand_action[env_ids] = 0.0
    self._requested_hand_action[env_ids] = 0.0
    self._authority_mask[env_ids] = False
    self._frozen_reach_action[env_ids] = 0.0
    self._contact_approach_correction_active[env_ids] = False
    self._contact_approach_correction_delta[env_ids] = 0.0
    self._teacher_thumb_contact_latched[env_ids] = False
    self._teacher_opposing_contact_latched[env_ids] = False
    self._contact_pose_hold_active[env_ids] = False
    self._closure_pose_hold_active[env_ids] = False
    self._closure_pose_hold_upper_targets[env_ids] = self._body_targets[
      env_ids, reach.UPPER_BODY_SLICE
    ]
    self._contact_pose_hold_upper_targets[env_ids] = self._body_targets[
      env_ids, reach.UPPER_BODY_SLICE
    ]
    self._approach_safety_intervention[env_ids] = False
    self._safe_precontact_upper_targets[env_ids] = self._body_targets[
      env_ids, reach.UPPER_BODY_SLICE
    ]
    self._last_contact_free_upper_targets[env_ids] = self._body_targets[
      env_ids, reach.UPPER_BODY_SLICE
    ]
    self._guarded_contact_release_upper_targets[env_ids] = self._body_targets[
      env_ids, reach.UPPER_BODY_SLICE
    ]
    self._guarded_contact_active[env_ids] = False


def _contact_action(env: ManagerBasedRlEnv) -> WorkyardContactAction:
  action = env.action_manager.get_term("workyard")
  if not isinstance(action, WorkyardContactAction):
    raise TypeError("Workyard contact action term is unavailable")
  return action


class WorkyardContactObservation(reach.WorkyardResidualObservation):
  def __call__(self, env: ManagerBasedRlEnv) -> torch.Tensor:
    reach_observation = super().__call__(env)
    action = _contact_action(env)
    observation = torch.cat((
      reach_observation,
      action.coordination,
      action.hand_action,
    ), dim=-1)
    if observation.shape != (env.num_envs, HAND_OBSERVATION_SIZE):
      raise RuntimeError(f"Contact observation drifted: {observation.shape}")
    finite = torch.isfinite(observation)
    if not bool(finite.all().item()) and not hasattr(
      env, "numerical_recovery_count"
    ):
      indexes = (~finite).nonzero(as_tuple=False)[:32]
      diagnostic = {
        "protocol": "hear-contact-observation-nonfinite-v1",
        "common_step": int(env.common_step_counter),
        "entries": [
          {
            "environment": int(environment),
            "feature": int(feature),
            "term": (
              "frozen_reach_observation" if feature < REACH_OBSERVATION_SIZE
              else "hand_coordination" if feature < REACH_OBSERVATION_SIZE + 8
              else "previous_authorized_hand_action"
            ),
          }
          for environment, feature in indexes.cpu().tolist()
        ],
        "physics_nonfinite_environments": torch.where(
          _nonfinite_physics_state(env)
        )[0][:32].cpu().tolist(),
        "maximum_finite_qvel": _maximum_finite_absolute(env.sim.data.qvel),
        "maximum_finite_qacc": _maximum_finite_absolute(env.sim.data.qacc),
      }
      raise RuntimeError(
        "Contact observation became non-finite: "
        + json.dumps(diagnostic, ensure_ascii=False, sort_keys=True)
      )
    return observation


def _maximum_finite_absolute(value: torch.Tensor) -> float:
  finite = value[torch.isfinite(value)]
  return float(finite.abs().max().item()) if finite.numel() > 0 else float("inf")


def _nonfinite_physics_state(env: ManagerBasedRlEnv) -> torch.Tensor:
  data = env.sim.data
  invalid = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
  for value in (
    data.qpos,
    data.qvel,
    data.qacc,
    data.qacc_warmstart,
    data.sensordata,
  ):
    invalid |= ~torch.isfinite(value).all(dim=-1)
  return invalid


def numerical_instability(env: ManagerBasedRlEnv) -> torch.Tensor:
  """Terminate an exploding vector world before it contaminates the learner."""
  data = env.sim.data
  return (
    _nonfinite_physics_state(env)
    | (data.qvel.abs().amax(dim=-1) > MAXIMUM_NUMERICALLY_STABLE_QVEL)
    | (data.qacc.abs().amax(dim=-1) > MAXIMUM_NUMERICALLY_STABLE_QACC)
  )


def hand_teacher_action_tracking(env: ManagerBasedRlEnv) -> torch.Tensor:
  action = _contact_action(env)
  teacher = action.compute_hand_teacher_action()
  error = (action.hand_action - teacher).square().mean(dim=-1)
  return torch.exp(-4.0 * error)


def contact_force_band(env: ManagerBasedRlEnv) -> torch.Tensor:
  command = base._workyard_command(env)
  found, force, surfaces, _ = base._hand_contact_summary(env)
  rows = torch.arange(env.num_envs, device=env.device)
  active_force = force[rows, command.active_hand]
  active_surfaces = surfaces[rows, command.active_hand]
  active_found = found[rows, command.active_hand]
  lower = torch.clamp(active_force / 4.0, 0.0, 1.0)
  upper = torch.exp(-0.02 * torch.relu(active_force - 12.0).square())
  surface_score = torch.clamp(active_surfaces / 2.0, 0.0, 1.0)
  authorized = command.teacher_stage >= CONTACT_STAGE_INDEX
  return lower * upper * surface_score * active_found.float() * authorized.float()


def opposing_surface_reward(env: ManagerBasedRlEnv) -> torch.Tensor:
  command = base._workyard_command(env)
  _, _, _, opposed = base._hand_contact_summary(env)
  rows = torch.arange(env.num_envs, device=env.device)
  authorized = command.teacher_stage >= CONTACT_STAGE_INDEX
  return opposed[rows, command.active_hand].float() * authorized.float()


def excessive_contact_force(env: ManagerBasedRlEnv) -> torch.Tensor:
  command = base._workyard_command(env)
  _, force, _, _ = base._hand_contact_summary(env)
  rows = torch.arange(env.num_envs, device=env.device)
  active_force = force[rows, command.active_hand]
  return (torch.relu(active_force - 18.0) / 18.0).square()


def object_motion_cost(env: ManagerBasedRlEnv) -> torch.Tensor:
  command = base._workyard_command(env)
  velocity = command.rod.data.root_link_vel_w
  return velocity[:, :3].square().sum(dim=-1) + 0.1 * velocity[:, 3:].square().sum(dim=-1)


def inactive_hand_motion_cost(env: ManagerBasedRlEnv) -> torch.Tensor:
  action = _contact_action(env)
  command = base._workyard_command(env)
  coordination = action.coordination.reshape(env.num_envs, 2, 4)
  inactive = 1 - command.active_hand
  rows = torch.arange(env.num_envs, device=env.device)
  return coordination[rows, inactive].square().mean(dim=-1)


def object_lost_before_grasp(env: ManagerBasedRlEnv) -> torch.Tensor:
  command = base._workyard_command(env)
  if not isinstance(command, WorkyardContactCommand):
    raise TypeError("Contact command is unavailable for object-loss termination")
  position = command.rod.data.root_link_pos_w
  horizontal = torch.linalg.vector_norm(
    position[:, :2] - command.initial_rod_position[:, :2], dim=-1
  )
  fell_below_stand = position[:, 2] < base.SOURCE_POSITION[2] + 0.025
  escaped_workspace = horizontal > 0.22
  return (fell_below_stand | escaped_workspace) & ~command.completed_target


def make_workyard_contact_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  cfg = reach.make_workyard_residual_env_cfg(play=play)
  robot_cfg = cfg.scene.entities["robot"]
  robot_cfg.spec_fn = _load_contact_actuated_g1_spec
  cfg.scene.entities["pickup_stand"].spec_fn = _make_contact_source_spec
  cfg.scene.num_envs = 1 if play else 2048
  cfg.episode_length_s = 8.0
  if not play:
    cfg.sim.nan_guard.enabled = True
    cfg.sim.nan_guard.buffer_size = 16
    cfg.sim.nan_guard.max_envs_to_dump = 3
    cfg.sim.nan_guard.output_dir = os.environ.get(
      "HEAR_WORKYARD_CONTACT_NAN_DUMP_DIR",
      "/tmp/hear-workyard-contact-nan-dumps",
    )
  cfg.actions = {
    "workyard": WorkyardContactActionCfg(
      entity_name="robot",
      upper_body_scale=0.5,
      hand_synergy_step=0.0075,
      hand_max_closing_joint_lead_rad=0.25,
      contact_force_stop_n=6.0,
      contact_force_emergency_release_n=12.0,
    )
  }
  cfg.commands = {
    "workyard": WorkyardContactCommandCfg(
      resampling_time_range=(20.0, 20.0),
      debug_vis=False,
      object_position_jitter_m=0.03,
      target_position_jitter_m=0.0,
      contact_base_assist_enabled=False,
      pregrasp_shell_radius_m=0.10,
      pregrasp_lateral_clearance_m=0.16,
      wrist_frame_safe_pregrasp=False,
      # Terminal 4D DLS aligns the measured wrist bearing before entering the
      # palm/finger pocket.  This remains deterministic Harness authority; the
      # learned 8D hand action cannot alter these pose coordinates.
      # Real MuJoCo mesh scans place the cylindrical rod in a strong, shallow
      # opposing-contact basin around 9.4/2.4 cm.  The approach retains the
      # proven collision-free lateral bearing; low-force first contact inside
      # this pocket becomes the terminal constraint and freezes the measured
      # arm before closure.  A centred approach is invalid because the open
      # index/middle links occupy that ray before the typed closure latch.
      contact_pocket_forward_m=0.094,
      contact_pocket_lateral_m=0.024,
      contact_pocket_vertical_m=0.0,
      contact_alignment_radius_m=0.150,
      contact_alignment_tolerance_m=0.030,
      contact_retreat_tolerance_m=0.010,
      contact_alignment_bearing_tolerance_rad=0.18,
      # The cylinder is rotationally symmetric and the final pocket is defined
      # in the measured wrist frame.  A 0.56 rad axis gate stranded the
      # mirrored left wrist at its bounded joint authority even after position
      # and bearing converged.  0.85 rad still rejects a transverse hand while
      # admitting the full measured left/right terminal distribution.
      contact_alignment_axis_tolerance_rad=0.85,
      contact_alignment_max_force_n=0.5,
      contact_preshape_ready_coordination=0.0,
      contact_closure_tolerance_m=0.035,
      contact_closure_vertical_tolerance_m=0.040,
      contact_base_assist_gain_s_inv=1.5,
      contact_base_assist_min_speed_m_s=0.12,
      contact_base_assist_max_speed_m_s=0.12,
    )
  }
  cfg.observations = {
    "actor": ObservationGroupCfg(
      terms={
        "workyard": ObservationTermCfg(
          func=WorkyardContactObservation,
          delay_min_lag=0,
          delay_max_lag=0 if play else 1,
          clip=(-20.0, 20.0),
        )
      },
      concatenate_terms=True,
      enable_corruption=False,
    ),
    "critic": ObservationGroupCfg(
      terms={"workyard": ObservationTermCfg(func=WorkyardContactObservation)},
      concatenate_terms=True,
      enable_corruption=False,
    ),
  }
  cfg.rewards = {
    "upright_support": RewardTermCfg(func=base.upright_support, weight=1.0),
    "teacher_lower_body_tracking": RewardTermCfg(
      func=reach.teacher_lower_body_tracking, weight=2.0
    ),
    "teacher_waist_tracking": RewardTermCfg(
      func=reach.teacher_waist_tracking, weight=0.5
    ),
    "wrist_position_tracking": RewardTermCfg(
      func=reach.wrist_position_tracking, weight=4.0
    ),
    "wrist_distance_progress": RewardTermCfg(
      func=reach.WristDistanceProgress, weight=20.0
    ),
    "qualified_hand_contact": RewardTermCfg(
      func=base.qualified_hand_contact, weight=2.0
    ),
    "opposing_contact": RewardTermCfg(func=opposing_surface_reward, weight=4.0),
    "grasp_verified": RewardTermCfg(func=base.grasp_verified, weight=12.0),
    "contact_force_band": RewardTermCfg(func=contact_force_band, weight=3.0),
    "hand_teacher_action_tracking": RewardTermCfg(
      func=hand_teacher_action_tracking, weight=0.5
    ),
    "dynamic_com_support": RewardTermCfg(func=reach.dynamic_com_support, weight=1.0),
    "object_motion": RewardTermCfg(func=object_motion_cost, weight=-0.05),
    "excessive_contact_force": RewardTermCfg(
      func=excessive_contact_force, weight=-1.0
    ),
    "inactive_hand_motion": RewardTermCfg(
      func=inactive_hand_motion_cost, weight=-2.0
    ),
    "action_rate": RewardTermCfg(func=base.mdp.action_rate_l2, weight=-0.02),
    "actuator_saturation": RewardTermCfg(
      func=base.JointActuatorSaturationCost, weight=-0.5
    ),
    "fall": RewardTermCfg(func=base.fall_cost, weight=-25.0),
    "non_foot_collision": RewardTermCfg(
      func=base.non_foot_collision_cost, weight=-2.0
    ),
    "foot_slip": RewardTermCfg(func=base.foot_slip_cost, weight=-0.2),
  }
  cfg.terminations = {
    "time_out": TerminationTermCfg(func=base.mdp.time_out, time_out=True),
    "fall": TerminationTermCfg(func=base.fell_over),
    "non_foot_ground": TerminationTermCfg(func=base.illegal_ground_contact),
    "numerical_instability": TerminationTermCfg(func=numerical_instability),
    "grasp_success": TerminationTermCfg(func=base.stage_completed),
    "object_lost": TerminationTermCfg(func=object_lost_before_grasp),
  }
  cfg.curriculum = {}
  return cfg


def workyard_contact_ppo_runner_cfg():
  cfg = reach.workyard_residual_ppo_runner_cfg()
  cfg.experiment_name = "hear_workyard_contact_g1"
  cfg.max_iterations = 3_000
  return cfg


def register_workyard_contact_task() -> None:
  if TASK_ID in list_tasks():
    return
  register_mjlab_task(
    task_id=TASK_ID,
    env_cfg=make_workyard_contact_env_cfg(play=False),
    play_env_cfg=make_workyard_contact_env_cfg(play=True),
    rl_cfg=workyard_contact_ppo_runner_cfg(),
  )


register_workyard_contact_task()
