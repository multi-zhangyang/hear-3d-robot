from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import inspect
import json
import os
import shutil
import sys
import tarfile
import traceback
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from types import ModuleType


MJLAB_VERSION = "1.5.3"
EXECUTION_ROOT = Path(os.environ.get("HEAR_WORKYARD_EXECUTION_ROOT", "/content"))
REMOTE_BUNDLE = Path(os.environ.get(
  "HEAR_WORKYARD_BUNDLE",
  str(EXECUTION_ROOT / "hear-workyard-residual-bundle.tar.gz"),
))
REMOTE_CONFIG = Path(os.environ.get(
  "HEAR_WORKYARD_CONFIG",
  str(EXECUTION_ROOT / "hear-workyard-residual-config.json"),
))
REMOTE_ROOT = Path(os.environ.get(
  "HEAR_WORKYARD_SOURCE_ROOT",
  str(EXECUTION_ROOT / "hear-workyard-residual-source"),
))
REMOTE_REPORT = Path(os.environ.get(
  "HEAR_WORKYARD_REPORT",
  str(EXECUTION_ROOT / "hear-workyard-residual-report.json"),
))


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser()
  parser.add_argument(
    "--mode", choices=("smoke", "teacher", "train"), default="smoke"
  )
  parser.add_argument("--iterations", type=int, default=20)
  parser.add_argument("--dagger-steps", type=int, default=400)
  parser.add_argument("--dagger-learning-rate", type=float, default=3.0e-4)
  parser.add_argument("--dagger-beta-initial", type=float, default=1.0)
  parser.add_argument("--dagger-beta-final", type=float, default=0.1)
  parser.add_argument(
    "--ppo-retention-mode",
    choices=("baseline", "critic_warmup_rollout_teacher"),
    default="critic_warmup_rollout_teacher",
  )
  parser.add_argument("--critic-warmup-iterations", type=int, default=5)
  parser.add_argument("--ppo-actor-learning-rate", type=float, default=1.0e-5)
  parser.add_argument("--ppo-critic-learning-rate", type=float, default=3.0e-4)
  parser.add_argument("--rollout-teacher-coefficient", type=float, default=1.0)
  parser.add_argument("--teacher-maximum-action-std", type=float, default=0.15)
  parser.add_argument("--teacher-dispersion-coefficient", type=float, default=1.0)
  parser.add_argument("--num-envs", type=int, default=64)
  parser.add_argument("--rollout-steps", type=int, default=64)
  parser.add_argument("--eval-envs", type=int, default=64)
  parser.add_argument("--eval-steps", type=int, default=600)
  parser.add_argument("--seed", type=int, default=42)
  parser.add_argument("--execution-profile", default="colab-pro-l4-formal-v1")
  parser.add_argument("--output-root", default="/content/hear-workyard-residual")
  parser.add_argument(
    "--archive", default="/content/hear-workyard-residual-artifacts.tar.gz"
  )
  if REMOTE_CONFIG.is_file():
    configured = json.loads(REMOTE_CONFIG.read_text(encoding="utf-8"))
    valid_destinations = {action.dest for action in parser._actions}
    unknown = sorted(set(configured) - valid_destinations)
    if unknown:
      parser.error("unknown remote configuration keys: " + ", ".join(unknown))
    parser.set_defaults(**configured)
  args, unknown = parser.parse_known_args()
  if unknown and not (len(unknown) == 2 and unknown[0] == "-f"):
    parser.error("unrecognized arguments: " + " ".join(unknown))
  for name in (
    "iterations", "dagger_steps", "num_envs", "rollout_steps", "eval_envs",
    "eval_steps"
  ):
    if getattr(args, name) <= 0:
      parser.error(f"--{name.replace('_', '-')} must be positive")
  if args.seed < 0:
    parser.error("--seed must be non-negative")
  if args.dagger_learning_rate <= 0.0:
    parser.error("--dagger-learning-rate must be positive")
  for name in (
    "ppo_actor_learning_rate", "ppo_critic_learning_rate",
    "rollout_teacher_coefficient",
    "teacher_maximum_action_std", "teacher_dispersion_coefficient",
  ):
    if getattr(args, name) <= 0.0:
      parser.error(f"--{name.replace('_', '-')} must be positive")
  if args.critic_warmup_iterations < 0:
    parser.error("--critic-warmup-iterations must be non-negative")
  if args.teacher_maximum_action_std > 1.0:
    parser.error("--teacher-maximum-action-std must not exceed 1")
  if not (
    0.0 <= args.dagger_beta_final <= args.dagger_beta_initial <= 1.0
  ):
    parser.error("DAgger beta must satisfy 0 <= final <= initial <= 1")
  return args


def run(command: list[str]) -> None:
  import subprocess

  subprocess.run(command, check=True)


def sha256(path: Path) -> str:
  digest = hashlib.sha256()
  with path.open("rb") as handle:
    for chunk in iter(lambda: handle.read(1024 * 1024), b""):
      digest.update(chunk)
  return digest.hexdigest()


def extract_bundle() -> None:
  if not REMOTE_BUNDLE.is_file():
    raise FileNotFoundError(f"Residual Workyard bundle is missing: {REMOTE_BUNDLE}")
  REMOTE_ROOT.mkdir(parents=True, exist_ok=False)
  with tarfile.open(REMOTE_BUNDLE, "r:gz") as archive:
    root = REMOTE_ROOT.resolve()
    for member in archive.getmembers():
      target = (REMOTE_ROOT / member.name).resolve()
      if target != root and root not in target.parents:
        raise RuntimeError(f"Unsafe residual Workyard bundle member: {member.name}")
    archive.extractall(REMOTE_ROOT, filter="data")


def load_modules() -> tuple[ModuleType, ModuleType]:
  training_root = REMOTE_ROOT / "training"
  sys.path.insert(0, str(training_root))
  import workyard_mjlab_env as workyard
  import workyard_residual_mjlab_env as residual

  return workyard, residual


def tensorboard_curves(log_dir: Path) -> dict[str, list[dict[str, float | int]]]:
  from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

  event_directories = sorted({
    event.parent for event in log_dir.rglob("events.out.tfevents.*")
  })
  curves: dict[str, list[dict[str, float | int]]] = {}
  for directory in event_directories:
    accumulator = EventAccumulator(str(directory), size_guidance={"scalars": 0})
    accumulator.Reload()
    for tag in accumulator.Tags().get("scalars", []):
      curves.setdefault(tag, []).extend(
        {"step": int(event.step), "value": float(event.value)}
        for event in accumulator.Scalars(tag)
      )
  for values in curves.values():
    values.sort(key=lambda value: int(value["step"]))
  if not curves:
    raise RuntimeError(
      "Residual PPO produced no readable TensorBoard scalar curves under "
      + str(log_dir)
    )
  return curves


def validate_contract(module: ModuleType) -> tuple[dict[str, object], Path, list[str]]:
  contract_path = REMOTE_ROOT / "training" / "workyard-task-v4.json"
  contract = json.loads(contract_path.read_text(encoding="utf-8"))
  if contract.get("protocol") != "hear-workyard-residual-training-contract-v4":
    raise RuntimeError("Residual Workyard contract protocol is invalid")
  if contract["environment"]["task_id"] != module.TASK_ID:
    raise RuntimeError("Residual task registration disagrees with its contract")
  if contract["observation"]["size"] != module.OBSERVATION_SIZE:
    raise RuntimeError("Residual observation dimension disagrees with its contract")
  if contract["action"]["size"] != module.ACTION_SIZE:
    raise RuntimeError("Residual action dimension disagrees with its contract")
  if contract["teacher"]["actuation"] != module.BODY_ACTUATION_CONTRACT:
    raise RuntimeError("Residual teacher actuation boundary is invalid")
  if tuple(contract["student"]["entry_state"]["root_position_world"]) != (
    module.RESIDUAL_ENTRY_ROOT_POSITION
  ):
    raise RuntimeError("Residual reach entry state disagrees with its contract")
  if contract["reach_teacher"] != module.REACH_TEACHER_CONTRACT:
    mismatched = sorted(
      key for key in set(contract["reach_teacher"]) | set(module.REACH_TEACHER_CONTRACT)
      if contract["reach_teacher"].get(key) != module.REACH_TEACHER_CONTRACT.get(key)
    )
    raise RuntimeError(
      "Adaptive reach teacher disagrees with its contract: " + ", ".join(mismatched)
    )
  if (
    contract["evaluation"]["dynamic_com_settling_control_steps"]
      != module.DYNAMIC_COM_SETTLING_STEPS
  ):
    raise RuntimeError("Dynamic-CoM settling window disagrees with its contract")
  if (
    contract["warm_start"]["actor_distribution"]
      != "beta_bounded_minus_one_one"
    or contract["ppo_retention"]["protocol"] != "hear-ppo-retention-v2"
    or contract["ppo_retention"]["default_teacher_loss_coefficient"] != 1.0
    or contract["ppo_retention"]["maximum_action_std"] != 0.15
  ):
    raise RuntimeError("PPO retention boundary disagrees with its contract")
  observation_source = inspect.getsource(module.WorkyardResidualObservation.__call__)
  forbidden = contract["student"]["forbidden_observations"]
  leaked = [name for name in forbidden if name in observation_source]
  if leaked:
    raise RuntimeError(
      "Teacher-only state leaked into the residual actor observation: "
      + ", ".join(leaked)
    )
  inference_source = inspect.getsource(module.FrozenLocomotionTeacher.infer)
  reach_source = inspect.getsource(module.BatchedTaskSpaceReachTeacher.infer)
  process_source = inspect.getsource(module.WorkyardResidualAction.process_actions)
  cpu_round_trip_tokens = [".cpu(", ".numpy(", ".tolist("]
  cpu_round_trips = [
    token for token in cpu_round_trip_tokens
    if token in inference_source or token in reach_source or token in process_source
  ]
  if cpu_round_trips:
    raise RuntimeError(
      "Teacher control loop contains a CPU round trip: " + ", ".join(cpu_round_trips)
    )
  return contract, contract_path, leaked


def rollout(
  module: ModuleType,
  num_envs: int,
  steps: int,
  seed: int,
  policy=None,
  use_reach_teacher: bool = False,
  checkpoint: Path | None = None,
) -> dict[str, object]:
  import torch
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
  from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls

  if policy is not None and checkpoint is not None:
    raise ValueError("Residual rollout accepts either a policy or a checkpoint")

  env_cfg = load_env_cfg(module.TASK_ID, play=True)
  env_cfg.scene.num_envs = num_envs
  env_cfg.seed = seed
  agent_cfg = load_rl_cfg(module.TASK_ID)
  raw_env = ManagerBasedRlEnv(cfg=env_cfg, device="cuda:0")
  env = RslRlVecEnvWrapper(raw_env, clip_actions=agent_cfg.clip_actions)
  environment_closed = False
  try:
    if checkpoint is not None:
      runner_cls = load_runner_cls(module.TASK_ID) or MjlabOnPolicyRunner
      evaluation_runner = runner_cls(env, asdict(agent_cfg), device="cuda:0")
      evaluation_runner.load(
        str(checkpoint),
        load_cfg={"actor": True},
        strict=True,
        map_location="cuda:0",
      )
      policy = evaluation_runner.get_inference_policy(device="cuda:0")
    observations = env.get_observations()
    if tuple(observations["actor"].shape) != (num_envs, module.OBSERVATION_SIZE):
      raise RuntimeError(
        f"Residual actor observation has shape {tuple(observations['actor'].shape)}"
      )
    action_term = module._residual_action(raw_env)
    robot = raw_env.scene["robot"]
    reach_contact_sensors = {
      name: module.base._contact_sensor(raw_env, name)
      for name in module.REACH_CONTACT_DIAGNOSTIC_SENSOR_NAMES
    }
    reach_contact_maximum_force_by_primary = {
      name: torch.zeros(
        len(sensor.primary_names), dtype=torch.float32, device="cuda:0"
      )
      for name, sensor in reach_contact_sensors.items()
    }
    reach_contact_found_count_by_primary = {
      name: torch.zeros(
        len(sensor.primary_names), dtype=torch.long, device="cuda:0"
      )
      for name, sensor in reach_contact_sensors.items()
    }
    action = torch.zeros(
      (num_envs, module.ACTION_SIZE), dtype=torch.float32, device="cuda:0"
    )
    reward_sum = torch.zeros(num_envs, dtype=torch.float32, device="cuda:0")
    active = torch.ones(num_envs, dtype=torch.bool, device="cuda:0")
    success = torch.zeros(num_envs, dtype=torch.bool, device="cuda:0")
    termination_count = 0
    fall_count = 0
    non_foot_collision_count = 0
    maximum_action = 0.0
    action_element_count = 0
    clipped_action_element_count = 0
    action_frame_count = 0
    clipped_action_frame_count = 0
    maximum_teacher_action = 0.0
    maximum_observation = float(observations["actor"].abs().max().item())
    maximum_joint_velocity = 0.0
    maximum_frozen_joint_velocity = 0.0
    maximum_frozen_joint_velocity_after_settling = 0.0
    maximum_upper_body_joint_velocity = 0.0
    maximum_hand_joint_velocity = 0.0
    maximum_frozen_velocity_by_joint = torch.zeros(15, device="cuda:0")
    maximum_upper_velocity_by_joint = torch.zeros(14, device="cuda:0")
    maximum_hand_velocity_by_joint = torch.zeros(14, device="cuda:0")
    maximum_teacher_action_by_joint = torch.zeros(29, device="cuda:0")
    maximum_frozen_target_delta_by_joint = torch.zeros(15, device="cuda:0")
    frozen_velocity_trace: list[float] = []
    peak_frozen_velocity = {
      "value_rad_s": 0.0,
      "control_step": 0,
      "environment_index": 0,
      "joint_index": 0,
      "joint_name": module.FROZEN_TEACHER_JOINT_NAMES[0],
    }
    maximum_force_ratio = 0.0
    maximum_body_force_ratio_by_joint = torch.zeros(29, device="cuda:0")
    maximum_body_constraint_force_by_joint = torch.zeros(29, device="cuda:0")
    maximum_body_bias_force_by_joint = torch.zeros(29, device="cuda:0")
    upper_tracking_squared_error_sum_by_joint = torch.zeros(
      14, dtype=torch.float64, device="cuda:0"
    )
    maximum_upper_tracking_error_by_joint = torch.zeros(14, device="cuda:0")
    upper_tracking_sample_count = 0
    minimum_root_height = float("inf")
    maximum_frozen_composition_error = 0.0
    maximum_upper_composition_error = 0.0
    frozen_tracking_squared_error_sum = 0.0
    frozen_tracking_sample_count = 0
    teacher_frame_count = 0
    residual_frame_count = 0
    reach_teacher_label_count = 0
    reach_teacher_label_total = 0
    previous_frozen_targets = action_term.body_targets[:, :15].clone()
    command = module.base._workyard_command(raw_env)
    initial_active_hand = command.active_hand.clone()
    initial_wrist_error = module.active_wrist_position_error(raw_env).clone()
    initial_wrist_delta = module.active_wrist_position_delta(raw_env).clone()
    minimum_wrist_error = initial_wrist_error.clone()
    minimum_wrist_error_step = torch.zeros(
      num_envs, dtype=torch.long, device="cuda:0"
    )
    minimum_wrist_delta = initial_wrist_delta.clone()
    final_wrist_error = initial_wrist_error.clone()
    final_wrist_delta = initial_wrist_delta.clone()
    wrist_error_trace: list[dict[str, float | int | None]] = []

    def append_wrist_error_trace(
      control_step: int,
      wrist_error,
      first_episode_active,
    ) -> None:
      active_error = wrist_error[first_episode_active]
      point: dict[str, float | int | None] = {
        "control_step": control_step,
        "active_first_episode_environment_count": int(active_error.numel()),
        "mean": None,
        "p50": None,
        "p90": None,
      }
      if active_error.numel():
        quantiles = torch.quantile(active_error, torch.tensor(
          (0.50, 0.90), dtype=active_error.dtype, device=active_error.device
        ))
        point.update({
          "mean": float(active_error.mean().item()),
          "p50": float(quantiles[0].item()),
          "p90": float(quantiles[1].item()),
        })
      wrist_error_trace.append(point)

    append_wrist_error_trace(0, initial_wrist_error, active)
    reach_teacher_active_joint_samples = torch.zeros(
      module.ACTION_SIZE, dtype=torch.long, device="cuda:0"
    )
    reach_teacher_authority_saturation_count = torch.zeros_like(
      reach_teacher_active_joint_samples
    )
    reach_teacher_soft_limit_saturation_count = torch.zeros_like(
      reach_teacher_active_joint_samples
    )
    reach_teacher_command_lead_saturation_count = torch.zeros_like(
      reach_teacher_active_joint_samples
    )
    reach_teacher_absolute_action_sum = torch.zeros(
      module.ACTION_SIZE, dtype=torch.float64, device="cuda:0"
    )
    reach_teacher_maximum_unclamped_action_by_joint = torch.zeros(
      module.ACTION_SIZE, device="cuda:0"
    )
    reach_teacher_maximum_joint_target_step_by_joint = torch.zeros(
      module.ACTION_SIZE, device="cuda:0"
    )
    reach_teacher_maximum_solver_target_slew_by_joint = torch.zeros(
      module.ACTION_SIZE, device="cuda:0"
    )
    maximum_absolute_unclamped_reach_teacher_action = 0.0
    reach_teacher_active_arm_samples = 0
    reach_teacher_minimum_singular_value = float("inf")
    reach_teacher_singular_value_sum = 0.0
    reach_teacher_singularity_threshold_count = 0
    reach_teacher_holding_target_count = 0
    feasible_posture_feature_sample_count = 0
    feasible_posture_feature_clamp_count = 0
    feasible_posture_action_sample_count = 0
    feasible_posture_action_clamp_count = 0
    joint_ids = robot.indexing.joint_ids
    initial_dynamic_com = action_term.dynamic_com.compute()
    initial_foot_position = initial_dynamic_com.foot_position_w.clone()
    settled_foot_position = None
    minimum_support_margin = float("inf")
    maximum_capture_point_norm = 0.0
    maximum_foot_planar_displacement = 0.0
    maximum_foot_slip_speed = 0.0
    double_support_loss_count = 0
    no_foot_contact_count = 0
    dynamic_com_frame_count = 0
    settled_minimum_support_margin = float("inf")
    settled_maximum_capture_point_norm = 0.0
    settled_maximum_foot_planar_displacement = 0.0
    settled_maximum_foot_slip_speed = 0.0
    settled_double_support_loss_count = 0
    settled_no_foot_contact_count = 0
    settled_dynamic_com_frame_count = 0
    initial_dynamic_com_trace: list[dict[str, float | int]] = []
    maximum_hand_target_error = 0.0

    with torch.inference_mode():
      for control_step in range(steps):
        teacher_label_before_step = action_term.compute_reach_teacher_action()
        if use_reach_teacher:
          action = teacher_label_before_step
        else:
          action = policy(observations) if policy is not None else torch.zeros_like(action)
        if action.shape != (num_envs, module.ACTION_SIZE) or not torch.isfinite(action).all():
          raise RuntimeError("Residual policy emitted an invalid action")
        active_before_step = active.clone()
        active_hand_before_step = command.active_hand.clone()
        left_active = (active_hand_before_step == 0).unsqueeze(-1).expand(-1, 7)
        right_active = (active_hand_before_step == 1).unsqueeze(-1).expand(-1, 7)
        active_joint_mask = torch.cat((left_active, right_active), dim=-1)
        active_joint_mask &= active_before_step.unsqueeze(-1)
        reach_teacher_active_joint_samples += active_joint_mask.sum(dim=0)
        authority_saturation = action_term.reach_teacher.authority_saturation
        soft_limit_saturation = action_term.reach_teacher.soft_limit_saturation
        command_lead_saturation = action_term.reach_teacher.command_lead_saturation
        reach_teacher_authority_saturation_count += (
          authority_saturation & active_joint_mask
        ).sum(dim=0)
        reach_teacher_soft_limit_saturation_count += (
          soft_limit_saturation & active_joint_mask
        ).sum(dim=0)
        reach_teacher_command_lead_saturation_count += (
          command_lead_saturation & active_joint_mask
        ).sum(dim=0)
        reach_teacher_absolute_action_sum += torch.where(
          active_joint_mask,
          teacher_label_before_step.abs().to(torch.float64),
          0.0,
        ).sum(dim=0)
        active_unclamped_action = torch.where(
          active_joint_mask,
          action_term.reach_teacher.unclamped_action.abs(),
          0.0,
        )
        reach_teacher_maximum_unclamped_action_by_joint = torch.maximum(
          reach_teacher_maximum_unclamped_action_by_joint,
          active_unclamped_action.amax(dim=0),
        )
        active_joint_target_step = torch.where(
          active_joint_mask,
          action_term.reach_teacher.joint_target_step,
          0.0,
        )
        reach_teacher_maximum_joint_target_step_by_joint = torch.maximum(
          reach_teacher_maximum_joint_target_step_by_joint,
          active_joint_target_step.amax(dim=0),
        )
        active_solver_target_slew = torch.where(
          active_joint_mask,
          action_term.reach_teacher.solver_target_slew,
          0.0,
        )
        reach_teacher_maximum_solver_target_slew_by_joint = torch.maximum(
          reach_teacher_maximum_solver_target_slew_by_joint,
          active_solver_target_slew.amax(dim=0),
        )
        maximum_absolute_unclamped_reach_teacher_action = max(
          maximum_absolute_unclamped_reach_teacher_action,
          float(active_unclamped_action.max().item()),
        )
        active_arm_index = active_hand_before_step.unsqueeze(-1)
        active_singular_value = action_term.reach_teacher.minimum_singular_value.gather(
          1, active_arm_index
        ).squeeze(-1)
        active_holding_target = action_term.reach_teacher.holding_target.gather(
          1, active_arm_index
        ).squeeze(-1)
        active_posture_feature_clamped = (
          action_term.reach_teacher.feasible_posture_feature_clamped.gather(
            1, active_arm_index.unsqueeze(-1).expand(-1, 1, 2)
          ).squeeze(1)
        )
        active_posture_action_clamped = (
          action_term.reach_teacher.feasible_posture_action_clamped.gather(
            1, active_arm_index.unsqueeze(-1).expand(-1, 1, 7)
          ).squeeze(1)
        )
        active_arm_sample_count = int(active_before_step.sum().item())
        if active_arm_sample_count:
          active_values = active_singular_value[active_before_step]
          reach_teacher_active_arm_samples += active_arm_sample_count
          reach_teacher_minimum_singular_value = min(
            reach_teacher_minimum_singular_value,
            float(active_values.min().item()),
          )
          reach_teacher_singular_value_sum += float(active_values.sum().item())
          reach_teacher_singularity_threshold_count += int((
            active_values < action_term.reach_teacher.singularity_threshold
          ).sum().item())
          reach_teacher_holding_target_count += int(
            active_holding_target[active_before_step].sum().item()
          )
          feasible_posture_feature_sample_count += active_arm_sample_count * 2
          feasible_posture_feature_clamp_count += int(
            active_posture_feature_clamped[active_before_step].sum().item()
          )
          feasible_posture_action_sample_count += active_arm_sample_count * 7
          feasible_posture_action_clamp_count += int(
            active_posture_action_clamped[active_before_step].sum().item()
          )
        active_action = action[active_before_step]
        if active_action.numel():
          clipped_elements = active_action.abs() > 1.0
          action_element_count += active_action.numel()
          clipped_action_element_count += int(clipped_elements.sum().item())
          action_frame_count += active_action.shape[0]
          clipped_action_frame_count += int(clipped_elements.any(dim=-1).sum().item())
        dynamic_com = action_term.dynamic_com.compute()
        active_dynamic_frames = int(active_before_step.sum().item())
        if active_dynamic_frames:
          step_minimum_support_margin = float(
            dynamic_com.support_margin[active_before_step].min().item()
          )
          step_maximum_capture_point_norm = float(torch.linalg.vector_norm(
            dynamic_com.capture_point_pelvis[active_before_step], dim=-1
          ).max().item())
          minimum_support_margin = min(
            minimum_support_margin,
            step_minimum_support_margin,
          )
          maximum_capture_point_norm = max(
            maximum_capture_point_norm,
            step_maximum_capture_point_norm,
          )
          foot_displacement = torch.linalg.vector_norm(
            dynamic_com.foot_position_w[..., :2] - initial_foot_position[..., :2],
            dim=-1,
          )
          maximum_foot_planar_displacement = max(
            maximum_foot_planar_displacement,
            float(foot_displacement[active_before_step].max().item()),
          )
          step_maximum_foot_slip_speed = float(
            dynamic_com.foot_planar_speed[active_before_step].max().item()
          )
          maximum_foot_slip_speed = max(
            maximum_foot_slip_speed,
            step_maximum_foot_slip_speed,
          )
          step_double_support_loss_count = int(
            ((~dynamic_com.double_support) & active_before_step).sum().item()
          )
          step_no_foot_contact_count = int(
            (dynamic_com.no_foot_contact & active_before_step).sum().item()
          )
          double_support_loss_count += step_double_support_loss_count
          no_foot_contact_count += step_no_foot_contact_count
          dynamic_com_frame_count += active_dynamic_frames
          if control_step < 16:
            initial_dynamic_com_trace.append({
              "control_step": control_step,
              "active_frames": active_dynamic_frames,
              "minimum_support_margin_m": step_minimum_support_margin,
              "maximum_capture_point_norm_m": step_maximum_capture_point_norm,
              "maximum_foot_slip_speed_m_s": step_maximum_foot_slip_speed,
              "double_support_loss_count": step_double_support_loss_count,
              "no_foot_contact_count": step_no_foot_contact_count,
            })
          if control_step == module.DYNAMIC_COM_SETTLING_STEPS:
            settled_foot_position = dynamic_com.foot_position_w.clone()
          if control_step >= module.DYNAMIC_COM_SETTLING_STEPS:
            if settled_foot_position is None:
              raise RuntimeError("Dynamic-CoM settling reference was not captured")
            settled_foot_displacement = torch.linalg.vector_norm(
              dynamic_com.foot_position_w[..., :2]
                - settled_foot_position[..., :2],
              dim=-1,
            )
            settled_minimum_support_margin = min(
              settled_minimum_support_margin, step_minimum_support_margin
            )
            settled_maximum_capture_point_norm = max(
              settled_maximum_capture_point_norm, step_maximum_capture_point_norm
            )
            settled_maximum_foot_planar_displacement = max(
              settled_maximum_foot_planar_displacement,
              float(settled_foot_displacement[active_before_step].max().item()),
            )
            settled_maximum_foot_slip_speed = max(
              settled_maximum_foot_slip_speed, step_maximum_foot_slip_speed
            )
            settled_double_support_loss_count += step_double_support_loss_count
            settled_no_foot_contact_count += step_no_foot_contact_count
            settled_dynamic_com_frame_count += active_dynamic_frames
        wrist_error_before_step = module.active_wrist_position_error(raw_env)
        wrist_delta_before_step = module.active_wrist_position_delta(raw_env)
        maximum_action = max(maximum_action, float(action.abs().max().item()))
        observations, rewards, dones, _ = env.step(action)
        for sensor_name, sensor in reach_contact_sensors.items():
          force = sensor.data.force_history
          if force is None:
            force = sensor.data.force
          if force is None or sensor.data.found is None:
            raise RuntimeError(
              f"Reach contact diagnostic {sensor_name} has incomplete fields"
            )
          force_norm = torch.linalg.vector_norm(force, dim=-1)
          if force_norm.ndim == 3:
            force_norm = force_norm.amax(dim=-1)
          reach_contact_maximum_force_by_primary[sensor_name] = torch.maximum(
            reach_contact_maximum_force_by_primary[sensor_name],
            force_norm.amax(dim=0),
          )
          reach_contact_found_count_by_primary[sensor_name] += (
            sensor.data.found > 0
          ).sum(dim=0)
        label_is_finite = torch.isfinite(teacher_label_before_step).all(dim=-1)
        reach_teacher_label_count += int(label_is_finite.sum().item())
        reach_teacher_label_total += num_envs
        if not torch.isfinite(observations["actor"]).all():
          raise RuntimeError("Residual rollout observation became non-finite")
        if not torch.isfinite(rewards).all():
          raise RuntimeError("Residual rollout reward became non-finite")
        reward_sum += rewards * active
        termination_count += int((dones.bool() & active_before_step).sum().item())
        success |= (
          raw_env.termination_manager.get_term("wrist_target_success").bool()
          & active_before_step
        )
        fall_count += int((
          raw_env.termination_manager.get_term("fall").bool()
          & active_before_step
        ).sum().item())
        non_foot_collision_count += int(
          (
            raw_env.termination_manager.get_term("non_foot_ground").bool()
            & active_before_step
          ).sum().item()
        )
        active &= ~dones.bool()
        wrist_error_after_step = module.active_wrist_position_error(raw_env)
        wrist_delta_after_step = module.active_wrist_position_delta(raw_env)
        # RSL-RL resets terminal environments inside env.step.  Preserve the
        # last pre-reset physical error instead of reporting the next episode's
        # freshly sampled initial state as the final result.
        terminal_safe_wrist_error = torch.where(
          dones.bool(), wrist_error_before_step, wrist_error_after_step
        )
        terminal_safe_wrist_delta = torch.where(
          dones.bool().unsqueeze(-1), wrist_delta_before_step, wrist_delta_after_step
        )
        # Once an evaluation environment terminates, RSL-RL immediately starts
        # another episode in that slot.  Freeze first-episode evidence instead
        # of letting later reset episodes overwrite final/minimum reach metrics.
        final_wrist_error = torch.where(
          active_before_step, terminal_safe_wrist_error, final_wrist_error
        )
        final_wrist_delta = torch.where(
          active_before_step.unsqueeze(-1), terminal_safe_wrist_delta, final_wrist_delta
        )
        improved = active_before_step & (final_wrist_error < minimum_wrist_error)
        minimum_wrist_error = torch.where(
          improved, final_wrist_error, minimum_wrist_error
        )
        minimum_wrist_error_step = torch.where(
          improved,
          torch.full_like(minimum_wrist_error_step, control_step + 1),
          minimum_wrist_error_step,
        )
        minimum_wrist_delta = torch.where(
          improved.unsqueeze(-1), final_wrist_delta, minimum_wrist_delta
        )
        completed_control_steps = control_step + 1
        if completed_control_steps % 10 == 0 or completed_control_steps == steps:
          append_wrist_error_trace(
            completed_control_steps,
            terminal_safe_wrist_error,
            active_before_step,
          )

        frozen_error = (
          action_term.body_targets[:, :15]
          - action_term.teacher_body_targets[:, :15]
        ).abs()
        upper_error = (
          action_term.body_targets[:, 15:]
          - action_term.teacher.default_joint_positions[:, 15:]
          - action_term.upper_body_residual
        ).abs()
        maximum_frozen_composition_error = max(
          maximum_frozen_composition_error, float(frozen_error.max().item())
        )
        maximum_upper_composition_error = max(
          maximum_upper_composition_error, float(upper_error.max().item())
        )
        maximum_hand_target_error = max(
          maximum_hand_target_error,
          float(action_term.hand_targets.abs().max().item()),
        )
        frozen_position_error = (
          robot.data.joint_pos[:, action_term._body_ids[:15]]
          - action_term.teacher_body_targets[:, :15]
        )
        frozen_tracking_squared_error_sum += float(
          frozen_position_error.square().sum().item()
        )
        frozen_tracking_sample_count += frozen_position_error.numel()
        upper_position_error = (
          robot.data.joint_pos[:, action_term._body_ids[15:]]
          - action_term.body_targets[:, 15:]
        )
        upper_tracking_squared_error_sum_by_joint += (
          upper_position_error.square().sum(dim=0).to(torch.float64)
        )
        maximum_upper_tracking_error_by_joint = torch.maximum(
          maximum_upper_tracking_error_by_joint,
          upper_position_error.abs().amax(dim=0),
        )
        upper_tracking_sample_count += num_envs
        teacher_frame_count += num_envs
        residual_frame_count += num_envs

        effort = robot.data.qfrc_actuator
        ranges = raw_env.sim.model.jnt_actfrcrange[:, joint_ids]
        directional_limits = torch.where(effort >= 0.0, ranges[..., 1], -ranges[..., 0])
        force_ratio = effort.abs() / directional_limits
        maximum_force_ratio = max(
          maximum_force_ratio,
          float(force_ratio.max().item()),
        )
        maximum_body_force_ratio_by_joint = torch.maximum(
          maximum_body_force_ratio_by_joint,
          force_ratio[:, action_term._body_ids].amax(dim=0),
        )
        body_constraint_force = robot.data._joint_dof_field(
          "qfrc_constraint"
        )[:, action_term._body_ids]
        body_bias_force = robot.data._joint_dof_field(
          "qfrc_bias"
        )[:, action_term._body_ids]
        maximum_body_constraint_force_by_joint = torch.maximum(
          maximum_body_constraint_force_by_joint,
          body_constraint_force.abs().amax(dim=0),
        )
        maximum_body_bias_force_by_joint = torch.maximum(
          maximum_body_bias_force_by_joint,
          body_bias_force.abs().amax(dim=0),
        )
        maximum_teacher_action = max(
          maximum_teacher_action, float(action_term.teacher_action.abs().max().item())
        )
        maximum_observation = max(
          maximum_observation, float(observations["actor"].abs().max().item())
        )
        maximum_joint_velocity = max(
          maximum_joint_velocity, float(robot.data.joint_vel.abs().max().item())
        )
        body_velocity = robot.data.joint_vel[:, action_term._body_ids]
        hand_velocity = robot.data.joint_vel[:, action_term._hand_ids]
        frozen_velocity = body_velocity[:, :15].abs()
        upper_velocity = body_velocity[:, 15:].abs()
        hand_velocity_abs = hand_velocity.abs()
        maximum_frozen_velocity_by_joint = torch.maximum(
          maximum_frozen_velocity_by_joint, frozen_velocity.amax(dim=0)
        )
        maximum_upper_velocity_by_joint = torch.maximum(
          maximum_upper_velocity_by_joint, upper_velocity.amax(dim=0)
        )
        maximum_hand_velocity_by_joint = torch.maximum(
          maximum_hand_velocity_by_joint, hand_velocity_abs.amax(dim=0)
        )
        maximum_teacher_action_by_joint = torch.maximum(
          maximum_teacher_action_by_joint,
          action_term.teacher_action.abs().amax(dim=0),
        )
        frozen_target_delta = (
          action_term.body_targets[:, :15] - previous_frozen_targets
        ).abs()
        maximum_frozen_target_delta_by_joint = torch.maximum(
          maximum_frozen_target_delta_by_joint,
          frozen_target_delta.amax(dim=0),
        )
        previous_frozen_targets.copy_(action_term.body_targets[:, :15])
        step_peak, flat_index = frozen_velocity.reshape(-1).max(dim=0)
        step_peak_value = float(step_peak.item())
        if control_step < 16:
          frozen_velocity_trace.append(step_peak_value)
        if step_peak_value > peak_frozen_velocity["value_rad_s"]:
          flattened = int(flat_index.item())
          joint_index = flattened % 15
          peak_frozen_velocity = {
            "value_rad_s": step_peak_value,
            "control_step": control_step,
            "environment_index": flattened // 15,
            "joint_index": joint_index,
            "joint_name": module.FROZEN_TEACHER_JOINT_NAMES[joint_index],
          }
        maximum_frozen_joint_velocity = max(
          maximum_frozen_joint_velocity,
          step_peak_value,
        )
        if control_step >= 10:
          maximum_frozen_joint_velocity_after_settling = max(
            maximum_frozen_joint_velocity_after_settling, step_peak_value
          )
        maximum_upper_body_joint_velocity = max(
          maximum_upper_body_joint_velocity,
          float(body_velocity[:, 15:].abs().max().item()),
        )
        maximum_hand_joint_velocity = max(
          maximum_hand_joint_velocity,
          float(hand_velocity.abs().max().item()),
        )
        minimum_root_height = min(
          minimum_root_height, float(robot.data.root_link_pos_w[:, 2].min().item())
        )

    dynamic_com_settling_applied = settled_dynamic_com_frame_count > 0
    if dynamic_com_settling_applied:
      gated_minimum_support_margin = settled_minimum_support_margin
      gated_maximum_capture_point_norm = settled_maximum_capture_point_norm
      gated_maximum_foot_planar_displacement = (
        settled_maximum_foot_planar_displacement
      )
      gated_maximum_foot_slip_speed = settled_maximum_foot_slip_speed
      gated_double_support_loss_count = settled_double_support_loss_count
      gated_no_foot_contact_count = settled_no_foot_contact_count
      gated_dynamic_com_frame_count = settled_dynamic_com_frame_count
    else:
      gated_minimum_support_margin = minimum_support_margin
      gated_maximum_capture_point_norm = maximum_capture_point_norm
      gated_maximum_foot_planar_displacement = maximum_foot_planar_displacement
      gated_maximum_foot_slip_speed = maximum_foot_slip_speed
      gated_double_support_loss_count = double_support_loss_count
      gated_no_foot_contact_count = no_foot_contact_count
      gated_dynamic_com_frame_count = dynamic_com_frame_count

    teacher_identity = dict(action_term.teacher.identity)
    teacher_identity["device"] = str(action_term.teacher_action.device)
    reach_teacher_sample_total = int(reach_teacher_active_joint_samples.sum().item())
    reach_teacher_authority_saturation_total = int(
      reach_teacher_authority_saturation_count.sum().item()
    )
    reach_teacher_soft_limit_saturation_total = int(
      reach_teacher_soft_limit_saturation_count.sum().item()
    )
    reach_teacher_command_lead_saturation_total = int(
      reach_teacher_command_lead_saturation_count.sum().item()
    )
    reach_teacher_joint_diagnostics = {}
    for joint_index, joint_name in enumerate(module.UPPER_BODY_JOINT_NAMES):
      joint_samples = int(reach_teacher_active_joint_samples[joint_index].item())
      authority_count = int(
        reach_teacher_authority_saturation_count[joint_index].item()
      )
      soft_limit_count = int(
        reach_teacher_soft_limit_saturation_count[joint_index].item()
      )
      command_lead_count = int(
        reach_teacher_command_lead_saturation_count[joint_index].item()
      )
      reach_teacher_joint_diagnostics[joint_name] = {
        "active_samples": joint_samples,
        "authority_saturation_count": authority_count,
        "authority_saturation_rate": (
          authority_count / joint_samples if joint_samples else 0.0
        ),
        "soft_limit_saturation_count": soft_limit_count,
        "soft_limit_saturation_rate": (
          soft_limit_count / joint_samples if joint_samples else 0.0
        ),
        "command_lead_saturation_count": command_lead_count,
        "command_lead_saturation_rate": (
          command_lead_count / joint_samples if joint_samples else 0.0
        ),
        "mean_absolute_action": (
          float(reach_teacher_absolute_action_sum[joint_index].item())
          / joint_samples if joint_samples else 0.0
        ),
        "maximum_absolute_unclamped_action": float(
          reach_teacher_maximum_unclamped_action_by_joint[joint_index].item()
        ),
        "maximum_solver_target_slew_rad": float(
          reach_teacher_maximum_solver_target_slew_by_joint[joint_index].item()
        ),
        "maximum_actual_joint_target_step_rad": float(
          reach_teacher_maximum_joint_target_step_by_joint[joint_index].item()
        ),
      }

    hand_trajectory = {}
    for hand_index, hand_name in enumerate(("left", "right")):
      hand_mask = initial_active_hand == hand_index
      hand_count = int(hand_mask.sum().item())
      hand_trajectory[hand_name] = {
        "environment_count": hand_count,
        "success_count": int((success & hand_mask).sum().item()),
        "success_rate": (
          float(success[hand_mask].float().mean().item()) if hand_count else 0.0
        ),
        "initial_mean": (
          float(initial_wrist_error[hand_mask].mean().item()) if hand_count else 0.0
        ),
        "minimum_mean_over_episode": (
          float(minimum_wrist_error[hand_mask].mean().item()) if hand_count else 0.0
        ),
        "final_mean": (
          float(final_wrist_error[hand_mask].mean().item()) if hand_count else 0.0
        ),
        "final_rebound_from_minimum_mean": (
          float((final_wrist_error - minimum_wrist_error)[hand_mask].mean().item())
          if hand_count else 0.0
        ),
      }
    return {
      "environment_count": num_envs,
      "control_steps": steps,
      "mean_reward_sum": float(reward_sum.mean().item()),
      "success_count": int(success.sum().item()),
      "success_rate": float(success.float().mean().item()),
      "wrist_position_error_m": {
        "initial_mean": float(initial_wrist_error.mean().item()),
        "initial_maximum": float(initial_wrist_error.max().item()),
        "minimum_mean_over_episode": float(minimum_wrist_error.mean().item()),
        "minimum_best_environment": float(minimum_wrist_error.min().item()),
        "minimum_step_mean": float(minimum_wrist_error_step.float().mean().item()),
        "minimum_step_p90": float(torch.quantile(
          minimum_wrist_error_step.float(), 0.90
        ).item()),
        "minimum_mean_absolute_xyz": [
          float(value) for value in minimum_wrist_delta.abs().mean(dim=0).tolist()
        ],
        "final_mean": float(final_wrist_error.mean().item()),
        "final_maximum": float(final_wrist_error.max().item()),
        "final_rebound_from_minimum_mean": float(
          (final_wrist_error - minimum_wrist_error).mean().item()
        ),
        "initial_mean_absolute_xyz": [
          float(value) for value in initial_wrist_delta.abs().mean(dim=0).tolist()
        ],
        "final_mean_absolute_xyz": [
          float(value) for value in final_wrist_delta.abs().mean(dim=0).tolist()
        ],
        "trace_protocol": "hear-active-first-episode-sparse-wrist-error-trace-v1",
        "sparse_active_first_episode_trace": wrist_error_trace,
        "by_initial_active_hand": hand_trajectory,
      },
      "termination_count": termination_count,
      "fall_count": fall_count,
      "fall_rate": fall_count / num_envs,
      "non_foot_collision_count": non_foot_collision_count,
      "maximum_absolute_student_action": maximum_action,
      "action_clipping": {
        "limit": 1.0,
        "element_count": action_element_count,
        "clipped_element_count": clipped_action_element_count,
        "clipped_element_rate": (
          clipped_action_element_count / action_element_count
          if action_element_count else 0.0
        ),
        "frame_count": action_frame_count,
        "clipped_frame_count": clipped_action_frame_count,
        "clipped_frame_rate": (
          clipped_action_frame_count / action_frame_count
          if action_frame_count else 0.0
        ),
      },
      "maximum_absolute_teacher_action": maximum_teacher_action,
      "maximum_absolute_observation": maximum_observation,
      "maximum_joint_velocity_rad_s": maximum_joint_velocity,
      "maximum_frozen_joint_velocity_rad_s": maximum_frozen_joint_velocity,
      "maximum_frozen_joint_velocity_after_settling_rad_s": (
        maximum_frozen_joint_velocity_after_settling
      ),
      "maximum_upper_body_joint_velocity_rad_s": maximum_upper_body_joint_velocity,
      "maximum_hand_joint_velocity_rad_s": maximum_hand_joint_velocity,
      "maximum_frozen_joint_velocity_by_joint_rad_s": {
        name: float(value)
        for name, value in zip(
          module.FROZEN_TEACHER_JOINT_NAMES,
          maximum_frozen_velocity_by_joint.tolist(),
          strict=True,
        )
      },
      "maximum_upper_body_joint_velocity_by_joint_rad_s": {
        name: float(value)
        for name, value in zip(
          module.UPPER_BODY_JOINT_NAMES,
          maximum_upper_velocity_by_joint.tolist(),
          strict=True,
        )
      },
      "maximum_hand_joint_velocity_by_joint_rad_s": {
        name: float(value)
        for name, value in zip(
          module.base.HAND_JOINT_NAMES,
          maximum_hand_velocity_by_joint.tolist(),
          strict=True,
        )
      },
      "maximum_teacher_action_by_joint": {
        name: float(value)
        for name, value in zip(
          module.base.BODY_JOINT_NAMES,
          maximum_teacher_action_by_joint.tolist(),
          strict=True,
        )
      },
      "maximum_frozen_target_delta_by_joint_rad": {
        name: float(value)
        for name, value in zip(
          module.FROZEN_TEACHER_JOINT_NAMES,
          maximum_frozen_target_delta_by_joint.tolist(),
          strict=True,
        )
      },
      "maximum_frozen_target_slew_rad_s": float(
        maximum_frozen_target_delta_by_joint.max().item()
        / action_term._env.step_dt
      ),
      "peak_frozen_joint_velocity": peak_frozen_velocity,
      "initial_frozen_joint_velocity_trace_rad_s": frozen_velocity_trace,
      "maximum_joint_actuator_force_ratio": maximum_force_ratio,
      "maximum_body_joint_actuator_force_ratio_by_joint": {
        name: float(maximum_body_force_ratio_by_joint[index].item())
        for index, name in enumerate((
          *module.FROZEN_TEACHER_JOINT_NAMES,
          *module.UPPER_BODY_JOINT_NAMES,
        ))
      },
      "maximum_body_joint_constraint_force_nm_by_joint": {
        name: float(maximum_body_constraint_force_by_joint[index].item())
        for index, name in enumerate((
          *module.FROZEN_TEACHER_JOINT_NAMES,
          *module.UPPER_BODY_JOINT_NAMES,
        ))
      },
      "maximum_body_joint_bias_force_nm_by_joint": {
        name: float(maximum_body_bias_force_by_joint[index].item())
        for index, name in enumerate((
          *module.FROZEN_TEACHER_JOINT_NAMES,
          *module.UPPER_BODY_JOINT_NAMES,
        ))
      },
      "upper_body_joint_tracking": {
        "measurement_window": "all_control_frames",
        "sample_count_per_joint": upper_tracking_sample_count,
        "rms_error_rad_by_joint": {
          name: float((
            upper_tracking_squared_error_sum_by_joint[index]
            / upper_tracking_sample_count
          ).sqrt().item())
          for index, name in enumerate(module.UPPER_BODY_JOINT_NAMES)
        },
        "maximum_absolute_error_rad_by_joint": {
          name: float(maximum_upper_tracking_error_by_joint[index].item())
          for index, name in enumerate(module.UPPER_BODY_JOINT_NAMES)
        },
      },
      "minimum_root_height_m": minimum_root_height,
      "frozen_teacher_joint_rms_error_rad": (
        frozen_tracking_squared_error_sum / frozen_tracking_sample_count
      ) ** 0.5,
      "composition": {
        "maximum_frozen_joint_command_error": maximum_frozen_composition_error,
        "maximum_upper_body_command_error": maximum_upper_composition_error,
        "frozen_joint_count": 15,
        "upper_body_residual_joint_count": 14,
        "hand_synergy_count": 0,
        "maximum_fixed_open_hand_target_error_rad": maximum_hand_target_error,
      },
      "attribution": {
        "teacher_frames": teacher_frame_count,
        "upper_body_residual_frames": residual_frame_count,
        "teacher_frame_ratio": 1.0,
        "upper_body_residual_frame_ratio": 1.0,
      },
      "teacher": teacher_identity,
      "reach_teacher": {
        **action_term.reach_teacher.identity,
        "label_count": reach_teacher_label_count,
        "label_coverage": (
          reach_teacher_label_count / reach_teacher_label_total
          if reach_teacher_label_total else 0.0
        ),
        "executed_teacher_only": use_reach_teacher,
        "authority_diagnostics": {
          "active_joint_samples": reach_teacher_sample_total,
          "authority_saturation_count": reach_teacher_authority_saturation_total,
          "authority_saturation_rate": (
            reach_teacher_authority_saturation_total / reach_teacher_sample_total
            if reach_teacher_sample_total else 0.0
          ),
          "soft_limit_saturation_count": reach_teacher_soft_limit_saturation_total,
          "soft_limit_saturation_rate": (
            reach_teacher_soft_limit_saturation_total / reach_teacher_sample_total
            if reach_teacher_sample_total else 0.0
          ),
          "command_lead_saturation_count": (
            reach_teacher_command_lead_saturation_total
          ),
          "command_lead_saturation_rate": (
            reach_teacher_command_lead_saturation_total / reach_teacher_sample_total
            if reach_teacher_sample_total else 0.0
          ),
          "maximum_absolute_unclamped_action": (
            maximum_absolute_unclamped_reach_teacher_action
          ),
          "by_joint": reach_teacher_joint_diagnostics,
        },
        "adaptive_solver_diagnostics": {
          "active_arm_samples": reach_teacher_active_arm_samples,
          "minimum_singular_value_m_per_rad": (
            reach_teacher_minimum_singular_value
            if reach_teacher_active_arm_samples else 0.0
          ),
          "mean_singular_value_m_per_rad": (
            reach_teacher_singular_value_sum / reach_teacher_active_arm_samples
            if reach_teacher_active_arm_samples else 0.0
          ),
          "below_singularity_threshold_count": (
            reach_teacher_singularity_threshold_count
          ),
          "below_singularity_threshold_rate": (
            reach_teacher_singularity_threshold_count
            / reach_teacher_active_arm_samples
            if reach_teacher_active_arm_samples else 0.0
          ),
          "holding_target_count": reach_teacher_holding_target_count,
          "holding_target_rate": (
            reach_teacher_holding_target_count / reach_teacher_active_arm_samples
            if reach_teacher_active_arm_samples else 0.0
          ),
          "maximum_solver_target_slew_rad": float(
            reach_teacher_maximum_solver_target_slew_by_joint.max().item()
          ),
          "maximum_actual_joint_target_step_rad": float(
            reach_teacher_maximum_joint_target_step_by_joint.max().item()
          ),
        },
        "feasible_posture_diagnostics": {
          "feature_sample_count": feasible_posture_feature_sample_count,
          "feature_clamp_count": feasible_posture_feature_clamp_count,
          "feature_clamp_rate": (
            feasible_posture_feature_clamp_count
            / feasible_posture_feature_sample_count
            if feasible_posture_feature_sample_count else 0.0
          ),
          "normalized_action_sample_count": feasible_posture_action_sample_count,
          "normalized_action_clamp_count": feasible_posture_action_clamp_count,
          "normalized_action_clamp_rate": (
            feasible_posture_action_clamp_count
            / feasible_posture_action_sample_count
            if feasible_posture_action_sample_count else 0.0
          ),
        },
      },
      "reach_contact_diagnostics": {
        "protocol": "hear-upper-body-collision-diagnostics-v1",
        "sample_count_per_primary": num_envs * steps,
        "sensors": {
          sensor_name: {
            "maximum_force_n_by_primary": {
              primary_name: float(
                reach_contact_maximum_force_by_primary[sensor_name][index].item()
              )
              for index, primary_name in enumerate(sensor.primary_names)
            },
            "contact_sample_rate_by_primary": {
              primary_name: float(
                reach_contact_found_count_by_primary[sensor_name][index].item()
              ) / (num_envs * steps)
              for index, primary_name in enumerate(sensor.primary_names)
            },
          }
          for sensor_name, sensor in reach_contact_sensors.items()
        },
      },
      "dynamic_com": {
        "protocol": module.DYNAMIC_COM_PROTOCOL,
        "measurement_window": (
          "post_settling_active_frames"
          if dynamic_com_settling_applied else "reset_inclusive_fallback"
        ),
        "settling_control_steps_excluded": module.DYNAMIC_COM_SETTLING_STEPS,
        "minimum_support_margin_m": gated_minimum_support_margin,
        "maximum_capture_point_norm_m": gated_maximum_capture_point_norm,
        "maximum_foot_planar_displacement_m": (
          gated_maximum_foot_planar_displacement
        ),
        "maximum_foot_slip_speed_m_s": gated_maximum_foot_slip_speed,
        "double_support_loss_count": gated_double_support_loss_count,
        "double_support_loss_rate": (
          gated_double_support_loss_count / gated_dynamic_com_frame_count
          if gated_dynamic_com_frame_count else 0.0
        ),
        "no_foot_contact_count": gated_no_foot_contact_count,
        "no_foot_contact_rate": (
          gated_no_foot_contact_count / gated_dynamic_com_frame_count
          if gated_dynamic_com_frame_count else 0.0
        ),
        "frame_count": gated_dynamic_com_frame_count,
        "reset_inclusive": {
          "minimum_support_margin_m": minimum_support_margin,
          "maximum_capture_point_norm_m": maximum_capture_point_norm,
          "maximum_foot_planar_displacement_m": maximum_foot_planar_displacement,
          "maximum_foot_slip_speed_m_s": maximum_foot_slip_speed,
          "double_support_loss_count": double_support_loss_count,
          "double_support_loss_rate": (
            double_support_loss_count / dynamic_com_frame_count
            if dynamic_com_frame_count else 0.0
          ),
          "no_foot_contact_count": no_foot_contact_count,
          "no_foot_contact_rate": (
            no_foot_contact_count / dynamic_com_frame_count
            if dynamic_com_frame_count else 0.0
          ),
          "frame_count": dynamic_com_frame_count,
        },
        "initial_trace": initial_dynamic_com_trace,
      },
      "finite": True,
      "environment_closed": True,
    }
  finally:
    env.close()
    environment_closed = True
    if not environment_closed:
      raise RuntimeError("Residual Workyard environment did not close")


def smoke(args: argparse.Namespace, module: ModuleType) -> dict[str, object]:
  return rollout(module, args.num_envs, args.rollout_steps, args.seed)


def teacher_rollout(args: argparse.Namespace, module: ModuleType) -> dict[str, object]:
  return rollout(
    module,
    args.num_envs,
    args.rollout_steps,
    args.seed,
    use_reach_teacher=True,
  )


def dagger_warm_start(
  runner,
  env,
  raw_env,
  module: ModuleType,
  args: argparse.Namespace,
  checkpoint: Path,
  curves_path: Path,
) -> dict[str, object]:
  """Train the deployable actor online against analytic labels, then hand it to PPO."""
  import torch

  actor = runner.alg.get_policy()
  actor.train()
  optimizer = torch.optim.Adam(actor.parameters(), lr=args.dagger_learning_rate)
  observations = env.get_observations().to("cuda:0")
  action_term = module._residual_action(raw_env)
  loss_curve: list[dict[str, float | int]] = []
  label_count = 0
  finite_label_count = 0
  teacher_execution_count = 0
  total_execution_count = 0

  for step in range(args.dagger_steps):
    fraction = step / max(1, args.dagger_steps - 1)
    beta = args.dagger_beta_initial + fraction * (
      args.dagger_beta_final - args.dagger_beta_initial
    )
    # no_grad keeps labels and next observations as ordinary tensors.  PyTorch
    # inference tensors cannot be saved by Smooth-L1/Linear backward even when
    # they do not require gradients.
    with torch.no_grad():
      teacher_action = action_term.compute_reach_teacher_action()
      actor.update_normalization(observations)
    predicted_action, policy_action_std = (
      runner.alg.deterministic_policy_statistics(observations)
    )
    imitation_loss = torch.nn.functional.smooth_l1_loss(
      predicted_action, teacher_action
    )
    dispersion_penalty = torch.relu(
      policy_action_std - args.teacher_maximum_action_std
    ).square().mean()
    loss = (
      imitation_loss
      + args.teacher_dispersion_coefficient * dispersion_penalty
    )
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    gradient_norm = torch.nn.utils.clip_grad_norm_(actor.parameters(), 1.0)
    optimizer.step()

    with torch.no_grad():
      student_action = actor(observations).clamp(-1.0, 1.0)
      teacher_mask = torch.rand(
        (raw_env.num_envs, 1), device=raw_env.device
      ) < beta
      executed_action = torch.where(teacher_mask, teacher_action, student_action)
      observations, _, _, _ = env.step(executed_action)
      observations = observations.to("cuda:0")
    finite = torch.isfinite(teacher_action).all(dim=-1)
    finite_label_count += int(finite.sum().item())
    label_count += raw_env.num_envs
    teacher_execution_count += int(teacher_mask.sum().item())
    total_execution_count += raw_env.num_envs
    loss_curve.append({
      "step": step,
      "loss": float(loss.detach().item()),
      "imitation_loss": float(imitation_loss.detach().item()),
      "dispersion_penalty": float(dispersion_penalty.detach().item()),
      "mean_policy_action_std": float(policy_action_std.detach().mean().item()),
      "gradient_norm": float(gradient_norm.detach().item()),
      "teacher_beta": beta,
    })

  runner.save(str(checkpoint), infos={
    "protocol": "hear-online-dagger-warm-start-v1",
    "steps": args.dagger_steps,
  })
  curve_payload = {
    "protocol": "hear-online-dagger-curves-v1",
    "loss": loss_curve,
  }
  curves_path.write_text(
    json.dumps(curve_payload, ensure_ascii=False, indent=2) + "\n",
    encoding="utf-8",
  )
  normalizer = getattr(actor, "obs_normalizer", None)
  normalizer_count = getattr(normalizer, "count", None)
  return {
    "protocol": "hear-online-dagger-warm-start-v1",
    "steps": args.dagger_steps,
    "samples": args.dagger_steps * raw_env.num_envs,
    "learning_rate": args.dagger_learning_rate,
    "loss": "smooth_l1_plus_excess_std_penalty",
    "actor_distribution": "beta_bounded_minus_one_one",
    "maximum_action_std": args.teacher_maximum_action_std,
    "dispersion_coefficient": args.teacher_dispersion_coefficient,
    "initial_loss": loss_curve[0]["loss"],
    "final_loss": loss_curve[-1]["loss"],
    "minimum_loss": min(float(point["loss"]) for point in loss_curve),
    "final_mean_policy_action_std": float(
      loss_curve[-1]["mean_policy_action_std"]
    ),
    "label_coverage": finite_label_count / label_count if label_count else 0.0,
    "teacher_execution_ratio": (
      teacher_execution_count / total_execution_count if total_execution_count else 0.0
    ),
    "teacher_beta": {
      "initial": args.dagger_beta_initial,
      "final": args.dagger_beta_final,
    },
    "actor_normalizer_sample_count": (
      int(normalizer_count.item()) if normalizer_count is not None else None
    ),
    "checkpoint": {
      "file": checkpoint.name,
      "bytes": checkpoint.stat().st_size,
      "sha256": sha256(checkpoint),
    },
    "curves": {
      "file": curves_path.name,
      "bytes": curves_path.stat().st_size,
      "sha256": sha256(curves_path),
    },
  }


def ppo_with_retention(
  runner,
  env,
  raw_env,
  module: ModuleType,
  args: argparse.Namespace,
) -> dict[str, object]:
  """Train PPO with learner-rollout teacher labels in every PPO minibatch."""
  import torch

  actor = runner.alg.get_policy()
  if not hasattr(runner.alg, "configure_rollout_teacher"):
    raise RuntimeError("Residual runner did not construct HearRetentionPPO")
  distribution = runner.alg.distribution_identity()
  if not distribution["structurally_bounded"]:
    raise RuntimeError("Residual actor is not structurally bounded to [-1, 1]")
  action_term = module._residual_action(raw_env)
  teacher_coefficient = (
    0.0 if args.ppo_retention_mode == "baseline"
    else args.rollout_teacher_coefficient
  )
  runner.alg.configure_rollout_teacher(
    action_term.compute_reach_teacher_action,
    teacher_coefficient,
    args.teacher_maximum_action_std,
    args.teacher_dispersion_coefficient,
  )
  actor_normalizer = getattr(actor, "obs_normalizer", None)
  normalizer_count = getattr(actor_normalizer, "count", None)
  normalizer_count_before = (
    int(normalizer_count.item()) if normalizer_count is not None else None
  )
  if args.ppo_retention_mode == "baseline":
    runner.learn(
      num_learning_iterations=args.iterations,
      init_at_random_ep_len=True,
    )
    normalizer_count_after = (
      int(normalizer_count.item()) if normalizer_count is not None else None
    )
    return {
      "protocol": "hear-ppo-retention-v2",
      "mode": "baseline",
      "actor_distribution": distribution,
      "critic_warmup_iterations": 0,
      "actor_update_iterations": args.iterations,
      "actor_normalizer_frozen_after_dagger": False,
      "actor_normalizer_sample_count_before": normalizer_count_before,
      "actor_normalizer_sample_count_after": normalizer_count_after,
      "rollout_teacher_loss_coefficient": 0.0,
      "rollout_teacher_label_coverage": float(
        runner.alg.last_retention_metrics["rollout_teacher_label_coverage"]
      ),
      "curve": [],
    }

  if args.critic_warmup_iterations >= args.iterations:
    raise ValueError(
      "Retention PPO requires at least one actor-update iteration after "
      "critic warm-up"
    )

  critic = runner.alg._raw_critic
  actor_parameters = list(actor.parameters())
  critic_parameters = list(critic.parameters())
  runner.alg.optimizer = torch.optim.Adam([
    {"params": actor_parameters, "lr": args.ppo_actor_learning_rate},
    {"params": critic_parameters, "lr": args.ppo_critic_learning_rate},
  ])
  runner.alg.schedule = "fixed"
  runner.alg.learning_rate = args.ppo_critic_learning_rate

  anchor_parameters = {
    name: parameter.detach().clone()
    for name, parameter in actor.named_parameters()
  }
  original_algorithm_update = runner.alg.update
  original_normalization_update = actor.update_normalization
  update_index = 0
  curve: list[dict[str, object]] = []

  def actor_relative_parameter_drift() -> float:
    delta_squared = torch.zeros((), device=raw_env.device)
    anchor_squared = torch.zeros((), device=raw_env.device)
    for name, parameter in actor.named_parameters():
      anchor = anchor_parameters[name]
      delta_squared += (parameter.detach() - anchor).square().sum()
      anchor_squared += anchor.square().sum()
    return float(torch.sqrt(delta_squared / anchor_squared.clamp_min(1.0e-12)).item())

  def guarded_update() -> dict[str, float]:
    nonlocal update_index
    critic_warmup = update_index < args.critic_warmup_iterations
    if critic_warmup:
      for parameter in actor_parameters:
        parameter.requires_grad_(False)
    try:
      losses = original_algorithm_update()
    finally:
      if critic_warmup:
        for parameter in actor_parameters:
          parameter.requires_grad_(True)

    retention = dict(runner.alg.last_retention_metrics)
    point = {
      "iteration": update_index,
      "critic_warmup": critic_warmup,
      "rollout_teacher_label_count": retention["rollout_teacher_label_count"],
      "rollout_teacher_finite_label_count": retention[
        "rollout_teacher_finite_label_count"
      ],
      "rollout_teacher_label_coverage": retention[
        "rollout_teacher_label_coverage"
      ],
      "joint_teacher_loss": retention["joint_teacher_loss"],
      "teacher_mean_smooth_l1": retention["teacher_mean_smooth_l1"],
      "teacher_dispersion_penalty": retention["teacher_dispersion_penalty"],
      "mean_policy_action_std": retention["mean_policy_action_std"],
      "teacher_smooth_l1_by_action": retention[
        "teacher_smooth_l1_by_action"
      ],
      "actor_relative_parameter_drift": actor_relative_parameter_drift(),
    }
    curve.append(point)
    losses["critic_warmup"] = float(critic_warmup)
    losses["rollout_teacher"] = float(point["joint_teacher_loss"])
    losses["rollout_teacher_mean"] = float(point["teacher_mean_smooth_l1"])
    losses["rollout_teacher_dispersion"] = float(
      point["teacher_dispersion_penalty"]
    )
    losses["rollout_policy_action_std"] = float(point["mean_policy_action_std"])
    losses["actor_relative_parameter_drift"] = float(
      point["actor_relative_parameter_drift"]
    )
    update_index += 1
    return losses

  # DAgger already fitted the actor projection on mixed teacher/student states.
  # PPO may adapt the critic normalizer, but silently shifting the actor
  # normalizer would change the deployable policy even during critic-only warm-up.
  actor.update_normalization = lambda _observations: None
  runner.alg.update = guarded_update
  try:
    runner.learn(
      num_learning_iterations=args.iterations,
      init_at_random_ep_len=True,
    )
  finally:
    runner.alg.update = original_algorithm_update
    actor.update_normalization = original_normalization_update
    for parameter in actor_parameters:
      parameter.requires_grad_(True)

  normalizer_count_after = (
    int(normalizer_count.item()) if normalizer_count is not None else None
  )
  label_count = sum(int(point["rollout_teacher_label_count"]) for point in curve)
  finite_label_count = sum(
    int(point["rollout_teacher_finite_label_count"]) for point in curve
  )
  return {
    "protocol": "hear-ppo-retention-v2",
    "mode": "critic_warmup_rollout_teacher",
    "actor_distribution": distribution,
    "optimizer": "adam_separate_actor_critic_groups",
    "schedule": "fixed",
    "critic_warmup_iterations": args.critic_warmup_iterations,
    "actor_update_iterations": args.iterations - args.critic_warmup_iterations,
    "actor_learning_rate": args.ppo_actor_learning_rate,
    "critic_learning_rate": args.ppo_critic_learning_rate,
    "actor_normalizer_frozen_after_dagger": True,
    "actor_normalizer_sample_count_before": normalizer_count_before,
    "actor_normalizer_sample_count_after": normalizer_count_after,
    "teacher_supervision": "every_stored_learner_rollout_state",
    "loss_coupling": "same_ppo_minibatch",
    "teacher_action_scope": "authorized_14d_actor_action",
    "rollout_teacher_loss": "smooth_l1_plus_excess_std_penalty",
    "rollout_teacher_loss_coefficient": args.rollout_teacher_coefficient,
    "teacher_maximum_action_std": args.teacher_maximum_action_std,
    "teacher_dispersion_coefficient": args.teacher_dispersion_coefficient,
    "rollout_teacher_label_count": label_count,
    "rollout_teacher_finite_label_count": finite_label_count,
    "rollout_teacher_label_coverage": (
      finite_label_count / label_count if label_count else 0.0
    ),
    "final_mean_policy_action_std": float(curve[-1]["mean_policy_action_std"]),
    "final_actor_relative_parameter_drift": actor_relative_parameter_drift(),
    "curve": curve,
  }


def train(
  args: argparse.Namespace,
  module: ModuleType,
  contract: dict[str, object],
) -> tuple[
  dict[str, object], dict[str, object]
]:
  import torch
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
  from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
  from mjlab.utils.os import dump_yaml

  output_root = Path(args.output_root).resolve()
  staging = output_root / "artifacts"
  log_dir = output_root / "logs"
  staging.mkdir(parents=True, exist_ok=False)
  log_dir.mkdir(parents=True, exist_ok=False)

  env_cfg = load_env_cfg(module.TASK_ID, play=False)
  env_cfg.scene.num_envs = args.num_envs
  env_cfg.seed = args.seed
  agent_cfg = load_rl_cfg(module.TASK_ID)
  agent_cfg.max_iterations = args.iterations
  agent_cfg.save_interval = max(1, args.iterations // 4)
  agent_cfg.seed = args.seed
  agent_cfg.logger = "tensorboard"
  agent_cfg.run_name = "task_space_reach_dagger_then_ppo"
  agent_cfg.upload_model = False
  if args.ppo_retention_mode == "critic_warmup_rollout_teacher":
    agent_cfg.algorithm.learning_rate = args.ppo_critic_learning_rate
    agent_cfg.algorithm.schedule = "fixed"
  dump_yaml(staging / "env.yaml", asdict(env_cfg))
  dump_yaml(staging / "agent.yaml", asdict(agent_cfg))

  raw_env = ManagerBasedRlEnv(cfg=env_cfg, device="cuda:0")
  env = RslRlVecEnvWrapper(raw_env, clip_actions=agent_cfg.clip_actions)
  training_environment_closed = False
  runner = None
  tensorboard_writer = None
  try:
    action_term = module._residual_action(raw_env)
    teacher_identity = dict(action_term.teacher.identity)
    reach_teacher_identity = dict(action_term.reach_teacher.identity)
    runner_cls = load_runner_cls(module.TASK_ID) or MjlabOnPolicyRunner
    runner = runner_cls(env, asdict(agent_cfg), str(log_dir), "cuda:0")
    dagger_checkpoint = staging / "workyard_reach_dagger_warm_start.pt"
    dagger_curves = staging / "dagger-curves.json"
    dagger = dagger_warm_start(
      runner,
      env,
      raw_env,
      module,
      args,
      dagger_checkpoint,
      dagger_curves,
    )
    ppo_retention = ppo_with_retention(
      runner,
      env,
      raw_env,
      module,
      args,
    )
    tensorboard_writer = getattr(runner.logger, "writer", None)
    checkpoint = staging / "workyard_reach_dagger_ppo.pt"
    runner.save(str(checkpoint))
    policy = runner.get_inference_policy(device="cuda:0")
  finally:
    if tensorboard_writer is None and runner is not None:
      tensorboard_writer = getattr(runner.logger, "writer", None)
    if tensorboard_writer is not None:
      if hasattr(tensorboard_writer, "flush"):
        tensorboard_writer.flush()
      if hasattr(tensorboard_writer, "close"):
        tensorboard_writer.close()
    env.close()
    training_environment_closed = True

  checkpoint_data = torch.load(checkpoint, map_location="cpu", weights_only=False)
  checkpoint_iteration = int(checkpoint_data.get("iter", -1))
  completed_iterations = checkpoint_iteration + 1
  if completed_iterations < args.iterations:
    raise RuntimeError(
      f"Residual checkpoint contains {completed_iterations} iterations, "
      f"below {args.iterations}"
    )
  curves = tensorboard_curves(log_dir)
  curves_path = staging / "training-curves.json"
  curves_path.write_text(
    json.dumps(curves, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
  )
  ppo_evaluation = rollout(
    module, args.eval_envs, args.eval_steps, args.seed + 10_000, policy
  )
  dagger_evaluation = rollout(
    module,
    args.eval_envs,
    args.eval_steps,
    args.seed + 10_000,
    checkpoint=dagger_checkpoint,
  )
  dagger["evaluation"] = dagger_evaluation
  checkpoint_comparison = {
    "protocol": "hear-dagger-ppo-checkpoint-comparison-v1",
    "identical_seed": args.seed + 10_000,
    "identical_environment_count": args.eval_envs,
    "identical_control_steps": args.eval_steps,
    "ppo_minus_dagger": {
      "success_rate": (
        ppo_evaluation["success_rate"] - dagger_evaluation["success_rate"]
      ),
      "minimum_mean_wrist_error_m": (
        ppo_evaluation["wrist_position_error_m"]["minimum_mean_over_episode"]
        - dagger_evaluation["wrist_position_error_m"]["minimum_mean_over_episode"]
      ),
      "final_mean_wrist_error_m": (
        ppo_evaluation["wrist_position_error_m"]["final_mean"]
        - dagger_evaluation["wrist_position_error_m"]["final_mean"]
      ),
      "minimum_support_margin_m": (
        ppo_evaluation["dynamic_com"]["minimum_support_margin_m"]
        - dagger_evaluation["dynamic_com"]["minimum_support_margin_m"]
      ),
      "maximum_foot_slip_speed_m_s": (
        ppo_evaluation["dynamic_com"]["maximum_foot_slip_speed_m_s"]
        - dagger_evaluation["dynamic_com"]["maximum_foot_slip_speed_m_s"]
      ),
      "action_clipped_element_rate": (
        ppo_evaluation["action_clipping"]["clipped_element_rate"]
        - dagger_evaluation["action_clipping"]["clipped_element_rate"]
      ),
    },
  }
  threshold = contract["evaluation"]
  retention_delta = checkpoint_comparison["ppo_minus_dagger"]
  ppo_retention_gates_passed = (
    ppo_retention["mode"] == "critic_warmup_rollout_teacher"
    and ppo_retention["protocol"] == "hear-ppo-retention-v2"
    and ppo_retention["actor_distribution"]["structurally_bounded"] is True
    and ppo_retention["rollout_teacher_label_coverage"]
      >= threshold["ppo_rollout_teacher_label_coverage_minimum"]
    and ppo_retention["final_mean_policy_action_std"]
      <= threshold["ppo_rollout_mean_action_std_maximum"]
    and retention_delta["success_rate"]
      >= threshold["ppo_success_rate_delta_minimum"]
    and retention_delta["minimum_mean_wrist_error_m"]
      <= threshold["ppo_minimum_mean_wrist_error_delta_maximum_m"]
    and retention_delta["minimum_support_margin_m"]
      >= threshold["ppo_minimum_support_margin_delta_minimum_m"]
    and retention_delta["maximum_foot_slip_speed_m_s"]
      <= threshold["ppo_maximum_foot_slip_delta_maximum_m_s"]
    and ppo_evaluation["action_clipping"]["clipped_element_rate"]
      <= threshold["ppo_action_clipped_element_rate_maximum"]
  )
  selected_checkpoint = staging / "workyard_reach_selected.pt"
  selected_source = checkpoint if ppo_retention_gates_passed else dagger_checkpoint
  shutil.copy2(selected_source, selected_checkpoint)
  selected_evaluation = (
    ppo_evaluation if ppo_retention_gates_passed else dagger_evaluation
  )
  checkpoint_selection = {
    "protocol": "hear-retention-checkpoint-selection-v1",
    "ppo_retention_gates_passed": ppo_retention_gates_passed,
    "rollback_applied": not ppo_retention_gates_passed,
    "selected_source": "ppo" if ppo_retention_gates_passed else "dagger",
    "selected_checkpoint": {
      "file": selected_checkpoint.name,
      "bytes": selected_checkpoint.stat().st_size,
      "sha256": sha256(selected_checkpoint),
    },
  }
  training = {
    "seed": args.seed,
    "environment_count": args.num_envs,
    "iterations_requested": args.iterations,
    "completed_iterations": completed_iterations,
    "dagger": dagger,
    "ppo_retention": ppo_retention,
    "checkpoint_comparison": checkpoint_comparison,
    "checkpoint_selection": checkpoint_selection,
    "steps_per_environment_per_iteration": agent_cfg.num_steps_per_env,
    "ppo_environment_steps": (
      args.num_envs * args.iterations * agent_cfg.num_steps_per_env
    ),
    "dagger_environment_steps": args.num_envs * args.dagger_steps,
    "total_environment_steps": args.num_envs * (
      args.dagger_steps + args.iterations * agent_cfg.num_steps_per_env
    ),
    "teacher": teacher_identity,
    "reach_teacher": reach_teacher_identity,
    "scalar_curve_tags": sorted(curves),
    "environment_closed": training_environment_closed,
    "checkpoint": {
      "file": checkpoint.name,
      "bytes": checkpoint.stat().st_size,
      "sha256": sha256(checkpoint),
    },
    "training_curves": {
      "file": curves_path.name,
      "bytes": curves_path.stat().st_size,
      "sha256": sha256(curves_path),
    },
  }
  archive_path = Path(args.archive).resolve()
  if archive_path.exists():
    raise FileExistsError(f"Residual training archive already exists: {archive_path}")
  archive_path.parent.mkdir(parents=True, exist_ok=True)
  with tarfile.open(archive_path, "w:gz") as archive:
    archive.add(staging, arcname="hear-workyard-residual")
  return training, selected_evaluation


def execute(args: argparse.Namespace) -> dict[str, object]:
  skip_install = os.environ.get("HEAR_WORKYARD_SKIP_DEPENDENCY_INSTALL") == "1"
  if skip_install:
    installed_mjlab = importlib.metadata.version("mjlab")
    if installed_mjlab != MJLAB_VERSION:
      raise RuntimeError(
        f"Pinned local mjlab runtime is {installed_mjlab}, expected {MJLAB_VERSION}"
      )
  else:
    run([sys.executable, "-m", "pip", "install", "--quiet", f"mjlab=={MJLAB_VERSION}"])
  os.environ.update({
    "MUJOCO_GL": "egl",
    "WANDB_MODE": "disabled",
    "PYTHONUNBUFFERED": "1",
  })

  import mujoco
  import torch
  from mjlab.tasks.registry import list_tasks

  if not torch.cuda.is_available():
    raise RuntimeError("Residual Workyard requires an NVIDIA CUDA runtime")
  extract_bundle()
  _, module = load_modules()
  if module.TASK_ID not in list_tasks():
    raise RuntimeError(f"Residual Workyard task was not registered: {module.TASK_ID}")
  contract, contract_path, leaked = validate_contract(module)
  threshold = contract["evaluation"]

  training = None
  teacher_preflight = None
  if args.mode == "smoke":
    evaluation = smoke(args, module)
  elif args.mode == "teacher":
    evaluation = teacher_rollout(args, module)
  else:
    teacher_preflight = rollout(
      module,
      min(args.num_envs, args.eval_envs, 128),
      args.rollout_steps,
      args.seed,
      use_reach_teacher=True,
    )
    preflight_error = teacher_preflight["wrist_position_error_m"]
    preflight_useful = (
      teacher_preflight["finite"] is True
      and teacher_preflight["fall_count"] == 0
      and teacher_preflight["success_rate"]
        >= threshold["analytic_teacher_preflight_success_rate_minimum"]
      and preflight_error["minimum_mean_over_episode"]
        <= threshold["analytic_teacher_preflight_minimum_mean_error_maximum_m"]
    )
    if not preflight_useful:
      raise RuntimeError(
        "Analytic reach teacher preflight rejected DAgger: "
        + json.dumps({
          "wrist_position_error_m": preflight_error,
          "success_rate": teacher_preflight["success_rate"],
          "fall_count": teacher_preflight["fall_count"],
        }, ensure_ascii=False)
      )
    training, evaluation = train(args, module, contract)
  structural_acceptance = (
    evaluation["finite"] is True
    and evaluation["environment_closed"] is True
    and evaluation["teacher"]["gradient_parameter_count"] == 0
    and evaluation["teacher"]["device"].startswith("cuda")
    and evaluation["teacher"]["actuation_protocol"]
      == module.TEACHER_ACTUATION_PROTOCOL
    and evaluation["teacher"]["source_actuation_joint_count"] == 15
    and evaluation["teacher"]["task_tracking_actuation_joint_count"] == 14
    and evaluation["teacher"]["frozen_actuation_protocol"]
      == module.FROZEN_TEACHER_ACTUATION_PROTOCOL
    and evaluation["teacher"]["upper_body_actuation_protocol"]
      == module.UPPER_BODY_ACTUATION_PROTOCOL
    and evaluation["reach_teacher"]["actor_observation_exposure"] is False
    and evaluation["reach_teacher"]["execution_authority"] == "none"
    and evaluation["reach_teacher"]["label_coverage"] == 1.0
    and evaluation["composition"]["maximum_frozen_joint_command_error"] <= 1e-6
    and evaluation["composition"]["maximum_upper_body_command_error"] <= 1e-6
    and evaluation["composition"]["maximum_fixed_open_hand_target_error_rad"] <= 1e-6
  )
  held_out_episode_requirement_met = args.eval_envs >= threshold["episodes"]
  ppo_retention_passed = False
  if training is not None:
    retention_delta = training["checkpoint_comparison"]["ppo_minus_dagger"]
    ppo_retention_passed = (
      training["checkpoint_selection"]["ppo_retention_gates_passed"] is True
      and training["ppo_retention"]["mode"] == "critic_warmup_rollout_teacher"
      and training["ppo_retention"]["protocol"] == "hear-ppo-retention-v2"
      and training["ppo_retention"]["actor_distribution"][
        "structurally_bounded"
      ] is True
      and training["ppo_retention"]["rollout_teacher_label_coverage"]
        >= threshold["ppo_rollout_teacher_label_coverage_minimum"]
      and training["ppo_retention"]["final_mean_policy_action_std"]
        <= threshold["ppo_rollout_mean_action_std_maximum"]
      and retention_delta["success_rate"]
        >= threshold["ppo_success_rate_delta_minimum"]
      and retention_delta["minimum_mean_wrist_error_m"]
        <= threshold["ppo_minimum_mean_wrist_error_delta_maximum_m"]
      and retention_delta["minimum_support_margin_m"]
        >= threshold["ppo_minimum_support_margin_delta_minimum_m"]
      and retention_delta["maximum_foot_slip_speed_m_s"]
        <= threshold["ppo_maximum_foot_slip_delta_maximum_m_s"]
      and evaluation["action_clipping"]["clipped_element_rate"]
        <= threshold["ppo_action_clipped_element_rate_maximum"]
    )
  selected_checkpoint_safety_gate_passed = False
  if training is not None:
    selection = training["checkpoint_selection"]
    selected_checkpoint_safety_gate_passed = (
      (
        ppo_retention_passed
        and selection["selected_source"] == "ppo"
        and selection["rollback_applied"] is False
      )
      or (
        not ppo_retention_passed
        and selection["selected_source"] == "dagger"
        and selection["rollback_applied"] is True
      )
    )
  phase_one_accepted = (
    args.mode == "train"
    and held_out_episode_requirement_met
    and evaluation["success_rate"] >= threshold["wrist_target_success_rate_minimum"]
    and evaluation["fall_rate"] <= threshold["fall_rate_maximum"]
    and evaluation["frozen_teacher_joint_rms_error_rad"]
      <= threshold["teacher_frozen_joint_rms_error_maximum_rad"]
    and evaluation["maximum_joint_velocity_rad_s"]
      <= threshold["maximum_joint_velocity_rad_s"]
    and evaluation["maximum_joint_actuator_force_ratio"]
      <= threshold["maximum_requested_effort_utilization"]
    and evaluation["attribution"]["teacher_frame_ratio"]
      >= threshold["teacher_frame_ratio_minimum"]
    and evaluation["attribution"]["upper_body_residual_frame_ratio"]
      >= threshold["upper_body_residual_frame_ratio_minimum"]
    and training is not None
    and training["dagger"]["label_coverage"]
      >= threshold["reach_teacher_label_coverage_minimum"]
    and selected_checkpoint_safety_gate_passed
    and evaluation["dynamic_com"]["minimum_support_margin_m"]
      >= threshold["minimum_support_margin_m"]
    and evaluation["dynamic_com"]["maximum_capture_point_norm_m"]
      <= threshold["maximum_capture_point_norm_m"]
    and evaluation["dynamic_com"]["maximum_foot_planar_displacement_m"]
      <= threshold["maximum_foot_planar_displacement_m"]
    and evaluation["dynamic_com"]["maximum_foot_slip_speed_m_s"]
      <= threshold["maximum_foot_slip_speed_m_s"]
    and evaluation["dynamic_com"]["double_support_loss_rate"]
      <= threshold["double_support_loss_rate_maximum"]
    and evaluation["dynamic_com"]["no_foot_contact_rate"]
      <= threshold["no_foot_contact_rate_maximum"]
  )
  report = {
    "protocol": "hear-workyard-residual-run-v4",
    "ready": True,
    "mode": args.mode,
    "created_at": datetime.now(timezone.utc).isoformat(),
    "task": module.TASK_ID,
    "framework": {
      "mjlab": importlib.metadata.version("mjlab"),
      "mujoco": mujoco.__version__,
      "torch": torch.__version__,
      "algorithm": "online DAgger + PPO (RSL-RL)" if args.mode == "train" else None,
    },
    "accelerator": {
      "device": torch.cuda.get_device_name(0),
      "cuda_version": torch.version.cuda,
    },
    "execution": {
      "profile": args.execution_profile,
      "dependency_install": "preinstalled_pinned" if skip_install else "ephemeral_pip",
    },
    "bundle": {
      "sha256": sha256(REMOTE_BUNDLE),
      "contract_sha256": sha256(contract_path),
      "environment_sha256": sha256(
        REMOTE_ROOT / "training" / "workyard_residual_mjlab_env.py"
      ),
      "retention_algorithm_sha256": sha256(
        REMOTE_ROOT / "training" / "hear_retention_ppo.py"
      ),
    },
    "contract": {
      "protocol": contract["protocol"],
      "observation_size": contract["observation"]["size"],
      "action_size": contract["action"]["size"],
      "phase": contract["student"]["phase"],
      "teacher_state_directly_exposed": False,
      "forbidden_direct_references": leaked,
      "cpu_round_trip_per_control_step": False,
      "teacher_actuation_protocol": module.TEACHER_ACTUATION_PROTOCOL,
      "reach_teacher_protocol": module.REACH_TEACHER_PROTOCOL,
      "dynamic_com_protocol": module.DYNAMIC_COM_PROTOCOL,
      "entry_state_protocol": contract["student"]["entry_state"]["protocol"],
    },
    "training": training,
    "teacher_preflight": teacher_preflight,
    "evaluation": evaluation,
    "acceptance": {
      "scope": (
        "structural_smoke"
        if args.mode == "smoke"
        else "analytic_teacher_rollout"
        if args.mode == "teacher"
        else "dagger_ppo_reach_candidate"
      ),
      "structural_invariants_passed": structural_acceptance,
      "held_out_episode_requirement": threshold["episodes"],
      "held_out_episode_requirement_met": held_out_episode_requirement_met,
      "ppo_retention_passed": ppo_retention_passed,
      "selected_checkpoint_safety_gate_passed": (
        selected_checkpoint_safety_gate_passed
      ),
      "phase_one_accepted": phase_one_accepted,
      "hand_checkpoint_expansion_authorized": phase_one_accepted,
      "waist_checkpoint_expansion_authorized": False,
      "deployment_accepted": False,
      "reason": (
        "smoke validates composition and data flow only"
        if args.mode == "smoke"
        else "analytic teacher rollout validates reach guidance without training"
        if args.mode == "teacher"
        else (
          (
            "held-out reach gate passed with selected "
            f"{training['checkpoint_selection']['selected_source']} checkpoint; "
            "hand contact/grasp expansion may begin"
          )
          if phase_one_accepted
          else "held-out phase-one thresholds are not yet satisfied"
        )
      ),
    },
  }
  if not structural_acceptance:
    raise RuntimeError("Residual Workyard structural acceptance failed")
  return report


def main() -> None:
  report: dict[str, object]
  mode = "unknown"
  try:
    args = parse_args()
    mode = args.mode
    report = execute(args)
  except BaseException as error:
    report = {
      "protocol": "hear-workyard-residual-run-v4",
      "ready": False,
      "mode": mode,
      "created_at": datetime.now(timezone.utc).isoformat(),
      "error": {
        "type": type(error).__name__,
        "message": str(error),
        "traceback": traceback.format_exc(),
      },
    }
  REMOTE_REPORT.parent.mkdir(parents=True, exist_ok=True)
  REMOTE_REPORT.write_text(
    json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
  )
  print(json.dumps(report, ensure_ascii=False))
  if not report["ready"]:
    raise SystemExit(1)


if __name__ == "__main__":
  main()
