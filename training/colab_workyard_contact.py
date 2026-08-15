"""Train and independently qualify the HEAR 8D contact hand policy.

The 29D whole-body reach policy, locomotion reference, and terminal DLS
executor are immutable Harness-owned components. Online DAgger and retention PPO optimize
only the bounded 8D hand actor.  Checkpoint selection uses a private comparison
split; the selected checkpoint is then evaluated once on the independent
500-seed gate declared by workyard-contact-task-v1.json.
"""

from __future__ import annotations

import argparse
import base64
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
DEVICE = "cuda:0"
EXECUTION_ROOT = Path(os.environ.get("HEAR_WORKYARD_EXECUTION_ROOT", "/content"))
REMOTE_BUNDLE = Path(os.environ.get(
  "HEAR_WORKYARD_BUNDLE",
  str(EXECUTION_ROOT / "hear-workyard-contact-bundle.tar.gz"),
))
REMOTE_CONFIG = Path(os.environ.get(
  "HEAR_WORKYARD_CONFIG",
  str(EXECUTION_ROOT / "hear-workyard-contact-config.json"),
))
REMOTE_ROOT = Path(os.environ.get(
  "HEAR_WORKYARD_SOURCE_ROOT",
  str(EXECUTION_ROOT / "hear-workyard-contact-source"),
))
REMOTE_REPORT = Path(os.environ.get(
  "HEAR_WORKYARD_REPORT",
  str(EXECUTION_ROOT / "hear-workyard-contact-report.json"),
))


class ContactQualificationError(RuntimeError):
  """Carries the rejected physical gate into the downloaded failure report."""

  def __init__(self, message: str, evidence: dict[str, object]):
    super().__init__(message)
    self.evidence = evidence


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser()
  parser.add_argument(
    "--mode", choices=("smoke", "teacher", "pilot", "train"), default="smoke"
  )
  parser.add_argument("--iterations", type=int, default=1_000)
  parser.add_argument("--dagger-steps", type=int, default=800)
  parser.add_argument("--dagger-learning-rate", type=float, default=3.0e-4)
  parser.add_argument("--dagger-beta-initial", type=float, default=1.0)
  parser.add_argument("--dagger-beta-final", type=float, default=0.1)
  parser.add_argument("--critic-warmup-iterations", type=int, default=5)
  parser.add_argument("--ppo-actor-learning-rate", type=float, default=1.0e-5)
  parser.add_argument("--ppo-critic-learning-rate", type=float, default=3.0e-4)
  parser.add_argument("--rollout-teacher-coefficient", type=float, default=1.0)
  parser.add_argument("--teacher-maximum-action-std", type=float, default=0.15)
  parser.add_argument("--teacher-dispersion-coefficient", type=float, default=1.0)
  parser.add_argument("--num-envs", type=int, default=2_048)
  parser.add_argument("--rollout-steps", type=int, default=64)
  parser.add_argument("--comparison-envs", type=int, default=128)
  parser.add_argument("--comparison-steps", type=int, default=400)
  parser.add_argument("--final-eval-envs", type=int, default=500)
  parser.add_argument("--final-eval-steps", type=int, default=400)
  parser.add_argument("--seed", type=int, default=42)
  parser.add_argument("--episode-seeds", type=int, nargs="+", default=None)
  parser.add_argument("--execution-profile", default="colab-pro-l4-formal-v1")
  parser.add_argument("--output-root", default="/content/hear-workyard-contact")
  parser.add_argument(
    "--archive", default="/content/hear-workyard-contact-artifacts.tar.gz"
  )
  parser.add_argument(
    "--artifact-stream",
    action=argparse.BooleanOptionalAction,
    default=False,
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
    "iterations", "dagger_steps", "num_envs", "rollout_steps",
    "comparison_envs", "comparison_steps", "final_eval_envs",
    "final_eval_steps",
  ):
    if getattr(args, name) <= 0:
      parser.error(f"--{name.replace('_', '-')} must be positive")
  if args.seed < 0 or args.critic_warmup_iterations < 0:
    parser.error("seed and critic warm-up iterations must be non-negative")
  for name in (
    "dagger_learning_rate", "ppo_actor_learning_rate",
    "ppo_critic_learning_rate", "rollout_teacher_coefficient",
    "teacher_maximum_action_std", "teacher_dispersion_coefficient",
  ):
    if getattr(args, name) <= 0.0:
      parser.error(f"--{name.replace('_', '-')} must be positive")
  if args.teacher_maximum_action_std > 1.0:
    parser.error("--teacher-maximum-action-std must not exceed 1")
  if not 0.0 <= args.dagger_beta_final <= args.dagger_beta_initial <= 1.0:
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
    raise FileNotFoundError(f"Contact Workyard bundle is missing: {REMOTE_BUNDLE}")
  REMOTE_ROOT.mkdir(parents=True, exist_ok=False)
  with tarfile.open(REMOTE_BUNDLE, "r:gz") as archive:
    root = REMOTE_ROOT.resolve()
    for member in archive.getmembers():
      target = (REMOTE_ROOT / member.name).resolve()
      if target != root and root not in target.parents:
        raise RuntimeError(f"Unsafe contact bundle member: {member.name}")
    archive.extractall(REMOTE_ROOT, filter="data")


def load_module() -> ModuleType:
  training_root = REMOTE_ROOT / "training"
  sys.path.insert(0, str(training_root))
  import workyard_contact_mjlab_env as contact

  return contact


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
    raise RuntimeError(f"Contact PPO produced no TensorBoard curves under {log_dir}")
  return curves


def _read_json(path: Path) -> dict[str, object]:
  value = json.loads(path.read_text(encoding="utf-8"))
  if not isinstance(value, dict):
    raise TypeError(f"Expected a JSON object: {path}")
  return value


def validate_contract(
  module: ModuleType,
  require_qualified_preflight: bool,
) -> tuple[dict[str, object], Path, dict[str, object]]:
  contract_path = REMOTE_ROOT / "training" / "workyard-contact-task-v1.json"
  contract = _read_json(contract_path)
  if contract.get("protocol") != "hear-workyard-contact-training-contract-v2":
    raise RuntimeError("Contact Workyard contract protocol is invalid")
  if contract["environment"]["task_id"] != module.TASK_ID:
    raise RuntimeError("Contact task registration disagrees with its contract")
  if (
    contract["learner"]["observation"]["size"] != module.HAND_OBSERVATION_SIZE
    or contract["learner"]["action"]["size"] != module.HAND_ACTION_SIZE
    or contract["composition"]["logical_composed_action_size"]
      != module.COMPOSED_ACTION_SIZE
  ):
    raise RuntimeError("Contact observation/action composition drifted")
  if tuple(contract["learner"]["action"]["names"]) != module.HAND_SYNERGY_NAMES:
    raise RuntimeError("Contact hand synergy order drifted")
  if contract["environment"]["excluded_stages"] != ["lift", "carry", "place"]:
    raise RuntimeError("Contact learner expanded beyond verified grasp")
  if (
    contract["harness_executor"]["hand_contact_solref_time_constant_s"]
    != module.HAND_CONTACT_SOLREF_TIME_CONSTANT_S
  ):
    raise RuntimeError("Contact collision-skin compliance drifted")

  observation_source = inspect.getsource(module.WorkyardContactObservation.__call__)
  leaked = [
    name for name in contract["learner"]["observation"]["forbidden_terms"]
    if name in observation_source
  ]
  if leaked:
    raise RuntimeError("Teacher-private contact state leaked to the actor: " + ", ".join(leaked))
  control_sources = "\n".join((
    inspect.getsource(module.FrozenReachPolicy.infer),
    inspect.getsource(module.WorkyardContactAction.compute_hand_teacher_action),
    inspect.getsource(module.WorkyardContactAction.process_actions),
  ))
  cpu_round_trips = [
    token for token in (".cpu(", ".numpy(", ".tolist(")
    if token in control_sources
  ]
  if cpu_round_trips:
    raise RuntimeError("Contact control loop contains a CPU round trip: " + ", ".join(cpu_round_trips))

  qualified = contract["qualified_inputs"]
  locomotion = qualified["locomotion_teacher"]
  locomotion_root = REMOTE_ROOT / locomotion["root"]
  locomotion_jit = locomotion_root / locomotion["jit"]
  locomotion_report_path = locomotion_root / locomotion["report"]
  reach = qualified["reach_policy"]
  reach_root = REMOTE_ROOT / reach["root"]
  reach_jit = reach_root / reach["jit"]
  reach_report_path = reach_root / reach["report"]
  preflight_path = REMOTE_ROOT / qualified["analytic_teacher_preflight"]["report"]
  required_paths = [
    locomotion_jit, locomotion_report_path, reach_jit, reach_report_path,
  ]
  if require_qualified_preflight:
    required_paths.append(preflight_path)
  for path in required_paths:
    if not path.is_file():
      raise FileNotFoundError(f"Qualified contact input is missing: {path}")

  locomotion_report = _read_json(locomotion_report_path)
  locomotion_identity = locomotion_report.get("teacher_jit")
  if not isinstance(locomotion_identity, dict) or (
    locomotion_identity.get("file") != locomotion_jit.name
    or locomotion_identity.get("bytes") != locomotion_jit.stat().st_size
    or locomotion_identity.get("sha256") != sha256(locomotion_jit)
    or locomotion_identity.get("input_size") != 99
    or locomotion_identity.get("output_size") != 29
    or locomotion_identity.get("batch_dynamic") is not True
  ):
    raise RuntimeError("Frozen locomotion identity is invalid")
  reach_report = _read_json(reach_report_path)
  reach_source = reach_report.get("source")
  reach_identity = reach_report.get("policy")
  reach_deployment = reach_report.get("deployment")
  if not isinstance(reach_source, dict) or not isinstance(reach_identity, dict) or (
    reach_report.get("protocol") != reach["protocol"]
    or not isinstance(reach_deployment, dict)
    or reach_deployment.get("protocol")
      != "hear-typescript-mujoco-reach-deployment-gate-v1"
    or reach_deployment.get("accepted") is not True
    or reach_deployment.get("controller_mode") != "learned_policy_only"
    or reach_deployment.get("terminal_assistance_step_count") != 0
    or reach_deployment.get("minimum_support_margin_m", float("-inf")) < -0.04
    or reach_deployment.get("maximum_foot_planar_displacement_m", float("inf")) > 0.08
    or reach_deployment.get("maximum_foot_slip_speed_m_s", float("inf")) > 0.20
    or reach_deployment.get("double_support_loss_rate_maximum", float("inf")) > 0.10
    or reach_deployment.get("no_foot_contact_rate_maximum", float("inf")) > 0.01
    or reach["source_checkpoint_sha256"] is None
    or reach["jit_sha256"] is None
    or reach_source.get("checkpoint_sha256") != reach["source_checkpoint_sha256"]
    or reach_identity.get("file") != reach_jit.name
    or reach_identity.get("bytes") != reach_jit.stat().st_size
    or reach_identity.get("sha256") != sha256(reach_jit)
    or reach_identity.get("sha256") != reach["jit_sha256"]
    or reach_identity.get("input")
      != "hear-workyard-whole-body-reach-observation-v5"
    or reach_identity.get("input_size") != module.REACH_OBSERVATION_SIZE
    or reach_identity.get("output") != "bounded-whole-body-reach-mean"
    or reach_identity.get("output_size") != module.reach.ACTION_SIZE
    or reach_identity.get("batch_dynamic") is not True
    or reach_identity.get("gradient_parameter_count") != 0
  ):
    raise RuntimeError("Qualified whole-body reach identity is invalid or unpinned")
  if require_qualified_preflight:
    preflight = _read_json(preflight_path)
    gate = preflight.get("gate")
    evaluation = preflight.get("evaluation")
    preflight_contract = preflight.get("contract")
    frozen_reach = None
    if isinstance(evaluation, dict):
      frozen_reach = evaluation.get("frozen_reach")
    if not isinstance(frozen_reach, dict) and isinstance(preflight_contract, dict):
      frozen_reach = preflight_contract.get("frozen_reach")
    if not isinstance(gate, dict) or (
      gate.get("protocol") != qualified["analytic_teacher_preflight"]["protocol"]
      or gate.get("passed") is not True
      or not isinstance(gate.get("checks"), dict)
      or not all(gate["checks"].values())
      or not isinstance(frozen_reach, dict)
      or frozen_reach.get("gradient_parameter_count") != 0
      or frozen_reach.get("jit_sha256") != sha256(reach_jit)
    ):
      raise RuntimeError(
        "Bundled analytic contact preflight is not qualified for pinned Reach"
      )
  # The inherited G1 spec factory resolves its qualified locomotion identity
  # before an environment action term exists, so it cannot see the per-run
  # action_cfg paths configured below.  Mirror only the already hash-validated
  # files into that historical runtime location inside the ephemeral bundle.
  runtime_locomotion_root = (
    REMOTE_ROOT / "assets" / "humanoid" / "controllers"
    / "mjlab-g1-velocity"
  )
  runtime_locomotion_root.mkdir(parents=True, exist_ok=True)
  runtime_locomotion_jit = runtime_locomotion_root / locomotion_jit.name
  runtime_locomotion_report = runtime_locomotion_root / locomotion_report_path.name
  shutil.copy2(locomotion_jit, runtime_locomotion_jit)
  shutil.copy2(locomotion_report_path, runtime_locomotion_report)
  if (
    sha256(runtime_locomotion_jit) != sha256(locomotion_jit)
    or sha256(runtime_locomotion_report) != sha256(locomotion_report_path)
  ):
    raise RuntimeError("Ephemeral locomotion runtime mirror changed artifact identity")
  artifact_paths = {
    "locomotion_jit": locomotion_jit,
    "locomotion_report": locomotion_report_path,
    "reach_jit": reach_jit,
    "reach_report": reach_report_path,
    "analytic_preflight": preflight_path if require_qualified_preflight else None,
  }
  evidence = {
    "locomotion_jit_sha256": sha256(locomotion_jit),
    "reach_jit_sha256": sha256(reach_jit),
    "analytic_preflight_sha256": (
      sha256(preflight_path) if require_qualified_preflight else None
    ),
    "forbidden_observation_references": leaked,
    "cpu_round_trip_per_control_step": False,
    "paths": artifact_paths,
  }
  return contract, contract_path, evidence


def configure_artifacts(env_cfg, evidence: dict[str, object]) -> None:
  paths = evidence["paths"]
  action_cfg = env_cfg.actions["workyard"]
  action_cfg.teacher_jit_path = str(paths["locomotion_jit"])
  action_cfg.teacher_report_path = str(paths["locomotion_report"])
  action_cfg.reach_policy_jit_path = str(paths["reach_jit"])
  action_cfg.reach_policy_report_path = str(paths["reach_report"])


def numerically_guarded_env_class(base_class, module: ModuleType, contract):
  """Recover a single corrupt vector world without exposing NaNs to PPO.

  MuJoCo Warp can very rarely lose one randomized world after millions of
  contact steps.  A 2048-world learner must treat that as a scoped episode
  failure, not let one row poison a global gradient update.  Recovery remains
  bounded, counted, and forbidden during held-out evaluation.
  """
  import torch

  guard = contract["training"]["numerical_guard"]
  maximum_count = int(guard["maximum_recovery_count"])
  maximum_rate = float(guard["maximum_recovery_rate"])

  class NumericallyGuardedContactEnv(base_class):
    def __init__(self, *args, **kwargs):
      super().__init__(*args, **kwargs)
      self.numerical_recovery_count = 0
      self.numerical_recovery_diagnostics: list[dict[str, object]] = []

    def step(self, action):
      result = super().step(action)
      observations, rewards, terminated, time_outs, extras = result
      invalid = ~torch.isfinite(rewards)
      invalid |= module._nonfinite_physics_state(self)
      for value in observations.values():
        invalid |= ~torch.isfinite(value).reshape(self.num_envs, -1).all(dim=-1)
      if not bool(invalid.any().item()):
        return result

      environment_ids = torch.where(invalid)[0]
      entry = {
        "common_step": int(self.common_step_counter),
        "environment_ids": environment_ids[:32].cpu().tolist(),
        "environment_count": int(environment_ids.numel()),
        "episode_lengths": self.episode_length_buf[environment_ids[:32]].cpu().tolist(),
        "nonfinite_reward_count": int((~torch.isfinite(rewards)).sum().item()),
        "nonfinite_actor_observation_count": int((
          ~torch.isfinite(observations["actor"])
        ).sum().item()),
        "nonfinite_critic_observation_count": int((
          ~torch.isfinite(observations["critic"])
        ).sum().item()),
        "maximum_finite_qvel": module._maximum_finite_absolute(self.sim.data.qvel),
        "maximum_finite_qacc": module._maximum_finite_absolute(self.sim.data.qacc),
      }
      self.numerical_recovery_count += int(environment_ids.numel())
      if len(self.numerical_recovery_diagnostics) < 32:
        self.numerical_recovery_diagnostics.append(entry)
      processed_environment_steps = max(
        1, int(self.common_step_counter) * self.num_envs
      )
      recovery_rate = self.numerical_recovery_count / processed_environment_steps
      if self.numerical_recovery_count > maximum_count or recovery_rate > maximum_rate:
        raise RuntimeError(
          "Contact numerical recovery budget exhausted: "
          + json.dumps({
            "protocol": guard["protocol"],
            "count": self.numerical_recovery_count,
            "rate": recovery_rate,
            "maximum_count": maximum_count,
            "maximum_rate": maximum_rate,
            "latest": entry,
          }, ensure_ascii=False, sort_keys=True)
        )

      reset_observations, _ = self.reset(env_ids=environment_ids)
      rewards[environment_ids] = 0.0
      terminated[environment_ids] = True
      time_outs[environment_ids] = False
      extras["hear_numerical_recovery"] = {
        "protocol": guard["protocol"],
        "total_count": self.numerical_recovery_count,
        "rate": recovery_rate,
        "latest": entry,
      }
      return reset_observations, rewards, terminated, time_outs, extras

    def numerical_recovery_report(self) -> dict[str, object]:
      processed_environment_steps = max(
        1, int(self.common_step_counter) * self.num_envs
      )
      count = int(self.numerical_recovery_count)
      rate = count / processed_environment_steps
      return {
        "protocol": guard["protocol"],
        "scope": guard["scope"],
        "count": count,
        "rate": rate,
        "processed_environment_steps": processed_environment_steps,
        "maximum_recovery_count": maximum_count,
        "maximum_recovery_rate": maximum_rate,
        "within_budget": count <= maximum_count and rate <= maximum_rate,
        "diagnostics": self.numerical_recovery_diagnostics,
      }

  return NumericallyGuardedContactEnv


def contact_rollout(
  module: ModuleType,
  evidence: dict[str, object],
  num_envs: int,
  steps: int,
  episode_seed_first: int,
  *,
  episode_seeds: list[int] | None = None,
  checkpoint: Path | None = None,
  analytic_teacher: bool = False,
) -> dict[str, object]:
  import torch
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
  from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
  from workyard_observation_parity import capture_workyard_observation_fixture

  if (checkpoint is None) == (not analytic_teacher):
    raise ValueError("Contact rollout requires exactly one action source")
  if episode_seeds is not None:
    if len(episode_seeds) != num_envs or len(set(episode_seeds)) != num_envs:
      raise ValueError("Explicit episode seeds must be unique and match num_envs")
    if any(seed < 0 for seed in episode_seeds):
      raise ValueError("Explicit episode seeds must be non-negative")
  evaluation_seed = episode_seeds[0] if episode_seeds else episode_seed_first
  torch.manual_seed(evaluation_seed)
  env_cfg = load_env_cfg(module.TASK_ID, play=True)
  env_cfg.scene.num_envs = num_envs
  env_cfg.seed = evaluation_seed
  if episode_seeds is None:
    env_cfg.commands["workyard"].evaluation_episode_seed_base = episode_seed_first
  else:
    env_cfg.commands["workyard"].evaluation_episode_seeds = tuple(episode_seeds)
  configure_artifacts(env_cfg, evidence)
  agent_cfg = load_rl_cfg(module.TASK_ID)
  raw_env = ManagerBasedRlEnv(cfg=env_cfg, device=DEVICE)
  env = RslRlVecEnvWrapper(raw_env, clip_actions=agent_cfg.clip_actions)
  environment_closed = False
  evaluation_runner = None
  try:
    policy = None
    if checkpoint is not None:
      runner_cls = load_runner_cls(module.TASK_ID) or MjlabOnPolicyRunner
      evaluation_runner = runner_cls(env, asdict(agent_cfg), device=DEVICE)
      evaluation_runner.load(
        str(checkpoint),
        load_cfg={"actor": True},
        strict=True,
        map_location=DEVICE,
      )
      policy = evaluation_runner.get_inference_policy(device=DEVICE)
    observations = env.get_observations()
    expected_shape = (num_envs, module.HAND_OBSERVATION_SIZE)
    if tuple(observations["actor"].shape) != expected_shape:
      raise RuntimeError(f"Contact actor observation is {observations['actor'].shape}")
    action_term = module._contact_action(raw_env)
    command = module.base._workyard_command(raw_env)
    initial_active_hand = command.active_hand.clone()
    expected_seeds = torch.tensor(
      episode_seeds,
      dtype=torch.long,
      device=DEVICE,
    ) if episode_seeds is not None else torch.arange(
      episode_seed_first,
      episode_seed_first + num_envs,
      dtype=torch.long,
      device=DEVICE,
    )
    hand_sensors = (
      module.base._contact_sensor(raw_env, "left_hand_object_contact"),
      module.base._contact_sensor(raw_env, "right_hand_object_contact"),
    )
    primary_names = tuple(tuple(sensor.primary_names) for sensor in hand_sensors)
    primary_seen = [
      torch.zeros(
        len(sensor.primary_names) * sensor.cfg.num_slots,
        dtype=torch.bool,
        device=DEVICE,
      )
      for sensor in hand_sensors
    ]
    active = torch.ones(num_envs, dtype=torch.bool, device=DEVICE)
    success = torch.zeros_like(active)
    lost = torch.zeros_like(active)
    fell = torch.zeros_like(active)
    non_foot_ground = torch.zeros_like(active)
    numerical_instability = torch.zeros_like(active)
    opposing_seen = torch.zeros_like(active)
    reached_contact = torch.zeros_like(active)
    reached_pocket = torch.zeros_like(active)
    first_contact_step = torch.full(
      (num_envs,), -1, dtype=torch.long, device=DEVICE
    )
    first_pocket_step = torch.full_like(first_contact_step, -1)
    first_opposing_step = torch.full_like(first_contact_step, -1)
    first_thumb_latch_step = torch.full_like(first_contact_step, -1)
    first_opposing_digit_latch_step = torch.full_like(first_contact_step, -1)
    first_thumb_latch_force = torch.zeros(num_envs, device=DEVICE)
    first_opposing_digit_latch_force = torch.zeros(num_envs, device=DEVICE)
    first_thumb_latch_rod_local = torch.zeros((num_envs, 3), device=DEVICE)
    first_opposing_digit_latch_rod_local = torch.zeros(
      (num_envs, 3), device=DEVICE
    )
    first_thumb_latch_coordination = torch.zeros((num_envs, 4), device=DEVICE)
    first_opposing_digit_latch_coordination = torch.zeros(
      (num_envs, 4), device=DEVICE
    )
    success_step = torch.full_like(first_contact_step, -1)
    last_active_stage = torch.zeros(
      num_envs, dtype=torch.long, device=DEVICE
    )
    last_active_wrist_delta = torch.zeros(
      (num_envs, 3), dtype=torch.float32, device=DEVICE
    )
    last_active_bearing_error = torch.zeros(
      num_envs, dtype=torch.float32, device=DEVICE
    )
    last_active_axis_error = torch.zeros_like(last_active_bearing_error)
    last_active_coordination = torch.zeros(
      (num_envs, 4), dtype=torch.float32, device=DEVICE
    )
    last_active_authority_saturation = torch.zeros(
      (num_envs, 7), dtype=torch.bool, device=DEVICE
    )
    last_active_soft_limit_saturation = torch.zeros_like(
      last_active_authority_saturation
    )
    last_active_command_lead_saturation = torch.zeros_like(
      last_active_authority_saturation
    )
    last_active_minimum_singular_value = torch.zeros(
      num_envs, dtype=torch.float32, device=DEVICE
    )
    last_active_joint_position = torch.zeros(
      (num_envs, 7), dtype=torch.float32, device=DEVICE
    )
    last_active_authority_lower_margin = torch.zeros_like(
      last_active_joint_position
    )
    last_active_authority_upper_margin = torch.zeros_like(
      last_active_joint_position
    )
    last_active_primary_delta = torch.zeros_like(last_active_joint_position)
    last_active_secondary_delta = torch.zeros_like(last_active_joint_position)
    last_active_projected_posture_delta = torch.zeros_like(
      last_active_joint_position
    )
    last_active_joint_correction = torch.zeros_like(last_active_joint_position)
    last_active_instantaneous_target = torch.zeros_like(
      last_active_joint_position
    )
    last_active_authority_limited_target = torch.zeros_like(
      last_active_joint_position
    )
    precontact_object_contact = torch.zeros_like(active)
    base_assist_seen = torch.zeros_like(active)
    maximum_force = torch.zeros(num_envs, device=DEVICE)
    force_peak_step = torch.full(
      (num_envs,), -1, dtype=torch.long, device=DEVICE
    )
    force_peak_stage = torch.zeros(
      num_envs, dtype=torch.long, device=DEVICE
    )
    force_peak_coordination = torch.zeros(num_envs, device=DEVICE)
    force_peak_opposed = torch.zeros_like(active)
    force_peak_hand_target_lag = torch.zeros(num_envs, device=DEVICE)
    force_peak_wrist_delta = torch.zeros((num_envs, 3), device=DEVICE)
    force_peak_bearing_error = torch.zeros(num_envs, device=DEVICE)
    force_peak_axis_error = torch.zeros(num_envs, device=DEVICE)
    force_peak_measured_rod_local = torch.zeros((num_envs, 3), device=DEVICE)
    force_peak_pocket_local_error = torch.zeros(num_envs, device=DEVICE)
    force_peak_target_offset_drift = torch.zeros(num_envs, device=DEVICE)
    force_peak_wrist_orientation_drift = torch.zeros(num_envs, device=DEVICE)
    pocket_entry_valid = torch.zeros_like(active)
    pocket_entry_target_offset_w = torch.zeros((num_envs, 3), device=DEVICE)
    pocket_entry_wrist_quat_w = torch.zeros((num_envs, 4), device=DEVICE)
    pocket_entry_wrist_quat_w[:, 0] = 1.0
    maximum_stage = torch.zeros(num_envs, dtype=torch.long, device=DEVICE)
    reward_sum = torch.zeros(num_envs, device=DEVICE)
    minimum_opposing_normal_dot = 1.0
    maximum_requested_action = 0.0
    maximum_applied_action = 0.0
    maximum_unauthorized_action = 0.0
    maximum_unauthorized_requested_action = 0.0
    maximum_inactive_coordination = 0.0
    maximum_inactive_arm_correction = 0.0
    maximum_outside_contact_correction = 0.0
    finite = True
    termination_count = 0
    completed_steps = 0
    parity_fixtures: list[dict[str, object]] = []
    parity_captured: set[tuple[str, str]] = set()
    parity_indexes: dict[str, int] = {}
    if analytic_teacher:
      for side_index, side in enumerate(("left", "right")):
        indexes = torch.where(initial_active_hand == side_index)[0]
        if indexes.numel():
          parity_indexes[side] = int(indexes[0].item())

    with torch.inference_mode():
      for step_index in range(steps):
        active_before = active.clone()
        if not bool(active_before.any().item()):
          break
        completed_steps = step_index + 1
        stage_before = command.teacher_stage.clone()
        maximum_stage = torch.maximum(maximum_stage, stage_before)
        contact_now = (
          (stage_before >= module.CONTACT_STAGE_INDEX) & active_before
        )
        pocket_now = command.contact_pocket_active & active_before
        first_contact_step = torch.where(
          contact_now & (first_contact_step < 0),
          torch.full_like(first_contact_step, step_index),
          first_contact_step,
        )
        first_pocket_step = torch.where(
          pocket_now & (first_pocket_step < 0),
          torch.full_like(first_pocket_step, step_index),
          first_pocket_step,
        )
        reached_contact |= contact_now
        reached_pocket |= pocket_now
        rows = torch.arange(num_envs, device=DEVICE)
        root_pos = command.robot.data.root_link_pos_w
        root_quat = command.robot.data.root_link_quat_w
        wrist_pose = command.robot.data.body_link_pose_w[:, command._wrist_body_ids]
        active_wrist_pose = wrist_pose[rows, command.active_hand]
        target_position_p = command.wrist_targets_pelvis.reshape(
          num_envs, 2, 7
        )[..., :3][rows, command.active_hand]
        target_position_w = root_pos + module.base.quat_apply(
          root_quat, target_position_p
        )
        rod_position_w = command.rod.data.root_link_pos_w
        newly_entered_pocket = pocket_now & ~pocket_entry_valid
        pocket_entry_target_offset_w = torch.where(
          newly_entered_pocket.unsqueeze(-1),
          target_position_w - rod_position_w,
          pocket_entry_target_offset_w,
        )
        pocket_entry_wrist_quat_w = torch.where(
          newly_entered_pocket.unsqueeze(-1),
          active_wrist_pose[:, 3:7],
          pocket_entry_wrist_quat_w,
        )
        pocket_entry_valid |= pocket_now
        active_wrist_delta = module.reach.active_wrist_position_delta(raw_env)
        active_bearing_error = action_term.reach_teacher.wrist_bearing_error[
          rows, command.active_hand
        ]
        active_axis_error = action_term.reach_teacher.wrist_axis_alignment_error[
          rows, command.active_hand
        ]
        coordination_before = action_term.coordination.reshape(num_envs, 2, 4)[
          rows, command.active_hand
        ]
        reach_teacher = action_term.reach_teacher
        active_authority_saturation = reach_teacher.authority_saturation.reshape(
          num_envs, 2, 7
        )[rows, command.active_hand]
        active_soft_limit_saturation = reach_teacher.soft_limit_saturation.reshape(
          num_envs, 2, 7
        )[rows, command.active_hand]
        active_command_lead_saturation = (
          reach_teacher.command_lead_saturation.reshape(num_envs, 2, 7)[
            rows, command.active_hand
          ]
        )
        active_minimum_singular_value = reach_teacher.minimum_singular_value[
          rows, command.active_hand
        ]
        active_joint_position = action_term._entity.data.joint_pos[
          :, action_term._body_ids[module.reach.UPPER_BODY_SLICE]
        ].reshape(num_envs, 2, 7)[rows, command.active_hand]
        active_authority_lower = reach_teacher._authority_lower[
          rows, command.active_hand
        ]
        active_authority_upper = reach_teacher._authority_upper[
          rows, command.active_hand
        ]
        active_primary_delta = reach_teacher.primary_delta[
          rows, command.active_hand
        ]
        active_secondary_delta = reach_teacher.secondary_delta[
          rows, command.active_hand
        ]
        active_projected_posture_delta = reach_teacher.projected_posture_delta[
          rows, command.active_hand
        ]
        active_joint_correction = reach_teacher.joint_correction[
          rows, command.active_hand
        ]
        active_instantaneous_target = reach_teacher.instantaneous_target[
          rows, command.active_hand
        ]
        active_authority_limited_target = reach_teacher.authority_limited_target[
          rows, command.active_hand
        ]
        last_active_stage = torch.where(
          active_before, stage_before, last_active_stage
        )
        last_active_wrist_delta = torch.where(
          active_before.unsqueeze(-1),
          active_wrist_delta,
          last_active_wrist_delta,
        )
        last_active_bearing_error = torch.where(
          active_before, active_bearing_error, last_active_bearing_error
        )
        last_active_axis_error = torch.where(
          active_before, active_axis_error, last_active_axis_error
        )
        last_active_coordination = torch.where(
          active_before.unsqueeze(-1),
          coordination_before,
          last_active_coordination,
        )
        last_active_authority_saturation = torch.where(
          active_before.unsqueeze(-1),
          active_authority_saturation,
          last_active_authority_saturation,
        )
        last_active_soft_limit_saturation = torch.where(
          active_before.unsqueeze(-1),
          active_soft_limit_saturation,
          last_active_soft_limit_saturation,
        )
        last_active_command_lead_saturation = torch.where(
          active_before.unsqueeze(-1),
          active_command_lead_saturation,
          last_active_command_lead_saturation,
        )
        last_active_minimum_singular_value = torch.where(
          active_before,
          active_minimum_singular_value,
          last_active_minimum_singular_value,
        )
        last_active_joint_position = torch.where(
          active_before.unsqueeze(-1),
          active_joint_position,
          last_active_joint_position,
        )
        last_active_authority_lower_margin = torch.where(
          active_before.unsqueeze(-1),
          active_joint_position - active_authority_lower,
          last_active_authority_lower_margin,
        )
        last_active_authority_upper_margin = torch.where(
          active_before.unsqueeze(-1),
          active_authority_upper - active_joint_position,
          last_active_authority_upper_margin,
        )
        last_active_primary_delta = torch.where(
          active_before.unsqueeze(-1),
          active_primary_delta,
          last_active_primary_delta,
        )
        last_active_secondary_delta = torch.where(
          active_before.unsqueeze(-1),
          active_secondary_delta,
          last_active_secondary_delta,
        )
        last_active_projected_posture_delta = torch.where(
          active_before.unsqueeze(-1),
          active_projected_posture_delta,
          last_active_projected_posture_delta,
        )
        last_active_joint_correction = torch.where(
          active_before.unsqueeze(-1),
          active_joint_correction,
          last_active_joint_correction,
        )
        last_active_instantaneous_target = torch.where(
          active_before.unsqueeze(-1),
          active_instantaneous_target,
          last_active_instantaneous_target,
        )
        last_active_authority_limited_target = torch.where(
          active_before.unsqueeze(-1),
          active_authority_limited_target,
          last_active_authority_limited_target,
        )
        if analytic_teacher:
          for side, fixture_index in parity_indexes.items():
            milestones = []
            if step_index == 0:
              milestones.append("initial")
            if int(stage_before[fixture_index].item()) >= module.CONTACT_STAGE_INDEX:
              milestones.append("contact")
            if bool(command.contact_pocket_active[fixture_index].item()):
              milestones.append("pocket")
            for milestone in milestones:
              key = (side, milestone)
              if key in parity_captured:
                continue
              parity_fixtures.append(capture_workyard_observation_fixture(
                module,
                raw_env,
                observations["actor"],
                fixture_index,
                int(expected_seeds[fixture_index].item()),
                step_index,
                milestone,
              ))
              parity_captured.add(key)
        if analytic_teacher:
          proposed_action = action_term.compute_hand_teacher_action()
        else:
          proposed_action = policy(observations)
        if tuple(proposed_action.shape) != (num_envs, module.HAND_ACTION_SIZE):
          raise RuntimeError(f"Contact policy action is {proposed_action.shape}")
        if not torch.isfinite(proposed_action).all():
          raise RuntimeError("Contact policy produced a non-finite action")
        if float(proposed_action.abs().max().item()) > 1.0 + 1.0e-6:
          raise RuntimeError("Contact policy escaped the structural [-1, 1] bound")
        observations, rewards, dones, _ = env.step(proposed_action)
        if not torch.isfinite(observations["actor"]).all() or not torch.isfinite(rewards).all():
          finite = False
          raise RuntimeError("Contact rollout became non-finite")
        reward_sum += rewards * active_before
        maximum_requested_action = max(
          maximum_requested_action,
          float(action_term.requested_hand_action.abs().max().item()),
        )
        maximum_applied_action = max(
          maximum_applied_action,
          float(action_term.hand_action.abs().max().item()),
        )
        unauthorized = torch.where(
          action_term.authority_mask,
          torch.zeros_like(action_term.hand_action),
          action_term.hand_action,
        )
        unauthorized_requested = torch.where(
          action_term.authority_mask,
          torch.zeros_like(action_term.requested_hand_action),
          action_term.requested_hand_action,
        )
        maximum_unauthorized_action = max(
          maximum_unauthorized_action, float(unauthorized.abs().max().item())
        )
        maximum_unauthorized_requested_action = max(
          maximum_unauthorized_requested_action,
          float(unauthorized_requested.abs().max().item()),
        )
        coordination = action_term.coordination.reshape(num_envs, 2, 4)
        inactive_coordination = coordination[rows, 1 - command.active_hand]
        maximum_inactive_coordination = max(
          maximum_inactive_coordination,
          float(inactive_coordination.abs().max().item()),
        )
        correction = action_term.contact_approach_correction_delta.reshape(
          num_envs, 2, 7
        )
        inactive_correction = correction[rows, 1 - command.active_hand]
        maximum_inactive_arm_correction = max(
          maximum_inactive_arm_correction,
          float(inactive_correction.abs().max().item()),
        )
        outside_contact = torch.where(
          action_term.contact_approach_correction_active[:, None, None],
          torch.zeros_like(correction),
          correction,
        )
        maximum_outside_contact_correction = max(
          maximum_outside_contact_correction,
          float(outside_contact.abs().max().item()),
        )
        found, force, _, opposed = module.base._hand_contact_summary(raw_env)
        active_found = found[rows, command.active_hand]
        active_force = force[rows, command.active_hand]
        active_opposed = opposed[rows, command.active_hand]
        precontact = (stage_before < module.CONTACT_STAGE_INDEX) & active_before
        precontact_object_contact |= active_found & precontact
        eligible_force = torch.where(
          active_before, active_force, torch.zeros_like(active_force)
        )
        new_force_peak = eligible_force > maximum_force
        maximum_force = torch.maximum(maximum_force, eligible_force)
        force_peak_step = torch.where(
          new_force_peak,
          torch.full_like(force_peak_step, step_index),
          force_peak_step,
        )
        force_peak_stage = torch.where(
          new_force_peak, stage_before, force_peak_stage
        )
        active_coordination_peak = coordination[
          rows, command.active_hand
        ].abs().amax(dim=-1)
        force_peak_coordination = torch.where(
          new_force_peak,
          active_coordination_peak,
          force_peak_coordination,
        )
        force_peak_opposed = torch.where(
          new_force_peak, active_opposed, force_peak_opposed
        )
        measured_hand_positions = action_term._entity.data.joint_pos[
          :, action_term._hand_ids
        ].reshape(num_envs, 2, 7)[rows, command.active_hand]
        active_hand_targets = action_term.hand_targets.reshape(
          num_envs, 2, 7
        )[rows, command.active_hand]
        active_hand_target_lag = (
          active_hand_targets - measured_hand_positions
        ).abs().amax(dim=-1)
        force_peak_hand_target_lag = torch.where(
          new_force_peak,
          active_hand_target_lag,
          force_peak_hand_target_lag,
        )
        peak_wrist_delta = module.reach.active_wrist_position_delta(raw_env)
        peak_wrist_pose = command.robot.data.body_link_pose_w[
          :, command._wrist_body_ids
        ][rows, command.active_hand]
        peak_rod_position_w = command.rod.data.root_link_pos_w
        peak_measured_rod_local = module.base.quat_apply_inverse(
          peak_wrist_pose[:, 3:7],
          peak_rod_position_w - peak_wrist_pose[:, :3],
        )
        active_thumb_latched = action_term.teacher_thumb_contact_latched[
          rows, command.active_hand
        ]
        active_opposing_digit_latched = (
          action_term.teacher_opposing_contact_latched[rows, command.active_hand]
        )
        new_thumb_latch = (
          active_thumb_latched & active_before & (first_thumb_latch_step < 0)
        )
        new_opposing_digit_latch = (
          active_opposing_digit_latched
          & active_before
          & (first_opposing_digit_latch_step < 0)
        )
        first_thumb_latch_step = torch.where(
          new_thumb_latch,
          torch.full_like(first_thumb_latch_step, step_index),
          first_thumb_latch_step,
        )
        first_opposing_digit_latch_step = torch.where(
          new_opposing_digit_latch,
          torch.full_like(first_opposing_digit_latch_step, step_index),
          first_opposing_digit_latch_step,
        )
        first_thumb_latch_force = torch.where(
          new_thumb_latch, active_force, first_thumb_latch_force
        )
        first_opposing_digit_latch_force = torch.where(
          new_opposing_digit_latch,
          active_force,
          first_opposing_digit_latch_force,
        )
        first_thumb_latch_rod_local = torch.where(
          new_thumb_latch.unsqueeze(-1),
          peak_measured_rod_local,
          first_thumb_latch_rod_local,
        )
        first_opposing_digit_latch_rod_local = torch.where(
          new_opposing_digit_latch.unsqueeze(-1),
          peak_measured_rod_local,
          first_opposing_digit_latch_rod_local,
        )
        active_coordination_after = coordination[rows, command.active_hand]
        first_thumb_latch_coordination = torch.where(
          new_thumb_latch.unsqueeze(-1),
          active_coordination_after,
          first_thumb_latch_coordination,
        )
        first_opposing_digit_latch_coordination = torch.where(
          new_opposing_digit_latch.unsqueeze(-1),
          active_coordination_after,
          first_opposing_digit_latch_coordination,
        )
        desired_pocket_local = torch.zeros_like(peak_measured_rod_local)
        desired_pocket_local[:, 0] = command.cfg.contact_pocket_forward_m
        desired_pocket_local[:, 1] = torch.where(
          command.active_hand == 0,
          torch.full_like(active_force, -command.cfg.contact_pocket_lateral_m),
          torch.full_like(active_force, command.cfg.contact_pocket_lateral_m),
        )
        desired_pocket_local[:, 2] = command.cfg.contact_pocket_vertical_m
        peak_pocket_local_error = torch.linalg.vector_norm(
          peak_measured_rod_local - desired_pocket_local,
          dim=-1,
        )
        peak_root_pos = command.robot.data.root_link_pos_w
        peak_root_quat = command.robot.data.root_link_quat_w
        peak_target_position_p = command.wrist_targets_pelvis.reshape(
          num_envs, 2, 7
        )[..., :3][rows, command.active_hand]
        peak_target_position_w = peak_root_pos + module.base.quat_apply(
          peak_root_quat, peak_target_position_p
        )
        peak_target_offset_drift = torch.linalg.vector_norm(
          (peak_target_position_w - peak_rod_position_w)
            - pocket_entry_target_offset_w,
          dim=-1,
        )
        peak_quat_dot = torch.sum(
          peak_wrist_pose[:, 3:7] * pocket_entry_wrist_quat_w,
          dim=-1,
        ).abs().clamp(0.0, 1.0)
        peak_wrist_orientation_drift = 2.0 * torch.acos(peak_quat_dot)
        peak_bearing_error = action_term.reach_teacher.wrist_bearing_error[
          rows, command.active_hand
        ]
        peak_axis_error = action_term.reach_teacher.wrist_axis_alignment_error[
          rows, command.active_hand
        ]
        force_peak_wrist_delta = torch.where(
          new_force_peak.unsqueeze(-1), peak_wrist_delta, force_peak_wrist_delta
        )
        force_peak_bearing_error = torch.where(
          new_force_peak, peak_bearing_error, force_peak_bearing_error
        )
        force_peak_axis_error = torch.where(
          new_force_peak, peak_axis_error, force_peak_axis_error
        )
        force_peak_measured_rod_local = torch.where(
          new_force_peak.unsqueeze(-1),
          peak_measured_rod_local,
          force_peak_measured_rod_local,
        )
        force_peak_pocket_local_error = torch.where(
          new_force_peak, peak_pocket_local_error, force_peak_pocket_local_error
        )
        force_peak_target_offset_drift = torch.where(
          new_force_peak,
          peak_target_offset_drift,
          force_peak_target_offset_drift,
        )
        force_peak_wrist_orientation_drift = torch.where(
          new_force_peak,
          peak_wrist_orientation_drift,
          force_peak_wrist_orientation_drift,
        )
        opposing_seen |= active_opposed & active_before
        first_opposing_step = torch.where(
          active_opposed & active_before & (first_opposing_step < 0),
          torch.full_like(first_opposing_step, step_index),
          first_opposing_step,
        )
        base_assist_seen |= command.contact_base_assist_active & active_before
        for side, sensor in enumerate(hand_sensors):
          sensor_found = sensor.data.found
          sensor_normal = sensor.data.normal
          if sensor_found is None or sensor_normal is None:
            raise RuntimeError("Contact sensor diagnostics are unavailable")
          relevant = active_before & (command.active_hand == side)
          if not bool(relevant.any().item()):
            continue
          contacted = sensor_found[relevant] > 0
          primary_seen[side] |= contacted.any(dim=0)
          normals = torch.nn.functional.normalize(
            sensor_normal[relevant], dim=-1, eps=1.0e-6
          )
          dot = torch.einsum("bik,bjk->bij", normals, normals)
          valid_pairs = contacted.unsqueeze(2) & contacted.unsqueeze(1)
          eye = torch.eye(dot.shape[-1], dtype=torch.bool, device=DEVICE).unsqueeze(0)
          valid_dot = torch.where(
            valid_pairs & ~eye,
            dot,
            torch.full_like(dot, float("inf")),
          )
          candidate = float(valid_dot.amin().item())
          if candidate != float("inf"):
            minimum_opposing_normal_dot = min(minimum_opposing_normal_dot, candidate)
        grasp_term = raw_env.termination_manager.get_term("grasp_success").bool()
        lost_term = raw_env.termination_manager.get_term("object_lost").bool()
        fall_term = raw_env.termination_manager.get_term("fall").bool()
        ground_term = raw_env.termination_manager.get_term("non_foot_ground").bool()
        numerical_term = raw_env.termination_manager.get_term(
          "numerical_instability"
        ).bool()
        success |= grasp_term & active_before
        success_step = torch.where(
          grasp_term & active_before & (success_step < 0),
          torch.full_like(success_step, step_index),
          success_step,
        )
        lost |= lost_term & active_before
        fell |= fall_term & active_before
        non_foot_ground |= ground_term & active_before
        numerical_instability |= numerical_term & active_before
        done = dones.bool() & active_before
        termination_count += int(done.sum().item())
        active &= ~dones.bool()
  finally:
    env.close()
    environment_closed = True

  top_force_count = min(10, num_envs)
  top_force_values, top_force_indexes = torch.topk(
    maximum_force, k=top_force_count
  )
  top_force_episodes = [
    {
      "episode_seed": int(expected_seeds[index].item()),
      "active_hand": (
        "left" if int(initial_active_hand[index].item()) == 0 else "right"
      ),
      "maximum_force_n": float(force.item()),
      "control_step": int(force_peak_step[index].item()),
      "teacher_stage": int(force_peak_stage[index].item()),
      "maximum_coordination": float(force_peak_coordination[index].item()),
      "maximum_joint_target_lag_rad": float(
        force_peak_hand_target_lag[index].item()
      ),
      "opposing_contact": bool(force_peak_opposed[index].item()),
      "wrist_delta_m": [
        float(value) for value in force_peak_wrist_delta[index].tolist()
      ],
      "bearing_error_rad": float(force_peak_bearing_error[index].item()),
      "axis_alignment_error_rad": float(force_peak_axis_error[index].item()),
      "measured_rod_local_m": [
        float(value) for value in force_peak_measured_rod_local[index].tolist()
      ],
      "pocket_local_error_m": float(force_peak_pocket_local_error[index].item()),
      "target_offset_drift_from_pocket_entry_m": float(
        force_peak_target_offset_drift[index].item()
      ),
      "wrist_orientation_drift_from_pocket_entry_rad": float(
        force_peak_wrist_orientation_drift[index].item()
      ),
      "success": bool(success[index].item()),
    }
    for force, index in zip(top_force_values, top_force_indexes, strict=True)
  ]
  failure_indexes = torch.where(~success)[0][:32]
  failure_episodes = [
    {
      "episode_seed": int(expected_seeds[index].item()),
      "active_hand": (
        "left" if int(initial_active_hand[index].item()) == 0 else "right"
      ),
      "last_teacher_stage": int(last_active_stage[index].item()),
      "first_contact_step": int(first_contact_step[index].item()),
      "first_pocket_step": int(first_pocket_step[index].item()),
      "first_opposing_step": int(first_opposing_step[index].item()),
      "first_thumb_latch_step": int(first_thumb_latch_step[index].item()),
      "first_opposing_digit_latch_step": int(
        first_opposing_digit_latch_step[index].item()
      ),
      "first_thumb_latch_force_n": float(
        first_thumb_latch_force[index].item()
      ),
      "first_opposing_digit_latch_force_n": float(
        first_opposing_digit_latch_force[index].item()
      ),
      "first_thumb_latch_rod_local_m": [
        float(value) for value in first_thumb_latch_rod_local[index].tolist()
      ],
      "first_opposing_digit_latch_rod_local_m": [
        float(value)
        for value in first_opposing_digit_latch_rod_local[index].tolist()
      ],
      "first_thumb_latch_coordination": [
        float(value)
        for value in first_thumb_latch_coordination[index].tolist()
      ],
      "first_opposing_digit_latch_coordination": [
        float(value)
        for value in first_opposing_digit_latch_coordination[index].tolist()
      ],
      "last_wrist_delta_m": [
        float(value) for value in last_active_wrist_delta[index].tolist()
      ],
      "last_wrist_planar_error_m": float(
        torch.linalg.vector_norm(last_active_wrist_delta[index, :2]).item()
      ),
      "last_wrist_vertical_error_m": float(
        last_active_wrist_delta[index, 2].abs().item()
      ),
      "last_bearing_error_rad": float(
        last_active_bearing_error[index].item()
      ),
      "last_axis_alignment_error_rad": float(
        last_active_axis_error[index].item()
      ),
      "last_active_coordination": [
        float(value) for value in last_active_coordination[index].tolist()
      ],
      "last_authority_saturation": [
        bool(value) for value in last_active_authority_saturation[index].tolist()
      ],
      "last_soft_limit_saturation": [
        bool(value) for value in last_active_soft_limit_saturation[index].tolist()
      ],
      "last_command_lead_saturation": [
        bool(value) for value in last_active_command_lead_saturation[index].tolist()
      ],
      "last_minimum_singular_value": float(
        last_active_minimum_singular_value[index].item()
      ),
      "last_active_joint_position_rad": [
        float(value) for value in last_active_joint_position[index].tolist()
      ],
      "last_authority_lower_margin_rad": [
        float(value)
        for value in last_active_authority_lower_margin[index].tolist()
      ],
      "last_authority_upper_margin_rad": [
        float(value)
        for value in last_active_authority_upper_margin[index].tolist()
      ],
      "last_primary_delta_rad": [
        float(value) for value in last_active_primary_delta[index].tolist()
      ],
      "last_secondary_delta_rad": [
        float(value) for value in last_active_secondary_delta[index].tolist()
      ],
      "last_projected_posture_delta_rad": [
        float(value)
        for value in last_active_projected_posture_delta[index].tolist()
      ],
      "last_joint_correction_rad": [
        float(value) for value in last_active_joint_correction[index].tolist()
      ],
      "last_instantaneous_target_rad": [
        float(value)
        for value in last_active_instantaneous_target[index].tolist()
      ],
      "last_authority_limited_target_rad": [
        float(value)
        for value in last_active_authority_limited_target[index].tolist()
      ],
      "maximum_force_n": float(maximum_force[index].item()),
      "reached_contact": bool(reached_contact[index].item()),
      "reached_pocket": bool(reached_pocket[index].item()),
      "opposing_contact": bool(opposing_seen[index].item()),
    }
    for index in failure_indexes
  ]

  def milestone_summary(side_index: int) -> dict[str, object]:
    side = initial_active_hand == side_index

    def summarize(values: torch.Tensor) -> dict[str, object]:
      reached = side & (values >= 0)
      selected = values[reached]
      return {
        "count": int(reached.sum().item()),
        "mean_control_step": (
          float(selected.float().mean().item()) if selected.numel() else None
        ),
        "maximum_control_step": (
          int(selected.max().item()) if selected.numel() else None
        ),
      }

    return {
      "contact": summarize(first_contact_step),
      "pocket": summarize(first_pocket_step),
      "opposing_contact": summarize(first_opposing_step),
      "success": summarize(success_step),
    }

  return {
    "protocol": "hear-workyard-contact-policy-evaluation-v2",
    "action_source": "analytic_teacher" if analytic_teacher else "learned_checkpoint",
    "environment_count": num_envs,
    "episode_count": num_envs,
    "control_step_limit": steps,
    "completed_control_steps": completed_steps,
    "termination_count": termination_count,
    "seed_allocation": {
      "protocol": (
        "explicit_episode_seed_list_v1"
        if episode_seeds is not None else "per_environment_deterministic_v1"
      ),
      "first": int(expected_seeds[0].item()),
      "last": int(expected_seeds[-1].item()),
      "count": int(expected_seeds.numel()),
      **(
        {"seeds": [int(seed) for seed in expected_seeds.tolist()]}
        if episode_seeds is not None else {}
      ),
    },
    "success_count": int(success.sum().item()),
    "success_rate": float(success.float().mean().item()),
    "initial_active_hand_count": {
      side: int((initial_active_hand == index).sum().item())
      for index, side in enumerate(("left", "right"))
    },
    "success_count_by_active_hand": {
      side: int((success & (initial_active_hand == index)).sum().item())
      for index, side in enumerate(("left", "right"))
    },
    "success_rate_by_active_hand": {
      side: (
        float((success & (initial_active_hand == index)).sum().item())
        / max(1, int((initial_active_hand == index).sum().item()))
      )
      for index, side in enumerate(("left", "right"))
    },
    "opposing_contact_count": int(opposing_seen.sum().item()),
    "opposing_contact_count_by_active_hand": {
      side: int((opposing_seen & (initial_active_hand == index)).sum().item())
      for index, side in enumerate(("left", "right"))
    },
    "reached_contact_count": int(reached_contact.sum().item()),
    "reached_pocket_count": int(reached_pocket.sum().item()),
    "object_lost_count": int(lost.sum().item()),
    "fall_count": int(fell.sum().item()),
    "non_foot_ground_count": int(non_foot_ground.sum().item()),
    "numerical_instability_count": int(numerical_instability.sum().item()),
    "precontact_object_contact_count": int(precontact_object_contact.sum().item()),
    "maximum_active_hand_force_n": float(maximum_force.max().item()),
    "top_force_episodes": top_force_episodes,
    "failure_episodes": failure_episodes,
    "milestones_by_active_hand": {
      side: milestone_summary(index)
      for index, side in enumerate(("left", "right"))
    },
    "observation_parity_fixtures": parity_fixtures,
    "minimum_opposing_normal_dot": minimum_opposing_normal_dot,
    "maximum_teacher_stage": int(maximum_stage.max().item()),
    "active_hand_contact_primaries_seen": {
      side: [
        name
        for slot, name in enumerate(
          name
          for name in primary_names[index]
          for _ in range(hand_sensors[index].cfg.num_slots)
        )
        if bool(primary_seen[index][slot].item())
      ]
      for index, side in enumerate(("left", "right"))
    },
    "maximum_requested_hand_action": maximum_requested_action,
    "maximum_applied_hand_action": maximum_applied_action,
    "maximum_unauthorized_hand_action": maximum_unauthorized_action,
    "maximum_unauthorized_requested_hand_action": (
      maximum_unauthorized_requested_action
    ),
    "maximum_inactive_hand_coordination": maximum_inactive_coordination,
    "maximum_inactive_arm_contact_approach_correction": (
      maximum_inactive_arm_correction
    ),
    "maximum_outside_contact_approach_correction": (
      maximum_outside_contact_correction
    ),
    "contact_base_assist_environment_count": int(base_assist_seen.sum().item()),
    "mean_reward_sum": float(reward_sum.mean().item()),
    "frozen_locomotion": action_term.teacher.identity,
    "frozen_reach": action_term.frozen_reach.identity,
    "finite": finite,
    "environment_closed": environment_closed,
  }


def analytic_preflight_gate(
  evaluation: dict[str, object], contract: dict[str, object]
) -> dict[str, object]:
  threshold = contract["qualified_inputs"]["analytic_teacher_preflight"]
  primaries = evaluation["active_hand_contact_primaries_seen"]
  checks = {
    "environment_count": evaluation["environment_count"]
      >= threshold["minimum_environment_count"],
    "success_rate": evaluation["success_rate"] >= threshold["minimum_success_rate"],
    "success_rate_per_active_hand": all(
      rate >= threshold["minimum_success_rate_per_active_hand"]
      for rate in evaluation["success_rate_by_active_hand"].values()
    ),
    "both_active_hands_succeeded": all(
      count >= 1 for count in evaluation["success_count_by_active_hand"].values()
    ),
    "success_is_verified_opposition": (
      evaluation["opposing_contact_count"] >= evaluation["success_count"] > 0
    ),
    "opposing_normals": evaluation["minimum_opposing_normal_dot"]
      <= threshold["minimum_opposing_normal_dot_maximum"],
    "both_hands_exposed_thumb_and_finger_surfaces": all(
      any("_thumb_" in name for name in primaries[side])
      and any("_thumb_" not in name for name in primaries[side])
      for side in ("left", "right")
    ),
    "contact_force": evaluation["maximum_active_hand_force_n"]
      <= threshold["maximum_contact_force_n"],
    "no_object_loss": evaluation["object_lost_count"] == 0,
    "no_fall": evaluation["fall_count"] == 0,
    "no_non_foot_ground_contact": evaluation["non_foot_ground_count"] == 0,
    "no_numerical_instability": evaluation["numerical_instability_count"] == 0,
    "no_precontact_object_contact": (
      evaluation["precontact_object_contact_count"] == 0
    ),
    "base_assist_disabled": (
      evaluation["contact_base_assist_environment_count"] == 0
    ),
    "learner_authority_partition": (
      evaluation["maximum_unauthorized_hand_action"] == 0.0
      and evaluation["maximum_inactive_hand_coordination"] == 0.0
      and evaluation["maximum_inactive_arm_contact_approach_correction"] == 0.0
      and evaluation["maximum_outside_contact_approach_correction"] == 0.0
    ),
    "frozen_gradient_partition": (
      evaluation["frozen_locomotion"]["gradient_parameter_count"] == 0
      and evaluation["frozen_reach"]["gradient_parameter_count"] == 0
    ),
    "finite_and_closed": (
      evaluation["finite"] is True and evaluation["environment_closed"] is True
    ),
  }
  return {
    "protocol": "hear-workyard-contact-fresh-preflight-v1",
    "checks": checks,
    "passed": all(checks.values()),
  }


def final_evaluation_gate(
  evaluation: dict[str, object], contract: dict[str, object]
) -> dict[str, object]:
  threshold = contract["evaluation"]
  checks = {
    "independent_episode_count": (
      evaluation["episode_count"] == threshold["episodes"]
    ),
    "held_out_seed_protocol": (
      evaluation["seed_allocation"]["protocol"]
      == threshold["held_out_seed_protocol"]
      and evaluation["seed_allocation"]["first"] == threshold["held_out_seed_first"]
      and evaluation["seed_allocation"]["last"] == threshold["held_out_seed_last"]
    ),
    "verified_grasp_success_rate": evaluation["success_rate"]
      >= threshold["verified_grasp_success_rate_minimum"],
    "verified_grasp_success_rate_per_active_hand": all(
      rate >= threshold[
        "verified_grasp_success_rate_per_active_hand_minimum"
      ]
      for rate in evaluation["success_rate_by_active_hand"].values()
    ),
    "both_active_hands_succeeded": all(
      count >= threshold["success_count_per_active_hand_minimum"]
      for count in evaluation["success_count_by_active_hand"].values()
    ),
    "success_is_verified_opposition": (
      evaluation["opposing_contact_count"] >= evaluation["success_count"] > 0
    ),
    "opposing_normals": evaluation["minimum_opposing_normal_dot"]
      <= threshold["minimum_opposing_normal_dot_maximum"],
    "contact_force": evaluation["maximum_active_hand_force_n"]
      <= threshold["maximum_contact_force_n"],
    "no_object_loss": evaluation["object_lost_count"]
      <= threshold["object_lost_count_maximum"],
    "no_fall": evaluation["fall_count"] <= threshold["fall_count_maximum"],
    "no_non_foot_ground_contact": evaluation["non_foot_ground_count"]
      <= threshold["non_foot_ground_count_maximum"],
    "no_numerical_instability": evaluation["numerical_instability_count"]
      <= threshold["numerical_instability_count_maximum"],
    "no_precontact_object_contact": evaluation["precontact_object_contact_count"]
      <= threshold["precontact_object_contact_count_maximum"],
    "no_unauthorized_hand_action": evaluation["maximum_unauthorized_hand_action"]
      <= threshold["unauthorized_hand_action_maximum"],
    "inactive_hand_stationary": evaluation["maximum_inactive_hand_coordination"]
      <= threshold["inactive_hand_coordination_maximum"],
    "frozen_gradient_partition": (
      evaluation["frozen_locomotion"]["gradient_parameter_count"]
        == threshold["frozen_gradient_parameter_count"]
      and evaluation["frozen_reach"]["gradient_parameter_count"]
        == threshold["frozen_gradient_parameter_count"]
    ),
    "base_assist_disabled": evaluation["contact_base_assist_environment_count"]
      <= threshold["base_assist_environment_count_maximum"],
    "finite": evaluation["finite"] is threshold["finite_required"],
    "environment_closed": evaluation["environment_closed"]
      is threshold["environment_closed_required"],
  }
  return {
    "protocol": "hear-workyard-contact-independent-500-gate-v1",
    "checks": checks,
    "passed": all(checks.values()),
  }


def dagger_warm_start(
  runner,
  env,
  raw_env,
  module: ModuleType,
  args: argparse.Namespace,
  checkpoint: Path,
  curves_path: Path,
) -> dict[str, object]:
  """Online DAgger with extra weight on dimensions currently owned by Harness."""
  import torch

  actor = runner.alg.get_policy()
  actor.train()
  optimizer = torch.optim.Adam(actor.parameters(), lr=args.dagger_learning_rate)
  observations = env.get_observations().to(DEVICE)
  action_term = module._contact_action(raw_env)
  curve: list[dict[str, float | int]] = []
  label_count = 0
  finite_label_count = 0
  authorized_label_count = 0
  authorized_state_count = 0
  teacher_execution_count = 0
  execution_count = 0

  for step in range(args.dagger_steps):
    fraction = step / max(1, args.dagger_steps - 1)
    beta = args.dagger_beta_initial + fraction * (
      args.dagger_beta_final - args.dagger_beta_initial
    )
    with torch.no_grad():
      teacher_action = action_term.compute_hand_teacher_action()
      authority = action_term.hand_teacher_authority_mask()
      actor.update_normalization(observations)
    predicted_action, policy_action_std = (
      runner.alg.deterministic_policy_statistics(observations)
    )
    element_imitation = torch.nn.functional.smooth_l1_loss(
      predicted_action, teacher_action, reduction="none"
    )
    # Background zero-action labels remain useful, while four authorized
    # active-hand dimensions receive enough mass not to be drowned by approach.
    imitation_weight = 1.0 + 3.0 * authority.to(dtype=element_imitation.dtype)
    imitation_loss = (
      (element_imitation * imitation_weight).sum() / imitation_weight.sum()
    )
    dispersion_penalty = torch.relu(
      policy_action_std - args.teacher_maximum_action_std
    ).square().mean()
    loss = imitation_loss + args.teacher_dispersion_coefficient * dispersion_penalty
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
      observations = observations.to(DEVICE)
    finite = torch.isfinite(teacher_action).all(dim=-1)
    finite_label_count += int(finite.sum().item()) * module.HAND_ACTION_SIZE
    label_count += raw_env.num_envs * module.HAND_ACTION_SIZE
    authorized_label_count += int(authority.sum().item())
    authorized_state_count += int(authority.any(dim=-1).sum().item())
    teacher_execution_count += int(teacher_mask.sum().item())
    execution_count += raw_env.num_envs
    curve.append({
      "step": step,
      "loss": float(loss.detach().item()),
      "imitation_loss": float(imitation_loss.detach().item()),
      "dispersion_penalty": float(dispersion_penalty.detach().item()),
      "mean_policy_action_std": float(policy_action_std.detach().mean().item()),
      "gradient_norm": float(gradient_norm.detach().item()),
      "teacher_beta": beta,
      "authorized_state_fraction": float(authority.any(dim=-1).float().mean().item()),
      "authorized_dimension_fraction": float(authority.float().mean().item()),
    })

  runner.save(str(checkpoint), infos={
    "protocol": "hear-contact-online-dagger-warm-start-v1",
    "steps": args.dagger_steps,
  })
  curves_path.write_text(
    json.dumps({
      "protocol": "hear-contact-online-dagger-curves-v1",
      "loss": curve,
    }, ensure_ascii=False, indent=2) + "\n",
    encoding="utf-8",
  )
  normalizer = getattr(actor, "obs_normalizer", None)
  normalizer_count = getattr(normalizer, "count", None)
  return {
    "protocol": "hear-contact-online-dagger-warm-start-v1",
    "steps": args.dagger_steps,
    "samples": args.dagger_steps * raw_env.num_envs,
    "learning_rate": args.dagger_learning_rate,
    "loss": "authority_weighted_smooth_l1_plus_excess_std_penalty",
    "authority_weight": 4.0,
    "background_weight": 1.0,
    "actor_distribution": "beta_bounded_minus_one_one",
    "maximum_action_std": args.teacher_maximum_action_std,
    "initial_loss": curve[0]["loss"],
    "final_loss": curve[-1]["loss"],
    "minimum_loss": min(float(point["loss"]) for point in curve),
    "final_mean_policy_action_std": float(curve[-1]["mean_policy_action_std"]),
    "label_coverage": finite_label_count / label_count if label_count else 0.0,
    "authorized_label_count": authorized_label_count,
    "authorized_state_count": authorized_state_count,
    "teacher_execution_ratio": (
      teacher_execution_count / execution_count if execution_count else 0.0
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
  """PPO with an analytic 8D label on every learner-visited rollout state."""
  import torch

  actor = runner.alg.get_policy()
  if not hasattr(runner.alg, "configure_rollout_teacher"):
    raise RuntimeError("Contact runner did not construct HearRetentionPPO")
  distribution = runner.alg.distribution_identity()
  if not distribution["structurally_bounded"]:
    raise RuntimeError("Contact actor is not structurally bounded to [-1, 1]")
  action_term = module._contact_action(raw_env)
  runner.alg.configure_rollout_teacher(
    action_term.compute_hand_teacher_action,
    args.rollout_teacher_coefficient,
    args.teacher_maximum_action_std,
    args.teacher_dispersion_coefficient,
  )
  if args.critic_warmup_iterations >= args.iterations:
    raise ValueError("Retention PPO needs an actor-update iteration after warm-up")
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
    name: parameter.detach().clone() for name, parameter in actor.named_parameters()
  }
  actor_normalizer = getattr(actor, "obs_normalizer", None)
  normalizer_count = getattr(actor_normalizer, "count", None)
  normalizer_count_before = (
    int(normalizer_count.item()) if normalizer_count is not None else None
  )
  original_update = runner.alg.update
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
      losses = original_update()
    finally:
      if critic_warmup:
        for parameter in actor_parameters:
          parameter.requires_grad_(True)
    retention = dict(runner.alg.last_retention_metrics)
    point = {
      "iteration": update_index,
      "critic_warmup": critic_warmup,
      **retention,
      "actor_relative_parameter_drift": actor_relative_parameter_drift(),
    }
    curve.append(point)
    losses["critic_warmup"] = float(critic_warmup)
    losses["actor_relative_parameter_drift"] = float(
      point["actor_relative_parameter_drift"]
    )
    update_index += 1
    return losses

  actor.update_normalization = lambda _observations: None
  runner.alg.update = guarded_update
  try:
    runner.learn(
      num_learning_iterations=args.iterations,
      init_at_random_ep_len=True,
    )
  finally:
    runner.alg.update = original_update
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
    "protocol": "hear-contact-ppo-retention-v1",
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
    "teacher_action_scope": "authorized_8d_hand_action",
    "loss_coupling": "same_ppo_minibatch",
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


def _selection_gate(
  ppo_evaluation: dict[str, object],
  dagger_evaluation: dict[str, object],
  retention: dict[str, object],
  contract: dict[str, object],
) -> tuple[dict[str, object], dict[str, object]]:
  threshold = contract["training"]["checkpoint_selection"]
  delta = {
    "success_rate": ppo_evaluation["success_rate"] - dagger_evaluation["success_rate"],
    "maximum_active_hand_force_n": (
      ppo_evaluation["maximum_active_hand_force_n"]
      - dagger_evaluation["maximum_active_hand_force_n"]
    ),
    "object_lost_count": (
      ppo_evaluation["object_lost_count"] - dagger_evaluation["object_lost_count"]
    ),
  }
  checks = {
    "retention_protocol": retention["protocol"] == "hear-contact-ppo-retention-v1",
    "bounded_actor": retention["actor_distribution"]["structurally_bounded"] is True,
    "full_rollout_teacher_coverage": retention["rollout_teacher_label_coverage"]
      >= threshold["ppo_rollout_teacher_label_coverage_minimum"],
    "bounded_policy_dispersion": retention["final_mean_policy_action_std"]
      <= threshold["ppo_mean_action_std_maximum"],
    "success_retained": delta["success_rate"]
      >= threshold["ppo_success_rate_delta_minimum"],
    "both_active_hands_succeeded": all(
      count >= threshold["ppo_success_count_per_active_hand_minimum"]
      for count in ppo_evaluation["success_count_by_active_hand"].values()
    ),
    "success_rate_per_active_hand": all(
      rate >= threshold["ppo_success_rate_per_active_hand_minimum"]
      for rate in ppo_evaluation["success_rate_by_active_hand"].values()
    ),
    "contact_force": ppo_evaluation["maximum_active_hand_force_n"]
      <= threshold["ppo_maximum_contact_force_n"],
    "no_object_loss": ppo_evaluation["object_lost_count"]
      <= threshold["ppo_object_lost_count_maximum"],
    "no_fall": ppo_evaluation["fall_count"] <= threshold["ppo_fall_count_maximum"],
    "no_non_foot_ground": ppo_evaluation["non_foot_ground_count"]
      <= threshold["ppo_non_foot_ground_count_maximum"],
    "no_numerical_instability": ppo_evaluation["numerical_instability_count"]
      <= threshold["ppo_numerical_instability_count_maximum"],
    "no_precontact_contact": ppo_evaluation["precontact_object_contact_count"]
      <= threshold["ppo_precontact_object_contact_count_maximum"],
    "authority_partition": (
      ppo_evaluation["maximum_unauthorized_hand_action"]
        <= threshold["ppo_unauthorized_hand_action_maximum"]
      and ppo_evaluation["maximum_inactive_hand_coordination"]
        <= threshold["ppo_inactive_hand_coordination_maximum"]
    ),
    "base_assist_disabled": ppo_evaluation["contact_base_assist_environment_count"]
      <= threshold["ppo_base_assist_environment_count_maximum"],
    "finite_and_closed": (
      ppo_evaluation["finite"] is True
      and ppo_evaluation["environment_closed"] is True
    ),
  }
  return delta, {
    "protocol": "hear-contact-retention-checkpoint-selection-gate-v1",
    "checks": checks,
    "passed": all(checks.values()),
  }


def train(
  args: argparse.Namespace,
  module: ModuleType,
  contract: dict[str, object],
  evidence: dict[str, object],
) -> tuple[dict[str, object], dict[str, object], dict[str, object]]:
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
  configure_artifacts(env_cfg, evidence)
  agent_cfg = load_rl_cfg(module.TASK_ID)
  agent_cfg.max_iterations = args.iterations
  agent_cfg.num_steps_per_env = contract["training"]["ppo"]["steps_per_environment"]
  agent_cfg.save_interval = max(1, args.iterations // 4)
  agent_cfg.seed = args.seed
  agent_cfg.logger = "tensorboard"
  agent_cfg.run_name = "contact_hand_dagger_then_retention_ppo"
  agent_cfg.upload_model = False
  agent_cfg.algorithm.learning_rate = args.ppo_critic_learning_rate
  agent_cfg.algorithm.schedule = "fixed"
  dump_yaml(staging / "env.yaml", asdict(env_cfg))
  dump_yaml(staging / "agent.yaml", asdict(agent_cfg))

  guarded_env = numerically_guarded_env_class(
    ManagerBasedRlEnv, module, contract
  )
  raw_env = guarded_env(cfg=env_cfg, device=DEVICE)
  env = RslRlVecEnvWrapper(raw_env, clip_actions=agent_cfg.clip_actions)
  training_environment_closed = False
  numerical_recovery = None
  runner = None
  tensorboard_writer = None
  try:
    action_term = module._contact_action(raw_env)
    frozen_locomotion_identity = dict(action_term.teacher.identity)
    frozen_reach_identity = dict(action_term.frozen_reach.identity)
    runner_cls = load_runner_cls(module.TASK_ID) or MjlabOnPolicyRunner
    runner = runner_cls(env, asdict(agent_cfg), str(log_dir), DEVICE)
    dagger_checkpoint = staging / "workyard_contact_dagger_warm_start.pt"
    dagger_curves = staging / "dagger-curves.json"
    dagger = dagger_warm_start(
      runner, env, raw_env, module, args, dagger_checkpoint, dagger_curves
    )
    retention = ppo_with_retention(runner, env, raw_env, module, args)
    tensorboard_writer = getattr(runner.logger, "writer", None)
    ppo_checkpoint = staging / "workyard_contact_dagger_ppo.pt"
    runner.save(str(ppo_checkpoint))
  finally:
    numerical_recovery = raw_env.numerical_recovery_report()
    if tensorboard_writer is None and runner is not None:
      tensorboard_writer = getattr(runner.logger, "writer", None)
    if tensorboard_writer is not None:
      if hasattr(tensorboard_writer, "flush"):
        tensorboard_writer.flush()
      if hasattr(tensorboard_writer, "close"):
        tensorboard_writer.close()
    env.close()
    training_environment_closed = True

  checkpoint_data = torch.load(
    ppo_checkpoint, map_location="cpu", weights_only=False
  )
  completed_iterations = int(checkpoint_data.get("iter", -1)) + 1
  if completed_iterations < args.iterations:
    raise RuntimeError(
      f"Contact checkpoint has {completed_iterations} PPO iterations, expected {args.iterations}"
    )
  curves = tensorboard_curves(log_dir)
  curves_path = staging / "training-curves.json"
  curves_path.write_text(
    json.dumps(curves, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
  )
  selection = contract["training"]["checkpoint_selection"]
  ppo_evaluation = contact_rollout(
    module,
    evidence,
    args.comparison_envs,
    args.comparison_steps,
    selection["held_out_seed_first"],
    checkpoint=ppo_checkpoint,
  )
  dagger_evaluation = contact_rollout(
    module,
    evidence,
    args.comparison_envs,
    args.comparison_steps,
    selection["held_out_seed_first"],
    checkpoint=dagger_checkpoint,
  )
  dagger["comparison_evaluation"] = dagger_evaluation
  delta, selection_gate = _selection_gate(
    ppo_evaluation, dagger_evaluation, retention, contract
  )
  selected_source = ppo_checkpoint if selection_gate["passed"] else dagger_checkpoint
  selected_checkpoint = staging / "workyard_contact_selected.pt"
  shutil.copy2(selected_source, selected_checkpoint)
  checkpoint_selection = {
    "protocol": "hear-contact-retention-checkpoint-selection-v1",
    "comparison_seed_first": selection["held_out_seed_first"],
    "comparison_seed_last": selection["held_out_seed_last"],
    "ppo_minus_dagger": delta,
    "gate": selection_gate,
    "rollback_applied": not selection_gate["passed"],
    "selected_source": "ppo" if selection_gate["passed"] else "dagger",
    "selected_checkpoint": {
      "file": selected_checkpoint.name,
      "bytes": selected_checkpoint.stat().st_size,
      "sha256": sha256(selected_checkpoint),
    },
  }
  evaluation_contract = contract["evaluation"]
  final_evaluation = contact_rollout(
    module,
    evidence,
    args.final_eval_envs,
    args.final_eval_steps,
    evaluation_contract["held_out_seed_first"],
    checkpoint=selected_checkpoint,
  )
  final_gate = final_evaluation_gate(final_evaluation, contract)
  training = {
    "protocol": "hear-workyard-contact-dagger-ppo-training-v1",
    "seed": args.seed,
    "environment_count": args.num_envs,
    "iterations_requested": args.iterations,
    "completed_iterations": completed_iterations,
    "steps_per_environment_per_iteration": agent_cfg.num_steps_per_env,
    "dagger": dagger,
    "ppo_retention": retention,
    "ppo_comparison_evaluation": ppo_evaluation,
    "checkpoint_selection": checkpoint_selection,
    "ppo_environment_steps": (
      args.num_envs * args.iterations * agent_cfg.num_steps_per_env
    ),
    "dagger_environment_steps": args.num_envs * args.dagger_steps,
    "total_environment_steps": args.num_envs * (
      args.dagger_steps + args.iterations * agent_cfg.num_steps_per_env
    ),
    "frozen_locomotion": frozen_locomotion_identity,
    "frozen_reach": frozen_reach_identity,
    "environment_closed": training_environment_closed,
    "numerical_recovery": numerical_recovery,
    "ppo_checkpoint": {
      "file": ppo_checkpoint.name,
      "bytes": ppo_checkpoint.stat().st_size,
      "sha256": sha256(ppo_checkpoint),
    },
    "training_curves": {
      "file": curves_path.name,
      "bytes": curves_path.stat().st_size,
      "sha256": sha256(curves_path),
    },
  }
  archive_path = Path(args.archive).resolve()
  if archive_path.exists():
    raise FileExistsError(f"Contact training archive already exists: {archive_path}")
  archive_path.parent.mkdir(parents=True, exist_ok=True)
  with tarfile.open(archive_path, "w:gz") as archive:
    archive.add(staging, arcname="hear-workyard-contact")
  return training, final_evaluation, final_gate


def _memory_report(torch) -> dict[str, object]:
  torch.cuda.synchronize()
  free_bytes, total_bytes = torch.cuda.mem_get_info()
  return {
    "total_bytes": int(total_bytes),
    "free_bytes_after_run": int(free_bytes),
    "allocated_bytes_after_run": int(torch.cuda.memory_allocated()),
    "reserved_bytes_after_run": int(torch.cuda.memory_reserved()),
    "peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
    "peak_reserved_bytes": int(torch.cuda.max_memory_reserved()),
  }


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
    raise RuntimeError("Contact Workyard requires an NVIDIA CUDA runtime")
  torch.cuda.reset_peak_memory_stats()
  extract_bundle()
  module = load_module()
  if module.TASK_ID not in list_tasks():
    raise RuntimeError(f"Contact task was not registered: {module.TASK_ID}")
  contract, contract_path, evidence = validate_contract(
    module, args.mode in ("pilot", "train")
  )
  if args.mode == "train":
    expected = contract["training"]
    selection = expected["checkpoint_selection"]
    evaluation = contract["evaluation"]
    mismatches = {
      "iterations": (args.iterations, expected["ppo"]["iterations"]),
      "dagger_steps": (args.dagger_steps, expected["dagger"]["steps"]),
      "num_envs": (args.num_envs, expected["ppo"]["environments"]),
      "comparison_envs": (args.comparison_envs, selection["comparison_episodes"]),
      "comparison_steps": (
        args.comparison_steps, selection["control_steps_per_episode"]
      ),
      "final_eval_envs": (args.final_eval_envs, evaluation["episodes"]),
      "final_eval_steps": (
        args.final_eval_steps, evaluation["control_steps_per_episode"]
      ),
    }
    drifted = [name for name, pair in mismatches.items() if pair[0] != pair[1]]
    if drifted:
      raise RuntimeError("Formal contact configuration drifted: " + ", ".join(drifted))
    if not args.execution_profile.startswith("colab-pro-"):
      raise RuntimeError("Formal contact training is authorized only on Colab Pro")
  if args.mode == "pilot" and not args.execution_profile.startswith("colab-pro-"):
    raise RuntimeError("Contact training pilots are authorized only on Colab Pro")

  training = None
  final_gate = None
  if args.mode == "smoke":
    evaluation = contact_rollout(
      module,
      evidence,
      args.num_envs,
      args.rollout_steps,
      args.seed,
      episode_seeds=args.episode_seeds,
      analytic_teacher=True,
    )
    fresh_preflight = None
  elif args.mode == "teacher":
    evaluation = contact_rollout(
      module,
      evidence,
      args.num_envs,
      args.rollout_steps,
      args.seed,
      episode_seeds=args.episode_seeds,
      analytic_teacher=True,
    )
    fresh_preflight = analytic_preflight_gate(evaluation, contract)
  else:
    preflight_count = contract["qualified_inputs"]["analytic_teacher_preflight"][
      "minimum_environment_count"
    ]
    fresh_preflight_evaluation = contact_rollout(
      module,
      evidence,
      preflight_count,
      contract["evaluation"]["control_steps_per_episode"],
      args.seed,
      analytic_teacher=True,
    )
    fresh_preflight = analytic_preflight_gate(
      fresh_preflight_evaluation, contract
    )
    fresh_preflight["evaluation"] = fresh_preflight_evaluation
    if not fresh_preflight["passed"]:
      raise ContactQualificationError(
        "Fresh analytic contact preflight rejected training: "
        + json.dumps(fresh_preflight["checks"], ensure_ascii=False),
        fresh_preflight,
      )
    training, evaluation, final_gate = train(
      args, module, contract, evidence
    )

  structural_checks = {
    "observation_size": module.HAND_OBSERVATION_SIZE == 262,
    "learned_action_size": module.HAND_ACTION_SIZE == 8,
    "logical_composition_size": module.COMPOSED_ACTION_SIZE == 37,
    "finite_and_closed": (
      evaluation["finite"] is True and evaluation["environment_closed"] is True
    ),
    "frozen_locomotion": (
      evaluation["frozen_locomotion"]["gradient_parameter_count"] == 0
    ),
    "frozen_reach": evaluation["frozen_reach"]["gradient_parameter_count"] == 0,
    "authority_partition": (
      evaluation["maximum_unauthorized_hand_action"] == 0.0
      and evaluation["maximum_inactive_hand_coordination"] == 0.0
      and evaluation["maximum_inactive_arm_contact_approach_correction"] == 0.0
      and evaluation["maximum_outside_contact_approach_correction"] == 0.0
    ),
    "base_assist_disabled": evaluation["contact_base_assist_environment_count"] == 0,
  }
  if not all(structural_checks.values()):
    raise RuntimeError(
      "Contact structural acceptance failed: "
      + json.dumps(structural_checks, ensure_ascii=False)
    )
  return {
    "protocol": "hear-workyard-contact-run-v2",
    "ready": True,
    "mode": args.mode,
    "created_at": datetime.now(timezone.utc).isoformat(),
    "task": module.TASK_ID,
    "framework": {
      "mjlab": importlib.metadata.version("mjlab"),
      "mujoco": mujoco.__version__,
      "torch": torch.__version__,
      "algorithm": (
        "authority-weighted online DAgger + rollout-retention PPO"
        if args.mode in ("pilot", "train") else None
      ),
    },
    "accelerator": {
      "device": torch.cuda.get_device_name(0),
      "cuda_version": torch.version.cuda,
      "memory": _memory_report(torch),
    },
    "execution": {
      "profile": args.execution_profile,
      "dependency_install": "preinstalled_pinned" if skip_install else "ephemeral_pip",
    },
    "bundle": {
      "sha256": sha256(REMOTE_BUNDLE),
      "contract_sha256": sha256(contract_path),
      "environment_sha256": sha256(
        REMOTE_ROOT / "training" / "workyard_contact_mjlab_env.py"
      ),
      "retention_algorithm_sha256": sha256(
        REMOTE_ROOT / "training" / "hear_retention_ppo.py"
      ),
      "locomotion_jit_sha256": evidence["locomotion_jit_sha256"],
      "reach_jit_sha256": evidence["reach_jit_sha256"],
      "analytic_preflight_sha256": evidence["analytic_preflight_sha256"],
    },
    "contract": {
      "protocol": contract["protocol"],
      "observation_size": contract["learner"]["observation"]["size"],
      "action_size": contract["learner"]["action"]["size"],
      "logical_composed_action_size": contract["composition"][
        "logical_composed_action_size"
      ],
      "terminal_stage": contract["environment"]["terminal_stage"],
      "excluded_stages": contract["environment"]["excluded_stages"],
      "teacher_state_directly_exposed": False,
      "forbidden_direct_references": evidence["forbidden_observation_references"],
      "cpu_round_trip_per_control_step": evidence["cpu_round_trip_per_control_step"],
      "hand_max_closing_joint_lead_rad": (
        contract["harness_executor"]["maximum_closing_joint_lead_rad"]
      ),
      "opposing_support_coordination": (
        contract["harness_executor"]["opposing_support_coordination"]
      ),
      "hand_contact_solref_time_constant_s": (
        contract["harness_executor"]["hand_contact_solref_time_constant_s"]
      ),
    },
    "fresh_analytic_preflight": fresh_preflight,
    # Canonical qualified-input envelope retained for the training contract
    # and both local/Colab bundle validators.  It is an exact protocol alias of
    # the fresh gate above, not a second evaluation or a relaxed decision.
    "gate": None if fresh_preflight is None else {
      "protocol": "hear-workyard-contact-analytic-teacher-preflight-v1",
      "checks": fresh_preflight["checks"],
      "passed": fresh_preflight["passed"],
    },
    "training": training,
    "evaluation": evaluation,
    "acceptance": {
      "scope": (
        "local_or_remote_structural_smoke"
        if args.mode == "smoke"
        else "fresh_analytic_teacher_preflight"
        if args.mode == "teacher"
        else "bounded_colab_training_pipeline_pilot"
        if args.mode == "pilot"
        else "independent_500_episode_verified_grasp_gate"
      ),
      "structural_checks": structural_checks,
      "structural_invariants_passed": all(structural_checks.values()),
      "final_gate": final_gate,
      "verified_grasp_policy_accepted": (
        args.mode == "train" and final_gate is not None and final_gate["passed"]
      ),
      "lift_carry_place_authorized": False,
      "deployment_accepted": False,
    },
  }


def emit_artifact_stream(sources: dict[str, Path]) -> list[dict[str, object]]:
  """Return durable artifacts over the already-authenticated exec websocket."""

  emitted: list[dict[str, object]] = []
  prefix = "@@HEAR_ARTIFACT_V1@@"
  for name, source in sources.items():
    if not source.is_file():
      continue
    size = source.stat().st_size
    digest = sha256(source)
    print(prefix + json.dumps({
      "kind": "begin",
      "name": name,
      "bytes": size,
      "sha256": digest,
    }, separators=(",", ":")), flush=True)
    index = 0
    with source.open("rb") as handle:
      while chunk := handle.read(192 * 1024):
        print(prefix + json.dumps({
          "kind": "chunk",
          "name": name,
          "index": index,
          "data": base64.b64encode(chunk).decode("ascii"),
        }, separators=(",", ":")), flush=True)
        index += 1
    print(prefix + json.dumps({
      "kind": "end",
      "name": name,
      "chunks": index,
    }, separators=(",", ":")), flush=True)
    emitted.append({"name": name, "bytes": size, "sha256": digest})
  if "training-report.json" not in {value["name"] for value in emitted}:
    raise RuntimeError("Artifact stream did not emit the training report")
  return emitted


def main() -> None:
  report: dict[str, object]
  mode = "unknown"
  archive_path: Path | None = None
  artifact_stream_enabled = False
  try:
    args = parse_args()
    mode = args.mode
    archive_path = Path(args.archive).resolve()
    artifact_stream_enabled = args.artifact_stream
    report = execute(args)
  except BaseException as error:
    report = {
      "protocol": "hear-workyard-contact-run-v2",
      "ready": False,
      "mode": mode,
      "created_at": datetime.now(timezone.utc).isoformat(),
      "error": {
        "type": type(error).__name__,
        "message": str(error),
        "traceback": traceback.format_exc(),
        **(
          {"qualification": error.evidence}
          if isinstance(error, ContactQualificationError) else {}
        ),
      },
    }
  REMOTE_REPORT.parent.mkdir(parents=True, exist_ok=True)
  REMOTE_REPORT.write_text(
    json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
  )
  streamed_artifacts = None
  if artifact_stream_enabled:
    streamed_artifacts = emit_artifact_stream({
      "training-report.json": REMOTE_REPORT,
      **(
        {"training-artifacts.tar.gz": archive_path}
        if archive_path is not None else {}
      ),
    })
  evaluation = report.get("evaluation")
  acceptance = report.get("acceptance")
  fresh_preflight = report.get("fresh_analytic_preflight")
  console_summary = {
    "protocol": report["protocol"],
    "ready": report["ready"],
    "mode": report["mode"],
    "report_path": str(REMOTE_REPORT),
    "fresh_preflight_passed": (
      fresh_preflight.get("passed") if isinstance(fresh_preflight, dict) else None
    ),
    "success_rate": (
      evaluation.get("success_rate") if isinstance(evaluation, dict) else None
    ),
    "success_rate_by_active_hand": (
      evaluation.get("success_rate_by_active_hand")
      if isinstance(evaluation, dict) else None
    ),
    "maximum_active_hand_force_n": (
      evaluation.get("maximum_active_hand_force_n")
      if isinstance(evaluation, dict) else None
    ),
    "final_gate_passed": (
      acceptance.get("final_gate", {}).get("passed")
      if isinstance(acceptance, dict)
      and isinstance(acceptance.get("final_gate"), dict)
      else None
    ),
    "error": report.get("error") if not report["ready"] else None,
    "streamed_artifacts": streamed_artifacts,
  }
  print(json.dumps(console_summary, ensure_ascii=False))
  if not report["ready"]:
    raise SystemExit(1)


if __name__ == "__main__":
  main()
