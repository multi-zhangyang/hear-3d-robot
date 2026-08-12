from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import os
import sys
import tarfile
import traceback
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from types import ModuleType


MJLAB_VERSION = "1.5.3"
REMOTE_BUNDLE = Path("/content/hear-workyard-baseline-bundle.tar.gz")
REMOTE_CONFIG = Path("/content/hear-workyard-baseline-config.json")
REMOTE_ROOT = Path("/content/hear-workyard-baseline-source")
REPORT_PATH = Path("/content/hear-workyard-baseline-report.json")


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser()
  parser.add_argument("--iterations", type=int, default=20)
  parser.add_argument("--num-envs", type=int, default=64)
  parser.add_argument("--eval-envs-per-stage", type=int, default=8)
  parser.add_argument("--eval-steps", type=int, default=300)
  parser.add_argument("--seed", type=int, default=42)
  parser.add_argument("--output-root", default="/content/hear-workyard-baseline")
  parser.add_argument(
    "--archive", default="/content/hear-workyard-baseline-artifacts.tar.gz"
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
  for name in ("iterations", "num_envs", "eval_envs_per_stage", "eval_steps"):
    if getattr(args, name) <= 0:
      parser.error(f"--{name.replace('_', '-')} must be positive")
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
    raise FileNotFoundError(f"Workyard training bundle is missing: {REMOTE_BUNDLE}")
  REMOTE_ROOT.mkdir(parents=True, exist_ok=False)
  with tarfile.open(REMOTE_BUNDLE, "r:gz") as archive:
    root = REMOTE_ROOT.resolve()
    for member in archive.getmembers():
      target = (REMOTE_ROOT / member.name).resolve()
      if target != root and root not in target.parents:
        raise RuntimeError(f"Unsafe Workyard bundle member: {member.name}")
    archive.extractall(REMOTE_ROOT, filter="data")


def load_workyard_module() -> ModuleType:
  path = REMOTE_ROOT / "training" / "workyard_mjlab_env.py"
  spec = importlib.util.spec_from_file_location("hear_workyard_mjlab_env", path)
  if spec is None or spec.loader is None:
    raise RuntimeError(f"Cannot load Workyard environment module: {path}")
  module = importlib.util.module_from_spec(spec)
  sys.modules[spec.name] = module
  spec.loader.exec_module(module)
  return module


def tensorboard_curves(log_dir: Path) -> dict[str, list[dict[str, float | int]]]:
  from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

  accumulator = EventAccumulator(str(log_dir), size_guidance={"scalars": 0})
  accumulator.Reload()
  curves = {}
  for tag in accumulator.Tags().get("scalars", []):
    curves[tag] = [
      {"step": int(event.step), "value": float(event.value)}
      for event in accumulator.Scalars(tag)
    ]
  if not curves:
    raise RuntimeError("PPO baseline produced no TensorBoard scalar curves")
  return curves


def evaluate_policy(
  module: ModuleType,
  policy,
  envs_per_stage: int,
  steps: int,
  seed: int,
) -> dict[str, object]:
  import torch
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.rl import RslRlVecEnvWrapper
  from mjlab.tasks.registry import load_env_cfg, load_rl_cfg

  stage_count = len(module.TEACHER_STAGES)
  num_envs = envs_per_stage * stage_count
  env_cfg = load_env_cfg(module.TASK_ID, play=True)
  env_cfg.scene.num_envs = num_envs
  env_cfg.seed = seed
  agent_cfg = load_rl_cfg(module.TASK_ID)
  raw_env = ManagerBasedRlEnv(cfg=env_cfg, device="cuda:0")
  env = RslRlVecEnvWrapper(raw_env, clip_actions=agent_cfg.clip_actions)
  environment_closed = False
  try:
    target_stage = torch.arange(stage_count, device="cuda:0").repeat_interleave(
      envs_per_stage
    )
    command = module._workyard_command(raw_env)
    command.teacher_target_stage[:] = target_stage
    observations = env.get_observations()
    active = torch.ones(num_envs, dtype=torch.bool, device="cuda:0")
    reward_sum = torch.zeros(num_envs, device="cuda:0")
    steps_to_done = torch.full(
      (num_envs,), steps, dtype=torch.long, device="cuda:0"
    )
    max_stage_reached = command.teacher_stage.clone()
    termination_names = tuple(raw_env.termination_manager.active_terms)
    termination_cause = torch.full(
      (num_envs,), -1, dtype=torch.long, device="cuda:0"
    )
    maximum_action = 0.0
    maximum_force_ratio = 0.0
    maximum_joint_velocity = 0.0
    robot = raw_env.scene["robot"]
    joint_ids = robot.indexing.joint_ids

    with torch.inference_mode():
      for step_index in range(steps):
        max_stage_reached = torch.maximum(max_stage_reached, command.teacher_stage)
        actions = policy(observations)
        if not torch.isfinite(actions).all():
          raise RuntimeError("PPO baseline emitted a non-finite evaluation action")
        maximum_action = max(maximum_action, float(actions.abs().max().item()))
        observations, rewards, dones, _ = env.step(actions)
        if not torch.isfinite(observations["actor"]).all():
          raise RuntimeError("PPO evaluation observation became non-finite")
        if not torch.isfinite(rewards).all():
          raise RuntimeError("PPO evaluation reward became non-finite")
        reward_sum += rewards * active

        effort = robot.data.qfrc_actuator
        ranges = raw_env.sim.model.jnt_actfrcrange[:, joint_ids]
        directional_limits = torch.where(
          effort >= 0.0, ranges[..., 1], -ranges[..., 0]
        )
        maximum_force_ratio = max(
          maximum_force_ratio,
          float((effort.abs() / directional_limits).max().item()),
        )
        maximum_joint_velocity = max(
          maximum_joint_velocity, float(robot.data.joint_vel.abs().max().item())
        )

        first_done = dones & active
        if torch.any(first_done):
          steps_to_done[first_done] = step_index + 1
          for cause_index, name in enumerate(termination_names):
            caused = raw_env.termination_manager.get_term(name) & first_done
            unset = termination_cause < 0
            termination_cause[caused & unset] = cause_index
          succeeded = (
            raw_env.termination_manager.get_term("stage_success") & first_done
          )
          max_stage_reached[succeeded] = torch.maximum(
            max_stage_reached[succeeded], target_stage[succeeded] + 1
          )
          active[first_done] = False
        if not torch.any(active):
          break

    stage_reports = []
    for stage_index, stage_name in enumerate(module.TEACHER_STAGES):
      mask = target_stage == stage_index
      cause_counts = {
        name: int(((termination_cause == index) & mask).sum().item())
        for index, name in enumerate(termination_names)
      }
      horizon_count = int((active & mask).sum().item())
      cause_counts["evaluation_horizon"] = horizon_count
      success_count = cause_counts["stage_success"]
      stage_reports.append({
        "target_stage": stage_name,
        "target_stage_index": stage_index,
        "episodes": envs_per_stage,
        "success_count": success_count,
        "success_rate": success_count / envs_per_stage,
        "mean_reward_sum": float(reward_sum[mask].mean().item()),
        "mean_steps_observed": float(steps_to_done[mask].float().mean().item()),
        "maximum_teacher_stage_reached": int(max_stage_reached[mask].max().item()),
        "termination_distribution": cause_counts,
      })

    return {
      "environment_count": num_envs,
      "environments_per_stage": envs_per_stage,
      "maximum_steps_per_episode": steps,
      "stage_metrics": stage_reports,
      "maximum_absolute_action": maximum_action,
      "maximum_joint_actuator_force_ratio": maximum_force_ratio,
      "maximum_joint_velocity_rad_s": maximum_joint_velocity,
      "finite": True,
      "environment_closed": True,
    }
  finally:
    env.close()
    environment_closed = True
    if not environment_closed:
      raise RuntimeError("Workyard evaluation environment did not close")


def train(args: argparse.Namespace) -> dict[str, object]:
  run([sys.executable, "-m", "pip", "install", "--quiet", f"mjlab=={MJLAB_VERSION}"])
  os.environ.update({
    "MUJOCO_GL": "egl",
    "WANDB_MODE": "disabled",
    "PYTHONUNBUFFERED": "1",
  })

  import mujoco
  import torch
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
  from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
  from mjlab.utils.os import dump_yaml

  if not torch.cuda.is_available():
    raise RuntimeError("Workyard PPO baseline requires an NVIDIA CUDA runtime")

  extract_bundle()
  module = load_workyard_module()
  contract_path = REMOTE_ROOT / "training" / "workyard-task-v2.json"
  contract = json.loads(contract_path.read_text(encoding="utf-8"))
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
  agent_cfg.run_name = "workyard_short_baseline"
  agent_cfg.upload_model = False
  dump_yaml(staging / "env.yaml", asdict(env_cfg))
  dump_yaml(staging / "agent.yaml", asdict(agent_cfg))

  raw_env = ManagerBasedRlEnv(cfg=env_cfg, device="cuda:0")
  env = RslRlVecEnvWrapper(raw_env, clip_actions=agent_cfg.clip_actions)
  training_environment_closed = False
  try:
    runner_cls = load_runner_cls(module.TASK_ID) or MjlabOnPolicyRunner
    runner = runner_cls(env, asdict(agent_cfg), str(log_dir), "cuda:0")
    runner.learn(
      num_learning_iterations=args.iterations,
      init_at_random_ep_len=True,
    )
    checkpoint = staging / "workyard_ppo_baseline.pt"
    runner.save(str(checkpoint))
    policy = runner.get_inference_policy(device="cuda:0")
    command = module._workyard_command(raw_env)
    curriculum = {
      "maximum_stage_index": int(command.max_stage),
      "maximum_stage": module.TEACHER_STAGES[command.max_stage],
      "episode_counts": {
        stage: int(command.episode_counts[index].item())
        for index, stage in enumerate(module.TEACHER_STAGES)
      },
      "success_counts": {
        stage: int(command.success_counts[index].item())
        for index, stage in enumerate(module.TEACHER_STAGES)
      },
    }
  finally:
    env.close()
    training_environment_closed = True

  if not checkpoint.is_file():
    raise RuntimeError("PPO baseline did not write its final checkpoint")
  checkpoint_data = torch.load(checkpoint, map_location="cpu", weights_only=False)
  checkpoint_iteration = int(checkpoint_data.get("iter", -1))
  completed_iterations = checkpoint_iteration + 1
  if completed_iterations < args.iterations:
    raise RuntimeError(
      f"PPO checkpoint contains {completed_iterations} completed iterations, "
      f"below {args.iterations}"
    )

  curves = tensorboard_curves(log_dir)
  curves_path = staging / "training-curves.json"
  curves_path.write_text(
    json.dumps(curves, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
  )
  evaluation = evaluate_policy(
    module,
    policy,
    args.eval_envs_per_stage,
    args.eval_steps,
    args.seed + 10_000,
  )
  report = {
    "protocol": "hear-workyard-ppo-baseline-v1",
    "ready": True,
    "created_at": datetime.now(timezone.utc).isoformat(),
    "task": module.TASK_ID,
    "purpose": "short learning-signal baseline; not a deployment policy",
    "framework": {
      "mjlab": importlib.metadata.version("mjlab"),
      "mujoco": mujoco.__version__,
      "torch": torch.__version__,
      "algorithm": "PPO (RSL-RL)",
    },
    "accelerator": {
      "device": torch.cuda.get_device_name(0),
      "cuda_version": torch.version.cuda,
    },
    "contract": {
      "sha256": sha256(contract_path),
      "observation_size": contract["observation"]["size"],
      "action_size": contract["action"]["size"],
    },
    "training": {
      "seed": args.seed,
      "environment_count": args.num_envs,
      "iterations_requested": args.iterations,
      "checkpoint_iteration_zero_based": checkpoint_iteration,
      "completed_iterations": completed_iterations,
      "steps_per_environment_per_iteration": agent_cfg.num_steps_per_env,
      "total_environment_steps": (
        args.num_envs * args.iterations * agent_cfg.num_steps_per_env
      ),
      "curriculum": curriculum,
      "scalar_curve_tags": sorted(curves),
      "environment_closed": training_environment_closed,
    },
    "evaluation": evaluation,
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
  report_path = staging / "training-report.json"
  report_path.write_text(
    json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
  )

  archive_path = Path(args.archive).resolve()
  if archive_path.exists():
    raise FileExistsError(f"Training archive already exists: {archive_path}")
  archive_path.parent.mkdir(parents=True, exist_ok=True)
  with tarfile.open(archive_path, "w:gz") as archive:
    archive.add(staging, arcname="hear-workyard-baseline")
  return report


def main() -> None:
  report: dict[str, object]
  try:
    report = train(parse_args())
  except BaseException as error:
    report = {
      "protocol": "hear-workyard-ppo-baseline-v1",
      "ready": False,
      "created_at": datetime.now(timezone.utc).isoformat(),
      "error": {
        "type": type(error).__name__,
        "message": str(error),
        "traceback": traceback.format_exc(),
      },
    }
  REPORT_PATH.write_text(
    json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
  )
  print(json.dumps(report, ensure_ascii=False))
  if not report["ready"]:
    raise SystemExit(1)


if __name__ == "__main__":
  main()
