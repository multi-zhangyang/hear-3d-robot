from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import shutil
import subprocess
import sys
import tarfile
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

MJLAB_VERSION = "1.5.3"
TASK_ID = "Mjlab-Velocity-Flat-Unitree-G1"


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser()
  parser.add_argument("--iterations", type=int, default=1000)
  parser.add_argument("--num-envs", type=int, default=4096)
  parser.add_argument("--eval-envs", type=int, default=256)
  parser.add_argument("--eval-steps", type=int, default=600)
  parser.add_argument("--seed", type=int, default=42)
  parser.add_argument("--output-root", default="/content/hear-g1-training")
  parser.add_argument(
    "--archive", default="/content/hear-g1-training-artifacts.tar.gz"
  )
  args = parser.parse_args()
  for name in ("iterations", "num_envs", "eval_envs", "eval_steps"):
    if getattr(args, name) <= 0:
      parser.error(f"--{name.replace('_', '-')} must be positive")
  return args


def run(command: list[str], env: dict[str, str] | None = None) -> None:
  subprocess.run(command, check=True, env=env)


def sha256(path: Path) -> str:
  digest = hashlib.sha256()
  with path.open("rb") as handle:
    for chunk in iter(lambda: handle.read(1024 * 1024), b""):
      digest.update(chunk)
  return digest.hexdigest()


def latest_artifacts(log_root: Path) -> tuple[Path, Path, Path]:
  run_dirs = sorted(
    (path for path in (log_root / "g1_velocity").glob("*") if path.is_dir()),
    key=lambda path: path.stat().st_mtime_ns,
  )
  if not run_dirs:
    raise RuntimeError("mjlab produced no training run directory")
  run_dir = run_dirs[-1]
  checkpoints = sorted(
    run_dir.glob("model_*.pt"),
    key=lambda path: int(path.stem.rsplit("_", 1)[-1]),
  )
  onnx_models = sorted(run_dir.glob("*.onnx"))
  if not checkpoints or not onnx_models:
    raise RuntimeError("mjlab produced no checkpoint or ONNX policy")
  return run_dir, checkpoints[-1], onnx_models[-1]


def evaluate(checkpoint: Path, num_envs: int, steps: int) -> dict[str, float | int]:
  import torch
  import mjlab.tasks  # noqa: F401
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
  from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls

  device = "cuda:0"
  env_cfg = load_env_cfg(TASK_ID, play=True)
  env_cfg.scene.num_envs = num_envs
  agent_cfg = load_rl_cfg(TASK_ID)
  raw_env = ManagerBasedRlEnv(cfg=env_cfg, device=device)
  env = RslRlVecEnvWrapper(raw_env, clip_actions=agent_cfg.clip_actions)
  try:
    runner_cls = load_runner_cls(TASK_ID) or MjlabOnPolicyRunner
    runner = runner_cls(env, asdict(agent_cfg), device=device)
    runner.load(
      str(checkpoint),
      load_cfg={"actor": True},
      strict=True,
      map_location=device,
    )
    policy = runner.get_inference_policy(device=device)
    observations = env.get_observations()
    robot = env.unwrapped.scene["robot"]
    start = robot.data.root_link_pose_w[:, :3].clone()
    reward_total = torch.zeros(num_envs, device=device)
    termination_count = 0
    minimum_height = math.inf
    maximum_action = 0.0
    with torch.inference_mode():
      for _ in range(steps):
        actions = policy(observations)
        if not torch.isfinite(actions).all():
          raise RuntimeError("trained policy emitted a non-finite action")
        maximum_action = max(maximum_action, float(actions.abs().max().item()))
        observations, rewards, dones, _ = env.step(actions)
        reward_total += rewards
        termination_count += int(dones.sum().item())
        minimum_height = min(
          minimum_height,
          float(robot.data.root_link_pose_w[:, 2].min().item()),
        )
    end = robot.data.root_link_pose_w[:, :3]
    displacement = torch.linalg.vector_norm(end[:, :2] - start[:, :2], dim=1)
    return {
      "environment_count": num_envs,
      "physics_steps": steps,
      "mean_episode_reward_sum": float(reward_total.mean().item()),
      "mean_planar_displacement_m": float(displacement.mean().item()),
      "maximum_planar_displacement_m": float(displacement.max().item()),
      "minimum_root_height_m": minimum_height,
      "termination_count": termination_count,
      "maximum_absolute_action": maximum_action,
    }
  finally:
    env.close()


def onnx_identity(path: Path) -> dict[str, object]:
  import onnx

  model = onnx.load(path)
  onnx.checker.check_model(model)
  metadata = {entry.key: entry.value for entry in model.metadata_props}
  return {
    "inputs": [value.name for value in model.graph.input],
    "outputs": [value.name for value in model.graph.output],
    "metadata": metadata,
  }


def main() -> None:
  args = parse_args()
  run([
    sys.executable,
    "-m",
    "pip",
    "install",
    "--quiet",
    f"mjlab=={MJLAB_VERSION}",
  ])

  import torch

  if not torch.cuda.is_available():
    raise RuntimeError("mjlab G1 training requires an NVIDIA CUDA runtime")
  output_root = Path(args.output_root).resolve()
  job_root = output_root / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
  log_root = job_root / "logs"
  staging = job_root / "artifacts"
  staging.mkdir(parents=True)

  environment = os.environ.copy()
  environment.update({
    "MUJOCO_GL": "egl",
    "WANDB_MODE": "disabled",
    "PYTHONUNBUFFERED": "1",
  })
  run([
    sys.executable,
    "-m",
    "mjlab.scripts.train",
    TASK_ID,
    "--env.scene.num-envs",
    str(args.num_envs),
    "--agent.max-iterations",
    str(args.iterations),
    "--agent.save-interval",
    str(max(1, min(100, args.iterations))),
    "--agent.seed",
    str(args.seed),
    "--agent.logger",
    "tensorboard",
    "--agent.upload-model",
    "False",
    "--agent.run-name",
    "hear_g1_velocity",
    "--log-root",
    str(log_root),
  ], environment)

  run_dir, checkpoint, onnx_model = latest_artifacts(log_root)
  checkpoint_data = torch.load(checkpoint, map_location="cpu", weights_only=False)
  iteration = int(checkpoint_data.get("iter", -1))
  if iteration <= 0:
    raise RuntimeError("mjlab checkpoint has no positive training iteration")
  evaluation = evaluate(checkpoint, args.eval_envs, args.eval_steps)
  copied_checkpoint = staging / "g1_velocity.pt"
  copied_onnx = staging / "g1_velocity.onnx"
  shutil.copy2(checkpoint, copied_checkpoint)
  shutil.copy2(onnx_model, copied_onnx)
  for name in ("env.yaml", "agent.yaml"):
    source = run_dir / "params" / name
    if not source.is_file():
      raise RuntimeError(f"mjlab training configuration is missing: {name}")
    shutil.copy2(source, staging / name)

  report = {
    "version": 1,
    "created_at": datetime.now(timezone.utc).isoformat(),
    "framework": "mjlab",
    "framework_version": importlib.metadata.version("mjlab"),
    "task": TASK_ID,
    "seed": args.seed,
    "training": {
      "iterations_requested": args.iterations,
      "checkpoint_iteration": iteration,
      "environment_count": args.num_envs,
      "accelerator": torch.cuda.get_device_name(0),
      "cuda_version": torch.version.cuda,
    },
    "evaluation": evaluation,
    "checkpoint": {
      "file": copied_checkpoint.name,
      "bytes": copied_checkpoint.stat().st_size,
      "sha256": sha256(copied_checkpoint),
    },
    "onnx": {
      "file": copied_onnx.name,
      "bytes": copied_onnx.stat().st_size,
      "sha256": sha256(copied_onnx),
      **onnx_identity(copied_onnx),
    },
  }
  report_path = staging / "training-report.json"
  report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
  archive = Path(args.archive).resolve()
  if archive.exists():
    raise FileExistsError(f"training archive already exists: {archive}")
  archive.parent.mkdir(parents=True, exist_ok=True)
  with tarfile.open(archive, "w:gz") as handle:
    handle.add(staging, arcname="hear-g1-training")
  print(json.dumps({
    "archive": str(archive),
    "archive_sha256": sha256(archive),
    "report": report,
  }, ensure_ascii=False))


if __name__ == "__main__":
  main()
