from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import shutil
import sys
import tarfile
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from types import ModuleType


MJLAB_VERSION = "1.5.3"
EXECUTION_ROOT = Path(os.environ.get("HEAR_GETUP_EXECUTION_ROOT", "/content"))
REMOTE_BUNDLE = Path(os.environ.get(
  "HEAR_GETUP_BUNDLE", str(EXECUTION_ROOT / "hear-g1-getup-bundle.tar.gz")
))
REMOTE_CONFIG = Path(os.environ.get(
  "HEAR_GETUP_CONFIG", str(EXECUTION_ROOT / "hear-g1-getup-config.json")
))
REMOTE_ROOT = Path(os.environ.get(
  "HEAR_GETUP_SOURCE_ROOT", str(EXECUTION_ROOT / "hear-g1-getup-source")
))
REMOTE_REPORT = Path(os.environ.get(
  "HEAR_GETUP_REPORT", str(EXECUTION_ROOT / "hear-g1-getup-report.json")
))


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser()
  parser.add_argument("--mode", choices=("smoke", "train"), default="train")
  parser.add_argument("--iterations", type=int, default=5000)
  parser.add_argument("--num-envs", type=int, default=2048)
  parser.add_argument("--eval-envs", type=int, default=500)
  parser.add_argument("--eval-steps", type=int, default=750)
  parser.add_argument("--seed", type=int, default=42)
  parser.add_argument("--output-root", default="/content/hear-g1-getup")
  parser.add_argument("--archive", default="/content/hear-g1-getup-artifacts.tar.gz")
  if REMOTE_CONFIG.is_file():
    configured = json.loads(REMOTE_CONFIG.read_text(encoding="utf-8"))
    valid = {action.dest for action in parser._actions}
    unknown = sorted(set(configured) - valid)
    if unknown:
      parser.error("unknown remote configuration keys: " + ", ".join(unknown))
    parser.set_defaults(**configured)
  args, unknown = parser.parse_known_args()
  if unknown and not (len(unknown) == 2 and unknown[0] == "-f"):
    parser.error("unrecognized arguments: " + " ".join(unknown))
  for name in ("iterations", "num_envs", "eval_envs", "eval_steps"):
    if getattr(args, name) <= 0:
      parser.error(f"--{name.replace('_', '-')} must be positive")
  if args.mode == "train" and args.eval_envs < 500:
    parser.error("formal get-up evaluation requires at least 500 environments")
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
    raise FileNotFoundError(f"G1 get-up bundle is missing: {REMOTE_BUNDLE}")
  REMOTE_ROOT.mkdir(parents=True, exist_ok=False)
  with tarfile.open(REMOTE_BUNDLE, "r:gz") as archive:
    root = REMOTE_ROOT.resolve()
    for member in archive.getmembers():
      target = (REMOTE_ROOT / member.name).resolve()
      if target != root and root not in target.parents:
        raise RuntimeError(f"Unsafe G1 get-up bundle member: {member.name}")
    archive.extractall(REMOTE_ROOT, filter="data")


def load_environment() -> ModuleType:
  training_root = REMOTE_ROOT / "training"
  sys.path.insert(0, str(training_root))
  import g1_getup_mjlab_env as module
  return module


def validate_contract(module: ModuleType) -> tuple[dict[str, object], Path]:
  path = REMOTE_ROOT / "training" / "g1-getup-task-v1.json"
  contract = json.loads(path.read_text(encoding="utf-8"))
  if contract.get("protocol") != "hear-g1-getup-training-contract-v1":
    raise RuntimeError("G1 get-up training contract protocol is invalid")
  if contract["task_id"] != module.TASK_ID:
    raise RuntimeError("G1 get-up task registration differs from its contract")
  observation = contract["actor"]["observation"]
  action = contract["actor"]["action"]
  if (
    observation["protocol"] != module.OBSERVATION_PROTOCOL
    or observation["size"] != module.OBSERVATION_SIZE
    or sum(term[1] for term in observation["ordered_terms"])
      != module.OBSERVATION_SIZE
    or action["protocol"] != module.ACTION_PROTOCOL
    or action["size"] != module.ACTION_SIZE
  ):
    raise RuntimeError("G1 get-up tensor contract drifted")
  source = Path(module.__file__).read_text(encoding="utf-8")
  leaked = [
    token for token in observation["forbidden_terms"]
    if token in source[source.index("class G1GetupObservation"):source.index(
      "def _reset_pose_catalog"
    )]
  ]
  if leaked:
    raise RuntimeError("Forbidden state leaked into get-up Actor: " + ", ".join(leaked))
  success = contract["success"]
  if (
    success["continuous_control_steps"] != module.STABLE_CONTROL_STEPS
    or success["root_height_minimum_m"] != 0.7
    or success["projected_upright_minimum"] != 0.9
  ):
    raise RuntimeError("G1 get-up stable exit contract drifted")
  return contract, path


def train(
  args: argparse.Namespace,
  module: ModuleType,
  output_root: Path,
) -> tuple[Path, Path, Path, dict[str, object]]:
  import torch
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
  from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
  from mjlab.utils.os import dump_yaml

  staging = output_root / "artifacts"
  log_dir = output_root / "logs"
  staging.mkdir(parents=True, exist_ok=False)
  log_dir.mkdir(parents=True, exist_ok=False)
  env_cfg = load_env_cfg(module.TASK_ID, play=False)
  env_cfg.scene.num_envs = args.num_envs
  env_cfg.seed = args.seed
  agent_cfg = load_rl_cfg(module.TASK_ID)
  agent_cfg.max_iterations = args.iterations
  agent_cfg.save_interval = max(1, min(250, args.iterations // 4))
  agent_cfg.seed = args.seed
  agent_cfg.logger = "tensorboard"
  agent_cfg.run_name = "proprioceptive_getup"
  agent_cfg.upload_model = False
  dump_yaml(staging / "env.yaml", asdict(env_cfg))
  dump_yaml(staging / "agent.yaml", asdict(agent_cfg))

  raw_env = ManagerBasedRlEnv(cfg=env_cfg, device="cuda:0")
  env = RslRlVecEnvWrapper(raw_env, clip_actions=agent_cfg.clip_actions)
  runner_cls = load_runner_cls(module.TASK_ID) or MjlabOnPolicyRunner
  runner = runner_cls(env, asdict(agent_cfg), str(log_dir), "cuda:0")
  checkpoint = staging / "g1_getup.pt"
  onnx = staging / "g1_getup.onnx"
  try:
    runner.learn(
      num_learning_iterations=args.iterations,
      init_at_random_ep_len=True,
    )
    runner.save(str(checkpoint))
    runner.export_policy_to_onnx(str(staging), onnx.name)
  finally:
    writer = getattr(getattr(runner, "logger", None), "writer", None)
    if writer is not None:
      if hasattr(writer, "flush"):
        writer.flush()
      if hasattr(writer, "close"):
        writer.close()
    env.close()
  checkpoint_data = torch.load(checkpoint, map_location="cpu", weights_only=False)
  completed = int(checkpoint_data.get("iter", -1)) + 1
  if completed < args.iterations:
    raise RuntimeError(
      f"Get-up checkpoint contains {completed} iterations below {args.iterations}"
    )
  training = {
    "iterations": completed,
    "environment_count": args.num_envs,
    "steps_per_environment_per_iteration": agent_cfg.num_steps_per_env,
    "environment_steps": completed * args.num_envs * agent_cfg.num_steps_per_env,
    "accelerator": torch.cuda.get_device_name(0),
    "cuda_version": torch.version.cuda,
  }
  return staging, checkpoint, onnx, training


def evaluate(
  args: argparse.Namespace,
  module: ModuleType,
  checkpoint: Path,
) -> dict[str, object]:
  import torch
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
  from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls

  env_cfg = load_env_cfg(module.TASK_ID, play=True)
  env_cfg.scene.num_envs = args.eval_envs
  env_cfg.seed = args.seed + 10_000
  agent_cfg = load_rl_cfg(module.TASK_ID)
  raw_env = ManagerBasedRlEnv(cfg=env_cfg, device="cuda:0")
  env = RslRlVecEnvWrapper(raw_env, clip_actions=agent_cfg.clip_actions)
  try:
    runner_cls = load_runner_cls(module.TASK_ID) or MjlabOnPolicyRunner
    runner = runner_cls(env, asdict(agent_cfg), device="cuda:0")
    runner.load(
      str(checkpoint),
      load_cfg={"actor": True},
      strict=True,
      map_location="cuda:0",
    )
    policy = runner.get_inference_policy(device="cuda:0")
    observations = env.get_observations()
    if tuple(observations["actor"].shape) != (
      args.eval_envs, module.OBSERVATION_SIZE
    ):
      raise RuntimeError("Get-up evaluation observation shape drifted")
    categories = raw_env.getup_reset_category.clone()
    active = torch.ones(args.eval_envs, dtype=torch.bool, device="cuda:0")
    succeeded = torch.zeros_like(active)
    completion_step = torch.full(
      (args.eval_envs,), args.eval_steps, dtype=torch.long, device="cuda:0"
    )
    non_finite_action_count = 0
    maximum_action = 0.0
    with torch.inference_mode():
      for step in range(args.eval_steps):
        actions = policy(observations)
        finite = torch.isfinite(actions).all(dim=-1)
        non_finite_action_count += int((~finite & active).sum().item())
        if not finite.all():
          actions = torch.where(finite.unsqueeze(-1), actions, torch.zeros_like(actions))
        maximum_action = max(maximum_action, float(actions.abs().max().item()))
        active_before = active.clone()
        observations, _, dones, _ = env.step(actions)
        stable_done = raw_env.termination_manager.get_term("stable_standing")
        newly_succeeded = active_before & stable_done
        succeeded |= newly_succeeded
        completion_step[newly_succeeded] = step + 1
        active &= ~(dones.bool() & active_before)
        if not active.any():
          break

    def rate(mask: torch.Tensor) -> float:
      denominator = int(mask.sum().item())
      return float((succeeded & mask).sum().item() / denominator) if denominator else 0.0

    prone = categories == module.RESET_POSE_NAMES.index("prone")
    supine = categories == module.RESET_POSE_NAMES.index("supine")
    side = (categories == module.RESET_POSE_NAMES.index("left_side")) | (
      categories == module.RESET_POSE_NAMES.index("right_side")
    )
    times = completion_step.float() * module.CONTROL_STEP_SECONDS
    quantiles = torch.quantile(times, torch.tensor((0.50, 0.95), device="cuda:0"))
    return {
      "episode_count": args.eval_envs,
      "overall_success_rate": rate(torch.ones_like(succeeded)),
      "prone_success_rate": rate(prone),
      "supine_success_rate": rate(supine),
      "side_success_rate": rate(side),
      "reset_category_counts": {
        name: int((categories == index).sum().item())
        for index, name in enumerate(module.RESET_POSE_NAMES)
      },
      "median_recovery_seconds": float(quantiles[0].item()),
      "p95_recovery_seconds": float(quantiles[1].item()),
      "stable_exit_rate": rate(torch.ones_like(succeeded)),
      "non_finite_action_count": non_finite_action_count,
      "maximum_absolute_action": maximum_action,
    }
  finally:
    env.close()


def onnx_identity(path: Path) -> dict[str, object]:
  import onnx
  model = onnx.load(path)
  onnx.checker.check_model(model, full_check=True)
  inputs = [value.name for value in model.graph.input]
  outputs = [value.name for value in model.graph.output]
  if inputs != ["obs"] or outputs != ["actions"]:
    raise RuntimeError(f"Get-up ONNX has incompatible names: {inputs} -> {outputs}")
  return {"inputs": inputs, "outputs": outputs}


def accepted(evaluation: dict[str, object], gate: dict[str, object]) -> bool:
  return (
    evaluation["episode_count"] >= gate["minimum_episode_count"]
    and evaluation["overall_success_rate"] >= gate["minimum_overall_success_rate"]
    and evaluation["prone_success_rate"] >= gate["minimum_prone_success_rate"]
    and evaluation["supine_success_rate"] >= gate["minimum_supine_success_rate"]
    and evaluation["side_success_rate"] >= gate["minimum_side_success_rate"]
    and evaluation["median_recovery_seconds"] <= gate["maximum_median_recovery_seconds"]
    and evaluation["p95_recovery_seconds"] <= gate["maximum_p95_recovery_seconds"]
    and evaluation["stable_exit_rate"] >= gate["minimum_stable_exit_rate"]
    and evaluation["non_finite_action_count"] == gate["non_finite_action_count"]
  )


def write_archive(staging: Path, archive_path: Path) -> None:
  if archive_path.exists():
    raise FileExistsError(f"Get-up archive already exists: {archive_path}")
  archive_path.parent.mkdir(parents=True, exist_ok=True)
  with tarfile.open(archive_path, "w:gz") as archive:
    archive.add(staging, arcname="hear-g1-getup")


def main() -> None:
  args = parse_args()
  extract_bundle()
  run([sys.executable, "-m", "pip", "install", "--quiet", f"mjlab=={MJLAB_VERSION}"])
  import torch
  if not torch.cuda.is_available():
    raise RuntimeError("G1 get-up training requires an NVIDIA CUDA runtime")
  module = load_environment()
  contract, contract_path = validate_contract(module)
  output_root = Path(args.output_root).resolve()
  if output_root.exists():
    raise FileExistsError(f"Get-up output already exists: {output_root}")
  staging, checkpoint, onnx, training = train(args, module, output_root)
  evaluation = evaluate(args, module, checkpoint)
  gate_passed = accepted(evaluation, contract["deployment_gate"])
  evaluation["deployment_accepted"] = gate_passed
  identity = onnx_identity(onnx)
  environment_path = Path(module.__file__).resolve()
  report = {
    "protocol": "hear-g1-getup-policy-deployment-v1",
    "created_at": datetime.now(timezone.utc).isoformat(),
    "framework": {
      "name": "mjlab",
      "version": importlib.metadata.version("mjlab"),
      "task_id": module.TASK_ID,
    },
    "source": {
      "checkpoint": {
        "file": checkpoint.name,
        "bytes": checkpoint.stat().st_size,
        "sha256": sha256(checkpoint),
      },
      "training_contract_sha256": sha256(contract_path),
      "environment_sha256": sha256(environment_path),
      "seed": args.seed,
      "iterations": training["iterations"],
      "environment_count": training["environment_count"],
    },
    "policy": {
      "onnx": {
        "file": onnx.name,
        "bytes": onnx.stat().st_size,
        "sha256": sha256(onnx),
      },
      "runtime": "onnxruntime-web/wasm",
      "input": "obs",
      "input_protocol": module.OBSERVATION_PROTOCOL,
      "input_size": module.OBSERVATION_SIZE,
      "output": "actions",
      "output_protocol": module.ACTION_PROTOCOL,
      "output_size": module.ACTION_SIZE,
      "joint_names": list(module.BODY_JOINT_NAMES),
      "default_joint_positions": list(module.DEFAULT_JOINT_POSITIONS),
      "joint_lower_limits": list(module.JOINT_LOWER_LIMITS),
      "joint_upper_limits": list(module.JOINT_UPPER_LIMITS),
      "stiffness": list(module.JOINT_STIFFNESS),
      "damping": list(module.JOINT_DAMPING),
      "action_mapping": "neutral_piecewise_soft_joint_limits",
      "actor_inputs": "proprioception_only_no_reference_phase",
      **identity,
    },
    "training": training,
    "evaluation": evaluation,
  }
  report_path = staging / "getup-policy-report.json"
  report_path.write_text(
    json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
  )
  REMOTE_REPORT.write_text(
    json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
  )
  write_archive(staging, Path(args.archive).resolve())
  print(json.dumps({
    "report": str(REMOTE_REPORT),
    "archive": args.archive,
    "deployment_accepted": gate_passed,
    "evaluation": evaluation,
  }, ensure_ascii=False))
  if args.mode == "train" and not gate_passed:
    raise RuntimeError("G1 get-up policy did not pass the deployment gate")


if __name__ == "__main__":
  main()
