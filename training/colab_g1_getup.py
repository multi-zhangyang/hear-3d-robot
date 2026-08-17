from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.metadata
import json
import os
import shutil
import subprocess
import sys
import tarfile
import time
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
DRIVE_ROOT = Path("/content/drive/MyDrive")
HEARTBEAT_SECONDS = 900


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser()
  parser.add_argument(
    "--mode", choices=("smoke", "train", "evaluate"), default="train"
  )
  parser.add_argument("--iterations", type=int, default=5000)
  parser.add_argument("--num-envs", type=int, default=2048)
  parser.add_argument("--eval-envs", type=int, default=500)
  parser.add_argument("--eval-batch-size", type=int, default=64)
  parser.add_argument("--eval-steps", type=int, default=750)
  parser.add_argument("--seed", type=int, default=42)
  parser.add_argument("--output-root", default="/content/hear-g1-getup")
  parser.add_argument("--archive", default="/content/hear-g1-getup-artifacts.tar.gz")
  parser.add_argument("--worker", action="store_true")
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
  for name in (
    "iterations", "num_envs", "eval_envs", "eval_batch_size", "eval_steps"
  ):
    if getattr(args, name) <= 0:
      parser.error(f"--{name.replace('_', '-')} must be positive")
  if args.mode in ("train", "evaluate") and args.eval_envs < 500:
    parser.error("formal get-up evaluation requires at least 500 environments")
  if args.eval_batch_size > args.eval_envs:
    parser.error("--eval-batch-size cannot exceed --eval-envs")
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


def atomic_json(path: Path, value: dict[str, object]) -> None:
  path.parent.mkdir(parents=True, exist_ok=True)
  temporary = path.with_name(path.name + ".partial")
  temporary.write_text(
    json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
  )
  temporary.replace(path)


def assert_drive_writable(args: argparse.Namespace, output_root: Path) -> None:
  if args.mode != "train":
    return
  if not DRIVE_ROOT.is_dir():
    raise RuntimeError(
      "Formal G1 training requires a mounted /content/drive/MyDrive"
    )
  try:
    output_root.relative_to(DRIVE_ROOT.resolve())
  except ValueError as error:
    raise RuntimeError(
      f"Formal G1 training output must live in Google Drive: {output_root}"
    ) from error
  output_root.parent.mkdir(parents=True, exist_ok=True)
  probe = output_root.parent / f".hear-drive-probe-{os.getpid()}"
  payload = f"hear-drive-write-probe:{time.time_ns()}\n"
  try:
    probe.write_text(payload, encoding="utf-8")
    if probe.read_text(encoding="utf-8") != payload:
      raise RuntimeError("Google Drive write/read probe changed bytes")
  finally:
    probe.unlink(missing_ok=True)


def supervised_main(args: argparse.Namespace) -> None:
  extract_bundle()
  output_root = Path(args.output_root).resolve()
  assert_drive_writable(args, output_root)
  output_root.mkdir(parents=True, exist_ok=True)
  log_path = output_root / "training.log"
  state_path = output_root / "training-state.json"
  worker_path = REMOTE_ROOT / "training" / "colab_g1_getup.py"
  with log_path.open("a", encoding="utf-8", buffering=1) as log:
    log.write(
      f"\n[hear] supervised worker start {datetime.now(timezone.utc).isoformat()}\n"
    )
    process = subprocess.Popen(
      [sys.executable, "-u", str(worker_path), "--worker"],
      cwd=REMOTE_ROOT,
      stdout=log,
      stderr=subprocess.STDOUT,
      text=True,
    )
    started = time.monotonic()
    print(json.dumps({
      "event": "g1_getup_training_started",
      "worker_pid": process.pid,
      "drive_output": str(output_root),
      "heartbeat_seconds": HEARTBEAT_SECONDS,
    }), flush=True)
    try:
      while True:
        try:
          exit_code = process.wait(timeout=HEARTBEAT_SECONDS)
          break
        except subprocess.TimeoutExpired:
          state = {}
          try:
            state = json.loads(state_path.read_text(encoding="utf-8"))
          except (FileNotFoundError, json.JSONDecodeError):
            pass
          print(json.dumps({
            "event": "g1_getup_training_heartbeat",
            "elapsed_seconds": round(time.monotonic() - started),
            "stage": state.get("stage", "starting"),
            "checkpoint_iteration": state.get("checkpoint_iteration"),
            "drive_output": str(output_root),
          }), flush=True)
    except BaseException:
      process.terminate()
      try:
        process.wait(timeout=30)
      except subprocess.TimeoutExpired:
        process.kill()
        process.wait()
      raise
  if exit_code != 0:
    tail = log_path.read_text(encoding="utf-8", errors="replace").splitlines()[-80:]
    if tail:
      print("\n".join(tail), file=sys.stderr)
    raise RuntimeError(f"G1 get-up worker exited with code {exit_code}")
  print(json.dumps({
    "event": "g1_getup_training_complete",
    "elapsed_seconds": round(time.monotonic() - started),
    "drive_output": str(output_root),
  }), flush=True)


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
  state_path = output_root / "training-state.json"
  staging.mkdir(parents=True, exist_ok=True)
  log_dir.mkdir(parents=True, exist_ok=True)
  env_cfg = load_env_cfg(module.TASK_ID, play=False)
  env_cfg.scene.num_envs = args.num_envs
  env_cfg.seed = args.seed
  agent_cfg = load_rl_cfg(module.TASK_ID)
  agent_cfg.max_iterations = args.iterations
  agent_cfg.save_interval = max(1, min(100, args.iterations))
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
  original_save = runner.save

  def atomic_runner_save(path: str, infos=None) -> None:
    destination = Path(path)
    temporary = destination.with_name(destination.name + ".partial")
    temporary.unlink(missing_ok=True)
    original_save(str(temporary), infos)
    temporary.replace(destination)
    if destination.parent == log_dir and destination.stem.startswith("model_"):
      try:
        iteration = int(destination.stem.removeprefix("model_"))
      except ValueError:
        return
      atomic_json(state_path, {
        "protocol": "hear-g1-getup-training-state-v1",
        "stage": "training",
        "checkpoint": destination.name,
        "checkpoint_iteration": iteration,
        "target_iterations": args.iterations,
        "updated_at": datetime.now(timezone.utc).isoformat(),
      })

  runner.save = atomic_runner_save
  checkpoints = []
  for candidate in log_dir.glob("model_*.pt"):
    try:
      iteration = int(candidate.stem.removeprefix("model_"))
    except ValueError:
      continue
    checkpoints.append((iteration, candidate))
  completed_before_resume = 0
  if checkpoints:
    _, latest = max(checkpoints, key=lambda item: item[0])
    runner.load(str(latest), strict=True, map_location="cuda:0")
    completed_before_resume = int(runner.current_learning_iteration) + 1
    if completed_before_resume > args.iterations:
      raise RuntimeError(
        f"Drive checkpoint already exceeds target: {completed_before_resume}"
      )
    runner.current_learning_iteration = completed_before_resume
  remaining_iterations = args.iterations - completed_before_resume
  atomic_json(state_path, {
    "protocol": "hear-g1-getup-training-state-v1",
    "stage": "training",
    "resumed_iterations": completed_before_resume,
    "target_iterations": args.iterations,
    "updated_at": datetime.now(timezone.utc).isoformat(),
  })
  try:
    if remaining_iterations > 0:
      runner.learn(
        num_learning_iterations=remaining_iterations,
        init_at_random_ep_len=True,
      )
    runner.save(str(checkpoint))
    onnx_temporary = onnx.with_name(onnx.name + ".partial")
    onnx_temporary.unlink(missing_ok=True)
    runner.export_policy_to_onnx(str(staging), onnx_temporary.name)
    onnx_temporary.replace(onnx)
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


def load_trained_artifacts(
  args: argparse.Namespace,
  module: ModuleType,
  output_root: Path,
) -> tuple[Path, Path, Path, dict[str, object]]:
  import torch
  from mjlab.tasks.registry import load_rl_cfg

  staging = output_root / "artifacts"
  checkpoint = staging / "g1_getup.pt"
  onnx = staging / "g1_getup.onnx"
  for artifact in (checkpoint, onnx):
    if not artifact.is_file():
      raise FileNotFoundError(f"Trained G1 get-up artifact is missing: {artifact}")
  checkpoint_data = torch.load(checkpoint, map_location="cpu", weights_only=False)
  completed = int(checkpoint_data.get("iter", -1)) + 1
  if completed < args.iterations:
    raise RuntimeError(
      f"Get-up checkpoint contains {completed} iterations below {args.iterations}"
    )
  agent_cfg = load_rl_cfg(module.TASK_ID)
  return staging, checkpoint, onnx, {
    "iterations": completed,
    "environment_count": args.num_envs,
    "steps_per_environment_per_iteration": agent_cfg.num_steps_per_env,
    "environment_steps": completed * args.num_envs * agent_cfg.num_steps_per_env,
    "accelerator": torch.cuda.get_device_name(0),
    "cuda_version": torch.version.cuda,
  }


def evaluate(
  args: argparse.Namespace,
  module: ModuleType,
  checkpoint: Path,
) -> dict[str, object]:
  import torch
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
  from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls

  episode_count = 0
  success_count = 0
  category_counts = {name: 0 for name in module.RESET_POSE_NAMES}
  category_successes = {name: 0 for name in module.RESET_POSE_NAMES}
  recovery_seconds: list[float] = []
  non_finite_action_count = 0
  maximum_action = 0.0
  batch_index = 0
  while episode_count < args.eval_envs:
    batch_size = min(args.eval_batch_size, args.eval_envs - episode_count)
    env_cfg = load_env_cfg(module.TASK_ID, play=True)
    env_cfg.scene.num_envs = batch_size
    env_cfg.seed = args.seed + 10_000 + batch_index
    agent_cfg = load_rl_cfg(module.TASK_ID)
    raw_env = ManagerBasedRlEnv(cfg=env_cfg, device="cuda:0")
    env = RslRlVecEnvWrapper(raw_env, clip_actions=agent_cfg.clip_actions)
    runner = None
    policy = None
    observations = None
    categories = None
    active = None
    succeeded = None
    completion_step = None
    actions = None
    finite = None
    active_before = None
    stable_done = None
    newly_succeeded = None
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
        batch_size, module.OBSERVATION_SIZE
      ):
        raise RuntimeError("Get-up evaluation observation shape drifted")
      categories = raw_env.getup_reset_category.clone()
      active = torch.ones(batch_size, dtype=torch.bool, device="cuda:0")
      succeeded = torch.zeros_like(active)
      completion_step = torch.full(
        (batch_size,), args.eval_steps, dtype=torch.long, device="cuda:0"
      )
      with torch.inference_mode():
        for step in range(args.eval_steps):
          actions = policy(observations)
          finite = torch.isfinite(actions).all(dim=-1)
          non_finite_action_count += int((~finite & active).sum().item())
          if not finite.all():
            actions = torch.where(
              finite.unsqueeze(-1), actions, torch.zeros_like(actions)
            )
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
      for index, name in enumerate(module.RESET_POSE_NAMES):
        category_mask = categories == index
        category_counts[name] += int(category_mask.sum().item())
        category_successes[name] += int((succeeded & category_mask).sum().item())
      success_count += int(succeeded.sum().item())
      recovery_seconds.extend(
        float(value) * module.CONTROL_STEP_SECONDS
        for value in completion_step.cpu().tolist()
      )
      episode_count += batch_size
      batch_index += 1
      print(json.dumps({
        "event": "g1_getup_evaluation_batch_complete",
        "episodes_completed": episode_count,
        "episodes_target": args.eval_envs,
        "batch_size": batch_size,
        "successes": success_count,
      }), flush=True)
    finally:
      env.close()
      runner = None
      policy = None
      observations = None
      categories = None
      active = None
      succeeded = None
      completion_step = None
      actions = None
      finite = None
      active_before = None
      stable_done = None
      newly_succeeded = None
      del env, raw_env
      gc.collect()
      torch.cuda.empty_cache()

  def category_rate(*names: str) -> float:
    denominator = sum(category_counts[name] for name in names)
    numerator = sum(category_successes[name] for name in names)
    return numerator / denominator if denominator else 0.0

  return {
    "episode_count": episode_count,
    "overall_success_rate": success_count / episode_count,
    "prone_success_rate": category_rate("prone"),
    "supine_success_rate": category_rate("supine"),
    "side_success_rate": category_rate("left_side", "right_side"),
    "reset_category_counts": category_counts,
    "median_recovery_seconds": quantile(recovery_seconds, 0.50),
    "p95_recovery_seconds": quantile(recovery_seconds, 0.95),
    "stable_exit_rate": success_count / episode_count,
    "non_finite_action_count": non_finite_action_count,
    "maximum_absolute_action": maximum_action,
  }


def quantile(values: list[float], probability: float) -> float:
  if not values:
    raise ValueError("Cannot calculate a quantile from no values")
  ordered = sorted(values)
  position = (len(ordered) - 1) * probability
  lower = int(position)
  upper = min(lower + 1, len(ordered) - 1)
  fraction = position - lower
  return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


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


def worker_main(args: argparse.Namespace) -> None:
  run([sys.executable, "-m", "pip", "install", "--quiet", f"mjlab=={MJLAB_VERSION}"])
  import torch
  if not torch.cuda.is_available():
    raise RuntimeError("G1 get-up training requires an NVIDIA CUDA runtime")
  module = load_environment()
  contract, contract_path = validate_contract(module)
  output_root = Path(args.output_root).resolve()
  assert_drive_writable(args, output_root)
  output_root.mkdir(parents=True, exist_ok=True)
  environment_path = Path(module.__file__).resolve()
  resume_identity = {
    "protocol": "hear-g1-getup-resume-identity-v1",
    "task_id": module.TASK_ID,
    "mjlab_version": MJLAB_VERSION,
    "training_contract_sha256": sha256(contract_path),
    "environment_sha256": sha256(environment_path),
    "seed": args.seed,
    "environment_count": args.num_envs,
  }
  resume_path = output_root / "resume-identity.json"
  if resume_path.is_file():
    existing = json.loads(resume_path.read_text(encoding="utf-8"))
    if existing != resume_identity:
      raise RuntimeError(
        "Drive checkpoint identity differs from the requested training source"
      )
  elif args.mode == "evaluate":
    raise FileNotFoundError("G1 evaluation requires a Drive resume identity")
  else:
    atomic_json(resume_path, resume_identity)
  if args.mode == "evaluate":
    staging, checkpoint, onnx, training = load_trained_artifacts(
      args, module, output_root
    )
  else:
    staging, checkpoint, onnx, training = train(args, module, output_root)
  atomic_json(output_root / "training-state.json", {
    "protocol": "hear-g1-getup-training-state-v1",
    "stage": "evaluating",
    "checkpoint_iteration": training["iterations"] - 1,
    "target_iterations": args.iterations,
    "updated_at": datetime.now(timezone.utc).isoformat(),
  })
  evaluation = evaluate(args, module, checkpoint)
  gate_passed = accepted(evaluation, contract["deployment_gate"])
  evaluation["deployment_accepted"] = gate_passed
  identity = onnx_identity(onnx)
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
  atomic_json(output_root / "training-state.json", {
    "protocol": "hear-g1-getup-training-state-v1",
    "stage": "completed" if gate_passed else "rejected",
    "checkpoint_iteration": training["iterations"] - 1,
    "target_iterations": args.iterations,
    "deployment_accepted": gate_passed,
    "updated_at": datetime.now(timezone.utc).isoformat(),
  })
  print(json.dumps({
    "report": str(REMOTE_REPORT),
    "archive": args.archive,
    "deployment_accepted": gate_passed,
    "evaluation": evaluation,
  }, ensure_ascii=False))
  if args.mode in ("train", "evaluate") and not gate_passed:
    raise RuntimeError("G1 get-up policy did not pass the deployment gate")


def main() -> None:
  args = parse_args()
  if args.worker:
    worker_main(args)
  else:
    supervised_main(args)


if __name__ == "__main__":
  main()
