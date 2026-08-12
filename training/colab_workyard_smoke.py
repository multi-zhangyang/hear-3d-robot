from __future__ import annotations

import hashlib
import importlib.metadata
import importlib.util
import inspect
import json
import os
import subprocess
import sys
import tarfile
import traceback
from datetime import datetime, timezone
from pathlib import Path
from types import ModuleType


MJLAB_VERSION = "1.5.3"
REMOTE_BUNDLE = Path("/content/hear-workyard-smoke-bundle.tar.gz")
REMOTE_ROOT = Path("/content/hear-workyard-smoke")
REPORT_PATH = Path("/content/hear-workyard-smoke-report.json")
NUM_ENVS = 8
ROLLOUT_STEPS = 32


def sha256(path: Path) -> str:
  digest = hashlib.sha256()
  with path.open("rb") as handle:
    for chunk in iter(lambda: handle.read(1024 * 1024), b""):
      digest.update(chunk)
  return digest.hexdigest()


def run(command: list[str]) -> None:
  subprocess.run(command, check=True)


def extract_bundle() -> None:
  if not REMOTE_BUNDLE.is_file():
    raise FileNotFoundError(f"Workyard smoke bundle is missing: {REMOTE_BUNDLE}")
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


def smoke() -> dict[str, object]:
  run([
    sys.executable,
    "-m",
    "pip",
    "install",
    "--quiet",
    f"mjlab=={MJLAB_VERSION}",
  ])

  os.environ.update({
    "MUJOCO_GL": "egl",
    "WANDB_MODE": "disabled",
    "PYTHONUNBUFFERED": "1",
  })

  import mujoco
  import torch
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.rl import RslRlVecEnvWrapper
  from mjlab.tasks.registry import list_tasks, load_env_cfg, load_rl_cfg

  if not torch.cuda.is_available():
    raise RuntimeError("Workyard smoke requires an NVIDIA CUDA runtime")

  extract_bundle()
  contract_path = REMOTE_ROOT / "training" / "workyard-task-v2.json"
  contract = json.loads(contract_path.read_text(encoding="utf-8"))
  module = load_workyard_module()
  if module.TASK_ID not in list_tasks():
    raise RuntimeError(f"Workyard task was not registered: {module.TASK_ID}")
  if module.TASK_ID != contract["environment"]["task_id"]:
    raise RuntimeError("Registered Workyard task disagrees with its contract")

  observation_source = inspect.getsource(module.WorkyardObservation.__call__)
  forbidden = contract["task"]["student_forbidden_observations"]
  forbidden_direct_references = [
    name for name in forbidden if name in observation_source
  ]
  if forbidden_direct_references:
    raise RuntimeError(
      "Teacher-only state leaked into the student observation source: "
      + ", ".join(forbidden_direct_references)
    )

  env_cfg = load_env_cfg(module.TASK_ID, play=True)
  env_cfg.scene.num_envs = NUM_ENVS
  agent_cfg = load_rl_cfg(module.TASK_ID)
  raw_env = ManagerBasedRlEnv(cfg=env_cfg, device="cuda:0")
  env = RslRlVecEnvWrapper(raw_env, clip_actions=agent_cfg.clip_actions)
  environment_closed = False
  try:
    observations = env.get_observations()
    if "actor" not in observations.keys():
      raise RuntimeError(
        f"Workyard observation TensorDict has no actor group: {list(observations.keys())}"
      )
    actor_observations = observations["actor"]
    expected_observation_shape = (NUM_ENVS, module.OBSERVATION_SIZE)
    if tuple(actor_observations.shape) != expected_observation_shape:
      raise RuntimeError(
        f"Expected observation {expected_observation_shape}, got "
        f"{tuple(actor_observations.shape)}"
      )
    if not torch.isfinite(actor_observations).all():
      raise RuntimeError("Workyard reset observation contains non-finite values")

    actions = torch.zeros(
      (NUM_ENVS, module.ACTION_SIZE), dtype=torch.float32, device="cuda:0"
    )
    reward_sum = torch.zeros(NUM_ENVS, dtype=torch.float32, device="cuda:0")
    termination_count = 0
    maximum_absolute_observation = float(actor_observations.abs().max().item())
    maximum_absolute_reward = 0.0
    reward_term_rate_sum = {
      name: 0.0 for name in raw_env.reward_manager.active_terms
    }
    reward_term_rate_abs_max = {
      name: 0.0 for name in raw_env.reward_manager.active_terms
    }
    maximum_actuator_force_ratio = 0.0
    maximum_joint_velocity = 0.0
    maximum_foot_slip = 0.0
    robot = raw_env.scene["robot"]
    joint_ids = robot.indexing.joint_ids
    joint_force_ranges = raw_env.sim.model.jnt_actfrcrange[:, joint_ids]
    joint_force_limited = raw_env.sim.model.jnt_actfrclimited[joint_ids]
    if not bool(joint_force_limited.all().item()):
      raise RuntimeError("A controlled G1 joint has no enabled actuator-force limit")
    if not bool(
      (
        torch.isfinite(joint_force_ranges).all(dim=-1)
        & (joint_force_ranges[..., 0] < 0.0)
        & (joint_force_ranges[..., 1] > 0.0)
      ).all().item()
    ):
      raise RuntimeError("A controlled G1 joint has an invalid actuator-force range")
    with torch.inference_mode():
      for _ in range(ROLLOUT_STEPS):
        observations, rewards, dones, _ = env.step(actions)
        actor_observations = observations["actor"]
        if not torch.isfinite(actor_observations).all():
          raise RuntimeError("Workyard rollout observation contains non-finite values")
        if not torch.isfinite(rewards).all():
          raise RuntimeError("Workyard rollout reward contains non-finite values")
        reward_sum += rewards
        termination_count += int(dones.sum().item())
        maximum_absolute_observation = max(
          maximum_absolute_observation,
          float(actor_observations.abs().max().item()),
        )
        maximum_absolute_reward = max(
          maximum_absolute_reward,
          float(rewards.abs().max().item()),
        )
        step_reward = raw_env.reward_manager._step_reward
        for index, name in enumerate(raw_env.reward_manager.active_terms):
          weighted_rate = step_reward[:, index]
          reward_term_rate_sum[name] += float(weighted_rate.mean().item())
          reward_term_rate_abs_max[name] = max(
            reward_term_rate_abs_max[name],
            float(weighted_rate.abs().max().item()),
          )
        joint_effort = robot.data.qfrc_actuator
        joint_force_ranges = raw_env.sim.model.jnt_actfrcrange[:, joint_ids]
        directional_limits = torch.where(
          joint_effort >= 0.0,
          joint_force_ranges[..., 1],
          -joint_force_ranges[..., 0],
        )
        maximum_actuator_force_ratio = max(
          maximum_actuator_force_ratio,
          float((joint_effort.abs() / directional_limits).max().item()),
        )
        maximum_joint_velocity = max(
          maximum_joint_velocity,
          float(robot.data.joint_vel.abs().max().item()),
        )
        maximum_foot_slip = max(
          maximum_foot_slip,
          float(module._foot_contact_summary(raw_env)[2].max().item()),
        )
  finally:
    env.close()
    environment_closed = True

  return {
    "protocol": "hear-workyard-colab-smoke-v1",
    "ready": True,
    "created_at": datetime.now(timezone.utc).isoformat(),
    "task": module.TASK_ID,
    "framework": {
      "mjlab": importlib.metadata.version("mjlab"),
      "mujoco": mujoco.__version__,
      "torch": torch.__version__,
    },
    "accelerator": {
      "cuda_available": True,
      "cuda_version": torch.version.cuda,
      "device": torch.cuda.get_device_name(0),
    },
    "bundle": {
      "sha256": sha256(REMOTE_BUNDLE),
      "contract_sha256": sha256(contract_path),
      "environment_sha256": sha256(
        REMOTE_ROOT / "training" / "workyard_mjlab_env.py"
      ),
    },
    "observation": {
      "shape": [NUM_ENVS, module.OBSERVATION_SIZE],
      "groups": sorted(observations.keys()),
      "expected_size": contract["observation"]["size"],
      "finite": True,
      "teacher_state_directly_exposed": False,
      "forbidden_direct_references": forbidden_direct_references,
    },
    "action": {
      "shape": [NUM_ENVS, module.ACTION_SIZE],
      "expected_size": contract["action"]["size"],
      "finite": bool(torch.isfinite(actions).all().item()),
    },
    "rollout": {
      "environment_count": NUM_ENVS,
      "control_steps": ROLLOUT_STEPS,
      "mean_reward_sum": float(reward_sum.mean().item()),
      "termination_count": termination_count,
      "maximum_absolute_observation": maximum_absolute_observation,
      "maximum_absolute_reward": maximum_absolute_reward,
      "reward_terms": {
        name: {
          "mean_episode_contribution": reward_term_rate_sum[name]
          * raw_env.step_dt,
          "maximum_absolute_weighted_rate": reward_term_rate_abs_max[name],
        }
        for name in raw_env.reward_manager.active_terms
      },
      "actuator_force_authority": "joint.actuatorfrcrange",
      "all_joint_actuator_force_limits_enabled": bool(
        joint_force_limited.all().item()
      ),
      "minimum_joint_actuator_authority_nm": float(
        joint_force_ranges.abs().min().item()
      ),
      "maximum_joint_actuator_authority_nm": float(
        joint_force_ranges.abs().max().item()
      ),
      "maximum_actuator_force_ratio": maximum_actuator_force_ratio,
      "maximum_joint_velocity_rad_s": maximum_joint_velocity,
      "maximum_foot_slip_m_s": maximum_foot_slip,
      "environment_closed": environment_closed,
    },
  }


def main() -> None:
  report: dict[str, object]
  try:
    report = smoke()
  except BaseException as error:
    report = {
      "protocol": "hear-workyard-colab-smoke-v1",
      "ready": False,
      "created_at": datetime.now(timezone.utc).isoformat(),
      "error": {
        "type": type(error).__name__,
        "message": str(error),
        "traceback": traceback.format_exc(),
      },
    }
  REPORT_PATH.write_text(
    json.dumps(report, ensure_ascii=False, indent=2) + "\n",
    encoding="utf-8",
  )
  print(json.dumps(report, ensure_ascii=False))
  if not report["ready"]:
    raise SystemExit(1)


if __name__ == "__main__":
  main()
