"""Whole-body closed-loop Workyard reach training environment.

The locomotion policy remains a frozen CUDA reference, not a joint owner.  The
student receives a bounded 15D balance residual around that reference and owns
the 14D arm/wrist target.  This preserves the useful standing prior while
allowing PPO to compensate for the real coupled dynamics created by reaching.
Hands remain open until the contact/grasp phase.  A batched differential-IK
teacher supplies upper-body DAgger labels without becoming an actor input.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Final

import mujoco_warp as mjwarp
import torch
import torch.nn.functional as torch_f
import warp as wp

import workyard_mjlab_env as base
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp
from mjlab.managers.action_manager import ActionTerm, ActionTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.rl import RslRlModelCfg, RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.tasks.registry import list_tasks, register_mjlab_task
from mjlab.utils.lab_api.math import quat_apply

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


TASK_ID: Final = "Hear-Workyard-Whole-Body-Reach-G1-v5"
OBSERVATION_SIZE: Final = 246
ACTION_SIZE: Final = 29
BALANCE_ACTION_SIZE: Final = 15
UPPER_BODY_ACTION_SIZE: Final = 14
HAND_COORDINATION_SIZE: Final = 8
TEACHER_OBSERVATION_SIZE: Final = 99
TEACHER_ACTION_SIZE: Final = 29

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TEACHER_ROOT = (
  REPOSITORY_ROOT / "assets" / "humanoid" / "controllers" / "mjlab-g1-velocity"
)
DEFAULT_TEACHER_JIT = DEFAULT_TEACHER_ROOT / "g1_velocity_teacher.jit.pt"
DEFAULT_TEACHER_REPORT = DEFAULT_TEACHER_ROOT / "training-report.json"

UPPER_BODY_JOINT_NAMES: Final = base.BODY_JOINT_NAMES[15:]
FROZEN_TEACHER_JOINT_NAMES: Final = base.BODY_JOINT_NAMES[:15]
LOWER_BODY_SLICE: Final = slice(0, 12)
WAIST_SLICE: Final = slice(12, 15)
UPPER_BODY_SLICE: Final = slice(15, 29)
FROZEN_TEACHER_ACTUATION_PROTOCOL: Final = "mjlab-unitree-g1-source-actuation-v1"
UPPER_BODY_ACTUATION_PROTOCOL: Final = "hear-harness-task-tracking-pd-v1"
TEACHER_ACTUATION_PROTOCOL: Final = (
  "hear-whole-body-reach-residual-task-tracking-actuation-v2"
)
UPPER_BODY_TASK_TRACKING_STIFFNESS: Final = tuple(
  40.0 if "wrist" in name else 80.0 for name in UPPER_BODY_JOINT_NAMES
)
BODY_ACTUATION_CONTRACT: Final[dict[str, object]] = {
  "protocol": TEACHER_ACTUATION_PROTOCOL,
  "authority": "bounded_whole_body_student_with_frozen_locomotion_reference",
  "runtime_body_model": "arm_state_conditioned_station_keeping_and_reach",
  "body_joint_count": 29,
  "locomotion_reference_joint_count": 15,
  "balance_residual_joint_count": 15,
  "residual_task_tracking_joint_count": 14,
  "frozen_protocol": FROZEN_TEACHER_ACTUATION_PROTOCOL,
  "residual_protocol": UPPER_BODY_ACTUATION_PROTOCOL,
  "task_tracking_stiffness": {"arm": 80.0, "wrist": 40.0},
  "damping_scaling": "source_damping_sqrt_stiffness_ratio",
  "joint_effort_limits": "unitree_g1_joint_actuatorfrcrange_unchanged",
  "generic_xml_position_gains_permitted": False,
}
WRIST_BROAD_REACH_SCALE_M: Final = 0.35
RESIDUAL_ENTRY_ROOT_POSITION: Final = (0.63, 0.0, 0.79)
REACH_ENTRY_OBJECT_RELATIVE_PLACEMENTS: Final = (
  (0.34, 0.06, 0.45),
  (0.34, 0.12, 0.75),
  (0.40, 0.06, 0.60),
  (0.40, 0.12, 0.90),
  (0.46, 0.06, 0.75),
  (0.46, 0.12, 1.05),
  (0.405844400461645, 0.082924357517691, 0.614),
)
REACH_ENTRY_CORRELATION_PROTOCOL: Final = (
  "hand-signed-object-relative-root-placement-catalog-v1"
)
RUNTIME_GEOMETRY_TOP_TARGET_PROTOCOL: Final = (
  "typescript-pregrasp-geometry-top-wrist-target-v1"
)
# Measured from the same TypeScript solveG1PregraspPose + G1 hand geometry
# chain used by deployment.  MJLab axes are [app.z, app.x, app.y].  The left
# and right palm offsets mirror only across the forward axis; the 0.259 m
# vertical component includes the rod top and the distal hand contact offset.
RUNTIME_GEOMETRY_TOP_WRIST_OFFSET_M: Final = (
  (-0.02634, 0.0, 0.25899),
  (0.02634, 0.0, 0.25899),
)
OPEN_HAND_JOINT_TARGETS: Final = (0.0,) * len(base.HAND_JOINT_NAMES)
REACH_TEACHER_PROTOCOL: Final = "hear-batched-adaptive-reach-teacher-v15"
REACH_TEACHER_DIAGNOSTICS_PROTOCOL: Final = (
  "hear-reach-teacher-collision-aware-diagnostics-v15"
)
REACH_CONTACT_DIAGNOSTIC_SENSOR_NAMES: Final = (
  "upper_body_any_contact",
  "upper_body_torso_contact",
  "upper_body_rod_contact",
  "upper_body_stand_contact",
)
REACH_TEACHER_FEASIBLE_POSTURE_PROTOCOL: Final = (
  "offline-collision-aware-geometry-top-placement-catalog-v1"
)
REACH_TEACHER_FEASIBLE_POSTURE_LOOKUP_PROTOCOL: Final = (
  "active-hand-and-placement-index-v1"
)
REACH_TEACHER_FEASIBLE_POSTURE_AUTHORITY_RAD: Final = 0.7
# Per-arm [placement, joint] collision-aware IK solutions for the exact
# object-relative placement catalog used by both production qualification and
# training reset. The final catalog entry is the measured Windows mission pose.
REACH_TEACHER_FEASIBLE_POSTURE_ACTIONS: Final = (
  (
    (-0.7862444992046225, 0.12462721431360141, -0.013874699529727195, -0.3169021547072469, 0.009476888811408927, -0.9061360782324888, 0.0),
    (0.0806054112835693, 0.6640788113686761, 0.3329684995933437, -0.9488604000800469, 0.05396684333666417, -0.2845859557884664, 0.0),
    (-0.8840066060434625, 0.41390622798850196, 0.07618002326025768, -0.19473200820916614, -0.001069682318055284, 0.09441980184808962, 0.0),
    (-0.08656664256701126, 0.8246209604804577, 0.5080664349308687, -0.192277245735843, 0.014340888879353553, -0.9900139768028484, 0.0),
    (-0.9393374557486266, 0.7308023761110074, 0.012981646649759945, 0.2433215520202982, 0.004893162034773152, 0.04455280611854335, 0.0),
    (-0.2936266795373517, 0.8948423949265314, 0.9560053766336235, 0.26811089517848735, 0.053868357744537874, -0.14759861681120548, 0.0),
    (-0.8047941083074475, 0.6571059302344479, -0.07736620949411913, -0.16361414000402427, 0.016460280996407513, -0.048975288099952795, 0.0),
  ),
  (
    (-0.9048428538142181, -0.2541118325965608, 0.008748110463731801, -0.39388772130637156, -0.08652918971070603, 0.27345456061819867, 0.0),
    (-0.3199112807181031, -0.6309572923609497, -0.22410315752937252, -0.28090754162482057, 0.9955430550231495, -0.9956544183452068, 0.0),
    (-0.9223998101834354, -0.20466896467882723, -0.4871963422339207, -0.13716946581593742, -0.9735554229868433, 0.014918581821070335, 0.0),
    (-0.5157789378941124, -0.8868209143985271, -0.569200902106965, 0.17906404256933653, -0.0024554607356513393, 0.11084708461263983, 0.0),
    (-0.9794275195955516, -0.6083524067916396, -0.46707055891790267, 0.24511613795828924, -0.09751568486137403, 0.10602194804637059, 0.0),
    (-0.39208447036362504, -0.9999199482031169, -0.8398513183886178, 0.4030017258207021, -0.5827960619152892, 0.08514824276822514, 0.0),
    (-0.9956025740614802, -0.5984284564040137, -0.10167425826868942, 0.14033222955234173, 0.11085503785977265, 0.10908812594303263, 0.0),
  ),
)
REACH_TEACHER_BASE_DAMPING: Final = 0.015
REACH_TEACHER_SINGULARITY_DAMPING: Final = 0.12
REACH_TEACHER_SINGULARITY_THRESHOLD: Final = 0.05
REACH_TEACHER_POSTURE_ATTRACTOR_GAIN: Final = 0.15
REACH_TEACHER_TASK_SPACE_FEEDBACK_GAIN: Final = 1.0
REACH_TEACHER_MAX_CARTESIAN_STEP_M: Final = 0.08
REACH_TEACHER_MAX_JOINT_CORRECTION_RAD: Final = 0.20
REACH_TEACHER_MAX_SOLVER_TARGET_SLEW_RAD: Final = 0.03
REACH_TEACHER_MAX_COMMAND_LEAD_RAD: Final = 0.16
REACH_TEACHER_HOLD_ENTER_ERROR_M: Final = 0.05
REACH_TEACHER_HOLD_RELEASE_ERROR_M: Final = 0.075
REACH_ENTRY_SETTLING_CONTROL_STEPS: Final = 20
REACH_TEACHER_CONTRACT: Final[dict[str, object]] = {
  "protocol": REACH_TEACHER_PROTOCOL,
  "runtime": "mujoco_warp_torch_cuda",
  "solver": "target_conditioned_feasible_posture_servo_with_dls_diagnostics",
  "target_protocol": RUNTIME_GEOMETRY_TOP_TARGET_PROTOCOL,
  "pregrasp_shell_radius_m": base.PREGRASP_SHELL_RADIUS_M,
  "pregrasp_lateral_clearance_m": base.PREGRASP_LATERAL_CLEARANCE_M,
  "active_hand_allocation": "placement_catalog_hand-v1",
  "contact_target_activation": "contact_authority_only",
  "success_metric": "active_wrist_to_command_within_tolerance",
  "controlled_joint_count": UPPER_BODY_ACTION_SIZE,
  "task_dimension_per_arm": 3,
  "target_memory": "per_environment_measured_joint_anchored_anti_windup",
  "base_damping": REACH_TEACHER_BASE_DAMPING,
  "singularity_damping": REACH_TEACHER_SINGULARITY_DAMPING,
  "singularity_threshold": REACH_TEACHER_SINGULARITY_THRESHOLD,
  "feasible_posture_protocol": REACH_TEACHER_FEASIBLE_POSTURE_PROTOCOL,
  "feasible_posture_lookup_protocol": (
    REACH_TEACHER_FEASIBLE_POSTURE_LOOKUP_PROTOCOL
  ),
  "feasible_posture_authority_rad": (
    REACH_TEACHER_FEASIBLE_POSTURE_AUTHORITY_RAD
  ),
  "feasible_posture_placement_catalog": [
    list(placement) for placement in REACH_ENTRY_OBJECT_RELATIVE_PLACEMENTS
  ],
  "feasible_posture_normalized_actions": [
    [list(placement) for placement in arm]
    for arm in REACH_TEACHER_FEASIBLE_POSTURE_ACTIONS
  ],
  "feasible_posture_offline_validation": {
    "protocol": "hear-collision-aware-reach-posture-fit-v2",
    "sample_count": 14,
    "tolerance_m": 0.06,
    "collision_clearance_m": 0.005,
    "success_rate": 13 / 14,
    "kinematic_tolerance_rate": 13 / 14,
    "collision_clear_rate": 1.0,
    "mean_error_m": 0.018788770715006734,
    "p90_error_m": 0.05136783438503414,
    "maximum_error_m": 0.06791992759195634,
    "minimum_clearance_m": 0.03313698178789586,
  },
  "posture_attractor_gain": REACH_TEACHER_POSTURE_ATTRACTOR_GAIN,
  "task_space_feedback_gain": REACH_TEACHER_TASK_SPACE_FEEDBACK_GAIN,
  "max_cartesian_step_m": REACH_TEACHER_MAX_CARTESIAN_STEP_M,
  "max_joint_correction_rad": REACH_TEACHER_MAX_JOINT_CORRECTION_RAD,
  "max_solver_target_slew_rad": REACH_TEACHER_MAX_SOLVER_TARGET_SLEW_RAD,
  "max_command_lead_rad": REACH_TEACHER_MAX_COMMAND_LEAD_RAD,
  "hold_enter_error_m": REACH_TEACHER_HOLD_ENTER_ERROR_M,
  "hold_release_error_m": REACH_TEACHER_HOLD_RELEASE_ERROR_M,
  "entry_settling_control_steps": REACH_ENTRY_SETTLING_CONTROL_STEPS,
  "supervision": "online_dagger_and_ppo_rollout_labels_only",
  "actor_observation_exposure": False,
  "execution_authority": "none",
  "cpu_round_trip_per_label": False,
  "diagnostics_protocol": REACH_TEACHER_DIAGNOSTICS_PROTOCOL,
}
DYNAMIC_COM_PROTOCOL: Final = "hear-support-relative-dynamic-com-v1"
DYNAMIC_COM_SETTLING_STEPS: Final = 10
GRAVITY_M_S2: Final = 9.81


def _sha256(path: Path) -> str:
  digest = hashlib.sha256()
  with path.open("rb") as handle:
    for chunk in iter(lambda: handle.read(1024 * 1024), b""):
      digest.update(chunk)
  return digest.hexdigest()


def _metadata_vector(
  source: object,
  label: str,
  size: int,
  predicate=lambda _: True,
) -> tuple[float, ...]:
  if not isinstance(source, str):
    raise ValueError(f"Locomotion teacher {label} metadata is missing")
  try:
    values = tuple(float(value) for value in source.split(","))
  except ValueError as error:
    raise ValueError(f"Locomotion teacher {label} metadata is invalid") from error
  if len(values) != size or any(
    not math.isfinite(value) or not predicate(value) for value in values
  ):
    raise ValueError(f"Locomotion teacher {label} metadata is invalid")
  return values


def _teacher_source_actuation(
  report_path: Path,
) -> tuple[dict[str, float], ...]:
  """Resolve the exact mjlab G1 actuation used by the teacher source task.

  The HEAR model with hands ships XML position actuators with a generic kp=500.
  The velocity teacher was trained against mjlab's motor-specific stiffness,
  damping, armature, and effort limits.  Reusing only its policy weights while
  retaining the generic XML gains changes the closed-loop plant and produces
  unsafe reset transients.  Resolve the source model from the pinned mjlab
  package and cross-check its rounded values against the exported report.
  """
  from mjlab.asset_zoo.robots.unitree_g1.g1_constants import G1_ARTICULATION

  report = json.loads(report_path.read_text(encoding="utf-8"))
  onnx = report.get("onnx")
  metadata = onnx.get("metadata") if isinstance(onnx, dict) else None
  if not isinstance(metadata, dict):
    raise ValueError("Locomotion teacher report has no controller metadata")
  reported_stiffness = _metadata_vector(
    metadata.get("joint_stiffness"), "joint stiffness", TEACHER_ACTION_SIZE,
    lambda value: value > 0.0,
  )
  reported_damping = _metadata_vector(
    metadata.get("joint_damping"), "joint damping", TEACHER_ACTION_SIZE,
    lambda value: value > 0.0,
  )

  source: list[dict[str, float]] = []
  for index, joint_name in enumerate(base.BODY_JOINT_NAMES):
    matches = [
      actuator
      for actuator in G1_ARTICULATION.actuators
      if any(re.fullmatch(pattern, joint_name) for pattern in actuator.target_names_expr)
    ]
    if len(matches) != 1:
      raise ValueError(
        f"Teacher source actuation is ambiguous for {joint_name}: {len(matches)}"
      )
    actuator = matches[0]
    if (
      actuator.stiffness is None
      or actuator.damping is None
      or actuator.armature is None
      or actuator.effort_limit is None
    ):
      raise ValueError(f"Teacher source actuation is incomplete for {joint_name}")
    stiffness = float(actuator.stiffness)
    damping = float(actuator.damping)
    if (
      not math.isclose(stiffness, reported_stiffness[index], abs_tol=5e-4)
      or not math.isclose(damping, reported_damping[index], abs_tol=5e-4)
    ):
      raise ValueError(
        f"Teacher report and pinned mjlab actuation disagree for {joint_name}"
      )
    source.append({
      "stiffness": stiffness,
      "damping": damping,
      "armature": float(actuator.armature),
      "effort_limit": float(actuator.effort_limit),
    })
  return tuple(source)


def _load_hybrid_actuated_g1_spec() -> mujoco.MjSpec:
  """Load the G1 with actuation partitioned by the Harness ownership boundary.

  The frozen lower body and waist retain the exact plant used by the locomotion
  teacher.  Teacher upper-body actions are never applied, so the residual-owned
  arm/wrist joints instead use the production Harness task-tracking stiffness.
  Damping preserves the source damping ratio and the XML's joint-level hardware
  effort limits remain unchanged.
  """
  spec = base._load_g1_spec()
  source = _teacher_source_actuation(DEFAULT_TEACHER_REPORT)
  joints = {joint.name: joint for joint in spec.joints}
  actuators = {actuator.target: actuator for actuator in spec.actuators}
  for index, (joint_name, parameters) in enumerate(zip(
    base.BODY_JOINT_NAMES, source, strict=True
  )):
    joint = joints.get(joint_name)
    actuator = actuators.get(joint_name)
    if joint is None or actuator is None:
      raise ValueError(f"Teacher-aligned G1 is missing body actuation for {joint_name}")
    joint.armature = parameters["armature"]
    # The teacher source MJCF has no passive joint damping or friction loss;
    # its damping is entirely in the position actuator.
    joint.damping[:] = 0.0
    joint.frictionloss = 0.0
    stiffness = parameters["stiffness"]
    damping = parameters["damping"]
    if index >= UPPER_BODY_SLICE.start:
      stiffness = UPPER_BODY_TASK_TRACKING_STIFFNESS[index - UPPER_BODY_SLICE.start]
      damping *= math.sqrt(stiffness / parameters["stiffness"])
    actuator.set_to_position(kp=stiffness, kv=damping, inheritrange=True)
  return spec


def _reach_contact_diagnostic_sensors() -> tuple[ContactSensorCfg, ...]:
  """Observe collision blockers without exposing them to the student policy."""
  upper_body = ContactMatch(
    mode="body",
    pattern=r"^(?:left|right)_(?:shoulder|elbow|wrist|hand).+_link$",
    entity="robot",
  )

  def sensor(name: str, secondary: ContactMatch | None) -> ContactSensorCfg:
    return ContactSensorCfg(
      name=name,
      primary=upper_body,
      secondary=secondary,
      fields=("found", "force", "dist"),
      reduce="maxforce",
      num_slots=1,
      history_length=4,
    )

  return (
    sensor("upper_body_any_contact", None),
    sensor(
      "upper_body_torso_contact",
      ContactMatch(mode="body", pattern="torso_link", entity="robot"),
    ),
    sensor(
      "upper_body_rod_contact",
      ContactMatch(mode="body", pattern="assembly_rod", entity="assembly_rod"),
    ),
    sensor(
      "upper_body_stand_contact",
      ContactMatch(mode="body", pattern="pickup_stand", entity="pickup_stand"),
    ),
  )


class FrozenLocomotionTeacher:
  """Validated, no-gradient, dynamic-batch TorchScript velocity teacher."""

  def __init__(
    self,
    jit_path: Path,
    report_path: Path,
    device: torch.device | str,
    validation_batch_size: int,
  ):
    if not jit_path.is_file():
      raise FileNotFoundError(f"Frozen locomotion teacher is missing: {jit_path}")
    if not report_path.is_file():
      raise FileNotFoundError(f"Locomotion teacher report is missing: {report_path}")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report.get("version") != 1:
      raise ValueError("Locomotion teacher report version is unsupported")
    if report.get("task") != "Mjlab-Velocity-Flat-Unitree-G1":
      # Do not trust a same-named artifact from another policy family.
      raise ValueError("Locomotion teacher report has the wrong source task")
    jit = report.get("teacher_jit")
    if not isinstance(jit, dict):
      raise ValueError("Locomotion teacher report has no validated JIT artifact")
    if (
      jit.get("file") != jit_path.name
      or jit.get("bytes") != jit_path.stat().st_size
      or jit.get("sha256") != _sha256(jit_path)
      or jit.get("input") != "obs"
      or jit.get("input_size") != TEACHER_OBSERVATION_SIZE
      or jit.get("output") != "actions"
      or jit.get("output_size") != TEACHER_ACTION_SIZE
      or jit.get("batch_dynamic") is not True
      or jit.get("runtime") != "torchscript_cuda"
    ):
      raise ValueError("Locomotion teacher JIT identity does not match its report")

    onnx = report.get("onnx")
    metadata = onnx.get("metadata") if isinstance(onnx, dict) else None
    if not isinstance(metadata, dict):
      raise ValueError("Locomotion teacher report has no controller metadata")
    joint_names = metadata.get("joint_names")
    if not isinstance(joint_names, str) or tuple(joint_names.split(",")) != base.BODY_JOINT_NAMES:
      raise ValueError("Locomotion teacher joint order differs from Workyard")
    if metadata.get("observation_names") != (
      "base_lin_vel,base_ang_vel,projected_gravity,joint_pos,joint_vel,actions,command"
    ):
      raise ValueError("Locomotion teacher observation order is incompatible")

    self.default_joint_positions = torch.tensor(
      _metadata_vector(
        metadata.get("default_joint_pos"), "default joint position", TEACHER_ACTION_SIZE
      ),
      dtype=torch.float32,
      device=device,
    ).unsqueeze(0)
    expected_default = torch.tensor(
      base.BODY_DEFAULT_POSITIONS, dtype=torch.float32, device=device
    ).unsqueeze(0)
    if not torch.allclose(self.default_joint_positions, expected_default, atol=1e-5):
      raise ValueError("Locomotion teacher neutral pose differs from Workyard")
    self.action_scale = torch.tensor(
      _metadata_vector(
        metadata.get("action_scale"),
        "action scale",
        TEACHER_ACTION_SIZE,
        lambda value: value > 0.0,
      ),
      dtype=torch.float32,
      device=device,
    ).unsqueeze(0)

    loaded = torch.jit.load(str(jit_path), map_location=device).eval()
    parameter_count = sum(parameter.numel() for parameter in loaded.parameters())
    for parameter in loaded.parameters():
      parameter.requires_grad_(False)
    self.module = torch.jit.freeze(loaded)
    self.parameter_count = parameter_count
    self.gradient_parameter_count = sum(
      parameter.numel()
      for parameter in self.module.parameters()
      if parameter.requires_grad
    )
    if self.gradient_parameter_count != 0:
      raise RuntimeError("Frozen locomotion teacher still owns gradient parameters")
    self.identity = {
      "protocol": "hear-frozen-locomotion-teacher-v1",
      "jit_sha256": jit["sha256"],
      "report_sha256": _sha256(report_path),
      "parameter_count": parameter_count,
      "gradient_parameter_count": self.gradient_parameter_count,
      "runtime": "torchscript_cuda",
      "actuation_protocol": TEACHER_ACTUATION_PROTOCOL,
      "frozen_actuation_protocol": FROZEN_TEACHER_ACTUATION_PROTOCOL,
      "upper_body_actuation_protocol": UPPER_BODY_ACTUATION_PROTOCOL,
      "source_actuation_joint_count": len(FROZEN_TEACHER_JOINT_NAMES),
      "task_tracking_actuation_joint_count": len(UPPER_BODY_JOINT_NAMES),
    }

    with torch.inference_mode():
      probe = torch.zeros(
        (validation_batch_size, TEACHER_OBSERVATION_SIZE),
        dtype=torch.float32,
        device=device,
      )
      output = self.module(probe)
    if tuple(output.shape) != (validation_batch_size, TEACHER_ACTION_SIZE):
      raise RuntimeError(
        "Frozen locomotion teacher does not support the environment batch: "
        f"{tuple(output.shape)}"
      )
    if not torch.isfinite(output).all():
      raise RuntimeError("Frozen locomotion teacher validation output is non-finite")

  def infer(self, observation: torch.Tensor) -> torch.Tensor:
    if observation.ndim != 2 or observation.shape[1] != TEACHER_OBSERVATION_SIZE:
      raise ValueError(f"Expected teacher observation [B, 99], got {observation.shape}")
    with torch.inference_mode():
      action = self.module(observation)
    if action.shape != (observation.shape[0], TEACHER_ACTION_SIZE):
      raise RuntimeError(f"Locomotion teacher action shape drifted: {action.shape}")
    if not torch.isfinite(action).all():
      raise RuntimeError("Locomotion teacher emitted a non-finite action")
    return action


@dataclass(kw_only=True)
class WorkyardResidualCommandCfg(base.WorkyardCommandCfg):
  """Reach commands sampled from the real post-navigation deployment state."""

  def build(self, env: ManagerBasedRlEnv) -> "WorkyardResidualCommand":
    return WorkyardResidualCommand(self, env)


class WorkyardResidualCommand(base.WorkyardCommand):
  """Uses the runtime geometry-top wrist target instead of a side-only proxy."""

  def __init__(self, cfg: WorkyardResidualCommandCfg, env: ManagerBasedRlEnv):
    super().__init__(cfg, env)
    self.reach_entry_placement_index = torch.zeros(
      self.num_envs, dtype=torch.long, device=self.device
    )

  def _resample_command(self, env_ids: torch.Tensor) -> None:
    super()._resample_command(env_ids)
    count = len(env_ids)
    origins = self._env.scene.env_origins[env_ids]
    active_hand = torch.randint(0, 2, (count,), device=self.device)
    side = torch.where(
      active_hand == 1,
      torch.ones(count, dtype=torch.float32, device=self.device),
      -torch.ones(count, dtype=torch.float32, device=self.device),
    )
    placement_catalog = torch.tensor(
      REACH_ENTRY_OBJECT_RELATIVE_PLACEMENTS,
      dtype=torch.float32,
      device=self.device,
    )
    placement_index = torch.randint(
      0, len(REACH_ENTRY_OBJECT_RELATIVE_PLACEMENTS), (count,), device=self.device
    )
    placement = placement_catalog[placement_index]
    forward_offset = placement[:, 0]
    lateral_magnitude = placement[:, 1]
    yaw_magnitude = placement[:, 2]
    rod_pos = self.initial_rod_position[env_ids]
    root_pos = torch.empty((count, 3), dtype=torch.float32, device=self.device)
    root_pos[:, 0] = rod_pos[:, 0] - forward_offset
    root_pos[:, 1] = rod_pos[:, 1] + side * lateral_magnitude
    root_pos[:, 2] = RESIDUAL_ENTRY_ROOT_POSITION[2] + origins[:, 2]
    yaw = side * yaw_magnitude
    zero = torch.zeros_like(yaw)
    root_quat = base.quat_from_euler_xyz(zero, zero, yaw)
    self.robot.data.write_root_pose(
      torch.cat((root_pos, root_quat), dim=-1), env_ids
    )
    self.robot.data.write_root_velocity(
      torch.zeros((count, 6), dtype=torch.float32, device=self.device), env_ids
    )
    self.active_hand[env_ids] = active_hand
    self.reach_entry_placement_index[env_ids] = placement_index

  def _update_task_space_targets(self) -> None:
    super()._update_task_space_targets()
    wrist_pose = self.robot.data.body_link_pose_w[:, self._wrist_body_ids]
    rod_pos = self.rod.data.root_link_pos_w
    root_pos = self.robot.data.root_link_pos_w
    root_quat = self.robot.data.root_link_quat_w
    offsets = torch.tensor(
      RUNTIME_GEOMETRY_TOP_WRIST_OFFSET_M,
      dtype=rod_pos.dtype,
      device=self.device,
    ).unsqueeze(0).expand(self.num_envs, -1, -1)
    targets = rod_pos.unsqueeze(1) + offsets
    active_mask = torch_f.one_hot(
      self.active_hand, num_classes=2
    ).bool().unsqueeze(-1)
    target_pos = torch.where(active_mask, targets, wrist_pose[..., :3])
    pos_pelvis = base.quat_apply_inverse(
      root_quat.unsqueeze(1).expand(-1, 2, -1),
      target_pos - root_pos.unsqueeze(1),
    )
    # Phase-one reach owns position only.  Preserve each measured wrist
    # orientation so DAgger cannot learn an orientation proxy unavailable to
    # its position-only analytic teacher.
    quat_pelvis = base.quat_mul(
      base.quat_conjugate(root_quat).unsqueeze(1).expand(-1, 2, -1),
      wrist_pose[..., 3:7],
    )
    self.wrist_targets_pelvis[:] = torch.cat(
      (pos_pelvis, quat_pelvis), dim=-1
    ).reshape(self.num_envs, 14)


@dataclass(kw_only=True)
class WorkyardResidualActionCfg(ActionTermCfg):
  balance_scale: float = 0.12
  upper_body_scale: float = 0.7
  teacher_jit_path: str = str(DEFAULT_TEACHER_JIT)
  teacher_report_path: str = str(DEFAULT_TEACHER_REPORT)

  def build(self, env: ManagerBasedRlEnv) -> "WorkyardResidualAction":
    return WorkyardResidualAction(self, env)


@dataclass(frozen=True)
class SupportRelativeDynamicComState:
  observation: torch.Tensor
  capture_point_pelvis: torch.Tensor
  support_margin: torch.Tensor
  foot_position_w: torch.Tensor
  foot_planar_speed: torch.Tensor
  double_support: torch.Tensor
  no_foot_contact: torch.Tensor


class SupportRelativeDynamicCom:
  """GPU-only support-relative whole-body CoM state and safety evidence."""

  def __init__(self, env: ManagerBasedRlEnv, robot):
    self._env = env
    self.robot = robot
    foot_ids, names = robot.find_bodies(
      ("left_ankle_roll_link", "right_ankle_roll_link"), preserve_order=True
    )
    if tuple(names) != ("left_ankle_roll_link", "right_ankle_roll_link"):
      raise ValueError("Dynamic-CoM requires both G1 foot frames")
    self._foot_ids = torch.tensor(foot_ids, dtype=torch.long, device=env.device)
    self._root_body_id = robot.indexing.root_body_id
    # Four conservative sole points in each ankle-roll frame.  This is an
    # explicitly approximate support polygon; no world-axis shortcut is used.
    self._sole_corners = torch.tensor(
      (
        (-0.05, -0.03, 0.0),
        (-0.05, 0.03, 0.0),
        (0.12, -0.03, 0.0),
        (0.12, 0.03, 0.0),
      ),
      dtype=torch.float32,
      device=env.device,
    )

  def compute(self) -> SupportRelativeDynamicComState:
    root_pos = self.robot.data.root_link_pos_w
    root_quat = self.robot.data.root_link_quat_w
    foot_pose = self.robot.data.body_link_pose_w[:, self._foot_ids]
    foot_velocity = self.robot.data.body_link_lin_vel_w[:, self._foot_ids]
    foot_found, _, foot_slip = base._foot_contact_summary(self._env)
    contact = foot_found > 0.0
    no_contact = ~contact.any(dim=-1)
    effective_contact = torch.where(
      no_contact.unsqueeze(-1), torch.ones_like(foot_found), foot_found
    )
    weights = effective_contact / effective_contact.sum(dim=-1, keepdim=True)
    support_position_w = torch.sum(
      foot_pose[..., :3] * weights.unsqueeze(-1), dim=1
    )
    support_velocity_w = torch.sum(
      foot_velocity * weights.unsqueeze(-1), dim=1
    )

    # subtree_com/cvel at the entity root describe the whole articulated G1,
    # not merely the pelvis link.
    com_position_w = self.robot.data.data.subtree_com[:, self._root_body_id]
    com_velocity_w = self.robot.data.data.cvel[:, self._root_body_id, 3:6]
    relative_position_p = base.quat_apply_inverse(
      root_quat, com_position_w - support_position_w
    )
    relative_velocity_p = base.quat_apply_inverse(
      root_quat, com_velocity_w - support_velocity_w
    )
    observation = torch.cat(
      (relative_position_p[:, :2], relative_velocity_p[:, :2]), dim=-1
    )

    height = (com_position_w[:, 2] - support_position_w[:, 2]).clamp(0.25, 1.50)
    capture_point = observation[:, :2] + observation[:, 2:4] * torch.sqrt(
      height.unsqueeze(-1) / GRAVITY_M_S2
    )

    batch = self._env.num_envs
    corners = self._sole_corners.view(1, 1, 4, 3).expand(batch, 2, -1, -1)
    foot_quat = foot_pose[..., 3:7].unsqueeze(2).expand(-1, -1, 4, -1)
    corners_w = foot_pose[..., :3].unsqueeze(2) + quat_apply(foot_quat, corners)
    corners_p = base.quat_apply_inverse(
      root_quat[:, None, None, :].expand(-1, 2, 4, -1),
      corners_w - root_pos[:, None, None, :],
    )
    support_position_p = base.quat_apply_inverse(
      root_quat, support_position_w - root_pos
    )
    relative_corners = corners_p[..., :2] - support_position_p[:, None, None, :2]
    active_corners = effective_contact.bool().unsqueeze(-1).expand(-1, -1, 4)
    positive_inf = torch.full_like(relative_corners[..., 0], float("inf"))
    negative_inf = torch.full_like(relative_corners[..., 0], -float("inf"))
    minimum = torch.stack(
      (
        torch.where(active_corners, relative_corners[..., 0], positive_inf).amin(
          dim=(1, 2)
        ),
        torch.where(active_corners, relative_corners[..., 1], positive_inf).amin(
          dim=(1, 2)
        ),
      ),
      dim=-1,
    )
    maximum = torch.stack(
      (
        torch.where(active_corners, relative_corners[..., 0], negative_inf).amax(
          dim=(1, 2)
        ),
        torch.where(active_corners, relative_corners[..., 1], negative_inf).amax(
          dim=(1, 2)
        ),
      ),
      dim=-1,
    )
    support_margin = torch.stack(
      (
        capture_point[:, 0] - minimum[:, 0],
        maximum[:, 0] - capture_point[:, 0],
        capture_point[:, 1] - minimum[:, 1],
        maximum[:, 1] - capture_point[:, 1],
      ),
      dim=-1,
    ).amin(dim=-1)
    return SupportRelativeDynamicComState(
      observation=observation,
      capture_point_pelvis=capture_point,
      support_margin=support_margin,
      foot_position_w=foot_pose[..., :3],
      foot_planar_speed=foot_slip,
      double_support=contact.all(dim=-1),
      no_foot_contact=no_contact,
    )


class BatchedTaskSpaceReachTeacher:
  """Stateful 2x7-DoF adaptive-DLS teacher used only for supervision labels."""

  def __init__(
    self,
    env: ManagerBasedRlEnv,
    robot,
    upper_joint_ids: torch.Tensor,
    upper_body_scale: float,
  ):
    if upper_joint_ids.numel() != UPPER_BODY_ACTION_SIZE:
      raise ValueError("Reach teacher requires exactly fourteen arm/wrist joints")
    self._env = env
    self.robot = robot
    self._joint_ids = upper_joint_ids.reshape(2, 7)
    self._joint_dof_ids = tuple(
      robot.indexing.joint_v_adr[ids] for ids in self._joint_ids
    )
    wrist_ids, wrist_names = robot.find_bodies(
      ("left_wrist_yaw_link", "right_wrist_yaw_link"), preserve_order=True
    )
    if tuple(wrist_names) != ("left_wrist_yaw_link", "right_wrist_yaw_link"):
      raise ValueError("Reach teacher requires both G1 wrist frames")
    self._wrist_ids = torch.tensor(wrist_ids, dtype=torch.long, device=env.device)
    global_wrist_ids = robot.indexing.body_ids[self._wrist_ids]
    self._neutral = torch.tensor(
      base.BODY_DEFAULT_POSITIONS[15:], dtype=torch.float32, device=env.device
    ).reshape(1, 2, 7)
    self._soft_lower = robot.data.soft_joint_pos_limits[:, self._joint_ids, 0] + 0.02
    self._soft_upper = robot.data.soft_joint_pos_limits[:, self._joint_ids, 1] - 0.02
    self._action_scale = float(upper_body_scale)
    self._authority_lower = torch.maximum(
      self._soft_lower, self._neutral - self._action_scale
    )
    self._authority_upper = torch.minimum(
      self._soft_upper, self._neutral + self._action_scale
    )
    self._feasible_posture_actions = torch.tensor(
      REACH_TEACHER_FEASIBLE_POSTURE_ACTIONS,
      dtype=torch.float32,
      device=env.device,
    )
    self.base_damping = REACH_TEACHER_BASE_DAMPING
    self.singularity_damping = REACH_TEACHER_SINGULARITY_DAMPING
    self.singularity_threshold = REACH_TEACHER_SINGULARITY_THRESHOLD
    self.posture_attractor_gain = REACH_TEACHER_POSTURE_ATTRACTOR_GAIN
    self.task_space_feedback_gain = REACH_TEACHER_TASK_SPACE_FEEDBACK_GAIN
    self.wrist_bearing_feedback_gain = 0.0
    self.wrist_axis_alignment_feedback_gain = 0.0
    self.wrist_bearing_task_weight_m_per_rad = 0.12
    self.max_wrist_bearing_step_rad = 0.35
    self.max_cartesian_step_m = REACH_TEACHER_MAX_CARTESIAN_STEP_M
    self.max_joint_correction_rad = REACH_TEACHER_MAX_JOINT_CORRECTION_RAD
    self.max_solver_target_slew_rad = REACH_TEACHER_MAX_SOLVER_TARGET_SLEW_RAD
    # Optional phase-specific cap used by contact-capable subclasses.  The
    # generic reach teacher keeps one limit; a terminal executor may slow only
    # the final typed pocket without delaying reach/retreat/alignment.
    self.pocket_max_solver_target_slew_rad: float | None = None
    self.max_command_lead_rad = REACH_TEACHER_MAX_COMMAND_LEAD_RAD
    self.pocket_max_command_lead_rad: float | None = None
    self.hold_enter_error_m = REACH_TEACHER_HOLD_ENTER_ERROR_M
    self.hold_release_error_m = REACH_TEACHER_HOLD_RELEASE_ERROR_M

    nworld = env.num_envs
    nv = env.sim.mj_model.nv
    self._joint_target = self._neutral.expand(nworld, -1, -1).clone()
    self._feasible_posture = self._neutral.expand(nworld, -1, -1).clone()
    self._initialized = torch.zeros(nworld, dtype=torch.bool, device=env.device)
    self._last_episode_length = torch.full(
      (nworld,), -1, dtype=torch.long, device=env.device
    )
    self._cache_valid = torch.zeros(nworld, dtype=torch.bool, device=env.device)
    self._cached_action = torch.zeros(
      (nworld, 2, 7), dtype=torch.float32, device=env.device
    )
    self._holding_target = torch.zeros(
      (nworld, 2), dtype=torch.bool, device=env.device
    )
    self._unclamped_action = torch.zeros(
      (nworld, 2, 7), dtype=torch.float32, device=env.device
    )
    self._authority_saturation = torch.zeros(
      (nworld, 2, 7), dtype=torch.bool, device=env.device
    )
    self._soft_limit_saturation = torch.zeros_like(self._authority_saturation)
    self._command_lead_saturation = torch.zeros_like(self._authority_saturation)
    self._solver_target_slew = torch.zeros_like(self._unclamped_action)
    self._joint_target_step = torch.zeros_like(self._unclamped_action)
    self._minimum_singular_value = torch.zeros(
      (nworld, 2), dtype=torch.float32, device=env.device
    )
    self._primary_delta = torch.zeros_like(self._unclamped_action)
    self._secondary_delta = torch.zeros_like(self._unclamped_action)
    self._projected_posture_delta = torch.zeros_like(self._unclamped_action)
    self._joint_correction = torch.zeros_like(self._unclamped_action)
    self._instantaneous_target = torch.zeros_like(self._unclamped_action)
    self._authority_limited_target = torch.zeros_like(self._unclamped_action)
    self._lower_priority_authority_revoked = torch.zeros(
      (nworld, 2), dtype=torch.bool, device=env.device
    )
    self._wrist_bearing_error = torch.zeros(
      (nworld, 2), dtype=torch.float32, device=env.device
    )
    self._wrist_axis_alignment_error = torch.zeros_like(
      self._wrist_bearing_error
    )
    self._feasible_posture_feature_clamped = torch.zeros(
      (nworld, 2, 2), dtype=torch.bool, device=env.device
    )
    self._feasible_posture_action_clamped = torch.zeros(
      (nworld, 2, 7), dtype=torch.bool, device=env.device
    )
    self._task_identity = torch.eye(
      3, dtype=torch.float32, device=env.device
    ).unsqueeze(0)
    self._bearing_task_identity = torch.eye(
      4, dtype=torch.float32, device=env.device
    ).unsqueeze(0)
    self._pose_task_identity = torch.eye(
      6, dtype=torch.float32, device=env.device
    ).unsqueeze(0)
    self._joint_identity = torch.eye(
      7, dtype=torch.float32, device=env.device
    ).unsqueeze(0)
    self._jacp_wp = []
    self._jacr_wp = []
    self._point_wp = []
    self._body_wp = []
    self._jacp_torch = []
    self._jacr_torch = []
    self._point_torch = []
    with wp.ScopedDevice(env.sim.wp_device):
      for arm in range(2):
        jacp_wp = wp.zeros((nworld, 3, nv), dtype=float)
        jacr_wp = wp.zeros((nworld, 3, nv), dtype=float)
        point_wp = wp.zeros(nworld, dtype=wp.vec3)
        body_wp = wp.zeros(nworld, dtype=wp.int32)
        body_wp.fill_(int(global_wrist_ids[arm].item()))
        self._jacp_wp.append(jacp_wp)
        self._jacr_wp.append(jacr_wp)
        self._point_wp.append(point_wp)
        self._body_wp.append(body_wp)
        self._jacp_torch.append(wp.to_torch(jacp_wp))
        self._jacr_torch.append(wp.to_torch(jacr_wp))
        self._point_torch.append(wp.to_torch(point_wp).view(nworld, 3))
    self.identity = dict(REACH_TEACHER_CONTRACT)

  @property
  def unclamped_action(self) -> torch.Tensor:
    return self._unclamped_action.reshape(self._env.num_envs, UPPER_BODY_ACTION_SIZE)

  @property
  def authority_saturation(self) -> torch.Tensor:
    return self._authority_saturation.reshape(self._env.num_envs, UPPER_BODY_ACTION_SIZE)

  @property
  def soft_limit_saturation(self) -> torch.Tensor:
    return self._soft_limit_saturation.reshape(self._env.num_envs, UPPER_BODY_ACTION_SIZE)

  @property
  def command_lead_saturation(self) -> torch.Tensor:
    return self._command_lead_saturation.reshape(self._env.num_envs, UPPER_BODY_ACTION_SIZE)

  @property
  def joint_target_step(self) -> torch.Tensor:
    return self._joint_target_step.reshape(self._env.num_envs, UPPER_BODY_ACTION_SIZE)

  @property
  def solver_target_slew(self) -> torch.Tensor:
    return self._solver_target_slew.reshape(self._env.num_envs, UPPER_BODY_ACTION_SIZE)

  @property
  def holding_target(self) -> torch.Tensor:
    return self._holding_target

  @property
  def minimum_singular_value(self) -> torch.Tensor:
    return self._minimum_singular_value

  @property
  def primary_delta(self) -> torch.Tensor:
    return self._primary_delta

  @property
  def secondary_delta(self) -> torch.Tensor:
    return self._secondary_delta

  @property
  def projected_posture_delta(self) -> torch.Tensor:
    return self._projected_posture_delta

  @property
  def joint_correction(self) -> torch.Tensor:
    return self._joint_correction

  @property
  def instantaneous_target(self) -> torch.Tensor:
    return self._instantaneous_target

  @property
  def authority_limited_target(self) -> torch.Tensor:
    return self._authority_limited_target

  @property
  def wrist_bearing_error(self) -> torch.Tensor:
    return self._wrist_bearing_error

  @property
  def wrist_axis_alignment_error(self) -> torch.Tensor:
    return self._wrist_axis_alignment_error

  @property
  def feasible_posture_feature_clamped(self) -> torch.Tensor:
    return self._feasible_posture_feature_clamped

  @property
  def feasible_posture_action_clamped(self) -> torch.Tensor:
    return self._feasible_posture_action_clamped

  def rewind_active_arm_target(
    self,
    mask: torch.Tensor,
    upper_targets: torch.Tensor,
  ) -> None:
    """Synchronize the persistent DLS target after a guarded contact retreat."""
    if mask.shape != (self._env.num_envs,):
      raise ValueError(f"Reach rewind mask drifted: {mask.shape}")
    if upper_targets.shape != (self._env.num_envs, UPPER_BODY_ACTION_SIZE):
      raise ValueError(f"Reach rewind target drifted: {upper_targets.shape}")
    command = base._workyard_command(self._env)
    rows = torch.arange(self._env.num_envs, device=self._env.device)
    active = command.active_hand
    selected = upper_targets.reshape(self._env.num_envs, 2, 7)[rows, active]
    current = self._joint_target[rows, active]
    self._joint_target[rows, active] = torch.where(
      mask.unsqueeze(-1), selected, current
    )
    self._holding_target[rows, active] = torch.where(
      mask,
      torch.zeros_like(mask),
      self._holding_target[rows, active],
    )

  def infer(self) -> torch.Tensor:
    command = base._workyard_command(self._env)
    root_pos = self.robot.data.root_link_pos_w
    root_quat = self.robot.data.root_link_quat_w
    wrist_position_w = self.robot.data.body_link_pos_w[:, self._wrist_ids]
    wrist_quat_w = self.robot.data.body_link_quat_w[:, self._wrist_ids]
    target_position_p = command.wrist_targets_pelvis.reshape(
      self._env.num_envs, 2, 7
    )[..., :3]
    if isinstance(command, WorkyardResidualCommand):
      placement_index = command.reach_entry_placement_index
      raw_normalized_posture = self._feasible_posture_actions[
        :, placement_index, :
      ].permute(1, 0, 2) * (
        REACH_TEACHER_FEASIBLE_POSTURE_AUTHORITY_RAD / self._action_scale
      )
    else:
      # Contact training owns a different target distribution and is retrained
      # only after the reach deployment is qualified. Do not inject the removed
      # side-pregrasp regression into that task through this reach-v4 teacher.
      raw_normalized_posture = torch.zeros(
        (self._env.num_envs, 2, 7),
        dtype=torch.float32,
        device=self._env.device,
      )
    normalized_posture = raw_normalized_posture.clamp(-1.0, 1.0)
    requested_posture = self._neutral + self._action_scale * normalized_posture
    mapped_feasible_posture = torch.maximum(
      self._authority_lower,
      torch.minimum(requested_posture, self._authority_upper),
    )
    target_position_w = root_pos.unsqueeze(1) + quat_apply(
      root_quat.unsqueeze(1).expand(-1, 2, -1), target_position_p
    )
    episode_length = self._env.episode_length_buf
    same_control_step = self._cache_valid & (
      episode_length == self._last_episode_length
    )
    advance = ~same_control_step
    reset_state = advance & (
      (~self._initialized) | (episode_length <= self._last_episode_length)
    )
    joint_positions = self.robot.data.joint_pos[:, self._joint_ids]
    self._joint_target[:] = torch.where(
      reset_state[:, None, None], joint_positions, self._joint_target
    )
    self._feasible_posture[:] = torch.where(
      reset_state[:, None, None], mapped_feasible_posture, self._feasible_posture
    )
    self._holding_target[:] = torch.where(
      reset_state[:, None], False, self._holding_target
    )
    self._feasible_posture_feature_clamped[:] = torch.where(
      reset_state[:, None, None], False,
      self._feasible_posture_feature_clamped,
    )
    self._feasible_posture_action_clamped[:] = torch.where(
      reset_state[:, None, None], raw_normalized_posture != normalized_posture,
      self._feasible_posture_action_clamped,
    )
    new_output = torch.zeros_like(self._cached_action)

    for arm in range(2):
      frame_position = wrist_position_w[:, arm]
      self._point_torch[arm][:] = frame_position
      with wp.ScopedDevice(self._env.sim.wp_device):
        mjwarp.jac(
          self._env.sim.wp_model,
          self._env.sim.wp_data,
          self._jacp_wp[arm],
          self._jacr_wp[arm],
          self._point_wp[arm],
          self._body_wp[arm],
        )
      translational_jacobian = self._jacp_torch[arm][
        :, :, self._joint_dof_ids[arm]
      ].clone()
      rotational_jacobian = self._jacr_torch[arm][
        :, :, self._joint_dof_ids[arm]
      ].clone()
      wrist_frame_constraint = command.contact_pocket_active
      # The terminal target is not a fixed world-space wrist point.  It is the
      # coupled constraint p_wrist + R_wrist * pocket_offset = p_object.  Its
      # differential therefore contains the offset's rotational motion:
      #
      #   (J_position - skew(R * offset) J_rotation) dq = position_error
      #
      # Treating this as J_position alone makes every wrist correction move
      # the target while the shoulder chases it one control step later.  That
      # artificial servo race was the source of rare open-hand force spikes.
      wrist_to_object_offset_w = (
        command.rod.data.root_link_pos_w - target_position_w[:, arm]
      )
      offset_x = wrist_to_object_offset_w[:, 0]
      offset_y = wrist_to_object_offset_w[:, 1]
      offset_z = wrist_to_object_offset_w[:, 2]
      zero = torch.zeros_like(offset_x)
      offset_skew = torch.stack((
        zero, -offset_z, offset_y,
        offset_z, zero, -offset_x,
        -offset_y, offset_x, zero,
      ), dim=-1).reshape(-1, 3, 3)
      coupled_wrist_frame_jacobian = translational_jacobian - torch.bmm(
        offset_skew, rotational_jacobian
      )
      jacobian = torch.where(
        wrist_frame_constraint[:, None, None],
        coupled_wrist_frame_jacobian,
        translational_jacobian,
      )
      position_error = target_position_w[:, arm] - frame_position
      raw_error_norm = torch.linalg.vector_norm(
        position_error, dim=-1, keepdim=True
      ).clamp_min(1e-6)
      position_error = position_error * torch.clamp(
        self.max_cartesian_step_m / raw_error_norm, max=1.0
      )
      task_error = position_error
      task_identity = self._task_identity
      holding_error = raw_error_norm.squeeze(-1)
      bearing_error = torch.zeros_like(holding_error)
      axis_alignment_error = torch.zeros_like(holding_error)
      orientation_step = torch.zeros_like(frame_position)
      if self.wrist_bearing_feedback_gain > 0.0:
        # Keep wrist bearing and rod-axis control alive throughout open-hand
        # insertion.  Disabling it immediately after the alignment latch let
        # the longer pocket translation rotate the mirrored wrist by 0.4--0.8
        # rad before closure, turning a symmetric static grasp into one-sided
        # thumb contact.  The measured pose hold still takes over at the
        # closure latch, so this cannot inject corrections during grasp force.
        pose_alignment_mask = (
          command.contact_alignment_active | command.contact_pocket_active
        ).float()
        rod_local = base.quat_apply_inverse(
          wrist_quat_w[:, arm],
          command.rod.data.root_link_pos_w - frame_position,
        )
        current_bearing = torch.atan2(rod_local[:, 1], rod_local[:, 0])
        desired_lateral = (
          -command.cfg.contact_pocket_lateral_m
          if arm == 0 else command.cfg.contact_pocket_lateral_m
        )
        desired_bearing = math.atan2(
          desired_lateral, command.cfg.contact_pocket_forward_m
        )
        bearing_error = torch.remainder(
          current_bearing - desired_bearing + math.pi,
          2.0 * math.pi,
        ) - math.pi
        bearing_step = (
          bearing_error * self.wrist_bearing_feedback_gain
        ).clamp(
          -self.max_wrist_bearing_step_rad,
          self.max_wrist_bearing_step_rad,
        ) * pose_alignment_mask
        wrist_local_z = torch.zeros_like(frame_position)
        wrist_local_z[:, 2] = 1.0
        wrist_axis_w = base.quat_apply(wrist_quat_w[:, arm], wrist_local_z)
        orientation_task_jacobian = rotational_jacobian.clone()
        # Preserve the accepted retreat/alignment controller: before the
        # pocket latch, wrist orientation belongs to the three wrist joints.
        # Once the coupled pocket constraint is active, the hierarchical solve
        # may distribute pose motion over the complete seven-DoF arm because
        # its primary task explicitly preserves the object-relative pocket.
        orientation_task_jacobian[:, :, :4] = torch.where(
          wrist_frame_constraint[:, None, None],
          orientation_task_jacobian[:, :, :4],
          torch.zeros_like(orientation_task_jacobian[:, :, :4]),
        )
        bearing_weight = self.wrist_bearing_task_weight_m_per_rad
        if self.wrist_axis_alignment_feedback_gain > 0.0:
          rod_local_z = torch.zeros_like(frame_position)
          rod_local_z[:, 2] = 1.0
          rod_axis_w = base.quat_apply(
            command.rod.data.root_link_quat_w, rod_local_z
          )
          raw_axis_dot = torch.sum(wrist_axis_w * rod_axis_w, dim=-1)
          target_axis_w = rod_axis_w * torch.where(
            raw_axis_dot >= 0.0,
            torch.ones_like(raw_axis_dot),
            -torch.ones_like(raw_axis_dot),
          ).unsqueeze(-1)
          axis_cross = torch.linalg.cross(wrist_axis_w, target_axis_w, dim=-1)
          axis_cross_norm = torch.linalg.vector_norm(
            axis_cross, dim=-1
          ).clamp_min(1e-6)
          axis_dot = raw_axis_dot.abs().clamp(0.0, 1.0)
          axis_alignment_error = torch.atan2(axis_cross_norm, axis_dot)
          axis_error_vector = (
            axis_cross / axis_cross_norm.unsqueeze(-1)
            * axis_alignment_error.unsqueeze(-1)
            * self.wrist_axis_alignment_feedback_gain
            * pose_alignment_mask.unsqueeze(-1)
          )
          orientation_step = (
            axis_error_vector + wrist_axis_w * bearing_step.unsqueeze(-1)
          )
          orientation_step_norm = torch.linalg.vector_norm(
            orientation_step, dim=-1, keepdim=True
          ).clamp_min(1e-6)
          orientation_step *= torch.clamp(
            self.max_wrist_bearing_step_rad / orientation_step_norm,
            max=1.0,
          )
          jacobian = torch.cat(
            (jacobian, bearing_weight * orientation_task_jacobian), dim=1
          )
          task_error = torch.cat(
            (position_error, bearing_weight * orientation_step), dim=1
          )
          task_identity = self._pose_task_identity
          holding_error = torch.maximum(
            holding_error,
            bearing_weight * torch.maximum(
              bearing_error.abs(), axis_alignment_error
            ) * pose_alignment_mask,
          )
        else:
          bearing_jacobian = torch.einsum(
            "bti,bt->bi", orientation_task_jacobian, wrist_axis_w
          )
          jacobian = torch.cat(
            (jacobian, bearing_weight * bearing_jacobian.unsqueeze(1)), dim=1
          )
          task_error = torch.cat(
            (position_error, bearing_weight * bearing_step.unsqueeze(1)), dim=1
          )
          task_identity = self._bearing_task_identity
          holding_error = torch.maximum(
            holding_error,
            bearing_weight * bearing_error.abs() * pose_alignment_mask,
          )
      self._wrist_bearing_error[:, arm] = torch.where(
        advance, bearing_error, self._wrist_bearing_error[:, arm]
      )
      self._wrist_axis_alignment_error[:, arm] = torch.where(
        advance,
        axis_alignment_error,
        self._wrist_axis_alignment_error[:, arm],
      )
      joint_position = joint_positions[:, arm]
      active = (command.active_hand == arm) & (
        episode_length >= REACH_ENTRY_SETTLING_CONTROL_STEPS
      )
      # Generic reach keeps the established adaptive-DLS solve.  Terminal
      # wrist-frame contact uses a lexicographic task hierarchy instead: the
      # coupled pocket position is primary, orientation is solved only in its
      # null space, and posture attraction is tertiary.  A weighted stacked
      # solve can legally exchange centimetres of pocket error for radians of
      # axis error, which is unacceptable immediately before physical contact.
      jjt = torch.einsum("bti,bui->btu", jacobian, jacobian)
      singular_values_squared = torch.linalg.eigvalsh(jjt).clamp_min(0.0)
      generic_minimum_singular_value = singular_values_squared[:, 0].sqrt()
      singularity = torch.clamp(
        (self.singularity_threshold - generic_minimum_singular_value)
          / self.singularity_threshold,
        min=0.0,
        max=1.0,
      )
      damping = self.base_damping + self.singularity_damping * singularity.square()
      damped_task_matrix = jjt + task_identity * damping[:, None, None].square()
      damped_inverse_j = torch.linalg.solve(damped_task_matrix, jacobian)
      pseudoinverse = damped_inverse_j.transpose(1, 2)
      generic_task_delta = torch.einsum("bit,bt->bi", pseudoinverse, task_error)
      generic_nullspace = self._joint_identity - torch.bmm(
        pseudoinverse, jacobian
      )

      primary_jacobian = coupled_wrist_frame_jacobian
      primary_jjt = torch.einsum(
        "bti,bui->btu", primary_jacobian, primary_jacobian
      )
      primary_singular_values = torch.linalg.eigvalsh(
        primary_jjt
      ).clamp_min(0.0).sqrt()
      primary_minimum_singular_value = primary_singular_values[:, 0]
      primary_singularity = torch.clamp(
        (self.singularity_threshold - primary_minimum_singular_value)
          / self.singularity_threshold,
        min=0.0,
        max=1.0,
      )
      primary_damping = self.base_damping + (
        self.singularity_damping * primary_singularity.square()
      )
      primary_inverse_j = torch.linalg.solve(
        primary_jjt
          + self._task_identity * primary_damping[:, None, None].square(),
        primary_jacobian,
      )
      primary_pseudoinverse = primary_inverse_j.transpose(1, 2)
      primary_delta = torch.einsum(
        "bit,bt->bi", primary_pseudoinverse, position_error
      )
      primary_nullspace = self._joint_identity - torch.bmm(
        primary_pseudoinverse, primary_jacobian
      )

      command_lead_limit = torch.full_like(
        joint_position, self.max_command_lead_rad
      )
      if self.pocket_max_command_lead_rad is not None:
        command_lead_limit = torch.where(
          command.contact_pocket_active.unsqueeze(-1),
          torch.full_like(joint_position, self.pocket_max_command_lead_rad),
          command_lead_limit,
        )
      secondary_jacobian = torch.bmm(
        rotational_jacobian, primary_nullspace
      )
      secondary_error = orientation_step - torch.einsum(
        "bti,bi->bt", rotational_jacobian, primary_delta
      )
      secondary_jjt = torch.einsum(
        "bti,bui->btu", secondary_jacobian, secondary_jacobian
      )
      secondary_singular_values = torch.linalg.eigvalsh(
        secondary_jjt
      ).clamp_min(0.0).sqrt()
      secondary_minimum_singular_value = secondary_singular_values[:, 0]
      secondary_singularity = torch.clamp(
        (self.singularity_threshold - secondary_minimum_singular_value)
          / self.singularity_threshold,
        min=0.0,
        max=1.0,
      )
      secondary_damping = self.base_damping + (
        self.singularity_damping * secondary_singularity.square()
      )
      secondary_inverse_j = torch.linalg.solve(
        secondary_jjt
          + self._task_identity * secondary_damping[:, None, None].square(),
        secondary_jacobian,
      )
      secondary_pseudoinverse = secondary_inverse_j.transpose(1, 2)
      secondary_free_delta = torch.einsum(
        "bit,bt->bi", secondary_pseudoinverse, secondary_error
      )
      secondary_delta = torch.bmm(
        primary_nullspace, secondary_free_delta.unsqueeze(-1)
      ).squeeze(-1)
      hierarchical_task_delta = primary_delta + secondary_delta
      secondary_nullspace = self._joint_identity - torch.bmm(
        secondary_pseudoinverse, secondary_jacobian
      )
      hierarchical_nullspace = torch.bmm(
        primary_nullspace, secondary_nullspace
      )
      task_delta = torch.where(
        wrist_frame_constraint.unsqueeze(-1),
        hierarchical_task_delta,
        generic_task_delta,
      )
      nullspace_projector = torch.where(
        wrist_frame_constraint[:, None, None],
        hierarchical_nullspace,
        generic_nullspace,
      )
      minimum_singular_value = torch.where(
        wrist_frame_constraint,
        torch.minimum(
          primary_minimum_singular_value,
          secondary_minimum_singular_value,
        ),
        generic_minimum_singular_value,
      )
      posture_delta = self.posture_attractor_gain * (
        self._feasible_posture[:, arm] - joint_position
      )
      projected_posture_delta = torch.bmm(
        nullspace_projector, posture_delta.unsqueeze(-1)
      ).squeeze(-1)
      posture_delta = projected_posture_delta
      unconstrained_joint_correction = (
        self.task_space_feedback_gain * task_delta + posture_delta
      ).clamp(-self.max_joint_correction_rad, self.max_joint_correction_rad)
      lower_margin = joint_position - self._authority_lower[:, arm]
      upper_margin = self._authority_upper[:, arm] - joint_position
      braking_margin = 2.0 * command_lead_limit
      lower_priority_reverses_primary = (
        (primary_delta * hierarchical_task_delta < 0.0)
        & (secondary_delta.abs() > primary_delta.abs())
        & (primary_delta.abs() > 1.0e-4)
      ).any(dim=-1)
      boundary_direction_conflict = (
        (
          (lower_margin <= braking_margin)
          & (unconstrained_joint_correction < 0.0)
          & (primary_delta >= 0.0)
        )
        | (
          (upper_margin <= braking_margin)
          & (unconstrained_joint_correction > 0.0)
          & (primary_delta <= 0.0)
        )
      ).any(dim=-1)
      # The mirrored right chain has a measured infeasible orientation branch:
      # its sixth arm joint is driven toward the Harness lower boundary while
      # the primary pocket solve requires the opposite direction.  The left
      # chain uses the same secondary motion to maintain its viable grasp pose,
      # so early revocation is attached to the affected kinematic chain rather
      # than to an episode seed or object location.
      right_chain_hierarchy_conflict = (
        lower_priority_reverses_primary | boundary_direction_conflict
        if arm == 1
        else torch.zeros_like(lower_priority_reverses_primary)
      )
      lower_priority_conflict = wrist_frame_constraint & (
        right_chain_hierarchy_conflict
      )
      revoke_lower_priority = (
        self._lower_priority_authority_revoked[:, arm] | lower_priority_conflict
      ) & wrist_frame_constraint
      self._lower_priority_authority_revoked[:, arm] = torch.where(
        advance & active,
        revoke_lower_priority,
        self._lower_priority_authority_revoked[:, arm],
      )
      primary_only_correction = (
        self.task_space_feedback_gain * primary_delta
      ).clamp(-self.max_joint_correction_rad, self.max_joint_correction_rad)
      hierarchical_joint_correction = torch.where(
        revoke_lower_priority.unsqueeze(-1),
        primary_only_correction,
        unconstrained_joint_correction,
      )
      joint_correction = torch.where(
        wrist_frame_constraint.unsqueeze(-1),
        hierarchical_joint_correction,
        unconstrained_joint_correction,
      )
      # DLS yields an instantaneous correction around the measured joint state.
      # The persistent target only filters that correction; it must never
      # integrate another correction while the physical joint is lagging.
      instantaneous_target = joint_position + joint_correction
      lead_lower = joint_position - command_lead_limit
      lead_upper = joint_position + command_lead_limit
      lead_limited_target = instantaneous_target.clamp(lead_lower, lead_upper)
      soft_limited_target = lead_limited_target.clamp(
        self._soft_lower[:, arm], self._soft_upper[:, arm]
      )
      authority_limited_target = soft_limited_target.clamp(
        self._authority_lower[:, arm], self._authority_upper[:, arm]
      )
      diagnostic_zero = torch.zeros_like(primary_delta)
      self._primary_delta[:, arm] = torch.where(
        advance[:, None] & active[:, None],
        torch.where(wrist_frame_constraint[:, None], primary_delta, diagnostic_zero),
        self._primary_delta[:, arm],
      )
      self._secondary_delta[:, arm] = torch.where(
        advance[:, None] & active[:, None],
        torch.where(wrist_frame_constraint[:, None], secondary_delta, diagnostic_zero),
        self._secondary_delta[:, arm],
      )
      self._projected_posture_delta[:, arm] = torch.where(
        advance[:, None] & active[:, None],
        torch.where(
          wrist_frame_constraint[:, None], projected_posture_delta, posture_delta
        ),
        self._projected_posture_delta[:, arm],
      )
      self._joint_correction[:, arm] = torch.where(
        advance[:, None] & active[:, None],
        joint_correction,
        self._joint_correction[:, arm],
      )
      self._instantaneous_target[:, arm] = torch.where(
        advance[:, None] & active[:, None],
        instantaneous_target,
        self._instantaneous_target[:, arm],
      )
      self._authority_limited_target[:, arm] = torch.where(
        advance[:, None] & active[:, None],
        authority_limited_target,
        self._authority_limited_target[:, arm],
      )
      solver_target_slew_limit = torch.full_like(
        joint_position, self.max_solver_target_slew_rad
      )
      if self.pocket_max_solver_target_slew_rad is not None:
        solver_target_slew_limit = torch.where(
          command.contact_pocket_active.unsqueeze(-1),
          torch.full_like(
            joint_position, self.pocket_max_solver_target_slew_rad
          ),
          solver_target_slew_limit,
        )
      target_error = authority_limited_target - self._joint_target[:, arm]
      target_slew = torch.maximum(
        torch.minimum(target_error, solver_target_slew_limit),
        -solver_target_slew_limit,
      )
      proposed_target = self._joint_target[:, arm] + target_slew
      # Reapply hard boundaries after slew limiting. This is the anti-windup
      # boundary if the measured joint moved materially since the last command.
      bounded_target = proposed_target.clamp(lead_lower, lead_upper).clamp(
        self._soft_lower[:, arm], self._soft_upper[:, arm]
      ).clamp(self._authority_lower[:, arm], self._authority_upper[:, arm])
      previously_holding = self._holding_target[:, arm]
      raw_error = holding_error
      holding = (
        (previously_holding & (raw_error <= self.hold_release_error_m))
        | (raw_error <= self.hold_enter_error_m)
      )
      update = advance & active
      holding = torch.where(update, holding, previously_holding)
      next_target = torch.where(
        holding[:, None], self._joint_target[:, arm], bounded_target
      )
      target_step = torch.where(
        update[:, None],
        (next_target - self._joint_target[:, arm]).abs(),
        0.0,
      )
      self._solver_target_slew[:, arm] = torch.where(
        advance[:, None] & active[:, None],
        target_slew.abs(),
        self._solver_target_slew[:, arm],
      )
      self._joint_target_step[:, arm] = torch.where(
        advance[:, None], target_step, self._joint_target_step[:, arm]
      )
      self._joint_target[:, arm] = torch.where(
        update[:, None], next_target, self._joint_target[:, arm]
      )
      self._holding_target[:, arm] = holding
      unclamped_normalized = (
        soft_limited_target - self._neutral[:, arm]
      ) / self._action_scale
      active_unclamped = torch.where(active[:, None], unclamped_normalized, 0.0)
      self._unclamped_action[:, arm] = torch.where(
        advance[:, None], active_unclamped, self._unclamped_action[:, arm]
      )
      self._authority_saturation[:, arm] = torch.where(
        advance[:, None],
        active[:, None] & (
          (soft_limited_target < self._authority_lower[:, arm])
          | (soft_limited_target > self._authority_upper[:, arm])
        ),
        self._authority_saturation[:, arm],
      )
      self._soft_limit_saturation[:, arm] = torch.where(
        advance[:, None],
        active[:, None] & (
          (lead_limited_target < self._soft_lower[:, arm])
          | (lead_limited_target > self._soft_upper[:, arm])
        ),
        self._soft_limit_saturation[:, arm],
      )
      self._command_lead_saturation[:, arm] = torch.where(
        advance[:, None],
        active[:, None] & (
          (instantaneous_target < lead_lower) | (instantaneous_target > lead_upper)
        ),
        self._command_lead_saturation[:, arm],
      )
      self._minimum_singular_value[:, arm] = torch.where(
        advance, minimum_singular_value, self._minimum_singular_value[:, arm]
      )
      normalized = (
        (self._joint_target[:, arm] - self._neutral[:, arm]) / self._action_scale
      ).clamp(-1.0, 1.0)
      new_output[:, arm] = torch.where(active[:, None], normalized, 0.0)

    output = torch.where(
      advance[:, None, None], new_output, self._cached_action
    )
    self._cached_action.copy_(output)
    self._last_episode_length.copy_(torch.where(
      advance, episode_length, self._last_episode_length
    ))
    self._initialized |= advance
    self._cache_valid |= advance
    if not torch.isfinite(output).all():
      raise RuntimeError("Task-space reach teacher emitted a non-finite label")
    return output.reshape(self._env.num_envs, UPPER_BODY_ACTION_SIZE)

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    if env_ids is None:
      env_ids = slice(None)
    self._joint_target[env_ids] = self._neutral
    self._feasible_posture[env_ids] = self._neutral
    self._initialized[env_ids] = False
    self._last_episode_length[env_ids] = -1
    self._cache_valid[env_ids] = False
    self._cached_action[env_ids] = 0.0
    self._holding_target[env_ids] = False
    self._unclamped_action[env_ids] = 0.0
    self._authority_saturation[env_ids] = False
    self._soft_limit_saturation[env_ids] = False
    self._command_lead_saturation[env_ids] = False
    self._solver_target_slew[env_ids] = 0.0
    self._joint_target_step[env_ids] = 0.0
    self._minimum_singular_value[env_ids] = 0.0
    self._primary_delta[env_ids] = 0.0
    self._secondary_delta[env_ids] = 0.0
    self._projected_posture_delta[env_ids] = 0.0
    self._joint_correction[env_ids] = 0.0
    self._instantaneous_target[env_ids] = 0.0
    self._authority_limited_target[env_ids] = 0.0
    self._lower_priority_authority_revoked[env_ids] = False
    self._wrist_bearing_error[env_ids] = 0.0
    self._wrist_axis_alignment_error[env_ids] = 0.0
    self._feasible_posture_feature_clamped[env_ids] = False
    self._feasible_posture_action_clamped[env_ids] = False


class WorkyardResidualAction(base.WorkyardAction):
  """29D whole-body skill composed around a locomotion reference."""

  cfg: WorkyardResidualActionCfg

  def __init__(self, cfg: WorkyardResidualActionCfg, env: ManagerBasedRlEnv):
    ActionTerm.__init__(self, cfg, env)
    self._env = env
    body_ids, body_names = self._entity.find_joints(
      base.BODY_JOINT_NAMES, preserve_order=True
    )
    hand_ids, hand_names = self._entity.find_joints(
      base.HAND_JOINT_NAMES, preserve_order=True
    )
    if tuple(body_names) != base.BODY_JOINT_NAMES or tuple(hand_names) != base.HAND_JOINT_NAMES:
      raise ValueError("Residual Workyard joint order does not match its contract")
    self._body_ids = torch.tensor(body_ids, dtype=torch.long, device=self.device)
    self._hand_ids = torch.tensor(hand_ids, dtype=torch.long, device=self.device)
    self._raw_action = torch.zeros((self.num_envs, ACTION_SIZE), device=self.device)
    # Kept only for the base command interface.  Phase one gives it no action
    # slice and no update path, so grasp/contact authority cannot leak in.
    self.coordination = torch.zeros(
      (self.num_envs, HAND_COORDINATION_SIZE), device=self.device
    )
    self._teacher_action = torch.zeros(
      (self.num_envs, TEACHER_ACTION_SIZE), device=self.device
    )
    self._executed_teacher_equivalent_action = torch.zeros_like(
      self._teacher_action
    )
    self._teacher_body_targets = torch.zeros_like(self._teacher_action)
    self._body_targets = torch.zeros_like(self._teacher_action)
    self._hand_targets = torch.zeros(
      (self.num_envs, len(base.HAND_JOINT_NAMES)), device=self.device
    )
    self._balance_residual = torch.zeros(
      (self.num_envs, BALANCE_ACTION_SIZE), device=self.device
    )
    self._upper_body_residual = torch.zeros(
      (self.num_envs, UPPER_BODY_ACTION_SIZE), device=self.device
    )
    self._reach_teacher_action = torch.zeros_like(self._raw_action)
    self.teacher = FrozenLocomotionTeacher(
      Path(cfg.teacher_jit_path),
      Path(cfg.teacher_report_path),
      self.device,
      self.num_envs,
    )
    self.reach_teacher = BatchedTaskSpaceReachTeacher(
      env,
      self._entity,
      self._body_ids[UPPER_BODY_SLICE],
      cfg.upper_body_scale,
    )
    self.dynamic_com = SupportRelativeDynamicCom(env, self._entity)
    self._teacher_body_targets[:] = self.teacher.default_joint_positions
    self._body_targets[:] = self.teacher.default_joint_positions
    self._hand_targets[:] = torch.tensor(
      OPEN_HAND_JOINT_TARGETS, dtype=torch.float32, device=self.device
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

  @property
  def teacher_action(self) -> torch.Tensor:
    return self._teacher_action

  @property
  def teacher_body_targets(self) -> torch.Tensor:
    return self._teacher_body_targets

  @property
  def upper_body_residual(self) -> torch.Tensor:
    return self._upper_body_residual

  @property
  def balance_residual(self) -> torch.Tensor:
    return self._balance_residual

  @property
  def reach_teacher_action(self) -> torch.Tensor:
    return self._reach_teacher_action

  def compute_reach_teacher_action(self) -> torch.Tensor:
    self._reach_teacher_action.zero_()
    self._reach_teacher_action[:, UPPER_BODY_SLICE] = self.reach_teacher.infer()
    return self._reach_teacher_action

  def _teacher_observation(self) -> torch.Tensor:
    command = base._workyard_command(self._env)
    body_pos = self._entity.data.joint_pos[:, self._body_ids]
    body_vel = self._entity.data.joint_vel[:, self._body_ids]
    observation = torch.cat(
      (
        self._entity.data.root_link_lin_vel_b,
        self._entity.data.root_link_ang_vel_b,
        self._entity.data.projected_gravity_b,
        body_pos - self.teacher.default_joint_positions,
        body_vel,
        self._executed_teacher_equivalent_action,
        command.desired_base_twist,
      ),
      dim=-1,
    )
    if observation.shape != (self.num_envs, TEACHER_OBSERVATION_SIZE):
      raise RuntimeError(f"Locomotion teacher observation drifted: {observation.shape}")
    return observation

  def process_actions(self, actions: torch.Tensor) -> None:
    if actions.shape != (self.num_envs, ACTION_SIZE):
      raise ValueError(f"Expected residual Workyard action [B, 29], got {actions.shape}")
    self._reach_teacher_action[:] = self.compute_reach_teacher_action()
    self._raw_action[:] = actions.clamp(-1.0, 1.0)
    self._teacher_action[:] = self.teacher.infer(self._teacher_observation())
    self._teacher_body_targets[:] = (
      self.teacher.default_joint_positions
      + self._teacher_action * self.teacher.action_scale
    )
    self._balance_residual[:] = (
      self._raw_action[:, :BALANCE_ACTION_SIZE] * self.cfg.balance_scale
    )
    self._upper_body_residual[:] = (
      self._raw_action[:, UPPER_BODY_SLICE] * self.cfg.upper_body_scale
    )
    self._body_targets[:] = self.teacher.default_joint_positions
    self._body_targets[:, :BALANCE_ACTION_SIZE] = (
      self._teacher_body_targets[:, :BALANCE_ACTION_SIZE]
      + self._balance_residual
    )
    self._body_targets[:, UPPER_BODY_SLICE] += self._upper_body_residual
    # The source policy's recurrent state is represented by its previous-action
    # observation. Report the command that the hybrid plant actually executed,
    # not the teacher's discarded arm proposal. The equivalent source action is
    # clipped to the source policy's trained action domain; actual arm position
    # and velocity remain available in their dedicated observation terms.
    self._executed_teacher_equivalent_action[:] = (
      (self._body_targets - self.teacher.default_joint_positions)
      / self.teacher.action_scale
    ).clamp(-1.0, 1.0)

  def apply_actions(self) -> None:
    self._entity.set_joint_position_target(self._body_targets, joint_ids=self._body_ids)
    self._entity.set_joint_position_target(self._hand_targets, joint_ids=self._hand_ids)

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    if env_ids is None:
      env_ids = slice(None)
    self._raw_action[env_ids] = 0.0
    self.coordination[env_ids] = 0.0
    self._teacher_action[env_ids] = 0.0
    self._executed_teacher_equivalent_action[env_ids] = 0.0
    self._teacher_body_targets[env_ids] = self.teacher.default_joint_positions
    self._body_targets[env_ids] = self.teacher.default_joint_positions
    self._balance_residual[env_ids] = 0.0
    self._upper_body_residual[env_ids] = 0.0
    self._reach_teacher_action[env_ids] = 0.0
    self.reach_teacher.reset(env_ids)
    self._hand_targets[env_ids] = torch.tensor(
      OPEN_HAND_JOINT_TARGETS, dtype=torch.float32, device=self.device
    )


def _residual_action(env: ManagerBasedRlEnv) -> WorkyardResidualAction:
  action = env.action_manager.get_term("workyard")
  if not isinstance(action, WorkyardResidualAction):
    raise TypeError("Workyard residual action term is unavailable")
  return action


class WorkyardResidualObservation(base.WorkyardObservation):
  def __call__(self, env: ManagerBasedRlEnv) -> torch.Tensor:
    action = _residual_action(env)
    command = base._workyard_command(env)
    root_pos = self.robot.data.root_link_pos_w
    root_quat = self.robot.data.root_link_quat_w
    body_pos = self.robot.data.joint_pos[:, self.body_ids]
    body_vel = self.robot.data.joint_vel[:, self.body_ids]
    hand_pos = self.robot.data.joint_pos[:, self.hand_ids]
    hand_vel = self.robot.data.joint_vel[:, self.hand_ids]

    ee_pose = self.robot.data.body_link_pose_w[:, self.ee_ids]
    ee_pos_p = base.quat_apply_inverse(
      root_quat.unsqueeze(1).expand(-1, 4, -1), ee_pose[..., :3] - root_pos.unsqueeze(1)
    )
    ee_quat_p = base.quat_mul(
      base.quat_conjugate(root_quat).unsqueeze(1).expand(-1, 4, -1),
      ee_pose[..., 3:7],
    )
    ee_pose_p = torch.cat((ee_pos_p, ee_quat_p), dim=-1).reshape(env.num_envs, 28)

    foot_found, foot_force, foot_slip = base._foot_contact_summary(env)
    support = torch.cat((foot_found, foot_force, foot_slip), dim=-1)
    hand_found, hand_force, hand_surfaces, _ = base._hand_contact_summary(env)
    hand_contact = torch.cat(
      (hand_found.float(), hand_force, hand_surfaces), dim=-1
    )

    rod_pose = self.rod.data.root_link_pose_w
    rod_pos_p = base.quat_apply_inverse(root_quat, rod_pose[:, :3] - root_pos)
    rod_quat_p = base.quat_mul(base.quat_conjugate(root_quat), rod_pose[:, 3:7])
    rod_vel = self.rod.data.root_link_vel_w
    rod_twist_p = torch.cat(
      (
        base.quat_apply_inverse(root_quat, rod_vel[:, :3]),
        base.quat_apply_inverse(root_quat, rod_vel[:, 3:6]),
      ),
      dim=-1,
    )
    zone_delta = base.quat_apply_inverse(
      root_quat, command.target_position - rod_pose[:, :3]
    )
    zone_distance = torch.linalg.vector_norm(zone_delta[:, :2], dim=-1, keepdim=True)
    zone_inside = base._object_inside_zone(env).float().unsqueeze(-1)
    dynamic_com = action.dynamic_com.compute()

    observation = torch.cat(
      (
        body_pos - self.body_default,
        body_vel,
        action.raw_action,
        ((hand_pos - self.hand_min) / self.hand_span).clamp(0.0, 1.0),
        hand_vel,
        dynamic_com.observation,
        action.teacher_action,
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
      raise RuntimeError(f"Residual Workyard observation drifted: {observation.shape}")
    return observation


def teacher_lower_body_tracking(env: ManagerBasedRlEnv) -> torch.Tensor:
  action = _residual_action(env)
  position = action._entity.data.joint_pos[:, action._body_ids]
  error = position[:, LOWER_BODY_SLICE] - action.teacher_body_targets[:, LOWER_BODY_SLICE]
  return torch.exp(-error.square().mean(dim=-1) / (0.15 ** 2))


def teacher_waist_tracking(env: ManagerBasedRlEnv) -> torch.Tensor:
  action = _residual_action(env)
  position = action._entity.data.joint_pos[:, action._body_ids]
  error = position[:, WAIST_SLICE] - action.teacher_body_targets[:, WAIST_SLICE]
  return torch.exp(-error.square().mean(dim=-1) / (0.12 ** 2))


def active_wrist_position_delta(env: ManagerBasedRlEnv) -> torch.Tensor:
  command = base._workyard_command(env)
  root_pos = command.robot.data.root_link_pos_w
  root_quat = command.robot.data.root_link_quat_w
  wrist_pose = command.robot.data.body_link_pose_w[:, command._wrist_body_ids]
  wrist_position_p = base.quat_apply_inverse(
    root_quat.unsqueeze(1).expand(-1, 2, -1),
    wrist_pose[..., :3] - root_pos.unsqueeze(1),
  )
  targets = command.wrist_targets_pelvis.reshape(env.num_envs, 2, 7)[..., :3]
  rows = torch.arange(env.num_envs, device=env.device)
  active = command.active_hand
  return wrist_position_p[rows, active] - targets[rows, active]


def active_wrist_position_error(env: ManagerBasedRlEnv) -> torch.Tensor:
  return torch.linalg.vector_norm(active_wrist_position_delta(env), dim=-1)


def wrist_position_tracking(env: ManagerBasedRlEnv) -> torch.Tensor:
  command = base._workyard_command(env)
  rows = torch.arange(env.num_envs, device=env.device)
  active = command.active_hand
  error = active_wrist_position_error(env)
  tolerance = command.wrist_tolerances[rows, active].clamp_min(0.025)
  # The precision kernel alone is effectively zero at the initial reach
  # distance, so PPO cannot distinguish actions that move toward the target.
  # A broad reach kernel supplies dense far-field credit while the original
  # tolerance-conditioned kernel retains the exact near-field objective.
  broad_reach = torch.exp(-error / WRIST_BROAD_REACH_SCALE_M)
  precision = torch.exp(-error.square() / tolerance.square())
  return 0.75 * broad_reach + 0.25 * precision


def wrist_orientation_tracking(env: ManagerBasedRlEnv) -> torch.Tensor:
  command = base._workyard_command(env)
  root_quat = command.robot.data.root_link_quat_w
  wrist_quat = command.robot.data.body_link_quat_w[:, command._wrist_body_ids]
  wrist_quat_p = base.quat_mul(
    base.quat_conjugate(root_quat).unsqueeze(1).expand(-1, 2, -1), wrist_quat
  )
  targets = command.wrist_targets_pelvis.reshape(env.num_envs, 2, 7)[..., 3:7]
  rows = torch.arange(env.num_envs, device=env.device)
  active = command.active_hand
  return torch.abs(
    torch.sum(wrist_quat_p[rows, active] * targets[rows, active], dim=-1)
  ).square()


def upper_body_residual_l2(env: ManagerBasedRlEnv) -> torch.Tensor:
  return _residual_action(env).upper_body_residual.square().mean(dim=-1)


def balance_residual_l2(env: ManagerBasedRlEnv) -> torch.Tensor:
  return _residual_action(env).balance_residual.square().mean(dim=-1)


class WristDistanceProgress:
  """Signed one-step reach progress with reset-safe state."""

  def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
    del cfg
    self._previous_error = torch.zeros(env.num_envs, device=env.device)
    self._previous_episode_length = torch.full(
      (env.num_envs,), -1, dtype=torch.long, device=env.device
    )
    self._valid = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

  def __call__(self, env: ManagerBasedRlEnv) -> torch.Tensor:
    current_error = active_wrist_position_error(env)
    episode_length = env.episode_length_buf
    continuing = self._valid & (episode_length > self._previous_episode_length)
    progress = torch.where(
      continuing, self._previous_error - current_error, torch.zeros_like(current_error)
    ).clamp(-0.05, 0.05)
    self._previous_error.copy_(current_error)
    self._previous_episode_length.copy_(episode_length)
    self._valid[:] = True
    return progress


def reach_teacher_action_tracking(env: ManagerBasedRlEnv) -> torch.Tensor:
  action = _residual_action(env)
  error = (
    action.raw_action[:, UPPER_BODY_SLICE]
    - action.reach_teacher_action[:, UPPER_BODY_SLICE]
  ).square().mean(dim=-1)
  return torch.exp(-error / (0.35 ** 2))


def dynamic_com_support(env: ManagerBasedRlEnv) -> torch.Tensor:
  state = _residual_action(env).dynamic_com.compute()
  margin_score = torch.sigmoid(state.support_margin / 0.025)
  capture_norm = torch.linalg.vector_norm(state.capture_point_pelvis, dim=-1)
  centered_score = torch.exp(-capture_norm / 0.18)
  score = 0.75 * margin_score + 0.25 * centered_score
  return torch.where(state.no_foot_contact, torch.zeros_like(score), score)


def make_workyard_residual_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  cfg = base.make_workyard_env_cfg(play=play)
  cfg.scene.sensors = (*cfg.scene.sensors, *_reach_contact_diagnostic_sensors())
  robot_cfg = cfg.scene.entities["robot"]
  robot_cfg.spec_fn = _load_hybrid_actuated_g1_spec
  # A reach Skill begins at one complete reachable base placement selected by
  # the Harness. WorkyardResidualCommand samples the object-relative root
  # translation, active hand, and yaw as one correlated reset tuple.
  robot_cfg.init_state.pos = RESIDUAL_ENTRY_ROOT_POSITION
  cfg.scene.num_envs = 1 if play else 2048
  cfg.episode_length_s = 12.0
  cfg.actions = {
    "workyard": WorkyardResidualActionCfg(
      entity_name="robot",
      balance_scale=0.12,
      upper_body_scale=0.7,
    )
  }
  cfg.commands = {
    "workyard": WorkyardResidualCommandCfg(
      resampling_time_range=(20.0, 20.0),
      debug_vis=False,
      object_position_jitter_m=0.05,
      target_position_jitter_m=0.0,
    )
  }
  cfg.observations = {
    "actor": ObservationGroupCfg(
      terms={
        "workyard": ObservationTermCfg(
          func=WorkyardResidualObservation,
          delay_min_lag=0,
          delay_max_lag=0 if play else 1,
          clip=(-20.0, 20.0),
        )
      },
      concatenate_terms=True,
      enable_corruption=False,
    ),
    "critic": ObservationGroupCfg(
      terms={"workyard": ObservationTermCfg(func=WorkyardResidualObservation)},
      concatenate_terms=True,
      enable_corruption=False,
    ),
  }
  cfg.rewards = {
    "upright_support": RewardTermCfg(func=base.upright_support, weight=1.0),
    "teacher_lower_body_tracking": RewardTermCfg(
      func=teacher_lower_body_tracking, weight=2.0
    ),
    "teacher_waist_tracking": RewardTermCfg(func=teacher_waist_tracking, weight=0.5),
    "wrist_position_tracking": RewardTermCfg(func=wrist_position_tracking, weight=4.0),
    "wrist_distance_progress": RewardTermCfg(func=WristDistanceProgress, weight=25.0),
    "wrist_orientation_tracking": RewardTermCfg(
      func=wrist_orientation_tracking, weight=0.25
    ),
    "reach_teacher_action_tracking": RewardTermCfg(
      func=reach_teacher_action_tracking, weight=0.30
    ),
    "dynamic_com_support": RewardTermCfg(func=dynamic_com_support, weight=1.0),
    "balance_residual_l2": RewardTermCfg(func=balance_residual_l2, weight=-0.01),
    "upper_body_residual_l2": RewardTermCfg(func=upper_body_residual_l2, weight=-0.02),
    "action_rate": RewardTermCfg(func=mdp.action_rate_l2, weight=-0.02),
    "joint_limit_proximity": RewardTermCfg(
      func=mdp.joint_pos_limits,
      weight=-0.5,
      params={"asset_cfg": SceneEntityCfg("robot", joint_names=(".*",))},
    ),
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
    "time_out": TerminationTermCfg(func=mdp.time_out, time_out=True),
    "fall": TerminationTermCfg(func=base.fell_over),
    "non_foot_ground": TerminationTermCfg(func=base.illegal_ground_contact),
    "wrist_target_success": TerminationTermCfg(func=base.stage_completed),
  }
  cfg.curriculum = {}
  if not play:
    mass = cfg.events.get("object_mass")
    if mass is not None:
      mass.params["alpha_range"] = (0.5 * math.log(0.9), 0.5 * math.log(1.1))
    friction = cfg.events.get("object_friction")
    if friction is not None:
      friction.params["ranges"] = (0.9, 1.1)
    strength = cfg.events.get("actuator_strength")
    if strength is not None:
      strength.params["scale_range"] = (0.95, 1.05)
  return cfg


def workyard_residual_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
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
      class_name="hear_retention_ppo:HearRetentionPPO",
      value_loss_coef=1.0,
      use_clipped_value_loss=True,
      clip_param=0.2,
      entropy_coef=0.0,
      num_learning_epochs=5,
      num_mini_batches=8,
      learning_rate=3.0e-4,
      schedule="adaptive",
      gamma=0.99,
      lam=0.95,
      desired_kl=0.01,
      max_grad_norm=1.0,
    ),
    experiment_name="hear_workyard_residual_g1",
    save_interval=100,
    num_steps_per_env=32,
    max_iterations=5_000,
  )


def register_workyard_residual_task() -> None:
  if TASK_ID in list_tasks():
    return
  register_mjlab_task(
    task_id=TASK_ID,
    env_cfg=make_workyard_residual_env_cfg(play=False),
    play_env_cfg=make_workyard_residual_env_cfg(play=True),
    rl_cfg=workyard_residual_ppo_runner_cfg(),
  )


register_workyard_residual_task()
