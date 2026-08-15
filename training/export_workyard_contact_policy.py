"""Export an accepted HEAR contact checkpoint for runtime inference.

Only the checkpoint selected by a successful formal 500-seed gate is
exportable.  The exported actor is the deterministic mean of the bounded Beta
distribution used during training, including the learned observation
normalizer.  TorchScript remains the frozen training/runtime reference while
ONNX is the artifact consumed by the TypeScript MuJoCo controller.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from xml.etree import ElementTree

import numpy as np
import onnx
import torch
from onnx.reference import ReferenceEvaluator
from torch import nn
from torch.nn import functional as torch_f


PROTOCOL = "hear-frozen-contact-policy-export-v2"
SOURCE_REPORT_PROTOCOL = "hear-workyard-contact-run-v2"
OBSERVATION_PROTOCOL = "hear-workyard-contact-observation-v2"
ACTION_PROTOCOL = "hear-active-hand-synergy-action-v1"
OBSERVATION_SIZE = 262
ACTION_SIZE = 8
NORMALIZER_EPS = 1.0e-2
COORDINATION_STEP = 0.0075
MAXIMUM_CLOSING_JOINT_LEAD_RAD = 0.25
OPPOSING_SUPPORT_COORDINATION = 0.4
FORCE_RELEASE_THRESHOLD_N = 6.0
EMERGENCY_FORCE_RELEASE_THRESHOLD_N = 12.0
HAND_POSITION_KP = 2.5
HAND_VELOCITY_DAMPING = 0.3
HAND_CONTACT_COLLISION_COUNT = 14
HAND_CONTACT_PRIORITY = 2
HAND_CONTACT_SOLREF_TIME_CONSTANT_S = 0.04
HAND_CONTACT_SOLREF_DAMPING_RATIO = 1.0
HAND_JOINT_NAMES = (
  "left_hand_thumb_0_joint", "left_hand_thumb_1_joint",
  "left_hand_thumb_2_joint", "left_hand_middle_0_joint",
  "left_hand_middle_1_joint", "left_hand_index_0_joint",
  "left_hand_index_1_joint", "right_hand_thumb_0_joint",
  "right_hand_thumb_1_joint", "right_hand_thumb_2_joint",
  "right_hand_middle_0_joint", "right_hand_middle_1_joint",
  "right_hand_index_0_joint", "right_hand_index_1_joint",
)
HAND_CONTACT_MESH_NAMES = frozenset(
  joint_name.removesuffix("_joint") + "_link"
  for joint_name in HAND_JOINT_NAMES
  if not joint_name.endswith("_thumb_1_joint")
)


def sha256(path: Path) -> str:
  digest = hashlib.sha256()
  with path.open("rb") as source:
    for chunk in iter(lambda: source.read(1024 * 1024), b""):
      digest.update(chunk)
  return digest.hexdigest()


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser()
  parser.add_argument("--checkpoint", type=Path, required=True)
  parser.add_argument("--training-report", type=Path, required=True)
  parser.add_argument("--training-contract", type=Path, required=True)
  parser.add_argument("--training-environment", type=Path, required=True)
  parser.add_argument("--plant-xml", type=Path, required=True)
  parser.add_argument("--torchscript-output", type=Path, required=True)
  parser.add_argument("--onnx-output", type=Path, required=True)
  parser.add_argument("--report", type=Path, required=True)
  return parser.parse_args()


def validate_plant(path: Path) -> dict[str, object]:
  if not path.is_file():
    raise FileNotFoundError(f"Contact deployment plant is missing: {path}")
  root = ElementTree.parse(path).getroot()
  hand_default = root.find(".//default[@class='hand_pd']/position")
  hand_contact_default = root.find(
    ".//default[@class='hand_collision']/geom"
  )
  actuators = root.findall(".//actuator/position[@class='hand_pd']")
  hand_contact_geoms = root.findall(".//geom[@class='hand_collision']")
  joints = {actuator.get("joint") for actuator in actuators}
  hand_contact_meshes = {
    geom.get("mesh") for geom in hand_contact_geoms if geom.get("mesh")
  }
  hand_contact_box_count = sum(
    geom.get("type") == "box" for geom in hand_contact_geoms
  )
  solref = tuple(
    float(value)
    for value in (hand_contact_default.get("solref", "").split()
      if hand_contact_default is not None else ())
  )
  if (
    hand_default is None
    or float(hand_default.get("kp", "nan")) != HAND_POSITION_KP
    or float(hand_default.get("kv", "nan")) != HAND_VELOCITY_DAMPING
    or joints != set(HAND_JOINT_NAMES)
    or len(actuators) != len(HAND_JOINT_NAMES)
    or hand_contact_default is None
    or int(hand_contact_default.get("priority", "-1")) != HAND_CONTACT_PRIORITY
    or solref != (
      HAND_CONTACT_SOLREF_TIME_CONSTANT_S,
      HAND_CONTACT_SOLREF_DAMPING_RATIO,
    )
    or len(hand_contact_geoms) != HAND_CONTACT_COLLISION_COUNT
    or hand_contact_meshes != HAND_CONTACT_MESH_NAMES
    or hand_contact_box_count != 2
  ):
    raise ValueError(
      "Contact deployment plant drifted from trained hand actuation or contact skin"
    )
  return {
    "file": path.name,
    "bytes": path.stat().st_size,
    "sha256": sha256(path),
    "hand_contact_collision_count": len(hand_contact_geoms),
    "hand_contact_priority": HAND_CONTACT_PRIORITY,
    "hand_contact_solref_time_constant_s": HAND_CONTACT_SOLREF_TIME_CONSTANT_S,
    "hand_contact_solref_damping_ratio": HAND_CONTACT_SOLREF_DAMPING_RATIO,
  }


class FrozenContactActor(nn.Module):
  """RSL-RL Beta actor reduced to its deterministic bounded mean."""

  __constants__ = ("normalizer_eps", "action_size")

  def __init__(self, state: dict[str, torch.Tensor]):
    super().__init__()
    self.normalizer_eps = NORMALIZER_EPS
    self.action_size = ACTION_SIZE
    self.register_buffer("observation_mean", state["obs_normalizer._mean"].clone())
    self.register_buffer("observation_std", state["obs_normalizer._std"].clone())
    self.linear0 = nn.Linear(OBSERVATION_SIZE, 512)
    self.linear2 = nn.Linear(512, 512)
    self.linear4 = nn.Linear(512, 256)
    self.linear6 = nn.Linear(256, 2 * ACTION_SIZE)
    for layer, prefix in (
      (self.linear0, "mlp.0"),
      (self.linear2, "mlp.2"),
      (self.linear4, "mlp.4"),
      (self.linear6, "mlp.6"),
    ):
      with torch.no_grad():
        layer.weight.copy_(state[f"{prefix}.weight"])
        layer.bias.copy_(state[f"{prefix}.bias"])
    self.requires_grad_(False)
    self.eval()

  def forward(self, observation: torch.Tensor) -> torch.Tensor:
    normalized = (
      observation - self.observation_mean
    ) / (self.observation_std + self.normalizer_eps)
    hidden = torch_f.elu(self.linear0(normalized))
    hidden = torch_f.elu(self.linear2(hidden))
    hidden = torch_f.elu(self.linear4(hidden))
    parameters = self.linear6(hidden).reshape(-1, 2, self.action_size)
    alpha = torch_f.softplus(parameters[:, 0]) + 1.0
    beta = torch_f.softplus(parameters[:, 1]) + 1.0
    return 2.0 * alpha / (alpha + beta) - 1.0


def validate_source(
  checkpoint_path: Path,
  report_path: Path,
  contract_path: Path,
  environment_path: Path,
) -> tuple[dict[str, object], dict[str, torch.Tensor]]:
  if not checkpoint_path.is_file():
    raise FileNotFoundError(f"Selected contact checkpoint is missing: {checkpoint_path}")
  if not report_path.is_file():
    raise FileNotFoundError(f"Contact qualification report is missing: {report_path}")
  if not contract_path.is_file():
    raise FileNotFoundError(f"Contact training contract is missing: {contract_path}")
  if not environment_path.is_file():
    raise FileNotFoundError(
      f"Contact training environment is missing: {environment_path}"
    )
  report = json.loads(report_path.read_text(encoding="utf-8"))
  bundle = report.get("bundle")
  source_contract = report.get("contract")
  acceptance = report.get("acceptance")
  training = report.get("training")
  selection = training.get("checkpoint_selection") if isinstance(training, dict) else None
  frozen_reach = training.get("frozen_reach") if isinstance(training, dict) else None
  selected = selection.get("selected_checkpoint") if isinstance(selection, dict) else None
  final_gate = acceptance.get("final_gate") if isinstance(acceptance, dict) else None
  if (
    report.get("protocol") != SOURCE_REPORT_PROTOCOL
    or report.get("mode") != "train"
    or report.get("ready") is not True
    or not isinstance(acceptance, dict)
    or acceptance.get("verified_grasp_policy_accepted") is not True
    or not isinstance(final_gate, dict)
    or final_gate.get("protocol")
      != "hear-workyard-contact-independent-500-gate-v1"
    or final_gate.get("passed") is not True
    or not isinstance(final_gate.get("checks"), dict)
    or not all(value is True for value in final_gate["checks"].values())
    or not isinstance(selection, dict)
    or selection.get("selected_source") not in ("dagger", "ppo")
    or not isinstance(selected, dict)
    or selected.get("file") != checkpoint_path.name
    or selected.get("bytes") != checkpoint_path.stat().st_size
    or selected.get("sha256") != sha256(checkpoint_path)
    or not isinstance(bundle, dict)
    or bundle.get("contract_sha256") != sha256(contract_path)
    or bundle.get("environment_sha256") != sha256(environment_path)
    or not isinstance(frozen_reach, dict)
    or frozen_reach.get("protocol")
      != "hear-frozen-qualified-whole-body-reach-runtime-v2"
    or frozen_reach.get("gradient_parameter_count") != 0
    or frozen_reach.get("execution_authority")
      != "frozen_29d_whole_body_reach"
    or frozen_reach.get("jit_sha256") != bundle.get("reach_jit_sha256")
    or not isinstance(frozen_reach.get("source_checkpoint_sha256"), str)
    or not isinstance(frozen_reach.get("report_sha256"), str)
  ):
    raise ValueError(
      "Contact checkpoint is not backed by an accepted formal 500-seed gate"
    )

  # Reports produced by the bundle revision immediately before these summary
  # fields were introduced remain exportable through the stronger file-hash
  # binding above.  New reports must agree with the same qualified settings.
  if isinstance(source_contract, dict):
    reported_lead = source_contract.get("hand_max_closing_joint_lead_rad")
    reported_support = source_contract.get("opposing_support_coordination")
    if reported_lead is not None and reported_lead != MAXIMUM_CLOSING_JOINT_LEAD_RAD:
      raise ValueError("Contact training report declares an incompatible hand lead guard")
    if reported_support is not None and reported_support != OPPOSING_SUPPORT_COORDINATION:
      raise ValueError(
        "Contact training report declares incompatible opposing-finger support"
      )

  checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
  state = checkpoint.get("actor_state_dict")
  if not isinstance(state, dict):
    raise ValueError("Contact checkpoint has no actor state")
  required_shapes = {
    "obs_normalizer._mean": (1, OBSERVATION_SIZE),
    "obs_normalizer._std": (1, OBSERVATION_SIZE),
    "mlp.0.weight": (512, OBSERVATION_SIZE),
    "mlp.0.bias": (512,),
    "mlp.2.weight": (512, 512),
    "mlp.2.bias": (512,),
    "mlp.4.weight": (256, 512),
    "mlp.4.bias": (256,),
    "mlp.6.weight": (2 * ACTION_SIZE, 256),
    "mlp.6.bias": (2 * ACTION_SIZE,),
  }
  for name, shape in required_shapes.items():
    value = state.get(name)
    if not isinstance(value, torch.Tensor) or tuple(value.shape) != shape:
      raise ValueError(f"Contact actor tensor {name} has the wrong shape")
    if not torch.isfinite(value).all():
      raise ValueError(f"Contact actor tensor {name} is non-finite")
  if (state["obs_normalizer._std"] < 0.0).any():
    raise ValueError("Contact actor normalizer has a negative standard deviation")
  return report, state


def require_new_outputs(paths: tuple[Path, ...]) -> None:
  for path in paths:
    if path.exists():
      raise FileExistsError(f"Contact export output already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)


def main() -> None:
  args = parse_args()
  checkpoint_path = args.checkpoint.resolve()
  source_report_path = args.training_report.resolve()
  training_contract_path = args.training_contract.resolve()
  training_environment_path = args.training_environment.resolve()
  plant_xml_path = args.plant_xml.resolve()
  torchscript_path = args.torchscript_output.resolve()
  onnx_path = args.onnx_output.resolve()
  report_path = args.report.resolve()
  require_new_outputs((torchscript_path, onnx_path, report_path))

  source_report, state = validate_source(
    checkpoint_path,
    source_report_path,
    training_contract_path,
    training_environment_path,
  )
  plant_xml = validate_plant(plant_xml_path)
  actor = FrozenContactActor(state)
  scripted = torch.jit.script(actor)
  torch.manual_seed(42)
  normalized_probes = torch.cat((
    torch.zeros((1, OBSERVATION_SIZE)),
    torch.ones((1, OBSERVATION_SIZE)),
    torch.randn((14, OBSERVATION_SIZE)),
  ))
  # Keep export probes inside the actor's fitted observation distribution.
  # The ONNX reference Softplus implementation is intentionally simple and
  # can overflow for raw synthetic values on near-constant dimensions even
  # though Torch and production ONNX Runtime return the same finite policy.
  probes = actor.observation_mean + normalized_probes * (
    actor.observation_std + NORMALIZER_EPS
  )
  with torch.inference_mode():
    eager_output = actor(probes)
    scripted_output = scripted(probes)
  maximum_torchscript_error = float(
    (eager_output - scripted_output).abs().max().item()
  )
  if (
    tuple(scripted_output.shape) != (16, ACTION_SIZE)
    or not torch.isfinite(scripted_output).all()
    or float(scripted_output.abs().max().item()) > 1.0 + 1.0e-6
    or maximum_torchscript_error > 1.0e-6
  ):
    raise RuntimeError("Frozen contact TorchScript validation failed")
  torch.jit.save(scripted, str(torchscript_path))

  torch.onnx.export(
    actor,
    (probes[:1],),
    str(onnx_path),
    input_names=["observation"],
    output_names=["synergy_action"],
    dynamic_axes={
      "observation": {0: "batch"},
      "synergy_action": {0: "batch"},
    },
    opset_version=17,
    dynamo=False,
  )
  model = onnx.load(str(onnx_path))
  onnx.checker.check_model(model, full_check=True)
  reference = ReferenceEvaluator(model)
  onnx_output = reference.run(
    ["synergy_action"],
    {"observation": probes.numpy()},
  )[0]
  eager_numpy = eager_output.numpy()
  maximum_onnx_error = float(np.max(np.abs(onnx_output - eager_numpy)))
  if (
    onnx_output.shape != (16, ACTION_SIZE)
    or not np.isfinite(onnx_output).all()
    or float(np.max(np.abs(onnx_output))) > 1.0 + 1.0e-6
    or maximum_onnx_error > 1.0e-5
  ):
    raise RuntimeError("Frozen contact ONNX validation failed")

  source_evaluation = source_report["evaluation"]
  source_selection = source_report["training"]["checkpoint_selection"]
  frozen_reach = source_report["training"]["frozen_reach"]
  payload = {
    "protocol": PROTOCOL,
    "created_at": datetime.now(timezone.utc).isoformat(),
    "source": {
      "training_report_file": source_report_path.name,
      "training_report_sha256": sha256(source_report_path),
      "training_contract_sha256": sha256(training_contract_path),
      "training_environment_sha256": sha256(training_environment_path),
      "checkpoint_file": checkpoint_path.name,
      "checkpoint_bytes": checkpoint_path.stat().st_size,
      "checkpoint_sha256": sha256(checkpoint_path),
      "selected_source": source_selection["selected_source"],
      "formal_gate_protocol": source_report["acceptance"]["final_gate"]["protocol"],
      "formal_gate_passed": True,
      "held_out_episode_count": source_evaluation["episode_count"],
      "held_out_success_rate": source_evaluation["success_rate"],
      "maximum_active_hand_force_n": source_evaluation[
        "maximum_active_hand_force_n"
      ],
      "frozen_reach": {
        "protocol": "hear-frozen-contact-reach-binding-v1",
        "runtime_protocol": frozen_reach["protocol"],
        "source_checkpoint_sha256": frozen_reach[
          "source_checkpoint_sha256"
        ],
        "jit_sha256": frozen_reach["jit_sha256"],
        "report_sha256": frozen_reach["report_sha256"],
      },
    },
    "plant": {
      "protocol": "hear-workyard-contact-deployment-plant-v1",
      "g1_xml": plant_xml,
      "hand_joint_count": len(HAND_JOINT_NAMES),
      "hand_position_kp": HAND_POSITION_KP,
      "hand_velocity_damping": HAND_VELOCITY_DAMPING,
      "workyard_rod": {
        "shape": "cylinder",
        "radius_m": 0.03,
        "half_height_m": 0.11,
        "mass_kg": 0.35,
        "friction": [0.8, 0.012, 0.002],
      },
    },
    "policy": {
      "torchscript": {
        "file": torchscript_path.name,
        "bytes": torchscript_path.stat().st_size,
        "sha256": sha256(torchscript_path),
        "runtime": "torchscript",
      },
      "onnx": {
        "file": onnx_path.name,
        "bytes": onnx_path.stat().st_size,
        "sha256": sha256(onnx_path),
        "runtime": "onnxruntime",
        "opset": 17,
        "input": "observation",
        "output": "synergy_action",
      },
      "input_protocol": OBSERVATION_PROTOCOL,
      "input_size": OBSERVATION_SIZE,
      "output_protocol": ACTION_PROTOCOL,
      "output_size": ACTION_SIZE,
      "distribution": "beta_bounded_minus_one_one",
      "deterministic_statistic": "mean",
      "batch_dynamic": True,
      "normalizer_epsilon": NORMALIZER_EPS,
      "parameter_count": sum(parameter.numel() for parameter in actor.parameters()),
      "gradient_parameter_count": sum(
        parameter.numel() for parameter in actor.parameters()
        if parameter.requires_grad
      ),
    },
    "harness": {
      "terminal_pose_hold": "closure_latch_measured_active_arm",
      "coordination_step": COORDINATION_STEP,
      "maximum_closing_joint_lead_rad": MAXIMUM_CLOSING_JOINT_LEAD_RAD,
      "force_release_threshold_n": FORCE_RELEASE_THRESHOLD_N,
      "emergency_force_release_threshold_n": EMERGENCY_FORCE_RELEASE_THRESHOLD_N,
    },
    "validation": {
      "probe_batch_size": len(probes),
      "maximum_torchscript_error": maximum_torchscript_error,
      "maximum_onnx_error": maximum_onnx_error,
      "maximum_absolute_action": float(np.max(np.abs(onnx_output))),
      "finite": True,
      "onnx_checker_full": True,
    },
  }
  report_path.write_text(
    json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
    encoding="utf-8",
  )
  print(json.dumps(payload, ensure_ascii=False))


if __name__ == "__main__":
  main()
