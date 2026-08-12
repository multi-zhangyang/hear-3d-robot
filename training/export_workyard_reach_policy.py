"""Export an accepted HEAR reach checkpoint for deployment.

The contact/grasp phase must not reopen gradient authority over the accepted
14D reach primitive.  This exporter validates the qualification report and
selected checkpoint identity, reconstructs the deterministic bounded actor,
and emits batch-dynamic TorchScript and ONNX modules plus a cryptographic
identity report for the next Harness skill layer.  TorchScript remains the
training-side reference; ONNX is the portable artifact used by the TypeScript
MuJoCo runtime.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import onnx
import torch
from onnx.reference import ReferenceEvaluator
from torch import nn
from torch.nn import functional as torch_f


PROTOCOL = "hear-frozen-reach-policy-export-v1"
SOURCE_REPORT_PROTOCOL = "hear-workyard-residual-run-v4"
OBSERVATION_SIZE = 231
ACTION_SIZE = 14
NORMALIZER_EPS = 1.0e-2


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
  parser.add_argument("--output", type=Path, required=True)
  parser.add_argument("--onnx-output", type=Path, required=True)
  parser.add_argument("--report", type=Path, required=True)
  return parser.parse_args()


class FrozenReachActor(nn.Module):
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
    layers = (
      (self.linear0, "mlp.0"),
      (self.linear2, "mlp.2"),
      (self.linear4, "mlp.4"),
      (self.linear6, "mlp.6"),
    )
    with torch.no_grad():
      for layer, prefix in layers:
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
) -> tuple[dict[str, object], dict[str, torch.Tensor]]:
  if not checkpoint_path.is_file():
    raise FileNotFoundError(f"Selected reach checkpoint is missing: {checkpoint_path}")
  if not report_path.is_file():
    raise FileNotFoundError(f"Reach qualification report is missing: {report_path}")
  report = json.loads(report_path.read_text(encoding="utf-8"))
  acceptance = report.get("acceptance")
  training = report.get("training")
  selection = training.get("checkpoint_selection") if isinstance(training, dict) else None
  selected = selection.get("selected_checkpoint") if isinstance(selection, dict) else None
  if (
    report.get("protocol") != SOURCE_REPORT_PROTOCOL
    or report.get("mode") != "train"
    or not isinstance(acceptance, dict)
    or acceptance.get("phase_one_accepted") is not True
    or acceptance.get("hand_checkpoint_expansion_authorized") is not True
    or acceptance.get("waist_checkpoint_expansion_authorized") is not False
    or acceptance.get("selected_checkpoint_safety_gate_passed") is not True
    or not isinstance(selection, dict)
    or selection.get("selected_source") not in ("dagger", "ppo")
    or not isinstance(selected, dict)
    or selected.get("file") != checkpoint_path.name
    or selected.get("bytes") != checkpoint_path.stat().st_size
    or selected.get("sha256") != sha256(checkpoint_path)
  ):
    raise ValueError("Reach checkpoint is not backed by an accepted qualification report")

  checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
  state = checkpoint.get("actor_state_dict")
  if not isinstance(state, dict):
    raise ValueError("Reach checkpoint has no actor state")
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
      raise ValueError(f"Reach actor tensor {name} has the wrong shape")
    if not torch.isfinite(value).all():
      raise ValueError(f"Reach actor tensor {name} is non-finite")
  if (state["obs_normalizer._std"] < 0.0).any():
    raise ValueError("Reach actor normalizer has a negative standard deviation")
  return report, state


def main() -> None:
  args = parse_args()
  checkpoint_path = args.checkpoint.resolve()
  source_report_path = args.training_report.resolve()
  output_path = args.output.resolve()
  onnx_path = args.onnx_output.resolve()
  report_path = args.report.resolve()
  for path in (output_path, onnx_path, report_path):
    if path.exists():
      raise FileExistsError(f"Reach export output already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)

  source_report, state = validate_source(checkpoint_path, source_report_path)
  actor = FrozenReachActor(state)
  scripted = torch.jit.script(actor)
  torch.manual_seed(42)
  normalized_probes = torch.cat((
    torch.zeros((1, OBSERVATION_SIZE)),
    torch.ones((1, OBSERVATION_SIZE)),
    torch.randn((14, OBSERVATION_SIZE)),
  ))
  # Probe the learned operating distribution instead of feeding raw all-one
  # observations through dimensions whose fitted standard deviation is near
  # zero.  ONNX's reference Softplus implementation evaluates both branches
  # of its expression and can overflow on those artificial values even though
  # Torch and production ONNX Runtime evaluate the same finite policy.
  probes = actor.observation_mean + normalized_probes * (
    actor.observation_std + NORMALIZER_EPS
  )
  with torch.inference_mode():
    eager_output = actor(probes)
    scripted_output = scripted(probes)
  maximum_export_error = float((eager_output - scripted_output).abs().max().item())
  if (
    tuple(scripted_output.shape) != (16, ACTION_SIZE)
    or not torch.isfinite(scripted_output).all()
    or float(scripted_output.abs().max().item()) > 1.0 + 1.0e-6
    or maximum_export_error > 1.0e-6
  ):
    raise RuntimeError("Frozen reach policy export validation failed")
  torch.jit.save(scripted, str(output_path))

  torch.onnx.export(
    actor,
    (probes[:1],),
    str(onnx_path),
    input_names=["observation"],
    output_names=["reach_action"],
    dynamic_axes={
      "observation": {0: "batch"},
      "reach_action": {0: "batch"},
    },
    opset_version=17,
    dynamo=False,
  )
  model = onnx.load(str(onnx_path))
  onnx.checker.check_model(model, full_check=True)
  reference = ReferenceEvaluator(model)
  onnx_output = reference.run(
    ["reach_action"],
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
    raise RuntimeError("Frozen reach ONNX validation failed")

  source_acceptance = source_report["acceptance"]
  source_evaluation = source_report["evaluation"]
  payload = {
    "protocol": PROTOCOL,
    "created_at": datetime.now(timezone.utc).isoformat(),
    "source": {
      "training_report_file": source_report_path.name,
      "training_report_sha256": sha256(source_report_path),
      "checkpoint_file": checkpoint_path.name,
      "checkpoint_bytes": checkpoint_path.stat().st_size,
      "checkpoint_sha256": sha256(checkpoint_path),
      "selected_source": source_report["training"]["checkpoint_selection"][
        "selected_source"
      ],
      "phase_one_accepted": source_acceptance["phase_one_accepted"],
      "hand_checkpoint_expansion_authorized": source_acceptance[
        "hand_checkpoint_expansion_authorized"
      ],
      "waist_checkpoint_expansion_authorized": source_acceptance[
        "waist_checkpoint_expansion_authorized"
      ],
      "held_out_environment_count": source_evaluation["environment_count"],
      "held_out_success_rate": source_evaluation["success_rate"],
    },
    "policy": {
      "file": output_path.name,
      "bytes": output_path.stat().st_size,
      "sha256": sha256(output_path),
      "runtime": "torchscript_cuda",
      "input": "hear-workyard-residual-observation-v4",
      "input_size": OBSERVATION_SIZE,
      "output": "bounded-upper-body-residual-mean",
      "output_size": ACTION_SIZE,
      "distribution": "beta_bounded_minus_one_one",
      "deterministic_statistic": "mean",
      "batch_dynamic": True,
      "normalizer_epsilon": NORMALIZER_EPS,
      "parameter_count": sum(parameter.numel() for parameter in actor.parameters()),
      "gradient_parameter_count": sum(
        parameter.numel() for parameter in actor.parameters() if parameter.requires_grad
      ),
    },
    "onnx": {
      "file": onnx_path.name,
      "bytes": onnx_path.stat().st_size,
      "sha256": sha256(onnx_path),
      "runtime": "onnxruntime",
      "opset": 17,
      "input": "observation",
      "input_protocol": "hear-workyard-residual-observation-v4",
      "input_size": OBSERVATION_SIZE,
      "output": "reach_action",
      "output_protocol": "bounded-upper-body-residual-mean",
      "output_size": ACTION_SIZE,
      "batch_dynamic": True,
    },
    "validation": {
      "probe_batch_size": len(probes),
      "maximum_export_error": maximum_export_error,
      "maximum_onnx_error": maximum_onnx_error,
      "maximum_absolute_action": float(scripted_output.abs().max().item()),
      "finite": True,
      "onnx_checker_full": True,
    },
  }
  report_path.write_text(
    json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
  )
  print(json.dumps(payload, ensure_ascii=False))


if __name__ == "__main__":
  main()
