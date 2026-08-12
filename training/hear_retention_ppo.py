"""HEAR PPO with rollout-aligned analytic-teacher retention.

The Harness grants a policy exactly one bounded Skill action window (currently
14D reach or 8D hand synergy).  Every learner-visited state in an on-policy
rollout receives its analytic teacher label before the environment is stepped.
The label is carried inside
the rollout observation TensorDict under private keys, so RSL-RL's own
shuffling keeps it aligned with the exact PPO minibatch without patching or
forking RSL-RL.

The implementation intentionally supports the feed-forward PPO configuration
used by Workyard v4.  RND, symmetry augmentation, and recurrent policies must
be integrated explicitly before they can be enabled for this safety-sensitive
training path.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Final

import torch
import torch.nn as nn
import torch.nn.functional as torch_f
from tensordict import TensorDict

from rsl_rl.algorithms import PPO
from rsl_rl.env import VecEnv


TEACHER_ACTION_KEY: Final = "_hear_rollout_teacher_action"
TEACHER_VALID_KEY: Final = "_hear_rollout_teacher_valid"


class HearRetentionPPO(PPO):
  """Project-owned PPO objective with learner-rollout teacher supervision."""

  teacher_action_provider: Callable[[], torch.Tensor] | None = None
  teacher_loss_coefficient: float = 0.0
  teacher_maximum_action_std: float = 0.15
  teacher_dispersion_coefficient: float = 1.0

  @staticmethod
  def construct_algorithm(
    obs: TensorDict,
    env: VecEnv,
    cfg: dict,
    device: str,
  ) -> "HearRetentionPPO":
    """Construct through RSL-RL while reserving aligned private rollout fields."""
    if TEACHER_ACTION_KEY in obs or TEACHER_VALID_KEY in obs:
      raise ValueError("HEAR retention rollout keys collide with environment observations")
    actor_observation = obs.get("actor")
    if actor_observation is None or actor_observation.ndim != 2:
      raise ValueError("HEAR retention PPO requires a flat batched actor observation")

    storage_observation = obs.clone()
    storage_observation.set(
      TEACHER_ACTION_KEY,
      torch.zeros(
        (env.num_envs, env.num_actions),
        dtype=actor_observation.dtype,
        device=actor_observation.device,
      ),
    )
    storage_observation.set(
      TEACHER_VALID_KEY,
      torch.zeros(
        (env.num_envs, 1),
        dtype=actor_observation.dtype,
        device=actor_observation.device,
      ),
    )
    algorithm = PPO.construct_algorithm(storage_observation, env, cfg, device)
    if not isinstance(algorithm, HearRetentionPPO):
      raise TypeError("RSL-RL did not construct the configured HEAR retention PPO")
    algorithm.last_retention_metrics = algorithm._empty_retention_metrics()
    return algorithm

  def configure_rollout_teacher(
    self,
    provider: Callable[[], torch.Tensor],
    loss_coefficient: float,
    maximum_action_std: float,
    dispersion_coefficient: float,
  ) -> None:
    """Bind the environment-local analytic teacher after runner construction."""
    if not callable(provider):
      raise TypeError("Rollout teacher provider must be callable")
    if not torch.isfinite(torch.tensor(loss_coefficient)) or loss_coefficient < 0.0:
      raise ValueError("Rollout teacher loss coefficient must be finite and non-negative")
    if (
      not torch.isfinite(torch.tensor(maximum_action_std))
      or not 0.0 < maximum_action_std <= 1.0
    ):
      raise ValueError("Maximum teacher-guided action std must be inside (0, 1]")
    if (
      not torch.isfinite(torch.tensor(dispersion_coefficient))
      or dispersion_coefficient < 0.0
    ):
      raise ValueError("Teacher dispersion coefficient must be finite and non-negative")
    self.teacher_action_provider = provider
    self.teacher_loss_coefficient = float(loss_coefficient)
    self.teacher_maximum_action_std = float(maximum_action_std)
    self.teacher_dispersion_coefficient = float(dispersion_coefficient)

  def act(self, obs: TensorDict) -> torch.Tensor:
    """Sample the learner action and label the same pre-step state."""
    actions = super().act(obs)
    if self.teacher_action_provider is None:
      teacher_action = torch.zeros_like(actions)
      teacher_valid = torch.zeros(
        (actions.shape[0], 1), dtype=actions.dtype, device=actions.device
      )
    else:
      with torch.no_grad():
        teacher_action = self.teacher_action_provider().detach()
      if teacher_action.shape != actions.shape:
        raise RuntimeError(
          "Rollout teacher action shape disagrees with the authorized actor slice: "
          f"{tuple(teacher_action.shape)} != {tuple(actions.shape)}"
        )
      finite = torch.isfinite(teacher_action).all(dim=-1, keepdim=True)
      if finite.any() and teacher_action[finite.squeeze(-1)].abs().max() > 1.0 + 1.0e-6:
        raise RuntimeError("Rollout teacher exceeded the authorized [-1, 1] action range")
      teacher_valid = finite.to(dtype=actions.dtype)
      teacher_action = torch.where(
        finite,
        teacher_action.clamp(-1.0, 1.0),
        torch.zeros_like(teacher_action),
      )

    # The actor/critic observation sets reference only their public groups, so
    # these fields remain training-private and cannot become policy inputs.
    transition_observation = obs.clone()
    transition_observation.set(TEACHER_ACTION_KEY, teacher_action)
    transition_observation.set(TEACHER_VALID_KEY, teacher_valid)
    self.transition.observations = transition_observation
    return actions

  def deterministic_policy_statistics(
    self, obs: TensorDict
  ) -> tuple[torch.Tensor, torch.Tensor]:
    """Return bounded policy mean/std without consuming the sampling RNG."""
    actor = self._raw_actor
    if actor.distribution is None:
      raise RuntimeError("HEAR retention actor has no bounded distribution")
    latent = actor.get_latent(obs)
    actor.distribution.update(actor.mlp(latent))
    return actor.output_mean, actor.output_std

  def update(self) -> dict[str, float]:
    """Optimize PPO plus teacher mean/dispersion retention per minibatch."""
    rollout_metrics = self._rollout_label_metrics()
    if self.teacher_loss_coefficient == 0.0:
      losses = super().update()
      self.last_retention_metrics = rollout_metrics
      losses.update({
        "rollout_teacher": 0.0,
        "rollout_teacher_label_coverage": float(
          rollout_metrics["rollout_teacher_label_coverage"]
        ),
      })
      return losses

    if self.teacher_action_provider is None:
      raise RuntimeError("Retention PPO has no rollout teacher provider")
    if self.actor.is_recurrent or self.critic.is_recurrent:
      raise RuntimeError("HEAR retention PPO does not yet authorize recurrent models")
    if self.rnd is not None or self.symmetry is not None:
      raise RuntimeError("HEAR retention PPO requires explicit RND/symmetry integration")
    if self.schedule != "fixed":
      raise RuntimeError("HEAR retention PPO requires a fixed learning-rate schedule")
    if rollout_metrics["rollout_teacher_label_coverage"] <= 0.0:
      raise RuntimeError("Retention PPO rollout contains no valid teacher labels")

    mean_value_loss = 0.0
    mean_surrogate_loss = 0.0
    mean_entropy = 0.0
    mean_teacher_loss = 0.0
    mean_teacher_imitation_loss = 0.0
    mean_teacher_dispersion_penalty = 0.0
    mean_policy_action_std = 0.0
    mean_teacher_loss_by_action = torch.zeros(
      self.storage.actions_shape[-1], dtype=torch.float32, device=self.device
    )
    generator = self.storage.mini_batch_generator(
      self.num_mini_batches, self.num_learning_epochs
    )

    for batch in generator:
      if self.normalize_advantage_per_mini_batch:
        with torch.no_grad():
          batch.advantages = (
            batch.advantages - batch.advantages.mean()
          ) / (batch.advantages.std() + 1.0e-8)

      self.actor(batch.observations, stochastic_output=True)
      actions_log_prob = self.actor.get_output_log_prob(batch.actions)
      values = self.critic(batch.observations)
      entropy = self.actor.output_entropy

      ratio = torch.exp(actions_log_prob - torch.squeeze(batch.old_actions_log_prob))
      surrogate = -torch.squeeze(batch.advantages) * ratio
      surrogate_clipped = -torch.squeeze(batch.advantages) * torch.clamp(
        ratio, 1.0 - self.clip_param, 1.0 + self.clip_param
      )
      surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

      if self.use_clipped_value_loss:
        value_clipped = batch.values + (values - batch.values).clamp(
          -self.clip_param, self.clip_param
        )
        value_losses = (values - batch.returns).pow(2)
        value_losses_clipped = (value_clipped - batch.returns).pow(2)
        value_loss = torch.max(value_losses, value_losses_clipped).mean()
      else:
        value_loss = (batch.returns - values).pow(2).mean()

      teacher_action = batch.observations[TEACHER_ACTION_KEY]
      teacher_valid = batch.observations[TEACHER_VALID_KEY].squeeze(-1) > 0.5
      if not teacher_valid.any():
        raise RuntimeError("A retention PPO minibatch contains no valid teacher label")
      predicted_action = self.actor.output_mean
      element_teacher_imitation_loss = torch_f.smooth_l1_loss(
        predicted_action[teacher_valid],
        teacher_action[teacher_valid],
        reduction="none",
      )
      teacher_imitation_loss = element_teacher_imitation_loss.mean()
      policy_action_std = self.actor.output_std[teacher_valid]
      teacher_dispersion_penalty = torch.relu(
        policy_action_std - self.teacher_maximum_action_std
      ).square().mean()
      teacher_loss = (
        teacher_imitation_loss
        + self.teacher_dispersion_coefficient * teacher_dispersion_penalty
      )
      loss = (
        surrogate_loss
        + self.value_loss_coef * value_loss
        - self.entropy_coef * entropy.mean()
        + self.teacher_loss_coefficient * teacher_loss
      )

      self.optimizer.zero_grad()
      loss.backward()
      if self.is_multi_gpu:
        self.reduce_parameters()
      nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
      nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
      self.optimizer.step()

      mean_value_loss += value_loss.item()
      mean_surrogate_loss += surrogate_loss.item()
      mean_entropy += entropy.mean().item()
      mean_teacher_loss += teacher_loss.item()
      mean_teacher_imitation_loss += teacher_imitation_loss.item()
      mean_teacher_dispersion_penalty += teacher_dispersion_penalty.item()
      mean_policy_action_std += policy_action_std.detach().mean().item()
      mean_teacher_loss_by_action += (
        element_teacher_imitation_loss.detach().mean(dim=0)
      )

    num_updates = self.num_learning_epochs * self.num_mini_batches
    mean_value_loss /= num_updates
    mean_surrogate_loss /= num_updates
    mean_entropy /= num_updates
    mean_teacher_loss /= num_updates
    mean_teacher_imitation_loss /= num_updates
    mean_teacher_dispersion_penalty /= num_updates
    mean_policy_action_std /= num_updates
    mean_teacher_loss_by_action /= num_updates
    self.storage.clear()

    self.last_retention_metrics = {
      **rollout_metrics,
      "joint_teacher_loss": mean_teacher_loss,
      "teacher_mean_smooth_l1": mean_teacher_imitation_loss,
      "teacher_dispersion_penalty": mean_teacher_dispersion_penalty,
      "mean_policy_action_std": mean_policy_action_std,
      "teacher_smooth_l1_by_action": [
        float(value) for value in mean_teacher_loss_by_action.tolist()
      ],
    }
    return {
      "value": mean_value_loss,
      "surrogate": mean_surrogate_loss,
      "entropy": mean_entropy,
      "rollout_teacher": mean_teacher_loss,
      "rollout_teacher_label_coverage": float(
        rollout_metrics["rollout_teacher_label_coverage"]
      ),
    }

  def distribution_identity(self) -> dict[str, object]:
    distribution = self._raw_actor.distribution
    action_range = getattr(distribution, "action_range", None)
    return {
      "class_name": type(distribution).__name__,
      "action_range": list(action_range) if action_range is not None else None,
      "structurally_bounded": (
        type(distribution).__name__ == "BetaDistribution"
        and tuple(action_range or ()) == (-1.0, 1.0)
      ),
    }

  def _rollout_label_metrics(self) -> dict[str, object]:
    step_count = int(self.storage.step)
    valid = self.storage.observations[TEACHER_VALID_KEY][:step_count]
    total = int(valid.numel())
    finite = int((valid > 0.5).sum().item())
    return {
      "rollout_teacher_label_count": total,
      "rollout_teacher_finite_label_count": finite,
      "rollout_teacher_label_coverage": finite / total if total else 0.0,
      "joint_teacher_loss": 0.0,
      "teacher_mean_smooth_l1": 0.0,
      "teacher_dispersion_penalty": 0.0,
      "mean_policy_action_std": 0.0,
      "teacher_smooth_l1_by_action": [0.0] * self.storage.actions_shape[-1],
    }

  def _empty_retention_metrics(self) -> dict[str, object]:
    return {
      "rollout_teacher_label_count": 0,
      "rollout_teacher_finite_label_count": 0,
      "rollout_teacher_label_coverage": 0.0,
      "joint_teacher_loss": 0.0,
      "teacher_mean_smooth_l1": 0.0,
      "teacher_dispersion_penalty": 0.0,
      "mean_policy_action_std": 0.0,
      "teacher_smooth_l1_by_action": [0.0] * self.storage.actions_shape[-1],
    }
