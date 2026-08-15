"""Run the phase-two contact/grasp analytic-teacher gate locally or on Colab."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser()
  parser.add_argument("--num-envs", type=int, default=16)
  parser.add_argument("--steps", type=int, default=400)
  parser.add_argument("--seed", type=int, default=42)
  parser.add_argument("--output", type=Path)
  parser.add_argument("--locomotion-teacher-root", type=Path)
  parser.add_argument("--reach-policy-root", type=Path)
  parser.add_argument("--enforce-gate", action="store_true")
  args = parser.parse_args()
  if args.num_envs <= 0 or args.steps <= 0 or args.seed < 0:
    parser.error("num-envs/steps must be positive and seed must be non-negative")
  return args


def main() -> None:
  args = parse_args()
  training_root = Path(__file__).resolve().parent
  sys.path.insert(0, str(training_root))

  import torch
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.rl import RslRlVecEnvWrapper
  from mjlab.tasks.registry import load_env_cfg, load_rl_cfg

  import workyard_contact_mjlab_env as module

  torch.manual_seed(args.seed)
  repository_root = training_root.parent
  locomotion_teacher_root = (
    args.locomotion_teacher_root
    if args.locomotion_teacher_root is not None
    else repository_root / "artifacts" / "training" / "g1-residual-teacher"
  ).resolve()
  reach_policy_root = (
    args.reach_policy_root
    if args.reach_policy_root is not None
    else repository_root / "artifacts" / "training" / "workyard-reach-deployment-v3"
  ).resolve()
  artifact_paths = {
    "locomotion teacher JIT": locomotion_teacher_root / "g1_velocity_teacher.jit.pt",
    "locomotion teacher report": locomotion_teacher_root / "training-report.json",
    "reach policy JIT": reach_policy_root / "workyard_reach.jit.pt",
    "reach policy report": reach_policy_root / "reach-policy-report.json",
  }
  missing = [label for label, path in artifact_paths.items() if not path.is_file()]
  if missing:
    raise FileNotFoundError(
      "Contact teacher gate is missing qualified artifacts: " + ", ".join(missing)
    )

  env_cfg = load_env_cfg(module.TASK_ID, play=True)
  env_cfg.scene.num_envs = args.num_envs
  env_cfg.seed = args.seed
  action_cfg = env_cfg.actions["workyard"]
  action_cfg.teacher_jit_path = str(artifact_paths["locomotion teacher JIT"])
  action_cfg.teacher_report_path = str(artifact_paths["locomotion teacher report"])
  action_cfg.reach_policy_jit_path = str(artifact_paths["reach policy JIT"])
  action_cfg.reach_policy_report_path = str(artifact_paths["reach policy report"])
  agent_cfg = load_rl_cfg(module.TASK_ID)
  raw_env = ManagerBasedRlEnv(cfg=env_cfg, device="cuda:0")
  env = RslRlVecEnvWrapper(raw_env, clip_actions=agent_cfg.clip_actions)
  environment_closed = False
  try:
    observations = env.get_observations()
    if tuple(observations["actor"].shape) != (
      args.num_envs, module.HAND_OBSERVATION_SIZE
    ):
      raise RuntimeError(f"Contact observation shape drifted: {observations['actor'].shape}")
    action_term = module._contact_action(raw_env)
    command = module.base._workyard_command(raw_env)
    initial_active_hand = command.active_hand.clone()
    hand_contact_sensors = (
      module.base._contact_sensor(raw_env, "left_hand_object_contact"),
      module.base._contact_sensor(raw_env, "right_hand_object_contact"),
    )
    hand_contact_primary_names = tuple(
      tuple(sensor.primary_names) for sensor in hand_contact_sensors
    )
    primary_contact_seen = [
      torch.zeros(
        len(sensor.primary_names) * sensor.cfg.num_slots,
        dtype=torch.bool,
        device="cuda:0",
      )
      for sensor in hand_contact_sensors
    ]
    active = torch.ones(args.num_envs, dtype=torch.bool, device="cuda:0")
    success = torch.zeros_like(active)
    object_lost = torch.zeros_like(active)
    fell = torch.zeros_like(active)
    non_foot_ground = torch.zeros_like(active)
    maximum_stage = torch.zeros(
      args.num_envs, dtype=torch.long, device="cuda:0"
    )
    reached_contact = torch.zeros_like(active)
    reached_grasp = torch.zeros_like(active)
    maximum_force = torch.zeros(args.num_envs, device="cuda:0")
    maximum_surfaces = torch.zeros(args.num_envs, device="cuda:0")
    opposing_seen = torch.zeros_like(active)
    maximum_absolute_hand_action = 0.0
    maximum_absolute_reach_action = 0.0
    maximum_absolute_frozen_reach_action = 0.0
    maximum_absolute_contact_approach_correction = 0.0
    maximum_inactive_arm_contact_approach_correction = 0.0
    maximum_outside_pocket_contact_approach_correction = 0.0
    contact_approach_correction_seen = torch.zeros_like(active)
    contact_approach_correction_env_steps = 0
    teacher_thumb_contact_latch_seen = torch.zeros_like(active)
    teacher_opposing_contact_latch_seen = torch.zeros_like(active)
    contact_pose_hold_seen = torch.zeros_like(active)
    contact_pose_hold_env_steps = 0
    closure_pose_hold_seen = torch.zeros_like(active)
    closure_pose_hold_env_steps = 0
    contact_base_assist_seen = torch.zeros_like(active)
    contact_base_assist_env_steps = 0
    maximum_contact_base_assist_speed = 0.0
    contact_base_assist_first_root_position = torch.zeros(
      (args.num_envs, 3), device="cuda:0"
    )
    contact_base_assist_maximum_root_displacement = torch.zeros(
      args.num_envs, device="cuda:0"
    )
    contact_base_assist_maximum_forward_progress = torch.zeros(
      args.num_envs, device="cuda:0"
    )
    contact_base_assist_maximum_lateral_drift = torch.zeros(
      args.num_envs, device="cuda:0"
    )
    contact_base_assist_actual_speed_sum = 0.0
    contact_base_assist_command_projection_sum = 0.0
    contact_base_assist_motion_sample_count = 0
    maximum_contact_base_assist_actual_speed = 0.0
    maximum_inactive_coordination = 0.0
    maximum_unauthorized_hand_action = 0.0
    maximum_precontact_hand_target = 0.0
    precontact_object_contact = torch.zeros_like(active)
    maximum_precontact_object_force = 0.0
    approach_safety_intervention_seen = torch.zeros_like(active)
    approach_safety_intervention_steps = 0
    precontact_safety_intervention_steps = 0
    maximum_active_hand_coordination = 0.0
    minimum_opposing_normal_dot = 1.0
    contact_sample_count = torch.zeros(2, dtype=torch.long, device="cuda:0")
    contact_rod_local_sum = torch.zeros((2, 3), device="cuda:0")
    contact_rod_local_min = torch.full((2, 3), float("inf"), device="cuda:0")
    contact_rod_local_max = torch.full((2, 3), -float("inf"), device="cuda:0")
    contact_coordination_sum = torch.zeros((2, 4), device="cuda:0")
    minimum_wrist_error = torch.full(
      (args.num_envs,), float("inf"), device="cuda:0"
    )
    first_contact_step = torch.full(
      (args.num_envs,), -1, dtype=torch.long, device="cuda:0"
    )
    first_grasp_step = torch.full_like(first_contact_step, -1)
    contact_pocket_seen = torch.zeros_like(active)
    contact_pocket_env_steps = 0
    first_contact_pocket_step = torch.full_like(first_contact_step, -1)
    contact_alignment_seen = torch.zeros_like(active)
    contact_alignment_env_steps = 0
    contact_alignment_object_contact = torch.zeros_like(active)
    maximum_contact_alignment_object_force = 0.0
    contact_retreat_seen = torch.zeros_like(active)
    contact_retreat_env_steps = 0
    contact_retreat_object_contact = torch.zeros_like(active)
    maximum_contact_retreat_object_force = 0.0
    alignment_sample_count = torch.zeros(2, dtype=torch.long, device="cuda:0")
    alignment_rod_local_sum = torch.zeros((2, 3), device="cuda:0")
    alignment_rod_local_min = torch.full((2, 3), float("inf"), device="cuda:0")
    alignment_rod_local_max = torch.full((2, 3), -float("inf"), device="cuda:0")
    alignment_target_error_sum = torch.zeros(2, device="cuda:0")
    alignment_target_error_min = torch.full((2,), float("inf"), device="cuda:0")
    alignment_target_error_max = torch.zeros(2, device="cuda:0")
    alignment_bearing_absolute_error_min = torch.full(
      (2,), float("inf"), device="cuda:0"
    )
    alignment_bearing_absolute_error_max = torch.zeros(2, device="cuda:0")
    alignment_axis_error_min = torch.full((2,), float("inf"), device="cuda:0")
    alignment_axis_error_max = torch.zeros(2, device="cuda:0")
    hand_closure_ready_seen = torch.zeros_like(active)
    hand_closure_ready_env_steps = 0
    first_hand_closure_ready_step = torch.full_like(first_contact_step, -1)
    pocket_closure_ready_seen = torch.zeros_like(active)
    pocket_closure_ready_env_steps = 0
    preshape_ready_seen = torch.zeros_like(active)
    preshape_ready_env_steps = 0
    pocket_sample_count = torch.zeros(2, dtype=torch.long, device="cuda:0")
    pocket_rod_local_sum = torch.zeros((2, 3), device="cuda:0")
    pocket_rod_local_min = torch.full((2, 3), float("inf"), device="cuda:0")
    pocket_rod_local_max = torch.full((2, 3), -float("inf"), device="cuda:0")
    pocket_wrist_delta_sum = torch.zeros((2, 3), device="cuda:0")
    pocket_wrist_delta_min = torch.full((2, 3), float("inf"), device="cuda:0")
    pocket_wrist_delta_max = torch.full((2, 3), -float("inf"), device="cuda:0")
    pocket_wrist_error_sum = torch.zeros(2, device="cuda:0")
    pocket_wrist_error_min = torch.full((2,), float("inf"), device="cuda:0")
    pocket_wrist_error_max = torch.zeros(2, device="cuda:0")
    pocket_wrist_bearing_absolute_error_sum = torch.zeros(2, device="cuda:0")
    pocket_wrist_bearing_absolute_error_min = torch.full(
      (2,), float("inf"), device="cuda:0"
    )
    pocket_wrist_bearing_absolute_error_max = torch.zeros(2, device="cuda:0")
    pocket_coordination_sum = torch.zeros((2, 4), device="cuda:0")
    pocket_coordination_min = torch.full((2, 4), float("inf"), device="cuda:0")
    pocket_coordination_max = torch.full((2, 4), -float("inf"), device="cuda:0")
    pocket_joint_position_sum = torch.zeros((2, 7), device="cuda:0")
    pocket_joint_position_min = torch.full((2, 7), float("inf"), device="cuda:0")
    pocket_joint_position_max = torch.full((2, 7), -float("inf"), device="cuda:0")
    pocket_joint_target_sum = torch.zeros((2, 7), device="cuda:0")
    pocket_joint_target_min = torch.full((2, 7), float("inf"), device="cuda:0")
    pocket_joint_target_max = torch.full((2, 7), -float("inf"), device="cuda:0")
    pocket_joint_absolute_error_sum = torch.zeros((2, 7), device="cuda:0")
    pocket_joint_absolute_error_max = torch.zeros((2, 7), device="cuda:0")
    pocket_upper_joint_position_sum = torch.zeros((2, 7), device="cuda:0")
    pocket_upper_joint_target_sum = torch.zeros((2, 7), device="cuda:0")
    pocket_upper_joint_absolute_error_max = torch.zeros((2, 7), device="cuda:0")
    pocket_reach_unclamped_action_absolute_max = torch.zeros(
      (2, 7), device="cuda:0"
    )
    pocket_reach_authority_saturation_seen = torch.zeros(
      (2, 7), dtype=torch.bool, device="cuda:0"
    )
    pocket_reach_soft_limit_saturation_seen = torch.zeros_like(
      pocket_reach_authority_saturation_seen
    )
    pocket_reach_command_lead_saturation_seen = torch.zeros_like(
      pocket_reach_authority_saturation_seen
    )
    pocket_reach_minimum_singular_value = torch.full(
      (2,), float("inf"), device="cuda:0"
    )
    reward_sum = torch.zeros(args.num_envs, device="cuda:0")
    termination_count = 0
    completed_control_steps = 0

    with torch.inference_mode():
      for step_index in range(args.steps):
        active_before = active.clone()
        if not bool(active_before.any().item()):
          break
        completed_control_steps = step_index + 1
        stage_before = command.teacher_stage.clone()
        maximum_stage = torch.maximum(maximum_stage, stage_before)
        reached_contact |= stage_before >= module.CONTACT_STAGE_INDEX
        reached_grasp |= stage_before >= module.GRASP_STAGE_INDEX
        first_contact_step = torch.where(
          (first_contact_step < 0) & (stage_before >= module.CONTACT_STAGE_INDEX),
          torch.full_like(first_contact_step, step_index),
          first_contact_step,
        )
        first_grasp_step = torch.where(
          (first_grasp_step < 0) & (stage_before >= module.GRASP_STAGE_INDEX),
          torch.full_like(first_grasp_step, step_index),
          first_grasp_step,
        )
        teacher_action = action_term.compute_hand_teacher_action()
        observations, rewards, dones, _ = env.step(teacher_action)
        if not torch.isfinite(observations["actor"]).all():
          raise RuntimeError("Contact teacher produced a non-finite observation")
        if not torch.isfinite(rewards).all():
          raise RuntimeError("Contact teacher produced a non-finite reward")
        reward_sum += rewards * active_before
        maximum_absolute_hand_action = max(
          maximum_absolute_hand_action,
          float(action_term.hand_action.abs().max().item()),
        )
        maximum_absolute_reach_action = max(
          maximum_absolute_reach_action,
          float(action_term.raw_action.abs().max().item()),
        )
        maximum_absolute_frozen_reach_action = max(
          maximum_absolute_frozen_reach_action,
          float(action_term.frozen_reach_action.abs().max().item()),
        )
        correction_active = (
          action_term.contact_approach_correction_active & active_before
        )
        contact_approach_correction_seen |= correction_active
        contact_approach_correction_env_steps += int(
          correction_active.sum().item()
        )
        correction_delta = action_term.contact_approach_correction_delta.reshape(
          args.num_envs, 2, 7
        )
        maximum_absolute_contact_approach_correction = max(
          maximum_absolute_contact_approach_correction,
          float(correction_delta.abs().max().item()),
        )
        inactive_correction = correction_delta[
          torch.arange(args.num_envs, device="cuda:0"),
          1 - command.active_hand,
        ]
        maximum_inactive_arm_contact_approach_correction = max(
          maximum_inactive_arm_contact_approach_correction,
          float(inactive_correction.abs().max().item()),
        )
        outside_pocket_correction = torch.where(
          action_term.contact_approach_correction_active[:, None, None],
          torch.zeros_like(correction_delta),
          correction_delta,
        )
        maximum_outside_pocket_contact_approach_correction = max(
          maximum_outside_pocket_contact_approach_correction,
          float(outside_pocket_correction.abs().max().item()),
        )
        latch_rows = torch.arange(args.num_envs, device="cuda:0")
        teacher_thumb_contact_latch_seen |= (
          action_term.teacher_thumb_contact_latched[
            latch_rows, command.active_hand
          ] & active_before
        )
        teacher_opposing_contact_latch_seen |= (
          action_term.teacher_opposing_contact_latched[
            latch_rows, command.active_hand
          ] & active_before
        )
        contact_pose_hold = action_term.contact_pose_hold_active & active_before
        contact_pose_hold_seen |= contact_pose_hold
        contact_pose_hold_env_steps += int(contact_pose_hold.sum().item())
        closure_pose_hold = action_term.closure_pose_hold_active & active_before
        closure_pose_hold_seen |= closure_pose_hold
        closure_pose_hold_env_steps += int(closure_pose_hold.sum().item())
        contact_base_assist = command.contact_base_assist_active & active_before
        contact_base_assist_seen |= contact_base_assist
        contact_base_assist_env_steps += int(contact_base_assist.sum().item())
        maximum_contact_base_assist_speed = max(
          maximum_contact_base_assist_speed,
          float(
            torch.linalg.vector_norm(
              command.desired_base_twist[:, :2], dim=-1
            ).max().item()
          ),
        )
        # `seen` was updated above, so identify the first sample from the
        # absence of a stored root position instead of the updated mask.
        unstored_assist = contact_base_assist & (
          contact_base_assist_first_root_position.abs().sum(dim=-1) == 0.0
        )
        root_position = command.robot.data.root_link_pos_w
        contact_base_assist_first_root_position[:] = torch.where(
          unstored_assist.unsqueeze(-1),
          root_position,
          contact_base_assist_first_root_position,
        )
        root_displacement = torch.linalg.vector_norm(
          root_position[:, :2]
            - contact_base_assist_first_root_position[:, :2],
          dim=-1,
        )
        contact_base_assist_maximum_root_displacement = torch.maximum(
          contact_base_assist_maximum_root_displacement,
          torch.where(
            contact_base_assist,
            root_displacement,
            torch.zeros_like(root_displacement),
          ),
        )
        planar_displacement = (
          root_position[:, :2]
          - contact_base_assist_first_root_position[:, :2]
        )
        forward_progress = torch.sum(
          planar_displacement * command.contact_base_assist_direction_w,
          dim=-1,
        )
        lateral_direction_w = torch.stack((
          -command.contact_base_assist_direction_w[:, 1],
          command.contact_base_assist_direction_w[:, 0],
        ), dim=-1)
        lateral_drift = torch.abs(torch.sum(
          planar_displacement * lateral_direction_w, dim=-1
        ))
        contact_base_assist_maximum_forward_progress = torch.maximum(
          contact_base_assist_maximum_forward_progress,
          torch.where(
            contact_base_assist,
            forward_progress,
            torch.zeros_like(forward_progress),
          ),
        )
        contact_base_assist_maximum_lateral_drift = torch.maximum(
          contact_base_assist_maximum_lateral_drift,
          torch.where(
            contact_base_assist,
            lateral_drift,
            torch.zeros_like(lateral_drift),
          ),
        )
        if bool(contact_base_assist.any().item()):
          actual_velocity = command.robot.data.root_link_lin_vel_b[
            contact_base_assist, :2
          ]
          desired_velocity = command.desired_base_twist[
            contact_base_assist, :2
          ]
          actual_speed = torch.linalg.vector_norm(actual_velocity, dim=-1)
          desired_direction = desired_velocity / torch.linalg.vector_norm(
            desired_velocity, dim=-1, keepdim=True
          ).clamp_min(1e-6)
          command_projection = torch.sum(
            actual_velocity * desired_direction, dim=-1
          )
          contact_base_assist_actual_speed_sum += float(actual_speed.sum().item())
          contact_base_assist_command_projection_sum += float(
            command_projection.sum().item()
          )
          contact_base_assist_motion_sample_count += int(actual_speed.numel())
          maximum_contact_base_assist_actual_speed = max(
            maximum_contact_base_assist_actual_speed,
            float(actual_speed.max().item()),
          )
        safety_intervention = (
          action_term.approach_safety_intervention & active_before
        )
        approach_safety_intervention_seen |= safety_intervention
        approach_safety_intervention_steps += int(
          safety_intervention.sum().item()
        )
        precontact_safety_intervention_steps += int(
          (safety_intervention & (stage_before < module.CONTACT_STAGE_INDEX))
          .sum().item()
        )
        unauthorized = torch.where(
          action_term.authority_mask,
          torch.zeros_like(action_term.hand_action),
          action_term.hand_action,
        )
        maximum_unauthorized_hand_action = max(
          maximum_unauthorized_hand_action,
          float(unauthorized.abs().max().item()),
        )
        coordination = action_term.coordination.reshape(args.num_envs, 2, 4)
        rows = torch.arange(args.num_envs, device="cuda:0")
        inactive_side = 1 - command.active_hand
        inactive_coordination = coordination[rows, inactive_side]
        active_coordination = coordination[rows, command.active_hand]
        maximum_inactive_coordination = max(
          maximum_inactive_coordination,
          float(inactive_coordination.abs().max().item()),
        )
        maximum_active_hand_coordination = max(
          maximum_active_hand_coordination,
          float(active_coordination.abs().max().item()),
        )
        precontact = stage_before < module.CONTACT_STAGE_INDEX
        if bool(precontact.any().item()):
          maximum_precontact_hand_target = max(
            maximum_precontact_hand_target,
            float(action_term.hand_targets[precontact].abs().max().item()),
          )
        found, force, surfaces, opposed = module.base._hand_contact_summary(raw_env)
        active_hand = command.active_hand
        active_found = found[rows, active_hand]
        active_force = force[rows, active_hand]
        active_surfaces = surfaces[rows, active_hand]
        active_opposed = opposed[rows, active_hand]
        precontact_active = precontact & active_before
        precontact_object_contact |= active_found & precontact_active
        if bool(precontact_active.any().item()):
          maximum_precontact_object_force = max(
            maximum_precontact_object_force,
            float(active_force[precontact_active].max().item()),
          )
        maximum_force = torch.maximum(
          maximum_force,
          torch.where(active_before, active_force, torch.zeros_like(active_force)),
        )
        maximum_surfaces = torch.maximum(
          maximum_surfaces,
          torch.where(active_before, active_surfaces, torch.zeros_like(active_surfaces)),
        )
        opposing_seen |= active_opposed & active_before
        wrist_pose = command.robot.data.body_link_pose_w[:, command._wrist_body_ids]
        active_wrist_pose = wrist_pose[rows, active_hand]
        rod_local = module.base.quat_apply_inverse(
          active_wrist_pose[:, 3:7],
          command.rod.data.root_link_pos_w - active_wrist_pose[:, :3],
        )
        contacting = active_found & active_before
        for side in range(2):
          side_contacting = contacting & (active_hand == side)
          if bool(side_contacting.any().item()):
            local_samples = rod_local[side_contacting]
            coordination_samples = coordination[
              side_contacting, side
            ]
            contact_sample_count[side] += side_contacting.sum()
            contact_rod_local_sum[side] += local_samples.sum(dim=0)
            contact_rod_local_min[side] = torch.minimum(
              contact_rod_local_min[side], local_samples.amin(dim=0)
            )
            contact_rod_local_max[side] = torch.maximum(
              contact_rod_local_max[side], local_samples.amax(dim=0)
            )
            contact_coordination_sum[side] += coordination_samples.sum(dim=0)
        for side, sensor in enumerate(hand_contact_sensors):
          sensor_found = sensor.data.found
          sensor_normal = sensor.data.normal
          if sensor_found is None or sensor_normal is None:
            raise RuntimeError("Hand contact diagnostics are unavailable")
          relevant = active_before & (command.active_hand == side)
          if bool(relevant.any().item()):
            contacted = sensor_found[relevant] > 0
            primary_contact_seen[side] |= contacted.any(dim=0)
            normals = torch.nn.functional.normalize(
              sensor_normal[relevant], dim=-1, eps=1e-6
            )
            dot = torch.einsum("bik,bjk->bij", normals, normals)
            valid_pairs = contacted.unsqueeze(2) & contacted.unsqueeze(1)
            eye = torch.eye(
              dot.shape[-1], dtype=torch.bool, device="cuda:0"
            ).unsqueeze(0)
            valid_dot = torch.where(
              valid_pairs & ~eye, dot, torch.full_like(dot, float("inf"))
            )
            candidate = float(valid_dot.amin().item())
            if candidate != float("inf"):
              minimum_opposing_normal_dot = min(
                minimum_opposing_normal_dot, candidate
              )
        wrist_error = module.reach.active_wrist_position_error(raw_env)
        minimum_wrist_error = torch.minimum(
          minimum_wrist_error,
          torch.where(active_before, wrist_error, minimum_wrist_error),
        )
        wrist_delta = module.reach.active_wrist_position_delta(raw_env)
        closure_ready = module.hand_closure_ready(raw_env) & active_before
        hand_closure_ready_seen |= closure_ready
        hand_closure_ready_env_steps += int(closure_ready.sum().item())
        first_hand_closure_ready_step = torch.where(
          (first_hand_closure_ready_step < 0) & closure_ready,
          torch.full_like(first_hand_closure_ready_step, step_index),
          first_hand_closure_ready_step,
        )
        preshape_ready = (
          coordination[rows, command.active_hand, 2:].amin(dim=-1) >= 0.70
        ) & active_before
        preshape_ready_seen |= preshape_ready
        preshape_ready_env_steps += int(preshape_ready.sum().item())
        pocket_active = command.contact_pocket_active & active_before
        alignment_active = command.contact_alignment_active & active_before
        retreat_active = command.contact_retreat_active & active_before
        contact_retreat_seen |= retreat_active
        contact_retreat_env_steps += int(retreat_active.sum().item())
        contact_retreat_object_contact |= active_found & retreat_active
        if bool(retreat_active.any().item()):
          maximum_contact_retreat_object_force = max(
            maximum_contact_retreat_object_force,
            float(active_force[retreat_active].max().item()),
          )
        contact_alignment_seen |= alignment_active
        contact_alignment_env_steps += int(alignment_active.sum().item())
        contact_alignment_object_contact |= active_found & alignment_active
        if bool(alignment_active.any().item()):
          maximum_contact_alignment_object_force = max(
            maximum_contact_alignment_object_force,
            float(active_force[alignment_active].max().item()),
          )
        pocket_cfg = command.cfg
        final_planar = torch.tensor(
          (
            (pocket_cfg.contact_pocket_forward_m, -pocket_cfg.contact_pocket_lateral_m),
            (pocket_cfg.contact_pocket_forward_m, pocket_cfg.contact_pocket_lateral_m),
          ),
          device="cuda:0",
        )
        alignment_planar = final_planar / torch.linalg.vector_norm(
          final_planar, dim=-1, keepdim=True
        ) * pocket_cfg.contact_alignment_radius_m
        alignment_target_local = torch.cat((
          alignment_planar,
          torch.full(
            (2, 1), pocket_cfg.contact_pocket_vertical_m, device="cuda:0"
          ),
        ), dim=-1)
        for side in range(2):
          side_alignment = alignment_active & (command.active_hand == side)
          if not bool(side_alignment.any().item()):
            continue
          local_samples = rod_local[side_alignment]
          target_error_samples = torch.linalg.vector_norm(
            local_samples - alignment_target_local[side], dim=-1
          )
          bearing_error_samples = action_term.reach_teacher.wrist_bearing_error[
            side_alignment, side
          ].abs()
          axis_error_samples = (
            action_term.reach_teacher.wrist_axis_alignment_error[
              side_alignment, side
            ]
          )
          alignment_sample_count[side] += side_alignment.sum()
          alignment_rod_local_sum[side] += local_samples.sum(dim=0)
          alignment_rod_local_min[side] = torch.minimum(
            alignment_rod_local_min[side], local_samples.amin(dim=0)
          )
          alignment_rod_local_max[side] = torch.maximum(
            alignment_rod_local_max[side], local_samples.amax(dim=0)
          )
          alignment_target_error_sum[side] += target_error_samples.sum()
          alignment_target_error_min[side] = torch.minimum(
            alignment_target_error_min[side], target_error_samples.amin()
          )
          alignment_target_error_max[side] = torch.maximum(
            alignment_target_error_max[side], target_error_samples.amax()
          )
          alignment_bearing_absolute_error_min[side] = torch.minimum(
            alignment_bearing_absolute_error_min[side],
            bearing_error_samples.amin(),
          )
          alignment_bearing_absolute_error_max[side] = torch.maximum(
            alignment_bearing_absolute_error_max[side],
            bearing_error_samples.amax(),
          )
          alignment_axis_error_min[side] = torch.minimum(
            alignment_axis_error_min[side], axis_error_samples.amin()
          )
          alignment_axis_error_max[side] = torch.maximum(
            alignment_axis_error_max[side], axis_error_samples.amax()
          )
        contact_pocket_seen |= pocket_active
        contact_pocket_env_steps += int(pocket_active.sum().item())
        first_contact_pocket_step = torch.where(
          (first_contact_pocket_step < 0) & pocket_active,
          torch.full_like(first_contact_pocket_step, step_index),
          first_contact_pocket_step,
        )
        pocket_closure_ready = pocket_active & closure_ready
        pocket_closure_ready_seen |= pocket_closure_ready
        pocket_closure_ready_env_steps += int(pocket_closure_ready.sum().item())
        hand_joint_position = action_term._entity.data.joint_pos[
          :, action_term._hand_ids
        ].reshape(args.num_envs, 2, 7)
        hand_joint_target = action_term.hand_targets.reshape(args.num_envs, 2, 7)
        upper_joint_position = action_term._entity.data.joint_pos[
          :, action_term._body_ids[module.reach.UPPER_BODY_SLICE]
        ].reshape(args.num_envs, 2, 7)
        upper_joint_target = action_term.body_targets[
          :, module.reach.UPPER_BODY_SLICE
        ].reshape(args.num_envs, 2, 7)
        reach_unclamped_action = action_term.reach_teacher.unclamped_action.reshape(
          args.num_envs, 2, 7
        )
        reach_authority_saturation = (
          action_term.reach_teacher.authority_saturation.reshape(
            args.num_envs, 2, 7
          )
        )
        reach_soft_limit_saturation = (
          action_term.reach_teacher.soft_limit_saturation.reshape(
            args.num_envs, 2, 7
          )
        )
        reach_command_lead_saturation = (
          action_term.reach_teacher.command_lead_saturation.reshape(
            args.num_envs, 2, 7
          )
        )
        for side in range(2):
          side_pocket = pocket_active & (command.active_hand == side)
          if not bool(side_pocket.any().item()):
            continue
          side_count = side_pocket.sum()
          rod_samples = rod_local[side_pocket]
          wrist_delta_samples = wrist_delta[side_pocket]
          wrist_error_samples = wrist_error[side_pocket]
          wrist_bearing_absolute_error_samples = (
            action_term.reach_teacher.wrist_bearing_error[
              side_pocket, side
            ].abs()
          )
          coordination_samples = coordination[side_pocket, side]
          joint_position_samples = hand_joint_position[side_pocket, side]
          joint_target_samples = hand_joint_target[side_pocket, side]
          joint_absolute_error_samples = (
            joint_position_samples - joint_target_samples
          ).abs()
          upper_joint_position_samples = upper_joint_position[side_pocket, side]
          upper_joint_target_samples = upper_joint_target[side_pocket, side]
          upper_joint_absolute_error_samples = (
            upper_joint_position_samples - upper_joint_target_samples
          ).abs()
          pocket_sample_count[side] += side_count
          pocket_rod_local_sum[side] += rod_samples.sum(dim=0)
          pocket_rod_local_min[side] = torch.minimum(
            pocket_rod_local_min[side], rod_samples.amin(dim=0)
          )
          pocket_rod_local_max[side] = torch.maximum(
            pocket_rod_local_max[side], rod_samples.amax(dim=0)
          )
          pocket_wrist_delta_sum[side] += wrist_delta_samples.sum(dim=0)
          pocket_wrist_delta_min[side] = torch.minimum(
            pocket_wrist_delta_min[side], wrist_delta_samples.amin(dim=0)
          )
          pocket_wrist_delta_max[side] = torch.maximum(
            pocket_wrist_delta_max[side], wrist_delta_samples.amax(dim=0)
          )
          pocket_wrist_error_sum[side] += wrist_error_samples.sum()
          pocket_wrist_error_min[side] = torch.minimum(
            pocket_wrist_error_min[side], wrist_error_samples.amin()
          )
          pocket_wrist_error_max[side] = torch.maximum(
            pocket_wrist_error_max[side], wrist_error_samples.amax()
          )
          pocket_wrist_bearing_absolute_error_sum[side] += (
            wrist_bearing_absolute_error_samples.sum()
          )
          pocket_wrist_bearing_absolute_error_min[side] = torch.minimum(
            pocket_wrist_bearing_absolute_error_min[side],
            wrist_bearing_absolute_error_samples.amin(),
          )
          pocket_wrist_bearing_absolute_error_max[side] = torch.maximum(
            pocket_wrist_bearing_absolute_error_max[side],
            wrist_bearing_absolute_error_samples.amax(),
          )
          pocket_coordination_sum[side] += coordination_samples.sum(dim=0)
          pocket_coordination_min[side] = torch.minimum(
            pocket_coordination_min[side], coordination_samples.amin(dim=0)
          )
          pocket_coordination_max[side] = torch.maximum(
            pocket_coordination_max[side], coordination_samples.amax(dim=0)
          )
          pocket_joint_position_sum[side] += joint_position_samples.sum(dim=0)
          pocket_joint_position_min[side] = torch.minimum(
            pocket_joint_position_min[side], joint_position_samples.amin(dim=0)
          )
          pocket_joint_position_max[side] = torch.maximum(
            pocket_joint_position_max[side], joint_position_samples.amax(dim=0)
          )
          pocket_joint_target_sum[side] += joint_target_samples.sum(dim=0)
          pocket_joint_target_min[side] = torch.minimum(
            pocket_joint_target_min[side], joint_target_samples.amin(dim=0)
          )
          pocket_joint_target_max[side] = torch.maximum(
            pocket_joint_target_max[side], joint_target_samples.amax(dim=0)
          )
          pocket_joint_absolute_error_sum[side] += (
            joint_absolute_error_samples.sum(dim=0)
          )
          pocket_joint_absolute_error_max[side] = torch.maximum(
            pocket_joint_absolute_error_max[side],
            joint_absolute_error_samples.amax(dim=0),
          )
          pocket_upper_joint_position_sum[side] += (
            upper_joint_position_samples.sum(dim=0)
          )
          pocket_upper_joint_target_sum[side] += (
            upper_joint_target_samples.sum(dim=0)
          )
          pocket_upper_joint_absolute_error_max[side] = torch.maximum(
            pocket_upper_joint_absolute_error_max[side],
            upper_joint_absolute_error_samples.amax(dim=0),
          )
          pocket_reach_unclamped_action_absolute_max[side] = torch.maximum(
            pocket_reach_unclamped_action_absolute_max[side],
            reach_unclamped_action[side_pocket, side].abs().amax(dim=0),
          )
          pocket_reach_authority_saturation_seen[side] |= (
            reach_authority_saturation[side_pocket, side].any(dim=0)
          )
          pocket_reach_soft_limit_saturation_seen[side] |= (
            reach_soft_limit_saturation[side_pocket, side].any(dim=0)
          )
          pocket_reach_command_lead_saturation_seen[side] |= (
            reach_command_lead_saturation[side_pocket, side].any(dim=0)
          )
          pocket_reach_minimum_singular_value[side] = torch.minimum(
            pocket_reach_minimum_singular_value[side],
            action_term.reach_teacher.minimum_singular_value[
              side_pocket, side
            ].amin(),
          )
        grasp_term = raw_env.termination_manager.get_term("grasp_success").bool()
        lost_term = raw_env.termination_manager.get_term("object_lost").bool()
        fall_term = raw_env.termination_manager.get_term("fall").bool()
        ground_term = raw_env.termination_manager.get_term("non_foot_ground").bool()
        success |= grasp_term & active_before
        object_lost |= lost_term & active_before
        fell |= fall_term & active_before
        non_foot_ground |= ground_term & active_before
        termination_count += int((dones.bool() & active_before).sum().item())
        active &= ~dones.bool()

    payload = {
      "protocol": "hear-workyard-contact-teacher-gate-v1",
      "created_at": datetime.now(timezone.utc).isoformat(),
      "task": module.TASK_ID,
      "accelerator": {
        "device": torch.cuda.get_device_name(0),
        "cuda_version": torch.version.cuda,
      },
      "contract": {
        "reach_observation_size": module.REACH_OBSERVATION_SIZE,
        "hand_observation_size": module.HAND_OBSERVATION_SIZE,
        "learned_hand_action_size": module.HAND_ACTION_SIZE,
        "composed_action_size": module.COMPOSED_ACTION_SIZE,
        "hand_synergy_names": list(module.HAND_SYNERGY_NAMES),
        "frozen_locomotion": action_term.teacher.identity,
        "frozen_reach": action_term.frozen_reach.identity,
        "contact_approach_executor": {
          "protocol": "hear-contact-pocket-dls-executor-v1",
          "runtime": "deterministic_no_gradient",
          "activation": "typed_contact_alignment_or_pocket_only",
          "controlled_action_dimensions": 7,
          "authority": "active_arm_only",
          "hand_learner_wrist_authority": False,
          "waist_authority": False,
          "locomotion_authority": False,
          "checkpoint_mutation_authority": False,
          "contact_force_pose_hold_threshold_n": 2.0,
          "contact_force_stop_n": action_term.cfg.contact_force_stop_n,
          "hand_max_closing_joint_lead_rad": (
            action_term.cfg.hand_max_closing_joint_lead_rad
          ),
          "finger_closure_coordination": (
            action_term.cfg.finger_preshape_coordination
          ),
          "thumb_opposition_teacher_target": 0.0,
          "thumb_flexion_after_finger_contact_target": 0.95,
          "finger_preshape_ready_coordination": (
            command.cfg.contact_preshape_ready_coordination
          ),
          "contact_closure_planar_tolerance_m": (
            command.cfg.contact_closure_tolerance_m
          ),
          "contact_closure_vertical_tolerance_m": (
            command.cfg.contact_closure_vertical_tolerance_m
          ),
          "contact_retreat_tolerance_m": command.cfg.contact_retreat_tolerance_m,
          "contact_alignment_max_force_n": (
            command.cfg.contact_alignment_max_force_n
          ),
          "task_space_feedback_gain": (
            action_term.reach_teacher.task_space_feedback_gain
          ),
          "posture_attractor_gain": (
            action_term.reach_teacher.posture_attractor_gain
          ),
          "wrist_bearing_feedback_gain": (
            action_term.reach_teacher.wrist_bearing_feedback_gain
          ),
          "wrist_axis_alignment_feedback_gain": (
            action_term.reach_teacher.wrist_axis_alignment_feedback_gain
          ),
          "rod_axis_alignment": "unsigned_parallel_or_antiparallel",
          "orientation_joint_scope": "active_wrist_roll_pitch_yaw_only",
          "pocket_translation_joint_scope": "active_shoulder_elbow_only",
          "pocket_wrist_target": (
            "terminal_dls_until_measured_closure_gate_pose_hold"
          ),
          "closure_pose_hold": (
            "first_measured_closure_gate_pose_then_physical_contact_pose"
          ),
          "closure_authority_latch": (
            "episode_local_harness_state_after_first_geometry_gate"
          ),
          "contact_base_assist": {
            "controller": "frozen_locomotion_teacher",
            "enabled": command.cfg.contact_base_assist_enabled,
            "learner_authority": False,
            "gain_s_inv": command.cfg.contact_base_assist_gain_s_inv,
            "minimum_speed_m_s": (
              command.cfg.contact_base_assist_min_speed_m_s
            ),
            "maximum_speed_m_s": command.cfg.contact_base_assist_max_speed_m_s,
            "direction_frame": "world_xy_latched_at_pocket_entry",
          },
          "wrist_bearing_task_weight_m_per_rad": (
            action_term.reach_teacher.wrist_bearing_task_weight_m_per_rad
          ),
          "max_wrist_bearing_step_rad": (
            action_term.reach_teacher.max_wrist_bearing_step_rad
          ),
          "max_joint_correction_rad": (
            action_term.reach_teacher.max_joint_correction_rad
          ),
          "max_solver_target_slew_rad": (
            action_term.reach_teacher.max_solver_target_slew_rad
          ),
          "max_command_lead_rad": action_term.reach_teacher.max_command_lead_rad,
          "hold_enter_error_m": action_term.reach_teacher.hold_enter_error_m,
          "hold_release_error_m": action_term.reach_teacher.hold_release_error_m,
        },
      },
      "evaluation": {
        "environment_count": args.num_envs,
        "control_step_limit": args.steps,
        "completed_control_steps": completed_control_steps,
        "termination_count": termination_count,
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
        "opposing_contact_count_by_active_hand": {
          side: int(
            (opposing_seen & (initial_active_hand == index)).sum().item()
          )
          for index, side in enumerate(("left", "right"))
        },
        "reached_contact_count": int(reached_contact.sum().item()),
        "reached_grasp_count": int(reached_grasp.sum().item()),
        "object_lost_count": int(object_lost.sum().item()),
        "fall_count": int(fell.sum().item()),
        "non_foot_ground_count": int(non_foot_ground.sum().item()),
        "minimum_wrist_error_mean_m": float(minimum_wrist_error.mean().item()),
        "maximum_active_hand_force_n": float(maximum_force.max().item()),
        "maximum_active_contact_surfaces": float(maximum_surfaces.max().item()),
        "opposing_contact_count": int(opposing_seen.sum().item()),
        "minimum_opposing_normal_dot": minimum_opposing_normal_dot,
        "active_hand_contact_primaries_seen": {
          side: [
            name
            for slot, name in enumerate(
              name
              for name in hand_contact_primary_names[index]
              for _ in range(hand_contact_sensors[index].cfg.num_slots)
            )
            if bool(primary_contact_seen[index][slot].item())
          ]
          for index, side in enumerate(("left", "right"))
        },
        "active_hand_contact_samples": {
          side: {
            "sample_count": int(contact_sample_count[index].item()),
            "mean_rod_position_wrist_frame_m": (
              (contact_rod_local_sum[index] / contact_sample_count[index])
              .cpu().tolist()
              if int(contact_sample_count[index].item()) > 0 else None
            ),
            "minimum_rod_position_wrist_frame_m": (
              contact_rod_local_min[index].cpu().tolist()
              if int(contact_sample_count[index].item()) > 0 else None
            ),
            "maximum_rod_position_wrist_frame_m": (
              contact_rod_local_max[index].cpu().tolist()
              if int(contact_sample_count[index].item()) > 0 else None
            ),
            "mean_coordination": (
              (contact_coordination_sum[index] / contact_sample_count[index])
              .cpu().tolist()
              if int(contact_sample_count[index].item()) > 0 else None
            ),
          }
          for index, side in enumerate(("left", "right"))
        },
        "first_contact_step_min": int(
          first_contact_step[first_contact_step >= 0].min().item()
        ) if bool((first_contact_step >= 0).any().item()) else None,
        "first_contact_step_max": int(
          first_contact_step[first_contact_step >= 0].max().item()
        ) if bool((first_contact_step >= 0).any().item()) else None,
        "first_grasp_step_min": int(
          first_grasp_step[first_grasp_step >= 0].min().item()
        ) if bool((first_grasp_step >= 0).any().item()) else None,
        "first_grasp_step_max": int(
          first_grasp_step[first_grasp_step >= 0].max().item()
        ) if bool((first_grasp_step >= 0).any().item()) else None,
        "contact_pocket_diagnostics": {
          "retreat_active_environment_count": int(
            contact_retreat_seen.sum().item()
          ),
          "retreat_active_env_steps": contact_retreat_env_steps,
          "retreat_object_contact_environment_count": int(
            contact_retreat_object_contact.sum().item()
          ),
          "maximum_retreat_object_force_n": maximum_contact_retreat_object_force,
          "alignment_active_environment_count": int(
            contact_alignment_seen.sum().item()
          ),
          "alignment_active_env_steps": contact_alignment_env_steps,
          "alignment_object_contact_environment_count": int(
            contact_alignment_object_contact.sum().item()
          ),
          "maximum_alignment_object_force_n": (
            maximum_contact_alignment_object_force
          ),
          "alignment_samples": {
            side: {
              "sample_count": int(alignment_sample_count[index].item()),
              "mean_rod_position_wrist_frame_m": (
                (alignment_rod_local_sum[index] / alignment_sample_count[index])
                .cpu().tolist()
                if int(alignment_sample_count[index].item()) > 0 else None
              ),
              "minimum_rod_position_wrist_frame_m": (
                alignment_rod_local_min[index].cpu().tolist()
                if int(alignment_sample_count[index].item()) > 0 else None
              ),
              "maximum_rod_position_wrist_frame_m": (
                alignment_rod_local_max[index].cpu().tolist()
                if int(alignment_sample_count[index].item()) > 0 else None
              ),
              "mean_target_error_m": (
                float((
                  alignment_target_error_sum[index]
                  / alignment_sample_count[index]
                ).item())
                if int(alignment_sample_count[index].item()) > 0 else None
              ),
              "minimum_target_error_m": (
                float(alignment_target_error_min[index].item())
                if int(alignment_sample_count[index].item()) > 0 else None
              ),
              "maximum_target_error_m": (
                float(alignment_target_error_max[index].item())
                if int(alignment_sample_count[index].item()) > 0 else None
              ),
              "minimum_absolute_wrist_bearing_error_rad": (
                float(alignment_bearing_absolute_error_min[index].item())
                if int(alignment_sample_count[index].item()) > 0 else None
              ),
              "maximum_absolute_wrist_bearing_error_rad": (
                float(alignment_bearing_absolute_error_max[index].item())
                if int(alignment_sample_count[index].item()) > 0 else None
              ),
              "minimum_wrist_axis_alignment_error_rad": (
                float(alignment_axis_error_min[index].item())
                if int(alignment_sample_count[index].item()) > 0 else None
              ),
              "maximum_wrist_axis_alignment_error_rad": (
                float(alignment_axis_error_max[index].item())
                if int(alignment_sample_count[index].item()) > 0 else None
              ),
            }
            for index, side in enumerate(("left", "right"))
          },
          "active_environment_count": int(contact_pocket_seen.sum().item()),
          "active_env_steps": contact_pocket_env_steps,
          "first_active_step_min": int(
            first_contact_pocket_step[first_contact_pocket_step >= 0].min().item()
          ) if bool((first_contact_pocket_step >= 0).any().item()) else None,
          "first_active_step_max": int(
            first_contact_pocket_step[first_contact_pocket_step >= 0].max().item()
          ) if bool((first_contact_pocket_step >= 0).any().item()) else None,
          "hand_closure_ready_environment_count": int(
            hand_closure_ready_seen.sum().item()
          ),
          "hand_closure_ready_env_steps": hand_closure_ready_env_steps,
          "first_hand_closure_ready_step_min": int(
            first_hand_closure_ready_step[
              first_hand_closure_ready_step >= 0
            ].min().item()
          ) if bool((first_hand_closure_ready_step >= 0).any().item()) else None,
          "first_hand_closure_ready_step_max": int(
            first_hand_closure_ready_step[
              first_hand_closure_ready_step >= 0
            ].max().item()
          ) if bool((first_hand_closure_ready_step >= 0).any().item()) else None,
          "preshape_ready_environment_count": int(preshape_ready_seen.sum().item()),
          "preshape_ready_env_steps": preshape_ready_env_steps,
          "pocket_and_closure_ready_environment_count": int(
            pocket_closure_ready_seen.sum().item()
          ),
          "pocket_and_closure_ready_env_steps": pocket_closure_ready_env_steps,
          "samples": {
            side: {
              "sample_count": int(pocket_sample_count[index].item()),
              "mean_rod_position_wrist_frame_m": (
                (pocket_rod_local_sum[index] / pocket_sample_count[index])
                .cpu().tolist()
                if int(pocket_sample_count[index].item()) > 0 else None
              ),
              "minimum_rod_position_wrist_frame_m": (
                pocket_rod_local_min[index].cpu().tolist()
                if int(pocket_sample_count[index].item()) > 0 else None
              ),
              "maximum_rod_position_wrist_frame_m": (
                pocket_rod_local_max[index].cpu().tolist()
                if int(pocket_sample_count[index].item()) > 0 else None
              ),
              "mean_wrist_position_delta_pelvis_frame_m": (
                (pocket_wrist_delta_sum[index] / pocket_sample_count[index])
                .cpu().tolist()
                if int(pocket_sample_count[index].item()) > 0 else None
              ),
              "minimum_wrist_position_delta_pelvis_frame_m": (
                pocket_wrist_delta_min[index].cpu().tolist()
                if int(pocket_sample_count[index].item()) > 0 else None
              ),
              "maximum_wrist_position_delta_pelvis_frame_m": (
                pocket_wrist_delta_max[index].cpu().tolist()
                if int(pocket_sample_count[index].item()) > 0 else None
              ),
              "mean_wrist_position_error_m": (
                float(
                  (pocket_wrist_error_sum[index] / pocket_sample_count[index]).item()
                ) if int(pocket_sample_count[index].item()) > 0 else None
              ),
              "minimum_wrist_position_error_m": (
                float(pocket_wrist_error_min[index].item())
                if int(pocket_sample_count[index].item()) > 0 else None
              ),
              "maximum_wrist_position_error_m": (
                float(pocket_wrist_error_max[index].item())
                if int(pocket_sample_count[index].item()) > 0 else None
              ),
              "mean_absolute_wrist_bearing_error_rad": (
                float((
                  pocket_wrist_bearing_absolute_error_sum[index]
                  / pocket_sample_count[index]
                ).item())
                if int(pocket_sample_count[index].item()) > 0 else None
              ),
              "minimum_absolute_wrist_bearing_error_rad": (
                float(pocket_wrist_bearing_absolute_error_min[index].item())
                if int(pocket_sample_count[index].item()) > 0 else None
              ),
              "maximum_absolute_wrist_bearing_error_rad": (
                float(pocket_wrist_bearing_absolute_error_max[index].item())
                if int(pocket_sample_count[index].item()) > 0 else None
              ),
              "mean_coordination": (
                (pocket_coordination_sum[index] / pocket_sample_count[index])
                .cpu().tolist()
                if int(pocket_sample_count[index].item()) > 0 else None
              ),
              "minimum_coordination": (
                pocket_coordination_min[index].cpu().tolist()
                if int(pocket_sample_count[index].item()) > 0 else None
              ),
              "maximum_coordination": (
                pocket_coordination_max[index].cpu().tolist()
                if int(pocket_sample_count[index].item()) > 0 else None
              ),
              "hand_joint_names": list(
                module.base.HAND_JOINT_NAMES[index * 7:(index + 1) * 7]
              ),
              "mean_hand_joint_position_rad": (
                (pocket_joint_position_sum[index] / pocket_sample_count[index])
                .cpu().tolist()
                if int(pocket_sample_count[index].item()) > 0 else None
              ),
              "minimum_hand_joint_position_rad": (
                pocket_joint_position_min[index].cpu().tolist()
                if int(pocket_sample_count[index].item()) > 0 else None
              ),
              "maximum_hand_joint_position_rad": (
                pocket_joint_position_max[index].cpu().tolist()
                if int(pocket_sample_count[index].item()) > 0 else None
              ),
              "mean_hand_joint_target_rad": (
                (pocket_joint_target_sum[index] / pocket_sample_count[index])
                .cpu().tolist()
                if int(pocket_sample_count[index].item()) > 0 else None
              ),
              "minimum_hand_joint_target_rad": (
                pocket_joint_target_min[index].cpu().tolist()
                if int(pocket_sample_count[index].item()) > 0 else None
              ),
              "maximum_hand_joint_target_rad": (
                pocket_joint_target_max[index].cpu().tolist()
                if int(pocket_sample_count[index].item()) > 0 else None
              ),
              "mean_absolute_hand_joint_tracking_error_rad": (
                (
                  pocket_joint_absolute_error_sum[index]
                  / pocket_sample_count[index]
                ).cpu().tolist()
                if int(pocket_sample_count[index].item()) > 0 else None
              ),
              "maximum_absolute_hand_joint_tracking_error_rad": (
                pocket_joint_absolute_error_max[index].cpu().tolist()
                if int(pocket_sample_count[index].item()) > 0 else None
              ),
              "upper_joint_names": list(
                module.reach.UPPER_BODY_JOINT_NAMES[index * 7:(index + 1) * 7]
              ),
              "mean_upper_joint_position_rad": (
                (pocket_upper_joint_position_sum[index] / pocket_sample_count[index])
                .cpu().tolist()
                if int(pocket_sample_count[index].item()) > 0 else None
              ),
              "mean_upper_joint_target_rad": (
                (pocket_upper_joint_target_sum[index] / pocket_sample_count[index])
                .cpu().tolist()
                if int(pocket_sample_count[index].item()) > 0 else None
              ),
              "maximum_absolute_upper_joint_tracking_error_rad": (
                pocket_upper_joint_absolute_error_max[index].cpu().tolist()
                if int(pocket_sample_count[index].item()) > 0 else None
              ),
              "maximum_absolute_reach_unclamped_action": (
                pocket_reach_unclamped_action_absolute_max[index].cpu().tolist()
                if int(pocket_sample_count[index].item()) > 0 else None
              ),
              "reach_authority_saturation_joint_names": (
                [
                  name
                  for joint, name in enumerate(
                    module.reach.UPPER_BODY_JOINT_NAMES[
                      index * 7:(index + 1) * 7
                    ]
                  )
                  if bool(
                    pocket_reach_authority_saturation_seen[index, joint].item()
                  )
                ]
                if int(pocket_sample_count[index].item()) > 0 else None
              ),
              "reach_soft_limit_saturation_joint_names": (
                [
                  name
                  for joint, name in enumerate(
                    module.reach.UPPER_BODY_JOINT_NAMES[
                      index * 7:(index + 1) * 7
                    ]
                  )
                  if bool(
                    pocket_reach_soft_limit_saturation_seen[index, joint].item()
                  )
                ]
                if int(pocket_sample_count[index].item()) > 0 else None
              ),
              "reach_command_lead_saturation_joint_names": (
                [
                  name
                  for joint, name in enumerate(
                    module.reach.UPPER_BODY_JOINT_NAMES[
                      index * 7:(index + 1) * 7
                    ]
                  )
                  if bool(
                    pocket_reach_command_lead_saturation_seen[index, joint].item()
                  )
                ]
                if int(pocket_sample_count[index].item()) > 0 else None
              ),
              "minimum_reach_solver_singular_value": (
                float(pocket_reach_minimum_singular_value[index].item())
                if int(pocket_sample_count[index].item()) > 0 else None
              ),
            }
            for index, side in enumerate(("left", "right"))
          },
        },
        "mean_reward_sum": float(reward_sum.mean().item()),
        "maximum_absolute_hand_action": maximum_absolute_hand_action,
        "maximum_absolute_reach_action": maximum_absolute_reach_action,
        "maximum_absolute_frozen_reach_action": (
          maximum_absolute_frozen_reach_action
        ),
        "contact_approach_correction_environment_count": int(
          contact_approach_correction_seen.sum().item()
        ),
        "contact_approach_correction_env_steps": (
          contact_approach_correction_env_steps
        ),
        "maximum_absolute_contact_approach_correction": (
          maximum_absolute_contact_approach_correction
        ),
        "maximum_inactive_arm_contact_approach_correction": (
          maximum_inactive_arm_contact_approach_correction
        ),
        "maximum_outside_pocket_contact_approach_correction": (
          maximum_outside_pocket_contact_approach_correction
        ),
        "teacher_thumb_contact_latch_environment_count": int(
          teacher_thumb_contact_latch_seen.sum().item()
        ),
        "teacher_opposing_contact_latch_environment_count": int(
          teacher_opposing_contact_latch_seen.sum().item()
        ),
        "contact_pose_hold_environment_count": int(
          contact_pose_hold_seen.sum().item()
        ),
        "contact_pose_hold_env_steps": contact_pose_hold_env_steps,
        "closure_pose_hold_environment_count": int(
          closure_pose_hold_seen.sum().item()
        ),
        "closure_pose_hold_env_steps": closure_pose_hold_env_steps,
        "contact_base_assist_environment_count": int(
          contact_base_assist_seen.sum().item()
        ),
        "contact_base_assist_env_steps": contact_base_assist_env_steps,
        "maximum_contact_base_assist_speed_m_s": (
          maximum_contact_base_assist_speed
        ),
        "maximum_contact_base_assist_root_displacement_m": float(
          contact_base_assist_maximum_root_displacement.max().item()
        ),
        "mean_contact_base_assist_maximum_forward_progress_m": float(
          contact_base_assist_maximum_forward_progress[
            contact_base_assist_seen
          ].mean().item()
        ) if bool(contact_base_assist_seen.any().item()) else 0.0,
        "minimum_contact_base_assist_maximum_forward_progress_m": float(
          contact_base_assist_maximum_forward_progress[
            contact_base_assist_seen
          ].min().item()
        ) if bool(contact_base_assist_seen.any().item()) else 0.0,
        "maximum_contact_base_assist_lateral_drift_m": float(
          contact_base_assist_maximum_lateral_drift.max().item()
        ),
        "mean_contact_base_assist_actual_speed_m_s": (
          contact_base_assist_actual_speed_sum
          / contact_base_assist_motion_sample_count
          if contact_base_assist_motion_sample_count > 0 else 0.0
        ),
        "maximum_contact_base_assist_actual_speed_m_s": (
          maximum_contact_base_assist_actual_speed
        ),
        "mean_contact_base_assist_command_projection_m_s": (
          contact_base_assist_command_projection_sum
          / contact_base_assist_motion_sample_count
          if contact_base_assist_motion_sample_count > 0 else 0.0
        ),
        "maximum_unauthorized_hand_action": maximum_unauthorized_hand_action,
        "maximum_inactive_hand_coordination": maximum_inactive_coordination,
        "maximum_precontact_hand_target_rad": maximum_precontact_hand_target,
        "precontact_object_contact_count": int(
          precontact_object_contact.sum().item()
        ),
        "maximum_precontact_object_force_n": maximum_precontact_object_force,
        "approach_safety_intervention_environment_count": int(
          approach_safety_intervention_seen.sum().item()
        ),
        "approach_safety_intervention_env_steps": approach_safety_intervention_steps,
        "precontact_safety_intervention_env_steps": precontact_safety_intervention_steps,
        "maximum_active_hand_coordination": maximum_active_hand_coordination,
        "finite": True,
        "environment_closed": False,
      },
    }
  finally:
    env.close()
    environment_closed = True

  payload["evaluation"]["environment_closed"] = environment_closed
  evaluation = payload["evaluation"]
  primaries = evaluation["active_hand_contact_primaries_seen"]
  thresholds = {
    "environment_count_minimum": 16,
    "success_rate_minimum": 0.50,
    "success_count_per_active_hand_minimum": 1,
    "maximum_active_hand_force_n": 30.0,
    "minimum_opposing_normal_dot_maximum": -0.15,
  }
  checks = {
    "environment_count": evaluation["environment_count"]
      >= thresholds["environment_count_minimum"],
    "success_rate": evaluation["success_rate"]
      >= thresholds["success_rate_minimum"],
    "both_active_hands_succeeded": all(
      count >= thresholds["success_count_per_active_hand_minimum"]
      for count in evaluation["success_count_by_active_hand"].values()
    ),
    "success_is_verified_opposition": (
      evaluation["opposing_contact_count"] >= evaluation["success_count"] > 0
    ),
    "opposing_normals": evaluation["minimum_opposing_normal_dot"]
      <= thresholds["minimum_opposing_normal_dot_maximum"],
    "both_hands_exposed_thumb_and_finger_surfaces": all(
      any("_thumb_" in name for name in primaries[side])
      and any("_thumb_" not in name for name in primaries[side])
      for side in ("left", "right")
    ),
    "contact_force": evaluation["maximum_active_hand_force_n"]
      <= thresholds["maximum_active_hand_force_n"],
    "no_object_loss": evaluation["object_lost_count"] == 0,
    "no_fall": evaluation["fall_count"] == 0,
    "no_non_foot_ground_contact": evaluation["non_foot_ground_count"] == 0,
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
      and evaluation["maximum_outside_pocket_contact_approach_correction"] == 0.0
    ),
    "frozen_gradient_partition": (
      action_term.teacher.gradient_parameter_count == 0
      and action_term.frozen_reach.gradient_parameter_count == 0
    ),
    "finite_and_closed": evaluation["finite"] and environment_closed,
  }
  payload["gate"] = {
    "protocol": "hear-workyard-contact-analytic-teacher-preflight-v1",
    "thresholds": thresholds,
    "checks": checks,
    "passed": all(checks.values()),
  }
  encoded = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
  if args.output is not None:
    output = args.output.resolve()
    if output.exists():
      raise FileExistsError(f"Contact teacher output already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(encoded, encoding="utf-8")
  print(encoded, end="")
  if args.enforce_gate and not payload["gate"]["passed"]:
    raise SystemExit(2)


if __name__ == "__main__":
  main()
