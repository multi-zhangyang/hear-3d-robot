"""Serialize cross-runtime Workyard observation fixtures from live MJLab state.

The fixture contains physical quantities, command history and the exact Python
231D/247D tensors.  It deliberately contains no teacher-private curriculum
state and grants no execution authority; TypeScript rebuilds the observations
from this payload to prove the deployment contract before formal training.
"""

from __future__ import annotations

from typing import Any

import torch


PROTOCOL = "hear-workyard-observation-parity-fixture-v1"
OBJECT_ID = "assembly_rod"
TARGET_ZONE_ID = "assembly_bay"


def capture_workyard_observation_fixture(
  module,
  raw_env,
  actor_observation: torch.Tensor,
  environment_index: int,
  control_step: int,
  milestone: str,
) -> dict[str, Any]:
  action = module._contact_action(raw_env)
  command = module.base._workyard_command(raw_env)
  robot = action._entity
  index = int(environment_index)
  side_index = int(command.active_hand[index].item())
  active_hand = ("left", "right")[side_index]
  body_positions = robot.data.joint_pos[index, action._body_ids]
  body_velocities = robot.data.joint_vel[index, action._body_ids]
  hand_positions = robot.data.joint_pos[index, action._hand_ids]
  hand_velocities = robot.data.joint_vel[index, action._hand_ids]
  root_position = robot.data.root_link_pos_w[index]
  root_quaternion = robot.data.root_link_quat_w[index]
  root_linear_velocity = robot.data.root_link_lin_vel_b[index]
  root_angular_velocity = robot.data.root_link_ang_vel_b[index]

  end_effector_names = (
    "left_wrist_yaw_link",
    "right_wrist_yaw_link",
    "left_ankle_roll_link",
    "right_ankle_roll_link",
    "torso_link",
  )
  end_effector_ids, resolved_names = robot.find_bodies(
    end_effector_names, preserve_order=True
  )
  if tuple(resolved_names) != end_effector_names:
    raise RuntimeError("Parity fixture cannot resolve Workyard end effectors")
  poses = robot.data.body_link_pose_w[index, end_effector_ids]
  linear_velocities = robot.data.body_link_lin_vel_w[index, end_effector_ids]
  angular_velocities = robot.data.body_link_ang_vel_w[index, end_effector_ids]
  end_effectors = {
    name: {
      "position": _app_vector(poses[offset, :3]),
      "rotation": _app_quaternion(poses[offset, 3:7]),
      "linearVelocity": _app_vector(linear_velocities[offset]),
      "angularVelocity": _app_vector(angular_velocities[offset]),
    }
    for offset, name in enumerate(end_effector_names)
  }

  feet_found, feet_force, _ = module.base._foot_contact_summary(raw_env)
  contacts = _hand_contacts(module, raw_env, index, command)
  rod_pose = command.rod.data.root_link_pose_w[index]
  rod_velocity = command.rod.data.root_link_vel_w[index]
  target_position = command.target_position[index]
  com_position = robot.data.data.subtree_com[
    index, robot.indexing.root_body_id
  ]
  com_velocity = robot.data.data.cvel[
    index, robot.indexing.root_body_id, 3:6
  ]
  wrist_targets = command.wrist_targets_pelvis[index].reshape(2, 7)
  wrist_tolerances = command.wrist_tolerances[index]
  requested_capabilities = [
    name
    for capability_index, name in enumerate(module.base.POLICY_CAPABILITIES)
    if float(command.requested_capabilities[index, capability_index].item()) > 0.5
  ]
  grasp_requirements = command.grasp_requirements[index]
  reach_observation = action._reach_observation_builder(raw_env)[index]
  coordination = action.coordination[index].reshape(2, 4)
  if reach_observation.shape != (module.REACH_OBSERVATION_SIZE,):
    raise RuntimeError("Parity fixture reach observation drifted")
  if actor_observation[index].shape != (module.HAND_OBSERVATION_SIZE,):
    raise RuntimeError("Parity fixture contact observation drifted")

  maximum_steps = int(raw_env.max_episode_length)
  step_index = int(raw_env.episode_length_buf[index].item())
  task = {
    "window": {
      "maximumSteps": maximum_steps,
      "stepIndex": step_index,
    },
    "requestedCapabilities": requested_capabilities,
    "command": {
      "baseTwist": {
        "forwardMetersPerSecond": float(
          command.desired_base_twist[index, 0].item()
        ),
        "lateralMetersPerSecond": float(
          command.desired_base_twist[index, 1].item()
        ),
        "yawRadiansPerSecond": float(
          command.desired_base_twist[index, 2].item()
        ),
      },
      "endEffectors": [
        {
          "body": f"{side}_wrist_yaw_link",
          "frame": "pelvis",
          "position": _app_vector(wrist_targets[side_offset, :3]),
          "orientation": _app_quaternion(wrist_targets[side_offset, 3:7]),
          "tolerance": float(wrist_tolerances[side_offset].item()),
          "orientationTolerance": float(
            command.cfg.contact_alignment_axis_tolerance_rad
          ),
        }
        for side_offset, side in enumerate(("left", "right"))
      ],
      "grasps": [{
        "objectId": OBJECT_ID,
        "hand": active_hand,
        "minimumNormalForceN": float(grasp_requirements[0].item()),
        "minimumDistinctContactSurfaces": int(grasp_requirements[1].item()),
      }],
    },
  }
  state = {
    "jointPositions": _list(body_positions),
    "jointVelocities": _list(body_velocities),
    "rootQuaternion": _list(root_quaternion),
    "rootAngularVelocity": _list(root_angular_velocity),
    "environment": {
      "protocol": "humanoid-policy-environment-v1",
      "authority": "mujoco_state",
      "rootVelocityFrame": "pelvis_imu",
      "rootLinearVelocity": _list(root_linear_velocity),
      "rootAngularVelocity": _list(root_angular_velocity),
      "rootPosition": _app_vector(root_position),
      "endEffectors": end_effectors,
      "hands": {
        name: {
          "position": float(hand_positions[joint_index].item()),
          "velocity": float(hand_velocities[joint_index].item()),
          "target": float(action._hand_targets[index, joint_index].item()),
        }
        for joint_index, name in enumerate(module.base.HAND_JOINT_NAMES)
      },
      "contacts": contacts,
      "objects": [{
        "id": OBJECT_ID,
        "position": _app_vector(rod_pose[:3]),
        "rotation": _app_quaternion(rod_pose[3:7]),
        "linearVelocity": _app_vector(rod_velocity[:3]),
        "angularVelocity": _app_vector(rod_velocity[3:6]),
      }],
      "zones": [{
        "id": TARGET_ZONE_ID,
        "center": _app_vector(target_position),
        "size": {"x": 1.4, "y": 0.29, "z": 1.4},
      }],
      "feet": {
        "left": {
          "touching": bool(feet_found[index, 0].item()),
          "normalForce": float(feet_force[index, 0].item()),
        },
        "right": {
          "touching": bool(feet_found[index, 1].item()),
          "normalForce": float(feet_force[index, 1].item()),
        },
      },
      "centerOfMass": _app_vector(com_position),
      "centerOfMassVelocity": _app_vector(com_velocity),
    },
  }
  return {
    "protocol": PROTOCOL,
    "episode_seed": int(
      command.cfg.evaluation_episode_seed_base + index
    ),
    "active_hand": active_hand,
    "control_step": int(control_step),
    "milestone": milestone,
    "input": {
      "state": state,
      "options": {"taskCommand": task},
      "previousTeacherAction": _list(action.teacher_action[index]),
      "previousReachAction": _list(action.raw_action[index]),
      "coordination": {
        side: {
          "thumb_opposition": float(coordination[side_offset, 0].item()),
          "thumb_curl": float(coordination[side_offset, 1].item()),
          "index_curl": float(coordination[side_offset, 2].item()),
          "middle_curl": float(coordination[side_offset, 3].item()),
        }
        for side_offset, side in enumerate(("left", "right"))
      },
      "previousAuthorizedAction": _list(action.hand_action[index]),
    },
    "expected": {
      "reachObservation": _list(reach_observation),
      "contactObservation": _list(actor_observation[index]),
    },
  }


def _hand_contacts(module, raw_env, index: int, command) -> list[dict[str, Any]]:
  contacts: list[dict[str, Any]] = []
  rod_position = command.rod.data.root_link_pos_w[index]
  for side in ("left", "right"):
    sensor = module.base._contact_sensor(
      raw_env, f"{side}_hand_object_contact"
    )
    if sensor.data.found is None or sensor.data.force is None:
      raise RuntimeError("Parity fixture hand contact sensor is incomplete")
    expanded_names = tuple(
      name
      for name in sensor.primary_names
      for _ in range(sensor.cfg.num_slots)
    )
    found = sensor.data.found[index] > 0
    force = sensor.data.force[index]
    normals = sensor.data.normal[index] if sensor.data.normal is not None else None
    for slot, name in enumerate(expanded_names):
      if not bool(found[slot].item()):
        continue
      normal = (
        _app_vector(normals[slot])
        if normals is not None
        else {"x": 0.0, "y": 1.0, "z": 0.0}
      )
      contacts.append({
        "position": _app_vector(rod_position),
        "normal": normal,
        "normalForce": float(torch.linalg.vector_norm(force[slot]).item()),
        "firstBody": name,
        "secondBody": None,
        "firstObject": None,
        "secondObject": OBJECT_ID,
        "firstSolid": None,
        "secondSolid": None,
        "firstHandLink": name,
        "secondHandLink": None,
      })
  return contacts


def _app_vector(value: torch.Tensor) -> dict[str, float]:
  values = _list(value)
  return {"x": values[1], "y": values[2], "z": values[0]}


def _app_quaternion(value: torch.Tensor) -> dict[str, float]:
  values = _list(value)
  return {
    "x": values[2],
    "y": values[3],
    "z": values[1],
    "w": values[0],
  }


def _list(value: torch.Tensor) -> list[float]:
  return [float(item) for item in value.detach().cpu().tolist()]
