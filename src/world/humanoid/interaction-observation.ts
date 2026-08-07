import type { Scenario, Vec3 } from "../../domain/schema.js";
import {
  inverseQuaternion,
  rotateVector,
  subtract,
  vectorLength
} from "../geometry.js";
import type {
  HumanoidCarriedObjectLifecycleCheckpoint
} from "./carried-object-lifecycle.js";
import type { HumanoidGraspContract } from "./grasp-tracker.js";
import type { HumanoidWorldGraspState } from "./grasp-world-state.js";
import { G1_HAND_CONTACT_SURFACE_NAMES } from "./morphology.js";
import type { HumanoidObjectToken } from "./object-memory.js";
import { humanoidObjectZoneRelation } from "./object-zone-relation.js";
import type { HumanoidSimulationSnapshot } from "./simulation.js";
import {
  createHumanoidObjectWorldModel,
  type HumanoidObjectWorldModel
} from "./object-world-model.js";
import {
  createHumanoidSkillCatalog,
  type HumanoidSkillCatalog
} from "./skill-catalog.js";
import type { HumanoidSolidToken } from "./solid-observation.js";

type Hand = "left" | "right";

export interface HumanoidInteractionObservation {
  frame: number;
  world_revision: number;
  object_world_model: HumanoidObjectWorldModel;
  skill_catalog: HumanoidSkillCatalog;
  grasp_authority: {
    contract_sha256: string;
    minimum_distinct_contact_surfaces: number;
    minimum_lift_m: number;
    minimum_lifted_hold_frames: number;
    minimum_relative_pose_stable_frames: number;
    hand_surfaces: Record<Hand, string[]>;
  };
  carrying: {
    phase: HumanoidCarriedObjectLifecycleCheckpoint["phase"];
    bindings: Array<{
      object_id: string;
      hand: Hand;
      source_frame: number;
      source_world_revision: number;
    }>;
    continuation_verified: boolean | null;
    unauthorized_contact_count: number;
    transition_frame: number;
    transition_world_revision: number;
    transition_reason: HumanoidCarriedObjectLifecycleCheckpoint["transition_reason"];
  };
  zones: Array<{
    zone_id: string;
    center: Vec3;
    size: Vec3;
    robot_planar_distance_m: number;
    robot_inside_horizontal: boolean;
  }>;
  manipulable_objects: Array<{
    object_id: string;
    authority: "head_sensor" | "carried_contact";
    size: Vec3;
    world_position: Vec3;
    pelvis_relative_position: Vec3;
    hand_relations: Record<Hand, {
      distance_m: number;
      object_from_wrist_world: Vec3;
      object_from_wrist_pelvis: Vec3;
    }>;
    grasp: Array<{
      hand: Hand;
      phase: string;
      verified: boolean;
      reason: string;
    }>;
    zone_relations: Array<{
      zone_id: string;
      inside_without_tolerance: boolean;
      center_offset: Vec3;
      minimum_horizontal_clearance_m: number;
      support_height_error_m: number;
    }>;
  }>;
}

export function createHumanoidInteractionObservation(input: {
  frame: number;
  worldRevision: number;
  scenario: Scenario;
  robot: HumanoidSimulationSnapshot;
  objectTokens: readonly HumanoidObjectToken[];
  solidTokens?: readonly HumanoidSolidToken[];
  grasp: HumanoidWorldGraspState;
  graspContract: HumanoidGraspContract;
  carried: HumanoidCarriedObjectLifecycleCheckpoint;
}): HumanoidInteractionObservation {
  const pelvis = input.robot.links.pelvis;
  const inversePelvis = inverseQuaternion(pelvis.rotation);
  const visible = new Set(input.objectTokens.flatMap((token) => (
    token.status === "visible" && token.observable ? [token.id] : []
  )));
  const carriedByObject = new Map(
    input.carried.active_binding_set?.bindings.map((binding) => [
      binding.object_id,
      binding
    ]) ?? []
  );
  const observableObjectIds = new Set([...visible, ...carriedByObject.keys()]);
  const zones = [...input.scenario.zones]
    .sort((left, right) => compare(left.id, right.id));
  const descriptors = new Map(input.scenario.objects.map((object) => [object.id, object]));
  const manipulableObjects = [...observableObjectIds]
    .sort(compare)
    .flatMap((objectId) => {
      const descriptor = descriptors.get(objectId);
      const object = input.robot.objects[objectId];
      if (!descriptor?.portable || !object) return [];
      const hands = Object.fromEntries((["left", "right"] as const).map((hand) => {
        const wrist = input.robot.links[
          hand === "left" ? "left_wrist_yaw_link" : "right_wrist_yaw_link"
        ];
        const offset = subtract(object.position, wrist.position);
        return [hand, {
          distance_m: vectorLength(offset),
          object_from_wrist_world: offset,
          object_from_wrist_pelvis: rotateVector(inversePelvis, offset)
        }];
      })) as HumanoidInteractionObservation["manipulable_objects"][number]["hand_relations"];
      return [{
        object_id: objectId,
        authority: visible.has(objectId) ? "head_sensor" as const : "carried_contact" as const,
        size: { ...descriptor.size },
        world_position: { ...object.position },
        pelvis_relative_position: rotateVector(
          inversePelvis,
          subtract(object.position, pelvis.position)
        ),
        hand_relations: hands,
        grasp: input.grasp.assessments
          .filter((assessment) => assessment.frame === input.frame
            && assessment.object_id === objectId)
          .map((assessment) => ({
            hand: assessment.hand,
            phase: assessment.phase,
            verified: assessment.grasp_verified,
            reason: assessment.reason
          })),
        zone_relations: zones.map((zone) => {
          const relation = humanoidObjectZoneRelation({
            object: {
              position: object.position,
              rotation: object.rotation,
              size: descriptor.size
            },
            zone,
            tolerance: 0
          });
          return {
            zone_id: zone.id,
            inside_without_tolerance: relation.inside,
            center_offset: relation.centerOffset,
            minimum_horizontal_clearance_m: relation.horizontalClearance.minimum,
            support_height_error_m: relation.supportHeightError
          };
        })
      }];
    });

  const objectWorldModel = createHumanoidObjectWorldModel({
    frame: input.frame,
    worldRevision: input.worldRevision,
    scenario: input.scenario,
    robot: input.robot,
    objectTokens: input.objectTokens
  });
  return {
    frame: input.frame,
    world_revision: input.worldRevision,
    object_world_model: objectWorldModel,
    skill_catalog: createHumanoidSkillCatalog(objectWorldModel, input.solidTokens ?? []),
    grasp_authority: {
      contract_sha256: input.grasp.contractSha256,
      minimum_distinct_contact_surfaces:
        input.graspContract.minimum_distinct_contact_links,
      minimum_lift_m: input.graspContract.minimum_lift_m,
      minimum_lifted_hold_frames: input.graspContract.minimum_lifted_hold_frames,
      minimum_relative_pose_stable_frames:
        input.graspContract.minimum_relative_pose_stable_frames,
      hand_surfaces: {
        left: surfaces("left"),
        right: surfaces("right")
      }
    },
    carrying: {
      phase: input.carried.phase,
      bindings: (input.carried.active_binding_set?.bindings ?? []).map((binding) => ({
        object_id: binding.object_id,
        hand: binding.hand,
        source_frame: binding.source_frame,
        source_world_revision: binding.source_world_revision
      })),
      continuation_verified: input.carried.last_continuation?.continued ?? null,
      unauthorized_contact_count: input.carried.last_unauthorized_contacts.length,
      transition_frame: input.carried.transition_frame,
      transition_world_revision: input.carried.transition_world_revision,
      transition_reason: input.carried.transition_reason
    },
    zones: zones.map((zone) => ({
      zone_id: zone.id,
      center: { ...zone.center },
      size: { ...zone.size },
      robot_planar_distance_m: Math.hypot(
        input.robot.rootPosition.x - zone.center.x,
        input.robot.rootPosition.z - zone.center.z
      ),
      robot_inside_horizontal:
        Math.abs(input.robot.rootPosition.x - zone.center.x) <= zone.size.x / 2
        && Math.abs(input.robot.rootPosition.z - zone.center.z) <= zone.size.z / 2
    })),
    manipulable_objects: manipulableObjects
  };
}

function surfaces(hand: Hand): string[] {
  return G1_HAND_CONTACT_SURFACE_NAMES.filter((surface) => (
    surface.startsWith(`${hand}_`)
  ));
}

function compare(left: string, right: string): number {
  return left < right ? -1 : left > right ? 1 : 0;
}
