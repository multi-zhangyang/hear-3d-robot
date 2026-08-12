import { describe, expect, it } from "vitest";
import { ScenarioSchema } from "../../domain/schema.js";
import {
  DEFAULT_HUMANOID_GRASP_CONTRACT
} from "./grasp-registry.js";
import { humanoidGraspContractSha256 } from "./grasp-tracker.js";
import { createHumanoidInteractionObservation } from "./interaction-observation.js";
import type {
  HumanoidCarriedObjectLifecycleCheckpoint
} from "./carried-object-lifecycle.js";
import { G1_HAND_CONTACT_SURFACE_NAMES } from "./morphology.js";
import type { HumanoidObjectToken } from "./object-memory.js";
import type { HumanoidSimulationSnapshot } from "./simulation.js";

const scenario = ScenarioSchema.parse({
  title: "Interaction geometry",
  seed: 3,
  bounds: { width: 8, depth: 8 },
  visibility_radius: 5,
  robot: { x: 2, z: 2, yaw: 0 },
  obstacles: [],
  objects: [{
    id: "parcel",
    kind: "parcel",
    color: "#b37a48",
    position: { x: 2.4, y: 0.2, z: 2.3 },
    size: { x: 0.3, y: 0.4, z: 0.2 },
    portable: true
  }],
  zones: [{
    id: "delivery",
    color: "#56c69a",
    center: { x: 4, y: 0, z: 4 },
    size: { x: 1.5, y: 0.04, z: 1.5 }
  }],
  default_goal: {
    summary: "放置包裹",
    predicates: [{
      type: "object_placed",
      object_id: "parcel",
      zone_id: "delivery",
      tolerance: 0.05
    }]
  }
});

describe("humanoid interaction observation", () => {
  it("publishes factual pelvis, wrist, zone and grasp-authority geometry", () => {
    const observation = createHumanoidInteractionObservation({
      frame: 12,
      worldRevision: 12,
      scenario,
      robot: robotSnapshot(),
      objectTokens: [objectToken("visible")],
      grasp: {
        contractSha256: humanoidGraspContractSha256(DEFAULT_HUMANOID_GRASP_CONTRACT),
        assessments: []
      },
      graspContract: DEFAULT_HUMANOID_GRASP_CONTRACT,
      carried: idleLifecycle()
    });

    expect(observation.grasp_authority).toMatchObject({
      minimum_distinct_contact_surfaces: 2,
      minimum_lift_m: 0.04,
      minimum_lifted_hold_frames: 8
    });
    expect(observation.grasp_authority.hand_surfaces.left.length).toBeGreaterThan(2);
    expect(observation.grasp_authority.hand_surfaces.left.every((surface) => (
      surface.startsWith("left_")
    ))).toBe(true);
    expect(observation.zones).toEqual([expect.objectContaining({
      zone_id: "delivery",
      robot_planar_distance_m: Math.hypot(2, 2),
      robot_inside_horizontal: false
    })]);
    expect(observation.manipulable_objects).toHaveLength(1);
    const object = observation.manipulable_objects[0]!;
    expect(object).toMatchObject({
      object_id: "parcel",
      authority: "head_sensor",
      zone_relations: [expect.objectContaining({
        zone_id: "delivery",
        inside_without_tolerance: false
      })]
    });
    expect(object.pelvis_relative_position.x).toBeCloseTo(0.4);
    expect(object.pelvis_relative_position.y).toBeCloseTo(-0.6);
    expect(object.pelvis_relative_position.z).toBeCloseTo(0.3);
    expect(object.hand_relations.left.object_from_wrist_world.x).toBeCloseTo(0.2);
    expect(object.hand_relations.left.object_from_wrist_world.y).toBeCloseTo(-0.5);
    expect(object.hand_relations.left.object_from_wrist_world.z).toBeCloseTo(0.2);
  });

  it("does not turn remembered sensor history into current interaction geometry", () => {
    const observation = createHumanoidInteractionObservation({
      frame: 12,
      worldRevision: 12,
      scenario,
      robot: robotSnapshot(),
      objectTokens: [objectToken("remembered")],
      grasp: {
        contractSha256: humanoidGraspContractSha256(DEFAULT_HUMANOID_GRASP_CONTRACT),
        assessments: []
      },
      graspContract: DEFAULT_HUMANOID_GRASP_CONTRACT,
      carried: idleLifecycle()
    });

    expect(observation.manipulable_objects).toEqual([]);
  });

  it("keeps a physically carried object observable outside the head sensor", () => {
    const carried = carriedLifecycle();
    const observation = createHumanoidInteractionObservation({
      frame: 12,
      worldRevision: 12,
      scenario,
      robot: robotSnapshot(),
      objectTokens: [objectToken("remembered")],
      grasp: {
        contractSha256: humanoidGraspContractSha256(DEFAULT_HUMANOID_GRASP_CONTRACT),
        assessments: []
      },
      graspContract: DEFAULT_HUMANOID_GRASP_CONTRACT,
      carried
    });

    expect(observation.carrying).toMatchObject({
      phase: "acquired",
      bindings: [{ object_id: "parcel", hand: "left" }],
      continuation_verified: null
    });
    expect(observation.manipulable_objects).toEqual([
      expect.objectContaining({
        object_id: "parcel",
        authority: "carried_contact"
      })
    ]);
    expect(observation.object_world_model.objects).toEqual([
      expect.objectContaining({
        id: "parcel",
        status: "visible",
        authority: "carried_contact",
        pose: expect.objectContaining({
          position: { x: 2.4, y: 0.2, z: 2.3 }
        })
      })
    ]);
  });
});

function robotSnapshot(): HumanoidSimulationSnapshot {
  return {
    rootPosition: { x: 2, y: 0.8, z: 2 },
    rootRotation: { x: 0, y: 0, z: 0, w: 1 },
    links: {
      pelvis: link({ x: 2, y: 0.8, z: 2 }),
      left_wrist_yaw_link: link({ x: 2.2, y: 0.7, z: 2.1 }),
      right_wrist_yaw_link: link({ x: 1.8, y: 0.7, z: 2.1 })
    },
    objects: {
      parcel: {
        id: "parcel",
        ...link({ x: 2.4, y: 0.2, z: 2.3 })
      }
    }
  } as unknown as HumanoidSimulationSnapshot;
}

function link(position: { x: number; y: number; z: number }) {
  return {
    position,
    rotation: { x: 0, y: 0, z: 0, w: 1 },
    linearVelocity: { x: 0, y: 0, z: 0 },
    angularVelocity: { x: 0, y: 0, z: 0 }
  };
}

function objectToken(status: "visible" | "remembered"): HumanoidObjectToken {
  return {
    id: "parcel",
    status,
    observable: status === "visible"
  } as HumanoidObjectToken;
}

function idleLifecycle() {
  return {
    protocol: "humanoid-carried-object-lifecycle-v1" as const,
    phase: "idle" as const,
    active_binding_set: null,
    last_continuation: null,
    last_unauthorized_contacts: [],
    transition_frame: 0,
    transition_world_revision: 0,
    transition_reason: "initialized" as const
  };
}

function carriedLifecycle(): HumanoidCarriedObjectLifecycleCheckpoint {
  const contractSha256 = humanoidGraspContractSha256(
    DEFAULT_HUMANOID_GRASP_CONTRACT
  );
  const registrySha256 = "a".repeat(64);
  const allowedHandSurfaces = G1_HAND_CONTACT_SURFACE_NAMES.filter((surface) => (
    surface.startsWith("left_")
  ));
  return {
    protocol: "humanoid-carried-object-lifecycle-v1",
    phase: "acquired",
    active_binding_set: {
      protocol: "humanoid-carried-object-binding-set-v1",
      source_frame: 12,
      source_world_revision: 12,
      grasp_contract_sha256: contractSha256,
      grasp_registry_checkpoint_sha256: registrySha256,
      bindings: [{
        protocol: "humanoid-carried-object-binding-v1",
        object_id: "parcel",
        hand: "left",
        grasp_contract_sha256: contractSha256,
        grasp_registry_checkpoint_sha256: registrySha256,
        grasp_assessment_sha256: "b".repeat(64),
        source_frame: 12,
        source_world_revision: 12,
        verified_contact_surfaces: allowedHandSurfaces.slice(0, 2),
        allowed_hand_surfaces: allowedHandSurfaces
      }]
    },
    last_continuation: null,
    last_unauthorized_contacts: [],
    transition_frame: 12,
    transition_world_revision: 12,
    transition_reason: "grasp_acquired"
  };
}
