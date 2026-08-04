import { describe, expect, it } from "vitest";
import {
  DEFAULT_HUMANOID_GRASP_CONTRACT,
  HumanoidGraspRegistry,
  HumanoidGraspRegistryCheckpointSchema
} from "./grasp-registry.js";
import {
  HumanoidGraspContractSchema,
  humanoidGraspContractSha256,
  type HumanoidGraspAssessment,
  type HumanoidGraspContract
} from "./grasp-tracker.js";
import type {
  HumanoidContactSnapshot,
  HumanoidSimulationSnapshot
} from "./simulation.js";
import type { HumanoidMotionOptionContract } from "./motion-option.js";

const CONTRACT = {
  protocol: "humanoid-grasp-contract-v1",
  world_up: { x: 0, y: 1, z: 0 },
  minimum_distinct_contact_links: 2,
  minimum_contact_normal_force_n: 5,
  maximum_opposing_normal_dot: -0.5,
  maximum_opposing_position_dot: -0.5,
  minimum_opposing_contact_separation_m: 0.02,
  minimum_contact_radial_distance_m: 0.005,
  maximum_relative_translation_drift_m: 0.01,
  maximum_relative_rotation_drift_rad: 0.05,
  minimum_relative_pose_stable_frames: 2,
  minimum_lift_m: 0.05,
  minimum_lifted_hold_frames: 2,
  minimum_support_normal_force_n: 2,
  minimum_support_up_dot: 0.7
} as const satisfies HumanoidGraspContract;

describe("HumanoidGraspRegistry", () => {
  it("provides a frozen, schema-valid host default contract", () => {
    expect(HumanoidGraspContractSchema.parse(DEFAULT_HUMANOID_GRASP_CONTRACT))
      .toEqual(DEFAULT_HUMANOID_GRASP_CONTRACT);
    expect(Object.isFrozen(DEFAULT_HUMANOID_GRASP_CONTRACT)).toBe(true);
    expect(Object.isFrozen(DEFAULT_HUMANOID_GRASP_CONTRACT.world_up)).toBe(true);
    expect(humanoidGraspContractSha256(DEFAULT_HUMANOID_GRASP_CONTRACT))
      .toMatch(/^[a-f0-9]{64}$/);
  });

  it("captures an initial support baseline for both hands and sorts output", () => {
    const registry = new HumanoidGraspRegistry({
      contract: CONTRACT,
      portableObjectIds: ["parcel", "crate"]
    });
    const assessments = registry.observe(0, snapshot({
      objectIds: ["fixed", "crate"],
      supported: true,
      leftContacts: true
    }));

    expect(assessments.map(({ object_id, hand }) => [object_id, hand])).toEqual([
      ["crate", "left"],
      ["crate", "right"]
    ]);
    expect(assessments.every((assessment) => (
      assessment.evidence.support.baseline_projection_m === 0.5
    ))).toBe(true);
    expect(registry.checkpoint()).toMatchObject({
      protocol: "humanoid-grasp-registry-checkpoint-v1",
      contract_sha256: humanoidGraspContractSha256(CONTRACT),
      portable_object_ids: ["crate", "parcel"],
      last_frame: 0
    });
    expect(registry.checkpoint().tracker.tracks.map((track) => (
      [track.object_id, track.hand]
    ))).toEqual([
      ["crate", "left"],
      ["crate", "right"]
    ]);
  });

  it("continuously verifies a real opposed-contact lift and hold", () => {
    const registry = registryForCrate();
    registry.observe(0, snapshot({ supported: true, leftContacts: true }));
    registry.observe(1, snapshot({ supported: true, leftContacts: true }));
    registry.observe(2, snapshot({
      objectHeight: 0.56,
      supported: false,
      leftContacts: true
    }));
    const assessments = registry.observe(3, snapshot({
      objectHeight: 0.56,
      supported: false,
      leftContacts: true
    }));

    expect(leftAssessment(assessments)).toMatchObject({
      frame: 3,
      phase: "verified",
      grasp_verified: true,
      reason: "grasp_verified",
      evidence: {
        contact: { status: "opposed" },
        support: { status: "unsupported", lift_m: expect.closeTo(0.06) },
        relative_pose: { stable_frames: 4 },
        lifted_hold_frames: 2
      }
    });
  });

  it("is idempotent within a frame and rejects backwards time", () => {
    const registry = registryForCrate();
    const first = registry.observe(4, snapshot({
      supported: true,
      leftContacts: true
    }));
    const checkpoint = registry.checkpoint();
    const repeated = registry.observe(4, snapshot({
      objectHeight: 0.8,
      supported: false,
      leftContacts: false
    }));

    expect(repeated).toEqual(first);
    expect(registry.checkpoint()).toEqual(checkpoint);
    expect(() => registry.observe(3, snapshot())).toThrow(/moved backwards/);
  });

  it("resets tracks on frame gaps and never returns stale unloaded evidence", () => {
    const registry = registryForCrate();
    registry.observe(0, snapshot({ supported: true, leftContacts: true }));
    const afterGap = registry.observe(2, snapshot({
      supported: true,
      leftContacts: true
    }));
    expect(leftAssessment(afterGap)).toMatchObject({
      frame: 2,
      grasp_verified: false,
      reason: "frame_discontinuity",
      reset_reason: "frame_discontinuity",
      evidence: { relative_pose: { stable_frames: 1 } }
    });
    expect(registry.assessmentsForFrame(0)).toEqual([]);

    expect(registry.observe(3, snapshot({ objectIds: [] }))).toEqual([]);
    expect(registry.assessmentsForFrame(2)).toEqual([]);
    const reloaded = registry.observe(4, snapshot({
      supported: true,
      leftContacts: true
    }));
    expect(leftAssessment(reloaded)).toMatchObject({
      reason: "frame_discontinuity",
      reset_reason: "frame_discontinuity"
    });
  });

  it("restores exact progress and continues from the next physical frame", () => {
    const authority = registryForCrate();
    authority.observe(0, snapshot({ supported: true, leftContacts: true }));
    authority.observe(1, snapshot({ supported: true, leftContacts: true }));
    const checkpoint = authority.checkpoint();
    const restored = new HumanoidGraspRegistry({
      contract: CONTRACT,
      portableObjectIds: ["crate"],
      checkpoint
    });

    expect(restored.checkpoint()).toEqual(checkpoint);
    restored.observe(2, snapshot({
      objectHeight: 0.56,
      supported: false,
      leftContacts: true
    }));
    const resumed = restored.observe(3, snapshot({
      objectHeight: 0.56,
      supported: false,
      leftContacts: true
    }));
    expect(leftAssessment(resumed).grasp_verified).toBe(true);
  });

  it("rejects corrupt hashes, contracts, objects, assessment keys, and frames", () => {
    const registry = registryForCrate();
    registry.observe(0, snapshot({ supported: true, leftContacts: true }));
    const checkpoint = registry.checkpoint();

    expect(() => HumanoidGraspRegistryCheckpointSchema.parse({
      ...checkpoint,
      contract_sha256: "0".repeat(64)
    })).toThrow(/contract hash does not match/);

    const unknownTrack = structuredClone(checkpoint);
    unknownTrack.tracker.tracks[0]!.object_id = "unknown";
    expect(() => HumanoidGraspRegistryCheckpointSchema.parse(unknownTrack))
      .toThrow(/unknown object/);

    const wrongFrame = structuredClone(checkpoint);
    wrongFrame.last_assessments[0]!.frame = 1;
    expect(() => HumanoidGraspRegistryCheckpointSchema.parse(wrongFrame))
      .toThrow(/not from the registry frame/);

    const duplicateAssessment = structuredClone(checkpoint);
    duplicateAssessment.last_assessments.push(
      structuredClone(duplicateAssessment.last_assessments[1]!)
    );
    expect(() => HumanoidGraspRegistryCheckpointSchema.parse(duplicateAssessment))
      .toThrow(/unique and deterministically sorted/);

    const mismatchedAssessment = structuredClone(checkpoint);
    mismatchedAssessment.portable_object_ids.push("parcel");
    mismatchedAssessment.last_assessments[0]!.object_id = "parcel";
    expect(() => HumanoidGraspRegistryCheckpointSchema.parse(mismatchedAssessment))
      .toThrow(/does not match its tracker track/);

    expect(() => new HumanoidGraspRegistry({
      contract: { ...CONTRACT, minimum_lift_m: 0.08 },
      portableObjectIds: ["crate"],
      checkpoint
    })).toThrow(/contract does not match/);
    expect(() => new HumanoidGraspRegistry({
      contract: CONTRACT,
      portableObjectIds: ["crate", "parcel"],
      checkpoint
    })).toThrow(/object set does not match/);
    expect(() => new HumanoidGraspRegistry({
      contract: CONTRACT,
      portableObjectIds: ["crate", "crate"]
    })).toThrow(/Duplicate portable object id/);
  });

  it("forks rollout state without advancing the authority registry", () => {
    const authority = registryForCrate();
    authority.observe(0, snapshot({ supported: true, leftContacts: true }));
    authority.observe(1, snapshot({ supported: true, leftContacts: true }));
    const authorityCheckpoint = authority.checkpoint();
    const rollout = authority.fork();

    rollout.observe(2, snapshot({
      objectHeight: 0.56,
      supported: false,
      leftContacts: true
    }));
    const rolloutResult = rollout.observe(3, snapshot({
      objectHeight: 0.56,
      supported: false,
      leftContacts: true
    }));

    expect(leftAssessment(rolloutResult).grasp_verified).toBe(true);
    expect(rollout.lastFrame).toBe(3);
    expect(authority.lastFrame).toBe(1);
    expect(authority.checkpoint()).toEqual(authorityCheckpoint);
  });

  it("binds only current assessments to matching authority predicates", () => {
    const registry = registryForCrate();
    registry.observe(0, snapshot({ supported: true, leftContacts: true }));
    const option = graspOption(registry.contractSha256, "crate");
    const bindings = registry.bindingsForOption(option, 0);

    expect(bindings).toMatchObject([{
      predicate_index: 1,
      contract_sha256: registry.contractSha256,
      assessment: { frame: 0, object_id: "crate", hand: "left" }
    }]);
    expect(registry.bindingsForOption(option, 1)).toEqual([]);

    expect(() => registry.bindingsForOption(
      graspOption("0".repeat(64), "crate"),
      0
    )).toThrow(/contract hash does not match authority/);
    expect(() => registry.bindingsForOption(
      graspOption(registry.contractSha256, "parcel"),
      0
    )).toThrow(/unknown portable object/);

    const duplicate = graspOption(registry.contractSha256, "crate");
    duplicate.predicates.push(structuredClone(duplicate.predicates[1]!));
    expect(() => registry.bindingsForOption(duplicate, 0))
      .toThrow(/Duplicate humanoid grasp predicate/);
  });

  it("rejects mismatched physical snapshot object identities before tracking", () => {
    const registry = registryForCrate();
    const invalid = snapshot();
    invalid.objects.crate!.id = "parcel";
    expect(() => registry.observe(0, invalid)).toThrow(/key does not match its id/);
    expect(registry.lastFrame).toBeNull();
    expect(registry.checkpoint().tracker.tracks).toEqual([]);
  });
});

function registryForCrate(): HumanoidGraspRegistry {
  return new HumanoidGraspRegistry({
    contract: CONTRACT,
    portableObjectIds: ["crate"]
  });
}

function graspOption(
  contractSha256: string,
  objectId: string
): HumanoidMotionOptionContract {
  return {
    option_id: "grasp-crate",
    stable_steps: 1,
    predicates: [{
      type: "root_near_point",
      target: { x: 0, y: 0.8, z: 0 },
      tolerance_m: 0.2
    }, {
      type: "grasp_verified",
      object_id: objectId,
      hand: "left",
      grasp_contract_sha256: contractSha256
    }]
  };
}

function leftAssessment(
  assessments: readonly HumanoidGraspAssessment[]
): HumanoidGraspAssessment {
  const assessment = assessments.find(({ hand }) => hand === "left");
  if (!assessment) throw new Error("Missing left-hand assessment");
  return assessment;
}

function snapshot(options: {
  objectIds?: readonly string[];
  objectHeight?: number;
  supported?: boolean;
  leftContacts?: boolean;
} = {}): HumanoidSimulationSnapshot {
  const objectIds = options.objectIds ?? ["crate"];
  const height = options.objectHeight ?? 0.5;
  const objects = Object.fromEntries(objectIds.map((id, index) => [id, {
    id,
    position: { x: index, y: height, z: 0 },
    rotation: quaternion(),
    linearVelocity: vector(),
    angularVelocity: vector()
  }]));
  const contacts: HumanoidContactSnapshot[] = [];
  if (objectIds.includes("crate") && options.leftContacts) {
    contacts.push(
      contact({
        position: { x: -0.03, y: height, z: 0 },
        normal: { x: 1, y: 0, z: 0 },
        normalForce: 10,
        firstHandLink: "left_hand_thumb_2_link",
        secondObject: "crate"
      }),
      contact({
        position: { x: 0.03, y: height, z: 0 },
        normal: { x: -1, y: 0, z: 0 },
        normalForce: 11,
        firstHandLink: "left_hand_index_1_link",
        secondObject: "crate"
      })
    );
  }
  if (objectIds.includes("crate") && options.supported) {
    contacts.push(contact({
      position: { x: 0, y: height - 0.03, z: 0 },
      normal: { x: 0, y: 1, z: 0 },
      normalForce: 15,
      secondObject: "crate"
    }));
  }
  const link = (x: number) => ({
    position: { x, y: height, z: 0 },
    rotation: quaternion(),
    linearVelocity: vector(),
    angularVelocity: vector()
  });
  return {
    objects,
    links: {
      left_wrist_yaw_link: link(-0.1),
      right_wrist_yaw_link: link(0.1)
    },
    contacts
  } as unknown as HumanoidSimulationSnapshot;
}

function contact(
  overrides: Partial<HumanoidContactSnapshot>
): HumanoidContactSnapshot {
  return {
    position: vector(),
    normal: { x: 0, y: 1, z: 0 },
    normalForce: 0,
    firstBody: null,
    secondBody: null,
    firstObject: null,
    secondObject: null,
    firstHandLink: null,
    secondHandLink: null,
    ...overrides
  };
}

function vector(x = 0, y = 0, z = 0): { x: number; y: number; z: number } {
  return { x, y, z };
}

function quaternion(): { x: number; y: number; z: number; w: number } {
  return { x: 0, y: 0, z: 0, w: 1 };
}
