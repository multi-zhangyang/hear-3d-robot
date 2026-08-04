import { describe, expect, it } from "vitest";
import {
  HumanoidCarriedObjectBindingSchema,
  HumanoidCarriedObjectBindingSetSchema,
  HumanoidCarriedObjectContinuationEvidenceSchema,
  admitHumanoidCarriedObjectBindings,
  humanoidCarriedObjectBindingSetSha256,
  humanoidCarriedObjectBindingSha256,
  humanoidCarriedObjectContinuationEvidence,
  humanoidCarriedObjectContactConstraints,
  humanoidCarriedObjectUnauthorizedContacts,
  humanoidGraspAssessmentSha256,
  humanoidGraspRegistryCheckpointSha256,
  verifyHumanoidCarriedObjectBindingSet
} from "./carried-object-binding.js";
import {
  HumanoidCarriedObjectLifecycle,
  HumanoidCarriedObjectLifecycleCheckpointSchema
} from "./carried-object-lifecycle.js";
import { HumanoidGraspRegistry } from "./grasp-registry.js";
import type { HumanoidGraspContract } from "./grasp-tracker.js";
import type {
  HumanoidContactSnapshot,
  HumanoidSimulationSnapshot
} from "./simulation.js";

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

describe("humanoid carried-object binding", () => {
  it("derives a canonical binding only from the current verified registry assessment", () => {
    const registry = verifiedRegistry();
    const state = admitHumanoidCarriedObjectBindings({
      registry,
      currentFrame: 3,
      currentWorldRevision: 41,
      requests: [{ object_id: "crate", hand: "left" }]
    });
    const assessment = registry.assessmentsForFrame(3).find((entry) => (
      entry.object_id === "crate" && entry.hand === "left"
    ));
    if (!assessment) throw new Error("Missing verified assessment fixture");

    expect(state).toMatchObject({
      protocol: "humanoid-carried-object-binding-set-v1",
      source_frame: 3,
      source_world_revision: 41,
      grasp_contract_sha256: registry.contractSha256,
      grasp_registry_checkpoint_sha256: humanoidGraspRegistryCheckpointSha256(
        registry.checkpoint()
      ),
      bindings: [{
        protocol: "humanoid-carried-object-binding-v1",
        object_id: "crate",
        hand: "left",
        grasp_assessment_sha256: humanoidGraspAssessmentSha256(assessment),
        verified_contact_surfaces: [
          "left_hand_index_1_link",
          "left_hand_palm_link"
        ]
      }]
    });
    expect(state.bindings[0]!.allowed_hand_surfaces).toHaveLength(8);
    expect(state.bindings[0]!.allowed_hand_surfaces.every((surface) => (
      surface.startsWith("left_hand_")
    ))).toBe(true);
    const constraints = humanoidCarriedObjectContactConstraints(state);
    expect(constraints).toHaveLength(8);
    expect(constraints.every((constraint) => (
      constraint.object_id === "crate"
        && constraint.hand_surface.startsWith("left_hand_")
        && constraint.required === false
    ))).toBe(true);

    const roundTrip = HumanoidCarriedObjectBindingSetSchema.parse(
      JSON.parse(JSON.stringify(state))
    );
    expect(roundTrip).toEqual(state);
    expect(humanoidCarriedObjectBindingSetSha256(roundTrip)).toBe(
      humanoidCarriedObjectBindingSetSha256(state)
    );
    expect(humanoidCarriedObjectBindingSha256(state.bindings[0]!))
      .toMatch(/^[a-f0-9]{64}$/);
    expect(verifyHumanoidCarriedObjectBindingSet({
      registry,
      currentFrame: 3,
      currentWorldRevision: 41,
      state: roundTrip
    })).toEqual(state);
  });

  it("creates no implicit binding when no carried object is requested", () => {
    const registry = verifiedRegistry();
    const state = admitHumanoidCarriedObjectBindings({
      registry,
      currentFrame: 3,
      currentWorldRevision: 41,
      requests: []
    });
    expect(state.bindings).toEqual([]);
    expect(humanoidCarriedObjectContactConstraints(state)).toEqual([]);
  });

  it("rejects stale, missing, unknown, and unverified grasp authority", () => {
    const verified = verifiedRegistry();
    expect(() => admitHumanoidCarriedObjectBindings({
      registry: verified,
      currentFrame: 4,
      currentWorldRevision: 42,
      requests: [{ object_id: "crate", hand: "left" }]
    })).toThrow(/requires current grasp frame/);
    expect(() => admitHumanoidCarriedObjectBindings({
      registry: verified,
      currentFrame: 3,
      currentWorldRevision: 42,
      requests: [{ object_id: "parcel", hand: "left" }]
    })).toThrow(/unknown object/);

    const missing = registryForCrate();
    missing.observe(0, snapshot({ objectIds: [] }));
    expect(() => admitHumanoidCarriedObjectBindings({
      registry: missing,
      currentFrame: 0,
      currentWorldRevision: 1,
      requests: [{ object_id: "crate", hand: "left" }]
    })).toThrow(/no current grasp assessment/);

    const unverified = registryForCrate();
    unverified.observe(0, snapshot({ supported: true, leftContacts: true }));
    expect(() => admitHumanoidCarriedObjectBindings({
      registry: unverified,
      currentFrame: 0,
      currentWorldRevision: 1,
      requests: [{ object_id: "crate", hand: "left" }]
    })).toThrow(/requires a verified grasp assessment/);
  });

  it("rejects duplicate objects, conflicting hands, and one hand carrying two objects", () => {
    const registry = verifiedRegistry();
    expect(() => admitHumanoidCarriedObjectBindings({
      registry,
      currentFrame: 3,
      currentWorldRevision: 41,
      requests: [
        { object_id: "crate", hand: "left" },
        { object_id: "crate", hand: "left" }
      ]
    })).toThrow(/Duplicate carried-object request/);
    expect(() => admitHumanoidCarriedObjectBindings({
      registry,
      currentFrame: 3,
      currentWorldRevision: 41,
      requests: [
        { object_id: "crate", hand: "left" },
        { object_id: "crate", hand: "right" }
      ]
    })).toThrow(/multiple hands/);
    expect(() => admitHumanoidCarriedObjectBindings({
      registry,
      currentFrame: 3,
      currentWorldRevision: 41,
      requests: [
        { object_id: "crate", hand: "left" },
        { object_id: "parcel", hand: "left" }
      ]
    })).toThrow(/multiple carried objects/);
  });

  it("rejects forged assessment surfaces and structurally corrupt binding sets", () => {
    const authority = verifiedRegistry();
    const checkpoint = authority.checkpoint();
    const verified = checkpoint.last_assessments.find((assessment) => (
      assessment.object_id === "crate" && assessment.hand === "left"
    ));
    if (!verified) throw new Error("Missing verified assessment fixture");
    verified.evidence.contact.distinct_force_qualified_links.push(
      "right_hand_palm_link"
    );
    verified.evidence.contact.force_qualified_contact_count += 1;
    verified.evidence.contact.observed_contact_count += 1;
    const forged = new HumanoidGraspRegistry({
      contract: CONTRACT,
      portableObjectIds: ["crate"],
      checkpoint
    });
    expect(() => admitHumanoidCarriedObjectBindings({
      registry: forged,
      currentFrame: 3,
      currentWorldRevision: 41,
      requests: [{ object_id: "crate", hand: "left" }]
    })).toThrow(/opposite hand/);

    const state = admitHumanoidCarriedObjectBindings({
      registry: authority,
      currentFrame: 3,
      currentWorldRevision: 41,
      requests: [{ object_id: "crate", hand: "left" }]
    });
    const duplicate = structuredClone(state);
    duplicate.bindings.push(structuredClone(duplicate.bindings[0]!));
    expect(() => HumanoidCarriedObjectBindingSetSchema.parse(duplicate))
      .toThrow(/Duplicate carried-object binding/);

    const wrongSource = structuredClone(state);
    wrongSource.bindings[0]!.source_world_revision += 1;
    expect(() => HumanoidCarriedObjectBindingSetSchema.parse(wrongSource))
      .toThrow(/source does not match/);

    expect(() => HumanoidCarriedObjectBindingSchema.parse({
      ...state.bindings[0],
      allowed_hand_surfaces: [
        "left_hand_index_1_link",
        "right_hand_palm_link"
      ]
    })).toThrow(/opposite hand/);
  });

  it("reports environment, other-object, and non-authorized humanoid collisions", () => {
    const registry = verifiedRegistry();
    const state = admitHumanoidCarriedObjectBindings({
      registry,
      currentFrame: 3,
      currentWorldRevision: 41,
      requests: [{ object_id: "crate", hand: "left" }]
    });
    const contacts = [
      contact({
        firstBody: "left_wrist_yaw_link",
        firstHandLink: "left_hand_palm_link",
        secondObject: "crate",
        normalForce: 8
      }),
      contact({ firstObject: "crate", normalForce: 2 }),
      contact({ firstObject: "crate", secondObject: "parcel", normalForce: 3 }),
      contact({ firstBody: "torso_link", secondObject: "crate", normalForce: 4 }),
      contact({
        firstBody: "right_wrist_yaw_link",
        firstHandLink: "right_hand_palm_link",
        secondObject: "crate",
        normalForce: 5
      })
    ];

    expect(humanoidCarriedObjectUnauthorizedContacts(state, contacts)).toMatchObject([
      {
        object_id: "crate",
        contact_index: 1,
        counterpart_kind: "environment",
        counterpart_object_id: null,
        counterpart_body: null,
        counterpart_hand_surface: null,
        normal_force_n: 2
      },
      {
        object_id: "crate",
        contact_index: 2,
        counterpart_kind: "object",
        counterpart_object_id: "parcel",
        normal_force_n: 3
      },
      {
        object_id: "crate",
        contact_index: 3,
        counterpart_kind: "humanoid",
        counterpart_body: "torso_link",
        normal_force_n: 4
      },
      {
        object_id: "crate",
        contact_index: 4,
        counterpart_kind: "humanoid",
        counterpart_body: "right_wrist_yaw_link",
        counterpart_hand_surface: "right_hand_palm_link",
        normal_force_n: 5
      }
    ]);

    expect(() => humanoidCarriedObjectUnauthorizedContacts(state, [contact({
      firstObject: "crate",
      secondObject: "parcel",
      secondBody: "torso_link"
    })])).toThrow(/ambiguous counterpart identity/);
  });

  it("provides per-frame continuation evidence and fails closed when grasp is lost", () => {
    const registry = verifiedRegistry();
    const state = admitHumanoidCarriedObjectBindings({
      registry,
      currentFrame: 3,
      currentWorldRevision: 41,
      requests: [{ object_id: "crate", hand: "left" }]
    });

    expect(humanoidCarriedObjectContinuationEvidence({
      state,
      registry,
      currentFrame: 4,
      currentWorldRevision: 42
    })).toMatchObject({
      continued: false,
      bindings: [{ continued: false, failure: "registry_not_current" }]
    });

    registry.observe(4, snapshot({
      objectHeight: 0.56,
      supported: false,
      leftContacts: true
    }));
    const continued = humanoidCarriedObjectContinuationEvidence({
      state,
      registry,
      currentFrame: 4,
      currentWorldRevision: 42
    });
    expect(HumanoidCarriedObjectContinuationEvidenceSchema.parse(continued))
      .toMatchObject({
        source_frame: 3,
        observed_frame: 4,
        continued: true,
        bindings: [{
          object_id: "crate",
          hand: "left",
          continued: true,
          failure: null,
          current_assessment_sha256: expect.stringMatching(/^[a-f0-9]{64}$/),
          verified_contact_surfaces: [
            "left_hand_index_1_link",
            "left_hand_palm_link"
          ]
        }]
      });

    registry.observe(5, snapshot({
      objectHeight: 0.5,
      supported: true,
      leftContacts: true
    }));
    expect(humanoidCarriedObjectContinuationEvidence({
      state,
      registry,
      currentFrame: 5,
      currentWorldRevision: 43
    })).toMatchObject({
      continued: true,
      bindings: [{
        continued: true,
        failure: null,
        verified_contact_surfaces: [
          "left_hand_index_1_link",
          "left_hand_palm_link"
        ]
      }]
    });
    expect(() => admitHumanoidCarriedObjectBindings({
      registry,
      currentFrame: 5,
      currentWorldRevision: 43,
      requests: [{ object_id: "crate", hand: "left" }]
    })).toThrow(/acquisition has insufficient lift evidence/);

    registry.observe(6, snapshot({
      objectHeight: 0.5,
      supported: true,
      leftContacts: false
    }));
    expect(humanoidCarriedObjectContinuationEvidence({
      state,
      registry,
      currentFrame: 6,
      currentWorldRevision: 44
    })).toMatchObject({
      continued: false,
      bindings: [{
        continued: false,
        failure: "grasp_not_verified",
        current_assessment_sha256: expect.stringMatching(/^[a-f0-9]{64}$/)
      }]
    });
  });

  it("keeps the original acquisition authority across continued carry frames and restore", () => {
    const registry = verifiedRegistry();
    const lifecycle = new HumanoidCarriedObjectLifecycle({
      registry,
      currentFrame: 3,
      currentWorldRevision: 41
    });
    const acquired = lifecycle.acquire({
      currentFrame: 3,
      currentWorldRevision: 41,
      requests: [{ object_id: "crate", hand: "left" }]
    });

    registry.observe(4, snapshot({
      objectHeight: 0.56,
      supported: false,
      leftContacts: true
    }));
    expect(lifecycle.observe({
      currentFrame: 4,
      currentWorldRevision: 42,
      contacts: []
    })).toMatchObject({ phase: "carrying", continuation: { continued: true } });
    expect(lifecycle.active).toMatchObject({
      source_frame: 3,
      source_world_revision: 41,
      bindings: [{ grasp_assessment_sha256: acquired.bindings[0]!.grasp_assessment_sha256 }]
    });

    registry.observe(5, snapshot({
      objectHeight: 0.5,
      supported: true,
      leftContacts: true
    }));
    lifecycle.observe({
      currentFrame: 5,
      currentWorldRevision: 43,
      contacts: []
    });
    const checkpoint = lifecycle.checkpoint();
    expect(HumanoidCarriedObjectLifecycleCheckpointSchema.parse(checkpoint))
      .toMatchObject({
        phase: "carrying",
        active_binding_set: { source_frame: 3 },
        last_continuation: { observed_frame: 5, continued: true }
      });
    const restored = new HumanoidCarriedObjectLifecycle({
      registry,
      currentFrame: 5,
      currentWorldRevision: 43,
      checkpoint
    });
    expect(restored.active).toEqual(lifecycle.active);

    registry.observe(6, snapshot({
      objectHeight: 0.5,
      supported: true,
      leftContacts: false
    }));
    expect(restored.observe({
      currentFrame: 6,
      currentWorldRevision: 44,
      contacts: []
    })).toMatchObject({ phase: "lost", continuation: { continued: false } });
    expect(restored.active).toBeNull();
  });

  it("distinguishes an authorized release from an accidental grasp loss", () => {
    const registry = verifiedRegistry();
    const lifecycle = new HumanoidCarriedObjectLifecycle({
      registry,
      currentFrame: 3,
      currentWorldRevision: 41
    });
    lifecycle.acquire({
      currentFrame: 3,
      currentWorldRevision: 41,
      requests: [{ object_id: "crate", hand: "left" }]
    });
    lifecycle.beginRelease({ currentFrame: 3, currentWorldRevision: 41 });
    registry.observe(4, snapshot({
      objectHeight: 0.5,
      supported: true,
      leftContacts: false
    }));

    expect(lifecycle.observe({
      currentFrame: 4,
      currentWorldRevision: 42,
      contacts: [{
        position: { x: 0, y: 0.5, z: 0 },
        normal: { x: 1, y: 0, z: 0 },
        normalForce: 0.1,
        firstBody: null,
        secondBody: null,
        firstObject: "crate",
        secondObject: null,
        firstHandLink: null,
        secondHandLink: "left_hand_index_1_link"
      }]
    })).toMatchObject({ phase: "release_pending", continuation: { continued: false } });
    expect(lifecycle.active).not.toBeNull();

    registry.observe(5, snapshot({
      objectHeight: 0.4,
      supported: true,
      leftContacts: false
    }));
    expect(lifecycle.observe({
      currentFrame: 5,
      currentWorldRevision: 43,
      contacts: []
    })).toMatchObject({ phase: "released", continuation: { continued: false } });
    expect(lifecycle.checkpoint()).toMatchObject({
      phase: "released",
      transition_reason: "release_completed",
      active_binding_set: null
    });
  });
});

function verifiedRegistry(): HumanoidGraspRegistry {
  const registry = registryForCrate();
  registry.observe(0, snapshot({ supported: true, leftContacts: true }));
  registry.observe(1, snapshot({ supported: true, leftContacts: true }));
  registry.observe(2, snapshot({
    objectHeight: 0.56,
    supported: false,
    leftContacts: true
  }));
  registry.observe(3, snapshot({
    objectHeight: 0.56,
    supported: false,
    leftContacts: true
  }));
  return registry;
}

function registryForCrate(): HumanoidGraspRegistry {
  return new HumanoidGraspRegistry({
    contract: CONTRACT,
    portableObjectIds: ["crate"]
  });
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
        firstBody: "left_wrist_yaw_link",
        firstHandLink: "left_hand_palm_link",
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
