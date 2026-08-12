import { describe, expect, it } from "vitest";
import { G1_HAND_CONTACT_SURFACE_NAMES } from "./morphology.js";
import type { HumanoidCarriedObjectBindingSet } from "./carried-object-binding.js";
import { authorizeHumanoidCarriedObjectRelease } from "./carried-object-release.js";

const sha = "a".repeat(64);
const bindingSet: HumanoidCarriedObjectBindingSet = {
  protocol: "humanoid-carried-object-binding-set-v1",
  source_frame: 12,
  source_world_revision: 12,
  grasp_contract_sha256: sha,
  grasp_registry_checkpoint_sha256: sha,
  bindings: [{
    protocol: "humanoid-carried-object-binding-v1",
    object_id: "parcel",
    hand: "left",
    grasp_contract_sha256: sha,
    grasp_registry_checkpoint_sha256: sha,
    grasp_assessment_sha256: sha,
    source_frame: 12,
    source_world_revision: 12,
    verified_contact_surfaces: [
      "left_hand_index_1_link",
      "left_hand_thumb_2_link"
    ],
    allowed_hand_surfaces: G1_HAND_CONTACT_SURFACE_NAMES.filter((surface) => (
      surface.startsWith("left_")
    )).sort()
  }]
};

const contract = {
  option_id: "place-parcel",
  predicates: [{
    type: "grasp_verified" as const,
    object_id: "parcel",
    hand: "left" as const,
    grasp_contract_sha256: sha
  }, {
    type: "object_released" as const,
    object_id: "parcel",
    hand: "left" as const
  }, {
    type: "object_settled_on_support" as const,
    object_id: "parcel"
  }, {
    type: "object_in_zone" as const,
    object_id: "parcel",
    zone_id: "destination",
    expected: true,
    tolerance_m: 0.05
  }],
  stable_steps: 8,
  phases: {
    precondition: {
      condition: { op: "predicate" as const, predicate_index: 0 },
      stable_steps: 1
    },
    during: null,
    terminal: {
      condition: {
        op: "all" as const,
        conditions: [{
          op: "not" as const,
          condition: { op: "predicate" as const, predicate_index: 0 }
        }, {
          op: "predicate" as const,
          predicate_index: 1
        }, {
          op: "predicate" as const,
          predicate_index: 2
        }, {
          op: "predicate" as const,
          predicate_index: 3
        }]
      }
    }
  }
};

describe("carried-object release authority", () => {
  it("binds explicit grasp, separation, destination and settled evidence", () => {
    expect(authorizeHumanoidCarriedObjectRelease({ contract, bindingSet }))
      .toMatchObject({
        protocol: "humanoid-carried-object-release-authority-v1",
        bindings: [{
          objectId: "parcel",
          hand: "left",
          graspPredicateIndex: 0,
          releasedPredicateIndex: 1,
          settledPredicateIndex: 2,
          destinationPredicateIndex: 3
        }]
      });
  });

  it("rejects release contracts that omit physical separation", () => {
    expect(() => authorizeHumanoidCarriedObjectRelease({
      bindingSet,
      contract: {
        ...contract,
        predicates: contract.predicates.filter((predicate) => (
          predicate.type !== "object_released"
        )),
        phases: {
          ...contract.phases,
          terminal: {
            condition: {
              op: "all",
              conditions: [{
                op: "not",
                condition: { op: "predicate", predicate_index: 0 }
              }, {
                op: "predicate",
                predicate_index: 1
              }, {
                op: "predicate",
                predicate_index: 2
              }]
            }
          }
        }
      }
    })).toThrow(/physical separation/);
  });
});
