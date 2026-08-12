import { describe, expect, it } from "vitest";
import { loadRuntimeCatalog } from "../config/load.js";
import {
  decodeWorkyardPolicyAction,
  dryRunWorkyardTrainingContract,
  loadWorkyardTrainingContract,
  WorkyardTrainingContractSchema
} from "./workyard-contract.js";

describe("Workyard task-conditioned training contract", () => {
  it("dry-runs against the real scenario and G1 morphology", async () => {
    const [catalog, contract] = await Promise.all([
      loadRuntimeCatalog(),
      loadWorkyardTrainingContract()
    ]);

    const report = dryRunWorkyardTrainingContract(
      contract,
      catalog.materialize("humanoid_workyard", 0)
    );

    expect(report).toMatchObject({
      scenario_id: "humanoid_workyard",
      target: {
        object_id: "assembly_rod",
        source_support_id: "pickup_stand",
        target_zone_id: "assembly_bay"
      },
      morphology: {
        body_joint_count: 29,
        hand_joint_count: 14
      },
      observation: { size: 221 },
      action: {
        size: 37,
        body_residual_count: 29,
        hand_synergy_count: 8
      },
      contract_ready: true,
      colab_smoke_ready: true,
      blockers: []
    });
    expect(report.curriculum).toEqual([
      "reach",
      "contact",
      "grasp",
      "lift",
      "carry",
      "place"
    ]);
  });

  it("decodes bounded body residuals and stateful hand synergy deltas", async () => {
    const contract = await loadWorkyardTrainingContract();
    const action = Array.from({ length: contract.action.size }, () => 0);
    action[0] = 1;
    action[29] = 1;
    const decoded = decodeWorkyardPolicyAction(contract, action, {
      left: {
        thumb_opposition: 0.98,
        thumb_curl: 0,
        index_curl: 0,
        middle_curl: 0
      },
      right: {
        thumb_opposition: 0,
        thumb_curl: 0,
        index_curl: 0,
        middle_curl: 0
      }
    });

    expect(decoded.body_joint_position_residuals).toHaveLength(29);
    expect(decoded.body_joint_position_residuals[0]).toBe(0.35);
    expect(decoded.hand_coordination.left.thumb_opposition).toBe(1);
    expect(Object.keys(decoded.hand_joint_targets)).toHaveLength(14);
  });

  it("rejects dimension drift before any GPU session can start", async () => {
    const contract = await loadWorkyardTrainingContract();
    const drifted = structuredClone(contract) as unknown as Record<string, unknown>;
    const observation = drifted.observation as Record<string, unknown>;
    observation.size = 220;

    expect(() => WorkyardTrainingContractSchema.parse(drifted)).toThrow(
      /does not match term size/
    );
  });
});
