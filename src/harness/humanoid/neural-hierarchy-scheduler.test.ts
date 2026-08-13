import { randomUUID } from "node:crypto";
import { describe, expect, it, vi } from "vitest";
import {
  HUMANOID_NEURAL_AGENT_IDS
} from "./neural-hierarchy-contract.js";
import {
  NeuralHierarchyScheduler,
  type NeuralSchedulerDispatch,
  type NeuralSchedulerEvent
} from "./neural-hierarchy-scheduler.js";

describe("NeuralHierarchyScheduler", () => {
  it("serializes all external wake events through the one Executive root", async () => {
    let releaseFirst!: () => void;
    const firstGate = new Promise<void>((resolve) => {
      releaseFirst = resolve;
    });
    let firstStarted!: () => void;
    const started = new Promise<void>((resolve) => {
      firstStarted = resolve;
    });
    const dispatched: NeuralSchedulerDispatch[] = [];
    const scheduler = new NeuralHierarchyScheduler({
      resolveAuthority: () => ({
        targetNodeId: HUMANOID_NEURAL_AGENT_IDS.executive,
        parentNodeId: null,
        authorityLeaseId: null,
        authorityPath: []
      }),
      dispatch: async (wake) => {
        dispatched.push(wake);
        if (dispatched.length === 1) {
          firstStarted();
          await firstGate;
        }
      }
    });

    scheduler.publish(event("run_started"));
    await started;
    scheduler.publish(event("world_revision_changed"));
    await Promise.resolve();
    expect(dispatched).toHaveLength(1);

    releaseFirst();
    await scheduler.waitForIdle();
    expect(dispatched).toHaveLength(2);
    expect(dispatched.every((wake) => (
      wake.executiveNodeId === HUMANOID_NEURAL_AGENT_IDS.executive
    ))).toBe(true);
    await scheduler.shutdown();
  });

  it("rejects a lower wake that omits any Executive-to-target authority hop", async () => {
    const dispatch = vi.fn(async () => undefined);
    const scheduler = new NeuralHierarchyScheduler({
      resolveAuthority: () => ({
        targetNodeId: HUMANOID_NEURAL_AGENT_IDS.predictive,
        parentNodeId: HUMANOID_NEURAL_AGENT_IDS.sensorimotorManager,
        authorityLeaseId: randomUUID(),
        authorityPath: [{
          parentNodeId: HUMANOID_NEURAL_AGENT_IDS.sensorimotorManager,
          childNodeId: HUMANOID_NEURAL_AGENT_IDS.predictive,
          authorityLeaseId: randomUUID()
        }]
      }),
      dispatch
    });

    scheduler.publish(event("rollout_completed"));
    await expect(scheduler.waitForIdle()).rejects.toThrow(
      "Scheduler authority path is discontinuous"
    );
    expect(dispatch).not.toHaveBeenCalled();
    await scheduler.shutdown();
  });
});

function event(kind: "run_started" | "world_revision_changed" | "rollout_completed"):
NeuralSchedulerEvent {
  return {
    event_id: randomUUID(),
    at: new Date().toISOString(),
    world_revision: 0,
    causal_signal_ids: [],
    kind
  };
}
