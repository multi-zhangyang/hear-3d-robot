import { describe, expect, it } from "vitest";
import {
  HumanoidEmbodiedSkillEventSchema,
  HumanoidEmbodiedSkillStatusSchema,
  legacyHumanoidEmbodiedSkillIdentity,
  type HumanoidEmbodiedSkillStatus
} from "./embodied-skill-call.js";
import { HumanoidSkillEventStream } from "./skill-event-stream.js";
import { restoreHumanoidSkillEventStreamStates } from "./skill-event-stream.js";

const identity = legacyHumanoidEmbodiedSkillIdentity({
  callId: "skill-event-test",
  runtimeKind: "legacy_motion",
  phase: "test",
  observedFrame: 0,
  observedWorldRevision: 0
});

describe("humanoid Skill event stream", () => {
  it("emits sparse physical progress and monotonic event identities", async () => {
    const events: unknown[] = [];
    const stream = new HumanoidSkillEventStream(identity, (event) => {
      events.push(event);
    });

    await stream.accepted(status("accepted", 0, 0));
    await stream.progress(status("executing", 0.05, 0));
    await stream.progress(status("executing", 0.08, 0));
    await stream.progress(status("executing", 0.08, 1));
    await stream.progress(status("executing", 0.04, 2));
    await stream.terminal("succeeded", status("succeeded", 1, 2));

    expect(events).toMatchObject([
      { sequence: 0, type: "accepted" },
      { sequence: 1, type: "progress" },
      { sequence: 2, type: "progress" },
      { sequence: 3, type: "succeeded" }
    ]);
  });

  it("rejects an event whose type contradicts physical status", () => {
    expect(HumanoidEmbodiedSkillEventSchema.safeParse({
      protocol: "humanoid-embodied-skill-event-v1",
      sequence: 0,
      type: "succeeded",
      status: status("executing", 0.5, 0)
    }).success).toBe(false);
  });

  it("restores a durable cursor without repeating accepted", async () => {
    const durable = [HumanoidEmbodiedSkillEventSchema.parse({
      protocol: "humanoid-embodied-skill-event-v1",
      sequence: 0,
      type: "accepted",
      status: status("accepted", 0, 0)
    })];
    const restored = restoreHumanoidSkillEventStreamStates(durable);
    const emitted: unknown[] = [];
    const stream = new HumanoidSkillEventStream(
      identity,
      (event) => emitted.push(event),
      restored.get(identity.callId)
    );

    await stream.accepted(status("accepted", 0, 0));
    await stream.progress(status("executing", 0.5, 1));
    await stream.terminal("succeeded", status("succeeded", 1, 2));

    expect(emitted).toMatchObject([
      { sequence: 1, type: "progress" },
      { sequence: 2, type: "succeeded" }
    ]);
  });

  it("commits its cursor only after the durable sink succeeds", async () => {
    const emitted: unknown[] = [];
    let fail = true;
    const stream = new HumanoidSkillEventStream(identity, (event) => {
      if (fail) throw new Error("journal unavailable");
      emitted.push(event);
    });

    await expect(stream.accepted(status("accepted", 0, 0))).rejects.toThrow(
      "journal unavailable"
    );
    fail = false;
    await stream.accepted(status("accepted", 0, 0));

    expect(emitted).toMatchObject([{ sequence: 0, type: "accepted" }]);
  });

  it("rejects a broken durable sequence during recovery", () => {
    expect(() => restoreHumanoidSkillEventStreamStates([
      HumanoidEmbodiedSkillEventSchema.parse({
        protocol: "humanoid-embodied-skill-event-v1",
        sequence: 1,
        type: "accepted",
        status: status("accepted", 0, 0)
      })
    ])).toThrow("non-contiguous event sequence");
  });
});

function status(
  state: HumanoidEmbodiedSkillStatus["state"],
  progress: number,
  stableSteps: number
): HumanoidEmbodiedSkillStatus {
  return HumanoidEmbodiedSkillStatusSchema.parse({
    protocol: "humanoid-embodied-skill-status-v1",
    callId: identity.callId,
    state,
    progress: {
      elapsedRatio: progress,
      physicalCompletionRatio: progress,
      satisfiedPredicateRatio: progress,
      stableSteps,
      requiredStableSteps: 2
    },
    confidence: {
      value: 1,
      basis: "observable_contract_evidence"
    },
    failure: null,
    recoverability: "not_applicable",
    worldFrame: stableSteps,
    worldRevision: stableSteps,
    controller: null
  });
}
