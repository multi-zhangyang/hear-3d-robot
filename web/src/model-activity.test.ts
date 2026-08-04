import { describe, expect, it } from "vitest";
import {
  createModelActivity,
  modelActivityFromJournal,
  reduceProviderActivity,
  reduceRuntimeModelActivity,
  settleModelActivity
} from "./model-activity";

describe("model activity lifecycle", () => {
  it("ends a real request when its usable response arrives", () => {
    const started = reduceRuntimeModelActivity(createModelActivity(true), {
      event_id: "event-1",
      run_id: "run-1",
      type: "model_request_started",
      at: "2026-08-03T00:00:00.000Z",
      data: { agent_id: "motion" }
    });
    expect(started).toMatchObject({
      phase: "active",
      agentId: "motion",
      startedAt: "2026-08-03T00:00:00.000Z",
      completedAt: null
    });

    const completed = reduceProviderActivity(started, {
      status: "usable_stream",
      at: "2026-08-03T00:00:02.000Z",
      source: "model",
      agentId: "motion"
    });
    expect(completed).toMatchObject({
      phase: "verified",
      agentId: "motion",
      startedAt: "2026-08-03T00:00:00.000Z",
      completedAt: "2026-08-03T00:00:02.000Z"
    });
  });

  it("folds completed journal calls without treating a live run as an active request", () => {
    const activity = modelActivityFromJournal(true, [{
      status: "contacted",
      agent_id: "sentry",
      at: "2026-08-03T00:00:00.000Z"
    }, {
      status: "usable_stream",
      agent_id: "sentry",
      at: "2026-08-03T00:00:01.000Z"
    }], true);
    expect(activity).toMatchObject({ phase: "verified", agentId: "sentry" });
  });

  it("shows recovery distinctly and settles stale terminal activity", () => {
    const interrupted = reduceProviderActivity(createModelActivity(true), {
      status: "transport_interrupted",
      at: "2026-08-03T00:00:00.000Z",
      source: null
    });
    expect(interrupted.phase).toBe("recovering");
    expect(settleModelActivity(interrupted, false).phase).toBe("error");
    expect(settleModelActivity({ ...interrupted, phase: "active" }, false).phase).toBe("ready");
  });

  it("ignores provider bookkeeping that is not a request transition", () => {
    const verified = { ...createModelActivity(true), phase: "verified" as const };
    expect(reduceProviderActivity(verified, {
      status: "context_compacted",
      at: "2026-08-03T00:00:00.000Z",
      source: null
    })).toBe(verified);
  });
});
