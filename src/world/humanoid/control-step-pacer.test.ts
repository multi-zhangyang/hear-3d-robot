import { afterEach, describe, expect, it, vi } from "vitest";
import { createHumanoidControlStepPacer } from "./control-step-pacer.js";

afterEach(() => {
  vi.useRealTimers();
});

describe("humanoid control step pacer", () => {
  it("paces authority steps without catch-up bursts", async () => {
    vi.useFakeTimers();
    const pacer = createHumanoidControlStepPacer({
      controlStepSeconds: 0.02,
      realtime: true
    });
    const first = pacer.waitForNextStep();
    await vi.advanceTimersByTimeAsync(19);
    let completed = false;
    void first.then(() => { completed = true; });
    await Promise.resolve();
    expect(completed).toBe(false);
    await vi.advanceTimersByTimeAsync(1);
    await first;

    await vi.advanceTimersByTimeAsync(50);
    const delayed = pacer.waitForNextStep();
    await vi.advanceTimersByTimeAsync(19);
    completed = false;
    void delayed.then(() => { completed = true; });
    await Promise.resolve();
    expect(completed).toBe(false);
    await vi.advanceTimersByTimeAsync(1);
    await delayed;
  });

  it("does not delay unpaced validation and observes cancellation", async () => {
    const controller = new AbortController();
    const pacer = createHumanoidControlStepPacer({
      controlStepSeconds: 0.02,
      realtime: false,
      signal: controller.signal
    });
    await pacer.waitForNextStep();
    controller.abort(new Error("stopped"));
    await expect(pacer.waitForNextStep()).rejects.toThrow("stopped");
  });
});
