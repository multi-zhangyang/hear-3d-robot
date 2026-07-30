import { describe, expect, it } from "vitest";
import { DelegationDrainRegistry } from "./delegation-drain.js";

describe("DelegationDrainRegistry", () => {
  it("drains every registered sibling without making interrupted siblings wait on themselves", async () => {
    const registry = new DelegationDrainRegistry();
    const first = registry.register("parent", "first");
    const second = registry.register("parent", "second");
    expect(first.sourceCallIds).toEqual(new Set(["first", "second"]));

    let firstDrained = false;
    const firstDrain = first.settleAndDrain().then(() => {
      firstDrained = true;
    });
    await Promise.resolve();
    expect(firstDrained).toBe(false);

    const secondDrain = second.settleAndDrain();
    await Promise.all([firstDrain, secondDrain]);
    expect(firstDrained).toBe(true);
  });

  it("does not let a retry cross into an interrupted batch before it drains", async () => {
    const registry = new DelegationDrainRegistry();
    const interrupted = registry.register("parent", "old_failure");
    const oldSibling = registry.register("parent", "old_sibling");
    const draining = interrupted.settleAndDrain();

    expect(() => registry.register("parent", "retry"))
      .toThrow("still draining an interrupted batch");

    let oldDrained = false;
    void draining.then(() => {
      oldDrained = true;
    });
    await Promise.resolve();
    expect(oldDrained).toBe(false);
    oldSibling.settle();
    await draining;
    expect(oldDrained).toBe(true);

    const retry = registry.register("parent", "retry");
    expect(retry.sourceCallIds).toEqual(new Set(["retry"]));
    retry.settle();
  });
});
