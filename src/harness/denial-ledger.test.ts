import { describe, expect, it } from "vitest";
import { DenialLedger, REPEATED_DENIAL_LIMIT } from "./denial-ledger.js";

/**
 * The ledger exists because of a real run that committed 594 identical IK
 * denials from four distinct targets, and a later one that spent eleven
 * consecutive reads at a single revision. These cases are the exact
 * distinctions it has to draw to stop both without ever blocking an agent that
 * is progressing.
 */

const base = {
  agentId: "agent_1",
  name: "solve_end_effector_pose",
  input: { position: { x: 2, y: 0.5, z: 1.5 } },
  worldRevision: 7
};

describe("DenialLedger", () => {
  it("allows the limit through before refusing", () => {
    const ledger = new DenialLedger();
    for (let attempt = 1; attempt < REPEATED_DENIAL_LIMIT; attempt += 1) {
      ledger.recordDenial(base);
      expect(ledger.exhausted(base)).toBe(false);
    }
    ledger.recordDenial(base);
    expect(ledger.exhausted(base)).toBe(true);
    expect(ledger.count(base)).toBe(REPEATED_DENIAL_LIMIT);
  });

  it("never blocks a different argument, however small the difference", () => {
    const ledger = new DenialLedger();
    for (let attempt = 0; attempt < REPEATED_DENIAL_LIMIT * 2; attempt += 1) {
      ledger.recordDenial(base);
    }
    // This is the corrected y the world's own recovery hint asks for.
    const corrected = { ...base, input: { position: { x: 2, y: 0.554, z: 1.5 } } };
    expect(ledger.exhausted(corrected)).toBe(false);
  });

  it("forgets everything once the world actually changes", () => {
    const ledger = new DenialLedger();
    for (let attempt = 0; attempt < REPEATED_DENIAL_LIMIT; attempt += 1) {
      ledger.recordDenial(base);
    }
    expect(ledger.exhausted(base)).toBe(true);

    // The base drove somewhere, so the same target is a different question.
    const moved = { ...base, worldRevision: 8 };
    expect(ledger.exhausted(moved)).toBe(false);
    expect(ledger.count(moved)).toBe(0);
  });

  it("scores each agent separately, so one leaf's loop does not gag a sibling", () => {
    const ledger = new DenialLedger();
    for (let attempt = 0; attempt < REPEATED_DENIAL_LIMIT; attempt += 1) {
      ledger.recordDenial(base);
    }
    expect(ledger.exhausted({ ...base, agentId: "agent_2" })).toBe(false);
  });

  it("does not confuse two capabilities called with the same arguments", () => {
    const ledger = new DenialLedger();
    for (let attempt = 0; attempt < REPEATED_DENIAL_LIMIT; attempt += 1) {
      ledger.recordDenial(base);
    }
    expect(ledger.exhausted({ ...base, name: "plan_base_path" })).toBe(false);
  });

  it("starts clean, so a first attempt is never refused", () => {
    expect(new DenialLedger().exhausted(base)).toBe(false);
  });
});
