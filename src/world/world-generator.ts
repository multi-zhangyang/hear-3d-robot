import { randomBytes } from "node:crypto";
import type { Scenario, ScenarioTemplate } from "../domain/schema.js";
import { materializeProceduralWorld } from "./procedural-world.js";

export function materializeScenario(
  template: ScenarioTemplate,
  seed: number
): Scenario {
  if (template.kind === "authored") {
    return { ...structuredClone(template.scenario), seed };
  }
  return materializeProceduralWorld(template, seed);
}

/** A run seed drawn from the platform's randomness, used once and then stored. */
export function drawSeed(): number {
  return randomBytes(4).readUInt32LE(0);
}
