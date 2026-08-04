import type { Scenario, Vec3 } from "../../domain/schema.js";
import { humanoidFixedObjectSolidId } from "./environment.js";
import type {
  HumanoidContactSnapshot,
  HumanoidSolidSensorSnapshot
} from "./simulation.js";

export interface HumanoidSolidToken {
  id: string;
  sourceId: string;
  kind: "block" | "fixed_object";
  center: Vec3;
  size: Vec3;
  currentContacts: HumanoidContactSnapshot[];
}

export function visibleHumanoidSolidTokens(input: {
  scenario: Scenario;
  sensed: HumanoidSolidSensorSnapshot;
  contacts: readonly HumanoidContactSnapshot[];
}): HumanoidSolidToken[] {
  const descriptors = new Map<string, Omit<HumanoidSolidToken, "currentContacts">>();
  for (const block of input.scenario.obstacles) {
    descriptors.set(block.id, {
      id: block.id,
      sourceId: block.id,
      kind: "block",
      center: { ...block.center },
      size: { ...block.size }
    });
  }
  for (const object of input.scenario.objects) {
    if (object.portable) continue;
    const id = humanoidFixedObjectSolidId(object.id);
    descriptors.set(id, {
      id,
      sourceId: object.id,
      kind: "fixed_object",
      center: { ...object.position },
      size: { ...object.size }
    });
  }
  return observableHumanoidSolidIds(input.sensed, input.contacts).map((id) => {
    const descriptor = descriptors.get(id);
    if (!descriptor) throw new Error(`Visible MuJoCo solid has no scenario descriptor: ${id}`);
    return {
      ...descriptor,
      currentContacts: input.contacts
        .filter((contact) => contact.firstSolid === id || contact.secondSolid === id)
        .map((contact) => structuredClone(contact))
    };
  });
}

export function observableHumanoidSolidIds(
  sensed: HumanoidSolidSensorSnapshot,
  contacts: readonly HumanoidContactSnapshot[]
): string[] {
  return [...new Set([
    ...Object.keys(sensed.solids),
    ...contacts.flatMap((contact) => [
      ...(contact.firstSolid ? [contact.firstSolid] : []),
      ...(contact.secondSolid ? [contact.secondSolid] : [])
    ])
  ])].sort();
}
