import { createHash } from "node:crypto";

export function humanoidMotionContactEvidenceSha256(input: {
  planId: string;
  intentSha256: string;
  artifactSha256: string;
  nextFrameIndex: number;
  satisfiedContactKeys: readonly string[];
}): string {
  if (!input.planId) throw new Error("Humanoid contact evidence requires a plan id");
  if (!/^[a-f0-9]{64}$/.test(input.intentSha256)
    || !/^[a-f0-9]{64}$/.test(input.artifactSha256)) {
    throw new Error("Humanoid contact evidence requires valid intent and artifact identities");
  }
  if (!Number.isSafeInteger(input.nextFrameIndex) || input.nextFrameIndex < 0) {
    throw new Error("Humanoid contact evidence frame must be a nonnegative integer");
  }
  const keys = [...input.satisfiedContactKeys].sort();
  if (keys.some((key) => key.length === 0)
    || new Set(keys).size !== keys.length) {
    throw new Error("Humanoid contact evidence keys must be nonempty and unique");
  }
  return createHash("sha256").update(JSON.stringify({
    protocol: "humanoid-motion-contact-evidence-v1",
    plan_id: input.planId,
    intent_sha256: input.intentSha256,
    artifact_sha256: input.artifactSha256,
    next_frame_index: input.nextFrameIndex,
    satisfied_contact_keys: keys
  })).digest("hex");
}
