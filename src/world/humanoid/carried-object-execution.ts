import {
  humanoidCarriedObjectContinuationEvidence,
  humanoidCarriedObjectUnauthorizedContacts
} from "./carried-object-binding.js";
import type { HumanoidCarriedObjectLifecycle } from "./carried-object-lifecycle.js";
import type { HumanoidCarriedObjectReleaseAuthority } from "./carried-object-release.js";
import { carryNavigationFailure } from "./navigation-execution.js";
import type { HumanoidSimulationSnapshot } from "./simulation.js";

export function carryMotionFailure(input: {
  continuation: ReturnType<typeof humanoidCarriedObjectContinuationEvidence>;
  unauthorized: ReturnType<typeof humanoidCarriedObjectUnauthorizedContacts>;
  snapshot: HumanoidSimulationSnapshot;
  releaseAuthority: HumanoidCarriedObjectReleaseAuthority | null;
  lifecyclePhase: ReturnType<HumanoidCarriedObjectLifecycle["checkpoint"]>["phase"];
}): string | undefined {
  if (!input.releaseAuthority) {
    return carryNavigationFailure(
      input.continuation,
      input.unauthorized,
      input.snapshot
    );
  }
  if (input.lifecyclePhase === "released") return undefined;
  if (input.lifecyclePhase === "release_pending"
    && input.unauthorized.length === 0) return undefined;
  return carryNavigationFailure(
    input.continuation,
    input.unauthorized,
    input.snapshot
  );
}
