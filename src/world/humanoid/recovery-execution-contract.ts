import { createHash } from "node:crypto";
import { z } from "zod";
import {
  NeuralSafetyInterruptSchema
} from "../../domain/neural-hierarchy.js";
import {
  HumanoidContactConstraintSchema
} from "./motion-plan-schema.js";
import {
  HUMANOID_RECOVERY_HANDOFF_STEPS,
  HUMANOID_RECOVERY_MAXIMUM_STEPS,
  HUMANOID_RECOVERY_STABLE_STEPS
} from "../../domain/humanoid-policy.js";

export {
  HUMANOID_RECOVERY_HANDOFF_STEPS,
  HUMANOID_RECOVERY_MAXIMUM_STEPS,
  HUMANOID_RECOVERY_STABLE_STEPS
} from "../../domain/humanoid-policy.js";

export const HumanoidRecoveryExecutionContractSchema = z.object({
  protocol: z.literal("humanoid-embodied-recovery-contract-v1"),
  safetyInterrupt: NeuralSafetyInterruptSchema,
  minimumSupportMarginMeters: z.number().finite().nonnegative().max(0.3),
  stableSteps: z.literal(HUMANOID_RECOVERY_STABLE_STEPS),
  handoffSteps: z.literal(HUMANOID_RECOVERY_HANDOFF_STEPS),
  maximumSteps: z.literal(HUMANOID_RECOVERY_MAXIMUM_STEPS),
  authorizedContacts: z.array(HumanoidContactConstraintSchema).max(16),
  standing: z.object({
    minimumRootHeightMeters: z.literal(0.7),
    minimumUpright: z.literal(0.9),
    maximumRootLinearSpeedMetersPerSecond: z.literal(0.35),
    maximumRootAngularSpeedRadiansPerSecond: z.literal(0.5),
    maximumJointSpeedRadiansPerSecond: z.literal(1.5),
    requireBothFeetContact: z.literal(true)
  }).strict(),
  safetyLimits: z.object({
    maximumPeakContactNormalForceN: z.literal(2500),
    maximumTotalContactNormalForceN: z.literal(4000),
    maximumTotalContactForceRiseRateNPerSecond: z.literal(100000),
    maximumJointSpeedRadiansPerSecond: z.literal(40),
    minimumJointLimitMarginRadians: z.literal(-0.1)
  }).strict()
}).strict().superRefine((contract, context) => {
  const interrupt = contract.safetyInterrupt;
  if (interrupt.kind !== "stationary_fall"
    || interrupt.status !== "acknowledged") {
    context.addIssue({
      code: "custom",
      path: ["safetyInterrupt"],
      message: "Recovery requires an acknowledged physical fall interrupt"
    });
  }
  const contactKeys = contract.authorizedContacts.map((contact) => (
    JSON.stringify(contact)
  ));
  if (new Set(contactKeys).size !== contactKeys.length) {
    context.addIssue({
      code: "custom",
      path: ["authorizedContacts"],
      message: "Recovery contact authorization cannot contain duplicates"
    });
  }
});

export type HumanoidRecoveryExecutionContract = z.infer<
  typeof HumanoidRecoveryExecutionContractSchema
>;

export const HumanoidRecoveryPlanSchema = z.object({
  id: z.string().trim().min(1),
  contract: HumanoidRecoveryExecutionContractSchema
}).strict();

export type HumanoidRecoveryPlan = z.infer<typeof HumanoidRecoveryPlanSchema>;

export function humanoidRecoveryContractSha256(
  contract: HumanoidRecoveryExecutionContract
): string {
  const parsed = HumanoidRecoveryExecutionContractSchema.parse(contract);
  return createHash("sha256").update(canonicalJson(parsed)).digest("hex");
}

function canonicalJson(value: unknown): string {
  if (value === null || typeof value !== "object") return JSON.stringify(value);
  if (Array.isArray(value)) return `[${value.map(canonicalJson).join(",")}]`;
  const record = value as Record<string, unknown>;
  return `{${Object.keys(record).sort().map((key) => (
    `${JSON.stringify(key)}:${canonicalJson(record[key])}`
  )).join(",")}}`;
}
