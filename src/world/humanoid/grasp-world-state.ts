import { z } from "zod";
import {
  HumanoidGraspAssessmentSchema,
  type HumanoidGraspAssessment
} from "./grasp-tracker.js";

const SHA256_PATTERN = /^[a-f0-9]{64}$/;

export const HumanoidWorldGraspStateSchema = z.object({
  contractSha256: z.string().regex(SHA256_PATTERN),
  assessments: z.array(HumanoidGraspAssessmentSchema)
}).strict().superRefine((state, context) => {
  const identities = new Set<string>();
  state.assessments.forEach((assessment, index) => {
    const identity = `${assessment.object_id}\0${assessment.hand}`;
    if (identities.has(identity)) {
      context.addIssue({
        code: "custom",
        path: ["assessments", index],
        message: "World grasp state contains a duplicate object-hand assessment"
      });
    }
    identities.add(identity);
  });
});

export interface HumanoidWorldGraspState {
  contractSha256: string;
  assessments: HumanoidGraspAssessment[];
}
