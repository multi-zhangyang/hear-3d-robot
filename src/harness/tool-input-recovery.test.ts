import { describe, expect, it } from "vitest";
import { z } from "zod";
import {
  createToolInputRecovery,
  invalidToolInputResult,
  preflightAgentToolInput,
  recoverInvalidToolInputOutput
} from "./tool-input-recovery.js";

describe("invalid tool input recovery", () => {
  it("keeps schema correction available when an Agent tool omits invocation metadata", () => {
    const output = invalidToolInputResult(
      Object.assign(new Error("Invalid JSON input for tool"), {
        name: "InvalidToolInputError",
        originalError: {
          issues: [{
            path: ["planning_transaction_id"],
            code: "custom",
            message: "Planning transaction is required"
          }]
        }
      }),
      "delegate_physics_executor"
    );

    expect(JSON.parse(output)).toMatchObject({
      accepted: false,
      code: "invalid_tool_input",
      tool: "delegate_physics_executor",
      automatic_actuation: false,
      next_response_contract: {
        mode: "corrected_tool_call_only",
        tool: "delegate_physics_executor",
        preserve_valid_fields: true,
        narration_allowed: false
      },
      validation_issues: [{
        path: "planning_transaction_id",
        message: "Planning transaction is required"
      }]
    });
  });

  it("does not relabel another tool's schema error", () => {
    const error = Object.assign(new Error("Invalid JSON input for tool"), {
      name: "InvalidToolInputError",
      toolInvocation: {
        details: { toolCall: { name: "another_tool" } }
      }
    });

    expect(() => invalidToolInputResult(error, "delegate_physics_executor"))
      .toThrow(error);
  });

  it("rejects schema-invalid agent delegation before SDK invocation", () => {
    const output = preflightAgentToolInput(
      JSON.stringify({ planning_transaction_id: null }),
      z.object({ planning_transaction_id: z.string().min(1) }).strict(),
      "delegate_physics_executor"
    );

    expect(JSON.parse(String(output))).toMatchObject({
      accepted: false,
      code: "invalid_tool_input",
      tool: "delegate_physics_executor",
      validation_issues: [{
        path: "planning_transaction_id",
        code: "invalid_type"
      }]
    });
  });

  it("rejects malformed JSON without invoking a delegated agent", () => {
    const output = preflightAgentToolInput(
      "{",
      z.object({ objective: z.string().min(1) }).strict(),
      "delegate_humanoid_sentry"
    );

    expect(JSON.parse(String(output))).toMatchObject({
      accepted: false,
      code: "invalid_tool_input",
      validation_issues: [{ path: "", code: "invalid_json" }]
    });
  });

  it("reports each actionable field error once", () => {
    const output = invalidToolInputResult(
      Object.assign(new Error("Invalid JSON input for tool"), {
        name: "InvalidToolInputError",
        originalError: {
          issues: [
            { path: ["candidates", 0, "contacts"], code: "custom", message: "Duplicate contact" },
            { path: ["candidates", 0, "contacts"], code: "custom", message: "Duplicate contact" },
            { path: ["candidates", 1, "contacts"], code: "custom", message: "Duplicate contact" }
          ]
        }
      }),
      "plan_whole_body_motion_candidates"
    );

    expect(JSON.parse(output).validation_issues).toEqual([
      {
        path: "candidates.0.contacts",
        code: "custom",
        message: "Duplicate contact"
      },
      {
        path: "candidates.1.contacts",
        code: "custom",
        message: "Duplicate contact"
      }
    ]);
  });

  it("recovers redacted SDK validation details without discarding result metadata", () => {
    const output = recoverInvalidToolInputOutput(
      JSON.stringify({
        accepted: false,
        code: "invalid_tool_input",
        tool: "recall_goal_history",
        validation_issues: [],
        historical_only: true
      }),
      JSON.stringify({ limit: 0 }),
      z.object({ limit: z.number().int().positive() }).strict(),
      "recall_goal_history",
      createToolInputRecovery()
    );

    expect(JSON.parse(output)).toMatchObject({
      accepted: false,
      code: "invalid_tool_input",
      historical_only: true,
      validation_issues: [{ path: "limit", code: "too_small" }]
    });
  });

  it("does not preflight an input that the SDK accepted", () => {
    const output = recoverInvalidToolInputOutput(
      "accepted-output",
      JSON.stringify({ optional_label: null }),
      z.object({ optional_label: z.string().optional() }).strict(),
      "strict_tool",
      createToolInputRecovery()
    );

    expect(output).toBe("accepted-output");
  });

  it("identifies a repeated invalid field without repairing the model input", () => {
    const recovery = createToolInputRecovery();
    const schema = z.object({
      termination: z.object({
        predicates: z.array(z.object({
          minimum_normal_force: z.number().positive()
        }))
      })
    });
    const input = JSON.stringify({
      termination: {
        predicates: [{ minimum_normal_force: 0 }]
      }
    });

    expect(JSON.parse(String(recovery.preflight(
      input,
      schema,
      "plan_whole_body_motion_candidates"
    )))).toMatchObject({
      code: "invalid_tool_input"
    });
    expect(JSON.parse(String(recovery.preflight(
      input,
      schema,
      "plan_whole_body_motion_candidates"
    )))).toMatchObject({
      accepted: false,
      code: "repeated_invalid_tool_input",
      tool: "plan_whole_body_motion_candidates",
      repeated_attempt: {
        count: 2,
        invalid_fields: [{
          path: "termination.predicates.0.minimum_normal_force",
          value: 0
        }]
      },
      automatic_actuation: false
    });
  });

  it("treats reordered JSON as the same invalid model arguments", () => {
    const recovery = createToolInputRecovery();
    const schema = z.object({
      root_height: z.number().positive(),
      intent: z.string()
    });

    recovery.preflight(
      JSON.stringify({ intent: "reach", root_height: 0 }),
      schema,
      "plan_whole_body_motion_candidates"
    );
    const output = recovery.preflight(
      JSON.stringify({ root_height: 0, intent: "reach" }),
      schema,
      "plan_whole_body_motion_candidates"
    );

    expect(JSON.parse(String(output))).toMatchObject({
      code: "repeated_invalid_tool_input",
      repeated_attempt: { count: 2 }
    });
  });

  it("does not let unrelated labels hide a repeated invalid field value", () => {
    const recovery = createToolInputRecovery();
    const schema = z.object({
      id: z.string(),
      objective: z.string(),
      minimum_normal_force: z.number().positive()
    });

    recovery.preflight(
      JSON.stringify({
        id: "candidate-v7",
        objective: "first label",
        minimum_normal_force: 0
      }),
      schema,
      "plan_whole_body_motion_candidates"
    );
    const output = recovery.preflight(
      JSON.stringify({
        id: "candidate-v8",
        objective: "renamed label",
        minimum_normal_force: 0
      }),
      schema,
      "plan_whole_body_motion_candidates"
    );

    expect(JSON.parse(String(output))).toMatchObject({
      code: "repeated_invalid_tool_input",
      repeated_attempt: {
        count: 2,
        invalid_fields: [{ path: "minimum_normal_force", value: 0 }]
      }
    });
  });

  it("reports a repeated null field as the rejected scalar value", () => {
    const recovery = createToolInputRecovery();
    const schema = z.object({ planning_transaction_id: z.string().min(1) });
    const input = JSON.stringify({ planning_transaction_id: null });

    recovery.preflight(input, schema, "delegate_physics_executor");
    const output = recovery.preflight(
      input,
      schema,
      "delegate_physics_executor"
    );

    expect(JSON.parse(String(output))).toMatchObject({
      code: "repeated_invalid_tool_input",
      repeated_attempt: {
        count: 2,
        invalid_fields: [{ path: "planning_transaction_id", value: null }]
      }
    });
  });

  it("resets repeated-input tracking after a valid tool call", () => {
    const recovery = createToolInputRecovery();
    const schema = z.object({ force: z.number().positive() });
    const invalid = JSON.stringify({ force: 0 });

    recovery.preflight(invalid, schema, "contact_tool");
    expect(recovery.preflight(
      JSON.stringify({ force: 1 }),
      schema,
      "contact_tool"
    )).toBeUndefined();
    const output = recovery.preflight(invalid, schema, "contact_tool");

    expect(JSON.parse(String(output))).toMatchObject({
      code: "invalid_tool_input"
    });
  });
});
