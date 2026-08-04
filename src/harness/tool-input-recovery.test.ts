import { describe, expect, it } from "vitest";
import { z } from "zod";
import {
  invalidToolInputResult,
  preflightAgentToolInput
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
});
