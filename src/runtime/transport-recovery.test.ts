import { readFile } from "node:fs/promises";
import { resolve } from "node:path";
import { describe, expect, it } from "vitest";
import { RunCheckpointSchema } from "../domain/schema.js";
import {
  canReplayInitialModelRequest,
  isTransportInterruption
} from "./transport-recovery.js";

describe("isTransportInterruption", () => {
  it("recognizes the socket drop that ended a real mission mid-stream", () => {
    // Recorded verbatim from runs/20260726T130935Z_fetch_red_block_a4fa67c3,
    // which lost its connection after three committed actions. `fetch` reports
    // it as a bare TypeError and hides the reason one level down in `cause`.
    const error = Object.assign(new TypeError("terminated"), {
      cause: Object.assign(new Error("other side closed"), {
        name: "SocketError",
        code: "UND_ERR_SOCKET"
      })
    });
    expect(isTransportInterruption(error)).toBe(true);
  });

  it("recognizes a connection reset carried directly on the error", () => {
    expect(isTransportInterruption(Object.assign(new Error("read ECONNRESET"), {
      code: "ECONNRESET"
    }))).toBe(true);
  });

  it("treats throttling and gateway failures as continuable", () => {
    for (const statusCode of [408, 409, 425, 429, 500, 502, 503, 504, 599]) {
      expect(isTransportInterruption(Object.assign(new Error("upstream"), { statusCode })))
        .toBe(true);
    }
    expect(isTransportInterruption(Object.assign(new Error("busy"), { status: 429 }))).toBe(true);
    expect(isTransportInterruption({
      error: { statusCode: 503, message: "nested compatible-provider failure" }
    })).toBe(true);
  });

  it.each([
    ["statusCode", 400, false],
    ["status", 401, false],
    ["statusCode", 408, true],
    ["status", 409, true],
    ["statusCode", 425, true],
    ["status", 429, true],
    ["statusCode", 503, true],
    ["status", 503, true]
  ] as const)(
    "uses %s=%i instead of blindly retrying an undici response-status error",
    (field, status, expected) => {
      const error = Object.assign(new Error(`HTTP ${status}`), {
        code: "UND_ERR_RESPONSE_STATUS_CODE",
        [field]: status
      });
      expect(isTransportInterruption(error)).toBe(expected);
    }
  );

  it("does not let a terminal response status fall through to a nested socket code", () => {
    const error = Object.assign(new Error("HTTP 400"), {
      code: "UND_ERR_RESPONSE_STATUS_CODE",
      statusCode: 400,
      cause: Object.assign(new Error("socket closed"), { code: "ECONNRESET" })
    });
    expect(isTransportInterruption(error)).toBe(false);
  });

  it("does not continue through failures that would repeat identically", () => {
    // Replaying these loops on an error only a human can clear, so they must
    // end the run rather than consume the recovery budget.
    expect(isTransportInterruption(Object.assign(new Error("unauthorized"), { status: 401 })))
      .toBe(false);
    expect(isTransportInterruption(Object.assign(new Error("bad request"), { statusCode: 400 })))
      .toBe(false);
    expect(isTransportInterruption(new Error("Mission Coordinator returned no text"))).toBe(false);
    expect(isTransportInterruption(new TypeError("x is not a function"))).toBe(false);
    expect(isTransportInterruption(null)).toBe(false);
    expect(isTransportInterruption("boom")).toBe(false);
  });

  it("finds the transport failure inside an aggregate of retried attempts", () => {
    const aggregate = new AggregateError(
      [new Error("first"), Object.assign(new Error("second"), { code: "ETIMEDOUT" })],
      "all attempts failed"
    );
    expect(isTransportInterruption(aggregate)).toBe(true);
  });

  it("terminates on a self-referencing cause chain instead of hanging", () => {
    const error = new Error("looping") as Error & { cause?: unknown };
    error.cause = error;
    expect(isTransportInterruption(error)).toBe(false);
  });

  it("retries an opening request only while mission authority is unchanged", async () => {
    const checkpoint = RunCheckpointSchema.parse(JSON.parse(await readFile(resolve(
      "tests/fixtures/runs/20000101T000000Z_fetch_red_block_00000000/checkpoint.json"
    ), "utf8")));
    const telemetryOnly = structuredClone(checkpoint);
    telemetryOnly.total_model_calls += 1;
    const root = telemetryOnly.nodes[telemetryOnly.root_id];
    if (!root) throw new Error("Fixture hierarchy root is missing");
    root.model_calls_used += 1;
    root.updated_at = new Date().toISOString();
    telemetryOnly.updated_at = root.updated_at;

    expect(canReplayInitialModelRequest(checkpoint, telemetryOnly)).toBe(true);

    const inputJournalChanged = structuredClone(telemetryOnly);
    inputJournalChanged.context_memory = {
      ...inputJournalChanged.context_memory,
      active_scope_id: inputJournalChanged.root_id,
      active_estimated_tokens: 120,
      scopes: {
        [inputJournalChanged.root_id]: {
          scope_id: inputJournalChanged.root_id,
          agent_id: inputJournalChanged.root_id,
          agent_name: "Mission Coordinator",
          raw_item_count: 1,
          raw_chain_hash: "a".repeat(64),
          compacted_item_count: 0,
          retained_item_count: 1,
          retained_chain_hash: "a".repeat(64),
          active_estimated_tokens: 120,
          compaction_count: 0,
          summary: null,
          summary_origin: null,
          summary_world_revision: null,
          summary_voxel_revision: null,
          last_compacted_at: null
        }
      }
    };
    expect(canReplayInitialModelRequest(checkpoint, inputJournalChanged)).toBe(true);

    const actionChanged = structuredClone(telemetryOnly);
    const receipt = Object.values(actionChanged.committed_actions)[0];
    if (!receipt) throw new Error("Fixture action receipt is missing");
    actionChanged.committed_actions[`${receipt.transaction_id}:new`] = {
      ...receipt,
      transaction_id: `${receipt.transaction_id}:new`
    };
    expect(canReplayInitialModelRequest(checkpoint, actionChanged)).toBe(false);

    const worldChanged = structuredClone(telemetryOnly);
    worldChanged.world.world_revision += 1;
    expect(canReplayInitialModelRequest(checkpoint, worldChanged)).toBe(false);

    const hierarchyChanged = structuredClone(telemetryOnly);
    hierarchyChanged.nodes[hierarchyChanged.root_id]!.steps_used += 1;
    expect(canReplayInitialModelRequest(checkpoint, hierarchyChanged)).toBe(false);

    const compacted = structuredClone(inputJournalChanged);
    compacted.context_memory.total_compactions = 1;
    expect(canReplayInitialModelRequest(checkpoint, compacted)).toBe(false);
  });
});
