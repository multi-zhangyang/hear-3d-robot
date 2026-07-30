import { describe, expect, it } from "vitest";
import {
  parseRuntimeEventCursor,
  runtimeEventCursor,
  runtimeEventCursorMatches
} from "./event-cursor.js";

describe("durable runtime event cursors", () => {
  it("binds one journal ordinal to its run and event identity", () => {
    const cursor = runtimeEventCursor("run-a", "event-a", 42);
    const parsed = parseRuntimeEventCursor(cursor);

    expect(parsed).toMatchObject({ kind: "versioned", index: 42 });
    if (parsed.kind !== "versioned") throw new Error("Expected a versioned cursor");
    expect(runtimeEventCursorMatches(parsed, "run-a", "event-a")).toBe(true);
    expect(runtimeEventCursorMatches(parsed, "run-b", "event-a")).toBe(false);
    expect(runtimeEventCursorMatches(parsed, "run-a", "event-b")).toBe(false);
  });

  it("distinguishes legacy IDs from malformed reserved cursors", () => {
    expect(parseRuntimeEventCursor("old-event-id")).toEqual({ kind: "legacy" });
    expect(parseRuntimeEventCursor("v1:not-an-index:broken")).toEqual({ kind: "invalid" });
    expect(() => runtimeEventCursor("run", "event", Number.MAX_SAFE_INTEGER + 1))
      .toThrow(/safe integer/);
  });
});
