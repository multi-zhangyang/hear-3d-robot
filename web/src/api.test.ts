import { afterEach, describe, expect, it, vi } from "vitest";
import { hasPassword, setPassword, subscribeToRun } from "./api";
import type { RuntimeEvent, StreamState } from "./types";

afterEach(() => {
  vi.unstubAllGlobals();
});

describe("run event subscription", () => {
  it("keeps authentication usable in memory when session storage rejects writes", () => {
    vi.stubGlobal("sessionStorage", {
      getItem: vi.fn(() => null),
      setItem: vi.fn(() => {
        throw new Error("storage quota exceeded");
      }),
      removeItem: vi.fn(() => {
        throw new Error("storage access denied");
      })
    });

    expect(() => setPassword("temporary-secret")).not.toThrow();
    expect(hasPassword()).toBe(true);
    expect(() => setPassword("")).not.toThrow();
    expect(hasPassword()).toBe(false);
  });

  it("refreshes an unknown cursor from an authoritative snapshot before reconnecting", async () => {
    const event: RuntimeEvent = {
      event_id: "event-after-refresh",
      run_id: "run-1",
      type: "framework_event",
      at: "2026-07-30T14:00:00.000Z",
      data: { type: "response_done" },
      durable: true
    };
    const fetchMock = vi.fn()
      .mockResolvedValueOnce(new Response(JSON.stringify({ error: "Unknown event cursor" }), {
        status: 409,
        headers: { "content-type": "application/json" }
      }))
      .mockResolvedValueOnce(new Response(`data: ${JSON.stringify(event)}\n\n`, {
        status: 200,
        headers: { "content-type": "text/event-stream" }
      }));
    vi.stubGlobal("fetch", fetchMock);
    const refreshCursor = vi.fn().mockResolvedValue({ cursor: "fresh-cursor", active: true });
    const failures: Error[] = [];
    const states: StreamState[] = [];
    let unsubscribe = (): void => undefined;

    const received = new Promise<RuntimeEvent>((resolve) => {
      unsubscribe = subscribeToRun(
        "run-1",
        (next) => {
          resolve(next);
          unsubscribe();
        },
        (error) => failures.push(error),
        (state) => states.push(state),
        "expired-cursor",
        refreshCursor
      );
    });

    await expect(received).resolves.toEqual(event);
    expect(refreshCursor).toHaveBeenCalledOnce();
    expect(failures).toEqual([]);
    expect(states).toContain("connected");
    expect(fetchMock).toHaveBeenCalledTimes(2);
    expect(String(fetchMock.mock.calls[0]?.[0])).toContain("after=expired-cursor");
    expect(String(fetchMock.mock.calls[1]?.[0])).toContain("after=fresh-cursor");
    expect((fetchMock.mock.calls[1]?.[1]?.headers as Record<string, string>)["last-event-id"])
      .toBe("fresh-cursor");
  });

  it("never reconnects without a cursor while an active snapshot has no event baseline", async () => {
    const event: RuntimeEvent = {
      event_id: "first-safe-event",
      run_id: "run-2",
      type: "framework_event",
      at: "2026-07-30T14:01:00.000Z",
      data: { type: "response_done" },
      durable: true
    };
    const fetchMock = vi.fn().mockResolvedValue(new Response(
      `data: ${JSON.stringify(event)}\n\n`,
      { status: 200, headers: { "content-type": "text/event-stream" } }
    ));
    vi.stubGlobal("fetch", fetchMock);
    const refreshCursor = vi.fn()
      .mockResolvedValueOnce({ cursor: null, active: true })
      .mockResolvedValueOnce({ cursor: "safe-baseline", active: true });
    let unsubscribe = (): void => undefined;

    const received = new Promise<RuntimeEvent>((resolve) => {
      unsubscribe = subscribeToRun(
        "run-2",
        (next) => {
          resolve(next);
          unsubscribe();
        },
        () => undefined,
        undefined,
        undefined,
        refreshCursor
      );
    });

    await expect(received).resolves.toEqual(event);
    expect(refreshCursor).toHaveBeenCalledTimes(2);
    expect(fetchMock).toHaveBeenCalledOnce();
    expect(String(fetchMock.mock.calls[0]?.[0])).toContain("after=safe-baseline");
    expect((fetchMock.mock.calls[0]?.[1]?.headers as Record<string, string>)["last-event-id"])
      .toBe("safe-baseline");
  });

  it("adopts the SSE id as the durable cursor when legacy JSON omits it", async () => {
    const streamCursor = `v1:9:${"a".repeat(64)}`;
    const event: RuntimeEvent = {
      event_id: "durable-event",
      run_id: "run-3",
      type: "framework_event",
      at: "2026-07-30T14:02:00.000Z",
      data: { type: "response_done" },
      durable: true
    };
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(new Response(
      `id: ${streamCursor}\nevent: framework_event\ndata: ${JSON.stringify(event)}\n\n`,
      { status: 200, headers: { "content-type": "text/event-stream" } }
    )));
    let unsubscribe = (): void => undefined;

    const received = new Promise<RuntimeEvent>((resolve) => {
      unsubscribe = subscribeToRun(
        "run-3",
        (next) => {
          resolve(next);
          unsubscribe();
        },
        () => undefined,
        undefined,
        "baseline"
      );
    });

    await expect(received).resolves.toEqual({ ...event, cursor: streamCursor });
  });

  it("stops when the SSE id disagrees with the JSON cursor", async () => {
    const event: RuntimeEvent = {
      event_id: "durable-event",
      run_id: "run-4",
      type: "framework_event",
      at: "2026-07-30T14:03:00.000Z",
      data: {},
      durable: true,
      cursor: `v1:10:${"b".repeat(64)}`
    };
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(new Response(
      `id: v1:10:${"c".repeat(64)}\ndata: ${JSON.stringify(event)}\n\n`,
      { status: 200, headers: { "content-type": "text/event-stream" } }
    )));
    const onEvent = vi.fn();

    const failure = new Promise<Error>((resolve) => {
      subscribeToRun("run-4", onEvent, resolve, undefined, "baseline");
    });

    await expect(failure).resolves.toMatchObject({
      name: "SyntaxError",
      message: "Runtime event cursor does not match its SSE id"
    });
    expect(onEvent).not.toHaveBeenCalled();
  });

  it("stops instead of retaining an unbounded incomplete SSE record", async () => {
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(new Response(
      `data: ${"x".repeat(8 * 1024 * 1024 + 1)}`,
      { status: 200, headers: { "content-type": "text/event-stream" } }
    )));
    const onEvent = vi.fn();

    const failure = new Promise<Error>((resolve) => {
      subscribeToRun("run-large", onEvent, resolve, undefined, "baseline");
    });

    await expect(failure).resolves.toMatchObject({
      name: "SyntaxError",
      message: "Runtime event stream record exceeds the client limit"
    });
    expect(onEvent).not.toHaveBeenCalled();
  });
});
