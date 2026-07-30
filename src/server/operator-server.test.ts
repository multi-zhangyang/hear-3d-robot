import { EventEmitter } from "node:events";
import { randomUUID } from "node:crypto";
import { cp, mkdtemp, rm, writeFile } from "node:fs/promises";
import { hostname, tmpdir } from "node:os";
import { basename, join, resolve } from "node:path";
import { describe, expect, it, vi } from "vitest";
import { loadRuntimeCatalog } from "../config/load.js";
import { createOperatorServer, EventStreamWriter, runtimeEventRecord } from "./operator-server.js";

const RUN_FIXTURE = resolve(
  process.cwd(),
  "tests/fixtures/runs/20000101T000000Z_fetch_red_block_00000000"
);

describe("Operator API", () => {
  it("does not publish a resumable SSE cursor for live-only frames", () => {
    const live = runtimeEventRecord({
      event_id: "live-frame",
      run_id: "run",
      type: "world_frames",
      at: "2026-07-30T00:00:00.000Z",
      data: { frames: [] },
      durable: false
    });
    const durable = runtimeEventRecord({
      event_id: "committed",
      run_id: "run",
      type: "action_committed",
      at: "2026-07-30T00:00:01.000Z",
      data: {},
      durable: true
    });

    expect(live).not.toContain("id: live-frame");
    expect(live).toContain("event: world_frames");
    expect(durable).toContain("id: committed");
  });

  it("protects runtime data and exposes the current hierarchy contract", async () => {
    const runsDir = await mkdtemp(join(tmpdir(), "hear-api-"));
    const catalog = await loadRuntimeCatalog();
    const app = await createOperatorServer({
      server: { host: "127.0.0.1", port: 8765, password: "secret", runsDir },
      catalog,
      providerError: "AI_MODEL is required",
      dev: true
    });

    try {
      const unauthorized = await app.inject({ method: "GET", url: "/api/bootstrap" });
      expect(unauthorized.statusCode).toBe(401);

      const headers = { authorization: "Bearer secret" };
      const bootstrap = await app.inject({ method: "GET", url: "/api/bootstrap", headers });
      expect(bootstrap.statusCode).toBe(200);
      expect(bootstrap.json()).toMatchObject({
        provider: { configured: false, error: "AI_MODEL is required" }
      });
      expect(bootstrap.json().capability_catalog).toEqual(expect.arrayContaining([
        "sense_scene",
        "survey_terrain",
        "plan_base_path",
        "execute_base_plan",
        "set_joint_targets",
        "set_gripper_target"
      ]));
      expect(bootstrap.json().scenarios).toEqual(expect.arrayContaining([
        expect.objectContaining({
          id: "voxel_expanse",
          kind: "generated",
          extent: { width: 80, depth: 80 },
          objects: expect.arrayContaining([expect.objectContaining({ id: "red_block" })])
        }),
        expect.objectContaining({
          id: "voxel_highlands",
          kind: "generated",
          extent: { width: 96, depth: 96 }
        })
      ]));

      const invalidGoal = await app.inject({
        method: "POST",
        url: "/api/runs",
        headers,
        payload: {
          mission: "Reach the arrival coordinate",
          scenario_id: "open_navigation",
          goal: {
            predicates: [
              { type: "robot_at", target: { x: 3, y: 0, z: 1 }, tolerance: 0.25 }
            ]
          },
          confirmed: true
        }
      });
      expect(invalidGoal.statusCode).toBe(400);

      const start = await app.inject({
        method: "POST",
        url: "/api/runs",
        headers,
        payload: {
          mission: "Reach the arrival coordinate",
          scenario_id: "open_navigation",
          goal: {
            summary: "Robot reaches the requested coordinate.",
            predicates: [
              { type: "robot_at", target: { x: 3, y: 0, z: 1 }, tolerance: 0.25 }
            ]
          },
          confirmed: true
        }
      });
      expect(start.statusCode).toBe(503);
      expect(start.json()).toEqual({ error: "AI_MODEL is required" });
    } finally {
      await app.close();
    }
  });

  it("loads bounded journal tails with an event high-water mark", async () => {
    const catalog = await loadRuntimeCatalog();
    const app = await createOperatorServer({
      server: {
        host: "127.0.0.1",
        port: 8765,
        password: "",
        runsDir: resolve(process.cwd(), "tests/fixtures/runs")
      },
      catalog,
      providerError: "AI_MODEL is required",
      dev: true
    });

    try {
      const response = await app.inject({
        method: "GET",
        url: "/api/runs/20000101T000000Z_fetch_red_block_00000000?actions=3&provider=4&framework=2"
      });
      expect(response.statusCode).toBe(200);
      const details = response.json();
      expect(details.actions.length).toBeLessThanOrEqual(3);
      expect(details.provider.length).toBeLessThanOrEqual(4);
      expect(details.framework).toHaveLength(2);
      expect(details.event_cursor).toEqual(expect.any(String));

      const unknownCursor = await app.inject({
        method: "GET",
        url: "/api/runs/20000101T000000Z_fetch_red_block_00000000/events?after=unknown"
      });
      expect(unknownCursor.statusCode).toBe(409);
      expect(unknownCursor.json().error).toMatch(/Unknown event cursor/);

      const missingRun = await app.inject({
        method: "GET",
        url: "/api/runs/missing_run/events"
      });
      expect(missingRun.statusCode).toBe(404);
    } finally {
      await app.close();
    }
  });

  it("closes live event streams before draining the Operator", async () => {
    const catalog = await loadRuntimeCatalog();
    const app = await createOperatorServer({
      server: {
        host: "127.0.0.1",
        port: 0,
        password: "",
        runsDir: resolve(process.cwd(), "tests/fixtures/runs")
      },
      catalog,
      providerError: "AI_MODEL is required",
      dev: true
    });
    let closed = false;
    try {
      const address = await app.listen({ host: "127.0.0.1", port: 0 });
      const response = await fetch(
        `${address}/api/runs/20000101T000000Z_fetch_red_block_00000000/events`
      );
      expect(response.status).toBe(200);
      expect(response.headers.get("content-type")).toContain("text/event-stream");

      const outcome = await Promise.race([
        app.close().then(() => "closed"),
        new Promise<string>((resolveTimeout) => {
          const timer = setTimeout(() => resolveTimeout("timeout"), 2_000);
          timer.unref();
        })
      ]);
      expect(outcome).toBe("closed");
      closed = true;
    } finally {
      if (!closed) await app.close();
    }
  });

  it("fences RunStore repairs and mutating requests after the Operator lease is replaced", async () => {
    const runsDir = await mkdtemp(join(tmpdir(), "hear-operator-fencing-"));
    await cp(RUN_FIXTURE, join(runsDir, basename(RUN_FIXTURE)), { recursive: true });
    const catalog = await loadRuntimeCatalog();
    const app = await createOperatorServer({
      server: { host: "127.0.0.1", port: 0, password: "", runsDir },
      catalog,
      providerError: "AI_MODEL is required",
      dev: true
    });
    const replacement = {
      version: 2,
      token: randomUUID(),
      pid: process.pid,
      hostname: hostname(),
      started_at: new Date().toISOString(),
      heartbeat_interval_ms: 1_000,
      lease_duration_ms: 10_000
    };
    await writeFile(
      join(runsDir, ".operator.lock"),
      `${JSON.stringify(replacement)}\n`,
      "utf8"
    );
    try {
      const journal = await app.inject({
        method: "GET",
        url: `/api/runs/${basename(RUN_FIXTURE)}/journal?name=events&from=0&limit=1`
      });
      expect(journal.statusCode).toBe(503);
      expect(journal.json().error).toMatch(/lease token was replaced/);

      const response = await app.inject({
        method: "POST",
        url: "/api/runs/missing/stop",
        payload: { confirmed: true }
      });
      expect(response.statusCode).toBe(503);
      expect(response.json().error).toMatch(/lease token was replaced/);
    } finally {
      await app.close();
      await rm(runsDir, { recursive: true, force: true });
    }
  });

  it("pauses ordered writes until a backpressured stream drains", async () => {
    const target = new ControlledEventStreamTarget([false, true]);
    const failures: Error[] = [];
    const writer = new EventStreamWriter(target, (error) => failures.push(error), {
      drainTimeoutMs: 1_000
    });

    const first = writer.write("first\n\n");
    const second = writer.write("second\n\n");
    expect(target.records).toEqual(["first\n\n"]);

    target.emit("drain");
    await expect(Promise.all([first, second])).resolves.toEqual([undefined, undefined]);
    expect(target.records).toEqual(["first\n\n", "second\n\n"]);
    expect(failures).toEqual([]);
    writer.close();
  });

  it("rejects pending writes on disconnect and bounds a stalled client queue", async () => {
    const disconnectedTarget = new ControlledEventStreamTarget([false]);
    const disconnectedWriter = new EventStreamWriter(disconnectedTarget, () => undefined);
    const pending = disconnectedWriter.write("pending\n\n");
    const disconnected = new Error("client disconnected");
    disconnectedWriter.close(disconnected);
    await expect(pending).rejects.toBe(disconnected);
    disconnectedTarget.emit("drain");
    expect(disconnectedTarget.records).toEqual(["pending\n\n"]);

    const stalledTarget = new ControlledEventStreamTarget([false]);
    const failures: Error[] = [];
    const stalledWriter = new EventStreamWriter(stalledTarget, (error) => failures.push(error), {
      maxPendingRecords: 2,
      maxPendingBytes: 64,
      drainTimeoutMs: 1_000
    });
    const first = stalledWriter.write("one\n\n");
    const second = stalledWriter.write("two\n\n");
    const overflow = stalledWriter.write("three\n\n");
    const outcomes = await Promise.allSettled([first, second, overflow]);
    expect(outcomes.every((outcome) => outcome.status === "rejected")).toBe(true);
    expect(failures).toHaveLength(1);
    expect(failures[0]?.message).toMatch(/too slow/);
    expect(stalledTarget.records).toEqual(["one\n\n"]);
  });

  it("fails a backpressured stream when drain never arrives", async () => {
    vi.useFakeTimers();
    try {
      const target = new ControlledEventStreamTarget([false]);
      const failures: Error[] = [];
      const writer = new EventStreamWriter(target, (error) => failures.push(error), {
        drainTimeoutMs: 50
      });
      const pending = writer.write("blocked\n\n");
      const observed = pending.catch((error: unknown) => error);

      await vi.advanceTimersByTimeAsync(50);

      await expect(observed).resolves.toMatchObject({ message: "Event stream drain timed out" });
      expect(failures).toHaveLength(1);
      expect(target.listenerCount("drain")).toBe(0);
    } finally {
      vi.useRealTimers();
    }
  });
});

class ControlledEventStreamTarget extends EventEmitter {
  readonly records: string[] = [];
  readonly #writeResults: boolean[];

  constructor(writeResults: boolean[]) {
    super();
    this.#writeResults = [...writeResults];
  }

  write(record: string): boolean {
    this.records.push(record);
    return this.#writeResults.shift() ?? true;
  }
}
