import { timingSafeEqual } from "node:crypto";
import { access } from "node:fs/promises";
import { resolve } from "node:path";
import { fileURLToPath } from "node:url";
import fastifyStatic from "@fastify/static";
import Fastify, { type FastifyInstance } from "fastify";
import { z, ZodError } from "zod";
import type { ProviderConfig, RuntimeCatalog, ServerConfig } from "../config/load.js";
import { GoalSchema } from "../domain/schema.js";
import { capabilityCatalog } from "../harness/agents.js";
import type { RuntimeEvent } from "../harness/runtime-context.js";
import { providerIdentity } from "../model/factory.js";
import type { MutationFence } from "../persistence/mutation-fence.js";
import {
  acquireOperatorLease,
  type OperatorLease,
  OperatorLeaseLostError
} from "./operator-lease.js";
import { RunConflictError, RunManager } from "./run-manager.js";

const bundledWebRoot = fileURLToPath(new URL("../../web/dist/", import.meta.url));

const StartRunInput = z.object({
  mission: z.string().trim().min(1),
  scenario_id: z.string().min(1),
  goal: GoalSchema,
  seed: z.number().int().min(0).max(0xffff_ffff).optional(),
  confirmed: z.literal(true)
}).strict();

const ResumeInput = z.object({ confirmed: z.literal(true) }).strict();
const StopInput = z.object({ confirmed: z.literal(true) }).strict();
const RunParams = z.object({ runId: z.string().min(1).max(160).regex(/^[a-zA-Z0-9_-]+$/) });
const DetailsQuery = z.object({
  actions: z.coerce.number().int().min(1).max(5000).default(500),
  provider: z.coerce.number().int().min(1).max(5000).default(400),
  framework: z.coerce.number().int().min(1).max(5000).default(300)
});
const EventsQuery = z.object({
  after: z.string().min(1).max(256).regex(/^[^\r\n\0]+$/).optional()
});
const JournalQuery = z.object({
  name: z.enum(["events", "provider", "framework", "actions", "hierarchy", "checker"]),
  from: z.coerce.number().int().nonnegative().default(0),
  limit: z.coerce.number().int().min(1).max(5000).default(500)
});

export async function createOperatorServer(input: {
  server: ServerConfig;
  catalog: RuntimeCatalog;
  provider?: ProviderConfig;
  providerError?: string;
  dev?: boolean;
}): Promise<FastifyInstance> {
  const app = Fastify({
    logger: {
      level: "info",
      redact: ["req.headers.authorization", "req.headers.x-hear-password"]
    }
  });
  let lease: OperatorLease | undefined;
  const mutationFence: MutationFence = {
    runMutation: (operation) => {
      const currentLease = lease;
      if (!currentLease) throw new OperatorLeaseLostError("Operator lease is unavailable");
      return currentLease.runMutation(operation);
    }
  };
  const manager = new RunManager({
    runsDir: input.server.runsDir,
    catalog: input.catalog,
    ...(input.provider ? { provider: input.provider } : {}),
    ...(input.providerError ? { providerError: input.providerError } : {}),
    mutationFence
  });
  const activeStreams = new Set<() => void>();
  let leaseLossDrain: Promise<void> | undefined;
  app.addHook("onRequest", async (request, reply) => {
    if (!request.url.startsWith("/api/") || request.url === "/api/health") return;
    if (input.server.password) {
      const supplied = bearer(request.headers.authorization)
        ?? stringHeader(request.headers["x-hear-password"]);
      if (!supplied || !sameSecret(supplied, input.server.password)) {
        await reply.code(401).send({ error: "Authentication required" });
        return;
      }
    }
    if (request.method !== "GET" && request.method !== "HEAD") {
      const currentLease = lease;
      if (!currentLease) throw new OperatorLeaseLostError("Operator lease is unavailable");
      await currentLease.assertOwned();
    }
  });

  app.get("/api/health", async () => ({ status: "ok" }));

  app.get("/api/bootstrap", async () => ({
    provider: input.provider
      ? { configured: true, ...providerIdentity(input.provider) }
      : { configured: false, error: input.providerError ?? "Provider is not configured" },
    authentication_required: Boolean(input.server.password),
    capability_catalog: capabilityCatalog(),
    // A generated world has no coordinates until a seed exists, so the console
    // is told what a world contains and how large it is, not where anything sits.
    scenarios: Object.entries(input.catalog.templates).map(([id, template]) => {
      const source = template.kind === "authored" ? template.scenario : template.generate;
      return {
        id,
        title: template.title,
        kind: template.kind,
        extent: template.kind === "authored"
          ? { width: template.scenario.bounds.width, depth: template.scenario.bounds.depth }
          : {
              width: template.generate.terrain.size * template.generate.terrain.cell,
              depth: template.generate.terrain.size * template.generate.terrain.cell
            },
        objects: source.objects.map(({ id: objectId, kind, color }) => ({ id: objectId, kind, color })),
        zones: source.zones.map(({ id: zoneId, color }) => ({ id: zoneId, color })),
        suggested_goal: source.default_goal
      };
    })
  }));

  app.get("/api/runs", async () => ({ runs: await manager.list() }));

  app.post("/api/runs", async (request, reply) => {
    const body = StartRunInput.parse(request.body);
    if (!input.catalog.templates[body.scenario_id]) {
      return reply.code(404).send({ error: `Unknown scenario: ${body.scenario_id}` });
    }
    const runId = await manager.start({
      mission: body.mission,
      scenarioId: body.scenario_id,
      goal: body.goal,
      ...(body.seed === undefined ? {} : { seed: body.seed })
    });
    return reply.code(202).send({ run_id: runId, status: "running" });
  });

  app.get("/api/runs/:runId", async (request) => {
    const { runId } = RunParams.parse(request.params);
    const limits = DetailsQuery.parse(request.query);
    return manager.details(runId, limits);
  });

  app.post("/api/runs/:runId/resume", async (request, reply) => {
    ResumeInput.parse(request.body);
    const { runId } = RunParams.parse(request.params);
    const resumedId = await manager.resume(runId);
    return reply.code(202).send({ run_id: resumedId, status: "running" });
  });

  app.post("/api/runs/:runId/stop", async (request, reply) => {
    StopInput.parse(request.body);
    const { runId } = RunParams.parse(request.params);
    manager.stop(runId);
    return reply.code(202).send({ run_id: runId, status: "stopping" });
  });

  app.get("/api/runs/:runId/journal", async (request) => {
    const { runId } = RunParams.parse(request.params);
    const { name, from, limit } = JournalQuery.parse(request.query);
    const page = await manager.journal(runId, name, from, limit);
    return {
      name,
      ...page
    };
  });

  app.get("/api/runs/:runId/events", async (request, reply) => {
    const { runId } = RunParams.parse(request.params);
    const query = EventsQuery.parse(request.query);
    const after = query.after ?? stringHeader(request.headers["last-event-id"]);
    let writeEvent: ((event: RuntimeEvent) => Promise<void>) | undefined;
    let writer: EventStreamWriter | undefined;
    let unsubscribe = (): void => undefined;
    let heartbeat: ReturnType<typeof setInterval> | undefined;
    let closed = false;
    const abort = new AbortController();
    const close = (
      reason = new Error("Event stream closed"),
      force = false
    ): void => {
      if (closed) return;
      closed = true;
      abort.abort(reason);
      if (heartbeat) clearInterval(heartbeat);
      unsubscribe();
      writer?.close(reason);
      activeStreams.delete(close);
      if (force && !reply.raw.destroyed) {
        reply.raw.destroy(reason);
      } else if (reply.raw.headersSent && !reply.raw.destroyed && !reply.raw.writableEnded) {
        reply.raw.end();
      }
    };
    activeStreams.add(close);
    reply.raw.once("close", () => close());
    reply.raw.once("error", (error) => close(error));
    if (reply.raw.destroyed) close();
    try {
      const startResponse = async (): Promise<void> => {
        abort.signal.throwIfAborted();
        reply.hijack();
        reply.raw.writeHead(200, {
          "content-type": "text/event-stream",
          "cache-control": "no-cache, no-transform",
          connection: "keep-alive",
          "x-accel-buffering": "no"
        });
        writer = new EventStreamWriter(reply.raw, (error) => close(error, true));
        await writer.write(": connected\n\n");
        abort.signal.throwIfAborted();
        heartbeat = setInterval(() => {
          if (!closed) void writer?.write(": heartbeat\n\n").catch(() => undefined);
        }, 15_000);
        writeEvent = (event) => {
          if (closed || !writer) return Promise.reject(abort.signal.reason);
          return writer.write(runtimeEventRecord(event));
        };
      };
      unsubscribe = await manager.subscribe(runId, after, async (event) => {
        if (closed) return;
        if (!writeEvent) throw new Error("Event stream received data before its cursor was ready");
        await writeEvent(event);
      }, abort.signal, startResponse);
      if (closed || reply.raw.destroyed) {
        unsubscribe();
        close();
        return;
      }
    } catch (error) {
      const disconnected = abort.signal.aborted;
      close();
      if (disconnected) return;
      throw error;
    }
  });

  if (!input.dev) {
    const webRoot = bundledWebRoot;
    try {
      await access(resolve(webRoot, "index.html"));
      // `wildcard: true` resolves each asset per request instead of registering
      // a route per file at boot. Without it a console rebuilt while the server
      // is up serves an index.html referencing hashed assets that have no route,
      // and the browser gets index.html back for every one of them.
      await app.register(fastifyStatic, { root: webRoot, wildcard: true });
      app.setNotFoundHandler(async (request, reply) => {
        if (request.url.startsWith("/api/")) {
          return reply.code(404).send({ error: "Not found" });
        }
        return reply.sendFile("index.html");
      });
    } catch {
      app.get("/", async (_request, reply) => reply.code(503).send({
        error: "Operator UI has not been built"
      }));
    }
  }

  app.setErrorHandler((error, _request, reply) => {
    if (error instanceof ZodError) {
      void reply.code(400).send({ error: "Invalid request", issues: error.issues });
      return;
    }
    if (error instanceof RunConflictError) {
      void reply.code(409).send({ error: error.message });
      return;
    }
    if (isMissingFile(error)) {
      void reply.code(404).send({ error: "Run not found" });
      return;
    }
    const status = typeof error === "object" && error !== null
      && "statusCode" in error && typeof error.statusCode === "number"
      ? error.statusCode
      : 500;
    const message = error instanceof Error ? error.message : String(error);
    void reply.code(status).send({ error: message });
  });

  lease = await acquireOperatorLease(input.server.runsDir);
  const onLeaseLost = (): void => {
    for (const close of [...activeStreams]) close();
    leaseLossDrain ??= manager.drain("Operator lost the runs-directory lease");
    void leaseLossDrain.catch((error) => {
      app.log.error({ err: error }, "Failed to drain after losing the Operator lease");
    });
  };
  lease.signal.addEventListener("abort", onLeaseLost, { once: true });
  try {
    await lease.assertOwned();
    await manager.recoverOrphanedRuns();
    await lease.assertOwned();
  } catch (error) {
    lease.signal.removeEventListener("abort", onLeaseLost);
    await lease.release();
    throw error;
  }
  app.addHook("preClose", async () => {
    for (const close of [...activeStreams]) close();
    await (leaseLossDrain ?? manager.drain());
  });
  app.addHook("onClose", async () => {
    lease.signal.removeEventListener("abort", onLeaseLost);
    await lease.release();
  });
  return app;
}

function bearer(value: string | undefined): string | undefined {
  if (!value?.startsWith("Bearer ")) return undefined;
  return value.slice("Bearer ".length);
}

function stringHeader(value: string | string[] | undefined): string | undefined {
  return typeof value === "string" ? value : value?.[0];
}

function sameSecret(left: string, right: string): boolean {
  const leftBuffer = Buffer.from(left);
  const rightBuffer = Buffer.from(right);
  return leftBuffer.length === rightBuffer.length && timingSafeEqual(leftBuffer, rightBuffer);
}

export function runtimeEventRecord(event: RuntimeEvent): string {
  const cursor = event.durable === false ? "" : `id: ${event.event_id}\n`;
  return `${cursor}event: ${event.type}\ndata: ${JSON.stringify(event)}\n\n`;
}

function isMissingFile(error: unknown): boolean {
  return error instanceof Error && "code" in error && error.code === "ENOENT";
}

interface EventStreamTarget {
  write(record: string): boolean;
  once(event: "drain", listener: () => void): unknown;
  off(event: "drain", listener: () => void): unknown;
}

export interface EventStreamWriterOptions {
  maxPendingRecords?: number;
  maxPendingBytes?: number;
  drainTimeoutMs?: number;
}

interface PendingEventStreamWrite {
  record: string;
  bytes: number;
  resolve: () => void;
  reject: (reason: Error) => void;
}

/** Serializes SSE records without allowing a slow socket to grow memory indefinitely. */
export class EventStreamWriter {
  readonly #target: EventStreamTarget;
  readonly #onFailure: (error: Error) => void;
  readonly #maxPendingRecords: number;
  readonly #maxPendingBytes: number;
  readonly #drainTimeoutMs: number;
  readonly #pending: PendingEventStreamWrite[] = [];
  #pendingBytes = 0;
  #waitingForDrain = false;
  #drainTimer: ReturnType<typeof setTimeout> | undefined;
  #closedError: Error | undefined;

  constructor(
    target: EventStreamTarget,
    onFailure: (error: Error) => void,
    options: EventStreamWriterOptions = {}
  ) {
    this.#target = target;
    this.#onFailure = onFailure;
    this.#maxPendingRecords = options.maxPendingRecords ?? 256;
    this.#maxPendingBytes = options.maxPendingBytes ?? 1024 * 1024;
    this.#drainTimeoutMs = options.drainTimeoutMs ?? 30_000;
    if (
      !Number.isSafeInteger(this.#maxPendingRecords) || this.#maxPendingRecords < 1
      || !Number.isSafeInteger(this.#maxPendingBytes) || this.#maxPendingBytes < 1
      || !Number.isSafeInteger(this.#drainTimeoutMs) || this.#drainTimeoutMs < 1
    ) {
      throw new Error("Event stream writer limits must be positive integers");
    }
  }

  write(record: string): Promise<void> {
    if (this.#closedError) return Promise.reject(this.#closedError);
    const bytes = Buffer.byteLength(record);
    if (
      this.#pending.length >= this.#maxPendingRecords
      || bytes > this.#maxPendingBytes - this.#pendingBytes
    ) {
      const error = new Error("Event stream client is too slow");
      this.#fail(error);
      return Promise.reject(error);
    }

    return new Promise<void>((resolveWrite, rejectWrite) => {
      this.#pending.push({ record, bytes, resolve: resolveWrite, reject: rejectWrite });
      this.#pendingBytes += bytes;
      this.#pump();
    });
  }

  close(reason = new Error("Event stream closed")): void {
    this.#settle(reason);
  }

  readonly #onDrain = (): void => {
    if (this.#closedError || !this.#waitingForDrain) return;
    this.#waitingForDrain = false;
    this.#clearDrainTimer();
    this.#resolveFirst();
    this.#pump();
  };

  #pump(): void {
    if (this.#closedError || this.#waitingForDrain) return;
    while (this.#pending.length > 0 && !this.#closedError) {
      const next = this.#pending[0]!;
      let accepted: boolean;
      try {
        accepted = this.#target.write(next.record);
      } catch (error) {
        this.#fail(error instanceof Error ? error : new Error(String(error)));
        return;
      }
      if (accepted) {
        this.#resolveFirst();
        continue;
      }
      this.#waitingForDrain = true;
      this.#target.once("drain", this.#onDrain);
      this.#drainTimer = setTimeout(() => {
        this.#fail(new Error("Event stream drain timed out"));
      }, this.#drainTimeoutMs);
      this.#drainTimer.unref();
      return;
    }
  }

  #resolveFirst(): void {
    const completed = this.#pending.shift();
    if (!completed) return;
    this.#pendingBytes -= completed.bytes;
    completed.resolve();
  }

  #fail(error: Error): void {
    if (this.#closedError) return;
    this.#settle(error);
    this.#onFailure(error);
  }

  #settle(error: Error): void {
    if (this.#closedError) return;
    this.#closedError = error;
    if (this.#waitingForDrain) {
      this.#target.off("drain", this.#onDrain);
      this.#waitingForDrain = false;
    }
    this.#clearDrainTimer();
    const pending = this.#pending.splice(0);
    this.#pendingBytes = 0;
    for (const write of pending) write.reject(error);
  }

  #clearDrainTimer(): void {
    if (!this.#drainTimer) return;
    clearTimeout(this.#drainTimer);
    this.#drainTimer = undefined;
  }
}
