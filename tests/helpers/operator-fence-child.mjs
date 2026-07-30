import { acquireOperatorLease } from "../../src/server/operator-lease.ts";
import { RunStore } from "../../src/persistence/run-store.ts";

const [runsDir, runDir, mode, blockText] = process.argv.slice(2);
if (!runsDir || !runDir || !mode || !blockText || !process.send) {
  throw new Error("Invalid operator fence child arguments");
}

const timing = {
  heartbeatIntervalMs: 25,
  leaseDurationMs: 150,
  reclaimConfirmationMs: 50
};
const blockMs = Number(blockText);
const lease = await acquireOperatorLease(runsDir, timing);
const store = await RunStore.open(runDir, { mutationFence: lease });
const staleCheckpoint = mode === "checkpoint" ? await store.readCheckpoint() : undefined;

process.on("message", (message) => {
  if (message !== "write") return;
  void writeAsStaleOwner();
});

await send({ type: "owned" });
Atomics.wait(new Int32Array(new SharedArrayBuffer(4)), 0, 0, blockMs);
await send({ type: "resumed" });

async function writeAsStaleOwner() {
  try {
    if (mode === "checkpoint" && staleCheckpoint) {
      await store.writeCheckpoint({
        ...staleCheckpoint,
        error: "stale-owner",
        updated_at: new Date().toISOString()
      });
    } else if (mode === "journal") {
      await store.append("events", { owner: "stale-owner" });
    } else {
      throw new Error(`Unknown fence child mode: ${mode}`);
    }
    await send({ type: "write_result", accepted: true });
  } catch (error) {
    await send({
      type: "write_result",
      accepted: false,
      error_name: error instanceof Error ? error.constructor.name : typeof error,
      error_message: error instanceof Error ? error.message : String(error)
    });
  } finally {
    await lease.release().catch(() => undefined);
    process.disconnect();
  }
}

function send(message) {
  return new Promise((resolve, reject) => {
    process.send(message, (error) => error ? reject(error) : resolve());
  });
}
