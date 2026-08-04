import { describe, expect, it } from "vitest";
import {
  HumanoidAuthorityAdmissionError,
  HumanoidAuthorityLoop,
  type HumanoidAuthorityCommand
} from "./authority-loop.js";

describe("HumanoidAuthorityLoop", () => {
  it("serializes stationary and queued command steps through one writer", async () => {
    let revision = 0;
    let activeWriters = 0;
    let maximumWriters = 0;
    const loop = new HumanoidAuthorityLoop<number>({
      identity: () => ({ revision, stateSha256: digest(revision) }),
      stationaryStep: async () => {
        activeWriters += 1;
        maximumWriters = Math.max(maximumWriters, activeWriters);
        revision += 1;
        activeWriters -= 1;
        return revision;
      }
    });
    let commandFrames = 0;
    const handle = await loop.submit(command({
      admissionRevision: 0,
      step: async () => {
        activeWriters += 1;
        maximumWriters = Math.max(maximumWriters, activeWriters);
        revision += 1;
        commandFrames += 1;
        activeWriters -= 1;
        return commandFrames === 2
          ? { snapshot: revision, done: true, result: "complete" }
          : { snapshot: revision, done: false };
      }
    }));

    const [first, second] = await Promise.all([loop.tick(), loop.tick()]);
    expect([first.source, second.source]).toEqual(["motion", "motion"]);
    await expect(handle.result).resolves.toBe("complete");
    expect(maximumWriters).toBe(1);
    expect((await loop.tick()).source).toBe("stationary");
    await loop.dispose();
  });

  it("publishes concurrent ticks in authoritative commit order", async () => {
    let revision = 0;
    const published: number[] = [];
    const loop = new HumanoidAuthorityLoop<number>({
      identity: () => ({ revision, stateSha256: digest(revision) }),
      stationaryStep: async () => ++revision
    });
    const handle = await loop.submit({
      ...command({
        admissionRevision: 0,
        step: async () => ({
          snapshot: ++revision,
          done: revision === 2,
          ...(revision === 2 ? { result: "complete" } : {})
        })
      }),
      frameSink: (snapshot) => {
        published.push(snapshot);
      }
    });

    await Promise.all([loop.tick(), loop.tick()]);
    await expect(handle.result).resolves.toBe("complete");
    await expect(handle.publication).resolves.toBeUndefined();
    expect(published).toEqual([1, 2]);
    await loop.dispose();
  });

  it("atomically rejects stale revision or deterministic state admission", async () => {
    let revision = 3;
    const loop = new HumanoidAuthorityLoop<number>({
      identity: () => ({ revision, stateSha256: digest(revision) }),
      stationaryStep: async () => ++revision
    });

    await expect(loop.submit(command({
      admissionRevision: 2,
      step: async () => ({ done: true, result: "unreachable" })
    }))).rejects.toBeInstanceOf(HumanoidAuthorityAdmissionError);
    await expect(loop.submit({
      ...command({
        admissionRevision: 3,
        step: async () => ({ done: true, result: "unreachable" })
      }),
      admission: { revision: 3, stateSha256: digest(99) }
    })).rejects.toBeInstanceOf(HumanoidAuthorityAdmissionError);
    expect(revision).toBe(3);
    await loop.dispose();
  });

  it("releases the writer lock before awaiting frame publication", async () => {
    let revision = 0;
    const entered = deferred();
    const release = deferred();
    const loop = new HumanoidAuthorityLoop<number>({
      identity: () => ({ revision, stateSha256: digest(revision) }),
      stationaryStep: async () => ++revision
    });
    const ticking = loop.tick(async () => {
      entered.resolve();
      await release.promise;
    });
    const committed = await ticking;
    await entered.promise;

    await expect(loop.capture(() => revision)).resolves.toBe(1);
    expect(committed).toMatchObject({
      source: "stationary",
      snapshot: 1
    });
    release.resolve();
    await loop.dispose();
  });

  it("separates physical command completion from its publication barrier", async () => {
    let revision = 0;
    const entered = deferred();
    const release = deferred();
    const loop = new HumanoidAuthorityLoop<number>({
      identity: () => ({ revision, stateSha256: digest(revision) }),
      stationaryStep: async () => ++revision
    });
    const base = command({
      admissionRevision: 0,
      step: async () => ({ snapshot: ++revision, done: true, result: "physical" })
    });
    const handle = await loop.submit({
      ...base,
      frameSink: async () => {
        entered.resolve();
        await release.promise;
      }
    });
    await loop.tick();
    await entered.promise;

    await expect(handle.result).resolves.toBe("physical");
    let publicationSettled = false;
    void handle.publication.finally(() => {
      publicationSettled = true;
    });
    await Promise.resolve();
    expect(publicationSettled).toBe(false);
    release.resolve();
    await expect(handle.publication).resolves.toBeUndefined();
    await loop.dispose();
  });

  it("cancels an admitted command without executing another physical frame", async () => {
    let revision = 0;
    const loop = new HumanoidAuthorityLoop<number>({
      identity: () => ({ revision, stateSha256: digest(revision) }),
      stationaryStep: async () => ++revision
    });
    const handle = await loop.submit(command({
      admissionRevision: 0,
      step: async () => ({ snapshot: ++revision, done: false })
    }));
    const stopped = new Error("operator stopped the command");

    await expect(loop.cancel(handle.id, stopped)).resolves.toBe(true);
    await expect(handle.result).rejects.toBe(stopped);
    await expect(handle.publication).resolves.toBeUndefined();
    expect(revision).toBe(0);
    expect((await loop.tick()).source).toBe("stationary");
    expect(revision).toBe(1);
    await loop.dispose();
  });

  it("latches publication failure before committing another physical frame", async () => {
    let revision = 0;
    const publicationError = new Error("frame journal unavailable");
    const loop = new HumanoidAuthorityLoop<number>({
      identity: () => ({ revision, stateSha256: digest(revision) }),
      stationaryStep: async () => ++revision
    });
    const base = command({
      admissionRevision: 0,
      step: async () => ({ snapshot: ++revision, done: false })
    });
    const handle = await loop.submit({
      ...base,
      frameSink: () => {
        throw publicationError;
      }
    });

    await loop.tick();
    await Promise.resolve();
    await expect(loop.tick()).rejects.toBe(publicationError);
    await expect(handle.result).rejects.toBe(publicationError);
    await expect(handle.publication).rejects.toBe(publicationError);
    expect(revision).toBe(1);
    await loop.dispose();
  });
});

function command(input: {
  admissionRevision: number;
  step: HumanoidAuthorityCommand<number, string>["step"];
}): HumanoidAuthorityCommand<number, string> {
  return {
    id: `command-${input.admissionRevision}`,
    source: "motion",
    admission: {
      revision: input.admissionRevision,
      stateSha256: digest(input.admissionRevision)
    },
    admit: () => undefined,
    step: input.step
  };
}

function digest(value: number): string {
  return value.toString(16).padStart(64, "0");
}

function deferred(): { promise: Promise<void>; resolve(): void } {
  let resolve!: () => void;
  const promise = new Promise<void>((done) => {
    resolve = done;
  });
  return { promise, resolve };
}
