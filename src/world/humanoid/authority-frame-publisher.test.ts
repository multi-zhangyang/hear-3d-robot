import { describe, expect, it } from "vitest";
import { HumanoidAuthorityFramePublisher } from "./authority-frame-publisher.js";

describe("HumanoidAuthorityFramePublisher", () => {
  it("flushes queued frames without closing the publisher", async () => {
    const entered = deferred();
    const release = deferred();
    const published: number[] = [];
    const publisher = new HumanoidAuthorityFramePublisher<number>();
    publisher.enqueue({
      source: "stationary",
      commandId: null,
      sink: async (frame) => {
        published.push(frame);
        if (frame === 1) {
          entered.resolve();
          await release.promise;
        }
      },
      snapshot: 1
    });
    await entered.promise;
    let flushed = false;
    const firstFlush = publisher.flush().then(() => {
      flushed = true;
    });
    await Promise.resolve();
    expect(flushed).toBe(false);
    release.resolve();
    await firstFlush;

    publisher.enqueue({
      source: "stationary",
      commandId: null,
      sink: (frame) => {
        published.push(frame);
      },
      snapshot: 2
    });
    await publisher.flush();

    expect(published).toEqual([1, 2]);
    await publisher.dispose();
  });

  it("coalesces stationary backlog while a sink is slow", async () => {
    const entered = deferred();
    const release = deferred();
    const published: number[] = [];
    const publisher = new HumanoidAuthorityFramePublisher<number>({
      maximumQueuedFrames: 2
    });
    const sink = async (frame: number): Promise<void> => {
      published.push(frame);
      if (frame === 1) {
        entered.resolve();
        await release.promise;
      }
    };
    publisher.enqueue({ source: "stationary", commandId: null, sink, snapshot: 1 });
    await entered.promise;
    publisher.enqueue({ source: "stationary", commandId: null, sink, snapshot: 2 });
    publisher.enqueue({ source: "stationary", commandId: null, sink, snapshot: 3 });
    publisher.enqueue({ source: "stationary", commandId: null, sink, snapshot: 4 });
    release.resolve();
    await publisher.dispose();

    expect(published).toEqual([1, 4]);
  });

  it("reports bounded execution overflow without blocking physical completion", async () => {
    const entered = deferred();
    const release = deferred();
    const publisher = new HumanoidAuthorityFramePublisher<number>({
      maximumQueuedFrames: 1
    });
    const barrier = publisher.openCommand("motion-a");
    const sink = async (): Promise<void> => {
      entered.resolve();
      await release.promise;
    };
    publisher.enqueue({ source: "motion", commandId: "motion-a", sink, snapshot: 1 });
    await entered.promise;
    publisher.enqueue({ source: "motion", commandId: "motion-a", sink, snapshot: 2 });
    publisher.enqueue({ source: "motion", commandId: "motion-a", sink, snapshot: 3 });
    publisher.closeCommand("motion-a");
    release.resolve();

    await expect(barrier).rejects.toThrow("publication queue exceeded");
    await publisher.dispose();
  });
});

function deferred(): { promise: Promise<void>; resolve(): void } {
  let resolve!: () => void;
  const promise = new Promise<void>((done) => {
    resolve = done;
  });
  return { promise, resolve };
}
