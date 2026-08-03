import { describe, expect, it, vi } from "vitest";
import type { HumanoidWorldSnapshot } from "../types";
import { HumanoidFrameBuffer } from "./humanoid-frame-buffer";

describe("HumanoidFrameBuffer", () => {
  it("orders authoritative frames and rejects stale physics time", () => {
    const buffer = new HumanoidFrameBuffer(4);
    buffer.reset(frame(0));

    expect(buffer.push([frame(3), frame(1), frame(2)])).toBe(3);
    expect(buffer.latest?.frame).toBe(3);
    expect(buffer.push([frame(2), frame(4, 0.01)])).toBe(0);
    expect(buffer.latest?.frame).toBe(3);
  });

  it("replaces a duplicate frame and remains bounded", () => {
    const buffer = new HumanoidFrameBuffer(3);
    buffer.reset(frame(0));
    buffer.push([frame(1), frame(2), frame(3)]);
    const terminal = frame(3, 9);

    expect(buffer.push([terminal])).toBe(1);
    expect(buffer.latest).toBe(terminal);
    expect(buffer.latest?.robot.simulatedTime).toBe(9);
  });

  it("snaps a historical run to its exact newest frame", () => {
    const buffer = new HumanoidFrameBuffer();
    buffer.reset(frame(0));
    const terminal = frame(5);
    buffer.push([terminal]);

    expect(buffer.sample(10, false)).toBe(terminal);
    expect(buffer.pending).toBe(false);
  });

  it("catches a live playhead up instead of accumulating unbounded lag", () => {
    const buffer = new HumanoidFrameBuffer();
    buffer.reset(frame(0));
    buffer.sample(0, true);
    const terminal = frame(20);
    buffer.push([terminal]);

    expect(buffer.sample(1, true)?.robot.simulatedTime).toBeGreaterThan(0.25);
    expect(buffer.sample(60_000, true)).toBe(terminal);
    expect(buffer.pending).toBe(false);
  });

  it("notifies once per accepted batch", () => {
    const buffer = new HumanoidFrameBuffer();
    buffer.reset(frame(0));
    const listener = vi.fn();
    const unsubscribe = buffer.subscribe(listener);

    buffer.push([frame(1), frame(2)]);
    buffer.push([frame(1)]);
    unsubscribe();
    buffer.push([frame(3)]);

    expect(listener).toHaveBeenCalledTimes(1);
  });
});

function frame(index: number, time = index / 50): HumanoidWorldSnapshot {
  return {
    frame: index,
    worldRevision: index,
    robot: { simulatedTime: time }
  } as unknown as HumanoidWorldSnapshot;
}
