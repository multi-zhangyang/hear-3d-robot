import { useEffect, useRef, useState } from "react";
import type { HumanoidFrameBuffer } from "../stage/humanoid-frame-buffer";
import type { HumanoidWorldSnapshot } from "../types";

export function useHumanoidHudFrame(
  buffer: HumanoidFrameBuffer,
  fallback: HumanoidWorldSnapshot,
  minimumIntervalMs = 100
): HumanoidWorldSnapshot {
  const [frame, setFrame] = useState(() => buffer.latest ?? fallback);
  const fallbackRef = useRef(fallback);
  fallbackRef.current = fallback;

  useEffect(() => {
    let timeout: number | null = null;
    let lastPublished = 0;
    const publish = (): void => {
      timeout = null;
      lastPublished = performance.now();
      setFrame(buffer.latest ?? fallbackRef.current);
    };
    const schedule = (): void => {
      const remaining = minimumIntervalMs - (performance.now() - lastPublished);
      if (remaining <= 0) publish();
      else if (timeout === null) timeout = window.setTimeout(publish, remaining);
    };
    publish();
    const unsubscribe = buffer.subscribe(schedule);
    return () => {
      unsubscribe();
      if (timeout !== null) window.clearTimeout(timeout);
    };
  }, [buffer, minimumIntervalMs]);

  return frame;
}
