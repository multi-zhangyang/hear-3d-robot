import { useEffect, useRef, useState } from "react";
import type { WorldSnapshot } from "../types";
import type { AuthoritativeFrameBuffer } from "./authoritative-frame-buffer";

/**
 * React telemetry intentionally runs slower than the Three.js pose consumer.
 * This keeps live HUD text current without making the application shell and
 * agent panels reconcile at physics-frame frequency.
 */
export function useAuthoritativeHudFrame(
  buffer: AuthoritativeFrameBuffer,
  fallback: WorldSnapshot,
  minimumIntervalMs = 100
): WorldSnapshot {
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
