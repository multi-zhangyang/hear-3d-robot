export interface HumanoidControlStepPacer {
  waitForNextStep(): Promise<void>;
}

export function createHumanoidControlStepPacer(input: {
  controlStepSeconds: number;
  realtime: boolean;
  signal?: AbortSignal;
}): HumanoidControlStepPacer {
  if (!Number.isFinite(input.controlStepSeconds) || input.controlStepSeconds <= 0) {
    throw new Error("Humanoid execution pacing requires a positive control step");
  }
  if (!input.realtime) {
    return {
      waitForNextStep: async () => {
        input.signal?.throwIfAborted();
      }
    };
  }
  const intervalMilliseconds = input.controlStepSeconds * 1_000;
  let nextBoundary = performance.now() + intervalMilliseconds;
  return {
    waitForNextStep: async () => {
      input.signal?.throwIfAborted();
      const now = performance.now();
      if (now >= nextBoundary) {
        nextBoundary = now + intervalMilliseconds;
      }
      const delay = Math.max(0, nextBoundary - now);
      await abortableDelay(delay, input.signal);
      nextBoundary += intervalMilliseconds;
    }
  };
}

function abortableDelay(milliseconds: number, signal?: AbortSignal): Promise<void> {
  if (!signal) {
    return new Promise((resolve) => setTimeout(resolve, milliseconds));
  }
  signal.throwIfAborted();
  return new Promise<void>((resolve, reject) => {
    const timer = setTimeout(() => {
      signal.removeEventListener("abort", abort);
      resolve();
    }, milliseconds);
    const abort = (): void => {
      clearTimeout(timer);
      signal.removeEventListener("abort", abort);
      reject(signal.reason);
    };
    signal.addEventListener("abort", abort, { once: true });
  });
}
