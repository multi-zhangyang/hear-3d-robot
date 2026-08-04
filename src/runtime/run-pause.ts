export class RunPauseRequestedError extends Error {
  readonly code = "run_pause_requested";

  constructor(message = "Run paused by operator") {
    super(message);
    this.name = "RunPauseRequestedError";
  }
}

export function isRunPauseRequested(value: unknown): value is RunPauseRequestedError {
  return value instanceof RunPauseRequestedError
    || value instanceof Error
      && "code" in value
      && value.code === "run_pause_requested";
}
