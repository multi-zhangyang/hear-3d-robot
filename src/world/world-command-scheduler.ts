import type {
  BodyChannel,
  JsonValue,
  WorldSnapshot
} from "../domain/schema.js";

export interface ScheduledCommandSource {
  id: string;
  agentId: string;
  agentName: string;
  skill: string;
  channels: BodyChannel[];
  focus?: JsonValue;
}

export type ActiveWorldCommand = NonNullable<WorldSnapshot["active_command"]>;
export type LastWorldCommand = NonNullable<WorldSnapshot["last_command"]>;

interface PendingAdvance {
  commandId: string;
  resolve: () => void;
  reject: (error: unknown) => void;
}

/**
 * Owns command identity, channel exclusion and physics-frame coalescing.
 *
 * It deliberately knows nothing about Rapier or robot joints. Independent body
 * commands that ask for their next frame in the same microtask are handed to
 * the world's step callback together; conflicting channels never enter the
 * active set in the first place.
 */
export class WorldCommandScheduler {
  readonly #step: (commandIds: string[]) => Promise<void>;
  readonly #active = new Map<string, ActiveWorldCommand>();
  readonly #pending = new Map<string, PendingAdvance>();
  #scheduled = false;
  #advanceChain: Promise<void> = Promise.resolve();
  #last: LastWorldCommand | null = null;
  #wallStartedMs = 0;
  #simulatedStarted = 0;

  constructor(step: (commandIds: string[]) => Promise<void>) {
    this.#step = step;
  }

  get size(): number {
    return this.#active.size;
  }

  get last(): LastWorldCommand | null {
    return this.#last ? structuredClone(this.#last) : null;
  }

  begin(command: ScheduledCommandSource, simulatedTime: number, target?: JsonValue): void {
    if (this.#active.has(command.id)) {
      throw new Error(`Command is already active: ${command.id}`);
    }
    const overlapping = [...this.#active.values()].filter((active) =>
      active.channels.some((channel) => command.channels.includes(channel))
    );
    if (overlapping.length > 0) {
      throw new Error(
        `Body channels are already active: ${overlapping.flatMap((entry) => entry.channels).join(", ")}`
      );
    }
    if (this.#active.size === 0) {
      this.#wallStartedMs = performance.now();
      this.#simulatedStarted = simulatedTime;
    }
    this.#active.set(command.id, {
      id: command.id,
      agent_id: command.agentId,
      agent_name: command.agentName,
      skill: command.skill,
      channels: [...command.channels],
      phase: "accepted",
      ...(target === undefined ? {} : { target: structuredClone(target) }),
      ...(command.focus === undefined ? {} : { focus: structuredClone(command.focus) })
    });
  }

  advance(commandId: string, phase: string): Promise<void> {
    const command = this.#active.get(commandId);
    if (!command) throw new Error(`Cannot advance inactive command: ${commandId}`);
    if (this.#pending.has(commandId)) {
      throw new Error(`Command already has a pending physics advance: ${commandId}`);
    }
    command.phase = phase;
    return new Promise<void>((resolve, reject) => {
      this.#pending.set(commandId, { commandId, resolve, reject });
      if (this.#scheduled) return;
      this.#scheduled = true;
      queueMicrotask(() => this.#queueBatch());
    });
  }

  complete(
    commandId: string,
    resultCode: string,
    accepted: boolean,
    frame: number,
    simulatedTime: number
  ): ActiveWorldCommand | null {
    const command = this.#active.get(commandId);
    if (!command) return null;
    this.#last = {
      ...command,
      accepted,
      result_code: resultCode,
      ended_at_frame: frame
    };
    this.#active.delete(commandId);
    if (this.#active.size === 0) {
      this.#wallStartedMs = 0;
      this.#simulatedStarted = simulatedTime;
    }
    return structuredClone(command);
  }

  focused(): ActiveWorldCommand | null {
    const command = [...this.#active.values()].at(-1);
    return command ? structuredClone(command) : null;
  }

  forChannel(channel: BodyChannel): ActiveWorldCommand | undefined {
    const command = [...this.#active.values()].find((entry) => entry.channels.includes(channel));
    return command ? structuredClone(command) : undefined;
  }

  get(commandId: string): ActiveWorldCommand | undefined {
    const command = this.#active.get(commandId);
    return command ? structuredClone(command) : undefined;
  }

  ids(): string[] {
    return [...this.#active.keys()];
  }

  snapshot(): ActiveWorldCommand[] {
    return [...this.#active.values()].map((command) => structuredClone(command));
  }

  restore(
    commands: readonly ActiveWorldCommand[],
    last: LastWorldCommand | null,
    simulatedTime: number
  ): void {
    if (this.#pending.size > 0) throw new Error("Cannot restore commands during a physics advance");
    this.#active.clear();
    for (const command of commands) {
      if (this.#active.has(command.id)) {
        throw new Error(`Checkpoint contains duplicate active command: ${command.id}`);
      }
      const overlapping = [...this.#active.values()].some((active) =>
        active.channels.some((channel) => command.channels.includes(channel))
      );
      if (overlapping) throw new Error("Checkpoint contains overlapping active body channels");
      this.#active.set(command.id, structuredClone(command));
    }
    this.#last = last ? structuredClone(last) : null;
    this.#wallStartedMs = commands.length > 0 ? performance.now() : 0;
    this.#simulatedStarted = simulatedTime;
  }

  paceDelayMs(simulatedTime: number): number {
    if (this.#wallStartedMs === 0) return 0;
    const target = this.#wallStartedMs
      + (simulatedTime - this.#simulatedStarted) * 1000;
    return Math.max(0, target - performance.now());
  }

  #queueBatch(): void {
    this.#scheduled = false;
    const batch = [...this.#pending.values()];
    this.#pending.clear();
    if (batch.length === 0) return;
    const advance = this.#advanceChain.then(() =>
      this.#step(batch.map((pending) => pending.commandId))
    );
    this.#advanceChain = advance.catch(() => undefined);
    void advance.then(
      () => batch.forEach((pending) => pending.resolve()),
      (error) => batch.forEach((pending) => pending.reject(error))
    );
  }
}
