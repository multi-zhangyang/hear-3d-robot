export interface DelegationRecoveryState {
  recovering: boolean;
}

export interface DelegationDrainHandle {
  readonly sourceCallIds: ReadonlySet<string>;
  readonly recoveryState: DelegationRecoveryState;
  settle(): void;
  settleAndDrain(): Promise<void>;
}

export class DelegationDrainRegistry {
  readonly #current = new Map<string, DelegationBatch>();

  register(
    parentId: string,
    sourceCallId: string,
    recoveringParentInvocation = false
  ): DelegationDrainHandle {
    let batch = this.#current.get(parentId);
    if (batch?.sealed) {
      throw new Error(`Delegation parent ${parentId} is still draining an interrupted batch`);
    }
    if (!batch) {
      batch = new DelegationBatch(recoveringParentInvocation, () => {
        if (this.#current.get(parentId) === batch) this.#current.delete(parentId);
      });
      this.#current.set(parentId, batch);
    } else if (recoveringParentInvocation) {
      batch.recoveryState.recovering = true;
    }
    return batch.register(sourceCallId);
  }
}

class DelegationBatch {
  readonly sourceCallIds = new Set<string>();
  readonly recoveryState: DelegationRecoveryState;
  readonly #pending = new Set<symbol>();
  readonly #drained: Promise<void>;
  readonly #resolveDrained: () => void;
  readonly #onDrained: () => void;
  sealed = false;

  constructor(recovering: boolean, onDrained: () => void) {
    this.recoveryState = { recovering };
    this.#onDrained = onDrained;
    let resolveDrained: (() => void) | undefined;
    this.#drained = new Promise<void>((resolve) => {
      resolveDrained = resolve;
    });
    this.#resolveDrained = resolveDrained!;
  }

  register(sourceCallId: string): DelegationDrainHandle {
    if (this.sealed) throw new Error("Cannot register delegation in a draining batch");
    if (this.sourceCallIds.has(sourceCallId)) {
      throw new Error(`Delegation call ${sourceCallId} is already running in this batch`);
    }
    const token = Symbol(sourceCallId);
    this.sourceCallIds.add(sourceCallId);
    this.#pending.add(token);
    let settled = false;
    const settle = (): void => {
      if (settled) return;
      settled = true;
      this.#pending.delete(token);
      if (this.#pending.size !== 0) return;
      this.#resolveDrained();
      this.#onDrained();
    };
    return {
      sourceCallIds: this.sourceCallIds,
      recoveryState: this.recoveryState,
      settle,
      settleAndDrain: async () => {
        this.sealed = true;
        settle();
        await this.#drained;
      }
    };
  }
}
