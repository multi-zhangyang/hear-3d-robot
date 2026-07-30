/**
 * Stops an agent from re-issuing an action whose outcome is already known.
 *
 * Identical requests from the same agent at the same world revision have a
 * deterministic outcome. Repeating either an accepted read or a denial cannot
 * add information, so the ledger treats both as the same stalled condition.
 *
 * This is not a step quota: only repeated outcomes without an intervening state
 * or argument change are limited. New arguments or a new world revision clear
 * the condition immediately.
 *
 * The ledger is deliberately in-memory. It describes one continuous stretch of
 * reasoning; a resumed run starts from a fresh checkpoint and deserves a fresh
 * allowance rather than inheriting a bound it cannot see the history behind.
 */
import type { JsonValue } from "../domain/schema.js";

/**
 * How many identical outcomes an agent may collect before the harness refuses
 * to run the action again. Three are allowed through so the agent sees a
 * denial's recovery text more than once before being cut off, and so a leaf
 * that reads, thinks, and reads again is never punished for one recheck.
 */
export const REPEATED_DENIAL_LIMIT = 3;

export interface DenialAttempt {
  agentId: string;
  name: string;
  input: JsonValue;
  worldRevision: number;
}

export class DenialLedger {
  readonly #counts = new Map<string, number>();
  #revision: number | null = null;

  /**
   * Number of times this exact action has already produced the same outcome at
   * the current world revision.
   */
  count(attempt: DenialAttempt): number {
    this.#syncRevision(attempt.worldRevision);
    return this.#counts.get(signature(attempt)) ?? 0;
  }

  /** True when running the action again cannot tell the agent anything new. */
  exhausted(attempt: DenialAttempt): boolean {
    return this.count(attempt) >= REPEATED_DENIAL_LIMIT;
  }

  /**
   * Records one uninformative outcome — a denial, or an observation that read a
   * world nothing has changed since.
   */
  recordDenial(attempt: DenialAttempt): number {
    this.#syncRevision(attempt.worldRevision);
    const key = signature(attempt);
    const next = (this.#counts.get(key) ?? 0) + 1;
    this.#counts.set(key, next);
    return next;
  }

  /**
   * A new world revision means every earlier outcome was decided against a
   * state that no longer exists, so none of them constrain what happens next.
   */
  #syncRevision(worldRevision: number): void {
    if (this.#revision === worldRevision) return;
    this.#revision = worldRevision;
    this.#counts.clear();
  }
}

/**
 * Agent ids and capability names are drawn from disjoint alphabets and neither
 * contains a newline, so joining on one keeps two different attempts from ever
 * colliding into a single key. (A NUL separator would do the same but makes git
 * classify this source file as binary.)
 */
function signature(attempt: DenialAttempt): string {
  return [attempt.agentId, attempt.name, JSON.stringify(attempt.input)].join("\n");
}
