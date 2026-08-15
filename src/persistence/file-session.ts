import { createHash } from "node:crypto";
import { readFile } from "node:fs/promises";
import type {
  AgentInputItem,
  Session,
  SessionHistoryTransaction,
  SessionHistoryTransactionArgs,
  SessionHistoryTransactionAwareSession
} from "@openai/agents";
import { writeTextAtomically } from "./atomic-file.js";
import {
  runFencedMutation,
  type MutationFence
} from "./mutation-fence.js";

interface LegacySessionData {
  version: 1;
  session_id: string;
  items: AgentInputItem[];
}

interface SessionData {
  version: 2;
  session_id: string;
  items: AgentInputItem[];
  applied_history_transactions: Record<string, string>;
}

export interface FileSessionItemsIdentity {
  itemCount: number;
  itemsSha256: string;
}

export class FileSession implements Session, SessionHistoryTransactionAwareSession {
  readonly #path: string;
  readonly #sessionId: string;
  readonly #mutationFence: MutationFence | undefined;
  #pending: Promise<void> = Promise.resolve();
  #cache: SessionData | undefined;
  #itemsIdentity: FileSessionItemsIdentity | undefined;

  constructor(path: string, sessionId: string, mutationFence?: MutationFence) {
    this.#path = path;
    this.#sessionId = sessionId;
    this.#mutationFence = mutationFence;
  }

  async getSessionId(): Promise<string> {
    return this.#sessionId;
  }

  async getItems(limit?: number): Promise<AgentInputItem[]> {
    return this.#serialize(async () => {
      const items = (await this.#read()).items;
      const selected = limit === undefined ? items : items.slice(-limit);
      return structuredClone(selected);
    });
  }

  /** Returns a cached content identity without cloning a long Session history. */
  async getItemsIdentity(): Promise<FileSessionItemsIdentity> {
    return this.#serialize(async () => {
      if (this.#itemsIdentity) return { ...this.#itemsIdentity };
      const items = (await this.#read()).items;
      this.#itemsIdentity = {
        itemCount: items.length,
        itemsSha256: createHash("sha256").update(canonicalJson(items)).digest("hex")
      };
      return { ...this.#itemsIdentity };
    });
  }

  async addItems(items: AgentInputItem[]): Promise<void> {
    await this.#mutate(async () => {
      const data = await this.#read();
      data.items = data.items.concat(structuredClone(items));
      await this.#write(data);
    });
  }

  /**
   * Applies the SDK's blocked-output persistence transaction atomically with
   * its operation identity. If the filesystem reports an error after rename,
   * a resumed RunState can safely retry the same operation without appending
   * the model/tool suffix twice.
   */
  async applyHistoryTransaction(
    args: SessionHistoryTransactionArgs
  ): Promise<void> {
    const operationId = args.operationId?.trim();
    if (!operationId) {
      throw new Error("Session history transaction operation ID is empty");
    }
    const transaction = snapshotHistoryTransaction(args.transaction);
    const transactionSha256 = createHash("sha256")
      .update(canonicalJson(transaction))
      .digest("hex");
    await this.#mutate(async () => {
      const data = await this.#read();
      const appliedSha256 = data.applied_history_transactions[operationId];
      if (appliedSha256 !== undefined) {
        if (appliedSha256 !== transactionSha256) {
          throw new Error(
            "Session history operation was already applied with a different transaction"
          );
        }
        return;
      }
      data.items = applyHistoryTransaction(data.items, transaction);
      data.applied_history_transactions[operationId] = transactionSha256;
      await this.#write(data);
    });
  }

  async popItem(): Promise<AgentInputItem | undefined> {
    let removed: AgentInputItem | undefined;
    await this.#mutate(async () => {
      const data = await this.#read();
      removed = data.items.pop();
      // popItem is a deliberate branch rollback. Operation receipts from the
      // removed branch must not suppress the same transaction when RunState
      // resumes against the restored history.
      data.applied_history_transactions = {};
      await this.#write(data);
    });
    return removed === undefined ? undefined : structuredClone(removed);
  }

  async clearSession(): Promise<void> {
    await this.#mutate(() => this.#write(this.#empty()));
  }

  /** Replaces the disposable SDK branch after its older prefix was compacted. */
  async replaceItems(items: AgentInputItem[]): Promise<void> {
    await this.#mutate(() => this.#write({
      version: 2,
      session_id: this.#sessionId,
      items: structuredClone(items),
      applied_history_transactions: {}
    }));
  }

  async #mutate(operation: () => Promise<void>): Promise<void> {
    await this.#serialize(operation);
  }

  async #serialize<T>(operation: () => Promise<T>): Promise<T> {
    const current = this.#pending.then(operation);
    this.#pending = current.then(() => undefined, () => undefined);
    return current;
  }

  async #read(): Promise<SessionData> {
    if (this.#cache) return structuredClone(this.#cache);
    try {
      const parsed = JSON.parse(await readFile(this.#path, "utf8")) as
        Partial<SessionData | LegacySessionData>;
      if ((parsed.version !== 1 && parsed.version !== 2)
        || parsed.session_id !== this.#sessionId
        || !Array.isArray(parsed.items)) {
        throw new Error("Invalid agent session file");
      }
      const applied = parsed.version === 2
        ? parsed.applied_history_transactions
        : {};
      if (!isHistoryTransactionIndex(applied)) {
        throw new Error("Invalid agent session transaction index");
      }
      this.#cache = {
        version: 2,
        session_id: parsed.session_id,
        items: structuredClone(parsed.items),
        applied_history_transactions: { ...applied }
      };
      return structuredClone(this.#cache);
    } catch (error) {
      if (isMissing(error)) return this.#empty();
      throw error;
    }
  }

  async #write(data: SessionData): Promise<void> {
    try {
      await runFencedMutation(
        this.#mutationFence,
        () => writeTextAtomically(this.#path, `${JSON.stringify(data)}\n`)
      );
      this.#cache = structuredClone(data);
      this.#itemsIdentity = undefined;
    } catch (error) {
      // A durability error may be reported after rename has already published
      // the new file. Drop the in-memory view so a later retry cannot overwrite
      // that visible state from a stale cache.
      this.#cache = undefined;
      this.#itemsIdentity = undefined;
      throw error;
    }
  }

  #empty(): SessionData {
    return {
      version: 2,
      session_id: this.#sessionId,
      items: [],
      applied_history_transactions: {}
    };
  }
}

function snapshotHistoryTransaction(
  transaction: SessionHistoryTransaction
): SessionHistoryTransaction {
  if (transaction.type === "append_items") {
    if (!Array.isArray(transaction.items)) {
      throw new Error("Invalid append session history transaction");
    }
    return {
      type: "append_items",
      items: structuredClone(transaction.items)
    };
  }
  if (transaction.type === "replace_suffix"
    && Array.isArray(transaction.expectedSuffix)
    && Array.isArray(transaction.replacement)) {
    return {
      type: "replace_suffix",
      expectedSuffix: structuredClone(transaction.expectedSuffix),
      replacement: structuredClone(transaction.replacement)
    };
  }
  throw new Error("Invalid replace-suffix session history transaction");
}

function applyHistoryTransaction(
  items: readonly AgentInputItem[],
  transaction: SessionHistoryTransaction
): AgentInputItem[] {
  if (transaction.type === "append_items") {
    return [...structuredClone(items), ...structuredClone(transaction.items)];
  }
  const suffixStart = items.length - transaction.expectedSuffix.length;
  const actualSuffix = suffixStart < 0 ? [] : items.slice(suffixStart);
  if (suffixStart < 0
    || canonicalJson(actualSuffix) !== canonicalJson(transaction.expectedSuffix)) {
    throw new Error(
      "Session history suffix no longer matches the transaction precondition"
    );
  }
  return [
    ...structuredClone(items.slice(0, suffixStart)),
    ...structuredClone(transaction.replacement)
  ];
}

function isHistoryTransactionIndex(
  value: unknown
): value is Record<string, string> {
  return value !== null
    && typeof value === "object"
    && !Array.isArray(value)
    && Object.entries(value).every(([operationId, sha256]) => (
      operationId.trim().length > 0
      && typeof sha256 === "string"
      && /^[a-f0-9]{64}$/.test(sha256)
    ));
}

function canonicalJson(value: unknown): string {
  if (value === null || typeof value !== "object") {
    return JSON.stringify(value) ?? "null";
  }
  if (Array.isArray(value)) return `[${value.map(canonicalJson).join(",")}]`;
  return `{${Object.entries(value)
    .sort(([left], [right]) => compareCodePoints(left, right))
    .map(([key, item]) => `${JSON.stringify(key)}:${canonicalJson(item)}`)
    .join(",")}}`;
}

function compareCodePoints(left: string, right: string): number {
  return left < right ? -1 : left > right ? 1 : 0;
}

function isMissing(error: unknown): boolean {
  return error instanceof Error && "code" in error && error.code === "ENOENT";
}
