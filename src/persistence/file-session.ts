import { createHash } from "node:crypto";
import { readFile } from "node:fs/promises";
import type { Session, AgentInputItem } from "@openai/agents";
import { writeTextAtomically } from "./atomic-file.js";
import {
  runFencedMutation,
  type MutationFence
} from "./mutation-fence.js";

interface SessionData {
  version: 1;
  session_id: string;
  items: AgentInputItem[];
}

export interface FileSessionItemsIdentity {
  itemCount: number;
  itemsSha256: string;
}

export class FileSession implements Session {
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

  async popItem(): Promise<AgentInputItem | undefined> {
    let removed: AgentInputItem | undefined;
    await this.#mutate(async () => {
      const data = await this.#read();
      removed = data.items.pop();
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
      version: 1,
      session_id: this.#sessionId,
      items: structuredClone(items)
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
      const parsed = JSON.parse(await readFile(this.#path, "utf8")) as Partial<SessionData>;
      if (parsed.version !== 1 || parsed.session_id !== this.#sessionId || !Array.isArray(parsed.items)) {
        throw new Error("Invalid agent session file");
      }
      this.#cache = structuredClone(parsed as SessionData);
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
    return { version: 1, session_id: this.#sessionId, items: [] };
  }
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
