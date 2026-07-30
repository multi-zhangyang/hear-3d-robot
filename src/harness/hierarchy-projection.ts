import { randomUUID } from "node:crypto";
import { isDeepStrictEqual } from "node:util";
import type {
  AgentSpec,
  JsonValue,
  TaskNode
} from "../domain/schema.js";
import type { DelegationRecoveryState } from "./delegation-drain.js";

export interface DelegationEntry {
  node: TaskNode;
  created: boolean;
  cached_output?: string;
}

export class HierarchyProjection {
  readonly #nodes: Record<string, TaskNode>;
  readonly rootId: string;
  readonly #activeIds: Set<string>;
  #focusId: string | null;

  static create(
    objective: string,
    capabilityCatalog: string[],
    goalPredicateCount = 0
  ): HierarchyProjection {
    const root = newNode({
      name: "Mission Coordinator",
      objective,
      success_criteria: ["All requested final-state predicates pass against the current physics state."],
      evidence_requirements: [],
      goal_predicate_indexes: Array.from({ length: goalPredicateCount }, (_, index) => index),
      capabilities: capabilityCatalog,
      may_delegate: true,
      references: []
    }, null, 0);
    root.status = "active";
    return new HierarchyProjection({ [root.id]: root }, root.id, root.id);
  }

  constructor(
    nodes: Record<string, TaskNode>,
    rootId: string,
    activeId: string | null,
    activeIds?: string[]
  ) {
    this.#nodes = structuredClone(nodes);
    this.rootId = rootId;
    // Checkpoints written before sibling concurrency only have the singular
    // focus. The v3 schema supplies an empty array for the missing plural
    // field, so an explicit nullish fallback is not enough here: restore the
    // focused node into the active set when the legacy array is absent/empty.
    const restoredActiveIds = activeIds && activeIds.length > 0
      ? activeIds
      : activeId ? [activeId] : [];
    this.#activeIds = new Set(restoredActiveIds);
    this.#focusId = activeId ?? this.#activeIds.values().next().value ?? null;
    this.#validate();
  }

  get activeId(): string | null {
    return this.#focusId;
  }

  get activeIds(): string[] {
    return [...this.#activeIds];
  }

  get(nodeId: string): TaskNode {
    const node = this.#nodes[nodeId];
    if (!node) throw new Error(`Unknown hierarchy node: ${nodeId}`);
    return node;
  }

  active(): TaskNode {
    if (!this.#focusId) throw new Error("The hierarchy has no focused active agent");
    return this.get(this.#focusId);
  }

  children(nodeId: string): TaskNode[] {
    return this.get(nodeId).child_ids.map((childId) => this.get(childId));
  }

  subtreeIds(nodeId: string): string[] {
    const node = this.get(nodeId);
    return [node.id, ...descendants(this.#nodes, node).map((descendant) => descendant.id)];
  }

  canDelegate(parentSpec: AgentSpec | null, parentId?: string): boolean {
    try {
      const parent = this.#parentFor(parentSpec, parentId);
      return (parent.status === "active" || parent.status === "waiting")
        && parent.may_delegate;
    } catch {
      return false;
    }
  }

  delegatingParent(parentSpec: AgentSpec | null, parentId?: string): TaskNode {
    const parent = this.#parentFor(parentSpec, parentId);
    if (parent.status !== "active" && parent.status !== "waiting") {
      throw new Error(`Hierarchy node ${parent.id} is not an open delegating parent`);
    }
    return structuredClone(parent);
  }

  enterChild(
    parentSpec: AgentSpec | null,
    spec: AgentSpec,
    sourceCallId: string,
    parentId?: string,
    concurrentSourceCallIds?: ReadonlySet<string>,
    recoveryState?: DelegationRecoveryState,
    currentWorldRevision?: number
  ): DelegationEntry {
    const parent = this.#parentFor(parentSpec, parentId);
    const children = this.children(parent.id);
    const existing = children.find(
      (candidate) => candidate.source_call_id === sourceCallId
    );
    if (existing) {
      if (!matchesSpec(existing, spec)) {
        throw new Error(`Delegation call ${sourceCallId} was reused with a different agent specification`);
      }
      return this.#reuseChild(parent, existing);
    }

    const abandoned = concurrentSourceCallIds
      ? children.filter((candidate) =>
          isUnfinished(candidate)
          && (!candidate.source_call_id
            || !concurrentSourceCallIds.has(candidate.source_call_id))
        )
      : [];
    if (abandoned.length > 0 && recoveryState) recoveryState.recovering = true;
    const recovering = recoveryState?.recovering === true || abandoned.length > 0;
    const concurrentDuplicate = children.find((candidate) =>
      isUnfinished(candidate)
      && matchesSpec(candidate, spec)
      && candidate.source_call_id !== undefined
      && concurrentSourceCallIds?.has(candidate.source_call_id)
    );
    if (concurrentDuplicate) {
      throw new Error(
        `Concurrent delegation duplicates open hierarchy node ${concurrentDuplicate.id}`
      );
    }

    // A nested SDK run that loses its transport cannot serialize the model
    // response containing the child's original call id. Its hierarchy node is
    // nevertheless authoritative and must be resumed, not duplicated, when
    // the model recreates the exact same grant with a fresh call id.
    const matchingOpen = recovering
      ? children.find((candidate) => isUnfinished(candidate) && matchesSpec(candidate, spec))
      : undefined;
    if (matchingOpen) {
      matchingOpen.source_call_id = sourceCallId;
      matchingOpen.updated_at = new Date().toISOString();
      return this.#reuseChild(parent, matchingOpen);
    }
    const matchingCompleted = recovering && currentWorldRevision !== undefined
      ? children.find((candidate) =>
          candidate.status === "completed"
          && matchesSpec(candidate, spec)
          && completedAtWorldRevision(candidate, currentWorldRevision)
        )
      : undefined;
    if (matchingCompleted) {
      this.#refreshParent(parent);
      return {
        node: structuredClone(matchingCompleted),
        created: false,
        cached_output: completedOutput(matchingCompleted)
      };
    }

    if (abandoned.length > 0) {
      throw new Error(
        `Hierarchy parent ${parent.id} has unfinished delegation(s) from an interrupted model `
        + `turn: ${abandoned.map((candidate) => `${candidate.id}:${candidate.name}`).join(", ")}. `
        + "Reissue an exact child specification from CURRENT HARNESS AUTHORITY before creating different work."
      );
    }

    if (parent.status !== "active" && parent.status !== "waiting") {
      throw new Error(`Parent hierarchy node ${parent.id} is not open`);
    }
    if (!parent.may_delegate) {
      throw new Error(`Agent ${parent.name} is not allowed to create child agents`);
    }
    const childGoalPredicates = spec.goal_predicate_indexes ?? [];
    if (new Set(childGoalPredicates).size !== childGoalPredicates.length) {
      throw new Error("Child goal predicate ownership contains duplicate indexes");
    }
    const unavailablePredicates = childGoalPredicates.filter(
      (index) => !parent.goal_predicate_indexes.includes(index)
    );
    if (unavailablePredicates.length > 0) {
      throw new Error(
        `Child goal predicate ownership exceeds parent authority: ${unavailablePredicates.join(", ")}`
      );
    }
    if (spec.may_delegate
      && parent.goal_predicate_indexes.length > 0
      && childGoalPredicates.length === 0) {
      throw new Error(
        "A supervisory child must own at least one structured goal predicate; use a leaf for bounded observation or planning"
      );
    }
    const unavailable = spec.capabilities.filter(
      (capability) => !parent.capabilities.includes(capability)
    );
    if (unavailable.length > 0) {
      throw new Error(`Child capability grant exceeds parent authority: ${unavailable.join(", ")}`);
    }
    if (spec.capabilities.length >= parent.capabilities.length) {
      throw new Error(
        `Child capability grant must strictly narrow parent authority: ${parent.capabilities.length} `
        + `to ${spec.capabilities.length}`
      );
    }

    const child = newNode(spec, parent.id, parent.depth + 1, sourceCallId);
    const now = child.created_at;
    parent.child_ids.push(child.id);
    parent.status = "waiting";
    parent.updated_at = now;
    child.status = "active";
    this.#nodes[child.id] = child;
    this.#activeIds.delete(parent.id);
    this.#activeIds.add(child.id);
    this.#focus(child.id);
    return { node: structuredClone(child), created: true };
  }

  #reuseChild(parent: TaskNode, child: TaskNode): DelegationEntry {
    if (child.status === "completed") {
      this.#refreshParent(parent);
      return {
        node: structuredClone(child),
        created: false,
        cached_output: completedOutput(child)
      };
    }
    if (child.status === "failed" || child.status === "blocked") {
      throw new Error(
        `Delegation call ${child.source_call_id ?? child.id} already ended as ${child.status}`
      );
    }
    const now = new Date().toISOString();
    parent.status = "waiting";
    parent.updated_at = now;
    this.#activeIds.delete(parent.id);
    if (child.status === "active") this.#activeIds.add(child.id);
    this.#focus(child.status === "active" ? child.id : undefined);
    return { node: structuredClone(child), created: false };
  }

  completeChild(childId: string, output: string): void {
    const child = this.get(childId);
    if (!child.parent_id) throw new Error("Root node cannot complete as a child");
    if (child.status !== "active" && child.status !== "waiting") {
      throw new Error(`Hierarchy node ${childId} is already ${child.status}`);
    }
    const unfinished = descendants(this.#nodes, child)
      .filter((node) => isUnfinished(node));
    if (unfinished.length > 0) {
      throw new Error(`Cannot complete ${childId} with unfinished descendants`);
    }
    const parent = this.get(child.parent_id);
    const now = new Date().toISOString();
    this.#activeIds.delete(child.id);
    child.status = "completed";
    child.last_result = { output };
    child.updated_at = now;
    parent.last_result = { child_id: child.id, child_name: child.name, output };
    parent.updated_at = now;
    this.#refreshParent(parent);
  }

  failChild(childId: string, error: string): void {
    this.#closeChild(childId, "failed", error);
  }

  blockChild(childId: string, reason: string): void {
    this.#closeChild(childId, "blocked", reason);
  }

  #closeChild(
    childId: string,
    status: "blocked" | "failed",
    reason: string
  ): void {
    const child = this.get(childId);
    if (!child.parent_id) throw new Error("Root node cannot close as a child");
    const now = new Date().toISOString();
    const result = status === "blocked" ? { blocked: reason } : { error: reason };
    for (const node of [child, ...descendants(this.#nodes, child)]) {
      if (!isUnfinished(node)) continue;
      this.#activeIds.delete(node.id);
      node.status = status;
      node.last_result = result;
      node.updated_at = now;
    }
    const parent = this.get(child.parent_id);
    parent.last_result = {
      child_id: child.id,
      child_name: child.name,
      ...result
    };
    parent.updated_at = now;
    this.#refreshParent(parent);
  }

  recordModelCall(nodeId: string): void {
    const node = this.get(nodeId);
    node.model_calls_used += 1;
    node.updated_at = new Date().toISOString();
  }

  recordToolResult(nodeId: string, result: JsonValue): void {
    const node = this.get(nodeId);
    node.steps_used += 1;
    node.last_result = structuredClone(result);
    node.updated_at = new Date().toISOString();
  }

  completeRoot(result: JsonValue, failed = false): void {
    const unfinished = Object.values(this.#nodes).filter(
      (node) => node.id !== this.rootId && isUnfinished(node)
    );
    if (unfinished.length > 0) {
      throw new Error(`Cannot complete root with unfinished nodes: ${unfinished.map((node) => node.id).join(", ")}`);
    }
    const root = this.get(this.rootId);
    root.status = failed ? "failed" : "completed";
    root.last_result = structuredClone(result);
    root.updated_at = new Date().toISOString();
    this.#activeIds.clear();
    this.#focusId = null;
  }

  reactivateRoot(): void {
    const now = new Date().toISOString();
    for (const node of Object.values(this.#nodes)) {
      if (node.id === this.rootId || !isUnfinished(node)) continue;
      node.status = "failed";
      node.last_result = { error: "Unfinished node closed before coordinator restart" };
      node.updated_at = now;
    }
    const root = this.get(this.rootId);
    root.status = "active";
    root.updated_at = now;
    this.#activeIds.clear();
    this.#activeIds.add(root.id);
    this.#focusId = root.id;
  }

  failActive(error: JsonValue): void {
    const now = new Date().toISOString();
    for (const node of Object.values(this.#nodes)) {
      if (node.id !== this.rootId && !isUnfinished(node)) continue;
      node.status = "failed";
      node.last_result = structuredClone(error);
      node.updated_at = now;
    }
    this.#activeIds.clear();
    this.#focusId = null;
  }

  snapshot(): Record<string, TaskNode> {
    return structuredClone(this.#nodes);
  }

  #parentFor(spec: AgentSpec | null, parentId?: string): TaskNode {
    if (parentId !== undefined) {
      const parent = this.get(parentId);
      if (spec !== null && !matchesSpec(parent, spec)) {
        throw new Error(`Hierarchy node ${parentId} does not match parent agent ${spec.name}`);
      }
      if (spec === null && parent.id !== this.rootId) {
        throw new Error(`Only the hierarchy root may omit its parent specification`);
      }
      return parent;
    }
    if (spec === null) return this.get(this.rootId);
    let currentId = this.#focusId;
    while (currentId) {
      const node = this.get(currentId);
      if (matchesSpec(node, spec)) return node;
      currentId = node.parent_id;
    }
    throw new Error(`No active hierarchy node matches parent agent ${spec.name}`);
  }

  #validate(): void {
    const root = this.get(this.rootId);
    if (root.parent_id !== null || root.depth !== 0) throw new Error("Invalid hierarchy root");
    if (this.#focusId !== null && !this.#nodes[this.#focusId]) {
      throw new Error("Hierarchy focus node does not exist");
    }
    for (const activeId of this.#activeIds) {
      if (!this.#nodes[activeId]) throw new Error("Hierarchy active node does not exist");
      if (this.#nodes[activeId]!.status !== "active") {
        throw new Error(`Hierarchy active set contains a ${this.#nodes[activeId]!.status} node`);
      }
    }
    if (this.#focusId !== null && !this.#activeIds.has(this.#focusId)) {
      throw new Error("Hierarchy focus is not in the active set");
    }
    const seen = new Set<string>();
    const visit = (node: TaskNode): void => {
      if (seen.has(node.id)) throw new Error("Hierarchy contains a cycle");
      seen.add(node.id);
      for (const childId of node.child_ids) {
        const child = this.get(childId);
        if (child.parent_id !== node.id || child.depth !== node.depth + 1) {
          throw new Error("Invalid hierarchy parent link");
        }
        if (child.capabilities.some((capability) => !node.capabilities.includes(capability))) {
          throw new Error("Hierarchy contains an invalid capability grant");
        }
        if (child.capabilities.length >= node.capabilities.length) {
          throw new Error("Hierarchy contains a child that does not narrow parent authority");
        }
        if (child.goal_predicate_indexes.some(
          (index) => !node.goal_predicate_indexes.includes(index)
        )) {
          throw new Error("Hierarchy contains an invalid goal predicate grant");
        }
        visit(child);
      }
    };
    visit(root);
    if (seen.size !== Object.keys(this.#nodes).length) {
      throw new Error("Hierarchy contains a disconnected node");
    }
  }

  #refreshParent(parent: TaskNode): void {
    const unfinishedChild = this.children(parent.id).some((child) => isUnfinished(child));
    if (unfinishedChild) {
      parent.status = "waiting";
      this.#activeIds.delete(parent.id);
      this.#focus();
      return;
    }
    parent.status = "active";
    parent.updated_at = new Date().toISOString();
    this.#activeIds.add(parent.id);
    this.#focus(parent.id);
  }

  #focus(preferred?: string): void {
    if (preferred && this.#activeIds.has(preferred)) {
      this.#focusId = preferred;
      return;
    }
    if (this.#focusId && this.#activeIds.has(this.#focusId)) return;
    this.#focusId = [...this.#activeIds].at(-1) ?? null;
  }
}

function newNode(
  spec: AgentSpec,
  parentId: string | null,
  depth: number,
  sourceCallId?: string
): TaskNode {
  const now = new Date().toISOString();
  return {
    id: `agent_${randomUUID().slice(0, 8)}`,
    name: spec.name,
    parent_id: parentId,
    child_ids: [],
    objective: spec.objective,
    success_criteria: [...spec.success_criteria],
    evidence_requirements: structuredClone(spec.evidence_requirements ?? []),
    goal_predicate_indexes: [...(spec.goal_predicate_indexes ?? [])],
    capabilities: [...spec.capabilities],
    may_delegate: spec.may_delegate,
    references: structuredClone(spec.references),
    depth,
    ...(sourceCallId ? { source_call_id: sourceCallId } : {}),
    status: "ready",
    steps_used: 0,
    model_calls_used: 0,
    created_at: now,
    updated_at: now
  };
}

function matchesSpec(node: TaskNode, spec: AgentSpec): boolean {
  return node.name === spec.name
    && node.objective === spec.objective
    && node.may_delegate === spec.may_delegate
    && sameStrings(node.success_criteria, spec.success_criteria)
    && isDeepStrictEqual(node.evidence_requirements, spec.evidence_requirements ?? [])
    && sameNumbers(node.goal_predicate_indexes, spec.goal_predicate_indexes ?? [])
    && sameStrings(node.capabilities, spec.capabilities)
    && sameReferences(node.references, spec.references);
}

function sameStrings(left: string[], right: string[]): boolean {
  return left.length === right.length && left.every((value, index) => value === right[index]);
}

function sameNumbers(left: number[], right: number[]): boolean {
  return left.length === right.length && left.every((value, index) => value === right[index]);
}

function sameReferences(
  left: AgentSpec["references"],
  right: AgentSpec["references"]
): boolean {
  return left.length === right.length && left.every((value, index) =>
    value.name === right[index]?.name
      && value.transaction_id === right[index]?.transaction_id
  );
}

function completedOutput(node: TaskNode): string {
  const result = node.last_result;
  if (typeof result === "object" && result !== null && !Array.isArray(result)) {
    const output = result.output;
    if (typeof output === "string" && output.trim() !== "") return output;
  }
  throw new Error(`Completed hierarchy node ${node.id} has no output`);
}

function completedAtWorldRevision(node: TaskNode, worldRevision: number): boolean {
  try {
    const parsed = JSON.parse(completedOutput(node)) as unknown;
    return typeof parsed === "object"
      && parsed !== null
      && !Array.isArray(parsed)
      && "world_revision" in parsed
      && parsed.world_revision === worldRevision;
  } catch {
    return false;
  }
}

function descendants(nodes: Record<string, TaskNode>, parent: TaskNode): TaskNode[] {
  return parent.child_ids.flatMap((childId) => {
    const child = nodes[childId];
    if (!child) throw new Error(`Unknown hierarchy node: ${childId}`);
    return [child, ...descendants(nodes, child)];
  });
}

function isUnfinished(node: TaskNode): boolean {
  return node.status === "ready" || node.status === "active" || node.status === "waiting";
}
