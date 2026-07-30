import { describe, expect, it } from "vitest";
import type { TaskNode } from "./types";
import { visibleHierarchyNodes } from "./MissionWorkspace";

describe("mission hierarchy HUD", () => {
  it("always retains the root, current agent, and every active ancestor", () => {
    const nodes = Array.from({ length: 12 }, (_, index) => node(
      `agent-${index}`,
      index === 0 ? null : `agent-${index - 1}`,
      index
    ));
    const byId = Object.fromEntries(nodes.map((entry) => [entry.id, entry]));
    const visible = visibleHierarchyNodes(
      nodes,
      byId,
      "agent-0",
      ["agent-11"],
      "agent-11",
      9
    );
    expect(visible.map((entry) => entry.id)).toEqual(nodes.map((entry) => entry.id));
  });

  it("fills remaining slots with recent nodes without evicting the active lineage", () => {
    const root = node("root", null, 0);
    const active = node("active", "root", 1);
    const completed = Array.from({ length: 12 }, (_, index) => node(`done-${index}`, "root", 1));
    const nodes = [root, active, ...completed];
    const visible = visibleHierarchyNodes(
      nodes,
      Object.fromEntries(nodes.map((entry) => [entry.id, entry])),
      root.id,
      [active.id],
      active.id,
      5
    );
    expect(visible.map((entry) => entry.id)).toEqual([
      "root",
      "active",
      "done-9",
      "done-10",
      "done-11"
    ]);
  });
});

function node(id: string, parentId: string | null, depth: number): TaskNode {
  return {
    id,
    name: id,
    parent_id: parentId,
    child_ids: [],
    objective: id,
    success_criteria: [id],
    goal_predicate_indexes: [],
    capabilities: [],
    may_delegate: false,
    references: [],
    depth,
    status: id === "active" || id === "agent-11" ? "active" : "completed",
    steps_used: 0,
    model_calls_used: 0,
    created_at: new Date(depth).toISOString(),
    updated_at: new Date(depth).toISOString()
  };
}
