import { createElement } from "react";
import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it } from "vitest";
import type { HumanoidRunCheckpoint } from "../types";
import { AgentFlowView } from "./AgentFlowView";
import { buildAgentTree } from "./agent-tree";

describe("agent hierarchy tree", () => {
  it("follows parent child identity instead of timestamps or stored depth", () => {
    const nodes = {
      root: treeNode("root", null, ["planner", "sentry"], "2026-08-03T00:00:03.000Z"),
      sentry: treeNode("sentry", "root", [], "2026-08-03T00:00:00.000Z"),
      planner: treeNode("planner", "root", ["worker"], "2026-08-03T00:00:02.000Z"),
      worker: treeNode("worker", "planner", [], "2026-08-03T00:00:01.000Z")
    };

    expect(buildAgentTree(nodes, "root").map((entry) => [entry.node.id, entry.depth]))
      .toEqual([
        ["root", 0],
        ["planner", 1],
        ["worker", 2],
        ["sentry", 1]
      ]);
  });

  it("appends inferred children and orphans once while breaking cycles", () => {
    const nodes = {
      root: treeNode("root", null, [], "2026-08-03T00:00:00.000Z"),
      inferred: treeNode("inferred", "root", [], "2026-08-03T00:00:01.000Z"),
      first: treeNode("first", "second", ["second"], "2026-08-03T00:00:02.000Z"),
      second: treeNode("second", "first", ["first"], "2026-08-03T00:00:03.000Z")
    };

    const entries = buildAgentTree(nodes, "root");
    expect(entries.map((entry) => entry.node.id)).toEqual([
      "root", "inferred", "first", "second"
    ]);
    expect(new Set(entries.map((entry) => entry.node.id)).size).toBe(4);
  });
});

describe("agent flow context meter", () => {
  it("renders pressure from the restored active worker budget", () => {
    const checkpoint = {
      status: "running",
      root_id: "worker",
      active_agent_id: "worker",
      active_agent_ids: ["worker"],
      nodes: {
        worker: {
          id: "worker",
          name: "工作智能体",
          child_ids: [],
          depth: 0,
          status: "active",
          capabilities: [],
          may_delegate: false,
          model_calls_used: 2,
          steps_used: 1,
          created_at: "2026-08-03T00:00:00.000Z"
        }
      },
      goal: { summary: "继续自主任务", predicates: [] },
      checker: null,
      world: { worldRevision: 0 },
      cycle_index: 0,
      active_cycle: null,
      embodied_memory: { recent_episodes: [] },
      context_memory: {
        version: 1,
        context_window_tokens: 65_536,
        compact_trigger_tokens: 40_000,
        compact_recent_model_turns: 4,
        compact_max_output_tokens: 2_048,
        active_scope_id: "worker",
        active_estimated_tokens: 100,
        total_compactions: 0,
        last_compacted_at: null,
        scopes: {
          worker: {
            scope_id: "worker",
            agent_id: "worker",
            agent_name: "工作智能体",
            raw_item_count: 0,
            raw_chain_hash: null,
            compacted_item_count: 0,
            retained_item_count: 0,
            retained_chain_hash: null,
            active_estimated_tokens: 4_000,
            context_window_tokens: 32_768,
            compact_trigger_tokens: 8_000,
            compact_recent_model_turns: 2,
            compact_max_output_tokens: 768,
            compaction_count: 0,
            summary: null,
            summary_origin: null,
            summary_world_revision: null,
            last_compacted_at: null
          }
        }
      },
      model_usage: {
        version: 1,
        total: usageTotals(600),
        by_agent: { worker: usageTotals(600) },
        updated_at: "2026-08-03T00:00:02.000Z"
      }
    } as unknown as HumanoidRunCheckpoint;

    const html = renderToStaticMarkup(createElement(AgentFlowView, {
      checkpoint,
      actions: [],
      framework: []
    }));

    expect(html).toContain("--context-load:50%");
    expect(html).toContain(
      "当前上下文估算为 4000 个令牌，上下文窗口为 32768 个令牌，压缩触发线为 8000 个令牌"
    );
    expect(html).toContain("/ 3.3万");
    expect(html).toContain("压缩线 8000");
    expect(html).toContain("600 模型令牌");
  });

  it("shows the Goal Manager and the live candidate-selection phase", () => {
    const root = treeNode("root", null, [], "2026-08-03T00:00:00.000Z");
    const goalManager = {
      ...treeNode("goal-manager", "root", [], "2026-08-03T00:00:01.000Z"),
      name: "自主目标管理智能体",
      status: "active" as const
    };
    const checkpoint = {
      version: 6,
      status: "running",
      root_id: "root",
      active_agent_id: "goal-manager",
      active_agent_ids: ["goal-manager"],
      nodes: { root, "goal-manager": goalManager },
      goal_dag: {
        status: "awaiting_model_selection",
        candidates: {
          first: { status: "proposed" },
          second: { status: "proposed" }
        }
      },
      checker: null,
      world: { worldRevision: 3 },
      cycle_index: 0,
      embodied_memory: { recent_episodes: [] },
      context_memory: {
        version: 1,
        context_window_tokens: 65_536,
        compact_trigger_tokens: 18_000,
        compact_recent_model_turns: 4,
        compact_max_output_tokens: 4_096,
        active_scope_id: "goal-manager",
        active_estimated_tokens: 900,
        total_compactions: 0,
        last_compacted_at: null,
        scopes: {}
      }
    } as unknown as HumanoidRunCheckpoint;

    const html = renderToStaticMarkup(createElement(AgentFlowView, {
      checkpoint,
      actions: [],
      framework: []
    }));

    expect(html).toContain("自主目标管理智能体");
    expect(html).toContain("2 个智能体");
    expect(html).toContain("目标管理智能体正在选择 · 2 个候选");
    expect(html).toContain("目标选择");
  });
});

function treeNode(
  id: string,
  parentId: string | null,
  childIds: string[],
  createdAt: string
): HumanoidRunCheckpoint["nodes"][string] {
  return {
    id,
    name: id,
    parent_id: parentId,
    child_ids: childIds,
    objective: id,
    success_criteria: [],
    evidence_requirements: [],
    goal_predicate_indexes: [],
    capabilities: [],
    may_delegate: childIds.length > 0,
    references: [],
    depth: 99,
    status: "ready",
    steps_used: 0,
    model_calls_used: 0,
    created_at: createdAt,
    updated_at: createdAt
  };
}

function usageTotals(totalTokens: number) {
  return {
    requests: 1,
    reported_requests: 1,
    input_tokens: Math.max(0, totalTokens - 120),
    output_tokens: Math.min(120, totalTokens),
    total_tokens: totalTokens,
    cached_input_tokens: 0,
    reasoning_tokens: 0
  };
}
