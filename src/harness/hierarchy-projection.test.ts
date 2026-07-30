import { describe, expect, it } from "vitest";
import type { AgentSpec } from "../domain/schema.js";
import { HierarchyProjection } from "./hierarchy-projection.js";

function spec(input: Partial<AgentSpec> & Pick<AgentSpec, "name">): AgentSpec {
  return {
    name: input.name,
    objective: input.objective ?? `Complete ${input.name}`,
    success_criteria: input.success_criteria ?? [`${input.name} reports a source-backed result`],
    capabilities: input.capabilities ?? [],
    may_delegate: input.may_delegate ?? false,
    references: input.references ?? []
  };
}

describe("HierarchyProjection", () => {
  it("keeps parallel siblings active until every delegated branch returns", () => {
    const hierarchy = HierarchyProjection.create(
      "Observe and move concurrently",
      ["sense_scene", "set_head_target", "plan_base_path", "execute_base_plan"]
    );
    const rootId = hierarchy.rootId;
    const observer = hierarchy.enterChild(null, spec({
      name: "Observer",
      capabilities: ["sense_scene"]
    }), "delegate_observer_parallel").node;
    const head = hierarchy.enterChild(null, spec({
      name: "Head mover",
      capabilities: ["set_head_target"]
    }), "delegate_head_parallel").node;

    expect(new Set(hierarchy.activeIds)).toEqual(new Set([observer.id, head.id]));
    expect(hierarchy.get(rootId).status).toBe("waiting");

    hierarchy.completeChild(observer.id, "Scene observed");
    expect(hierarchy.activeIds).toEqual([head.id]);
    expect(hierarchy.get(rootId).status).toBe("waiting");

    hierarchy.completeChild(head.id, "Head target reached");
    expect(hierarchy.activeIds).toEqual([rootId]);
    expect(hierarchy.get(rootId).status).toBe("active");

    const restored = new HierarchyProjection(
      hierarchy.snapshot(),
      rootId,
      hierarchy.activeId,
      hierarchy.activeIds
    );
    expect(restored.activeIds).toEqual([rootId]);
  });

  it("records recursive parent-child execution and returns control after completion", () => {
    const hierarchy = HierarchyProjection.create(
      "Complete the requested world state",
      ["sense_scene", "inspect_entity", "drive_base", "set_head_target"]
    );
    const rootId = hierarchy.rootId;
    const parentSpec = spec({
      name: "World state lead",
      capabilities: ["sense_scene", "inspect_entity", "drive_base"],
      may_delegate: true
    });
    const leafSpec = spec({
      name: "Scene observer",
      capabilities: ["sense_scene", "inspect_entity"]
    });

    const parent = hierarchy.enterChild(null, parentSpec, "delegate_parent").node;
    expect(hierarchy.activeId).toBe(parent.id);
    expect(hierarchy.get(rootId).status).toBe("waiting");
    expect(parent).toMatchObject({ parent_id: rootId, depth: 1, status: "active" });

    const leaf = hierarchy.enterChild(parentSpec, leafSpec, "delegate_leaf").node;
    expect(hierarchy.activeId).toBe(leaf.id);
    expect(hierarchy.get(parent.id).status).toBe("waiting");
    expect(leaf).toMatchObject({ parent_id: parent.id, depth: 2, status: "active" });
    expect(hierarchy.children(parent.id).map((node) => node.id)).toEqual([leaf.id]);

    hierarchy.completeChild(leaf.id, "Observed current entities");
    expect(hierarchy.activeId).toBe(parent.id);
    expect(hierarchy.get(leaf.id)).toMatchObject({
      status: "completed",
      last_result: { output: "Observed current entities" }
    });
    expect(hierarchy.get(parent.id)).toMatchObject({
      status: "active",
      last_result: {
        child_id: leaf.id,
        child_name: leafSpec.name,
        output: "Observed current entities"
      }
    });

    hierarchy.completeChild(parent.id, "Parent objective complete");
    expect(hierarchy.activeId).toBe(rootId);
    expect(hierarchy.get(rootId).status).toBe("active");

    const reused = hierarchy.enterChild(null, parentSpec, "delegate_parent");
    expect(reused).toMatchObject({
      created: false,
      cached_output: "Parent objective complete",
      node: { id: parent.id }
    });
    expect(hierarchy.children(rootId)).toHaveLength(1);
    expect(hierarchy.activeId).toBe(rootId);
  });

  it("returns control to the parent and closes unfinished descendants after failure", () => {
    const hierarchy = HierarchyProjection.create(
      "Inspect the world",
      ["sense_scene", "inspect_entity", "query_contacts"]
    );
    const rootId = hierarchy.rootId;
    const parentSpec = spec({
      name: "Inspection lead",
      capabilities: ["sense_scene", "inspect_entity"],
      may_delegate: true
    });
    const leafSpec = spec({ name: "Entity reader", capabilities: ["inspect_entity"] });
    const parent = hierarchy.enterChild(null, parentSpec, "delegate_parent_failure").node;
    const leaf = hierarchy.enterChild(parentSpec, leafSpec, "delegate_leaf_failure").node;

    hierarchy.failChild(leaf.id, "Entity was not visible");
    expect(hierarchy.activeId).toBe(parent.id);
    expect(hierarchy.get(leaf.id)).toMatchObject({
      status: "failed",
      last_result: { error: "Entity was not visible" }
    });
    expect(hierarchy.get(parent.id)).toMatchObject({
      status: "active",
      last_result: {
        child_id: leaf.id,
        child_name: leafSpec.name,
        error: "Entity was not visible"
      }
    });

    hierarchy.failChild(parent.id, "Inspection objective could not be completed");
    expect(hierarchy.activeId).toBe(rootId);
    expect(hierarchy.get(parent.id).status).toBe("failed");
    expect(hierarchy.get(rootId).status).toBe("active");
  });

  it("records an unmet child objective as blocked instead of failed", () => {
    const hierarchy = HierarchyProjection.create(
      "Reach the requested state",
      ["sense_scene", "read_proprioception"]
    );
    const child = hierarchy.enterChild(null, spec({
      name: "State observer",
      capabilities: ["sense_scene"]
    }), "delegate_blocked_child").node;

    hierarchy.blockChild(child.id, "The target is outside sensor range");

    expect(hierarchy.get(child.id)).toMatchObject({
      status: "blocked",
      last_result: { blocked: "The target is outside sensor range" }
    });
    expect(hierarchy.activeId).toBe(hierarchy.rootId);
  });

  it("prevents child agents from exceeding their parent's capability authority", () => {
    const capabilityHierarchy = HierarchyProjection.create(
      "Inspect before moving",
      ["sense_scene", "drive_base"]
    );
    const observerSpec = spec({
      name: "Observer",
      capabilities: ["sense_scene"],
      may_delegate: true
    });
    capabilityHierarchy.enterChild(null, observerSpec, "delegate_observer");
    expect(() => capabilityHierarchy.enterChild(observerSpec, spec({
      name: "Unauthorized mover",
      capabilities: ["drive_base"]
    }), "delegate_unauthorized")).toThrow(
      "Child capability grant exceeds parent authority: drive_base"
    );
  });

  it("rejects delegation that copies the parent's complete capability grant", () => {
    const hierarchy = HierarchyProjection.create(
      "Explore the world",
      ["survey_terrain", "plan_base_path", "execute_base_plan"]
    );
    const supervisor = spec({
      name: "Exploration supervisor",
      capabilities: ["survey_terrain", "plan_base_path"],
      may_delegate: true
    });
    hierarchy.enterChild(null, supervisor, "delegate_supervisor");

    expect(() => hierarchy.enterChild(supervisor, spec({
      name: "Copied supervisor",
      capabilities: ["survey_terrain", "plan_base_path"],
      may_delegate: true
    }), "delegate_copy")).toThrow("must strictly narrow parent authority");
  });
});
