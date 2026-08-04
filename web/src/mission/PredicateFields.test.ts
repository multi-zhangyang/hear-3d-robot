import { createElement } from "react";
import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it, vi } from "vitest";
import type { Bootstrap } from "../types";
import { PredicateFields } from "./PredicateFields";

describe("抓取条件编辑器", () => {
  it("只允许选择场景声明的可搬动物体", () => {
    const scenario: Bootstrap["scenarios"][number] = {
      id: "editor-world",
      title: "编辑器世界",
      kind: "authored",
      runtime: "humanoid_g1",
      extent: { width: 12, depth: 12 },
      chunk_grid: {
        manifest_version: 1,
        chunk_size: 12,
        columns: 1,
        rows: 1
      },
      objects: [
        { id: "portable-box", kind: "box", color: "#fff", portable: true },
        { id: "fixed-pillar", kind: "pillar", color: "#777", portable: false }
      ],
      zones: [{ id: "arrival", color: "#3ad" }],
      suggested_goal: {
        summary: "保持位置",
        predicates: [{
          type: "robot_at",
          target: { x: 2, y: 0, z: 2 },
          tolerance: 0.2
        }]
      }
    };
    const html = renderToStaticMarkup(createElement(PredicateFields, {
      predicate: { type: "object_grasped", object_id: "", hand: "either" },
      scenario,
      onChange: vi.fn()
    }));

    expect(html).toContain("选择可抓取物体");
    expect(html).toContain('value="portable-box"');
    expect(html).not.toContain("fixed-pillar");
    expect(html).toContain("抓取手");
    expect(html).toContain("任意手");
  });

  it("放置编辑器只显示可搬动物体、区域与几何容差", () => {
    const scenario: Bootstrap["scenarios"][number] = {
      id: "placement-world",
      title: "放置世界",
      kind: "authored",
      runtime: "humanoid_g1",
      extent: { width: 12, depth: 12 },
      chunk_grid: {
        manifest_version: 1,
        chunk_size: 12,
        columns: 1,
        rows: 1
      },
      objects: [
        { id: "portable-box", kind: "box", color: "#fff", portable: true },
        { id: "fixed-pillar", kind: "pillar", color: "#777", portable: false }
      ],
      zones: [{ id: "arrival", color: "#3ad" }],
      suggested_goal: {
        summary: "保持位置",
        predicates: [{
          type: "robot_at",
          target: { x: 2, y: 0, z: 2 },
          tolerance: 0.2
        }]
      }
    };
    const html = renderToStaticMarkup(createElement(PredicateFields, {
      predicate: {
        type: "object_placed",
        object_id: "",
        zone_id: "",
        tolerance: 0.05
      },
      scenario,
      onChange: vi.fn()
    }));

    expect(html).toContain("选择可搬动物体");
    expect(html).toContain('value="portable-box"');
    expect(html).not.toContain("fixed-pillar");
    expect(html).toContain("放置区域");
    expect(html).toContain('value="arrival"');
    expect(html).toContain("容差");
    expect(html).not.toContain("支撑力");
    expect(html).not.toContain("速度");
    expect(html).not.toContain("稳定帧");
  });
});
