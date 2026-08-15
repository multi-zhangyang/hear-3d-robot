import { hierarchy, tree as treeLayout } from "d3-hierarchy";
import { useMemo } from "react";
import {
  Background,
  BackgroundVariant,
  Controls,
  Handle,
  MarkerType,
  Position,
  ReactFlow,
  type Edge,
  type Node,
  type NodeProps
} from "@xyflow/react";
import type { Goal, TaskNode } from "../types";
import {
  agentNameLabel,
  nodeStatusLabel
} from "../ui-text";

interface HierarchyGraphProps {
  nodes: Record<string, TaskNode>;
  rootId: string;
  activeIds: readonly string[];
  selectedId: string | null;
  goal: Goal | null;
  onSelect: (nodeId: string) => void;
}

interface HierarchyDatum {
  node: TaskNode;
  children: HierarchyDatum[];
}

interface NeuralGraphNodeData extends Record<string, unknown> {
  node: TaskNode;
  goal: Goal | null;
  root: boolean;
  active: boolean;
  selected: boolean;
  pathSelected: boolean;
  executionKind: "model" | "service" | "writer" | "controller" | "plant";
}

interface HierarchyGraphSummary {
  modelAgents: number;
  runtimeNodes: number;
  controlEdges: number;
  physicalWriters: number;
}

// The production tree has nine siblings at its widest cortical band. Keeping
// nodes compact makes the complete 18-node authority structure readable at
// the default desktop fit instead of shrinking 15px labels into a thumbnail.
const GRAPH_NODE_WIDTH = 152;
const GRAPH_NODE_HEIGHT = 62;
const NODE_TYPES = { neural: NeuralGraphNode } as const;
const NON_MODEL_NODES = new Set([
  "humanoid-sensor-fusion",
  "humanoid-rollout-gate",
  "humanoid-executor",
  "humanoid-controller-reflex",
  "humanoid-mujoco-body"
]);

export function HierarchyGraph(props: HierarchyGraphProps): React.JSX.Element {
  const graph = useMemo(() => buildGraph(props), [
    props.activeIds,
    props.goal,
    props.nodes,
    props.rootId,
    props.selectedId
  ]);

  return (
    <div
      className="hierarchy-graph"
      aria-label={`${graph.nodes.length} 节点、${graph.edges.length} 条边的严格单父控制图`}
    >
      <ReactFlow
        nodes={graph.nodes}
        edges={graph.edges}
        nodeTypes={NODE_TYPES}
        nodesDraggable={false}
        nodesConnectable={false}
        elementsSelectable
        fitView
        fitViewOptions={{ padding: 0.055, minZoom: 0.45, maxZoom: 1 }}
        minZoom={0.32}
        maxZoom={1.35}
        panOnScroll
        selectionOnDrag={false}
        proOptions={{ hideAttribution: true }}
        onNodeClick={(_, node) => props.onSelect(node.id)}
      >
        <Background
          variant={BackgroundVariant.Dots}
          gap={22}
          size={1.2}
          color="rgba(255, 255, 255, 0.08)"
        />
        <Controls showInteractive={false} position="bottom-right" />
      </ReactFlow>
      <div className="hierarchy-graph-legend" aria-hidden="true">
        <span><i className="model" />{graph.summary.modelAgents} 模型 Agent</span>
        <span><i className="service" />{graph.summary.runtimeNodes} 运行时节点</span>
        <span><i className="edge" />{graph.summary.controlEdges} 条控制权边</span>
        <span><i className="writer" />{graph.summary.physicalWriters} 个物理写入者</span>
      </div>
    </div>
  );
}

function buildGraph(props: HierarchyGraphProps): {
  nodes: Array<Node<NeuralGraphNodeData>>;
  edges: Edge[];
  summary: HierarchyGraphSummary;
} {
  const root = props.nodes[props.rootId] ?? Object.values(props.nodes).find((node) => (
    node.parent_id === null
  ));
  if (!root) {
    return {
      nodes: [],
      edges: [],
      summary: { modelAgents: 0, runtimeNodes: 0, controlEdges: 0, physicalWriters: 0 }
    };
  }

  const datum = hierarchyDatum(root, props.nodes, new Set());
  const layout = treeLayout<HierarchyDatum>()
    .nodeSize([GRAPH_NODE_WIDTH + 5, GRAPH_NODE_HEIGHT + 20])
    .separation((left, right) => left.parent === right.parent ? 1 : 1.02)(hierarchy(datum));
  const descendants = layout.descendants();
  const minimumX = Math.min(...descendants.map((entry) => entry.x));
  const activeIds = controlPathIds(props.nodes, props.activeIds);
  const selectedPathIds = controlPathIds(
    props.nodes,
    props.selectedId ? [props.selectedId] : []
  );
  const graphNodes = descendants.map((entry): Node<NeuralGraphNodeData> => {
    const node = entry.data.node;
    const active = activeIds.has(node.id) || node.status === "active";
    const selected = node.id === props.selectedId;
    return {
      id: node.id,
      type: "neural",
      position: {
        x: entry.x - minimumX + 28,
        y: entry.y + 24
      },
      className: [
        active ? "is-active" : "",
        selected ? "is-selected" : "",
        selectedPathIds.has(node.id) ? "is-selected-path" : ""
      ]
        .filter(Boolean)
        .join(" "),
      width: GRAPH_NODE_WIDTH,
      height: GRAPH_NODE_HEIGHT,
      data: {
        node,
        goal: props.goal,
        root: node.id === props.rootId,
        active,
        selected,
        pathSelected: selectedPathIds.has(node.id),
        executionKind: executionKind(node.id)
      }
    };
  });
  const graphEdges = descendants.flatMap((entry): Edge[] => {
    if (!entry.parent) return [];
    const source = entry.parent.data.node;
    const target = entry.data.node;
    const active = activeIds.has(source.id) && activeIds.has(target.id);
    const pathSelected = selectedPathIds.has(source.id) && selectedPathIds.has(target.id);
    return [{
      id: `${source.id}->${target.id}`,
      source: source.id,
      target: target.id,
      type: "smoothstep",
      animated: active,
      className: [
        "hierarchy-edge",
        active ? "active" : "",
        pathSelected ? "selected-path" : ""
      ].filter(Boolean).join(" "),
      markerEnd: {
        type: MarkerType.ArrowClosed,
        width: 14,
        height: 14,
        color: active ? "#ededed" : pathSelected ? "#777777" : "#3a3a3a"
      }
    }];
  });
  const modelAgents = graphNodes.filter((node) => (
    node.data.executionKind === "model"
  )).length;
  const physicalWriters = graphNodes.filter((node) => (
    node.data.executionKind === "writer"
  )).length;
  return {
    nodes: graphNodes,
    edges: graphEdges,
    summary: {
      modelAgents,
      runtimeNodes: graphNodes.length - modelAgents,
      controlEdges: graphEdges.length,
      physicalWriters
    }
  };
}

function hierarchyDatum(
  node: TaskNode,
  nodes: Record<string, TaskNode>,
  ancestors: Set<string>
): HierarchyDatum {
  if (ancestors.has(node.id)) return { node, children: [] };
  const nextAncestors = new Set(ancestors).add(node.id);
  const explicitIds = new Set(node.child_ids);
  const children = [
    ...node.child_ids.flatMap((childId) => {
      const child = nodes[childId];
      return child && child.id !== node.id ? [child] : [];
    }),
    ...Object.values(nodes).filter((candidate) => (
      candidate.parent_id === node.id
      && candidate.id !== node.id
      && !explicitIds.has(candidate.id)
    )).sort(stableNodeOrder)
  ];
  return {
    node,
    children: children.map((child) => hierarchyDatum(child, nodes, nextAncestors))
  };
}

function NeuralGraphNode({ data }: NodeProps<Node<NeuralGraphNodeData>>): React.JSX.Element {
  const { node } = data;
  const kindLabel = data.executionKind === "model"
    ? data.root
      ? "ROOT"
      : node.id === "humanoid-motor-intent-compiler"
        ? "LLM BOUNDARY"
        : "AGENT"
    : data.executionKind === "writer"
      ? "WRITER"
    : data.executionKind === "controller"
      ? "CONTROL"
      : data.executionKind === "plant"
        ? "PLANT"
        : node.id === "humanoid-executor"
          ? "WRITER"
          : "RUNTIME";
  return (
    <article
      className={`neural-graph-node ${data.executionKind} ${data.active ? "active" : ""} ${data.selected ? "selected" : ""} ${data.pathSelected ? "path-selected" : ""}`}
      aria-label={`${agentNameLabel(node.name)}，${nodeStatusLabel(node.status)}`}
    >
      <Handle type="target" position={Position.Top} isConnectable={false} />
      <div className="neural-node-kicker">
        <span>{kindLabel}</span>
        <i className={node.status}>{nodeStatusLabel(node.status)}</i>
      </div>
      <strong>{agentNameLabel(node.name)}</strong>
      <footer>
        <span>L{node.depth}</span>
        <span>{node.model_calls_used} calls</span>
      </footer>
      <Handle type="source" position={Position.Bottom} isConnectable={false} />
    </article>
  );
}

function executionKind(nodeId: string): NeuralGraphNodeData["executionKind"] {
  if (nodeId === "humanoid-executor") return "writer";
  if (nodeId === "humanoid-controller-reflex") return "controller";
  if (nodeId === "humanoid-mujoco-body") return "plant";
  return NON_MODEL_NODES.has(nodeId) ? "service" : "model";
}

function controlPathIds(
  nodes: Record<string, TaskNode>,
  leafIds: readonly string[]
): Set<string> {
  const pathIds = new Set<string>();
  for (const leafId of leafIds) {
    const visited = new Set<string>();
    let cursor = nodes[leafId];
    while (cursor && !visited.has(cursor.id)) {
      visited.add(cursor.id);
      pathIds.add(cursor.id);
      cursor = cursor.parent_id ? nodes[cursor.parent_id] : undefined;
    }
  }
  return pathIds;
}

function stableNodeOrder(left: TaskNode, right: TaskNode): number {
  return left.created_at.localeCompare(right.created_at) || left.id.localeCompare(right.id);
}
