import type { TaskNode } from "../types";

export interface AgentTreeEntry {
  node: TaskNode;
  depth: number;
  ancestorContinuations: boolean[];
  isLastSibling: boolean;
}

export function buildAgentTree(
  nodes: Record<string, TaskNode>,
  rootId: string
): AgentTreeEntry[] {
  const ordered: AgentTreeEntry[] = [];
  const visited = new Set<string>();
  const stableNodes = Object.values(nodes).sort(stableNodeOrder);

  const childrenOf = (parent: TaskNode): TaskNode[] => {
    const explicit = [...new Set(parent.child_ids)].flatMap((childId) => {
      const child = nodes[childId];
      return child && child.id !== parent.id ? [child] : [];
    });
    const explicitIds = new Set(explicit.map((child) => child.id));
    const inferred = stableNodes.filter((candidate) => (
      candidate.id !== parent.id
      && candidate.parent_id === parent.id
      && !explicitIds.has(candidate.id)
    ));
    return [...explicit, ...inferred];
  };

  const visit = (
    node: TaskNode,
    depth: number,
    ancestorContinuations: boolean[],
    isLastSibling: boolean
  ): void => {
    if (visited.has(node.id)) return;
    visited.add(node.id);
    ordered.push({ node, depth, ancestorContinuations, isLastSibling });
    const children = childrenOf(node).filter((child) => !visited.has(child.id));
    children.forEach((child, index) => {
      const childIsLast = index === children.length - 1;
      visit(
        child,
        depth + 1,
        [...ancestorContinuations, !childIsLast],
        childIsLast
      );
    });
  };

  const root = nodes[rootId];
  if (root) visit(root, 0, [], true);

  for (const node of stableNodes) {
    if (visited.has(node.id) || node.parent_id !== null) continue;
    visit(node, 0, [], true);
  }
  for (const node of stableNodes) {
    if (!visited.has(node.id)) visit(node, 0, [], true);
  }
  return ordered;
}

function stableNodeOrder(left: TaskNode, right: TaskNode): number {
  return left.created_at.localeCompare(right.created_at) || left.id.localeCompare(right.id);
}
