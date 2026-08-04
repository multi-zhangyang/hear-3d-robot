import { useEffect } from "react";
import { modelActivityLabel, type ModelActivityPhase } from "../model-activity";

export type Workspace = "world" | "flow" | "journey" | "output";

const WORKSPACES: Array<{
  key: Workspace;
  number: string;
  glyph: string;
  label: string;
}> = [
  { key: "world", number: "1", glyph: "◈", label: "世界" },
  { key: "flow", number: "2", glyph: "⑂", label: "智能体流" },
  { key: "journey", number: "3", glyph: "➜", label: "行动历程" },
  { key: "output", number: "4", glyph: "✦", label: "输出" }
];

interface GameShellProps {
  workspace: Workspace;
  onWorkspace: (workspace: Workspace) => void;
  modelState: ModelActivityPhase;
  toolbar: React.ReactNode;
  onRefresh: () => void;
  onLogout: (() => void) | null;
  children: React.ReactNode;
}

export function GameShell(props: GameShellProps): React.JSX.Element {
  useWorkspaceKeys(props.onWorkspace);
  const modelOnline = props.modelState === "active" || props.modelState === "verified";
  return (
    <div className="game-shell">
      <header className="game-header">
        <div className="game-brand" aria-label="HEAR">
          <span className="brand-cube" aria-hidden="true"><i /><i /><i /></span>
          <span><b>HEAR</b><small>具身智能世界</small></span>
        </div>
        <div className="game-toolbar">{props.toolbar}</div>
        <div className="system-controls">
          <span className={`model-lamp ${modelOnline ? "online" : props.modelState}`}>
            {modelActivityLabel(props.modelState)}
          </span>
          <button type="button" aria-label="刷新" title="刷新" onClick={props.onRefresh}>↻</button>
          {props.onLogout && (
            <button type="button" aria-label="退出登录" title="退出登录" onClick={props.onLogout}>×</button>
          )}
        </div>
      </header>

      <main className="game-main">{props.children}</main>

      <nav className="game-hotbar" aria-label="工作区">
        {WORKSPACES.map((item) => (
          <button
            key={item.key}
            type="button"
            className={props.workspace === item.key ? "hotbar-slot active" : "hotbar-slot"}
            aria-label={item.label}
            aria-current={props.workspace === item.key ? "page" : undefined}
            onClick={() => props.onWorkspace(item.key)}
          >
            <small>{item.number}</small>
            <b aria-hidden="true">{item.glyph}</b>
            <span>{item.label}</span>
          </button>
        ))}
      </nav>
    </div>
  );
}

function useWorkspaceKeys(onWorkspace: (workspace: Workspace) => void): void {
  useEffect(() => {
    const handle = (event: KeyboardEvent): void => {
      const target = event.target;
      if (target instanceof HTMLInputElement
        || target instanceof HTMLTextAreaElement
        || target instanceof HTMLSelectElement
        || (target instanceof HTMLElement && target.isContentEditable)) return;
      const workspace = WORKSPACES.find((item) => item.number === event.key)?.key;
      if (workspace) onWorkspace(workspace);
      else if (event.key === "Escape") onWorkspace("world");
    };
    window.addEventListener("keydown", handle);
    return () => window.removeEventListener("keydown", handle);
  }, [onWorkspace]);
}
