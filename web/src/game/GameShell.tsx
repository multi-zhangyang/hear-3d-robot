import { useEffect } from "react";
import { modelActivityLabel, type ModelActivityPhase } from "../model-activity";

export type Workspace = "world" | "flow" | "journey" | "output";

const WORKSPACES: Array<{
  key: Workspace;
  number: string;
  label: string;
}> = [
  { key: "world", number: "1", label: "仿真" },
  { key: "flow", number: "2", label: "层级" },
  { key: "journey", number: "3", label: "动作" },
  { key: "output", number: "4", label: "日志" }
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
  const modelLabel = modelActivityLabel(props.modelState);
  return (
    <div className="game-shell">
      <header className="game-header">
        <div className="game-brand" aria-label="HEAR">
          <span className="brand-mark" aria-hidden="true">
            <svg viewBox="0 0 24 24">
              <path d="M12 7.5v4M6.5 15v-3.5h11V15" />
              <circle cx="12" cy="5.5" r="2" />
              <circle cx="6.5" cy="17.5" r="2" />
              <circle cx="17.5" cy="17.5" r="2" />
            </svg>
          </span>
          <span><b>HEAR</b></span>
        </div>
        <div className="game-toolbar">{props.toolbar}</div>
        <div className="system-controls">
          <span
            className={`model-lamp ${modelOnline ? "online" : props.modelState}`}
            aria-label={modelLabel}
            title={modelLabel}
          ><i /></span>
          <button type="button" aria-label="刷新" title="刷新" onClick={props.onRefresh}>
            <SystemIcon name="refresh" />
          </button>
          {props.onLogout && (
            <button type="button" aria-label="退出登录" title="退出登录" onClick={props.onLogout}>
              <SystemIcon name="logout" />
            </button>
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
            <WorkspaceIcon workspace={item.key} />
            <span>{item.label}</span>
          </button>
        ))}
      </nav>
    </div>
  );
}

function WorkspaceIcon({ workspace }: { workspace: Workspace }): React.JSX.Element {
  if (workspace === "world") {
    return (
      <svg viewBox="0 0 24 24" aria-hidden="true">
        <path d="m12 3.5 7 4v8l-7 4-7-4v-8l7-4Z" />
        <path d="m5.5 7.8 6.5 3.8 6.5-3.8M12 11.6v7.5" />
      </svg>
    );
  }
  if (workspace === "flow") {
    return (
      <svg viewBox="0 0 24 24" aria-hidden="true">
        <path d="M12 7v4M6 15v-4h12v4" />
        <circle cx="12" cy="5" r="2" />
        <circle cx="6" cy="17" r="2" />
        <circle cx="18" cy="17" r="2" />
      </svg>
    );
  }
  if (workspace === "journey") {
    return (
      <svg viewBox="0 0 24 24" aria-hidden="true">
        <path d="M4 12h14M14 7l5 5-5 5" />
      </svg>
    );
  }
  return (
    <svg viewBox="0 0 24 24" aria-hidden="true">
      <path d="M6 6h12M6 12h12M6 18h8" />
    </svg>
  );
}

function SystemIcon({ name }: { name: "refresh" | "logout" }): React.JSX.Element {
  return name === "refresh" ? (
    <svg viewBox="0 0 24 24" aria-hidden="true">
      <path d="M19 8a8 8 0 1 0 1 7M19 8V3m0 5h-5" />
    </svg>
  ) : (
    <svg viewBox="0 0 24 24" aria-hidden="true">
      <path d="M10 5H5v14h5M14 8l4 4-4 4M8 12h10" />
    </svg>
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
