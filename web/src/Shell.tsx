import { useState } from "react";
import { summarizeFailure } from "./failure-summary";
import type { RunListItem } from "./types";
import { UiButton } from "./ui/Button";
import { CloseIcon } from "./ui/Icons";
import { runStatusLabel } from "./ui-text";

export function Login(props: {
  hasStoredPassword: boolean;
  onLogin: (password: string) => Promise<void>;
}): React.JSX.Element {
  const [value, setValue] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(props.hasStoredPassword ? "已保存的登录凭据无效" : null);
  const submit = async (): Promise<void> => {
    if (busy || !value) return;
    setBusy(true);
    setError(null);
    try {
      await props.onLogin(value);
    } catch {
      setError("登录失败，请检查操作密码。仍无法登录时，请确认服务端认证配置。");
    } finally {
      setBusy(false);
    }
  };
  return (
    <div className="login-shell">
      <div className="login-panel">
        <span className="login-mark" aria-hidden="true"><i /><i /><i /></span>
        <h1>HEAR</h1>
        <form onSubmit={(event) => {
          event.preventDefault();
          void submit();
        }}>
          <label htmlFor="operator-password">操作密码</label>
          <div className={error ? "login-input invalid" : "login-input"}>
            <span aria-hidden="true">◇</span>
            <input
              type="password"
              autoFocus
              id="operator-password"
              value={value}
              onChange={(event) => setValue(event.target.value)}
            />
          </div>
          {error && <p className="login-error" role="alert">{error}</p>}
          <UiButton block type="submit" tone="primary" busy={busy} disabled={!value}>登录</UiButton>
        </form>
      </div>
    </div>
  );
}

export function RunStatus({ status }: { status: RunListItem["status"] }): React.JSX.Element {
  return <span className={`run-status run-status-${status}`}><i />{runStatusLabel(status)}</span>;
}

/**
 * A run stores its failure as the full structured error, which is the right
 * thing to keep and the wrong thing to render. This shows the sentence a reader
 * can act on, the fields worth naming beside it, and the raw evidence behind a
 * disclosure — so nothing is hidden and nothing is a wall of JSON.
 */
export function FailureAlert(props: {
  title: string;
  error: string;
  onClose?: () => void;
}): React.JSX.Element {
  const summary = summarizeFailure(props.error);
  const headline = /[\u3400-\u9fff]/.test(summary.headline)
    ? summary.headline
    : "运行发生错误，请查看状态信息后重试。";
  return (
    <aside className="failure-alert" role="alert">
      <span className="failure-mark" aria-hidden="true">!</span>
      <div className="failure-body">
        <div className="failure-heading">
          <span className="failure-title">{props.title}</span>
          {summary.facts.map((fact) => (
            <small key={fact.label}>{fact.label} {fact.value}</small>
          ))}
        </div>
        <p className="failure-headline">{headline}</p>
      </div>
      {props.onClose && (
        <button type="button" className="failure-close" aria-label="关闭" onClick={props.onClose}>
          <CloseIcon />
        </button>
      )}
    </aside>
  );
}

export function CenteredSpin(): React.JSX.Element {
  return <div className="centered-spin" role="status" aria-label="正在加载"><i /><i /><i /></div>;
}
