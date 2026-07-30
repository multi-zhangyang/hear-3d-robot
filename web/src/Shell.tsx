import { LockOutlined, RobotOutlined } from "@ant-design/icons";
import { Alert, Button, Form, Input, Space, Spin, Tag, Typography } from "antd";
import { useState } from "react";
import { summarizeFailure } from "./failure-summary";
import type { RunListItem } from "./types";
import { runStatusLabel } from "./ui-text";

export function Login(props: {
  hasStoredPassword: boolean;
  onLogin: (password: string) => Promise<void>;
}): React.JSX.Element {
  const [value, setValue] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(props.hasStoredPassword ? "已保存的登录凭据无效" : null);
  const submit = async (): Promise<void> => {
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
        <RobotOutlined className="login-mark" />
        <Typography.Title level={2}>HEAR</Typography.Title>
        <Form layout="vertical" onFinish={() => void submit()}>
          <Form.Item
            label="操作密码"
            // Form.Item only wires label-to-input when the field is registered
            // through `name`. This one is controlled by local state, so the
            // association has to be made explicitly or the password box has no
            // accessible name at all.
            htmlFor="operator-password"
            help={error}
            {...(error ? { validateStatus: "error" as const } : {})}
          >
            <Input.Password
              autoFocus
              id="operator-password"
              prefix={<LockOutlined />}
              value={value}
              onChange={(event) => setValue(event.target.value)}
            />
          </Form.Item>
          <Button block htmlType="submit" type="primary" loading={busy} disabled={!value}>登录</Button>
        </Form>
      </div>
    </div>
  );
}

export function RunStatus({ status }: { status: RunListItem["status"] }): React.JSX.Element {
  const color = status === "succeeded" ? "success"
    : status === "running" || status === "starting" ? "processing"
      : status === "local_artifact" ? "default"
        : "error";
  return <Tag color={color}>{runStatusLabel(status)}</Tag>;
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
    <Alert
      className="failure-alert"
      type="error"
      showIcon
      {...(props.onClose ? { closable: true, onClose: props.onClose } : {})}
      message={
        <Space size={8} wrap>
          <span className="failure-title">{props.title}</span>
          {summary.facts.map((fact) => (
            <Tag key={fact.label} color="error" bordered={false}>{fact.label} {fact.value}</Tag>
          ))}
        </Space>
      }
      description={
        <div className="failure-body">
          <p className="failure-headline">{headline}</p>
        </div>
      }
    />
  );
}

export function CenteredSpin(): React.JSX.Element {
  return <div className="centered-spin" role="status" aria-label="正在加载"><Spin size="large" /></div>;
}
