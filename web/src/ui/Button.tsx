import type { ButtonHTMLAttributes } from "react";

interface UiButtonProps extends ButtonHTMLAttributes<HTMLButtonElement> {
  busy?: boolean;
  block?: boolean;
  tone?: "default" | "primary" | "danger";
}

export function UiButton({
  busy = false,
  block = false,
  tone = "default",
  className,
  disabled,
  children,
  ...props
}: UiButtonProps): React.JSX.Element {
  const classes = [
    "ui-button",
    `ui-button-${tone}`,
    block ? "ui-button-block" : "",
    className ?? ""
  ].filter(Boolean).join(" ");
  return (
    <button
      {...props}
      className={classes}
      disabled={disabled || busy}
      aria-busy={busy || undefined}
    >
      {busy && <i className="ui-button-spinner" aria-hidden="true" />}
      <span>{children}</span>
    </button>
  );
}
