import { CloseIcon } from "../ui/Icons";

interface OverlayPanelProps {
  title: string;
  onClose: () => void;
  children: React.ReactNode;
}

export function OverlayPanel(props: OverlayPanelProps): React.JSX.Element {
  return (
    <section className="overlay-panel" aria-label={`${props.title}面板`}>
      <header className="overlay-panel-header">
        <b>{props.title}</b>
        <button type="button" aria-label="关闭面板" title="关闭" onClick={props.onClose}>
          <CloseIcon />
        </button>
      </header>
      <div className="overlay-panel-body">{props.children}</div>
    </section>
  );
}
