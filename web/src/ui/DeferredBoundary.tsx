import { Component, type ReactNode } from "react";

interface DeferredBoundaryProps {
  children: ReactNode;
  resetKey: string;
  modal?: boolean;
}

interface DeferredBoundaryState {
  failed: boolean;
}

export class DeferredBoundary extends Component<DeferredBoundaryProps, DeferredBoundaryState> {
  override state: DeferredBoundaryState = { failed: false };

  static getDerivedStateFromError(): DeferredBoundaryState {
    return { failed: true };
  }

  override componentDidUpdate(previous: DeferredBoundaryProps): void {
    if (this.state.failed && previous.resetKey !== this.props.resetKey) {
      this.setState({ failed: false });
    }
  }

  override render(): ReactNode {
    if (!this.state.failed) return this.props.children;
    return (
      <div className={this.props.modal ? "deferred-failure modal" : "deferred-failure"} role="alert">
        <span>视图加载失败</span>
        <button type="button" onClick={() => globalThis.location.reload()}>重新加载</button>
      </div>
    );
  }
}
