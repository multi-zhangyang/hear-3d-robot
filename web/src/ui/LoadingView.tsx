interface LoadingViewProps {
  label: string;
  modal?: boolean;
}

export function LoadingView(props: LoadingViewProps): React.JSX.Element {
  const indicator = (
    <div className="panel-loading" role="status" aria-label={props.label}>
      <i /><i /><i />
    </div>
  );
  return props.modal ? <div className="modal-loading">{indicator}</div> : indicator;
}
