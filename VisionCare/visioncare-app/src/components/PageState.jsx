export function PageLoader({ title = 'Loading data...', subtitle = 'Please wait while we sync with the backend.' }) {
  return (
    <div className="page-state page-state-loading">
      <div className="page-spinner" />
      <div className="page-state-title">{title}</div>
      <div className="page-state-subtitle">{subtitle}</div>
    </div>
  );
}

export function PageError({ title = 'Something went wrong', subtitle = 'We could not load this view.', action }) {
  return (
    <div className="page-state page-state-error">
      <div className="page-state-icon">!</div>
      <div className="page-state-title">{title}</div>
      <div className="page-state-subtitle">{subtitle}</div>
      {action || null}
    </div>
  );
}

export function EmptyState({ title = 'Nothing here yet', subtitle = 'This section does not have any data yet.', action }) {
  return (
    <div className="page-state page-state-empty">
      <div className="page-state-icon">+</div>
      <div className="page-state-title">{title}</div>
      <div className="page-state-subtitle">{subtitle}</div>
      {action || null}
    </div>
  );
}
