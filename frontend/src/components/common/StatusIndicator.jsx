import React from 'react';

export function StatusIndicator({ status = 'online', label, showDot = true, size = 'sm' }) {
  const isHealthy = status === 'online' || status === 'ready' || status === 'healthy' || status === true;

  const dotClasses = isHealthy
    ? 'bg-emerald-500'
    : 'bg-red-500';

  const textClasses = isHealthy ? 'text-emerald-700' : 'text-red-700';

  return (
    <div className="inline-flex items-center gap-2">
      {showDot && <span className={`w-2 h-2 rounded-full ${dotClasses}`} />}
      {label && (
        <span className={`text-xs font-medium ${textClasses}`}>
          {label}
        </span>
      )}
    </div>
  );
}

export default StatusIndicator;
