import React from 'react';
import { AlertTriangle, RefreshCw } from 'lucide-react';

export function ErrorAlert({ message, onRetry = null }) {
  const handleRetryClick = (e) => {
    if (e && typeof e.preventDefault === 'function') {
      e.preventDefault();
      e.stopPropagation();
    }
    if (typeof onRetry === 'function') {
      onRetry();
    }
  };

  return (
    <div className="rounded-xl border border-red-200 bg-red-50 p-4 my-4 flex items-start gap-3">
      <AlertTriangle className="w-5 h-5 text-red-600 shrink-0 mt-0.5" />
      <div className="flex-1">
        <h4 className="text-sm font-semibold text-red-800">Service Communication Error</h4>
        <p className="mt-1 text-xs text-red-700 leading-relaxed">{message || 'An unexpected error occurred while communicating with the backend.'}</p>
        {onRetry && (
          <button
            type="button"
            onClick={handleRetryClick}
            className="mt-3 inline-flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-white hover:bg-red-100 text-red-800 text-xs font-medium border border-red-200 transition-colors shadow-sm"
          >
            <RefreshCw className="w-3.5 h-3.5" />
            Retry Request
          </button>
        )}
      </div>
    </div>
  );
}

export default ErrorAlert;
