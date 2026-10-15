import React from 'react';
import { SEVERITY_CONFIG } from '../../utils/constants';

export function SeverityBadge({ severity = 'informational', size = 'md' }) {
  const normSev = (severity || 'informational').toLowerCase();
  const config = SEVERITY_CONFIG[normSev] || SEVERITY_CONFIG.informational;

  const sizeClasses = {
    sm: 'text-xs px-2 py-0.5',
    md: 'text-xs px-2.5 py-1',
    lg: 'text-sm px-3 py-1.5',
  };

  return (
    <span
      className={`inline-flex items-center gap-1.5 font-medium rounded-full border ${config.bg} ${config.border} ${config.text} ${sizeClasses[size] || sizeClasses.md}`}
    >
      <span className={`w-1.5 h-1.5 rounded-full ${config.dot}`} />
      {config.label}
    </span>
  );
}

export default SeverityBadge;
