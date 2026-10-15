import React from 'react';
import { NavLink } from 'react-router-dom';
import { SeverityBadge } from '../common/SeverityBadge';
import { Badge } from '../common/Badge';
import { formatDate, truncateId } from '../../utils/formatters';
import { DETECTOR_NAMES } from '../../utils/constants';

export function ExplanationTable({ explanations = [], isLoading = false, onSelectExplanation = null }) {
  if (isLoading) {
    return (
      <div className="w-full overflow-hidden rounded-xl border border-[#E5E7EB] bg-white">
        <div className="animate-pulse p-6 space-y-4">
          <div className="h-4 bg-slate-200 rounded w-1/4" />
          <div className="space-y-3">
            {[...Array(5)].map((_, i) => (
              <div key={i} className="h-10 bg-slate-100 rounded" />
            ))}
          </div>
        </div>
      </div>
    );
  }

  if (!explanations || explanations.length === 0) {
    return (
      <div className="w-full rounded-xl border border-[#E5E7EB] bg-white p-8 text-center text-xs text-[#667085]">
        No advisory threat explanations match the specified query filters.
      </div>
    );
  }

  return (
    <div className="w-full overflow-x-auto rounded-xl border border-[#E5E7EB] bg-white shadow-sm">
      <table className="w-full text-left text-xs border-collapse">
        <thead>
          <tr className="border-b border-[#E5E7EB] bg-[#F8F9FA] text-[#667085] font-semibold uppercase tracking-wider text-[10px]">
            <th className="py-3 px-4">Prediction ID</th>
            <th className="py-3 px-4">Severity</th>
            <th className="py-3 px-4">Detector</th>
            <th className="py-3 px-4">Summary</th>
            <th className="py-3 px-4">Provider</th>
            <th className="py-3 px-4">Mode</th>
            <th className="py-3 px-4">Human Review</th>
            <th className="py-3 px-4">Generated (UTC)</th>
            <th className="py-3 px-4 text-right">Actions</th>
          </tr>
        </thead>
        <tbody className="divide-y divide-[#E5E7EB] text-[#202124]">
          {explanations.map((exp) => (
            <tr
              key={exp.id || exp.prediction_id}
              onClick={() => onSelectExplanation && onSelectExplanation(exp)}
              className="hover:bg-[#F8F9FA] cursor-pointer transition-colors"
            >
              <td className="py-3 px-4 font-mono font-medium text-[#202124]">
                <NavLink
                  to={`/alerts/${exp.prediction_id}`}
                  onClick={(e) => e.stopPropagation()}
                  className="hover:text-[#F26522] underline decoration-slate-300 underline-offset-2"
                  title={exp.prediction_id}
                >
                  {truncateId(exp.prediction_id, 10, 4)}
                </NavLink>
              </td>
              <td className="py-3 px-4">
                <SeverityBadge severity={exp.severity} size="sm" />
              </td>
              <td className="py-3 px-4 text-[#202124] font-medium whitespace-nowrap">
                {DETECTOR_NAMES[exp.detector_name] || exp.detector_name}
              </td>
              <td className="py-3 px-4 text-[#202124] max-w-xs truncate" title={exp.summary}>
                {exp.summary}
              </td>
              <td className="py-3 px-4 font-mono text-[11px] text-[#667085]">
                {exp.explanation_provider}
              </td>
              <td className="py-3 px-4">
                {exp.is_fallback ? (
                  <Badge variant="warning" size="sm">Rule Fallback</Badge>
                ) : (
                  <Badge variant="blue" size="sm">LLM Grounded</Badge>
                )}
              </td>
              <td className="py-3 px-4">
                {exp.requires_human_review ? (
                  <Badge variant="malicious" size="sm">Required</Badge>
                ) : (
                  <Badge variant="default" size="sm">Routine</Badge>
                )}
              </td>
              <td className="py-3 px-4 text-[#667085] whitespace-nowrap">
                {formatDate(exp.generated_at)}
              </td>
              <td className="py-3 px-4 text-right">
                <button
                  onClick={(e) => {
                    e.stopPropagation();
                    if (onSelectExplanation) onSelectExplanation(exp);
                  }}
                  className="px-2.5 py-1 rounded bg-slate-100 hover:bg-[#F26522] hover:text-white text-slate-700 text-[11px] font-medium transition-colors"
                >
                  Inspect
                </button>
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

export default ExplanationTable;
