import React from 'react';
import { useNavigate } from 'react-router-dom';
import { ArrowRight, ShieldAlert, CheckCircle2 } from 'lucide-react';
import { Badge } from '../common/Badge';
import { formatDate, formatMs, formatPercent, truncateId } from '../../utils/formatters';
import { DETECTOR_NAMES } from '../../utils/constants';

export function PredictionTable({ predictions = [], isLoading = false, onRowClick = null }) {
  const navigate = useNavigate();

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

  if (!predictions || predictions.length === 0) {
    return (
      <div className="w-full rounded-xl border border-[#E5E7EB] bg-white p-8 text-center text-xs text-[#667085]">
        No intrusion detection predictions match the current criteria.
      </div>
    );
  }

  const handleRowAction = (predictionId) => {
    if (onRowClick) {
      onRowClick(predictionId);
    } else {
      navigate(`/alerts/${predictionId}`);
    }
  };

  return (
    <div className="w-full overflow-x-auto rounded-xl border border-[#E5E7EB] bg-white shadow-sm">
      <table className="w-full text-left text-xs border-collapse">
        <thead>
          <tr className="border-b border-[#E5E7EB] bg-[#F8F9FA] text-[#667085] font-semibold uppercase tracking-wider text-[10px]">
            <th className="py-3 px-4">Prediction ID</th>
            <th className="py-3 px-4">Timestamp (UTC)</th>
            <th className="py-3 px-4">Detector</th>
            <th className="py-3 px-4">Class</th>
            <th className="py-3 px-4">Malicious Score</th>
            <th className="py-3 px-4">Threshold</th>
            <th className="py-3 px-4">Latency</th>
            <th className="py-3 px-4 text-right">Actions</th>
          </tr>
        </thead>
        <tbody className="divide-y divide-[#E5E7EB] text-[#202124]">
          {predictions.map((pred) => {
            const isMalicious = pred.predicted_class === 'malicious';
            const malProb = pred.malicious_probability;

            return (
              <tr
                key={pred.prediction_id || pred.id}
                onClick={() => handleRowAction(pred.prediction_id)}
                className="hover:bg-[#F8F9FA] cursor-pointer transition-colors group"
              >
                <td className="py-3 px-4 font-mono font-medium text-[#202124]">
                  <span title={pred.prediction_id}>
                    {truncateId(pred.prediction_id, 10, 4)}
                  </span>
                </td>
                <td className="py-3 px-4 text-[#667085] whitespace-nowrap">
                  {formatDate(pred.created_at)}
                </td>
                <td className="py-3 px-4 text-[#202124] font-medium">
                  {DETECTOR_NAMES[pred.detector_name] || pred.detector_name}
                </td>
                <td className="py-3 px-4">
                  <Badge variant={isMalicious ? 'malicious' : 'benign'} size="sm">
                    {isMalicious ? (
                      <span className="flex items-center gap-1">
                        <ShieldAlert className="w-3 h-3 text-red-600" />
                        MALICIOUS
                      </span>
                    ) : (
                      <span className="flex items-center gap-1">
                        <CheckCircle2 className="w-3 h-3 text-emerald-600" />
                        BENIGN
                      </span>
                    )}
                  </Badge>
                </td>
                <td className="py-3 px-4 font-mono">
                  {malProb !== null && malProb !== undefined ? (
                    <span
                      className={
                        malProb >= 0.7
                          ? 'text-red-600 font-semibold'
                          : malProb >= 0.4
                          ? 'text-amber-700 font-medium'
                          : 'text-emerald-700'
                      }
                    >
                      {formatPercent(malProb * 100)}
                    </span>
                  ) : (
                    <span className="text-slate-400">—</span>
                  )}
                </td>
                <td className="py-3 px-4 font-mono text-[#667085]">
                  {pred.threshold_used !== undefined ? Number(pred.threshold_used).toFixed(2) : '0.50'}
                </td>
                <td className="py-3 px-4 font-mono text-[#667085] whitespace-nowrap">
                  {formatMs(pred.inference_time_ms)}
                </td>
                <td className="py-3 px-4 text-right">
                  <button
                    onClick={(e) => {
                      e.stopPropagation();
                      handleRowAction(pred.prediction_id);
                    }}
                    className="inline-flex items-center gap-1 px-2.5 py-1 rounded bg-slate-100 hover:bg-[#F26522] hover:text-white text-slate-700 text-[11px] font-medium transition-colors"
                  >
                    <span>Details</span>
                    <ArrowRight className="w-3 h-3" />
                  </button>
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}

export default PredictionTable;
