import React, { useEffect, useState, useCallback } from 'react';
import { useSearchParams, NavLink } from 'react-router-dom';
import {
  Brain,
  Filter,
  RotateCcw,
  ExternalLink,
  ShieldAlert,
} from 'lucide-react';
import { explanationService } from '../services/explanationService';
import { ExplanationTable } from '../components/tables/ExplanationTable';
import { Pagination } from '../components/common/Pagination';
import { ErrorAlert } from '../components/common/ErrorAlert';
import { Modal } from '../components/common/Modal';
import { SeverityBadge } from '../components/common/SeverityBadge';
import { Badge } from '../components/common/Badge';
import { formatDate } from '../utils/formatters';
import { DETECTOR_NAMES } from '../utils/constants';

export function ExplanationsPage() {
  const [searchParams, setSearchParams] = useSearchParams();

  const page = Number(searchParams.get('page')) || 1;
  const pageSize = Number(searchParams.get('page_size')) || 20;
  const severity = searchParams.get('severity') || '';
  const isFallback = searchParams.get('is_fallback') || '';
  const requiresHumanReview = searchParams.get('requires_human_review') || '';
  const detectorName = searchParams.get('detector_name') || '';

  const [explanations, setExplanations] = useState([]);
  const [totalItems, setTotalItems] = useState(0);
  const [totalPages, setTotalPages] = useState(1);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState(null);

  // Selected explanation for inspector modal
  const [selectedExp, setSelectedExp] = useState(null);

  const fetchExplanations = useCallback(async (signal) => {
    const validSignal = (signal && typeof signal === 'object' && signal instanceof AbortSignal) ? signal : undefined;
    setIsLoading(true);
    setError(null);
    try {
      const res = await explanationService.getHistoricalExplanations(
        {
          page,
          page_size: pageSize,
          severity: severity || undefined,
          is_fallback: isFallback === 'true' ? true : isFallback === 'false' ? false : undefined,
          requires_human_review:
            requiresHumanReview === 'true'
              ? true
              : requiresHumanReview === 'false'
              ? false
              : undefined,
          detector_name: detectorName || undefined,
          sort_by: 'generated_at',
          sort_order: 'desc',
        },
        validSignal
      );
      setExplanations(res.items || []);
      setTotalItems(res.total_items || 0);
      setTotalPages(res.total_pages || 1);
    } catch (err) {
      if (err.name !== 'CanceledError' && err.name !== 'AbortError' && err.code !== 'ERR_CANCELED' && err.message !== 'canceled') {
        setError(err.message || 'Failed to fetch historical explanations');
      }
    } finally {
      setIsLoading(false);
    }
  }, [page, pageSize, severity, isFallback, requiresHumanReview, detectorName]);

  useEffect(() => {
    const controller = new AbortController();
    fetchExplanations(controller.signal);
    return () => controller.abort();
  }, [fetchExplanations]);

  const updateParam = (newParams) => {
    const updated = new URLSearchParams(searchParams);
    Object.keys(newParams).forEach((k) => {
      if (newParams[k] !== undefined && newParams[k] !== null && newParams[k] !== '') {
        updated.set(k, newParams[k]);
      } else {
        updated.delete(k);
      }
    });
    setSearchParams(updated);
  };

  const handleResetFilters = () => {
    setSearchParams({ page: '1', page_size: String(pageSize) });
  };

  return (
    <div className="space-y-6">
      {/* Header and Filters Bar */}
      <div className="bg-white border border-[#E5E7EB] rounded-xl p-5 shadow-sm">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 mb-4">
          <div>
            <h2 className="text-base font-bold text-[#202124] flex items-center gap-2">
              <Brain className="w-5 h-5 text-[#F26522]" />
              <span>AI Threat Explanations Archive</span>
            </h2>
            <p className="text-xs text-[#667085]">
              Correlated LLM advisory explanations and deterministic fallback logs stored in PostgreSQL
            </p>
          </div>

          <button
            onClick={handleResetFilters}
            className="inline-flex items-center gap-1 text-xs text-[#667085] hover:text-[#202124] transition-colors self-start sm:self-auto"
          >
            <RotateCcw className="w-3.5 h-3.5" />
            <span>Reset Filters</span>
          </button>
        </div>

        {/* Quick Filter Selectors */}
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-3 pt-2">
          {/* Severity */}
          <div>
            <label className="block text-[11px] font-medium uppercase tracking-wider text-[#667085] mb-1">
              Severity Level
            </label>
            <select
              value={severity}
              onChange={(e) => updateParam({ severity: e.target.value, page: 1 })}
              className="w-full bg-white border border-[#E5E7EB] rounded-lg px-3 py-1.5 text-xs text-[#202124] focus:outline-none focus:border-[#F26522]"
            >
              <option value="">All Severities</option>
              <option value="critical">Critical</option>
              <option value="high">High</option>
              <option value="medium">Medium</option>
              <option value="low">Low</option>
              <option value="informational">Informational</option>
            </select>
          </div>

          {/* Fallback Mode */}
          <div>
            <label className="block text-[11px] font-medium uppercase tracking-wider text-[#667085] mb-1">
              Generation Engine
            </label>
            <select
              value={isFallback}
              onChange={(e) => updateParam({ is_fallback: e.target.value, page: 1 })}
              className="w-full bg-white border border-[#E5E7EB] rounded-lg px-3 py-1.5 text-xs text-[#202124] focus:outline-none focus:border-[#F26522]"
            >
              <option value="">All Engines</option>
              <option value="false">Grounded LLM Only</option>
              <option value="true">Rule Fallback Only</option>
            </select>
          </div>

          {/* Human Review */}
          <div>
            <label className="block text-[11px] font-medium uppercase tracking-wider text-[#667085] mb-1">
              SOC Human Review
            </label>
            <select
              value={requiresHumanReview}
              onChange={(e) => updateParam({ requires_human_review: e.target.value, page: 1 })}
              className="w-full bg-white border border-[#E5E7EB] rounded-lg px-3 py-1.5 text-xs text-[#202124] focus:outline-none focus:border-[#F26522]"
            >
              <option value="">All Records</option>
              <option value="true">Review Required</option>
              <option value="false">Routine Automated</option>
            </select>
          </div>

          {/* Detector */}
          <div>
            <label className="block text-[11px] font-medium uppercase tracking-wider text-[#667085] mb-1">
              Origin Detector
            </label>
            <select
              value={detectorName}
              onChange={(e) => updateParam({ detector_name: e.target.value, page: 1 })}
              className="w-full bg-white border border-[#E5E7EB] rounded-lg px-3 py-1.5 text-xs text-[#202124] focus:outline-none focus:border-[#F26522]"
            >
              <option value="">All Models</option>
              {Object.keys(DETECTOR_NAMES).map((k) => (
                <option key={k} value={k}>
                  {DETECTOR_NAMES[k]}
                </option>
              ))}
            </select>
          </div>
        </div>
      </div>

      {error && <ErrorAlert message={error} onRetry={() => fetchExplanations()} />}

      {/* Explanations Table */}
      <div className="bg-white border border-[#E5E7EB] rounded-xl shadow-sm">
        <div className="p-4 sm:p-5 border-b border-[#E5E7EB] flex items-center justify-between">
          <span className="text-xs font-semibold text-[#202124]">
            Advisory Telemetry Log
          </span>
          <span className="text-xs font-mono text-[#667085]">
            {totalItems.toLocaleString()} matching explanations
          </span>
        </div>

        <ExplanationTable
          explanations={explanations}
          isLoading={isLoading}
          onSelectExplanation={setSelectedExp}
        />

        <div className="p-4">
          <Pagination
            currentPage={page}
            totalPages={totalPages}
            totalItems={totalItems}
            pageSize={pageSize}
            onPageChange={(p) => updateParam({ page: p })}
            onPageSizeChange={(s) => updateParam({ page_size: s, page: 1 })}
          />
        </div>
      </div>

      {/* Detailed Explanation Inspector Modal */}
      <Modal
        isOpen={!!selectedExp}
        onClose={() => setSelectedExp(null)}
        title="Advisory Cybersecurity Threat Explanation"
        maxWidth="max-w-3xl"
      >
        {selectedExp && (
          <div className="space-y-5 text-xs">
            {/* Modal Subheader */}
            <div className="flex flex-wrap items-center justify-between gap-2 p-3 bg-[#F8F9FA] rounded-lg border border-[#E5E7EB]">
              <div className="flex items-center gap-2">
                <SeverityBadge severity={selectedExp.severity} size="md" />
                {selectedExp.is_fallback ? (
                  <Badge variant="warning">Rule Fallback</Badge>
                ) : (
                  <Badge variant="blue">Grounded LLM ({selectedExp.explanation_provider})</Badge>
                )}
                {selectedExp.requires_human_review && (
                  <Badge variant="malicious">Review Required</Badge>
                )}
              </div>
              <NavLink
                to={`/alerts/${selectedExp.prediction_id}`}
                className="inline-flex items-center gap-1 text-[#F26522] hover:text-[#202124] font-mono transition-colors"
              >
                <span>Flow: {selectedExp.prediction_id}</span>
                <ExternalLink className="w-3 h-3" />
              </NavLink>
            </div>

            {/* Summary */}
            <div>
              <h4 className="font-semibold uppercase tracking-wider text-[#667085] text-[11px] mb-1">
                Advisory Summary
              </h4>
              <p className="p-3 rounded-lg bg-[#F8F9FA] border border-[#E5E7EB] text-[#202124] leading-relaxed">
                {selectedExp.summary}
              </p>
            </div>

            {/* Severity Rationale */}
            <div>
              <h4 className="font-semibold uppercase tracking-wider text-[#667085] text-[11px] mb-1">
                Severity Justification
              </h4>
              <p className="text-[#202124] leading-relaxed">
                {selectedExp.severity_rationale}
              </p>
            </div>

            {/* Evidence & Uncertainties */}
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <div className="p-3 rounded-lg bg-[#F8F9FA] border border-[#E5E7EB]">
                <h4 className="font-semibold uppercase tracking-wider text-emerald-700 text-[11px] mb-2">
                  Observed Supporting Evidence
                </h4>
                <ul className="space-y-1 list-disc list-inside text-[#202124]">
                  {(selectedExp.supporting_evidence || []).map((ev, i) => (
                    <li key={i}>{ev}</li>
                  ))}
                </ul>
              </div>

              <div className="p-3 rounded-lg bg-[#F8F9FA] border border-[#E5E7EB]">
                <h4 className="font-semibold uppercase tracking-wider text-amber-800 text-[11px] mb-2">
                  Model Limitations & Caveats
                </h4>
                <ul className="space-y-1 list-disc list-inside text-[#202124]">
                  {(selectedExp.uncertainties || []).map((un, i) => (
                    <li key={i}>{un}</li>
                  ))}
                </ul>
              </div>
            </div>

            {/* Recommended Actions */}
            <div>
              <h4 className="font-semibold uppercase tracking-wider text-sky-700 text-[11px] mb-1">
                Actionable Investigation Recommendations
              </h4>
              <ol className="p-3 rounded-lg bg-[#F8F9FA] border border-[#E5E7EB] space-y-1.5 list-decimal list-inside text-[#202124]">
                {(selectedExp.recommended_actions || []).map((act, i) => (
                  <li key={i}>{act}</li>
                ))}
              </ol>
            </div>

            {/* Disclaimer */}
            <div className="pt-2 border-t border-[#E5E7EB] text-[10px] text-[#667085] italic">
              {selectedExp.disclaimer}
            </div>
          </div>
        )}
      </Modal>
    </div>
  );
}

export default ExplanationsPage;
