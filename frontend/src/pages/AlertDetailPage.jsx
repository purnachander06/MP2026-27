import React, { useEffect, useState, useCallback } from 'react';
import { useParams, NavLink } from 'react-router-dom';
import {
  ArrowLeft,
  ShieldAlert,
  ShieldCheck,
  Cpu,
  Clock,
  Info,
  Layers,
  FileCode,
  FileText,
  FileSpreadsheet,
} from 'lucide-react';
import { predictionService } from '../services/predictionService';
import { explanationService } from '../services/explanationService';
import { LoadingSpinner } from '../components/common/LoadingSpinner';
import { ErrorAlert } from '../components/common/ErrorAlert';
import { Badge } from '../components/common/Badge';
import { SeverityBadge } from '../components/common/SeverityBadge';
import { formatDate, formatPercent, formatMs } from '../utils/formatters';
import { DETECTOR_NAMES } from '../utils/constants';

export function AlertDetailPage() {
  const { predictionId } = useParams();
  const [detail, setDetail] = useState(null);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState(null);

  // Explanation generation state
  const [isGeneratingExp, setIsGeneratingExp] = useState(false);
  const [operatorContext, setOperatorContext] = useState('');
  const [genError, setGenError] = useState(null);

  const fetchDetail = useCallback(async (signal) => {
    const validSignal = (signal && typeof signal === 'object' && signal instanceof AbortSignal) ? signal : undefined;
    setIsLoading(true);
    setError(null);
    try {
      const data = await predictionService.getPredictionDetail(predictionId, validSignal);
      setDetail(data);
    } catch (err) {
      if (err.name !== 'CanceledError' && err.name !== 'AbortError' && err.code !== 'ERR_CANCELED' && err.message !== 'canceled') {
        setError(err.message || `Failed to fetch prediction detail for ${predictionId}`);
      }
    } finally {
      setIsLoading(false);
    }
  }, [predictionId]);

  useEffect(() => {
    const controller = new AbortController();
    fetchDetail(controller.signal);
    return () => controller.abort();
  }, [fetchDetail]);

  const handleGenerateExplanation = async (e) => {
    e.preventDefault();
    if (!detail?.prediction) return;
    setIsGeneratingExp(true);
    setGenError(null);

    const p = detail.prediction;
    try {
      const newExp = await explanationService.generateExplanation({
        prediction_id: p.prediction_id,
        predicted_class: p.predicted_class,
        detector_name: p.detector_name,
        malicious_probability: p.malicious_probability || 0.0,
        confidence_score: p.confidence_score || 0.0,
        validated_evidence: p.validated_evidence || {},
        model_version: p.model_version || '1.0.0',
        operator_context: operatorContext.trim() || null,
      });

      // Append new explanation to view
      setDetail((prev) => ({
        ...prev,
        explanations: [newExp, ...(prev.explanations || [])],
      }));
      setOperatorContext('');
    } catch (err) {
      setGenError(err.message || 'Failed to generate advisory explanation');
    } finally {
      setIsGeneratingExp(false);
    }
  };

  if (isLoading) {
    return <LoadingSpinner text={`Retrieving authoritative record for ${predictionId}...`} />;
  }

  if (error || !detail?.prediction) {
    return (
      <div className="space-y-4">
        <NavLink
          to="/alerts"
          className="inline-flex items-center gap-1.5 text-xs text-slate-400 hover:text-slate-200 transition-colors"
        >
          <ArrowLeft className="w-4 h-4" />
          Back to Threat Alerts
        </NavLink>
        <ErrorAlert message={error || 'Prediction record not found.'} onRetry={() => fetchDetail()} />
      </div>
    );
  }

  const pred = detail.prediction;
  const explanations = detail.explanations || [];
  const isMalicious = pred.predicted_class === 'malicious';

  return (
    <div className="space-y-6">
      {/* Navigation header */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3">
        <NavLink
          to="/alerts"
          className="inline-flex items-center gap-1.5 text-xs text-[#667085] hover:text-[#202124] transition-colors"
        >
          <ArrowLeft className="w-4 h-4" />
          Back to Threat Alerts Archive
        </NavLink>
        <span className="text-xs font-mono text-[#667085]">
          Internal UUID: {pred.id}
        </span>
      </div>

      {/* Primary Alert Banner */}
      <div className="bg-white border border-[#E5E7EB] rounded-xl p-6 shadow-sm">
        <div className="flex flex-col md:flex-row items-start md:items-center justify-between gap-4">
          <div>
            <div className="flex items-center gap-3">
              <span className="text-xl font-bold font-mono text-[#202124] tracking-tight">
                {pred.prediction_id}
              </span>
              <Badge variant={isMalicious ? 'malicious' : 'benign'} size="lg">
                {isMalicious ? (
                  <span className="flex items-center gap-1.5">
                    <ShieldAlert className="w-4 h-4 text-red-600" />
                    MALICIOUS FLOW
                  </span>
                ) : (
                  <span className="flex items-center gap-1.5">
                    <ShieldCheck className="w-4 h-4 text-emerald-600" />
                    BENIGN ROUTINE FLOW
                  </span>
                )}
              </Badge>
            </div>
            <p className="mt-1 text-xs text-[#667085] flex items-center gap-4">
              <span className="flex items-center gap-1">
                <Cpu className="w-3.5 h-3.5 text-[#667085]" />
                {DETECTOR_NAMES[pred.detector_name] || pred.detector_name} (v{pred.model_version})
              </span>
              <span className="flex items-center gap-1">
                <Clock className="w-3.5 h-3.5 text-[#667085]" />
                {formatDate(pred.created_at)}
              </span>
            </p>
          </div>

          <div className="flex items-center gap-4 text-right">
            <div>
              <span className="text-[10px] uppercase tracking-wider text-[#667085] block">Malicious Probability</span>
              <span className={`text-2xl font-bold font-mono ${isMalicious ? 'text-red-600' : 'text-[#202124]'}`}>
                {pred.malicious_probability !== null ? formatPercent(pred.malicious_probability * 100) : '—'}
              </span>
            </div>
            <div className="border-l border-[#E5E7EB] pl-4">
              <span className="text-[10px] uppercase tracking-wider text-[#667085] block">Inference Latency</span>
              <span className="text-lg font-mono font-semibold text-[#F26522]">
                {formatMs(pred.inference_time_ms)}
              </span>
            </div>
          </div>
        </div>

        {/* Uncalibrated Score Notice */}
        {!pred.is_calibrated && (
          <div className="mt-5 p-3 rounded-lg bg-amber-50 border border-amber-200 flex items-start gap-2.5">
            <Info className="w-4 h-4 text-amber-700 shrink-0 mt-0.5" />
            <p className="text-xs text-amber-800 leading-relaxed">
              <strong>Notice — Uncalibrated Score:</strong> Probability scores generated by this model are raw classifier outputs. They quantify relative statistical anomaly score in the IoT-23 feature space, not a calibrated Bayesian likelihood of real-world compromise.
            </p>
          </div>
        )}
      </div>

      {/* Model Telemetry & Leakage Isolation Strip */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Validated Evidence */}
        <div className="bg-white border border-[#E5E7EB] rounded-xl p-5 shadow-sm flex flex-col">
          <div className="flex items-center gap-2 mb-2">
            <FileCode className="w-4 h-4 text-[#F26522]" />
            <h3 className="text-sm font-bold text-[#202124]">Validated Telemetry Evidence</h3>
          </div>
          <p className="text-xs text-[#667085] mb-3">
            Grounded non-null connection features verified against IoT-23 schema:
          </p>

          <div className="flex-1 bg-[#F8F9FA] rounded-lg p-4 font-mono text-xs border border-[#E5E7EB] overflow-x-auto text-[#202124]">
            <pre>{JSON.stringify(pred.validated_evidence || {}, null, 2)}</pre>
          </div>
        </div>

        {/* Stripped Leakage Fields */}
        <div className="bg-white border border-[#E5E7EB] rounded-xl p-5 shadow-sm flex flex-col justify-between">
          <div>
            <div className="flex items-center gap-2 mb-2">
              <Layers className="w-4 h-4 text-sky-600" />
              <h3 className="text-sm font-bold text-[#202124]">Data Leakage Prevention Guard</h3>
            </div>
            <p className="text-xs text-[#667085] mb-4">
              Fields separated during parsing to prevent identity memorization (IP addresses, Zeek UIDs, target labels):
            </p>

            <div className="space-y-2">
              {pred.stripped_leakage_fields && pred.stripped_leakage_fields.length > 0 ? (
                pred.stripped_leakage_fields.map((field) => (
                  <div
                    key={field}
                    className="flex items-center justify-between p-2 rounded bg-[#F8F9FA] border border-[#E5E7EB] text-xs font-mono"
                  >
                    <span className="text-[#202124]">{field}</span>
                    <span className="text-[10px] text-amber-800 bg-amber-50 px-2 py-0.5 rounded border border-amber-200">
                      Excluded from Evidence
                    </span>
                  </div>
                ))
              ) : (
                <div className="p-3 text-xs text-[#667085] italic bg-[#F8F9FA] rounded border border-[#E5E7EB]">
                  No extraneous metadata columns were present in this input flow.
                </div>
              )}
            </div>
          </div>

          <div className="mt-4 pt-4 border-t border-[#E5E7EB] text-[11px] text-[#667085]">
            Decision Threshold: <span className="font-mono text-[#202124] font-semibold">{Number(pred.threshold_used).toFixed(4)}</span> &bull; Model Confidence: <span className="font-mono text-[#202124] font-semibold">{formatPercent((pred.confidence_score || 0) * 100)}</span>
          </div>
        </div>
      </div>

      {/* Linked Explanations Section */}
      <div className="bg-white border border-[#E5E7EB] rounded-xl p-5 shadow-sm space-y-6">
        <div className="flex items-center justify-between pb-3 border-b border-[#E5E7EB]">
          <div>
            <h3 className="text-sm font-bold text-[#202124] flex items-center gap-2">
              <FileText className="w-4 h-4 text-[#F26522]" />
              <span>Advisory Threat Explanations ({explanations.length})</span>
            </h3>
            <p className="text-xs text-[#667085]">
              Correlated LLM-grounded reasoning or deterministic fallback rules
            </p>
          </div>
        </div>

        {explanations.length === 0 ? (
          <div className="p-6 text-center rounded-xl bg-[#F8F9FA] border border-[#E5E7EB] space-y-4">
            <p className="text-xs text-[#667085]">
              No advisory explanation has been generated for this prediction yet.
            </p>

            {/* Explanation generation form */}
            <form onSubmit={handleGenerateExplanation} className="max-w-md mx-auto space-y-3 text-left">
              <div>
                <label className="block text-[11px] font-medium text-[#202124] mb-1">
                  Optional Operator Context / SOC Notes:
                </label>
                <textarea
                  value={operatorContext}
                  onChange={(e) => setOperatorContext(e.target.value)}
                  placeholder="e.g. Host is a smart thermostat running firmware v2.1"
                  rows={2}
                  maxLength={500}
                  className="w-full bg-white border border-[#E5E7EB] rounded-lg p-2.5 text-xs text-[#202124] focus:outline-none focus:border-[#F26522]"
                />
              </div>

              {genError && <p className="text-xs text-red-600">{genError}</p>}

              <button
                type="submit"
                disabled={isGeneratingExp}
                className="w-full inline-flex items-center justify-center gap-2 px-4 py-2 rounded-lg bg-[#F26522] hover:bg-[#F26522]/90 text-white text-xs font-semibold disabled:opacity-50 transition-colors shadow-sm"
              >
                <FileText className="w-4 h-4" />
                {isGeneratingExp ? 'Consulting LLM Provider...' : 'Generate Advisory Explanation'}
              </button>
            </form>
          </div>
        ) : (
          <div className="space-y-6">
            {explanations.map((exp, idx) => (
              <div
                key={exp.id || idx}
                className="p-5 rounded-xl bg-[#F8F9FA] border border-[#E5E7EB] space-y-4"
              >
                <div className="flex flex-wrap items-center justify-between gap-2 pb-3 border-b border-[#E5E7EB]">
                  <div className="flex items-center gap-2">
                    <SeverityBadge severity={exp.severity} size="lg" />
                    {exp.is_fallback ? (
                      <Badge variant="warning">Rule Fallback Engine</Badge>
                    ) : (
                      <Badge variant="blue">Grounded LLM ({exp.explanation_provider})</Badge>
                    )}
                    {exp.requires_human_review && (
                      <Badge variant="malicious">Human Review Required</Badge>
                    )}
                  </div>
                  <span className="text-xs text-[#667085] font-mono">
                    {formatDate(exp.generated_at)}
                  </span>
                </div>

                {/* Summary */}
                <div>
                  <h4 className="text-xs font-semibold uppercase tracking-wider text-[#667085] mb-1">Summary Assessment</h4>
                  <p className="text-xs text-[#202124] leading-relaxed bg-white p-3 rounded-lg border border-[#E5E7EB]">
                    {exp.summary}
                  </p>
                </div>

                {/* Severity Rationale */}
                <div>
                  <h4 className="text-xs font-semibold uppercase tracking-wider text-[#667085] mb-1">Severity Rationale</h4>
                  <p className="text-xs text-[#202124] leading-relaxed">
                    {exp.severity_rationale}
                  </p>
                </div>

                {/* Evidence & Uncertainties */}
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  <div>
                    <h4 className="text-xs font-semibold uppercase tracking-wider text-emerald-700 mb-1">Supporting Evidence</h4>
                    <ul className="text-xs text-[#202124] space-y-1 list-disc list-inside">
                      {(exp.supporting_evidence || []).map((ev, i) => (
                        <li key={i}>{ev}</li>
                      ))}
                    </ul>
                  </div>

                  <div>
                    <h4 className="text-xs font-semibold uppercase tracking-wider text-amber-800 mb-1">Uncertainties & Limitations</h4>
                    <ul className="text-xs text-[#202124] space-y-1 list-disc list-inside">
                      {(exp.uncertainties || []).map((un, i) => (
                        <li key={i}>{un}</li>
                      ))}
                    </ul>
                  </div>
                </div>

                {/* Recommended Actions */}
                <div>
                  <h4 className="text-xs font-semibold uppercase tracking-wider text-sky-700 mb-1">Recommended Investigation Actions</h4>
                  <ul className="text-xs text-[#202124] space-y-1 list-decimal list-inside">
                    {(exp.recommended_actions || []).map((act, i) => (
                      <li key={i}>{act}</li>
                    ))}
                  </ul>
                </div>

                {/* Disclaimer */}
                <p className="text-[10px] text-[#667085] italic pt-2 border-t border-[#E5E7EB]">
                  {exp.disclaimer || 'Advisory automated explanation grounded in flow telemetry. Not proof of compromise.'}
                </p>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}

export default AlertDetailPage;
