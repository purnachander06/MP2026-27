import React, { useEffect, useState, useCallback } from 'react';
import {
  Cpu,
  CheckCircle2,
  XCircle,
  FileText,
  TrendingUp,
  Info,
} from 'lucide-react';
import { modelService } from '../services/modelService';
import { LoadingSpinner } from '../components/common/LoadingSpinner';
import { ErrorAlert } from '../components/common/ErrorAlert';
import { Badge } from '../components/common/Badge';
import {
  DETECTOR_NAMES,
  DETECTOR_DESCRIPTIONS,
  OFFLINE_BENCHMARK_RESULTS,
} from '../utils/constants';

export function ModelsPage() {
  const [modelData, setModelData] = useState(null);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState(null);

  const fetchModels = useCallback(async (signal) => {
    const validSignal = (signal && typeof signal === 'object' && signal instanceof AbortSignal) ? signal : undefined;
    setIsLoading(true);
    setError(null);
    try {
      const data = await modelService.getModels(validSignal);
      setModelData(data);
    } catch (err) {
      if (err.name === 'CanceledError' || err.name === 'AbortError' || err.code === 'ERR_CANCELED' || err.message === 'canceled') {
        return;
      }
      setError(err.message || 'Failed to fetch registered models');
    } finally {
      setIsLoading(false);
    }
  }, []);

  useEffect(() => {
    const controller = new AbortController();
    fetchModels(controller.signal);
    return () => controller.abort();
  }, [fetchModels]);

  if (isLoading) {
    return <LoadingSpinner text="Checking model checkpoints and registry metadata..." />;
  }

  if (error) {
    return <ErrorAlert message={error} onRetry={() => fetchModels()} />;
  }

  const models = modelData?.models || [];
  const defaultDetector = modelData?.default_detector || 'random_forest';

  return (
    <div className="space-y-8">
      {/* Registered Live Models Header */}
      <div className="bg-white border border-[#E5E7EB] rounded-xl p-5 shadow-sm">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2">
          <div>
            <h2 className="text-base font-bold text-[#202124] flex items-center gap-2">
              <Cpu className="w-5 h-5 text-[#F26522]" />
              <span>Registered Detection Models</span>
            </h2>
            <p className="text-xs text-[#667085]">
              Machine learning classifiers deployed and loaded by the Threat Detection Adapter
            </p>
          </div>
          <span className="text-xs font-mono text-[#202124] bg-[#F8F9FA] px-3 py-1 rounded-full border border-[#E5E7EB]">
            Default Active: <strong className="text-[#F26522]">{defaultDetector}</strong>
          </span>
        </div>

        {/* Live Model Cards Grid */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4 mt-5">
          {models.map((m) => {
            const isAvail = m.is_available;
            const isDefault = m.detector_name === defaultDetector;

            return (
              <div
                key={m.detector_name}
                className="bg-[#F8F9FA] border border-[#E5E7EB] rounded-xl p-4 flex flex-col justify-between hover:border-slate-300 transition-colors"
              >
                <div>
                  <div className="flex items-start justify-between gap-2">
                    <div>
                      <h3 className="text-sm font-bold text-[#202124]">
                        {DETECTOR_NAMES[m.detector_name] || m.display_name || m.detector_name}
                      </h3>
                      <span className="text-[10px] font-mono text-[#667085] block mt-0.5">
                        ID: {m.detector_name} &bull; v{m.version}
                      </span>
                    </div>
                    {isAvail ? (
                      <span className="inline-flex items-center gap-1 text-[11px] text-emerald-700 bg-emerald-50 px-2 py-0.5 rounded border border-emerald-200 font-medium">
                        <CheckCircle2 className="w-3 h-3" />
                        Ready
                      </span>
                    ) : (
                      <span className="inline-flex items-center gap-1 text-[11px] text-red-700 bg-red-50 px-2 py-0.5 rounded border border-red-200 font-medium">
                        <XCircle className="w-3 h-3" />
                        Missing
                      </span>
                    )}
                  </div>

                  <p className="text-xs text-[#667085] mt-3 leading-relaxed">
                    {DETECTOR_DESCRIPTIONS[m.detector_name] || m.description || 'Pre-trained intrusion detection classifier.'}
                  </p>
                </div>

                <div className="mt-4 pt-3 border-t border-[#E5E7EB] space-y-1.5 text-xs font-mono text-[#667085]">
                  <div className="flex justify-between">
                    <span>Architecture:</span>
                    <span className="text-[#202124]">{m.model_type}</span>
                  </div>
                  <div className="flex justify-between">
                    <span>Decision Threshold:</span>
                    <span className="text-[#202124]">{m.threshold}</span>
                  </div>
                  <div className="flex justify-between truncate" title={m.file_path}>
                    <span>Checkpoint:</span>
                    <span className="text-[#667085] truncate max-w-[140px]">{m.file_path}</span>
                  </div>
                </div>
              </div>
            );
          })}
        </div>
      </div>

      {/* Research Offline Benchmark Evaluation Section */}
      <div className="bg-white border border-[#E5E7EB] rounded-xl p-5 shadow-sm space-y-4">
        <div>
          <div className="flex items-center gap-2">
            <TrendingUp className="w-5 h-5 text-sky-600" />
            <h2 className="text-base font-bold text-[#202124]">
              Offline Test-Set Benchmark Evaluation (4,800 Samples)
            </h2>
          </div>
          <p className="text-xs text-[#667085] mt-1">
            Comparative evaluation performed across held-out IoT-23 evaluation test flows (Phases 2–5).
          </p>
        </div>

        {/* Clear Disclaimer Alert */}
        <div className="p-3.5 rounded-lg bg-sky-50 border border-sky-200 flex items-start gap-2.5">
          <Info className="w-4 h-4 text-sky-700 shrink-0 mt-0.5" />
          <p className="text-xs text-sky-800 leading-relaxed">
            <strong>Evaluation Ground Truth Disclosure:</strong> The metrics below represent offline supervised scientific benchmark results measured against ground-truth labels on the test split (4,800 records). They are strictly distinct from live production telemetry in the database where ground-truth labels are unknown.
          </p>
        </div>

        {/* Benchmark Comparison Table */}
        <div className="overflow-x-auto rounded-lg border border-[#E5E7EB]">
          <table className="w-full text-left text-xs border-collapse">
            <thead>
              <tr className="bg-[#F8F9FA] border-b border-[#E5E7EB] text-[#667085] uppercase text-[10px] tracking-wider">
                <th className="py-3 px-4">Detector Algorithm</th>
                <th className="py-3 px-4">Paradigm</th>
                <th className="py-3 px-4">Accuracy</th>
                <th className="py-3 px-4">Macro F1</th>
                <th className="py-3 px-4">Malicious Recall</th>
                <th className="py-3 px-4">Benign Recall</th>
                <th className="py-3 px-4">ROC-AUC</th>
                <th className="py-3 px-4">Comm. Payload</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-[#E5E7EB] font-mono text-[#202124]">
              {OFFLINE_BENCHMARK_RESULTS.map((bench) => {
                const isProposed = bench.detector === 'gsfs';
                return (
                  <tr
                    key={bench.detector}
                    className={`hover:bg-[#F8F9FA] transition-colors ${
                      isProposed ? 'bg-orange-50/50 font-semibold' : ''
                    }`}
                  >
                    <td className="py-3 px-4 font-sans font-medium text-[#202124]">
                      {bench.displayName}
                      {isProposed && (
                        <span className="ml-2 text-[10px] px-1.5 py-0.5 rounded bg-orange-100 text-[#F26522] border border-orange-200">
                          Proposed
                        </span>
                      )}
                    </td>
                    <td className="py-3 px-4 font-sans text-[#667085]">{bench.paradigm}</td>
                    <td className="py-3 px-4 text-[#202124]">{bench.accuracy.toFixed(2)}%</td>
                    <td className="py-3 px-4 text-[#202124]">{bench.macroF1.toFixed(4)}</td>
                    <td className="py-3 px-4 text-emerald-700 font-semibold">{bench.maliciousRecall.toFixed(2)}%</td>
                    <td className="py-3 px-4 text-[#202124]">{bench.benignRecall.toFixed(2)}%</td>
                    <td className="py-3 px-4 text-[#202124]">{bench.rocAuc.toFixed(4)}</td>
                    <td className="py-3 px-4 text-[#667085]">
                      {bench.payloadMb > 0 ? `${bench.payloadMb.toFixed(2)} MB` : 'N/A (Centralized)'}
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>

        {/* Scientific Findings Summary */}
        <div className="bg-[#F8F9FA] p-4 rounded-lg border border-[#E5E7EB] space-y-2 text-xs text-[#202124]">
          <h4 className="font-semibold text-[#202124] uppercase tracking-wider text-[11px]">
            Key Research Findings (Phase 5 Validation):
          </h4>
          <ul className="space-y-1.5 list-disc list-inside text-[#667085]">
            <li>
              <strong className="text-[#202124]">Intrusion Detection Performance Retention:</strong> GSFS matches the identical 88.71% test accuracy and <strong className="text-emerald-700">99.79% Malicious Recall</strong> of FedAvg, demonstrating that adaptive update pooling does not compromise intrusion sensitivity.
            </li>
            <li>
              <strong className="text-[#202124]">Communication & Straggler Resilience:</strong> By filtering non-productive updates and buffering parameter deltas, GSFS prevents degraded client updates from polluting the global federated model.
            </li>
          </ul>
        </div>
      </div>
    </div>
  );
}

export default ModelsPage;
