import React, { useEffect, useState, useCallback } from 'react';
import { useOutletContext, NavLink } from 'react-router-dom';
import {
  ShieldAlert,
  ShieldCheck,
  AlertTriangle,
  Flame,
  Cpu,
  ArrowRight,
} from 'lucide-react';
import { analyticsService } from '../services/analyticsService';
import { predictionService } from '../services/predictionService';
import { KpiCard } from '../components/common/KpiCard';
import { LoadingSpinner } from '../components/common/LoadingSpinner';
import { ErrorAlert } from '../components/common/ErrorAlert';
import { AlertTrendChart } from '../components/charts/AlertTrendChart';
import { ThreatDistributionChart } from '../components/charts/ThreatDistributionChart';
import { PredictionTable } from '../components/tables/PredictionTable';
import { formatNumber, formatPercent, formatMs } from '../utils/formatters';
import { DETECTOR_NAMES } from '../utils/constants';

export function OverviewPage() {
  const { refreshTrigger } = useOutletContext() || {};
  const [summary, setSummary] = useState(null);
  const [alertsOverTime, setAlertsOverTime] = useState([]);
  const [threatBreakdown, setThreatBreakdown] = useState([]);
  const [detectors, setDetectors] = useState([]);
  const [recentPredictions, setRecentPredictions] = useState([]);
  const [bucketSize, setBucketSize] = useState('hour');
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState(null);

  const fetchOverviewData = useCallback(async (signal) => {
    setIsLoading(true);
    setError(null);
    try {
      const [sumRes, timeRes, breakRes, detRes, recRes] = await Promise.all([
        analyticsService.getSummary({}, signal),
        analyticsService.getAlertsOverTime({ bucket: bucketSize }, signal),
        analyticsService.getThreatBreakdown({}, signal),
        analyticsService.getDetectorUsage({}, signal),
        predictionService.getHistoricalPredictions({ page: 1, page_size: 6, sort_by: 'created_at', sort_order: 'desc' }, signal),
      ]);

      setSummary(sumRes);
      setAlertsOverTime(timeRes.items || []);
      setThreatBreakdown(breakRes.items || []);
      setDetectors(detRes.items || []);
      setRecentPredictions(recRes.items || []);
    } catch (err) {
      if (err.name !== 'CanceledError' && err.name !== 'AbortError' && err.code !== 'ERR_CANCELED' && err.message !== 'canceled') {
        setError(err.message || 'Failed to fetch SOC overview data');
      }
    } finally {
      setIsLoading(false);
    }
  }, [bucketSize]);

  useEffect(() => {
    const controller = new AbortController();
    fetchOverviewData(controller.signal);
    return () => controller.abort();
  }, [fetchOverviewData, refreshTrigger]);

  if (isLoading && !summary) {
    return <LoadingSpinner text="Connecting to PostgreSQL and aggregating threat metrics..." />;
  }

  if (error && !summary) {
    return <ErrorAlert message={error} onRetry={() => fetchOverviewData()} />;
  }

  const malPct = summary?.malicious_percentage || 0;
  const humanReviewCount = summary?.requires_human_review || 0;
  const fallbackCount = summary?.fallback_explanations || 0;

  return (
    <div className="space-y-6">
      {/* KPI Cards Grid */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-5 gap-4">
        <KpiCard
          title="Total Predictions"
          value={formatNumber(summary?.total_predictions)}
          description="Durable detector classifications recorded in database"
          icon={Cpu}
          variant="default"
        />
        <KpiCard
          title="Malicious Flows"
          value={formatNumber(summary?.malicious_predictions)}
          description={`${formatPercent(malPct)} of observed network traffic`}
          icon={Flame}
          variant="critical"
          badgeText={formatPercent(malPct)}
        />
        <KpiCard
          title="Benign Connections"
          value={formatNumber(summary?.benign_predictions)}
          description="Legitimate routine IoT connection telemetry"
          icon={ShieldCheck}
          variant="success"
        />
        <KpiCard
          title="Human Review"
          value={formatNumber(humanReviewCount)}
          description="Threat alerts flagged for SOC analyst triage"
          icon={ShieldAlert}
          variant="warning"
        />
        <KpiCard
          title="Fallback Explanations"
          value={formatNumber(fallbackCount)}
          description="Deterministic rules executed during LLM offline"
          icon={AlertTriangle}
          variant={fallbackCount > 0 ? 'accent' : 'default'}
        />
      </div>

      {/* Charts Row */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Trend Area Chart (2 cols) */}
        <div className="lg:col-span-2 bg-white border border-[#E5E7EB] rounded-xl p-5 shadow-sm">
          <div className="flex items-center justify-between mb-2">
            <div>
              <h2 className="text-sm font-bold text-[#202124] tracking-tight">Threat Ingestion Trend</h2>
              <p className="text-xs text-[#667085]">Time-bucketed threat volume across the IoT network</p>
            </div>
          </div>
          <AlertTrendChart
            data={alertsOverTime}
            bucketSize={bucketSize}
            onBucketChange={setBucketSize}
          />
        </div>

        {/* Threat Distribution Bar Chart (1 col) */}
        <div className="bg-white border border-[#E5E7EB] rounded-xl p-5 shadow-sm flex flex-col justify-between">
          <div>
            <h2 className="text-sm font-bold text-[#202124] tracking-tight">Threats by Detector</h2>
            <p className="text-xs text-[#667085] mb-3">Model classifications (Malicious vs Benign)</p>
          </div>
          <ThreatDistributionChart data={threatBreakdown} />
        </div>
      </div>

      {/* Active Detector Utilization Strip */}
      <div className="bg-white border border-[#E5E7EB] rounded-xl p-5 shadow-sm">
        <h2 className="text-sm font-bold text-[#202124] tracking-tight mb-1">Fleet Detector Status</h2>
        <p className="text-xs text-[#667085] mb-4">Volume, classification breakdown, and recorded inference latency per model</p>

        {detectors.length === 0 ? (
          <p className="text-xs text-[#667085] italic">No detector activity recorded yet.</p>
        ) : (
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-5 gap-3">
            {detectors.map((det) => (
              <div
                key={det.detector_name}
                className="p-3 rounded-lg bg-[#F8F9FA] border border-[#E5E7EB] flex flex-col justify-between"
              >
                <div>
                  <span className="text-xs font-semibold text-[#202124] block truncate" title={DETECTOR_NAMES[det.detector_name] || det.detector_name}>
                    {DETECTOR_NAMES[det.detector_name] || det.detector_name}
                  </span>
                  <div className="flex items-baseline gap-2 mt-1">
                    <span className="text-lg font-mono font-bold text-[#202124]">
                      {det.total_predictions.toLocaleString()}
                    </span>
                    <span className="text-[10px] text-[#667085]">flows</span>
                  </div>
                </div>

                <div className="mt-3 pt-2 border-t border-[#E5E7EB] flex items-center justify-between text-[11px]">
                  <span className="text-[#667085]">Latency:</span>
                  <span className="font-mono text-[#F26522] font-medium">{formatMs(det.mean_inference_time_ms)}</span>
                </div>
              </div>
            ))}
          </div>
        )}
      </div>

      {/* Recent Alerts Table */}
      <div className="bg-white border border-[#E5E7EB] rounded-xl p-5 shadow-sm">
        <div className="flex items-center justify-between mb-4">
          <div>
            <h2 className="text-sm font-bold text-[#202124] tracking-tight">Recent Threat Alerts</h2>
            <p className="text-xs text-[#667085]">Latest connection records evaluated by the detection adapter</p>
          </div>
          <NavLink
            to="/alerts"
            className="inline-flex items-center gap-1.5 text-xs text-[#F26522] hover:text-[#202124] font-medium transition-colors"
          >
            <span>View All Alerts</span>
            <ArrowRight className="w-3.5 h-3.5" />
          </NavLink>
        </div>

        <PredictionTable
          predictions={recentPredictions}
          isLoading={isLoading}
        />
      </div>
    </div>
  );
}

export default OverviewPage;
