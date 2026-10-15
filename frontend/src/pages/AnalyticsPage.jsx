import React, { useEffect, useState, useCallback } from 'react';
import {
  BarChart3,
  Calendar,
  Layers,
  FileText,
  Zap,
  Info,
} from 'lucide-react';
import { analyticsService } from '../services/analyticsService';
import { LoadingSpinner } from '../components/common/LoadingSpinner';
import { ErrorAlert } from '../components/common/ErrorAlert';
import { KpiCard } from '../components/common/KpiCard';
import { AlertTrendChart } from '../components/charts/AlertTrendChart';
import { ThreatDistributionChart } from '../components/charts/ThreatDistributionChart';
import { DetectorLatencyChart } from '../components/charts/DetectorLatencyChart';
import { SeverityBreakdownChart } from '../components/charts/SeverityBreakdownChart';
import { formatNumber, formatPercent } from '../utils/formatters';

export function AnalyticsPage() {
  const [summary, setSummary] = useState(null);
  const [alertsOverTime, setAlertsOverTime] = useState([]);
  const [threatBreakdown, setThreatBreakdown] = useState([]);
  const [detectorUsage, setDetectorUsage] = useState([]);
  const [explanationQuality, setExplanationQuality] = useState(null);

  const [bucketSize, setBucketSize] = useState('hour');
  const [startTime, setStartTime] = useState('');
  const [endTime, setEndTime] = useState('');

  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState(null);

  const fetchAnalytics = useCallback(async (signal) => {
    const validSignal = (signal && typeof signal === 'object' && signal instanceof AbortSignal) ? signal : undefined;
    setIsLoading(true);
    setError(null);

    const params = {};
    if (startTime) params.start_time = new Date(startTime).toISOString();
    if (endTime) params.end_time = new Date(endTime).toISOString();

    try {
      const [sumRes, timeRes, breakRes, detRes, expRes] = await Promise.all([
        analyticsService.getSummary(params, validSignal),
        analyticsService.getAlertsOverTime({ ...params, bucket: bucketSize }, validSignal),
        analyticsService.getThreatBreakdown(params, validSignal),
        analyticsService.getDetectorUsage(params, validSignal),
        analyticsService.getExplanationQuality(params, validSignal),
      ]);

      setSummary(sumRes);
      setAlertsOverTime(timeRes.items || []);
      setThreatBreakdown(breakRes.items || []);
      setDetectorUsage(detRes.items || []);
      setExplanationQuality(expRes);
    } catch (err) {
      if (err.name !== 'CanceledError' && err.name !== 'AbortError' && err.code !== 'ERR_CANCELED' && err.message !== 'canceled') {
        setError(err.message || 'Failed to fetch analytics metrics');
      }
    } finally {
      setIsLoading(false);
    }
  }, [bucketSize, startTime, endTime]);

  useEffect(() => {
    const controller = new AbortController();
    fetchAnalytics(controller.signal);
    return () => controller.abort();
  }, [fetchAnalytics]);

  const handleApplyDates = (e) => {
    e.preventDefault();
    fetchAnalytics();
  };

  const handleResetDates = () => {
    setStartTime('');
    setEndTime('');
  };

  if (isLoading && !summary) {
    return <LoadingSpinner text="Computing threat distributions and latency metrics..." />;
  }

  return (
    <div className="space-y-6">
      {/* Date Filter & Control Banner */}
      <div className="bg-white border border-[#E5E7EB] rounded-xl p-5 shadow-sm">
        <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
          <div>
            <h2 className="text-base font-bold text-[#202124] flex items-center gap-2">
              <BarChart3 className="w-5 h-5 text-[#F26522]" />
              <span>SOC Threat Analytics & Telemetry Metrics</span>
            </h2>
            <p className="text-xs text-[#667085]">
              Computed live from persisted PostgreSQL detection predictions and explanation records
            </p>
          </div>

          {/* Date Picker Form */}
          <form onSubmit={handleApplyDates} className="flex flex-wrap items-center gap-2 text-xs">
            <div className="flex items-center gap-1.5 bg-white border border-[#E5E7EB] rounded-lg px-2.5 py-1.5">
              <Calendar className="w-3.5 h-3.5 text-[#667085]" />
              <input
                type="datetime-local"
                value={startTime}
                onChange={(e) => setStartTime(e.target.value)}
                className="bg-transparent text-[#202124] focus:outline-none"
                placeholder="Start"
              />
            </div>
            <span className="text-[#667085]">to</span>
            <div className="flex items-center gap-1.5 bg-white border border-[#E5E7EB] rounded-lg px-2.5 py-1.5">
              <Calendar className="w-3.5 h-3.5 text-[#667085]" />
              <input
                type="datetime-local"
                value={endTime}
                onChange={(e) => setEndTime(e.target.value)}
                className="bg-transparent text-[#202124] focus:outline-none"
                placeholder="End"
              />
            </div>

            <button
              type="submit"
              className="px-3 py-1.5 rounded-lg bg-[#F26522] hover:bg-[#F26522]/90 text-white font-medium transition-colors shadow-sm"
            >
              Filter
            </button>
            {(startTime || endTime) && (
              <button
                type="button"
                onClick={handleResetDates}
                className="px-2.5 py-1.5 rounded-lg bg-slate-100 hover:bg-slate-200 text-slate-700 font-medium transition-colors"
              >
                Clear
              </button>
            )}
          </form>
        </div>
      </div>

      {error && <ErrorAlert message={error} onRetry={() => fetchAnalytics()} />}

      {/* KPI Overview Grid */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
        <KpiCard
          title="Total Flows Ingested"
          value={formatNumber(summary?.total_predictions)}
          description="Total predictions processed in this observation window"
          icon={Layers}
          variant="default"
        />
        <KpiCard
          title="Fleet Malicious Ratio"
          value={formatPercent(summary?.malicious_percentage)}
          description={`${formatNumber(summary?.malicious_predictions)} malicious of ${formatNumber(summary?.total_predictions)} total flows`}
          icon={Zap}
          variant="critical"
        />
        <KpiCard
          title="Explanation Fallback Rate"
          value={formatPercent(explanationQuality?.fallback_rate_pct)}
          description={`${formatNumber(explanationQuality?.fallback_count)} heuristic fallback activations`}
          icon={FileText}
          variant={explanationQuality?.fallback_count > 0 ? 'warning' : 'default'}
        />
        <KpiCard
          title="SOC Human Review Rate"
          value={formatPercent(explanationQuality?.human_review_rate_pct)}
          description={`${formatNumber(explanationQuality?.human_review_count)} alerts requiring analyst verification`}
          icon={BarChart3}
          variant="accent"
        />
      </div>

      {/* Analytics Charts Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Time-Series Trend */}
        <div className="bg-white border border-[#E5E7EB] rounded-xl p-5 shadow-sm">
          <div className="flex items-center justify-between mb-2">
            <div>
              <h3 className="text-sm font-bold text-[#202124]">Threat Activity Trend Over Time</h3>
              <p className="text-xs text-[#667085]">Total, malicious, and benign volume chronologically grouped</p>
            </div>
          </div>
          <AlertTrendChart
            data={alertsOverTime}
            bucketSize={bucketSize}
            onBucketChange={setBucketSize}
          />
        </div>

        {/* Threat Distribution by Model */}
        <div className="bg-white border border-[#E5E7EB] rounded-xl p-5 shadow-sm">
          <div className="mb-2">
            <h3 className="text-sm font-bold text-[#202124]">Intrusion Class Breakdown by Model</h3>
            <p className="text-xs text-[#667085]">Classification distribution across centralized and federated detectors</p>
          </div>
          <ThreatDistributionChart data={threatBreakdown} />
        </div>

        {/* Model Inference Latency */}
        <div className="bg-white border border-[#E5E7EB] rounded-xl p-5 shadow-sm">
          <div className="mb-2">
            <h3 className="text-sm font-bold text-[#202124]">Mean Inference Latency by Detector</h3>
            <p className="text-xs text-[#667085]">Execution time benchmark (ms per flow)</p>
          </div>
          <DetectorLatencyChart data={detectorUsage} />
        </div>

        {/* Severity Distribution */}
        <div className="bg-white border border-[#E5E7EB] rounded-xl p-5 shadow-sm">
          <div className="mb-2">
            <h3 className="text-sm font-bold text-[#202124]">Advisory Severity Level Distribution</h3>
            <p className="text-xs text-[#667085]">Proportion of assigned cybersecurity risk categories</p>
          </div>
          <SeverityBreakdownChart severityData={explanationQuality?.severity_breakdown || {}} />
        </div>
      </div>

      {/* Operational Methodology Notice */}
      <div className="rounded-xl border border-[#E5E7EB] bg-white p-4 flex items-start gap-3 shadow-sm">
        <Info className="w-4 h-4 text-[#667085] shrink-0 mt-0.5" />
        <p className="text-xs text-[#667085] leading-relaxed">
          <strong className="text-[#202124]">Methodology Note:</strong> Accuracy, Precision, and Recall require verified ground-truth labels and are documented under the Offline Benchmark in the <em>Detection Models</em> section. The metrics above represent observed operational distributions in production.
        </p>
      </div>
    </div>
  );
}

export default AnalyticsPage;
