import React, { useEffect, useState, useRef, useCallback } from 'react';
import {
  Activity,
  Pause,
  Play,
  RefreshCw,
  Info,
  Clock,
  Radio,
} from 'lucide-react';
import { predictionService } from '../services/predictionService';
import { analyticsService } from '../services/analyticsService';
import { PredictionTable } from '../components/tables/PredictionTable';
import { formatTimeOnly, formatNumber, formatPercent } from '../utils/formatters';

export function MonitoringPage() {
  const [predictions, setPredictions] = useState([]);
  const [summary, setSummary] = useState(null);
  const [isPolling, setIsPolling] = useState(true);
  const [pollInterval, setPollInterval] = useState(12); // seconds
  const [lastRefreshed, setLastRefreshed] = useState(null);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState(null);
  const isFetchingRef = useRef(false);

  const fetchStreamData = useCallback(async (signal) => {
    if (isFetchingRef.current) return;
    isFetchingRef.current = true;
    try {
      const [sumRes, predRes] = await Promise.all([
        analyticsService.getSummary({}, signal),
        predictionService.getHistoricalPredictions(
          { page: 1, page_size: 15, sort_by: 'created_at', sort_order: 'desc' },
          signal
        ),
      ]);
      setSummary(sumRes);
      setPredictions(predRes.items || []);
      setLastRefreshed(new Date().toISOString());
      setError(null);
    } catch (err) {
      if (err.name !== 'CanceledError' && err.name !== 'AbortError' && err.code !== 'ERR_CANCELED' && err.message !== 'canceled') {
        setError(err.message || 'Stream fetch failed');
      }
    } finally {
      setIsLoading(false);
      isFetchingRef.current = false;
    }
  }, []);

  // Polling setup with tab visibility handling
  useEffect(() => {
    const controller = new AbortController();
    fetchStreamData(controller.signal);

    let intervalId = null;

    const startPolling = () => {
      if (intervalId) clearInterval(intervalId);
      intervalId = setInterval(() => {
        if (document.visibilityState === 'visible' && isPolling) {
          fetchStreamData();
        }
      }, pollInterval * 1000);
    };

    const handleVisibilityChange = () => {
      if (document.visibilityState === 'visible' && isPolling) {
        fetchStreamData();
      }
    };

    if (isPolling) {
      startPolling();
      document.addEventListener('visibilitychange', handleVisibilityChange);
    }

    return () => {
      controller.abort();
      if (intervalId) clearInterval(intervalId);
      document.removeEventListener('visibilitychange', handleVisibilityChange);
    };
  }, [fetchStreamData, isPolling, pollInterval]);

  const togglePolling = () => {
    setIsPolling((prev) => !prev);
  };

  return (
    <div className="space-y-6">
      {/* Stream Controls Header */}
      <div className="bg-white border border-[#E5E7EB] rounded-xl p-4 sm:p-5 flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4 shadow-sm">
        <div className="flex items-center gap-3">
          <div className="p-2.5 rounded-lg bg-[#F26522]/10 text-[#F26522] border border-[#F26522]/20">
            <Activity className="w-5 h-5 animate-pulse" />
          </div>
          <div>
            <h2 className="text-sm font-bold text-[#202124] flex items-center gap-2">
              <span>Live Telemetry Polling Stream</span>
              <span className={`inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-[10px] font-semibold uppercase ${
                isPolling ? 'bg-emerald-50 text-emerald-700 border border-emerald-200' : 'bg-amber-50 text-amber-800 border border-amber-200'
              }`}>
                <Radio className="w-2.5 h-2.5 animate-ping" />
                {isPolling ? 'Streaming' : 'Paused'}
              </span>
            </h2>
            <p className="text-xs text-[#667085]">
              Near-real-time polling interval: {pollInterval}s &bull; Pauses automatically when tab is hidden
            </p>
          </div>
        </div>

        <div className="flex items-center gap-2 self-stretch sm:self-auto justify-end">
          {/* Interval Selector */}
          <select
            value={pollInterval}
            onChange={(e) => setPollInterval(Number(e.target.value))}
            className="bg-white border border-[#E5E7EB] rounded-lg px-2.5 py-1.5 text-xs text-[#202124] focus:outline-none focus:border-[#F26522]"
          >
            <option value={5}>Every 5s</option>
            <option value={10}>Every 10s</option>
            <option value={15}>Every 15s</option>
            <option value={30}>Every 30s</option>
          </select>

          {/* Pause / Resume Button */}
          <button
            onClick={togglePolling}
            className={`inline-flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-medium border transition-colors ${
              isPolling
                ? 'bg-amber-50 hover:bg-amber-100 text-amber-800 border-amber-200'
                : 'bg-emerald-50 hover:bg-emerald-100 text-emerald-700 border-emerald-200'
            }`}
          >
            {isPolling ? (
              <>
                <Pause className="w-3.5 h-3.5" />
                Pause
              </>
            ) : (
              <>
                <Play className="w-3.5 h-3.5" />
                Resume
              </>
            )}
          </button>

          {/* Manual Pulse Button */}
          <button
            onClick={() => fetchStreamData()}
            className="inline-flex items-center gap-1 px-3 py-1.5 rounded-lg bg-white hover:bg-[#F8F9FA] text-[#202124] text-xs font-medium border border-[#E5E7EB] transition-colors shadow-sm"
            title="Poll now"
          >
            <RefreshCw className="w-3.5 h-3.5" />
          </button>
        </div>
      </div>

      {/* Scope Disclaimer Banner */}
      <div className="rounded-xl border border-sky-200 bg-sky-50 p-4 flex items-start gap-3">
        <Info className="w-4 h-4 text-sky-700 shrink-0 mt-0.5" />
        <p className="text-xs text-sky-800 leading-relaxed">
          <strong className="text-sky-900">Architecture Scope:</strong> This live stream monitors detector outputs and telemetry events persisted by the backend service. It reflects predictions generated via API or automated test suites and does not collect packets directly from local host network interfaces.
        </p>
      </div>

      {/* Live Stream Statistics Cards */}
      <div className="grid grid-cols-1 sm:grid-cols-4 gap-4">
        <div className="p-4 rounded-xl bg-white border border-[#E5E7EB] shadow-sm">
          <span className="text-[11px] uppercase tracking-wider text-[#667085]">Total Stream Records</span>
          <p className="text-2xl font-bold font-mono text-[#202124] mt-1">
            {formatNumber(summary?.total_predictions)}
          </p>
        </div>
        <div className="p-4 rounded-xl bg-white border border-red-200 shadow-sm">
          <span className="text-[11px] uppercase tracking-wider text-red-700">Malicious Detected</span>
          <p className="text-2xl font-bold font-mono text-red-600 mt-1">
            {formatNumber(summary?.malicious_predictions)} ({formatPercent(summary?.malicious_percentage)})
          </p>
        </div>
        <div className="p-4 rounded-xl bg-white border border-emerald-200 shadow-sm">
          <span className="text-[11px] uppercase tracking-wider text-emerald-700">Benign Flows</span>
          <p className="text-2xl font-bold font-mono text-emerald-700 mt-1">
            {formatNumber(summary?.benign_predictions)}
          </p>
        </div>
        <div className="p-4 rounded-xl bg-white border border-[#E5E7EB] shadow-sm flex flex-col justify-between">
          <span className="text-[11px] uppercase tracking-wider text-[#667085]">Last Telemetry Poll</span>
          <p className="text-sm font-mono text-[#202124] mt-1 flex items-center gap-1.5">
            <Clock className="w-3.5 h-3.5 text-[#667085]" />
            {lastRefreshed ? formatTimeOnly(lastRefreshed) : 'Polling...'}
          </p>
        </div>
      </div>

      {/* Incoming Stream Event Log */}
      <div className="bg-white border border-[#E5E7EB] rounded-xl p-5 shadow-sm">
        <div className="flex items-center justify-between mb-4">
          <div>
            <h3 className="text-sm font-bold text-[#202124]">Live Prediction Buffer</h3>
            <p className="text-xs text-[#667085]">Most recent connection flows ingested and classified by the detection pipeline</p>
          </div>
          <span className="text-xs text-[#667085] font-mono">
            {predictions.length} flows loaded
          </span>
        </div>

        {error && (
          <div className="mb-4 text-xs text-red-700 bg-red-50 p-3 rounded-lg border border-red-200">
            {error}
          </div>
        )}

        <PredictionTable
          predictions={predictions}
          isLoading={isLoading && predictions.length === 0}
        />
      </div>
    </div>
  );
}

export default MonitoringPage;
