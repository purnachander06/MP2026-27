import React, { useEffect, useState, useCallback } from 'react';
import { useSearchParams } from 'react-router-dom';
import {
  Filter,
  RotateCcw,
  Search,
} from 'lucide-react';
import { predictionService } from '../services/predictionService';
import { PredictionTable } from '../components/tables/PredictionTable';
import { Pagination } from '../components/common/Pagination';
import { ErrorAlert } from '../components/common/ErrorAlert';
import { DETECTOR_NAMES } from '../utils/constants';

export function AlertsPage() {
  const [searchParams, setSearchParams] = useSearchParams();

  // Query states from searchParams or defaults
  const page = Number(searchParams.get('page')) || 1;
  const pageSize = Number(searchParams.get('page_size')) || 20;
  const detectorName = searchParams.get('detector_name') || '';
  const predictedClass = searchParams.get('predicted_class') || '';
  const minProb = searchParams.get('min_malicious_probability') || '';
  const startTime = searchParams.get('start_time') || '';
  const endTime = searchParams.get('end_time') || '';
  const sortBy = searchParams.get('sort_by') || 'created_at';
  const sortOrder = searchParams.get('sort_order') || 'desc';

  const [predictions, setPredictions] = useState([]);
  const [totalItems, setTotalItems] = useState(0);
  const [totalPages, setTotalPages] = useState(1);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState(null);

  // Local filter form state
  const [filterForm, setFilterForm] = useState({
    detector_name: detectorName,
    predicted_class: predictedClass,
    min_malicious_probability: minProb,
    start_time: startTime,
    end_time: endTime,
    sort_by: sortBy,
    sort_order: sortOrder,
  });

  const fetchAlerts = useCallback(async (signal) => {
    const validSignal = (signal && typeof signal === 'object' && signal instanceof AbortSignal) ? signal : undefined;
    setIsLoading(true);
    setError(null);
    try {
      const res = await predictionService.getHistoricalPredictions(
        {
          page,
          page_size: pageSize,
          detector_name: detectorName || undefined,
          predicted_class: predictedClass || undefined,
          min_malicious_probability: minProb ? Number(minProb) : undefined,
          start_time: startTime ? new Date(startTime).toISOString() : undefined,
          end_time: endTime ? new Date(endTime).toISOString() : undefined,
          sort_by: sortBy,
          sort_order: sortOrder,
        },
        validSignal
      );
      setPredictions(res.items || []);
      setTotalItems(res.total_items || 0);
      setTotalPages(res.total_pages || 1);
    } catch (err) {
      if (err.name !== 'CanceledError' && err.name !== 'AbortError' && err.code !== 'ERR_CANCELED' && err.message !== 'canceled') {
        setError(err.message || 'Failed to fetch historical alerts');
      }
    } finally {
      setIsLoading(false);
    }
  }, [page, pageSize, detectorName, predictedClass, minProb, startTime, endTime, sortBy, sortOrder]);

  useEffect(() => {
    const controller = new AbortController();
    fetchAlerts(controller.signal);
    return () => controller.abort();
  }, [fetchAlerts]);

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

  const handleApplyFilters = (e) => {
    e.preventDefault();
    updateParam({
      ...filterForm,
      page: 1, // Reset to page 1 on filter change
    });
  };

  const handleResetFilters = () => {
    const resetState = {
      detector_name: '',
      predicted_class: '',
      min_malicious_probability: '',
      start_time: '',
      end_time: '',
      sort_by: 'created_at',
      sort_order: 'desc',
    };
    setFilterForm(resetState);
    setSearchParams({ page: '1', page_size: String(pageSize) });
  };

  const handlePageChange = (newPage) => {
    updateParam({ page: newPage });
  };

  const handlePageSizeChange = (newSize) => {
    updateParam({ page_size: newSize, page: 1 });
  };

  return (
    <div className="space-y-6">
      {/* Filters Card */}
      <div className="bg-white border border-[#E5E7EB] rounded-xl p-5 shadow-sm">
        <div className="flex items-center justify-between mb-4">
          <div className="flex items-center gap-2 text-sm font-bold text-[#202124]">
            <Filter className="w-4 h-4 text-[#F26522]" />
            <span>Server-Side Alert Telemetry Filters</span>
          </div>
          <button
            onClick={handleResetFilters}
            className="inline-flex items-center gap-1 text-xs text-[#667085] hover:text-[#202124] transition-colors"
          >
            <RotateCcw className="w-3.5 h-3.5" />
            <span>Reset</span>
          </button>
        </div>

        <form onSubmit={handleApplyFilters} className="space-y-4">
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
            {/* Classification Class */}
            <div>
              <label
                htmlFor="filter-predicted-class"
                className="block text-[11px] font-medium uppercase tracking-wider text-[#667085] mb-1"
              >
                Predicted Class
              </label>
              <select
                id="filter-predicted-class"
                value={filterForm.predicted_class}
                onChange={(e) => setFilterForm({ ...filterForm, predicted_class: e.target.value })}
                className="w-full bg-white border border-[#E5E7EB] rounded-lg px-3 py-2 text-xs text-[#202124] focus:outline-none focus:border-[#F26522]"
              >
                <option value="">All Outcomes</option>
                <option value="malicious">Malicious Only</option>
                <option value="benign">Benign Only</option>
              </select>
            </div>

            {/* Model Detector */}
            <div>
              <label
                htmlFor="filter-detector-name"
                className="block text-[11px] font-medium uppercase tracking-wider text-[#667085] mb-1"
              >
                Detector Model
              </label>
              <select
                id="filter-detector-name"
                value={filterForm.detector_name}
                onChange={(e) => setFilterForm({ ...filterForm, detector_name: e.target.value })}
                className="w-full bg-white border border-[#E5E7EB] rounded-lg px-3 py-2 text-xs text-[#202124] focus:outline-none focus:border-[#F26522]"
              >
                <option value="">All Models</option>
                {Object.keys(DETECTOR_NAMES).map((k) => (
                  <option key={k} value={k}>
                    {DETECTOR_NAMES[k]}
                  </option>
                ))}
              </select>
            </div>

            {/* Min Malicious Probability */}
            <div>
              <label
                htmlFor="filter-min-probability"
                className="block text-[11px] font-medium uppercase tracking-wider text-[#667085] mb-1"
              >
                Min Malicious Probability
              </label>
              <select
                id="filter-min-probability"
                value={filterForm.min_malicious_probability}
                onChange={(e) => setFilterForm({ ...filterForm, min_malicious_probability: e.target.value })}
                className="w-full bg-white border border-[#E5E7EB] rounded-lg px-3 py-2 text-xs text-[#202124] focus:outline-none focus:border-[#F26522]"
              >
                <option value="">Any Score (0.00+)</option>
                <option value="0.50">Suspicious (≥ 0.50)</option>
                <option value="0.75">High Threat (≥ 0.75)</option>
                <option value="0.90">Critical Confirmed (≥ 0.90)</option>
              </select>
            </div>

            {/* Sort Field */}
            <div>
              <label
                htmlFor="filter-sort-by"
                className="block text-[11px] font-medium uppercase tracking-wider text-[#667085] mb-1"
              >
                Sort By
              </label>
              <div className="flex gap-2">
                <select
                  id="filter-sort-by"
                  value={filterForm.sort_by}
                  onChange={(e) => setFilterForm({ ...filterForm, sort_by: e.target.value })}
                  className="w-2/3 bg-white border border-[#E5E7EB] rounded-lg px-3 py-2 text-xs text-[#202124] focus:outline-none focus:border-[#F26522]"
                >
                  <option value="created_at">Timestamp</option>
                  <option value="malicious_probability">Malicious Score</option>
                  <option value="inference_time_ms">Latency</option>
                  <option value="prediction_id">Prediction ID</option>
                </select>
                <select
                  id="filter-sort-order"
                  aria-label="Sort direction"
                  value={filterForm.sort_order}
                  onChange={(e) => setFilterForm({ ...filterForm, sort_order: e.target.value })}
                  className="w-1/3 bg-white border border-[#E5E7EB] rounded-lg px-2 py-2 text-xs text-[#202124] focus:outline-none focus:border-[#F26522]"
                >
                  <option value="desc">DESC</option>
                  <option value="asc">ASC</option>
                </select>
              </div>
            </div>
          </div>

          <div className="flex justify-end pt-2">
            <button
              type="submit"
              className="inline-flex items-center gap-1.5 px-4 py-2 rounded-lg bg-[#F26522] hover:bg-[#F26522]/90 text-white text-xs font-semibold shadow-sm transition-colors"
            >
              <Search className="w-3.5 h-3.5" />
              <span>Apply Filters</span>
            </button>
          </div>
        </form>
      </div>

      {error && <ErrorAlert message={error} onRetry={() => fetchAlerts()} />}

      {/* Table Container */}
      <div className="bg-white border border-[#E5E7EB] rounded-xl shadow-sm">
        <div className="p-4 sm:p-5 border-b border-[#E5E7EB] flex items-center justify-between">
          <div>
            <h2 className="text-sm font-bold text-[#202124]">Historical Prediction Archive</h2>
            <p className="text-xs text-[#667085]">Click any row to view validated flow telemetry and linked explanations</p>
          </div>
          <span className="text-xs font-mono text-[#667085]">
            {totalItems.toLocaleString()} matching records
          </span>
        </div>

        <PredictionTable
          predictions={predictions}
          isLoading={isLoading}
        />

        <div className="p-4">
          <Pagination
            currentPage={page}
            totalPages={totalPages}
            totalItems={totalItems}
            pageSize={pageSize}
            onPageChange={handlePageChange}
            onPageSizeChange={handlePageSizeChange}
          />
        </div>
      </div>
    </div>
  );
}

export default AlertsPage;
