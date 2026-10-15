import { apiClient } from './api';

const sanitizeSignal = (signal) =>
  (typeof AbortSignal !== 'undefined' && signal instanceof AbortSignal) ? signal : undefined;

export const predictionService = {
  getHistoricalPredictions: async (params = {}, signal) => {
    const cleanParams = {};
    Object.keys(params).forEach((key) => {
      const val = params[key];
      if (val !== undefined && val !== null && val !== '') {
        cleanParams[key] = val;
      }
    });
    return apiClient.get('/api/v1/history/predictions', { params: cleanParams, signal: sanitizeSignal(signal) });
  },

  getPredictionDetail: async (predictionId, signal) => {
    return apiClient.get(`/api/v1/history/predictions/${predictionId}`, { signal: sanitizeSignal(signal) });
  },

  predictSingle: async (features, detectorName = 'random_forest', operatorContext = null, signal) => {
    return apiClient.post('/api/v1/predictions', {
      features,
      detector_name: detectorName,
      operator_context: operatorContext,
    }, { signal: sanitizeSignal(signal) });
  },

  predictBatch: async (records, detectorName = 'random_forest', signal) => {
    return apiClient.post('/api/v1/predictions/batch', {
      records,
      detector_name: detectorName,
    }, { signal: sanitizeSignal(signal) });
  },

  analyzeThreat: async (features, detectorName = 'random_forest', operatorContext = null, signal) => {
    return apiClient.post('/api/v1/analyze', {
      features,
      detector_name: detectorName,
      operator_context: operatorContext,
    }, { signal: sanitizeSignal(signal) });
  },
};
