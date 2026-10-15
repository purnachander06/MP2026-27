import { apiClient } from './api';

const sanitizeSignal = (signal) =>
  (typeof AbortSignal !== 'undefined' && signal instanceof AbortSignal) ? signal : undefined;

export const explanationService = {
  getHistoricalExplanations: async (params = {}, signal) => {
    const cleanParams = {};
    Object.keys(params).forEach((key) => {
      const val = params[key];
      if (val !== undefined && val !== null && val !== '') {
        cleanParams[key] = val;
      }
    });
    return apiClient.get('/api/v1/history/explanations', { params: cleanParams, signal: sanitizeSignal(signal) });
  },

  generateExplanation: async (payload, signal) => {
    return apiClient.post('/api/v1/explanations', payload, { signal: sanitizeSignal(signal) });
  },
};
