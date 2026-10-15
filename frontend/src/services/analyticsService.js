import { apiClient } from './api';

const sanitizeSignal = (signal) =>
  (typeof AbortSignal !== 'undefined' && signal instanceof AbortSignal) ? signal : undefined;

export const analyticsService = {
  getSummary: async (params = {}, signal) => {
    return apiClient.get('/api/v1/analytics/summary', { params, signal: sanitizeSignal(signal) });
  },

  getAlertsOverTime: async (params = {}, signal) => {
    return apiClient.get('/api/v1/analytics/alerts-over-time', { params, signal: sanitizeSignal(signal) });
  },

  getThreatBreakdown: async (params = {}, signal) => {
    return apiClient.get('/api/v1/analytics/threat-breakdown', { params, signal: sanitizeSignal(signal) });
  },

  getDetectorUsage: async (params = {}, signal) => {
    return apiClient.get('/api/v1/analytics/detectors', { params, signal: sanitizeSignal(signal) });
  },

  getExplanationQuality: async (params = {}, signal) => {
    return apiClient.get('/api/v1/analytics/explanations', { params, signal: sanitizeSignal(signal) });
  },
};
