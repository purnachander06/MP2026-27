import { apiClient } from './api';

const sanitizeSignal = (signal) =>
  (typeof AbortSignal !== 'undefined' && signal instanceof AbortSignal) ? signal : undefined;

export const healthService = {
  getHealth: async (signal) => {
    return apiClient.get('/health', { signal: sanitizeSignal(signal) });
  },
  getReadiness: async (signal) => {
    return apiClient.get('/ready', { signal: sanitizeSignal(signal) });
  },
};
