import { apiClient } from './api';

const sanitizeSignal = (signal) =>
  (typeof AbortSignal !== 'undefined' && signal instanceof AbortSignal) ? signal : undefined;

export const modelService = {
  getModels: async (signal) => {
    return apiClient.get('/models', { signal: sanitizeSignal(signal) });
  },
};
