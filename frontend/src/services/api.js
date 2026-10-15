import axios from 'axios';

let rawBaseUrl = import.meta.env.VITE_API_BASE_URL || 'http://127.0.0.1:8000';
// Replace localhost/0.0.0.0 with 127.0.0.1 to avoid Windows IPv6 resolution issues (::1 connection refused)
const API_BASE_URL = rawBaseUrl.replace(/:\/\/(localhost|0\.0\.0\.0):/, '://127.0.0.1:');

export const apiClient = axios.create({
  baseURL: API_BASE_URL,
  timeout: 15000,
  headers: {
    'Content-Type': 'application/json',
    Accept: 'application/json',
  },
});

// Request interceptor to sanitize AbortSignal against accidental non-signal objects (e.g. React MouseEvents)
apiClient.interceptors.request.use(
  (config) => {
    if (config.signal) {
      const isAbortSignal = typeof AbortSignal !== 'undefined'
        ? (config.signal instanceof AbortSignal)
        : (typeof config.signal.addEventListener === 'function' && typeof config.signal.aborted === 'boolean');
      if (!isAbortSignal) {
        delete config.signal;
        config.signal = undefined;
      }
    }
    return config;
  },
  (error) => Promise.reject(error)
);

apiClient.interceptors.response.use(
  (response) => response.data,
  (error) => {
    // Preserve cancellation errors so React component lifecycles cleanly ignore unmounted requests
    const isCancel = axios.isCancel(error) ||
      error.code === 'ERR_CANCELED' ||
      error.name === 'CanceledError' ||
      error.name === 'AbortError' ||
      error.message === 'canceled';

    if (isCancel) {
      const cancelError = new Error('canceled');
      cancelError.name = 'CanceledError';
      cancelError.code = 'ERR_CANCELED';
      cancelError.isCanceled = true;
      return Promise.reject(cancelError);
    }

    let normalizedError = {
      name: 'ApiError',
      message: 'Network error or service unavailable. Verify FastAPI is running.',
      status: 0,
      details: null,
    };

    if (error.response) {
      normalizedError.status = error.response.status;
      const data = error.response.data;
      if (data && typeof data === 'object') {
        normalizedError.message = data.detail || data.message || `Request failed with status ${error.response.status}`;
        normalizedError.details = data.details || null;
      } else {
        normalizedError.message = `HTTP ${error.response.status}: ${error.response.statusText}`;
      }
    } else if (error.request) {
      normalizedError.message = `Cannot connect to backend at ${API_BASE_URL}. Ensure server is running.`;
    } else {
      normalizedError.message = error.message;
    }

    return Promise.reject(normalizedError);
  }
);

export default apiClient;
