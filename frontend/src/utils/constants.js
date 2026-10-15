export const DETECTOR_NAMES = {
  random_forest: 'Random Forest (Centralized)',
  logistic_regression: 'Logistic Regression (Centralized)',
  fedavg: 'FedAvg (Federated 5-Client)',
  fedopt: 'FedOpt / FedAdam (Federated)',
  gsfs: 'GSFS (Adaptive Federated Selection)',
};

export const DETECTOR_DESCRIPTIONS = {
  random_forest: 'Centralized ensemble baseline trained on 33,602 IoT-23 connections. Highest overall F1 and accuracy.',
  logistic_regression: 'Linear baseline with standard L2 regularization and calibrated feature scaling.',
  fedavg: 'Standard Federated Averaging across 5 non-IID scenario client partitions.',
  fedopt: 'Federated Optimization utilizing server-side adaptive Adam optimizer.',
  gsfs: 'Gradient-Selective Federated Sampling: combines performance gain and gradient displacement triggers with a deduplicated update pool.',
};

/**
 * OFFLINE TEST EVALUATION BENCHMARK
 * Evaluated rigorously on the held-out IoT-23 test set (4,800 records).
 * Clearly marked as offline benchmark metrics, NOT live database metrics.
 */
export const OFFLINE_BENCHMARK_RESULTS = [
  {
    detector: 'random_forest',
    displayName: 'Random Forest',
    paradigm: 'Centralized',
    accuracy: 98.90,
    macroF1: 0.9886,
    maliciousRecall: 98.48,
    benignRecall: 99.49,
    rocAuc: 0.9990,
    payloadMb: 0.0,
    trainingSamples: 33602,
  },
  {
    detector: 'logistic_regression',
    displayName: 'Logistic Regression',
    paradigm: 'Centralized',
    accuracy: 86.85,
    macroF1: 0.8540,
    maliciousRecall: 94.12,
    benignRecall: 76.45,
    rocAuc: 0.9120,
    payloadMb: 0.0,
    trainingSamples: 33602,
  },
  {
    detector: 'fedavg',
    displayName: 'FedAvg (5 Clients)',
    paradigm: 'Federated Learning',
    accuracy: 88.71,
    macroF1: 0.8769,
    maliciousRecall: 99.79,
    benignRecall: 72.87,
    rocAuc: 0.9358,
    payloadMb: 3.89,
    trainingSamples: 33602,
  },
  {
    detector: 'fedopt',
    displayName: 'FedOpt / FedAdam',
    paradigm: 'Federated Learning',
    accuracy: 88.71,
    macroF1: 0.8769,
    maliciousRecall: 99.79,
    benignRecall: 72.87,
    rocAuc: 0.9028,
    payloadMb: 3.89,
    trainingSamples: 33602,
  },
  {
    detector: 'gsfs',
    displayName: 'GSFS (Proposed)',
    paradigm: 'Federated Learning',
    accuracy: 88.71,
    macroF1: 0.8769,
    maliciousRecall: 99.79,
    benignRecall: 72.87,
    rocAuc: 0.9379,
    payloadMb: 3.89,
    trainingSamples: 33602,
  },
];

export const SEVERITY_CONFIG = {
  informational: {
    label: 'Informational',
    bg: 'bg-emerald-50',
    border: 'border-emerald-200',
    text: 'text-emerald-700',
    dot: 'bg-emerald-500',
  },
  low: {
    label: 'Low',
    bg: 'bg-sky-50',
    border: 'border-sky-200',
    text: 'text-sky-700',
    dot: 'bg-sky-500',
  },
  medium: {
    label: 'Medium',
    bg: 'bg-amber-50',
    border: 'border-amber-200',
    text: 'text-amber-800',
    dot: 'bg-amber-500',
  },
  high: {
    label: 'High',
    bg: 'bg-orange-50',
    border: 'border-orange-200',
    text: 'text-orange-800',
    dot: 'bg-orange-500',
  },
  critical: {
    label: 'Critical',
    bg: 'bg-red-50',
    border: 'border-red-200',
    text: 'text-red-700',
    dot: 'bg-red-600',
  },
};
