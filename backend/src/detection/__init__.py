"""Baseline threat detection package for IoT-23 dataset."""

from src.detection.data_loader import (
    load_dataset_split,
    load_feature_metadata,
    validate_feature_dataframe,
    DEFAULT_FEATURE_COLUMNS,
    DEFAULT_NUMERIC_FEATURES,
    DEFAULT_CATEGORICAL_FEATURES,
    LABEL_MAPPING,
    INVERSE_LABEL_MAPPING,
)
from src.detection.pipeline import (
    build_preprocessor,
    build_logistic_regression_pipeline,
    build_random_forest_pipeline,
    get_model_pipeline,
)
from src.detection.evaluate import (
    compute_metrics,
    find_optimal_threshold,
    evaluate_per_scenario,
    plot_confusion_matrix,
    plot_roc_and_pr_curves,
)
from src.detection.predict import ThreatDetector
from src.detection.adapter import ThreatDetectionAdapter

__all__ = [
    "load_dataset_split",
    "load_feature_metadata",
    "validate_feature_dataframe",
    "DEFAULT_FEATURE_COLUMNS",
    "DEFAULT_NUMERIC_FEATURES",
    "DEFAULT_CATEGORICAL_FEATURES",
    "LABEL_MAPPING",
    "INVERSE_LABEL_MAPPING",
    "build_preprocessor",
    "build_logistic_regression_pipeline",
    "build_random_forest_pipeline",
    "get_model_pipeline",
    "compute_metrics",
    "find_optimal_threshold",
    "evaluate_per_scenario",
    "plot_confusion_matrix",
    "plot_roc_and_pr_curves",
    "ThreatDetector",
    "ThreatDetectionAdapter",
]
