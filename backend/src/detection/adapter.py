"""Model Prediction Adapter for Phase 6 Threat Detection & Explanation.

Provides a unified, production-grade interface to load and execute predictions across
all validated intrusion detection models (Centralized Random Forest, Centralized
Logistic Regression, FedAvg, FedOpt, and GSFS) with strict input validation,
leakage prevention, and feature schema compliance.
"""

from dataclasses import dataclass, field
import json
import logging
from pathlib import Path
import time
from typing import Dict, List, Optional, Union, Any, Tuple
import uuid
import joblib
import numpy as np
import pandas as pd
import torch

from src.detection.data_loader import (
    DEFAULT_FEATURE_COLUMNS,
    DEFAULT_NUMERIC_FEATURES,
    DEFAULT_CATEGORICAL_FEATURES,
    LABEL_MAPPING,
    INVERSE_LABEL_MAPPING,
)
from src.federated.model import IoTThreatClassifier

logger = logging.getLogger("ThreatDetectionAdapter")

# Explicit list of excluded/leakage fields that must never be used as features
FORBIDDEN_LEAKAGE_COLUMNS = {
    "ts",
    "uid",
    "id.orig_h",
    "id.resp_h",
    "label",
    "detailed_label",
    "scenario",
}


@dataclass
class ModelMetadata:
    """Metadata describing a registered detection model."""
    name: str
    display_name: str
    model_type: str  # 'sklearn_pipeline' or 'pytorch_federated'
    file_path: Path
    threshold: float
    description: str
    version: str = "1.0.0"
    is_available: bool = False
    details: Dict[str, Any] = field(default_factory=dict)


class ThreatDetectionAdapter:
    """
    Unified adapter managing model loading, schema validation, and inference
    for centralized and federated IoT intrusion detection models.
    """

    def __init__(
        self,
        models_dir: Union[str, Path] = "models",
        default_detector: str = "random_forest",
    ):
        self.models_dir = Path(models_dir)
        self.default_detector = default_detector.lower().strip()

        # Cache for instantiated models and shared preprocessors
        self._model_cache: Dict[str, Any] = {}
        self._shared_nn_preprocessor: Optional[Any] = None

        # Registered models
        self.registry: Dict[str, ModelMetadata] = {
            "random_forest": ModelMetadata(
                name="random_forest",
                display_name="Centralized Random Forest (Reference Detector)",
                model_type="sklearn_pipeline",
                file_path=self.models_dir / "best_model_pipeline.joblib",
                threshold=0.60,
                description="Best performing centralized ensemble model trained on 33,602 IoT-23 flows.",
                version="1.0.0",
            ),
            "logistic_regression": ModelMetadata(
                name="logistic_regression",
                display_name="Centralized Logistic Regression",
                model_type="sklearn_pipeline",
                file_path=self.models_dir / "baseline_lr_pipeline.joblib",
                threshold=0.50,
                description="Linear baseline model with StandardScaler and balanced class weighting.",
                version="1.0.0",
            ),
            "fedavg": ModelMetadata(
                name="fedavg",
                display_name="Federated FedAvg Classifier",
                model_type="pytorch_federated",
                file_path=self.models_dir / "federated" / "fedavg_model.pt",
                threshold=0.50,
                description="Federated 2-layer MLP (125->64->32->2) trained across 5 Non-IID clients using FedAvg.",
                version="1.0.0",
            ),
            "fedopt": ModelMetadata(
                name="fedopt",
                display_name="Federated FedOpt (FedAdam) Classifier",
                model_type="pytorch_federated",
                file_path=self.models_dir / "federated" / "fedopt_model.pt",
                threshold=0.50,
                description="Federated 2-layer MLP trained using server-side adaptive FedAdam optimization.",
                version="1.0.0",
            ),
            "gsfs": ModelMetadata(
                name="gsfs",
                display_name="Federated GSFS Classifier",
                model_type="pytorch_federated",
                file_path=self.models_dir / "federated" / "gsfs_model.pt",
                threshold=0.50,
                description="Federated 2-layer MLP trained with Gradient-Selective Federated Sampling and Update Pool.",
                version="1.0.0",
            ),
        }

        # Check file availability on initialization
        self._refresh_availability()

    def _refresh_availability(self) -> None:
        """Scan filesystem and update availability flags."""
        # Fallback for random_forest if best_model_pipeline is missing but baseline_rf exists
        rf_meta = self.registry["random_forest"]
        if not rf_meta.file_path.exists():
            alt_rf = self.models_dir / "baseline_rf_pipeline.joblib"
            if alt_rf.exists():
                rf_meta.file_path = alt_rf

        for meta in self.registry.values():
            meta.is_available = meta.file_path.exists()
            if meta.is_available:
                try:
                    meta.details["file_size_bytes"] = meta.file_path.stat().st_size
                except Exception:
                    pass

    def list_available_models(self) -> List[Dict[str, Any]]:
        """Return list of supported detectors and their readiness status."""
        self._refresh_availability()
        result = []
        for name, meta in self.registry.items():
            result.append({
                "detector_name": meta.name,
                "display_name": meta.display_name,
                "model_type": meta.model_type,
                "is_available": meta.is_available,
                "threshold": meta.threshold,
                "version": meta.version,
                "description": meta.description,
                "file_path": str(meta.file_path.name),
            })
        return result

    def get_model(self, detector_name: str) -> Tuple[Any, ModelMetadata]:
        """
        Retrieve and cache model instance by detector name.
        """
        name = detector_name.lower().strip()
        if name not in self.registry:
            supported = list(self.registry.keys())
            raise ValueError(f"Unknown detector '{detector_name}'. Supported detectors: {supported}")

        meta = self.registry[name]
        if not meta.file_path.exists():
            raise FileNotFoundError(
                f"Model checkpoint for '{name}' not found at {meta.file_path.resolve()}."
            )

        if name not in self._model_cache:
            logger.info(f"Loading detector '{name}' from {meta.file_path}...")
            start_t = time.time()
            if meta.model_type == "sklearn_pipeline":
                loaded_model = joblib.load(meta.file_path)
                self._model_cache[name] = loaded_model
            elif meta.model_type == "pytorch_federated":
                # Ensure shared preprocessor is available
                preproc = self._get_nn_preprocessor()
                classifier = IoTThreatClassifier(input_dim=125, hidden_dims=(64, 32), dropout_rate=0.15)
                checkpoint = torch.load(meta.file_path, map_location="cpu")
                state_dict = (
                    checkpoint["model_state_dict"]
                    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint
                    else checkpoint
                )
                classifier.load_state_dict(state_dict)
                classifier.eval()
                self._model_cache[name] = (classifier, preproc)
            else:
                raise ValueError(f"Unsupported model type '{meta.model_type}'.")

            elapsed = time.time() - start_t
            logger.info(f"Loaded detector '{name}' in {elapsed:.3f}s.")

        return self._model_cache[name], meta

    def _get_nn_preprocessor(self) -> Any:
        """
        Retrieve fitted ColumnTransformer with StandardScaler for neural networks.
        Extracts preprocessor from baseline_lr_pipeline.joblib.
        """
        if self._shared_nn_preprocessor is not None:
            return self._shared_nn_preprocessor

        lr_path = self.models_dir / "baseline_lr_pipeline.joblib"
        if not lr_path.exists():
            raise FileNotFoundError(
                f"Required preprocessor source '{lr_path}' not found. Cannot preprocess for neural networks."
            )

        lr_pipe = joblib.load(lr_path)
        if hasattr(lr_pipe, "named_steps") and "preprocessor" in lr_pipe.named_steps:
            self._shared_nn_preprocessor = lr_pipe.named_steps["preprocessor"]
            return self._shared_nn_preprocessor

        raise ValueError("Failed to extract fitted preprocessor from baseline LR pipeline.")

    def sanitize_input_record(self, raw_record: Dict[str, Any]) -> Tuple[Dict[str, Any], List[str]]:
        """
        Validate, clean, and align input telemetry record.

        - Strips forbidden leakage columns (ts, uid, IPs, labels, scenario).
        - Enforces approved feature schema (DEFAULT_FEATURE_COLUMNS).
        - Populates missing fields with None so imputers handle them safely.
        - Returns (sanitized_record, list_of_stripped_columns).
        """
        if not isinstance(raw_record, dict):
            raise TypeError(f"Record must be a dictionary, got {type(raw_record).__name__}.")

        stripped_cols = []
        sanitized = {}

        # 1. Filter out forbidden leakage attributes
        for key in list(raw_record.keys()):
            k_lower = key.lower().strip()
            if k_lower in FORBIDDEN_LEAKAGE_COLUMNS:
                stripped_cols.append(key)

        # 2. Extract approved feature columns, filling missing with None
        for col in DEFAULT_FEATURE_COLUMNS:
            val = raw_record.get(col, None)

            # Normalize missing token strings
            if val in ("-", "(empty)", "none", "null", ""):
                val = None

            # Type check numeric features
            if col in DEFAULT_NUMERIC_FEATURES and val is not None:
                try:
                    val = float(val)
                except (ValueError, TypeError):
                    logger.warning(f"Invalid non-numeric value for {col}: {val}. Setting to None.")
                    val = None

            # String check categorical features
            if col in DEFAULT_CATEGORICAL_FEATURES and val is not None:
                val = str(val).strip().lower()

            sanitized[col] = val

        return sanitized, stripped_cols

    def extract_evidence(self, sanitized: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract meaningful, validated telemetry evidence for prompt grounding.
        Only includes fields actually present and non-null in the input.
        """
        evidence = {}
        for col, val in sanitized.items():
            if val is not None:
                evidence[col] = val
        return evidence

    def predict_single(
        self,
        record: Dict[str, Any],
        detector_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Run threat detection on a single flow record.
        """
        selected_detector = detector_name or self.default_detector
        batch_results = self.predict_batch([record], detector_name=selected_detector, max_batch_size=1)
        return batch_results[0]

    def predict_batch(
        self,
        records: List[Dict[str, Any]],
        detector_name: Optional[str] = None,
        max_batch_size: int = 100,
    ) -> List[Dict[str, Any]]:
        """
        Run threat detection on a batch of flow records.

        Args:
            records: List of raw telemetry dictionaries.
            detector_name: Configurable detector choice.
            max_batch_size: Safety limit to prevent memory exhaustion.

        Returns:
            List of structured prediction result dictionaries.
        """
        if not records:
            raise ValueError("Input records batch cannot be empty.")

        if len(records) > max_batch_size:
            raise ValueError(
                f"Batch size {len(records)} exceeds maximum allowed limit of {max_batch_size}."
            )

        selected_detector = detector_name or self.default_detector
        model_obj, meta = self.get_model(selected_detector)

        # 1. Sanitize all records
        sanitized_records = []
        all_stripped = []
        for i, rec in enumerate(records):
            try:
                sanitized, stripped = self.sanitize_input_record(rec)
                sanitized_records.append(sanitized)
                all_stripped.append(stripped)
            except Exception as e:
                raise ValueError(f"Record {i} validation failed: {str(e)}") from e

        df = pd.DataFrame(sanitized_records, columns=DEFAULT_FEATURE_COLUMNS)

        # 2. Model inference
        start_inf = time.time()
        if meta.model_type == "sklearn_pipeline":
            pipeline = model_obj
            if hasattr(pipeline, "predict_proba"):
                probs = pipeline.predict_proba(df)
                mal_probs = probs[:, 1]
                ben_probs = probs[:, 0]
            else:
                preds = pipeline.predict(df)
                mal_probs = preds.astype(float)
                ben_probs = 1.0 - mal_probs

        elif meta.model_type == "pytorch_federated":
            classifier, preproc = model_obj
            X_transformed = preproc.transform(df)
            X_tensor = torch.tensor(X_transformed, dtype=torch.float32)
            with torch.no_grad():
                probs_tensor = classifier.predict_proba(X_tensor)
                probs = probs_tensor.cpu().numpy()
                ben_probs = probs[:, 0]
                mal_probs = probs[:, 1]
        else:
            raise ValueError(f"Unknown model type {meta.model_type}")

        elapsed_inf = time.time() - start_inf

        # 3. Format structured results
        results = []
        threshold = meta.threshold

        for i in range(len(records)):
            m_prob = float(mal_probs[i])
            b_prob = float(ben_probs[i])
            is_malicious = bool(m_prob >= threshold)
            pred_class = "malicious" if is_malicious else "benign"
            pred_int = 1 if is_malicious else 0
            conf_score = float(max(m_prob, b_prob))

            pred_id = f"pred-{uuid.uuid4().hex[:12]}"
            evidence = self.extract_evidence(sanitized_records[i])

            res = {
                "prediction_id": pred_id,
                "detector_name": meta.name,
                "model_version": meta.version,
                "predicted_class": pred_class,
                "prediction": pred_int,
                "malicious_probability": round(m_prob, 5),
                "benign_probability": round(b_prob, 5),
                "threshold_used": threshold,
                "confidence_score": round(conf_score, 5),
                "is_calibrated": False,  # Model output is uncalibrated probability score
                "validated_evidence": evidence,
                "stripped_leakage_fields": all_stripped[i],
                "inference_time_ms": round((elapsed_inf / len(records)) * 1000, 3),
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            }
            results.append(res)

        return results
