"""Inference and threat detection module.

Loads the serialized scikit-learn pipeline and provides robust predictions
with input validation and probability estimation.
"""

from pathlib import Path
from typing import Dict, List, Optional, Union, Any
import json
import logging
import joblib
import numpy as np
import pandas as pd

from src.detection.data_loader import (
    DEFAULT_FEATURE_COLUMNS,
    LABEL_MAPPING,
    INVERSE_LABEL_MAPPING,
)

logger = logging.getLogger(__name__)


class ThreatDetector:
    """Production-grade threat detector loading a trained baseline pipeline."""

    def __init__(
        self,
        model_path: Union[str, Path] = "models/baseline_rf_pipeline.joblib",
        metadata_path: Union[str, Path] = "models/model_metadata.json",
    ):
        self.model_path = Path(model_path)
        self.metadata_path = Path(metadata_path)

        if not self.model_path.exists():
            raise FileNotFoundError(f"Model file not found: {self.model_path.resolve()}")

        self.pipeline = joblib.load(self.model_path)

        if self.metadata_path.exists():
            with self.metadata_path.open(encoding="utf-8") as f:
                self.metadata = json.load(f)
        else:
            self.metadata = {
                "ml_feature_columns": DEFAULT_FEATURE_COLUMNS,
                "threshold": 0.50,
            }

        self.feature_columns: List[str] = self.metadata.get("ml_feature_columns", DEFAULT_FEATURE_COLUMNS)
        self.threshold: float = float(self.metadata.get("selected_threshold", 0.50))

    def predict_record(self, record: Dict[str, Any]) -> Dict[str, Any]:
        """
        Predict threat for a single connection record.

        Args:
            record: Dictionary containing flow features.

        Returns:
            Dictionary containing prediction (0 or 1), label_name ('benign' or 'malicious'),
            malicious_probability, and confidence.
        """
        df = pd.DataFrame([record])
        batch_results = self.predict_batch(df)
        return {
            "prediction": int(batch_results["prediction"].iloc[0]),
            "label_name": str(batch_results["label_name"].iloc[0]),
            "malicious_probability": float(batch_results["malicious_probability"].iloc[0]),
            "threshold_used": self.threshold,
        }

    def predict_batch(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Predict threats for a batch DataFrame of records.

        Validates presence of feature columns, gracefully fills missing optional features,
        and applies the tuned decision threshold.
        """
        # Ensure all expected feature columns exist in df (fill missing with None)
        input_df = df.copy()
        for col in self.feature_columns:
            if col not in input_df.columns:
                input_df[col] = None

        # Restrict strictly to feature columns in the exact trained order
        X = input_df[self.feature_columns]

        # Probabilities
        if hasattr(self.pipeline, "predict_proba"):
            probs = self.pipeline.predict_proba(X)
            # Index 1 is malicious probability
            mal_prob = probs[:, 1]
            preds = (mal_prob >= self.threshold).astype(int)
        else:
            preds = self.pipeline.predict(X)
            mal_prob = preds.astype(float)

        label_names = [INVERSE_LABEL_MAPPING.get(p, "unknown") for p in preds]

        out_df = pd.DataFrame({
            "prediction": preds,
            "label_name": label_names,
            "malicious_probability": np.round(mal_prob, 5),
        })

        return out_df
