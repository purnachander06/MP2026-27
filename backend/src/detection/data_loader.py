"""Data loading and schema enforcement for threat detection models.

Guarantees:
- Strict isolation of ML features from target and metadata columns (preventing data leakage).
- Validation of expected feature schema against feature_metadata.json.
- Deterministic binary target mapping: 'benign' -> 0, 'malicious' -> 1.
- Separate extraction of scenario and trace metadata for client partitioning and evaluation.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
import json
import logging
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

# Canonical feature schema
DEFAULT_NUMERIC_FEATURES = [
    "id.orig_p",
    "id.resp_p",
    "duration",
    "orig_bytes",
    "resp_bytes",
    "missed_bytes",
    "orig_pkts",
    "orig_ip_bytes",
    "resp_pkts",
    "resp_ip_bytes",
]

DEFAULT_CATEGORICAL_FEATURES = [
    "proto",
    "service",
    "conn_state",
    "local_orig",
    "local_resp",
    "history",
]

DEFAULT_FEATURE_COLUMNS = DEFAULT_NUMERIC_FEATURES + DEFAULT_CATEGORICAL_FEATURES

EXCLUDED_COLUMNS = [
    "uid",
    "id.orig_h",
    "id.resp_h",
    "ts",
    "tunnel_parents",
    "scenario",
    "label",
    "detailed_label",
]

LABEL_MAPPING = {
    "benign": 0,
    "malicious": 1,
}
INVERSE_LABEL_MAPPING = {v: k for k, v in LABEL_MAPPING.items()}


def load_feature_metadata(metadata_path: Union[str, Path] = "processed/feature_metadata.json") -> Dict[str, Any]:
    """Load feature schema and leakage prevention contract."""
    p = Path(metadata_path)
    if not p.exists():
        for cand in [Path("../processed/feature_metadata.json"), Path("../../processed/feature_metadata.json")]:
            if cand.exists():
                p = cand
                break
    if not p.exists():
        logger.warning(f"Metadata file {p} not found. Using default feature definitions.")
        return {
            "ml_feature_columns": DEFAULT_FEATURE_COLUMNS,
            "numeric_feature_columns": DEFAULT_NUMERIC_FEATURES,
            "categorical_feature_columns": DEFAULT_CATEGORICAL_FEATURES,
            "target_columns": ["label", "detailed_label"],
            "metadata_columns": ["scenario", "ts", "uid", "id.orig_h", "id.resp_h", "tunnel_parents"],
        }
    with p.open(encoding="utf-8") as f:
        return json.load(f)


def validate_feature_dataframe(df: pd.DataFrame, expected_features: Optional[List[str]] = None) -> None:
    """
    Validate that DataFrame contains required ML features and no leaky columns are masquerading as features.
    """
    if expected_features is None:
        expected_features = DEFAULT_FEATURE_COLUMNS

    missing_cols = [c for c in expected_features if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Input DataFrame is missing required feature columns: {missing_cols}")


def load_dataset_split(
    file_path: Union[str, Path],
    metadata_path: Union[str, Path] = "processed/feature_metadata.json",
    require_target: bool = True,
) -> Tuple[pd.DataFrame, Optional[pd.Series], pd.DataFrame]:
    """
    Load a dataset split and cleanly isolate features (X), target (y), and metadata.

    Returns:
        X: pd.DataFrame containing only ml_feature_columns.
        y: pd.Series containing binary integer targets (0=benign, 1=malicious), or None if require_target is False.
        meta_df: pd.DataFrame containing scenario and traceability metadata (never fed into model).
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset split not found: {path.resolve()}")

    df = pd.read_csv(path)
    meta_info = load_feature_metadata(metadata_path)
    feature_cols = meta_info.get("ml_feature_columns", DEFAULT_FEATURE_COLUMNS)

    # 1. Enforce leakage prevention: verify features don't include excluded columns
    for leaky in EXCLUDED_COLUMNS:
        if leaky in feature_cols:
            raise ValueError(f"Security/Leakage violation: {leaky} is in ml_feature_columns!")

    validate_feature_dataframe(df, feature_cols)

    # 2. Extract features X
    X = df[feature_cols].copy()

    # 3. Extract target y
    y = None
    if require_target:
        if "label" not in df.columns:
            raise ValueError(f"Target column 'label' missing from {path}")
        raw_labels = df["label"].astype(str).str.lower().str.strip()
        y = raw_labels.map(LABEL_MAPPING)
        if y.isnull().any():
            invalid_labels = df.loc[y.isnull(), "label"].unique()
            raise ValueError(f"Found unrecognized labels that cannot be mapped: {invalid_labels}")
        y = y.astype(int)

    # 4. Extract metadata for evaluation / FL client partitioning
    meta_cols_present = [c for c in df.columns if c not in feature_cols and c != "label"]
    meta_df = df[meta_cols_present].copy()

    return X, y, meta_df
