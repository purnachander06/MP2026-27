"""Modular scikit-learn Pipelines and ColumnTransformer for threat detection.

Features:
- Preprocessing fitted on training data only via Pipeline.
- Missing-value imputation: median for numeric features, constant for categorical features.
- Safe OneHotEncoder with handle_unknown='ignore' so unseen categories at test/inference time do not crash.
- Model-specific scaling: StandardScaler for Logistic Regression.
- Class weighting applied to address severe class imbalance.
- Reproducible random state enforcement.
"""

from typing import List, Optional, Tuple, Dict, Any
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from src.detection.data_loader import (
    DEFAULT_NUMERIC_FEATURES,
    DEFAULT_CATEGORICAL_FEATURES,
)


def build_preprocessor(
    numeric_features: Optional[List[str]] = None,
    categorical_features: Optional[List[str]] = None,
    scale_numeric: bool = False,
) -> ColumnTransformer:
    """
    Construct ColumnTransformer for numeric and categorical feature preprocessing.

    Args:
        numeric_features: List of numeric column names.
        categorical_features: List of categorical column names.
        scale_numeric: Whether to apply StandardScaler (required for linear models).
    """
    if numeric_features is None:
        numeric_features = DEFAULT_NUMERIC_FEATURES
    if categorical_features is None:
        categorical_features = DEFAULT_CATEGORICAL_FEATURES

    # Numeric pipeline
    numeric_steps = [
        ("imputer", SimpleImputer(strategy="median")),
    ]
    if scale_numeric:
        numeric_steps.append(("scaler", StandardScaler()))
    numeric_transformer = Pipeline(steps=numeric_steps)

    # Categorical pipeline: impute missing with 'missing', encode with handle_unknown='ignore'
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )

    return preprocessor


def build_logistic_regression_pipeline(
    random_state: int = 42,
    numeric_features: Optional[List[str]] = None,
    categorical_features: Optional[List[str]] = None,
    max_iter: int = 1000,
    C: float = 1.0,
    **kwargs,
) -> Pipeline:
    """Build Logistic Regression pipeline with StandardScaler and balanced class weights."""
    import inspect
    lr_params = inspect.signature(LogisticRegression.__init__).parameters
    filtered_kwargs = {k: v for k, v in kwargs.items() if k in lr_params}

    preprocessor = build_preprocessor(
        numeric_features=numeric_features,
        categorical_features=categorical_features,
        scale_numeric=True,
    )
    classifier = LogisticRegression(
        C=C,
        max_iter=max_iter,
        class_weight="balanced",
        random_state=random_state,
        solver="lbfgs",
        **filtered_kwargs,
    )
    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", classifier),
        ]
    )


def build_random_forest_pipeline(
    random_state: int = 42,
    numeric_features: Optional[List[str]] = None,
    categorical_features: Optional[List[str]] = None,
    n_estimators: int = 100,
    max_depth: Optional[int] = 16,
    min_samples_split: int = 5,
    **kwargs,
) -> Pipeline:
    """Build Random Forest pipeline with median imputation, one-hot encoding, and balanced class weights."""
    import inspect
    rf_params = inspect.signature(RandomForestClassifier.__init__).parameters
    filtered_kwargs = {k: v for k, v in kwargs.items() if k in rf_params}

    preprocessor = build_preprocessor(
        numeric_features=numeric_features,
        categorical_features=categorical_features,
        scale_numeric=False,
    )
    classifier = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        class_weight="balanced",
        random_state=random_state,
        n_jobs=-1,
        **filtered_kwargs,
    )
    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", classifier),
        ]
    )


def get_model_pipeline(
    model_name: str,
    random_state: int = 42,
    numeric_features: Optional[List[str]] = None,
    categorical_features: Optional[List[str]] = None,
    **kwargs,
) -> Pipeline:
    """Factory function to instantiate baseline pipelines."""
    name_clean = model_name.lower().strip().replace(" ", "_").replace("-", "_")
    if name_clean in ("logistic_regression", "lr", "logistic"):
        return build_logistic_regression_pipeline(
            random_state=random_state,
            numeric_features=numeric_features,
            categorical_features=categorical_features,
            **kwargs,
        )
    elif name_clean in ("random_forest", "rf", "forest"):
        return build_random_forest_pipeline(
            random_state=random_state,
            numeric_features=numeric_features,
            categorical_features=categorical_features,
            **kwargs,
        )
    else:
        raise ValueError(f"Unknown model name '{model_name}'. Supported: 'logistic_regression', 'random_forest'.")
