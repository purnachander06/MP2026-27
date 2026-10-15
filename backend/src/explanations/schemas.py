"""Pydantic schemas for Phase 6 Threat Detection & LLM Explanations.

Defines validated request/response models with strict type constraints,
field descriptions, and documented severity levels.
"""

from enum import Enum
from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field, field_validator


class SeverityLevel(str, Enum):
    """
    Documented Cybersecurity Severity Scale:
    - informational: Standard benign/routine telemetry with expected protocol patterns.
    - low: Minor deviation or low-confidence anomaly; unlikely to indicate compromise.
    - medium: Moderate anomaly or suspicious connection state (e.g. repeated SYNs, unusual ports)
             requiring routine investigation.
    - high: High-confidence malicious flow with significant behavioral anomalies
            (e.g. C2 beaconing, port sweep, unauthorized protocol encapsulation).
    - critical: Confirmed high-volume flood, mass scanning, active data exfiltration, or
               destructive exploit attempt threatening host/network integrity.
    """
    INFORMATIONAL = "informational"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ThreatPredictionRequest(BaseModel):
    """Request payload for single-flow threat detection."""
    features: Dict[str, Any] = Field(
        ...,
        description="Dictionary of IoT connection telemetry features matching IoT-23 schema.",
        examples=[{
            "id.orig_p": 443,
            "id.resp_p": 5353,
            "proto": "udp",
            "service": "dns",
            "duration": 0.05,
            "orig_bytes": 120,
            "resp_bytes": 240,
            "conn_state": "SF",
            "history": "Dd",
            "orig_pkts": 2,
            "resp_pkts": 2
        }],
    )
    detector_name: Optional[str] = Field(
        default="random_forest",
        description="Configurable detector: 'random_forest', 'logistic_regression', 'fedavg', 'fedopt', 'gsfs'.",
    )
    operator_context: Optional[str] = Field(
        default=None,
        max_length=500,
        description="Optional untrusted operator notes or SOC context (sanitized prior to processing).",
    )


class BatchThreatPredictionRequest(BaseModel):
    """Request payload for batch threat detection."""
    records: List[Dict[str, Any]] = Field(
        ...,
        min_length=1,
        max_length=100,
        description="Batch list of flow telemetry records (maximum 100 flows per request).",
    )
    detector_name: Optional[str] = Field(
        default="random_forest",
        description="Configurable detector choice applied across the batch.",
    )


class ThreatPredictionResponse(BaseModel):
    """Standardized threat detection output produced by the prediction adapter."""
    prediction_id: str = Field(..., description="Unique correlation identifier for this prediction.")
    detector_name: str = Field(..., description="Identifier of the detector model used.")
    model_version: str = Field(..., description="Model version string.")
    predicted_class: str = Field(..., description="Binary classification outcome: 'benign' or 'malicious'.")
    prediction: int = Field(..., ge=0, le=1, description="Integer class label (0 for benign, 1 for malicious).")
    malicious_probability: float = Field(
        ..., ge=0.0, le=1.0,
        description="Uncalibrated probability score for the malicious class."
    )
    benign_probability: float = Field(
        ..., ge=0.0, le=1.0,
        description="Uncalibrated probability score for the benign class."
    )
    threshold_used: float = Field(..., ge=0.0, le=1.0, description="Decision threshold applied.")
    confidence_score: float = Field(
        ..., ge=0.0, le=1.0,
        description="Max class probability score representing model confidence."
    )
    is_calibrated: bool = Field(
        default=False,
        description="Whether probability scores are formally calibrated (False indicates raw scores)."
    )
    validated_evidence: Dict[str, Any] = Field(
        default_factory=dict,
        description="Sanitized non-null flow features grounded in input telemetry."
    )
    stripped_leakage_fields: List[str] = Field(
        default_factory=list,
        description="Input fields excluded/stripped to prevent data leakage."
    )
    inference_time_ms: float = Field(..., ge=0.0, description="Inference latency in milliseconds.")
    timestamp: str = Field(..., description="ISO 8601 UTC timestamp of prediction generation.")


class ThreatExplanationRequest(BaseModel):
    """Request payload to generate an LLM explanation for an existing prediction."""
    prediction_id: str = Field(..., description="Prediction identifier to correlate with.")
    predicted_class: str = Field(..., description="Classifier predicted label: 'benign' or 'malicious'.")
    detector_name: str = Field(..., description="Detector model identifier that produced the prediction.")
    malicious_probability: float = Field(..., ge=0.0, le=1.0, description="Predicted malicious probability score.")
    confidence_score: float = Field(..., ge=0.0, le=1.0, description="Model confidence score.")
    validated_evidence: Dict[str, Any] = Field(..., description="Grounded telemetry evidence from input flow.")
    model_version: Optional[str] = Field(default="1.0.0", description="Model version string.")
    operator_context: Optional[str] = Field(
        default=None,
        max_length=500,
        description="Optional operator context or notes.",
    )


class ThreatExplanationResponse(BaseModel):
    """Validated structured cybersecurity threat explanation response."""
    prediction_id: str = Field(..., description="Correlated prediction identifier.")
    predicted_class: str = Field(..., description="Grounded class label matching ML detector.")
    detector_name: str = Field(..., description="Detector model that produced the original prediction.")
    confidence_score: Optional[float] = Field(
        default=None,
        ge=0.0, le=1.0,
        description="Model confidence score from the classifier."
    )
    severity: SeverityLevel = Field(
        ...,
        description="Assigned cybersecurity severity level based on evidence and risk."
    )
    severity_rationale: str = Field(
        ...,
        min_length=5,
        description="Justification explaining why the severity rating was assigned."
    )
    summary: str = Field(
        ...,
        min_length=10,
        description="Concise advisory explanation of observed traffic and prediction rationale."
    )
    supporting_evidence: List[str] = Field(
        ...,
        min_length=1,
        description="List of specific observed telemetry points supporting the assessment."
    )
    uncertainties: List[str] = Field(
        ...,
        min_length=1,
        description="Explicit uncertainties, model limitations, or false-positive risks."
    )
    recommended_actions: List[str] = Field(
        ...,
        min_length=1,
        description="Actionable, low-risk, reversible investigation and triage recommendations."
    )
    requires_human_review: bool = Field(
        ...,
        description="Flag indicating whether a human security analyst must review this alert."
    )
    explanation_provider: str = Field(
        ...,
        description="Provider that generated the explanation (e.g. 'mock_engine', 'groq', 'fallback')."
    )
    model_version: str = Field(..., description="Version of the detection model.")
    generated_at: str = Field(..., description="ISO 8601 UTC timestamp of explanation generation.")
    disclaimer: str = Field(
        default="Advisory automated explanation grounded in flow telemetry. Not proof of compromise. Low-risk human triage required.",
        description="Standard advisory legal and operational disclaimer."
    )
    is_fallback: bool = Field(
        default=False,
        description="True if generated by the deterministic fallback engine due to provider failure."
    )


class ThreatAnalysisRequest(BaseModel):
    """End-to-end request: runs detection model and generates correlated explanation."""
    features: Dict[str, Any] = Field(
        ...,
        description="IoT connection flow features."
    )
    detector_name: Optional[str] = Field(
        default="random_forest",
        description="Configurable detector: 'random_forest', 'logistic_regression', 'fedavg', 'fedopt', 'gsfs'."
    )
    operator_context: Optional[str] = Field(
        default=None,
        max_length=500,
        description="Optional operator notes."
    )


class ThreatAnalysisResponse(BaseModel):
    """End-to-end response containing both verified prediction and grounded explanation."""
    prediction: ThreatPredictionResponse = Field(..., description="Original ML prediction result.")
    explanation: ThreatExplanationResponse = Field(..., description="Correlated LLM advisory explanation.")


class ModelInfoItem(BaseModel):
    """Information item describing an available detector model."""
    detector_name: str
    display_name: str
    model_type: str
    is_available: bool
    threshold: float
    version: str
    description: str
    file_path: str


class ModelListResponse(BaseModel):
    """Response containing list of registered models."""
    models: List[ModelInfoItem]
    default_detector: str
    total_models: int


class HealthResponse(BaseModel):
    """Service health status response."""
    status: str = "healthy"
    version: str = "1.0.0"
    service: str = "iot-threat-explanation-api"
    timestamp: str


class ReadinessResponse(BaseModel):
    """Service readiness response checking model artifact presence and database connectivity."""
    status: str = "ready"
    models_available: int
    models_total: int
    default_detector_ready: bool
    database_connected: bool = True
    database_error: Optional[str] = None
    details: Dict[str, bool]


# =====================================================================
# Phase 7: Historical & Analytics Models
# =====================================================================

class HistoricalPredictionItem(BaseModel):
    """Authoritative persisted prediction record item."""
    id: str
    prediction_id: str
    detector_name: str
    model_version: str
    predicted_class: str
    prediction: int
    malicious_probability: Optional[float] = None
    benign_probability: Optional[float] = None
    threshold_used: float
    confidence_score: Optional[float] = None
    is_calibrated: bool = False
    validated_evidence: Dict[str, Any] = Field(default_factory=dict)
    stripped_leakage_fields: List[str] = Field(default_factory=list)
    inference_time_ms: float = 0.0
    created_at: str

    model_config = {"from_attributes": True}


class HistoricalExplanationItem(BaseModel):
    """Authoritative persisted explanation record item."""
    id: str
    prediction_id: str
    predicted_class: str
    detector_name: str
    confidence_score: Optional[float] = None
    severity: str
    severity_rationale: str
    summary: str
    supporting_evidence: List[str] = Field(default_factory=list)
    uncertainties: List[str] = Field(default_factory=list)
    recommended_actions: List[str] = Field(default_factory=list)
    requires_human_review: bool
    explanation_provider: str
    model_version: str
    is_fallback: bool
    disclaimer: str
    operator_context: Optional[str] = None
    generated_at: str

    model_config = {"from_attributes": True}


class HistoricalPredictionDetailResponse(BaseModel):
    """Prediction detail along with correlated explanations."""
    prediction: HistoricalPredictionItem
    explanations: List[HistoricalExplanationItem] = Field(default_factory=list)


class PaginatedPredictionsResponse(BaseModel):
    """Paginated response envelope for historical predictions."""
    items: List[HistoricalPredictionItem]
    page: int
    page_size: int
    total_items: int
    total_pages: int


class PaginatedExplanationsResponse(BaseModel):
    """Paginated response envelope for historical explanations."""
    items: List[HistoricalExplanationItem]
    page: int
    page_size: int
    total_items: int
    total_pages: int


class AnalyticsSummaryResponse(BaseModel):
    """High-level SOC dashboard summary KPIs."""
    total_predictions: int
    malicious_predictions: int
    benign_predictions: int
    malicious_percentage: float
    total_explanations: int
    requires_human_review: int
    human_review_percentage: float
    fallback_explanations: int
    fallback_percentage: float
    detector_counts: Dict[str, int]
    latest_event_timestamp: Optional[str] = None


class AlertTimeSeriesItem(BaseModel):
    """Individual time-bucket aggregated prediction counts."""
    bucket_start: str
    total_predictions: int
    malicious_count: int
    benign_count: int


class AlertTimeSeriesResponse(BaseModel):
    """Time-series alert trend response."""
    bucket_size: str
    items: List[AlertTimeSeriesItem]


class ThreatBreakdownItem(BaseModel):
    """Threat counts by detector and class."""
    detector_name: str
    predicted_class: str
    count: int


class ThreatBreakdownResponse(BaseModel):
    """Categorized threat counts across detectors and classes."""
    items: List[ThreatBreakdownItem]
    total_records: int


class DetectorUsageItem(BaseModel):
    """Inference metrics, volume, and latency per detector."""
    detector_name: str
    total_predictions: int
    malicious_count: int
    benign_count: int
    mean_inference_time_ms: float
    first_seen: Optional[str] = None
    last_seen: Optional[str] = None


class DetectorUsageResponse(BaseModel):
    """Detector utilization and performance response."""
    items: List[DetectorUsageItem]


class ExplanationQualityResponse(BaseModel):
    """Explanation metrics, fallback rate, human review rate, and provider breakdown."""
    total_explanations: int
    fallback_count: int
    fallback_rate_pct: float
    human_review_count: int
    human_review_rate_pct: float
    severity_breakdown: Dict[str, int]
    provider_usage: Dict[str, int]

