"""Explanations package exposing schemas, prompt builder, providers, and service."""

from src.explanations.schemas import (
    SeverityLevel,
    ThreatPredictionRequest,
    BatchThreatPredictionRequest,
    ThreatPredictionResponse,
    ThreatExplanationRequest,
    ThreatExplanationResponse,
    ThreatAnalysisRequest,
    ThreatAnalysisResponse,
    ModelInfoItem,
    ModelListResponse,
    HealthResponse,
    ReadinessResponse,
)
from src.explanations.prompt_builder import (
    SYSTEM_EXPLANATION_PROMPT,
    build_explanation_prompt,
    sanitize_untrusted_text,
)
from src.explanations.providers.base import BaseExplanationProvider
from src.explanations.providers.mock_provider import MockExplanationProvider
from src.explanations.providers.groq_provider import GroqExplanationProvider
from src.explanations.providers.gemini_provider import GeminiExplanationProvider
from src.explanations.service import ThreatExplanationService

__all__ = [
    "SeverityLevel",
    "ThreatPredictionRequest",
    "BatchThreatPredictionRequest",
    "ThreatPredictionResponse",
    "ThreatExplanationRequest",
    "ThreatExplanationResponse",
    "ThreatAnalysisRequest",
    "ThreatAnalysisResponse",
    "ModelInfoItem",
    "ModelListResponse",
    "HealthResponse",
    "ReadinessResponse",
    "SYSTEM_EXPLANATION_PROMPT",
    "build_explanation_prompt",
    "sanitize_untrusted_text",
    "BaseExplanationProvider",
    "MockExplanationProvider",
    "GroqExplanationProvider",
    "GeminiExplanationProvider",
    "ThreatExplanationService",
]
