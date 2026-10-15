"""Threat Explanation Service orchestrating LLM providers with robust fallback guarantees."""

import logging
import os
import time
from typing import Optional, Dict, Any

from src.explanations.providers.base import BaseExplanationProvider
from src.explanations.providers.mock_provider import MockExplanationProvider
from src.explanations.providers.groq_provider import GroqExplanationProvider
from src.explanations.providers.gemini_provider import GeminiExplanationProvider
from src.explanations.schemas import (
    ThreatExplanationResponse,
    ThreatPredictionResponse,
    SeverityLevel,
)

logger = logging.getLogger("ThreatExplanationService")


class ThreatExplanationService:
    """
    Production-grade explanation service managing provider selection,
    timeout handling, schema validation, and deterministic fallbacks.
    """

    def __init__(
        self,
        provider: Optional[BaseExplanationProvider] = None,
        provider_type: Optional[str] = None,
    ):
        # Fallback provider is always available
        self.fallback_provider = MockExplanationProvider(provider_name="fallback_deterministic_engine")

        if provider is not None:
            self.primary_provider = provider
        else:
            p_type = (provider_type or os.getenv("LLM_PROVIDER", "mock")).lower().strip()
            self.primary_provider = self._create_provider(p_type)

        logger.info(
            f"Initialized ThreatExplanationService with primary provider: '{self.primary_provider.provider_name}' "
            f"(available: {self.primary_provider.is_available()})."
        )

    def _create_provider(self, p_type: str) -> BaseExplanationProvider:
        """Instantiate configured LLM provider adapter."""
        if p_type == "groq":
            groq_p = GroqExplanationProvider()
            if groq_p.is_available():
                return groq_p
            logger.warning("Groq API key not found. Defaulting to mock explanation provider.")
            return MockExplanationProvider(provider_name="mock_threat_explainer_v1")

        elif p_type == "gemini":
            gem_p = GeminiExplanationProvider()
            if gem_p.is_available():
                return gem_p
            logger.warning("Gemini API key not found. Defaulting to mock explanation provider.")
            return MockExplanationProvider(provider_name="mock_threat_explainer_v1")

        elif p_type in ("mock", "test"):
            return MockExplanationProvider(provider_name="mock_threat_explainer_v1")

        else:
            logger.warning(f"Unknown LLM provider '{p_type}'. Defaulting to mock provider.")
            return MockExplanationProvider(provider_name="mock_threat_explainer_v1")

    async def explain_prediction(
        self,
        prediction: ThreatPredictionResponse,
        operator_context: Optional[str] = None,
    ) -> ThreatExplanationResponse:
        """
        Generate a validated threat explanation for a classified flow.

        If the primary provider fails, times out, or produces invalid output,
        a deterministic fallback explanation is automatically returned, preserving
        the original prediction and evidence.
        """
        pid = prediction.prediction_id
        logger.info(f"[{pid}] Generating threat explanation using provider '{self.primary_provider.provider_name}'...")
        start_t = time.time()

        # Try primary provider if available
        if self.primary_provider.is_available():
            try:
                explanation = await self.primary_provider.generate_explanation(
                    prediction=prediction,
                    operator_context=operator_context,
                )
                elapsed = time.time() - start_t
                logger.info(f"[{pid}] Primary provider explanation succeeded in {elapsed:.3f}s.")
                return explanation

            except Exception as e:
                logger.warning(
                    f"[{pid}] Primary provider '{self.primary_provider.provider_name}' failed: {e}. "
                    f"Activating deterministic fallback explanation."
                )

        # Fallback execution
        fallback_start = time.time()
        explanation = await self.fallback_provider.generate_explanation(
            prediction=prediction,
            operator_context=operator_context,
        )
        # Update metadata flags
        explanation.is_fallback = True
        explanation.explanation_provider = self.fallback_provider.provider_name
        explanation.uncertainties.append(
            f"Primary explanation provider was unavailable or failed; deterministic fallback explanation generated."
        )

        elapsed = time.time() - fallback_start
        logger.info(f"[{pid}] Fallback explanation generated in {elapsed:.3f}s.")
        return explanation
