"""Abstract base class for LLM threat explanation providers."""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from src.explanations.schemas import ThreatExplanationResponse, ThreatPredictionResponse


class BaseExplanationProvider(ABC):
    """
    Abstract interface for cybersecurity threat explanation providers.
    Supports asynchronous execution with bounded timeouts and error handling.
    """

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Return the unique identifier of the provider."""
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """Check whether the provider is configured and ready for execution."""
        pass

    @abstractmethod
    async def generate_explanation(
        self,
        prediction: ThreatPredictionResponse,
        operator_context: Optional[str] = None,
    ) -> ThreatExplanationResponse:
        """
        Generate a validated threat explanation for the given prediction.

        Args:
            prediction: Validated prediction result from the detection adapter.
            operator_context: Optional untrusted analyst notes.

        Returns:
            Structured ThreatExplanationResponse instance.

        Raises:
            Exception on unrecoverable provider failure, timeout, or invalid output.
        """
        pass
