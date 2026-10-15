"""FastAPI dependency injection and bounded in-memory prediction cache."""

from collections import OrderedDict
import logging
from typing import Optional, Dict, Any
from src.api.config import settings
from src.detection.adapter import ThreatDetectionAdapter
from src.explanations.service import ThreatExplanationService
from src.explanations.schemas import ThreatPredictionResponse

logger = logging.getLogger("ApiDependencies")


class PredictionStore:
    """
    Thread-safe, bounded in-memory LRU cache storing recent trusted predictions
    so the explanation endpoint can verify prediction authenticity.
    """

    def __init__(self, max_capacity: int = 1000):
        self.max_capacity = max_capacity
        self._store: OrderedDict[str, ThreatPredictionResponse] = OrderedDict()

    def save(self, prediction: ThreatPredictionResponse) -> None:
        """Store a verified prediction."""
        if len(self._store) >= self.max_capacity:
            self._store.popitem(last=False)  # Evict oldest entry
        self._store[prediction.prediction_id] = prediction

    def get(self, prediction_id: str) -> Optional[ThreatPredictionResponse]:
        """Retrieve a stored prediction by ID."""
        if prediction_id in self._store:
            self._store.move_to_end(prediction_id)  # Mark recently accessed
            return self._store[prediction_id]
        return None

    def clear(self) -> None:
        """Clear cache."""
        self._store.clear()


# Global singletons
_adapter_instance: Optional[ThreatDetectionAdapter] = None
_service_instance: Optional[ThreatExplanationService] = None
_prediction_store_instance = PredictionStore(max_capacity=1000)


def get_detection_adapter() -> ThreatDetectionAdapter:
    """Retrieve singleton ThreatDetectionAdapter."""
    global _adapter_instance
    if _adapter_instance is None:
        _adapter_instance = ThreatDetectionAdapter(
            models_dir=settings.MODELS_DIR,
            default_detector=settings.DEFAULT_DETECTOR,
        )
    return _adapter_instance


def get_explanation_service() -> ThreatExplanationService:
    """Retrieve singleton ThreatExplanationService."""
    global _service_instance
    if _service_instance is None:
        _service_instance = ThreatExplanationService(
            provider_type=settings.LLM_PROVIDER,
        )
    return _service_instance


from fastapi import Depends
from sqlalchemy.ext.asyncio import AsyncSession
from src.db.session import get_db
from src.db.repositories.predictions import PredictionRepository
from src.db.repositories.explanations import ExplanationRepository
from src.db.repositories.analytics import AnalyticsRepository


def get_prediction_store() -> PredictionStore:
    """Retrieve singleton PredictionStore (in-memory LRU cache)."""
    return _prediction_store_instance


def get_prediction_repo(session: AsyncSession = Depends(get_db)) -> PredictionRepository:
    """Retrieve request-scoped PredictionRepository."""
    return PredictionRepository(session)


def get_explanation_repo(session: AsyncSession = Depends(get_db)) -> ExplanationRepository:
    """Retrieve request-scoped ExplanationRepository."""
    return ExplanationRepository(session)


def get_analytics_repo(session: AsyncSession = Depends(get_db)) -> AnalyticsRepository:
    """Retrieve request-scoped AnalyticsRepository."""
    return AnalyticsRepository(session)
