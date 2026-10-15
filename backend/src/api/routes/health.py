"""Health, readiness, and model registry endpoints."""

import time
from fastapi import APIRouter, Depends
from src.api.config import settings
from src.api.dependencies import get_detection_adapter
from src.detection.adapter import ThreatDetectionAdapter
from src.explanations.schemas import HealthResponse, ReadinessResponse, ModelListResponse

router = APIRouter(tags=["System Health & Metadata"])


@router.get("/health", response_model=HealthResponse)
async def get_health():
    """
    Service health check endpoint.
    Returns operational status and version without exposing internal secrets or configurations.
    """
    return HealthResponse(
        status="healthy",
        version="1.0.0",
        service="iot-threat-explanation-api",
        timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    )


from src.db.session import ping_database


@router.get("/ready", response_model=ReadinessResponse)
async def get_readiness(adapter: ThreatDetectionAdapter = Depends(get_detection_adapter)):
    """
    Readiness check verifying availability of local model checkpoints on disk
    and active database connectivity. Does not initiate external LLM calls.
    """
    models = adapter.list_available_models()
    details = {m["detector_name"]: m["is_available"] for m in models}
    available_count = sum(1 for v in details.values() if v)
    total_count = len(details)

    default_ready = details.get(settings.DEFAULT_DETECTOR, False)
    db_connected, db_error = await ping_database()

    status = "ready" if (default_ready and db_connected) else "not_ready"

    return ReadinessResponse(
        status=status,
        models_available=available_count,
        models_total=total_count,
        default_detector_ready=default_ready,
        database_connected=db_connected,
        database_error=db_error,
        details=details,
    )


@router.get("/models", response_model=ModelListResponse)
async def list_models(adapter: ThreatDetectionAdapter = Depends(get_detection_adapter)):
    """
    List registered threat detection models, types, readiness status, and thresholds.
    """
    models = adapter.list_available_models()
    return ModelListResponse(
        models=models,
        default_detector=settings.DEFAULT_DETECTOR,
        total_models=len(models),
    )
