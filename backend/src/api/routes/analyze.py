"""End-to-end analysis route combining intrusion detection with correlated LLM explanation."""

import logging
from fastapi import APIRouter, Depends, HTTPException, status
from src.api.dependencies import (
    get_detection_adapter,
    get_explanation_service,
    get_prediction_store,
    get_prediction_repo,
    get_explanation_repo,
    PredictionStore,
)
from src.db.repositories.predictions import PredictionRepository
from src.db.repositories.explanations import ExplanationRepository
from src.detection.adapter import ThreatDetectionAdapter
from src.explanations.service import ThreatExplanationService
from src.explanations.schemas import (
    ThreatAnalysisRequest,
    ThreatAnalysisResponse,
    ThreatPredictionResponse,
)

logger = logging.getLogger("AnalyzeRoutes")
router = APIRouter(prefix="/api/v1/analyze", tags=["End-to-End Threat Analysis"])


@router.post("", response_model=ThreatAnalysisResponse, status_code=status.HTTP_200_OK)
async def analyze_threat(
    request: ThreatAnalysisRequest,
    adapter: ThreatDetectionAdapter = Depends(get_detection_adapter),
    service: ThreatExplanationService = Depends(get_explanation_service),
    store: PredictionStore = Depends(get_prediction_store),
    pred_repo: PredictionRepository = Depends(get_prediction_repo),
    exp_repo: ExplanationRepository = Depends(get_explanation_repo),
):
    """
    Unified end-to-end endpoint:
    1. Validates telemetry features and strips forbidden leakage columns.
    2. Executes selected intrusion detection model to obtain trustworthy classification.
    3. Persists authoritative prediction to the database and in-memory cache.
    4. Feeds validated prediction and grounded evidence into LLM explanation service.
    5. Persists the generated or fallback explanation linked to the prediction.
    6. Returns synchronized pair: ThreatPredictionResponse and ThreatExplanationResponse.
    """
    # 1. Model inference & Prediction persistence
    try:
        raw_result = adapter.predict_single(
            record=request.features,
            detector_name=request.detector_name,
        )
        prediction_resp = ThreatPredictionResponse(**raw_result)
        store.save(prediction_resp)
        await pred_repo.create(prediction_resp.model_dump())

    except FileNotFoundError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Model artifact not found: {str(e)}",
        )
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Validation failed: {str(e)}",
        )
    except Exception as e:
        logger.error(f"Inference phase failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Inference failure occurred during threat analysis.",
        )

    # 2. Explanation generation with guaranteed fallback & Explanation persistence
    try:
        explanation_resp = await service.explain_prediction(
            prediction=prediction_resp,
            operator_context=request.operator_context,
        )
    except Exception as e:
        logger.error(f"Explanation phase failed, utilizing fallback: {e}", exc_info=True)
        explanation_resp = await service.fallback_provider.generate_explanation(
            prediction=prediction_resp,
            operator_context=request.operator_context,
        )
        explanation_resp.is_fallback = True

    try:
        await exp_repo.create(explanation_resp.model_dump())
    except Exception as e:
        logger.error(f"Failed to persist explanation for {prediction_resp.prediction_id}: {e}", exc_info=True)
        # Note: Prediction is already safely persisted.

    return ThreatAnalysisResponse(
        prediction=prediction_resp,
        explanation=explanation_resp,
    )

