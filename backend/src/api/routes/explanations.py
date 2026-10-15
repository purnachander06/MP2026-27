"""Threat explanation endpoint generating grounded LLM advisory explanations."""

import logging
import time
from fastapi import APIRouter, Depends, HTTPException, status
from src.api.dependencies import (
    get_explanation_service,
    get_prediction_store,
    get_prediction_repo,
    get_explanation_repo,
    PredictionStore,
)
from src.db.repositories.predictions import PredictionRepository
from src.db.repositories.explanations import ExplanationRepository
from src.explanations.service import ThreatExplanationService
from src.explanations.schemas import (
    ThreatExplanationRequest,
    ThreatExplanationResponse,
    ThreatPredictionResponse,
)

logger = logging.getLogger("ExplanationRoutes")
router = APIRouter(prefix="/api/v1/explanations", tags=["Threat Explanations"])


@router.post("", response_model=ThreatExplanationResponse, status_code=status.HTTP_200_OK)
async def explain_prediction(
    request: ThreatExplanationRequest,
    service: ThreatExplanationService = Depends(get_explanation_service),
    store: PredictionStore = Depends(get_prediction_store),
    pred_repo: PredictionRepository = Depends(get_prediction_repo),
    exp_repo: ExplanationRepository = Depends(get_explanation_repo),
):
    """
    Generate an LLM-assisted, evidence-grounded threat explanation for an existing prediction.

    Security & Authenticity Guarantees:
    - Verifies prediction existence against in-memory cache and authoritative database.
    - If unknown/fabricated, returns HTTP 404 Not Found.
    - If found, enforces authoritative ML predicted_class and confidence from database,
      preventing clients from spoofing or altering classification outcome.
    - Persists the generated explanation in PostgreSQL.
    """
    # 1. Check in-memory store
    stored_prediction = store.get(request.prediction_id)

    # 2. If not in memory, query authoritative database
    if stored_prediction is None:
        db_pred = await pred_repo.get_by_prediction_id(request.prediction_id)
        if db_pred is not None:
            stored_prediction = ThreatPredictionResponse(
                prediction_id=db_pred.prediction_id,
                detector_name=db_pred.detector_name,
                model_version=db_pred.model_version,
                predicted_class=db_pred.predicted_class,
                prediction=db_pred.prediction,
                malicious_probability=db_pred.malicious_probability,
                benign_probability=db_pred.benign_probability,
                threshold_used=db_pred.threshold_used,
                confidence_score=db_pred.confidence_score,
                is_calibrated=db_pred.is_calibrated,
                validated_evidence=db_pred.validated_evidence,
                stripped_leakage_fields=db_pred.stripped_leakage_fields,
                inference_time_ms=db_pred.inference_time_ms,
                timestamp=db_pred.created_at.isoformat() if db_pred.created_at else "",
            )
            store.save(stored_prediction)

    # 3. If still not found, reject fabricated/unknown prediction ID with 404
    if stored_prediction is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Prediction with ID '{request.prediction_id}' not found.",
        )

    # 4. Enforce trusted model output
    trusted_pred = stored_prediction
    if request.predicted_class.lower() != trusted_pred.predicted_class.lower():
        logger.warning(
            f"Client attempted to override predicted class for {request.prediction_id}: "
            f"client='{request.predicted_class}' vs trusted='{trusted_pred.predicted_class}'. Enforcing trusted."
        )
    pred_to_explain = trusted_pred

    # 5. Generate explanation
    try:
        explanation = await service.explain_prediction(
            prediction=pred_to_explain,
            operator_context=request.operator_context,
        )
        # 6. Persist explanation in database
        await exp_repo.create(explanation.model_dump())
        return explanation

    except Exception as e:
        logger.error(f"Failed to generate explanation for {request.prediction_id}: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Explanation generation failed: {str(e)}",
        )

