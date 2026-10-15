"""Historical query endpoints providing paginated threat predictions and explanations with resilient fallbacks."""

import logging
from datetime import datetime
from typing import Optional
from fastapi import APIRouter, Depends, HTTPException, Query, status

from src.api.dependencies import get_prediction_repo, get_explanation_repo
from src.db.repositories.predictions import PredictionRepository, ALLOWED_PREDICTION_SORT_FIELDS
from src.db.repositories.explanations import ExplanationRepository, ALLOWED_EXPLANATION_SORT_FIELDS
from src.explanations.schemas import (
    HistoricalPredictionItem,
    HistoricalExplanationItem,
    HistoricalPredictionDetailResponse,
    PaginatedPredictionsResponse,
    PaginatedExplanationsResponse,
)

logger = logging.getLogger("HistoryRoutes")
router = APIRouter(prefix="/api/v1/history", tags=["Historical Threat Telemetry"])


def _map_pred_record(r) -> HistoricalPredictionItem:
    return HistoricalPredictionItem(
        id=str(r.id),
        prediction_id=r.prediction_id,
        detector_name=r.detector_name,
        model_version=r.model_version,
        predicted_class=r.predicted_class,
        prediction=r.prediction,
        malicious_probability=r.malicious_probability,
        benign_probability=r.benign_probability,
        threshold_used=r.threshold_used,
        confidence_score=r.confidence_score,
        is_calibrated=r.is_calibrated,
        validated_evidence=r.validated_evidence or {},
        stripped_leakage_fields=r.stripped_leakage_fields or [],
        inference_time_ms=r.inference_time_ms,
        created_at=r.created_at.isoformat() if r.created_at else "",
    )


def _map_exp_record(r) -> HistoricalExplanationItem:
    return HistoricalExplanationItem(
        id=str(r.id),
        prediction_id=r.prediction_id,
        predicted_class=r.predicted_class,
        detector_name=r.detector_name,
        confidence_score=r.confidence_score,
        severity=r.severity,
        severity_rationale=r.severity_rationale,
        summary=r.summary,
        supporting_evidence=r.supporting_evidence or [],
        uncertainties=r.uncertainties or [],
        recommended_actions=r.recommended_actions or [],
        requires_human_review=r.requires_human_review,
        explanation_provider=r.explanation_provider,
        model_version=r.model_version,
        is_fallback=r.is_fallback,
        disclaimer=r.disclaimer,
        operator_context=r.operator_context,
        generated_at=r.generated_at.isoformat() if r.generated_at else "",
    )


@router.get("/predictions", response_model=PaginatedPredictionsResponse, status_code=status.HTTP_200_OK)
async def list_predictions(
    page: int = Query(1, ge=1, description="Page number starting at 1."),
    page_size: int = Query(20, ge=1, le=100, description="Items per page (max 100)."),
    detector_name: Optional[str] = Query(None, description="Filter by detector name."),
    predicted_class: Optional[str] = Query(None, description="Filter by class ('benign' or 'malicious')."),
    start_time: Optional[datetime] = Query(None, description="Filter predictions on or after start_time (UTC)."),
    end_time: Optional[datetime] = Query(None, description="Filter predictions on or before end_time (UTC)."),
    min_malicious_probability: Optional[float] = Query(None, ge=0.0, le=1.0, description="Minimum malicious score."),
    max_malicious_probability: Optional[float] = Query(None, ge=0.0, le=1.0, description="Maximum malicious score."),
    sort_by: str = Query("created_at", description="Sort field allowlist: created_at, malicious_probability, inference_time_ms, prediction_id."),
    sort_order: str = Query("desc", description="Sort direction: 'asc' or 'desc'."),
    repo: PredictionRepository = Depends(get_prediction_repo),
):
    """
    List historical threat predictions with server-side filtering, sorting, and pagination.
    """
    if start_time and end_time and start_time > end_time:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="start_time must be earlier than or equal to end_time.",
        )

    if sort_by not in ALLOWED_PREDICTION_SORT_FIELDS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid sort_by field '{sort_by}'. Allowed: {list(ALLOWED_PREDICTION_SORT_FIELDS.keys())}",
        )

    if sort_order.lower() not in ("asc", "desc"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="sort_order must be 'asc' or 'desc'.",
        )

    try:
        items, total_items, total_pages = await repo.list_paginated(
            page=page,
            page_size=page_size,
            detector_name=detector_name,
            predicted_class=predicted_class,
            start_time=start_time,
            end_time=end_time,
            min_malicious_probability=min_malicious_probability,
            max_malicious_probability=max_malicious_probability,
            sort_by=sort_by,
            sort_order=sort_order,
        )

        return PaginatedPredictionsResponse(
            items=[_map_pred_record(r) for r in items],
            page=page,
            page_size=page_size,
            total_items=total_items,
            total_pages=total_pages,
        )
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        logger.warning(f"Database query failed for list_predictions ({e}). Returning empty list.")
        return PaginatedPredictionsResponse(
            items=[],
            page=page,
            page_size=page_size,
            total_items=0,
            total_pages=0,
        )


@router.get("/predictions/{prediction_id}", response_model=HistoricalPredictionDetailResponse, status_code=status.HTTP_200_OK)
async def get_prediction_detail(
    prediction_id: str,
    repo: PredictionRepository = Depends(get_prediction_repo),
):
    """
    Retrieve authoritative persisted prediction details including all associated explanations.
    """
    try:
        record = await repo.get_with_explanations(prediction_id)
        if record is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Prediction with ID '{prediction_id}' not found.",
            )

        return HistoricalPredictionDetailResponse(
            prediction=_map_pred_record(record),
            explanations=[_map_exp_record(exp) for exp in record.explanations],
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.warning(f"Database query failed for get_prediction_detail ({e}).")
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Prediction '{prediction_id}' not found.")


@router.get("/explanations", response_model=PaginatedExplanationsResponse, status_code=status.HTTP_200_OK)
async def list_explanations(
    page: int = Query(1, ge=1, description="Page number."),
    page_size: int = Query(20, ge=1, le=100, description="Items per page (max 100)."),
    prediction_id: Optional[str] = Query(None, description="Filter by parent prediction ID."),
    severity: Optional[str] = Query(None, description="Filter by severity level ('critical', 'high', etc.)."),
    is_fallback: Optional[bool] = Query(None, description="Filter by fallback generation flag."),
    requires_human_review: Optional[bool] = Query(None, description="Filter by human review requirement."),
    detector_name: Optional[str] = Query(None, description="Filter by detector model used."),
    start_time: Optional[datetime] = Query(None, description="Filter explanations generated on/after."),
    end_time: Optional[datetime] = Query(None, description="Filter explanations generated on/before."),
    sort_by: str = Query("generated_at", description="Sort field allowlist: generated_at, severity, confidence_score."),
    sort_order: str = Query("desc", description="Sort direction: 'asc' or 'desc'."),
    repo: ExplanationRepository = Depends(get_explanation_repo),
):
    """
    List historical LLM threat explanations with filtering, pagination, and sorting.
    """
    if start_time and end_time and start_time > end_time:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="start_time must be earlier than or equal to end_time.",
        )

    if sort_by not in ALLOWED_EXPLANATION_SORT_FIELDS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid sort_by field '{sort_by}'. Allowed: {list(ALLOWED_EXPLANATION_SORT_FIELDS.keys())}",
        )

    if sort_order.lower() not in ("asc", "desc"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="sort_order must be 'asc' or 'desc'.",
        )

    try:
        items, total_items, total_pages = await repo.list_paginated(
            page=page,
            page_size=page_size,
            severity=severity,
            is_fallback=is_fallback,
            requires_human_review=requires_human_review,
            detector_name=detector_name,
            start_time=start_time,
            end_time=end_time,
            sort_by=sort_by,
            sort_order=sort_order,
        )

        return PaginatedExplanationsResponse(
            items=[_map_exp_record(r) for r in items],
            page=page,
            page_size=page_size,
            total_items=total_items,
            total_pages=total_pages,
        )
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        logger.warning(f"Database query failed for list_explanations ({e}). Returning empty list.")
        return PaginatedExplanationsResponse(
            items=[],
            page=page,
            page_size=page_size,
            total_items=0,
            total_pages=0,
        )