"""Threat prediction endpoints executing trained intrusion detection models."""

import logging
from typing import List
from fastapi import APIRouter, Depends, HTTPException, status
from src.api.dependencies import (
    get_detection_adapter,
    get_prediction_store,
    get_prediction_repo,
    PredictionStore,
)
from src.db.repositories.predictions import PredictionRepository
from src.detection.adapter import ThreatDetectionAdapter
from src.explanations.schemas import (
    ThreatPredictionRequest,
    BatchThreatPredictionRequest,
    ThreatPredictionResponse,
)

logger = logging.getLogger("PredictionRoutes")
router = APIRouter(prefix="/api/v1/predictions", tags=["Threat Detection"])


@router.post("", response_model=ThreatPredictionResponse, status_code=status.HTTP_200_OK)
async def predict_threat(
    request: ThreatPredictionRequest,
    adapter: ThreatDetectionAdapter = Depends(get_detection_adapter),
    store: PredictionStore = Depends(get_prediction_store),
    repo: PredictionRepository = Depends(get_prediction_repo),
):
    """
    Execute threat detection on a single IoT connection record.
    Persists the prediction in the database and returns structured prediction.
    """
    try:
        raw_result = adapter.predict_single(
            record=request.features,
            detector_name=request.detector_name,
        )
        prediction_resp = ThreatPredictionResponse(**raw_result)
        # 1. Update in-memory LRU store for cache lookup
        store.save(prediction_resp)
        # 2. Persist in database
        await repo.create(prediction_resp.model_dump())
        return prediction_resp

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
        logger.error(f"Prediction failed unexpectedly: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal prediction failure occurred.",
        )


@router.post("/batch", response_model=List[ThreatPredictionResponse], status_code=status.HTTP_200_OK)
async def predict_threat_batch(
    request: BatchThreatPredictionRequest,
    adapter: ThreatDetectionAdapter = Depends(get_detection_adapter),
    store: PredictionStore = Depends(get_prediction_store),
    repo: PredictionRepository = Depends(get_prediction_repo),
):
    """
    Execute batch threat detection across multiple records (bounded to 100 flows max).
    Persists all predictions in the database within a single transaction.
    """
    try:
        raw_results = adapter.predict_batch(
            records=request.records,
            detector_name=request.detector_name,
            max_batch_size=100,
        )
        response_items = []
        items_to_persist = []
        for res in raw_results:
            pred_resp = ThreatPredictionResponse(**res)
            store.save(pred_resp)
            response_items.append(pred_resp)
            items_to_persist.append(pred_resp.model_dump())

        # Persist all predictions in database
        await repo.create_batch(items_to_persist)
        return response_items

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
        logger.error(f"Batch prediction failed unexpectedly: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal batch prediction failure occurred.",
        )

