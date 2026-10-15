"""SOC dashboard analytics endpoints aggregating live threat and explanation metrics with resilient fallbacks."""

import logging
from datetime import datetime
from typing import Optional
from fastapi import APIRouter, Depends, HTTPException, Query, status

from src.api.dependencies import get_analytics_repo
from src.db.repositories.analytics import AnalyticsRepository
from src.explanations.schemas import (
    AnalyticsSummaryResponse,
    AlertTimeSeriesResponse,
    AlertTimeSeriesItem,
    ThreatBreakdownResponse,
    ThreatBreakdownItem,
    DetectorUsageResponse,
    DetectorUsageItem,
    ExplanationQualityResponse,
)

logger = logging.getLogger("AnalyticsRoutes")
router = APIRouter(prefix="/api/v1/analytics", tags=["SOC Dashboard Analytics"])

DEFAULT_DETECTORS = ["random_forest", "logistic_regression", "fedavg", "fedopt", "gsfs"]


@router.get("/summary", response_model=AnalyticsSummaryResponse, status_code=status.HTTP_200_OK)
async def get_summary_analytics(
    start_time: Optional[datetime] = Query(None, description="Start of observation window (UTC)."),
    end_time: Optional[datetime] = Query(None, description="End of observation window (UTC)."),
    repo: AnalyticsRepository = Depends(get_analytics_repo),
):
    """
    Compute live summary metrics: total alerts, malicious ratio, fallback rate, detector volume.
    Safely returns zero values when database is unseeded or temporarily offline.
    """
    if start_time and end_time and start_time > end_time:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="start_time must be earlier than or equal to end_time.",
        )

    try:
        data = await repo.get_summary(start_time=start_time, end_time=end_time)
        return AnalyticsSummaryResponse(**data)
    except Exception as e:
        logger.warning(f"Database query failed for analytics summary ({e}). Returning zeroed metrics.")
        return AnalyticsSummaryResponse(
            total_predictions=0,
            malicious_predictions=0,
            benign_predictions=0,
            malicious_percentage=0.0,
            total_explanations=0,
            requires_human_review=0,
            human_review_percentage=0.0,
            fallback_explanations=0,
            fallback_percentage=0.0,
            detector_counts={d: 0 for d in DEFAULT_DETECTORS},
            latest_event_timestamp=None,
        )


@router.get("/alerts-over-time", response_model=AlertTimeSeriesResponse, status_code=status.HTTP_200_OK)
async def get_alerts_time_series(
    bucket: str = Query("hour", description="Time aggregation bucket: 'hour', 'day', 'week'."),
    start_time: Optional[datetime] = Query(None, description="Start timestamp (UTC)."),
    end_time: Optional[datetime] = Query(None, description="End timestamp (UTC)."),
    repo: AnalyticsRepository = Depends(get_analytics_repo),
):
    """
    Generate chronological time-bucketed series of predictions and threat counts.
    """
    if start_time and end_time and start_time > end_time:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="start_time must be earlier than or equal to end_time.",
        )

    bucket_norm = bucket.lower().strip()
    if bucket_norm not in ("hour", "day", "week"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid bucket size '{bucket}'. Allowed values: 'hour', 'day', 'week'.",
        )

    try:
        rows = await repo.get_alerts_over_time(
            bucket=bucket_norm,
            start_time=start_time,
            end_time=end_time,
        )
        items = [
            AlertTimeSeriesItem(
                bucket_start=r["bucket_start"],
                total_predictions=r["total_predictions"],
                malicious_count=r["malicious_count"],
                benign_count=r["benign_count"],
            )
            for r in rows
        ]
        return AlertTimeSeriesResponse(bucket_size=bucket_norm, items=items)
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        logger.warning(f"Database query failed for alert time series ({e}). Returning empty bucket list.")
        return AlertTimeSeriesResponse(bucket_size=bucket_norm, items=[])


@router.get("/threat-breakdown", response_model=ThreatBreakdownResponse, status_code=status.HTTP_200_OK)
async def get_threat_breakdown(
    start_time: Optional[datetime] = Query(None, description="Start timestamp (UTC)."),
    end_time: Optional[datetime] = Query(None, description="End timestamp (UTC)."),
    repo: AnalyticsRepository = Depends(get_analytics_repo),
):
    """
    Break down threat detection volume by detector model and classification class.
    """
    if start_time and end_time and start_time > end_time:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="start_time must be earlier than or equal to end_time.",
        )

    try:
        rows = await repo.get_threat_breakdown(start_time=start_time, end_time=end_time)
        items = [
            ThreatBreakdownItem(
                detector_name=r["detector_name"],
                predicted_class=r["predicted_class"],
                count=r["count"],
            )
            for r in rows
        ]
        total_records = sum(item.count for item in items)
        return ThreatBreakdownResponse(items=items, total_records=total_records)
    except Exception as e:
        logger.warning(f"Database query failed for threat breakdown ({e}). Returning empty breakdown.")
        return ThreatBreakdownResponse(items=[], total_records=0)


@router.get("/detectors", response_model=DetectorUsageResponse, status_code=status.HTTP_200_OK)
async def get_detector_usage(
    start_time: Optional[datetime] = Query(None, description="Start timestamp (UTC)."),
    end_time: Optional[datetime] = Query(None, description="End timestamp (UTC)."),
    repo: AnalyticsRepository = Depends(get_analytics_repo),
):
    """
    Compute per-detector volume, class distribution, and average inference latency.
    """
    if start_time and end_time and start_time > end_time:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="start_time must be earlier than or equal to end_time.",
        )

    try:
        rows = await repo.get_detector_usage(start_time=start_time, end_time=end_time)
        items = [
            DetectorUsageItem(
                detector_name=r["detector_name"],
                total_predictions=r["total_predictions"],
                malicious_count=r["malicious_count"],
                benign_count=r["benign_count"],
                mean_inference_time_ms=r["mean_inference_time_ms"],
                first_seen=r["first_seen"],
                last_seen=r["last_seen"],
            )
            for r in rows
        ]
        return DetectorUsageResponse(items=items)
    except Exception as e:
        logger.warning(f"Database query failed for detector usage ({e}). Returning default active detectors.")
        return DetectorUsageResponse(
            items=[
                DetectorUsageItem(
                    detector_name=d,
                    total_predictions=0,
                    malicious_count=0,
                    benign_count=0,
                    mean_inference_time_ms=0.5,
                    first_seen="",
                    last_seen="",
                )
                for d in DEFAULT_DETECTORS
            ]
        )


@router.get("/explanations", response_model=ExplanationQualityResponse, status_code=status.HTTP_200_OK)
async def get_explanation_analytics(
    start_time: Optional[datetime] = Query(None, description="Start timestamp (UTC)."),
    end_time: Optional[datetime] = Query(None, description="End timestamp (UTC)."),
    repo: AnalyticsRepository = Depends(get_analytics_repo),
):
    """
    Compute explanation quality metrics: fallback rate, human review rate, severity breakdown.
    """
    if start_time and end_time and start_time > end_time:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="start_time must be earlier than or equal to end_time.",
        )

    try:
        data = await repo.get_explanation_quality(start_time=start_time, end_time=end_time)
        return ExplanationQualityResponse(**data)
    except Exception as e:
        logger.warning(f"Database query failed for explanation quality ({e}). Returning default quality stats.")
        return ExplanationQualityResponse(
            total_explanations=0,
            fallback_count=0,
            fallback_rate_pct=0.0,
            human_review_count=0,
            human_review_rate_pct=0.0,
            severity_breakdown={"informational": 0, "low": 0, "medium": 0, "high": 0, "critical": 0},
            provider_usage={"groq:openai/gpt-oss-20b": 0},
        )