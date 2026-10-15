"""Repository performing asynchronous CRUD and paginated filtering on PredictionRecord."""

from datetime import datetime
import math
from typing import Dict, List, Optional, Tuple, Any
from sqlalchemy import select, func, and_, desc, asc
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from src.db.models import PredictionRecord

ALLOWED_PREDICTION_SORT_FIELDS = {
    "created_at": PredictionRecord.created_at,
    "malicious_probability": PredictionRecord.malicious_probability,
    "inference_time_ms": PredictionRecord.inference_time_ms,
    "prediction_id": PredictionRecord.prediction_id,
}


class PredictionRepository:
    """Async repository for prediction records."""

    def __init__(self, session: AsyncSession):
        self.session = session

    async def create(self, data: Dict[str, Any]) -> PredictionRecord:
        """Create and persist a single prediction record."""
        record = PredictionRecord(
            prediction_id=data["prediction_id"],
            detector_name=data["detector_name"],
            model_version=data.get("model_version", "1.0.0"),
            predicted_class=data["predicted_class"],
            prediction=data["prediction"],
            malicious_probability=data["malicious_probability"],
            benign_probability=data["benign_probability"],
            threshold_used=data["threshold_used"],
            confidence_score=data["confidence_score"],
            is_calibrated=data.get("is_calibrated", False),
            validated_evidence=data.get("validated_evidence", {}),
            stripped_leakage_fields=data.get("stripped_leakage_fields", []),
            inference_time_ms=data.get("inference_time_ms", 0.0),
        )
        self.session.add(record)
        await self.session.flush()
        return record

    async def create_batch(self, items: List[Dict[str, Any]]) -> List[PredictionRecord]:
        """Create and persist a batch of prediction records in a single transaction."""
        records = [
            PredictionRecord(
                prediction_id=data["prediction_id"],
                detector_name=data["detector_name"],
                model_version=data.get("model_version", "1.0.0"),
                predicted_class=data["predicted_class"],
                prediction=data["prediction"],
                malicious_probability=data["malicious_probability"],
                benign_probability=data["benign_probability"],
                threshold_used=data["threshold_used"],
                confidence_score=data["confidence_score"],
                is_calibrated=data.get("is_calibrated", False),
                validated_evidence=data.get("validated_evidence", {}),
                stripped_leakage_fields=data.get("stripped_leakage_fields", []),
                inference_time_ms=data.get("inference_time_ms", 0.0),
            )
            for data in items
        ]
        self.session.add_all(records)
        await self.session.flush()
        return records

    async def get_by_prediction_id(self, prediction_id: str) -> Optional[PredictionRecord]:
        """Retrieve a prediction by its prediction_id."""
        stmt = select(PredictionRecord).where(PredictionRecord.prediction_id == prediction_id)
        result = await self.session.execute(stmt)
        return result.scalar_one_or_none()

    async def get_with_explanations(self, prediction_id: str) -> Optional[PredictionRecord]:
        """Retrieve a prediction along with all associated explanations."""
        stmt = (
            select(PredictionRecord)
            .where(PredictionRecord.prediction_id == prediction_id)
            .options(selectinload(PredictionRecord.explanations))
        )
        result = await self.session.execute(stmt)
        return result.scalar_one_or_none()

    async def list_paginated(
        self,
        page: int = 1,
        page_size: int = 20,
        detector_name: Optional[str] = None,
        predicted_class: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        min_malicious_probability: Optional[float] = None,
        max_malicious_probability: Optional[float] = None,
        sort_by: str = "created_at",
        sort_order: str = "desc",
    ) -> Tuple[List[PredictionRecord], int, int]:
        """
        Query predictions with server-side filtering, count, and pagination.

        Returns:
            Tuple of (items, total_items, total_pages).
        """
        # Validate sort field
        if sort_by not in ALLOWED_PREDICTION_SORT_FIELDS:
            allowed = list(ALLOWED_PREDICTION_SORT_FIELDS.keys())
            raise ValueError(f"Invalid sort_by '{sort_by}'. Supported sort fields: {allowed}")

        sort_col = ALLOWED_PREDICTION_SORT_FIELDS[sort_by]
        order_clause = desc(sort_col) if sort_order.lower() == "desc" else asc(sort_col)

        # Build filter predicates
        filters = []
        if detector_name:
            filters.append(PredictionRecord.detector_name == detector_name.lower().strip())
        if predicted_class:
            filters.append(PredictionRecord.predicted_class == predicted_class.lower().strip())
        if start_time:
            filters.append(PredictionRecord.created_at >= start_time)
        if end_time:
            filters.append(PredictionRecord.created_at <= end_time)
        if min_malicious_probability is not None:
            filters.append(PredictionRecord.malicious_probability >= min_malicious_probability)
        if max_malicious_probability is not None:
            filters.append(PredictionRecord.malicious_probability <= max_malicious_probability)

        # 1. Total count query
        count_stmt = select(func.count(PredictionRecord.id))
        if filters:
            count_stmt = count_stmt.where(and_(*filters))
        total_items = (await self.session.execute(count_stmt)).scalar() or 0

        total_pages = max(1, math.ceil(total_items / page_size)) if total_items > 0 else 0
        current_page = max(1, page)
        offset = (current_page - 1) * page_size

        # 2. Paginated items query with stable tie-breaker
        items_stmt = select(PredictionRecord)
        if filters:
            items_stmt = items_stmt.where(and_(*filters))
        items_stmt = items_stmt.order_by(order_clause, desc(PredictionRecord.id)).offset(offset).limit(page_size)

        result = await self.session.execute(items_stmt)
        items = list(result.scalars().all())

        return items, total_items, total_pages
