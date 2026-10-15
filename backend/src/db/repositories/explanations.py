"""Repository performing asynchronous CRUD and paginated filtering on ExplanationRecord."""

from datetime import datetime
import math
from typing import Dict, List, Optional, Tuple, Any
from sqlalchemy import select, func, and_, desc, asc
from sqlalchemy.ext.asyncio import AsyncSession

from src.db.models import ExplanationRecord

ALLOWED_EXPLANATION_SORT_FIELDS = {
    "generated_at": ExplanationRecord.generated_at,
    "severity": ExplanationRecord.severity,
    "confidence_score": ExplanationRecord.confidence_score,
    "prediction_id": ExplanationRecord.prediction_id,
}


class ExplanationRepository:
    """Async repository for explanation records."""

    def __init__(self, session: AsyncSession):
        self.session = session

    async def create(self, data: Dict[str, Any]) -> ExplanationRecord:
        """Create and persist an explanation record."""
        record = ExplanationRecord(
            prediction_id=data["prediction_id"],
            predicted_class=data["predicted_class"],
            detector_name=data["detector_name"],
            confidence_score=data.get("confidence_score"),
            severity=data["severity"] if isinstance(data["severity"], str) else data["severity"].value,
            severity_rationale=data["severity_rationale"],
            summary=data["summary"],
            supporting_evidence=data.get("supporting_evidence", []),
            uncertainties=data.get("uncertainties", []),
            recommended_actions=data.get("recommended_actions", []),
            requires_human_review=data.get("requires_human_review", False),
            explanation_provider=data["explanation_provider"],
            model_version=data.get("model_version", "1.0.0"),
            is_fallback=data.get("is_fallback", False),
            disclaimer=data.get("disclaimer", "Advisory automated explanation grounded in flow telemetry."),
            operator_context=data.get("operator_context"),
        )
        self.session.add(record)
        await self.session.flush()
        return record

    async def get_by_id(self, record_id: str) -> Optional[ExplanationRecord]:
        """Retrieve explanation by internal ID."""
        stmt = select(ExplanationRecord).where(ExplanationRecord.id == record_id)
        result = await self.session.execute(stmt)
        return result.scalar_one_or_none()

    async def list_by_prediction_id(self, prediction_id: str) -> List[ExplanationRecord]:
        """Retrieve all explanations generated for a specific prediction."""
        stmt = (
            select(ExplanationRecord)
            .where(ExplanationRecord.prediction_id == prediction_id)
            .order_by(desc(ExplanationRecord.generated_at))
        )
        result = await self.session.execute(stmt)
        return list(result.scalars().all())

    async def list_paginated(
        self,
        page: int = 1,
        page_size: int = 20,
        detector_name: Optional[str] = None,
        severity: Optional[str] = None,
        is_fallback: Optional[bool] = None,
        requires_human_review: Optional[bool] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        sort_by: str = "generated_at",
        sort_order: str = "desc",
    ) -> Tuple[List[ExplanationRecord], int, int]:
        """
        Query explanations with filtering, count, and pagination.

        Returns:
            Tuple of (items, total_items, total_pages).
        """
        if sort_by not in ALLOWED_EXPLANATION_SORT_FIELDS:
            allowed = list(ALLOWED_EXPLANATION_SORT_FIELDS.keys())
            raise ValueError(f"Invalid sort_by '{sort_by}'. Supported sort fields: {allowed}")

        sort_col = ALLOWED_EXPLANATION_SORT_FIELDS[sort_by]
        order_clause = desc(sort_col) if sort_order.lower() == "desc" else asc(sort_col)

        filters = []
        if detector_name:
            filters.append(ExplanationRecord.detector_name == detector_name.lower().strip())
        if severity:
            filters.append(ExplanationRecord.severity == severity.lower().strip())
        if is_fallback is not None:
            filters.append(ExplanationRecord.is_fallback == is_fallback)
        if requires_human_review is not None:
            filters.append(ExplanationRecord.requires_human_review == requires_human_review)
        if start_time:
            filters.append(ExplanationRecord.generated_at >= start_time)
        if end_time:
            filters.append(ExplanationRecord.generated_at <= end_time)

        # 1. Total count query
        count_stmt = select(func.count(ExplanationRecord.id))
        if filters:
            count_stmt = count_stmt.where(and_(*filters))
        total_items = (await self.session.execute(count_stmt)).scalar() or 0

        total_pages = max(1, math.ceil(total_items / page_size)) if total_items > 0 else 0
        current_page = max(1, page)
        offset = (current_page - 1) * page_size

        # 2. Paginated items query with stable tie-breaker
        items_stmt = select(ExplanationRecord)
        if filters:
            items_stmt = items_stmt.where(and_(*filters))
        items_stmt = items_stmt.order_by(order_clause, desc(ExplanationRecord.id)).offset(offset).limit(page_size)

        result = await self.session.execute(items_stmt)
        items = list(result.scalars().all())

        return items, total_items, total_pages
