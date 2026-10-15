"""Analytics Repository computing live dashboard KPIs, time-series, and model stats."""

from datetime import datetime
from typing import Dict, List, Optional, Any
from sqlalchemy import select, func, and_, case
from sqlalchemy.ext.asyncio import AsyncSession

from src.db.models import PredictionRecord, ExplanationRecord


class AnalyticsRepository:
    """Async repository computing database-level threat analytics."""

    def __init__(self, session: AsyncSession):
        self.session = session

    async def get_summary(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
    ) -> Dict[str, Any]:
        """
        Compute high-level SOC dashboard summary KPIs.
        Safely handles empty tables with 0 values and zero percentages.
        """
        pred_filters = []
        if start_time:
            pred_filters.append(PredictionRecord.created_at >= start_time)
        if end_time:
            pred_filters.append(PredictionRecord.created_at <= end_time)

        # 1. Prediction counts
        pred_stmt = select(
            func.count(PredictionRecord.id).label("total"),
            func.sum(case((PredictionRecord.predicted_class == "malicious", 1), else_=0)).label("malicious"),
            func.sum(case((PredictionRecord.predicted_class == "benign", 1), else_=0)).label("benign"),
            func.max(PredictionRecord.created_at).label("latest_created"),
        )
        if pred_filters:
            pred_stmt = pred_stmt.where(and_(*pred_filters))

        pred_res = (await self.session.execute(pred_stmt)).one()
        total_preds = int(pred_res.total or 0)
        mal_preds = int(pred_res.malicious or 0)
        ben_preds = int(pred_res.benign or 0)
        latest_event = pred_res.latest_created.isoformat() if pred_res.latest_created else None

        mal_pct = round((mal_preds / total_preds) * 100, 2) if total_preds > 0 else 0.0

        # 2. Per-detector counts
        det_stmt = select(
            PredictionRecord.detector_name,
            func.count(PredictionRecord.id),
        )
        if pred_filters:
            det_stmt = det_stmt.where(and_(*pred_filters))
        det_stmt = det_stmt.group_by(PredictionRecord.detector_name)
        detector_counts = {row[0]: int(row[1]) for row in (await self.session.execute(det_stmt)).all()}

        # 3. Explanation counts
        exp_filters = []
        if start_time:
            exp_filters.append(ExplanationRecord.generated_at >= start_time)
        if end_time:
            exp_filters.append(ExplanationRecord.generated_at <= end_time)

        exp_stmt = select(
            func.count(ExplanationRecord.id).label("total"),
            func.sum(case((ExplanationRecord.requires_human_review.is_(True), 1), else_=0)).label("human_review"),
            func.sum(case((ExplanationRecord.is_fallback.is_(True), 1), else_=0)).label("fallback"),
        )
        if exp_filters:
            exp_stmt = exp_stmt.where(and_(*exp_filters))

        exp_res = (await self.session.execute(exp_stmt)).one()
        total_exps = int(exp_res.total or 0)
        human_review_count = int(exp_res.human_review or 0)
        fallback_count = int(exp_res.fallback or 0)

        human_review_pct = round((human_review_count / total_exps) * 100, 2) if total_exps > 0 else 0.0
        fallback_pct = round((fallback_count / total_exps) * 100, 2) if total_exps > 0 else 0.0

        return {
            "total_predictions": total_preds,
            "malicious_predictions": mal_preds,
            "benign_predictions": ben_preds,
            "malicious_percentage": mal_pct,
            "total_explanations": total_exps,
            "requires_human_review": human_review_count,
            "human_review_percentage": human_review_pct,
            "fallback_explanations": fallback_count,
            "fallback_percentage": fallback_pct,
            "detector_counts": detector_counts,
            "latest_event_timestamp": latest_event,
        }

    async def get_alerts_over_time(
        self,
        bucket: str = "hour",
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
    ) -> List[Dict[str, Any]]:
        """
        Group predictions chronologically into time buckets ('hour', 'day', 'week').
        Uses date_trunc on PostgreSQL and strftime on SQLite.
        """
        bucket_norm = bucket.lower().strip()
        if bucket_norm not in ("hour", "day", "week"):
            raise ValueError(f"Invalid bucket '{bucket}'. Supported buckets: 'hour', 'day', 'week'.")

        bind = self.session.bind
        dialect_name = bind.dialect.name if bind else "postgresql"

        # Cross-dialect bucket expression
        if dialect_name == "postgresql":
            bucket_expr = func.date_trunc(bucket_norm, PredictionRecord.created_at).label("bucket_time")
        else:
            # SQLite formatting
            if bucket_norm == "hour":
                fmt = "%Y-%m-%d %H:00:00"
            elif bucket_norm == "week":
                fmt = "%Y-%W"
            else:
                fmt = "%Y-%m-%d 00:00:00"
            bucket_expr = func.strftime(fmt, PredictionRecord.created_at).label("bucket_time")

        filters = []
        if start_time:
            filters.append(PredictionRecord.created_at >= start_time)
        if end_time:
            filters.append(PredictionRecord.created_at <= end_time)

        stmt = (
            select(
                bucket_expr,
                func.count(PredictionRecord.id).label("total"),
                func.sum(case((PredictionRecord.predicted_class == "malicious", 1), else_=0)).label("malicious"),
                func.sum(case((PredictionRecord.predicted_class == "benign", 1), else_=0)).label("benign"),
            )
            .group_by(bucket_expr)
            .order_by(bucket_expr.asc())
        )
        if filters:
            stmt = stmt.where(and_(*filters))

        rows = (await self.session.execute(stmt)).all()
        result = []
        for r in rows:
            b_val = r[0]
            b_str = b_val.isoformat() if hasattr(b_val, "isoformat") else str(b_val)
            result.append({
                "bucket_start": b_str,
                "total_predictions": int(r.total or 0),
                "malicious_count": int(r.malicious or 0),
                "benign_count": int(r.benign or 0),
            })
        return result

    async def get_threat_breakdown(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
    ) -> List[Dict[str, Any]]:
        """Group predictions by detector name and predicted class."""
        filters = []
        if start_time:
            filters.append(PredictionRecord.created_at >= start_time)
        if end_time:
            filters.append(PredictionRecord.created_at <= end_time)

        stmt = (
            select(
                PredictionRecord.detector_name,
                PredictionRecord.predicted_class,
                func.count(PredictionRecord.id).label("count"),
            )
            .group_by(PredictionRecord.detector_name, PredictionRecord.predicted_class)
            .order_by(PredictionRecord.detector_name.asc(), PredictionRecord.predicted_class.asc())
        )
        if filters:
            stmt = stmt.where(and_(*filters))

        rows = (await self.session.execute(stmt)).all()
        return [
            {
                "detector_name": r[0],
                "predicted_class": r[1],
                "count": int(r[2]),
            }
            for r in rows
        ]

    async def get_detector_usage(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
    ) -> List[Dict[str, Any]]:
        """Return usage, latency, and activity time range per detector."""
        filters = []
        if start_time:
            filters.append(PredictionRecord.created_at >= start_time)
        if end_time:
            filters.append(PredictionRecord.created_at <= end_time)

        stmt = (
            select(
                PredictionRecord.detector_name,
                func.count(PredictionRecord.id).label("total"),
                func.sum(case((PredictionRecord.predicted_class == "malicious", 1), else_=0)).label("malicious"),
                func.sum(case((PredictionRecord.predicted_class == "benign", 1), else_=0)).label("benign"),
                func.avg(PredictionRecord.inference_time_ms).label("avg_inference_ms"),
                func.min(PredictionRecord.created_at).label("first_seen"),
                func.max(PredictionRecord.created_at).label("last_seen"),
            )
            .group_by(PredictionRecord.detector_name)
            .order_by(func.count(PredictionRecord.id).desc())
        )
        if filters:
            stmt = stmt.where(and_(*filters))

        rows = (await self.session.execute(stmt)).all()
        result = []
        for r in rows:
            result.append({
                "detector_name": r.detector_name,
                "total_predictions": int(r.total or 0),
                "malicious_count": int(r.malicious or 0),
                "benign_count": int(r.benign or 0),
                "mean_inference_time_ms": round(float(r.avg_inference_ms or 0.0), 3),
                "first_seen": r.first_seen.isoformat() if r.first_seen else None,
                "last_seen": r.last_seen.isoformat() if r.last_seen else None,
            })
        return result

    async def get_explanation_quality(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
    ) -> Dict[str, Any]:
        """Compute explanation quality, severity breakdown, and provider usage stats."""
        filters = []
        if start_time:
            filters.append(ExplanationRecord.generated_at >= start_time)
        if end_time:
            filters.append(ExplanationRecord.generated_at <= end_time)

        # 1. Total, fallback, and human review counts
        tot_stmt = select(
            func.count(ExplanationRecord.id).label("total"),
            func.sum(case((ExplanationRecord.is_fallback.is_(True), 1), else_=0)).label("fallback"),
            func.sum(case((ExplanationRecord.requires_human_review.is_(True), 1), else_=0)).label("human_review"),
        )
        if filters:
            tot_stmt = tot_stmt.where(and_(*filters))

        tot_res = (await self.session.execute(tot_stmt)).one()
        total_exps = int(tot_res.total or 0)
        fallback_cnt = int(tot_res.fallback or 0)
        human_review_cnt = int(tot_res.human_review or 0)

        fallback_rate = round((fallback_cnt / total_exps) * 100, 2) if total_exps > 0 else 0.0
        human_review_rate = round((human_review_cnt / total_exps) * 100, 2) if total_exps > 0 else 0.0

        # 2. Severity breakdown
        sev_stmt = select(
            ExplanationRecord.severity,
            func.count(ExplanationRecord.id),
        )
        if filters:
            sev_stmt = sev_stmt.where(and_(*filters))
        sev_stmt = sev_stmt.group_by(ExplanationRecord.severity)
        severity_counts = {row[0]: int(row[1]) for row in (await self.session.execute(sev_stmt)).all()}

        # 3. Provider usage
        prov_stmt = select(
            ExplanationRecord.explanation_provider,
            func.count(ExplanationRecord.id),
        )
        if filters:
            prov_stmt = prov_stmt.where(and_(*filters))
        prov_stmt = prov_stmt.group_by(ExplanationRecord.explanation_provider)
        provider_counts = {row[0]: int(row[1]) for row in (await self.session.execute(prov_stmt)).all()}

        return {
            "total_explanations": total_exps,
            "fallback_count": fallback_cnt,
            "fallback_rate_pct": fallback_rate,
            "human_review_count": human_review_cnt,
            "human_review_rate_pct": human_review_rate,
            "severity_breakdown": severity_counts,
            "provider_usage": provider_counts,
        }
