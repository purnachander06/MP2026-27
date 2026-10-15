"""SQLAlchemy ORM models for Phase 7 predictions and threat explanations."""

from datetime import datetime, timezone
import uuid
from typing import List, Dict, Any, Optional
from sqlalchemy import (
    Column,
    String,
    Integer,
    Float,
    Boolean,
    Text,
    DateTime,
    ForeignKey,
    Index,
)
from sqlalchemy.orm import relationship, Mapped, mapped_column

from src.db.base import Base, CrossDialectJson


def utc_now() -> datetime:
    """Return current timezone-aware UTC datetime."""
    return datetime.now(timezone.utc)


class PredictionRecord(Base):
    """
    Persisted record of an intrusion detection model prediction.
    Stores authoritative outputs and sanitized input evidence.
    """
    __tablename__ = "predictions"

    id: Mapped[str] = mapped_column(
        String(36),
        primary_key=True,
        default=lambda: str(uuid.uuid4()),
    )
    prediction_id: Mapped[str] = mapped_column(
        String(64),
        unique=True,
        index=True,
        nullable=False,
    )
    detector_name: Mapped[str] = mapped_column(String(64), nullable=False)
    model_version: Mapped[str] = mapped_column(String(32), nullable=False, default="1.0.0")
    predicted_class: Mapped[str] = mapped_column(String(16), nullable=False)
    prediction: Mapped[int] = mapped_column(Integer, nullable=False)
    malicious_probability: Mapped[float] = mapped_column(Float, nullable=False)
    benign_probability: Mapped[float] = mapped_column(Float, nullable=False)
    threshold_used: Mapped[float] = mapped_column(Float, nullable=False)
    confidence_score: Mapped[float] = mapped_column(Float, nullable=False)
    is_calibrated: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    validated_evidence: Mapped[Dict[str, Any]] = mapped_column(CrossDialectJson, nullable=False, default=dict)
    stripped_leakage_fields: Mapped[List[str]] = mapped_column(CrossDialectJson, nullable=False, default=list)
    inference_time_ms: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=utc_now,
        index=True,
    )

    # Relationships
    explanations: Mapped[List["ExplanationRecord"]] = relationship(
        "ExplanationRecord",
        back_populates="prediction",
        cascade="all, delete-orphan",
        order_by="desc(ExplanationRecord.generated_at)",
    )

    __table_args__ = (
        Index("idx_predictions_class_created", "predicted_class", "created_at"),
        Index("idx_predictions_det_created", "detector_name", "created_at"),
    )

    def to_dict(self) -> Dict[str, Any]:
        """Convert ORM record to dictionary format."""
        return {
            "prediction_id": self.prediction_id,
            "detector_name": self.detector_name,
            "model_version": self.model_version,
            "predicted_class": self.predicted_class,
            "prediction": self.prediction,
            "malicious_probability": self.malicious_probability,
            "benign_probability": self.benign_probability,
            "threshold_used": self.threshold_used,
            "confidence_score": self.confidence_score,
            "is_calibrated": self.is_calibrated,
            "validated_evidence": self.validated_evidence,
            "stripped_leakage_fields": self.stripped_leakage_fields,
            "inference_time_ms": self.inference_time_ms,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }


class ExplanationRecord(Base):
    """
    Persisted record of an LLM or fallback threat explanation.
    Linked to a parent prediction via foreign key on prediction_id.
    """
    __tablename__ = "explanations"

    id: Mapped[str] = mapped_column(
        String(36),
        primary_key=True,
        default=lambda: str(uuid.uuid4()),
    )
    prediction_id: Mapped[str] = mapped_column(
        String(64),
        ForeignKey("predictions.prediction_id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    predicted_class: Mapped[str] = mapped_column(String(16), nullable=False)
    detector_name: Mapped[str] = mapped_column(String(64), nullable=False)
    confidence_score: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    severity: Mapped[str] = mapped_column(String(16), nullable=False, index=True)
    severity_rationale: Mapped[str] = mapped_column(Text, nullable=False)
    summary: Mapped[str] = mapped_column(Text, nullable=False)
    supporting_evidence: Mapped[List[str]] = mapped_column(CrossDialectJson, nullable=False, default=list)
    uncertainties: Mapped[List[str]] = mapped_column(CrossDialectJson, nullable=False, default=list)
    recommended_actions: Mapped[List[str]] = mapped_column(CrossDialectJson, nullable=False, default=list)
    requires_human_review: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False, index=True)
    explanation_provider: Mapped[str] = mapped_column(String(64), nullable=False)
    model_version: Mapped[str] = mapped_column(String(32), nullable=False, default="1.0.0")
    is_fallback: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False, index=True)
    disclaimer: Mapped[str] = mapped_column(Text, nullable=False)
    operator_context: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    generated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=utc_now,
        index=True,
    )

    # Relationships
    prediction: Mapped["PredictionRecord"] = relationship(
        "PredictionRecord",
        back_populates="explanations",
    )

    __table_args__ = (
        Index("idx_explanations_sev_generated", "severity", "generated_at"),
        Index("idx_explanations_det_generated", "detector_name", "generated_at"),
    )

    def to_dict(self) -> Dict[str, Any]:
        """Convert ORM record to dictionary format."""
        return {
            "prediction_id": self.prediction_id,
            "predicted_class": self.predicted_class,
            "detector_name": self.detector_name,
            "confidence_score": self.confidence_score,
            "severity": self.severity,
            "severity_rationale": self.severity_rationale,
            "summary": self.summary,
            "supporting_evidence": self.supporting_evidence,
            "uncertainties": self.uncertainties,
            "recommended_actions": self.recommended_actions,
            "requires_human_review": self.requires_human_review,
            "explanation_provider": self.explanation_provider,
            "model_version": self.model_version,
            "is_fallback": self.is_fallback,
            "disclaimer": self.disclaimer,
            "operator_context": self.operator_context,
            "generated_at": self.generated_at.isoformat() if self.generated_at else None,
        }
