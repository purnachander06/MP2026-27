"""Initial Phase 7 schema: predictions and explanations tables

Revision ID: 001_initial_schema
Revises: None
Create Date: 2026-10-13 14:30:00.000000

"""
from typing import Sequence, Union
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = "001_initial_schema"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # 1. Create predictions table
    op.create_table(
        "predictions",
        sa.Column("id", sa.String(length=36), primary_key=True),
        sa.Column("prediction_id", sa.String(length=64), nullable=False, unique=True),
        sa.Column("detector_name", sa.String(length=64), nullable=False),
        sa.Column("model_version", sa.String(length=32), nullable=False, server_default="1.0.0"),
        sa.Column("predicted_class", sa.String(length=16), nullable=False),
        sa.Column("prediction", sa.Integer(), nullable=False),
        sa.Column("malicious_probability", sa.Float(), nullable=False),
        sa.Column("benign_probability", sa.Float(), nullable=False),
        sa.Column("threshold_used", sa.Float(), nullable=False),
        sa.Column("confidence_score", sa.Float(), nullable=False),
        sa.Column("is_calibrated", sa.Boolean(), nullable=False, server_default=sa.text("false")),
        sa.Column("validated_evidence", sa.JSON(), nullable=False),
        sa.Column("stripped_leakage_fields", sa.JSON(), nullable=False),
        sa.Column("inference_time_ms", sa.Float(), nullable=False, server_default=sa.text("0.0")),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
    )
    op.create_index("ix_predictions_prediction_id", "predictions", ["prediction_id"])
    op.create_index("ix_predictions_created_at", "predictions", ["created_at"])
    op.create_index("idx_predictions_class_created", "predictions", ["predicted_class", "created_at"])
    op.create_index("idx_predictions_det_created", "predictions", ["detector_name", "created_at"])

    # 2. Create explanations table
    op.create_table(
        "explanations",
        sa.Column("id", sa.String(length=36), primary_key=True),
        sa.Column(
            "prediction_id",
            sa.String(length=64),
            sa.ForeignKey("predictions.prediction_id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("predicted_class", sa.String(length=16), nullable=False),
        sa.Column("detector_name", sa.String(length=64), nullable=False),
        sa.Column("confidence_score", sa.Float(), nullable=True),
        sa.Column("severity", sa.String(length=16), nullable=False),
        sa.Column("severity_rationale", sa.Text(), nullable=False),
        sa.Column("summary", sa.Text(), nullable=False),
        sa.Column("supporting_evidence", sa.JSON(), nullable=False),
        sa.Column("uncertainties", sa.JSON(), nullable=False),
        sa.Column("recommended_actions", sa.JSON(), nullable=False),
        sa.Column("requires_human_review", sa.Boolean(), nullable=False, server_default=sa.text("false")),
        sa.Column("explanation_provider", sa.String(length=64), nullable=False),
        sa.Column("model_version", sa.String(length=32), nullable=False, server_default="1.0.0"),
        sa.Column("is_fallback", sa.Boolean(), nullable=False, server_default=sa.text("false")),
        sa.Column("disclaimer", sa.Text(), nullable=False),
        sa.Column("operator_context", sa.Text(), nullable=True),
        sa.Column("generated_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
    )
    op.create_index("ix_explanations_prediction_id", "explanations", ["prediction_id"])
    op.create_index("ix_explanations_severity", "explanations", ["severity"])
    op.create_index("ix_explanations_requires_human_review", "explanations", ["requires_human_review"])
    op.create_index("ix_explanations_is_fallback", "explanations", ["is_fallback"])
    op.create_index("ix_explanations_generated_at", "explanations", ["generated_at"])
    op.create_index("idx_explanations_sev_generated", "explanations", ["severity", "generated_at"])
    op.create_index("idx_explanations_det_generated", "explanations", ["detector_name", "generated_at"])


def downgrade() -> None:
    op.drop_table("explanations")
    op.drop_table("predictions")
