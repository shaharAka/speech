from datetime import datetime
from typing import Optional

from sqlalchemy import Boolean, DateTime, ForeignKey, Integer, Text as TextCol, func
from sqlalchemy.orm import Mapped, mapped_column

from app.models.base import Base


class CoachingReport(Base):
    __tablename__ = "coaching_reports"

    id: Mapped[int] = mapped_column(primary_key=True)
    training_run_id: Mapped[int] = mapped_column(
        ForeignKey("training_runs.id"), unique=True, nullable=False
    )
    round_number: Mapped[int] = mapped_column(Integer, nullable=False)
    next_round_number: Mapped[int] = mapped_column(Integer, nullable=False)

    # Coaching content (stored as JSON text)
    summary_text: Mapped[str] = mapped_column(TextCol, nullable=False)
    insights_json: Mapped[str] = mapped_column(TextCol, nullable=False, default="[]")
    recommendations_json: Mapped[str] = mapped_column(TextCol, nullable=False, default="[]")
    wer_trajectory_json: Mapped[str] = mapped_column(TextCol, nullable=False, default="[]")
    error_analysis_json: Mapped[str] = mapped_column(TextCol, nullable=False, default="{}")
    difficulty_distribution_json: Mapped[str] = mapped_column(TextCol, nullable=False, default="{}")
    suggested_next_params_json: Mapped[Optional[str]] = mapped_column(TextCol, nullable=True)

    texts_generated: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    is_round1_noise: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)

    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
