from datetime import datetime
from typing import Optional

from sqlalchemy import DateTime, Float, ForeignKey, Integer, String, Text as TextCol, func
from sqlalchemy.orm import Mapped, mapped_column

from app.models.base import Base


class TrainingRun(Base):
    __tablename__ = "training_runs"

    id: Mapped[int] = mapped_column(primary_key=True)
    status: Mapped[str] = mapped_column(String(20), nullable=False, default="pending")
    base_model_version_id: Mapped[int] = mapped_column(ForeignKey("model_versions.id"))
    result_model_version_id: Mapped[Optional[int]] = mapped_column(
        ForeignKey("model_versions.id"), nullable=True
    )
    num_samples: Mapped[int] = mapped_column(Integer, nullable=False)
    num_epochs: Mapped[int] = mapped_column(Integer, default=5)
    lora_rank: Mapped[int] = mapped_column(Integer, default=32)
    learning_rate: Mapped[float] = mapped_column(Float, default=1e-4)
    train_wer: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    eval_wer: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    training_loss: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    celery_task_id: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    error_message: Mapped[Optional[str]] = mapped_column(TextCol, nullable=True)
    coaching_status: Mapped[Optional[str]] = mapped_column(String(20), nullable=True)
    started_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    completed_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
