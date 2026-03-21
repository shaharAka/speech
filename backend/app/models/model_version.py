from datetime import datetime
from typing import Optional

from sqlalchemy import Boolean, DateTime, Float, Integer, String, func
from sqlalchemy.orm import Mapped, mapped_column

from app.models.base import Base


class ModelVersion(Base):
    __tablename__ = "model_versions"

    id: Mapped[int] = mapped_column(primary_key=True)
    version_tag: Mapped[str] = mapped_column(String(50), unique=True, nullable=False)
    display_name: Mapped[str] = mapped_column(String(200), nullable=False)
    base_model_name: Mapped[str] = mapped_column(String(200), nullable=False)
    model_path: Mapped[str] = mapped_column(String(500), nullable=False)
    adapter_path: Mapped[Optional[str]] = mapped_column(String(500), nullable=True)
    is_active: Mapped[bool] = mapped_column(Boolean, default=False)
    is_base: Mapped[bool] = mapped_column(Boolean, default=False)
    eval_wer: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    eval_wer_improvement: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    num_training_samples: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
