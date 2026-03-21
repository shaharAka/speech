from datetime import datetime
from typing import TYPE_CHECKING

from sqlalchemy import DateTime, Float, ForeignKey, Integer, String, Text as TextCol, func
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.models.base import Base

if TYPE_CHECKING:
    from app.models.model_version import ModelVersion
    from app.models.recording import Recording


class Transcription(Base):
    __tablename__ = "transcriptions"

    id: Mapped[int] = mapped_column(primary_key=True)
    recording_id: Mapped[int] = mapped_column(ForeignKey("recordings.id"), unique=True)
    raw_text: Mapped[str] = mapped_column(TextCol, nullable=False)
    normalized_text: Mapped[str] = mapped_column(TextCol, nullable=False)
    reference_text: Mapped[str] = mapped_column(TextCol, nullable=False)
    wer_score: Mapped[float] = mapped_column(Float, nullable=False)
    cer_score: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    word_diff_json: Mapped[str] = mapped_column(TextCol, nullable=False)
    model_version_id: Mapped[int] = mapped_column(ForeignKey("model_versions.id"))
    inference_time_ms: Mapped[int] = mapped_column(Integer, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())

    recording: Mapped["Recording"] = relationship(back_populates="transcription")
    model_version: Mapped["ModelVersion"] = relationship()
