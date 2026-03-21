from datetime import datetime
from typing import TYPE_CHECKING, Optional

from sqlalchemy import Boolean, DateTime, ForeignKey, Integer, String, func
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.models.base import Base

if TYPE_CHECKING:
    from app.models.text import Text
    from app.models.transcription import Transcription


class Recording(Base):
    __tablename__ = "recordings"

    id: Mapped[int] = mapped_column(primary_key=True)
    text_id: Mapped[int] = mapped_column(ForeignKey("texts.id"), nullable=False)
    audio_path: Mapped[str] = mapped_column(String(500), nullable=False)
    audio_duration_ms: Mapped[int] = mapped_column(Integer, nullable=False)
    sample_rate: Mapped[int] = mapped_column(Integer, default=16000)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
    used_in_training: Mapped[bool] = mapped_column(Boolean, default=False)

    text: Mapped["Text"] = relationship(back_populates="recordings")
    transcription: Mapped[Optional["Transcription"]] = relationship(
        back_populates="recording", uselist=False
    )
