from datetime import datetime

from pydantic import BaseModel


class RecordingResponse(BaseModel):
    id: int
    text_id: int
    audio_duration_ms: int
    sample_rate: int
    created_at: datetime
    used_in_training: bool
    has_transcription: bool = False

    model_config = {"from_attributes": True}


class RecordingListResponse(BaseModel):
    items: list[RecordingResponse]
    total: int
