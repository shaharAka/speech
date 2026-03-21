from datetime import datetime

from pydantic import BaseModel


class TranscribeRequest(BaseModel):
    recording_id: int


class WordDiffEntry(BaseModel):
    ref_word: str | None = None
    hyp_word: str | None = None
    status: str  # "correct", "substitution", "insertion", "deletion"


class TranscriptionResponse(BaseModel):
    id: int
    recording_id: int
    raw_text: str
    normalized_text: str
    reference_text: str
    wer_score: float
    cer_score: float
    word_diff: list[WordDiffEntry]
    model_version_id: int
    inference_time_ms: int
    created_at: datetime

    model_config = {"from_attributes": True}


class StatsResponse(BaseModel):
    total_recordings: int
    total_transcriptions: int
    average_wer: float | None
    best_wer: float | None
    recent_wer_trend: list[float]
    recordings_needed_for_training: int
