from datetime import datetime

from pydantic import BaseModel


class TrainingStartRequest(BaseModel):
    num_epochs: int = 5
    lora_rank: int = 32
    learning_rate: float = 1e-4


class TrainingRunResponse(BaseModel):
    id: int
    status: str
    base_model_version_id: int
    result_model_version_id: int | None
    num_samples: int
    num_epochs: int
    lora_rank: int
    learning_rate: float
    train_wer: float | None
    eval_wer: float | None
    training_loss: float | None
    error_message: str | None
    coaching_status: str | None = None
    started_at: datetime | None
    completed_at: datetime | None
    created_at: datetime

    model_config = {"from_attributes": True}


class DataStatsResponse(BaseModel):
    total_recordings: int
    usable_recordings: int
    min_required: int
    is_ready: bool


class CoachingReportResponse(BaseModel):
    id: int
    training_run_id: int
    round_number: int
    next_round_number: int
    summary_text: str
    insights: list[dict]
    recommendations: list[dict]
    wer_trajectory: list[dict]
    difficulty_distribution: dict
    suggested_next_params: dict | None
    texts_generated: int
    is_round1_noise: bool
    created_at: datetime
