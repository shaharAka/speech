from datetime import datetime

from pydantic import BaseModel


class ModelVersionResponse(BaseModel):
    id: int
    version_tag: str
    display_name: str
    base_model_name: str
    is_active: bool
    is_base: bool
    eval_wer: float | None
    eval_wer_improvement: float | None
    num_training_samples: int | None
    created_at: datetime

    model_config = {"from_attributes": True}
