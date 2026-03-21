from datetime import datetime

from pydantic import BaseModel, Field


class TextCreate(BaseModel):
    title: str = Field(min_length=1, max_length=200)
    content: str = Field(min_length=1)
    difficulty: str = Field(default="medium", pattern="^(easy|medium|hard)$")


class TextUpdate(BaseModel):
    title: str | None = None
    content: str | None = None
    difficulty: str | None = None


class TextResponse(BaseModel):
    id: int
    title: str
    content: str
    difficulty: str
    category: str
    word_count: int
    is_builtin: bool
    round: int = 1
    created_at: datetime
    recording_count: int = 0

    model_config = {"from_attributes": True}


class TextListResponse(BaseModel):
    items: list[TextResponse]
    total: int


class RoundProgressResponse(BaseModel):
    current_round: int
    total_texts: int
    practiced_texts: int
    is_complete: bool
    performance_summary: dict | None = None


class GenerateRoundResponse(BaseModel):
    round: int
    texts_created: int
