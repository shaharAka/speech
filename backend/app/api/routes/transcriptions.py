import json

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db
from app.schemas.transcription import (
    StatsResponse,
    TranscribeRequest,
    TranscriptionResponse,
    WordDiffEntry,
)
from app.services.transcription_service import get_stats, get_transcription, transcribe_recording

router = APIRouter()


@router.post("", response_model=TranscriptionResponse, status_code=201)
async def transcribe(body: TranscribeRequest, db: AsyncSession = Depends(get_db)):
    try:
        transcription = await transcribe_recording(db, body.recording_id)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))

    word_diff = json.loads(transcription.word_diff_json)

    return TranscriptionResponse(
        id=transcription.id,
        recording_id=transcription.recording_id,
        raw_text=transcription.raw_text,
        normalized_text=transcription.normalized_text,
        reference_text=transcription.reference_text,
        wer_score=transcription.wer_score,
        cer_score=transcription.cer_score,
        word_diff=[WordDiffEntry(**w) for w in word_diff],
        model_version_id=transcription.model_version_id,
        inference_time_ms=transcription.inference_time_ms,
        created_at=transcription.created_at,
    )


@router.get("/{transcription_id}", response_model=TranscriptionResponse)
async def get_transcription_detail(
    transcription_id: int, db: AsyncSession = Depends(get_db)
):
    transcription = await get_transcription(db, transcription_id)
    if not transcription:
        raise HTTPException(status_code=404, detail="Transcription not found")

    word_diff = json.loads(transcription.word_diff_json)

    return TranscriptionResponse(
        id=transcription.id,
        recording_id=transcription.recording_id,
        raw_text=transcription.raw_text,
        normalized_text=transcription.normalized_text,
        reference_text=transcription.reference_text,
        wer_score=transcription.wer_score,
        cer_score=transcription.cer_score,
        word_diff=[WordDiffEntry(**w) for w in word_diff],
        model_version_id=transcription.model_version_id,
        inference_time_ms=transcription.inference_time_ms,
        created_at=transcription.created_at,
    )


@router.get("/stats/overview", response_model=StatsResponse)
async def stats(db: AsyncSession = Depends(get_db)):
    return await get_stats(db)
