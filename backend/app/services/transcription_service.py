import json

from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.hebrew_utils import normalize_hebrew
from app.models.recording import Recording
from app.models.transcription import Transcription
from app.services.comparison_service import (
    compute_cer,
    compute_wer,
    compute_word_diff,
    word_diff_to_json,
)
from app.services.model_manager import get_active_model
from app.services.storage_service import storage_service
from app.services.whisper_service import whisper_service


async def transcribe_recording(
    db: AsyncSession,
    recording_id: int,
) -> Transcription:
    """Run Whisper on a recording and store the transcription with WER analysis."""
    # Get recording with text
    result = await db.execute(
        select(Recording).where(Recording.id == recording_id)
    )
    recording = result.scalar_one_or_none()
    if recording is None:
        raise ValueError(f"Recording {recording_id} not found")

    # Load the text content
    from app.models.text import Text

    text_result = await db.execute(select(Text).where(Text.id == recording.text_id))
    text = text_result.scalar_one()

    # Get active model
    active_model = await get_active_model(db)
    if active_model is None:
        raise RuntimeError("No active model version found")

    # Transcribe
    audio_path = str(storage_service.get_full_audio_path(recording.audio_path))
    result = await whisper_service.transcribe(audio_path)

    # Compare
    reference = text.content
    hypothesis = result.text
    normalized = normalize_hebrew(hypothesis)

    wer = compute_wer(reference, hypothesis)
    cer = compute_cer(reference, hypothesis)
    diff = compute_word_diff(reference, hypothesis)

    # Check if transcription already exists
    existing = await db.execute(
        select(Transcription).where(Transcription.recording_id == recording_id)
    )
    transcription = existing.scalar_one_or_none()

    if transcription:
        # Update existing
        transcription.raw_text = hypothesis
        transcription.normalized_text = normalized
        transcription.reference_text = reference
        transcription.wer_score = wer
        transcription.cer_score = cer
        transcription.word_diff_json = word_diff_to_json(diff)
        transcription.model_version_id = active_model.id
        transcription.inference_time_ms = result.inference_time_ms
    else:
        transcription = Transcription(
            recording_id=recording_id,
            raw_text=hypothesis,
            normalized_text=normalized,
            reference_text=reference,
            wer_score=wer,
            cer_score=cer,
            word_diff_json=word_diff_to_json(diff),
            model_version_id=active_model.id,
            inference_time_ms=result.inference_time_ms,
        )
        db.add(transcription)

    await db.flush()
    return transcription


async def get_transcription(db: AsyncSession, transcription_id: int) -> Transcription | None:
    result = await db.execute(
        select(Transcription).where(Transcription.id == transcription_id)
    )
    return result.scalar_one_or_none()


async def get_stats(db: AsyncSession) -> dict:
    """Get aggregate statistics."""
    from app.config import settings

    # Total recordings
    rec_count = await db.execute(select(func.count(Recording.id)))
    total_recordings = rec_count.scalar_one()

    # Total transcriptions
    trans_count = await db.execute(select(func.count(Transcription.id)))
    total_transcriptions = trans_count.scalar_one()

    # Average WER
    avg_wer_result = await db.execute(select(func.avg(Transcription.wer_score)))
    avg_wer = avg_wer_result.scalar_one()

    # Best WER
    best_wer_result = await db.execute(select(func.min(Transcription.wer_score)))
    best_wer = best_wer_result.scalar_one()

    # Recent WER trend (last 20 transcriptions)
    recent = await db.execute(
        select(Transcription.wer_score)
        .order_by(Transcription.created_at.desc())
        .limit(20)
    )
    recent_wer = [r for (r,) in recent.all()]
    recent_wer.reverse()  # Chronological order

    recordings_needed = max(0, settings.min_recordings_for_training - total_recordings)

    return {
        "total_recordings": total_recordings,
        "total_transcriptions": total_transcriptions,
        "average_wer": round(avg_wer, 4) if avg_wer else None,
        "best_wer": round(best_wer, 4) if best_wer else None,
        "recent_wer_trend": recent_wer,
        "recordings_needed_for_training": recordings_needed,
    }
