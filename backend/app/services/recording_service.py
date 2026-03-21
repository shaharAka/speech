import os
import tempfile

from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.core.audio_utils import convert_to_wav_16k, get_audio_duration_ms
from app.models.recording import Recording
from app.services.storage_service import storage_service


async def create_recording(
    db: AsyncSession,
    text_id: int,
    audio_content: bytes,
    original_filename: str,
) -> Recording:
    """Save uploaded audio, convert to WAV 16kHz, create DB record."""
    # Save to temp file
    suffix = os.path.splitext(original_filename)[1] or ".webm"
    fd, temp_path = tempfile.mkstemp(suffix=suffix)
    try:
        os.write(fd, audio_content)
    finally:
        os.close(fd)

    # Create a placeholder recording to get an ID
    recording = Recording(
        text_id=text_id,
        audio_path="",
        audio_duration_ms=0,
    )
    db.add(recording)
    await db.flush()  # Get the ID

    # Convert to WAV 16kHz mono
    relative_path = storage_service.get_audio_relative_path(recording.id)
    output_path = storage_service.get_full_audio_path(relative_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    convert_to_wav_16k(temp_path, str(output_path))
    os.unlink(temp_path)

    # Update recording with actual info
    duration_ms = get_audio_duration_ms(str(output_path))
    recording.audio_path = relative_path
    recording.audio_duration_ms = duration_ms

    return recording


async def get_recording(db: AsyncSession, recording_id: int) -> Recording | None:
    result = await db.execute(
        select(Recording)
        .options(selectinload(Recording.text), selectinload(Recording.transcription))
        .where(Recording.id == recording_id)
    )
    return result.scalar_one_or_none()


async def list_recordings(
    db: AsyncSession,
    text_id: int | None = None,
    limit: int = 50,
    offset: int = 0,
) -> tuple[list[Recording], int]:
    query = select(Recording).options(selectinload(Recording.transcription))
    count_query = select(func.count(Recording.id))

    if text_id is not None:
        query = query.where(Recording.text_id == text_id)
        count_query = count_query.where(Recording.text_id == text_id)

    query = query.order_by(Recording.created_at.desc()).limit(limit).offset(offset)

    result = await db.execute(query)
    recordings = list(result.scalars().all())

    count_result = await db.execute(count_query)
    total = count_result.scalar_one()

    return recordings, total


async def delete_recording(db: AsyncSession, recording_id: int) -> bool:
    recording = await get_recording(db, recording_id)
    if recording is None:
        return False

    # Delete audio file
    audio_path = storage_service.get_full_audio_path(recording.audio_path)
    if audio_path.exists():
        audio_path.unlink()

    await db.delete(recording)
    return True
