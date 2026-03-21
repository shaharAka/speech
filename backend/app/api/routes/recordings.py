from fastapi import APIRouter, Depends, HTTPException, UploadFile
from fastapi.responses import FileResponse
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db
from app.schemas.recording import RecordingListResponse, RecordingResponse
from app.services import recording_service
from app.services.storage_service import storage_service

router = APIRouter()


@router.post("", response_model=RecordingResponse, status_code=201)
async def upload_recording(
    text_id: int,
    audio: UploadFile,
    db: AsyncSession = Depends(get_db),
):
    content = await audio.read()
    if not content:
        raise HTTPException(status_code=400, detail="Empty audio file")

    rec = await recording_service.create_recording(
        db=db,
        text_id=text_id,
        audio_content=content,
        original_filename=audio.filename or "recording.webm",
    )

    return RecordingResponse(
        id=rec.id,
        text_id=rec.text_id,
        audio_duration_ms=rec.audio_duration_ms,
        sample_rate=rec.sample_rate,
        created_at=rec.created_at,
        used_in_training=rec.used_in_training,
        has_transcription=False,
    )


@router.get("", response_model=RecordingListResponse)
async def list_recordings(
    text_id: int | None = None,
    limit: int = 50,
    offset: int = 0,
    db: AsyncSession = Depends(get_db),
):
    recordings, total = await recording_service.list_recordings(
        db, text_id=text_id, limit=limit, offset=offset
    )

    items = [
        RecordingResponse(
            id=r.id,
            text_id=r.text_id,
            audio_duration_ms=r.audio_duration_ms,
            sample_rate=r.sample_rate,
            created_at=r.created_at,
            used_in_training=r.used_in_training,
            has_transcription=r.transcription is not None,
        )
        for r in recordings
    ]

    return RecordingListResponse(items=items, total=total)


@router.get("/{recording_id}", response_model=RecordingResponse)
async def get_recording(recording_id: int, db: AsyncSession = Depends(get_db)):
    rec = await recording_service.get_recording(db, recording_id)
    if not rec:
        raise HTTPException(status_code=404, detail="Recording not found")

    return RecordingResponse(
        id=rec.id,
        text_id=rec.text_id,
        audio_duration_ms=rec.audio_duration_ms,
        sample_rate=rec.sample_rate,
        created_at=rec.created_at,
        used_in_training=rec.used_in_training,
        has_transcription=rec.transcription is not None,
    )


@router.get("/{recording_id}/audio")
async def get_audio(recording_id: int, db: AsyncSession = Depends(get_db)):
    rec = await recording_service.get_recording(db, recording_id)
    if not rec:
        raise HTTPException(status_code=404, detail="Recording not found")

    audio_path = storage_service.get_full_audio_path(rec.audio_path)
    if not audio_path.exists():
        raise HTTPException(status_code=404, detail="Audio file not found")

    return FileResponse(str(audio_path), media_type="audio/wav")


@router.delete("/{recording_id}", status_code=204)
async def delete_recording(recording_id: int, db: AsyncSession = Depends(get_db)):
    deleted = await recording_service.delete_recording(db, recording_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Recording not found")
