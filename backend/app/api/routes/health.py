from fastapi import APIRouter

from app.services.whisper_service import whisper_service

router = APIRouter()


@router.get("/health")
async def health_check():
    return {
        "status": "ok",
        "model_loaded": whisper_service.is_loaded,
        "model_path": whisper_service.model_path,
    }
