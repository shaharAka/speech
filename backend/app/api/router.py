from fastapi import APIRouter

from app.api.routes import health, models, recordings, texts, transcriptions, training

api_router = APIRouter()
api_router.include_router(health.router, tags=["health"])
api_router.include_router(texts.router, prefix="/texts", tags=["texts"])
api_router.include_router(recordings.router, prefix="/recordings", tags=["recordings"])
api_router.include_router(
    transcriptions.router, prefix="/transcriptions", tags=["transcriptions"]
)
api_router.include_router(training.router, prefix="/training", tags=["training"])
api_router.include_router(models.router, prefix="/models", tags=["models"])
