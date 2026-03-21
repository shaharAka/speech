from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.router import api_router
from app.config import settings
from app.services.whisper_service import whisper_service


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: create tables (SQLite dev mode)
    from app.core.database import engine
    from app.models import Base

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    # Seed base model version
    from app.core.database import async_session_factory
    from app.services.model_manager import ensure_base_model

    async with async_session_factory() as session:
        await ensure_base_model(session)
        await session.commit()

    # Load the active Whisper model (fine-tuned if available, otherwise base)
    from app.services.model_manager import get_active_model

    async with async_session_factory() as session:
        active_model = await get_active_model(session)

    model_path = settings.hf_ct2_model_id  # default: base model
    if active_model and active_model.model_path and not active_model.is_base:
        model_path = active_model.model_path

    await whisper_service.load_model(
        model_path=model_path,
        device=settings.whisper_device,
        compute_type=settings.whisper_compute_type,
    )
    yield
    # Shutdown: nothing to clean up


app = FastAPI(
    title="Hebrew Speech Trainer",
    description="Fine-tune Whisper for personalized Hebrew speech recognition",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(api_router, prefix="/api")
