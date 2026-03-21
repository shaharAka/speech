import logging

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.models.model_version import ModelVersion
from app.services.whisper_service import whisper_service

logger = logging.getLogger(__name__)


async def ensure_base_model(db: AsyncSession) -> ModelVersion:
    """Ensure the base ivrit-ai model exists in the database."""
    result = await db.execute(
        select(ModelVersion).where(ModelVersion.is_base == True)  # noqa: E712
    )
    base = result.scalar_one_or_none()
    if base:
        return base

    base = ModelVersion(
        version_tag="base",
        display_name="ivrit-ai Whisper Large V3 Turbo (Hebrew)",
        base_model_name=settings.hf_ct2_model_id,
        model_path=settings.hf_ct2_model_id,
        is_active=True,
        is_base=True,
    )
    db.add(base)
    await db.flush()
    logger.info(f"Created base model version: {base.version_tag}")
    return base


async def get_active_model(db: AsyncSession) -> ModelVersion | None:
    result = await db.execute(
        select(ModelVersion).where(ModelVersion.is_active == True)  # noqa: E712
    )
    return result.scalar_one_or_none()


async def activate_model(db: AsyncSession, model_id: int) -> ModelVersion | None:
    """Set a model version as active and hot-swap the Whisper model."""
    # Deactivate all
    result = await db.execute(select(ModelVersion))
    for mv in result.scalars():
        mv.is_active = False

    # Activate the requested one
    result = await db.execute(
        select(ModelVersion).where(ModelVersion.id == model_id)
    )
    model = result.scalar_one_or_none()
    if model is None:
        return None

    model.is_active = True

    # Hot-swap the model
    await whisper_service.swap_model(model.model_path)
    logger.info(f"Activated model version: {model.version_tag}")

    return model


async def list_model_versions(db: AsyncSession) -> list[ModelVersion]:
    result = await db.execute(
        select(ModelVersion).order_by(ModelVersion.created_at.desc())
    )
    return list(result.scalars().all())
