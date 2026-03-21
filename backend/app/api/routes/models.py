from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db
from app.schemas.model_version import ModelVersionResponse
from app.services.model_manager import activate_model, get_active_model, list_model_versions

router = APIRouter()


@router.get("", response_model=list[ModelVersionResponse])
async def list_models(db: AsyncSession = Depends(get_db)):
    versions = await list_model_versions(db)
    return [ModelVersionResponse.model_validate(v) for v in versions]


@router.get("/active", response_model=ModelVersionResponse)
async def get_active(db: AsyncSession = Depends(get_db)):
    model = await get_active_model(db)
    if not model:
        raise HTTPException(status_code=404, detail="No active model")
    return ModelVersionResponse.model_validate(model)


@router.post("/{model_id}/activate", response_model=ModelVersionResponse)
async def activate(model_id: int, db: AsyncSession = Depends(get_db)):
    model = await activate_model(db, model_id)
    if not model:
        raise HTTPException(status_code=404, detail="Model version not found")
    return ModelVersionResponse.model_validate(model)
