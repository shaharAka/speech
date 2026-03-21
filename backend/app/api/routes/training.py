import json

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.core.database import get_db
from app.models.coaching_report import CoachingReport
from app.models.recording import Recording
from app.models.training_run import TrainingRun
from app.schemas.training import (
    CoachingReportResponse,
    DataStatsResponse,
    TrainingRunResponse,
    TrainingStartRequest,
)
from app.services.model_manager import get_active_model

router = APIRouter()


def _run_to_response(r: TrainingRun) -> TrainingRunResponse:
    return TrainingRunResponse(
        id=r.id,
        status=r.status,
        base_model_version_id=r.base_model_version_id,
        result_model_version_id=r.result_model_version_id,
        num_samples=r.num_samples,
        num_epochs=r.num_epochs,
        lora_rank=r.lora_rank,
        learning_rate=r.learning_rate,
        train_wer=r.train_wer,
        eval_wer=r.eval_wer,
        training_loss=r.training_loss,
        error_message=r.error_message,
        coaching_status=r.coaching_status,
        started_at=r.started_at,
        completed_at=r.completed_at,
        created_at=r.created_at,
    )


def _report_to_response(report: CoachingReport) -> CoachingReportResponse:
    return CoachingReportResponse(
        id=report.id,
        training_run_id=report.training_run_id,
        round_number=report.round_number,
        next_round_number=report.next_round_number,
        summary_text=report.summary_text,
        insights=json.loads(report.insights_json),
        recommendations=json.loads(report.recommendations_json),
        wer_trajectory=json.loads(report.wer_trajectory_json),
        difficulty_distribution=json.loads(report.difficulty_distribution_json),
        suggested_next_params=json.loads(report.suggested_next_params_json)
        if report.suggested_next_params_json else None,
        texts_generated=report.texts_generated,
        is_round1_noise=report.is_round1_noise,
        created_at=report.created_at,
    )


@router.post("/start", response_model=TrainingRunResponse, status_code=201)
async def start_training(
    body: TrainingStartRequest,
    db: AsyncSession = Depends(get_db),
):
    # Check recording count
    count_result = await db.execute(select(func.count(Recording.id)))
    total = count_result.scalar_one()

    if total < settings.min_recordings_for_training:
        raise HTTPException(
            status_code=400,
            detail=f"Need at least {settings.min_recordings_for_training} recordings, have {total}",
        )

    active_model = await get_active_model(db)
    if not active_model:
        raise HTTPException(status_code=500, detail="No active model")

    run = TrainingRun(
        status="pending",
        base_model_version_id=active_model.id,
        num_samples=total,
        num_epochs=body.num_epochs,
        lora_rank=body.lora_rank,
        learning_rate=body.learning_rate,
    )
    db.add(run)
    await db.flush()

    # Dispatch Celery task for fine-tuning
    from app.tasks.fine_tune_task import fine_tune_task

    task = fine_tune_task.delay(run.id)
    run.celery_task_id = task.id

    return _run_to_response(run)


@router.get("/runs", response_model=list[TrainingRunResponse])
async def list_training_runs(db: AsyncSession = Depends(get_db)):
    result = await db.execute(
        select(TrainingRun).order_by(TrainingRun.created_at.desc())
    )
    return [_run_to_response(r) for r in result.scalars().all()]


@router.get("/runs/{run_id}", response_model=TrainingRunResponse)
async def get_training_run(run_id: int, db: AsyncSession = Depends(get_db)):
    result = await db.execute(
        select(TrainingRun).where(TrainingRun.id == run_id)
    )
    run = result.scalar_one_or_none()
    if not run:
        raise HTTPException(status_code=404, detail="Training run not found")
    return _run_to_response(run)


@router.get("/runs/{run_id}/coaching-report", response_model=CoachingReportResponse)
async def get_coaching_report(run_id: int, db: AsyncSession = Depends(get_db)):
    result = await db.execute(
        select(CoachingReport).where(CoachingReport.training_run_id == run_id)
    )
    report = result.scalar_one_or_none()
    if not report:
        raise HTTPException(status_code=404, detail="Coaching report not found")
    return _report_to_response(report)


@router.get("/latest-coaching-report", response_model=CoachingReportResponse)
async def get_latest_coaching_report(db: AsyncSession = Depends(get_db)):
    result = await db.execute(
        select(CoachingReport).order_by(CoachingReport.created_at.desc()).limit(1)
    )
    report = result.scalar_one_or_none()
    if not report:
        raise HTTPException(status_code=404, detail="No coaching reports found")
    return _report_to_response(report)


@router.get("/data-stats", response_model=DataStatsResponse)
async def data_stats(db: AsyncSession = Depends(get_db)):
    count_result = await db.execute(select(func.count(Recording.id)))
    total = count_result.scalar_one()

    # Usable = recordings that have transcriptions
    from app.models.transcription import Transcription

    usable_result = await db.execute(select(func.count(Transcription.id)))
    usable = usable_result.scalar_one()

    return DataStatsResponse(
        total_recordings=total,
        usable_recordings=usable,
        min_required=settings.min_recordings_for_training,
        is_ready=usable >= settings.min_recordings_for_training,
    )


@router.get("/config")
async def training_config():
    """Return current training configuration (GCP settings, etc.)."""
    return {
        "gcp_enabled": settings.gcp_training_enabled,
        "gcp_project": settings.gcp_project_id,
        "gcp_region": settings.gcp_region,
        "gcp_zone": settings.gcp_zone,
        "gcp_bucket": settings.gcs_bucket,
        "gcp_machine_type": settings.gcp_machine_type,
        "gcp_gpu": settings.gcp_gpu_type,
        "gcp_spot": settings.gcp_use_spot,
        "estimated_cost_per_run": "$1-3 (spot L4 GPU)",
    }
