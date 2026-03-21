from celery import Celery

from app.config import settings

celery_app = Celery(
    "whisper_trainer",
    broker=settings.redis_url,
    backend=settings.redis_url,
)

celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    task_track_started=True,
    include=["app.tasks.fine_tune_task"],
)
