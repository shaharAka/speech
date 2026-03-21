import os
from datetime import datetime
from pathlib import Path

from app.config import settings


class StorageService:
    def __init__(self):
        self.audio_root = Path(settings.audio_storage_path)
        self.model_root = Path(settings.model_storage_path)

    def ensure_dirs(self) -> None:
        self.audio_root.mkdir(parents=True, exist_ok=True)
        self.model_root.mkdir(parents=True, exist_ok=True)

    def get_audio_path(self, recording_id: int) -> Path:
        now = datetime.now()
        dir_path = self.audio_root / str(now.year) / f"{now.month:02d}"
        dir_path.mkdir(parents=True, exist_ok=True)
        return dir_path / f"{recording_id}.wav"

    def get_audio_relative_path(self, recording_id: int) -> str:
        now = datetime.now()
        return f"{now.year}/{now.month:02d}/{recording_id}.wav"

    def get_full_audio_path(self, relative_path: str) -> Path:
        return self.audio_root / relative_path

    def get_model_dir(self, version_tag: str) -> Path:
        path = self.model_root / version_tag
        path.mkdir(parents=True, exist_ok=True)
        return path

    def audio_exists(self, relative_path: str) -> bool:
        return (self.audio_root / relative_path).exists()


storage_service = StorageService()
