from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    database_url: str = "sqlite+aiosqlite:///./storage/whisper.db"
    redis_url: str = "redis://localhost:6379/0"
    model_storage_path: str = "./storage/models"
    audio_storage_path: str = "./storage/audio"
    whisper_device: str = "cpu"
    whisper_compute_type: str = "int8"
    hf_model_id: str = "ivrit-ai/whisper-large-v3-turbo"
    hf_ct2_model_id: str = "ivrit-ai/whisper-large-v3-turbo-ct2"
    cors_origins: str = "http://localhost:3000"
    min_recordings_for_training: int = 50

    # Gemini API (for adaptive text generation)
    gemini_api_key: str = ""

    # GCP Training Configuration
    gcp_project_id: str = "whisper-489414"
    gcp_region: str = "us-central1"
    gcp_zone: str = "us-central1-a"
    gcs_bucket: str = "whisper-training-whisper-489414"
    gcp_training_enabled: bool = False  # Raw VM mode (disabled — use Vertex AI)
    gcp_machine_type: str = "g2-standard-8"
    gcp_gpu_type: str = "nvidia-l4"
    gcp_use_spot: bool = True

    # Vertex AI Training (preferred over raw VMs)
    vertex_training_enabled: bool = True

    model_config = {"env_file": ".env", "extra": "ignore"}


settings = Settings()
