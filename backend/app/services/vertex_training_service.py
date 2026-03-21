"""Vertex AI Custom Training Job management.

Replaces raw GCP VM creation with Vertex AI CustomJob, which uses
separate GPU quotas and handles infrastructure management automatically.

Reuses the GCS-based functions from gcp_training_service:
  - export_training_data, upload_training_data, upload_code_to_gcs
  - check_training_results, download_model
"""

import logging
import time

from google.cloud import aiplatform

from app.config import settings

logger = logging.getLogger(__name__)

# Regions to try if primary region is out of capacity
FALLBACK_REGIONS = [
    "us-central1",
    "us-east1",
    "us-west1",
    "us-east4",
    "europe-west4",
]


def submit_vertex_training_job(run_id: int) -> aiplatform.CustomJob:
    """Submit a Vertex AI CustomJob for Whisper fine-tuning.

    Uses the pre-built PyTorch GPU container and runs vertex_entrypoint.py.
    Returns the job object (already running asynchronously).
    """
    sa_email = f"whisper-trainer@{settings.gcp_project_id}.iam.gserviceaccount.com"
    job_display_name = f"whisper-train-run-{run_id}"

    worker_pool_specs = [
        {
            "machine_spec": {
                "machine_type": settings.gcp_machine_type,
                "accelerator_type": settings.gcp_gpu_type.upper().replace("-", "_"),
                "accelerator_count": 1,
            },
            "replica_count": 1,
            "container_spec": {
                "image_uri": "us-docker.pkg.dev/vertex-ai/training/pytorch-gpu.2-2:latest",
                "command": ["bash", "-c"],
                "args": [
                    # Install gsutil, download code, install package, run entrypoint
                    "pip install --quiet google-cloud-storage && "
                    f"gsutil -m cp -r gs://{settings.gcs_bucket}/code/* /opt/whisper/ && "
                    "cd /opt/whisper && "
                    "pip install --quiet -e '.[training]' ctranslate2 aiosqlite && "
                    f"python -m training.vertex_entrypoint "
                    f"--gcs-bucket={settings.gcs_bucket} --run-id={run_id}"
                ],
            },
        }
    ]

    # Deduplicate regions, primary first
    seen = set()
    regions = []
    for r in [settings.gcp_region] + FALLBACK_REGIONS:
        if r not in seen:
            seen.add(r)
            regions.append(r)

    last_error = None
    for region in regions:
        try:
            logger.info(f"Submitting Vertex AI job in region {region}...")
            aiplatform.init(
                project=settings.gcp_project_id,
                location=region,
                staging_bucket=f"gs://{settings.gcs_bucket}/vertex-staging",
            )

            job = aiplatform.CustomJob(
                display_name=job_display_name,
                worker_pool_specs=worker_pool_specs,
            )

            job.run(
                service_account=sa_email,
                sync=False,  # Non-blocking — we poll ourselves
            )

            logger.info(f"Vertex AI job submitted: {job.resource_name} in {region}")
            return job

        except Exception as e:
            err_str = str(e)
            if "quota" in err_str.lower() or "capacity" in err_str.lower() or "resource" in err_str.lower():
                logger.warning(f"Region {region} unavailable: {err_str[:200]}")
                last_error = e
                continue
            else:
                raise  # Non-capacity error — don't retry

    raise RuntimeError(f"No Vertex AI capacity in any region. Last error: {last_error}")


def poll_vertex_job(
    job: aiplatform.CustomJob,
    poll_interval: int = 60,
    max_wait: int = 4 * 60 * 60,
) -> str:
    """Poll a Vertex AI job until completion.

    Returns the final state string: 'SUCCEEDED', 'FAILED', 'CANCELLED', etc.
    """
    waited = 0
    while waited < max_wait:
        time.sleep(poll_interval)
        waited += poll_interval

        job.refresh()  # Fetch latest state from API
        state = job.state.name if job.state else "UNKNOWN"

        if waited % 300 == 0:
            logger.info(f"Vertex AI job state: {state} ({waited}s elapsed)")

        if state in ("PIPELINE_STATE_SUCCEEDED", "JOB_STATE_SUCCEEDED"):
            logger.info(f"Vertex AI job succeeded after {waited}s")
            return "SUCCEEDED"
        elif state in (
            "PIPELINE_STATE_FAILED", "JOB_STATE_FAILED",
            "PIPELINE_STATE_CANCELLED", "JOB_STATE_CANCELLED",
        ):
            error_msg = getattr(job, "error", None)
            logger.error(f"Vertex AI job {state}: {error_msg}")
            return "FAILED"

    raise TimeoutError(f"Vertex AI job timed out after {max_wait}s (state: {job.state})")
