"""GCP-based training job management.

Handles:
  - Exporting training data to GCS
  - Creating GPU VMs for training
  - Polling job status
  - Downloading trained models
"""

import json
import logging
import os
import shutil
import sqlite3
import subprocess
import tempfile
from datetime import datetime
from pathlib import Path

from app.config import settings

logger = logging.getLogger(__name__)


def _active_zone() -> str:
    """Return the zone where the training VM was actually created (may differ from config if fallback was used)."""
    return getattr(settings, "_active_training_zone", settings.gcp_zone)

# Sync DB URL for direct sqlite3 access
_sync_db_path = settings.database_url.replace("sqlite+aiosqlite:///", "").replace("sqlite:///", "")


def _run_cmd(cmd: list[str], check: bool = True, timeout: int = 300) -> subprocess.CompletedProcess:
    """Run a shell command and return result."""
    logger.info(f"Running: {' '.join(cmd[:5])}...")
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    if check and result.returncode != 0:
        logger.error(f"Command failed: {result.stderr}")
        raise RuntimeError(f"Command failed: {result.stderr[:500]}")
    return result


def _gsutil(args: list[str], **kwargs) -> subprocess.CompletedProcess:
    """Run gsutil command."""
    return _run_cmd(["gsutil"] + args, **kwargs)


def _gcloud(args: list[str], **kwargs) -> subprocess.CompletedProcess:
    """Run gcloud command."""
    return _run_cmd(
        ["gcloud", "--project", settings.gcp_project_id, "--quiet"] + args,
        **kwargs,
    )


def export_training_data(run_id: int, config: dict) -> tuple[str, int]:
    """
    Export training data from local DB to a temp directory.

    Returns:
        Tuple of (export_dir_path, sample_count)
    """
    export_dir = tempfile.mkdtemp(prefix=f"whisper_train_{run_id}_")

    db_path = os.path.abspath(_sync_db_path)
    audio_root = os.path.abspath(settings.audio_storage_path)

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row

    query = """
        SELECT r.id, r.audio_path, t.content as sentence
        FROM recordings r
        JOIN texts t ON r.text_id = t.id
        JOIN transcriptions tr ON tr.recording_id = r.id
        WHERE r.audio_path IS NOT NULL
    """
    rows = conn.execute(query).fetchall()
    conn.close()

    samples = []
    for row in rows:
        src_audio = os.path.join(audio_root, row["audio_path"])
        if not os.path.exists(src_audio):
            logger.warning(f"Audio file not found: {src_audio}")
            continue

        audio_filename = f"sample_{row['id']}.wav"
        shutil.copy2(src_audio, os.path.join(export_dir, audio_filename))
        samples.append({
            "id": row["id"],
            "audio_file": audio_filename,
            "sentence": row["sentence"],
        })

    manifest = {
        "config": config,
        "num_samples": len(samples),
        "run_id": run_id,
        "samples": samples,
    }

    # Generate TTS texts if augmentation is enabled
    if config.get("tts_enabled", False):
        try:
            from training.tts_text_generator import generate_tts_texts
            num_synth = config.get("tts_num_synthetic", 500)
            logger.info(f"Generating {num_synth} TTS texts via Gemini...")
            tts_texts = generate_tts_texts(
                num_texts=num_synth,
                gemini_api_key=settings.gemini_api_key,
            )
            manifest["tts_texts"] = tts_texts
            logger.info(f"Added {len(tts_texts)} TTS texts to manifest")
        except Exception as e:
            logger.error(f"TTS text generation failed: {e}")
            manifest["tts_texts"] = []

    with open(os.path.join(export_dir, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    return export_dir, len(samples)


def upload_training_data(export_dir: str, run_id: int) -> str:
    """Upload training data to GCS. Returns the GCS job directory."""
    job_dir = f"gs://{settings.gcs_bucket}/training-jobs/run_{run_id}"

    _gsutil(["-m", "cp", "-r", f"{export_dir}/*", f"{job_dir}/data/"], timeout=600)
    _gsutil(["cp", f"{export_dir}/manifest.json", f"{job_dir}/manifest.json"])

    logger.info(f"Uploaded training data to {job_dir}")
    return job_dir


def create_training_vm(run_id: int) -> str:
    """Create a GPU VM for training. Returns the VM name."""
    vm_name = f"whisper-train-run-{run_id}"
    sa_email = f"whisper-trainer@{settings.gcp_project_id}.iam.gserviceaccount.com"
    job_dir = f"gs://{settings.gcs_bucket}/training-jobs/run_{run_id}"

    training_image = f"us-central1-docker.pkg.dev/{settings.gcp_project_id}/whisper-training/trainer:latest"

    # Startup script — uses pre-built Docker image (no pip install needed)
    startup_script = f"""#!/bin/bash
set -euo pipefail
exec > /var/log/whisper-training.log 2>&1

echo "=== Whisper Training VM Starting ==="
echo "Run ID: {run_id}"
nvidia-smi
echo "GPU ready!"

# Download code and data from GCS
mkdir -p /opt/whisper /tmp/training-data /tmp/output
gsutil -m cp -r gs://{settings.gcs_bucket}/code/* /opt/whisper/
gsutil -m cp -r {job_dir}/data/* /tmp/training-data/
gsutil cp {job_dir}/manifest.json /tmp/training-data/manifest.json
echo "Data downloaded"

# Auth Docker for Artifact Registry
gcloud auth configure-docker us-central1-docker.pkg.dev --quiet
docker pull {training_image}
echo "Docker image ready"

echo "=== Starting Training ==="
# Run training inside pre-built container (all deps pre-installed, zero pip time)
docker run --rm --gpus all \
    -v /opt/whisper:/app:ro \
    -v /tmp/training-data:/tmp/training-data \
    -v /tmp/output:/tmp/output \
    -e PYTHONPATH=/app \
    -e HF_AUDIO_DECODER_BACKEND=soundfile \
    {training_image} \
    python3 -c "

python3 -c "
import json, os, sys, logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(name)s: %(message)s')
logger = logging.getLogger('train')

with open('/tmp/training-data/manifest.json') as f:
    manifest = json.load(f)

config = manifest['config']
samples = manifest['samples']
logger.info(f'Samples: {{len(samples)}}')

# === Phase 0: TTS Data Augmentation ===
tts_texts = manifest.get('tts_texts', [])
if tts_texts and config.get('tts_enabled', False):
    logger.info(f'Starting TTS augmentation with {{len(tts_texts)}} texts...')
    try:
        from training.tts_augmentation import run_tts_augmentation
        manifest = run_tts_augmentation(manifest, '/tmp/training-data', tts_texts, config)
        samples = manifest['samples']
        logger.info(f'After TTS: {{len(samples)}} total samples')
    except Exception as e:
        logger.error(f'TTS augmentation failed: {{e}}')
        import traceback; traceback.print_exc()
        # Continue with real data only

# === Phase 1: Build Dataset ===
# Tag real samples and apply oversampling for balance
real_samples = [s for s in samples if s.get('source', 'real') == 'real']
synth_samples = [s for s in samples if s.get('source') == 'synthetic']
weight = config.get('real_sample_weight', 8)
if synth_samples and weight > 1:
    balanced = real_samples * int(weight) + synth_samples
    logger.info(f'Balanced: {{len(real_samples)}}x{{int(weight)}} real + {{len(synth_samples)}} synth = {{len(balanced)}}')
else:
    balanced = samples

from datasets import Dataset, Audio
audio_paths = [os.path.join('/tmp/training-data', s['audio_file']) for s in balanced]
sentences = [s['sentence'] for s in balanced]
existing = [(a, s) for a, s in zip(audio_paths, sentences) if os.path.exists(a)]
logger.info(f'Valid: {{len(existing)}}/{{len(audio_paths)}}')

if len(existing) < 10:
    sys.exit(1)

audio_paths, sentences = zip(*existing)
dataset = Dataset.from_dict({{'audio': list(audio_paths), 'sentence': list(sentences)}})
dataset = dataset.cast_column('audio', Audio(sampling_rate=16000))
split = dataset.train_test_split(test_size=0.1, seed=42)

# === Phase 2: Training ===
from training.trainer import run_training
output_dir = '/tmp/output'
os.makedirs(output_dir, exist_ok=True)
os.chmod(output_dir, 0o777)

result = run_training(
    train_dataset=split['train'],
    eval_dataset=split['test'],
    output_dir=output_dir,
    config={{
        'num_train_epochs': config.get('num_epochs', 5),
        'lora_r': config.get('lora_rank', 32),
        'learning_rate': config.get('learning_rate', 1e-4),
        'lora_encoder_layers': config.get('lora_encoder_layers', [13, 14, 15]),
        'spec_augment_freq_mask': config.get('spec_augment_freq_mask', 15),
        'spec_augment_time_mask': config.get('spec_augment_time_mask', 100),
    }},
)

from training.export import merge_and_export
ct2_path = merge_and_export(result['adapter_path'], os.path.join(output_dir, 'final'))

results = {{'status': 'completed', 'train_loss': result.get('train_loss'), 'eval_wer': result.get('eval_wer'), 'ct2_path': ct2_path, 'epoch_metrics': result.get('epoch_metrics', [])}}
with open('/tmp/output/results.json', 'w') as f:
    json.dump(results, f, indent=2, default=str)
logger.info('Done!')
"

if [ $? -eq 0 ]; then
    echo "=== Uploading results ==="
    gsutil -m cp -r /tmp/output/final/ {job_dir}/model/
    gsutil cp /tmp/output/results.json {job_dir}/results.json
    [ -d /tmp/output/adapter ] && gsutil -m cp -r /tmp/output/adapter/ {job_dir}/adapter/
    echo "TRAINING_STATUS:COMPLETED"
else
    echo '{{"status": "failed", "error": "Training script failed"}}' > /tmp/output/results.json
    gsutil cp /tmp/output/results.json {job_dir}/results.json
    echo "TRAINING_STATUS:FAILED"
fi
"""

    # Write startup script to a temp file (avoids shell escaping issues)
    import tempfile
    startup_file = tempfile.NamedTemporaryFile(mode="w", suffix=".sh", delete=False)
    startup_file.write(startup_script)
    startup_file.close()

    # Build gcloud create command
    create_args = [
        "compute", "instances", "create", vm_name,
        "--zone", settings.gcp_zone,
        "--machine-type", settings.gcp_machine_type,
        f"--accelerator=type={settings.gcp_gpu_type},count=1",
        "--image-family", "pytorch-2-7-cu128-ubuntu-2204-nvidia-570",
        "--image-project", "deeplearning-platform-release",
        "--boot-disk-size", "200GB",  # extra space for Docker image + model weights
        "--boot-disk-type", "pd-ssd",
        "--service-account", sa_email,
        "--scopes", "cloud-platform",
        f"--metadata-from-file=startup-script={startup_file.name}",
        "--metadata", "install-nvidia-driver=True",
        "--maintenance-policy", "TERMINATE",
        "--no-restart-on-failure",
    ]

    if settings.gcp_use_spot:
        create_args.extend(["--provisioning-model", "SPOT", "--instance-termination-action", "STOP"])

    # Try multiple zones if the primary one is exhausted
    fallback_zones = [
        settings.gcp_zone,
        "us-central1-b", "us-central1-c",
        "us-east4-a", "us-east4-b", "us-east4-c",
        "us-east1-b", "us-east1-c", "us-east1-d",
        "us-west1-a", "us-west1-b",
        "us-west4-a", "us-west4-b",
        "europe-west4-a", "europe-west4-b",
    ]
    # Deduplicate while preserving order
    seen = set()
    zones_to_try = []
    for z in fallback_zones:
        if z not in seen:
            seen.add(z)
            zones_to_try.append(z)

    last_error = None
    try:
        for zone in zones_to_try:
            # Swap the zone in the args
            zone_idx = create_args.index("--zone") + 1
            create_args[zone_idx] = zone

            try:
                logger.info(f"Trying to create VM in zone {zone}...")
                _gcloud(create_args, timeout=120)
                logger.info(f"Created VM: {vm_name} in zone {zone}")
                # Store the actual zone used so polling/deletion works
                settings.__dict__["_active_training_zone"] = zone
                return vm_name
            except RuntimeError as e:
                err_str = str(e)
                retryable = (
                    "ZONE_RESOURCE_POOL_EXHAUSTED" in err_str
                    or "does not have enough resources" in err_str
                    or "does not exist in zone" in err_str
                    or "Machine type with name" in err_str
                    or "stockout" in err_str
                )
                if retryable:
                    logger.warning(f"Zone {zone} unavailable, trying next...")
                    last_error = e
                    continue
                else:
                    raise  # Non-retryable error — don't retry

        raise RuntimeError(
            f"No GPU capacity in any zone. Last error: {last_error}"
        )
    finally:
        os.unlink(startup_file.name)


def upload_code_to_gcs():
    """Upload training code to GCS so VMs can access it."""
    project_dir = Path(__file__).parent.parent.parent  # backend/
    code_gcs = f"gs://{settings.gcs_bucket}/code"

    # Upload app/ and training/ directories + pyproject.toml
    with tempfile.TemporaryDirectory() as tmp:
        # Copy just the needed code
        shutil.copytree(project_dir / "app", os.path.join(tmp, "app"))
        shutil.copytree(project_dir / "training", os.path.join(tmp, "training"))
        shutil.copy2(project_dir / "pyproject.toml", os.path.join(tmp, "pyproject.toml"))

        _gsutil(["-m", "rsync", "-r", tmp, code_gcs], timeout=120)

    logger.info(f"Uploaded code to {code_gcs}")


def check_vm_status(vm_name: str) -> str:
    """Check if a VM is running/stopped/terminated."""
    result = _gcloud(
        ["compute", "instances", "describe", vm_name,
         "--zone", _active_zone(),
         "--format", "value(status)"],
        check=False,
    )
    if result.returncode != 0:
        return "NOT_FOUND"
    return result.stdout.strip()


def check_training_results(run_id: int) -> dict | None:
    """Check if training results are available on GCS."""
    job_dir = f"gs://{settings.gcs_bucket}/training-jobs/run_{run_id}"
    result = _gsutil(
        ["cat", f"{job_dir}/results.json"],
        check=False, timeout=30,
    )
    if result.returncode != 0:
        return None
    try:
        return json.loads(result.stdout)
    except json.JSONDecodeError:
        return None


def download_model(run_id: int) -> str:
    """Download trained model from GCS to local storage."""
    job_dir = f"gs://{settings.gcs_bucket}/training-jobs/run_{run_id}"
    local_dir = os.path.join(os.path.abspath(settings.model_storage_path), f"run_{run_id}")
    os.makedirs(local_dir, exist_ok=True)

    _gsutil(
        ["-m", "cp", "-r", f"{job_dir}/model/final/ct2/*", f"{local_dir}/"],
        timeout=600,
    )

    # Ensure preprocessor_config.json exists (needed for correct mel feature count)
    preprocessor_path = os.path.join(local_dir, "preprocessor_config.json")
    if not os.path.exists(preprocessor_path):
        try:
            from huggingface_hub import hf_hub_download
            import shutil
            src = hf_hub_download("ivrit-ai/whisper-large-v3-turbo-ct2", "preprocessor_config.json")
            shutil.copy2(src, preprocessor_path)
            logger.info("Copied preprocessor_config.json from base model")
        except Exception as e:
            logger.warning(f"Could not copy preprocessor_config.json: {e}")

    logger.info(f"Downloaded model to {local_dir}")
    return local_dir


def delete_vm(vm_name: str):
    """Delete a training VM."""
    _gcloud(
        ["compute", "instances", "delete", vm_name,
         "--zone", _active_zone()],
        check=False, timeout=60,
    )
    logger.info(f"Deleted VM: {vm_name}")


def get_vm_logs(vm_name: str) -> str:
    """Get training logs from VM serial port."""
    result = _gcloud(
        ["compute", "instances", "get-serial-port-output", vm_name,
         "--zone", _active_zone()],
        check=False, timeout=30,
    )
    return result.stdout if result.returncode == 0 else "Logs not available"
