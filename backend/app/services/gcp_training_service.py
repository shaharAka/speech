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

    # Startup script — direct pip install (fast without TTS deps)
    startup_script = f"""#!/bin/bash
set -euo pipefail
exec > /var/log/whisper-training.log 2>&1

echo "=== Whisper Training VM Starting ==="
echo "Run ID: {run_id}"
nvidia-smi
echo "GPU ready!"

# Download code from GCS
mkdir -p /opt/whisper /tmp/training-data /tmp/output
gsutil -m cp -r gs://{settings.gcs_bucket}/code/* /opt/whisper/
cd /opt/whisper

# Download training data
gsutil -m cp -r {job_dir}/data/* /tmp/training-data/
gsutil cp {job_dir}/manifest.json /tmp/training-data/manifest.json
echo "Data downloaded"

# Install core training dependencies (~2 min, no heavy TTS)
pip install --quiet transformers peft "datasets<3.0" accelerate evaluate jiwer ctranslate2 aiosqlite pydantic pydantic-settings soundfile librosa 2>&1 | tail -10
pip uninstall -y torchcodec 2>/dev/null || true
echo "Pip done (core)"

# TTS data augmentation if tts_enabled
# Always generates fresh 500 samples using the best WER recordings as reference
if python3 -c "import json; m=json.load(open('/tmp/training-data/manifest.json')); exit(0 if m.get('config',{{}}).get('tts_enabled') else 1)" 2>/dev/null; then
    echo "TTS enabled — generating 500 samples using best-WER reference clips..."

    # Step 1: Set up isolated venv (avoids DL VM torch conflicts)
    python3 -m venv /tmp/tts-env
    /tmp/tts-env/bin/pip install --quiet torch torchaudio --index-url https://download.pytorch.org/whl/cu121 2>&1 | tail -3
    /tmp/tts-env/bin/pip install --quiet chatterbox-tts 2>&1 | tail -3
    echo "TTS venv ready"

    # Step 2: Extract TTS texts from manifest
    python3 -c "
import json
m = json.load(open('/tmp/training-data/manifest.json'))
texts = m.get('tts_texts', [])[:500]
json.dump(texts, open('/tmp/tts_texts.json', 'w', encoding='utf-8'), ensure_ascii=False)
print(f'TTS texts: {{len(texts)}}')
"

    # Step 3: Pick 5 reference clips — longest recordings for richest voice profile
    # (longer clips give TTS more speech characteristics to clone from)
    mkdir -p /tmp/tts-ref
    python3 -c "
import json, os, shutil
m = json.load(open('/tmp/training-data/manifest.json'))
samples = m['samples']
# Sort by file size descending — longest recordings have most voice data
samples = sorted(samples, key=lambda s: os.path.getsize(os.path.join('/tmp/training-data', s['audio_file'])), reverse=True)
for s in samples[:5]:
    src = os.path.join('/tmp/training-data', s['audio_file'])
    if os.path.exists(src):
        shutil.copy2(src, '/tmp/tts-ref/')
        sz = os.path.getsize(src) / 1024
        print(f'  {{s[\"audio_file\"]}}: {{sz:.0f}}KB')
print(f'Selected {{min(5, len(samples))}} longest clips as reference')
"

    # Step 4: Generate TTS samples
    mkdir -p /tmp/training-data/tts
    /tmp/tts-env/bin/python /opt/whisper/scripts/run_tts_generation.py \
        --reference-dir /tmp/tts-ref \
        --output-dir /tmp/training-data/tts \
        --texts-json /tmp/tts_texts.json 2>&1 | tail -20

    # Step 5: Verify and clean up
    TTS_COUNT=$(ls /tmp/training-data/tts/tts_*.wav 2>/dev/null | wc -l)
    echo "TTS generated: $TTS_COUNT samples"
    if [ "$TTS_COUNT" -lt 10 ]; then
        echo "TTS FAILED — only $TTS_COUNT samples. Aborting."
        exit 1
    fi

    rm -rf /tmp/tts-env /tmp/tts-ref
    echo "TTS venv cleaned up — GPU memory freed for training"
fi

export PYTHONPATH=/opt/whisper:${{PYTHONPATH:-}}
export HF_AUDIO_DECODER_BACKEND=soundfile

echo "=== Starting Training ==="
python3 -c "
import json, os, sys, logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(name)s: %(message)s')
logger = logging.getLogger('train')

with open('/tmp/training-data/manifest.json') as f:
    manifest = json.load(f)

config = manifest['config']
samples = manifest['samples']
logger.info(f'Samples: {{len(samples)}}')

# === Phase 0: Load Pre-Generated TTS Data ===
if config.get('tts_enabled', False):
    tts_dir = '/tmp/training-data/tts'
    tts_manifest_path = os.path.join(tts_dir, 'tts_manifest.json')
    if os.path.exists(tts_manifest_path):
        with open(tts_manifest_path, encoding='utf-8') as tf:
            tts_data = json.load(tf)
        tts_samples = tts_data.get('samples', [])
        # Update paths to point to tts subdirectory
        for s in tts_samples:
            s['audio_file'] = os.path.join('tts', s['audio_file'])
        samples.extend(tts_samples)
        logger.info(f'Loaded {{len(tts_samples)}} pre-generated TTS samples (total: {{len(samples)}})')
    else:
        logger.error('TTS enabled but no pre-generated data found at ' + tts_manifest_path)
        sys.exit(1)

# === Phase 1: Build Dataset ===
# CRITICAL: Split REAL data first, then augment only the training portion.
# This prevents synthetic (TTS) samples from leaking into the eval set.
real_samples = [s for s in samples if s.get('source', 'real') == 'real']
synth_samples = [s for s in samples if s.get('source') == 'synthetic']

from datasets import Dataset, Audio

# Step 1a: Build real-only dataset and split
real_audio = [os.path.join('/tmp/training-data', s['audio_file']) for s in real_samples]
real_sents = [s['sentence'] for s in real_samples]
real_existing = [(a, s) for a, s in zip(real_audio, real_sents) if os.path.exists(a)]
logger.info(f'Real samples: {{len(real_existing)}}')

if len(real_existing) < 10:
    logger.error(f'Not enough real samples: {{len(real_existing)}}')
    sys.exit(1)

r_audio, r_sents = zip(*real_existing)
real_ds = Dataset.from_dict({{'audio': list(r_audio), 'sentence': list(r_sents)}})
real_ds = real_ds.cast_column('audio', Audio(sampling_rate=16000))
real_split = real_ds.train_test_split(test_size=0.1, seed=42)
logger.info(f'Real split: {{len(real_split[\"train\"])}} train / {{len(real_split[\"test\"])}} eval (REAL ONLY)')

# Step 1b: Build training set = oversampled real train + synthetic
weight = config.get('real_sample_weight', 8)
if synth_samples:
    synth_audio = [os.path.join('/tmp/training-data', s['audio_file']) for s in synth_samples]
    synth_sents = [s['sentence'] for s in synth_samples]
    synth_existing = [(a, s) for a, s in zip(synth_audio, synth_sents) if os.path.exists(a)]
    logger.info(f'Synthetic samples: {{len(synth_existing)}}')
    s_audio, s_sents = zip(*synth_existing)
    synth_ds = Dataset.from_dict({{'audio': list(s_audio), 'sentence': list(s_sents)}})
    synth_ds = synth_ds.cast_column('audio', Audio(sampling_rate=16000))

    # Oversample real train data to balance with synthetic
    from datasets import concatenate_datasets
    real_train_repeated = concatenate_datasets([real_split['train']] * int(weight))
    train_ds = concatenate_datasets([real_train_repeated, synth_ds])
    logger.info(f'Training set: {{len(real_split[\"train\"])}}x{{weight}} real + {{len(synth_ds)}} synth = {{len(train_ds)}} total')
else:
    train_ds = real_split['train']
    logger.info(f'Training set: {{len(train_ds)}} real (no TTS)')

# Eval is ALWAYS real-only — no synthetic contamination
eval_ds = real_split['test']

# === Phase 2: Training ===
from training.trainer import run_training
output_dir = '/tmp/output'
os.makedirs(output_dir, exist_ok=True)
os.chmod(output_dir, 0o777)

result = run_training(
    train_dataset=train_ds,
    eval_dataset=eval_ds,
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
