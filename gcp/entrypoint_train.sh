#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────
# GPU VM entrypoint: download data → train → upload results
# This runs on the GCP VM automatically
# ─────────────────────────────────────────────────────────────
set -euo pipefail

BUCKET="${GCS_BUCKET}"
RUN_ID="${TRAINING_RUN_ID}"
JOB_DIR="gs://${BUCKET}/training-jobs/${RUN_ID}"

echo "=== Whisper Hebrew Training - Run #${RUN_ID} ==="
echo "    Bucket: ${BUCKET}"
echo "    Job dir: ${JOB_DIR}"
echo "    GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'none detected')"
echo ""

# 1. Download training data from GCS
echo ">>> Downloading training data..."
mkdir -p /app/data
gsutil -m cp -r "${JOB_DIR}/data/*" /app/data/
echo "    Downloaded $(ls /app/data/*.wav 2>/dev/null | wc -l) audio files"

# 2. Download training manifest (JSON with audio_path → sentence mappings)
gsutil cp "${JOB_DIR}/manifest.json" /app/data/manifest.json

# 3. Run training
echo ">>> Starting LoRA fine-tuning..."
python3 -c "
import json
import os
import sys
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('train')

# Load manifest
with open('/app/data/manifest.json') as f:
    manifest = json.load(f)

config = manifest['config']
samples = manifest['samples']

logger.info(f'Training config: {json.dumps(config, indent=2)}')
logger.info(f'Number of samples: {len(samples)}')

# Build HF Dataset from manifest
from datasets import Dataset, Audio

audio_paths = [os.path.join('/app/data', s['audio_file']) for s in samples]
sentences = [s['sentence'] for s in samples]

# Validate files exist
existing = [(a, s) for a, s in zip(audio_paths, sentences) if os.path.exists(a)]
logger.info(f'Valid samples: {len(existing)} / {len(audio_paths)}')

if len(existing) < 10:
    logger.error('Not enough valid samples for training')
    sys.exit(1)

audio_paths, sentences = zip(*existing)

dataset = Dataset.from_dict({'audio': list(audio_paths), 'sentence': list(sentences)})
dataset = dataset.cast_column('audio', Audio(sampling_rate=16000))

# Split
split = dataset.train_test_split(test_size=0.1, seed=42)
train_ds = split['train']
eval_ds = split['test']

logger.info(f'Train: {len(train_ds)}, Eval: {len(eval_ds)}')

# Run training
from training.trainer import run_training

output_dir = '/app/output'
os.makedirs(output_dir, exist_ok=True)

result = run_training(
    train_dataset=train_ds,
    eval_dataset=eval_ds,
    output_dir=output_dir,
    config={
        'num_train_epochs': config.get('num_epochs', 5),
        'lora_r': config.get('lora_rank', 32),
        'learning_rate': config.get('learning_rate', 1e-4),
    },
)

logger.info(f'Training result: {json.dumps({k: str(v) for k, v in result.items()})}')

# Export: merge LoRA + convert to CT2
from training.export import merge_and_export

ct2_path = merge_and_export(result['adapter_path'], os.path.join(output_dir, 'final'))
logger.info(f'CT2 model exported to: {ct2_path}')

# Write results manifest
results = {
    'status': 'completed',
    'train_loss': result.get('train_loss'),
    'eval_wer': result.get('eval_wer'),
    'adapter_path': result.get('adapter_path'),
    'ct2_path': ct2_path,
}
with open('/app/output/results.json', 'w') as f:
    json.dump(results, f, indent=2)

logger.info('Training pipeline complete!')
"

TRAIN_EXIT=$?

if [ $TRAIN_EXIT -eq 0 ]; then
    echo ">>> Training succeeded! Uploading results..."

    # 4. Upload results to GCS
    gsutil -m cp -r /app/output/final/ "${JOB_DIR}/model/"
    gsutil cp /app/output/results.json "${JOB_DIR}/results.json"

    # Upload adapter separately (useful for future fine-tuning)
    if [ -d "/app/output/adapter" ]; then
        gsutil -m cp -r /app/output/adapter/ "${JOB_DIR}/adapter/"
    fi

    echo ">>> Results uploaded to ${JOB_DIR}/"
    echo "STATUS:COMPLETED"
else
    echo ">>> Training failed with exit code ${TRAIN_EXIT}"

    # Upload error log
    echo "{\"status\": \"failed\", \"error\": \"Training script exited with code ${TRAIN_EXIT}\"}" > /app/output/results.json
    gsutil cp /app/output/results.json "${JOB_DIR}/results.json"

    echo "STATUS:FAILED"
fi

# 5. Self-destruct signal (if running on auto-delete VM)
if [ "${AUTO_DELETE:-false}" = "true" ]; then
    echo ">>> VM will self-delete..."
    ZONE=$(curl -s -H "Metadata-Flavor: Google" http://metadata.google.internal/computeMetadata/v1/instance/zone | rev | cut -d'/' -f1 | rev)
    INSTANCE=$(curl -s -H "Metadata-Flavor: Google" http://metadata.google.internal/computeMetadata/v1/instance/name)
    gcloud compute instances delete "${INSTANCE}" --zone="${ZONE}" --quiet || true
fi
