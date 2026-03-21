#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────
# Submit a training job to GCP GPU VM
#
# Usage: ./gcp/train_remote.sh [--run-id ID] [--epochs N] [--lr RATE] [--lora-rank R]
#
# What this does:
#   1. Exports training data from local SQLite DB
#   2. Uploads data to GCS
#   3. Creates a spot GPU VM
#   4. VM runs training, uploads results to GCS
#   5. Downloads the trained model locally
#   6. Deletes the VM
# ─────────────────────────────────────────────────────────────
set -euo pipefail

# ── Configuration ──
PROJECT_ID="whisper-489414"
REGION="us-central1"
ZONE="us-central1-a"
BUCKET_NAME="whisper-training-${PROJECT_ID}"
SA_EMAIL="whisper-trainer@${PROJECT_ID}.iam.gserviceaccount.com"

# VM Configuration
MACHINE_TYPE="g2-standard-8"          # 8 vCPU, 32GB RAM, 1x L4 GPU (24GB)
GPU_TYPE="nvidia-l4"
GPU_COUNT=1
BOOT_DISK_SIZE="100GB"
VM_IMAGE_FAMILY="ubuntu-2204-lts"     # Ubuntu + NVIDIA driver auto-install
VM_IMAGE_PROJECT="ubuntu-os-cloud"

# Training defaults
RUN_ID=""
NUM_EPOCHS=5
LEARNING_RATE="1e-4"
LORA_RANK=32

# Project paths
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "${SCRIPT_DIR}")"
BACKEND_DIR="${PROJECT_DIR}/backend"
DB_PATH="${BACKEND_DIR}/storage/whisper.db"

# ── Parse arguments ──
while [[ $# -gt 0 ]]; do
    case $1 in
        --run-id) RUN_ID="$2"; shift 2 ;;
        --epochs) NUM_EPOCHS="$2"; shift 2 ;;
        --lr) LEARNING_RATE="$2"; shift 2 ;;
        --lora-rank) LORA_RANK="$2"; shift 2 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

# Generate run ID if not provided
if [ -z "${RUN_ID}" ]; then
    RUN_ID="run_$(date +%Y%m%d_%H%M%S)"
fi

VM_NAME="whisper-train-${RUN_ID//[_.]/-}"
JOB_DIR="gs://${BUCKET_NAME}/training-jobs/${RUN_ID}"

echo "═══════════════════════════════════════════════════"
echo "  Whisper Hebrew Training - GCP Job Submission"
echo "═══════════════════════════════════════════════════"
echo "  Run ID:      ${RUN_ID}"
echo "  VM:          ${VM_NAME} (${MACHINE_TYPE} + ${GPU_TYPE})"
echo "  Epochs:      ${NUM_EPOCHS}"
echo "  LR:          ${LEARNING_RATE}"
echo "  LoRA rank:   ${LORA_RANK}"
echo "  GCS:         ${JOB_DIR}"
echo "═══════════════════════════════════════════════════"
echo ""

# ── Step 1: Export training data ──
echo ">>> Step 1: Exporting training data from local DB..."

EXPORT_DIR=$(mktemp -d)
trap "rm -rf ${EXPORT_DIR}" EXIT

python3 "${BACKEND_DIR}/gcp_export_data.py" \
    --db "${DB_PATH}" \
    --audio-root "${BACKEND_DIR}/storage/audio" \
    --output-dir "${EXPORT_DIR}" \
    --config "{\"num_epochs\": ${NUM_EPOCHS}, \"learning_rate\": ${LEARNING_RATE}, \"lora_rank\": ${LORA_RANK}}"

SAMPLE_COUNT=$(python3 -c "import json; print(len(json.load(open('${EXPORT_DIR}/manifest.json'))['samples']))")
echo "    Exported ${SAMPLE_COUNT} samples"

if [ "${SAMPLE_COUNT}" -lt 10 ]; then
    echo "ERROR: Need at least 10 recordings with transcriptions. Have ${SAMPLE_COUNT}."
    exit 1
fi

# ── Step 2: Upload to GCS ──
echo ">>> Step 2: Uploading training data to GCS..."
gsutil -m cp -r "${EXPORT_DIR}/"* "${JOB_DIR}/data/"
gsutil cp "${EXPORT_DIR}/manifest.json" "${JOB_DIR}/manifest.json"
echo "    Uploaded to ${JOB_DIR}/"

# ── Step 3: Create GPU VM ──
echo ">>> Step 3: Creating GPU VM: ${VM_NAME}..."

gcloud compute instances create "${VM_NAME}" \
    --project="${PROJECT_ID}" \
    --zone="${ZONE}" \
    --machine-type="${MACHINE_TYPE}" \
    --accelerator="type=${GPU_TYPE},count=${GPU_COUNT}" \
    --image-family="${VM_IMAGE_FAMILY}" \
    --image-project="${VM_IMAGE_PROJECT}" \
    --boot-disk-size="${BOOT_DISK_SIZE}" \
    --boot-disk-type=pd-ssd \
    --provisioning-model=SPOT \
    --instance-termination-action=STOP \
    --service-account="${SA_EMAIL}" \
    --scopes=cloud-platform \
    --metadata="install-nvidia-driver=True" \
    --maintenance-policy=TERMINATE \
    --no-restart-on-failure

echo "    VM created. Waiting for it to be ready..."
sleep 30

# Wait for SSH to be available
for i in $(seq 1 20); do
    if gcloud compute ssh "${VM_NAME}" --zone="${ZONE}" --command="echo ready" 2>/dev/null; then
        break
    fi
    echo "    Waiting for SSH... (attempt $i/20)"
    sleep 15
done

# ── Step 4: Install deps and run training on VM ──
echo ">>> Step 4: Setting up training environment on VM..."

# Upload code to VM
gcloud compute scp --recurse \
    "${BACKEND_DIR}/app" \
    "${BACKEND_DIR}/training" \
    "${BACKEND_DIR}/pyproject.toml" \
    "${VM_NAME}:/tmp/whisper-code/" \
    --zone="${ZONE}"

# Run training script on VM
gcloud compute ssh "${VM_NAME}" --zone="${ZONE}" --command="bash -s" <<'REMOTE_SCRIPT'
set -euo pipefail

echo "=== VM Setup ==="
echo "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo 'initializing...')"

# Install system deps
sudo apt-get update -qq && sudo apt-get install -y -qq ffmpeg > /dev/null 2>&1

# Set up Python environment
cd /tmp/whisper-code
pip install --quiet --no-cache-dir -e ".[training]" 2>&1 | tail -5
pip install --quiet --no-cache-dir ctranslate2 2>&1 | tail -3
pip install --quiet --no-cache-dir aiosqlite 2>&1 | tail -1

echo "=== Downloading training data from GCS ==="
REMOTE_SCRIPT

# Pass env vars and run the actual training
gcloud compute ssh "${VM_NAME}" --zone="${ZONE}" --command="
export GCS_BUCKET='${BUCKET_NAME}'
export TRAINING_RUN_ID='${RUN_ID}'
export JOB_DIR='${JOB_DIR}'

cd /tmp/whisper-code

# Download training data
mkdir -p /tmp/training-data
gsutil -m cp -r '${JOB_DIR}/data/*' /tmp/training-data/
gsutil cp '${JOB_DIR}/manifest.json' /tmp/training-data/manifest.json

echo '=== Starting Training ==='
nvidia-smi

python3 -c \"
import json, os, sys, logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(name)s: %(message)s')
logger = logging.getLogger('train')

# Load manifest
with open('/tmp/training-data/manifest.json') as f:
    manifest = json.load(f)

config = manifest['config']
samples = manifest['samples']
logger.info(f'Config: {json.dumps(config, indent=2)}')
logger.info(f'Samples: {len(samples)}')

# Build HF Dataset
from datasets import Dataset, Audio

audio_paths = [os.path.join('/tmp/training-data', s['audio_file']) for s in samples]
sentences = [s['sentence'] for s in samples]

existing = [(a, s) for a, s in zip(audio_paths, sentences) if os.path.exists(a)]
logger.info(f'Valid: {len(existing)}/{len(audio_paths)}')

if len(existing) < 10:
    sys.exit(1)

audio_paths, sentences = zip(*existing)
dataset = Dataset.from_dict({'audio': list(audio_paths), 'sentence': list(sentences)})
dataset = dataset.cast_column('audio', Audio(sampling_rate=16000))

split = dataset.train_test_split(test_size=0.1, seed=42)
logger.info(f'Train: {len(split[\\\"train\\\"])}, Eval: {len(split[\\\"test\\\"])}')

from training.trainer import run_training

output_dir = '/tmp/output'
os.makedirs(output_dir, exist_ok=True)

result = run_training(
    train_dataset=split['train'],
    eval_dataset=split['test'],
    output_dir=output_dir,
    config={
        'num_train_epochs': config.get('num_epochs', 5),
        'lora_r': config.get('lora_rank', 32),
        'learning_rate': config.get('learning_rate', 1e-4),
    },
)

logger.info(f'Result: {result}')

from training.export import merge_and_export
ct2_path = merge_and_export(result['adapter_path'], os.path.join(output_dir, 'final'))

results = {
    'status': 'completed',
    'train_loss': result.get('train_loss'),
    'eval_wer': result.get('eval_wer'),
    'ct2_path': ct2_path,
}
with open('/tmp/output/results.json', 'w') as f:
    json.dump(results, f, indent=2, default=str)

logger.info('Done!')
\"

echo '=== Uploading results to GCS ==='
gsutil -m cp -r /tmp/output/final/ '${JOB_DIR}/model/'
gsutil cp /tmp/output/results.json '${JOB_DIR}/results.json'
[ -d /tmp/output/adapter ] && gsutil -m cp -r /tmp/output/adapter/ '${JOB_DIR}/adapter/'
echo 'Upload complete!'
"

TRAIN_EXIT=$?

# ── Step 5: Download results ──
if [ ${TRAIN_EXIT} -eq 0 ]; then
    echo ">>> Step 5: Downloading trained model..."
    MODEL_DIR="${BACKEND_DIR}/storage/models/${RUN_ID}"
    mkdir -p "${MODEL_DIR}"

    gsutil -m cp -r "${JOB_DIR}/model/ct2/*" "${MODEL_DIR}/" 2>/dev/null || true
    gsutil cp "${JOB_DIR}/results.json" "${MODEL_DIR}/results.json" 2>/dev/null || true

    echo "    Model downloaded to: ${MODEL_DIR}"
    echo ""
    echo "    Results:"
    cat "${MODEL_DIR}/results.json" 2>/dev/null || echo "    (results.json not found)"
else
    echo ">>> Training failed on VM. Check logs:"
    echo "    gcloud compute ssh ${VM_NAME} --zone=${ZONE} --command='cat /tmp/output/*.log'"
fi

# ── Step 6: Delete VM ──
echo ""
echo ">>> Step 6: Cleaning up VM..."
gcloud compute instances delete "${VM_NAME}" \
    --zone="${ZONE}" \
    --project="${PROJECT_ID}" \
    --quiet

echo ""
echo "═══════════════════════════════════════════════════"
echo "  Training job complete!"
echo "  Run ID:  ${RUN_ID}"
echo "  Model:   ${BACKEND_DIR}/storage/models/${RUN_ID}/"
echo "  GCS:     ${JOB_DIR}/"
echo "═══════════════════════════════════════════════════"
