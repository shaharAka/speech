#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────
# One-time GCP project setup for Whisper Hebrew training
# Usage: ./gcp/setup.sh
# ─────────────────────────────────────────────────────────────
set -euo pipefail

PROJECT_ID="whisper-489414"
REGION="us-central1"
ZONE="us-central1-a"
BUCKET_NAME="whisper-training-${PROJECT_ID}"
SA_NAME="whisper-trainer"
SA_EMAIL="${SA_NAME}@${PROJECT_ID}.iam.gserviceaccount.com"

echo "=== Setting up GCP project: ${PROJECT_ID} ==="

# Set defaults
gcloud config set project "${PROJECT_ID}"
gcloud config set compute/region "${REGION}"
gcloud config set compute/zone "${ZONE}"

# 1. Enable required APIs
echo ">>> Enabling APIs..."
gcloud services enable \
  compute.googleapis.com \
  storage.googleapis.com \
  iam.googleapis.com \
  artifactregistry.googleapis.com \
  --project="${PROJECT_ID}"

# 2. Create GCS bucket (multi-regional for training data + model artifacts)
echo ">>> Creating GCS bucket: ${BUCKET_NAME}"
if gsutil ls -b "gs://${BUCKET_NAME}" 2>/dev/null; then
  echo "    Bucket already exists"
else
  gsutil mb -p "${PROJECT_ID}" -l "${REGION}" "gs://${BUCKET_NAME}"
fi

# Create folder structure
gsutil cp /dev/null "gs://${BUCKET_NAME}/training-data/.keep" 2>/dev/null || true
gsutil cp /dev/null "gs://${BUCKET_NAME}/models/.keep" 2>/dev/null || true
gsutil cp /dev/null "gs://${BUCKET_NAME}/logs/.keep" 2>/dev/null || true

# 3. Create service account for training VMs
echo ">>> Creating service account: ${SA_NAME}"
if gcloud iam service-accounts describe "${SA_EMAIL}" --project="${PROJECT_ID}" 2>/dev/null; then
  echo "    Service account already exists"
else
  gcloud iam service-accounts create "${SA_NAME}" \
    --display-name="Whisper Training Worker" \
    --project="${PROJECT_ID}"
fi

# Grant permissions
echo ">>> Granting IAM roles..."
for ROLE in roles/storage.objectAdmin roles/compute.instanceAdmin.v1 roles/logging.logWriter; do
  gcloud projects add-iam-policy-binding "${PROJECT_ID}" \
    --member="serviceAccount:${SA_EMAIL}" \
    --role="${ROLE}" \
    --quiet 2>/dev/null
done

# 4. Create firewall rule for SSH (needed for gcloud compute ssh)
echo ">>> Configuring firewall..."
if gcloud compute firewall-rules describe allow-ssh --project="${PROJECT_ID}" 2>/dev/null; then
  echo "    SSH firewall rule already exists"
else
  gcloud compute firewall-rules create allow-ssh \
    --project="${PROJECT_ID}" \
    --direction=INGRESS \
    --priority=1000 \
    --network=default \
    --action=ALLOW \
    --rules=tcp:22 \
    --source-ranges=0.0.0.0/0
fi

echo ""
echo "=== Setup complete! ==="
echo "  Project:  ${PROJECT_ID}"
echo "  Region:   ${REGION}"
echo "  Bucket:   gs://${BUCKET_NAME}"
echo "  SA:       ${SA_EMAIL}"
echo ""
echo "Next: Run ./gcp/train_remote.sh to submit a training job"
