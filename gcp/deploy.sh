#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────
# Deploy Whisper Hebrew Trainer to GCP VM
#
# Creates a VM, installs Docker, deploys the app.
# Your daughter can access it at http://<VM_IP>
#
# Usage: ./gcp/deploy.sh
# ─────────────────────────────────────────────────────────────
set -euo pipefail

PROJECT_ID="whisper-489414"
ZONE="us-central1-a"
VM_NAME="whisper-app"
MACHINE_TYPE="e2-standard-2"  # 2 vCPU, 8GB RAM — good for Whisper CPU int8
SA_EMAIL="whisper-trainer@${PROJECT_ID}.iam.gserviceaccount.com"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "${SCRIPT_DIR}")"

echo "═══════════════════════════════════════════════════"
echo "  Deploying Whisper Hebrew Trainer to GCP"
echo "═══════════════════════════════════════════════════"
echo "  VM:       ${VM_NAME} (${MACHINE_TYPE})"
echo "  Zone:     ${ZONE}"
echo "  Project:  ${PROJECT_ID}"
echo "═══════════════════════════════════════════════════"
echo ""

# ── Step 1: Create firewall rule for HTTP ──
echo ">>> Step 1: Configuring firewall..."
if gcloud compute firewall-rules describe allow-http --project="${PROJECT_ID}" 2>/dev/null; then
  echo "    HTTP firewall rule already exists"
else
  gcloud compute firewall-rules create allow-http \
    --project="${PROJECT_ID}" \
    --direction=INGRESS \
    --priority=1000 \
    --network=default \
    --action=ALLOW \
    --rules=tcp:80 \
    --source-ranges=0.0.0.0/0 \
    --target-tags=http-server \
    --quiet
fi

# ── Step 2: Create VM (or reset if exists) ──
echo ">>> Step 2: Creating VM..."
EXISTING=$(gcloud compute instances describe "${VM_NAME}" --zone="${ZONE}" --project="${PROJECT_ID}" --format="value(status)" 2>/dev/null || echo "NONE")

if [ "${EXISTING}" = "NONE" ]; then
  gcloud compute instances create "${VM_NAME}" \
    --project="${PROJECT_ID}" \
    --zone="${ZONE}" \
    --machine-type="${MACHINE_TYPE}" \
    --image-family=ubuntu-2204-lts \
    --image-project=ubuntu-os-cloud \
    --boot-disk-size=50GB \
    --boot-disk-type=pd-balanced \
    --service-account="${SA_EMAIL}" \
    --scopes=cloud-platform \
    --tags=http-server \
    --quiet
  echo "    VM created. Waiting for SSH..."
  sleep 20
elif [ "${EXISTING}" = "TERMINATED" ] || [ "${EXISTING}" = "STOPPED" ]; then
  gcloud compute instances start "${VM_NAME}" --zone="${ZONE}" --project="${PROJECT_ID}" --quiet
  echo "    VM started. Waiting for SSH..."
  sleep 15
else
  echo "    VM already running."
fi

# Wait for SSH
for i in $(seq 1 15); do
  if gcloud compute ssh "${VM_NAME}" --zone="${ZONE}" --project="${PROJECT_ID}" --command="echo ok" 2>/dev/null; then
    break
  fi
  echo "    Waiting for SSH... (attempt $i/15)"
  sleep 10
done

# ── Step 3: Install Docker on VM ──
echo ">>> Step 3: Installing Docker on VM..."
gcloud compute ssh "${VM_NAME}" --zone="${ZONE}" --project="${PROJECT_ID}" --command="bash -s" <<'INSTALL_DOCKER'
set -e
if command -v docker &>/dev/null; then
  echo "Docker already installed: $(docker --version)"
else
  echo "Installing Docker..."
  curl -fsSL https://get.docker.com | sh
  sudo usermod -aG docker $USER
  echo "Docker installed."
fi

# Ensure docker compose plugin
if ! docker compose version 2>/dev/null; then
  sudo apt-get update -qq
  sudo apt-get install -y -qq docker-compose-plugin
fi

echo "Docker Compose: $(docker compose version)"
INSTALL_DOCKER

# ── Step 4: Upload project files ──
echo ">>> Step 4: Uploading project to VM..."

# Create a clean tarball (exclude .git, node_modules, .venv, storage)
cd "${PROJECT_DIR}"
tar czf /tmp/whisper-deploy.tar.gz \
  --exclude='.git' \
  --exclude='node_modules' \
  --exclude='.venv' \
  --exclude='backend/storage' \
  --exclude='frontend/.next' \
  --exclude='__pycache__' \
  --exclude='.DS_Store' \
  --exclude='._*' \
  .

gcloud compute scp /tmp/whisper-deploy.tar.gz "${VM_NAME}:~/whisper-deploy.tar.gz" \
  --zone="${ZONE}" --project="${PROJECT_ID}"

rm /tmp/whisper-deploy.tar.gz

# Extract on VM
gcloud compute ssh "${VM_NAME}" --zone="${ZONE}" --project="${PROJECT_ID}" --command="
  mkdir -p ~/whisper
  cd ~/whisper
  tar xzf ~/whisper-deploy.tar.gz
  rm ~/whisper-deploy.tar.gz
  echo 'Project extracted.'
  ls -la
"

# ── Step 5: Build and start with Docker Compose ──
echo ">>> Step 5: Building and starting the app..."
gcloud compute ssh "${VM_NAME}" --zone="${ZONE}" --project="${PROJECT_ID}" --command="
  cd ~/whisper
  sudo docker compose -f docker-compose.prod.yml build --no-cache
  sudo docker compose -f docker-compose.prod.yml up -d

  echo ''
  echo 'Waiting for services to start...'
  sleep 10
  sudo docker compose -f docker-compose.prod.yml ps
"

# ── Step 6: Get the external IP ──
EXTERNAL_IP=$(gcloud compute instances describe "${VM_NAME}" \
  --zone="${ZONE}" --project="${PROJECT_ID}" \
  --format="get(networkInterfaces[0].accessConfigs[0].natIP)")

echo ""
echo "═══════════════════════════════════════════════════"
echo "  ✅ Deployment complete!"
echo "═══════════════════════════════════════════════════"
echo ""
echo "  App URL:  http://${EXTERNAL_IP}"
echo ""
echo "  Note: First load takes ~2 minutes while the"
echo "  Whisper model downloads (~1.5GB)."
echo ""
echo "  To check logs:"
echo "    gcloud compute ssh ${VM_NAME} --zone=${ZONE} --command='cd ~/whisper && sudo docker compose -f docker-compose.prod.yml logs -f'"
echo ""
echo "  To redeploy after changes:"
echo "    ./gcp/deploy.sh"
echo ""
echo "  To stop (save costs):"
echo "    gcloud compute instances stop ${VM_NAME} --zone=${ZONE}"
echo ""
echo "  Estimated cost: ~\$50/month (on-demand)"
echo "═══════════════════════════════════════════════════"
