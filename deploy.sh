#!/usr/bin/env bash
# Deploy the Meeting Transcriber to Cloud Run.
#
# Usage:
#   PROJECT_ID=my-project GCS_BUCKET=my-bucket ./deploy.sh
#
# Optional overrides (environment variables):
#   REGION          Cloud Run region and Chirp 3 region (default: asia-southeast1)
#   GEMINI_LOCATION Vertex AI region for Gemini (default: asia-southeast1)
#   SERVICE_NAME    Cloud Run service name (default: transcribe)
#
# --- One-time prerequisites ---
#
# 1. Enable required APIs:
#    gcloud services enable \
#      speech.googleapis.com \
#      storage.googleapis.com \
#      aiplatform.googleapis.com \
#      run.googleapis.com \
#      cloudbuild.googleapis.com
#
# 2. Create the GCS bucket for audio uploads:
#    gcloud storage buckets create gs://$GCS_BUCKET --location=$REGION
#
# 3. Grant the Cloud Run service account the required roles.
#    Cloud Run uses the Compute Engine default SA unless you configure a custom one.
#    SA="${PROJECT_ID}-compute@developer.gserviceaccount.com"
#
#    # Read/write audio files in GCS
#    gcloud projects add-iam-policy-binding $PROJECT_ID \
#      --member="serviceAccount:$SA" --role="roles/storage.objectAdmin"
#
#    # Submit and poll Chirp 3 BatchRecognize jobs
#    gcloud projects add-iam-policy-binding $PROJECT_ID \
#      --member="serviceAccount:$SA" --role="roles/speech.editor"
#
#    # Call Gemini via Vertex AI
#    gcloud projects add-iam-policy-binding $PROJECT_ID \
#      --member="serviceAccount:$SA" --role="roles/aiplatform.user"
#
# --- Chirp 3 region availability ---
# Chirp 3 is only available in select regions (as of 2025):
#   asia-southeast1, europe-west2, europe-west3, northamerica-northeast1
# Verify current availability:
#   https://cloud.google.com/speech-to-text/v2/docs/locations

set -euo pipefail

PROJECT_ID="${PROJECT_ID:?Set PROJECT_ID}"
GCS_BUCKET="${GCS_BUCKET:?Set GCS_BUCKET}"
REGION="${REGION:-asia-southeast1}"
GEMINI_LOCATION="${GEMINI_LOCATION:-asia-southeast1}"
SERVICE_NAME="${SERVICE_NAME:-transcribe}"

echo "Deploying '$SERVICE_NAME' to Cloud Run in $REGION..."

gcloud run deploy "$SERVICE_NAME" \
  --source . \
  --project "$PROJECT_ID" \
  --region "$REGION" \
  --allow-unauthenticated \
  --set-env-vars "PROJECT_ID=${PROJECT_ID},GCS_BUCKET=${GCS_BUCKET},LOCATION=${REGION},GEMINI_LOCATION=${GEMINI_LOCATION}"

echo ""
echo "Deployed! Service URL:"
gcloud run services describe "$SERVICE_NAME" \
  --project "$PROJECT_ID" \
  --region "$REGION" \
  --format="value(status.url)"
