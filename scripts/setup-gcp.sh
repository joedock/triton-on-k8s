#!/bin/bash
# setup-gcp.sh
# Configures GCP project, IAM, GKE cluster (no GPU node pool), GCS bucket,
# and Workload Identity binding for triton-on-k8s.
#
# Prerequisites:
#   - GCP project triton-prod-joedock exists
#   - gcloud authenticated to that project
#   - Billing account linked
#   - Compute Engine API enabled
#
# Run create-gpu-pool.sh after GPU quota is approved.

set -e  # Exit immediately if any command fails

PROJECT_ID="triton-prod-joedock"
ZONE="us-central1-a"
REGION="us-central1"
CLUSTER_NAME="triton-prod"
NAMESPACE="triton"
KSA_NAME="triton-ksa"
GSA_NAME="triton-gcs-reader"
GSA_EMAIL="${GSA_NAME}@${PROJECT_ID}.iam.gserviceaccount.com"
BUCKET_NAME="triton-models-joedock-prod"

echo "==> Enabling required APIs..."
gcloud services enable container.googleapis.com storage.googleapis.com iamcredentials.googleapis.com

echo "==> Creating GKE cluster with Workload Identity..."
if gcloud container clusters describe ${CLUSTER_NAME} --zone ${ZONE} >/dev/null 2>&1; then
  echo "==> Cluster ${CLUSTER_NAME} already exists, skipping creation."
else
  gcloud container clusters create ${CLUSTER_NAME} \
    --zone ${ZONE} \
    --num-nodes 1 \
    --machine-type e2-standard-4 \
    --workload-pool=${PROJECT_ID}.svc.id.goog \
    --no-enable-autoupgrade
fi

echo "==> Creating Google Service Account (if not exists)..."
gcloud iam service-accounts describe ${GSA_EMAIL} >/dev/null 2>&1 || \
gcloud iam service-accounts create ${GSA_NAME} \
    --display-name="Triton GCS Model Repository Reader"

echo "==> Creating GCS bucket (if not exists)..."
gsutil ls gs://${BUCKET_NAME} >/dev/null 2>&1 || \
gsutil mb -p ${PROJECT_ID} -l ${REGION} gs://${BUCKET_NAME}

echo "==> Granting least-privilege bucket access..."
gsutil iam ch serviceAccount:${GSA_EMAIL}:objectViewer gs://${BUCKET_NAME}
gsutil iam ch serviceAccount:${GSA_EMAIL}:legacyBucketReader gs://${BUCKET_NAME}

echo "==> Getting cluster credentials..."
gcloud container clusters get-credentials ${CLUSTER_NAME} --zone ${ZONE}

echo "==> Uploading model repository to GCS (if not already uploaded)..."
gsutil ls gs://${BUCKET_NAME}/model_repository/ >/dev/null 2>&1 || \
gsutil -m cp -r model_repository/ gs://${BUCKET_NAME}/

echo "==> Creating namespace (if not exists)..."
kubectl get namespace ${NAMESPACE} >/dev/null 2>&1 || \
kubectl create namespace ${NAMESPACE}

echo "==> Creating KSA (if not exists)..."
kubectl get serviceaccount ${KSA_NAME} -n ${NAMESPACE} >/dev/null 2>&1 || \
kubectl create serviceaccount ${KSA_NAME} -n ${NAMESPACE}

echo "==> Binding KSA to GSA via Workload Identity..."
gcloud iam service-accounts add-iam-policy-binding ${GSA_EMAIL} \
  --role roles/iam.workloadIdentityUser \
  --member "serviceAccount:${PROJECT_ID}.svc.id.goog[${NAMESPACE}/${KSA_NAME}]"

echo "==> Annotating KSA with GSA email..."
kubectl annotate serviceaccount ${KSA_NAME} \
  -n ${NAMESPACE} \
  iam.gke.io/gcp-service-account=${GSA_EMAIL} \
  --overwrite

echo "==> Setup complete. Run create-gpu-pool.sh after GPU quota is approved."
