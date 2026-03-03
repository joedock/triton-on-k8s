# 🚀 Triton Inference Server on Kubernetes

A production-grade AI inference platform built on Google Kubernetes Engine (GKE) using NVIDIA Triton Inference Server. This project demonstrates GPU-accelerated model serving at scale with full observability, autoscaling, and benchmarking.

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    Client / User                         │
│              (curl, Python client, browser)              │
└──────────────────────┬──────────────────────────────────┘
                       │ HTTP (8000) / gRPC (8001)
┌──────────────────────▼──────────────────────────────────┐
│              GKE Cluster (us-central1-a)                 │
│                                                          │
│  ┌─────────────────────────────────────────────────┐    │
│  │           GCP LoadBalancer                       │    │
│  └──────────────────┬──────────────────────────────┘    │
│                     │                                    │
│  ┌──────────────────▼──────────────────────────────┐    │
│  │         Triton Inference Server Pod              │    │
│  │         nvcr.io/nvidia/tritonserver:23.10-py3    │    │
│  │              GPU: NVIDIA T4 (spot)               │    │
│  └──────────────────┬──────────────────────────────┘    │
│                     │                                    │
│  ┌──────────────────▼──────────────────────────────┐    │
│  │        Model Repository (GCS Bucket)             │    │
│  │     gs://triton-model-repo/model_repository      │    │
│  └─────────────────────────────────────────────────┘    │
│                                                          │
│  ┌─────────────────────────────────────────────────┐    │
│  │     Prometheus + Grafana (kube-prometheus-stack) │    │
│  │          ServiceMonitor → Triton :8002           │    │
│  └─────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────┘
```

---

## 📊 Benchmark Results

### CPU vs GPU Performance (ResNet50 — 50 inferences)

| Metric | CPU (e2-standard-4) | GPU (NVIDIA T4) | Improvement |
|---|---|---|---|
| Avg Request Latency | ~183ms | ⏳ pending | — |
| Avg Compute Time | ~182ms | ⏳ pending | — |
| Avg Queue Time | ~0.4ms | ⏳ pending | — |
| Success Rate | 100% | ⏳ pending | — |
| Throughput | ⏳ pending | ⏳ pending | — |

> 🔄 GPU benchmark results pending quota approval. Will be updated with before/after comparison.

---

## 🛠️ Tech Stack

| Component | Technology |
|---|---|
| Kubernetes | GKE 1.34 |
| Inference Server | NVIDIA Triton 2.39.0 |
| Model Format | TorchScript (PyTorch LibTorch backend) |
| Model | ResNet50 (ImageNet — 1000 classes) |
| Model Storage | Google Cloud Storage |
| Node Type | n1-standard-4 + NVIDIA T4 (spot) |
| Observability | Prometheus + Grafana (kube-prometheus-stack) |
| Autoscaling | KEDA (coming soon) |

---

## 📁 Repository Structure

```
triton-on-k8s/
├── README.md
├── export_model.py              # Export ResNet50 to TorchScript
├── kubernetes/
│   ├── triton-deployment.yaml   # Triton Deployment manifest
│   ├── triton-service.yaml      # LoadBalancer Service
│   └── triton-servicemonitor.yaml  # Prometheus ServiceMonitor
├── model_repository/
│   └── resnet50/
│       ├── config.pbtxt         # Triton model config
│       └── 1/
│           └── model.pt         # TorchScript model (not committed — lives in GCS)
└── scripts/
    └── infer.py                 # Inference client script
```

---

## 🚀 Quick Start

### Prerequisites

- Google Cloud account with billing enabled
- `gcloud` CLI installed and authenticated
- `kubectl` and `helm` installed
- Python 3.12+ with virtualenv

### 1. Clone and Set Up Environment

```bash
git clone https://github.com/joedock/triton-on-k8s.git
cd triton-on-k8s

python3 -m venv venv
source venv/bin/activate
pip install tritonclient[all] numpy torch torchvision Pillow
```

### 2. Create GCP Project and Enable APIs

```bash
gcloud projects create triton-demo-project --name="Triton Demo"
gcloud config set project triton-demo-project

gcloud services enable \
  container.googleapis.com \
  compute.googleapis.com \
  storage.googleapis.com
```

### 3. Create GKE Cluster with GPU Spot Node Pool

```bash
# CPU control plane node
gcloud container clusters create triton-demo \
  --zone us-central1-a \
  --num-nodes 1 \
  --machine-type e2-standard-4 \
  --disk-size 50GB \
  --no-enable-autoupgrade \
  --project triton-demo-project

# GPU spot node pool (T4 — ~$0.11/hr)
gcloud container node-pools create gpu-pool \
  --cluster triton-demo \
  --zone us-central1-a \
  --num-nodes 1 \
  --machine-type n1-standard-4 \
  --accelerator type=nvidia-tesla-t4,count=1 \
  --spot \
  --disk-size 50GB \
  --no-enable-autoupgrade \
  --project triton-demo-project

# Get credentials
gcloud container clusters get-credentials triton-demo \
  --zone us-central1-a \
  --project triton-demo-project
```

### 4. Export Model and Upload to GCS

```bash
# Export ResNet50 to TorchScript
python3 export_model.py

# Create GCS bucket and upload
gsutil mb -p triton-demo-project -l us-central1 gs://triton-model-repo-joedock
gsutil -m cp -r model_repository/ gs://triton-model-repo-joedock/
```

### 5. Deploy Triton

```bash
# Create namespace and GCS credentials secret
kubectl create namespace triton
kubectl create secret generic gcs-credentials \
  --from-file=gcs-key.json=gcs-key.json \
  --namespace triton

# Deploy
kubectl apply -f kubernetes/
```

### 6. Send an Inference Request

```bash
# Get external IP
export TRITON_IP=$(kubectl get svc triton-inference-server -n triton \
  -o jsonpath='{.status.loadBalancer.ingress[0].ip}')

# Health check
curl http://$TRITON_IP:8000/v2/health/ready

# Run inference
python3 scripts/infer.py $TRITON_IP
```

---

## 📈 Observability

### Install Prometheus + Grafana

```bash
helm repo add prometheus-community \
  https://prometheus-community.github.io/helm-charts

helm install monitoring prometheus-community/kube-prometheus-stack \
  --namespace monitoring \
  --create-namespace \
  --set grafana.adminPassword=admin123 \
  --set prometheus.prometheusSpec.serviceMonitorSelectorNilUsesHelmValues=false
```

### Key Triton Metrics

| Metric | Description |
|---|---|
| `nv_inference_request_success` | Total successful inferences |
| `nv_inference_request_failure` | Total failed inferences |
| `nv_inference_request_duration_us` | End-to-end request duration (µs) |
| `nv_inference_queue_duration_us` | Time spent waiting in queue (µs) |
| `nv_inference_compute_infer_duration_us` | GPU/CPU compute time (µs) |
| `nv_inference_pending_request_count` | Current queue depth |

### Grafana Dashboard Queries

```promql
# Inference rate (req/s)
rate(nv_inference_request_success[2m])

# Average latency (ms)
rate(nv_inference_request_duration_us[2m]) / rate(nv_inference_request_success[2m]) / 1000

# Compute time (ms)
rate(nv_inference_compute_infer_duration_us[2m]) / rate(nv_inference_count[2m]) / 1000

# Queue time (ms)
rate(nv_inference_queue_duration_us[2m]) / rate(nv_inference_count[2m]) / 1000
```

---

## 💡 Key Concepts Demonstrated

**Dynamic Batching** — Triton automatically groups concurrent requests into batches to maximize GPU utilization. Configured via `preferred_batch_size: [4, 8]` in `config.pbtxt`.

**Model Versioning** — The `1/` directory structure supports multiple concurrent model versions with traffic splitting.

**GPU Scheduling** — Kubernetes `nodeSelector` and `tolerations` ensure Triton pods land on GPU nodes while preventing non-GPU workloads from consuming expensive GPU resources.

**IAM Least Privilege** — A dedicated GCS service account with `objectViewer` and `legacyBucketReader` roles only — no broader permissions.

**Spot/Preemptible Nodes** — GPU node pool uses spot pricing for ~68% cost reduction during development and testing.

---

## 💰 Cost Management

```bash
# Scale GPU pool to 0 when not testing
gcloud container clusters resize triton-demo \
  --node-pool gpu-pool \
  --num-nodes 0 \
  --zone us-central1-a

# Scale back up when ready
gcloud container clusters resize triton-demo \
  --node-pool gpu-pool \
  --num-nodes 1 \
  --zone us-central1-a
```

| Resource | Approx Cost |
|---|---|
| GKE cluster management | $0.10/hr |
| e2-standard-4 CPU node | $0.13/hr |
| n1-standard-4 + T4 spot | $0.11/hr |
| GCS bucket (100MB) | ~$0.002/month |

---

## 🗺️ Roadmap

- [x] ResNet50 model export and repository setup
- [x] GKE cluster with GPU spot node pool
- [x] Triton Inference Server deployment
- [x] Live inference over public LoadBalancer
- [x] Prometheus + Grafana observability stack
- [ ] GPU benchmark results (pending quota approval)
- [ ] KEDA autoscaling based on queue depth
- [ ] TensorRT optimization (CPU → TRT: expected 2-5x speedup)
- [ ] Load testing with `hey` and throughput benchmarks
- [ ] Ensemble pipeline (preprocess → infer → postprocess)

---

## 📚 References

- [NVIDIA Triton Inference Server Documentation](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/index.html)
- [GKE GPU Node Pools](https://cloud.google.com/kubernetes-engine/docs/how-to/gpus)
- [kube-prometheus-stack](https://github.com/prometheus-community/helm-charts/tree/main/charts/kube-prometheus-stack)
- [NVIDIA GPU Operator](https://docs.nvidia.com/datacenter/cloud-native/gpu-operator/overview.html)

---

## 👤 Author

Joe | Cloud Native Architect | Kubstronaut | NVIDIA-Certified AI Infrastructure

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue)](https://linkedin.com/in/joe-dockery-jackson-4a0a8a8)
[![GitHub](https://img.shields.io/badge/GitHub-Follow-black)](https://github.com/joedock)
