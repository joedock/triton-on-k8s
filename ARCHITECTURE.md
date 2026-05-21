# triton-on-k8s — Architecture Document

A production-grade NVIDIA Triton Inference Server deployment on
Google Kubernetes Engine, with GPU-accelerated model serving,
observability, and cost-optimized infrastructure.

This document explains *why* the system is built the way it is —
the decisions, the alternatives considered, the failure modes,
and what would change for production scale.

---

## 1. Problem Statement
This project exists as a learning artifact to demonstrate production-style
patterns for GPU inference at scale. While it is a portfolio piece, the architectural
patterns utilized — NVIDIA Triton for serving, Kubernetes for GPU scheduling,
event-driven autoscaling concepts, and managed Prometheus for observability — are
identical to those used by organizations running high-volume, real-world machine
learning workloads. It solves the problem of bridging the gap between a data scientist's
local PyTorch model and a robust, scalable, cloud-native serving layer.

## 2. System Overview
```text
┌─────────────────────────────────────────────────────────┐
│                    Client / User                        │
│              (curl, Python client, browser)             │
└──────────────────────┬──────────────────────────────────┘
                       │ HTTP (8000) / gRPC (8001)
┌──────────────────────▼──────────────────────────────────┐
│              GKE Cluster (us-central1-a)                │
│                                                         │
│  ┌─────────────────────────────────────────────────┐    │
│  │           GCP LoadBalancer                      │    │
│  └──────────────────┬──────────────────────────────┘    │
│                     │                                   │
│  ┌──────────────────▼──────────────────────────────┐    │
│  │         Triton Inference Server Pod             │    │
│  │         nvcr.io/nvidia/tritonserver:23.10-py3   │    │
│  │              GPU: NVIDIA T4 (spot)              │    │
│  │              Metrics: :8002                     │    │
│  └──────────────────┬──────────────────────────────┘    │
│                     │                                   │
│  ┌──────────────────▼──────────────────────────────┐    │
│  │        Model Repository (GCS Bucket)            │    │
│  │     gs://triton-models-joedock-prod/...         │    │
│  └─────────────────────────────────────────────────┘    │
│                                                         │
│  ┌─────────────────────────────────────────────────┐    │
│  │      Google Managed Prometheus (gmp-system)     │    │
│  │   PodMonitoring → Triton :8002                  │    │
│  │   ClusterPodMonitoring → DCGM :9400 (managed)   │    │
│  └─────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────┘
```
The request flow begins when a client submits an HTTP or gRPC inference request to
the external IP address exposed by the Kubernetes `LoadBalancer` Service.
The Service routes this traffic to the NVIDIA Triton Inference Server pod
running on a GPU-enabled node within the GKE cluster.

Upon receiving the request, Triton manages the execution. If dynamic batching is
enabled, Triton holds the request in a queue for a specified microsecond window
to group it with incoming concurrent requests. The batch is then loaded into the
GPU's VRAM, the forward pass of the model is executed using the libtorch backend,
and the resulting output tensors are returned through the reverse path to the client.

In parallel, Google Managed Prometheus (GMP) — auto-provisioned by GKE on
GPU-enabled clusters — scrapes Triton's `/metrics` endpoint on port 8002 via
a `PodMonitoring` resource, and the GKE-managed DCGM exporter via a
`ClusterPodMonitoring` resource. Both metric streams are queryable through
Cloud Monitoring or GMP's Prometheus-compatible HTTP API.

## 3. Architecture Decisions

### 3.1 GKE over EKS or AKS
Google Kubernetes Engine (GKE) was selected primarily for its mature Autopilot
and standard tier scheduling capabilities, combined with favorable spot pricing for
NVIDIA T4 and L4 GPUs. While Amazon EKS or Azure AKS are viable alternatives,
GKE provides a seamless integration with Google Cloud Storage (GCS) and Workload
Identity, reducing the friction of building a proof-of-concept.

### 3.2 NVIDIA Triton over FastAPI + PyTorch
Deploying a raw FastAPI application wrapping a PyTorch model is a common
starting point, but it provides zero inference-specific infrastructure. Triton was
chosen because it natively provides dynamic batching, multi-framework support,
concurrent model execution, GPU memory management, and out-of-the-box
Prometheus metrics. Production inference workloads require these features
to maximize expensive GPU utilization, making a dedicated serving layer like
Triton (or alternatives like Ray Serve or BentoML) essential.

### 3.3 TorchScript over ONNX or TensorRT
TorchScript was selected as the model format because it offers the fastest,
lowest-friction export path from a native PyTorch training loop while
maintaining compatibility with Triton's `libtorch` backend. ONNX would
offer broader runtime portability, and TensorRT could provide a 2x to 5x
throughput speedup. However, TensorRT requires a highly specific and
hardware-dependent compilation pipeline. TorchScript allowed for rapid
functional validation, with TensorRT optimization slated as a future roadmap item.

### 3.4 GCS for the Model Repository
Storing the model weights in a GCS object bucket — rather than baking
them into the Docker container or utilizing a PersistentVolume (PV) — decouples
the infrastructure lifecycle from the machine learning lifecycle. This pattern
allows data scientists to push new model versions to the bucket independently.
Triton can dynamically poll the bucket for rolling updates without requiring a new
container build or complex PV mounting logic across multiple nodes.

### 3.5 Spot GPU Nodes
The GPU nodepools are provisioned using GCP Spot VMs. The architectural
trade-off here is a roughly 68% reduction in compute costs against the inherent
risk of sudden node preemption. Because Triton inference requests are stateless
and largely idempotent (from the server's perspective), a preempted node simply
drops transient connections that the client can retry. This is an acceptable trade-off
for development, though production environments would require a hybrid baseline.

### 3.6 Deployment over StatefulSet or DaemonSet
The Triton workload is managed via a standard Kubernetes `Deployment`.
Because the workload is entirely stateless (the model state lives in GCS and
inference state is ephemeral), a `StatefulSet` is unnecessary. A `DaemonSet` is
also inappropriate, as the goal is to run $N$ replicas scaled to traffic demand across
the cluster, rather than strictly forcing one pod per node regardless of load.

### 3.7 Google Managed Prometheus over kube-prometheus-stack
On GPU-enabled GKE clusters, Google auto-provisions two observability
components: a DCGM exporter in the `gke-managed-system` namespace
exposing NVIDIA GPU telemetry, and Google Managed Prometheus (GMP)
in the `gmp-system` namespace. GMP discovers scrape targets via the
`PodMonitoring` and `ClusterPodMonitoring` CRDs (Google's own resources,
distinct from the Prometheus Operator's `ServiceMonitor`/`PodMonitor`),
and serves metrics through Cloud Monitoring and a Prometheus-compatible
HTTP API.

Three observability paths were considered:

**Option A — GMP only (chosen).** Use the managed stack as-is. DCGM is
already scraped via the GKE-managed `gke-managed-dcgm-exporter`
`ClusterPodMonitoring`. Triton is scraped via a `PodMonitoring` resource
in the `triton` namespace. Queries hit GMP's Prometheus-compatible API
or Cloud Monitoring's UI. Zero install effort beyond the Triton
`PodMonitoring`, no version drift, lowest operational cost. Trade-off:
queries and dashboards live in GCP-specific tooling, and PromQL
behaviour through GMP's frontend has minor compatibility quirks
(see §8).

**Option B — kube-prometheus-stack alongside GMP.** Run both stacks.
The self-hosted path is cloud-portable; GMP provides managed visibility.
Rejected because GKE's managed DCGM exporter doesn't expose a Service
(only pods scraped via `ClusterPodMonitoring`), making a clean
`ServiceMonitor` configuration impractical. The alternative — deploying
a second DCGM exporter — duplicates work and creates dual-scrape cost
without clear benefit at lab scale.

**Option C — replace GMP with kube-prometheus-stack.** Fully portable
and standard. Rejected because it fights GKE defaults, adds maintenance
burden, and doesn't materially improve the portfolio story.

On a non-GCP cluster (EKS, AKS, or bare-metal CoreWeave), the chosen
path would invert: deploy DCGM exporter via the NVIDIA GPU Operator
and ingest via kube-prometheus-stack. The `kubernetes/triton-servicemonitor.yaml`
manifest is retained in the repo as the cloud-portable alternative
and carries a header comment marking it as such.

### 3.8 LoadBalancer Service over Ingress
A Type `LoadBalancer` Service is used for direct, low-friction exposure during development
and testing. In a true production environment, this would be replaced with a robust `Ingress`
controller providing TLS termination, authentication, and rate limiting in front of a Type `ClusterIP` Service.

### 3.9 GKE-Managed DCGM over NVIDIA GPU Operator
The natural assumption when adding GPU telemetry to Kubernetes is to install
the NVIDIA GPU Operator, which bundles drivers, the container toolkit, the
device plugin, node feature discovery, DCGM exporter, and MIG management
as a single Helm release. On GKE this is the wrong default. GKE auto-provisions
DCGM exporter on any GPU-enabled cluster, exposes GPU metrics on port 9400,
and pre-wires it into GMP via the `gke-managed-dcgm-exporter`
`ClusterPodMonitoring`. Driver installation and container runtime GPU support
are managed by GKE's COS images.

Installing the full GPU Operator alongside this managed stack creates a
device-plugin conflict: GKE's `nvidia-gpu-device-plugin` DaemonSet and the
Operator's `nvidia-device-plugin-daemonset` both attempt to claim the same
GPU. Resolution requires either tainting GPU nodes to exclude one plugin
or editing GKE's DaemonSet's node selector — both add fragility without
adding portfolio value at lab scale.

The chosen path consumes the managed DCGM exporter directly and skips
the GPU Operator install. Verified end-to-end:
`DCGM_FI_DEV_GPU_UTIL{modelName="Tesla T4"}` returns data from GMP's
API against the running GPU node.

DCGM exporter logs three categories of non-actionable errors on GKE's
COS host: `[[SysMon]] Incompatible hardware vendor for sysmon` (CPU
topology module fails to load on COS), `[[NvSwitch]] Not attached to
NvSwitches` (expected on single-GPU non-NVLink nodes, fires every 30s),
and `Failed to load module 9` (same root cause as the sysmon failure).
None affect GPU metric collection.

On a non-GCP cluster, the GPU Operator becomes the right choice — there
is no managed alternative to consume, and the conflict it creates on GKE
does not exist elsewhere.

## 4. Configuration Rationale

## 4.1 Prometheus Annotations
The pod template includes `prometheus.io/scrape` and `prometheus.io/port` annotations.
These annotations are only used when a Prometheus scrape job is configured with `relabel_configs` that consumes them. They were more common before the Prometheus Operator (and now GMP) became preferred discovery methods. Under the current GMP-based setup, discovery happens via the `PodMonitoring` CRD, which makes these annotations dead code. They will be removed in a follow-up commit.

## 4.2 Triton CLI Flags
`--model-repository=gs://triton-models-joedock-prod/model_repository`

This flag tells Triton exactly where to pull its served models from — in this case, natively hitting a Google Cloud Storage URI. There is no default; if no repository path is passed at startup, Triton won't launch. This specific GCS path decouples model storage from the Triton pods running on GKE, keeping the compute layer stateless while pulling artifacts from a centralized bucket. This is also the production target — relying on a highly available GCS bucket is the practical way to horizontally scale inference nodes without duplicating model files across local volumes.

`--strict-model-config=false`

This flag tells Triton to stop demanding a complete `config.pbtxt` for every model and instead auto-generate missing configs from internal model metadata. Out of the box, this defaults to `true`. It was flipped to `false` here purely as a development convenience to iterate fast without writing boilerplate configs for every tweak. It would never stay this way in production. The strict default is mandatory there — it forces every served model to have an explicit, version-controlled config file. Relying on autoconfig in production introduces implicit, under-the-hood behaviour that is a nightmare to debug. If Triton's auto-detection logic changes silently between versions, there is zero operational record of what config it actually used to serve the model.

`--log-verbose=1`

Detailed logging is on so the granular mechanics of request routing and model loading are visible. By default it's 0 (disabled). This will not stay on in a high-throughput production environment — the volume of verbose logs would hammer disk I/O and inflate log ingestion costs without giving much actionable value during steady-state operations.

### 4.3 Model Repository Structure & Lifecycle
**Repository Layout**
A defining characteristic of Triton's model repository layout is why the `config.pbtxt` file sits at the root of the model folder, outside the numbered version directories.

The config describes the model's contract — its name, backend, inputs, outputs, batching behaviour, instance groups — which is shared across all versions. Each numbered version directory holds only the version-specific binary (`model.pt`). Adding a v2 means creating a `2/` directory with a new `model.pt`; Triton picks it up based on the `version_policy` in the shared config. The separation lets versions roll forward without rewriting the contract.

**Lifecycle Mode**
Triton is running in the default `NONE` mode — all models in the repository are loaded at startup, with no runtime add/remove. This is appropriate for the current single-model demo. For production with multiple served models, rolling updates, or canary deployments, this would switch to `EXPLICIT` and integrate `POST /v2/repository/models/{name}/load` calls into a CI/CD pipeline. `POLL` mode is unsuitable for production due to a lack of atomicity, auditability, and rollback control.

## 5. IAM and Security
Securing this workload means keeping a tight lid on who (or what) can access proprietary model weights in GCS.

**Role Bindings and Least Privilege**
When setting up Triton's permissions, the service account is strictly limited to what it actually needs. Exactly two roles are attached: `roles/storage.objectViewer` and `roles/storage.legacyBucketReader`. Together, these let Triton list what's in the model repository bucket and read the actual files. Broader roles like `storage.objectAdmin` or `storage.admin` were explicitly rejected — Triton has no business writing, modifying, or deleting objects in storage. `roles/owner` was a hard pass — that's a massive, unnecessary blast radius.

**The gcs-key.json Risk (Resolved)**
Triton previously authenticated by mounting a static service account key (`gcs-key.json`) via a Kubernetes Secret. It was the fastest way to get the artifact running, but it was a long-lived credential sitting on disk. A botched volume mount, an accidental commit to git history, or a leak into an image registry layer would expose the proprietary model weights.

**Workload Identity as the Fix (Implemented)**
The production fix is GKE Workload Identity. Instead of passing around a physical JSON file, the Kubernetes Service Account (KSA) is annotated to map directly to the Google Service Account (GSA). When Triton needs to pull a model, the GKE metadata server intercepts the request under the hood and hands back a short-lived, ephemeral token. The win: no physical key file exists on the cluster — nothing to leak, nothing to manually rotate.

**The IAM-as-Code Gap**
The IAM setup was initially built with manual `gcloud` commands. Those commands are now captured in `scripts/setup-gcp.sh` so there is a reproducible record, but it's still a stopgap. The right end state expresses the service account creation, role bindings, and Workload Identity mapping in Terraform. That shifts from procedural shell scripts to declarative infrastructure-as-code, which lets static analysis tools (`tfsec`, `checkov`) run against PRs to catch security gaps before they hit the cluster.

## 6. Cost Engineering
Deploying GPUs is highly capital-intensive.

| Resource | Hourly Rate | Monthly Estimate |
| :--- | :--- | :--- |
| T4 GPU (On-Demand) | ~$0.35 | ~$250 |
| T4 GPU (Spot) | ~$0.11 | ~$80 |

*Note: Example estimates based on us-central1 pricing.*

The system utilizes an aggressive scale-to-zero pattern. By pairing GKE Cluster Autoscaler with Spot VMs, the cluster scales down to zero active GPUs when there are no pending pods, eliminating hardware costs during downtime. Modern, specialized GPU cloud providers (like CoreWeave) have built their entire business models around this type of cost-aware, dynamic GPU scheduling.

## 7. Failure Modes
A Staff-level architecture must account for how the system degrades and recovers under stress.

**Spot Node Preemption**
If the underlying GCP Spot VM is reclaimed, the Triton pod is forcefully terminated. In-flight requests will drop and return connection errors to the client. The Kubernetes scheduler will immediately request a new Spot node and recreate the pod, restoring service once the new node provisions.

**GCS Bucket Unreachable**
If GCP object storage experiences an outage or IAM permissions are revoked, new Triton pods will fail to start. The server will fail to download the model repository, crash, and enter a `CrashLoopBackOff` state. Existing, running pods that have already loaded the model into VRAM may continue to serve traffic, assuming they are not configured to actively poll GCS for updates.

**Corrupted Model File**
If an incompatible or corrupted `.pt` file is uploaded to GCS, Triton will detect the failure during the model load phase. The pod will start, but the specific model will be marked `UNAVAILABLE`. The Kubernetes readiness probe will fail, preventing the `Service` from routing user traffic to the corrupted instance.

**Triton Server Crash**
If the Triton process encounters a fatal error (e.g., an unhandled CUDA exception) mid-request, the pod will terminate. The client will receive a 5xx error. Kubernetes will automatically restart the container, and the workload will self-heal.

**Concurrent Load Exceeds GPU Batch Capacity**
If the incoming request rate drastically exceeds the hardware's capacity, Triton's internal queues will fill up. Clients will experience severe latency spikes. If the load sustains, requests will hit the timeout thresholds and fail, requiring the autoscaler to provision additional GPU nodes to absorb the traffic.

**GMP Ingestion Disruption**
If Google Managed Prometheus experiences an outage or scrape interruption, the inference workload is entirely unaffected. Triton continues serving traffic and emitting metrics on `:8002`, but the system suffers a temporary loss of observability and historical data until GMP recovers. Because GMP is a managed service, recovery is Google's responsibility — no operator intervention is required on the cluster.

**Control Plane Upgrades**
During a GKE control plane upgrade, the Kubernetes API becomes temporarily unavailable. Existing Triton pods on the worker nodes will continue to serve inference traffic without interruption. However, scaling events, deployment updates, or pod rescheduling cannot occur until the control plane returns.

## 8. Observability

Inference serving is highly sensitive to latency. Because GPU compute operates fundamentally differently than standard CPU-bound microservices, observability requires a precise, hardware-aware metrics pipeline.

The telemetry chain on this deployment uses Google Managed Prometheus rather than a self-hosted Prometheus instance. The flow is:

1. Triton exposes Prometheus-format metrics on port `8002`.
2. GKE's DCGM exporter (in the `gke-managed-system` namespace) exposes GPU telemetry on port `9400`.
3. A `PodMonitoring` resource in the `triton` namespace selects Triton pods by the `app: triton` label and tells GMP to scrape `:8002` every 15s.
4. A `ClusterPodMonitoring` resource (`gke-managed-dcgm-exporter`, managed by GKE) tells GMP to scrape DCGM `:9400`.
5. GMP ingests both streams and stores time series in Cloud Monitoring.
6. Queries hit GMP's Prometheus-compatible HTTP API at `https://monitoring.googleapis.com/v1/projects/{PROJECT}/location/global/prometheus/api/v1/query` or Cloud Monitoring's Metrics Explorer.

To extract meaningful signals from this pipeline, five critical PromQL queries are used. System throughput is measured with `rate(nv_inference_request_success[5m])`, which calculates the per-second rate of successful inference requests averaged over a rolling five-minute window. Full lifecycle latency (from request arrival to response dispatch) is measured with `rate(nv_inference_request_duration_us[5m]) / rate(nv_inference_request_success[5m]) / 1000` — dividing total accumulated microsecond duration by the success rate, then dividing again by 1000 to convert into a normalized, human-readable average request latency in milliseconds.

Overall latency breaks down into two components. Queue time — how long requests sit idle waiting for compute — is tracked with `rate(nv_inference_queue_duration_us[5m]) / rate(nv_inference_request_success[5m]) / 1000`. Compute latency — actual GPU execution time per request — sums input loading, model inference, and output extraction: `(rate(nv_inference_compute_input_duration_us[5m]) + rate(nv_inference_compute_infer_duration_us[5m]) + rate(nv_inference_compute_output_duration_us[5m])) / rate(nv_inference_request_success[5m]) / 1000`. Overall hardware saturation comes from `nv_gpu_utilization`, which returns the percentage of time over the past sample period during which one or more kernels were actively executing on the GPU.

GPU-specific telemetry from DCGM complements Triton's inference metrics. The metrics that matter most for an inference workload: `DCGM_FI_DEV_GPU_UTIL` (overall GPU utilization percentage, useful as a saturation signal but not granular enough for tuning), `DCGM_FI_DEV_FB_USED` (framebuffer memory used — directly relevant when scaling model size or batch dimensions against T4's 16GB), `DCGM_FI_DEV_GPU_TEMP` (thermal pressure during sustained load tests), and `DCGM_FI_PROF_PIPE_TENSOR_ACTIVE` (tensor core utilization — the metric that tells you whether you're actually using the hardware that makes GPU inference fast, versus running on CUDA cores at a fraction of the throughput).

The queue duration query is particularly vital for tuning system performance, as it directly reveals the impact of Triton's `max_queue_delay_microseconds` configuration flag. When dynamic batching is enabled, Triton intentionally holds requests in the queue up to this defined microsecond limit to form larger, more efficient batches. Tracking queue duration shows exactly how much latency is being sacrificed for throughput. The Week 2 benchmarking experiment confirmed this. Testing at three queue delay values under 8 concurrent requests produced the following batching ratios: 100us → 1.09x, 2000us → 2.67x, 5000us → 2.67x. The optimal value of 2000us was selected — it captures the full batching benefit available under this load pattern with no additional latency cost compared to 5000us.

**A note on querying GMP versus standard Prometheus.** GMP exposes a Prometheus-compatible API but is not Prometheus. Label matchers in URL-encoded query strings can behave subtly differently — for example, a filtered query like `up{job="triton-inference-server"}` may return `null` rather than an empty vector when the label match fails in the URL parsing layer. The reliable debugging move is to drop the filter, inspect the full label set on the unfiltered metric, then re-add the filter once the exact label values are known. This was observed end-to-end during Triton's first scrape verification.

## 9. Open Questions and Verification Pending

### 9.1 Prometheus Annotation Verification — RESOLVED
GMP discovery uses the `PodMonitoring` CRD; the `prometheus.io/scrape` and
`prometheus.io/port` pod template annotations are dead code on this cluster
and will be removed in a follow-up commit.

### 9.2 Redundant `tritonserver` in Container Args — RESOLVED
Fixed in commit bdc1f83. Promoted to explicit `command:` field.

### 9.3 Max Queue Delay Benchmarking — RESOLVED
Benchmark complete. Testing at three queue delay values under 8 concurrent
requests produced batching ratios of 100us → 1.09x, 2000us → 2.67x,
5000us → 2.67x. Selected value: 2000us — full batching benefit at minimum
queue-latency cost. See §8.

### 9.4 ServiceMonitor Label Verification — N/A
This cluster uses GMP `PodMonitoring`, not kube-prometheus-stack
`ServiceMonitor`. The `release: monitoring` label is not consulted by GMP
and is not present in the deployed `triton-podmonitoring.yaml`. The
`triton-servicemonitor.yaml` manifest is retained as the cloud-portable
alternative — see §3.7.

### 9.5 IAM Posture and Infrastructure-as-Code Gap — RESOLVED
Captured in `scripts/setup-gcp.sh` (commit 070b28f). Script runs idempotently
with check-then-create pattern for all resources.

### 9.6 GCP Project Recreation and Credential Hygiene — RESOLVED
New project `triton-prod-joedock` created. Workload Identity replaces
`gcs-key.json` entirely. Verified live: metadata server returns
`triton-gcs-reader@triton-prod-joedock.iam.gserviceaccount.com` from inside
a running pod. No static credential anywhere in the chain.

### 9.7 GPU Quota — Two Separate Approvals Required
GCP GPU provisioning requires both regional (`NVIDIA_T4_GPUS` in `us-central1`)
AND global (`GPUS_ALL_REGIONS`) quota approvals. Both must be non-zero.
Initial setup only requested regional; global required a separate same-day
request. Note for future deployments: verify both quotas before attempting
GPU node pool scale-up.

### 9.8 GKE Cluster Idempotency — RESOLVED
`setup-gcp.sh` updated in commit 070b28f with check-then-create pattern for
all resources. Script now runs cleanly from any partial state without manual
intervention.

### 9.9 Dynamic Batching Baseline — RESOLVED
At `max_queue_delay_microseconds: 100` and 8 concurrent requests,
inference_count (60) exceeded exec_count (55) — confirming the dynamic
batcher grouped 5 requests into shared executions (1.09x batching ratio).
Full curve characterized in §9.3.

### 9.10 GMP PodMonitoring End-to-End Verification — RESOLVED
`triton-podmonitoring.yaml` deployed; `PodMonitoring` `Status` reports
`ConfigurationCreateSuccess`. Verified `up{job="triton-inference-server"}=1`
and `nv_inference_request_success{model="resnet50",version="1"}` returns
data through GMP's Prometheus-compatible API. Label-matcher behaviour
through the URL-encoded query path noted in §8.

## 10. What I'd Do Differently at Production Scale
This architecture serves as a robust foundation. If this system were operating as a tier-one production service handling live traffic, the following production roadmap would be implemented:

* **KEDA-Based Autoscaling:** Scaling pods dynamically based on Triton's internal queue depth metrics rather than basic CPU/Memory triggers.
* **TensorRT Optimization:** Compiling the PyTorch models into TensorRT engines for a 2x-5x throughput increase.
* **Model Versioning & Canary Deployments:** Utilizing Triton's native traffic splitting to safely roll out new model weights.
* **Multi-Region Deployment:** Expanding to multiple GCP regions with global load balancing for high availability.
* **Ingress + TLS + Auth:** Securing the public endpoint with a proper Ingress controller, certificates, and API authentication.
* **Dedicated GPU Nodepools:** Sizing an on-demand baseline pool to predicted load to eliminate preemption risks for core traffic.
* **Distributed Tracing:** Injecting OpenTelemetry trace IDs to follow requests from the edge through the Triton backend via Tempo or Jaeger.
* **Cost Dashboards:** Implementing strict budget alerts and granular cost-per-inference tracking.
* **MLOps Pipeline Integration:** Establishing a separate model evaluation pipeline (e.g., MLflow) that automatically promotes validated models to the GCS repository.
* **Terraform for IAM and Cluster:** Migrating `setup-gcp.sh` to declarative IaC for PR-time static analysis and reproducibility across environments.

## 11. References
* [NVIDIA Triton Inference Server Documentation](https://github.com/triton-inference-server/server)
* [NVIDIA GPU Operator](https://docs.nvidia.com/datacenter/cloud-native/gpu-operator/latest/index.html)
* [GKE GPU Scheduling](https://cloud.google.com/kubernetes-engine/docs/how-to/gpus)
* [Google Managed Prometheus](https://cloud.google.com/stackdriver/docs/managed-prometheus)
* [GMP PodMonitoring CRD reference](https://cloud.google.com/stackdriver/docs/managed-prometheus/setup-managed#gmp-pod-monitoring)
* [kube-prometheus-stack](https://github.com/prometheus-community/helm-charts/tree/main/charts/kube-prometheus-stack) (retained as cloud-portable alternative)
* [Prometheus Operator Custom Resource Definitions](https://prometheus-operator.dev/docs/operator/design/)
