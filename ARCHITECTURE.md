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
patterns utilized—NVIDIA Triton for serving, Kubernetes for GPU scheduling, 
event-driven autoscaling concepts, and Prometheus Operator for observability—are 
identical to those used by organizations running high-volume, real-world machine l
earning workloads. It solves the problem of bridging the gap between a data scientist's 
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
│  └──────────────────┬──────────────────────────────┘    │
│                     │                                   │
│  ┌──────────────────▼──────────────────────────────┐    │
│  │        Model Repository (GCS Bucket)            │    │
│  │     gs://triton-model-repo/model_repository     │    │
│  └─────────────────────────────────────────────────┘    │
│                                                         │
│  ┌─────────────────────────────────────────────────┐    │
│  │     Prometheus + Grafana (kube-prometheus-stack)│    │
│  │          ServiceMonitor → Triton :8002          │    │
│  └─────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────┘
```
The request flow begins when a client submits an HTTP or gRPC inference request to 
the external IP address exposed by the Kubernetes `LoadBalancer` Service. 
The Service routes this traffic to one of the available NVIDIA Triton Inference Server 
pods running on a GPU-enabled node within the GKE cluster.

Upon receiving the request, Triton manages the execution. If dynamic batching is
 enabled, Triton holds the request in a queue for a specified microsecond window 
 to group it with incoming concurrent requests. The batch is then loaded into the 
 GPU's VRAM, the forward pass of the model is executed using the libtorch backend, 
 and the resulting output tensors are returned through the reverse path to the client.

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
Storing the model weights in a GCS object bucket—rather than baking 
them into the Docker container or utilizing a PersistentVolume (PV)—decouples 
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
also inappropriate, as the goal is to run $N$ replicas scaled to traffic demand across t
he cluster, rather than strictly forcing one pod per node regardless of load.

### 3.7 Prometheus Operator over Standalone Prometheus
The observability stack uses the `kube-prometheus-stack` (Prometheus Operator) 
rather than a standalone Prometheus deployment. This allows metrics discovery to 
be managed declaratively via `ServiceMonitor` Custom Resource Definitions (CRDs) 
rather than fragile, annotation-based scraping configurations.

### 3.8 LoadBalancer Service over Ingress
A Type `LoadBalancer` Service is used for direct, low-friction exposure during development 
and testing. In a true production environment, this would be replaced with a robust `Ingress` 
controller providing TLS termination, authentication, and rate limiting in front of a Type `ClusterIP` Service. 

## 4. Configuration Rationale

## 4.1 Prometheus Annotations
The pod template includes `prometheus.io/scrape` and `prometheus.io/port` annotations.  
These annotations are only used when a Prometheus scrape job is configured with `relabel_configs` the consumes them.  These were more common before the Prometheus Operator became the preferred method.  The current setup, discovery happens via the ServiceMonitor CRD, which makes these annotations likely redundant.  I will verify by inspecting the generated Prometheus scrape config once kube-prometheus-stack is reinstalled on the new cluster.  If they are not being used I will drop them from the config.

## 4.2 Triton CLI Flags
--model-repository=gs://triton-models-joedock-prod/model_repository
This flag tells Triton exactly where to pull its served models from—in our case, natively hitting a Google Cloud Storage (GCS) URI. There is no default value here; if you don't explicitly pass a repository path at startup, Triton won't even launch. I set this specific GCS path to decouple our model storage from the Triton pods running on GKE, keeping the compute layer stateless while pulling artifacts from a centralized bucket. We are definitely keeping this in production. In fact, this URI is already our production target. Relying on a highly available GCS bucket is really the only way we can horizontally scale these inference nodes without dealing with the headache of duplicating model files across local volumes.

--strict-model-config=false
This flag tells Triton to stop demanding a complete config.pbtxt for every single model and instead try to auto-generate the missing configs by scraping internal model metadata. Out of the box, this defaults to true. I've flipped it to false here purely as a development convenience so we can iterate fast without writing boilerplate configs for every tweak. But I would absolutely never keep this in production. As we've drilled a dozen times, that strict default is mandatory. It forces every served model to have an explicit, version-controlled config file. Relying on autoconfig in production introduces implicit, under-the-hood behavior that is a nightmare to debug. If Triton's auto-detection logic changes silently, we'd have zero operational record of what config it actually used to serve the model.

--log-verbose=1
This flag turns on detailed logging so we can actually see the granular mechanics of request routing and model loading. By default, it's set to 0 (disabled). I've bumped it to 1 for now because we need that deep visibility to trace inference requests and troubleshoot during this initial deployment phase. That said, this will not stay on in a high-throughput production environment. The sheer volume of verbose logs would hammer our disk I/O and unnecessarily spike our log ingestion costs without giving us much actionable value during normal steady-state ops.

### 4.3 Model Repository Structure & Lifecycle
**Repository Layout**
A defining characteristic of Triton's model repository layout is why the `config.pbtxt` file sits at the root of the model folder, outside of the numbered version directories. 

Because the config describes the model's contract — its name, backend, inputs, outputs, batching behavior, instance groups — which is shared across all versions. Each numbered version directory holds only the version-specific binary (`model.pt` for me). When I add a v2, I create a `2/` directory with a new `model.pt` and Triton picks it up based on the `version_policy` in the shared config. The separation lets me roll forward versions without rewriting the contract.

**Lifecycle Mode**
Triton is running in the default `NONE` mode — all models in the repository are loaded at startup, with no runtime add/remove. This is appropriate for the current single-model demo. For production with multiple served models, rolling updates, or canary deployments, I'd switch to `EXPLICIT` and integrate the `POST /v2/repository/models/{name}/load` calls into a CI/CD pipeline. `POLL` mode is unsuitable for production due to a lack of atomicity, auditability, and rollback control.

## 5. IAM and Security
Securing this workload means keeping a very tight lid on who (or what) can access our proprietary model weights over in GCS.

Role Bindings and Least Privilege
When setting up the permissions for Triton, I kept things strictly limited to what it actually needs to function. I attached exactly two roles to the service account: roles/storage.objectViewer and roles/storage.legacyBucketReader. Together, these just let Triton list what's in the model repository bucket and read the actual files. I explicitly rejected broader roles like storage.objectAdmin or storage.admin because Triton has absolutely no business writing, modifying, or deleting objects in our storage. And obviously, tossing roles/owner at it was a hard pass—that's a massive, unnecessary blast radius.

The gcs-key.json Risk
Right now, Triton authenticates by mounting a static service account key (gcs-key.json) via a Kubernetes Secret. It was the fastest way to get the artifact running yesterday, but I wouldn't leave it like this. It’s a long-lived credential sitting on disk. If someone botches a volume mount, accidentally commits that file to our git history, or leaks it into an image registry layer, our proprietary model weights are completely exposed.

Workload Identity as the Fix
The actual production fix here is GKE Workload Identity. Instead of passing around a physical JSON file, we just annotate our Kubernetes Service Account (KSA) so it maps directly to the Google Service Account (GSA). When Triton needs to pull a model, the GKE metadata server intercepts the request under the hood and hands back a short-lived, ephemeral token. The massive win here is that no physical key file ever exists on the cluster—meaning there's nothing to leak and nothing for us to manually rotate.

The IAM-as-Code Gap
When I was building this out, I initially just hammered out the IAM setup using manual gcloud commands. I’ve since grabbed those commands and put them into our deployment script so we at least have a reproducible record, but it’s still a stopgap. To do this right, we need to express the service account creation, the role bindings, and the Workload Identity mapping in Terraform. That shifts us from running procedural shell scripts to actual declarative infrastructure-as-code, which lets us run static analysis tools like tfsec or checkov against our PRs to catch security gaps before they ever hit the cluster.

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

**Prometheus Pod Dies**
If the `kube-prometheus-stack` pod crashes or runs out of memory, the inference workload is entirely unaffected. Triton will continue serving traffic and emitting metrics, but the system will suffer a temporary loss of observability and historical data scraping until the monitoring pod restarts.

**Control Plane Upgrades**
During a GKE control plane upgrade, the Kubernetes API becomes temporarily unavailable. Existing Triton pods on the worker nodes will continue to serve inference traffic without interruption. However, scaling events, deployment updates, or pod rescheduling cannot occur until the control plane returns.

## 8. Observability

Inference serving is highly sensitive to latency. Because GPU compute operates fundamentally differently than standard CPU-bound microservices, observability requires a precise, hardware-aware metrics pipeline. 

The telemetry architecture in this deployment follows a strict seven-step chain from the hardware to the dashboard. During operations or troubleshooting, this flow can be traced fluently:

1. Triton exposes Prometheus-format metrics on port `8002`.
2. The `triton-servicemonitor.yaml` tells the Prometheus Operator (via the strict `release: monitoring` label) to scrape that port via the Kubernetes Service.
3. The Operator dynamically updates the Prometheus instance's configuration to add Triton as a scrape target.
4. Prometheus pulls metrics from the Triton pod every scrape interval (currently 15s).
5. Prometheus stores the time series in its TSDB (Time Series Database).
6. Grafana queries Prometheus via PromQL.
7. The dashboard panel renders the query result.

To extract meaningful signals from this pipeline, we rely on five critical PromQL queries. To measure system throughput, we use `rate(nv_inference_request_success[5m])`, which calculates the per-second rate of successful inference requests averaged over a rolling five-minute window. To measure the full lifecycle latency from the moment a request hits the server to the moment the response is dispatched, we query `rate(nv_inference_request_duration_us[5m]) / rate(nv_inference_request_success[5m]) / 1000`. By dividing the total accumulated microsecond duration by the success rate, and dividing again by 1000, we convert the raw metric into a normalized, human-readable average request latency in milliseconds.

We break that overall latency down further into two components. First, to understand how long requests are sitting idle waiting for compute, we track queue time using `rate(nv_inference_queue_duration_us[5m]) / rate(nv_inference_request_success[5m]) / 1000`. Second, to measure the actual GPU execution time per request, we calculate compute latency by summing the rates of input loading, model inference, and output extraction, dividing by the success rate, and converting to milliseconds: `(rate(nv_inference_compute_input_duration_us[5m]) + rate(nv_inference_compute_infer_duration_us[5m]) + rate(nv_inference_compute_output_duration_us[5m])) / rate(nv_inference_request_success[5m]) / 1000`. Finally, we track overall hardware saturation simply by querying `nv_gpu_utilization`, which returns the percentage of time over the past sample period during which one or more kernels were actively executing on the GPU.

The queue duration query is particularly vital for tuning system performance, as it directly reveals the impact of Triton's `max_queue_delay_microseconds` configuration flag. When dynamic batching is enabled, Triton intentionally holds requests in the queue up to this defined microsecond limit to form larger, more efficient batches. Tracking the queue duration metric allows us to see exactly how much latency is being sacrificed for throughput. The Week 2 benchmarking experiment confirmed this. Testing at three queue delay values under 8 concurrent requests produced the following batching ratios: 100us → 1.09x, 2000us → 2.67x, 5000us → 2.67x. The optimal value of 2000us was selected — it captures the full batching benefit available under this load pattern with no additional latency cost compared to 5000us.

## 9. Open Questions and Verification Pending



### 9.1 Prometheus Annotation Verification
Verify the actual effect of the `prometheus.io/scrape` annotations using `kubectl exec` into the Prometheus pod to inspect the generated scrape config. If not consumed by the Operator, remove them.

### 9.2 Redundant `tritonserver` in Container Args — RESOLVED
Fixed in commit bdc1f83. Promoted to explicit command: field.

### 9.3 Max Queue Delay Benchmarking
`max_queue_delay_microseconds: 100` is likely too aggressive. At 100us, 
the dynamic batching window is shorter than typical inter-request arrival times 
in this deployment, meaning the batcher rarely forms batches larger than 
1. The configured `preferred_batch_size: [4, 8]` is therefore mostly inactive. 
Week 2 task: re-run benchmarks at 100us, 1000us, and 5000us to characterize 
the trade-off between p95 latency and throughput, and choose a value based on 
data rather than the original (likely tutorial-derived) value.

### 9.4 ServiceMonitor Label Verification
Verify the `release: monitoring` label behavior next week when the cluster is 
redeployed to ensure the `ServiceMonitor` is actively picked up and scraped 
by the Prometheus instance.

### 9.5 IAM Posture and Infrastructure-as-Code Gap — RESOLVED  
Captured in scripts/setup-gcp.sh (commit 070b28f). Script runs idempotently with check-then-create pattern for all resources.

### 9.6 GCP Project Recreation and Credential Hygiene — RESOLVED
New project triton-prod-joedock created. Workload Identity replaces
gcs-key.json entirely. Verified live: metadata server returns
triton-gcs-reader@triton-prod-joedock.iam.gserviceaccount.com
from inside running pod. No static credential anywhere in the chain.

### 9.7 GPU Quota — Two Separate Approvals Required
GCP GPU provisioning requires both regional (NVIDIA_T4_GPUS in
us-central1) AND global (GPUS_ALL_REGIONS) quota approvals. Both
must be non-zero. Initial setup only requested regional; global
required a separate same-day request. Note for future deployments:
verify both quotas before attempting GPU node pool scale-up.

### 9.8 GKE Cluster Idempotency — RESOLVED
setup-gcp.sh updated in commit 070b28f with check-then-create
pattern for all resources. Script now runs cleanly from any
partial state without manual intervention.

### 9.9 Dynamic Batching Baseline
At max_queue_delay_microseconds: 100 and 8 concurrent requests,
inference_count (60) exceeded exec_count (55) — confirming the
dynamic batcher grouped 5 requests into shared executions (1.09x
batching ratio). Week 2 task: benchmark at 1000us and 5000us
under sustained concurrent load to quantify the improvement curve.

## 10. What I'd Do Differently at Production Scale
This architecture serves as a robust foundation. If this system were operating as a tier-one production service handling live traffic, the following production roadmap would be implemented:

* **KEDA-Based Autoscaling:** Scaling pods dynamically based on Triton's internal queue depth metrics rather than basic CPU/Memory triggers.
* **TensorRT Optimization:** Compiling the PyTorch models into TensorRT engines for a 2x-5x throughput increase.
* **Model Versioning & Canary Deployments:** Utilizing Triton's native traffic splitting to safely roll out new model weights.
* **Multi-Region Deployment:** Expanding to multiple GCP regions with global load balancing for high availability.
* **Workload Identity:** Completely eliminating static Service Account keys in favor of dynamic GKE credentials.
* **Ingress + TLS + Auth:** Securing the public endpoint with a proper Ingress controller, certificates, and API authentication.
* **Dedicated GPU Nodepools:** Sizing an on-demand baseline pool to predicted load to eliminate preemption risks for core traffic.
* **Distributed Tracing:** Injecting OpenTelemetry trace IDs to follow requests from the edge through the Triton backend via Tempo or Jaeger.
* **Cost Dashboards:** Implementing strict budget alerts and granular cost-per-inference tracking.
* **MLOps Pipeline Integration:** Establishing a separate model evaluation pipeline (e.g., MLflow) that automatically promotes validated models to the GCS repository.

## 11. References
* [NVIDIA Triton Inference Server Documentation](https://github.com/triton-inference-server/server)
* [NVIDIA GPU Operator](https://docs.nvidia.com/datacenter/cloud-native/gpu-operator/latest/index.html)
* [GKE GPU Scheduling](https://cloud.google.com/kubernetes-engine/docs/how-to/gpus)
* [kube-prometheus-stack](https://github.com/prometheus-community/helm-charts/tree/main/charts/kube-prometheus-stack)
* [Prometheus Operator Custom Resource Definitions](https://prometheus-operator.dev/docs/operator/design/)
