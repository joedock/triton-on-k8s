#!/bin/bash
set -e

gcloud container node-pools create gpu-pool \
  --cluster triton-prod \
  --zone us-central1-a \
  --num-nodes 0 \
  --machine-type n1-standard-4 \
  --accelerator type=nvidia-tesla-t4,count=1 \
  --spot \
  --workload-metadata=GKE_METADATA \
  --no-enable-autoupgrade

echo ""
echo "GPU node pool created at 0 nodes (no GPU running, no cost)."
echo "To provision a GPU and start serving inference:"
echo "  gcloud container clusters resize triton-prod \\"
echo "    --node-pool gpu-pool \\"
echo "    --num-nodes 1 \\"
echo "    --zone us-central1-a"
echo ""
echo "To scale back to zero (stop GPU, stop billing):"
echo "  gcloud container clusters resize triton-prod \\"
echo "    --node-pool gpu-pool \\"
echo "    --num-nodes 0 \\"
echo "    --zone us-central1-a"
