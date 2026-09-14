#!/usr/bin/env bash
set -euo pipefail
EXPERIMENT=putback_fixed exec "${REPO_ROOT:-/mnt/vepfs02/output/kevin_wang/memorywam/code/fastwam_multimodal_rate_controlled_k8}/ops/train_segment_compress_experiment_8gpu.sh"
