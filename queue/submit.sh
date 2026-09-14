#!/usr/bin/env bash
# Entry point executed inside a Volcano job's allocated single eight-GPU node.
set -euo pipefail
source "$(dirname "$0")/../scripts/comparison_env.sh"
cd "${COMPARISON_ROOT}"
export PYTHONPATH="${PYTHONPATH}:/mnt/vepfs01/output/spidy.wang/starwam"
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1
export MEMORY_BACKEND="${MEMORY_BACKEND:-helios}"
export NUM_GPUS="${NUM_GPUS:-8}"
export BATCH_SIZE="${BATCH_SIZE:-16}" GRAD_ACCUM="${GRAD_ACCUM:-1}"
export MAX_STEPS="${MAX_STEPS:-30000}"
export WANDB_ENABLED="${WANDB_ENABLED:-true}"
export WANDB_PROJECT="${WANDB_PROJECT:-memorywam}" WANDB_MODE="${WANDB_MODE:-online}"
export WANDB_BASE_URL="${WANDB_BASE_URL:-https://api.wandb.ai}"
export WANDB_INIT_TIMEOUT="${WANDB_INIT_TIMEOUT:-120}"
export OUTPUT_DIR="${OUTPUT_DIR:-${COMPARISON_ROOT}/runs/${RMBENCH_TASK}_${MEMORY_BACKEND}_$(date +%Y%m%d_%H%M%S)}"
export WANDB_DIR="${OUTPUT_DIR}"
export TOS_UPLOAD="${TOS_UPLOAD:-1}"
export TOS_BUCKET="${TOS_BUCKET:-ai-dev}" TOS_PREFIX="${TOS_PREFIX:-spidy.wang/memorywam_ckpt}"
mkdir -p "${OUTPUT_DIR}"
# Preserve the scheduler's GPU assignment. Credentials are supplied by the caller.
unset ACCELERATE_USE_CPU STARWAM_MEMORY_LIMIT_GIB
if [[ "${1:-}" == --config ]]; then
  shift
  exec bash scripts/train_comparison.sh --config "wandb.workspace=${WANDB_ENTITY:-null}" "$@"
fi
if [[ "${TOS_UPLOAD}" == 1 ]]; then
  python -B queue/upload_checkpoints.py --doctor
fi
bash scripts/train_comparison.sh "wandb.workspace=${WANDB_ENTITY:-null}" "$@" 2>&1 | tee "${OUTPUT_DIR}/train.log"
# Only upload after successful completion, when all weight files are closed.
if [[ "${TOS_UPLOAD}" == 1 ]]; then
  python -B queue/upload_checkpoints.py --run "${OUTPUT_DIR}" 2>&1 | tee "${OUTPUT_DIR}/tos-upload.log"
fi
