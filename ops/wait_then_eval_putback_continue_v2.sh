#!/usr/bin/env bash
set -euo pipefail

repo_root="${REPO_ROOT:-/mnt/vepfs02/output/kevin_wang/memorywam/code/fastwam_memory_fullkv}"
train_output="${TRAIN_OUTPUT:-/mnt/vepfs02/output/kevin_wang/memorywam/train/fastwam_memorywam_fullkv_putback_continue2k_from_step2000_v2_20260730}"
checkpoint="${train_output}/checkpoints/weights/step_002000.pt"
run_tag="${RUN_TAG:-continue4k_contractfix_v2_20260730}"
poll_seconds="${POLL_SECONDS:-30}"

while [[ ! -s "${checkpoint}" ]]; do
  if ! pgrep -f "output_dir=${train_output}" >/dev/null; then
    echo "Training exited before checkpoint was created: ${checkpoint}" >&2
    exit 1
  fi
  sleep "${poll_seconds}"
done

echo "Checkpoint ready: ${checkpoint}"
cd "${repo_root}"
exec env \
  CHECKPOINT="${checkpoint}" \
  RUN_TAG="${run_tag}" \
  DIFFSYNTH_MODEL_BASE_PATH="/mnt/vepfs02/datasets/kevin_wang/code/projects/FastWAM/checkpoints" \
  ACTION_DELTA_FRACTION=null \
  bash ops/eval_putback_five_seed.sh
