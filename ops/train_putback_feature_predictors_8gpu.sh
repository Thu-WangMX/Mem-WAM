#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/mnt/vepfs02/output/kevin_wang/memorywam/code/fastwam_dynamic_multiframe_selection}"
PYTHON_BIN="${PYTHON_BIN:-/root/kevin_wang/envs/fastwam_guidemem/bin/python}"
DATASET="${DATASET:?set DATASET to the immutable predictor dataset artifact}"
OUTPUT_ROOT="${OUTPUT_ROOT:?set OUTPUT_ROOT to a new output directory}"

if [[ -e "${OUTPUT_ROOT}" ]]; then
  echo "refusing to overwrite OUTPUT_ROOT=${OUTPUT_ROOT}" >&2
  exit 2
fi
mkdir -p "${OUTPUT_ROOT}"
cd "${PROJECT_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}/src:${PROJECT_ROOT}"
export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"

# The predictors are intentionally lightweight. Each gets a disjoint four-GPU
# visibility group for cloud-job compatibility but executes on its local cuda:0;
# the expensive four-phase feature extraction is the stage that uses all 8 GPUs.
CUDA_VISIBLE_DEVICES=0,1,2,3 "${PYTHON_BIN}" scripts/train_putback_feature_predictor.py \
  --dataset "${DATASET}" \
  --mode visual_only \
  --output-root "${OUTPUT_ROOT}/visual_only" \
  --device cuda:0 \
  >"${OUTPUT_ROOT}/visual_only.log" 2>&1 &
VISUAL_PID=$!

CUDA_VISIBLE_DEVICES=4,5,6,7 "${PYTHON_BIN}" scripts/train_putback_feature_predictor.py \
  --dataset "${DATASET}" \
  --mode visual_action \
  --output-root "${OUTPUT_ROOT}/visual_action" \
  --device cuda:0 \
  >"${OUTPUT_ROOT}/visual_action.log" 2>&1 &
ACTION_PID=$!

cleanup() {
  kill "${VISUAL_PID}" "${ACTION_PID}" 2>/dev/null || true
}
trap cleanup INT TERM

VISUAL_STATUS=0
ACTION_STATUS=0
wait "${VISUAL_PID}" || VISUAL_STATUS=$?
wait "${ACTION_PID}" || ACTION_STATUS=$?
if [[ "${VISUAL_STATUS}" -ne 0 || "${ACTION_STATUS}" -ne 0 ]]; then
  echo "predictor training failed: visual=${VISUAL_STATUS} action=${ACTION_STATUS}" >&2
  exit 1
fi

cmp "${OUTPUT_ROOT}/visual_only/split_manifest.json" \
  "${OUTPUT_ROOT}/visual_action/split_manifest.json"
sha256sum "${OUTPUT_ROOT}/visual_only/split_manifest.json" \
  >"${OUTPUT_ROOT}/locked_split.sha256"
