#!/usr/bin/env bash
set -euo pipefail

repo_root="${REPO_ROOT:-/mnt/vepfs02/output/kevin_wang/memorywam/code/fastwam_memory_fullkv}"
runtime_bin="${RUNTIME_BIN:-/root/kevin_wang/envs/fastwam_guidemem/bin}"
output_dir="${OUTPUT_DIR:-/mnt/vepfs02/output/kevin_wang/memorywam/train/fastwam_memorywam_fullkv_putback_rgb_v3_b1x8_2k_seed42_20260731_v2}"

export PATH="${runtime_bin}:${PATH}"
export PYTHONPATH="${repo_root}/src${PYTHONPATH:+:${PYTHONPATH}}"
export DIFFSYNTH_MODEL_BASE_PATH="${DIFFSYNTH_MODEL_BASE_PATH:-/mnt/vepfs02/datasets/kevin_wang/code/projects/FastWAM/checkpoints}"
export DIFFSYNTH_SKIP_DOWNLOAD=true
export RMBENCH_PUTBACK_LEROBOT="${RMBENCH_PUTBACK_LEROBOT:-/mnt/vepfs02/output/kevin_wang/memorywam/data/rmbench_lerobot_v21_rgb_v3/put_back_block}"
export FASTWAM_PUTBACK_STATS="${FASTWAM_PUTBACK_STATS:-/mnt/vepfs02/output/kevin_wang/memorywam/data/fastwam_putback_stats_rgb_v3/dataset_stats.json}"
export FASTWAM_FULL_KV_LATENTS="${FASTWAM_FULL_KV_LATENTS:-/mnt/vepfs02/output/kevin_wang/memorywam/data/fastwam_putback_temporal_fullkv_rgb_v3_stride16}"
export FASTWAM_PUTBACK_TEXT_CACHE="${FASTWAM_PUTBACK_TEXT_CACHE:-/mnt/vepfs02/output/kevin_wang/memorywam/data/fastwam_putback_text_cache_rgb_v3}"
export FASTWAM_ACTION_DIT_INIT="${FASTWAM_ACTION_DIT_INIT:-/mnt/vepfs02/output/kevin_wang/memorywam/model_assets/fastwam_fullkv_official_init/ActionDiT_linear_interp_Wan22_alphascale_1024hdim.pt}"
export FASTWAM_FULL_KV_OUTPUT="${output_dir}"
dataset_manifest="${RMBENCH_DATASET_MANIFEST:-$(dirname "${RMBENCH_PUTBACK_LEROBOT}")/_build_manifest.json}"
rgb_report="${RMBENCH_RGB_REPORT:-$(dirname "${RMBENCH_PUTBACK_LEROBOT}")/_build_logs/rgb_contract.json}"
export PYTHONUTF8=1
export LANG=C.UTF-8
export LC_ALL=C.UTF-8

for required in \
  "${dataset_manifest}" \
  "${rgb_report}" \
  "${FASTWAM_PUTBACK_STATS}" \
  "${FASTWAM_FULL_KV_LATENTS}/manifest.json" \
  "${FASTWAM_ACTION_DIT_INIT}"; do
  if [[ ! -s "${required}" ]]; then
    echo "Missing prepared v3 artifact: ${required}" >&2
    exit 2
  fi
done

if [[ -e "${output_dir}" ]]; then
  echo "Refusing to overwrite existing run: ${output_dir}" >&2
  exit 2
fi

mkdir -p "${output_dir}"
cd "${repo_root}"

exec accelerate launch \
  --config_file scripts/accelerate_configs/accelerate_zero2_ds.yaml \
  --num_processes 8 \
  scripts/train.py \
  task=rmbench_putback_memorywam_fullkv_5k \
  "output_dir=${output_dir}" \
  max_steps=2000 \
  save_every=1000 \
  save_training_state=false \
  save_final_checkpoint=false \
  2>&1
