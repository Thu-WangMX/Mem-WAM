#!/usr/bin/env bash
set -euo pipefail

repo_root="${REPO_ROOT:-/mnt/vepfs02/output/kevin_wang/memorywam/code/fastwam_native_cache_consolidation}"
runtime_bin="${RUNTIME_BIN:-/root/kevin_wang/envs/fastwam_guidemem/bin}"
output_dir="${OUTPUT_DIR:-/mnt/vepfs02/output/kevin_wang/memorywam/train/native_cache_consolidation_putback_smoke_manual}"

export PATH="${runtime_bin}:${PATH}"
export PYTHONPATH="${repo_root}/src:${repo_root}${PYTHONPATH:+:${PYTHONPATH}}"
export DIFFSYNTH_MODEL_BASE_PATH="${DIFFSYNTH_MODEL_BASE_PATH:-/mnt/vepfs02/datasets/kevin_wang/code/projects/FastWAM/checkpoints}"
export DIFFSYNTH_SKIP_DOWNLOAD=true
export RMBENCH_PUTBACK_LEROBOT="${RMBENCH_PUTBACK_LEROBOT:-/mnt/vepfs02/output/kevin_wang/memorywam/data/rmbench_lerobot_v21_rgb_v3/put_back_block}"
export MEMORYWAM_PUTBACK_STATS="${MEMORYWAM_PUTBACK_STATS:-/mnt/vepfs02/output/kevin_wang/memorywam/data/fastwam_putback_stats_rgb_v3/dataset_stats.json}"
export MEMORYWAM_PUTBACK_CONTINUOUS_LATENTS="${MEMORYWAM_PUTBACK_CONTINUOUS_LATENTS:-/mnt/vepfs02/output/kevin_wang/memorywam/data/fastwam_putback_continuous_episode_stride16_v4}"
export MEMORYWAM_PUTBACK_TEXT_CACHE="${MEMORYWAM_PUTBACK_TEXT_CACHE:-/mnt/vepfs02/output/kevin_wang/memorywam/data/fastwam_putback_text_cache_rgb_v3}"
export FASTWAM_ACTION_DIT_INIT="${FASTWAM_ACTION_DIT_INIT:-/mnt/vepfs02/output/kevin_wang/memorywam/model_assets/fastwam_fullkv_official_init/ActionDiT_linear_interp_Wan22_alphascale_1024hdim.pt}"
export NATIVE_CACHE_BASE_CHECKPOINT="${NATIVE_CACHE_BASE_CHECKPOINT:-/mnt/vepfs02/output/kevin_wang/memorywam/train/memorywam_fullattention_putback_fsdp_unchunked_resume20k_to30k_save2k_seed42_cloud/checkpoints/weights/step_024000.pt}"
export NATIVE_CACHE_SMOKE_OUTPUT="${output_dir}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export PYTHONUTF8=1
export LANG=C.UTF-8
export LC_ALL=C.UTF-8

for required in \
  "$(dirname "${RMBENCH_PUTBACK_LEROBOT}")/_build_manifest.json" \
  "$(dirname "${RMBENCH_PUTBACK_LEROBOT}")/_build_logs/rgb_contract.json" \
  "${MEMORYWAM_PUTBACK_STATS}" \
  "${MEMORYWAM_PUTBACK_CONTINUOUS_LATENTS}/manifest.json" \
  "${MEMORYWAM_PUTBACK_TEXT_CACHE}" \
  "${FASTWAM_ACTION_DIT_INIT}" \
  "${NATIVE_CACHE_BASE_CHECKPOINT}"; do
  if [[ ! -e "${required}" ]]; then
    echo "Missing required artifact: ${required}" >&2
    exit 2
  fi
done

if [[ "$(tr ',' '\n' <<<"${CUDA_VISIBLE_DEVICES}" | wc -l)" -ne 8 ]]; then
  echo "Smoke requires exactly eight visible GPUs: ${CUDA_VISIBLE_DEVICES}" >&2
  exit 2
fi
if [[ -e "${output_dir}" ]]; then
  echo "Refusing to overwrite existing smoke output: ${output_dir}" >&2
  exit 2
fi

mkdir -p "${output_dir}"
cd "${repo_root}"

exec accelerate launch \
  --config_file scripts/accelerate_configs/accelerate_fsdp_full_shard.yaml \
  --num_processes 8 \
  scripts/train.py \
  task=rmbench_putback_native_cache_smoke \
  2>&1
