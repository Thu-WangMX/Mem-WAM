#!/usr/bin/env bash
set -euo pipefail

repo_root="${REPO_ROOT:-/mnt/vepfs02/output/kevin_wang/memorywam/code/fastwam_memory_fullkv}"
runtime_bin="${RUNTIME_BIN:-/root/kevin_wang/envs/fastwam_guidemem/bin}"
output_dir="${OUTPUT_DIR:-/mnt/vepfs02/output/kevin_wang/memorywam/train/fastwam_fullattention_putback_memorywamrope_logit_b1x8_seed42_20260801}"
max_steps="${MAX_STEPS:-1000}"
save_every="${SAVE_EVERY:-1000}"
resume_state="${RESUME_STATE:-}"

export PATH="${runtime_bin}:${PATH}"
export PYTHONPATH="${repo_root}/src:${repo_root}${PYTHONPATH:+:${PYTHONPATH}}"
export DIFFSYNTH_MODEL_BASE_PATH="${DIFFSYNTH_MODEL_BASE_PATH:-/mnt/vepfs02/datasets/kevin_wang/code/projects/FastWAM/checkpoints}"
export DIFFSYNTH_SKIP_DOWNLOAD=true
export RMBENCH_PUTBACK_LEROBOT="${RMBENCH_PUTBACK_LEROBOT:-/mnt/vepfs02/output/kevin_wang/memorywam/data/rmbench_lerobot_v21_rgb_v3/put_back_block}"
export FASTWAM_PUTBACK_STATS="${FASTWAM_PUTBACK_STATS:-/mnt/vepfs02/output/kevin_wang/memorywam/data/fastwam_putback_stats_rgb_v3/dataset_stats.json}"
export FASTWAM_FULL_KV_LATENTS="${FASTWAM_FULL_KV_LATENTS:-/mnt/vepfs02/output/kevin_wang/memorywam/data/fastwam_putback_temporal_fullkv_rgb_v3_stride16}"
export FASTWAM_PUTBACK_TEXT_CACHE="${FASTWAM_PUTBACK_TEXT_CACHE:-/mnt/vepfs02/output/kevin_wang/memorywam/data/fastwam_putback_text_cache_rgb_v3}"
export FASTWAM_ACTION_DIT_INIT="${FASTWAM_ACTION_DIT_INIT:-/mnt/vepfs02/output/kevin_wang/memorywam/model_assets/fastwam_fullkv_official_init/ActionDiT_linear_interp_Wan22_alphascale_1024hdim.pt}"
export FASTWAM_FULL_KV_OUTPUT="${output_dir}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export PYTHONUTF8=1
export LANG=C.UTF-8
export LC_ALL=C.UTF-8

for required in \
  "$(dirname "${RMBENCH_PUTBACK_LEROBOT}")/_build_manifest.json" \
  "$(dirname "${RMBENCH_PUTBACK_LEROBOT}")/_build_logs/rgb_contract.json" \
  "${FASTWAM_PUTBACK_STATS}" \
  "${FASTWAM_FULL_KV_LATENTS}/manifest.json" \
  "${FASTWAM_ACTION_DIT_INIT}"; do
  if [[ ! -s "${required}" ]]; then
    echo "Missing required artifact: ${required}" >&2
    exit 2
  fi
done

if [[ "$(tr ',' '\n' <<<"${CUDA_VISIBLE_DEVICES}" | wc -l)" -ne 8 ]]; then
  echo "This run requires exactly eight visible GPUs: ${CUDA_VISIBLE_DEVICES}" >&2
  exit 2
fi
if [[ ! "${max_steps}" =~ ^[1-9][0-9]*$ ]]; then
  echo "MAX_STEPS must be a positive integer, got: ${max_steps}" >&2
  exit 2
fi
if [[ -n "${resume_state}" ]]; then
  if [[ ! -s "${resume_state}/trainer_state.json" ]]; then
    echo "Invalid RESUME_STATE directory: ${resume_state}" >&2
    exit 2
  fi
  resume_override="resume=${resume_state}"
else
  if [[ -e "${output_dir}" ]]; then
    echo "Refusing to overwrite existing fresh run: ${output_dir}" >&2
    exit 2
  fi
  resume_override="resume=null"
fi

mkdir -p "${output_dir}"
cd "${repo_root}"

exec accelerate launch \
  --config_file scripts/accelerate_configs/accelerate_zero2_ds.yaml \
  --num_processes 8 \
  scripts/train.py \
  task=rmbench_putback_memorywam_fullkv_5k \
  "output_dir=${output_dir}" \
  model.video_scheduler.training_sampling_scheme=logit_normal \
  model.action_scheduler.training_sampling_scheme=logit_normal \
  model.action_rope_spatial_mode=memorywam \
  "max_steps=${max_steps}" \
  "save_every=${save_every}" \
  save_training_state=true \
  save_final_checkpoint=false \
  "${resume_override}" \
  log_every=10 \
  seed=42
