#!/usr/bin/env bash
set -euo pipefail

mode="${1:?Usage: bash ops/train_episode0_logit_ab.sh <memorywam|center|origin>}"
if [[ "${mode}" != "memorywam" && "${mode}" != "center" && "${mode}" != "origin" ]]; then
  echo "Action RoPE mode must be memorywam, center, or origin, got: ${mode}" >&2
  exit 2
fi

repo_root="/mnt/vepfs02/output/kevin_wang/memorywam/code/fastwam_memory_fullkv"
python_bin="/root/kevin_wang/envs/fastwam_guidemem/bin/python"
run_root="/mnt/vepfs02/output/kevin_wang/memorywam/train"
run_name="fastwam_fullkv_episode0_logit_${mode}_b1x8_500_seed42_20260801_v3"

export PATH="$(dirname "${python_bin}"):${PATH}"
export PYTHONPATH="${repo_root}/src:${repo_root}${PYTHONPATH:+:${PYTHONPATH}}"
export DIFFSYNTH_MODEL_BASE_PATH="/mnt/vepfs02/datasets/kevin_wang/code/projects/FastWAM/checkpoints"
export DIFFSYNTH_SKIP_DOWNLOAD=true
export RMBENCH_PUTBACK_LEROBOT="/mnt/vepfs02/output/kevin_wang/memorywam/data/rmbench_lerobot_v21_rgb_v3/put_back_block"
export FASTWAM_PUTBACK_STATS="/mnt/vepfs02/output/kevin_wang/memorywam/data/fastwam_putback_stats_rgb_v3/dataset_stats.json"
export FASTWAM_FULL_KV_LATENTS="/mnt/vepfs02/output/kevin_wang/memorywam/data/fastwam_putback_temporal_fullkv_rgb_v3_stride16"
export FASTWAM_PUTBACK_TEXT_CACHE="/mnt/vepfs02/output/kevin_wang/memorywam/data/fastwam_putback_text_cache_rgb_v3"
export FASTWAM_ACTION_DIT_INIT="/mnt/vepfs02/output/kevin_wang/memorywam/model_assets/fastwam_fullkv_official_init/ActionDiT_linear_interp_Wan22_alphascale_1024hdim.pt"
export PYTHONUTF8=1
export LANG=C.UTF-8
export LC_ALL=C.UTF-8

cd "${repo_root}"
exec accelerate launch \
  --config_file scripts/accelerate_configs/accelerate_zero2_ds.yaml \
  --num_processes 8 \
  scripts/train.py \
  task=rmbench_putback_memorywam_fullkv_5k \
  "output_dir=${run_root}/${run_name}" \
  '+data.train.episode_indices=[0]' \
  model.video_scheduler.training_sampling_scheme=logit_normal \
  model.action_scheduler.training_sampling_scheme=logit_normal \
  "model.action_rope_spatial_mode=${mode}" \
  max_steps=500 \
  log_every=10 \
  save_every=250 \
  save_training_state=false \
  save_final_checkpoint=false \
  resume=null \
  seed=42
