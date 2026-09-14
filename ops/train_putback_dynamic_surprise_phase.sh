#!/usr/bin/env bash
set -euo pipefail

repo_root="${REPO_ROOT:-/mnt/vepfs02/output/kevin_wang/memorywam/code/fastwam_dynamic_surprise_memory}"
runtime_bin="${RUNTIME_BIN:-/root/kevin_wang/envs/fastwam_guidemem/bin}"
output_dir="${OUTPUT_DIR:-/mnt/vepfs02/output/kevin_wang/memorywam/train/dynamic_surprise_memory_putback_k8_25k_seed42}"
: "${PHASE_END_STEP:?Set PHASE_END_STEP to a 5k boundary from 5000 through 40000}"
: "${DYNAMIC_SURPRISE_MANIFEST:?Set DYNAMIC_SURPRISE_MANIFEST}"
: "${DYNAMIC_SURPRISE_BOUNDARY_STEP:?Set DYNAMIC_SURPRISE_BOUNDARY_STEP}"
resume_state="${RESUME_STATE:-}"
save_steps="${SAVE_STEPS:-[]}"

export PATH="${runtime_bin}:${PATH}"
export PYTHONPATH="${repo_root}/src:${repo_root}${PYTHONPATH:+:${PYTHONPATH}}"
export DIFFSYNTH_MODEL_BASE_PATH="${DIFFSYNTH_MODEL_BASE_PATH:-/mnt/vepfs02/datasets/kevin_wang/code/projects/FastWAM/checkpoints}"
export DIFFSYNTH_SKIP_DOWNLOAD=true
export RMBENCH_PUTBACK_LEROBOT="${RMBENCH_PUTBACK_LEROBOT:-/mnt/vepfs02/output/kevin_wang/memorywam/data/rmbench_lerobot_v21_rgb_v3/put_back_block}"
export MEMORYWAM_PUTBACK_STATS="${MEMORYWAM_PUTBACK_STATS:-/mnt/vepfs02/output/kevin_wang/memorywam/data/fastwam_putback_stats_rgb_v3/dataset_stats.json}"
export MEMORYWAM_PUTBACK_CONTINUOUS_LATENTS="${MEMORYWAM_PUTBACK_CONTINUOUS_LATENTS:-/mnt/vepfs02/output/kevin_wang/memorywam/data/fastwam_putback_continuous_episode_stride16_v4}"
export MEMORYWAM_PUTBACK_TEXT_CACHE="${MEMORYWAM_PUTBACK_TEXT_CACHE:-/mnt/vepfs02/output/kevin_wang/memorywam/data/fastwam_putback_text_cache_rgb_v3}"
export FASTWAM_ACTION_DIT_INIT="${FASTWAM_ACTION_DIT_INIT:-/mnt/vepfs02/output/kevin_wang/memorywam/model_assets/fastwam_fullkv_official_init/ActionDiT_linear_interp_Wan22_alphascale_1024hdim.pt}"
export DYNAMIC_SURPRISE_OUTPUT="${output_dir}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"

die() { echo "ERROR: $*" >&2; exit 2; }
[[ "${PHASE_END_STEP}" =~ ^(5000|10000|15000|20000|25000|30000|35000|40000)$ ]] || die "invalid PHASE_END_STEP"
[[ "${DYNAMIC_SURPRISE_BOUNDARY_STEP}" =~ ^(-1|5000|10000|15000|20000|25000|30000|35000)$ ]] || die "invalid boundary step"
[[ -s "${DYNAMIC_SURPRISE_MANIFEST}/manifest.json" ]] || die "missing manifest"
IFS=',' read -r -a visible_gpus <<<"${CUDA_VISIBLE_DEVICES}"
[[ ${#visible_gpus[@]} -eq 8 ]] || die "formal phase requires eight GPUs"

resume_args=()
if [[ -n "${resume_state}" ]]; then
  [[ -s "${resume_state}/trainer_state.json" ]] || die "invalid resume state: ${resume_state}"
  resume_args+=("resume=$(readlink -f "${resume_state}")")
elif [[ "${PHASE_END_STEP}" != 5000 ]]; then
  die "noninitial phase requires RESUME_STATE"
elif [[ -e "${output_dir}" ]]; then
  die "refusing to overwrite ${output_dir}"
fi

cd "${repo_root}"
"${runtime_bin}/python" - "${DYNAMIC_SURPRISE_MANIFEST}" "${DYNAMIC_SURPRISE_BOUNDARY_STEP}" <<'PY'
import sys
from fastwam.memory.dynamic_surprise import DynamicSurpriseManifestStore
DynamicSurpriseManifestStore(sys.argv[1], expected_boundary_step=int(sys.argv[2]), expected_episode_count=50)
PY

echo "phase_end=${PHASE_END_STEP} boundary_step=${DYNAMIC_SURPRISE_BOUNDARY_STEP} manifest=${DYNAMIC_SURPRISE_MANIFEST} resume=${resume_state:-none}"
mkdir -p "${output_dir}"
exec > >(tee -a "${output_dir}/train.log") 2>&1
echo "phase_start=$(date --iso-8601=seconds)"
accelerate launch \
  --config_file scripts/accelerate_configs/accelerate_fsdp_full_shard.yaml \
  --num_processes 8 \
  scripts/train.py \
  task=rmbench_putback_dynamic_surprise_k8_25k \
  "max_steps=${PHASE_END_STEP}" \
  "save_steps=${save_steps}" \
  "${resume_args[@]}"
