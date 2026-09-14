#!/usr/bin/env bash
set -euo pipefail

repo_root="${REPO_ROOT:-/mnt/vepfs02/output/kevin.wang/memorywam/code/fastwam_dynamic_multiframe_selection}"
runtime_bin="${RUNTIME_BIN:-/root/kevin_wang/envs/fastwam_guidemem/bin}"
source_run="${SOURCE_RUN:-/mnt/vepfs02/output/kevin.wang/memorywam/train/embodied_information_k8_putback_e2e_40k_seed42}"
resume_state="${RESUME_STATE:-${source_run}/checkpoints/state/step_010000}"
output_dir="${OUTPUT_DIR:-/mnt/vepfs01/output/kevin.wang/memorywam/train/embodied_information_k8_putback_e2e_40k_seed42_resume10k}"
manifest_root="${MANIFEST_ROOT:-/mnt/vepfs02/output/kevin.wang/memorywam/data/putback_embodied_information_planning_segments_v1}"
preflight_only="${PREFLIGHT_ONLY:-0}"

export PATH="${runtime_bin}:${PATH}"
export PYTHONPATH="${repo_root}/src:${repo_root}${PYTHONPATH:+:${PYTHONPATH}}"
export DIFFSYNTH_MODEL_BASE_PATH="${DIFFSYNTH_MODEL_BASE_PATH:-/mnt/vepfs02/output/kevin.wang/memorywam/model_assets/diffsynth_official}"
export DIFFSYNTH_SKIP_DOWNLOAD=true
export RMBENCH_PUTBACK_LEROBOT="${RMBENCH_PUTBACK_LEROBOT:-/mnt/vepfs02/output/kevin.wang/memorywam/data/rmbench_lerobot_v21_rgb_v3/put_back_block}"
export MEMORYWAM_PUTBACK_STATS="${MEMORYWAM_PUTBACK_STATS:-/mnt/vepfs02/output/kevin.wang/memorywam/data/fastwam_putback_stats_rgb_v3/dataset_stats.json}"
export MEMORYWAM_PUTBACK_CONTINUOUS_LATENTS="${MEMORYWAM_PUTBACK_CONTINUOUS_LATENTS:-/mnt/vepfs02/output/kevin.wang/memorywam/data/fastwam_putback_continuous_episode_stride16_v4}"
export MEMORYWAM_PUTBACK_TEXT_CACHE="${MEMORYWAM_PUTBACK_TEXT_CACHE:-/mnt/vepfs02/output/kevin.wang/memorywam/data/fastwam_putback_text_cache_rgb_v3}"
export FASTWAM_ACTION_DIT_INIT="${FASTWAM_ACTION_DIT_INIT:-/mnt/vepfs02/output/kevin.wang/memorywam/model_assets/fastwam_fullkv_official_init/ActionDiT_linear_interp_Wan22_alphascale_1024hdim.pt}"
export DYNAMIC_SURPRISE_MANIFEST="${manifest_root}"
export DYNAMIC_SURPRISE_BOUNDARY_STEP=-1
export EMBODIED_INFORMATION_OUTPUT="${output_dir}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export PYTHONUTF8=1 LANG=C.UTF-8 LC_ALL=C.UTF-8
export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"

die() { echo "ERROR: $*" >&2; exit 2; }
[[ "${preflight_only}" == "0" || "${preflight_only}" == "1" ]] || die "PREFLIGHT_ONLY must be 0 or 1"
[[ -d "${repo_root}" && -x "${runtime_bin}/python" ]] || die "repository or runtime missing"
[[ "${resume_state}" == */checkpoints/state/step_010000 ]] || die "resume must be the complete 10k state"
[[ ! -e "${output_dir}" ]] || die "refusing to overwrite output: ${output_dir}"
IFS=',' read -r -a visible_gpus <<<"${CUDA_VISIBLE_DEVICES}"
[[ ${#visible_gpus[@]} -eq 8 ]] || die "formal resume requires exactly eight GPUs"

for required in \
  "${resume_state}/pytorch_model_fsdp_0/.metadata" \
  "${resume_state}/optimizer_0/.metadata" \
  "${resume_state}/scheduler.bin" \
  "${resume_state}/trainer_state.json" \
  "${source_run}/checkpoints/weights/step_010000.pt" \
  "${DIFFSYNTH_MODEL_BASE_PATH}/Wan-AI/Wan2.2-TI2V-5B/diffusion_pytorch_model-00001-of-00003.safetensors" \
  "${DIFFSYNTH_MODEL_BASE_PATH}/Wan-AI/Wan2.2-TI2V-5B/diffusion_pytorch_model-00002-of-00003.safetensors" \
  "${DIFFSYNTH_MODEL_BASE_PATH}/Wan-AI/Wan2.2-TI2V-5B/diffusion_pytorch_model-00003-of-00003.safetensors" \
  "${DIFFSYNTH_MODEL_BASE_PATH}/DiffSynth-Studio/Wan-Series-Converted-Safetensors/models_t5_umt5-xxl-enc-bf16.safetensors" \
  "${DIFFSYNTH_MODEL_BASE_PATH}/DiffSynth-Studio/Wan-Series-Converted-Safetensors/Wan2.2_VAE.safetensors" \
  "${FASTWAM_ACTION_DIT_INIT}" \
  "${manifest_root}/manifest.json"; do
  [[ -s "${required}" ]] || die "missing complete resume artifact: ${required}"
done
[[ -d "${DIFFSYNTH_MODEL_BASE_PATH}/Wan-AI/Wan2.1-T2V-1.3B/google/umt5-xxl" ]] || \
  die "missing tokenizer under new model base"
for rank in {0..7}; do
  model_shard="$(find "${resume_state}/pytorch_model_fsdp_0" -maxdepth 1 -name "__${rank}_0.distcp" -print -quit)"
  optimizer_shard="$(find "${resume_state}/optimizer_0" -maxdepth 1 -name "__${rank}_0.distcp" -print -quit)"
  [[ -n "${model_shard}" && $(stat -c %s "${model_shard}") -gt 1000000000 ]] || die "model shard ${rank} is incomplete"
  [[ -n "${optimizer_shard}" && $(stat -c %s "${optimizer_shard}") -gt 1000000000 ]] || die "optimizer shard ${rank} is incomplete"
done

mkdir -p "$(dirname "${output_dir}")"
available_bytes="$(df --output=avail -B1 "$(dirname "${output_dir}")" | tail -1 | tr -d ' ')"
[[ "${available_bytes}" -ge 214748364800 ]] || die "output filesystem needs at least 200 GiB free"

cd "${repo_root}"
"${runtime_bin}/python" - "${resume_state}" "${source_run}/checkpoints/weights/step_010000.pt" <<'PY'
import json, sys
from fastwam.utils.compact_checkpoint import validate_portable_checkpoint
state, portable = sys.argv[1:]
payload = json.load(open(state + "/trainer_state.json"))
assert payload["global_step"] == 10000, payload
validate_portable_checkpoint(portable, expected_step=10000)
print("resume_checkpoint_validation=pass")
PY

resolved_config="$("${runtime_bin}/python" scripts/train.py task=rmbench_putback_embodied_information_k8_40k \
  output_dir="${output_dir}" resume="${resume_state}" save_every=5000 \
  save_training_state_every=10000 keep_training_state_checkpoints=1 --cfg job)"
printf '%s\n' "${resolved_config}" | "${runtime_bin}/python" -c '
import sys
from omegaconf import OmegaConf
cfg=OmegaConf.create(sys.stdin.read())
expected={"max_steps":40000,"save_every":5000,"save_training_state_every":10000,
"keep_training_state_checkpoints":1,"seed":42,"batch_size":1,
"gradient_accumulation_steps":1,"learning_rate":2e-4,"mixed_precision":"bf16",
"model.native_cache.mode":"layerwise","model.native_cache.memory_tokens":8,
"model.native_cache.group_size":4,"model.native_cache.anchor_frames":2,
"model.native_cache.recent_frames":4,"model.native_cache.recursive":False}
for path,want in expected.items():
    got=OmegaConf.select(cfg,path)
    if got != want: raise SystemExit(f"config mismatch {path}: {got!r} != {want!r}")
'

echo "resume_state=${resume_state}"
echo "output_dir=${output_dir}"
echo "portable_steps=step_015000,step_020000,step_025000,step_030000,step_035000,step_040000"
echo "full_training_state_steps=step_020000,step_030000,step_040000 keep_latest=1"
if [[ "${preflight_only}" == "1" ]]; then echo "preflight_status=ok"; exit 0; fi

mkdir -p "${output_dir}"
exec > >(tee -a "${output_dir}/train.log") 2>&1
echo "training_resume_start=$(date --iso-8601=seconds)"
accelerate launch \
  --config_file scripts/accelerate_configs/accelerate_fsdp_full_shard.yaml \
  --num_processes 8 \
  scripts/train.py task=rmbench_putback_embodied_information_k8_40k \
  output_dir="${output_dir}" \
  resume="${resume_state}" \
  save_every=5000 \
  save_training_state_every=10000 \
  keep_training_state_checkpoints=1
